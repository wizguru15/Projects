

# STEP 1 — IMPORTS

import pandas as pd
import requests
import zipfile
import io
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch




# STEP 2 — DOWNLOAD & LOAD NHTSA COMPLAINTS DATASET

# Source  : NHTSA Office of Defects Investigation (ODI)
# URL     : https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip

def load_dataset(sample_size=2000):

    url = "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip"

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            filename = [f for f in z.namelist() if f.endswith(".txt")][0]
            with z.open(filename) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    encoding="latin-1",
                    header=None,
                    on_bad_lines="skip",
                    low_memory=False,
                )

    except Exception as e:
        print(f"⚠️  Direct download failed ({e}). Using fallback CSV from HuggingFace...")
        # Fallback: smaller NHTSA-derived dataset hosted on HuggingFace
        from datasets import load_dataset
        hf_data = load_dataset("nateraw/us-accidents", split="train")
        # Restructure to match expected format
        df = pd.DataFrame({
            "complaint": hf_data["Description"][:sample_size],
            "component": ["VEHICLE"] * sample_size,
            "make": ["UNKNOWN"] * sample_size,
            "model": ["UNKNOWN"] * sample_size,
            "year": ["2020"] * sample_size,
        })
        return df

    col_names = [
        "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
        "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC",
        "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
        "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
        "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN",
        "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED", "DOT",
        "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN",
        "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME",
        "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
        "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"
    ]

    # Assign only as many column names as there are columns in the file
    df.columns = col_names[:len(df.columns)]

    keep_cols = [c for c in ["CDESCR", "COMPDESC", "MAKETXT", "MODELTXT", "YEARTXT"] if c in df.columns]
    df = df[keep_cols].copy()
    df.columns = ["complaint", "component", "make", "model", "year"][:len(keep_cols)]

    df = df.dropna(subset=["complaint"])
    df["complaint"] = df["complaint"].astype(str).str.strip()
    df = df[df["complaint"].str.len() > 30]   # Drop very short complaints
    df = df[df["complaint"].str.len() < 1000]  # Drop excessively long ones

    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    return df



# STEP 3 — BUILD SEVERITY CLASSIFIER

CRITICAL_KEYWORDS  = ["fire", "stall", "accident", "crash", "death", "injury",
                       "rollover", "brake fail", "no brakes", "sudden acceleration",
                       "airbag deploy", "explosion", "smoke", "caught fire"]

HIGH_KEYWORDS      = ["stalls", "hesitation", "loss of power", "vibration",
                       "overheating", "warning light", "won't start", "leaking",
                       "grinding", "shaking", "rough idle", "misfire", "noise"]

MEDIUM_KEYWORDS    = ["intermittent", "occasional", "sometimes", "minor",
                       "rattle", "squeak", "slight", "small leak", "slow"]

def classify_severity(complaint_text):

    text = complaint_text.lower()
    if any(kw in text for kw in CRITICAL_KEYWORDS):
        return "CRITICAL", "Immediate — Do Not Drive"
    elif any(kw in text for kw in HIGH_KEYWORDS):
        return "HIGH",     "Within 24-48 Hours"
    elif any(kw in text for kw in MEDIUM_KEYWORDS):
        return "MEDIUM",   "Within 1 Week"
    else:
        return "LOW",      "Next Scheduled Service"



# STEP 4 — FORMAT AS INSTRUCTION 

def build_inspection_report(row):

    severity, urgency = classify_severity(str(row.get("complaint", "")))
    component = str(row.get("component", "UNSPECIFIED")).strip().title()
    make      = str(row.get("make",  "Unknown")).strip().title()
    model     = str(row.get("model", "Unknown")).strip().title()
    year      = str(row.get("year",  "Unknown")).strip()
    complaint = str(row.get("complaint", "")).strip()

    report = (
        f"VEHICLE INSPECTION REPORT\n"
        f"{'─'*40}\n"
        f"Vehicle       : {year} {make} {model}\n"
        f"Component     : {component}\n"
        f"Defect Type   : Functional Failure / Owner Reported Defect\n"
        f"Severity      : {severity}\n"
        f"Urgency       : {urgency}\n"
        f"{'─'*40}\n"
        f"COMPLAINT SUMMARY:\n{complaint[:300]}\n"
        f"{'─'*40}\n"
        f"RECOMMENDED ACTION:\n"
        f"Inspect {component} system immediately. Verify complaint under "
        f"controlled conditions. Document findings and escalate if severity "
        f"is confirmed. Check for related TSBs or open recalls before repair."
    )
    return report


def format_prompt(row):

    complaint = str(row.get("complaint", "")).strip()[:400]
    report    = build_inspection_report(row)

    return {
        "text": (
            f"### Instruction:\n"
            f"You are an automotive inspection system. Convert the raw vehicle "
            f"defect complaint below into a structured professional inspection report.\n\n"
            f"Raw Complaint:\n{complaint}\n\n"
            f"### Response:\n{report}"
        )
    }


# STEP 5 — LOAD & PREPARE DATASET

df       = load_dataset(sample_size=2000)
records  = df.to_dict(orient="records")
hf_data  = Dataset.from_list([format_prompt(r) for r in records])

split    = hf_data.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
test_ds  = split["test"]


# STEP 6 — QUANTIZATION CONFIG

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",               # NormalFloat4 — best quality/compression
    bnb_4bit_compute_dtype=torch.bfloat16,    # Compute in fp16 for GPU speed
    bnb_4bit_use_double_quant=True,          # Quantize the quant constants too (~saves 0.4 bits)
)

# STEP 7 — LOAD MODEL & TOKENIZER

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False 

# STEP 8 — LORA / PEFT CONFIGURATION

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                            # Rank (higher = more capacity)
    lora_alpha=32,                                   # Scaling (2 × r is standard)
    lora_dropout=0.05,                               # Prevent overfitting
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # All attention heads
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# STEP 9 — TRAINING ARGUMENTS

from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir                  = "./checkpoints",

    num_train_epochs            = 1,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 4,
    gradient_checkpointing      = True,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 10,
    fp16                        = False,
    bf16                        = True,

    logging_steps               = 25,
    eval_steps                  = 100,
    save_steps                  = 100,
    eval_strategy               = "steps",
    save_strategy               = "steps",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "none",
    optim                       = "paged_adamw_8bit",

    dataset_text_field          = "text",
    max_length                  = 512,   
    packing                     = False,
)

trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = train_ds,
    eval_dataset     = test_ds,
    args             = sft_config,
)

trainer.train()
print("\n Fine-tuning complete!")


# STEP 10 — SAVE FINE-TUNED MODEL

SAVE_PATH = "automotive_inspection_model"

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Fine Tuned Model saved to: {SAVE_PATH}/")

