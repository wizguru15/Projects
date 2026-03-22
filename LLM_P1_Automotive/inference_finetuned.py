import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# STEP 1 — LOAD YOUR FINE-TUNED MODEL

def load_model(
    base_model   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path = "./automotive_inspection_model"   
):
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map  = "cpu",          # CPU — no GPU needed for inference
        torch_dtype = torch.float32,  # float32 for CPU stability
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval() 

    print("Fine-tuned model is ready!\n")
    return model, tokenizer


# STEP 2 — GENERATE INSPECTION REPORT

def generate_report(complaint, model, tokenizer,vehicle="Unknown", max_new_tokens=300):

    prompt = (
        f"### Instruction:\n"
        f"You are an automotive inspection system. Convert the raw vehicle "
        f"defect complaint below into a structured professional inspection report.\n\n"
        f"Vehicle    : {vehicle}\n"         
        f"Raw Complaint:\n{complaint}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 512,
    ).to("cpu")

    with torch.no_grad(): 
        outputs = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            temperature        = 0.3,   
            do_sample          = True,
            pad_token_id       = tokenizer.eos_token_id,
            repetition_penalty = 1.2, 
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip()
    return response


# STEP 3 — TEST COMPLAINTS

TEST_COMPLAINTS = [
    {
        "vehicle"   : "2019 Toyota Camry",
        "complaint" : "WHILE DRIVING AT HIGHWAY SPEED THE ENGINE SUDDENLY STALLED "
                      "WITHOUT WARNING. VEHICLE LOST POWER STEERING AND POWER BRAKES. "
                      "HAD DIFFICULTY PULLING TO THE SIDE OF THE ROAD. ENGINE WOULD "
                      "NOT RESTART FOR 20 MINUTES. THIS HAS HAPPENED 3 TIMES.",
    },
    {
        "vehicle"   : "2021 Ford F-150",
        "complaint" : "BRAKE PEDAL GOES TO THE FLOOR ON FIRST APPLICATION AFTER "
                      "VEHICLE HAS BEEN SITTING OVERNIGHT. SECOND PUMP OF BRAKES "
                      "RESTORES NORMAL FEEL. DEALER CANNOT DUPLICATE. BRAKE FLUID "
                      "LEVEL NORMAL. NO VISIBLE LEAKS.",
    },
    {
        "vehicle"   : "2020 Honda Accord",
        "complaint" : "VEHICLE HESITATES AND JERKS WHEN ACCELERATING FROM A STOP. "
                      "CHECK ENGINE LIGHT CAME ON. DEALER SAID IT WAS A TRANSMISSION "
                      "SOLENOID AND REPLACED IT BUT PROBLEM RETURNED WITHIN 2 WEEKS. "
                      "FUEL ECONOMY DROPPED FROM 28 TO 21 MPG.",
    },
]

if __name__ == "__main__":

    print("  AUTOMOTIVE INSPECTION REPORT GENERATOR")

    model, tokenizer = load_model()

    # Run on all test complaints
    for i, sample in enumerate(TEST_COMPLAINTS, 1):
        print(f"  COMPLAINT #{i} — {sample['vehicle']}")
        print(f"{'━' * 65}")
        print(f"Raw Complaint:\n{sample['complaint']}\n")
        print(f"Generated Inspection Report:\n")

        report = generate_report(sample["complaint"], model, tokenizer)
        print(report)
        print()

    print(" TRY YOUR OWN COMPLAINT")
    user_input = input("Enter a vehicle complaint (or press Enter to skip): ").strip()

    if user_input:
        print(f"\n Generated Inspection Report:\n")
        report = generate_report(user_input, model, tokenizer)
        print(report)
