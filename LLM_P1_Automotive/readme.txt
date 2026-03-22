# Automotive Vehicle Inspection Report Generator
### Fine-Tuned LLM with QLoRA (4-bit Quantization + LoRA/PEFT) on Real NHTSA Data

## The Real Problem

 Technicians waste 15–20 minutes per vehicle manually converting raw customer descriptions into structured inspection reports.

**This project solves that with a fine-tuned LLM.**


## 📦 Dataset

**Source:** [NHTSA Office of Defects Investigation (ODI)](https://www.nhtsa.gov/vehicle-safety/complaints)  
**Direct URL:** `https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip`  
**License:** U.S. Public Domain (government data)  
**Size:** 1.5 Million+ records since 1995  

### Key Columns Used

| Column        | Description 
| `CDESCR`      | Raw complaint text from vehicle owner 
| `COMPDESC`    | Component (ENGINE, BRAKES, POWER TRAIN, etc.) 
| `MAKETXT`     | Vehicle make (FORD, TOYOTA, BMW) 
| `MODELTXT`    | Vehicle model 
| `YEARTXT`     | Model year 


## Architecture

NHTSA ODI Complaint (raw text)
         ↓
 Severity Classifier (rule-based)
   CRITICAL / HIGH / MEDIUM / LOW
         ↓
 Instruction Prompt Formatter
   ### Instruction → ### Response
         ↓
 ┌────────────────────────────────────┐
 │  TinyLlama-1.1B                    │
 │  + 4-bit NF4 Quantization          │  ← 75% memory reduction
 │  + LoRA (r=16, all attn heads)     │  ← 0.19% params trained
 │  + paged_adamw_8bit optimizer      │  ← memory-efficient
 └────────────────────────────────────┘
         ↓
  SFTTrainer (2 epochs, 2000 samples)
         ↓
  Structured Inspection Report
  ┌─────────────────────────────┐
  │ Component    : Engine       │
  │ Severity     : CRITICAL     │
  │ Urgency      : Immediate    │
  │ Diagnostic   : 1. Check...  │
  │ TSB/Recall   : Yes          │
  │ Priority     : 9/10         │
  └─────────────────────────────┘

Base Model:[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)


## Extend This Project

- **Scale up**: use full 1.5M NHTSA records
- **Better model**: `mistralai/Mistral-7B-Instruct-v0.2`
---

