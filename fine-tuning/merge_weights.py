# merge model with weights
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch
import os

# Define paths
base_model = "meta-llama/Llama-3.2-3B-Instruct" #3.2-3B 3.1-8B
local_adapters_path = "/n/netscratch/cga/Lab/anasuto/immigration/adapters/Llama-32-3B-multi"
local_model = "/n/netscratch/cga/Lab/anasuto/immigration/llm/Llama-32-3B-multi"  # Path for merged model

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": 0})

# Load fine-tuned model from local path
print(f"Loading fine-tuned model from local path: {local_adapters_path}")
finetuned_model = PeftModel.from_pretrained(model, local_adapters_path)

# Save adapters and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
os.makedirs(local_adapters_path, exist_ok=True)
finetuned_model.save_pretrained(local_adapters_path)
tokenizer.save_pretrained(local_adapters_path)

# Merge LoRA weights into the base model
print("Merging LoRA weights into the base model...")
merged_model = finetuned_model.merge_and_unload()
os.makedirs(local_model, exist_ok=True)
merged_model.save_pretrained(local_model)
tokenizer.save_pretrained(local_model)

print(f"Model saved to: {local_model}")