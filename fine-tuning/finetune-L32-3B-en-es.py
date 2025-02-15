
from huggingface_hub import HfApi, HfFolder
HfFolder.save_token("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")
api = HfApi()
api.whoami()  # This should print your user info


import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import wandb


# Model from Hugging Face hub
base_model = "meta-llama/Llama-3.2-3B-Instruct"

# load datasets in two or multiple languages
dataset_name_1 = "andreanasuto/migTest-es"
dataset_name_2 = "andreanasuto/migTest-en"


dataset_1 = load_dataset(dataset_name_1, split={'train': 'train', 'test': 'test'})
dataset_2 = load_dataset(dataset_name_2, split={'train': 'train', 'test': 'test'})

dataset_train = concatenate_datasets([dataset_1['train'], dataset_2['train']])
dataset_test = concatenate_datasets([dataset_1['test'], dataset_2['test']])


# Fine-tuned model
new_model = "Llama-32-3B-en-es"


compute_dtype = torch.bfloat16 #getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},
)

model.config.use_cache = False
model.config.pretraining_tp = 1

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"


# Define hyperparameters
learning_rate = 2e-4
gradient_accumulation_steps = 1
per_device_train_batch_size = 16
num_train_epochs = 10
max_seq_length = 1024
max_steps = -1
optimizer = "paged_adamw_32bit"
max_grad_norm = 0.3
max_length = 1024

# Initialize W&B with a custom run name
wandb.init(
    project="mig",
    reinit=True,
    name="llama-32-3b-mig",
    config={
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "max_steps": max_steps,
        "optimizer": optimizer,
        "max_grad_norm": max_grad_norm,
    }
)


peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=64, 
    lora_alpha=32, 
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

model.gradient_checkpointing_enable()

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_train_epochs, 
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    optim=optimizer,
    weight_decay=0.001,
    save_steps=25,
    logging_steps=25,
    fp16=False,
    gradient_checkpointing=True,
    bf16=False,
    max_steps=max_steps,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    max_grad_norm=max_grad_norm,
    dataloader_num_workers = 4,
    dataloader_pin_memory = True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    peft_config=peft_config,
    #dataset_text_field="text",
    #max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_params,
    #packing=False,
)

# Check for existing checkpoints and resume if found
#last_checkpoint = None
#if os.path.exists(training_params.output_dir) and os.listdir(training_params.output_dir):
#    last_checkpoint = max(
#        [os.path.join(training_params.output_dir, d) for d in os.listdir(training_params.output_dir)],
#        key=os.path.getctime,
#    )
#    print(f"Resuming training from checkpoint: {last_checkpoint}")

# Train the model (resume if checkpoint exists)
#trainer.train(resume_from_checkpoint=last_checkpoint)

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

trainer.model.push_to_hub(new_model)
trainer.tokenizer.push_to_hub(new_model)

exit(0)