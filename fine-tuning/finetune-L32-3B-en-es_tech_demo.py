# ============================================================================
# AUTHENTICATION & SETUP
# ============================================================================
from huggingface_hub import HfApi, HfFolder

# Save HuggingFace authentication token for model access and pushing results
# NOTE: In production, use environment variables instead of hardcoding tokens
HfFolder.save_token("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")
api = HfApi()
api.whoami()  # Verify authentication

# ============================================================================
# IMPORTS
# ============================================================================
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
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType #peft [parameter-efficient fine-tuning]
from trl import SFTTrainer
import wandb

# ============================================================================
# MODEL & DATASET CONFIGURATION
# ============================================================================

# Base Model Selection
# WHY LLAMA 3.2-3B? Smaller model suitable for fine-tuning with limited resources
# ALTERNATIVES: Llama-3.2-1B (lighter), Llama-3.1-8B (more capable but heavier)
base_model = "meta-llama/Llama-3.2-3B-Instruct"

# Multilingual Dataset Configuration
# Loading separate Spanish and English datasets for bilingual fine-tuning
# Q: Why separate datasets? A: Allows for language-specific data management
dataset_name_1 = "andreanasuto/migTest-es"  # Spanish dataset
dataset_name_2 = "andreanasuto/migTest-en"  # English dataset

# Load and merge datasets
dataset_1 = load_dataset(dataset_name_1, split={'train': 'train', 'test': 'test'})
dataset_2 = load_dataset(dataset_name_2, split={'train': 'train', 'test': 'test'})

# Concatenate for joint training (enables cross-lingual learning)
dataset_train = concatenate_datasets([dataset_1['train'], dataset_2['train']])
dataset_test = concatenate_datasets([dataset_1['test'], dataset_2['test']])

# Output model name
new_model = "Llama-32-3B-en-es"

# ============================================================================
# QUANTIZATION CONFIGURATION (4-bit for memory efficiency)
# ============================================================================

# Compute dtype for model operations
# bfloat16: Better numerical stability than float16, supported on modern GPUs
# ALTERNATIVE: torch.float16 (more compatible but less stable)
compute_dtype = torch.bfloat16

# 4-bit Quantization Configuration
# WHY QUANTIZATION? Reduces memory usage by ~75%, enabling training on consumer GPUs
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Use 4-bit precision
    bnb_4bit_quant_type="nf4",             # NF4 (NormalFloat4): optimal for neural nets
                                            # ALTERNATIVE: "fp4" (standard 4-bit float)
    bnb_4bit_compute_dtype=compute_dtype,  # Computation precision (bf16)
    bnb_4bit_use_double_quant=False,       # Double quantization saves more memory
                                            # Set True if you need even more memory savings
)

# Device selection
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ============================================================================
# MODEL LOADING & PREPARATION
# ============================================================================

# Load pre-trained model with quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},  # Load entire model on GPU 0
                         # For multi-GPU: use "auto" or specific mapping
)

# Model configuration
model.config.use_cache = False          # Disable KV cache during training (saves memory)
model.config.pretraining_tp = 1         # Tensor parallelism degree (1 = no parallelism)

# Enable gradient checkpointing for memory efficiency
# TRADEOFF: Reduces memory by ~30% but increases training time by ~20%
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)  # Prepare quantized model for training

# ============================================================================
# TOKENIZER SETUP
# ============================================================================

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding
# NOTE: padding_side defaults to "right" for causal LM

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Learning Rate
# WHY 2e-4? Standard for LoRA fine-tuning, balances speed and stability
# RANGE: 1e-5 (conservative) to 5e-4 (aggressive)
learning_rate = 2e-4

# Gradient Accumulation
# Simulates larger batch sizes: effective_batch = batch_size × accumulation_steps
# INCREASE if you have memory issues or want larger effective batch size
gradient_accumulation_steps = 1

# Batch Size
# WHY 16? Balances training speed and memory usage for 3B model
# ADJUST: Reduce to 8 or 4 if OOM errors occur
per_device_train_batch_size = 16

# Training Duration
# WHY 10 epochs? Enough for domain adaptation without overfitting
# MONITOR: Use validation loss to determine if more/fewer epochs needed
num_train_epochs = 10

# Sequence Length
# WHY 1024? Balances context window and memory usage
# ALTERNATIVES: 512 (save memory), 2048+ (longer context but more memory)
max_seq_length = 1024
max_steps = -1  # -1 means train for full epochs (not step-limited)

# Optimizer
# paged_adamw_32bit: Memory-efficient AdamW variant for large models
# ALTERNATIVE: "adamw_torch" (standard but uses more memory)
optimizer = "paged_adamw_32bit"

# Gradient Clipping
# Prevents exploding gradients (common in fine-tuning)
# RANGE: 0.3-1.0 (higher = less aggressive clipping)
max_grad_norm = 0.3

max_length = 1024

# ============================================================================
# WEIGHTS & BIASES TRACKING
# ============================================================================

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

# ============================================================================
# LoRA CONFIGURATION (Parameter-Efficient Fine-Tuning)
# ============================================================================

# LoRA: Low-Rank Adaptation - only trains small adapter layers
# ADVANTAGE: Reduces trainable parameters by 99%+, enabling faster training
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type: causal language modeling
    inference_mode=False,           # Training mode
    
    # Rank (r): Dimension of low-rank decomposition
    # WHY 64? Balances expressiveness and parameter count
    # RANGE: 8 (lightweight), 16-32 (standard), 64-128 (more expressive)
    # TRADEOFF: Higher r = more parameters but potentially better performance
    r=64,
    
    # Alpha: Scaling factor for LoRA updates
    # WHY 32? Common choice is alpha = 2×r for balanced learning
    # EFFECT: Controls magnitude of adapter contributions
    lora_alpha=32,
    
    # Dropout: Regularization to prevent overfitting
    # WHY 0.1? Standard dropout rate for fine-tuning
    # RANGE: 0.0 (no dropout) to 0.2 (aggressive regularization)
    lora_dropout=0.1,
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# ============================================================================
# PARAMETER COUNTING (for transparency)
# ============================================================================

def print_trainable_parameters(model):
    """
    Prints trainable vs total parameters
    Expected: ~1% trainable with LoRA (vs 100% for full fine-tuning)
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

print_trainable_parameters(model)
model.gradient_checkpointing_enable()

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    optim=optimizer,
    
    # Weight Decay: L2 regularization
    # WHY 0.001? Prevents overfitting without excessive regularization
    weight_decay=0.001,
    
    # Checkpointing & Logging
    save_steps=25,      # Save checkpoint every 25 steps (adjust based on dataset size)
    logging_steps=25,   # Log metrics every 25 steps
    
    # Mixed Precision Training
    fp16=False,         # float16 mixed precision (set True for V100/T4 GPUs)
    bf16=False,         # bfloat16 mixed precision (set True for A100/H100 GPUs)
                        # NOTE: Can't use both; bf16 preferred for Ampere+ GPUs
    
    gradient_checkpointing=True,  # Memory optimization
    max_steps=max_steps,
    
    # Learning Rate Schedule
    warmup_ratio=0.03,            # Warm up for first 3% of training
    lr_scheduler_type="constant", # Keep LR constant after warmup
                                  # ALTERNATIVES: "cosine" (decay), "linear" (linear decay)
    
    # Optimization
    group_by_length=True,          # Group sequences by length (efficiency boost)
    max_grad_norm=max_grad_norm,
    
    # Weights & Biases integration
    report_to="wandb",
    
    # Data loading optimization
    dataloader_num_workers=4,      # Parallel data loading (adjust to CPU cores)
    dataloader_pin_memory=True     # Faster GPU transfer
)

# ============================================================================
# TRAINER INITIALIZATION
# ============================================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    peft_config=peft_config,
    # dataset_text_field="text",  # Specify if dataset has specific text column
    # max_seq_length=max_seq_length,  # Handled by tokenizer
    tokenizer=tokenizer,
    args=training_params,
    # packing=False,  # Set True to pack multiple samples per sequence (efficiency)
)

# ============================================================================
# CHECKPOINT RESUME (Optional - currently commented out)
# ============================================================================
# Uncomment to enable automatic resume from last checkpoint
# Useful for interrupted training runs

# last_checkpoint = None
# if os.path.exists(training_params.output_dir) and os.listdir(training_params.output_dir):
#     last_checkpoint = max(
#         [os.path.join(training_params.output_dir, d) 
#          for d in os.listdir(training_params.output_dir)],
#         key=os.path.getctime,
#     )
#     print(f"Resuming training from checkpoint: {last_checkpoint}")
# trainer.train(resume_from_checkpoint=last_checkpoint)

# ============================================================================
# TRAINING
# ============================================================================

trainer.train()

# ============================================================================
# SAVE & UPLOAD MODEL
# ============================================================================

# Save locally
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Push to HuggingFace Hub (requires authentication)
trainer.model.push_to_hub(new_model)
trainer.tokenizer.push_to_hub(new_model)

exit(0)