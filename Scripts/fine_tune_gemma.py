import sys
import os

print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os

# Define the paths based on our project structure
dataset_path = "../data/processed/neuro_mirror_data.jsonl"
output_dir = "../models/gemma-3n-fine-tuned/"
model_name = "google/gemma-3-7b"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# 1. Load the model with BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
)
model.config.use_cache = False
model.config.pretraining_tp = 1 # Recommended for Gemma

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Configure the model for QLoRA fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load our custom dataset and format it for the model
dataset = load_dataset("json", data_files=dataset_path, split="train")

def formatting_prompts_func(examples):
    formatted_texts = []
    for messages in examples['messages']:
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(formatted_text)
    return {"text": formatted_texts}

dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=os.cpu_count())

# 4. Train the model using SFTTrainer
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    max_steps=-1,
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=training_args,
)

# Start training
trainer.train()

# 5. Save the fine-tuned model
trainer.model.save_pretrained(output_dir)