import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
import os

# Define the paths based on our project structure
dataset_path = "../data/processed/neuro_mirror_data.jsonl"
output_dir = "../models/gemma-3n-fine-tuned/"
model_name = "google/gemma-3-7b"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# 1. Load the base Gemma model and prepare it for QLoRA
# We'll use 4-bit quantization for efficiency.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 2. Configure the model for QLoRA fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 3. Load our custom dataset and format it for the model
def formatting_prompts_func(examples):
    formatted_texts = []
    for messages in examples['messages']:
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(formatted_text)
    return {"text": formatted_texts}

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=os.cpu_count())

# 4. Train the model using Hugging Face's Trainer
from trl import SFTTrainer

training_args = FastLanguageModel.get_training_arguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    learning_rate=2e-4,
    num_train_epochs=3,
    max_steps=-1,
    seed=3407,
    output_dir=output_dir,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=os.cpu_count(),
    packing=False,
    args=training_args,
)

# Start training
trainer.train()

# 5. Save the fine-tuned model
# This will save the LoRA adapter weights, which are small in size.
model.save_pretrained_merged(output_dir, tokenizer=tokenizer)