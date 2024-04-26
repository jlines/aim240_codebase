# !pip install -q peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

import json
import datetime
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# Model from Hugging Face hub
base_model = "meta-llama/Llama-2-7b-chat-hf"
base_model = "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
guanaco_dataset = "CheshireAI/guanaco-unchained"

solid_dataset_path = "data/finetune_dataset.txt"

# Fine-tuned model
new_model = "llama-2-7b-chat-ssn-agent-aws"

with open("credentials.json") as credentials_file:
    credentials = json.loads(credentials_file.read())

access_token = credentials["HUGGINGFACE_API_KEY"]

guanaco_dataset = load_dataset(guanaco_dataset, split="train")
solid_dataset = load_dataset("text", data_files=solid_dataset_path, split="train")

dataset = concatenate_datasets([guanaco_dataset, solid_dataset])

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,  # True may use less mem
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # added due to meduim article
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=32,  # this may act as a learning rate in the Low Rank Adapter
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=4,  # may reduce to 1 to reduce mem
    per_device_eval_batch_size=4,  # may reduce to 1 to reduce mem
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,  # 256 will reduce mem requirements
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and True:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

start = datetime.datetime.now()

trainer.train()

end = datetime.datetime.now()
elapsed = end - start

print(f"Training took {elapsed}")

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
