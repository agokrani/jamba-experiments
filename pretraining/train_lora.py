from datasets import load_dataset
from peft import LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,  BitsAndBytesConfig
from functools import partial

import mamba_ssm

def load_data(train_dataset, tokenizer, max_length):
    dataset = load_dataset("json", data_files=train_dataset)
    return dataset.map(
            partial(tokenize, max_length=max_length, tokenizer=tokenizer),
            batched=False,
            num_proc=os.cpu_count()//PartialState().num_processes,    # multithreaded
    )

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int4_skip_modules=["mamba"] #Maybe not necessary (per axoltl) but to test.
)
tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

training_args = TrainingArguments(
        output_dir="/workspace/models/jamba-pt-esther",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim = "adamw_8bit",
        max_grad_norm = 0.3,
        weight_decay = 0.001,
        warmup_ratio = 0.03,
        gradient_checkpointing=True,
        logging_dir='/workspace/logs',
        logging_steps=1,
        max_steps=50,
        group_by_length=True,
        lr_scheduler_type = "linear",
        learning_rate=2e-3,
        save_total_limit=2
)

lora_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        init_lora_weights=False,
        r=16,
        target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
model = AutoModelForCausalLM.from_pretrained(
        "ai21labs/Jamba-v0.1", 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config, 
        cache_dir="/workspace/models",
        use_mamba_kernels=True
)

