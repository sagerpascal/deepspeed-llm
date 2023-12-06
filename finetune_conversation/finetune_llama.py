# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass, field
from typing import Optional
from peft import LoraConfig, AdaLoraConfig, LoHaConfig, LoKrConfig
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from trl import SFTTrainer

# This example fine-tunes Llama v2 model on Guanace dataset
# using QLoRA. At the end of the script we perform merging the weights
# Use it by correctly passing --model_name argument when running the
# script.
#
# Versions used:
# accelerate == 0.21.0
# peft == 0.4.0
# bitsandbytes == 0.40.2
# transformers == 4.31.0
# trl == 0.4.7

# For models that have `config.pretraining_tp > 1` install:
# pip install git+https://github.com/huggingface/transformers.git

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=1e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)

    lora_method: Optional[str] = field(
        default="lora",
        metadata={"help": "Choose lora method, either lora or adalora, loha, lokr"},
    )
    # shared across lora flavours where applicable, use default settings otherwise
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)

    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_quant: Optional[int] = field(
        default=1,
        metadata={"help": "Choose quantization method should quantisation be used"},
    )
    # BNB options start here
    bnb_use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    bnb_use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )

    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=300, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    wandb_project: str = field(
        default="sds_llama",
        metadata={"help": "wand project this run will log to"},
    )

    wandb_run: str = field(
        default="run",
        metadata={"help": "wandb name for this run"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

os.environ["WANDB_PROJECT"] = script_args.wandb_project
os.environ["WANDB_NAME"] = script_args.wandb_run

def create_and_prepare_model(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}
    if args.use_quant == 1:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.bnb_use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.bnb_use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
    else:
        bnb_config = None


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_flash_attention_2=False,
        use_auth_token=True
    )


    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    if script_args.lora_method == "lora":
        peft_config = LoraConfig(
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            # hardcoded for llamav2 needs to change for other model
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif script_args.lora_method == "adalora":
        peft_config = AdaLoraConfig(
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            # hardcoded for llamav2 needs to change for other model
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_r=script_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

    elif script_args.lora_method == "loha":
        peft_config = LoHaConfig(
            target_modules=['q_proj','k_proj','v_proj','o_proj'],  # hardcoded for llamav2 needs to change for other model
            alpha=script_args.lora_alpha,
            module_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            task_type="CAUSAL_LM",
        )

    elif script_args.lora_method == "lokr":
        peft_config = LoKrConfig(
            target_modules = ['q_proj','k_proj','v_proj','o_proj'], # hardcoded for llamav2 needs to change for other model
            alpha=script_args.lora_alpha,
            module_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            task_type="CAUSAL_LM",
        )


    else:
        raise("unknown lora config")

    return model, peft_config, tokenizer


training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False
dataset = load_dataset(script_args.dataset_name, split="train")

# Fix weird overflow issue with fp16 training
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints_"+script_args.lora_method+"_"+str(script_args.use_quant))
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "merged_final_checkpoints_"+script_args.lora_method+"_"+str(script_args.use_quant))
    model.save_pretrained(output_merged_dir, safe_serialization=True)
