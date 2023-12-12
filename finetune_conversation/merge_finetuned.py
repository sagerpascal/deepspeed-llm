import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
)
import torch

@dataclass
class ScriptArguments:
    in_adaptder: Optional[str] = field(
        default="final_checkpoints_lora_0"
    )
    # BNB options start here
    base_model: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf"
    )
    out_path: Optional[str] = field(
        default=False
    )


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    use_flash_attention_2=False,
    use_auth_token=True
)

from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(args.in_adapter, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = "merged_"+args.in_adapter
model.save_pretrained(output_merged_dir, safe_serialization=True)