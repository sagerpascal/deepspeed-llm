import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from awq import AutoAWQForCausalLM


os.environ["WANDB_PROJECT"] = "sds_llama_infrence"
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    quant_method: Optional[str] = field(
        default="awq",
        metadata={"help": "Choose which quantization method: no, bnb, awq, gptq"},
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

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

if args.quant_method == 'no':
    model_name = "lmsys/vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
    )

elif args.quant_method == 'bnb':
    model_name = "lmsys/vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": 0}

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.bnb_use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_use_nested_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        )

elif args.quant_method == 'awq':
    # TODO use code snippets from hugginface hub
    ### AWQ
    ### https://huggingface.co/TheBloke/vicuna-7B-v1.5-AWQ
    model_name_or_path = "TheBloke/vicuna-7B-v1.5-AWQ"
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

elif args.quant_method == 'gptq':
    # TODO use code snippets from hugginface hub
    ### gptq
    ### https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ
    model_name_or_path = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
    # To use a different branch, change revision
    # For example: revision="main"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


else:
    raise("unknown quantization method")



# TODO inference benchmark here
text = ("Why is it good to obtain a PhD?")
inputs = tokenizer(text, return_tensors="pt").to(0)

out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
