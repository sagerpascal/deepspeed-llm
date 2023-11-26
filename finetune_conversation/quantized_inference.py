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

os.environ["WANDB_PROJECT"] = "sds_llama"
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    quant_method: Optional[str] = field(
        default="no",
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
    raise("not implemented")
elif args.quant_method == 'gptq':
    # TODO use code snippets from hugginface hub
    ### gptq
    ### https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ
    raise ("not implemented")
else:
    raise("unknown quantization method")



# TODO inference benchmark here

