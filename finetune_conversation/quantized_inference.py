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

def load_model(model_id: str=None, load_only_tokenizer: bool=False):

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

    if model_id is not None:
        # Model id is expected to follow this pattern:
        # vicuna-7b-v1.5-<quant>-<4bit>-<nested>-<dtype>-<quant_type>

        model_config = model_id.split("-")
        model_type = model_config[0]
        model_size = model_config[1]
        model_version = model_config[2]
        quant_method = model_config[3]
        act_4bit = model_config[4]
        act_quant = model_config[5]
        compute_dtype = model_config[6]
        quant_type = model_config[7]

        assert model_type == "vicuna", "Only vicuna models are supported"
        assert model_size in ["7b"], "Only 7b models are supported"
        assert model_version in ["v1.5"], "Only v1.5 models are supported"
        assert quant_method in ["no", "bnb", "awq", "gptq"], "Only no, bnb, awq, gptq quantization methods are supported"
        assert act_4bit in ["activate_4bit", "deactivate_4bit"], "Only activate_4bit and deactivate_4bit are supported"
        assert act_quant in ["activate_nested", "deactivate_nested"], "Only activate_nested and deactivate_nested are supported"
        assert compute_dtype in ["float16", "bfloat16"], "Only float16 and bfloat16 are supported"
        assert quant_type in ["nf4", "fp4"], "Only nf4 and fp4 are supported"

        args = ScriptArguments(
            quant_method=quant_method,
            bnb_use_4bit=act_4bit == "activate_4bit",
            bnb_use_nested_quant=act_quant == "activate_nested",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
        )

    else:

        parser = HfArgumentParser(ScriptArguments)
        args = parser.parse_args_into_dataclasses()[0]

    if args.quant_method == 'no':
        model_name = "lmsys/vicuna-7b-v1.3"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        device_map = {"": 0}

        if not load_only_tokenizer:
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

        if not load_only_tokenizer:
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

        if not load_only_tokenizer:
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

        if not load_only_tokenizer:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                         device_map="auto",
                                                         trust_remote_code=True,
                                                         revision="main")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    else:
        raise("unknown quantization method")

    if load_only_tokenizer:
        return tokenizer

    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = load_model()

    # TODO inference benchmark here
    text = ("Why is it good to obtain a PhD?")
    inputs = tokenizer(text, return_tensors="pt").to(0)

    out = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
