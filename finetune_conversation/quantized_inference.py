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
#from awq import AutoAWQForCausalLM


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
        # Llama-2-7b-<quant>-<4bit>-<nested>-<dtype>-<quant_type>


        model_config = model_id.split("-")
        model_type = "-".join(model_config[:3])
        quant_method = model_config[3]
        act_4bit = model_config[4]
        act_quant = model_config[5]
        compute_dtype = model_config[6]
        quant_type = model_config[7]

        assert model_type.lower() in ["llama-2-7b", "vicuna-7b-v1.5"], "Only vicuna-7b or llama-2-7b models are supported"
        if model_type.lower() == "llama-2-7b":
            assert quant_method == "no", "Quantization method is not supported for llama models"

        else:
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
        model_type = ""
        parser = HfArgumentParser(ScriptArguments)
        args = parser.parse_args_into_dataclasses()[0]

    if args.quant_method == 'no':
        if model_type.lower() == "llama-2-7b":
            model_name = "meta-llama/Llama-2-7b-hf"

        else:
            model_name = "meta-llama/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        device_map = {"": 0}

        if not load_only_tokenizer:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
            )

    elif args.quant_method == 'bnb':
        model_name = "meta-llama/Llama-2-7b-chat-hf"
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
        model_name_or_path = "TheBloke/Llama-2-7B-Chat-AWQ"

        if not load_only_tokenizer:
            model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                                      trust_remote_code=False, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

    elif args.quant_method == 'gptq':
        # TODO use code snippets from hugginface hub
        ### gptq
        ### https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ
        model_name_or_path = "TheBloke/Llama-2-7B-Chat-GPTQ"
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

    # qual_eval_texts = ["### Human: Can you recommend me a recipe for chocolate chip cookies?### Assistant:",
    #                    "### Human: What is photosynthesis?### Assistant:",
    #                    "### Human: Where can we get a good lunch in Zürich on a small budget?### Assistant:",
    #                    "### Human: Explain datascience to me.### Assistant:",
    #                    "### Human: Since when is Paris the capital of Germany?### Assistant:",
    #                    "### Human: Can you add 9123 to 1234?### Assistant:",
    #                    "### Human: When was the French Revolution?### Assistant:"
    #                    ]

    qual_eval_texts = ["Can you recommend me a recipe for chocolate chip cookies?",
                       "What is photosynthesis?",
                       "Where can we get a good lunch in Zürich on a small budget?",
                       "Explain datascience to me.",
                       "Since when is Paris the capital of Germany?",
                       "Can you add 9123 to 1234?",
                       "When was the French Revolution?"
                       ]
    for eval_text in qual_eval_texts:
        # TODO inference benchmark here

        eval_text = "[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct."+ \
        " If you don't know the answer to a question, please don't share false information. \n <</SYS>> "+eval_text+"[/INST]"
        #eval_text = "A chat between a human and an assistant. \n ### Human: \n "+eval_text + ""
        text = (eval_text)
        inputs = tokenizer(text, return_tensors="pt").to(0)

        out = model.generate(**inputs, max_new_tokens=250)
        print(tokenizer.decode(out[0], skip_special_tokens=True))
