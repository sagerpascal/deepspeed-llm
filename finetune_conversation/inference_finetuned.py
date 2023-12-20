import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer
)



os.environ["WANDB_PROJECT"] = "sds_llama_infrence"

def load_model(model_id: str=None, load_only_tokenizer: bool=False):

    @dataclass
    class ScriptArguments:
        """
        These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
        """

        lora_method: Optional[str] = field(
            default="lora",
            metadata={"help": "Choose which low-rank finetuning method: lora, adalora, lokr, qlora, qadalora"},
        )


    if model_id is not None:
        assert model_id[:28] == '/cluster/data/tugg/finalized', "Model id is expected to point to a stored model in /cluster/data/tugg"
        model_path = model_id

    else:

        parser = HfArgumentParser(ScriptArguments)
        args = parser.parse_args_into_dataclasses()[0]

        prefix = '/cluster/data/tugg/finalized'
        if args.lora_method == 'lora':
            model_path = os.path.join(prefix, 'merged_final_checkpoints_lora_0')
        elif args.lora_method == 'adalora':
            model_path = os.path.join(prefix, 'merged_final_checkpoints_adalora_0')
        elif args.lora_method == 'lokr':
            model_path = os.path.join(prefix, 'merged_final_checkpoints_lokr_0')
        elif args.lora_method == 'qlora':
            model_path = os.path.join(prefix, 'merged_final_checkpoints_lora_1')
        elif args.lora_method == 'qadalora':
            model_path = os.path.join(prefix, 'merged_final_checkpoints_adalora_1')
        else:
            raise ("unknown lora method")


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": 0}

    if not load_only_tokenizer:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
        )



    if load_only_tokenizer:
        return tokenizer

    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = load_model()

    qual_eval_texts = ["### Human: Can you recommend me a recipe for chocolate chip cookies?### Assistant:",
                       "### Human: What is photosynthesis?### Assistant:",
                       "### Human: Where can we get a good lunch in Zürich on a small budget?### Assistant:",
                       "### Human: Explain datascience to me.### Assistant:",
                       "### Human: Since when is Paris the capital of Germany?### Assistant:",
                       "### Human: Can you add 9123 to 1234?### Assistant:",
                       "### Human: When was the French Revolution?### Assistant:"
                       ]

    # TODO inference benchmark here
    text = ("### Human: Can you explain to me why it makes sense to optain a PhD?")
    inputs = tokenizer(text, return_tensors="pt").to(0)

    out = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
