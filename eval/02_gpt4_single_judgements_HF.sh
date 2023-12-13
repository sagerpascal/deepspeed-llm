#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

cp /deepspeed-llm/data/mt_bench/model_answer/* /FastChat/fastchat/llm_judge/data/mt_bench/model_answer/
python gen_judgment.py --model-list vicuna-7b-v1.5-awq-activate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-no-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-gptq-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-fp4 vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-fp4 vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-fp4 vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-nf4 vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-fp4
python show_result.py --model-list vicuna-7b-v1.5-awq-activate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-no-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-gptq-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-fp4 vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-fp4 vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-fp4 vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-nf4 vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-fp4


# python gen_judgment.py --model-list vicuna-7b-v1.5-awq-activate_4bit-deactivate_nested-float16-nf4
# python show_result.py --model-list vicuna-7b-v1.5-awq-activate_4bit-deactivate_nested-float16-nf4
