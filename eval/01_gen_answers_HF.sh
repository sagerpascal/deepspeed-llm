#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for config in vicuna-7b-v1.5-awq-activate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-no-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-gptq-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-fp4 vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-nf4 vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-fp4 vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-nf4 vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-fp4 vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-nf4 vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-fp4
do
  python eval/gen_model_answers.py --model-id $config
done
