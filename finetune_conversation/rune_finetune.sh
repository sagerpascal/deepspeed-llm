#!/bin/bash
python finetune_llama.py --use_quant 0 --lora_method lora
python finetune_llama.py --use_quant 0 --lora_method adalora
python finetune_llama.py --use_quant 0 --lora_method loha
python finetune_llama.py --use_quant 0 --lora_method lokr
python finetune_llama.py --use_quant 1 --lora_method lora
python finetune_llama.py --use_quant 1 --lora_method adalora
