#!/bin/bash
python finetune_llama.py --use_quant 0 --lora_method lora --wandb_project dgx_metrics --wandb_run lora --merge_and_push 1
python finetune_llama.py --use_quant 0 --lora_method adalora --wandb_project dgx_metrics --wandb_run adalora --merge_and_push 1
python finetune_llama.py --use_quant 0 --lora_method loha --wandb_project dgx_metrics --wandb_run loha --merge_and_push 1
python finetune_llama.py --use_quant 0 --lora_method lokr --wandb_project dgx_metrics --wandb_run lokr --merge_and_push 1
python finetune_llama.py --use_quant 1 --lora_method lora --wandb_project dgx_metrics --wandb_run qlora --merge_and_push 1
python finetune_llama.py --use_quant 1 --lora_method adalora --wandb_project dgx_metrics --wandb_run qadalora --merge_and_push 1
