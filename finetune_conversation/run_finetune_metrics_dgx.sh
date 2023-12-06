#!/bin/bash
python finetune_llama.py --use_quant 0 --lora_method lora --wandb_project dgx_metrics --wandb_run lora
python finetune_llama.py --use_quant 0 --lora_method adalora --wandb_project dgx_metrics --wandb_run adalora
python finetune_llama.py --use_quant 0 --lora_method loha --wandb_project dgx_metrics --wandb_run loha
python finetune_llama.py --use_quant 0 --lora_method lokr --wandb_project dgx_metrics --wandb_run lokr
python finetune_llama.py --use_quant 1 --lora_method lora --wandb_project dgx_metrics --wandb_run qlora
python finetune_llama.py --use_quant 1 --lora_method adalora --wandb_project dgx_metrics --wandb_run qadalora
