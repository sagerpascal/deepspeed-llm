#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

cp /deepspeed-llm/data/mt_bench/model_answer/* /FastChat/fastchat/llm_judge/data/mt_bench/model_answer/
python gen_judgment.py --model-list /cluster/data/tugg/finalized/merged_final_checkpoints_lora_0 /cluster/data/tugg/finalized/merged_final_checkpoints_adalora_0 /cluster/data/tugg/finalized/merged_final_checkpoints_lokr_0 /cluster/data/tugg/finalized/merged_final_checkpoints_lora_1 /cluster/data/tugg/finalized/merged_final_checkpoints_adalora_1
python show_result.py --model-list /cluster/data/tugg/finalized/merged_final_checkpoints_lora_0 /cluster/data/tugg/finalized/merged_final_checkpoints_adalora_0 /cluster/data/tugg/finalized/merged_final_checkpoints_lokr_0 /cluster/data/tugg/finalized/merged_final_checkpoints_lora_1 /cluster/data/tugg/finalized/merged_final_checkpoints_adalora_1

