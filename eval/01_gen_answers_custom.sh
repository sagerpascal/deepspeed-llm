#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for config in /cluster/data/tugg/finalized/merged_final_checkpoints_lora_0 /cluster/data/tugg/finalized/merged_final_checkpoints_adalora_0 /cluster/data/tugg/finalized/merged_final_checkpoints_lokr_0 /cluster/data/tugg/finalized/merged_final_checkpoints_lora_1 /cluster/data/tugg/finalized/merged_final_checkpoints_adalora_1
do
  python eval/gen_model_answers.py --model-id $config
done
