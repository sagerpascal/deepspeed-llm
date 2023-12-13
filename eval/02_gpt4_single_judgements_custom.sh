#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

cp /deepspeed-llm/data/mt_bench/model_answer/* /FastChat/fastchat/llm_judge/data/mt_bench/model_answer/
python gen_judgment.py --model-list /cluster/data/tugg/lora /cluster/data/tugg/adalora /cluster/data/tugg/lokr /cluster/data/tugg/qlora /cluster/data/tugg/qadalora
python show_result.py --model-list /cluster/data/tugg/lora /cluster/data/tugg/adalora /cluster/data/tugg/lokr /cluster/data/tugg/qlora /cluster/data/tugg/qadalora

