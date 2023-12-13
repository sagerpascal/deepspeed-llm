#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for config in /cluster/data/tugg/lora /cluster/data/tugg/adalora /cluster/data/tugg/lokr /cluster/data/tugg/qlora /cluster/data/tugg/qadalora
do
  python eval/gen_model_answers.py --model-id $config
done
