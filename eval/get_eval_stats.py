import warnings
from pathlib import Path

import pandas as pd
import wandb

from finetune_conversation.quantized_inference import load_model

runs = {
    'vicuna-7b-v1.5-awq-activate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/7id0iq73',
    'vicuna-7b-v1.5-no-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/nb7hi44o',
    'vicuna-7b-v1.5-gptq-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/pbbmonlf',
    'vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/6ixam464',
    'vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-fp4': 'sagerpascal/mt-bench/rumn5zu0',
    'vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-nf4': 'sagerpascal/mt-bench/f7jb2247',
    'vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-fp4': 'sagerpascal/mt-bench/exw4ffah',
    'vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/192wafhr',
    'vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-fp4': 'sagerpascal/mt-bench/mc421tmk',
    'vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-nf4': 'sagerpascal/mt-bench/6b889cu1',
    'vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-fp4': 'sagerpascal/mt-bench/lrfm1cp5',
}


def import_wandb_data(run_path, model_id):
    api = wandb.Api()
    run = api.run(run_path)
    answers = api.artifact(f"{run_path.split('/')[0]}/{run_path.split('/')[1]}/{model_id}_answers:latest").download(
        root="tmp")
    answers = pd.read_json(path_or_buf=Path(answers) / (model_id + ".jsonl"), lines=True)
    return run, answers


def extract_relevant_columns(df: pd.DataFrame):
    df = df[['_runtime', '_timestamp', 'system.gpu.0.powerWatts', 'system.gpu.0.memory', 'system.gpu.0.memoryAllocated',
             'system.gpu.0.gpu']]
    return df


def get_processed_tokens_per_s(df: pd.DataFrame, answers, tokenizer):
    responses = [dict(a[1][0]) for a in answers['choices'].items()]
    responses = [(r['turns'][0], r['turns'][1]) for r in responses]
    n_gen_tokens = [len(tokenizer(r[0])['input_ids']) + len(tokenizer(r[1])['input_ids']) for r in responses]
    return sum(n_gen_tokens) / df['_runtime'].max()


def calculate_statistics(df: pd.DataFrame, answers, tokenizer):
    runtime = df['_runtime'].max()
    avg_gpu_memory = df['system.gpu.0.memory'].mean()
    max_gpu_memory = df['system.gpu.0.memory'].max()
    avg_gpu_memory_allocated = df['system.gpu.0.memoryAllocated'].mean()
    max_gpu_memory_allocated = df['system.gpu.0.memoryAllocated'].max()
    gpu_power_consumption_Wh = df['system.gpu.0.powerWatts'].mean() * runtime / 3600

    return {
        'Runtime': runtime,
        'Avg. GPU Memory [%]': avg_gpu_memory,
        'Max. GPU Memory [%]': max_gpu_memory,
        'Avg. GPU Memory Allocated [%]': avg_gpu_memory_allocated,
        'Max. GPU Memory Allocated [%]': max_gpu_memory_allocated,
        'GPU Power Consumption [Wh]': gpu_power_consumption_Wh,
        'Tokens/s': get_processed_tokens_per_s(df, answers, tokenizer)
    }

def print_statistics(model_id, stats):
    print(f"Statistics for {model_id}:")
    for k, v in stats.items():
        print(f"\t{k: <40}: {v: <15}")
    print()

def main():
    for model_id, run_path in runs.items():
        run, answers = import_wandb_data(run_path, model_id)
        tokenizer = load_model(model_id, load_only_tokenizer=True)
        system_metrics = extract_relevant_columns(run.history(stream="events"))
        stats = calculate_statistics(system_metrics, answers, tokenizer)
        print_statistics(model_id, stats)

    warnings.warn(
        "This metrics are simplified. For example, we calculate tokens/s by dividing the number of tokens through "
        "the runtime and ignore the fact that during a run also the model is loaded, tokens are fed into the model"
        " etc.")


if __name__ == '__main__':
    main()
