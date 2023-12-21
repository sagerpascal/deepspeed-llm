import warnings
from pathlib import Path

import pandas as pd
import wandb

from eval.gen_model_answers import load_model


runs_a100 = {
    'Llama-2-7b-no-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/01v5xef5',
    'Llama-2-7b-chat-no-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/73qn3kyp',
    'Llama-2-7b-chat-bnb-activate_4bit-deactivate_nested-float16-fp4': 'sagerpascal/mt-bench/w0xb5gcj',
    'Llama-2-7b-chat-bnb-activate_4bit-activate_nested-float16-fp4': 'sagerpascal/mt-bench/kfrndzxy',
    'Llama-2-7b-chat-awq-activate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/csl8rgnj',
    'Llama-2-7b-chat-gptq-activate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/xrxcbahk',
    # 'vicuna-7b-v1.5-awq-activate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/7id0iq73',
    # 'vicuna-7b-v1.5-no-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/nb7hi44o',
    # 'vicuna-7b-v1.5-gptq-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/pbbmonlf',
    # 'vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/6ixam464',
    # 'vicuna-7b-v1.5-bnb-deactivate_4bit-deactivate_nested-float16-fp4': 'sagerpascal/mt-bench/rumn5zu0',
    # 'vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-nf4': 'sagerpascal/mt-bench/f7jb2247',
    # 'vicuna-7b-v1.5-bnb-deactivate_4bit-activate_nested-float16-fp4': 'sagerpascal/mt-bench/exw4ffah',
    # 'vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-nf4': 'sagerpascal/mt-bench/192wafhr',
    # 'vicuna-7b-v1.5-bnb-activate_4bit-deactivate_nested-float16-fp4': 'sagerpascal/mt-bench/mc421tmk',
    # 'vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-nf4': 'sagerpascal/mt-bench/6b889cu1',
    # 'vicuna-7b-v1.5-bnb-activate_4bit-activate_nested-float16-fp4': 'sagerpascal/mt-bench/lrfm1cp5',
    'tugg-merged_final_checkpoints_lora_0': 'sagerpascal/mt-bench/lo350jdw',
    'tugg-merged_final_checkpoints_adalora_0': 'sagerpascal/mt-bench/qp74qd91',
    'tugg-merged_final_checkpoints_adalora_1': 'sagerpascal/mt-bench/34t0wrzg',
    'tugg-merged_final_checkpoints_lokr_0': 'sagerpascal/mt-bench/5zwcs4om',
    'tugg-merged_final_checkpoints_lora_1': 'sagerpascal/mt-bench/khnf5usr',
}

runs_t4 = {

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
             'system.gpu.0.gpu', 'system.gpu.0.powerPercent']]
    return df


def get_processed_tokens_per_s(df: pd.DataFrame, answers, tokenizer):
    processing_time_df = df[df['system.gpu.0.powerPercent'] > 5]
    processing_time = processing_time_df['_runtime'].max() - processing_time_df['_runtime'].min()

    responses = [dict(a[1][0]) for a in answers['choices'].items()]
    responses = [(r['turns'][0], r['turns'][1]) for r in responses]
    n_gen_tokens = [len(tokenizer(r[0])['input_ids']) + len(tokenizer(r[1])['input_ids']) for r in responses]
    return sum(n_gen_tokens) / processing_time


def calculate_statistics(df: pd.DataFrame, answers, tokenizer, gpu_memory_gb):
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
        'Avg. GPU Memory [GB]': avg_gpu_memory / 100 * gpu_memory_gb,
        'Max. GPU Memory [GB]': max_gpu_memory / 100 * gpu_memory_gb,
        'Avg. GPU Memory Allocated [GB]': avg_gpu_memory_allocated / 100 * gpu_memory_gb,
        'Max. GPU Memory Allocated [GB]': max_gpu_memory_allocated / 100 * gpu_memory_gb,
        'GPU Power Consumption [Wh]': gpu_power_consumption_Wh,
        'Tokens/s': get_processed_tokens_per_s(df, answers, tokenizer)
    }


def print_statistics(model_id, stats):
    print(f"Statistics for {model_id}:")
    for k, v in stats.items():
        print(f"\t{k: <40}: {v: <15}")
    print()


def print_statistics(model_id, stats):
    print(f"Statistics for {model_id}:")
    for k, v in stats.items():
        print(f"\t{k: <40}: {v: <15}")
    print()

def store_statistics(results, filepath):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(filepath, index=False)


def add_statistics_to_dict(dict_, model_id, stats):
    if 'model' not in dict_:
        dict_['model'] = []
    dict_['model'].append(model_id)
    for k, v in stats.items():
        if k not in dict_:
            dict_[k] = []
        dict_[k].append(v)
    return dict_


def run_to_model_id(run_id):
    if run_id.startswith("tugg-"):
        prefix = '/cluster/data/tugg/finalized'
        return prefix + "/" + run_id[len("tugg-"):]
    else:
        return run_id


def main():
    for runs, gpu_memory, filepath in [(runs_a100, 40, "results/eval_stats_a100.csv"), (runs_t4, 12, "results/eval_stats_t4.csv")]:
        results = {}
        for run_id, run_path in runs.items():
            run, answers = import_wandb_data(run_path, run_id)
            tokenizer = load_model(run_to_model_id(run_id), load_only_tokenizer=True)
            system_metrics = extract_relevant_columns(run.history(stream="events"))
            stats = calculate_statistics(system_metrics, answers, tokenizer, gpu_memory)
            print_statistics(run_id, stats)
            results = add_statistics_to_dict(results, run_id, stats)

        store_statistics(results, filepath=filepath)

if __name__ == '__main__':
    main()
