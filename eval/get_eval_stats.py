import warnings
from pathlib import Path

import pandas as pd
import wandb

from eval.gen_model_answers import load_model

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
    'tugg-merged_final_checkpoints_lora_0': 'sagerpascal/mt-bench/lo350jdw',
    'tugg-merged_final_checkpoints_adalora_0': 'sagerpascal/mt-bench/qp74qd91',
    'tugg-merged_final_checkpoints_adalora_1': 'sagerpascal/mt-bench/34t0wrzg',
    'tugg-merged_final_checkpoints_lokr_0': 'sagerpascal/mt-bench/5zwcs4om',
    'tugg-merged_final_checkpoints_lora_1': 'sagerpascal/mt-bench/khnf5usr',
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


def print_statistics(model_id, stats):
    print(f"Statistics for {model_id}:")
    for k, v in stats.items():
        print(f"\t{k: <40}: {v: <15}")
    print()

def store_statistics(results):
    df = pd.DataFrame.from_dict(results)
    df.to_csv("results/eval_stats.csv", index=False)


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
    results = {}
    for run_id, run_path in runs.items():
        run, answers = import_wandb_data(run_path, run_id)
        tokenizer = load_model(run_to_model_id(run_id), load_only_tokenizer=True)
        system_metrics = extract_relevant_columns(run.history(stream="events"))
        stats = calculate_statistics(system_metrics, answers, tokenizer)
        print_statistics(run_id, stats)
        results = add_statistics_to_dict(results, run_id, stats)

    store_statistics(results)


    warnings.warn(
        "This metrics are simplified. For example, we calculate tokens/s by dividing the number of tokens through "
        "the runtime and ignore the fact that during a run also the model is loaded, tokens are fed into the model"
        " etc. However, we think that this is a good approximation, as loading the model takes about ~20s and "
        "evaluation between 30-60min.")


if __name__ == '__main__':
    # run = wandb.init(project="mt-bench", group="upload-results")
    # for fname in ["tugg-merged_final_checkpoints_adalora_0",
    #               "tugg-merged_final_checkpoints_adalora_1",
    #               "tugg-merged_final_checkpoints_lokr_0",
    #               "tugg-merged_final_checkpoints_lora_0",
    #               "tugg-merged_final_checkpoints_lora_1"]:
    #     artifact = wandb.Artifact(name=fname + "_answers", type="jsonl")
    #     artifact.add_file(local_path=f"/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/{fname}" + ".jsonl")
    #     run.log_artifact(artifact)
    # run.finish()

    main()
