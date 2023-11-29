# Evaluation using LLM Judge

## Installation


1. Create a folder for evaluation

```bash
mkdir eval
cd eval
```

2. Clone the Github Repository into this folder

```bash
git clone https://github.com/lm-sys/FastChat.git
```

3. Copy the `Dockerfile` into this folder

4. Build the Docker Image

```bash
docker build -t fastchat .
```

5. Run the Docker Image

```bash
nvidia-docker run -it --shm-size=16g -v eval/FastChat:/FastChat fastchat
```

## Evaluation

In the Docker Image, run the following command to evaluate a registered model:

```bash
cd fastchat/llm-judge
python download_mt_bench_pregenerated.py
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
```

If the model is not registered, you can either register it or create a `<model-id>.jsonl` file with the model responses
to the tasks in the `MT-Bench` dataset following the format:

```json
{"question_id": 1, "answer_id": "xxx", "model_id": "xxx", "choices": [{"index": 0, "turns": ["...", "..."]}], "tstamp": 1.1}
```

##### Generate GPT-4 judgments

```bash
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
python show_result.py --model-list [LIST-OF-MODEL-ID]
```
