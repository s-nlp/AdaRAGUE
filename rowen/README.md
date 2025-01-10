
# Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models

Based on [official implementation](https://github.com/dhx20150812/Rowen)

Used [vLLM](https://github.com/vllm-project/vllm) instances for using LLM:

```bash
vllm serve Qwen2.5-72B-Instruct-AWQ --max-model-len 12000 --gpu-memory-utilization 0.65 --dtype=auto

vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32000 --port 8001 --gpu-memory-utilization 0.349 --dtype=auto --enforce_eager --max-num-seqs 2
```

The main entrypoint is `run_standart.py` scipt. `run.sh` used for generating results from the paper.

For using Dashscope or OpenAI API, Proxy URI can be required. It can be used by environment variable `PROXY_URI`. 
For example: `PROXY_URI=socks5://username:password@addr:port python3 run_standart.py`


You can use the following environment variables for usage different models: 

```bash
MAIN_MODEL_NAME # meta-llama/Llama-3.1-8B-Instruct
MAIN_MODEL_BACKEND # vllm or openai
QWEN_BASE_URL # http://localhost:8000/v1
MAIN_MODEL_BASE_URL # http://localhost:8001/v1
PROXY_URI
```