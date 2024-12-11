for mode in Hybrid CM CL
do
    python3 run_standart.py --dataset_name natural_questions --mode $mode
    python3 run_standart.py --dataset_name trivia_qa --mode $mode
    python3 run_standart.py --dataset_name squad --mode $mode
    python3 run_standart.py --dataset_name 2wiki_multihop_qa --mode $mode
    python3 run_standart.py --dataset_name hotpot_qa --mode $mode
    python3 run_standart.py --dataset_name musique --mode $mode
done

# vllm serve Qwen2.5-72B-Instruct-AWQ --max-model-len 12000 --gpu-memory-utilization 0.65 --dtype=auto
# vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32000 --port 8001 --gpu-memory-utilization 0.349 --dtype=auto --enforce_eager --max-num-seqs 2
