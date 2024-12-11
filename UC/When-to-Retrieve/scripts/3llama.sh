#!/bin/bash

# Define the directory path
DIR="./examples/llama3_s_musique_dense/"
DATA_FILE="./data/s_musique.jsonl"
mkdir -p $DIR


python run_llm.py --source $DATA_FILE --ra none --type prior_pun_exp --outfile "${DIR}test.jsonl" --model llama3

python collect.py --mode preprocess --source $DATA_FILE --input "${DIR}test.jsonl" --output "${DIR}test_new.jsonl" --confidence "${DIR}confidence.jsonl" --answer "${DIR}answer.jsonl"

python run_llm.py --source "${DIR}test.jsonl" --ra none --type post_punish --outfile "${DIR}post_confidence.jsonl" --idx "${DIR}confidence.jsonl" --model llama3

python run_llm.py --source $DATA_FILE --ra none --type qa --outfile "${DIR}post_answer.jsonl" --idx "${DIR}answer.jsonl" --model llama3


python collect.py --mode evaluate --source $DATA_FILE --input "${DIR}test.jsonl" --output "${DIR}test_new.jsonl" --confidence "${DIR}post_confidence.jsonl" --answer "${DIR}post_answer.jsonl"

python run_llm.py --source $DATA_FILE --ra dense --type qa --outfile "${DIR}gold_static.jsonl" --model llama3

python collect.py --mode eval_rag --input "${DIR}gold_static.jsonl"

python collect.py --mode eval_adaptive_rag --input "${DIR}gold_static.jsonl" --origin "${DIR}test_new.jsonl" --source $DATA_FILE --confidence "${DIR}post_confidence.jsonl" --answer "${DIR}post_answer.jsonl"
