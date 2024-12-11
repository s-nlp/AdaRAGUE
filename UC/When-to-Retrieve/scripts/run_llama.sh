#python run_llm.py --source ./data/nq_sample.jsonl --ra none --type prior_pun_exp --outfile ./examples/nq_llama_pun_exp.jsonl --model llama2

#python collect.py --mode preprocess --source ./data/nq_sample.jsonl --input ./examples/nq_llama_pun_exp.jsonl --output ./examples/nq_llama_pun_exp_new.jsonl --confidence ./examples/nq_llama_pun_exp_confidence.jsonl --answer ./examples/nq_llama_pun_exp_answer.jsonl

#python run_llm.py --source ./examples/nq_llama_pun_exp.jsonl --ra none --type post --outfile ./examples/nq_llama_pun_exp_post_confidence.jsonl --idx ./examples/nq_llama_pun_exp_confidence.jsonl --model llama2

#python run_llm.py --source ./data/nq_sample.jsonl --ra none --type qa --outfile ./examples/nq_llama_pun_exp_post_answer.jsonl --idx ./examples/nq_llama_pun_exp_answer.jsonl --model llama2


#python collect.py --mode evaluate --source ./data/nq_sample.jsonl --input ./examples/nq_llama_pun_exp.jsonl --output ./examples/nq_llama_pun_exp_new.jsonl --confidence ./examples/nq_llama_pun_exp_post_confidence.jsonl --answer ./examples/nq_llama_pun_exp_post_answer.jsonl

#python run_llm.py --source ./data/nq_sample.jsonl --ra dpr --type qa --outfile ./examples/llama_gold_static.jsonl --model llama2

#python collect.py --mode eval_rag --input ./examples/llama_gold_static.jsonl

python collect.py --mode eval_adaptive_rag --input ./examples/llama_gold_static.jsonl --origin ./examples/nq_llama_pun_exp_new.jsonl --source ./data/nq_sample.jsonl --confidence ./examples/nq_llama_pun_exp_post_confidence.jsonl --answer ./examples/nq_llama_pun_exp_post_answer.jsonl
