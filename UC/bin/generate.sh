CUDA_VISIBLE_DEVICES=2 python generate.py \
	--model_path VityaVitalich/Llama3.1-8b-instruct \
	--output_path data/raw/s_nq_context_response \
	--data_path data/datasets/s_nq \
	--cache_dir /home/data/v.moskvoretskii/cache/ \
	--use_context_col retrieved_contexts \
	--prompt_type when_to_retrieve \
	--number_output_seq 1 \
	--critic_col none
