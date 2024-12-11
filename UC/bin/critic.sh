CUDA_VISIBLE_DEVICES=2 python generate.py \
	--model_path VityaVitalich/Llama3-8b-instruct \
	--output_path data/raw/nq_no_context_response_critic.pickle \
    	--data_path data/datasets/nq \
    	--cache_dir /home/data/v.moskvoretskii/cache/ \
	--use_context_col none \
	--answer_col no_context_response \
	--prompt_type critic
