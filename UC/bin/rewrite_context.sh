CUDA_VISIBLE_DEVICES=0 python rewrite_context.py \
	--model_path VityaVitalich/Llama3-8b-instruct \
	--output_path ./data/raw/nq_rewritten_context.pickle \
    	--data_path ./data/datasets/nq \
    	--cache_dir /home/data/v.moskvoretskii/cache/ \
    	--batch_size 8 \
        --document_col document_relevant;
