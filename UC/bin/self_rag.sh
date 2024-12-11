CUDA_VISIBLE_DEVICES=1 python self_rag.py \
	--output_path data/raw/popqa_self_rag_no_context.pickle \
    	--data_path data/datasets/popqa_longtail \
    	--cache_dir /home/data/v.moskvoretskii/cache/ \
	--context_col_name none
