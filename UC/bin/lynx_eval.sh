CUDA_VISIBLE_DEVICES=0 python lynx_eval.py \
     	--output_path data/raw/lynx_rewritten_context.pickle \
     	--data_path data/datasets/tqa_with_rewritten_response \
    	--cache_dir /home/data/v.moskvoretskii/cache/ \
    	--batch_size 2 \
	--use_context generated;
