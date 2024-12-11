CUDA_VISIBLE_DEVICES=1 python eval.py \
	--model_path VityaVitalich/Llama3-8b-instruct \
	--output_path data/raw/nq_self_eval_rel_context.pickle \
    	--data_path data/datasets/nq \
	--predict_col rel_context_response \
	--gt_col ground_truth \
	--cache_dir /home/data/v.moskvoretskii/cache/
