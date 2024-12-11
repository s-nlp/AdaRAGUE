CUDA_VISIBLE_DEVICES=2 python uncertainty.py \
	--model_path VityaVitalich/Llama3.1-8b-instruct \
    --cache_dir /home/data/v.moskvoretskii/cache/ \
    --data_path data/datasets/s_nq \
    --question_column question_text \
    --context_column none \
    --output_column none \
    --batch_size 32 \
