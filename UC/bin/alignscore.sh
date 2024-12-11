CUDA_VISIBLE_DEVICES=1 python alignscore.py \
    --data_path data/datasets/nq \
    --pred_column no_context_response \
    --gt_column ground_truth \
    --batch_size 64 \
    --eval_mode nli