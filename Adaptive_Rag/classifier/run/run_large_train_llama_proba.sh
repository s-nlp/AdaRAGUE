DATE=2024_11_28/14_25_04
MODEL=t5-large
LLM_NAME=llama_8b_it
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
GPU=0

for EPOCH in 25
do  
    TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/epoch/${EPOCH}/${DATE}

    # predict
    PREDICT_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/epoch/${EPOCH}/${DATE}/predict
    mkdir -p ${PREDICT_OUTPUT_DIR}
    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
        --question_column question \
        --answer_column answer \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${PREDICT_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval
done