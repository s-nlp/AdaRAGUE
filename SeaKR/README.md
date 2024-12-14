# SeaKR


## Install Environment

```bash
conda create -n seakr python=3.10
conda activate seakr
pip install beir==1.0.1 spacy==3.7.2 aiofiles tenacity
python -m spacy download en_core_web_sm
```

SeaKR modified the vllm to get the uncertainty measures.
```bash
cd vllm_uncertainty
pip install -e .
```

## Run SeaKR on Multihop QA

For multihop QA datasets, we use the same files as [dragin](https://github.com/oneal2000/DRAGIN). You can download and unzip it into the `data/multihop_data` folder. We provide a packed multihop data files here: [multihop_data.zip](https://drive.google.com/file/d/1xDqaPa8Kpnb95l7nHpwKWsBQUP9Ck7cn/view?usp=sharing).
We use an asynchronous reasoning engine to accelerate multi hop reasoning.

### 2WikiHop
```bash
python main_multihop.py \
    --n_shot 10 \
    --retriever_port $YOUR_ELASTIC_SERVICE_PORT \
    --dataset_name twowikihop \
    --eigen_threshold -6.0 \
    --save_dir "outputs/twowikihop" \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --served_model_name llama2-7b-chat \
    --max_reasoning_steps 7 \
    --max_docs 5
```
If you want to use it on your own data -> you have to bring your dataset to the form of such dataset

### HotpotQA
```bash
python main_multihop.py \
    --n_shot 10 \
    --retriever_port $YOUR_ELASTIC_SERVICE_PORT \
    --dataset_name hotpotqa \
    --eigen_threshold -6.0 \
    --save_dir "outputs/hotpotqa" \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --served_model_name llama2-7b-chat \
    --max_reasoning_steps 7 \
    --max_docs 5
```
If you want to use it on your own data -> you have to bring your dataset to the form of such dataset

### IIRC
```bash
python main_multihop.py \
    --n_shot 10 \
    --retriever_port $YOUR_ELASTIC_SERVICE_PORT \
    --dataset_name iirc \
    --eigen_threshold -6.0 \
    --save_dir "outputs/iirc" \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --served_model_name llama2-7b-chat \
    --max_reasoning_steps 7 \
    --max_docs 5
```
If you want to use it on your own data -> you have to bring your dataset to the form of such dataset


### Evaluation
For single and multi hop evaluation you can use code from `eval_multihop.ipynb` and `eval_singlehop.ipynb`
For bootstrap evaluation on single task
```bash
python bootstrap_CI_upd.py --input_file "path_to_jsonl"\
                            --pred_col "predict"\
                            --gt_col "answer"\
                            --n_rounds 1000
```
For bootstrap evaluation multihop task
```bash
python bootstrap_multihop.py --input_file "path_to_jsonl"\
                            --pred_cols "Final Answer" "Final Step Answer" "Final Read Answer"\
                            --gt_col "ground_truth"\
                            --n_rounds 1000
```


## Run SeaKR on Single QA

The original files are from [DPR](https://github.com/facebookresearch/DPR). We provide a packed version containing top 10 retrieved documents [singlehop_data.zip](https://drive.google.com/file/d/1hn4Om_KkIGJpgG2wJjUu1mpPv9oq8M6G/view?usp=sharing). You can download and unzip it into the `data` folder. 

If you want to use it on your own data on single hop -> you have to bring your dataset to the form of such dataset

```bash
python main_simpleqa.py \
    --dataset_name tq \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --selected_intermediate_layer 15 \
    --output_dir $OUTPUT_DIR
```

All results on our dataset is availabel in `results` folder.