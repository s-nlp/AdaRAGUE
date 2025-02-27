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

If there any problem with building `vllm_uncertainty`:
### Error - `Failed to detect a default CUDA architecture`.
* check `nvcc --version`
    if not: `conda install -c nvidia cudatoolkit=11.7`

* Double-check that when you run `which nvcc` and `which ptxas`, they both point to the correct (and same) installation. If use conda:
    `which nvcc` -> `......../miniconda3/envs/seakr/bin/nvcc`
    `which ptxas` -> `......./miniconda3/envs/seakr/bin/ptxas`
    if not: 
    ```bash
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    ```

* Double check that `pxtas` and `nvcc` has one version.
    `nvcc -V`:
    ```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Thu_Mar_28_02:18:24_PDT_2024
    Cuda compilation tools, release 12.4, V12.4.131
    Build cuda_12.4.r12.4/compiler.34097967_0```

    `pxtas --version`:
    ```bash
    ptxas: NVIDIA (R) Ptx optimizing assembler
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Thu_Mar_28_02:14:54_PDT_2024
    Cuda compilation tools, release 12.4, V12.4.131
    Build cuda_12.4.r12.4/compiler.34097967_0
    ```

* check gcc version `gcc --version`
    if not `conda install conda-forge::gcc`

## Run SeaKR on Multihop QA

For multihop QA datasets Baseline, we use the same files as [dragin](https://github.com/oneal2000/DRAGIN) and [SeaKR](https://github.com/THU-KEG/SeaKR). You can download and unzip it into the `data/multihop_data` folder. SeaKR paper provides a packed multihop data files here: [multihop_data.zip](https://drive.google.com/file/d/1xDqaPa8Kpnb95l7nHpwKWsBQUP9Ck7cn/view?usp=sharing).

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

The original files are from [DPR](https://github.com/facebookresearch/DPR). SeaKR paper provides a packed version containing top 10 retrieved documents [singlehop_data.zip](https://drive.google.com/file/d/1hn4Om_KkIGJpgG2wJjUu1mpPv9oq8M6G/view?usp=sharing). You can download and unzip it into the `data` folder. 

If you want to use it on your own data on single hop -> you have to bring your dataset to the form of such dataset

```bash
python main_simpleqa.py \
    --dataset_name tq \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --selected_intermediate_layer 15 \
    --output_dir $OUTPUT_DIR
```

All results on our `dataset` is availabel in `results` folder.

## Citation
```BibTex
@article{yao2024seakr,
  title={Seakr: Self-aware knowledge retrieval for adaptive retrieval augmented generation},
  author={Yao, Zijun and Qi, Weijian and Pan, Liangming and Cao, Shulin and Hu, Linmei and Liu, Weichuan and Hou, Lei and Li, Juanzi},
  journal={arXiv preprint arXiv:2406.19215},
  year={2024}
}
```
