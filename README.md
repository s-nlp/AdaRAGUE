# This is repository for experiments for AdaRAG experiments

# Retriever
Followed by [dragin](https://github.com/oneal2000/DRAGIN) and [SeaKR](https://github.com/THU-KEG/SeaKR). Use the Wikipedia dump and elastic search to build the retriever

#### Download Wikipedia dump

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

#### Run Elasticsearch service

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
```

#### build the index

```bash
python build_wiki_index.py --data_path $YOUR_WIKIPEDIA_TSV_PATH --index_name wiki --port $YOUR_ELASTIC_SERVICE_PORT
```


# Experiments

## IRCoT

## Adaptive RAG

## FLARE

## DRAGIN

## Rowen

Install all required dependences from `rowen/pyproject.toml` and use `run.sh` to reproduce results. More details in [Rowen README.md](./rowen/README.md)

## SeaKR
### Running experiments
All code with our evaluator is available in SeaKR folder. We copied the original SeaKR repository and made some in `vllm_uncertainty`. For running experiments use a code from README in `SeaKR` folder.

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

# Uncertainty Estimation Methods




