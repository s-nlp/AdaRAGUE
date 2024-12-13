# To run retriever

1) pip install beir==1.0.1

2) chmod 777 ./run_elastic.sh
* If Restricted use my archive from google drive
* Or uncomment `wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz`

3) run python preparation script from SeaKR (https://github.com/THU-KEG/SeaKR?tab=readme-ov-file)
* `python build_wiki_index.py --data_path $YOUR_WIKIPEDIA_TSV_PATH --index_name wiki --port $YOUR_ELASTIC_SERVICE_PORT`

* use 9200 as `YOUR_ELASTIC_SERVICE_PORT`
* use `data/dpr/psgs_w100.tsv` as `YOUR_ELASTIC_SERVICE_PORT`

`python build_wiki_index.py --data_path ./data/dpr/psgs_w100.tsv --index_name wiki --port 9200`