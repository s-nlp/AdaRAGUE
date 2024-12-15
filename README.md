# This is repository for AdaRAG experiments

This repository contains all code to reproduce `AdaRAG` results and use needed method for your purposes. Repo has a following structure:

```plain
data/ # all train and test dataset
├── adaptive_rag_2wikimultihopqa/
   ├── train.csv # train set
   └── test.csv  # test set
├── adaptive_rag_hotpotqa/
   ├── train.csv # train set
   └── test.csv  # test set
├── adaptive_rag_musique/
   ├── train.csv # train set
   └── test.csv  # test set
├── adaptive_rag_natural_questions/
   ├── train.csv # train set
   └── test.csv  # test set
├── adaptive_rag_squad/
   ├── train.csv # train set
   └── test.csv  # test set
├── adaptive_rag_trivia_qa/
   ├── train.csv # train set
   └── test.csv  # test set

standard_retriver/ # unified retriever for all methods
├── README.md # all info about running retriever
├── build_wiki_index.py # python script to build and run elastic search
└── run.sh # bash script to run retriever

Method/ # all posible method like `SeaKR`, `rowen` etc.
├── requirements.txt or pyproject.toml # with all needed requirements for method
└── README.md # with all info about how to run current method
```

To use any method and reproduce the results you need:
* Run the retriever
* Use one of the methods below

# Retriever
Followed by [dragin](https://github.com/oneal2000/DRAGIN) and [SeaKR](https://github.com/THU-KEG/SeaKR). Use the Wikipedia dump and elastic search to build the retriever
All info about how to run a retriever is available in [README.md](./standard_retriever/README.md).

You can use your own retriever.

# Datasets

We used standard datasets (Natural Questions, HotpotQA, 2wikimultihopqa, Squad, Trivia_Q, Musique) for evaluation with 500 sample from every [dataset](./data/)

# Methods

## Adaptive RAG and IRCoT

All code with our evaluator and dependencpy installation is available in Adaptive RAG folder. We slightly changed the original Adaptive repository. For running experiments with `Adaptive` and `IRCoT` use a code from [README.md](./Adaptive_Rag/README.md)

## FLARE and DRAGIN

All code with our evaluator and dependencpy installation is available in `DRAGIN` folder. We copied the original DRAGIN repository. For running experiments with FLARE and DRAGIN use a code from DRAGIN [README.md](./dragin/README.md)

## Rowen

Install all required dependences from `rowen/pyproject.toml` and use `run.sh` to reproduce results. More details in [Rowen README.md](./rowen/README.md)

## SeaKR

All code with our evaluator and dependencpy installation is available in `SeaKR` folder. We copied the original SeaKR repository and made some in `vllm_uncertainty`. For running experiments use a code from [SeaKR README.md](./SeaKR/README.md)

## Uncertainty Estimation Methods

Install all required dependences from `UC/requirements.txt` and use scripts from `bin/*.sh` to reproduce results. More details in [UC README.md](./UC/README.md)
