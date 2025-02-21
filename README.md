# AdaRAG Uncertanity Estimation
<p align="center">
   <img alt="GitHub License" src="https://img.shields.io/github/license/s-nlp/AdaRAGUE">
   <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/s-nlp/AdaRAGUE">
   <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/s-nlp/AdaRAGUE">
</p>

<p align="center">
📃 <a href="https://arxiv.org/abs/2501.12835" target="_self">Paper</a> • 🤗 <a href="/data/" target="_self">Dataset</a>  
</p>

This repository contains all code to reproduce [Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](https://arxiv.org/abs/2501.12835) results and use needed method for your purposes. Repo has a following structure:

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

All code, including our evaluator and dependency installation, is available in the `Adaptive RAG` folder. We made slight modifications to the original Adaptive repository. To run experiments with `Adaptive` and `IRCoT`, refer to the instructions in [README.md](./Adaptive_Rag/README.md).

## FLARE and DRAGIN

All code, including our evaluator and dependency installation, is available in the `DRAGIN` folder. We copied the original DRAGIN repository. To run experiments with FLARE and DRAGIN, follow the instructions in the DRAGIN [README.md](./dragin/README.md).

## Rowen

Install all required dependencies from `rowen/pyproject.toml` and use `run.sh` to reproduce results. More details can be found in the [Rowen README.md](./rowen/README.md).

## SeaKR

All code, including our evaluator and dependency installation, is available in the `SeaKR` folder. We copied the original SeaKR repository and made some modifications in `vllm_uncertainty`. To run experiments, follow the instructions in the [SeaKR README.md](./SeaKR/README.md).

## Uncertainty Estimation Methods

Install all required dependencies from `UC/requirements.txt` and use the scripts from `bin/*.sh` to reproduce results. More details can be found in the [UC README.md](./UC/README.md).

# Citation
If you use this code or refer to ideas from our paper, please cite:

```bibtex
@article{moskvoretskii2025adaptive,
  title={Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home},
  author={Moskvoretskii, Viktor and Lysyuk, Maria and Salnikov, Mikhail and Ivanov, Nikolay and Pletenev, Sergey and Galimzianova, Daria and Krayko, Nikita and Konovalov, Vasily and Nikishina, Irina and Panchenko, Alexander},
  journal={arXiv preprint arXiv:2501.12835},
  year={2025}
}
```
