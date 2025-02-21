# Uncertainty Estimation for Adaptive Retrieval

This repository contains the code for uncertainty estimation used to evaluate adaptive retrieval.

## Installation

Our uncertainty estimation methods are based on the LM-Polygraph library. Install it locally using the provided version:

```bash
cd lm-polygraph
pip install -e .
```

## Running Uncertainty Estimation

To perform uncertainty estimation, use the following script. The arguments are defined as follows:

- `model_path`: Path to the Hugging Face (HF) model, either local or hosted on the HF Hub.
- `cache_dir`: Directory where the HF model will be stored.
- `data_path`: Local directory containing the dataset.
- `question_column`: Column name that contains the questions.
- `context_column`: Leave as `none`.
- `output_column`: Leave as `none`.
- `batch_size`: Batch size for processing.

```bash
python uncertainty.py \
    --model_path VityaVitalich/Llama3.1-8b-instruct \
    --cache_dir /home/data/v.moskvoretskii/cache/ \
    --data_path data/datasets/s_nq \
    --question_column question_text \
    --context_column none \
    --output_column none \
    --batch_size 32
```

To select an uncertainty estimation method, refer to lines **39-60** in `uncertainty.py`, where the available methods are defined:

```python
estimators = [
    MaximumTokenProbability(),
    TokenEntropy(),
    Perplexity(),
    PTrue(),
    Verbalized1S(confidence_regex=r"Probability:\s*(0(\.\d+)?|1(\.0+)?)"),
    MeanPointwiseMutualInformation(),
    MeanConditionalPointwiseMutualInformation(),
    RenyiNeg(),
    FisherRao(),
    SemanticEntropy(),
    ClaimConditionedProbability(),
    SAR(),
    SentenceSAR(),
    NumSemSets(),
    EigValLaplacian(),
    DegMat(),
    Eccentricity(),
    LexicalSimilarity(),
#   PIKnow(),  # Uncomment to enable
]
```

The script appends new columns to the HF dataset, named after the uncertainty estimation methods used, with their corresponding values.

## Adaptive Retrieval Using Uncertainty Estimation

Once uncertainty estimations are computed, they can be used to guide adaptive retrieval. To do this, we require:
- Model responses **with retrieval** and **without retrieval**.
- Ground-truth references.

We typically use outputs from the **AdaptiveRAG** paper, which we have reproduced.

### Running Adaptive Retrieval Analysis

The script below evaluates retrieval performance and trains ML models based on uncertainty estimations:

- `data_path`: Path to the HF dataset containing uncertainty estimates.
- `no_context_col`: Column containing model responses **without retrieval**.
- `with_context_col`: Column containing model responses **with retrieval**.
- `gt_col`: Column containing reference answers (a list of reference strings per question).

```bash
python analyze_uncertainty.py \
    --data_path data/datasets/s_nq \
    --no_context_col no_context_response \
    --with_context_col context_response \
    --gt_col reference
```

Results are logged in `logs/{data_name}_detailed.log`, containing all computed metrics.
