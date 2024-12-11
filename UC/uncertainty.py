import argparse
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import (
    MaximumTokenProbability,
    Perplexity,
    PTrue,
    MeanTokenEntropy,
    Verbalized1S,
    Verbalized2S,
    TokenEntropy,
    MeanPointwiseMutualInformation,
    MeanConditionalPointwiseMutualInformation,
    RenyiNeg,
    FisherRao,
    SemanticEntropy,
    ClaimConditionedProbability,
    SAR,
    SentenceSAR,
    NumSemSets,
    EigValLaplacian,
    DegMat,
    Eccentricity,
    LexicalSimilarity,
)
from lm_polygraph.estimators.p_know import PIKnow
from lm_polygraph.utils.manager import estimate_uncertainty
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils import UEManager
from copy import deepcopy

verbalized_1s_top1_template = "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words orexplanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\n"


# Define estimators
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
 #   PIKnow(),
]

def load_model_and_tokenizer(model_path, cache_dir):
    print(f"Loading model from {model_path}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Move model to GPU

    return model, tokenizer


def load_dataset_hf(data_path):
    print(f"Loading dataset from {data_path}")

    # Load dataset from Hugging Face datasets
    dataset = load_from_disk(data_path)
    return dataset


def construct_inputs_outputs_hf(
    dataset, question_column, context_column, output_column, verbalized=False
):
    print("Constructing inputs and outputs")

    inputs = []
    outputs = []

    # Create input prompt based on question and context
    for row in dataset:
        question = row[question_column]
        # context = row[context_column] if context_column != 'none' else ""
        if verbalized:
            prompt = (
                f"{verbalized_1s_top1_template} The question is: {question}".strip()
            )
        else:
            prompt = f"{question}".strip()
        inputs.append(prompt)

        # Construct output list if the column is provided, otherwise empty
        output = row[output_column] if output_column != 'none' else ""
        outputs.append(output)

    return inputs, outputs


def save_estimations(train_estimations, test_estimations, dataset, data_path):
    print(f"Saving results")
    temp_path = f"{data_path}_temp"

    for split, estimation in zip(['train', 'test'], (train_estimations, test_estimations)):
        for k, v in estimation.items():
            if k[1] in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns(k[1])
            dataset[split] = dataset[split].add_column(k[1], v)


    dataset.save_to_disk(temp_path)
    shutil.rmtree(data_path)
    shutil.move(temp_path, data_path)


def main(args):
    # Load model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer(args.model_path, args.cache_dir)

    # Wrap the model with WhiteboxModel
    model = WhiteboxModel(base_model, tokenizer, model_path=args.model_path)


    # Load dataset from Hugging Face datasets format
    dataset = load_dataset_hf(args.data_path)

    # Construct inputs and outputs
    train_inputs, train_outputs = construct_inputs_outputs_hf(
        dataset['train'],
        args.question_column,
        args.context_column,
        args.output_column,
        verbalized=True,
    )
    # Create Dataset for UEManager
    train_ds = Dataset(train_inputs, train_outputs, batch_size=args.batch_size)

    for estimator in list(estimators):
        if str(estimator) in dataset['train'].column_names:
            estimators.remove(estimator)
    print(estimators)

    print("Running uncertainty estimation")
    man = UEManager(
        train_ds,
        model,
        estimators,
        [],
        [],
        [],
        ignore_exceptions=False,
        verbose=True,
    )

    # Run the manager and get the results
    res = man()
    train_estimations = deepcopy(man.estimations)

    # Construct inputs and outputs
    test_inputs, test_outputs = construct_inputs_outputs_hf(
        dataset['test'],
        args.question_column,
        args.context_column,
        args.output_column,
        verbalized=True,
    )
    # Create Dataset for UEManager
    test_ds = Dataset(test_inputs, test_outputs, batch_size=args.batch_size)

    for estimator in list(estimators):
        if str(estimator) in dataset['test'].column_names:
            estimators.remove(estimator)

    print("Running uncertainty estimation")
    man = UEManager(
        test_ds,
        model,
        estimators,
        [],
        [],
        [],
        ignore_exceptions=False,
        verbose=True,
    )

    # Run the manager and get the results
    res = man()
    test_estimations = deepcopy(man.estimations)

    # Save the estimations
    save_estimations(train_estimations, test_estimations, dataset, args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncertainty estimations for transformer models."
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True, help="Directory for model cache"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the Hugging Face dataset"
    )
    parser.add_argument(
        "--question_column", type=str, required=True, help="Column name for questions"
    )
    parser.add_argument(
        "--context_column", type=str, help="Optional column name for context"
    )
    parser.add_argument(
        "--output_column", type=str, help="Optional column name for output (answers)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for the dataset"
    )

    args = parser.parse_args()

    main(args)
