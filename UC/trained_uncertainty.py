import argparse
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import (
    MahalanobisDistanceSeq,
    RelativeMahalanobisDistanceSeq,
    RDESeq,
)
from lm_polygraph.utils.manager import estimate_uncertainty
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils import UEManager
from copy import deepcopy
from functools import partial
import numpy as np

# Define estimators
estimators = [
    MahalanobisDistanceSeq("decoder"),
    RelativeMahalanobisDistanceSeq("decoder"),
    RDESeq("decoder"),
]

background_train_dataset_path = "allenai/c4"
background_train_dataset_text_column = "text"
background_train_dataset_label_column = "url"
background_train_dataset_data_files = "en/c4-train.00000-of-01024.json.gz"
background_load_from_disk = False

subsample_background_train_dataset = 1000
seed = 0


def load_model_and_tokenizer(model_path, cache_dir):
    print(f"Loading model from {model_path}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir, torch_dtype="auto", device_map="auto"
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
        output = row[output_column] if output_column != "none" else ""
        outputs.append(output)

    return inputs, outputs


def add_trained_uc(sample, desired_uc_names, all_results):

    idx = sample["question_id"]
    try:
        resulting_sample = all_results[idx]
        for uc_name in desired_uc_names:
            sample[uc_name] = resulting_sample[uc_name]
    except KeyError:
        for uc_name in desired_uc_names:
            sample[uc_name] = np.nan

    return sample


def main(args):
    # Load model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer(args.model_path, args.cache_dir)

    # Wrap the model with WhiteboxModel
    model = WhiteboxModel(base_model, tokenizer, model_path=args.model_path)

    dataset = load_dataset_hf(args.data_path)
    # split to add training set, that will be used to tune trained UC
    ds_train_val = dataset["train"].train_test_split(test_size=0.5, seed=0)

    # concat real test and the one that is not used to train our UC (we will tune some algorithm atop)
    new_test_ds = concatenate_datasets([ds_train_val["test"], dataset["test"]])
    new_train_ds = ds_train_val["train"]

    train_inputs, train_outputs = construct_inputs_outputs_hf(
        new_train_ds,
        args.question_column,
        args.context_column,
        args.output_column,
        verbalized=False,
    )
    # Create Dataset for UEManager
    train_ds = Dataset(train_inputs, train_outputs, batch_size=args.batch_size)

    test_inputs, test_outputs = construct_inputs_outputs_hf(
        new_test_ds,
        args.question_column,
        args.context_column,
        args.output_column,
        verbalized=False,
    )
    # Create Dataset for UEManager
    test_ds = Dataset(test_inputs, test_outputs, batch_size=args.batch_size)

    for estimator in list(estimators):
        if (str(estimator) in dataset["train"].column_names) and (
            str(estimator) in dataset["test"].column_names
        ):
            estimators.remove(estimator)
    print(estimators)

    background_train_dataset = Dataset.load(
        background_train_dataset_path,
        background_train_dataset_text_column,
        background_train_dataset_label_column,
        batch_size=args.batch_size,
        data_files=background_train_dataset_data_files,
        split="train",
        size=100_000,
        load_from_disk=background_load_from_disk,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    background_train_dataset.subsample(subsample_background_train_dataset, seed=seed)

    print("Running uncertainty estimation")
    man = UEManager(
        test_ds,
        model,
        estimators,
        [],
        [],
        [],
        ignore_exceptions=False,
        train_data=train_ds,
        verbose=True,
        background_train_data=background_train_dataset,
    )

    # Run the manager and get the results
    res = man()
    new_test_estimations = deepcopy(man.estimations)

    desired_uc_names = []
    for k, v in new_test_estimations.items():
        if k[1] in new_test_ds.column_names:
            new_test_ds = new_test_ds.remove_columns(k[1])
        new_test_ds = new_test_ds.add_column(k[1], v)
        desired_uc_names.append(k[1])

    all_results = {
        sample["question_id"]: {
            uc_name: sample[uc_name] for uc_name in desired_uc_names
        }
        for sample in new_test_ds
    }

    dataset = dataset.map(
        partial(
            add_trained_uc, desired_uc_names=desired_uc_names, all_results=all_results
        )
    )
    # assert counter == (len(dataset['train']) // 2), 'Missing more ids than should be'

    # Save the estimations
    print("saving estimations")
    temp_path = f"{args.data_path}_temp"
    dataset.save_to_disk(temp_path)
    shutil.rmtree(args.data_path)
    shutil.move(temp_path, args.data_path)


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
