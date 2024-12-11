import pickle
import argparse
import os
import shutil
from datasets import load_from_disk, DatasetDict


def extract_assistant_responses(decoded_pairs, keyword="assistant"):
    new_data = []
    for _, generations in decoded_pairs:
        batch_generations = []
        if isinstance(generations, str):
            generations = [generations]
        for generation in generations:
            # Extract the generation starting from the "assistant" keyword
            if isinstance(generation, dict):
                # obtained from pipeline
                generation = generation["generated_text"]
            start_index = generation.lower().find(keyword)
            # if start_index != -1:
            #     response = generation[start_index + len(keyword):].strip()
            # else:
            #     response = generation  # If "assistant" keyword is not found, keep the full generation
            response = generation

            if "\sep\sep" in response:
                response = response.split("\sep\sep")
            if "SCORE" in keyword:
                # it means it is evaluation
                if "PASS" in response:
                    response = 1
                else:
                    response = 0

            batch_generations.append(response)
        new_data.extend(
            batch_generations
        )  # Assuming the dataset is row-wise, we need individual responses for each record.

    return new_data


def main(dataset_path, input_path, new_column_name, keyword):
    # Load the Hugging Face dataset
    dataset = load_from_disk(dataset_path)
    subset = 'train' if 'train' in input_path else 'test'

    # Load existing generations from pickle file
    with open(input_path, "rb") as f:
        decoded_pairs = pickle.load(f)

    # Extract relevant responses starting after the keyword "assistant"
    extracted_responses = extract_assistant_responses(decoded_pairs, keyword=keyword)

    # Check if the lengths match
    if len(dataset[subset]) != len(extracted_responses):
        raise ValueError(
            f"The number of extracted responses ({len(extracted_responses)}) does not match the number of rows in the dataset ({len(dataset)})."
        )

    # Add the new column to the dataset
    if new_column_name in dataset[subset].column_names:
        dataset[subset] = dataset[subset].remove_columns(new_column_name)
    dataset[subset] = dataset[subset].add_column(new_column_name, extracted_responses)

    temp_path = f"{dataset_path}_temp"
    # Save the modified dataset
    dataset.save_to_disk(temp_path)
    shutil.rmtree(dataset_path)
    shutil.move(temp_path, dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract generations from an existing pickle file and add as a new column to a HF dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input pickle file containing generations.",
    )
    parser.add_argument(
        "--new_column_name", type=str, required=True, help="Name for the new column."
    )
    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        help="Name for the keyword to split decoded sentence",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path '{args.dataset_path}' does not exist.")

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file '{args.input_path}' does not exist.")

    main(args.dataset_path, args.input_path, args.new_column_name, args.keyword)
