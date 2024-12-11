import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
import shutil

def calculate_tokens(dataset_path, model_name, context_col, question_col):
    # Load the dataset
    dataset = load_from_disk(dataset_path)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Process dataset rows
    def tokenize_entry(entry):
        all_context = entry.get(context_col, "")
        question = entry.get(question_col, "")

        if isinstance(all_context, list):
            all_context = [f'Passage {i}: {passage}' for i, passage in enumerate(all_context)]
            all_context = '\n'.join(all_context)
        prompt_with_context = f"Given the following information: \n{all_context}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}\nAnswer:"
        prompt_without_context = f"Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: "

        tokens_with_context = tokenizer(prompt_with_context, truncation=True, return_tensors='pt')
        tokens_without_context = tokenizer(prompt_without_context, truncation=True, return_tensors='pt')

        entry['tokens_all_context'] = len(tokens_with_context['input_ids'][0])
        entry['tokens_no_context'] = len(tokens_without_context['input_ids'][0])

        return entry

    # Map the tokenization function to the dataset
    dataset = dataset.map(lambda entry: tokenize_entry(entry))

    temp_path = f"{dataset_path}_temp"
    # Save the modified dataset
    dataset.save_to_disk(temp_path)
    shutil.rmtree(dataset_path)
    shutil.move(temp_path, dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate tokens with and without context for a Hugging Face dataset.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the Hugging Face dataset.")
    parser.add_argument('--model_name', type=str, required=True, help="Hugging Face model name for tokenization.")
    parser.add_argument('--context_col', type=str, required=True, help="Name of the context column in the dataset.")
    parser.add_argument('--question_col', type=str, required=True, help="Name of the question column in the dataset.")

    args = parser.parse_args()

    calculate_tokens(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        context_col=args.context_col,
        question_col=args.question_col
    )
