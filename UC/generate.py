import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import datasets
from tqdm import tqdm
import argparse
import pickle
from functools import partial
from collections.abc import Iterable


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, str):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def generate_prompt(sample, use_context_col, tokenizer, *args, **kwargs):
    system_prompt = "You are a helpful assistant tasked with answering questions."
    user_prompt = ""

    if use_context_col != "none":
        user_prompt += "Answer the question based on the information in the documents provided, the documets are divided with lines.\n\n"

    user_prompt += "Answer the following question based on your internal knowledge with one or few words.\n\n"

    if use_context_col != "none":
        all_context = sample[use_context_col]
        if isinstance(context, list):
            all_context = [f'Passage {i}: {passage}' for i, passage in enumerate(all_context)]
            all_context = '\n'.join(all_context)
        user_prompt += f"Document: {all_context}\n\n"

    user_prompt += f'Question: {sample["question"]}\n\nAnswer:'

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def generate_when_to_retrieve_prompt(sample, use_context_col, tokenizer, *args, **kwargs):

    if use_context_col != "none":
        all_context = sample[use_context_col]
        if isinstance(all_context, list):
            all_context = [f'Passage {i}: {passage}' for i, passage in enumerate(all_context)]
            all_context = '\n'.join(all_context)
        user_prompt = f'Given the following information: \n{all_context}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\n'
    else:
        user_prompt = "Answer the following question based on your internal knowledge with one or few words.\n"
    
    user_prompt += f'Question: {sample["question_text"]}\nAnswer: '

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def critic_prompt(sample, tokenizer, answer_col, *args, **kwargs):
    system_prompt = "You are a helpful critic assistant with task of finding factual errors in answers to the question and explaining them."
    user_prompt = "You will be provided a QUESTION and ANSWER to this question.\n\n"
    user_prompt += "Explain and write reason about what is incorrect only, so the next model can improve by looking at your arguments. Pay attention to facts the question is about.\n"
    user_prompt += "If nothing can be improved, and the answer is correct write that the answer is correct.\n"
    user_prompt += "Write in bullet points, be short. Check several times and then give me an answer. Consult with your memory.\n"
    user_prompt += (
        "Do not add more details. Do not write the answer. It is not your task.\n\n"
    )
    user_prompt += "Below is two examples on how you should answer. Do not include this in your output.\n\n"

    few_shot1 = "QUESTION: Which Portuguese soccer player has the most goals? \n\nANSWER: Christiano Ronaldo has scored 900 goals in a career spanning 22 years\n\n"
    few_shot1 += "CRITIC: Yes, Christiano Ronaldo is the correct answer. Other details are unimportant.\n\n"

    few_shot2 = "QUESTION: In Harry Potter literature series wrote by J.K. Rowling, which follows Harry Potter and the Philosopher's Stone?\n\n"
    few_shot2 += "ANSWER: Harry Potter and the Goblet of Fire\n\n"
    few_shot2 += """CRITIC: Lets Break down the answer.Here is the Harry Potter series listed in chronological order of publication:

Harry Potter and the Philosopher's Stone (Sorcerer's Stone in the U.S.) – 1997
Harry Potter and the Chamber of Secrets – 1998
Harry Potter and the Prisoner of Azkaban – 1999
Harry Potter and the Goblet of Fire – 2000
Harry Potter and the Order of the Phoenix – 2003
Harry Potter and the Half-Blood Prince – 2005
Harry Potter and the Deathly Hallows – 2007
Therefore the answer is incorrect. Harry Potter and the Goblet of Fire is the forth book in the series\n\n"""

    user_prompt += few_shot1
    user_prompt += few_shot2

    user_prompt += "Now I provide you a QUESTION and ANSWER.\n\n"

    user_prompt += f'QUESTION: {sample["question"]}\n\n'
    user_prompt += f"ANSWER: {sample[answer_col]}\n\n"

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def revision_prompt(sample, tokenizer, critic_col, answer_col, *args, **kwargs):
    system_prompt = "You are a helpful assistant tasked with answering questions and revising previous answers.\n"

    result = []
    for criticism in sample[critic_col]:
        user_prompt = "Below you will be provided with QUESTION, INITIAL ANSWER and CRITICISM to the answer.\n"
        user_prompt += "Revise the ANSWER according to the CRITICISM\n"

        user_prompt += "Provide the REVISED ANSWER in a single sentence.\n\n"

        user_prompt += f'QUESTION: {sample["question"]}\n\n'
        user_prompt += f"INITIAL ANSWER: {sample[answer_col]}\n\n"
        user_prompt += f"CRITICISM: {criticism}\n\n"
        user_prompt += f"REVISED ANSWER:"

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        result.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    return result

def generate_vllm(queries, num_samples=1):

    global model

    preds = model.generate(queries, sampling_params)

    cur_sample = 0
    output = []
    answers = []
    for query, pred in zip(queries, preds):
        outputs = pred.outputs
        samples_for_query = []
        for sample in outputs:
            samples_for_query.append(sample.text)

        samples_for_query = "\sep\sep".join(samples_for_query)
        answers.append(samples_for_query)
        cur_sample += 1

        if cur_sample == num_samples:
            answers = "\sep\sep".join(answers)
            output.append((query, answers))
            cur_sample = 0
            answers = []

    return output

prompt_mapper = {
    "generate": generate_prompt,
    "critic": critic_prompt,
    "revise": revision_prompt,
    "when_to_retrieve": generate_when_to_retrieve_prompt
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run generation for Truthful QA dataset with and without context"
    )
    parser.add_argument("--model_path", type=str, help="path to model")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--use_context_col", type=str, default="none")
    parser.add_argument("--cache_dir", type=str, default="/home/cache/")
    parser.add_argument("--prompt_type", type=str, default="generate")
    parser.add_argument("--answer_col", type=str, default="no_context_response")
    parser.add_argument(
        "--critic_col", type=str, default="no_context_response_critic_2shot"
    )
    parser.add_argument("--number_output_seq", type=int, default=1)

    args = parser.parse_args()

    model = LLM(args.model_path, download_dir=args.cache_dir, dtype=torch.bfloat16)
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024,
        skip_special_tokens=False,
        n=args.number_output_seq,
    )

    data = datasets.load_from_disk(args.data_path)

    prompt_function = partial(
        prompt_mapper[args.prompt_type],
        use_context_col=args.use_context_col,
        tokenizer=tokenizer,
        answer_col=args.answer_col,
        critic_col=args.critic_col,
    )

    train_queries = [prompt_function(sample) for sample in data['train']]
    train_queries = flatten_list(train_queries)
    train_num_samples = len(train_queries) // len(data['train'])
    train_answers = generate_vllm(train_queries, train_num_samples)
    train_output_path = f'{args.output_path}_train.pickle' 

    with open(train_output_path, "wb") as f:
        pickle.dump(train_answers, f)

    test_queries = [prompt_function(sample) for sample in data['test']]
    test_queries = flatten_list(test_queries)
    test_num_samples = len(test_queries) // len(data['test'])
    test_answers = generate_vllm(test_queries, test_num_samples)
    test_output_path = f'{args.output_path}_test.pickle' 

    with open(test_output_path, "wb") as f:
        pickle.dump(test_answers, f)
