import torch
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

import datasets
from tqdm import tqdm
import argparse
from peft import LoraConfig, PeftModel
from pathlib import Path
import pickle
from functools import partial




def tokenize(sample, target_col_name, use_reasoning=False):
    # prompt template https://arxiv.org/pdf/2307.03172#page=3.10
    # llama instruct 3 format
    prompt = 'Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.'
    prompt += "\n\n--\n"
    prompt += f'QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):\n{sample["question"]}'
    prompt += "\n\n--\n"
    prompt += f'DOCUMENT:\n{sample["source_text"]}'
    prompt += "\n\n--\n"
    prompt += f'ANSWER:\n{sample[target_col_name]}'
    prompt += "\n\n--\n\n"
    prompt += 'Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":\n'
    prompt += '{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}'

    new_sample = {}
    new_sample['role'] = 'user'
    new_sample['content'] = prompt
    
    return new_sample

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run generation for Truthful QA dataset with and without context')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--use_context', type=str)
    parser.add_argument('--cache_dir', type=str, default='/home/cache/')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()


    model_name = 'PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct'
    pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=600,
            device_map="auto",
            return_full_text=False,
            model_kwargs={'cache_dir': args.cache_dir},
            #tokenizer_kwargs={'cache_dir': '/home/data/v.moskvoretskii/cache/'}
            torch_dtype=torch.bfloat16
    )
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    # messages = [
    #     {"role": "user", "content": prompt},
    # ]

    # result = pipe(messages)
    # print(result[0]['generated_text'])



    data = datasets.load_from_disk(args.data_path)

    if args.use_context == 'retrieved':
        target_col_name = 'context_response'
    elif args.use_context == 'none':
        target_col_name = 'no_context_response'
    elif args.use_context == 'generated':
        target_col_name = 'gen_context_response'
    
    data = data.map(partial(tokenize, target_col_name=target_col_name))

    drop_columns = list(data.column_names)
    drop_columns.remove('role')
    drop_columns.remove('content')

    data = data.remove_columns(drop_columns)

    gen_pairs = []
   # all_data = [elem for elem in data]
    with torch.no_grad():
        for elem in tqdm(data):
        #for out in tqdm(pipe(all_data, batch_size=args.batch_size), total=len(all_data)):
            out = pipe([elem])
            gen_pairs.append((elem, out))


    with open(args.output_path, 'wb') as f:
        pickle.dump(gen_pairs, f)
      
