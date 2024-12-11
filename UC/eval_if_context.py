import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import datasets
from tqdm import tqdm
import argparse
from peft import LoraConfig, PeftModel
from pathlib import Path
import pickle
from functools import partial

def tokenize(sample, use_context, use_reasoning=False):
    # prompt template https://arxiv.org/pdf/2307.03172#page=3.10
    # llama instruct 3 format
    new_sample = {}
    prompt = '<|begin_of_text|>'
    prompt += '<|start_header_id|>user<|end_header_id|>\n\n'

    prompt += 'Given the following Question, evaluate your abilities to answer it without the use of external information. If you have enough knowledge to answer it, respond with [[NO CONTEXT]], however if you have not enough knowledge on this topic and require external information provided, respond with [[CONTEXT]].\n\n'

    

    prompt += f'Question: {sample["question"]}\n Answer:' # main prompt
    prompt += '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

    new_sample["input_ids"] = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
    new_sample["attention_mask"] = torch.ones_like(new_sample['input_ids'])
    return new_sample

def collator(data):

    global tokenizer

    input_ids = []
    att_masks = []
    for batch in data:
        input_ids.append(torch.tensor(batch['input_ids']).squeeze(0))
        att_masks.append(torch.tensor(batch['attention_mask']).squeeze(0))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    att_masks = torch.nn.utils.rnn.pad_sequence(
        att_masks, batch_first=True, padding_value=0
    )

    
    return (input_ids, att_masks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run generation for Truthful QA dataset with and without context')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='/home/cache/')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        cache_dir=args.cache_dir,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    data = datasets.load_from_disk(args.data_path)
    data = data.map(partial(tokenize, use_context=args.use_context))
    
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    gen_pairs = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            output = model.generate(input_ids=batch[0].to('cuda'), attention_mask=batch[1].to('cuda'),  max_new_tokens=64, temperature=0.6, do_sample=True, top_p=0.9)
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            question = tokenizer.batch_decode(batch[0], skip_special_tokens=True)
            gen_pairs.append((question, decoded))


    with open(args.output_path, 'wb') as f:
        pickle.dump(gen_pairs, f)
      
