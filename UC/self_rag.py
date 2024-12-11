from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import datasets
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run generation for Truthful QA dataset with and without context')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--cache_dir', type=str, default='/home/cache/')
    parser.add_argument('--context_col_name', type=str)
    args = parser.parse_args()

    model = LLM("selfrag/selfrag_llama2_7b", download_dir=args.cache_dir, dtype="half")
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
    
    def format_prompt(input, paragraph='none'):
      prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
      if paragraph != 'none':
        prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
      return prompt
    
    data = datasets.load_from_disk(args.data_path)
    queries = list(data['question'])
    if args.context_col_name != 'none':
        contexts = list(data[args.context_col_name])
    else:
        contexts = ['none']*len(queries)
    
    preds = model.generate([format_prompt(query, context) for query, context in zip(queries, contexts)], sampling_params)
    output = []
    for query, pred in zip(queries, preds):
      output.append(([query], [pred.outputs[0].text]))

    with open(args.output_path, 'wb') as f:
        pickle.dump(output, f)
