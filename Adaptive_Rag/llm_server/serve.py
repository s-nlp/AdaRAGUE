import os
from fuzzywuzzy import fuzz
# from vllm import LLM
os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(os.getcwd()) + '/cache' # '/data/soyeong/cache'
os.environ["HF_TOKEN"] = "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"
import time
from functools import lru_cache

from fastapi import FastAPI

if "TRANSFORMERS_CACHE" not in os.environ:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from constants import TRANSFORMERS_CACHE

    os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser(
        os.sep.join(TRANSFORMERS_CACHE.split("/"))  # before importing transformers
    )

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

@lru_cache(maxsize=None)
def get_model_and_tokenizer():

    model_shortname = os.environ["MODEL_NAME"]

    valid_model_shortnames = [
        "gpt-j-6B",
        "opt-66b",
        "gpt-neox-20b",
        "T0pp",
        "opt-125m",
        "flan-t5-base",
        "flan-t5-large",
        "flan-t5-xl",
        "flan-t5-xxl",
        "flan-t5-base-bf16",
        "flan-t5-large-bf16",
        "flan-t5-xl-bf16",
        "flan-t5-xxl-bf16",
        "llama-8b-it"
    ]
    assert model_shortname in valid_model_shortnames, f"Model name {model_shortname} not in {valid_model_shortnames}"

    if model_shortname == "gpt-j-6B":

        model_name = "EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision="sharded",
            device_map="auto",  # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    elif model_shortname == "llama-8b-it":

        access_token = os.environ.get("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")
        hf_token = "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"
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                                  cache_dir="/home/jovyan/shares/SR003.nfs2/.cache/models/transformers/",
                                                  token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        EOS_TOKEN = tokenizer.eos_token
        # #loading the model using AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir="/home/jovyan/shares/SR003.nfs2/.cache/models/transformers/",
            token=access_token,
            device_map="auto", 
            # load_in_8bit=True,
            # torch_dtype= torch.bfloat16, 
            attn_implementation = "eager")
        # model = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
        

    elif model_shortname == "opt-66b":

        model_name = "facebook/opt-66b"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main", device_map="auto")
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname == "gpt-neox-20b":

        model_name = "EleutherAI/gpt-neox-20b"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision="main",
            device_map="auto",  # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "T0pp":

        model_name = "bigscience/T0pp"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            revision="sharded",
            device_map="auto",  # torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "opt-125m":

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main", device_map="auto")
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname.startswith("flan-t5") and "bf16" not in model_shortname:
        model_name = "google/" + model_shortname
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="main", device_map=hf_device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname.startswith("flan-t5") and model_shortname.endswith("-bf16"):

        assert torch.cuda.is_bf16_supported()
        assert is_torch_bf16_gpu_available()
        model_name = "google/" + model_shortname.replace("-bf16", "")
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map=hf_device_map, torch_dtype=torch.bfloat16
        )
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    elif model_shortname == "ul2":
        model_name = "google/" + model_shortname
        model = T5ForConditionalGeneration.from_pretrained(
            # Don't use auto here. It's slower cpu loading I guess.
            "google/ul2"  # , low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained("google/ul2")

    return model, tokenizer


class EOSReachedCriteria(StoppingCriteria):
    # Use this when EOS is not a single id, but a sequence of ids, e.g. for a custom EOS text.
    def __init__(self, tokenizer: AutoTokenizer, eos_text: str):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert (
            len(self.tokenizer.encode(eos_text)) < 10
        ), "EOS text can't be longer then 10 tokens. It makes stopping_criteria check slow."

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0][-10:])
        condition1 = decoded_text.endswith(self.eos_text)
        condition2 = decoded_text.strip().endswith(self.eos_text.strip())
        return condition1 or condition2


app = FastAPI()


@app.get("/")
async def index():
    model_shortname = os.environ["MODEL_NAME"]
    return {"message": f"Hello! This is a server for {model_shortname}. " "Go to /generate/ for generation requests."}


@app.get("/generate/")
async def generate(
    prompt: str,
    max_input: int = None,
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    eos_text: str = None,
    keep_prompt: bool = False,
):
    start_time = time.time()

    model_shortname = os.environ["MODEL_NAME"]

    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=max_input).cuda()

    stopping_criteria_list = StoppingCriteriaList()
    if eos_text:
        stopping_criteria = EOSReachedCriteria(tokenizer=tokenizer, eos_text=eos_text)
        stopping_criteria_list = StoppingCriteriaList([stopping_criteria])

    # T0pp, ul2 and flan are the only encoder-decoder model, and so don't have prompt part in its generation.
    is_encoder_decoder = model_shortname in ["T0pp", "ul2"] or model_shortname.startswith("flan-t5")
    # max_length_ = max_length if is_encoder_decoder else inputs.shape[1]+max_length
    generated_output = model.generate(
        inputs,
        # max_length=max_length_,
        max_new_tokens=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        stopping_criteria=stopping_criteria_list,
        output_scores=False,  # make it configurable later. It turns in generated_output["scores"]
    )
    generated_ids = generated_output["sequences"]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # print('PROMPT full', prompt)
    # with open('ircot_prompts.txt', 'a+') as f:
    #     f.write('PROMPT full\n')
    #     f.write(prompt+'\n')
    cur_promp_len =  len(prompt.split("A: "))
    # print("PROMPT length", cur_promp_len)
    # print("PROMPT splits", prompt.split("A: "))
    # print("GEN texts", generated_texts)
    # with open('ircot_prompts.txt', 'a+') as f:
    #     f.write('GEN TEXT\n')
    #     f.write(str(generated_texts)+'\n')
    # with open('ircot_prompts.txt', 'a+') as f:
    #     f.write('PROMPT length\n')
    #     f.write(str(cur_promp_len)+'\n')
    # with open('ircot_prompts.txt', 'a+') as f:
    #     f.write('GEN length\n')
    #     f.write(str(len(generated_texts[0].split("A: ")))+'\n')
    # print("PREV", [
    #             generated_text[generated_text.index(generated_text.split("A: ")[cur_promp_len]) :] for generated_text in generated_texts
    #         ])
    # with open('ircot_prompts.txt', 'a+') as f:
    #     f.write('PREV-1\n')
    #     f.write(str([
    #             generated_text[generated_text.index(generated_text.split("A: ")[cur_promp_len-1]) :] for generated_text in generated_texts
    #         ])+'\n')
    # with open('ircot_prompts.txt', 'a+') as f:
    #     f.write('PREV\n')
    #     f.write(str([
    #             generated_text[generated_text.index(generated_text.split("A: ")[cur_promp_len]) :] for generated_text in generated_texts
    #         ])+'\n')
    # with open('ircot_prompts.txt', 'a+') as f:
    #     f.write('16th\n')
    #     f.write(str([
    #             generated_text[generated_text.index(generated_text.split("A: ")[16]) :] for generated_text in generated_texts
    #         ])+'\n')
    # print("CUR",[
    #             generated_text[generated_text.index(generated_text.split("A: ")[cur_promp_len]) :] for generated_text in generated_texts
    #         ])
    # print('PROMPT length', len(prompt.split("A: ")))
    
    # print('PROMPT part', prompt.split('\n')[-2:])
    # print('PROMPT join', '\n'.join(prompt.split('\n')[-2:]))
    # print("INCLUSION", '\n'.join(prompt.split('\n')[-2:]) in generated_texts[0])

    # print("INDEX", generated_text.index(prompt))
    generated_num_tokens = [len(generated_ids_) for generated_ids_ in generated_ids]
    if not keep_prompt and not is_encoder_decoder:
        # cur_promp_len =  len(prompt.split("A: "))
        # # oner_qa
        # generated_texts = [
        #         generated_text[generated_text.index(generated_text.split("A: ")[cur_promp_len]) :] for generated_text in generated_texts
        #     ]
        
        # #irCoT
        # generated_texts = [
        #         generated_text[generated_text.index(generated_text.split("A: ")[cur_promp_len-1]) :] for generated_text in generated_texts
        #     ]
        # print("SPLIT", generated_texts)
        try:
            prompt_part = '\n'.join(prompt.split('\n')[-2:])
            # with open('ircot_prompts.txt', 'a+') as f:
            #     f.write('PROMPT PART\n')
            #     f.write(str(prompt_part)+'\n')
            generated_texts = [
                generated_text[generated_text.index(prompt_part) + len(prompt_part) :] for generated_text in generated_texts
            ]
            # with open('ircot_prompts.txt', 'a+') as f:
            #     f.write('GEN TEXT\n')
            #     f.write(str(generated_texts)+'\n')
        except:
            try:
                prompt_part = '\n'.join(prompt.split('\n')[-2:]).replace(" '", "\'")
                # with open('ircot_prompts.txt', 'a+') as f:
                #     f.write('PROMPT PART\n')
                #     f.write(str(prompt_part)+'\n')
                generated_texts = [
                generated_text[generated_text.index(prompt_part) + len(prompt_part) :] for generated_text in generated_texts]
                # with open('ircot_prompts.txt', 'a+') as f:
                #     f.write('FIN TEXT\n')
                #     f.write(str(generated_texts)+'\n')
            except:
                # print("FUZZ RATIO", fuzz.ratio(prompt_part, "Q: "+generated_texts[0].split("Q: ")[-1].split("\nA: ")[0]))
                prompt_part = generated_texts[0].split("Q: ")[-1].split("\nA: ")[0]
                # with open('ircot_prompts.txt', 'a+') as f:
                #     f.write('PROMPT PART\n')
                #     f.write(str(prompt_part)+'\n')
                if fuzz.ratio(prompt_part, "Q: "+generated_texts[0].split("Q: ")[-1].split("\nA: ")[0]) > 90:
                    generated_texts = [
                generated_text[generated_text.index(generated_text.split("A: ")[-1]) :] for generated_text in generated_texts
                    ]
                # with open('ircot_prompts.txt', 'a+') as f:
                #     f.write('FIN TEXT\n')
                #     f.write(str(generated_texts)+'\n')
                    
    elif keep_prompt and is_encoder_decoder:
        generated_texts = [prompt + generated_text for generated_text in generated_texts]

    end_time = time.time()
    run_time_in_seconds = end_time - start_time
    return {
        "generated_num_tokens": generated_num_tokens,
        "generated_texts": generated_texts,
        "run_time_in_seconds": run_time_in_seconds,
        "model_name": model_shortname,
    }


print("\nLoading model and tokenizer.")
get_model_and_tokenizer()
print("Loaded model and tokenizer.\n")
