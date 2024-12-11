import datasets
import os
import click
from enum import Enum
import asyncio
from typing import List
import threading
import signal
import sys
import time
import json
from elasticsearch import Elasticsearch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from run_truthfulqa import (
    chain_of_thought_reasoning,
    construct_input_message,
    SEMANTICALLY_EQUIVALENT_PERTERBATIONS_TEMPLATE,
    async_machine_translate,
    CROSS_CONSISTENCY_CHECK_TEMPLATE,
    TRUTHFULQA_INITIAL_ANSWER_TEMPLATE,
    TRUTHFULQA_INITIAL_ANSWER_TEMPLATE_ZH,
    TRUTHFULQA_FINAL_ANSWER_TEMPLATE,
    TRUTHFULQA_FINAL_ANSWER_TEMPLATE_ZH,
    qwen_chain_of_thought_reasoning,
    TRUTHFULQA_REPAIR_HALLUCINATION_TEMPLATE,
)
from utils import (
    single_run,
    remove_prefix,
    is_supported,
    run_api,
    Sqlite3CacheProvider,
    LLMClientSingleton,
)

class DatasetName(Enum):
    NATURAL_QUESTIONS = 'natural_questions'
    TRIVIA_QA = 'trivia_qa'
    SQUAD = 'squad'
    WIKI_MULTIHOP_QA = '2wiki_multihop_qa'
    HOTPOT_QA = 'hotpot_qa'
    MUSIQUE = 'musique'

    @classmethod
    def get_full_path(cls, name):
        mapping = {
            cls.NATURAL_QUESTIONS.value: 'VityaVitalich/adaptive_rag_natural_questions',
            cls.TRIVIA_QA.value: 'VityaVitalich/adaptive_rag_trivia_qa',
            cls.SQUAD.value: 'VityaVitalich/adaptive_rag_squad',
            cls.WIKI_MULTIHOP_QA.value: 'VityaVitalich/adaptive_rag_2wikimultihopqa',
            cls.HOTPOT_QA.value: 'VityaVitalich/adaptive_rag_hotpotqa',
            cls.MUSIQUE.value: 'VityaVitalich/adaptive_rag_musique',
        }
        return mapping.get(name, name)
    
    
async def chain_of_thought_reasoning_language_consistency(questions: List[str], use_chinese=False):
    # first query
    messages = [
        construct_input_message(
            TRUTHFULQA_INITIAL_ANSWER_TEMPLATE.format(question=question)
            if not use_chinese
            else TRUTHFULQA_INITIAL_ANSWER_TEMPLATE_ZH.format(question=question)
        )
        for question in questions
    ]
    long_answers = await run_api(messages=messages)
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    TRUTHFULQA_INITIAL_ANSWER_TEMPLATE.format(question=question)
                    if not use_chinese
                    else TRUTHFULQA_INITIAL_ANSWER_TEMPLATE_ZH.format(question=question)
                ),
            },
            {"role": "assistant", "content": long_answer},
            {
                "role": "user",
                "content": (
                    TRUTHFULQA_FINAL_ANSWER_TEMPLATE
                    if not use_chinese
                    else TRUTHFULQA_FINAL_ANSWER_TEMPLATE_ZH
                ),
            },
        ]
        for question, long_answer in zip(questions, long_answers)
    ]
    # second query
    short_answers = await run_api(messages=messages)
    return short_answers



def generate_perturbations(item, number_of_clusters):
    semantic_equivalent_queries_str = single_run(
        messages=construct_input_message(
            prompt=SEMANTICALLY_EQUIVALENT_PERTERBATIONS_TEMPLATE.format(
                question=item['question_text'], k=number_of_clusters
            )
        ),
        temperature=1.0,
    )
    return [
        remove_prefix(query).strip()
        for query in semantic_equivalent_queries_str.split("\n")
    ]


async def search_wiki(elastic_search, query, index='wiki', limit=5):
    """
    Search the 'wiki' index in Elasticsearch for the given query.
    
    :param query: The search query string.
    :param index: The name of the index to search.
    :param limit: The maximum number of search results to return.
    :return: The search results.
    """
    search_body = {
        'query': {
            'multi_match': {
                'query': query,
                'fields': ['title', 'body']
            }
        },
        'size': limit
    }
    
    response = elastic_search.search(index=index, body=search_body)
    return response


async def get_context(perturbated_queries, elasitc_search):
    contexts = await asyncio.gather(*[
        search_wiki(elasitc_search, query, limit=1)
        for query in perturbated_queries
    ])

    def _format_context(context):
        if len(context['hits']['hits']) > 0:
            return f"{context['hits']['hits'][0]['_source']['title']}: {context['hits']['hits'][0]['_source']['txt']}"

    queries_contexts = [(query, _format_context(context)) for query, context in zip(perturbated_queries, contexts)]
    queries_contexts = filter(lambda x: x[1] is not None, queries_contexts)

    return [
        f"{query}? {context}"
        for query, context in queries_contexts
    ]


async def evaluate_language_consistency(state):
    state["chinese_perturbated_queries"] = await async_machine_translate(state["perturbated_queries"])
    state["target_perturbated_answers"] = await chain_of_thought_reasoning_language_consistency(
        state["chinese_perturbated_queries"], use_chinese=True
    )
    
    # Calculate language_consistency_check_results
    tasks = [
        run_api(
            [
                construct_input_message(
                    CROSS_CONSISTENCY_CHECK_TEMPLATE.format(
                        q=query,
                        a1=en_answer,
                        a2=zh_answer,
                    )
                )
            ]
        )
        for query, en_answer, zh_answer in zip(
            state["perturbated_queries"],
            state["original_long_answer"],
            state["target_perturbated_answers"],
        )
    ]
    results = await asyncio.gather(*tasks)
    language_consistency_check_results = [
        is_supported(out[0].lower()) for out in results
    ]
    
    
    state["valid_semantic_equivalent_QA_pair"] = [
        {
            "query": query,
            "source_answer": en_answer,
            "target_answer": zh_answer,
            "is_consistent": consistency,
        }
        for query, en_answer, zh_answer, consistency in zip(
            state["perturbated_queries"],
            state["original_long_answer"],
            state["target_perturbated_answers"],
            language_consistency_check_results,
        )
    ]
    state["language_consistency_check_score"] = (
        sum(language_consistency_check_results)
        * 1.0
        / len(language_consistency_check_results)
    )
    return state


async def evaluate_model_consistency(state, qwen_model_name):
    state["cross_model_perturbated_answers"] = [
        qwen_chain_of_thought_reasoning(query, qwen_model_name)[1]
        for query in state["perturbated_queries"]
    ]
    
    tasks = [
        run_api(
            [
                construct_input_message(
                    CROSS_CONSISTENCY_CHECK_TEMPLATE.format(
                        q=query,
                        a1=en_answer,
                        a2=cross_model_answer,
                    )
                )
            ]
        )
        for query, en_answer, cross_model_answer in zip(
            state["perturbated_queries"],
            state["original_long_answer"],
            state["cross_model_perturbated_answers"],
        )
    ]
    results = await asyncio.gather(*tasks)
    cross_model_consistency_check_results = [
        is_supported(out[0].lower()) for out in results
    ]
    
    state["cross_model_semantic_equivalent_pairs"] = [
        {
            "query": query,
            "source_answer": en_answer,
            "cross_model_answer": cross_model_answer,
            "is_consistent": consistency,
        }
        for query, en_answer, cross_model_answer, consistency in zip(
            state["perturbated_queries"],
            state["original_long_answer"],
            state["cross_model_perturbated_answers"],
            cross_model_consistency_check_results,
        )
    ]
    state["cross_model_consistency_check_score"] = (
        sum(cross_model_consistency_check_results)
        * 1.0
        / len(cross_model_consistency_check_results)
    )
    return state


def process_item(item, number_of_clusters, qwen_model_name, cm_consistency_alpha, consistency_threshold, elasitc_search, mode):
    state = {
        'dataset': item.get('dataset'),
        'question_id': item['question_id'],
        'question_text': item['question_text'],
        'reference': item['reference'],
        'retriever_call_num': 0,
        'main_llm_call_num': 0,
        'qwen_llm_call_num': 0,
    }    
    state['original_long_answer'], state['original_short_answer'] = chain_of_thought_reasoning(item['question_text'])
    state["perturbated_queries"] = generate_perturbations(item, number_of_clusters)
    state["main_llm_call_num"] += 3
    
    if mode in ['CL', 'Hybrid']:
        state = asyncio.run(evaluate_language_consistency(state))
        state["main_llm_call_num"] += 3 * len(state["perturbated_queries"])
    if mode in ['CM', 'Hybrid']:
        state = asyncio.run(evaluate_model_consistency(state, qwen_model_name))
        state["qwen_llm_call_num"] += 2 * len(state["perturbated_queries"])
        state["main_llm_call_num"] += len(state["perturbated_queries"])
    
    if mode in ['Hybrid']:
        state['consistency_check_score'] = (
            state['language_consistency_check_score'] + 
            cm_consistency_alpha * state['cross_model_consistency_check_score']
        )
    elif mode == 'CL':
        state['consistency_check_score'] = state['language_consistency_check_score']
    elif mode == 'CM':
        state['consistency_check_score'] = state['cross_model_consistency_check_score']
    else:
        raise ValueError(f"Invalid mode: {mode}")


    if state['consistency_check_score'] >= consistency_threshold:
        # print(f"Consistency check passed. No need to repair. {state['consistency_check_score']} | {consistency_threshold}")
        state['repaired_answer'] = state['original_short_answer']
    else:
        # print(f"Consistency check failed. Need to repair. {state['consistency_check_score']} | {consistency_threshold} | {state['consistency_check_score'] >= consistency_threshold}")
        state["retrieved_evidences"] = asyncio.run(get_context(state["perturbated_queries"], elasitc_search))
        state["retriever_call_num"] = len(state["retrieved_evidences"])
        
        state['repaired_answer'] = single_run(
            messages=construct_input_message(
                prompt=TRUTHFULQA_REPAIR_HALLUCINATION_TEMPLATE.format(
                    question=item['question_text'],
                    initial_long_answer=state['original_long_answer'],
                    initial_short_answer=state['original_short_answer'],
                    evidences="\n".join(state["retrieved_evidences"]),
                )
            )
        )
        state["main_llm_call_num"] += 1
    
        
    return state


def print_cache_size():
    cache_provider_openai = Sqlite3CacheProvider()
    cache_provider_qwen = Sqlite3CacheProvider("qwen_cache.db")
    cache_provider_serper = Sqlite3CacheProvider("google_serper_cache.db")
    cache_provider_vllm = Sqlite3CacheProvider("localvllm_cache.db")

    previous_sizes = {
        "openai": None,
        "qwen": None,
        "serper": None,
        "vllm": None,
    }

    while True:
        current_sizes = {
            "openai": cache_provider_openai.get_size(),
            "qwen": cache_provider_qwen.get_size(),
            "serper": cache_provider_serper.get_size(),
            "vllm": cache_provider_vllm.get_size(),
        }

        for key in current_sizes:
            if current_sizes[key] != previous_sizes[key]:
                print(f"{key.capitalize()} Cache size: {current_sizes[key]}")
                previous_sizes[key] = current_sizes[key]

        time.sleep(10)
        

@click.command()
@click.option('--dataset_name', type=click.Choice([e.value for e in DatasetName]), default=DatasetName.NATURAL_QUESTIONS.value)
@click.option('--k', default=6, help='Number of clusters')
@click.option('--qwen_model_name', type=str, default='Qwen2.5-72B-Instruct-AWQ', help='Qwen model name')
@click.option('--cm_consistency_alpha', type=float, default=1.0, help='Cross-model consistency alpha value')
@click.option('--consistency_threshold', type=float, default=0.5, help='Consistency threshold value')
@click.option('--max_workers', type=int, default=16, help='Number of workers')
@click.option('--mode', type=click.Choice(['CL', 'CM', 'Hybrid']), default='Hybrid', help='Mode of execution')
@click.option('--save_path', type=str, default=None, help='Path to save results JSONL format')
@click.option('--elastic_search_url', type=str, default='http://localhost:9200', help='Elasticsearch')
def main(dataset_name, k, qwen_model_name, cm_consistency_alpha, consistency_threshold, max_workers, mode, save_path, elastic_search_url):
    full_path = DatasetName.get_full_path(dataset_name)
    ds = datasets.load_dataset(full_path)
    elasitc_search = Elasticsearch([elastic_search_url])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _proc_item = lambda item: process_item(item, k, qwen_model_name, cm_consistency_alpha, consistency_threshold, elasitc_search, mode)
        results = list(tqdm(
            executor.map(_proc_item, ds['test']),
            total=len(ds['test'])
        ))
    

    if save_path is None or save_path == '':
        save_path = f"./runs/results_dataset_{dataset_name}_mode_{mode}_k_{k}_alpha_{cm_consistency_alpha}_th_{consistency_threshold}.jsonl"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path, "a+") as f:
        for line in results:
            f.write(json.dumps(line) + '\n')

    


if __name__ == '__main__':
    cache_monitor_thread = threading.Thread(target=print_cache_size, daemon=True)
    cache_monitor_thread.start()

    def signal_handler(sig, frame):
        print('Terminating...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main()
