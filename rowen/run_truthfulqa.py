import argparse
import asyncio
import json
import os
from typing import List

import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from tqdm import tqdm

from google_serper import GoogleSerperAPIWrapper
from prompt_template import *
from utils import *
import concurrent.futures

google_search = GoogleSerperAPIWrapper()


def construct_input_message(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


def chain_of_thought_reasoning(question: str):
    messages = construct_input_message(
        TRUTHFULQA_INITIAL_ANSWER_TEMPLATE.format(question=question)
    )
    # first query
    long_answer = single_run(messages)
    messages.append({"role": "assistant", "content": long_answer})
    # second query
    messages.append({"role": "user", "content": TRUTHFULQA_FINAL_ANSWER_TEMPLATE})
    short_answer = single_run(messages)
    return long_answer, short_answer


def async_chain_of_thought_reasoning(questions: List[str], use_chinese=False):
    # first query
    messages = [
        construct_input_message(
            TRUTHFULQA_INITIAL_ANSWER_TEMPLATE.format(question=question)
            if not use_chinese
            else TRUTHFULQA_INITIAL_ANSWER_TEMPLATE_ZH.format(question=question)
        )
        for question in questions
    ]
    long_answers = asyncio.run(run_api(messages=messages))
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
    short_answers = asyncio.run(run_api(messages=messages))
    return short_answers


def qwen_chain_of_thought_reasoning(question: str, model_name: str):
    messages = construct_input_message(
        TRUTHFULQA_INITIAL_ANSWER_TEMPLATE.format(question=question)
    )
    # first query
    long_answer = single_qwen_run(messages, model_name)
    messages.append({"role": "assistant", "content": long_answer})
    # second query
    messages.append({"role": "user", "content": TRUTHFULQA_FINAL_ANSWER_TEMPLATE})
    short_answer = single_qwen_run(messages, model_name)
    return long_answer, short_answer


def machine_translate(source_text, backward=False):
    prompt = (
        "Translate the following sentences into {}. Only response with translated sentence and do not include any useless content: \n".format(
            "Chinese" if not backward else "English"
        )
        + source_text
    )
    return single_run(messages=construct_input_message(prompt))


async def async_machine_translate(source_texts, backward=False):
    async def machine_translate(source_text, backward=False, retry=3):
        for _ in range(retry):
            try:
                prompt = (
                    "Translate the following sentences into {}. Only response with translated sentence and do not include any useless content: \n".format(
                        "Chinese" if not backward else "English"
                    )
                    + source_text
                )
                return single_run(messages=construct_input_message(prompt=prompt))
            except Exception as e:
                logging.error(e)
                await asyncio.sleep(10)
        return None

    responses = [
        machine_translate(source_text, backward) for source_text in source_texts
    ]
    return await asyncio.gather(*responses)


def evaluate_answer_factualness(eval_filepath="", batch_size=20):
    with open(eval_filepath, "r") as f:
        data = json.load(f)

    # compute BLEU scores
    bleu_score = corpus_bleu(
        [
            [item["Best Answer"].split(" "), item["Correct Answers"].split(" ")]
            for item in data
        ],
        [item["repaired_answer"].split(" ") for item in data],
    )
    print("BLEU Score: ", bleu_score)

    # compute Rouge-L scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(
        [item["repaired_answer"][:10000] for item in data],
        [item["Best Answer"] + " " + item["Correct Answers"] for item in data],
        avg=True,
    )["rouge-l"]["f"]
    print("Rouge-L Score: ", rouge_scores)

    # # compute the percentage of factual responses
    # response_list = []
    # for idx in tqdm(range(0, len(data), batch_size)):
    #     batch_input = [
    #         "Q: {0}\nA: {1}\nTrue:".format(
    #             item["Question"], sent_tokenize(item["repaired_answer"])[0]
    #         )
    #         for item in data[idx : idx + batch_size]
    #     ]
    #     response_list.extend(asyncio.run(run_completion_api(batch_input)))

    # check_results = []
    # for response in response_list:
    #     logprobs = response["choices"][0]["logprobs"]
    #     output_str = logprobs["tokens"][0]
    #     output_dict = logprobs["top_logprobs"][0]
    #     yes_prob = np.exp(output_dict[" yes"]) if " yes" in output_dict else 0.0
    #     check_results.append(
    #         {
    #             "text": output_str,
    #             "logprobs": logprobs,
    #             "yes_prob": yes_prob,
    #             "final_decision": bool(yes_prob >= 0.5),
    #         }
    #     )

    # assert len(check_results) == len(data)
    # for idx, item in enumerate(data):
    #     item["gpt_judge_results"] = check_results[idx]

    # factual_response_percentage = (
    #     sum([item["gpt_judge_results"]["final_decision"] for item in data])
    #     * 1.0
    #     / len(data)
    # )
    # print("Percentage of Factual Response: ", factual_response_percentage)

    # file_name, file_extension = os.path.splitext(eval_filepath)
    # with open(file_name + "_eval" + file_extension, "w") as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)




def reduce_hallucination_pipleline(args):
    with open(
        "./dataset/TruthfulQA.json",
        "r",
    ) as f:
        ori_data = json.load(f)

    def _proc_item(item):
        item_copy = item.copy()

        question = item["Question"]

        (
            original_long_answer,
            original_short_answer,
        ) = chain_of_thought_reasoning(question)
        item_copy["original_long_answer"] = original_long_answer
        item_copy["original_short_answer"] = original_short_answer

        semantic_equivalent_queries_str = single_run(
            messages=construct_input_message(
                prompt=SEMANTICALLY_EQUIVALENT_PERTERBATIONS_TEMPLATE.format(
                    question=question, k=args.k
                )
            ),
            temperature=1.0,
        )
        perturbated_queries = [
            remove_prefix(query).strip()
            for query in semantic_equivalent_queries_str.split("\n")
        ]
        item_copy["perturbated_queries"] = perturbated_queries
        
        perturbated_answers = async_chain_of_thought_reasoning(perturbated_queries)
        item_copy["perturbated_answers"] = perturbated_answers

        if args.mode != "model": #Language
            chinese_perturbated_queries = asyncio.run(
                async_machine_translate(perturbated_queries)
            )
            item_copy["chinese_perturbated_queries"] = chinese_perturbated_queries

            target_perturbated_answers = async_chain_of_thought_reasoning(
                chinese_perturbated_queries, use_chinese=True
            )
            item_copy["target_perturbated_answers"] = target_perturbated_answers

            language_consistency_check_results = [
                is_supported(out.lower())
                for out in asyncio.run(
                    run_api(
                        [
                            construct_input_message(
                                CROSS_CONSISTENCY_CHECK_TEMPLATE.format(
                                    q=query,
                                    a1=en_answer,
                                    a2=zh_answer,
                                )
                            )
                            for query, en_answer, zh_answer in zip(
                                perturbated_queries,
                                perturbated_answers,
                                target_perturbated_answers,
                            )
                        ]
                    )
                )
            ]

            item_copy["valid_semantic_equivalent_QA_pair"] = [
                {
                    "query": query,
                    "source_answer": en_answer,
                    "target_answer": zh_answer,
                    "is_consistent": consistency,
                }
                for query, en_answer, zh_answer, consistency in zip(
                    perturbated_queries,
                    perturbated_answers,
                    target_perturbated_answers,
                    language_consistency_check_results,
                )
            ]

            language_consistency_check_score = (
                sum(language_consistency_check_results)
                * 1.0
                / len(language_consistency_check_results)
            )
            item_copy["language_consistency_check_score"] = (
                language_consistency_check_score
            )

        if args.mode != "language": #Model
            cross_model_perturbated_answers = [
                qwen_chain_of_thought_reasoning(query, args.qwen_model_name)[1]
                for query in perturbated_queries
            ]
            item_copy["cross_model_perturbated_answers"] = (
                cross_model_perturbated_answers
            )

            cross_model_consistency_check_results = [
                is_supported(out.lower())
                for out in asyncio.run(
                    run_api(
                        [
                            construct_input_message(
                                CROSS_MODEL_CONSISTENCY_CHECK_TEMPLATE.format(
                                    q=query,
                                    a1=en_answer,
                                    a2=zh_answer,
                                )
                            )
                            for query, en_answer, zh_answer in zip(
                                perturbated_queries,
                                perturbated_answers,
                                cross_model_perturbated_answers,
                            )
                        ]
                    )
                )
            ]

            item_copy["cross_model_semantic_equivalent_pairs"] = [
                {
                    "query": query,
                    "source_answer": en_answer,
                    "cross_model_answer": cross_model_answer,
                    "is_consistent": consistency,
                }
                for query, en_answer, cross_model_answer, consistency in zip(
                    perturbated_queries,
                    perturbated_answers,
                    cross_model_perturbated_answers,
                    cross_model_consistency_check_results,
                )
            ]

            cross_model_consistency_check_score = (
                sum(cross_model_consistency_check_results)
                * 1.0
                / len(cross_model_consistency_check_results)
            )
            item_copy["cross_model_consistency_check_score"] = (
                cross_model_consistency_check_score
            )

        if args.mode == "language":
            consistency_check_score = language_consistency_check_score
        elif args.mode == "model":
            consistency_check_score = cross_model_consistency_check_score
        else:
            consistency_check_score = (
                language_consistency_check_score
                + args.alpha * cross_model_consistency_check_score
            )

        if consistency_check_score >= args.threshold:
            item_copy["repaired_answer"] = original_short_answer
            # generation_data.append(item_copy)
            # continue
            return item_copy

        search_outputs = asyncio.run(google_search.run(perturbated_queries))
        retrieved_evidences = [
            query.rstrip("?") + "? " + output["content"]
            for query, result in zip(perturbated_queries, search_outputs)
            for output in result
        ]
        item_copy["retrieved_evidences"] = retrieved_evidences

        repaired_answer = single_run(
            messages=construct_input_message(
                prompt=TRUTHFULQA_REPAIR_HALLUCINATION_TEMPLATE.format(
                    question=question,
                    initial_long_answer=original_long_answer,
                    initial_short_answer=original_short_answer,
                    evidences="\n".join(retrieved_evidences),
                )
            )
        )
        item_copy["repaired_answer"] = repaired_answer

        # generation_data.append(item_copy)
        return item_copy

    def print_cache_size():
        cache_provider_openai = Sqlite3CacheProvider()
        cache_provider_qwen = Sqlite3CacheProvider("qwen_cache.db")
        cache_provider_serper = Sqlite3CacheProvider("google_serper_cache.db")
        cache_provider_vllm = Sqlite3CacheProvider("localvllm_cache.db")
        while True:
            print(f"OpenAI Cache size:        {cache_provider_openai.get_size()}")
            print(f"Qwen2 Cache size:         {cache_provider_qwen.get_size()}")
            print(f"Vllm LLama3 Cache size:   {cache_provider_vllm.get_size()}")
            print(f"Google Serper Cache size: {cache_provider_serper.get_size()}\n")
            time.sleep(60)
    threading.Thread(target=print_cache_size, daemon=True).start()
    
    def _proc_item_tc(item):
        try:
            return _proc_item(item)
        except Exception as e:
            print(e)
            return None
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        generation_data = list(tqdm(executor.map(_proc_item_tc, ori_data), total=len(ori_data)))
    # generation_data = [_proc_item(item) for item in tqdm(ori_data)]
    
    save_path = f"truthfulqa_llama318binsturct_k{args.k}_threshold{args.threshold}_alpha{args.alpha}_qwen-{args.qwen_model_name}_mode-{args.mode}.json"
    with open(
        save_path,
        "w",
    ) as f:
        json.dump(generation_data, f, indent=4, ensure_ascii=False)

    evaluate_answer_factualness(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parsing example")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold value")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value")
    parser.add_argument(
        "--qwen_model_name", type=str, default="Qwen2.5-72B-Instruct-AWQ", help="Qwen model name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["language", "model", "hybrid"],
        default="hybrid",
        help="Mode of operation",
    )
    args = parser.parse_args()
    
    reduce_hallucination_pipleline(args)
