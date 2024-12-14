# Set dataset:
local dataset = "2wikimultihopqa";
local retrieval_corpus_name = dataset;
local add_pinned_paras = if dataset == "iirc" then true else false;
local valid_qids = ["228546780bdd11eba7f7acde48001122", "97954d9408b011ebbd84ac1f6bf848b6", "a5995da508ab11ebbd82ac1f6bf848b6", "1ceeab380baf11ebab90acde48001122", "35bf3490096d11ebbdafac1f6bf848b6", "f86b4a28091711ebbdaeac1f6bf848b6", "f44939100bda11eba7f7acde48001122", "e5150a5a0bda11eba7f7acde48001122", "c6805b2908a911ebbd80ac1f6bf848b6", "13cda43c09b311ebbdb0ac1f6bf848b6", "f1ccdfee094011ebbdaeac1f6bf848b6", "028eaef60bdb11eba7f7acde48001122", "8727d1280bdc11eba7f7acde48001122", "79a863dc0bdc11eba7f7acde48001122", "c6f63bfb089e11ebbd78ac1f6bf848b6"];
local prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 0, # don't drop in reading phase.
    "shuffle": false,
    "model_length_limit": 1000000, # don't drop in reading phase.
    "tokenizer_model_name": "google/flan-t5-xl",
};

# (Potentially) Hyper-parameters:
# null means it's unused.
local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 15;
local rc_context_type_ = "gold_with_n_distractors"; # Choices: no, gold, gold_with_n_distractors
local distractor_count = "1"; # Choices: 1, 2, 3
local rc_context_type = (
    if rc_context_type_ == "gold_with_n_distractors"
    then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local rc_qa_type = "direct"; # Choices: direct, cot
local qa_question_prefix = (
    if std.endsWith(rc_context_type, "cot")
    then "Answer the following question by reasoning step-by-step.\n"
    else "Answer the following question.\n"
);

{
    "start_state": "generate_titles",
    "end_state": "[EOQ]",
    "models": {
        "generate_titles": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "generate_main_question",
            "retrieval_type": "bm25",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "original_question",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "end_state": "[EOQ]",
        },

        "generate_main_question": {
            "name": "copy_question",
            "next_model": "answer_main_question",
            "eoq_after_n_calls": 1,
            "end_state": "[EOQ]",
        },
        "answer_main_question": {
            "name": "llmqa",
            "next_model": if std.endsWith(rc_qa_type, "cot") then "extract_answer" else null,
            "prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_"+rc_qa_type+"_qa_flan_t5.txt",
            "question_prefix": qa_question_prefix,
            "prompt_reader_args": prompt_reader_args,
            "end_state": "[EOQ]",
            "gen_model": "llm_api",
            "model_name": "google/flan-t5-xl",
            "model_tokens_limit": 6000,
            "max_length": 200,
            "add_context": true,
        },
        "extract_answer": {
            "name": "answer_extractor",
            "query_source": "last_answer",
            "regex": ".* answer is:? (.*)\\.?",
            "match_all_on_failure": true,
            "remove_last_fullstop": true,
        }
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": add_pinned_paras,
    },
    "prediction_type": "answer"
}