# TrustGen

## Generation

Genertion is performed with generate.py, example could be found in bin/generate.sh. 

- model_path. Path to HF model
- output_path. Path to save output in pickle format
- data_path should provide path to locally saved HF dataset.
- prompt_type argument == generate, other are experimental.
- use_context_col controls the name of column to use as context, if 'none', no context is used then.
- cache_dir controls the directory to search for HF model
- answer_col not used in generate mode
- critic_col not used in generate mode
- number_output_seq number of sequences to generate for each query

## Adding results of generation

Results of generation are saved in .pickle format. They are packed as List[Tuple(query, List[generations])]. To add them into dataset run the add_column.py 

- dataset_path path to locally saved HF dataset
- input_path path to .pickle saved file
- new_column_name name of new column
- keyword is not used for now

# Estimation of uncertainty

To estimate uncertainty lm-polygraph fork is used. It should be installed from the lm-polygraph folder inside this repo.

Later uncertainty.py should be run

- model_path is model from HF
- cache_dir where model is stored
- data_path path to locally saved HF dataset
- question_column name of column with question
- context_column name of column with context
- output_column name of column with answer
- batch_size used during estimation

# Evaluate classifiers based on uncertainty
Run the analyze_uncertainty.py script. outputs are saved ot logs/ folder

- data_path path to locally saved HF dataset with calculated uncertainty estimations
- no_context_col name of column with answer given without use of context
- with_context_col name of column with answer given with the use of context
- gt_col name of column with ground truth answers

# Evaluation like in Prompting Method

This script evaluates question-answering model outputs using three metrics:
1. **Accuracy**: Checks if any ground truth answer is in the prediction.
2. **Exact Match (EM)**: Measures if the prediction exactly matches any ground truth.
3. **F1 Score (F1)**: Calculates token overlap between prediction and ground truth answers.
 
### Example:
```bash
python eval.py input_file.jsonl prediction_column ground_truth_column
```

- ```input_file.jsonl```: Path to the JSONL input file.
- ```prediction_column```: String name of the column with model predictions (string).
- ```ground_truth_column```: String name of the column with list of ground truth answers, each is a possible correct answer.

You can test it yourself by running test example
```
python eval.py test/test_eval.jsonl 'pred_answer' 'gt_answers'
```
# Bootstrap metrics

This script evaluates mean, std and confidence interval on test using bootstrap

### Example:
```bash
python bootrstrap_CI_upd.py --input_file "preds_jsonl/nq_adaptive_rag.jsonl"\
                          --pred_col 'pred_answer'\
                          --gt_col 'gt_answers'\
                          --n_rounds 1000
```

- ```input_file.jsonl```: Path to the JSONL input file.
- ```pred_col```: String name of the column with model predictions (string).
- ```gt_col```: String name of the column with list of ground truth answers, each is a possible correct answer.
- ```n_rounds```: Number of repetition rounds in bootstrap.

