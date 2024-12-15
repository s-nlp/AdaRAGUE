import string
import re
import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Union

class MultiHopEvaluator:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        # Implement alias retrieval if needed
        # For now, return an empty list
        return []
    
    @classmethod
    def normalize_answer(cls, s: str) -> str:
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        if not isinstance(s, str):
            return ""
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ) -> Dict[str, int]:
        if not prediction:
            return {'correct': 0, 'incorrect': 1}
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
    
        correct = int(any(cls.normalize_answer(prediction) == cls.normalize_answer(gt) for gt in ground_truths))
        return {'correct': correct, 'incorrect': 1 - correct}
    
    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ) -> Dict[str, float]:
        final_metric = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        if not prediction:
            return final_metric
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
                
        for gt in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_gt = cls.normalize_answer(gt)
            
            # Skip if either prediction or ground truth is a simple yes/no/noanswer and they don't match
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_gt:
                continue
            if normalized_gt in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_gt:
                continue
            
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_gt.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = num_same / len(prediction_tokens)
            recall = num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Update final metrics with the maximum values
            final_metric['f1'] = max(f1, final_metric['f1'])
            final_metric['precision'] = max(precision, final_metric['precision'])
            final_metric['recall'] = max(recall, final_metric['recall'])
        return final_metric
    
    @classmethod
    def has_answer(cls, answers: Union[str, List[str]], prediction: str, match_type="string") -> int:
        """
        Check if any of the answers are present in the prediction.
        """
        if not prediction:
            return 0
        if isinstance(answers, str):
            answers = [answers]
        normalized_prediction = cls.normalize_answer(prediction)
        prediction_tokens = normalized_prediction.split()
        for answer in answers:
            normalized_answer = cls.normalize_answer(answer)
            answer_tokens = normalized_answer.split()
            # Check if answer_tokens are in prediction_tokens
            for i in range(len(prediction_tokens) - len(answer_tokens) + 1):
                if prediction_tokens[i:i+len(answer_tokens)] == answer_tokens:
                    return 1
        return 0
    
    def eval_answer(self, results_df: pd.DataFrame, answer_col: str = "Final Answer") -> None:
        """
        Evaluate EM, F1, and Accuracy for a specific answer column and add the results to the DataFrame.
        """
        em_list = []
        f1_list = []
        accuracy_list = []
        for _, row in results_df.iterrows():
            prediction = row[answer_col]
            ground_truth = row['ground_truth']
            ground_truths = [ground_truth] if isinstance(ground_truth, str) else ground_truth
            em = self.exact_match_score(prediction, ground_truths)['correct']
            f1 = self.f1_score(prediction, ground_truths)['f1']
            acc = self.has_answer(ground_truths, prediction)
            em_list.append(em)
            f1_list.append(f1)
            accuracy_list.append(acc)
        
        # Add metrics as new columns
        results_df[f'{answer_col}_EM'] = em_list
        results_df[f'{answer_col}_F1'] = f1_list
        results_df[f'{answer_col}_Accuracy'] = accuracy_list
        
        # Print average metrics
        print(f"{answer_col}")
        print(f"EM: {np.mean(em_list):.4f}\t F1: {np.mean(f1_list):.4f}\t Accuracy: {np.mean(accuracy_list):.4f}")
    
    def compute_bootstrap(
        self, 
        results_df: pd.DataFrame, 
        answer_col: str, 
        n_rounds: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform bootstrapping to compute 95% confidence intervals for EM, F1, and Accuracy.
        Returns a dictionary with metrics and their statistics.
        """
        metrics = ['EM', 'F1', 'Accuracy']
        bootstrap_results = {metric: [] for metric in metrics}
        
        for _ in tqdm(range(n_rounds), desc=f"Bootstrapping for {answer_col}"):
            sampled_df = results_df.sample(frac=1.0, replace=True)
            for metric in metrics:
                bootstrap_results[metric].append(sampled_df[f'{answer_col}_{metric}'].mean())
        
        # Compute confidence intervals
        ci = {}
        for metric in metrics:
            lower = np.percentile(bootstrap_results[metric], 2.5)
            upper = np.percentile(bootstrap_results[metric], 97.5)
            mean = np.mean(bootstrap_results[metric])
            median = np.median(bootstrap_results[metric])
            std = np.std(bootstrap_results[metric])
            ci[metric] = {
                'Mean': mean,
                'Median': median,
                'Std Dev': std,
                '95% CI': (lower, upper)
            }
        return ci

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM answers with metrics and bootstrap confidence intervals.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument(
        "--pred_cols", 
        type=str, 
        nargs='+', 
        default=["Final Answer", "Final Step Answer", "Final Read Answer"], 
        help="Column names for predictions"
    )
    parser.add_argument("--gt_col", type=str, required=True, help="Column name for ground truth answers")
    parser.add_argument("--n_rounds", type=int, default=1000, help="Number of bootstrap rounds")
    args = parser.parse_args()
    
    # Load the data
    try:
        results_df = pd.read_json(args.input_file, lines=True)
    except ValueError as e:
        print(f"Error reading the input file: {e}")
        return
    
    # Ensure ground truth column exists
    if args.gt_col not in results_df.columns:
        print(f"Ground truth column '{args.gt_col}' not found in the data.")
        return
    
    evaluator = MultiHopEvaluator()
    
    for answer_col in args.pred_cols:
        if answer_col not in results_df.columns:
            print(f"Prediction column '{answer_col}' not found in the data. Skipping...")
            continue
        
        # Evaluate metrics
        evaluator.eval_answer(results_df=results_df, answer_col=answer_col)
        
        # Perform bootstrapping
        ci = evaluator.compute_bootstrap(results_df=results_df, answer_col=answer_col, n_rounds=args.n_rounds)
        
        # Print bootstrapped confidence intervals
        for metric, stats in ci.items():
            lower, upper = stats['95% CI']
            print(f"{answer_col} {metric} - Mean: {stats['Mean']:.4f}, Median: {stats['Median']:.4f}, Std Dev: {stats['Std Dev']:.4f}, 95% CI: ({lower:.4f}, {upper:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main()
