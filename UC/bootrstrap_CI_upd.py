import json
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

from eval_utils import has_answer, EM_compute, F1_compute
            
def process_jsonl(input_file, pred_col, gt_col):
    
    total_has_answer = 0
    total_em = 0
    total_f1 = 0
    count = 0
    
    has_answer_arr = []
    em_arr = []
    f1_arr = []

    df_init = pd.read_json(input_file, lines=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = json.loads(line)
            prediction = item.get(pred_col, "")
            ground_truths = item.get(gt_col, [])
            has_ans = has_answer(ground_truths, prediction)
            has_answer_arr.append(has_ans)
            em = EM_compute(ground_truths, prediction)
            em_arr.append(em)
            f1 = F1_compute(ground_truths, prediction)
            f1_arr.append(f1)

            total_has_answer += has_ans
            total_em += em
            total_f1 += f1
            count += 1
    
    mean_has_answer = total_has_answer / count if count > 0 else 0
    mean_em = total_em / count if count > 0 else 0
    mean_f1 = total_f1 / count if count > 0 else 0

    df_init['has_answer'] =  has_answer_arr 
    df_init['em'] = em_arr
    df_init['f1'] = f1_arr
    
    return mean_has_answer, mean_em, mean_f1, df_init

def get_bootstrap_result(df_init, num_round):
    
    rows_acc = []
    rows_em = []
    rows_f1 = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        
        sampled_df = df_init.sample(frac=1.0, replace=True)        
        rows_acc.append(np.mean(sampled_df['has_answer']))
        rows_em.append(np.mean(sampled_df['em']))
        rows_f1.append(np.mean(sampled_df['f1']))
        
    df_acc = pd.DataFrame(rows_acc)
    df_em = pd.DataFrame(rows_em)
    df_f1 = pd.DataFrame(rows_f1)
    
    return df_acc[df_acc.median().sort_values(ascending=False).index],\
    df_em[df_em.median().sort_values(ascending=False).index], \
    df_f1[df_f1.median().sort_values(ascending=False).index]


def get_init_result(args, comp_metrics):
    
    json_path = args.input_file
    pred_answer=args.pred_col
    gt_answers=args.gt_col
    acc, em, f1, df = comp_metrics(json_path, pred_answer, gt_answers)

    return  acc, em, f1, df

def main():
    parser = argparse.ArgumentParser(description="Compute has_answer, EM, and F1 for JSONL file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("--pred_col", type=str, help="Column name for predictions")
    parser.add_argument("--gt_col", type=str, help="Column name for ground truth answers")
    parser.add_argument("--n_rounds", type=int, help="Number of bootstrap rounds")

        
    args = parser.parse_args()

    np.random.seed(42)
    
    # results = process_jsonl(args.input_file, args.pred_col, args.gt_col)
    acc_init, em_init, f1_init, df_init = get_init_result(args, process_jsonl)
    acc, em, f1 = get_bootstrap_result(df_init, args.n_rounds)
    
    lower_bounds = []
    upper_bounds = []
    
    lower_bounds.append(np.percentile(acc, 2.5))
    upper_bounds.append(np.percentile(acc, 97.5))
    
    lower_bounds.append(np.percentile(em, 2.5))
    upper_bounds.append(np.percentile(em, 97.5))
    
    lower_bounds.append(np.percentile(f1, 2.5))
    upper_bounds.append(np.percentile(f1, 97.5))

    decimal = 3

    print("-"*50)
    
    for i in range(3):
    
        if i == 0:
            interval = str((round(lower_bounds[i], decimal), round(upper_bounds[i] , decimal)))
            print(f" Accuracy (init, mean, median): {round(acc_init, decimal) : ^5}, {round(np.mean(acc), decimal) : ^5}, {round(np.median(acc), decimal) : ^5} |Std: {round(np.std(acc)[0], decimal)} | 95% CI: {interval : ^12}")
        elif i == 1:
            interval = str((round(lower_bounds[i] , decimal), round(upper_bounds[i] , decimal)))
            print(f" EM (init, mean, median): {round(em_init, decimal) : ^5}, {round(np.mean(em), decimal) : ^5}, {round(np.median(em), decimal) : ^5} | Std: {round(np.std(em)[0], decimal)} | 95% CI: {interval : ^12}")
        elif i == 2:
            interval = str((round(lower_bounds[i], decimal), round(upper_bounds[i], decimal)))
            print(f" F1 (init, mean, median): {round(f1_init, decimal) : ^5}, {round(np.mean(f1), decimal) : ^5}, {round(np.median(f1), decimal) : ^5}  | Std: {round(np.std(f1)[0], decimal)} |95% CI: {interval : ^12}")

    #save if needed

if __name__ == "__main__":
    main()
