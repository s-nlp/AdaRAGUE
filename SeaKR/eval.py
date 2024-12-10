import json
import argparse
from tqdm import tqdm
from eval_utils import has_answer, EM_compute, F1_compute
            
def process_jsonl(input_file, pred_col, gt_col):
    results = []
    total_has_answer = 0
    total_em = 0
    total_f1 = 0
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            item = json.loads(line)
            prediction = item.get(pred_col, "")
            ground_truths = item.get(gt_col, [])
            
            has_ans = has_answer(ground_truths, prediction)
            em = EM_compute(ground_truths, prediction)
            f1 = F1_compute(ground_truths, prediction)

            total_has_answer += has_ans
            total_em += em
            total_f1 += f1
            count += 1

            results.append({
                "prediction": prediction,
                "ground_truths": ground_truths,
                "has_answer": has_ans,
                "EM": em,
                "F1": f1
            })
    
    mean_has_answer = total_has_answer / count if count > 0 else 0
    mean_em = total_em / count if count > 0 else 0
    mean_f1 = total_f1 / count if count > 0 else 0

    print(f"Mean Accuracy: {mean_has_answer:.4f}")
    print(f"Mean EM: {mean_em:.4f}")
    print(f"Mean F1: {mean_f1:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Compute has_answer, EM, and F1 for JSONL file.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("pred_col", type=str, help="Column name for predictions")
    parser.add_argument("gt_col", type=str, help="Column name for ground truth answers")
    args = parser.parse_args()

    results = process_jsonl(args.input_file, args.pred_col, args.gt_col)
    #save if needed

if __name__ == "__main__":
    main()
