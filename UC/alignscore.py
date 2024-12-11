import argparse
import numpy as np
from datasets import load_from_disk
from alignscore import AlignScorer
import shutil

ckpt_path = "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"


def save_estimations(scores, dataset, data_path, score_column_name="align_score"):
    print(f"Saving results to {data_path}")

    # Add the scores as a new column in the dataset
    dataset = dataset.add_column(score_column_name, scores.tolist())

    # Save the updated dataset
    temp_path = f"{data_path}_temp"
    dataset.save_to_disk(temp_path)
    shutil.rmtree(data_path)
    shutil.move(temp_path, data_path)


def main(args):
    # Load dataset
    dataset = load_from_disk(args.data_path)

    claims = dataset[args.pred_column]
    contexts = dataset[args.gt_column]

    scorer = AlignScorer(
        model="roberta-large",
        batch_size=args.bs,
        device="cuda",
        ckpt_path=ckpt_path,
        evaluation_mode=args.eval_mode,
    )

    scores = np.array(
        scorer.score(
            claims=claims,
            contexts=contexts,
        )
    )

    new_name = f"alignscore_{pred_column}"
    save_estimations(scores, dataset, args.data_path, score_column_name=new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a column of predictions against ground truth using AlignScorer."
    )

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the Hugging Face dataset"
    )
    parser.add_argument(
        "--pred_column",
        type=str,
        required=True,
        help="Column name for predictions (claims)",
    )
    parser.add_argument(
        "--gt_column",
        type=str,
        required=True,
        help="Column name for ground truth (contexts)",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_mode", type=str, default="nli_sp")

    args = parser.parse_args()

    main(args)
