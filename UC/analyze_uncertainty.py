from joblib import Parallel, delayed
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import spearmanr
from functools import partial
import datasets
from tqdm import tqdm
from utils import add_metric, need_context, ThresholdOptimizerClassifier, add_all_context, add_no_context, add_ideal, calc_stats, unc_based_KM
from eval_utils import has_answer, EM_compute, F1_compute

np.set_printoptions(suppress=True, precision=3)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(
        class_weight={0: 1, 1: 1}, max_iter=15000
    ),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "MLP": MLPClassifier(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Threshold": ThresholdOptimizerClassifier()
}

metric_names = [
    "Perplexity",
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "MedianTokenEntropy",
    "MinTokenEntropy",
    "MeanMaximumTokenProbability",
    "MaxMaximumTokenProbability",
    "MedianMaximumTokenProbability",
    "MinMaximumTokenProbability",
    "PTrue",
    "Verbalized1S",
    "MeanPointwiseMutualInformation",
    "MeanConditionalPointwiseMutualInformation",
    "RenyiNeg",
    "FisherRao",
    "SemanticEntropy",
    "CCP",
    "SAR",
    "SentenceSAR",
    "NumSemSets",
    "EigValLaplacian_NLI_score_entail",
    "DegMat_NLI_score_entail",
    "Eccentricity_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    "PIKnow",
    "LinRegTokenMahalanobisDistance",
]



def save_results_to_file(results, file_path, headers):
    with open(file_path, "w") as f:
        # Write header
        f.write(tabulate(results, headers=headers, tablefmt="grid"))
        f.write("\n")


def train_and_predict(
    X_train, y_train, X_test, y_test, col, ds_test, classifier, classifier_name, with_context_name, no_context_name
):
    X_train = np.nan_to_num(X_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)
    corr = 0 if col == "hybrid" else spearmanr(X_test, y_test)[0]

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_prob = (
        classifier.predict_proba(X_test)[:, 1]
        if hasattr(classifier, "predict_proba")
        else y_pred
    )

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = (
        roc_auc_score(y_test, y_pred_prob)
        if hasattr(classifier, "predict_proba")
        else np.nan
    )
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix = conf_matrix / np.sum(conf_matrix)
    final_accuracy, mean_tokens, em, f1 = unc_based_KM(
        ds_test.add_column(f"pred_{col}", list(y_pred)),
        with_context_name,
        no_context_name,
        f"pred_{col}",
    )

    retrieval_calls = np.mean(y_pred)

    detailed_metrics = [
        col,
        classifier_name,
        np.around(roc_auc, 3),
        np.around(accuracy, 3),
        np.around(corr, 3),
        conf_matrix.tolist(),
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]
    general_metrics = [
        col,
        classifier_name,
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]

    return detailed_metrics, general_metrics


def main(args):
    ds = datasets.load_from_disk(args.data_path)
    with_context_name, no_context_name, gt_name = (
        args.with_context_col,
        args.no_context_col,
        args.gt_col,
    )

    desired_metrics = [(has_answer, 'InAccuracy'), (EM_compute, 'EM'), (F1_compute, 'F1')]
    for metric_func, metric_name in desired_metrics:
        key_args = {
            'gt_col': gt_name,
            'metric': metric_func,
            'metric_name': metric_name
        }
        ds = ds.map(partial(add_metric, target_col=with_context_name, **key_args))
        ds = ds.map(partial(add_metric, target_col=no_context_name, **key_args))


    ds = ds.map(
        partial(
            need_context,
            with_context_col=with_context_name,
            without_context_col=no_context_name,
            need_context_col="gt_need_retrieval",
        )
    )
    ds = ds.map(partial(calc_stats, col_name='TokenEntropy'))
    ds = ds.map(partial(calc_stats, col_name='MaximumTokenProbability'))


    y_train, y_test = np.array(ds["train"]["gt_need_retrieval"]), np.array(
        ds["test"]["gt_need_retrieval"]
    )

    train_data = pd.DataFrame(
        {
            name: ds["train"][name]
            for name in metric_names
            if (
                (name in ds["train"].column_names) and (name in ds["test"].column_names)
            )
        }
    )

    test_data = pd.DataFrame(
        {
            name: ds["test"][name]
            for name in metric_names
            if (
                (name in ds["train"].column_names) and (name in ds["test"].column_names)
            )
        }
    )

    detailed_results = []
    general_results = []

    def run_all_classifiers_for_feature(col):
        X_train, X_test = train_data[[col]].values, test_data[[col]].values
        return Parallel(n_jobs=-1)(
            delayed(train_and_predict)(
                X_train, y_train, X_test, y_test, col, ds["test"], clf, clf_name, with_context_name, no_context_name
            )
            for clf_name, clf in classifiers.items()
        )

    with tqdm(
        total=len(train_data.columns) * len(classifiers), desc="Processing features"
    ) as pbar:
        for col in train_data.columns:
            classifier_results = run_all_classifiers_for_feature(col)
            for detailed_metrics, general_metrics in classifier_results:
                detailed_results.append(detailed_metrics)
                general_results.append(general_metrics)
            pbar.update(len(classifiers))

    # Hybrid row (all features combined)
    X_train = np.array(train_data.values)
    X_test = np.array(test_data.values)
    for clf_name, clf in classifiers.items():
        if clf_name == 'Threshold':
            continue
        detailed_metrics, general_metrics = train_and_predict(
            X_train, y_train, X_test, y_test, "hybrid", ds["test"], clf, clf_name
        )
        detailed_results.append(detailed_metrics)
        general_results.append(general_metrics)

    # All Context and No Context rows
    detailed_metrics, general_metrics = add_all_context(ds['test'], without_context_col=no_context_name, with_context_col=with_context_name)
    general_results.append(general_metrics)
    detailed_results.append(detailed_metrics)

    detailed_metrics, general_metrics = add_no_context(ds['test'], without_context_col=no_context_name, with_context_col=with_context_name)
    general_results.append(general_metrics)
    detailed_results.append(detailed_metrics)
    

    # Ideal
    detailed_metrics, general_metrics = add_ideal(ds['test'], without_context_col=no_context_name, with_context_col=with_context_name)
    general_results.append(general_metrics)
    detailed_results.append(detailed_metrics)

    # Define headers for both files
    detailed_headers = [
        "Feature",
        "Classifier",
        "ROC AUC",
        "Accuracy",
        "Correlation",
        "Norm Confusion Matrix",
        "In-Accuracy",
        "EM",
        "F1"
        "Mean Tokens",
        "Retrieval Calls"
    ]
    general_headers = ["Feature", "Classifier", "In-Accuracy", "EM", "F1", "Mean Tokens", "Retrieval Calls"]

    # Save detailed and general results to respective files
    data_name = args.data_path.split("/")[-1]
    save_results_to_file(
        detailed_results, f"logs/{data_name}_detailed.log", detailed_headers
    )
    save_results_to_file(
        general_results, f"logs/{data_name}_general.log", general_headers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncertainty estimations for transformer models."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the Hugging Face dataset"
    )
    parser.add_argument(
        "--no_context_col", type=str, required=True, help="Column name for questions"
    )
    parser.add_argument(
        "--with_context_col", type=str, help="Optional column name for context"
    )
    parser.add_argument(
        "--gt_col", type=str, help="Optional column name for output (answers)"
    )
    args = parser.parse_args()

    main(args)
