import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from eval_utils import has_answer
from functools import partial

class ThresholdOptimizerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.threshold_ = None

    def fit(self, X, y):
        """
        Fit the classifier by finding the optimal threshold within the range of X that maximizes accuracy on the training data.
        
        Parameters:
        X : array-like of shape (n_samples, 1)
            Feature data (assumes a single feature for thresholding).
        y : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        
        Returns:
        self : object
            Fitted estimator.
        """
        # Ensure X is a 1D array of values (not probabilities, can be any range)
        if X.shape[1] > 1:
            return self
        X = np.ravel(X)

        # Determine range for threshold search between min and max of X
        min_val, max_val = X.min(), X.max()
        
        best_threshold = min_val
        best_accuracy = 0.0

        # Try different thresholds within the range of X values
        for threshold in np.linspace(min_val, max_val, num=100):
            y_pred = (X >= threshold).astype(int)
            accuracy = accuracy_score(y, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold_ = best_threshold
        return self

    def predict(self, X):
        """
        Predict binary labels based on the learned threshold.
        
        Parameters:
        X : array-like of shape (n_samples, 1)
            Feature data to predict (assumes a single feature for thresholding).
        
        Returns:
        y_pred : ndarray of shape (n_samples,)
            Predicted binary labels.
        """
        if self.threshold_ is None:
            return np.nan
            #raise ValueError("The model has not been fitted yet. Please call 'fit' before 'predict'.")
        
        X = np.ravel(X)
        return (X >= self.threshold_).astype(int)


def KM(ds, target_col, gt_col):
    total_match = 0
    for sample in ds:
        corr_ans = sample[gt_col]
        model_ans = sample[target_col]
        is_corr = has_answer(corr_ans, model_ans)
        total_match += is_corr
    return total_match / len(ds)


def add_metric(sample, target_col, gt_col, metric, metric_name):
    corr_ans = sample[gt_col]
    model_ans = sample[target_col]
    is_corr = metric(corr_ans, model_ans)
    sample[f"{metric_name}_{target_col}"] = is_corr
    return sample

def need_context(sample, with_context_col, without_context_col, need_context_col):
    # if sample[f'KM_{with_context_col}'] > sample[f'KM_{without_context_col}']:
    #     need_context = 1
    # else:
    #     need_context = 0
    if not sample[f"InAccuracy_{without_context_col}"]:
        need_context = 1
    else:
        need_context = 0
    sample[need_context_col] = need_context
    return sample

def ideal_unc(elem, without_context_col):
    elem['ideal_unc'] = not elem[f"InAccuracy_{without_context_col}"]
    return elem

def add_ideal(ds, without_context_col, with_context_col):

    ds = ds.map(partial(ideal_unc, without_context_col=without_context_col))
    final_accuracy, mean_tokens, em, f1 = unc_based_KM(ds, with_context_col=with_context_col, without_context_col=without_context_col, unc_col='ideal_unc')
    retrieval_calls = np.mean(ds['ideal_unc'])
    detailed_metrics = [
        'Ideal',
        '',
        '',
        '',
        '',
        '',
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]
    general_metrics = [
        'Ideal',
        '',
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]

    return detailed_metrics, general_metrics

def add_all_context(ds, without_context_col, with_context_col):
    ds = ds.add_column('unc_all_context', np.ones(len(ds)))
    final_accuracy, mean_tokens, em, f1 = unc_based_KM(ds, with_context_col=with_context_col, without_context_col=without_context_col, unc_col='unc_all_context')
    retrieval_calls = 1

    detailed_metrics = [
        'All Context',
        '',
        '',
        '',
        '',
        '',
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]
    general_metrics = [
        'All Context',
        '',
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]

    return detailed_metrics, general_metrics

def add_no_context(ds, without_context_col, with_context_col):
    ds = ds.add_column('unc_no_context', np.zeros(len(ds)))
    final_accuracy, mean_tokens, em, f1 = unc_based_KM(ds, with_context_col=with_context_col, without_context_col=without_context_col, unc_col='unc_no_context')
    retrieval_calls = 0

    detailed_metrics = [
        'No Context',
        '',
        '',
        '',
        '',
        '',
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]
    general_metrics = [
        'No Context',
        '',
        np.around(final_accuracy, 3),
        np.around(em, 3),
        np.around(f1, 3),
        np.around(mean_tokens, 1),
        np.around(retrieval_calls, 2)
    ]

    return detailed_metrics, general_metrics


def calc_stats(sample, col_name):
    entropy = sample[col_name]
    sample[f"Max{col_name}"] = np.max(entropy)
    sample[f"Median{col_name}"] = np.median(entropy)
    sample[f"Min{col_name}"] = np.min(entropy)
    sample[f"Mean{col_name}"] = np.mean(entropy)
    sample[f"Sum{col_name}"] = np.sum(entropy)

    return sample



def unc_based_KM(ds, with_context_col, without_context_col, unc_col):
    in_acc, tokens, em, f1 = [], [], [], []
    for elem in ds:
        need_retrieval = elem[unc_col]
        if need_retrieval:
            ans_col = with_context_col
            tokens_col = "tokens_all_context"
        else:
            ans_col = without_context_col
            tokens_col = "tokens_no_context"

        in_accuracy_col = f"InAccuracy_{ans_col}"
        em_col = f"EM_{ans_col}"
        f1_col = f"F1_{ans_col}"
        num_tokens = elem[tokens_col]

        in_acc.append(elem[in_accuracy_col])
        tokens.append(num_tokens)
        em.append(elem[em_col])
        f1.append(elem[f1_col])
    return np.mean(in_acc), np.mean(tokens), np.mean(em), np.mean(f1)