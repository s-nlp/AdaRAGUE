import datasets
import numpy as np
import argparse
import pandas as pd
from functools import partial
from sklearn.metrics import confusion_matrix


def add_KM(sample, target_col, gt_col):
    corr_ans = sample[gt_col]
    is_corr = 0

    for ref in corr_ans:
        if ref in sample[target_col]:
            is_corr= 1
        
    sample[f'KM_{target_col}'] = is_corr
    
    return sample

def add_self_rag_retrieval(sample):
    sample['self_rag_need_retrieval'] = '[Retrieval]' in sample['self_rag_no_context_response']
    return sample

def need_context(sample, with_context_col, without_context_col, need_context_col):

    if sample[f'KM_{with_context_col}'] > sample[f'KM_{without_context_col}']:
        need_context = 1
    else:
        need_context = 0

    sample[need_context_col] = need_context
    return sample    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics for provided data')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--interest_col', nargs='+', type=str)
    parser.add_argument('--gt_col', type=str)
    args = parser.parse_args()

    logs = []
    ds = datasets.load_from_disk(args.input_path)
    for col in args.interest_col:
        ds = ds.map(partial(add_KM, target_col=col, gt_col=args.gt_col))
        col_name = f'KM_{col}'
        logs.append(f'KM for {col} is {np.mean(ds[col_name])}\n')

    for col in args.interest_col:
        try:
            col_name = f'self_eval_{col}'
            logs.append(f'Self-Eval for {col} is {np.mean(ds[col_name])}\n')
        except KeyError:
            logs.append(f'Self-Eval for {col} is missing\n')

    ds = ds.map(add_self_rag_retrieval)
    ds = ds.map(partial(need_context, with_context_col='self_rag_all_context_response', without_context_col='self_rag_no_context_response',
                        need_context_col='gt_self_rag_need_retrieval'))
    self_rag_cm = confusion_matrix(np.array(ds['gt_self_rag_need_retrieval']), np.array(ds['self_rag_need_retrieval']))
    self_rag_cm = self_rag_cm / self_rag_cm.sum()
    logs.append(f'Self-RAG Confusion Matrix: {self_rag_cm}\n')

    print(''.join(logs))

