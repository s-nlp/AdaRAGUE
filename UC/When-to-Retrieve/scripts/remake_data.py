import datasets
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dataset remake for prompting')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    try:
        test = datasets.load(args.data_path)['test']
    except:
        test = datasets.load_from_disk(args.data_path)['test']


    cols_to_drop = ['dataset', 'question_id', 'contexts', 'answers_objects', 'reasoning_steps']
    for col in cols_to_drop:
        if col not in test.column_names:
            cols_to_drop.remove(col)

    task = args.data_path.split('/')[-1]
    print(args.data_path.split('/'))

    test = test.remove_columns(cols_to_drop)
    test = test.rename_column('retrieved_contexts', 'dense_ctxs')
    if 'gold_ctxs' in test.column_names:
        test = test.rename_column('gold_context', 'gold_ctxs')
    test = test.rename_column('question_text', 'question')
    test = test.add_column('id', list(range(len(test))))
    test = test.add_column('task', [task]*len(test))

    test.to_json(f'{args.save_dir}/{task}.jsonl', lines=True)
