from utils.utils import F1_compute, EM_compute

def adaptive_retrieval_score(model_data, ra_data):
    """
    compute scores for results with adaptive RAG
    Input:
        - model_data: data without ra
        - ra_data: data with ra
    """
    accuracy_ls = []
    em_ls = []
    f1_ls = []
    num_calls = 0
    for idx in range(len(model_data)):
        if 'has_answer' not in model_data[idx]:
            continue
        
        if model_data[idx]['Giveup'] == True:
            sample = ra_data[idx]
            num_calls += 1
        else:
            sample = model_data[idx]

        accuracy = sample['has_answer']
        f1 = F1_compute(sample['reference'], sample['Res'])
        em = EM_compute(sample['reference'], sample['Res'])

        accuracy_ls.append(accuracy)
        f1_ls.append(f1)
        em_ls.append(em)

    print(f'percentage retrieval calls {num_calls / len(accuracy_ls)}')
    print(f'count: {len(accuracy_ls)}')
    print(f'Accuracy: {sum(accuracy_ls) / len(accuracy_ls)}')
    print(f'F1: {sum(f1_ls) / len(f1_ls)}')
    print(f'EM: {sum(em_ls) / len(em_ls)}')

    return sum(accuracy_ls) / len(accuracy_ls)


def rag_score(data):
    """
    compute scores for results with static RAG
    """
    score_list = []
    em_ls = []
    f1_ls = []
    for idx in range(len(data)):
        sample = data[idx]
        if 'has_answer' not in sample:
            continue
        score_list.append(sample['has_answer'])

        f1 = F1_compute(sample['reference'], sample['Res'])
        em = EM_compute(sample['reference'], sample['Res'])

        f1_ls.append(f1)
        em_ls.append(em)
    print(f'count: {len(score_list)}')
    print(f'has answer: {sum(score_list) / len(score_list)}')
    print(f'F1: {sum(f1_ls) / len(f1_ls)}')
    print(f'EM: {sum(em_ls) / len(em_ls)}')
    return sum(score_list) / len(score_list)

def compute_giveup_score(data):
    """
    compute scores for results with any strategy
    """
    giveup_list, score_list, align = [], [], []
    overconf_count = 0
    conserv_count = 0
    em_ls = []
    f1_ls = []
    for idx in range(len(data)):
        sample = data[idx]
        if 'has_answer' not in sample: # filter
            continue
        score_list.append(sample['has_answer'])
        if sample['has_answer'] != sample['Giveup']:
            align.append(1)

        if sample['Giveup'] == True:
            if sample['has_answer'] == 1:
                conserv_count +=1
        else:
            if sample['has_answer'] == 0:
                overconf_count += 1
        giveup_list.append(sample['Giveup'])

        f1 = F1_compute(sample['reference'], sample['Res'])
        em = EM_compute(sample['reference'], sample['Res'])

        f1_ls.append(f1)
        em_ls.append(em)
    print(f'conut: {len(giveup_list)}')
    print(f'uncertain ratio: {sum(giveup_list) / len(giveup_list)}')
    print(f'has answer: {sum(score_list) / len(score_list)}')
    print(f'overconf: {format(overconf_count / len(giveup_list), ".4f")}')
    print(f'conserv: {format(conserv_count / len(giveup_list), ".4f")}')
    print(f'alignment: {format(sum(align) / len(giveup_list), ".4f")}')
    print(f'F1: {sum(f1_ls) / len(f1_ls)}')
    print(f'EM: {sum(em_ls) / len(em_ls)}')





    
