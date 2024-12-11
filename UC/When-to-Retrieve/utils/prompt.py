from utils.utils import remove_punc
prompt_dict = {
    'qa': {
        'none': 'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_explain': {
        'none': 'Answer the following question based on your internal knowledge with one or few words and explain why you give this answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words and explain why you give this answer.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_cot': {
        'none': 'Answer the following question based on your internal knowledge and let\'s think step by step.\n 1.Answer the question with one or few words.\nQuestion: {question}{paras}{prediction}',
        'ra': '',
        'tail': '\nAnswer: ',
    },
    'qa_gene': {
        'none': 'Generate a short document that helps answer the following question based on your internal knowledge and answer the question with one or few words.\nQuestion: {question}{paras}{prediction}',
        'ra': '',
        'tail': '\nAnswer: ',
    },
    'prior': {
        'none': 'Answer the following question based on your internal knowledge with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior_punish': {
        'none': 'Answer the following question based on your internal knowledge with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior_explain': {
        'none': 'Answer the following question based on your internal knowledge with one or few words and explain why you give this answer. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words and explain why you give this answer. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior_pun_exp': {
        'none': 'Answer the following question based on your internal knowledge with one or few words and explain why you give this answer. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}Answer the following question based on the given information or your internal knowledge with one or few words and explain why you give this answer. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior_cot': {
        'none': 'Answer the following question based on your internal knowledge and let\'s think step by step.\n 1.Answer the question with one or few words.\n 2.If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}{prediction}',
        'ra': '',
        'tail': '\nAnswer: ',
    },
    'prior_gene':{
        'none': 'Generate a short document that helps answer the following question based on your internal knowledge and answer the question with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}{prediction}',
        'ra': '',
        'tail': '\nAnswer: '
    },
    'post': {
        'none': 'If you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}\nAnswer: {prediction}',
        'ra': 'Given the following information: \n{paras}\nIf you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}\nAnswer: {prediction}',
        'tail': '\nJudgement is: ',
    },
    'post_punish': {
        'none': 'If you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nQuestion: {question}{paras}\nAnswer: {prediction}',
        'ra': 'Given the following information: \n{paras}\nIf you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nQuestion: {question}\nAnswer: {prediction}',
        'tail': '\nJudgement is: ',
    },

}

def get_prompt(sample, args, max_tokens=None, tokenizer=None):
    paras = ""
    ref_key = 'question' if 'question' in sample else 'parent_question'
    if ref_key not in sample:
        if 'Question:' in sample['prompt']:
            question = sample['prompt'].split('Question:')[-1]
            question = question.split('\nAnswer:')[0]
            sample['question'] = question
            ref_key = 'question'
    prompt_template = prompt_dict[args.type]['none'] # prior
    if args.ra != 'none':
        ra_dict = args.ra
        i = 0
        doc = []
        for k, v in ra_dict.items():
            v = min(v, len(sample[k]))
            for j in range(v):
                doc.append(("Passage-%d" % i) + sample[k][j])
                i += 1
        paras = '\n'.join(doc)
        prompt_template = prompt_dict[args.type]['ra']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prediction = sample['Res'] if args.type == 'post' or args.type == 'post_evidence' else ""
    prompt = prompt_template.format(question=sample[ref_key], paras=paras, prediction=prediction) + tail


    if max_tokens:
        # Tokenize the prompt to check length
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        token_count = tokens.size(1)

        # Truncate if the token count exceeds the max_tokens limit
        if token_count > max_tokens:
    
            # Calculate the allowed token count for `paras`
            static_prompt_text = prompt_template.format(question=sample[ref_key], paras="", prediction=prediction) + tail
            static_tokens = tokenizer(static_prompt_text, return_tensors="pt").input_ids[0]
            static_token_count = len(static_tokens)
    
            # Determine the maximum token length for `paras`
            allowed_paras_tokens = max_tokens - static_token_count
    
            # Tokenize `paras` and truncate it to fit within allowed tokens
            paras_tokens = tokenizer(paras, return_tensors="pt").input_ids[0]
            truncated_paras_tokens = paras_tokens[:allowed_paras_tokens]
    
            # Decode truncated `paras` back to text
            truncated_paras = tokenizer.decode(truncated_paras_tokens, skip_special_tokens=True)
    
            # Rebuild the prompt with truncated `paras`
            prompt = prompt_template.format(question=sample[ref_key], paras=truncated_paras, prediction=prediction) + tail
            print(f'the prompt was truncated since the first iter documents tokens = {len(paras_tokens)}')
    return prompt


