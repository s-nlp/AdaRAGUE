TRUTHFULQA_INITIAL_ANSWER_TEMPLATE = """When responding to my question, please first evaluate the validity of the information or the assumption underlying the question. Once you've established its truth or existence, then proceed to deliver a detailed explanation or answer. Prioritize accuracy and fact-checking before diving into elaboration or conjecture
My Question: {question}
Your answer:"""

TRUTHFULQA_INITIAL_ANSWER_TEMPLATE_ZH = """在回答我的问题时，请先评估问题所依据的信息或假设的有效性。一旦你确认其真实性或存在性，再进行详细的解释或回答。在深入阐述或推测之前，请优先确保准确性和事实核实。

我的问题：{question}

你的回答：
"""

TRUTHFULQA_FINAL_ANSWER_TEMPLATE = """Based on the above thought process, provide brief and direct answers to the above questions. You must directly answer the question and do not involve too much useless information.
Your Short Answer:"""

TRUTHFULQA_FINAL_ANSWER_TEMPLATE_ZH = """根据上述思考过程，对上述问题提供简短而直接的答案。 你必须使用一句话直接回答问题，不要添加太多无用的信息。

你的简短回答:"""

SEMANTICALLY_EQUIVALENT_PERTERBATIONS_TEMPLATE = """For the question [QUERIED QUESTION], provide {k} semantically equivalent questions.
QUERIED QUESTION: {question}
SEMANTICALLY EQUIVALENT PERTERBATIONS:
"""

SEMANTIC_EQUIVALENCE_CHECK_TEMPLATE = """Are the following two inputs semantically equivalent? Provide your best guess. Given ONLY the guess (True or False), no other words or explanation.
[QUERIED INPUT]: {queried_input}
[GENERATED INPUT]: {generated_input}
Your Answer:"""


CROSS_CONSISTENCY_CHECK_TEMPLATE = """Given the question Q, and two potential answers: answer A in English and answer B in Chinese. Your task is to determine if the content and meaning of A and B are equivalent in the context of answering Q. Consider linguistic nuances, cultural variations, and the overall conveyance of information. Respond with a binary classification. If A and B are equivalent, output 'True.', otherwise output 'False'

Question: {q}
Answer A (in English): {a1}
Answwer B (in Chinese): {a2}
Your response (True or False):"""

CROSS_MODEL_CONSISTENCY_CHECK_TEMPLATE = """Are the following two Question-Answer(QA) pairs semantically equivalent? Provide your best guess that it is correct (True or False). Given ONLY the guess (True or False), no other words or explanation. 

The first QA pair is:
Question: {q}
Answer: {a1}

The second QA pair is:
Question: {q}
Answer: {a2}

Your Guess (True or False):"""

TRUTHFULQA_REPAIR_HALLUCINATION_TEMPLATE = """You are an advanced AI with the ability to process and synthesize information efficiently. When provided with a question, you should consider the following in your response:

1. The specific question asked: {question}
2. The reasoning process: {initial_long_answer}
3. The short answer previously given related to the question: {initial_short_answer}
4. Evidences that have been retrieved: {evidences}

Use this information to generate a concise, accurate, and relevant answer that reflects the latest understanding of the topic based on the input provided. Your response should be clear, direct, and provide the most up-to-date information available while maintaining coherence with the previous discussion points.

Please provide your revised short response:"""

ENTAILMENT_TEMPLATE = """You are given a piece of text. Your task is to identify whether the text is factual or not.
When you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences can be helpful.
The following is the given text:
[text]: {response}
The following is the provided evidences:
[evidences]: {information}
You are forced to answer with True or False. Do not include any explanations.
[Your response]:"""


STRATEGYQA_CHAIN_OF_THOUGHT_TEMPLATE = """Answer the following StrategyQA question. Use the provided guidelines to ensure a comprehensive response:

Question: {question}

Guidelines for Answering:
1. Question Analysis:
- Identify the key components of the question.
- Determine if the question has any implicit assumptions or requires understanding of unstated information.
2. Decomposition:
- Break down the question into smaller, manageable parts.
- For each part, outline what information or type of reasoning is required.
3. Knowledge Integration:
- Access external knowledge sources relevant to the question.
- Integrate this knowledge with the information provided in the question.
4. Inference and Reasoning:
- Employ logical reasoning to make connections between different pieces of information.
- Make inferences where necessary to fill in gaps in information.
5. Answer Synthesis:
- Combine the insights from the previous steps to formulate a concise and accurate answer.
- Ensure that the answer addresses the core of the question and is backed by logical reasoning.
6. Answer Validation:
- Cross-check the answer for consistency and accuracy with respect to the question and the information gathered.
- Adjust the answer if necessary based on this validation.
7. Output:
- Present the final answer clearly and confidently.
- Please note that you must give a definite response and do not answer the question ambiguously. 
- If applicable, briefly outline the key reasoning steps that led to the answer.

Your final answer:"""

STRATEGYQA_CHAIN_OF_THOUGHT_TEMPLATE_ZH = """回答以下 StrategyQA 问题。 请根据提供的指南来逐步思考，以确保生成正确的回复：

问题：{question}

回答指南：
1.问题分析：
- 确定问题的关键组成部分。
- 确定问题是否有任何隐含的假设
2、分解：
- 将问题分解为较小的、易于管理的部分。
- 对于每个部分，概括需要哪些信息
3、知识整合：
- 访问与问题相关的外部知识源。
- 将这些知识与问题中提供的信息相结合。
4. 推理与推理：
- 利用逻辑推理在不同信息之间建立联系。
- 必要时进行推断以填补信息空白。
5.答案综合：
- 结合前面步骤的思考过程来制定简洁而准确的答案。
- 确保答案解决问题的核心并有逻辑推理支持。
6.答案验证：
- 交叉检查答案与问题和收集的信息的一致性和准确性。
- 如有必要，根据此验证调整答案。
7. 输出：
- 清晰、自信地给出最终答案。
- 请注意，您必须给出明确的答复，不要含糊地回答问题。
- 如果适用，请简要概述得出答案的关键推理步骤。

您的最终答案："""

STRATEGYQA_FINAL_ANSWER_TEMPLATE = """Based on the above thought process, provide the final answers to the question: {question}. You must directly answer with True or False, and do not involve any other explanations.
Your Fianl Answer (True or False):"""

STRATEGYQA_FINAL_ANSWER_TEMPLATE_ZH = """根据上述思考过程，对问题【{question}】提供简短而直接的答案。 你必须直接使用是或否来回答问题，不要包含其他无关的内容，不要解释你的回答。

你的回答（是或否）:"""

STRATEGYQA_SEMANTICALLY_EQUIVALENT_PERTERBATIONS_TEMPLATE = """For the following StrategyQA question [QUERIED QUESTION], please understand the question: ensure you fully comprehend what the question is asking, including any implicit assumptions or requirements, and then provide {k} semantically equivalent questions.

QUERIED QUESTION: {question}

SEMANTICALLY EQUIVALENT PERTERBATIONS:"""

STRATEGYQA_RETRIEVAL_REFORMULATION_TEMPLATE = """Transform the following StrategyQA question into a query suitable for information retrieval and then use the query to find relevant evidence. Follow these steps for a short but comprehensive approach:

Original Question: {question}

Rewrite the Question for Retrieval:
1. Simplify the question into key terms and concepts.
2. Remove any elements that are not essential for a search query.
3. If necessary, rephrase the question to focus on the most critical aspects for retrieval.

Your Short Rephrased Query:"""


STRATEGYQA_REPAIR_HALLUCINATION_TEMPLATE = """You are an advanced AI with the ability to process and synthesize information efficiently. When provided with a question, you should consider the following in your response:

1. The specific question asked: {question}
2. The reasoning process: {initial_long_answer}
3. The short answer previously given related to the question: {initial_short_answer}
4. Evidences that have been retrieved: {evidences}

Use this information to provide a concise, accurate, and relevant answer that reflects the latest understanding of the topic based on the input provided. Your response should be clear, direct, and provide the most up-to-date information available while maintaining coherence with the previous discussion points.

You must directly answer the above question with True or False, and do not involve any other explanations.

Your final answer (True or False):"""
