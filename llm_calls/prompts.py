DEFAULT_PROMPT_TEMPLATE = """
Statement 1: {text1}
Statement 2: {text2}

Question: Do the ideas entailed in Statement 1 imply the ideas entailed in Statement 2? Answer with YES or NO, state a brief reasoning for your answer, then give a score from 1 to 10 evaluating how confident you are with your answer (1 means 'I am absolutely not confident about my answer' and 10 means 'I am completely sure about my assessment'), and give a brief comment on your level of certainty on your answer.

Please provide your response as a valid JSON object in the following format:
{{
  "answer": "Your YES or NO answer goes here (as a string)",
  "reasoning": "The reasoning behind each of your answers goes here (as a string)",
  "score": "A score between 0 and 10 based on your confidence (as an integer)",
  "comment": "Additional comments on your confidence here (as a string)"
}}
"""

TEST_PROMPT_TOT_JSON_BATCH = """
You are working on assessing whether statements made in certain contexts entail one another. In natural language, logical implication makes sense in context. For example, if a statement speaks about a ruler and another one speaks about a king, they may be talking about the same figure of authority, if both statements talk about or within a monarchy. Also, from the entire argument and context, one can deduce the author's perspective and specific use of terms within a topic, which is important to determine whether two ideas are entailed, contradictory, or unrelated.

So please, make sure to take into account the entire argument and its context when making your assessments. Also, it is important not to overstep out of the context, nor being too strict with the logic, when thinking of counterexamples.

Now, this is a brief context for Argument 1:

Context 1: {context_1}

Please consider the following argument:

Argument 1: {argument_1}

This is a brief context for Argument 2:
Context 2: {context_2}

Consider then the following argument:

Argument 2: {argument_2}

From these arguments, we can infer the following statements, respectively.

Statement 1: {text1}
Statement 2: {text2}

Question: Considering both arguments, their historical context, and authors' perspectives, does Statement 1 entail Statement 2? Give 3 YES or NO answers, and state a brief reasoning for each of your answers (provide different reasons per answer) and support them with evidence from the arguments, context, and any background knowledge you have explicitly. Then, review your answers and reasons and give a single final answer, YES or NO, and give a score from 0 to 4 evaluating how confident you are with your final answer (0 means 'I am absolutely not confident about my answer' and 4 means 'I am completely sure about my assessment'). Finally, please provide a brief comment on your level of certainty with your answer.

IMPORTANT: you are assessing the relationship between the STATEMENTS, NOT the ARGUMENTS. Two arguments can express opposing or contradictory views, but share similar or equivalent premises.

Please provide your response as a valid JSON object in the following format:

{{
  "sentence_id_1": {sentence_id_1},
  "sentence_id_2": {sentence_id_2},
  "answers": "Your 3 YES or NO answers go here (as a string)",
  "reasoning": "The reasoning behind each of your answers go here (as a string)",
  "score": "A score between 0 and 4 based on your confidence (as an integer)",
  "comment": "Additional comments on your confidence here (as a string)",
  "final_answer": "The final YES or NO answer/conclusion here (as a string)"
}}

Ensure the response is strictly a valid JSON object with no extra characters or formatting.
"""

PROMPT_TOT_DEEPSEEK = """
You are working on assessing whether statements made in certain contexts entail one another. In natural language, logical implication makes sense in context. For example, if a statement speaks about a ruler and another one speaks about a king, they may be talking about the same figure of authority, if both statements talk about or within a monarchy. Also, from the entire argument and context, one can deduce the author's perspective and specific use of terms within a topic, which is important to determine whether two ideas are entailed, contradictory, or unrelated.

So please, make sure to take into account the entire argument and its context when making your assessments. Also, it is important not to overstep out of the context, nor being too strict with the logic, when thinking of counterexamples.

Now, this is a brief context for Argument 1:

Context 1: {context_1}

Please consider the following argument:

Argument 1: {argument_1}

This is a brief context for Argument 2:
Context 2: {context_2}

Consider then the following argument:

Argument 2: {argument_2}

From these arguments, we can infer the following statements, respectively.

Statement 1: {text1}
Statement 2: {text2}

Question: Considering both arguments, their historical context, and authors' perspectives, does Statement 1 entail Statement 2? Give 3 YES or NO answers, and state a brief reasoning for each of your answers (provide different reasons per answer) and support them with evidence from the arguments, context, and any background knowledge you have explicitly. Then, review your answers and reasons and give a single final answer, YES or NO, and give a score from 0 to 4 evaluating how confident you are with your final answer (0 means 'I am absolutely not confident about my answer' and 4 means 'I am completely sure about my assessment'). Finally, please provide a brief comment on your level of certainty with your answer.

IMPORTANT: you are assessing the relationship between the STATEMENTS, NOT the ARGUMENTS. Two arguments can express opposing or contradictory views, but share similar or equivalent premises.

Please provide your response as a valid JSON object in the following format:

{{
  "sentence_id_1": {sentence_id_1},
  "sentence_id_2": {sentence_id_2},
  "answers": "Your 3 YES or NO answers go here (as a string)",
  "reasoning": "The reasoning behind each of your answers go here (as a string)",
  "score": "A score between 0 and 4 based on your confidence (as an integer)",
  "comment": "Additional comments on your confidence here (as a string)",
  "final_answer": "The final YES or NO answer/conclusion here (as a string)"
}}

Ensure the response is strictly a valid JSON object with no extra characters or formatting.
"""

PROMPT_TOT_DEEPSEEK_BB = """
You are working on assessing whether statements made in certain contexts entail one another. In natural language, logical implication makes sense in context. For example, if a statement speaks about a ruler and another one speaks about a king, they may be talking about the same figure of authority, if both statements talk about or within a monarchy. Also, from the entire argument and context, one can deduce the author's perspective and specific use of terms within a topic, which is important to determine whether two ideas are entailed, contradictory, or unrelated.

So please, make sure to take into account the entire argument and its context when making your assessments. Also, it is important not to overstep out of the context, nor being too strict with the logic, when thinking of counterexamples.

Now, this is a brief context for Argument 1:

Context 1: {context_1}

Please consider the following argument:

Argument 1: {argument_1}

Consider then the following argument:

Argument 2: {argument_2}

From these arguments, we can infer the following statements, respectively.

Statement 1: {text1}
Statement 2: {text2}

Question: Considering both arguments, their historical context, and authors' perspectives, does Statement 1 entail Statement 2? Give 3 YES or NO answers, and state a brief reasoning for each of your answers (provide different reasons per answer) and support them with evidence from the arguments, context, and any background knowledge you have explicitly. Then, review your answers and reasons and give a single final answer, YES or NO, and give a score from 0 to 4 evaluating how confident you are with your final answer (0 means 'I am absolutely not confident about my answer' and 4 means 'I am completely sure about my assessment'). Finally, please provide a brief comment on your level of certainty with your answer.

IMPORTANT: you are assessing the relationship between the STATEMENTS, NOT the ARGUMENTS. Two arguments can express opposing or contradictory views, but share similar or equivalent premises.

Please provide your response as a valid JSON object in the following format:

{{
  "sentence_id_1": {sentence_id_1},
  "sentence_id_2": {sentence_id_2},
  "answers": "Your 3 YES or NO answers go here (as a string)",
  "reasoning": "The reasoning behind each of your answers go here (as a string)",
  "score": "A score between 0 and 4 based on your confidence (as an integer)",
  "comment": "Additional comments on your confidence here (as a string)",
  "final_answer": "The final YES or NO answer/conclusion here (as a string)"
}}

Ensure the response is strictly a valid JSON object with no extra characters or formatting.
"""

PROMPT_TOT_DEEPSEEK_SB = """
You are working on assessing whether statements made in certain contexts entail one another. In natural language, logical implication makes sense in context. For example, if a statement speaks about a ruler and another one speaks about a king, they may be talking about the same figure of authority, if both statements talk about or within a monarchy. Also, from the entire argument and context, one can deduce the author's perspective and specific use of terms within a topic, which is important to determine whether two ideas are entailed, contradictory, or unrelated.

So please, make sure to take into account the entire argument and its context when making your assessments. Also, it is important not to overstep out of the context, nor being too strict with the logic, when thinking of counterexamples.

Now, please consider the following:

This is a brief context for Argument 1:
Context 2: {context_1}

Argument 1: {argument_1}

Consider then the following argument:

Argument 2: {argument_2}

From these arguments, we can infer the following statements, respectively.

Statement 1: {text1}
Statement 2: {text2}

Question: Considering both arguments, their historical context, and authors' perspectives, does Statement 1 entail Statement 2? Give 3 YES or NO answers, and state a brief reasoning for each of your answers (provide different reasons per answer) and support them with evidence from the arguments, context, and any background knowledge you have explicitly. Then, review your answers and reasons and give a single final answer, YES or NO, and give a score from 0 to 4 evaluating how confident you are with your final answer (0 means 'I am absolutely not confident about my answer' and 4 means 'I am completely sure about my assessment'). Finally, please provide a brief comment on your level of certainty with your answer.

IMPORTANT: you are assessing the relationship between the STATEMENTS, NOT the ARGUMENTS. Two arguments can express opposing or contradictory views, but share similar or equivalent premises.

Please provide your response as a valid JSON object in the following format:

{{
  "sentence_id_1": {sentence_id_1},
  "sentence_id_2": {sentence_id_2},
  "answers": "Your 3 YES or NO answers go here (as a string)",
  "reasoning": "The reasoning behind each of your answers go here (as a string)",
  "score": "A score between 0 and 4 based on your confidence (as an integer)",
  "comment": "Additional comments on your confidence here (as a string)",
  "final_answer": "The final YES or NO answer/conclusion here (as a string)"
}}

Ensure the response is strictly a valid JSON object with no extra characters or formatting.
"""

PROMPT_TOT_DEEPSEEK_BS = """
You are working on assessing whether statements made in certain contexts entail one another. In natural language, logical implication makes sense in context. For example, if a statement speaks about a ruler and another one speaks about a king, they may be talking about the same figure of authority, if both statements talk about or within a monarchy. Also, from the entire argument and context, one can deduce the author's perspective and specific use of terms within a topic, which is important to determine whether two ideas are entailed, contradictory, or unrelated.

So please, make sure to take into account the entire argument and its context when making your assessments. Also, it is important not to overstep out of the context, nor being too strict with the logic, when thinking of counterexamples.

Now, please consider the following argument:

Argument 1: {argument_1}

This is a brief context for Argument 2:
Context 2: {context_2}

Consider then the following argument:

Argument 2: {argument_2}

From these arguments, we can infer the following statements, respectively.

Statement 1: {text1}
Statement 2: {text2}

Question: Considering both arguments, their historical context, and authors' perspectives, does Statement 1 entail Statement 2? Give 3 YES or NO answers, and state a brief reasoning for each of your answers (provide different reasons per answer) and support them with evidence from the arguments, context, and any background knowledge you have explicitly. Then, review your answers and reasons and give a single final answer, YES or NO, and give a score from 0 to 4 evaluating how confident you are with your final answer (0 means 'I am absolutely not confident about my answer' and 4 means 'I am completely sure about my assessment'). Finally, please provide a brief comment on your level of certainty with your answer.

IMPORTANT: you are assessing the relationship between the STATEMENTS, NOT the ARGUMENTS. Two arguments can express opposing or contradictory views, but share similar or equivalent premises.

Please provide your response as a valid JSON object in the following format:

{{
  "sentence_id_1": {sentence_id_1},
  "sentence_id_2": {sentence_id_2},
  "answers": "Your 3 YES or NO answers go here (as a string)",
  "reasoning": "The reasoning behind each of your answers go here (as a string)",
  "score": "A score between 0 and 4 based on your confidence (as an integer)",
  "comment": "Additional comments on your confidence here (as a string)",
  "final_answer": "The final YES or NO answer/conclusion here (as a string)"
}}

Ensure the response is strictly a valid JSON object with no extra characters or formatting.
"""