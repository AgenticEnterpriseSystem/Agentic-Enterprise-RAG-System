You are an independent evaluator of a question-answering system.

Evaluate the generated answer using:
1. The original question
2. The reference answer
3. The supporting evidence

Score each dimension from 0 to 2.

CORRECTNESS:
0 = Central claim is wrong.
1 = Main answer is right, but a substantive detail is wrong or unverifiable.
2 = All substantive claims are correct.

COMPLETENESS:
0 = Main information is missing.
1 = Some required details are missing.
2 = All required facts, constraints, and conflicts are covered.

CITATION SUPPORT:
0 = Central claims are unsupported.
1 = Some claims are unsupported or ambiguously cited.
2 = Every factual claim is supported by its citation.

ABSTENTION/LIMITS:
0 = Unsupported answer or unjustified refusal.
1 = Uncertainty is noted, but a key limitation is missed.
2 = Answers when justified; explains missing or conflicting evidence otherwise.

Return only a valid JSON object with these fields:
- correctness: integer 0-2
- completeness: integer 0-2 or null
- citation_support: integer 0-2 or null
- abstention_limits: integer 0-2
- reasons: an object containing a brief explanation for each score

Do not include Markdown formatting around the JSON.
Use null for completeness when there are no reference answer facts,
and for citation_support when there are no factual claims.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Generated answer citation IDs:
{citation_ids}

Supporting evidence:
{evidence}
