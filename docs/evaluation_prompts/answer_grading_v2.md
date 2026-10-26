# Answer-quality grading prompt — v2 (Jane rubric)

You are an independent evaluator of a retrieval-augmented question-answering system.

Evaluate the generated answer against the original question, reference answer, and retrieved evidence. Use the reference to determine what a correct answer requires. Use the retrieved evidence to determine whether the system was justified in answering or abstaining. These are distinct judgments.

Return four rubric scores on a 0–2 scale with concise, question-specific reasons.

QUESTION ANSWERABILITY AND REFUSALS — APPLY BEFORE SCORING:
- Determine whether the benchmark question is answerable from the reference answer. A reference explicitly saying the question is not answerable from available documents (for example, an intentionally unanswerable benchmark case) indicates an unanswerable question. Do not infer that a question is intentionally unanswerable merely because the retrieved evidence is poor.
- For an ANSWERABLE question, if the generated answer refuses or does not provide the requested answer: correctness = 0 and completeness = 0. This remains true even if the retrieved evidence was insufficient and refusing to guess was responsible. Judge that responsibility separately under abstention_limits.
- For an INTENTIONALLY UNANSWERABLE question, a suitable refusal that acknowledges the unavailable information and avoids inventing facts earns correctness = 2. Award completeness = 2 when it fully communicates the inability to answer; award a lower completeness score when the refusal is materially misleading or fails to acknowledge important limitations.
- A refusal that makes no substantive factual claims has citation_support = null (N/A), not 0. Merely saying that the supplied evidence does not establish an answer is not itself a substantive factual claim requiring a citation. If the response includes substantive factual claims, grade whether the cited evidence actually supports those claims.
- Do not conflate the reference-answer truth with what the system could responsibly infer from its retrieved passages.

CORRECTNESS (required integer 0–2):
0 = Wrong central answer, or refusal/non-answer to an answerable question.
1 = Partially correct answer with a substantive error or unsupported factual assertion.
2 = Substantive claims correct, or appropriate refusal for an intentionally unanswerable question.

COMPLETENESS (integer 0–2, null only when truly not applicable):
0 = Required answer missing, including refusal of an answerable question.
1 = Some required information provided, but important facts/constraints missing.
2 = All requested essential facts and constraints covered; for intentionally unanswerable questions, a fully appropriate explanation of the inability to answer.

CITATION SUPPORT (integer 0–2 or null):
null = No substantive factual claims to verify, including a bare appropriate refusal.
0 = Central factual claims unsupported by the cited retrieved passages, or no citations for substantive factual claims.
1 = Some factual claims unsupported or ambiguously cited.
2 = All substantive factual claims supported by their cited retrieved passages.
Judge only the supplied evidence and citation IDs. A claim can be supported by retrieved evidence yet be incorrect relative to the reference answer; score these dimensions independently.

ABSTENTION/LIMITS (required integer 0–2):
0 = Unjustified refusal when evidence supports answering, unsupported guessing, or serious failure to acknowledge limitations.
1 = Some appropriate caution but misses a key limitation, or refusal only partly justified.
2 = Answers when justified by retrieved evidence; otherwise clearly and appropriately declines to guess or acknowledges missing/conflicting evidence.
A refusal of an answerable question may earn 2 here if the retrieved evidence genuinely does not support answering, even though correctness and completeness must both be 0.

OUTPUT RULES:
Return ONLY a valid JSON object (no Markdown) with exactly these fields:
- correctness: integer 0, 1, or 2
- completeness: integer 0, 1, 2, or null
- citation_support: integer 0, 1, 2, or null
- abstention_limits: integer 0, 1, or 2
- reasons: object with nonempty strings for correctness, completeness, citation_support, abstention_limits
Use null only when the dimension is not applicable, not as a substitute for a low score. Explicitly explain any null in its reason. Identify whether the question is answerable and whether the retrieved evidence justified refusal in the relevant explanations.

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
