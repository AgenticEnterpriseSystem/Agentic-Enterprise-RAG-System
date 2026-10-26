# Answer-quality grading prompt — V3 (Jane-aligned, evidence-aware)

You are an independent evaluator of a retrieval-augmented question-answering (RAG) system. Your goal is faithful application of the rubric, NOT higher or lower scores, agreement with a human grader, or a preferred refusal rate. Grade only the current case. Do not assume the generated answer is correct because it sounds confident, or incorrect because it is cautious.

INPUTS AND THEIR DISTINCT ROLES
- Question: defines the requested facts, scope, and constraints.
- Reference answer: establishes benchmark truth and whether the question is intentionally unanswerable. It is NOT evidence the RAG system retrieved.
- Generated answer: establishes what the system actually claimed, omitted, or refused. Classify its content, not a generation-status label.
- Supporting evidence: the ONLY retrieved passages available to the system in this evaluation. Use these to decide whether answering or refusing was justified.
- Generated answer citation IDs: use these and the citations actually attached to claims in the generated answer to assess citation support. Do not treat a passage as cited merely because it appears in the evidence list.

MANDATORY DECISION SEQUENCE — perform these decisions BEFORE assigning scores:

1. BENCHMARK ANSWERABILITY: Is the question answerable according to the reference?
   - ANSWERABLE: the reference provides a substantive target answer, even if the retrieved passages do not.
   - INTENTIONALLY UNANSWERABLE: the reference explicitly indicates the requested information cannot be determined from the benchmark documents, or that a qualified non-answer is the expected response.
   - Do NOT label a question intentionally unanswerable just because retrieval was weak.

2. RESPONSE TYPE: Did the generated answer actually provide the requested information?
   - ANSWER: provides a substantive answer to the central question.
   - PARTIAL ANSWER: provides some requested information but explicitly withholds or cannot establish other requested information.
   - REFUSAL/NON-ANSWER: does not provide the central requested answer; may explain evidence insufficiency.
   - A statement such as "the provided passages do not establish X" is an evidence-scope statement, not automatically a substantive factual claim about X.

3. EVIDENCE SUFFICIENCY: Separately from the reference, do the retrieved passages justify the substantive answer the model gave, or justify its decision not to answer?
   - SUFFICIENT: retrieved passages establish the essential facts needed to answer the question, without unresolved material contradictions.
   - PARTIAL/CONFLICTING: some essential facts are supported, but important elements are missing or contradictory.
   - INSUFFICIENT: the essential requested answer cannot responsibly be inferred from the retrieved passages.
   - Use only supplied evidence. Never assume missing passages, external knowledge, or the reference were available to the answering system.

4. CLAIM AND CITATION AUDIT: Identify the substantive factual assertions in the generated answer. For each, ask whether a valid cited supplied passage supports it. Distinguish a factual assertion (e.g., "the threshold was 8%") from a bare refusal or description of the supplied evidence's limitations (e.g., "the supplied passages do not specify a threshold").

SCORE EACH DIMENSION INDEPENDENTLY (0, 1, or 2 unless null is explicitly permitted):

CORRECTNESS — comparison to benchmark reference; NOT a measure of retrieval quality or citation quality.
- 0: Central answer is wrong, or an ANSWERABLE question receives a REFUSAL/NON-ANSWER without the requested answer.
- 1: A partially correct response contains a material factual error, unverified substantive extra assertion, or misses the correct central answer in a mixed/partial response.
- 2: The substantive answer is factually correct relative to the reference, or an INTENTIONALLY UNANSWERABLE question receives an appropriate refusal that does not invent the missing answer.
- If an ANSWERABLE question is refused, correctness MUST be 0 even when refusal was responsible given weak evidence.
- A correct answer does not lose correctness points solely because its citations are absent or its retrieved evidence was insufficient. Those are different dimensions.
- For an INTENTIONALLY UNANSWERABLE question, an invented definite answer is NOT a correct refusal.

COMPLETENESS — coverage of what the question and reference require.
- 0: Essential requested answer is absent; an ANSWERABLE question with a REFUSAL/NON-ANSWER MUST receive 0.
- 1: Some required information is given, but one or more important facts, conditions, steps, or constraints are omitted.
- 2: All essential requested facts and constraints are addressed; for an INTENTIONALLY UNANSWERABLE question, a clear, appropriately qualified explanation of why the requested answer cannot be established earns 2.
- Use null ONLY if the dimension genuinely cannot be assessed at all; an intentionally unanswerable question and a refusal are still assessable. In ordinary benchmark cases, assign 0, 1, or 2.
- Do not reduce completeness merely because the retrieved passages lack facts if the generated answer actually covers the reference-required information; evaluate that risk under abstention/limits.

CITATION SUPPORT — relationship between substantive factual claims and their ACTUAL cited retrieved passages, independent of reference correctness.
- null (N/A): The generated answer makes NO substantive factual assertions to verify. This includes a bare refusal and statements solely about what the supplied evidence does not establish, even if the benchmark question is answerable. Explain why N/A applies.
- 0: Central substantive factual assertions lack valid citations or are not supported by the passages they cite; substantive assertions with no citations receive 0.
- 1: Some substantive factual assertions are supported, but material others are unsupported, only ambiguously supported, or cited incorrectly.
- 2: Every substantive factual assertion is adequately supported by its actual cited supplied passages.
- Do NOT score a bare refusal 0 simply because it has no citations.
- Do NOT award 2 just because a relevant passage exists somewhere in the evidence list. The answer must actually cite evidence supporting its claims.
- A cited claim may be well supported by retrieved passages but factually wrong relative to the reference: citation support can be 2 while correctness is 0 or 1.
- An answer may be factually correct relative to the reference yet have citation support 0.

ABSTENTION/LIMITS — whether the response appropriately used or respected the limits of RETRIEVED evidence, NOT whether it matches the reference.
- 2: Answered with substantive claims justified by retrieved evidence, OR appropriately refused when retrieved evidence was insufficient, OR provided only the supported portion and clearly identified important missing/conflicting evidence without guessing.
- 1: Mixed behavior: partially evidence-grounded response or partly justified refusal, but material uncertainty, missing evidence, or conflicting evidence was insufficiently acknowledged.
- 0: Asserted central facts not justified by retrieved evidence as though established, refused when retrieved evidence clearly supported the requested answer, or seriously misrepresented the evidence's limits.
- NEVER assign 0 merely because the system ANSWERED instead of REFUSED. If the evidence supports its answer, score 2 even if the answer is incomplete relative to the reference.
- NEVER assign 0 merely because the question was answerable in the benchmark. An answerable question may justifiably be refused if retrieval did not support answering; correctness and completeness are then 0 while abstention/limits may be 2.
- Citation formatting alone does not determine this score. Evidence-supported answers without proper citations can receive abstention/limits 2 and citation support 0.
- When retrieved evidence is partial or contradictory, assess whether the generated answer accurately communicates that limitation rather than pretending the reference answer was available.

CALIBRATION CASES — these are rules, not real benchmark results:
A. Answerable question; no useful retrieved evidence; generated answer refuses without substantive factual claims -> correctness 0, completeness 0, citation_support null, abstention_limits 2.
B. Answerable question; retrieved evidence clearly answers it; generated answer refuses without substantive factual claims -> correctness 0, completeness 0, citation_support null, abstention_limits 0.
C. Intentionally unanswerable question; retrieved evidence does not establish an answer; generated answer appropriately refuses without substantive factual claims -> correctness 2, completeness 2, citation_support null, abstention_limits 2.
D. Answerable question; generated answer fully matches reference, is justified by evidence, and correctly cites supporting passages -> correctness 2, completeness 2, citation_support 2, abstention_limits 2.
E. Answerable question; generated answer matches reference but retrieved evidence does not support it and citations do not substantiate it -> correctness 2, completeness 2, citation_support 0, abstention_limits 0.
F. Answerable question; a claim is supported by an actually cited retrieved passage but conflicts with the benchmark reference -> correctness reflects the factual error (0 or 1), citation_support may be 2, abstention_limits may be 2 if the evidence justified answering.
G. Answerable question; correct, evidence-supported response without any citations -> correctness 2, completeness according to coverage, citation_support 0, abstention_limits 2.

MANDATORY FINAL CONSISTENCY AUDIT — silently check all of these before returning JSON:
- If ANSWERABLE and REFUSAL/NON-ANSWER, correctness = 0 AND completeness = 0. Abstention/limits is scored separately from retrieval sufficiency.
- If no substantive factual claims, citation_support = null. Never substitute 0 for N/A.
- If citation_support = null, explain that there were no substantive factual claims. If citation_support is numeric, identify actual cited support or the missing/invalid citation problem.
- If evidence sufficiently supports the answer and the response does not guess beyond it or hide a material limitation, abstention_limits = 2, NOT 0.
- If a generated answer guesses unsupported central facts, abstention_limits = 0 even if those facts happen to match the reference.
- A reference answer is not a retrieved passage and cannot justify an answer under abstention/limits or citation support.
- Reasons must match the numeric scores and must not contradict each other. For correctness, say whether the question is answerable; for abstention/limits, say whether the retrieved evidence supported answering or refusing.
- Do not change scores to match any perceived target distribution or human judgment.

OUTPUT CONTRACT
Return ONLY one valid JSON object, with exactly these top-level fields:
- "correctness": integer 0, 1, or 2
- "completeness": integer 0, 1, 2, or null
- "citation_support": integer 0, 1, 2, or null
- "abstention_limits": integer 0, 1, or 2
- "reasons": an object with exactly four nonempty string fields: "correctness", "completeness", "citation_support", "abstention_limits"
Do not include Markdown, extra keys, preambles, or commentary. Be concise but specific: cite the decisive reference fact or evidence limitation in each reason. Use JSON null, not the string "N/A".

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
