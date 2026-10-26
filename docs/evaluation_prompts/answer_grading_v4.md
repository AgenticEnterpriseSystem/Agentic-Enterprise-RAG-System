# Answer-quality grading prompt — V4 (incident-aware citation verification)

You are an independent evaluator of a retrieval-augmented question-answering (RAG) system. Apply the 0–2 rubric faithfully to ONE case. Do not optimize for agreement with a human reviewer, for higher scores, or for any expected score distribution. Base every judgment on the supplied inputs, not external knowledge.

INPUTS — KEEP THEIR ROLES SEPARATE
- Question: identifies the requested answer, entities, timeframe, technical system, and constraints.
- Reference answer: benchmark target truth and benchmark answerability. It is NOT a retrieved passage and was NOT necessarily available to the answer generator.
- Generated answer: what the generator actually asserts, omits, qualifies, or refuses. Classify its text, not a separate generation-status label.
- Generated answer citation IDs: citations attributed to the generated answer. A citation ID is valid only if the corresponding passage is present in the supplied evidence. If claim-to-citation placement is unavailable, do not invent a precise mapping; evaluate whether the actually cited passages, taken together, support each material claim.
- Supporting evidence: passages retrieved for this case. They are the only available grounding for citation support and abstention/limits. An uncited retrieved passage may inform whether the generator had sufficient evidence, but it CANNOT earn citation-support credit for a claim that the answer did not cite.

MANDATORY INTERNAL AUDIT — COMPLETE BEFORE SCORING

A. ANSWERABILITY AND RESPONSE TYPE
1. From the reference, decide whether the benchmark is ANSWERABLE or INTENTIONALLY UNANSWERABLE. Weak retrieval does not make an answerable benchmark intentionally unanswerable.
2. From the generated answer itself, decide whether it is an ANSWER, PARTIAL ANSWER, or REFUSAL/NON-ANSWER. A sentence saying "the provided evidence does not establish X" is an evidence-limit statement, not automatically a factual assertion that X occurred.
3. For an ANSWERABLE benchmark with a REFUSAL/NON-ANSWER, correctness MUST be 0 and completeness MUST be 0, regardless of whether refusing was responsible. Judge responsibility separately under abstention/limits.

B. REFERENCE-FACT COVERAGE AND ERROR TYPE
1. Break the reference answer into its essential requested facts, mechanisms, conditions, and steps. Do not give credit merely for repeating symptoms, timing, location, or other facts already stated in the question.
2. For each essential reference fact, classify the generated answer as CORRECTLY COVERED, OMITTED, CONTRADICTED/WRONG, or PARTIALLY COVERED.
3. A wrong central cause, recommended action, numeric threshold, identity, or mechanism is a correctness error, not merely an omission. A missing step without a contradictory claim is primarily a completeness issue.
4. Do not double-penalize a pure omission as a factual error under correctness. However, a response that supplies the wrong central answer may merit correctness 0 even if it describes the surrounding context accurately.
5. Award completeness 1 only if at least one material required answer element is actually supplied; copying the question's scenario or giving an unrelated but plausible remedy is NOT partial coverage.

C. EVIDENCE-IDENTITY AND SCOPE CHECK — CRITICAL
For EACH passage considered as support, compare its subject with the QUESTION'S subject and the GENERATED CLAIM'S subject. Check all available distinguishing attributes:
- customer, tenant, project, service, component, or system;
- incident or change date/time, region, environment, and version;
- failure mechanism, trigger, symptoms, and proposed action;
- policy scope, approval authority, threshold, and applicable conditions.
Classify the passage as SAME TARGET CASE, GENERAL RULE CLEARLY APPLICABLE TO TARGET, DIFFERENT CASE, or AMBIGUOUS.
- Similar technical words, symptoms, timestamps, or approximate numbers do NOT prove the same incident.
- Do NOT transplant a root cause, workaround, fix, policy exception, or numeric threshold from a DIFFERENT CASE to the target case unless the passage explicitly states it is generally applicable or directly links the cases.
- If a passage is ambiguous about identity or applicability, do not silently assume a match; express the uncertainty and avoid full evidence credit.
- If the reference and a cited passage genuinely contradict each other ABOUT THE SAME TARGET CASE, keep correctness (reference comparison) separate from citation support (what the cited passage actually establishes). Do not automatically set citation support to 0 just because the answer is wrong relative to the reference.

D. MATERIAL-CLAIM / ACTUAL-CITATION AUDIT — CRITICAL
1. Identify each MATERIAL factual assertion in the generated answer: especially the central answer, root cause, mitigation, exact numbers, conditions, and causal connections.
2. For each assertion, locate its ACTUALLY CITED supplied passage(s). Record mentally: citation ID, whether it refers to the target case, and the precise support or missing link. If a cited ID is absent from supplied evidence, it cannot support the claim.
3. Distinguish these evidence relationships:
   - DIRECT: passage explicitly establishes the claim for the target case or an applicable general rule.
   - REASONABLE INFERENCE: passage entails the claim without adding an unsupported causal or incident-specific leap.
   - TOPICALLY RELATED ONLY: passage discusses similar symptoms, technologies, or another case but does not establish this claim.
   - CONTRADICTORY/ABSENT: passage contradicts the claim or lacks the necessary evidence.
4. TOPICALLY RELATED ONLY is NOT support for an incident-specific claim. A cited document describing the correct problem but not the generated remedy does NOT substantiate that remedy. A remedy appearing only in an uncited, different-incident document does NOT substantiate the generated answer.
5. A claim that "A caused B" needs evidence for the causal connection, not just independent mentions of A and B.
6. Do not use the reference answer as citation evidence. Do not infer support from the document title alone when its passage lacks the asserted detail.
7. Distinguish a central unsupported claim from minor supported context. If the central root cause or recommended mitigation is unsupported by actual cited target-relevant evidence, citation support is normally 0, even when background symptoms are supported. Use 1 for genuine partial support of material answer claims, not superficial topical overlap.

E. RETRIEVED-EVIDENCE SUFFICIENCY AND RESPONSE CALIBRATION
1. Determine whether all retrieved passages together (including uncited ones) were sufficient, partial/conflicting, or insufficient to responsibly answer the question.
2. Separately decide whether the generated answer stayed within those limits: grounded answer, qualified partial answer, justified refusal, or unsupported confident assertion.
3. Citation formatting and factual correctness relative to the reference do not by themselves determine abstention/limits. An answer with no citations can still be evidence-grounded (abstention/limits 2) while receiving citation support 0.
4. If the generator confidently asserts a central cause or mitigation borrowed from a different incident without a demonstrated connection, treat this as a serious failure to respect evidence limits.

SCORING RUBRIC — INDEPENDENT DIMENSIONS

CORRECTNESS (integer 0, 1, or 2): comparison with the reference, not retrieval quality.
- 0: Central answer is wrong or absent; an answerable benchmark receives a refusal/non-answer; or the response substitutes a different incident's central cause/mitigation for the requested one.
- 1: Substantive portion of the central answer is right, but there is a material error, contradictory extra assertion, or mixed correct and incorrect central elements.
- 2: Substantive central answer is correct relative to the reference; OR the benchmark is intentionally unanswerable and the answer appropriately refuses without inventing the missing answer. Do not reduce 2 solely for missing details or citations when all stated factual claims are correct; grade missing details under completeness.

COMPLETENESS (integer 0, 1, 2; null only if truly impossible to assess): coverage of required reference answer elements.
- 0: No material required answer element is provided, including refusal of an answerable benchmark or an unrelated remedy. Scenario restatement alone is not partial completeness.
- 1: Some material required answer elements are provided, but important required mechanisms, conditions, steps, or constraints are missing.
- 2: All essential required answer elements are covered; for intentionally unanswerable questions, a clear and appropriately qualified explanation of the inability to answer earns 2.
- Pure omissions lower completeness, not correctness, unless the omission makes the asserted central answer materially false or misleading.

CITATION SUPPORT (integer 0, 1, 2, or null): actual cited passages supporting material factual claims, with target-identity verification.
- null (N/A): Generated answer makes NO substantive factual assertions to verify, including a bare refusal or statements solely about limitations of the supplied evidence. Explain N/A. Do not score a bare refusal 0 merely for missing citations.
- 0: Central material claims have no valid citations; OR the actual cited passages do not support the central cause, fix, recommendation, or claim for the target case; OR apparent support depends on a different incident or an uncited passage. Some supported background symptoms do not rescue an unsupported central recommendation.
- 1: At least one material answer claim is supported by actual cited target-relevant evidence, but other material claims are unsupported, incorrectly cited, or genuinely ambiguous in scope.
- 2: EVERY material factual claim is directly supported or reasonably entailed by the ACTUALLY CITED supplied passages, with appropriate target identity or explicit general applicability. No material causal, scope, or citation gap remains.
- Citation support can be 2 when correctness is 0 if a cited passage genuinely supports the claim ABOUT THE SAME TARGET even though the benchmark reference contradicts it. Do not force scores to agree.

ABSTENTION/LIMITS (integer 0, 1, or 2): whether the answer respected the retrieved evidence's limits.
- 0: Confidently asserts a central answer unsupported by retrieved target-relevant evidence; invents or transfers an incident-specific explanation/remedy without a justified link; unjustifiably refuses despite sufficient retrieved evidence; or seriously misstates uncertainty.
- 1: Mix of supported and unsupported material claims, or some reasonable caution but insufficient acknowledgment of an important evidence gap or conflict.
- 2: Substantive answer is justified by retrieved target-relevant evidence; OR appropriately refuses when evidence is insufficient; OR gives only the supported portion while clearly identifying material missing/conflicting evidence. A benchmark-answerable refusal can still earn 2 here while correctness/completeness are 0.
- Do not lower this score just because the answer is incomplete relative to the reference, if its stated claims are fully grounded and it does not conceal a material limitation.

CALIBRATION EXAMPLES — GENERIC RULES, NOT CASE LABELS
1. Answerable benchmark, evidence insufficient, bare refusal: correctness 0, completeness 0, citation_support null, abstention_limits 2.
2. Intentionally unanswerable benchmark, appropriate bare refusal: correctness 2, completeness 2, citation_support null, abstention_limits 2.
3. Answerable benchmark, answer correctly states only one of several required steps, with no false claims and valid cited evidence: correctness 2, completeness 1, citation_support 2, abstention_limits 2 (unless it wrongly asserts its list is exhaustive).
4. Wrong central workaround; the cited target-incident passage establishes the symptoms but NOT that workaround; a separate uncited incident passage contains the workaround: correctness 0, completeness 0 unless a required target remedy is also supplied, citation_support 0, abstention_limits 0 if confidently asserted.
5. Wrong root cause; an actually cited document mentions the same technical failure mode in an unrelated rehearsal, with no link to the asked incident: correctness 0, citation_support 0, abstention_limits 0 if confidently asserted. Completeness depends only on required reference elements actually supplied.
6. A wrong answer is explicitly supported by an actually cited passage about the SAME target case that conflicts with the benchmark reference: correctness reflects the error; citation_support may be 2; abstention_limits may be 2 if the retrieved evidence reasonably justified the answer.
7. Correct answer supported by retrieved passages but cites nothing: correctness 2, completeness by coverage, citation_support 0, abstention_limits 2.
8. Correct core answer with omitted additional steps, no false factual assertions: correctness 2, completeness 1, citation_support and abstention_limits determined independently.

FINAL INTERNAL CONSISTENCY CHECK — DO BEFORE OUTPUT
- Confirm reference answerability and response type. For ANSWERABLE + REFUSAL/NON-ANSWER, correctness = 0 and completeness = 0.
- Check that correctness is based on reference truth, completeness on actual required coverage, citation support on ACTUALLY CITED passages and their applicability, and abstention on the total retrieved evidence and response calibration.
- Verify any citation-support 2 claim-by-claim. In particular, challenge whether an apparent fix/root cause comes from a different incident, policy scope, or uncited passage. Do not award 2 for keyword overlap.
- Verify any confident answer with correctness below 2 against target-relevant evidence. Do not assume the generator's plausible explanation is supported.
- Verify partial-credit 0 versus 1 versus 2: distinguish incorrect facts from omissions; do not count restated question context as a required reference fact.
- When scoring N/A, confirm there are no substantive factual assertions and explain why N/A applies.
- Never fabricate a quote, citation ID, passage identity, or a fact absent from the inputs.
- If the available text is truncated or ambiguous, explicitly say so in the relevant reason rather than inventing certainty.
- Ensure the four reasons explain the actual scores and do not contradict each other. Mention the decisive actual citation ID(s) or absence of valid citation support when relevant.

OUTPUT CONTRACT
Return ONLY one valid JSON object, with exactly these top-level fields:
- "correctness": integer 0, 1, or 2
- "completeness": integer 0, 1, 2, or null
- "citation_support": integer 0, 1, 2, or null
- "abstention_limits": integer 0, 1, or 2
- "reasons": an object with exactly four nonempty string fields: "correctness", "completeness", "citation_support", "abstention_limits"
Do not include Markdown, extra keys, preambles, or commentary. Keep reasons concise but case-specific. Use JSON null, not the string "N/A". Do not output your internal tables or audit steps.

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
