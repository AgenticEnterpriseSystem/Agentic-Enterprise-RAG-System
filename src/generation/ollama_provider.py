
"""Ollama Cloud adapter for RAG generation and answer evaluation."""

from ollama import Client

from src.contracts import Usage
from src.generation.generator import Completion, ProviderError


GENERATION_RULES = """
STRICT OUTPUT RULES:
Return exactly one JSON object with these keys:
status, text, citation_ids.

The status MUST be exactly 'answered' or 'insufficient_evidence'.
NEVER use 'success', 'complete', or 'ok' as a status.

When status is 'answered', citation_ids MUST contain at least
one ID copied exactly from the supplied evidence.

When evidence is insufficient, use 'insufficient_evidence'
and explain the limitation.

Do not invent citation IDs.

IMPORTANT: Every citation_id must be copied CHARACTER FOR CHARACTER
from a citation_id field in the supplied evidence JSON.

Citation IDs may contain colons and passage offsets, such as :0:3000.
You MUST preserve those suffixes exactly.

Never shorten a citation ID to just the document ID.
Before returning your answer, verify that every citation_id
exactly matches an ID in the supplied evidence.
"""


class OllamaProvider:
    def __init__(self, model="gpt-oss:20b-cloud"):
        self.model = model
        self.client = Client(host="http://localhost:11434")

    def complete(self, system: str, prompt: str) -> Completion:
        # Evaluation has its own JSON schema and scoring instructions.
        # Add citation rules only when generating RAG answers.
        if "independent answer-quality evaluator" in system:
            system_message = system
        else:
            system_message = system + "\n" + GENERATION_RULES

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0},
            )
        except Exception as error:
            raise ProviderError(retryable=False) from error

        return Completion(
            text=response.message.content or "",
            usage=Usage(
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
                api_calls=1,
                estimated_cost_usd=None,
            ),
        )
