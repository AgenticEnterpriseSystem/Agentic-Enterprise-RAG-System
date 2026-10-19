"""Shared defaults. Model and budget choices must be recorded with each run."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
# Full corpus source: the upstream generator repository at a pinned commit.
CORPUS_REPO_URL = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git"
CORPUS_COMMIT = "d36685e273713975ee20299bbf1ab64165575b3c"
CORPUS_DIR = ROOT / ".cache" / "corpus"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    corpus_commit: str = CORPUS_COMMIT
    seed: int = 5980
    top_k: int = Field(default=10, ge=1, le=100)
    max_search_rounds: int = Field(default=3, ge=1, le=10)
    max_context_chars: int = Field(default=24000, ge=1)
    timeout_seconds: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=8000, ge=1)
    max_cost_usd: float = Field(default=0, ge=0)  # Paid calls disabled by default.


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# BGE v1.5 English models share the query prefix and 512-token limit. Revisions not
# pinned here are resolved once when an index is built and stored in its manifest.
EMBEDDING_MODELS = {
    "bge-small": EMBEDDING_MODEL,
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}
EMBEDDING_REVISIONS = {EMBEDDING_MODEL: EMBEDDING_REVISION}


def runtime_versions():
    """Record installed inference/data libraries without importing model stacks."""
    import importlib.metadata
    import platform

    packages = {}
    for name in ("numpy", "bm25s", "pyarrow", "sentence-transformers",
                 "faiss-cpu", "pydantic", "anthropic", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages}
