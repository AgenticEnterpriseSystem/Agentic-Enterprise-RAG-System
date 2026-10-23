"""Shared defaults. Model and budget choices must be recorded with each run."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
# Full corpus source: the upstream generator repository at a pinned commit.
CORPUS_REPO_URL = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git"
CORPUS_COMMIT = "d36685e273713975ee20299bbf1ab64165575b3c"
CORPUS_DIR = ROOT / ".cache" / "corpus"


class RetrievalSettings(BaseModel):
    """Shared retrieval settings; web environment and CLI options use these defaults."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    retrieval: Literal['bm25', 'hybrid'] = 'bm25'
    corpus: Path = CORPUS_DIR / 'documents.parquet'
    bm25_index: Path = CORPUS_DIR / 'indexes/bm25'
    dense_index: Path = CORPUS_DIR / 'indexes/dense'
    device: str = 'cpu'
    fp16: bool | None = None
    bm25_weight: float = Field(default=.7, ge=0, le=1)
    candidate_k: int = Field(default=50, ge=1, le=100)
    rerank_k: int = Field(default=50, ge=1, le=200)
    rrf_k: float = Field(default=60.0, ge=0)

    def hybrid_config(self):
        from src.retrieval.hybrid import HybridConfig
        return HybridConfig(candidate_k=self.candidate_k, rerank_k=self.rerank_k,
                            rrf_k=self.rrf_k, bm25_weight=self.bm25_weight,
                            dense_weight=1 - self.bm25_weight)


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
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Dense models by label, each at an exact Hub commit with the prefixes from its model
# card. Qwen3 takes a task instruction; ours describes this benchmark's searches.
EMBEDDING_MODELS = {
    "bge-small": {"name": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
                  "query_prefix": BGE_QUERY_PREFIX},
    "bge-base": {"name": "BAAI/bge-base-en-v1.5",
                 "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
                 "query_prefix": BGE_QUERY_PREFIX},
    "bge-large": {"name": "BAAI/bge-large-en-v1.5",
                  "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                  "query_prefix": BGE_QUERY_PREFIX},
    "gte-modernbert-base": {"name": "Alibaba-NLP/gte-modernbert-base",
                            "revision": "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
                            "query_prefix": ""},
    "arctic-embed-l-v2.0": {"name": "Snowflake/snowflake-arctic-embed-l-v2.0",
                            "revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
                            "query_prefix": "query: "},
    "qwen3-embedding-0.6b": {"name": "Qwen/Qwen3-Embedding-0.6B",
                             "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                             "query_prefix": "Instruct: Given a question about a company's "
                                             "internal documents, retrieve the documents "
                                             "that answer it\nQuery:"},
}

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
