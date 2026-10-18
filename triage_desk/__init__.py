"""Version-aware evidence retrieval for an engineering issue triage workspace."""

from .core import TriageDesk, chunk_sources, tokenize
from .embeddings import MODEL_NAME, EmbeddingUnavailable
from .generation import GenerationUnavailable, provider_status

__all__ = [
    "TriageDesk", "chunk_sources", "tokenize", "MODEL_NAME",
    "EmbeddingUnavailable", "GenerationUnavailable", "provider_status",
]
