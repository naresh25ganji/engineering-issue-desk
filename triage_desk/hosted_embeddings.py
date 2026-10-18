"""Explicit opt-in compatible hosted embeddings with isolated vector caches.

Configuration is resolved only when an index is constructed/used at runtime.
This module never loads an environment file and never silently loads a local
model. Requests retain numeric usage and safe model/timing metadata, not inputs,
credentials, headers, error bodies, or arbitrary provider response text.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import math
from numbers import Real
from pathlib import Path
import re
import threading
import time
from typing import Sequence

import httpx
import numpy as np

from .embeddings import DenseIndex, EmbeddingUnavailable
from .generation import GenerationUnavailable, _configuration, _usage_counts

HOSTED_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
HOSTED_DIMENSIONS = 1024
BATCH_SIZE = 32
QUERY_CACHE_SIZE = 128
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+\-]{0,199}\Z")


def _embedding_endpoint(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[:-len("/chat/completions")]
    return base if base.endswith("/embeddings") else base + "/embeddings"


def _safe_model(value, key: str) -> str:
    if not isinstance(value, str) or not _MODEL_ID.fullmatch(value):
        return ""
    if (key and key in value) or re.match(r"(?:sk-|gh[pousr]_|github_pat_|AKIA)", value):
        return ""
    return value


class HostedDenseIndex(DenseIndex):
    """DenseIndex cosine search backed exclusively by a hosted embeddings API.

    The local BAAI implementation remains the application's independent default.
    This class is selected explicitly via RetrievalLab's dense_factory injection.
    Each endpoint/model/dimension/corpus combination has a separate fingerprint;
    no local or prebuilt vector file can be reused under this index's identity.
    """
    def __init__(self, chunks: list[dict], cache_dir: Path, model: str | None = None,
                 dimensions: int = HOSTED_DIMENSIONS):
        requested = (os.environ.get('TRIAGE_EMBEDDING_MODEL') or HOSTED_MODEL_NAME) if model is None else model
        if not isinstance(requested, str) or not _MODEL_ID.fullmatch(requested):
            raise ValueError("Embedding model must be a nonempty provider model identifier")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("Embedding dimensions must be a positive integer")
        super().__init__(chunks, cache_dir)
        original_fingerprint = self.fingerprint
        try:
            key, _, base = _configuration()
        except GenerationUnavailable as exc:
            raise EmbeddingUnavailable("Hosted embeddings are not configured. Set the provider key/base using the existing runtime configuration; no local fallback was selected.") from None
        if not _safe_model(requested, key):
            raise ValueError("Embedding model identifier is invalid")
        self.model_name, self.dimensions = requested, dimensions
        self.endpoint = _embedding_endpoint(base)
        identity = {"format": "hosted-embeddings-v1", "endpoint": self.endpoint,
                    "model": self.model_name, "dimensions": self.dimensions,
                    "chunk_fingerprint": original_fingerprint}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.index_path = self.cache_dir / f"vectors-{self.fingerprint}.npz"
        self.prebuilt_path = None
        self.calls: list[dict] = []
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_lock = threading.RLock()

    def _runtime_key(self) -> str:
        try:
            key, _, base = _configuration()
        except GenerationUnavailable:
            raise EmbeddingUnavailable("Hosted embedding configuration is unavailable; no local fallback was selected.") from None
        if _embedding_endpoint(base) != self.endpoint:
            raise EmbeddingUnavailable("Embedding endpoint changed after index construction. Create a new index so its cache identity matches the endpoint.")
        return key

    def _make_model(self):
        raise EmbeddingUnavailable("HostedDenseIndex does not load a local embedding model")

    def _request(self, texts: list[str], kind: str) -> np.ndarray:
        started = time.perf_counter()
        metadata = {"stage": "embedding", "kind": kind, "attempted": False, "status": "not_attempted",
                    "requested_model": self.model_name, "returned_model": "",
                    "dimensions": self.dimensions, "input_count": len(texts),
                    "usage": {}, "latency_ms": 0.0, "endpoint": self.endpoint}
        try:
            key = self._runtime_key()
            with httpx.Client(timeout=httpx.Timeout(45, connect=10), follow_redirects=False) as client:
                metadata["attempted"] = True
                self.calls.append(metadata)
                response = client.post(self.endpoint, headers={"Authorization": f"Bearer {key}"},
                                       json={"model": self.model_name, "input": texts, "dimensions": self.dimensions})
                status = response.status_code
                metadata["http_status"] = status
                if not 200 <= status < 300:
                    metadata["status"] = "provider_http_error"
                    raise EmbeddingUnavailable(f"Hosted embeddings returned HTTP {status}; no embedding results were accepted.")
                body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Expected an embedding response object")
            metadata["returned_model"] = _safe_model(body.get("model"), key)
            metadata["usage"] = _usage_counts(body.get("usage"))
            data = body.get("data")
            if not isinstance(data, list) or len(data) != len(texts):
                raise ValueError("Incomplete embedding response")
            ordered: list[list[float] | None] = [None] * len(texts)
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("Invalid embedding item")
                index, embedding = item.get("index"), item.get("embedding")
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(texts) or ordered[index] is not None:
                    raise ValueError("Missing, duplicate or out-of-range embedding index")
                if not isinstance(embedding, list) or len(embedding) != self.dimensions:
                    raise ValueError("Unexpected embedding dimension")
                if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in embedding):
                    raise ValueError("Invalid embedding values")
                ordered[index] = embedding
            if any(value is None for value in ordered):
                raise ValueError("Missing embedding index")
            # Normalize using float64 norms to avoid overflow from otherwise
            # finite large numbers. Only complete, finite float32 vectors leave
            # this method; provider text is never interpolated into errors.
            values = np.asarray(ordered, dtype=np.float64)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Invalid embedding norms")
            result = (values / norms).astype(np.float32)
            if result.shape != (len(texts), self.dimensions) or not np.isfinite(result).all():
                raise ValueError("Invalid normalized embedding matrix")
            result = self._normalise(result)
            metadata["status"] = "success"
            return result
        except EmbeddingUnavailable:
            if metadata["status"] == "not_attempted":
                metadata["status"] = "configuration_error"
            raise
        except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            metadata["status"] = "provider_error" if isinstance(exc, (httpx.HTTPError, OSError)) else "invalid_response"
            metadata["failure_type"] = type(exc).__name__
            raise EmbeddingUnavailable(f"Hosted embedding request failed ({type(exc).__name__}); no embedding results were accepted.") from None
        finally:
            metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)):
            raise ValueError("Embedding inputs must be a sequence of nonempty texts")
        values = list(texts)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Embedding inputs must be nonempty text strings")
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        batches = []
        # Deliberately serial: bounded request size and no implicit retries or
        # hidden additional credit consumption. Failed runs save no partial index.
        for start in range(0, len(values), BATCH_SIZE):
            batches.append(self._request(values[start:start + BATCH_SIZE], "documents"))
        result = np.concatenate(batches, axis=0)
        if result.shape != (len(values), self.dimensions):
            raise EmbeddingUnavailable("Hosted document embeddings did not cover the complete index")
        return result

    def embed_query(self, question: str) -> np.ndarray:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Embedding query must be nonempty text")
        self._runtime_key()  # A changed endpoint must not reuse old query vectors.
        identity = hashlib.sha256(question.encode()).hexdigest()
        with self._query_lock:
            if identity in self._query_cache:
                self._query_cache.move_to_end(identity)
                return self._query_cache[identity].copy()
        value = self._request([question], "query")[0]
        with self._query_lock:
            self._query_cache[identity] = value.copy()
            self._query_cache.move_to_end(identity)
            while len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return value.copy()

    def vectors(self) -> np.ndarray:
        if not self.chunks:
            self._vectors = np.empty((0, self.dimensions), dtype=np.float32)
            self.index_origin = "empty"
            return self._vectors
        values = super().vectors()
        if values.shape != (len(self.chunks), self.dimensions):
            # This is only a derived cache under this hosted fingerprint, never
            # a local/prebuilt index. Rebuild with the same hosted provider.
            self._vectors, self.cache_hit, self.index_origin = None, False, "not_loaded"
            try:
                self.index_path.unlink(missing_ok=True)
            except OSError:
                raise EmbeddingUnavailable("The hosted vector cache has the wrong shape and could not be replaced") from None
            values = super().vectors()
        if values.shape != (len(self.chunks), self.dimensions):
            raise EmbeddingUnavailable("Hosted vector index shape does not match the configured model dimensions")
        return values

    def scores(self, question: str) -> np.ndarray:
        if not self.chunks:
            return np.empty(0, dtype=float)
        return super().scores(question)
