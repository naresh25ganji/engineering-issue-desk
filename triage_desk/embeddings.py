"""A lazy, real dense embedding index. Failures never select another search mode."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
_INDEX_VERSION = 1


class EmbeddingUnavailable(RuntimeError):
    """The requested semantic retrieval could not be performed."""


class DenseIndex:
    def __init__(self, chunks: list[dict], cache_dir: Path, prebuilt_dir: Path | None = None):
        self.chunks = chunks
        self.cache_dir = Path(cache_dir)
        self._model = None
        self._vectors: np.ndarray | None = None
        identity = {
            "format": _INDEX_VERSION, "model": MODEL_NAME,
            "chunks": [(c["chunk_id"], hashlib.sha256(f"{c['title']}\n{c['section']}\n{c['text']}".encode()).hexdigest()) for c in chunks],
        }
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.index_path = self.cache_dir / f"vectors-{self.fingerprint}.npz"
        self.prebuilt_path = Path(prebuilt_dir) / self.index_path.name if prebuilt_dir is not None else None
        self.cache_hit = False
        self.index_origin = "not_loaded"

    def _make_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                model_cache = self.cache_dir / "models"
                kwargs = {"model_name": MODEL_NAME, "cache_dir": str(model_cache), "threads": 2}
                # Cached snapshots are useful offline, and should not trigger a remote
                # metadata lookup on every new application process.
                if any(model_cache.rglob("*.onnx")):
                    try:
                        self._model = TextEmbedding(**kwargs, local_files_only=True)
                    except Exception:
                        self._model = TextEmbedding(**kwargs)
                else:
                    self._model = TextEmbedding(**kwargs)
            except Exception as exc:
                raise EmbeddingUnavailable(
                    "Semantic search could not load BAAI/bge-small-en-v1.5. "
                    "Install fastembed in this environment and allow the first model download "
                    "(about 65 MB), then retry. Keyword baseline remains available as an explicit mode. "
                    f"Failure type: {type(exc).__name__}."
                ) from exc
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        try:
            result = np.asarray(list(self._make_model().embed(list(texts), batch_size=32)), dtype=np.float32)
            return self._normalise(result)
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(
                "Dense document embedding failed. Check available memory and the model cache, "
                f"then retry. Failure type: {type(exc).__name__}."
            ) from exc

    def embed_query(self, question: str) -> np.ndarray:
        try:
            result = np.asarray(list(self._make_model().query_embed(question)), dtype=np.float32)
            return self._normalise(result)[0]
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(
                "Dense query embedding failed; semantic/hybrid results were not produced. "
                f"Retry after checking the model installation. Failure type: {type(exc).__name__}."
            ) from exc

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Embedding output must be a finite matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Embedding output contains an empty vector")
        return vectors / norms

    def vectors(self) -> np.ndarray:
        if self._vectors is not None:
            return self._vectors
        candidates = [(self.index_path, "derived_cache")]
        if self.prebuilt_path is not None:
            candidates.append((self.prebuilt_path, "bundled_index"))
        for path, origin in candidates:
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=False) as stored:
                    vectors = np.asarray(stored["vectors"], dtype=np.float32)
                    identity = str(stored["fingerprint"].item())
                if identity != self.fingerprint or vectors.ndim != 2 or len(vectors) != len(self.chunks) or vectors.shape[1] < 1:
                    raise ValueError("Index identity or shape mismatch")
                self._vectors = self._normalise(vectors)
                self.cache_hit = True
                self.index_origin = origin
                return self._vectors
            except (ValueError, KeyError, OSError, EOFError, BadZipFile):
                # An outdated/corrupt bundle is never used for a changed corpus.
                # Try another matching index, then rebuild from source if needed.
                pass
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EmbeddingUnavailable("The vector cache directory is not writable. Select a writable local cache directory and retry.") from exc
        if not self.chunks:
            self._vectors = np.empty((0, 384), dtype=np.float32)
            return self._vectors
        vectors = self.embed_documents([f"{c['title']}\n{c['section']}\n{c['text']}" for c in self.chunks])
        if len(vectors) != len(self.chunks):
            raise EmbeddingUnavailable("Embedding provider returned an incomplete document index. Retry indexing.")
        handle = tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False)
        temporary_path = Path(handle.name)
        try:
            with handle:
                np.savez_compressed(handle, vectors=vectors, fingerprint=np.asarray(self.fingerprint))
            os.replace(temporary_path, self.index_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._vectors = vectors
        self.index_origin = "computed"
        return vectors

    def scores(self, question: str) -> np.ndarray:
        vectors = self.vectors()
        query = self.embed_query(question)
        if vectors.shape[1] != len(query):
            raise EmbeddingUnavailable("Dense index dimensions do not match the query model. Remove the derived vector cache and retry.")
        return vectors @ query
