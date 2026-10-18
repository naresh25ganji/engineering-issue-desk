"""Comparable chunking/retrieval experiments with literal source provenance.

Structured chunking is deterministic Markdown/sentence-aware splitting, not
embedding-based semantic segmentation. Code fences and tables remain whole when
they fit; oversized blocks are bounded and explicitly labelled as partial.
FlashRank is optional, lazy, local inference, and never a made-up score fallback.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from numbers import Real
from pathlib import Path
import re
from typing import Callable

import numpy as np

from .core import KINDS, MODES, _BM25, _quote, _validate_question
from .embeddings import DenseIndex, EmbeddingUnavailable, MODEL_NAME

STRATEGIES = {"fixed", "structured"}
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
TARGET_WORDS = 180
OVERLAP_WORDS = 35
STRUCTURED_MAX_CHARS = 2400
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


class RerankerUnavailable(RuntimeError):
    """The requested actual reranker could not produce complete valid scores."""


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    kind: str
    section: str
    group: int
    part: int = 0
    parts: int = 1
    split_reason: str = ""
    unterminated: bool = False


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _check_limits(target_words: int, max_chars: int) -> None:
    if isinstance(target_words, bool) or not isinstance(target_words, int) or target_words < 20:
        raise ValueError("target_words must be an integer of at least 20")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 100:
        raise ValueError("max_chars must be an integer of at least 100")


def _blocks(text: str) -> list[_Block]:
    lines = text.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    offsets.append(len(text))
    rows, headings, index = [], [], 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        begin = index
        heading = _HEADING.match(lines[index].rstrip("\r\n"))
        if heading:
            level = len(heading.group(1))
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, heading.group(2)))
            kind, index = "heading", index + 1
            unterminated = False
        elif fence := _FENCE.match(lines[index]):
            marker = fence.group(1)
            index += 1
            while index < len(lines):
                closing = re.match(r"^\s*" + re.escape(marker[0]) + r"{" + str(len(marker)) + r",}\s*$", lines[index])
                index += 1
                if closing:
                    break
            else:
                closing = False
            kind, unterminated = "code", not bool(closing)
        elif index + 1 < len(lines) and "|" in lines[index] and _TABLE_RULE.match(lines[index + 1]):
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                index += 1
            kind, unterminated = "table", False
        else:
            index += 1
            while index < len(lines) and lines[index].strip():
                if _HEADING.match(lines[index].rstrip("\r\n")) or _FENCE.match(lines[index]):
                    break
                if index + 1 < len(lines) and "|" in lines[index] and _TABLE_RULE.match(lines[index + 1]):
                    break
                index += 1
            kind = "list" if re.match(r"\s*(?:[-*+]\s|\d+[.)]\s)", lines[begin]) else "paragraph"
            unterminated = False
        section = " > ".join(title for _, title in headings) or "Overview"
        rows.append(_Block(offsets[begin], offsets[index], kind, section, len(rows), unterminated=unterminated))
    return rows


def _bounded_pieces(text: str, block: _Block, max_words: int, max_chars: int) -> list[_Block]:
    content = text[block.start:block.end]
    if _word_count(content) <= max_words and len(content) <= max_chars:
        return [block]
    # Prefer full lines for structural blocks and full sentences for prose.
    if block.kind in {"code", "table", "list", "heading"}:
        preferred = [block.start + m.end() for m in re.finditer(r"\n", content)]
    else:
        preferred = [block.start + m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9`])", content)]
    spans, cursor = [], block.start
    while cursor < block.end:
        if not text[cursor:block.end].strip():
            break
        hard_end = min(block.end, cursor + max_chars)
        words = list(re.finditer(r"\S+", text[cursor:hard_end]))
        if len(words) > max_words:
            hard_end = cursor + words[max_words - 1].end()
        good = [end for end in preferred if cursor < end <= hard_end and text[cursor:end].strip()]
        end = max(good) if good else hard_end
        # A missing preferred boundary means a long sentence/line or token had
        # to be split. Never synthesize code fences or repeat table headers.
        spans.append((cursor, end, "" if good or end == block.end else "_hard_boundary"))
        cursor = end
    count = len(spans)
    return [replace(block, start=start, end=end, part=i, parts=count,
                    split_reason=f"oversized_{block.kind}" + suffix)
            for i, (start, end, suffix) in enumerate(spans)]


def _chunk(source: dict, strategy: str, start: int, end: int, index: int,
           section: str, blocks: list[_Block] | None = None) -> dict:
    text = source["text"][start:end]
    identity = f"retrieval-lab-v1\0{strategy}\0{source['source_id']}\0{start}\0{end}\0{text}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    details = [{"kind": b.kind, "source_start": b.start, "source_end": b.end,
                "block_id": f"{source['source_id']}:block-{b.group}",
                "part_index": b.part, "parts": b.parts, "split_reason": b.split_reason,
                "unterminated_fence": b.unterminated} for b in (blocks or [])]
    return {**source, "text": text, "section": section, "section_index": index,
            "chunk_id": f"{source['source_id']}:lab-{strategy}:{digest}", "chunk_index": index,
            "chunk_strategy": strategy, "source_start": start, "source_end": end,
            "word_count": _word_count(text), "character_count": len(text),
            "blocks": details,
            "structure_complete": strategy == "structured" and all(b.parts == 1 and not b.unterminated for b in (blocks or [])),
            "split_blocks": any(b.parts > 1 or b.unterminated for b in (blocks or []))}


def fixed_chunks(sources: list[dict], target_words: int = TARGET_WORDS,
                 overlap_words: int = OVERLAP_WORDS) -> list[dict]:
    """Literal fixed word windows; intentionally ignores structure boundaries."""
    if isinstance(target_words, bool) or not isinstance(target_words, int) or target_words < 20:
        raise ValueError("target_words must be an integer of at least 20")
    if isinstance(overlap_words, bool) or not isinstance(overlap_words, int) or not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words must be a nonnegative integer below target_words")
    output = []
    for source in sources:
        text = source["text"]
        words = list(re.finditer(r"\S+", text))
        headings = [(block.start, block.section) for block in _blocks(text) if block.kind == "heading"]
        cursor, part = 0, 0
        while cursor < len(words):
            end = min(cursor + target_words, len(words))
            start_char, end_char = words[cursor].start(), words[end - 1].end()
            section = next((title for offset, title in reversed(headings) if offset <= start_char), "Overview")
            output.append(_chunk(source, "fixed", start_char, end_char, part, section))
            if end == len(words):
                break
            cursor, part = end - overlap_words, part + 1
    return output


def structured_chunks(sources: list[dict], target_words: int = TARGET_WORDS,
                      max_chars: int = STRUCTURED_MAX_CHARS) -> list[dict]:
    """Bounded Markdown blocks, splitting oversized prose at sentences first.

    No overlap is inserted: each complete block is kept together where possible.
    Text is always a literal contiguous source span, including internal whitespace.
    Partial oversized blocks retain group/part metadata for honest presentation.
    """
    _check_limits(target_words, max_chars)
    output = []
    for source in sources:
        text, pending, index = source["text"], [], 0
        pieces = [piece for block in _blocks(text) for piece in _bounded_pieces(text, block, target_words, max_chars)]
        for piece in pieces:
            combined = text[pending[0].start:piece.end] if pending else text[piece.start:piece.end]
            section_changed = bool(pending and piece.section != pending[-1].section)
            if pending and (section_changed or len(combined) > max_chars or _word_count(combined) > target_words):
                output.append(_chunk(source, "structured", pending[0].start, pending[-1].end, index, pending[0].section, pending))
                index, pending = index + 1, []
            pending.append(piece)
        if pending:
            output.append(_chunk(source, "structured", pending[0].start, pending[-1].end, index, pending[0].section, pending))
    return output


class _FlashRank:
    def __init__(self, cache_dir: Path):
        self.cache_dir, self._ranker, self._request = Path(cache_dir), None, None

    def rerank(self, query: str, passages: list[dict]) -> list[dict]:
        try:
            if self._ranker is None:
                from flashrank import Ranker, RerankRequest
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._ranker = Ranker(model_name=RERANK_MODEL, cache_dir=str(self.cache_dir), max_length=512)
                self._request = RerankRequest
            return self._ranker.rerank(self._request(query=query, passages=passages))
        except Exception as exc:
            raise RerankerUnavailable(
                f"FlashRank {RERANK_MODEL} could not run ({type(exc).__name__}). "
                "Install flashrank and prepare its local model cache; reranked results were not produced."
            ) from exc


class RetrievalLab:
    """Explicit experimental retrieval; no answer generation or environment reads.

    Optional dense_factory(chunks, cache_dir) supplies an index exposing
    scores(question) and fingerprint; default is the shared local DenseIndex.
    Optional reranker_factory(cache_dir) supplies rerank(query, passages), mainly
    for offline tests. Production default always uses the actual FlashRank model.
    """
    def __init__(self, sources: list[dict], cache_dir: Path, *, dense_factory: Callable | None = None,
                 reranker_factory: Callable | None = None):
        if not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
            raise ValueError("sources must be a list of source dictionaries")
        seen = set()
        for source in sources:
            if any(not isinstance(source.get(k), str) or not source[k].strip() for k in ("source_id", "title", "text", "url", "kind")):
                raise ValueError("Every source needs nonempty source_id/title/text/url/kind strings")
            if source["source_id"] in seen or source["kind"] not in KINDS:
                raise ValueError("Source IDs must be unique and kinds must be recognized")
            seen.add(source["source_id"])
        self.sources = list(sources)
        self.cache_dir = Path(cache_dir)
        self.chunks_by_strategy = {"fixed": fixed_chunks(sources), "structured": structured_chunks(sources)}
        factory = dense_factory or DenseIndex
        self.dense_by_strategy = {name: factory(chunks, self.cache_dir) for name, chunks in self.chunks_by_strategy.items()}
        self._keywords = {name: _BM25(chunks) for name, chunks in self.chunks_by_strategy.items()}
        self._reranker_factory = reranker_factory or _FlashRank
        self._reranker = None

    def chunks(self, strategy: str = "structured") -> list[dict]:
        if strategy not in STRATEGIES:
            raise ValueError("Chunk strategy must be fixed or structured")
        return self.chunks_by_strategy[strategy]

    def _rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        try:
            if self._reranker is None:
                self._reranker = self._reranker_factory(self.cache_dir / "reranker")
            passages = [{"id": c["chunk_id"], "text": f"{c['title']}\n{c['section']}\n{c['text']}"} for c in candidates]
            results = self._reranker.rerank(question, passages)
            if not isinstance(results, list) or len(results) != len(candidates):
                raise ValueError("Reranker did not return a score for every candidate")
            by_id, scored = {c["chunk_id"]: c for c in candidates}, {}
            for item in results:
                if not isinstance(item, dict) or item.get("id") not in by_id or item["id"] in scored:
                    raise ValueError("Reranker returned an unknown or repeated candidate ID")
                value = item.get("score")
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError("Reranker returned a nonfinite or nonnumeric score")
                scored[item["id"]] = {**by_id[item["id"]], "rerank_score": float(value), "rerank_model": RERANK_MODEL}
            return sorted(scored.values(), key=lambda c: (-c["rerank_score"], c["chunk_id"]))
        except RerankerUnavailable:
            raise
        except Exception as exc:
            raise RerankerUnavailable(f"FlashRank reranking failed validation ({type(exc).__name__}); no reranked results were produced.") from exc

    def search(self, question: str, *, strategy: str = "structured", mode: str = "hybrid",
               rerank: bool = False, top_k: int = 6, candidate_k: int = 18, source_kinds=None) -> list[dict]:
        question = _validate_question(question)
        if strategy not in STRATEGIES or mode not in MODES:
            raise ValueError("Choose fixed/structured chunking and keyword/semantic/hybrid retrieval")
        if not isinstance(rerank, bool):
            raise ValueError("rerank must be a boolean")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 30:
            raise ValueError("top_k must be an integer from 1 to 30")
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or not top_k <= candidate_k <= 100:
            raise ValueError("candidate_k must be an integer between top_k and 100")
        if source_kinds is None:
            kinds = KINDS
        elif isinstance(source_kinds, (list, tuple, set, frozenset)) and all(isinstance(k, str) for k in source_kinds) and set(source_kinds) <= KINDS:
            kinds = set(source_kinds)
        else:
            raise ValueError("Source kinds must contain only recognized evidence kinds")
        chunks = self.chunks_by_strategy[strategy]
        eligible = [i for i, c in enumerate(chunks) if c["kind"] in kinds]
        if not eligible:
            return []
        keyword = self._keywords[strategy].scores(question) if mode in {"keyword", "hybrid"} else np.zeros(len(chunks))
        dense = np.asarray(self.dense_by_strategy[strategy].scores(question), dtype=float) if mode in {"semantic", "hybrid"} else np.zeros(len(chunks))
        if dense.shape != (len(chunks),) or not np.isfinite(dense).all():
            raise EmbeddingUnavailable("Dense backend did not return one finite score for every chunk; no fallback mode was selected")
        word_order = sorted((i for i in eligible if keyword[i] > 0), key=lambda i: (-keyword[i], chunks[i]["chunk_id"]))
        dense_order = sorted(eligible, key=lambda i: (-dense[i], chunks[i]["chunk_id"]))
        if mode == "keyword":
            order, scores = word_order, keyword
        elif mode == "semantic":
            order, scores = dense_order, dense
        else:
            scores = np.zeros(len(chunks))
            for ranking in (word_order[:60], dense_order[:60]):
                for rank, index in enumerate(ranking, 1):
                    scores[index] += 1 / (60 + rank)
            order = sorted(set(word_order[:60] + dense_order[:60]), key=lambda i: (-scores[i], chunks[i]["chunk_id"]))
        candidates, counts = [], Counter()
        for index in order:
            chunk = chunks[index]
            if counts[chunk["source_id"]] >= 3:
                continue
            counts[chunk["source_id"]] += 1
            candidates.append({**chunk, "score": round(float(scores[index]), 6),
                               "keyword_score": round(float(keyword[index]), 6), "dense_score": round(float(dense[index]), 6),
                               "retrieval_rank": len(candidates) + 1,
                               "quote": _quote(chunk["text"], question)})
            if len(candidates) >= candidate_k:
                break
        if rerank:
            candidates = self._rerank(question, candidates)
        output, counts = [], Counter()
        for candidate in candidates:
            if counts[candidate["source_id"]] >= 2:
                continue
            counts[candidate["source_id"]] += 1
            output.append({**candidate, "citation_id": f"S{len(output) + 1}"})
            if len(output) >= top_k:
                break
        return output

    @staticmethod
    def to_documents(evidence: list[dict]):
        """Optional LangChain adapter; construction never calls a model."""
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise RuntimeError("Install langchain-core to use the Document adapter") from exc
        return [Document(page_content=item["text"], metadata={key: value for key, value in item.items() if key != "text"})
                for item in evidence]

    def description(self) -> dict:
        return {"embedding_model_default": MODEL_NAME, "reranker_model": RERANK_MODEL,
                "embedding_models": {name: getattr(index, 'model_name', MODEL_NAME) for name, index in self.dense_by_strategy.items()},
                "chunk_counts": {key: len(value) for key, value in self.chunks_by_strategy.items()},
                "chunk_settings": {"fixed_target_words": TARGET_WORDS, "fixed_overlap_words": OVERLAP_WORDS,
                                   "structured_max_words": TARGET_WORDS, "structured_max_chars": STRUCTURED_MAX_CHARS,
                                   "structured_overlap": "No duplicated overlap; full blocks where bounded"},
                "structured_split_chunks": sum(c["split_blocks"] for c in self.chunks_by_strategy["structured"]),
                "index_fingerprints": {name: getattr(index, "fingerprint", None) for name, index in self.dense_by_strategy.items()},
                "dense_backend_classes": {name: f"{type(index).__module__}.{type(index).__qualname__}" for name, index in self.dense_by_strategy.items()},
                "candidate_source_cap": 3, "output_source_cap": 2,
                "score_note": "score remains the first-stage score; rerank_score is a separate actual model score, not a calibrated confidence",
                "chunking_note": "Structured is deterministic paragraph/sentence/Markdown-aware, not embedding-based semantic chunking. Oversized blocks are explicitly partial."}
