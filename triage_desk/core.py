"""Deterministic retrieval and conservative triage, independent of a web framework."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import numpy as np

from .embeddings import DenseIndex, MODEL_NAME
from .generation import GenerationUnavailable, synthesize

KINDS = {"documentation", "issue", "release", "contributing"}
MODES = {"keyword", "semantic", "hybrid"}
STOP_WORDS = set("a an the and or but if then to of for in on at by is are was were be been being it its this that these those i me my we our you your how what which when where why can could would should do does did have has had with from as about please tell explain using use used streamlit app application issue problem help need want get getting not no just also into any all some there here than while will work works working".split())


def tokenize(text: str) -> list[str]:
    """Keep API identifiers while also matching their word components."""
    terms = []
    for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", text.lower()):
        token = token.rstrip(".-")
        values = [token] + (re.split(r"[_.-]+", token) if re.search(r"[_.-]", token) else [])
        for value in dict.fromkeys(values):
            if len(value) > 1 and value not in STOP_WORDS:
                terms.append(value)
    return terms


def _sections(text: str) -> list[tuple[str, str]]:
    rows, body, title = [], [], "Overview"
    fenced = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line) if not fenced else None
        if heading:
            if "".join(body).strip():
                rows.append((title, "".join(body).strip()))
            title, body = heading.group(1).strip(), []
        else:
            body.append(line)
    if "".join(body).strip():
        rows.append((title, "".join(body).strip()))
    return rows


def chunk_sources(sources: list[dict], target_words: int = 180, overlap_words: int = 35) -> list[dict]:
    if not isinstance(target_words, int) or not isinstance(overlap_words, int) or target_words < 20 or not 0 <= overlap_words < target_words:
        raise ValueError("Chunk size must be at least 20 words, with a smaller nonnegative overlap")
    chunks = []
    for source in sources:
        for section_index, (section, text) in enumerate(_sections(source["text"])):
            words = list(re.finditer(r"\S+", text))
            start, part = 0, 0
            while start < len(words):
                end = min(start + target_words, len(words))
                passage = text[words[start].start():words[end - 1].end()]
                identity = f"{source['source_id']}\0{section}\0{part}\0{passage}"
                digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
                chunk = {**source, "text": passage, "section": section,
                         "section_index": section_index, "word_count": end - start,
                         "chunk_id": f"{source['source_id']}:{digest}", "chunk_index": part}
                chunks.append(chunk)
                if end == len(words):
                    break
                start, part = end - overlap_words, part + 1
    return chunks


class _BM25:
    def __init__(self, chunks: list[dict]):
        self.documents = [Counter(tokenize(f"{c['title']} {c['title']} {c['section']} {c['text']}")) for c in chunks]
        self.lengths = np.asarray([sum(d.values()) for d in self.documents], dtype=float)
        self.average = float(np.mean(self.lengths)) if len(self.lengths) else 1.
        self.df = Counter(t for d in self.documents for t in d)

    def scores(self, question: str) -> np.ndarray:
        count, output = len(self.documents), np.zeros(len(self.documents), dtype=float)
        for term in set(tokenize(question)):
            document_frequency = self.df.get(term, 0)
            if not document_frequency:
                continue
            idf = math.log(1 + (count - document_frequency + .5) / (document_frequency + .5))
            for index, document in enumerate(self.documents):
                frequency = document.get(term, 0)
                if frequency:
                    output[index] += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * self.lengths[index] / max(self.average, 1)))
        return output


def _validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter an engineering question or issue report")
    if len(question) > 6000:
        raise ValueError("Keep the question under 6,000 characters; put lengthy logs in a minimal reproduction instead")
    if "\x00" in question:
        raise ValueError("Question contains an invalid control character")
    return question.strip()


def _load_sources(data_dir: Path) -> list[dict]:
    path = data_dir / "sources.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load the source corpus at {path}. Run the corpus snapshot script first.") from exc
    if not isinstance(payload, list):
        raise ValueError("sources.json must be a list of source records")
    sources, seen = [], set()
    required = {"source_id", "title", "url", "kind", "text", "source_commit", "snapshot_at", "version_scope"}
    for index, original in enumerate(payload):
        if not isinstance(original, dict) or required - original.keys():
            raise ValueError(f"Source record {index + 1} is missing required provenance fields")
        record = dict(original)
        text_fields = required - {"source_commit"}
        if any(not isinstance(record[k], str) for k in text_fields) or not (
            isinstance(record["source_commit"], str) or (record.get("kind") == "issue" and record["source_commit"] is None)
        ):
            raise ValueError(f"Source record {index + 1} has a non-text required field")
        if not record["source_id"].strip() or record["source_id"] in seen:
            raise ValueError("Source IDs must be nonempty and unique")
        if record["kind"] not in KINDS:
            raise ValueError(f"Unknown source kind in {record['source_id']}")
        parsed = urlsplit(record["url"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError(f"Invalid source URL in {record['source_id']}")
        if not record["text"].strip():
            raise ValueError(f"Source {record['source_id']} has no text")
        seen.add(record["source_id"])
        sources.append(record)
    return sources


def _intent(question: str) -> tuple[str, str]:
    low = question.lower()
    if re.search(r"\b(feature request|enhancement|add support|please add|would like|new feature)\b", low):
        category = "feature request"
    elif re.search(r"\b(error|exception|crash(?:es|ing)?|bug|broken|fails?|failing|resets?|unexpected|doesn.t work|does not work|not working|traceback)\b", low):
        category = "bug report"
    else:
        category = "question"
    groups = [
        ("File uploads", r"file_uploader|file upload|upload(?:ed|ing)? (?:a |the |my )?(?:csv|file|document)|csv upload"),
        ("Caching", r"cache|caching|ttl|hash_funcs|persist"),
        ("Session state", r"session[_. ]state|st\.session_state"),
        ("Widgets and forms", r"widget|selectbox|checkbox|slider|button|form|text_input|number_input"),
        ("Execution and reruns", r"fragment|rerun|dialog|execution|multipage|pages|navigation"),
        ("Data display", r"dataframe|data_editor|pandas|table|chart|plot"),
        ("Configuration and deployment", r"deploy|config|toml|port|server|docker|secrets|cloud|cors"),
    ]
    component = next((name for name, pattern in groups if re.search(pattern, low)), "General / needs maintainer review")
    return category, component


def _missing(question: str, version: str, category: str) -> list[str]:
    if category != "bug report":
        return []
    low = question.lower()
    output = []
    if not version and not re.search(r"\b(?:streamlit\s*(?:version|==|v|:)?\s*)?1\.\d+(?:\.\d+)?\b", low):
        output.append("Installed Streamlit version")
    if not re.search(r"python(?:\s+version)?\s*[:=v]*\s*3\.\d+", low):
        output.append("Python version")
    if not re.search(r"\b(windows|macos|mac os|linux|ubuntu|debian|fedora)\b", low):
        output.append("Operating system / deployment environment")
    if "```" not in question and not re.search(r"\b(import streamlit|st\.[a-z_]+\()", low):
        output.append("Minimal reproducible code")
    if not re.search(r"(?:steps to reproduce|reproduction steps|steps)\s*:\s*\S|^\s*1[.)]\s+\S", low, re.M):
        output.append("Exact steps to reproduce")
    if not ("expected" in low and re.search(r"actual|instead|observed", low)):
        output.append("Expected behavior and actual behavior")
    if re.search(r"error|exception|traceback|crash", low) and not re.search(r"traceback \(most recent|file .+ line \d|error:", low):
        output.append("Full error message / traceback, with secrets removed")
    return output


def _guard(question: str, version: str) -> tuple[str | None, str]:
    low = question.lower()
    private = re.search(r"\b(private repositor(?:y|ies)|private repo|internal secrets|api[_ -]?keys?|access tokens?|system prompt|hidden instructions|environment variables|\.env file|secret value|admin password)\b", low)
    seeking = re.search(r"\b(reveal|print|show|dump|exfiltrate|expose|read|list|give|what is|tell me)\b", low)
    attack = re.search(r"ignore (?:all |the |previous |prior )*(?:instructions|rules|docs|documents|system)|(?:bypass|override) (?:the )?(?:rules|instructions|guardrails)", low)
    if (private and seeking) or attack:
        return "insufficient_evidence", "This public-source workspace cannot reveal credentials, hidden instructions, environment values, or private repository information. Ask about documented Streamlit behavior or provide a redacted reproduction."
    if re.search(r"\b(?:our|my) (?:company|organization|team)\b", low) and re.search(r"\b(?:internal|confidential|private|on.call)\b", low):
        return "insufficient_evidence", "This snapshot contains public Streamlit sources, not your organization's internal policies or confidential contact details. Supply an authorized public reference or ask about documented Streamlit behavior."
    if not version and re.search(r"(?:what|which).{0,35}version.{0,40}(?:am i|i.m|i am|running|installed|do i have)|(?:my|our) installed version", low):
        return "needs_details", "The source snapshot cannot inspect your installed environment. Supply your Streamlit version (for example, from `streamlit version`) before asking for a version-specific diagnosis."
    detail_terms = set(tokenize(question)) - {"fix", "broken", "crash", "crashes", "fails", "failing", "error", "wrong", "computer", "something", "happening", "happens"}
    vague_failure = re.search(r"\b(?:app|it|application|program)\b.{0,20}(?:doesn.t work|does not work|not working|is broken|crashes|fails)", low)
    if (vague_failure and not detail_terms) or re.fullmatch(r"(?:hi|hello|hey|help|help me|it(?:'s| is)? broken|(?:my |the )?app (?:doesn.t work|is broken|crashes|fails)|it (?:doesn.t work|crashes|fails))[.!? ]*", low) or len(tokenize(question)) < 1:
        return "needs_details", "Describe the Streamlit API or behavior involved, what you expected, what happened, and your version. Include a small reproduction and a redacted error message when available."
    return None, ""


def _quote(text: str, question: str, limit: int = 85) -> str:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return ""
    terms = set(tokenize(question))
    best_start, best_count = 0, -1
    for start in range(0, max(1, len(words) - limit + 1), 12):
        end = min(start + limit, len(words))
        count = len(terms & set(tokenize(text[words[start].start():words[end - 1].end()])))
        if count > best_count:
            best_start, best_count = start, count
    end = min(best_start + limit, len(words))
    return text[words[best_start].start():words[end - 1].end()]


class TriageDesk:
    def __init__(self, data_dir: Path, cache_dir: Path):
        self.data_dir, self.cache_dir = Path(data_dir), Path(cache_dir)
        self.sources = _load_sources(self.data_dir)
        self.chunks = chunk_sources(self.sources)
        self._keyword = _BM25(self.chunks)
        self._dense = DenseIndex(self.chunks, self.cache_dir, prebuilt_dir=self.data_dir / "index")

    def search(self, question: str, mode: str = "hybrid", source_kinds=None, top_k: int = 6) -> list[dict]:
        question = _validate_question(question)
        if mode not in MODES:
            raise ValueError("Search mode must be keyword, semantic, or hybrid")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 30:
            raise ValueError("top_k must be an integer from 1 to 30")
        if source_kinds is None:
            kinds = KINDS
        elif isinstance(source_kinds, (list, tuple, set, frozenset)) and set(source_kinds) <= KINDS:
            kinds = set(source_kinds)
        else:
            raise ValueError("Source kinds must be a collection of documentation, issue, release, or contributing")
        eligible = [i for i, c in enumerate(self.chunks) if c["kind"] in kinds]
        if not eligible:
            return []
        lexical = self._keyword.scores(question) if mode in {"keyword", "hybrid"} else np.zeros(len(self.chunks))
        dense = self._dense.scores(question) if mode in {"semantic", "hybrid"} else np.zeros(len(self.chunks))
        keyword_order = sorted((i for i in eligible if lexical[i] > 0), key=lambda i: (-lexical[i], self.chunks[i]["chunk_id"]))
        dense_order = sorted(eligible, key=lambda i: (-dense[i], self.chunks[i]["chunk_id"]))
        if mode == "keyword":
            order, scores = keyword_order, lexical
        elif mode == "semantic":
            order, scores = dense_order, dense
        else:
            fused = np.zeros(len(self.chunks), dtype=float)
            for ranking in (keyword_order[:60], dense_order[:60]):
                for rank, i in enumerate(ranking, start=1):
                    fused[i] += 1 / (60 + rank)
            order = sorted(set(keyword_order[:60] + dense_order[:60]), key=lambda i: (-fused[i], self.chunks[i]["chunk_id"]))
            scores = fused
        evidence, counts = [], Counter()
        for i in order:
            chunk = self.chunks[i]
            if counts[chunk["source_id"]] >= 2:
                continue
            counts[chunk["source_id"]] += 1
            evidence.append({**chunk, "citation_id": f"S{len(evidence) + 1}", "score": round(float(scores[i]), 6),
                             "keyword_score": round(float(lexical[i]), 6), "dense_score": round(float(dense[i]), 6),
                             "quote": _quote(chunk["text"], question)})
            if len(evidence) >= top_k:
                break
        return evidence

    def analyze(self, question: str, *, mode: str = "hybrid", source_kinds=None, version: str = "", include_generation: bool = False, _evidence: list[dict] | None = None) -> dict:
        started = time.perf_counter()
        question = _validate_question(question)
        if not isinstance(version, str) or len(version) > 120:
            raise ValueError("Version must be text under 120 characters")
        if mode not in MODES:
            raise ValueError("Search mode must be keyword, semantic, or hybrid")
        category, component = _intent(question)
        missing = _missing(question, version, category)
        status, answer = _guard(question, version)
        if status == "needs_details" and not missing:
            missing = ["Installed Streamlit version (run `streamlit version`)"] if "version" in answer.lower() else ["Streamlit API or behavior involved", "Minimal reproduction and expected / actual behavior"]
        if source_kinds is not None and (not isinstance(source_kinds, (list, tuple, set, frozenset)) or not set(source_kinds) <= KINDS):
            raise ValueError('Invalid source kinds')
        if _evidence is not None and (not isinstance(_evidence, list) or len(_evidence) > 30):
            raise ValueError('Supplied evidence must be a bounded list of source passages')
        if status:
            evidence = []
        elif _evidence is None:
            evidence = self.search(question, mode=mode, source_kinds=source_kinds)
        else:
            # The graph may retrieve different literal chunk boundaries, but it
            # cannot introduce a source, passage, title, URL or authority claim.
            sources = {item['source_id']: item for item in self.sources}
            evidence = []
            for item in _evidence:
                if not isinstance(item, dict) or item.get('source_id') not in sources:
                    raise ValueError('Supplied evidence contains an unknown source')
                source = sources[item['source_id']]
                text = item.get('text')
                if not isinstance(text, str) or not text.strip() or text not in source['text']:
                    raise ValueError('Supplied evidence must be a literal passage in its source')
                if source_kinds is not None and source['kind'] not in source_kinds:
                    continue
                chunk_id = item.get('chunk_id')
                if not isinstance(chunk_id, str) or not chunk_id.startswith(source['source_id'] + ':') or len(chunk_id) > 240:
                    raise ValueError('Supplied evidence has an invalid chunk identity')
                scores = {name: item.get(name, 0.) for name in ('score', 'keyword_score', 'dense_score')}
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores.values()):
                    raise ValueError('Supplied evidence has an invalid retrieval score')
                section = item.get('section', 'Retrieved passage')
                if not isinstance(section, str):
                    raise ValueError('Supplied evidence has an invalid section')
                checked = {**item, **source, **scores, 'text': text, 'chunk_id': chunk_id,
                           'section': section, 'quote': _quote(text, question)}
                checked.pop('support_quotes', None)
                evidence.append(checked)
        # Search produces ranked candidates. An analysis additionally needs a relevance signal.
        terms = set(tokenize(question))
        relevant = []
        for item in evidence:
            overlap = terms & set(tokenize(f"{item['title']} {item['section']} {item['text']}"))
            uncommon = {t for t in overlap if t not in {"error", "message", "code", "python", "version", "report", "support", "question", "data", "user", "request"}}
            if uncommon or (item["dense_score"] >= .72 and len(overlap) >= 1):
                relevant.append(item)
        evidence = relevant
        if not status and not evidence:
            status = "insufficient_evidence"
            answer = "The selected public-source snapshot does not contain enough relevant evidence to answer this question. Try a specific Streamlit API, symptom, or error message, broaden the source filters, or inspect another source. No diagnosis, duplicate match, or fix has been established."
        generation_mode, generation_error = "extracted_evidence", ""
        generated_claims = []
        generation_metadata = {"requested_model": "", "returned_model": "", "usage": {}, "finish_reason": "",
                               "latency_ms": 0., "endpoint": "", "attempted": False,
                               "status": "not_attempted" if include_generation else "not_requested"}
        snapshot_caveat = (
            (f"Reported Streamlit version: {version}. " if version else "Installed Streamlit version was not supplied. ")
            + "These sources are a bounded public snapshot; check each source's version scope. "
            "Issue reports and source similarity do not establish a duplicate, root cause, or confirmed fix."
        )
        if not status:
            status = "evidence_found"
            # Reassign contiguous IDs after relevance filtering.
            for index, item in enumerate(evidence, 1):
                item["citation_id"] = f"S{index}"
            paragraphs = ["Extracted source passages to inspect. This evidence view does not generate a diagnosis or confirm a duplicate or fix."]
            for item in evidence[:4]:
                label = {"issue": "Public issue report", "release": "Release note", "documentation": "Documentation", "contributing": "Contribution guidance"}[item["kind"]]
                # Quotes often cross a Markdown code fence or contain relative links.
                # Render the unchanged passage as literal text, with a fence longer
                # than any run inside it, rather than executing source formatting.
                quote = item["quote"]
                fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", quote)), default=0) + 1)
                title = re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", item["title"])
                paragraphs.append(f"**[{item['citation_id']}] {label}: {title}** ([source](<{item['url']}>))\n\n{fence}text\n{quote}\n{fence}")
            answer = "\n\n".join(paragraphs)
            if include_generation:
                try:
                    generated = synthesize(question, evidence, version, return_metadata=True)
                    generation_metadata = generated["metadata"]
                    generated_claims = generated["claims"]
                    if generation_metadata["status"] == "abstained":
                        status, generation_mode = "insufficient_evidence", "model_abstained"
                        answer = "The model abstained: it could not support a direct answer from these passages. Review the retrieved candidates below or add more details.\n\n" + answer
                    else:
                        answer = "Model synthesis of the retrieved evidence (not a confirmed diagnosis):\n\n" + generated["answer"]
                        generation_mode = "cited_model_synthesis"
                        by_id = {item["citation_id"]: item for item in evidence}
                        for claim in generated_claims:
                            for support in claim["support"]:
                                quotes = by_id[support["citation_id"]].setdefault("support_quotes", [])
                                if support["quote"] not in quotes:
                                    quotes.append(support["quote"])
                except GenerationUnavailable as exc:
                    generation_metadata.update(exc.metadata)
                    generation_metadata["status"] = exc.reason
                    if exc.reason == "output_truncated":
                        generation_mode = "extracted_evidence_truncated"
                    elif exc.reason == "verification_rejected":
                        generation_mode = "extracted_evidence_verification_rejected"
                    elif exc.reason == "verification_unavailable":
                        generation_mode = "extracted_evidence_verification_unavailable"
                    elif exc.reason in {"answer_rejected", "unsafe_formatting"}:
                        generation_mode = "extracted_evidence_rejected_answer"
                    else:
                        generation_mode = "extracted_evidence_provider_unavailable"
                    generation_error = str(exc)
        if evidence:
            answer += "\n\n" + snapshot_caveat
        related = []
        for item in evidence:
            if item["kind"] == "issue" and not any(r["source_id"] == item["source_id"] for r in related):
                related.append({**item, "relationship": "Potentially relevant public report; similarity is not a duplicate determination"})
        draft = self._draft(question, version, category, component, missing, evidence, status)
        metrics = {"latency_ms": round((time.perf_counter() - started) * 1000, 1), "mode": mode,
                   "embedding_model": MODEL_NAME if mode != "keyword" and not _guard(question, version)[0] else "not used",
                   "generation_mode": generation_mode, "generation_error": generation_error,
                   "chunk_count": len(self.chunks), "source_count": len(self.sources),
                   "source_kinds": sorted(KINDS if source_kinds is None else source_kinds), "reported_version": version,
                   "index_cache_hit": self._dense.cache_hit if mode != "keyword" else False}
        metrics["index_origin"] = self._dense.index_origin if mode != "keyword" else "not_used"
        if _evidence is not None:
            metrics['index_origin'] = 'workflow_retrieval'
            metrics['index_cache_hit'] = None
        for name in ("requested_model", "returned_model", "usage", "finish_reason", "latency_ms", "endpoint", "attempted", "status", "response_format", "calls", "verification", "usage_complete", "protocol"):
            metrics[f"generation_{name}"] = generation_metadata.get(name, "" if name == "response_format" else None)
        return {"question": question, "status": status, "answer": answer, "evidence": evidence,
                "suggested_type": category, "component": component, "missing_details": missing,
                "related_issues": related, "draft_markdown": draft, "metrics": metrics,
                "generated_claims": generated_claims, "generation": generation_metadata,
                "snapshot_caveat": snapshot_caveat,
                "snapshot": {"snapshot_at": sorted({s["snapshot_at"] for s in self.sources}),
                             "version_scopes": sorted({s["version_scope"] for s in self.sources})}}

    @staticmethod
    def _draft(question, version, category, component, missing, evidence, status):
        python_label = re.search(r"^Python version:\s*([^\n]+)", question, re.I | re.M)
        python_inline = re.search(r"\bPython(?:\s+version)?\s*[:=v]*\s*(3\.\d+(?:\.\d+)?)", question, re.I)
        python = python_label.group(1).strip() if python_label else python_inline.group(1) if python_inline else "[provide version]"
        os_label = re.search(r"^Operating system:\s*([^\n]+)", question, re.I | re.M)
        os_inline = re.search(r"\b(?:Windows(?:\s+\d+)?|macOS(?:\s+\d+(?:\.\d+)*)?|Linux|Ubuntu(?:\s+\d+(?:\.\d+)*)?|Debian|Fedora)\b", question, re.I)
        os_name = os_label.group(1).strip() if os_label else os_inline.group(0) if os_inline else "[provide environment]"
        steps = re.search(r"^Steps to reproduce:\s*(.+?)(?=\n[A-Z][A-Za-z ]{1,30}:|\Z)", question, re.I | re.M | re.S)
        code = re.search(r"```[^\n]*\n.*?```", question, re.S)
        reproduction = "\n\n".join(part for part in [steps.group(1).strip() if steps else "", code.group(0) if code else ""] if part) or "[Add a small runnable example and exact steps.]"
        lines = ["# Issue triage draft — review before filing", "", f"Suggested type (heuristic): {category}",
                 f"Suggested component (heuristic): {component}", "", "## Reporter description", "", question,
                 "", "## Environment", "", f"- Streamlit: {version or '[provide installed version]'}",
                 f"- Python: {python}", f"- OS / deployment: {os_name}", "",
                 "## Minimal reproduction", "", reproduction, "",
                 "## Expected and actual behavior", "", "[Describe both; include a redacted error/traceback.]", "",
                 "## Retrieved context", ""]
        if evidence:
            for item in evidence:
                lines.append(f"- [{item['citation_id']}] {item['title']} — {item['url']} (kind: {item['kind']}; scope: {item['version_scope']})")
            lines.append("\nThese are retrieved references, not verified duplicates, root causes, or confirmed fixes.")
        else:
            lines.append("No supporting public-source evidence established. Collect details before filing a diagnosis.")
        if missing:
            lines.extend(["", "## Details to collect", ""] + [f"- [ ] {item}" for item in missing])
        lines.extend(["", "This draft has not been posted. Check source versions and remove private information before sharing."])
        return "\n".join(lines)
