"""Validated, reproducible ticket data preparation without model dependencies.

The held-out CSV is never included in either training export. Normalized text
duplicates are rejected rather than allowing the same ticket across the split.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from .config import LABELS, TEMPLATE


NORMALIZATION = "Unicode NFKC, casefold, collapse Unicode whitespace"
INPUT_PREFIX = "Support Ticket: "


@dataclass(frozen=True)
class Ticket:
    id: str
    text: str
    label: str
    text_sha256: str
    normalized_sha256: str


def _sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize only for identity/duplicate detection; retain original input text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _validate_tickets(tickets: Iterable[Ticket]) -> list[Ticket]:
    rows = list(tickets)
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    for ticket in rows:
        if not isinstance(ticket, Ticket):
            raise ValueError("Expected Ticket records.")
        if ticket.label not in LABELS:
            raise ValueError(f"Unknown label for ticket {ticket.id}: {ticket.label!r}")
        if not isinstance(ticket.text, str) or not normalize_text(ticket.text):
            raise ValueError(f"Missing ticket text for {ticket.id}.")
        if any(token in ticket.text for token in ("\x00", "<|im_start|>", "<|im_end|>")):
            raise ValueError(f"Reserved chat delimiter or invalid text in {ticket.id}.")
        normalized_hash = _sha256(normalize_text(ticket.text))
        if ticket.text_sha256 != _sha256(ticket.text) or ticket.normalized_sha256 != normalized_hash:
            raise ValueError(f"Text/hash mismatch for ticket {ticket.id}.")
        if ticket.id != f"ticket-{normalized_hash[:20]}":
            raise ValueError(f"Unstable ticket ID for {ticket.id}.")
        if ticket.id in seen_ids or normalized_hash in seen_text:
            raise ValueError(f"Normalized duplicate ticket detected: {ticket.id}.")
        seen_ids.add(ticket.id)
        seen_text[normalized_hash] = ticket.label
    return rows


def load_tickets(csv_path: str | Path) -> list[Ticket]:
    """Read original or exported CSV; unknown/blank labels and malformed rows fail.

    Original files use ``category_truth``. Exported split files use ``label``;
    both contain ``text``. Optional exported IDs/hashes are verified, not trusted.
    """
    return _load_tickets_bytes(Path(csv_path).read_bytes())


def _load_tickets_bytes(source: bytes) -> list[Ticket]:
    try:
        content = source.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Ticket CSV must be UTF-8.") from exc
    reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Ticket CSV needs a unique header row.")
    if "text" not in fields:
        raise ValueError("Ticket CSV is missing the text column.")
    label_fields = [field for field in ("category_truth", "label") if field in fields]
    if len(label_fields) != 1:
        raise ValueError("Ticket CSV needs exactly one label column: category_truth or label.")
    label_field = label_fields[0]
    tickets: list[Ticket] = []
    try:
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {line}.")
            label, text = row[label_field], row["text"]
            if label not in LABELS:
                raise ValueError(f"Unknown or missing label at row {line}: {label!r}")
            if not normalize_text(text) or "\x00" in text:
                raise ValueError(f"Missing or invalid ticket text at row {line}.")
            text_hash = _sha256(text)
            normalized_hash = _sha256(normalize_text(text))
            ticket = Ticket(f"ticket-{normalized_hash[:20]}", text, label, text_hash, normalized_hash)
            for key in ("id", "text_sha256", "normalized_sha256"):
                if key in row and row[key] != getattr(ticket, key):
                    raise ValueError(f"Exported {key} mismatch at row {line}.")
            tickets.append(ticket)
    except csv.Error as exc:
        raise ValueError(f"Malformed ticket CSV near line {reader.line_num}.") from exc
    if not tickets:
        raise ValueError("Ticket CSV contains no tickets.")
    return _validate_tickets(tickets)


def stratified_split(
    tickets: Iterable[Ticket], test_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[Ticket], list[Ticket]]:
    """Split independently by class, allocating test counts by largest remainder.

    The total held-out size is ceil(N * fraction). Label order resolves equal
    allocation remainders. IDs are sorted before seeded shuffling so CSV row
    reordering cannot change split membership. Each present class needs at least
    one training and one test example. This is not sklearn's split algorithm.
    """
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (float, int)):
        raise ValueError("test_fraction must be between zero and one.")
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    rows = _validate_tickets(tickets)
    if not rows:
        raise ValueError("Cannot split an empty dataset.")
    by_label = {label: sorted((row for row in rows if row.label == label), key=lambda row: row.id)
                for label in LABELS}
    present = [label for label in LABELS if by_label[label]]
    if any(len(by_label[label]) < 2 for label in present):
        raise ValueError("Each present label needs at least two independent tickets for a stratified split.")
    fraction = Fraction(str(test_fraction))
    total_test = math.ceil(len(rows) * fraction)
    if total_test < len(present) or total_test > len(rows) - len(present):
        raise ValueError("Requested fraction cannot preserve every present label in both splits.")
    desired = {label: len(by_label[label]) * fraction for label in present}
    counts = {label: max(1, min(len(by_label[label]) - 1, math.floor(desired[label])))
              for label in present}
    # Allocate/reduce bounded quotas until the exact total is reached. The
    # distance to the ideal is recomputed after each allocation for small sets.
    while sum(counts.values()) < total_test:
        eligible = [label for label in present if counts[label] < len(by_label[label]) - 1]
        chosen = max(eligible, key=lambda label: desired[label] - counts[label])
        counts[chosen] += 1
    while sum(counts.values()) > total_test:
        eligible = [label for label in present if counts[label] > 1]
        chosen = max(eligible, key=lambda label: counts[label] - desired[label])
        counts[chosen] -= 1
    rng = random.Random(seed)
    train, heldout = [], []
    for label in present:
        shuffled = by_label[label].copy()
        rng.shuffle(shuffled)
        heldout.extend(shuffled[:counts[label]])
        train.extend(shuffled[counts[label]:])
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(tickets: list[Ticket]) -> bytes:
    stream = io.StringIO(newline="")
    fields = ["id", "text", "label", "text_sha256", "normalized_sha256"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in tickets:
        writer.writerow({field: getattr(row, field) for field in fields})
    return stream.getvalue().encode("utf-8")


def prepare_dataset(
    csv_path: str | Path, output_dir: str | Path, *, system_prompt: str,
    template: str = TEMPLATE, seed: int = 42, test_fraction: float = 0.2,
) -> dict:
    """Export training-only formats, isolated split CSVs, and a hash manifest.

    Existing target files are never overwritten. The source is read only.
    """
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("A nonempty system_prompt is required.")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("A nonempty template is required.")
    source = Path(csv_path)
    source_bytes = source.read_bytes()
    rows = _load_tickets_bytes(source_bytes)
    missing = [label for label in LABELS if not any(row.label == label for row in rows)]
    if missing:
        raise ValueError(f"Dataset is missing required labels: {', '.join(missing)}")
    train, heldout = stratified_split(rows, test_fraction=test_fraction, seed=seed)
    messages = [
        {"messages": [{"role": "system", "content": system_prompt},
                      {"role": "user", "content": INPUT_PREFIX + row.text},
                      {"role": "assistant", "content": row.label}]}
        for row in train
    ]
    sharegpt = [
        {"system": system_prompt,
         "conversations": [{"from": "human", "value": INPUT_PREFIX + row.text},
                           {"from": "gpt", "value": row.label}]}
        for row in train
    ]
    registration = {"support_tickets": {
        "file_name": "train_sharegpt.json", "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
    }}
    files = {
        "train.csv": _csv_bytes(train),
        "heldout.csv": _csv_bytes(heldout),
        "train_sharegpt.json": _json_bytes(sharegpt),
        "train_messages.jsonl": b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                                         for row in messages),
        "dataset_info.json": _json_bytes(registration),
    }
    def counts(split):
        observed = Counter(row.label for row in split)
        return {label: observed[label] for label in LABELS}

    prompt_definition = {"system_prompt": system_prompt, "input_prefix": INPUT_PREFIX,
                         "template": template, "labels": list(LABELS)}
    manifest = {
        "manifest_version": 1,
        "source_file": source.name,
        "source_sha256": _sha256(source_bytes),
        "labels": list(LABELS),
        "normalization": NORMALIZATION,
        "duplicate_policy": "reject normalized exact duplicates, including conflicting labels",
        "split_algorithm": "bounded largest-remainder class allocation; sorted IDs; Python Random shuffle",
        "seed": seed,
        "test_fraction": test_fraction,
        "total_count": len(rows), "train_count": len(train), "heldout_count": len(heldout),
        "label_counts": {"all": counts(rows), "train": counts(train), "heldout": counts(heldout)},
        "split_ids": {"train": [row.id for row in train], "heldout": [row.id for row in heldout]},
        "text_hashes": {row.id: {"text_sha256": row.text_sha256, "normalized_sha256": row.normalized_sha256}
                        for row in sorted(rows, key=lambda item: item.id)},
        "template": template,
        "system_prompt": system_prompt,
        "prompt_sha256": _sha256(system_prompt),
        "prompt_hash_scope": "UTF-8 system prompt; full prompt definition hashed separately",
        "prompt_definition": prompt_definition,
        "prompt_definition_sha256": _sha256(json.dumps(prompt_definition, sort_keys=True, ensure_ascii=False)),
        "input_prefix": INPUT_PREFIX,
        "training_formats": ["train_sharegpt.json", "train_messages.jsonl"],
        "heldout_policy": "heldout.csv is excluded from both training exports",
        "files": {name: {"sha256": _sha256(content), "bytes": len(content)} for name, content in files.items()},
    }
    files["manifest.json"] = _json_bytes(manifest)
    destination = Path(output_dir)
    for name in files:
        if (destination / name).exists():
            raise FileExistsError(f"Refusing to overwrite prepared dataset file: {destination / name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        with (destination / name).open("xb") as handle:
            handle.write(content)
    return manifest
