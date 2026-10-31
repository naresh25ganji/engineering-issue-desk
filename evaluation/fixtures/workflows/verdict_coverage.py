# Frozen source excerpt: triage_desk/agent_review.py:288-301
def _validate_verdicts(verdicts: list[EvidenceVerdict], count: int) -> list[dict]:
    indices = [verdict.index for verdict in verdicts]
    if len(indices) != count or set(indices) != set(range(count)):
        raise ReviewUnavailable(code='incomplete_coverage')
    normalized = []
    for verdict in verdicts:
        # Removing paired single-backtick delimiters changes presentation only.
        # Check the resulting text so split/hidden URLs and HTML remain unsafe.
        # Fences, doubled or unmatched delimiters remain and are rejected.
        reason = _INLINE_CODE.sub(r'\1', verdict.reason).strip()
        if not reason or _REASON_UNSAFE.search(reason):
            raise ReviewUnavailable(code='invalid_reason')
        normalized.append({**verdict.model_dump(), 'reason': reason})
    return sorted(normalized, key=lambda verdict: verdict['index'])
