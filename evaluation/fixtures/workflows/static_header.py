# Frozen source excerpt: triage_desk/static_analysis.py:70-77
def _safe_label(value: str) -> str:
    if not isinstance(value, str):
        raise StaticAnalysisUnavailable('invalid_header')
    path = PurePosixPath(value)
    if (len(value) > 500 or not re.fullmatch(r'[A-Za-z0-9_./-]+\.py', value)
            or path.is_absolute() or any(part in {'.', '..'} or part.startswith('.') for part in value.split('/'))):
        raise StaticAnalysisUnavailable('invalid_header')
    return value
