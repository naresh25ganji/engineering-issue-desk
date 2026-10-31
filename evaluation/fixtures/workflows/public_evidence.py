# Frozen source excerpt: triage_desk/review_publication.py:56-73
def _public_source(source):
    if source.get('origin') not in {'snapshot', 'live'} or source.get('kind') not in {
        'documentation', 'issue', 'release', 'contributing', 'code',
    }:
        raise ValueError('Publication supports must contain public source evidence only.')
    url = canonical_http_url(source.get('url', ''))
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.hostname not in _PUBLIC_HOSTS
            or parts.port not in (None, 443) or parts.query):
        raise ValueError('Publication supports need approved public HTTPS source links.')
    path = unquote(parts.path)
    if (parts.hostname == 'github.com' and
            (not (path == '/streamlit/streamlit' or path.startswith('/streamlit/streamlit/'))
             or '\\' in path or any(part in {'.', '..'} for part in path.split('/')))):
        raise ValueError('GitHub publication evidence is limited to streamlit/streamlit.')
    if hashlib.sha256(source['text'].encode()).hexdigest() != source.get('sha256'):
        raise ValueError('The registered public evidence failed its integrity check.')
    return url
