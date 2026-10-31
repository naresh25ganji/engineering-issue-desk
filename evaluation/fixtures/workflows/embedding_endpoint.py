# Frozen source excerpt: triage_desk/hosted_embeddings.py:85-92
    def _runtime_key(self) -> str:
        try:
            key, _, base = _configuration()
        except GenerationUnavailable:
            raise EmbeddingUnavailable("Hosted embedding configuration is unavailable; no local fallback was selected.") from None
        if _embedding_endpoint(base) != self.endpoint:
            raise EmbeddingUnavailable("Embedding endpoint changed after index construction. Create a new index so its cache identity matches the endpoint.")
        return key
