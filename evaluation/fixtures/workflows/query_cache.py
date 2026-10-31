# Frozen source excerpt: triage_desk/hosted_embeddings.py:179-194
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
