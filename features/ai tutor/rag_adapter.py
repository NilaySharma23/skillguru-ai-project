from updated_rag import retrieve_with_rerank


class KritikaRAGAdapter:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def retrieve(self, query: str, top_k: int = 3):
        if not self.enabled:
            return []

        try:
            return retrieve_with_rerank(query, top_k=top_k) or []
        except Exception:
            # Silent failure by design
            return []
