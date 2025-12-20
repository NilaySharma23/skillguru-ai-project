
from RAG_PIPELINE import (
    retrieve_with_rerank,
)

class KritikaRAGAdapter:
    """
    Adapter to use Kritika's RAG pipeline safely inside the tutor.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def should_use_rag(self, weakness_score, difficulty):
        """
        Decide whether RAG is needed.
        """
        if not self.enabled:
            return False

        if weakness_score >= 3:
            return True

        if difficulty in ["beginner", "intermediate"]:
            return True

        return False

    def retrieve_context(self, query, top_k=3):
        """
        Retrieve grounded context using BM25 + Chroma.
        """
        try:
            chunks = retrieve_with_rerank(query, top_k=top_k)
            return "\n\n".join(chunks)
        except Exception as e:
            print(" RAG retrieval failed:", e)
            return ""
