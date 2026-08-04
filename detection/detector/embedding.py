
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from .schema import TraceEvent


class EmbeddingState:
    """Maintains embeddings and FAISS index for semantic similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.events: List[TraceEvent] = []

    def _serialize(self, e: TraceEvent) -> str:
        return (
            f"Q: {e.question}"
            f"SUBTASK: {e.subtask}"
            f"ACTION: {e.action}"
            f"RESP: {e.response}"
        )

    def update(self, e: TraceEvent, top_k: int = 5, similarity_threshold: float = 0.95) -> dict:
        text = self._serialize(e)
        emb = self.model.encode([text], normalize_embeddings=True)
        vec = np.array(emb, dtype="float32")
        self.index.add(vec)
        self.events.append(e)

        flags = []
        info = {}

        if self.index.ntotal > top_k:
            scores, ids = self.index.search(vec, top_k)
            best_score = float(scores[0][1])
            best_idx = int(ids[0][1])
            info["best_score"] = best_score
            info["best_neighbor_loop_idx"] = self.events[best_idx].loop_idx
            if best_score >= similarity_threshold:
                flags.append("semantic_repeat")

        info["flags"] = flags
        return info
