from __future__ import annotations

from typing import List, Sequence, Tuple

from backend.retrieval.embeddings import HashingEmbeddingModel, cosine_similarity
from backend.schemas.models import PatentChunk


class InMemoryVectorIndex:
    def __init__(self, chunks: Sequence[PatentChunk], embedding_model: HashingEmbeddingModel | None = None):
        self.chunks = list(chunks)
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.vectors = [self.embedding_model.embed(self._chunk_text(chunk)) for chunk in self.chunks]

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        query_vector = self.embedding_model.embed(query)
        scored = []
        for idx, vector in enumerate(self.vectors):
            score = cosine_similarity(query_vector, vector)
            if score > 0:
                scored.append((idx, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _chunk_text(self, chunk: PatentChunk) -> str:
        return " ".join([chunk.title, chunk.text, " ".join(chunk.cpc)])
