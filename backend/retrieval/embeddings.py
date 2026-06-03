from __future__ import annotations

import hashlib
import math
from typing import List

from backend.retrieval.hybrid import tokenize


class HashingEmbeddingModel:
    """Deterministic local embedding fallback.

    This is a lightweight stand-in for BGE/OpenAI embeddings. It preserves the
    RAG architecture and can be replaced without changing downstream retrieval.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))
