from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from backend.retrieval.hybrid import tokenize
from backend.schemas.models import PatentChunk


class BM25Index:
    def __init__(self, chunks: Sequence[PatentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_terms: List[List[str]] = [tokenize(chunk.text + " " + " ".join(chunk.cpc)) for chunk in self.chunks]
        self.doc_lengths = [len(terms) for terms in self.doc_terms]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_doc_freq = self._build_doc_freq(self.doc_terms)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float, List[str]]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        results = []
        query_unique = set(query_terms)
        for idx, terms in enumerate(self.doc_terms):
            term_counts = Counter(terms)
            score = 0.0
            matched_terms = []
            for term in query_unique:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                matched_terms.append(term)
                idf = self._idf(term)
                denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_lengths[idx] / max(self.avg_doc_length, 1))
                score += idf * (tf * (self.k1 + 1) / denominator)

            if score > 0:
                results.append((idx, score, sorted(matched_terms)))

        return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]

    def _build_doc_freq(self, docs: Sequence[Sequence[str]]) -> Dict[str, int]:
        freqs = defaultdict(int)
        for terms in docs:
            for term in set(terms):
                freqs[term] += 1
        return dict(freqs)

    def _idf(self, term: str) -> float:
        doc_count = len(self.doc_terms)
        freq = self.term_doc_freq.get(term, 0)
        return math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
