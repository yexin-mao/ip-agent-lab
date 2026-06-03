from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

from backend.schemas.models import PatentDocument, SearchResult


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def _term_overlap(query_terms: Sequence[str], text_terms: Sequence[str]) -> tuple[float, List[str]]:
    if not query_terms or not text_terms:
        return 0.0, []
    query_counter = Counter(query_terms)
    text_counter = Counter(text_terms)
    matched = sorted(set(query_counter).intersection(text_counter))
    overlap = sum(min(query_counter[t], text_counter[t]) for t in matched)
    score = overlap / max(len(query_terms), 1)
    return score, matched


def _cosine_counter(left: Iterable[str], right: Iterable[str]) -> float:
    left_counter = Counter(left)
    right_counter = Counter(right)
    if not left_counter or not right_counter:
        return 0.0
    common = set(left_counter).intersection(right_counter)
    dot = sum(left_counter[t] * right_counter[t] for t in common)
    left_norm = math.sqrt(sum(v * v for v in left_counter.values()))
    right_norm = math.sqrt(sum(v * v for v in right_counter.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalHybridRetriever:
    """Small local retriever for the MVP.

    It combines lexical overlap and bag-of-words cosine similarity. Replace this
    with BM25 + Qdrant + reranker when external services are added.
    """

    def __init__(self, documents: Sequence[PatentDocument]):
        self.documents = list(documents)

    def search(self, queries: Sequence[str], limit: int = 8) -> List[SearchResult]:
        query_text = " ".join(queries)
        query_terms = tokenize(query_text)
        results: List[SearchResult] = []

        for doc in self.documents:
            doc_text = " ".join([doc.title, doc.abstract, doc.claims, " ".join(doc.cpc)])
            doc_terms = tokenize(doc_text)
            overlap_score, matched_terms = _term_overlap(query_terms, doc_terms)
            cosine_score = _cosine_counter(query_terms, doc_terms)
            cpc_bonus = 0.08 if any(term.lower().startswith("h04") for term in matched_terms + doc.cpc) else 0.0
            score = min((0.6 * overlap_score) + (0.4 * cosine_score) + cpc_bonus, 1.0)

            if score <= 0:
                continue

            results.append(
                SearchResult(
                    document=doc,
                    score=round(score, 4),
                    matched_terms=matched_terms[:12],
                    retrieval_reason=(
                        f"Matched {len(matched_terms)} technical terms; "
                        f"lexical={overlap_score:.2f}, semantic_proxy={cosine_score:.2f}"
                    ),
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]
