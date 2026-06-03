from __future__ import annotations

from typing import List

from backend.retrieval.hybrid import LocalHybridRetriever
from backend.schemas.models import KeywordSet, SearchResult


class PatentSearchAgent:
    def __init__(self, retriever: LocalHybridRetriever):
        self.retriever = retriever

    def run(self, keywords: KeywordSet, limit: int = 8) -> List[SearchResult]:
        queries = keywords.query_groups + keywords.english_terms + keywords.classification_hints
        return self.retriever.search(queries=queries, limit=limit)
