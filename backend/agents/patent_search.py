from __future__ import annotations

from typing import List

from backend.retrieval.hybrid_rag import HybridRAGRetriever
from backend.schemas.models import GroupedPatentEvidence, KeywordSet


class PatentSearchAgent:
    def __init__(self, retriever: HybridRAGRetriever):
        self.retriever = retriever

    def run(self, keywords: KeywordSet, limit: int = 8) -> List[GroupedPatentEvidence]:
        return self.retriever.search(keywords=keywords, top_k_patents=limit)
