from __future__ import annotations

from typing import List, Protocol

from backend.schemas.models import GroupedPatentEvidence, KeywordSet


class PatentEvidenceRetriever(Protocol):
    def search(self, keywords: KeywordSet, top_k_patents: int = 8) -> List[GroupedPatentEvidence]:
        ...


class PatentSearchAgent:
    def __init__(self, retriever: PatentEvidenceRetriever):
        self.retriever = retriever

    def run(self, keywords: KeywordSet, limit: int = 8) -> List[GroupedPatentEvidence]:
        return self.retriever.search(keywords=keywords, top_k_patents=limit)
