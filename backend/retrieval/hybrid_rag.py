from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from backend.retrieval.bm25 import BM25Index
from backend.retrieval.hybrid import tokenize
from backend.retrieval.vector_store import InMemoryVectorIndex
from backend.schemas.models import (
    EvidenceChunkResult,
    GroupedPatentEvidence,
    KeywordSet,
    PatentChunk,
)


class HybridRAGRetriever:
    """Chunk-level hybrid retriever for patent evidence retrieval."""

    def __init__(self, chunks: Sequence[PatentChunk]):
        self.chunks = list(chunks)
        self.bm25 = BM25Index(self.chunks)
        self.vector_index = InMemoryVectorIndex(self.chunks)

    def search(
        self,
        keywords: KeywordSet,
        top_k_patents: int = 8,
        top_k_chunks_per_patent: int = 3,
        recall_k: int = 30,
        filters: Dict | None = None,
    ) -> List[GroupedPatentEvidence]:
        queries = self._queries_from_keywords(keywords)
        chunk_scores: Dict[int, float] = defaultdict(float)
        chunk_bm25_rank: Dict[int, int] = {}
        chunk_vector_rank: Dict[int, int] = {}
        chunk_matched_terms: Dict[int, set] = defaultdict(set)

        for query in queries:
            bm25_results = self.bm25.search(query, top_k=recall_k)
            vector_results = self.vector_index.search(query, top_k=recall_k)

            for rank, (chunk_idx, _score, matched_terms) in enumerate(bm25_results, start=1):
                chunk_scores[chunk_idx] += self._rrf(rank)
                chunk_bm25_rank[chunk_idx] = min(chunk_bm25_rank.get(chunk_idx, rank), rank)
                chunk_matched_terms[chunk_idx].update(matched_terms)

            for rank, (chunk_idx, _score) in enumerate(vector_results, start=1):
                chunk_scores[chunk_idx] += self._rrf(rank)
                chunk_vector_rank[chunk_idx] = min(chunk_vector_rank.get(chunk_idx, rank), rank)

        evidence_results = []
        for chunk_idx, score in chunk_scores.items():
            chunk = self.chunks[chunk_idx]
            if not self._passes_filters(chunk, filters):
                continue

            evidence_results.append(
                EvidenceChunkResult(
                    chunk=chunk,
                    score=round(score, 5),
                    bm25_rank=chunk_bm25_rank.get(chunk_idx),
                    vector_rank=chunk_vector_rank.get(chunk_idx),
                    matched_terms=sorted(chunk_matched_terms.get(chunk_idx, set()))[:12],
                    retrieval_reason=self._retrieval_reason(chunk_bm25_rank.get(chunk_idx), chunk_vector_rank.get(chunk_idx)),
                )
            )

        evidence_results = sorted(evidence_results, key=lambda item: item.score, reverse=True)
        grouped = self._group_by_patent(evidence_results, top_k_chunks_per_patent)
        return sorted(grouped, key=lambda item: item.score, reverse=True)[:top_k_patents]

    def _queries_from_keywords(self, keywords: KeywordSet) -> List[str]:
        queries = []
        queries.extend(keywords.query_groups)
        queries.append(" ".join(keywords.core_terms[:10]))
        queries.append(" ".join((keywords.core_terms + keywords.synonyms)[:16]))
        queries.append(" ".join(keywords.english_terms[:16]))
        queries.extend(keywords.classification_hints)
        return [query for query in queries if query and query.strip()]

    def _group_by_patent(
        self,
        evidence_results: Sequence[EvidenceChunkResult],
        top_k_chunks_per_patent: int,
    ) -> List[GroupedPatentEvidence]:
        grouped_chunks: Dict[str, List[EvidenceChunkResult]] = defaultdict(list)
        for result in evidence_results:
            if len(grouped_chunks[result.chunk.patent_id]) < top_k_chunks_per_patent:
                grouped_chunks[result.chunk.patent_id].append(result)

        grouped = []
        for patent_id, chunks in grouped_chunks.items():
            first = chunks[0].chunk
            grouped.append(
                GroupedPatentEvidence(
                    patent_id=patent_id,
                    title=first.title,
                    assignee=first.assignee,
                    publication_date=first.publication_date,
                    jurisdiction=first.jurisdiction,
                    cpc=first.cpc,
                    source_url=first.source_url,
                    score=round(sum(item.score for item in chunks), 5),
                    evidence_chunks=chunks,
                )
            )
        return grouped

    def _passes_filters(self, chunk: PatentChunk, filters: Dict | None) -> bool:
        if not filters:
            return True

        jurisdiction = filters.get("jurisdiction")
        if jurisdiction and chunk.jurisdiction != jurisdiction:
            return False

        section = filters.get("section")
        if section and chunk.section != section:
            return False

        cpc_prefix = filters.get("cpc_prefix")
        if cpc_prefix and not any(code.startswith(cpc_prefix) for code in chunk.cpc):
            return False

        return True

    def _retrieval_reason(self, bm25_rank: int | None, vector_rank: int | None) -> str:
        parts = []
        if bm25_rank is not None:
            parts.append(f"BM25 rank {bm25_rank}")
        if vector_rank is not None:
            parts.append(f"vector rank {vector_rank}")
        return "; ".join(parts) or "hybrid retrieval"

    def _rrf(self, rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)
