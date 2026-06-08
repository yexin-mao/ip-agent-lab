from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from backend.retrieval.bm25 import BM25Index
from backend.retrieval.chunker import PatentChunker
from backend.retrieval.embedding_factory import create_llamaindex_embedding
from backend.retrieval.hybrid import tokenize
from backend.schemas.models import (
    EvidenceChunkResult,
    GroupedPatentEvidence,
    KeywordSet,
    PatentChunk,
    PatentDocument,
    RetrievalQuery,
)

try:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import TextNode
except ImportError:  # pragma: no cover - exercised only when optional deps are missing.
    VectorStoreIndex = None
    BaseEmbedding = object
    TextNode = None


class LlamaIndexPatentRetriever:
    """Patent evidence retriever backed by LlamaIndex VectorStoreIndex."""

    def __init__(
        self,
        documents: Sequence[PatentDocument],
        embedding_model: BaseEmbedding | None = None,
    ):
        self._ensure_llamaindex_available()
        self.chunks = PatentChunker().chunk_documents(documents)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.chunk_index_by_id = {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}
        self.bm25 = BM25Index(self.chunks)
        self.embedding_model = embedding_model or create_llamaindex_embedding()
        self.index = VectorStoreIndex(
            nodes=self._build_nodes(self.chunks),
            embed_model=self.embedding_model,
        )

    def search(
        self,
        keywords: KeywordSet,
        top_k_patents: int = 8,
        top_k_chunks_per_patent: int = 3,
        recall_k: int = 30,
        filters: Dict | None = None,
    ) -> List[GroupedPatentEvidence]:
        queries = self._queries_from_keywords(keywords)
        chunk_scores: Dict[str, float] = defaultdict(float)
        chunk_bm25_rank: Dict[str, int] = {}
        chunk_vector_rank: Dict[str, int] = {}
        chunk_matched_terms: Dict[str, set] = defaultdict(set)
        chunk_query_sources: Dict[str, set] = defaultdict(set)
        chunk_retrieval_sources: Dict[str, set] = defaultdict(set)

        retriever = self.index.as_retriever(similarity_top_k=recall_k)
        for query in queries:
            bm25_results = self.bm25.search(query.query_text, top_k=recall_k)
            for rank, (chunk_idx, _score, matched_terms) in enumerate(bm25_results, start=1):
                chunk = self.chunks[chunk_idx]
                if not self._passes_filters(chunk, filters):
                    continue

                chunk_scores[chunk.chunk_id] += query.weight * self._rrf(rank)
                chunk_bm25_rank[chunk.chunk_id] = min(chunk_bm25_rank.get(chunk.chunk_id, rank), rank)
                chunk_matched_terms[chunk.chunk_id].update(matched_terms)
                chunk_query_sources[chunk.chunk_id].add(query.source)
                chunk_retrieval_sources[chunk.chunk_id].add("bm25")

            vector_results = retriever.retrieve(query.query_text)
            for rank, node_with_score in enumerate(vector_results, start=1):
                node = node_with_score.node
                chunk_id = str(node.metadata.get("chunk_id") or node.node_id)
                chunk = self.chunk_by_id.get(chunk_id)
                if not chunk or not self._passes_filters(chunk, filters):
                    continue

                chunk_scores[chunk_id] += query.weight * self._rrf(rank)
                chunk_vector_rank[chunk_id] = min(chunk_vector_rank.get(chunk_id, rank), rank)
                chunk_matched_terms[chunk_id].update(self._matched_terms(query.query_text, chunk.text))
                chunk_query_sources[chunk_id].add(query.source)
                chunk_retrieval_sources[chunk_id].add("vector")

        evidence_results = []
        for chunk_id, score in chunk_scores.items():
            chunk = self.chunk_by_id[chunk_id]
            bm25_rank = chunk_bm25_rank.get(chunk_id)
            vector_rank = chunk_vector_rank.get(chunk_id)
            evidence_results.append(
                EvidenceChunkResult(
                    chunk=chunk,
                    score=round(score, 5),
                    bm25_rank=bm25_rank,
                    vector_rank=vector_rank,
                    matched_terms=sorted(chunk_matched_terms.get(chunk_id, set()))[:12],
                    retrieval_reason=self._retrieval_reason(bm25_rank, vector_rank),
                    query_sources=sorted(chunk_query_sources.get(chunk_id, set())),
                    retrieval_sources=sorted(chunk_retrieval_sources.get(chunk_id, set())),
                )
            )

        evidence_results = sorted(evidence_results, key=lambda item: item.score, reverse=True)
        grouped = self._group_by_patent(evidence_results, top_k_chunks_per_patent)
        return sorted(grouped, key=lambda item: item.score, reverse=True)[:top_k_patents]

    def _build_nodes(self, chunks: Sequence[PatentChunk]):
        nodes = []
        for chunk in chunks:
            nodes.append(
                TextNode(
                    id_=chunk.chunk_id,
                    text=self._chunk_text(chunk),
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "patent_id": chunk.patent_id,
                        "title": chunk.title,
                        "section": chunk.section,
                        "assignee": chunk.assignee,
                        "publication_date": chunk.publication_date,
                        "jurisdiction": chunk.jurisdiction,
                        "cpc": ", ".join(chunk.cpc),
                        "source_url": chunk.source_url,
                    },
                )
            )
        return nodes

    def _chunk_text(self, chunk: PatentChunk) -> str:
        return " ".join([chunk.title, chunk.text, " ".join(chunk.cpc)])

    def _queries_from_keywords(self, keywords: KeywordSet) -> List[RetrievalQuery]:
        if keywords.retrieval_queries:
            return [
                query
                for query in keywords.retrieval_queries
                if query.query_text and query.query_text.strip()
            ]

        queries = []
        queries.extend(keywords.query_groups)
        queries.append(" ".join(keywords.core_terms[:10]))
        queries.append(" ".join((keywords.core_terms + keywords.synonyms)[:16]))
        queries.append(" ".join(keywords.english_terms[:16]))
        queries.extend(keywords.classification_hints)
        return [
            RetrievalQuery(query_text=query, source="legacy_keyword_set", weight=1.0)
            for query in queries
            if query and query.strip()
        ]

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

    def _matched_terms(self, query: str, text: str) -> List[str]:
        query_terms = set(tokenize(query))
        text_terms = set(tokenize(text))
        return sorted(query_terms.intersection(text_terms))

    def _retrieval_reason(self, bm25_rank: int | None, vector_rank: int | None) -> str:
        parts = []
        if bm25_rank is not None:
            parts.append(f"BM25 rank {bm25_rank}")
        if vector_rank is not None:
            parts.append(f"LlamaIndex vector rank {vector_rank}")
        return "; ".join(parts) or "LlamaIndex hybrid retrieval"

    def _rrf(self, rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _ensure_llamaindex_available(self) -> None:
        if VectorStoreIndex is None or TextNode is None:
            raise ImportError(
                "LlamaIndex retrieval requires llama-index-core. "
                "Install project dependencies with `pip install -r requirements.txt`."
            )
