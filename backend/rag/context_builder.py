from __future__ import annotations

from typing import List

from backend.schemas.models import GroupedPatentEvidence, RAGCitation


class EvidenceContextBuilder:
    def __init__(self, top_patents: int = 5, top_chunks_per_patent: int = 2, max_chars: int = 8000):
        self.top_patents = top_patents
        self.top_chunks_per_patent = top_chunks_per_patent
        self.max_chars = max_chars

    def build(self, evidence_results: List[GroupedPatentEvidence]) -> str:
        sections = []
        total_chars = 0

        for patent in evidence_results[: self.top_patents]:
            header = (
                f"Patent: {patent.patent_id}\n"
                f"Title: {patent.title}\n"
                f"Assignee: {patent.assignee or 'N/A'}\n"
                f"Publication date: {patent.publication_date or 'N/A'}\n"
                f"Jurisdiction: {patent.jurisdiction or 'N/A'}\n"
            )
            chunk_lines = []
            for evidence in patent.evidence_chunks[: self.top_chunks_per_patent]:
                chunk = evidence.chunk
                chunk_lines.append(
                    "\n".join([
                        f"[{chunk.chunk_id}]",
                        f"Section: {chunk.section}",
                        f"Retrieval: {evidence.retrieval_reason or 'N/A'}",
                        f"Matched terms: {', '.join(evidence.matched_terms) or 'N/A'}",
                        f"Text: {chunk.text}",
                    ])
                )

            block = header + "\n".join(chunk_lines)
            if total_chars + len(block) > self.max_chars:
                remaining = self.max_chars - total_chars
                if remaining > 300:
                    sections.append(block[:remaining])
                break

            sections.append(block)
            total_chars += len(block)

        return "\n\n---\n\n".join(sections)

    def citations(self, evidence_results: List[GroupedPatentEvidence], limit: int = 8) -> List[RAGCitation]:
        citations: List[RAGCitation] = []
        for patent in evidence_results[: self.top_patents]:
            for evidence in patent.evidence_chunks[: self.top_chunks_per_patent]:
                chunk = evidence.chunk
                citations.append(
                    RAGCitation(
                        patent_id=patent.patent_id,
                        chunk_id=chunk.chunk_id,
                        section=chunk.section,
                        quote=chunk.text[:260],
                        reason=evidence.retrieval_reason or "Retrieved as relevant evidence.",
                    )
                )
                if len(citations) >= limit:
                    return citations
        return citations
