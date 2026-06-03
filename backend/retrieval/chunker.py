from __future__ import annotations

import re
from typing import Iterable, List

from backend.schemas.models import PatentChunk, PatentDocument


class PatentChunker:
    """Convert patent-level records into evidence chunks for RAG retrieval."""

    def chunk_documents(self, documents: Iterable[PatentDocument]) -> List[PatentChunk]:
        chunks: List[PatentChunk] = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc))
        return chunks

    def chunk_document(self, doc: PatentDocument) -> List[PatentChunk]:
        chunks = []
        title_abstract = "\n".join(part for part in [doc.title, doc.abstract] if part.strip())
        if title_abstract:
            chunks.append(self._build_chunk(doc, "title_abstract", "title_abstract_1", title_abstract))

        claim_parts = self._split_claims(doc.claims)
        for idx, claim_text in enumerate(claim_parts, start=1):
            chunks.append(self._build_chunk(doc, "claim", f"claim_{idx}", claim_text))

        return chunks

    def _split_claims(self, claims: str) -> List[str]:
        cleaned = (claims or "").strip()
        if not cleaned:
            return []

        parts = re.split(r"(?=\b\d+\.\s+|\bclaim\s+\d+[:.]?\s+)", cleaned, flags=re.IGNORECASE)
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) <= 1:
            return [cleaned]
        return parts

    def _build_chunk(self, doc: PatentDocument, section: str, local_id: str, text: str) -> PatentChunk:
        return PatentChunk(
            chunk_id=f"{doc.patent_id}::{local_id}",
            patent_id=doc.patent_id,
            title=doc.title,
            section=section,
            text=text.strip(),
            assignee=doc.assignee,
            publication_date=doc.publication_date,
            jurisdiction=doc.jurisdiction,
            cpc=doc.cpc,
            source_url=doc.url,
        )
