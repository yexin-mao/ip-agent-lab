from __future__ import annotations

from typing import List

from backend.llm.client import LLMClient
from backend.rag.context_builder import EvidenceContextBuilder
from backend.schemas.models import (
    DisclosureAnalysis,
    GroupedPatentEvidence,
    NoveltyMatrixRow,
    RAGCitation,
    RAGGeneratedAnalysis,
)


class NoveltyRAGGenerationAgent:
    def __init__(self, llm_client: LLMClient | None = None, context_builder: EvidenceContextBuilder | None = None):
        self.llm_client = llm_client or LLMClient()
        self.context_builder = context_builder or EvidenceContextBuilder()

    def run(
        self,
        disclosure: DisclosureAnalysis,
        evidence_results: List[GroupedPatentEvidence],
        novelty_matrix: List[NoveltyMatrixRow],
    ) -> RAGGeneratedAnalysis:
        context = self.context_builder.build(evidence_results)
        llm_result = self._run_llm(disclosure, context, novelty_matrix)
        if llm_result:
            return llm_result
        return self._fallback(disclosure, evidence_results, novelty_matrix)

    def _run_llm(
        self,
        disclosure: DisclosureAnalysis,
        context: str,
        novelty_matrix: List[NoveltyMatrixRow],
    ) -> RAGGeneratedAnalysis | None:
        system_prompt = (
            "You are a patent prior-art analyst. Generate grounded novelty-search analysis "
            "only from the provided evidence. Do not provide legal advice. Return only valid JSON."
        )
        matrix_brief = [
            {
                "innovation_point_id": row.innovation_point_id,
                "patent_id": row.patent_id,
                "coverage": row.coverage.value,
                "evidence_chunk_id": row.evidence_chunk_id,
                "matched_terms": row.matched_terms,
            }
            for row in novelty_matrix[:12]
        ]
        user_prompt = f"""
Task: novelty search generation. Do not analyze infringement or FTO.

Disclosure:
- Title: {disclosure.title}
- Technical field: {disclosure.technical_field}
- Problem: {disclosure.problem}
- Solution: {disclosure.solution}
- Innovation points: {disclosure.innovation_points}

Innovation point evidence matrix:
{matrix_brief}

Retrieved evidence context:
{context}

Return this exact JSON shape:
{{
  "executive_summary": "2-4 sentence novelty-search summary grounded in evidence",
  "evidence_based_findings": ["finding with patent_id/chunk_id citation", "finding 2"],
  "risk_summary": "novelty risk only; no infringement conclusions",
  "recommended_next_steps": ["next search or review step"],
  "citations": [
    {{"patent_id": "US...", "chunk_id": "US...::claim_1", "section": "claim", "quote": "short quote", "reason": "why cited"}}
  ],
  "warnings": ["limitations"]
}}
"""
        raw = self.llm_client.chat_json(system_prompt, user_prompt, temperature=0.1)
        if not raw:
            return None

        try:
            return RAGGeneratedAnalysis(
                task_type="novelty_search",
                generation_mode="llm",
                executive_summary=str(raw.get("executive_summary") or ""),
                evidence_based_findings=_str_list(raw.get("evidence_based_findings")),
                risk_summary=str(raw.get("risk_summary") or ""),
                recommended_next_steps=_str_list(raw.get("recommended_next_steps")),
                citations=_citations(raw.get("citations")),
                warnings=_str_list(raw.get("warnings")),
            )
        except Exception:
            return None

    def _fallback(
        self,
        disclosure: DisclosureAnalysis,
        evidence_results: List[GroupedPatentEvidence],
        novelty_matrix: List[NoveltyMatrixRow],
    ) -> RAGGeneratedAnalysis:
        high_or_partial = [
            row for row in novelty_matrix
            if row.patent_id and row.coverage.value in {"full", "partial"}
        ]
        top_patents = ", ".join(item.patent_id for item in evidence_results[:3]) or "no strong prior art"
        findings = [
            f"{row.innovation_point_id} is {row.coverage.value} covered by {row.patent_id} via {row.evidence_chunk_id}."
            for row in high_or_partial[:6]
        ]
        if not findings:
            findings.append("No retrieved evidence chunk materially covered the extracted innovation points.")

        return RAGGeneratedAnalysis(
            task_type="novelty_search",
            generation_mode="fallback",
            executive_summary=(
                f"The search retrieved {len(evidence_results)} prior-art candidates for '{disclosure.title}'. "
                f"The strongest candidates are {top_patents}."
            ),
            evidence_based_findings=findings,
            risk_summary=(
                "Novelty risk is driven by innovation-point coverage in the evidence matrix; "
                "full or partial coverage should be manually reviewed."
            ),
            recommended_next_steps=[
                "Review full specifications and independent claims for the highest-covered patents.",
                "Expand the corpus and rerun retrieval before making a filing decision.",
            ],
            citations=self.context_builder.citations(evidence_results),
            warnings=["Fallback generation was used because no LLM response was available."],
        )


def _str_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _citations(value) -> List[RAGCitation]:
    citations = []
    if not isinstance(value, list):
        return citations
    for item in value:
        if not isinstance(item, dict):
            continue
        citations.append(
            RAGCitation(
                patent_id=str(item.get("patent_id") or ""),
                chunk_id=str(item.get("chunk_id") or ""),
                section=str(item.get("section") or ""),
                quote=str(item.get("quote") or ""),
                reason=str(item.get("reason") or ""),
            )
        )
    return citations[:8]
