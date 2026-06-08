from __future__ import annotations

from typing import List

from backend.llm.client import LLMClient
from backend.rag.context_builder import EvidenceContextBuilder
from backend.schemas.models import (
    DisclosureAnalysis,
    FTOClaimChartRow,
    GroupedPatentEvidence,
    RAGCitation,
    RAGGeneratedAnalysis,
)


class FTORAGGenerationAgent:
    def __init__(self, llm_client: LLMClient | None = None, context_builder: EvidenceContextBuilder | None = None):
        self.llm_client = llm_client or LLMClient()
        self.context_builder = context_builder or EvidenceContextBuilder(top_chunks_per_patent=3)

    def run(
        self,
        implementation: DisclosureAnalysis,
        claim_evidence_results: List[GroupedPatentEvidence],
        fto_claim_chart: List[FTOClaimChartRow],
    ) -> RAGGeneratedAnalysis:
        context = self.context_builder.build(claim_evidence_results)
        llm_result = self._run_llm(implementation, context, fto_claim_chart)
        if llm_result:
            return llm_result
        return self._fallback(implementation, claim_evidence_results, fto_claim_chart)

    def _run_llm(
        self,
        implementation: DisclosureAnalysis,
        context: str,
        fto_claim_chart: List[FTOClaimChartRow],
    ) -> RAGGeneratedAnalysis | None:
        system_prompt = (
            "You are a patent FTO analyst. Generate grounded claim-mapping analysis "
            "only from provided claim evidence. Do not analyze patentability or novelty. "
            "Do not provide legal advice. Return only valid JSON."
        )
        chart_brief = [
            {
                "element_id": row.element_id,
                "patent_id": row.patent_id,
                "mapping": row.mapping.value,
                "claim_chunk_id": row.claim_chunk_id,
                "matched_terms": row.matched_terms,
            }
            for row in fto_claim_chart[:14]
        ]
        user_prompt = f"""
Task: technical FTO generation. Do not analyze novelty.

Implementation / product features:
- Title: {implementation.title}
- Field: {implementation.technical_field}
- Implementation summary: {implementation.solution}
- Technical elements: {implementation.innovation_points}

Claim chart:
{chart_brief}

Claim evidence context:
{context}

Return this exact JSON shape:
{{
  "executive_summary": "2-4 sentence FTO summary grounded in claim evidence",
  "evidence_based_findings": ["claim mapping finding with patent_id/chunk_id citation"],
  "risk_summary": "FTO risk only; no validity or novelty conclusions",
  "recommended_next_steps": ["claim review or design-around step"],
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
                task_type="technical_fto",
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
        implementation: DisclosureAnalysis,
        claim_evidence_results: List[GroupedPatentEvidence],
        fto_claim_chart: List[FTOClaimChartRow],
    ) -> RAGGeneratedAnalysis:
        mapped = [
            row for row in fto_claim_chart
            if row.patent_id and row.mapping.value in {"full", "partial"}
        ]
        findings = [
            f"{row.element_id} is {row.mapping.value} mapped to {row.patent_id} claim evidence {row.claim_chunk_id}."
            for row in mapped[:6]
        ]
        if not findings:
            findings.append("No retrieved claim chunk materially mapped to the extracted implementation elements.")

        return RAGGeneratedAnalysis(
            task_type="technical_fto",
            generation_mode="fallback",
            executive_summary=(
                f"The FTO workflow reviewed {len(claim_evidence_results)} claim-evidence candidates "
                f"against implementation features for '{implementation.title}'."
            ),
            evidence_based_findings=findings,
            risk_summary=(
                "FTO risk is driven by claim-chart mappings; full or partial mappings require manual claim construction, "
                "jurisdiction, status, and expiration review."
            ),
            recommended_next_steps=[
                "Review independent claims for patents with full or partial element mappings.",
                "Check legal status, expiration, jurisdiction, and potential design-around options.",
            ],
            citations=self.context_builder.citations(claim_evidence_results),
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
