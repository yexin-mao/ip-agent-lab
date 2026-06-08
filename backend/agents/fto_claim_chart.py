from __future__ import annotations

from typing import Iterable, List

from backend.retrieval.hybrid import tokenize
from backend.schemas.models import (
    CoverageLevel,
    DisclosureAnalysis,
    EvidenceChunkResult,
    FTOClaimChartRow,
    GroupedPatentEvidence,
    RiskLevel,
)


class FTOClaimChartAgent:
    def __init__(self, max_patents: int = 3):
        self.max_patents = max_patents

    def run(
        self,
        disclosure: DisclosureAnalysis,
        evidence_results: List[GroupedPatentEvidence],
    ) -> List[FTOClaimChartRow]:
        elements = self._technical_elements(disclosure)
        if not elements:
            return [
                FTOClaimChartRow(
                    element_id="E-0",
                    technical_element="No technical element was extracted.",
                    reasoning="Cannot build an FTO claim chart without disclosure elements.",
                )
            ]

        rows: List[FTOClaimChartRow] = []
        for index, element in enumerate(elements, start=1):
            candidates = self._rank_claim_candidates(f"E-{index}", element, evidence_results[: self.max_patents])
            rows.extend(candidates or [self._missing_row(f"E-{index}", element)])
        return rows

    def _rank_claim_candidates(
        self,
        element_id: str,
        element: str,
        evidence_results: List[GroupedPatentEvidence],
    ) -> List[FTOClaimChartRow]:
        rows: List[FTOClaimChartRow] = []
        element_terms = self._content_terms(element)
        if not element_terms:
            return rows

        for patent in evidence_results:
            claim_chunks = [chunk for chunk in patent.evidence_chunks if chunk.chunk.section == "claim"]
            chunks = claim_chunks or patent.evidence_chunks
            best = None
            for evidence in chunks:
                row = self._map_element_to_claim(element_id, element, element_terms, patent, evidence)
                if row.mapping == CoverageLevel.none:
                    continue
                if best is None or row.score > best.score:
                    best = row
            if best:
                rows.append(best)

        return sorted(rows, key=lambda row: row.score, reverse=True)

    def _map_element_to_claim(
        self,
        element_id: str,
        element: str,
        element_terms: List[str],
        patent: GroupedPatentEvidence,
        evidence: EvidenceChunkResult,
    ) -> FTOClaimChartRow:
        claim_terms = set(self._content_terms(evidence.chunk.text))
        matched_terms = sorted(term for term in element_terms if term in claim_terms)
        ratio = len(matched_terms) / max(len(element_terms), 1)
        score = round(min((ratio * 0.85) + (min(max(evidence.score, 0.0), 1.0) * 0.15), 1.0), 4)
        mapping = self._mapping_level(ratio, matched_terms)

        return FTOClaimChartRow(
            element_id=element_id,
            technical_element=element,
            patent_id=patent.patent_id,
            title=patent.title,
            claim_chunk_id=evidence.chunk.chunk_id,
            claim_text=evidence.chunk.text,
            mapping=mapping,
            risk_level=self._risk_level(mapping),
            reasoning=self._reasoning(mapping, matched_terms),
            matched_terms=matched_terms[:12],
            score=score,
        )

    def _technical_elements(self, disclosure: DisclosureAnalysis) -> List[str]:
        candidates = disclosure.innovation_points or []
        if disclosure.solution:
            candidates = [disclosure.solution] + candidates
        return self._dedupe([item.strip() for item in candidates if item.strip()])[:8]

    def _mapping_level(self, ratio: float, matched_terms: List[str]) -> CoverageLevel:
        if len(matched_terms) < 2:
            return CoverageLevel.none
        if ratio >= 0.5 or len(matched_terms) >= 7:
            return CoverageLevel.full
        if ratio >= 0.18:
            return CoverageLevel.partial
        return CoverageLevel.none

    def _risk_level(self, mapping: CoverageLevel) -> RiskLevel:
        if mapping == CoverageLevel.full:
            return RiskLevel.high
        if mapping == CoverageLevel.partial:
            return RiskLevel.medium
        if mapping == CoverageLevel.none:
            return RiskLevel.low
        return RiskLevel.insufficient

    def _reasoning(self, mapping: CoverageLevel, matched_terms: List[str]) -> str:
        if mapping == CoverageLevel.full:
            return f"Claim language appears to map strongly to this element through terms: {', '.join(matched_terms[:8])}."
        if mapping == CoverageLevel.partial:
            return f"Claim language partially maps to this element through terms: {', '.join(matched_terms[:8])}."
        if mapping == CoverageLevel.none:
            return "Claim language does not materially map to this element based on available chunks."
        return "Claim evidence is insufficient for this element."

    def _missing_row(self, element_id: str, element: str) -> FTOClaimChartRow:
        return FTOClaimChartRow(
            element_id=element_id,
            technical_element=element,
            mapping=CoverageLevel.insufficient,
            risk_level=RiskLevel.insufficient,
            reasoning="No retrieved claim chunk had enough overlap with this technical element.",
        )

    def _content_terms(self, text: str) -> List[str]:
        stopwords = {
            "the",
            "and",
            "with",
            "for",
            "that",
            "this",
            "from",
            "into",
            "using",
            "based",
            "method",
            "system",
            "device",
            "comprising",
            "including",
            "configured",
            "一种",
            "通过",
            "用于",
            "包括",
            "实现",
        }
        terms = [term for term in tokenize(text) if len(term) > 1 and term not in stopwords]
        return self._dedupe(terms)

    def _dedupe(self, items: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
