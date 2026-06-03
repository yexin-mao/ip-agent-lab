from __future__ import annotations

from typing import List

from backend.retrieval.hybrid import tokenize
from backend.schemas.models import (
    DisclosureAnalysis,
    GroupedPatentEvidence,
    PriorArtComparison,
    RiskLevel,
)


class PriorArtCompareAgent:
    def run(
        self,
        disclosure: DisclosureAnalysis,
        search_results: List[GroupedPatentEvidence],
    ) -> List[PriorArtComparison]:
        comparisons = []
        invention_terms = set(tokenize(" ".join([
            disclosure.title,
            disclosure.solution,
            " ".join(disclosure.innovation_points),
            " ".join(disclosure.effects),
        ])))

        for result in search_results:
            evidence_texts = [chunk_result.chunk.text for chunk_result in result.evidence_chunks]
            doc_text = " ".join([result.title] + evidence_texts)
            doc_terms = set(tokenize(doc_text))
            overlap_terms = sorted(invention_terms.intersection(doc_terms))
            overlap_ratio = len(overlap_terms) / max(len(invention_terms), 1)
            relevance_score = round(min((0.55 * result.score) + (0.45 * overlap_ratio), 1.0), 4)

            risk = self._risk_level(relevance_score, overlap_terms)
            overlaps = self._build_overlaps(disclosure, doc_text, overlap_terms)
            differences = self._build_differences(disclosure, doc_text)
            evidence = self._extract_evidence(doc_text, overlap_terms)

            comparisons.append(
                PriorArtComparison(
                    patent_id=result.patent_id,
                    title=result.title,
                    relevance_score=relevance_score,
                    risk_level=risk,
                    overlaps=overlaps,
                    differences=differences,
                    evidence=evidence,
                    recommendation=self._recommendation(risk),
                    add_to_report=risk in {RiskLevel.high, RiskLevel.medium},
                )
            )

        return sorted(comparisons, key=lambda item: item.relevance_score, reverse=True)

    def _risk_level(self, score: float, overlap_terms: List[str]) -> RiskLevel:
        if len(overlap_terms) < 3:
            return RiskLevel.insufficient
        if score >= 0.45:
            return RiskLevel.high
        if score >= 0.25:
            return RiskLevel.medium
        return RiskLevel.low

    def _build_overlaps(self, disclosure: DisclosureAnalysis, doc_text: str, overlap_terms: List[str]) -> List[str]:
        overlaps = []
        lower_doc = doc_text.lower()
        for point in disclosure.innovation_points[:5]:
            point_terms = [term for term in tokenize(point) if term in lower_doc]
            if point_terms:
                overlaps.append(f"Innovation point overlaps on: {', '.join(point_terms[:6])}")
        if not overlaps and overlap_terms:
            overlaps.append(f"Shared technical vocabulary: {', '.join(overlap_terms[:8])}")
        return overlaps[:5]

    def _build_differences(self, disclosure: DisclosureAnalysis, doc_text: str) -> List[str]:
        differences = []
        lower_doc = doc_text.lower()
        for point in disclosure.innovation_points[:5]:
            point_tokens = tokenize(point)
            missing = [term for term in point_tokens if term not in lower_doc and len(term) > 3]
            if missing:
                differences.append(f"Potential differentiator in disclosure: {', '.join(missing[:5])}")
        return differences[:5] or ["No clear differentiator identified from available abstract/claim text."]

    def _extract_evidence(self, doc_text: str, overlap_terms: List[str]) -> List[str]:
        sentences = [part.strip() for part in doc_text.replace("\n", " ").split(".") if part.strip()]
        evidence = []
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(term in lower_sentence for term in overlap_terms[:10]):
                evidence.append(sentence[:280])
            if len(evidence) >= 3:
                break
        return evidence or sentences[:1]

    def _recommendation(self, risk: RiskLevel) -> str:
        if risk == RiskLevel.high:
            return "Prioritize manual review; compare independent claims against each innovation point."
        if risk == RiskLevel.medium:
            return "Review claim scope and publication date; may affect part of the novelty argument."
        if risk == RiskLevel.low:
            return "Keep as background prior art; limited direct novelty impact based on current evidence."
        return "Evidence is insufficient; retrieve full specification or claims before making a decision."
