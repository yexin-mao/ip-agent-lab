from __future__ import annotations

from typing import Iterable, List

from backend.retrieval.hybrid import tokenize
from backend.schemas.models import (
    CoverageLevel,
    DisclosureAnalysis,
    EvidenceChunkResult,
    GroupedPatentEvidence,
    NoveltyMatrixRow,
    RiskLevel,
)


class EvidenceJudgeAgent:
    def __init__(self, max_rows_per_point: int = 3):
        self.max_rows_per_point = max_rows_per_point

    def run(
        self,
        disclosure: DisclosureAnalysis,
        evidence_results: List[GroupedPatentEvidence],
    ) -> List[NoveltyMatrixRow]:
        if not disclosure.innovation_points:
            return [
                NoveltyMatrixRow(
                    innovation_point_id="IP-0",
                    innovation_point="No innovation point was extracted.",
                    reasoning="Cannot build a novelty matrix until the disclosure parser extracts innovation points.",
                )
            ]

        rows: List[NoveltyMatrixRow] = []
        for index, point in enumerate(disclosure.innovation_points, start=1):
            point_id = f"IP-{index}"
            candidates = self._rank_candidates(point_id, point, evidence_results)
            rows.extend(candidates[: self.max_rows_per_point])
            if not candidates:
                rows.append(
                    NoveltyMatrixRow(
                        innovation_point_id=point_id,
                        innovation_point=point,
                        reasoning="No retrieved evidence chunk had enough lexical overlap with this innovation point.",
                    )
                )

        return rows

    def _rank_candidates(
        self,
        point_id: str,
        point: str,
        evidence_results: List[GroupedPatentEvidence],
    ) -> List[NoveltyMatrixRow]:
        rows: List[NoveltyMatrixRow] = []
        point_terms = self._content_terms(point)
        if not point_terms:
            return []

        for patent in evidence_results:
            best_row = None
            for evidence in patent.evidence_chunks:
                row = self._judge_chunk(point_id, point, point_terms, patent, evidence)
                if row.coverage == CoverageLevel.none:
                    continue
                if best_row is None or row.score > best_row.score:
                    best_row = row
            if best_row:
                rows.append(best_row)

        return sorted(rows, key=lambda item: item.score, reverse=True)

    def _judge_chunk(
        self,
        point_id: str,
        point: str,
        point_terms: List[str],
        patent: GroupedPatentEvidence,
        evidence: EvidenceChunkResult,
    ) -> NoveltyMatrixRow:
        text_terms = set(self._content_terms(" ".join([patent.title, evidence.chunk.text])))
        matched_terms = sorted(term for term in point_terms if term in text_terms)
        coverage_ratio = len(matched_terms) / max(len(point_terms), 1)
        retrieval_boost = min(max(evidence.score, 0.0), 1.0) * 0.2
        score = round(min((0.8 * coverage_ratio) + retrieval_boost, 1.0), 4)
        coverage = self._coverage_level(coverage_ratio, matched_terms)
        risk = self._risk_level(coverage)

        return NoveltyMatrixRow(
            innovation_point_id=point_id,
            innovation_point=point,
            patent_id=patent.patent_id,
            title=patent.title,
            coverage=coverage,
            risk_level=risk,
            evidence_chunk_id=evidence.chunk.chunk_id,
            evidence_section=evidence.chunk.section,
            evidence_text=evidence.chunk.text,
            reasoning=self._reasoning(coverage, matched_terms),
            matched_terms=matched_terms[:12],
            query_sources=evidence.query_sources,
            retrieval_sources=evidence.retrieval_sources,
            score=score,
        )

    def _coverage_level(self, coverage_ratio: float, matched_terms: List[str]) -> CoverageLevel:
        if len(matched_terms) < 2:
            return CoverageLevel.none
        if coverage_ratio >= 0.55 or len(matched_terms) >= 8:
            return CoverageLevel.full
        if coverage_ratio >= 0.2:
            return CoverageLevel.partial
        return CoverageLevel.none

    def _risk_level(self, coverage: CoverageLevel) -> RiskLevel:
        if coverage == CoverageLevel.full:
            return RiskLevel.high
        if coverage == CoverageLevel.partial:
            return RiskLevel.medium
        if coverage == CoverageLevel.none:
            return RiskLevel.low
        return RiskLevel.insufficient

    def _reasoning(self, coverage: CoverageLevel, matched_terms: List[str]) -> str:
        if coverage == CoverageLevel.full:
            return f"Evidence strongly covers this innovation point through terms: {', '.join(matched_terms[:8])}."
        if coverage == CoverageLevel.partial:
            return f"Evidence partially covers this innovation point through terms: {', '.join(matched_terms[:8])}."
        if coverage == CoverageLevel.none:
            return "Evidence does not materially cover this innovation point based on current retrieved chunks."
        return "Evidence is insufficient for judging this innovation point."

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
            "一种",
            "通过",
            "用于",
            "包括",
            "实现",
        }
        terms = [term for term in tokenize(text) if len(term) > 1 and term not in stopwords]
        return self._dedupe(terms)

    def _dedupe(self, terms: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            result.append(term)
        return result
