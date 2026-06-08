from __future__ import annotations

from datetime import datetime
from typing import List

from backend.schemas.models import (
    DisclosureAnalysis,
    GroupedPatentEvidence,
    KeywordSet,
    NoveltyMatrixRow,
    RAGGeneratedAnalysis,
)


class NoveltyReportAgent:
    def run(
        self,
        task_id: str,
        disclosure: DisclosureAnalysis,
        keywords: KeywordSet,
        evidence_results: List[GroupedPatentEvidence],
        novelty_matrix: List[NoveltyMatrixRow],
        generated_analysis: RAGGeneratedAnalysis,
    ) -> str:
        lines = [
            "# Patent Novelty Search Report",
            "",
            f"Task ID: `{task_id}`",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## AI Use Notice",
            "This AI-assisted novelty report is not a legal opinion. Manual professional review is required.",
            "",
            "## Invention Summary",
            f"**Title:** {disclosure.title}",
            f"**Technical Field:** {disclosure.technical_field}",
            f"**Problem:** {disclosure.problem}",
            f"**Solution:** {disclosure.solution}",
            "",
            "## Innovation Points",
        ]
        lines.extend([f"- {point}" for point in disclosure.innovation_points] or ["- Not extracted"])
        lines.extend([
            "",
            "## Search Strategy",
            f"**Core terms:** {', '.join(keywords.core_terms) or 'N/A'}",
            f"**Expanded terms:** {', '.join(keywords.synonyms) or 'N/A'}",
            f"**Classification hints:** {', '.join(keywords.classification_hints) or 'N/A'}",
            "",
            "## Retrieved Prior Art",
        ])

        for idx, result in enumerate(evidence_results, start=1):
            lines.extend([
                f"### {idx}. {result.patent_id} - {result.title}",
                f"- Score: {result.score}",
                f"- Jurisdiction: {result.jurisdiction or 'N/A'}",
                f"- Date: {result.publication_date or 'N/A'}",
                "- Evidence chunks:",
            ])
            for evidence in result.evidence_chunks:
                chunk = evidence.chunk
                lines.extend([
                    f"  - `{chunk.chunk_id}` | section: `{chunk.section}` | score: {evidence.score}",
                    f"    - evidence: {chunk.text}",
                ])
            lines.append("")

        lines.extend([
            "## Innovation Point Evidence Matrix",
            "| Innovation Point | Patent | Coverage | Risk | Evidence | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for row in novelty_matrix:
            patent_ref = f"{row.patent_id} - {row.title}" if row.patent_id else "N/A"
            lines.append(
                "| "
                f"{self._cell(row.innovation_point_id + ': ' + row.innovation_point)} | "
                f"{self._cell(patent_ref)} | "
                f"{row.coverage.value} | "
                f"{row.risk_level.value} | "
                f"{self._cell(row.evidence_chunk_id or 'N/A')} | "
                f"{self._cell(row.reasoning)} |"
            )

        lines.extend([
            "",
            "## RAG-Generated Novelty Analysis",
            f"**Generation mode:** {generated_analysis.generation_mode}",
            "",
            "### Executive Summary",
            generated_analysis.executive_summary or "N/A",
            "",
            "### Evidence-Based Findings",
        ])
        lines.extend([f"- {item}" for item in generated_analysis.evidence_based_findings] or ["- N/A"])
        lines.extend([
            "",
            "### Risk Summary",
            generated_analysis.risk_summary or "N/A",
            "",
            "### Recommended Next Steps",
        ])
        lines.extend([f"- {item}" for item in generated_analysis.recommended_next_steps] or ["- N/A"])
        lines.extend([
            "",
            "### Citations",
        ])
        lines.extend([
            f"- `{citation.chunk_id}` ({citation.patent_id}, {citation.section}): {citation.quote}"
            for citation in generated_analysis.citations
        ] or ["- N/A"])
        return "\n".join(lines)

    def _cell(self, value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").strip()
