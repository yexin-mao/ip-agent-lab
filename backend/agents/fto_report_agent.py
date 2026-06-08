from __future__ import annotations

from datetime import datetime
from typing import List

from backend.schemas.models import (
    DisclosureAnalysis,
    FTOClaimChartRow,
    GroupedPatentEvidence,
    KeywordSet,
    RAGGeneratedAnalysis,
)


class FTOReportAgent:
    def run(
        self,
        task_id: str,
        implementation: DisclosureAnalysis,
        keywords: KeywordSet,
        claim_evidence_results: List[GroupedPatentEvidence],
        fto_claim_chart: List[FTOClaimChartRow],
        generated_analysis: RAGGeneratedAnalysis,
    ) -> str:
        lines = [
            "# Technical FTO Report",
            "",
            f"Task ID: `{task_id}`",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## AI Use Notice",
            "This AI-assisted FTO report is not a legal opinion. Manual claim review is required.",
            "",
            "## Implementation Summary",
            f"**Title:** {implementation.title}",
            f"**Technical Field:** {implementation.technical_field}",
            f"**Implementation / Product Features:** {implementation.solution}",
            "",
            "## Technical Elements",
        ]
        lines.extend([f"- {point}" for point in implementation.innovation_points] or ["- Not extracted"])
        lines.extend([
            "",
            "## Claim-Focused Search Strategy",
            f"**Core terms:** {', '.join(keywords.core_terms) or 'N/A'}",
            f"**Expanded terms:** {', '.join(keywords.synonyms) or 'N/A'}",
            "",
            "## Claim Evidence",
        ])

        for idx, result in enumerate(claim_evidence_results, start=1):
            lines.extend([
                f"### {idx}. {result.patent_id} - {result.title}",
                f"- Score: {result.score}",
                f"- Jurisdiction: {result.jurisdiction or 'N/A'}",
                f"- Date: {result.publication_date or 'N/A'}",
                "- Claim/evidence chunks:",
            ])
            for evidence in result.evidence_chunks:
                chunk = evidence.chunk
                lines.extend([
                    f"  - `{chunk.chunk_id}` | section: `{chunk.section}` | score: {evidence.score}",
                    f"    - evidence: {chunk.text}",
                ])
            lines.append("")

        lines.extend([
            "## FTO Claim Chart",
            "| Technical Element | Patent | Mapping | Risk | Claim Chunk | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for row in fto_claim_chart:
            patent_ref = f"{row.patent_id} - {row.title}" if row.patent_id else "N/A"
            lines.append(
                "| "
                f"{self._cell(row.element_id + ': ' + row.technical_element)} | "
                f"{self._cell(patent_ref)} | "
                f"{row.mapping.value} | "
                f"{row.risk_level.value} | "
                f"{self._cell(row.claim_chunk_id or 'N/A')} | "
                f"{self._cell(row.reasoning)} |"
            )

        lines.extend([
            "",
            "## RAG-Generated FTO Analysis",
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
