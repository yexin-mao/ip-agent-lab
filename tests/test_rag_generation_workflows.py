from __future__ import annotations

from pathlib import Path

from backend.agents.novelty_workflow import NoveltyWorkflow
from backend.agents.technical_fto_workflow import TechnicalFTOWorkflow
from backend.rag.context_builder import EvidenceContextBuilder


ROOT = Path(__file__).resolve().parents[1]


def sample_text() -> str:
    return (ROOT / "data" / "sample_disclosures" / "wireless_handover.txt").read_text(encoding="utf-8")


def test_novelty_workflow_generates_grounded_analysis_without_fto_chart(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_RETRIEVER", "llamaindex")
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")

    result = NoveltyWorkflow(ROOT / "data" / "sample_patents.json").run(sample_text(), limit=3)

    assert result.evidence_results
    assert result.novelty_matrix
    assert result.fto_claim_chart == []
    assert result.generated_analysis is not None
    assert result.generated_analysis.task_type == "novelty_search"
    assert result.generated_analysis.generation_mode == "fallback"
    assert "RAG-Generated Novelty Analysis" in (result.report_markdown or "")


def test_technical_fto_workflow_generates_claim_chart_and_fto_analysis(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_RETRIEVER", "llamaindex")
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")

    result = TechnicalFTOWorkflow(ROOT / "data" / "sample_patents.json").run(sample_text(), limit=3)

    assert result.claim_evidence_results
    assert result.fto_claim_chart
    assert result.generated_analysis is not None
    assert result.generated_analysis.task_type == "technical_fto"
    assert result.generated_analysis.generation_mode == "fallback"
    assert any(row.claim_chunk_id for row in result.fto_claim_chart)
    assert "RAG-Generated FTO Analysis" in (result.report_markdown or "")


def test_evidence_context_builder_outputs_chunk_citations(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_RETRIEVER", "llamaindex")
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")
    result = NoveltyWorkflow(ROOT / "data" / "sample_patents.json").run(sample_text(), limit=2)

    builder = EvidenceContextBuilder(top_patents=2, top_chunks_per_patent=1)
    context = builder.build(result.evidence_results)
    citations = builder.citations(result.evidence_results)

    assert "Patent:" in context
    assert "::" in context
    assert citations
    assert citations[0].chunk_id
