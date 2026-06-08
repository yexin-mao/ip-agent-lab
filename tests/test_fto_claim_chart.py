from __future__ import annotations

from pathlib import Path

from backend.agents.disclosure_parser import DisclosureParserAgent
from backend.agents.fto_claim_chart import FTOClaimChartAgent
from backend.agents.keyword_expander import KeywordExpansionAgent
from backend.agents.orchestrator import NoveltyOrchestrator
from backend.retrieval.llamaindex_retriever import LlamaIndexPatentRetriever
from backend.retrieval.local_corpus import load_patent_corpus
from backend.schemas.models import CoverageLevel


ROOT = Path(__file__).resolve().parents[1]


def test_fto_claim_chart_maps_elements_to_claim_chunks(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")
    disclosure = DisclosureParserAgent().run(
        (ROOT / "data" / "sample_disclosures" / "wireless_handover.txt").read_text(encoding="utf-8")
    )
    keywords = KeywordExpansionAgent().run(disclosure)
    documents = load_patent_corpus(ROOT / "data" / "sample_patents.json")
    evidence_results = LlamaIndexPatentRetriever(documents).search(keywords, top_k_patents=3)

    chart = FTOClaimChartAgent().run(disclosure, evidence_results)

    assert chart
    assert any(row.claim_chunk_id.endswith("claim_1") for row in chart if row.claim_chunk_id)
    assert any(row.mapping in {CoverageLevel.full, CoverageLevel.partial} for row in chart)
    assert any("claim" in row.claim_chunk_id for row in chart if row.claim_chunk_id)


def test_orchestrator_includes_fto_claim_chart_in_result_and_report(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_RETRIEVER", "llamaindex")
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")
    orchestrator = NoveltyOrchestrator(ROOT / "data" / "sample_patents.json")
    disclosure_text = (ROOT / "data" / "sample_disclosures" / "wireless_handover.txt").read_text(encoding="utf-8")

    result = orchestrator.run(disclosure_text, limit=3)

    assert result.fto_claim_chart
    assert "## FTO Claim Chart" in (result.report_markdown or "")
    assert result.fto_claim_chart[0].element_id.startswith("E-")
