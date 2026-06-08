from __future__ import annotations

from pathlib import Path

from backend.agents.disclosure_parser import DisclosureParserAgent
from backend.agents.evidence_judge import EvidenceJudgeAgent
from backend.agents.keyword_expander import KeywordExpansionAgent
from backend.agents.orchestrator import NoveltyOrchestrator
from backend.retrieval.llamaindex_retriever import LlamaIndexPatentRetriever
from backend.retrieval.local_corpus import load_patent_corpus
from backend.schemas.models import CoverageLevel


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_judge_builds_innovation_point_matrix(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")
    disclosure = DisclosureParserAgent().run(
        (ROOT / "data" / "sample_disclosures" / "wireless_handover.txt").read_text(encoding="utf-8")
    )
    keywords = KeywordExpansionAgent().run(disclosure)
    documents = load_patent_corpus(ROOT / "data" / "sample_patents.json")
    evidence_results = LlamaIndexPatentRetriever(documents).search(keywords, top_k_patents=3)

    matrix = EvidenceJudgeAgent().run(disclosure, evidence_results)

    assert matrix
    assert {row.innovation_point_id for row in matrix}
    assert any(row.patent_id == "US20240011234A1" for row in matrix)
    assert any(row.coverage in {CoverageLevel.full, CoverageLevel.partial} for row in matrix)
    assert any(row.evidence_chunk_id for row in matrix)


def test_orchestrator_includes_novelty_matrix_in_result_and_report(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_RETRIEVER", "llamaindex")
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")
    orchestrator = NoveltyOrchestrator(ROOT / "data" / "sample_patents.json")
    disclosure_text = (ROOT / "data" / "sample_disclosures" / "wireless_handover.txt").read_text(encoding="utf-8")

    result = orchestrator.run(disclosure_text, limit=3)

    assert result.novelty_matrix
    assert "## Innovation Point Evidence Matrix" in (result.report_markdown or "")
    assert result.novelty_matrix[0].innovation_point_id.startswith("IP-")
