from __future__ import annotations

from pathlib import Path

from backend.agents.disclosure_parser import DisclosureParserAgent
from backend.agents.keyword_expander import KeywordExpansionAgent
from backend.retrieval.llamaindex_retriever import LlamaIndexPatentRetriever
from backend.retrieval.local_corpus import load_patent_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_keyword_expander_builds_source_weighted_retrieval_queries() -> None:
    disclosure = DisclosureParserAgent().run(
        (ROOT / "data" / "sample_disclosures" / "wireless_handover.txt").read_text(encoding="utf-8")
    )

    keywords = KeywordExpansionAgent().run(disclosure)
    sources = {query.source for query in keywords.retrieval_queries}

    assert keywords.retrieval_queries
    assert "title" in sources
    assert "solution" in sources
    assert "innovation_point_1" in sources
    assert any(query.weight > 1.0 for query in keywords.retrieval_queries)


def test_llamaindex_retriever_returns_hybrid_source_metadata(monkeypatch) -> None:
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")
    documents = load_patent_corpus(ROOT / "data" / "sample_patents.json")
    disclosure = DisclosureParserAgent().run(
        (ROOT / "data" / "sample_disclosures" / "wireless_handover.txt").read_text(encoding="utf-8")
    )
    keywords = KeywordExpansionAgent().run(disclosure)

    results = LlamaIndexPatentRetriever(documents).search(keywords, top_k_patents=3)

    assert results
    assert results[0].patent_id == "US20240011234A1"
    first_chunk = results[0].evidence_chunks[0]
    assert first_chunk.query_sources
    assert first_chunk.retrieval_sources
    assert {"bm25", "vector"}.intersection(first_chunk.retrieval_sources)
