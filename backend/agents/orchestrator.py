from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from backend.agents.disclosure_parser import DisclosureParserAgent
from backend.agents.keyword_expander import KeywordExpansionAgent
from backend.agents.patent_search import PatentSearchAgent
from backend.agents.prior_art_compare import PriorArtCompareAgent
from backend.agents.report_agent import ReportAgent
from backend.retrieval.hybrid import LocalHybridRetriever
from backend.retrieval.local_corpus import load_patent_corpus
from backend.schemas.models import NoveltyTaskResult, TaskStatus


class NoveltyOrchestrator:
    def __init__(self, corpus_path: str | Path):
        documents = load_patent_corpus(corpus_path)
        retriever = LocalHybridRetriever(documents)
        self.parser = DisclosureParserAgent()
        self.keyword_expander = KeywordExpansionAgent()
        self.search_agent = PatentSearchAgent(retriever)
        self.compare_agent = PriorArtCompareAgent()
        self.report_agent = ReportAgent()

    def run(self, disclosure_text: str, task_id: str | None = None, limit: int = 8) -> NoveltyTaskResult:
        task_id = task_id or f"novelty-{uuid4().hex[:10]}"
        disclosure = self.parser.run(disclosure_text)
        keywords = self.keyword_expander.run(disclosure)
        search_results = self.search_agent.run(keywords, limit=limit)
        comparisons = self.compare_agent.run(disclosure, search_results)
        report = self.report_agent.run(task_id, disclosure, keywords, search_results, comparisons)

        return NoveltyTaskResult(
            task_id=task_id,
            status=TaskStatus.report_generated,
            created_at=datetime.now(),
            disclosure=disclosure,
            keywords=keywords,
            search_results=search_results,
            comparisons=comparisons,
            report_markdown=report,
        )
