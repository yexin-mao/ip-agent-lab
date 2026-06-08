from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from backend.agents.disclosure_parser import DisclosureParserAgent
from backend.agents.evidence_judge import EvidenceJudgeAgent
from backend.agents.keyword_expander import KeywordExpansionAgent
from backend.agents.novelty_rag_generation import NoveltyRAGGenerationAgent
from backend.agents.novelty_report_agent import NoveltyReportAgent
from backend.agents.patent_search import PatentSearchAgent
from backend.retrieval.chunker import PatentChunker
from backend.retrieval.hybrid_rag import HybridRAGRetriever
from backend.retrieval.llamaindex_retriever import LlamaIndexPatentRetriever
from backend.retrieval.local_corpus import load_patent_corpus
from backend.schemas.models import NoveltyTaskResult, TaskStatus


class NoveltyWorkflow:
    def __init__(self, corpus_path: str | Path):
        documents = load_patent_corpus(corpus_path)
        retriever_name = os.getenv("IP_AGENT_RETRIEVER", "llamaindex").strip().lower()
        if retriever_name == "llamaindex":
            retriever = LlamaIndexPatentRetriever(documents)
        else:
            retriever = HybridRAGRetriever(PatentChunker().chunk_documents(documents))

        self.parser = DisclosureParserAgent()
        self.keyword_expander = KeywordExpansionAgent()
        self.search_agent = PatentSearchAgent(retriever)
        self.evidence_judge = EvidenceJudgeAgent()
        self.rag_generation = NoveltyRAGGenerationAgent()
        self.report_agent = NoveltyReportAgent()

    def run(self, disclosure_text: str, task_id: str | None = None, limit: int = 8) -> NoveltyTaskResult:
        task_id = task_id or f"novelty-{uuid4().hex[:10]}"
        disclosure = self.parser.run(disclosure_text)
        keywords = self.keyword_expander.run(disclosure)
        evidence_results = self.search_agent.run(keywords, limit=limit)
        novelty_matrix = self.evidence_judge.run(disclosure, evidence_results)
        generated_analysis = self.rag_generation.run(disclosure, evidence_results, novelty_matrix)
        report = self.report_agent.run(
            task_id,
            disclosure,
            keywords,
            evidence_results,
            novelty_matrix,
            generated_analysis,
        )

        return NoveltyTaskResult(
            task_id=task_id,
            status=TaskStatus.report_generated,
            created_at=datetime.now(),
            disclosure=disclosure,
            keywords=keywords,
            evidence_results=evidence_results,
            novelty_matrix=novelty_matrix,
            generated_analysis=generated_analysis,
            comparisons=[],
            report_markdown=report,
        )
