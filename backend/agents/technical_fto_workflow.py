from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from backend.agents.disclosure_parser import DisclosureParserAgent
from backend.agents.fto_claim_chart import FTOClaimChartAgent
from backend.agents.fto_rag_generation import FTORAGGenerationAgent
from backend.agents.fto_report_agent import FTOReportAgent
from backend.agents.keyword_expander import KeywordExpansionAgent
from backend.agents.patent_search import PatentSearchAgent
from backend.retrieval.chunker import PatentChunker
from backend.retrieval.hybrid_rag import HybridRAGRetriever
from backend.retrieval.llamaindex_retriever import LlamaIndexPatentRetriever
from backend.retrieval.local_corpus import load_patent_corpus
from backend.schemas.models import GroupedPatentEvidence, TechnicalFTOResult, TaskStatus


class TechnicalFTOWorkflow:
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
        self.claim_chart = FTOClaimChartAgent()
        self.rag_generation = FTORAGGenerationAgent()
        self.report_agent = FTOReportAgent()

    def run(self, implementation_text: str, task_id: str | None = None, limit: int = 8) -> TechnicalFTOResult:
        task_id = task_id or f"technical-fto-{uuid4().hex[:10]}"
        implementation = self.parser.run(implementation_text)
        keywords = self.keyword_expander.run(implementation)
        evidence_results = self.search_agent.run(keywords, limit=limit)
        claim_evidence_results = self._prefer_claim_evidence(evidence_results)
        fto_claim_chart = self.claim_chart.run(implementation, claim_evidence_results)
        generated_analysis = self.rag_generation.run(implementation, claim_evidence_results, fto_claim_chart)
        report = self.report_agent.run(
            task_id,
            implementation,
            keywords,
            claim_evidence_results,
            fto_claim_chart,
            generated_analysis,
        )

        return TechnicalFTOResult(
            task_id=task_id,
            status=TaskStatus.report_generated,
            created_at=datetime.now(),
            implementation=implementation,
            keywords=keywords,
            claim_evidence_results=claim_evidence_results,
            fto_claim_chart=fto_claim_chart,
            generated_analysis=generated_analysis,
            report_markdown=report,
        )

    def _prefer_claim_evidence(self, evidence_results: list[GroupedPatentEvidence]) -> list[GroupedPatentEvidence]:
        preferred = []
        for patent in evidence_results:
            claim_chunks = [item for item in patent.evidence_chunks if item.chunk.section == "claim"]
            if not claim_chunks:
                preferred.append(patent)
                continue

            preferred.append(
                patent.model_copy(
                    update={
                        "evidence_chunks": claim_chunks,
                        "score": round(sum(item.score for item in claim_chunks), 5),
                    }
                )
            )
        return preferred
