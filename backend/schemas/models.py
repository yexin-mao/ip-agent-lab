from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    insufficient = "insufficient_evidence"


class CoverageLevel(str, Enum):
    full = "full"
    partial = "partial"
    none = "none"
    insufficient = "insufficient_evidence"


class TaskStatus(str, Enum):
    created = "created"
    parsing = "parsing"
    keyword_expanding = "keyword_expanding"
    searching = "searching"
    comparing = "comparing"
    review_required = "review_required"
    report_generated = "report_generated"
    failed = "failed"


class DisclosureAnalysis(BaseModel):
    title: str
    technical_field: str
    problem: str
    solution: str
    innovation_points: List[str] = Field(default_factory=list)
    effects: List[str] = Field(default_factory=list)
    applications: List[str] = Field(default_factory=list)
    key_terms: List[str] = Field(default_factory=list)
    parser_mode: str = "rule"
    parse_quality: str = "medium"
    warnings: List[str] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    query_text: str
    source: str = "keywords"
    weight: float = 1.0


class KeywordSet(BaseModel):
    core_terms: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    english_terms: List[str] = Field(default_factory=list)
    query_groups: List[str] = Field(default_factory=list)
    classification_hints: List[str] = Field(default_factory=list)
    retrieval_queries: List[RetrievalQuery] = Field(default_factory=list)


class PatentDocument(BaseModel):
    patent_id: str
    title: str
    abstract: str
    claims: str = ""
    assignee: str = ""
    publication_date: str = ""
    jurisdiction: str = ""
    cpc: List[str] = Field(default_factory=list)
    url: str = ""


class PatentChunk(BaseModel):
    chunk_id: str
    patent_id: str
    title: str
    section: str
    text: str
    assignee: str = ""
    publication_date: str = ""
    jurisdiction: str = ""
    cpc: List[str] = Field(default_factory=list)
    source_url: str = ""


class SearchResult(BaseModel):
    document: PatentDocument
    score: float
    matched_terms: List[str] = Field(default_factory=list)
    retrieval_reason: str = ""


class EvidenceChunkResult(BaseModel):
    chunk: PatentChunk
    score: float
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    matched_terms: List[str] = Field(default_factory=list)
    retrieval_reason: str = ""
    query_sources: List[str] = Field(default_factory=list)
    retrieval_sources: List[str] = Field(default_factory=list)


class GroupedPatentEvidence(BaseModel):
    patent_id: str
    title: str
    assignee: str = ""
    publication_date: str = ""
    jurisdiction: str = ""
    cpc: List[str] = Field(default_factory=list)
    source_url: str = ""
    score: float
    evidence_chunks: List[EvidenceChunkResult] = Field(default_factory=list)


class PriorArtComparison(BaseModel):
    patent_id: str
    title: str
    relevance_score: float
    risk_level: RiskLevel
    overlaps: List[str] = Field(default_factory=list)
    differences: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    recommendation: str
    add_to_report: bool = True


class NoveltyMatrixRow(BaseModel):
    innovation_point_id: str
    innovation_point: str
    patent_id: str = ""
    title: str = ""
    coverage: CoverageLevel = CoverageLevel.insufficient
    risk_level: RiskLevel = RiskLevel.insufficient
    evidence_chunk_id: str = ""
    evidence_section: str = ""
    evidence_text: str = ""
    reasoning: str = ""
    matched_terms: List[str] = Field(default_factory=list)
    query_sources: List[str] = Field(default_factory=list)
    retrieval_sources: List[str] = Field(default_factory=list)
    score: float = 0.0


class FTOClaimChartRow(BaseModel):
    element_id: str
    technical_element: str
    patent_id: str = ""
    title: str = ""
    claim_chunk_id: str = ""
    claim_text: str = ""
    mapping: CoverageLevel = CoverageLevel.insufficient
    risk_level: RiskLevel = RiskLevel.insufficient
    reasoning: str = ""
    matched_terms: List[str] = Field(default_factory=list)
    score: float = 0.0


class RAGCitation(BaseModel):
    patent_id: str = ""
    chunk_id: str = ""
    section: str = ""
    quote: str = ""
    reason: str = ""


class RAGGeneratedAnalysis(BaseModel):
    task_type: str
    generation_mode: str = "fallback"
    executive_summary: str = ""
    evidence_based_findings: List[str] = Field(default_factory=list)
    risk_summary: str = ""
    recommended_next_steps: List[str] = Field(default_factory=list)
    citations: List[RAGCitation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DesignPatentImage(BaseModel):
    image_id: str
    design_id: str
    title: str
    product_type: str
    view: str
    image_path: str
    assignee: str = ""
    publication_date: str = ""
    jurisdiction: str = ""
    source_url: str = ""


class DesignSearchResult(BaseModel):
    image: DesignPatentImage
    similarity_score: float
    risk_level: RiskLevel
    visual_overlaps: List[str] = Field(default_factory=list)
    visual_differences: List[str] = Field(default_factory=list)
    reasoning: str = ""


class DesignFTOResult(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    product_type: str = ""
    query_image_name: str = ""
    search_results: List[DesignSearchResult] = Field(default_factory=list)
    report_markdown: Optional[str] = None


class NoveltyTaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    disclosure: DisclosureAnalysis
    keywords: KeywordSet
    search_results: List[SearchResult] = Field(default_factory=list)
    evidence_results: List[GroupedPatentEvidence] = Field(default_factory=list)
    novelty_matrix: List[NoveltyMatrixRow] = Field(default_factory=list)
    fto_claim_chart: List[FTOClaimChartRow] = Field(default_factory=list)
    generated_analysis: Optional[RAGGeneratedAnalysis] = None
    comparisons: List[PriorArtComparison]
    report_markdown: Optional[str] = None


class TechnicalFTOResult(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    implementation: DisclosureAnalysis
    keywords: KeywordSet
    claim_evidence_results: List[GroupedPatentEvidence] = Field(default_factory=list)
    fto_claim_chart: List[FTOClaimChartRow] = Field(default_factory=list)
    generated_analysis: Optional[RAGGeneratedAnalysis] = None
    report_markdown: Optional[str] = None
