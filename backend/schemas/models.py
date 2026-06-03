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


class KeywordSet(BaseModel):
    core_terms: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    english_terms: List[str] = Field(default_factory=list)
    query_groups: List[str] = Field(default_factory=list)
    classification_hints: List[str] = Field(default_factory=list)


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


class SearchResult(BaseModel):
    document: PatentDocument
    score: float
    matched_terms: List[str] = Field(default_factory=list)
    retrieval_reason: str = ""


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


class NoveltyTaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    disclosure: DisclosureAnalysis
    keywords: KeywordSet
    search_results: List[SearchResult]
    comparisons: List[PriorArtComparison]
    report_markdown: Optional[str] = None
