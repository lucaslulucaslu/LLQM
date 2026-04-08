from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Verdict = Literal["supported", "likely_false", "unverified"]
ClaimCategory = Literal["timeline", "causal", "rumor"]


class Evidence(BaseModel):
    source_id: str
    url: str
    title: str
    excerpt: str
    published_at: str | None = None
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)


class Claim(BaseModel):
    text: str
    category: ClaimCategory = "timeline"
    evidence: list[Evidence] = Field(default_factory=list)
    support_count: int = 0
    contradiction_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TimelineEvent(BaseModel):
    timestamp: str | None = None
    summary: str
    source_urls: list[str] = Field(default_factory=list)


class RumorOrigin(BaseModel):
    first_seen_at: str | None = None
    first_seen_url: str | None = None
    first_seen_source: str | None = None
    provenance_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class InvestigationResult(BaseModel):
    query: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = ""
    timeline: list[TimelineEvent] = Field(default_factory=list)
    key_claims: list[Claim] = Field(default_factory=list)
    rumor_origin: RumorOrigin = Field(default_factory=RumorOrigin)
    uncertainty_notes: list[str] = Field(default_factory=list)


class RetrievedDocument(BaseModel):
    source_id: str
    url: str
    title: str
    content: str
    published_at: str | None = None
