from __future__ import annotations

from collections import defaultdict

from llqm.schemas.models import Claim, Evidence, RetrievedDocument, RumorOrigin, Verdict
from llqm.utils.date_extractor import date_sort_key
from llqm.utils.source_registry import resolve_source


def _split_candidate_claims(text: str) -> list[str]:
    parts = [segment.strip() for segment in text.replace("\n", " ").split(".")]
    return [part for part in parts if len(part.split()) >= 6][:3]


def extract_claims(documents: list[RetrievedDocument]) -> list[Claim]:
    """Extract coarse claims from docs while preserving evidence links."""
    claims_by_text: dict[str, Claim] = {}

    for doc in documents:
        source_id, trust_score = resolve_source(doc.url, doc.source_id)
        snippets = _split_candidate_claims(doc.content) or [doc.title]
        for snippet in snippets:
            claim_text = snippet.strip()
            if not claim_text:
                continue

            evidence = Evidence(
                source_id=source_id,
                url=doc.url,
                title=doc.title,
                excerpt=snippet,
                published_at=doc.published_at,
                trust_score=trust_score,
            )
            current = claims_by_text.get(claim_text)
            if current is None:
                claims_by_text[claim_text] = Claim(
                    text=claim_text,
                    category="rumor" if "rumor" in claim_text.lower() else "timeline",
                    evidence=[evidence],
                )
            else:
                current.evidence.append(evidence)

    return list(claims_by_text.values())


def score_claims(claims: list[Claim]) -> list[Claim]:
    """Score support strength and basic contradictions."""
    normalized_to_claims: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        normalized_to_claims[claim.text.lower()].append(claim)

    for claim in claims:
        evidence_sources = {item.source_id for item in claim.evidence}
        claim.support_count = len(evidence_sources)

        negation_hits = sum(
            1
            for item in claim.evidence
            if any(
                word in item.excerpt.lower()
                for word in ("false", "debunk", "misleading", "not true")
            )
        )
        claim.contradiction_count = negation_hits

        avg_trust = 0.0
        if claim.evidence:
            avg_trust = sum(item.trust_score for item in claim.evidence) / len(
                claim.evidence
            )

        support_component = min(1.0, claim.support_count / 3.0)
        contradiction_penalty = min(0.7, claim.contradiction_count * 0.25)
        claim.confidence = max(
            0.0,
            min(1.0, 0.6 * support_component + 0.4 * avg_trust - contradiction_penalty),
        )

    return claims


def detect_rumor_origin(documents: list[RetrievedDocument]) -> RumorOrigin:
    """Find earliest discoverable mention and estimate provenance confidence."""
    if not documents:
        return RumorOrigin(provenance_confidence=0.0)

    sorted_docs = sorted(documents, key=lambda d: date_sort_key(d.published_at))
    first = sorted_docs[0]
    source_id, trust_score = resolve_source(first.url, first.source_id)

    has_timestamp_coverage = sum(1 for doc in documents if doc.published_at) / len(
        documents
    )
    provenance_confidence = max(
        0.2, min(0.95, 0.5 * has_timestamp_coverage + 0.5 * trust_score)
    )

    return RumorOrigin(
        first_seen_at=first.published_at,
        first_seen_url=first.url,
        first_seen_source=source_id,
        provenance_confidence=provenance_confidence,
    )


def decide_verdict(claims: list[Claim]) -> tuple[Verdict, float, list[str]]:
    """Return verdict with overall confidence and uncertainty notes."""
    if not claims:
        return "unverified", 0.2, ["No sufficient evidence retrieved."]

    max_confidence = max(claim.confidence for claim in claims)
    supported_claims = [
        claim
        for claim in claims
        if claim.support_count >= 2 and claim.confidence >= 0.65
    ]
    contradiction_weight = sum(claim.contradiction_count for claim in claims)

    uncertainty_notes: list[str] = []
    if contradiction_weight > 0:
        uncertainty_notes.append("Conflicting evidence exists across sources.")
    if any(claim.support_count < 2 for claim in claims[:5]):
        uncertainty_notes.append("Some claims rely on single-source evidence.")

    if contradiction_weight >= len(claims) and max_confidence < 0.55:
        return "likely_false", max(0.25, max_confidence), uncertainty_notes
    if supported_claims:
        return "supported", max_confidence, uncertainty_notes
    return "unverified", max_confidence, uncertainty_notes
