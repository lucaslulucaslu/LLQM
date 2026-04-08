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
        source_id, trust_score = resolve_source(
            doc.url,
            doc.source_id,
            page_content=doc.content,
        )
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

        # Count evidence that explicitly debunks / contradicts the claim.
        # We look for negation patterns that indicate the *article itself*
        # is refuting the claim, not just mentioning the word "false".
        _negation_phrases = (
            "not true",
            "no evidence",
            "been debunked",
            "was debunked",
            "is false",
            "are false",
            "was false",
            "were false",
            "misleading claim",
            "baseless",
            "unfounded",
            "fabricated",
            "no proof",
            "has been denied",
            "was denied",
        )
        negation_hits = sum(
            1
            for item in claim.evidence
            if any(phrase in item.excerpt.lower() for phrase in _negation_phrases)
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
    source_id, trust_score = resolve_source(
        first.url,
        first.source_id,
        page_content=first.content,
    )

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
    """Return verdict with overall confidence and uncertainty notes.

    Strategy:
    - Weigh each claim by its confidence to compute an overall score.
    - A claim is "supportive" if support_count > contradiction_count.
    - A claim is "contradicting" if contradiction_count > 0 AND
      contradiction_count >= support_count.
    - The verdict reflects the *balance* of weighted support vs contradiction.
    """
    if not claims:
        return "unverified", 0.2, ["No sufficient evidence retrieved."]

    total_support_weight = 0.0
    total_contra_weight = 0.0
    total_weight = 0.0

    for claim in claims:
        w = max(claim.confidence, 0.1)  # avoid zero-weight claims
        total_weight += w
        if claim.support_count > claim.contradiction_count:
            total_support_weight += w
        elif (
            claim.contradiction_count > 0
            and claim.contradiction_count >= claim.support_count
        ):
            total_contra_weight += w
        # else: neutral / unverified — contributes to total_weight only

    support_ratio = total_support_weight / total_weight if total_weight else 0
    contra_ratio = total_contra_weight / total_weight if total_weight else 0

    max_confidence = max(claim.confidence for claim in claims)
    avg_confidence = sum(claim.confidence for claim in claims) / len(claims)

    uncertainty_notes: list[str] = []
    if total_contra_weight > 0:
        uncertainty_notes.append("Conflicting evidence exists across sources.")
    if any(claim.support_count < 2 for claim in claims[:5]):
        uncertainty_notes.append("Some claims rely on single-source evidence.")

    # "likely_false" requires strong contradiction signal
    if contra_ratio > 0.5 and support_ratio < 0.3:
        confidence = max(0.3, min(0.9, avg_confidence + contra_ratio * 0.2))
        return "likely_false", confidence, uncertainty_notes

    # "supported" requires meaningful support
    if support_ratio >= 0.4 and support_ratio > contra_ratio:
        return "supported", max_confidence, uncertainty_notes

    return "unverified", avg_confidence, uncertainty_notes
