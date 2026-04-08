"""Shared helpers for web routes."""

from __future__ import annotations

import json
from urllib.parse import urlparse

from llqm.schemas.models import InvestigationResult
from llqm.utils.source_registry import SOURCE_TRUST


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname.replace("www.", "") or url
    except Exception:
        return url


def _trust_for_domain(domain: str) -> float:
    entry = SOURCE_TRUST.get(domain)
    if entry is None:
        return 0.5
    if isinstance(entry, tuple):
        return float(entry[1])
    return float(entry)


def enrich_timeline_json(result: InvestigationResult) -> str:
    """Build enriched timeline JSON with authority and support info per event."""
    events_data = []
    # Pre-compute claim evidence URL sets
    claim_info: list[dict] = []
    for claim in result.key_claims:
        urls = {ev.url for ev in claim.evidence}
        claim_info.append(
            {
                "urls": urls,
                "support": claim.support_count,
                "contradiction": claim.contradiction_count,
            }
        )

    for event in result.timeline:
        d = event.model_dump(mode="json")

        # Authority from source trust scores
        trust_scores = []
        for url in event.source_urls:
            domain = _domain_from_url(url)
            trust_scores.append(_trust_for_domain(domain))
        d["authority"] = (
            round(sum(trust_scores) / len(trust_scores), 2) if trust_scores else None
        )
        d["source_count"] = len(event.source_urls)
        d["source_domains"] = list(
            dict.fromkeys(_domain_from_url(url) for url in event.source_urls)
        )  # unique, order-preserving

        # Support / contradiction from claims sharing source URLs
        event_urls = set(event.source_urls)
        support = 0
        contradiction = 0
        if event_urls:
            for ci in claim_info:
                if event_urls & ci["urls"]:
                    support += ci["support"]
                    contradiction += ci["contradiction"]
        d["support_count"] = support
        d["contradiction_count"] = contradiction

        events_data.append(d)

    return json.dumps(events_data)
