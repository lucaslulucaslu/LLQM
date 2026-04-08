from __future__ import annotations

from urllib.parse import urlparse


SOURCE_TRUST: dict[str, tuple[str, float]] = {
    "reuters.com": ("reuters", 0.92),
    "apnews.com": ("ap", 0.92),
    "bbc.com": ("bbc", 0.9),
    "nytimes.com": ("nytimes", 0.86),
    "x.com": ("x", 0.35),
    "twitter.com": ("x", 0.35),
    "reddit.com": ("reddit", 0.3),
    "youtube.com": ("youtube", 0.4),
}


def resolve_source(url: str, fallback_source_id: str) -> tuple[str, float]:
    """Map URL domain to normalized source id and trust score."""
    domain = urlparse(url).netloc.lower().replace("www.", "")
    for known_domain, (source_id, trust_score) in SOURCE_TRUST.items():
        if domain.endswith(known_domain):
            return source_id, trust_score
    return fallback_source_id, 0.5
