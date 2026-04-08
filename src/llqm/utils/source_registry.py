from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# 1. Known editorial trust scores (fast-path lookup)
# ---------------------------------------------------------------------------

SOURCE_TRUST: dict[str, tuple[str, float]] = {
    # Wire services
    "reuters.com": ("reuters", 0.92),
    "apnews.com": ("ap", 0.92),
    "afp.com": ("afp", 0.90),
    # Major newspapers / broadcasters
    "bbc.com": ("bbc", 0.90),
    "bbc.co.uk": ("bbc", 0.90),
    "nytimes.com": ("nytimes", 0.86),
    "washingtonpost.com": ("wapo", 0.85),
    "theguardian.com": ("guardian", 0.85),
    "economist.com": ("economist", 0.85),
    "ft.com": ("ft", 0.85),
    "wsj.com": ("wsj", 0.85),
    "cnn.com": ("cnn", 0.78),
    "nbcnews.com": ("nbc", 0.78),
    "abcnews.go.com": ("abc", 0.78),
    "cbsnews.com": ("cbs", 0.78),
    "npr.org": ("npr", 0.85),
    "pbs.org": ("pbs", 0.84),
    # Fact-checkers
    "snopes.com": ("snopes", 0.82),
    "politifact.com": ("politifact", 0.82),
    "factcheck.org": ("factcheck", 0.84),
    "fullfact.org": ("fullfact", 0.83),
    # Science / research
    "nature.com": ("nature", 0.90),
    "sciencemag.org": ("science", 0.90),
    "thelancet.com": ("lancet", 0.90),
    "pubmed.ncbi.nlm.nih.gov": ("pubmed", 0.88),
    # Social media / user-generated
    "x.com": ("x", 0.35),
    "twitter.com": ("x", 0.35),
    "reddit.com": ("reddit", 0.30),
    "youtube.com": ("youtube", 0.40),
    "facebook.com": ("facebook", 0.35),
    "instagram.com": ("instagram", 0.30),
    "tiktok.com": ("tiktok", 0.25),
    "threads.net": ("threads", 0.32),
    # Blog platforms (low trust — anyone can publish)
    "medium.com": ("medium", 0.38),
    "substack.com": ("substack", 0.40),
    "blogspot.com": ("blogspot", 0.30),
    "wordpress.com": ("wordpress", 0.30),
    "tumblr.com": ("tumblr", 0.25),
}


# ---------------------------------------------------------------------------
# 2. TLD-based trust signals
# ---------------------------------------------------------------------------

_TLD_TRUST: dict[str, float] = {
    ".gov": 0.88,
    ".mil": 0.88,
    ".edu": 0.82,
    ".int": 0.80,
    ".ac.uk": 0.80,
    ".gov.uk": 0.88,
    ".go.jp": 0.85,
    ".gouv.fr": 0.85,
}


def _tld_trust(domain: str) -> float | None:
    """Return a trust score based on the TLD, or None if not a special TLD."""
    for suffix, score in _TLD_TRUST.items():
        if domain.endswith(suffix):
            return score
    return None


# ---------------------------------------------------------------------------
# 3. Observable page-signal scoring
# ---------------------------------------------------------------------------

_LOW_TRUST_SUBDOMAIN_RE = re.compile(
    r"\.(blogspot|wordpress|tumblr|wixsite|weebly)\.", re.IGNORECASE
)

_POSITIVE_SIGNALS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r'"@type"\s*:\s*"NewsArticle"', re.I), 0.08),
    (re.compile(r'"@type"\s*:\s*"ReportageNewsArticle"', re.I), 0.10),
    (re.compile(r'<meta[^>]+name=["\']author["\']', re.I), 0.05),
    (re.compile(r'<meta[^>]+property=["\']article:published_time["\']', re.I), 0.04),
    (re.compile(r'"corrections?Policy"', re.I), 0.06),
    (re.compile(r'"ethicsPolicy"', re.I), 0.06),
]


def _signal_score(domain: str, page_content: str | None) -> tuple[float, float]:
    """Compute a trust score from observable signals.

    Returns (score, confidence) where confidence indicates how much
    signal we found (low confidence → should use LLM fallback).
    """
    # --- TLD boost ---
    tld = _tld_trust(domain)
    if tld is not None:
        return tld, 0.85

    # --- Low-trust subdomain patterns ---
    if _LOW_TRUST_SUBDOMAIN_RE.search(domain):
        return 0.30, 0.70

    base = 0.50
    confidence = 0.0

    # --- Page metadata signals ---
    if page_content:
        snippet = page_content[:5000]
        signal_bonus = 0.0
        for pattern, bonus in _POSITIVE_SIGNALS:
            if pattern.search(snippet):
                signal_bonus += bonus
                confidence += 0.10
        base += min(signal_bonus, 0.25)
        if signal_bonus > 0.10:
            confidence += 0.15

    return max(0.1, min(0.95, base)), confidence


# ---------------------------------------------------------------------------
# 4. LLM-assisted classification (fallback for unknown sources)
# ---------------------------------------------------------------------------

_SOURCE_CLASSIFY_SYSTEM = (
    "You are a media-literacy expert. Given a website domain name and "
    "optionally a snippet of page content, classify the source.\n\n"
    "Return JSON:\n"
    '{"source_type": "wire_agency|newspaper|broadcaster|fact_checker|'
    'academic|government|magazine|blog|social_media|forum|unknown",\n'
    ' "trust_score": 0.0-1.0,\n'
    ' "reason": "one sentence"}'
)


def _llm_classify_source(
    llm: Any,
    domain: str,
    page_snippet: str | None,
) -> tuple[float, str] | None:
    """Ask the LLM to classify an unknown source. Returns (score, type)."""
    snippet_text = page_snippet[:800] if page_snippet else "No page content available."
    try:
        data = llm.chat_json(
            system=_SOURCE_CLASSIFY_SYSTEM,
            user=f"Domain: {domain}\nPage snippet:\n{snippet_text}",
        )
        score = max(0.1, min(0.95, float(data.get("trust_score", 0.5))))
        source_type = str(data.get("source_type", "unknown"))
        return score, source_type
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. Domain-level cache
# ---------------------------------------------------------------------------

_domain_cache: dict[str, tuple[str, float]] = {}


def clear_trust_cache() -> None:
    """Reset the domain trust cache (useful in tests)."""
    _domain_cache.clear()


# ---------------------------------------------------------------------------
# 6. Public API — hybrid resolve_source
# ---------------------------------------------------------------------------


def resolve_source(
    url: str,
    fallback_source_id: str,
    *,
    llm: Any | None = None,
    page_content: str | None = None,
) -> tuple[str, float]:
    """Map URL domain to (source_id, trust_score).

    Resolution order:
    1. Known editorial dictionary (fast, high confidence)
    2. TLD-based rules (.gov, .edu, …)
    3. Observable page-signal heuristics
    4. LLM classification (if available and heuristic confidence is low)
    5. Default 0.5
    """
    domain = urlparse(url).netloc.lower().replace("www.", "")

    # --- Fast path: known domain ---
    for known_domain, (source_id, trust_score) in SOURCE_TRUST.items():
        if domain.endswith(known_domain):
            return source_id, trust_score

    # --- Check cache ---
    if domain in _domain_cache:
        return _domain_cache[domain]

    # --- Signal-based scoring ---
    score, confidence = _signal_score(domain, page_content)
    source_id = fallback_source_id

    # --- LLM fallback when heuristic confidence is low ---
    if confidence < 0.40 and llm is not None and getattr(llm, "available", False):
        result = _llm_classify_source(llm, domain, page_content)
        if result is not None:
            score, source_type = result
            source_id = source_type

    # --- Cache and return ---
    _domain_cache[domain] = (source_id, score)
    return source_id, score
