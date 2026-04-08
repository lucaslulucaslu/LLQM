"""Extract and normalise publication dates from HTML and free-text strings."""

from __future__ import annotations

import re
from datetime import datetime, timedelta


_DATE_PATTERNS = [
    # JSON-LD / schema.org structured data
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"'),
    re.compile(r'"dateCreated"\s*:\s*"([^"]+)"'),
    re.compile(r'"dateModified"\s*:\s*"([^"]+)"'),
    # <meta> open-graph / article tags
    re.compile(
        r'<meta[^>]+(?:property|name)="article:published_time"[^>]+content="([^"]+)"',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content="([^"]+)"[^>]+(?:property|name)="article:published_time"',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+(?:property|name)="date"[^>]+content="([^"]+)"',
        re.IGNORECASE,
    ),
    # <time datetime="...">
    re.compile(r"<time[^>]+datetime=\"([^\"]+)\"", re.IGNORECASE),
    # Plain-text date lines
    re.compile(
        r"(?:Published|Posted|Updated|Date)\s*:?\s*(\w+ \d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    ),
]


# -- Relative-date patterns (e.g. "2 hours ago", "3 days ago") -------------
_RELATIVE_RE = re.compile(
    r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)

_MONTH_ABBREV = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Patterns for free-text dates from Serper and similar APIs
_TEXT_DATE_FORMATS = (
    "%B %d, %Y",  # January 15, 2019
    "%B %d %Y",  # January 15 2019
    "%b %d, %Y",  # Jan 15, 2019
    "%b %d %Y",  # Jan 15 2019
    "%d %B %Y",  # 15 January 2019
    "%d %b %Y",  # 15 Jan 2019
    "%b %Y",  # Jan 2019
    "%B %Y",  # January 2019
    "%m/%d/%Y",  # 01/15/2019
    "%d/%m/%Y",  # 15/01/2019
    "%Y/%m/%d",  # 2019/01/15
)


def normalize_date(raw: str | None) -> str | None:
    """Normalise *any* date string to ``YYYY-MM-DD`` (or ``YYYY-MM``).

    Handles ISO dates, relative dates (``3 days ago``), month-year
    (``Jan 2019``), and common US/EU text formats.
    """
    if not raw:
        return None
    raw = raw.strip()

    # 1) Already ISO-like: 2024-01-15 or 2024-01-15T12:30:00Z
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]

    # 2) ISO month-only: 2024-01
    if re.match(r"^\d{4}-\d{2}$", raw):
        return raw

    # 3) Relative: "2 hours ago", "3 days ago", etc.
    rel = _RELATIVE_RE.search(raw)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2).lower()
        now = datetime.utcnow()
        if unit == "second":
            dt = now - timedelta(seconds=amount)
        elif unit == "minute":
            dt = now - timedelta(minutes=amount)
        elif unit == "hour":
            dt = now - timedelta(hours=amount)
        elif unit == "day":
            dt = now - timedelta(days=amount)
        elif unit == "week":
            dt = now - timedelta(weeks=amount)
        elif unit == "month":
            dt = now - timedelta(days=amount * 30)
        elif unit == "year":
            dt = now - timedelta(days=amount * 365)
        else:
            return None
        return dt.strftime("%Y-%m-%d")

    # 4) Try strptime with common formats
    for fmt in _TEXT_DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            if "%d" in fmt:
                return dt.strftime("%Y-%m-%d")
            return dt.strftime("%Y-%m")
        except ValueError:
            continue

    # 5) Bare "Month Year" / "Mon Year" not caught above (with extra text)
    m = re.search(r"(\w{3,9})\s+(\d{4})", raw)
    if m:
        month_str = m.group(1).lower()[:3]
        year = int(m.group(2))
        month_num = _MONTH_ABBREV.get(month_str)
        if month_num and 1900 <= year <= 2100:
            return f"{year}-{month_num:02d}"

    # 6) Bare year
    m = re.match(r"^(\d{4})$", raw)
    if m:
        year = int(m.group(1))
        if 1900 <= year <= 2100:
            return f"{year}-01"

    return None


def date_sort_key(date_str: str | None) -> str:
    """Return a string that sorts dates chronologically.

    Handles full dates (``YYYY-MM-DD``), partial dates (``YYYY-MM``),
    approximate dates (``~YYYY-MM``), and ``None`` (sorted last).
    """
    if not date_str:
        return "9999-99-99"
    cleaned = date_str.lstrip("~").strip()
    normalized = normalize_date(cleaned)
    if not normalized:
        return "9999-99-99"
    # Pad partial dates so they sort correctly: YYYY-MM → YYYY-MM-01
    if re.match(r"^\d{4}-\d{2}$", normalized):
        return normalized + "-01"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", normalized):
        return normalized
    return "9999-99-99"


def extract_date_from_html(html: str) -> str | None:
    """Try to extract a publication date from raw HTML content."""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(html)
        if match:
            result = normalize_date(match.group(1).strip())
            if result:
                return result
    return None
