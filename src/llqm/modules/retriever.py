"""Web retrieval backends: Serper API, DuckDuckGo, and composition helpers."""

from __future__ import annotations

import os
import re
from html import unescape
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from llqm.schemas.models import RetrievedDocument
from llqm.utils import load_dotenv_file
from llqm.utils.date_extractor import extract_date_from_html, normalize_date


class Retriever(Protocol):
    def retrieve(self, query: str, max_results: int = 5) -> list[RetrievedDocument]:
        """Return retrieved documents for a query."""
        ...


class NullRetriever:
    """Default retriever used during scaffolding / offline tests."""

    def retrieve(self, query: str, max_results: int = 5) -> list[RetrievedDocument]:
        _ = (query, max_results)
        return []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_duckduckgo_links(html_text: str, max_results: int) -> list[str]:
    pattern = re.compile(r'href="//duckduckgo\.com/l/\?([^\"]+)"')
    urls: list[str] = []
    for match in pattern.finditer(html_text):
        query = parse_qs(match.group(1))
        target = query.get("uddg", [""])[0]
        if not target:
            continue
        candidate = unescape(target)
        if candidate.startswith("http") and candidate not in urls:
            urls.append(candidate)
        if len(urls) >= max_results:
            break
    return urls


def _extract_text_excerpt(html_text: str, max_chars: int = 900) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", unescape(text)).strip()
    return text[:max_chars]


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.netloc.replace("www.", "")
    path = parsed.path.strip("/")
    if not path:
        return hostname
    return f"{hostname} / {path.split('/')[-1]}"


def _fetch_page(client: httpx.Client, url: str) -> tuple[str, str | None]:
    """Fetch a page and return ``(text_excerpt, extracted_date)``."""
    try:
        page = client.get(url)
        page.raise_for_status()
    except httpx.HTTPError:
        return "", None
    raw_html = page.text
    date = extract_date_from_html(raw_html)
    excerpt = _extract_text_excerpt(raw_html)
    return excerpt, date


# ---------------------------------------------------------------------------
# DuckDuckGo retriever
# ---------------------------------------------------------------------------


class LiveWebRetriever:
    """Fetch documents from DuckDuckGo HTML search results."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                )
            },
            follow_redirects=True,
        )

    def retrieve(self, query: str, max_results: int = 5) -> list[RetrievedDocument]:
        search_terms = [
            query,
            f"{query} rumor origin first reported",
            f"{query} site:reddit.com OR site:x.com OR site:youtube.com",
        ]

        urls: list[str] = []
        per_query_limit = max(2, max_results // len(search_terms) + 1)
        for term in search_terms:
            html_response = self._client.get(
                f"https://duckduckgo.com/html/?q={quote_plus(term)}"
            )
            html_response.raise_for_status()
            discovered = _parse_duckduckgo_links(html_response.text, per_query_limit)
            for item in discovered:
                if item not in urls:
                    urls.append(item)
                if len(urls) >= max_results:
                    break
            if len(urls) >= max_results:
                break

        documents: list[RetrievedDocument] = []
        for url in urls:
            excerpt, page_date = _fetch_page(self._client, url)
            if not excerpt:
                continue
            source_id = urlparse(url).netloc.replace("www.", "") or "web"
            documents.append(
                RetrievedDocument(
                    source_id=source_id,
                    url=url,
                    title=_title_from_url(url),
                    content=excerpt,
                    published_at=page_date,
                )
            )
        return documents


# ---------------------------------------------------------------------------
# Serper API retriever
# ---------------------------------------------------------------------------


class SerperRetriever:
    """Retriever backed by Serper API for higher-quality live search."""

    def __init__(self, api_key: str, timeout_seconds: float = 10.0) -> None:
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
        )
        self._fetch_client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                )
            },
            follow_redirects=True,
        )

    def retrieve(self, query: str, max_results: int = 5) -> list[RetrievedDocument]:
        response = self._client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": max(8, max_results * 2)},
        )
        response.raise_for_status()
        payload = response.json()

        candidates = payload.get("news") or payload.get("organic") or []
        documents: list[RetrievedDocument] = []

        for item in candidates[:max_results]:
            url = item.get("link")
            if not url:
                continue
            title = item.get("title") or _title_from_url(url)
            snippet = item.get("snippet") or ""
            serper_date = item.get("date")

            page_excerpt, page_date = _fetch_page(self._fetch_client, url)
            published_at = normalize_date(serper_date) or page_date

            source_id = urlparse(url).netloc.replace("www.", "") or "web"
            documents.append(
                RetrievedDocument(
                    source_id=source_id,
                    url=url,
                    title=title,
                    content=page_excerpt or snippet or title,
                    published_at=published_at,
                )
            )
        return documents


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class FallbackRetriever:
    """Try retrievers in order and merge distinct results across sources."""

    def __init__(self, retrievers: list[Retriever]) -> None:
        self._retrievers = retrievers

    def retrieve(self, query: str, max_results: int = 5) -> list[RetrievedDocument]:
        merged: dict[str, RetrievedDocument] = {}
        for retriever in self._retrievers:
            try:
                docs = retriever.retrieve(query, max_results=max_results)
            except Exception:
                continue
            for doc in docs:
                if doc.url not in merged:
                    merged[doc.url] = doc
                if len(merged) >= max_results:
                    return list(merged.values())
        return list(merged.values())


def build_retriever(use_live: bool = True) -> Retriever:
    """Build best available retriever based on env keys and mode."""
    if not use_live:
        return NullRetriever()

    load_dotenv_file()
    serper_key = os.environ.get("SERPER_API_KEY", "").strip()

    retrievers: list[Retriever] = []
    if serper_key:
        retrievers.append(SerperRetriever(api_key=serper_key))
    retrievers.append(LiveWebRetriever())

    return FallbackRetriever(retrievers)


# ---------------------------------------------------------------------------
# Article fetcher (seed a news-URL investigation)
# ---------------------------------------------------------------------------


def _extract_html_title(raw_html: str) -> str:
    m = re.search(r"<title[^>]*>([^<]+)</title>", raw_html, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def fetch_article(url: str) -> RetrievedDocument | None:
    """Fetch a single URL and return it as a RetrievedDocument, or None."""
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            raw_html = resp.read().decode(errors="replace")
    except Exception:
        return None
    title = _extract_html_title(raw_html) or _title_from_url(url)
    content = _extract_text_excerpt(raw_html, max_chars=2000)
    if not content:
        return None
    date = extract_date_from_html(raw_html)
    source_id = urlparse(url).netloc.replace("www.", "") or "web"
    return RetrievedDocument(
        source_id=source_id,
        url=url,
        title=title,
        content=content,
        published_at=date,
    )
