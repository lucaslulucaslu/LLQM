"""htmx partial endpoints for tabs and detail panels."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from llqm.utils.source_registry import SOURCE_TRUST
from llqm.web import state
from llqm.web.helpers import enrich_timeline_json

router = APIRouter()


@router.get("/result/{inv_id}/timeline", response_class=HTMLResponse)
async def timeline_partial(request: Request, inv_id: str):
    inv = state.get(inv_id)
    if not inv or not inv.result:
        return HTMLResponse("<p>Not found</p>", status_code=404)

    timeline_json = enrich_timeline_json(inv.result)
    html = request.app.state.templates.get_template("partials/timeline.html").render(
        inv=inv, timeline_json=timeline_json
    )
    return HTMLResponse(html)


@router.get("/result/{inv_id}/claims", response_class=HTMLResponse)
async def claims_partial(request: Request, inv_id: str):
    inv = state.get(inv_id)
    if not inv or not inv.result:
        return HTMLResponse("<p>Not found</p>", status_code=404)

    html = request.app.state.templates.get_template("partials/claims.html").render(
        inv=inv
    )
    return HTMLResponse(html)


@router.get("/result/{inv_id}/claim/{idx}", response_class=HTMLResponse)
async def claim_detail_partial(request: Request, inv_id: str, idx: int):
    inv = state.get(inv_id)
    if not inv or not inv.result:
        return HTMLResponse("<p>Not found</p>", status_code=404)

    claims = inv.result.key_claims
    if idx < 0 or idx >= len(claims):
        return HTMLResponse("<p>Claim not found</p>", status_code=404)

    html = request.app.state.templates.get_template(
        "partials/claim_detail.html"
    ).render(claim=claims[idx])
    return HTMLResponse(html)


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname.replace("www.", "") or url
    except Exception:
        return url


def _trust_for_domain(domain: str) -> float:
    entry = SOURCE_TRUST.get(domain)
    if entry is None:
        return 0.5
    # SOURCE_TRUST values are (source_id, trust_score) tuples
    if isinstance(entry, tuple):
        return float(entry[1])
    return float(entry)


@router.get("/result/{inv_id}/sources", response_class=HTMLResponse)
async def sources_partial(
    request: Request,
    inv_id: str,
    sort: str = Query("trust", pattern="^(trust|date)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    inv = state.get(inv_id)
    if not inv:
        return HTMLResponse("<p>Not found</p>", status_code=404)

    # Build source list from stored documents
    sources = []
    for doc in inv.documents:
        domain = _domain_from_url(doc.url)
        sources.append(
            {
                "domain": domain,
                "trust": _trust_for_domain(domain),
                "date": doc.published_at,
                "title": doc.title,
                "url": doc.url,
            }
        )

    # Sort
    reverse = order == "desc"
    if sort == "trust":
        sources.sort(key=lambda s: s["trust"], reverse=reverse)
    elif sort == "date":
        sources.sort(key=lambda s: s["date"] or "", reverse=reverse)

    html = request.app.state.templates.get_template("partials/sources.html").render(
        inv_id=inv_id, sources=sources, sort_by=sort, sort_order=order
    )
    return HTMLResponse(html)


@router.get("/result/{inv_id}/event/{idx}", response_class=HTMLResponse)
async def event_detail_partial(request: Request, inv_id: str, idx: int):
    inv = state.get(inv_id)
    if not inv or not inv.result:
        return HTMLResponse("<p>Not found</p>", status_code=404)

    timeline = inv.result.timeline
    if idx < 0 or idx >= len(timeline):
        return HTMLResponse("<p>Event not found</p>", status_code=404)

    html = request.app.state.templates.get_template(
        "partials/event_detail.html"
    ).render(event=timeline[idx])
    return HTMLResponse(html)
