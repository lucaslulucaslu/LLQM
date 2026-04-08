"""Investigation routes — start + result page."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from llqm.modules.llm import build_llm
from llqm.modules.retriever import build_retriever, fetch_article
from llqm.schemas.models import RetrievedDocument
from llqm.service.investigation_service import _article_to_query, investigate_stream
from llqm.web import state

router = APIRouter()


def _run_investigation(
    inv_id: str, query: str, seed_documents: list[RetrievedDocument]
) -> None:
    """Run investigation synchronously (called via asyncio.to_thread)."""
    node_icons = {
        "plan": "🧭",
        "retrieve": "🔍",
        "extract": "📑",
        "verify": "⚖️",
        "synthesize": "📝",
    }

    final_state = None
    try:
        retriever = build_retriever(use_live=True)
        llm = build_llm()

        gen = investigate_stream(
            query=query,
            retriever=retriever,
            llm=llm,
            seed_documents=seed_documents,
        )

        while True:
            event = next(gen)
            icon = node_icons.get(event.node, "🔄")
            state.update_progress(inv_id, icon, event.label, event.detail)
            final_state = event.state
    except StopIteration as exc:
        result = exc.value
        documents = final_state.get("documents", []) if final_state else []
        state.finish(inv_id, result, documents)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        state.fail(inv_id, str(exc))


@router.post("/investigate")
async def start_investigation(
    request: Request,
    mode: str = Form("rumor"),
    query: str = Form(""),
    url: str = Form(""),
):
    inv_id = uuid.uuid4().hex[:12]
    seed_documents: list[RetrievedDocument] = []

    if mode == "url" and url.strip():
        article = fetch_article(url.strip())
        if article is None:
            inv = state.StoredInvestigation(id=inv_id, query=url, mode="url")
            state.create(inv)
            state.fail(inv_id, f"Could not fetch or parse URL: {url}")
            return RedirectResponse(f"/result/{inv_id}", status_code=303)
        seed_documents = [article]
        llm = build_llm()
        actual_query = _article_to_query(llm, article)
    else:
        actual_query = query.strip()

    if not actual_query:
        return RedirectResponse("/", status_code=303)

    inv = state.StoredInvestigation(id=inv_id, query=actual_query, mode=mode)
    state.create(inv)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        None,
        _run_investigation,
        inv_id,
        actual_query,
        seed_documents,
    )

    return RedirectResponse(f"/result/{inv_id}", status_code=303)


@router.get("/result/{inv_id}", response_class=HTMLResponse)
async def result_page(request: Request, inv_id: str):
    inv = state.get(inv_id)
    if inv is None:
        return RedirectResponse("/", status_code=303)

    import json

    timeline_json = "[]"
    if inv.result and inv.result.timeline:
        timeline_json = json.dumps(
            [e.model_dump(mode="json") for e in inv.result.timeline]
        )

    return request.app.state.templates.TemplateResponse(
        request,
        "result.html",
        {"inv": inv, "timeline_json": timeline_json},
    )
