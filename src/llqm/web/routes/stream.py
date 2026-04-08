"""SSE streaming route for live investigation progress."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from llqm.web import state
from llqm.web.helpers import enrich_timeline_json

router = APIRouter()


def _sse_event(event_name: str, html: str) -> str:
    """Format a multi-line HTML string as a valid SSE message.

    SSE requires each line of data to be prefixed with 'data: '.
    """
    lines = html.replace("\r\n", "\n").split("\n")
    data_lines = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event_name}\n{data_lines}\n\n"


@router.get("/result/{inv_id}/stream")
async def stream_progress(request: Request, inv_id: str):
    """SSE endpoint that pushes progress events and a final 'done' event."""

    async def event_generator():
        seen = 0
        while True:
            if await request.is_disconnected():
                return

            inv = state.get(inv_id)
            if inv is None:
                return

            # Send any new progress items
            progress = inv.progress
            while seen < len(progress):
                p = progress[seen]
                html = request.app.state.templates.get_template(
                    "partials/progress.html"
                ).render(icon=p["node"], label=p["label"], detail=p["detail"])
                yield _sse_event("progress", html)
                seen += 1

            if inv.status == "done":
                inner = _extract_result_content(inv, request)
                yield _sse_event("done", inner)
                return

            if inv.status == "error":
                error_html = (
                    f'<div id="result-container" hx-swap-oob="innerHTML">'
                    f'<div class="max-w-2xl mx-auto mt-8">'
                    f'<div class="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">'
                    f"Investigation failed: {inv.error}</div>"
                    f'<a href="/" class="inline-block mt-4 text-[var(--accent)] hover:underline">&larr; Try again</a>'
                    f"</div></div>"
                )
                yield _sse_event("done", error_html)
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _extract_result_content(inv: state.StoredInvestigation, request: Request) -> str:
    """Render the done-state result HTML for SSE injection."""
    templates = request.app.state.templates

    timeline_json = "[]"
    if inv.result and inv.result.timeline:
        timeline_json = enrich_timeline_json(inv.result)

    # Render the verdict partial
    verdict_html = templates.get_template("partials/verdict.html").render(inv=inv)

    # Render the timeline partial
    timeline_html = templates.get_template("partials/timeline.html").render(
        inv=inv,
        timeline_json=timeline_json,
    )

    r = inv.result
    inv_id = inv.id

    return f"""<div id="result-container" hx-swap-oob="innerHTML">
    <div class="mb-6">{verdict_html}</div>

    <div class="border-b border-[var(--border)] mb-4">
      <nav class="flex gap-1" id="tab-nav">
        <button class="tab-btn active-tab px-4 py-2 text-sm font-medium rounded-t-lg
                       border border-b-0 border-[var(--border)]
                       bg-[var(--bg-secondary)] text-[var(--accent)]"
                hx-get="/result/{inv_id}/timeline"
                hx-target="#tab-content" hx-swap="innerHTML"
                onclick="setActiveTab(this)">Timeline</button>
        <button class="tab-btn px-4 py-2 text-sm font-medium rounded-t-lg
                       text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                hx-get="/result/{inv_id}/claims"
                hx-target="#tab-content" hx-swap="innerHTML"
                onclick="setActiveTab(this)">Claims</button>
        <button class="tab-btn px-4 py-2 text-sm font-medium rounded-t-lg
                       text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                hx-get="/result/{inv_id}/sources"
                hx-target="#tab-content" hx-swap="innerHTML"
                onclick="setActiveTab(this)">Sources</button>
      </nav>
    </div>

    <div class="flex gap-6">
      <div id="tab-content" class="flex-1 min-w-0">{timeline_html}</div>
      <div id="event-detail-panel"
           class="hidden w-96 shrink-0 bg-[var(--bg-secondary)] rounded-xl
                  border border-[var(--border)] p-4 self-start sticky top-20"></div>
    </div>

    <div class="mt-8 bg-[var(--bg-secondary)] rounded-xl border border-[var(--border)] p-4">
      <form action="/result/{inv_id}/followup" method="POST" class="flex gap-3">
        <input type="text" name="query"
               placeholder="Ask a follow-up question about this story…"
               class="flex-1 px-4 py-2 rounded-lg
                      bg-[var(--bg-primary)] border border-[var(--border)]
                      text-[var(--text-primary)] placeholder-[var(--text-muted)]
                      focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
               required>
        <button type="submit"
                class="px-5 py-2 rounded-lg font-semibold
                       bg-[var(--accent)] text-white
                       hover:opacity-90 transition-opacity whitespace-nowrap">
          Ask
        </button>
      </form>
    </div>

    <script>
    function setActiveTab(el) {{
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.remove('active-tab','border','border-b-0',
          'border-[var(--border)]','bg-[var(--bg-secondary)]','text-[var(--accent)]');
        btn.classList.add('text-[var(--text-secondary)]');
      }});
      el.classList.add('active-tab','border','border-b-0',
        'border-[var(--border)]','bg-[var(--bg-secondary)]','text-[var(--accent)]');
      el.classList.remove('text-[var(--text-secondary)]');
    }}
    htmx.process(document.body);
    </script>
    </div>"""
