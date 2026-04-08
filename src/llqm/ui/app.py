from __future__ import annotations

import html

import streamlit as st

from llqm.modules.llm import build_llm
from llqm.modules.retriever import build_retriever, fetch_article
from llqm.schemas.models import RetrievedDocument, TimelineEvent
from llqm.service.investigation_service import (
    _article_to_query,
    investigate_stream,
)


# ---------------------------------------------------------------------------
# Visual timeline renderer (pure HTML/CSS injected via st.markdown)
# ---------------------------------------------------------------------------

_TIMELINE_CSS = """
<style>
.tl-wrap {
    position: relative;
    padding: 20px 0 20px 0;
}
/* vertical spine */
.tl-wrap::before {
    content: '';
    position: absolute;
    left: 18px;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #4A90D9 0%, #7B68EE 100%);
    border-radius: 2px;
}
.tl-item {
    position: relative;
    margin: 0 0 28px 50px;
    padding: 0;
}
/* dot on the line */
.tl-item::before {
    content: '';
    position: absolute;
    left: -40px;
    top: 6px;
    width: 13px;
    height: 13px;
    background: #4A90D9;
    border: 3px solid #1E1E2E;
    border-radius: 50%;
    z-index: 1;
}
/* connector from dot to card */
.tl-item::after {
    content: '';
    position: absolute;
    left: -27px;
    top: 11px;
    width: 20px;
    height: 2px;
    background: #4A90D9;
}
.tl-date {
    font-size: 0.78em;
    font-weight: 600;
    color: #7B68EE;
    margin-bottom: 4px;
    letter-spacing: 0.03em;
}
.tl-card {
    background: #262636;
    border: 1px solid #363650;
    border-radius: 8px;
    padding: 12px 16px;
    transition: border-color 0.2s;
}
.tl-card:hover {
    border-color: #7B68EE;
}
.tl-summary {
    font-size: 0.92em;
    color: #E0E0E0;
    line-height: 1.45;
    margin: 0;
}
.tl-source {
    font-size: 0.75em;
    margin-top: 6px;
}
.tl-source a {
    color: #4A90D9;
    text-decoration: none;
}
.tl-source a:hover {
    text-decoration: underline;
}
</style>
"""


def _render_timeline_html(events: list[TimelineEvent]) -> str:
    """Build the HTML string for a visual timeline."""
    items: list[str] = []
    for event in events:
        date_label = html.escape(event.timestamp or "Date unknown")
        summary = html.escape(event.summary[:200])
        source_links = ""
        for url in event.source_urls[:2]:
            domain = html.escape(url.split("//")[-1].split("/")[0].replace("www.", ""))
            source_links += (
                f' <a href="{html.escape(url)}" target="_blank">{domain}</a>'
            )
        items.append(
            f'<div class="tl-item">'
            f'  <div class="tl-date">{date_label}</div>'
            f'  <div class="tl-card">'
            f'    <p class="tl-summary">{summary}</p>'
            f'    <div class="tl-source">Source:{source_links}</div>'
            f"  </div>"
            f"</div>"
        )
    return _TIMELINE_CSS + '<div class="tl-wrap">' + "\n".join(items) + "</div>"


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

_NODE_ICONS: dict[str, str] = {
    "plan": "🧭",
    "retrieve": "🔍",
    "extract": "📑",
    "verify": "⚖️",
    "synthesize": "📝",
}


def render_timeline_app() -> None:
    st.set_page_config(page_title="LLQM Investigator", layout="wide")
    st.title("LLQM Timeline Investigator")
    st.caption("News + rumor tracing with verdicts, confidence, and evidence links")

    # --- Dual-mode input: question OR news URL ---
    mode = st.radio(
        "Investigation mode",
        ["Rumor / Question", "News URL"],
        horizontal=True,
    )

    if mode == "News URL":
        url_input = st.text_input(
            "Paste a news article URL",
            placeholder="https://www.reuters.com/world/...",
        )
        go = st.button("Build full timeline", type="primary") and url_input.strip()
    else:
        query_input = st.text_input(
            "What do you want to investigate?",
            placeholder="Example: Did rumor X originate from account Y?",
        )
        go = st.button("Investigate", type="primary") and query_input.strip()  # type: ignore[possibly-undefined]

    if go:
        retriever = build_retriever(use_live=True)
        llm = build_llm()
        seed_documents: list[RetrievedDocument] = []

        # --- If URL mode, fetch the seed article first ---
        if mode == "News URL":
            with st.spinner("Fetching article..."):
                article = fetch_article(url_input.strip())
            if article is None:
                st.error(
                    "Could not fetch or parse that URL. Please check it and try again."
                )
                return
            seed_documents = [article]
            query = _article_to_query(llm, article)
            st.info(f"Research question derived from article: **{query}**")
        else:
            query = query_input.strip()  # type: ignore[possibly-undefined]

        # --- Streaming investigation with live progress ---
        status_box = st.status("Starting investigation...", expanded=True)
        result = None
        gen = investigate_stream(
            query=query,
            retriever=retriever,
            llm=llm,
            seed_documents=seed_documents,
        )
        try:
            while True:
                event = next(gen)
                icon = _NODE_ICONS.get(event.node, "🔄")
                status_box.update(label=f"{icon} {event.label}...", state="running")
                status_box.write(f"↳ {event.detail}")
        except StopIteration as exc:
            result = exc.value
            status_box.update(
                label="✅ Investigation complete",
                state="complete",
                expanded=False,
            )

        if result is None:
            st.error("Investigation did not produce a result.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", result.verdict)
        col2.metric("Confidence", f"{result.confidence:.2f}")
        col3.metric("Timeline events", str(len(result.timeline)))

        if result.summary:
            st.subheader("Story Summary")
            st.markdown(result.summary)

        if mode == "Rumor / Question":
            st.subheader("Rumor Origin")
            st.write(
                f"First seen source: {result.rumor_origin.first_seen_source or 'unknown'}"
            )
            st.write(
                f"First seen time: {result.rumor_origin.first_seen_at or 'unknown'}"
            )
            st.write(
                f"Provenance confidence: {result.rumor_origin.provenance_confidence:.2f}"
            )
            if result.rumor_origin.first_seen_url:
                st.link_button(
                    "Open first seen URL", result.rumor_origin.first_seen_url
                )

        st.subheader("Timeline")
        if not result.timeline:
            st.info("No timeline events were produced from current evidence.")
        else:
            st.markdown(
                _render_timeline_html(result.timeline),
                unsafe_allow_html=True,
            )

        st.subheader("Top Claims")
        for claim in result.key_claims:
            with st.expander(claim.text[:120]):
                st.write(f"Confidence: {claim.confidence:.2f}")
                st.write(
                    f"Support: {claim.support_count} | Contradictions: {claim.contradiction_count}"
                )
                for evidence in claim.evidence[:5]:
                    st.write(f"- {evidence.source_id}: {evidence.excerpt[:180]}")
                    st.link_button("Open evidence", evidence.url)

        if result.uncertainty_notes:
            st.subheader("Uncertainty Notes")
            for note in result.uncertainty_notes:
                st.warning(note)


def main() -> None:
    render_timeline_app()


if __name__ == "__main__":
    main()
