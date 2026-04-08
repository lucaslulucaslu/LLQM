"""LangGraph investigation pipeline with optional LLM-powered reasoning."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph

from llqm.modules.llm import LLMClient
from llqm.modules.retriever import NullRetriever, Retriever
from llqm.schemas.models import (
    Claim,
    Evidence,
    InvestigationResult,
    RetrievedDocument,
    RumorOrigin,
    TimelineEvent,
    Verdict,
)
from llqm.service.timeline_service import build_timeline
from llqm.service.verification_service import (
    decide_verdict,
    detect_rumor_origin,
    extract_claims,
    score_claims,
)


class AgentState(TypedDict, total=False):
    query: str
    max_iterations: int
    iteration: int
    search_queries: list[str]
    documents: list[RetrievedDocument]
    claims: list[Claim]
    timeline: list[TimelineEvent]
    rumor_origin: RumorOrigin
    verdict: str
    confidence: float
    uncertainty_notes: list[str]
    sufficient: bool
    sufficiency_score: float
    sufficiency_reason: str
    result: InvestigationResult


# ---------------------------------------------------------------------------
# System prompts for LLM-powered nodes
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are an investigative-research planner. Given a user question and "
    "optionally a summary of evidence already collected, produce 3-4 targeted "
    "web-search queries that will uncover new information.\n"
    "Focus on: earliest reports, primary sources, fact-checks, and "
    "social-media origins.\n"
    "If the user supplied a seed article, also generate queries that trace "
    "the story's background, key developments, and current status."
)

_EXTRACT_SYSTEM = (
    "You are an evidence analyst. Given numbered retrieved documents, extract:\n"
    "1. Key factual claims — reference documents by their [index].\n"
    "2. Timeline events with the most specific date you can determine.\n"
    "   If a date is approximate, prefix with '~' (e.g. '~2025-03').\n"
    "3. The likely rumor origin (earliest source).\n"
    "Return ONLY a JSON object."
)

_VERDICT_SYSTEM = (
    "You are a professional fact-checker. Assess the overall verdict based "
    "on the claims and evidence quality provided."
)

_STORY_SUMMARY_SYSTEM = (
    "You are a senior investigative journalist. Given a research question, "
    "a list of verified claims (with confidence scores), a chronological "
    "timeline, and a verdict, write a clear and concise narrative summary "
    "of the full story.\n\n"
    "Guidelines:\n"
    "- Start with one sentence stating what the story is about.\n"
    "- Cover the key developments in chronological order.\n"
    "- Note where evidence is strong and where gaps remain.\n"
    "- End with the current status and verdict.\n"
    "- Use neutral, factual language. 150-300 words.\n"
    "- Do NOT use markdown headings — just flowing paragraphs."
)

_SUFFICIENCY_SYSTEM = (
    "You are a research quality auditor. Given a research question, the "
    "claims and evidence gathered so far, and numeric coverage metrics, "
    "decide whether the collected evidence is SUFFICIENT to produce a "
    "well-supported answer, or whether another round of research is needed.\n\n"
    "Evaluation criteria (rate each 0-10):\n"
    "1. **Source diversity** — How many independent sources back the key claims?\n"
    "2. **Claim corroboration** — Are the major claims confirmed by ≥2 sources?\n"
    "3. **Contradiction resolution** — Are contradictions addressed or at least noted?\n"
    "4. **Temporal coverage** — Does the timeline cover the full span of the story?\n"
    "5. **Confidence level** — Is the overall confidence high enough for a verdict?\n\n"
    'Return JSON: {"sufficient": true/false, "overall_score": 0-10, '
    '"reason": "one sentence", "weakest_area": "which criterion is '
    'weakest and what query might help"}'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_verdict(value: str | None) -> Verdict:
    if value in {"supported", "likely_false", "unverified"}:
        return cast(Verdict, value)
    return "unverified"


def _truncate_docs(docs: list[RetrievedDocument], max_chars: int = 12000) -> str:
    """Build a text block of document summaries for LLM prompts."""
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(docs, 1):
        entry = (
            f"[{i}] {doc.title}\n"
            f"URL: {doc.url}\n"
            f"Date: {doc.published_at or 'unknown'}\n"
            f"Content: {doc.content[:600]}\n"
        )
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


def _heuristic_queries(query: str, iteration: int) -> list[str]:
    """Fallback query expansion without LLM."""
    if iteration <= 0:
        return [query]
    if iteration == 1:
        return [f"{query} timeline origin first report", f"{query} fact check"]
    return [f"{query} debunk evidence contradiction", f"{query} latest update"]


# ---------------------------------------------------------------------------
# LLM helpers (graceful fallback to heuristics on any failure)
# ---------------------------------------------------------------------------


def _llm_plan(
    llm: LLMClient,
    query: str,
    docs: list[RetrievedDocument],
) -> list[str]:
    evidence_summary = _truncate_docs(docs, max_chars=4000) if docs else "None yet."
    try:
        data = llm.chat_json(
            system=_PLAN_SYSTEM,
            user=(
                f"Research question: {query}\n\n"
                f"Evidence collected so far:\n{evidence_summary}\n\n"
                'Return JSON: {{"queries": ["q1", "q2", ...]}}'
            ),
        )
        queries = data.get("queries", [query])
        return [str(q) for q in queries][:4] or [query]
    except Exception:
        return [query]


def _llm_extract(
    llm: LLMClient,
    query: str,
    docs: list[RetrievedDocument],
) -> tuple[list[Claim], list[TimelineEvent], RumorOrigin]:
    """Use LLM to extract claims, timeline, and rumor origin from docs."""
    doc_text = _truncate_docs(docs)
    prompt = (
        f"Research question: {query}\n\n"
        f"Documents:\n{doc_text}\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "claims": [\n'
        '    {"text": "...", "category": "timeline|causal|rumor",\n'
        '     "confidence": 0.0-1.0, "source_doc_indices": [1],\n'
        '     "contradicting_doc_indices": []}\n'
        "  ],\n"
        '  "timeline": [\n'
        '    {"date": "YYYY-MM-DD or ~YYYY-MM", "summary": "...",\n'
        '     "source_doc_indices": [1]}\n'
        "  ],\n"
        '  "rumor_origin": {\n'
        '    "first_seen_date": "...", "first_seen_source": "...",\n'
        '    "first_seen_url": "...", "provenance_confidence": 0.0-1.0\n'
        "  }\n"
        "}"
    )
    try:
        data = llm.chat_json(system=_EXTRACT_SYSTEM, user=prompt)
    except Exception:
        return (
            score_claims(extract_claims(docs)),
            build_timeline(docs),
            detect_rumor_origin(docs),
        )

    # --- parse claims with document-index evidence linking ---
    claims: list[Claim] = []
    for c in data.get("claims", []):
        category_raw = c.get("category", "timeline")
        if category_raw not in ("timeline", "causal", "rumor"):
            category_raw = "timeline"

        evidence: list[Evidence] = []
        for idx in c.get("source_doc_indices", []):
            if isinstance(idx, int) and 1 <= idx <= len(docs):
                d = docs[idx - 1]
                evidence.append(
                    Evidence(
                        source_id=d.source_id,
                        url=d.url,
                        title=d.title,
                        excerpt=d.content[:200],
                        published_at=d.published_at,
                    )
                )
        for idx in c.get("contradicting_doc_indices", []):
            if isinstance(idx, int) and 1 <= idx <= len(docs):
                d = docs[idx - 1]
                evidence.append(
                    Evidence(
                        source_id=d.source_id,
                        url=d.url,
                        title=d.title,
                        excerpt=d.content[:200],
                        published_at=d.published_at,
                    )
                )

        claims.append(
            Claim(
                text=c.get("text", ""),
                category=category_raw,
                confidence=max(0.0, min(1.0, float(c.get("confidence", 0.5)))),
                support_count=len(c.get("source_doc_indices", [])),
                contradiction_count=len(c.get("contradicting_doc_indices", [])),
                evidence=evidence,
            )
        )

    # --- parse timeline ---
    timeline: list[TimelineEvent] = []
    for t in data.get("timeline", []):
        source_urls: list[str] = []
        for idx in t.get("source_doc_indices", []):
            if isinstance(idx, int) and 1 <= idx <= len(docs):
                source_urls.append(docs[idx - 1].url)
        timeline.append(
            TimelineEvent(
                timestamp=t.get("date"),
                summary=t.get("summary", ""),
                source_urls=source_urls,
            )
        )

    # --- parse rumor origin ---
    ro = data.get("rumor_origin", {})
    rumor_origin = RumorOrigin(
        first_seen_at=ro.get("first_seen_date"),
        first_seen_url=ro.get("first_seen_url"),
        first_seen_source=ro.get("first_seen_source"),
        provenance_confidence=max(
            0.0, min(1.0, float(ro.get("provenance_confidence", 0.5)))
        ),
    )

    return claims, timeline, rumor_origin


def _llm_verdict(
    llm: LLMClient,
    query: str,
    claims: list[Claim],
) -> tuple[Verdict, float, list[str]]:
    """Use LLM to determine verdict from claims."""
    claims_text = "\n".join(
        f"- {c.text} (confidence={c.confidence:.2f}, "
        f"support={c.support_count}, contradictions={c.contradiction_count})"
        for c in claims[:15]
    )
    try:
        data = llm.chat_json(
            system=_VERDICT_SYSTEM,
            user=(
                f"Question: {query}\n\nClaims:\n{claims_text}\n\n"
                "Return JSON: "
                '{"verdict": "supported|likely_false|unverified", '
                '"confidence": 0.0-1.0, "uncertainty_notes": [...]}'
            ),
        )
    except Exception:
        return decide_verdict(claims)

    verdict = _coerce_verdict(data.get("verdict"))
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    notes = [str(n) for n in data.get("uncertainty_notes", [])]
    return verdict, confidence, notes


def _compute_coverage_metrics(
    docs: list[RetrievedDocument],
    claims: list[Claim],
) -> dict[str, Any]:
    """Compute numeric evidence-quality metrics for the sufficiency check."""
    unique_sources = {d.source_id for d in docs}
    corroborated = sum(1 for c in claims if c.support_count >= 2)
    contradicted = sum(1 for c in claims if c.contradiction_count > 0)
    with_dates = sum(1 for d in docs if d.published_at)
    avg_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
    return {
        "total_documents": len(docs),
        "unique_sources": len(unique_sources),
        "total_claims": len(claims),
        "corroborated_claims": corroborated,
        "contradicted_claims": contradicted,
        "docs_with_dates": with_dates,
        "avg_claim_confidence": round(avg_confidence, 3),
    }


def _llm_sufficiency(
    llm: LLMClient,
    query: str,
    docs: list[RetrievedDocument],
    claims: list[Claim],
    confidence: float,
) -> tuple[bool, float, str]:
    """Ask the LLM whether the evidence collected is sufficient.

    Returns (sufficient, score_0_to_10, reason).
    """
    metrics = _compute_coverage_metrics(docs, claims)
    claims_text = "\n".join(
        f"- {c.text} (conf={c.confidence:.2f}, "
        f"support={c.support_count}, contradictions={c.contradiction_count})"
        for c in claims[:12]
    )
    try:
        data = llm.chat_json(
            system=_SUFFICIENCY_SYSTEM,
            user=(
                f"Research question: {query}\n\n"
                f"Coverage metrics: {metrics}\n"
                f"Current verdict confidence: {confidence:.2f}\n\n"
                f"Claims so far:\n{claims_text}\n\n"
                "Is the evidence sufficient?"
            ),
        )
    except Exception:
        return _heuristic_sufficiency(metrics, confidence)

    sufficient = bool(data.get("sufficient", False))
    score = max(0.0, min(10.0, float(data.get("overall_score", 5.0))))
    reason = str(data.get("reason", ""))
    return sufficient, score, reason


def _heuristic_sufficiency(
    metrics: dict[str, Any],
    confidence: float,
) -> tuple[bool, float, str]:
    """Fallback rule-based sufficiency check when no LLM is available."""
    score = 0.0
    # Source diversity: up to 3 pts
    score += min(3.0, metrics["unique_sources"] / 2.0)
    # Corroboration: up to 3 pts
    if metrics["total_claims"] > 0:
        ratio = metrics["corroborated_claims"] / metrics["total_claims"]
        score += ratio * 3.0
    # Confidence: up to 2 pts
    score += confidence * 2.0
    # Temporal coverage: up to 2 pts
    if metrics["total_documents"] > 0:
        date_ratio = metrics["docs_with_dates"] / metrics["total_documents"]
        score += date_ratio * 2.0

    sufficient = score >= 6.5
    reason = f"Heuristic score {score:.1f}/10"
    return sufficient, round(score, 1), reason


def _build_story_summary(
    llm: LLMClient | None,
    query: str,
    claims: list[Claim],
    timeline: list[TimelineEvent],
    verdict: Verdict,
    confidence: float,
) -> str:
    """Produce a narrative summary of the full story using the LLM."""
    if not (llm and llm.available):
        return ""

    claims_text = "\n".join(
        f"- {c.text} (confidence {c.confidence:.2f})" for c in claims[:12]
    )
    timeline_text = "\n".join(
        f"- {t.timestamp or '?'}: {t.summary}" for t in timeline[:15]
    )
    try:
        return llm.chat(
            system=_STORY_SUMMARY_SYSTEM,
            user=(
                f"Research question: {query}\n\n"
                f"Verdict: {verdict} (confidence {confidence:.0%})\n\n"
                f"Key claims:\n{claims_text}\n\n"
                f"Timeline:\n{timeline_text}"
            ),
        ).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_investigation_graph(
    retriever: Retriever,
    llm: LLMClient | None = None,
) -> Any:
    """Build a LangGraph investigation graph, optionally LLM-powered."""

    def plan_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        iteration = state.get("iteration", 0)

        if llm and llm.available:
            search_queries = _llm_plan(llm, query, state.get("documents", []))
        else:
            search_queries = _heuristic_queries(query, iteration)

        return {
            **state,
            "iteration": iteration,
            "search_queries": search_queries,
            "documents": state.get("documents", []),
        }

    def retrieve_node(state: AgentState) -> AgentState:
        next_iteration = state.get("iteration", 0) + 1
        queries = state.get("search_queries", [state.get("query", "")])

        new_docs: list[RetrievedDocument] = []
        for q in queries:
            new_docs.extend(retriever.retrieve(q, max_results=4))

        merged: dict[str, RetrievedDocument] = {
            doc.url: doc for doc in state.get("documents", [])
        }
        for doc in new_docs:
            if doc.url not in merged:
                merged[doc.url] = doc

        return {
            **state,
            "iteration": next_iteration,
            "documents": list(merged.values()),
        }

    def extract_node(state: AgentState) -> AgentState:
        docs = state.get("documents", [])
        if llm and llm.available:
            claims, timeline, origin = _llm_extract(
                llm,
                state.get("query", ""),
                docs,
            )
        else:
            claims = score_claims(extract_claims(docs))
            timeline = build_timeline(docs)
            origin = detect_rumor_origin(docs)
        return {
            **state,
            "claims": claims,
            "timeline": timeline,
            "rumor_origin": origin,
        }

    def verify_node(state: AgentState) -> AgentState:
        claims = state.get("claims", [])
        if llm and llm.available:
            verdict, confidence, notes = _llm_verdict(
                llm,
                state.get("query", ""),
                claims,
            )
        else:
            verdict, confidence, notes = decide_verdict(claims)

        # --- Sufficiency evaluation ---
        docs = state.get("documents", [])
        iteration = state.get("iteration", 0)

        # Always run at least 1 full iteration before checking sufficiency
        if iteration < 1:
            sufficient, suf_score, suf_reason = False, 0.0, "First pass"
        elif llm and llm.available:
            sufficient, suf_score, suf_reason = _llm_sufficiency(
                llm,
                state.get("query", ""),
                docs,
                claims,
                confidence,
            )
        else:
            metrics = _compute_coverage_metrics(docs, claims)
            sufficient, suf_score, suf_reason = _heuristic_sufficiency(
                metrics,
                confidence,
            )

        return {
            **state,
            "verdict": verdict,
            "confidence": confidence,
            "uncertainty_notes": notes,
            "sufficient": sufficient,
            "sufficiency_score": suf_score,
            "sufficiency_reason": suf_reason,
        }

    def synthesize_node(state: AgentState) -> AgentState:
        key_claims = sorted(
            state.get("claims", []),
            key=lambda c: (c.confidence, c.support_count),
            reverse=True,
        )[:10]

        summary = _build_story_summary(
            llm,
            state.get("query", ""),
            key_claims,
            state.get("timeline", []),
            _coerce_verdict(state.get("verdict")),
            state.get("confidence", 0.0),
        )

        result = InvestigationResult(
            query=state.get("query", ""),
            verdict=_coerce_verdict(state.get("verdict")),
            confidence=state.get("confidence", 0.0),
            summary=summary,
            timeline=state.get("timeline", []),
            key_claims=key_claims,
            rumor_origin=state.get("rumor_origin", RumorOrigin()),
            uncertainty_notes=state.get("uncertainty_notes", []),
        )
        return {**state, "result": result}

    def continue_or_stop(state: AgentState) -> str:
        # Hard cap: always stop at max_iterations
        if state.get("iteration", 0) >= state.get("max_iterations", 5):
            return "synthesize"
        # Dynamic: stop early when evidence is sufficient
        if state.get("sufficient", False):
            return "synthesize"
        return "plan"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("extract", extract_node)
    graph.add_node("verify", verify_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "verify")
    graph.add_conditional_edges(
        "verify",
        continue_or_stop,
        {"plan": "plan", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


_SUMMARIZE_ARTICLE_SYSTEM = (
    "You are a headline summariser. Given the title and opening text of a "
    "news article, produce a single concise research question that captures "
    "the core event or claim. Keep it under 30 words."
)


def _article_to_query(llm: LLMClient | None, doc: RetrievedDocument) -> str:
    """Derive a search-friendly research question from a seed article."""
    if llm and llm.available:
        try:
            return (
                llm.chat(
                    system=_SUMMARIZE_ARTICLE_SYSTEM,
                    user=f"Title: {doc.title}\n\nText: {doc.content[:1500]}",
                )
                .strip()
                .strip('"')
            )
        except Exception:
            pass
    # Fallback: use the article title directly
    return doc.title


def investigate(
    query: str,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    max_iterations: int = 5,
    seed_documents: list[RetrievedDocument] | None = None,
) -> InvestigationResult:
    """Run a full investigation and return structured results."""
    active_retriever = retriever or NullRetriever()
    graph = build_investigation_graph(active_retriever, llm=llm)

    final_state = graph.invoke(
        {
            "query": query,
            "max_iterations": max_iterations,
            "iteration": 0,
            "documents": seed_documents or [],
            "claims": [],
            "timeline": [],
            "uncertainty_notes": [],
        }
    )
    return final_state["result"]


# Human-readable labels for each graph node
_NODE_LABELS: dict[str, str] = {
    "plan": "Planning search queries",
    "retrieve": "Retrieving documents from the web",
    "extract": "Extracting claims and building timeline",
    "verify": "Verifying claims and assessing verdict",
    "synthesize": "Synthesizing final report",
}


class ProgressEvent:
    """A single progress tick emitted during streaming investigation."""

    __slots__ = ("node", "label", "iteration", "detail", "state")

    def __init__(
        self,
        node: str,
        label: str,
        iteration: int,
        detail: str,
        state: AgentState,
    ) -> None:
        self.node = node
        self.label = label
        self.iteration = iteration
        self.detail = detail
        self.state = state


def _describe_step(node: str, state: AgentState) -> str:
    """Build a short human-readable detail string for a completed node."""
    iteration = state.get("iteration", 0)
    max_it = state.get("max_iterations", 5)

    if node == "plan":
        queries = state.get("search_queries", [])
        n = len(queries)
        return f"Generated {n} search {'query' if n == 1 else 'queries'}"
    if node == "retrieve":
        n = len(state.get("documents", []))
        return f"Round {iteration}/{max_it} \u2014 {n} documents collected so far"
    if node == "extract":
        nc = len(state.get("claims", []))
        nt = len(state.get("timeline", []))
        return f"Found {nc} claims, {nt} timeline events"
    if node == "verify":
        v = state.get("verdict", "unverified")
        c = state.get("confidence", 0.0)
        suf = state.get("sufficient", False)
        suf_score = state.get("sufficiency_score", 0.0)
        suf_reason = state.get("sufficiency_reason", "")
        status = "sufficient \u2714" if suf else "need more evidence"
        parts = f"Verdict: {v} ({c:.0%}) \u2014 {status} (score {suf_score:.1f}/10)"
        if suf_reason:
            parts += f" \u2014 {suf_reason}"
        return parts
    if node == "synthesize":
        return "Done"
    return ""


def investigate_stream(
    query: str,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    max_iterations: int = 5,
    seed_documents: list[RetrievedDocument] | None = None,
) -> Generator[ProgressEvent, None, InvestigationResult]:
    """Stream progress events while investigating, then return the result.

    Usage::

        gen = investigate_stream(query, retriever, llm, max_iterations)
        try:
            while True:
                event = next(gen)
                # … update UI with event …
        except StopIteration as exc:
            result = exc.value
    """
    active_retriever = retriever or NullRetriever()
    graph = build_investigation_graph(active_retriever, llm=llm)

    init_state: AgentState = {
        "query": query,
        "max_iterations": max_iterations,
        "iteration": 0,
        "documents": seed_documents or [],
        "claims": [],
        "timeline": [],
        "uncertainty_notes": [],
    }

    final_state: AgentState = init_state
    for chunk in graph.stream(init_state):
        # LangGraph stream yields {node_name: updated_state_fragment}
        for node_name, node_output in chunk.items():
            if isinstance(node_output, dict):
                final_state = {**final_state, **node_output}
            label = _NODE_LABELS.get(node_name, node_name)
            detail = _describe_step(node_name, final_state)
            yield ProgressEvent(
                node=node_name,
                label=label,
                iteration=final_state.get("iteration", 0),
                detail=detail,
                state=final_state,
            )

    return final_state.get(
        "result",
        InvestigationResult(
            query=query,
            verdict="unverified",
            confidence=0.0,
        ),
    )
