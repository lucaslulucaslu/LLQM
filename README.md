# LLQM

LLQM (来龙去脉, Lái Lóng Qù Mài) — a timeline-first investigation agent that traces news events and rumors from origin to current status. Feed it a question or a news URL — it retrieves evidence from the web, extracts claims, builds a chronological timeline, and delivers a verdict with confidence scores and full citations.

## Features

- **LangGraph pipeline** — plan → retrieve → extract → verify → synthesize, with LLM-powered reasoning at every stage
- **Dynamic iteration** — the agent decides when it has enough evidence to stop (no fixed loop count); a configurable hard cap acts as a safety net
- **Dual input modes** — investigate a text question ("Did X originate from Y?") or paste a news article URL to build the full story
- **Hybrid source trust scoring** — 40+ known editorial sources, TLD-based rules (.gov/.edu), page-signal heuristics (schema.org, author tags, corrections policies), and LLM classification fallback for unknown domains
- **Narrative story summary** — LLM-generated overview of the full story arc, not just individual claims
- **Interactive web UI** — FastAPI + htmx with live SSE progress, vertical timeline, claim drill-down, source explorer, and dark/light theme toggle
- **Visual timeline** — alternating left/right timeline with month markers, corroboration indicators, source authority bars, and click-to-detail
- **Streaming progress** — real-time status updates via Server-Sent Events as each pipeline stage completes
- **Verdict system** — `supported`, `likely_false`, or `unverified` with weighted confidence scoring, uncertainty notes, and evidence links

## Quick Start

### Prerequisites

- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- (Optional) A [Serper API key](https://serper.dev/) for higher-quality web search; falls back to DuckDuckGo if absent

### Install

```bash
uv sync
```

### Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini        # optional, defaults to gpt-4o-mini
SERPER_API_KEY=...               # optional, enables Serper search
```

### Run

**Web UI (FastAPI):**

```bash
llqm-ui
# → opens http://127.0.0.1:8000

# Custom port
llqm-ui --port 8001
```

**CLI:**

```bash
# Investigate a rumor or question
llqm "Is the claim about X true?" --live

# Build a timeline from a news article URL
llqm --url https://www.bbc.com/news/article-id --live

# Adjust the safety cap on research rounds (default: 5)
llqm "What happened with X?" --live --max-iterations 3
```

## Architecture

```
src/llqm/
├── cli.py                  # CLI entry point
├── modules/
│   ├── llm.py              # OpenAI chat-completions client (httpx, no SDK)
│   └── retriever.py        # Serper, DuckDuckGo, fallback composition
├── schemas/
│   └── models.py           # Pydantic v2 data models
├── service/
│   ├── investigation_service.py   # LangGraph state machine
│   ├── timeline_service.py        # Chronological event builder
│   └── verification_service.py    # Claim scoring and verdict logic
├── web/
│   ├── app.py              # FastAPI application factory & entry point
│   ├── state.py            # Thread-safe in-memory investigation store
│   ├── helpers.py          # Shared helpers (enriched timeline JSON, trust)
│   ├── routes/
│   │   ├── home.py         # Landing page
│   │   ├── investigation.py # POST /investigate, GET /result/{id}
│   │   ├── stream.py       # SSE endpoint for live progress
│   │   └── partials.py     # htmx partial endpoints (timeline, claims, sources, events)
│   ├── templates/          # Jinja2 templates (base, result, partials/)
│   └── static/             # CSS (theme.css) and JS (app.js)
└── utils/
    ├── date_extractor.py   # Date normalization (ISO, relative, freetext)
    └── source_registry.py  # Hybrid trust scoring system
```

### Pipeline

```
+--------+   +----------+   +---------+   +----------+
|  Plan  |-->| Retrieve |-->| Extract |-->|  Verify  |
+--------+   +----------+   +---------+   +----+-----+
     ^                                         |
     |        need more evidence               | sufficient
     +-----------------------------------------+
                                               |
                                               v
                                        +-------------+
                                        | Synthesize  |--> Result
                                        +-------------+
```

| Node | Purpose |
|---|---|
| **Plan** | LLM generates 3–4 targeted search queries based on the question and evidence collected so far |
| **Retrieve** | Runs queries through Serper/DuckDuckGo, deduplicates results across rounds |
| **Extract** | LLM extracts claims (with evidence links), timeline events, and rumor origin from documents |
| **Verify** | LLM assesses verdict + confidence, then evaluates evidence sufficiency to decide: loop or stop |
| **Synthesize** | Ranks top claims, generates a narrative story summary, produces the final `InvestigationResult` |

### Evidence Sufficiency

After each verify step, the agent evaluates five criteria (scored 0–10):

1. **Source diversity** — independent sources backing key claims
2. **Claim corroboration** — major claims confirmed by ≥2 sources
3. **Contradiction resolution** — conflicts addressed or noted
4. **Temporal coverage** — timeline spans the full story
5. **Confidence level** — overall certainty in the verdict

If the score is high enough, research stops early. Otherwise it loops back to plan new queries. A hard cap (default 5 rounds) guarantees termination.

### Source Trust Scoring

Unknown sources are evaluated through a hybrid system:

| Layer | Example | Method |
|---|---|---|
| Known dictionary | reuters.com → 0.92 | Curated editorial scores for 40+ domains |
| TLD rules | cdc.gov → 0.88 | `.gov`, `.edu`, `.mil` get high trust |
| Page signals | `NewsArticle` JSON-LD → +0.08 | Schema.org markup, author tags, ethics policies |
| LLM fallback | unknown-site.com → classified | Only when heuristic confidence is low |
| Default | 0.50 | Neutral baseline |

Results are cached per domain to avoid redundant evaluation.

## Web UI

The web interface is built with **FastAPI + Jinja2 + htmx** (no build step, all assets via CDN):

- **Landing page** — enter a rumor/question or paste a URL, choose investigation mode
- **Live progress** — SSE-driven real-time updates as each pipeline stage runs
- **Verdict banner** — color-coded verdict with confidence meter, event/claim counts, rumor origin
- **Timeline tab** — alternating left/right vertical timeline with month markers, source domain labels, authority trust bars, and click-to-detail sidebar
- **Claims tab** — accordion with evidence drill-down, trust scores per source
- **Sources tab** — sortable table of all retrieved sources with domain trust scores
- **Dark / Light theme** — toggle with localStorage persistence

## Output

The `InvestigationResult` includes:

| Field | Description |
|---|---|
| `verdict` | `supported`, `likely_false`, or `unverified` |
| `confidence` | 0.0–1.0 overall confidence |
| `summary` | Narrative overview of the full story |
| `timeline` | Chronological events with dates and source URLs |
| `key_claims` | Top claims with evidence, support/contradiction counts, and confidence |
| `rumor_origin` | Earliest source, date, URL, and provenance confidence |
| `uncertainty_notes` | Caveats about evidence gaps or contradictions |

## Development

```bash
# Install dependencies
uv sync

# Run tests
python -m unittest discover -s tests -v

# Launch the UI in dev (with auto-reload)
uv run llqm-ui --reload
```

## License

MIT
