# LLQM

Timeline-first investigation agent for both news events and rumors.

## What This MVP Does

- Uses a LangGraph ReAct-style loop to retrieve evidence and synthesize findings.
- Produces timeline-oriented output with citations and uncertainty notes.
- Includes rumor-origin tracing (earliest discoverable source and timestamp).
- Emits a verdict: `supported`, `likely_false`, or `unverified`.
- Supports a live retriever that searches web sources for both news and rumor discussions.
- Uses `SERPER_API_KEY` automatically (when present in `.env`) for better real-time research.

## Current Status

This is an initial scaffold. The default retriever is a `NullRetriever`, and live retrieval is enabled with `--live`. The graph, models, verification logic, and CLI/UI are implemented.

## Run

```bash
pip install -e .
llqm "Did rumor X originate from source Y?"
llqm "Did rumor X originate from source Y?" --live
llqm-ui
```

Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=...
SERPER_API_KEY=...
```

## Output Contract

The result includes:

- `timeline`: ordered event summaries
- `key_claims`: extracted claims with evidence and confidence
- `rumor_origin`: first-seen info and provenance confidence
- `verdict`: supported / likely_false / unverified
- `uncertainty_notes`: caveats about source coverage or contradictions

## Next Implementation Steps

- Add API-backed retrieval adapters (NewsAPI/GDELT/etc.) to improve recency and source diversity.
- Add stronger contradiction detection and claim normalization.
- Add history storage and comparison mode for repeated investigations.
