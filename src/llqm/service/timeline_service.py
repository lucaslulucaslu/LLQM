from __future__ import annotations

from llqm.schemas.models import RetrievedDocument, TimelineEvent
from llqm.utils.date_extractor import date_sort_key


def build_timeline(documents: list[RetrievedDocument]) -> list[TimelineEvent]:
    """Create a simple chronology from retrieved documents."""
    sorted_docs = sorted(documents, key=lambda d: date_sort_key(d.published_at))
    timeline: list[TimelineEvent] = []
    for doc in sorted_docs:
        summary = doc.title.strip() or doc.content.strip()[:140]
        timeline.append(
            TimelineEvent(
                timestamp=doc.published_at,
                summary=summary,
                source_urls=[doc.url],
            )
        )
    return timeline
