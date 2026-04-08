"""In-memory investigation store."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from llqm.schemas.models import InvestigationResult, RetrievedDocument


@dataclass
class StoredInvestigation:
    id: str
    query: str
    mode: str = "rumor"  # "rumor" or "url"
    status: str = "running"  # "running" | "done" | "error"
    progress: list[dict[str, str]] = field(default_factory=list)
    result: InvestigationResult | None = None
    documents: list[RetrievedDocument] = field(default_factory=list)
    error: str | None = None
    parent_id: str | None = None  # links follow-up to original investigation


_lock = threading.Lock()
_store: dict[str, StoredInvestigation] = {}


def create(inv: StoredInvestigation) -> None:
    with _lock:
        _store[inv.id] = inv


def get(inv_id: str) -> StoredInvestigation | None:
    with _lock:
        return _store.get(inv_id)


def update_progress(inv_id: str, node: str, label: str, detail: str) -> None:
    with _lock:
        inv = _store.get(inv_id)
        if inv:
            inv.progress.append({"node": node, "label": label, "detail": detail})


def finish(
    inv_id: str, result: InvestigationResult, documents: list[RetrievedDocument]
) -> None:
    with _lock:
        inv = _store.get(inv_id)
        if inv:
            inv.status = "done"
            inv.result = result
            inv.documents = documents


def fail(inv_id: str, error: str) -> None:
    with _lock:
        inv = _store.get(inv_id)
        if inv:
            inv.status = "error"
            inv.error = error
