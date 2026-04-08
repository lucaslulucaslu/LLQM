from __future__ import annotations

import unittest

from llqm.modules.retriever import _parse_duckduckgo_links
from llqm.schemas.models import RetrievedDocument
from llqm.service.investigation_service import investigate


class FakeRetriever:
    def retrieve(self, query: str, max_results: int = 5) -> list[RetrievedDocument]:
        _ = (query, max_results)
        return [
            RetrievedDocument(
                source_id="social",
                url="https://x.com/post/1",
                title="Early rumor post",
                content="A rumor says Event A happened in secret.",
                published_at="2026-04-01T09:00:00",
            ),
            RetrievedDocument(
                source_id="reuters",
                url="https://www.reuters.com/world/event-a-update",
                title="Reuters update on Event A",
                content="Officials confirmed Event A details with evidence.",
                published_at="2026-04-02T10:00:00",
            ),
        ]


class AgentTests(unittest.TestCase):
    def test_investigation_returns_rumor_origin(self) -> None:
        result = investigate(
            "Did Event A really happen?", retriever=FakeRetriever(), max_iterations=1
        )

        self.assertEqual(result.rumor_origin.first_seen_url, "https://x.com/post/1")
        self.assertIn(result.verdict, {"supported", "unverified", "likely_false"})
        self.assertGreaterEqual(len(result.timeline), 1)

    def test_parse_duckduckgo_links(self) -> None:
        html = (
            '<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freuters.com%2Fstory"></a>'
            '<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fx.com%2Fpost%2F1"></a>'
        )
        links = _parse_duckduckgo_links(html, max_results=5)
        self.assertEqual(
            links,
            [
                "https://reuters.com/story",
                "https://x.com/post/1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
