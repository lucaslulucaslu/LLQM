from __future__ import annotations

import argparse
import json

from llqm.modules.llm import build_llm
from llqm.modules.retriever import build_retriever, fetch_article
from llqm.service.investigation_service import _article_to_query, investigate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLQM timeline-first investigation agent"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="News event or rumor to investigate",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="News article URL to build a full timeline from",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Safety cap on retrieval rounds (agent stops early when evidence is sufficient)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live web retrieval instead of NullRetriever",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.query and not args.url:
        print("Error: provide a query or --url")
        raise SystemExit(1)

    retriever = build_retriever(use_live=args.live or bool(args.url))
    llm = build_llm() if (args.live or args.url) else None

    seed_documents = []
    if args.url:
        article = fetch_article(args.url)
        if article is None:
            print(f"Error: could not fetch {args.url}")
            raise SystemExit(1)
        seed_documents = [article]
        query = args.query or _article_to_query(llm, article)
        print(f"Research question: {query}\n")
    else:
        query = args.query

    result = investigate(
        query=query,
        max_iterations=args.max_iterations,
        retriever=retriever,
        llm=llm,
        seed_documents=seed_documents,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
