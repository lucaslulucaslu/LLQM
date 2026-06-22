#!/usr/bin/env python3
"""Test LLQM against 10 famous disinformation campaigns."""

import json
from pathlib import Path

from llqm.modules.llm import build_llm
from llqm.modules.retriever import build_retriever
from llqm.service.investigation_service import investigate

queries = [
    "Do vaccines cause autism?",
    "Does 5G spread COVID-19?",
    "Was the Moon landing a hoax?",
    "Is the Earth flat?",
    "Was the 2020 election stolen from Trump?",
    "Is Pizzagate a real conspiracy?",
    "Did Joe Biden fall during the 2023 campaign?",
    "Did hydroxychloroquine cure COVID-19?",
    "Did a UFO crash at Roswell in 1947?",
    "Is JFK still alive?",
]

llm = build_llm()
retriever = build_retriever(use_live=True)
results = []
results_path = Path(__file__).with_name("disinformation_results.json")

print("Starting disinformation tests...\n")

for i, query in enumerate(queries, 1):
    print(f"[{i}/10] Testing: {query}")
    try:
        result = investigate(
            query=query,
            max_iterations=2,
            retriever=retriever,
            llm=llm,
        )
        
        test_result = {
            "query": query,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "summary": result.summary[:250] if result.summary else "",
            "key_claims_count": len(result.key_claims),
            "evidence_sources": len(result.key_claims[0].evidence) if result.key_claims else 0,
        }
        
        results.append(test_result)
        print(f"  → Verdict: {test_result['verdict']}, Confidence: {test_result['confidence']:.2f}\n")
        
    except Exception as e:
        test_result = {
            "query": query,
            "error": str(e)[:200]
        }
        results.append(test_result)
        print(f"  → ERROR: {str(e)[:100]}\n")

# Write results to JSON
with results_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

success_count = sum(1 for r in results if "verdict" in r)
print(f"Tests completed: {success_count}/{len(queries)}")

for r in results:
    if "verdict" in r:
        print(f"\n{r['query']}")
        print(f"  Verdict: {r['verdict']} (confidence: {r['confidence']:.2f})")

print(f"\n\nDetailed results saved to {results_path}")
