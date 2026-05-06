"""Small evaluation helpers for citation-oriented campus KB RAG."""

from __future__ import annotations

from typing import Dict, Iterable, List


def citation_hit_rate(results: Iterable[Dict]) -> float:
    rows = list(results)
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        expected = set(row.get("expected_doc_ids", []))
        retrieved = {c.get("doc_id") for c in row.get("citations", [])}
        if expected and expected.intersection(retrieved):
            hits += 1
    return hits / len(rows)


def refusal_accuracy(results: Iterable[Dict]) -> float:
    rows = [r for r in results if r.get("should_refuse")]
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        answer = row.get("answer", "")
        citations = row.get("citations", [])
        if not citations and "没有足够依据" in answer:
            correct += 1
    return correct / len(rows)


def summarize(results: List[Dict]) -> Dict[str, float]:
    return {
        "citation_hit_rate": citation_hit_rate(results),
        "refusal_accuracy": refusal_accuracy(results),
        "n": float(len(results)),
    }
