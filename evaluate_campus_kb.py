"""Evaluate retrieval and refusal behavior for the campus KB RAG system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from evaluation.metrics import summarize
from src.campus_kb_rag import CampusKBRAG
from src.campus_kb_rag.config import resolve_path


DEFAULT_EVAL_QUESTIONS = "data/campus_kb/eval_questions.jsonl"


def load_eval_questions(path: str | Path) -> List[Dict]:
    resolved = resolve_path(path)
    questions = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions.append(
                {
                    "question": item["question"],
                    "expected_doc_ids": item.get("expected_doc_ids", []),
                    "should_refuse": item.get("should_refuse", False),
                    "category": item.get("category", ""),
                    "id": item.get("id", ""),
                }
            )
    return questions


def per_category_summary(results: List[Dict]) -> Dict:
    cats: Dict[str, List[Dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r)
    return {cat: summarize(items) for cat, items in cats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NJU campus KB RAG.")
    parser.add_argument("--config", default=None, help="Path to campus_kb.yaml")
    parser.add_argument(
        "--questions",
        default=DEFAULT_EVAL_QUESTIONS,
        help="Path to eval_questions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/predictions/campus_kb_eval.json",
        help="Path to write evaluation output JSON",
    )
    args = parser.parse_args()

    rag = CampusKBRAG(config_path=args.config)
    rag.build_index(force=False)

    questions = load_eval_questions(args.questions)
    print(f"Loaded {len(questions)} evaluation questions.")

    results = []
    for case in questions:
        response = rag.ask(case["question"])
        results.append(
            {
                "id": case.get("id", ""),
                "category": case.get("category", ""),
                "question": case["question"],
                "expected_doc_ids": case["expected_doc_ids"],
                "should_refuse": case["should_refuse"],
                "answer": response["answer"],
                "citations": response["citations"],
            }
        )

    overall = summarize(results)
    by_category = per_category_summary(results)

    payload = {
        "summary": overall,
        "by_category": by_category,
        "n_questions": len(results),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Overall ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== By Category ===")
    print(json.dumps(by_category, ensure_ascii=False, indent=2))
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
