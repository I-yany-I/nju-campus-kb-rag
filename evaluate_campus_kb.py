"""Evaluate retrieval and refusal behavior for the campus KB RAG system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import summarize
from src.campus_kb_rag import CampusKBRAG


SEED_QUESTIONS = [
    {
        "question": "校园网外怎么访问校内资源？",
        "expected_doc_ids": ["nju-it-vpn"],
        "should_refuse": False,
    },
    {
        "question": "统一身份认证密码忘记了怎么办？",
        "expected_doc_ids": ["nju-it-auth-password"],
        "should_refuse": False,
    },
    {
        "question": "成绩单和在读证明应该找哪个部门？",
        "expected_doc_ids": ["nju-academic-transcript"],
        "should_refuse": False,
    },
    {
        "question": "补考和重修需要看什么通知？",
        "expected_doc_ids": ["nju-academic-retake"],
        "should_refuse": False,
    },
    {
        "question": "宿舍电费怎么充值？",
        "expected_doc_ids": [],
        "should_refuse": True,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate campus KB RAG.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="artifacts/predictions/campus_kb_eval.json")
    args = parser.parse_args()

    rag = CampusKBRAG(config_path=args.config)
    rag.build_index(force=False)

    results = []
    for case in SEED_QUESTIONS:
        response = rag.ask(case["question"])
        results.append(
            {
                **case,
                "answer": response["answer"],
                "citations": response["citations"],
            }
        )

    summary = summarize(results)
    payload = {"summary": summary, "results": results}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation details to {output_path}")


if __name__ == "__main__":
    main()
