"""Build the FAISS index for the campus KB RAG system."""

from __future__ import annotations

import argparse

from src.campus_kb_rag import CampusKBRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nanjing University campus KB index.")
    parser.add_argument("--config", default=None, help="主配置文件（可选；省略则用项目默认）")
    parser.add_argument("--force", action="store_true", help="Rebuild index even if cache exists.")
    args = parser.parse_args()

    rag = CampusKBRAG(config_path=args.config)
    rag.build_index(force=args.force)
    print("Campus KB index is ready.")


if __name__ == "__main__":
    main()
