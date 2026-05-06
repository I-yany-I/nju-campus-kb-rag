"""End-to-end campus KB RAG pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from src.campus_kb_rag.config import load_config, resolve_path
from src.campus_kb_rag.generator import CampusAnswerGenerator
from src.campus_kb_rag.retriever import CampusKBRetriever


class CampusKBRAG:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self._resolve_config_paths()
        self.retriever = CampusKBRetriever(self.config)
        self.generator = CampusAnswerGenerator(self.config)

    def build_index(self, force: bool = False) -> None:
        self.retriever.build(force=force)

    def ask(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        normalized = " ".join((query or "").split())
        if not normalized:
            return {"answer": "请输入具体的校园办事问题。", "citations": [], "retrieved": []}

        retrieved = self.retriever.search(normalized, top_k=top_k)
        evidence = self._filter_low_confidence(retrieved)
        answer = self.generator.generate(normalized, evidence)
        return {
            "query": normalized,
            "answer": answer,
            "citations": self._citations(evidence),
            "retrieved": retrieved,
        }

    def _filter_low_confidence(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not retrieved:
            return []
        prompt_cfg = self.config.get("prompt", {})
        refusal_doc_ids = set(prompt_cfg.get("refusal_doc_ids", []))
        if retrieved[0].get("doc_id") in refusal_doc_ids:
            return []
        threshold = float(prompt_cfg.get("refusal_threshold", 0.18))
        top_score = float(retrieved[0].get("score", 0.0))
        if top_score < threshold:
            return []
        return retrieved

    @staticmethod
    def _citations(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for i, item in enumerate(evidence, start=1):
            citations.append(
                {
                    "index": i,
                    "doc_id": item.get("doc_id"),
                    "title": item.get("title"),
                    "department": item.get("department"),
                    "source": item.get("source"),
                    "updated_at": item.get("updated_at"),
                    "score": item.get("score"),
                }
            )
        return citations

    def _resolve_config_paths(self) -> None:
        kb_cfg = self.config.setdefault("knowledge_base", {})
        idx_cfg = self.config.setdefault("index", {})
        kb_cfg["_resolved_path"] = str(resolve_path(kb_cfg["path"]))
        idx_cfg["_resolved_dir"] = str(resolve_path(idx_cfg["dir"]))
        idx_cfg["_resolved_faiss_path"] = str(resolve_path(idx_cfg["faiss_path"]))
        idx_cfg["_resolved_metadata_path"] = str(resolve_path(idx_cfg["metadata_path"]))
