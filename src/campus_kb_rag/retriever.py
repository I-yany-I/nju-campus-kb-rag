"""Hybrid retrieval for campus knowledge-base chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.campus_kb_rag.documents import KBChunk, chunk_documents, load_documents


def _tokenize(text: str) -> List[str]:
    lowered = (text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    cjk_bigrams = [lowered[i : i + 2] for i in range(len(lowered) - 1)]
    cjk_bigrams = [t for t in cjk_bigrams if re.search(r"[\u4e00-\u9fff]", t)]
    return ascii_tokens + cjk_chars + cjk_bigrams


def _rrf(ranked_lists: List[List[int]], rrf_k: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            if idx < 0:
                continue
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1.0)
    return scores


class CampusKBRetriever:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        kb_cfg = config["knowledge_base"]
        idx_cfg = config["index"]
        ret_cfg = config["retrieval"]

        self.kb_path = Path(kb_cfg["_resolved_path"])
        self.index_dir = Path(idx_cfg["_resolved_dir"])
        self.faiss_path = Path(idx_cfg["_resolved_faiss_path"])
        self.metadata_path = Path(idx_cfg["_resolved_metadata_path"])
        self.embedding_model_name = ret_cfg["embedding_model"]
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.chunks: List[KBChunk] = []
        self.index = None
        self.bm25 = None
        self._cross_encoder = None

    def build(self, force: bool = False) -> None:
        if not force and self.faiss_path.exists() and self.metadata_path.exists():
            self.load()
            return

        docs = load_documents(self.kb_path)
        kb_cfg = self.config["knowledge_base"]
        self.chunks = chunk_documents(
            docs,
            chunk_size=int(kb_cfg.get("chunk_size", 420)),
            chunk_overlap=int(kb_cfg.get("chunk_overlap", 80)),
        )
        if not self.chunks:
            raise ValueError(f"No chunks loaded from {self.kb_path}")

        embeddings = self.embedder.encode(
            [self._embed_text(c) for c in self.chunks],
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25Okapi([_tokenize(self._embed_text(c)) for c in self.chunks])

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.faiss_path))
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.faiss_path.exists() or not self.metadata_path.exists():
            self.build(force=True)
            return
        self.index = faiss.read_index(str(self.faiss_path))
        with self.metadata_path.open("r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [KBChunk(**item) for item in raw_chunks]
        self.bm25 = BM25Okapi([_tokenize(self._embed_text(c)) for c in self.chunks])

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            self.load()

        ret_cfg = self.config["retrieval"]
        dense_k = int(ret_cfg.get("dense_top_k", 12))
        bm25_k = int(ret_cfg.get("bm25_top_k", 12))
        final_k = int(top_k or ret_cfg.get("final_top_k", 5))

        dense_ids, dense_scores = self._dense_search(query, dense_k)
        if ret_cfg.get("hybrid_enabled", True):
            bm25_ids = self._bm25_search(query, bm25_k)
            fused = _rrf([dense_ids, bm25_ids], int(ret_cfg.get("rrf_k", 60)))
            candidate_ids = sorted(fused, key=lambda idx: -fused[idx])[: max(final_k, bm25_k)]
            scores = {idx: fused[idx] for idx in candidate_ids}
        else:
            candidate_ids = dense_ids
            scores = {idx: float(score) for idx, score in zip(dense_ids, dense_scores)}

        candidates = [self._result(idx, scores.get(idx, 0.0)) for idx in candidate_ids]
        ce_cfg = ret_cfg.get("cross_encoder", {})
        if ce_cfg.get("enabled", False) and candidates:
            candidates = self._rerank_cross_encoder(query, candidates, ce_cfg)
        return candidates[:final_k]

    def _dense_search(self, query: str, k: int) -> tuple[List[int], List[float]]:
        assert self.index is not None
        k = max(1, min(k, self.index.ntotal))
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k)
        ids = [int(i) for i in indices[0] if i >= 0]
        scores = [float(s) for s in distances[0][: len(ids)]]
        return ids, scores

    def _bm25_search(self, query: str, k: int) -> List[int]:
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float64)
        if scores.size == 0:
            return []
        k = max(1, min(k, scores.size))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [int(i) for i in idx]

    def _rerank_cross_encoder(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        ce_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        from sentence_transformers import CrossEncoder

        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(str(ce_cfg["model_name"]))
        pool = candidates[: int(ce_cfg.get("rerank_pool", len(candidates)))]
        pairs = [(query, item["text"][:1600]) for item in pool]
        ce_scores = self._cross_encoder.predict(pairs)
        for item, score in zip(pool, ce_scores):
            item["cross_encoder_score"] = float(score)
            item["score"] = float(score)
        pool.sort(key=lambda x: x.get("cross_encoder_score", x.get("score", 0.0)), reverse=True)
        return pool + candidates[len(pool) :]

    def _result(self, idx: int, score: float) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        item = chunk.to_dict()
        item["score"] = float(score)
        return item

    @staticmethod
    def _embed_text(chunk: KBChunk) -> str:
        tags = " ".join(chunk.tags or [])
        return f"{chunk.title} {chunk.department} {tags}\n{chunk.text}"
