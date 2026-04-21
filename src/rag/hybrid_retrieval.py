"""混合检索：稠密 FAISS + BM25，RRF 融合；可选 Cross-Encoder 重排。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.rag.bm25_utils import bm25_topk


def retrieve_dense(
    query: str,
    index: faiss.Index,
    embed_model,
    k: int,
) -> Tuple[List[int], np.ndarray]:
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    k = min(int(k), index.ntotal)
    if k <= 0:
        return [], np.array([])
    distances, indices = index.search(query_embedding, k)
    return indices[0].tolist(), distances[0]


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    rrf_k: int,
) -> Tuple[List[int], Dict[int, float]]:
    """RRF：多个排序列表融合为单一分数。"""
    scores: Dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            if idx < 0:
                continue
            scores[int(idx)] += 1.0 / (float(rrf_k) + float(rank) + 1.0)
    ordered = sorted(scores.keys(), key=lambda i: -scores[i])
    return ordered, dict(scores)


_ce_model = None
_ce_name = None


def _get_cross_encoder(model_name: str):
    global _ce_model, _ce_name
    if _ce_model is not None and _ce_name == model_name:
        return _ce_model
    from sentence_transformers import CrossEncoder

    _ce_model = CrossEncoder(model_name)
    _ce_name = model_name
    return _ce_model


def hybrid_retrieve(
    query: str,
    index: faiss.Index,
    embed_model,
    bm25,
    texts: List[str],
    settings: Dict[str, Any],
) -> Tuple[List[int], np.ndarray]:
    """
    返回用于 weighted_vote 的 (indices, scores)。
    scores 为融合分或 Cross-Encoder 分（越大越好）。
    """
    dense_k = int(settings["dense_topk"])
    merged_k = int(settings["merged_topk"])
    rrf_k = int(settings["rrf_k"])

    dense_idx, dense_sc = retrieve_dense(query, index, embed_model, dense_k)
    dense_ranked = [int(i) for i in dense_idx if i >= 0]

    if not settings.get("hybrid_enabled", True) or bm25 is None:
        return dense_ranked[:merged_k], np.asarray(dense_sc[:merged_k], dtype=np.float64)

    bm25_k = int(settings["bm25_topk"])
    bm25_idx, _bm25_sc = bm25_topk(bm25, query, bm25_k)
    bm25_ranked = [int(i) for i in bm25_idx]

    ordered, rrf_scores = reciprocal_rank_fusion([dense_ranked, bm25_ranked], rrf_k)
    merged_ids = ordered[:merged_k]
    rrf_vec = np.array([rrf_scores[i] for i in merged_ids], dtype=np.float64)

    if not settings.get("cross_encoder_enabled", False) or len(merged_ids) == 0:
        return merged_ids, rrf_vec

    pool_n = min(int(settings["cross_encoder_rerank_pool"]), len(merged_ids))
    pool = merged_ids[:pool_n]
    model_name = str(settings["cross_encoder_model"])
    ce = _get_cross_encoder(model_name)
    pairs = []
    for i in pool:
        t = texts[i] if 0 <= i < len(texts) else ""
        pairs.append((query, (t or "")[:2000]))
    ce_scores = np.asarray(ce.predict(pairs), dtype=np.float64)
    order = np.argsort(-ce_scores)
    vote_topk = int(settings["cross_encoder_vote_topk"])
    top = order[:vote_topk]
    final_ids = [pool[j] for j in top]
    raw = ce_scores[top].astype(np.float64)
    # Cross-Encoder  logits 可能为负，投票权重需非负
    final_scores = raw - float(np.min(raw)) + 1e-6
    return final_ids, final_scores
