"""BM25 稀疏检索：与稠密向量检索互补（关键词 / 专有名词）。"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def build_bm25_index(texts: List[str]) -> BM25Okapi:
    tokenized_corpus = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized_corpus)


def bm25_topk(bm25: BM25Okapi, query: str, k: int):
    """返回 (indices 降序, scores 对齐 indices)。"""
    scores = bm25.get_scores(tokenize(query))
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return [], np.array([])
    k = max(1, min(int(k), n))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist(), scores[idx]
