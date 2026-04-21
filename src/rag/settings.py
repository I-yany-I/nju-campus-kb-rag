"""
加载 config/rag.yaml，供 RAG 建库与检索使用。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.paths import PROJECT_ROOT

DEFAULT_RAG_YAML = PROJECT_ROOT / "config" / "rag.yaml"


def _defaults() -> Dict[str, Any]:
    return {
        "hybrid_enabled": True,
        "bm25_topk": 50,
        "dense_topk": 50,
        "rrf_k": 60,
        "merged_topk": 32,
        "cross_encoder_enabled": False,
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross_encoder_rerank_pool": 24,
        "cross_encoder_vote_topk": 8,
        "train_index_fraction": 1.0,
        "shuffle_seed": 42,
        "prefer_json_output": True,
        "_raw": {},
    }


def load_rag_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """返回扁平化配置 dict，便于 rag_pipeline 读取。"""
    p = path or DEFAULT_RAG_YAML
    if not p.exists():
        return _defaults()

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    h = raw.get("hybrid") or {}
    ce = raw.get("cross_encoder") or {}
    c = raw.get("corpus") or {}
    pr = raw.get("prompt") or {}
    d = _defaults()
    d.update(
        {
            "hybrid_enabled": bool(h.get("enabled", d["hybrid_enabled"])),
            "bm25_topk": int(h.get("bm25_topk", d["bm25_topk"])),
            "dense_topk": int(h.get("dense_topk", d["dense_topk"])),
            "rrf_k": int(h.get("rrf_k", d["rrf_k"])),
            "merged_topk": int(h.get("merged_topk", d["merged_topk"])),
            "cross_encoder_enabled": bool(ce.get("enabled", d["cross_encoder_enabled"])),
            "cross_encoder_model": str(ce.get("model_name", d["cross_encoder_model"])),
            "cross_encoder_rerank_pool": int(ce.get("rerank_pool", d["cross_encoder_rerank_pool"])),
            "cross_encoder_vote_topk": int(ce.get("vote_topk", d["cross_encoder_vote_topk"])),
            "train_index_fraction": float(c.get("train_index_fraction", d["train_index_fraction"])),
            "shuffle_seed": c.get("shuffle_seed", d["shuffle_seed"]),
            "prefer_json_output": bool(pr.get("prefer_json_output", d["prefer_json_output"])),
            "_raw": deepcopy(raw),
        }
    )
    return d


def manifest_dict(settings: Dict[str, Any], corpus_size: int) -> Dict[str, Any]:
    return {
        "hybrid_enabled": settings["hybrid_enabled"],
        "train_index_fraction": settings["train_index_fraction"],
        "shuffle_seed": settings["shuffle_seed"],
        "corpus_size": corpus_size,
        "cross_encoder_enabled": settings["cross_encoder_enabled"],
        "embed_model": "all-MiniLM-L6-v2",
    }


def manifests_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = (
        "hybrid_enabled",
        "train_index_fraction",
        "shuffle_seed",
        "corpus_size",
        "cross_encoder_enabled",
        "embed_model",
    )
    return all(a.get(k) == b.get(k) for k in keys)
