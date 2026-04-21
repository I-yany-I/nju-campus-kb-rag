"""
RAG 新闻分类：稠密向量（Sentence-BERT + FAISS）+ 可选 BM25 混合（RRF）+ 可选 Cross-Encoder 重排；
语料支持按 train 子集建库（config/rag.yaml），便于对比检索泄漏与消融。
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.llm.shared_pipeline import load_shared_generation_pipeline
from src.paths import VECTOR_STORE_DIR, ensure_project_dirs
from src.rag.bm25_utils import build_bm25_index
from src.rag.hybrid_retrieval import hybrid_retrieve
from src.rag.settings import load_rag_settings, manifest_dict, manifests_match

LABEL_TEXT = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.index"
EMBEDDINGS_FILE = VECTOR_STORE_DIR / "embeddings.npy"
MANIFEST_FILE = VECTOR_STORE_DIR / "rag_manifest.json"
BM25_FILE = VECTOR_STORE_DIR / "bm25_index.pkl"


def clean_query(text: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", text or "")


def load_rag_train_corpus(settings: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[int]]:
    """按 config 从 AG News train 构造建库语料（可 shuffle + 子集）。"""
    settings = settings or load_rag_settings()
    dataset = load_dataset("ag_news")
    texts = list(dataset["train"]["text"])
    labels = list(dataset["train"]["label"])
    seed = settings.get("shuffle_seed")
    if seed is not None:
        rnd = random.Random(int(seed))
        order = list(range(len(texts)))
        rnd.shuffle(order)
        texts = [texts[i] for i in order]
        labels = [labels[i] for i in order]
    frac = float(settings.get("train_index_fraction", 1.0))
    frac = min(1.0, max(0.01, frac))
    n = max(1, int(len(texts) * frac))
    return texts[:n], labels[:n]


def load_data() -> Tuple[List[str], List[int]]:
    """兼容旧代码：等价于当前 rag.yaml 下的建库语料。"""
    return load_rag_train_corpus(load_rag_settings())


def load_llm():
    return load_shared_generation_pipeline()


def _read_manifest() -> Optional[Dict[str, Any]]:
    if not MANIFEST_FILE.exists():
        return None
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(obj: Dict[str, Any]) -> None:
    ensure_project_dirs()
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_vector_index(
    texts: List[str],
    labels: Optional[List[int]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, SentenceTransformer, Any]:
    """
    构建 / 加载 FAISS；hybrid 时同时构建 BM25 并落盘。
    返回 (index, embed_model, bm25_or_None)
    """
    ensure_project_dirs()
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    settings = settings or load_rag_settings()
    labels = labels if labels is not None else []

    new_manifest = manifest_dict(settings, len(texts))
    old_manifest = _read_manifest()
    hybrid = bool(settings.get("hybrid_enabled", True))

    cache_ok = (
        old_manifest is not None
        and manifests_match(old_manifest, new_manifest)
        and INDEX_FILE.exists()
        and EMBEDDINGS_FILE.exists()
        and (not hybrid or BM25_FILE.exists())
    )

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if cache_ok:
        print("Loading cached FAISS index (manifest matches)...")
        index = faiss.read_index(str(INDEX_FILE))
        bm25 = None
        if hybrid:
            with open(BM25_FILE, "rb") as f:
                bm25 = pickle.load(f)
        return index, embed_model, bm25

    print("Building new vector index (manifest changed or cache missing)...")
    for p in (INDEX_FILE, EMBEDDINGS_FILE, MANIFEST_FILE, BM25_FILE):
        if p.exists():
            try:
                os.remove(p)
            except OSError:
                pass

    embeddings = embed_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    np.save(str(EMBEDDINGS_FILE), embeddings)

    bm25 = None
    if hybrid:
        print("Building BM25 index...")
        bm25 = build_bm25_index(texts)
        with open(BM25_FILE, "wb") as f:
            pickle.dump(bm25, f)

    _write_manifest(new_manifest)
    print("Vector index + manifest saved.")
    return index, embed_model, bm25


def weighted_vote(indices: List[int], scores, labels: List[int]):
    vote_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    sc = np.asarray(scores, dtype=np.float64).reshape(-1)
    for j, idx in enumerate(indices):
        if idx < 0 or idx >= len(labels):
            continue
        label = int(labels[idx])
        w = float(sc[j]) if j < len(sc) else 0.0
        vote_scores[label] += max(w, 0.0)
    best_label = max(vote_scores, key=vote_scores.get)
    return best_label, vote_scores


def rag_classify(
    query: str,
    texts: List[str],
    labels: List[int],
    index,
    embed_model,
    llm,
    bm25=None,
    settings: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    settings = settings or load_rag_settings()
    query = clean_query(query)

    similar_indices, similar_scores = hybrid_retrieve(
        query, index, embed_model, bm25, texts, settings
    )
    if not similar_indices:
        return "-1", []

    voted_label, vote_scores = weighted_vote(similar_indices, similar_scores, labels)

    context = ""
    examples: List[str] = []
    for i in similar_indices[:5]:
        if 0 <= i < len(texts):
            lbl = int(labels[i])
            context += f"Example: {texts[i]} -> {lbl} ({LABEL_TEXT[lbl]})\n"
            examples.append(texts[i])

    prompt = f"""
You are an expert news classifier.

Categories:
0 = World (politics, international relations, conflicts, government)
1 = Sports (games, teams, athletes, competitions)
2 = Business (companies, finance, markets, investments, corporate expansion)
3 = Sci/Tech (technology, science, software, AI, gadgets)

Examples:
{context}

News:
{query}

Rules:
- Focus on the main topic.
- Company expansion or investments → Business
- Scientific discovery or technology → Sci/Tech

Return ONLY one number: 0, 1, 2, or 3.
Do not output any explanation.

Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=5,
        do_sample=False,
        return_full_text=False,
    )
    output = result[0]["generated_text"].strip()
    match = re.search(r"[0-3]", output)
    llm_label = int(match.group(0)) if match else -1

    sorted_votes = sorted(vote_scores.values(), reverse=True)
    if len(sorted_votes) < 2:
        margin = 1.0
    else:
        margin = sorted_votes[0] - sorted_votes[1]
    if llm_label in [0, 1, 2, 3] and margin < 0.15:
        final_label = llm_label
    else:
        final_label = voted_label

    return str(final_label), examples


if __name__ == "__main__":
    print("Loading dataset (RAG corpus per config/rag.yaml)...")
    cfg = load_rag_settings()
    texts, labels = load_rag_train_corpus(cfg)
    print(f"Corpus size: {len(texts)} (fraction={cfg['train_index_fraction']}, hybrid={cfg['hybrid_enabled']})")

    print("Building / loading vector database...")
    index, embed_model, bm25 = build_vector_index(texts, labels, cfg)

    print("Loading LLM...")
    llm = load_llm()

    query = "Amazon announced plans to open three new distribution centers in Europe."
    label, examples = rag_classify(query, texts, labels, index, embed_model, llm, bm25=bm25, settings=cfg)
    print("\nPrediction:", label)
    print("\nSimilar News:")
    for e in examples:
        print("-", e[:120], "...")
