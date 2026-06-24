"""Tests for RAG pipeline: refusal logic, citation formatting, end-to-end ask flow.

Run from project root:
    python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campus_kb_rag.pipeline import CampusKBRAG


# ---------------------------------------------------------------------------
# Refusal logic (_filter_low_confidence)
# ---------------------------------------------------------------------------

class TestRefusalLogic:
    @pytest.fixture
    def rag_with_config(self):
        """Create a CampusKBRAG with a specific config, mocking heavy deps."""
        config = {
            "knowledge_base": {"path": "data/kb.jsonl"},
            "index": {"dir": "vector_store/", "faiss_path": "faiss.index", "metadata_path": "chunks.json"},
            "retrieval": {
                "embedding_model": "test-model",
                "hybrid_enabled": True,
                "dense_top_k": 12,
                "bm25_top_k": 12,
                "final_top_k": 5,
                "rrf_k": 60,
                "cross_encoder": {"enabled": False},
            },
            "prompt": {
                "refusal_threshold": 0.18,
                "refusal_doc_ids": ["nju-support-unknown"],
            },
            "generation": {"backend": "extractive"},
            "index": {"dir": "vector_store/", "faiss_path": "faiss.index", "metadata_path": "chunks.json"},
        }
        # Avoid loading models by patching the retriever init
        with patch("src.campus_kb_rag.pipeline.CampusKBRetriever", autospec=True):
            with patch("src.campus_kb_rag.pipeline.CampusAnswerGenerator", autospec=True):
                rag = CampusKBRAG.__new__(CampusKBRAG)
                rag.config = config
                return rag

    def test_empty_retrieved_returns_empty(self, rag_with_config):
        result = rag_with_config._filter_low_confidence([])
        assert result == []

    def test_refusal_doc_id_triggers_rejection(self, rag_with_config):
        retrieved = [{"doc_id": "nju-support-unknown", "score": 0.85, "text": "..."}]
        result = rag_with_config._filter_low_confidence(retrieved)
        assert result == []

    def test_score_below_threshold_rejected(self, rag_with_config):
        retrieved = [{"doc_id": "nju-it-vpn", "score": 0.10, "text": "..."}]
        result = rag_with_config._filter_low_confidence(retrieved)
        assert result == []

    def test_score_above_threshold_accepted(self, rag_with_config):
        retrieved = [{"doc_id": "nju-it-vpn", "score": 0.50, "text": "..."}]
        result = rag_with_config._filter_low_confidence(retrieved)
        assert len(result) == 1
        assert result[0]["doc_id"] == "nju-it-vpn"

    def test_threshold_exactly_at_boundary(self, rag_with_config):
        # score == threshold (0.18): code uses '<' not '<=', so equal is accepted
        retrieved = [{"doc_id": "nju-it-vpn", "score": 0.18, "text": "..."}]
        result = rag_with_config._filter_low_confidence(retrieved)
        assert len(result) == 1  # 0.18 < 0.18 is False → accepted

    def test_refusal_doc_id_checked_before_score(self, rag_with_config):
        """Even with high score, refusal doc_id should trigger rejection."""
        retrieved = [{"doc_id": "nju-support-unknown", "score": 0.99, "text": "..."}]
        result = rag_with_config._filter_low_confidence(retrieved)
        assert result == []

    def test_different_refusal_doc_ids(self, rag_with_config):
        rag_with_config.config["prompt"]["refusal_doc_ids"] = ["custom-refusal", "fallback-doc"]
        retrieved = [{"doc_id": "custom-refusal", "score": 0.90, "text": "..."}]
        result = rag_with_config._filter_low_confidence(retrieved)
        assert result == []

    def test_no_prompt_config_defaults(self, rag_with_config):
        """Without prompt config keys, should still work with defaults."""
        rag_with_config.config.pop("prompt")
        retrieved = [{"doc_id": "some-doc", "score": 0.50, "text": "..."}]
        result = rag_with_config._filter_low_confidence(retrieved)
        # Default threshold is 0.18, score 0.50 > 0.18 → accepted
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

class TestCitations:
    def test_formats_single_evidence(self):
        evidence = [
            {
                "doc_id": "nju-it-vpn",
                "title": "校园VPN使用说明",
                "department": "信息化中心",
                "source": "https://example.com",
                "updated_at": "2026-05-01",
                "score": 0.85,
            }
        ]
        citations = CampusKBRAG._citations(evidence)
        assert len(citations) == 1
        assert citations[0]["index"] == 1
        assert citations[0]["doc_id"] == "nju-it-vpn"
        assert citations[0]["title"] == "校园VPN使用说明"
        assert citations[0]["score"] == 0.85

    def test_formats_multiple_evidence(self):
        evidence = [
            {"doc_id": "doc-a", "title": "A", "department": "", "source": "", "updated_at": "", "score": 0.9},
            {"doc_id": "doc-b", "title": "B", "department": "", "source": "", "updated_at": "", "score": 0.7},
            {"doc_id": "doc-c", "title": "C", "department": "", "source": "", "updated_at": "", "score": 0.5},
        ]
        citations = CampusKBRAG._citations(evidence)
        assert len(citations) == 3
        assert [c["index"] for c in citations] == [1, 2, 3]

    def test_empty_evidence(self):
        assert CampusKBRAG._citations([]) == []

    def test_missing_keys_default_to_none(self):
        evidence = [{"doc_id": "test"}]
        citations = CampusKBRAG._citations(evidence)
        assert citations[0]["title"] is None
        assert citations[0]["department"] is None
