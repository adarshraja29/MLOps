"""Unit tests for academic text cleaner and ModelService singleton logic."""

import pytest
from backend.app.utils.text_cleaner import clean_academic_text, compute_text_stats
from backend.app.services.model_service import ModelService


def test_clean_academic_text_removes_latex_and_citations():
    raw_sample = (
        r"\section{1. Introduction}" "\n"
        r"Transformer architectures \cite{vaswani2017} have achieved state-of-the-art results "
        r"in computational linguistics [1, 2]. As demonstrated by (Devlin et al., 2018), "
        r"bidirectional representations provide significant gains. "
        r"Consider the formula $\mathcal{L}_{CE} = -\sum y \log \hat{y}$ \cite{loss2020}. "
        r"For further details see https://arxiv.org/abs/1706.03762."
    )

    cleaned = clean_academic_text(raw_sample)

    # Assert LaTeX and citations stripped
    assert r"\section" not in cleaned
    assert r"\cite" not in cleaned
    assert "[1, 2]" not in cleaned
    assert "https://" not in cleaned
    assert r"$\mathcal{L}_{CE}" not in cleaned
    assert "Transformer architectures" in cleaned
    assert "bidirectional representations" in cleaned


def test_compute_text_stats():
    text = "Natural language processing models summarize academic papers accurately."
    stats = compute_text_stats(text)

    assert stats["word_count"] == 8
    assert stats["char_count"] == len(text)
    assert stats["sentence_count"] == 1
    assert stats["estimated_read_time_seconds"] > 0


def test_model_service_singleton_identity():
    """Verify that multiple instantiations of ModelService return the exact same instance in memory."""
    instance_a = ModelService()
    instance_b = ModelService.get_instance()
    instance_c = ModelService()

    assert instance_a is instance_b
    assert instance_b is instance_c
    assert id(instance_a) == id(instance_b) == id(instance_c)


def test_mock_summarize_pipeline():
    """Verify inference pipeline execution in mock mode without downloading weights."""
    service = ModelService.get_instance()
    service.settings.USE_MOCK_MODEL = True

    academic_text = (
        "In this work, we investigate sequence-to-sequence transformers for academic text summarization. "
        "Our methodology evaluates multi-lingual compression on linguistics benchmarks. "
        "The empirical findings demonstrate consistent improvements over baseline architectures."
    )

    result = service.summarize(academic_text)

    assert "summary" in result
    assert result["compression_ratio"] >= 0.0
    assert result["device"] == "cpu"
    assert "mock-academic-summarizer" in result["model_identifier"]
    assert "rss_memory_mb" in result
