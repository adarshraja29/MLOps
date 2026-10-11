"""Unit tests for Statistical Metrics (ROUGE & Keyword Extraction) and Cache Synchronization."""

import uuid
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_service import ModelService
from backend.app.utils.text_cleaner import (
    compute_rouge_scores,
    extract_statistical_keywords,
)

client = TestClient(app)


def test_pure_python_keyword_extraction():
    """Verify statistical keyword extraction extracts scientific phrases without NLTK."""
    sample = (
        "Transformer architectures revolutionized sequence transduction in computational linguistics. "
        "Self-attention mechanisms capture bidirectional contextual embeddings effectively. "
        "Empirical benchmarks demonstrate superior language modeling across linguistic corpora."
    )
    keywords = extract_statistical_keywords(sample, top_k=5)

    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert len(keywords) <= 5
    # Should capture core scientific unigrams or bigrams
    text_joined = " ".join(keywords).lower()
    assert any(term in text_joined for term in ["transformer", "linguistics", "attention", "embeddings", "contextual", "sequence", "language"])


def test_rouge_scorer_fmeasure():
    """Verify ROUGE scorer computes precision, recall, and fmeasure without stemmer."""
    reference = "Transformer models utilize self-attention mechanisms for document summarization."
    hypothesis = "Self-attention mechanisms in transformer models summarize documents."

    scores = compute_rouge_scores(reference, hypothesis)

    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores

    for metric in ["rouge1", "rouge2", "rougeL"]:
        assert 0.0 <= scores[metric]["precision"] <= 1.0
        assert 0.0 <= scores[metric]["recall"] <= 1.0
        assert 0.0 <= scores[metric]["fmeasure"] <= 1.0

    assert scores["rouge1"]["fmeasure"] > 0.0


def test_summarization_includes_rouge_and_keywords_and_caches_them():
    """Verify that /api/v1/summarize generates ROUGE and keywords, and retrieves them from cache."""
    service = ModelService.get_instance()
    service.settings.USE_MOCK_MODEL = True

    unique_academic_corpus = (
        f"Recent advances in natural language processing ({uuid.uuid4().hex}) benchmark transfer learning across multiple tasks. "
        "Our methodology investigates sparse attention and parameter-efficient fine-tuning on academic corpora. "
        "The experimental results establish significant gains in computational linguistics evaluations."
    )

    payload = {
        "text": unique_academic_corpus,
        "max_summary_tokens": 128,
        "min_summary_tokens": 30,
    }

    # 1. First execution (Live / Mock Inference)
    res1 = client.post("/api/v1/summarize", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()

    assert "rouge_scores" in data1
    assert data1["rouge_scores"] is not None
    assert "rouge1" in data1["rouge_scores"]
    assert "rouge2" in data1["rouge_scores"]
    assert "rougeL" in data1["rouge_scores"]
    assert data1["rouge_scores"]["rouge1"]["fmeasure"] >= 0.0

    assert "keywords" in data1
    assert isinstance(data1["keywords"], list)
    assert len(data1["keywords"]) > 0
    assert data1["cache_hit"] is False

    # 2. Second execution (Zero-Compute Cache Hit)
    res2 = client.post("/api/v1/summarize", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()

    assert data2["cache_hit"] is True
    assert data2["latency_seconds"] == 0.0
    # Crucial rule: cache hit must restore saved ROUGE scores and keywords!
    assert data2["rouge_scores"] is not None
    assert data2["rouge_scores"]["rouge1"]["fmeasure"] == data1["rouge_scores"]["rouge1"]["fmeasure"]
    assert data2["keywords"] == data1["keywords"]
