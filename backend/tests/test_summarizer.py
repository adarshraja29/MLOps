"""Automated tests for summarization API endpoint and input boundary validation."""

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_service import ModelService

client = TestClient(app)


def test_summarize_valid_payload():
    """Verify successful summarization with proper request schema."""
    service = ModelService.get_instance()
    service.settings.USE_MOCK_MODEL = True

    payload = {
        "text": (
            "Large language models and encoder-decoder transformers have transformed "
            "computational linguistics. In this paper, we demonstrate a specialized "
            "methodology for condensing dense scientific abstracts while preserving key "
            "methodological findings and numerical metrics across benchmark corpora."
        ),
        "max_summary_tokens": 128,
        "min_summary_tokens": 20,
    }

    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert "input_stats" in data
    assert "summary_stats" in data
    assert "compression_ratio" in data
    assert "latency_seconds" in data
    assert data["device"] == "cpu"


def test_reject_short_input_boundary():
    """Verify HTTP 422 Unprocessable Entity when text is shorter than 50 characters."""
    payload = {
        "text": "Too short text.",
    }

    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("at least 50 characters" in str(err) or "min_length" in str(err) for err in errors)


def test_reject_massive_input_boundary():
    """Verify HTTP 422 Unprocessable Entity when text exceeds 50,000 characters (RAM protection)."""
    # Create text payload that exceeds 50,000 characters
    massive_text = "Computational linguistics analysis. " * 1500  # ~54,000 characters
    payload = {
        "text": massive_text,
    }

    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("at most 50000 characters" in str(err) or "max_length" in str(err) for err in errors)


def test_map_reduce_chunking_on_long_document():
    """Verify Map-Reduce chunking executes on inputs > 12,000 characters up to 50,000 chars."""
    service = ModelService.get_instance()
    service.settings.USE_MOCK_MODEL = True

    # 25,000 characters academic text with paragraph breaks
    paragraph = (
        "In this empirical evaluation of computational linguistics, we investigate "
        "algorithmic chunking using a Map-Reduce paradigm for long document summarization. "
        "Sequential passes with deterministic garbage collection prevent memory accumulation.\n\n"
    )
    long_academic_text = paragraph * 110  # ~25,000 characters
    assert len(long_academic_text) > 12000

    payload = {
        "text": long_academic_text,
        "max_summary_tokens": 200,
        "min_summary_tokens": 40,
    }

    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert data["input_stats"]["char_count"] > 12000
    assert data["compression_ratio"] > 0.0
    assert "keywords" in data
    assert "rouge_scores" in data


def test_reject_invalid_token_boundary():
    """Verify HTTP 422 when min_summary_tokens is greater than max_summary_tokens."""
    payload = {
        "text": "This is a valid length academic document abstract discussing neural transformers in linguistics.",
        "max_summary_tokens": 50,
        "min_summary_tokens": 100,  # Invalid: min > max
    }

    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 422
