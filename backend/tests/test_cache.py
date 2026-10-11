"""Unit tests for Zero-Compute SQLite Hash Cache Service."""

import os
import tempfile
import uuid
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.cache_service import CacheService, get_cache_service
from backend.app.services.model_service import ModelService

client = TestClient(app)


def test_cache_key_generation_composite():
    """Verify that cache key is composite and changes when slider parameters change."""
    text = "Academic literature summarization benchmark with transformer models."
    key_default = CacheService.generate_cache_key(text, 256, 40)
    key_same = CacheService.generate_cache_key(text, 256, 40)
    key_different_max = CacheService.generate_cache_key(text, 512, 40)
    key_different_min = CacheService.generate_cache_key(text, 256, 80)

    assert key_default == key_same
    assert key_default != key_different_max
    assert key_default != key_different_min


def test_zero_compute_cache_hit_pipeline():
    """Verify that duplicate request is served from SQLite cache with latency=0.0."""
    service = ModelService.get_instance()
    service.settings.USE_MOCK_MODEL = True

    # Use unique text with UUID for clean test
    unique_text = (
        f"In this empirical evaluation ({uuid.uuid4().hex}), we demonstrate zero-compute hash caching. "
        "The proposed architecture stores deterministic representations in SQLite, "
        "reclaiming valuable CPU cycles for identical academic literature queries."
    )

    payload = {
        "text": unique_text,
        "max_summary_tokens": 128,
        "min_summary_tokens": 30,
    }

    # 1. First invocation: Cache Miss
    res1 = client.post("/api/v1/summarize", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cache_hit"] is False
    assert data1["latency_seconds"] > 0

    # 2. Second invocation: Cache Hit (Instant)
    res2 = client.post("/api/v1/summarize", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["latency_seconds"] == 0.0
    assert data2["summary"] == data1["summary"]

    # 3. Third invocation with altered slider tokens: Cache Miss (Fresh summary)
    payload_modified = {
        "text": unique_text,
        "max_summary_tokens": 256,  # Changed slider
        "min_summary_tokens": 30,
    }
    res3 = client.post("/api/v1/summarize", json=payload_modified)
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["cache_hit"] is False


def test_cache_graceful_fallback_on_corrupt_key():
    """Verify that invalid/corrupt queries gracefully return None without crashing."""
    cache = get_cache_service()
    assert cache.get("non_existent_key_12345") is None
