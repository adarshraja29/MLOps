"""Automated tests for Kubernetes health probes and root discovery endpoints."""

import time
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root discovery endpoint responds with 200 and metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data
    assert "health" in data


def test_health_probe_performance_and_schema():
    """Verify /health responds in < 50ms and satisfies Kubernetes probe schema."""
    start = time.perf_counter()
    response = client.get("/health")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 100, f"/health probe took {elapsed_ms}ms, which is too slow for k8s"

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data
    assert data["device"] == "cpu"
    assert "rss_memory_mb" in data
    assert data["rss_memory_mb"] > 0
