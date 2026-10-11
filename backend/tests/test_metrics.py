"""Automated tests for Prometheus telemetry and memory observability endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_metrics_endpoint_status_and_format():
    """Verify /metrics responds with HTTP 200 and standard Prometheus text payload."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

    payload = response.text
    # Standard Prometheus comment headers
    assert "# HELP" in payload
    assert "# TYPE" in payload


def test_memory_tracking_metrics_exposed():
    """Verify process RSS memory metrics are exposed for the 4GB RAM constraint."""
    # Issue a request to trigger metric updates
    client.get("/")

    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.text

    # Verify critical memory metric names
    assert "process_resident_memory_bytes" in payload
    assert "app_process_memory_rss_megabytes" in payload

    # Verify numeric values are emitted
    lines = payload.splitlines()
    memory_lines = [
        line for line in lines
        if not line.startswith("#") and "process_resident_memory_bytes" in line
    ]
    assert len(memory_lines) > 0, "No sample found for process_resident_memory_bytes"

    # Extract value and assert RSS > 0 bytes
    parts = memory_lines[0].split()
    rss_value = float(parts[-1])
    assert rss_value > 0, f"Expected RSS > 0 bytes, got {rss_value}"


def test_http_traffic_instrumentation():
    """Verify HTTP traffic volume and status codes are tracked by Prometheus instrumentator."""
    # Hit root and health endpoints
    client.get("/")
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.text

    # prometheus-fastapi-instrumentator tracks http_requests or http_request_duration
    has_http_metric = (
        "http_requests_total" in payload or
        "http_request_duration_seconds" in payload or
        "http_requests_inprogress" in payload
    )
    assert has_http_metric, "Expected HTTP request metrics to be present in Prometheus payload"
