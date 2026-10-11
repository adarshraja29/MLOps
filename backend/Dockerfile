# ==============================================================================
# Stage 1: Build Dependencies (Multi-stage Slim Builder)
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /install

# Install build-essential only if C-extensions are required during wheel build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Create virtual environment and install pinned CPU-only packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies using CPU-only PyTorch index and purge pip cache aggressively
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete

# ==============================================================================
# Stage 2: Final Minimal Runtime Image
# ==============================================================================
FROM python:3.10-slim AS runtime

# System metadata & environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    PATH="/opt/venv/bin:$PATH" \
    DEVICE=cpu \
    FORCE_CPU=true \
    PORT=8000 \
    MAX_WORKERS=1

WORKDIR /workspace

# Create a non-root system user for security compliance
RUN groupadd -r appuser && useradd -r -g appuser -d /workspace -s /sbin/nologin appuser
# Create cache directory and grant permissions
RUN mkdir -p /workspace/.cache && chown -R appuser:appuser /workspace

# Copy installed virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the application source tree
COPY --chown=appuser:appuser app /workspace/backend/app

# Drop root privileges
USER appuser

# Lightweight container health check without installing curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run single worker to prevent duplicate model weights in RAM
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
