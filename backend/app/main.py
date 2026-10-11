"""FastAPI application entry point for Computational Linguistics Literature Summarizer.

Exposes:
- GET /health: Ultra-lightweight endpoint for Kubernetes Liveness & Readiness probes
- GET /metrics: Prometheus observability endpoint (latencies, HTTP codes, process RSS memory)
- POST /api/v1/summarize: Memory-safe scientific text summarization with SQLite caching & ROUGE
- POST /api/v1/extract-pdf: Lightweight PDF text extraction using pypdf
"""

import gc
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psutil
from pypdf import PdfReader
from prometheus_client import REGISTRY, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from backend.app.config import get_settings
from backend.app.models.schemas import (
    ExtractPdfResponse,
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from backend.app.services.model_service import ModelService
from backend.app.utils.text_cleaner import normalize_pdf_text

# Configure logger
logger = logging.getLogger("academic_summarizer.main")
logging.basicConfig(level=logging.INFO)

settings = get_settings()

# ==============================================================================
# Prometheus Memory Observability (Critical for 4GB Constraint)
# ==============================================================================
def get_current_rss_bytes() -> int:
    """Return process resident memory in bytes via psutil."""
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return 0


def get_current_rss_mb() -> float:
    """Return process resident memory in Megabytes via psutil."""
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


# Register custom Gauges if not already present in the default registry
if "process_resident_memory_bytes" not in REGISTRY._names_to_collectors:
    process_resident_memory_bytes_gauge = Gauge(
        "process_resident_memory_bytes",
        "Resident memory size (RSS) of the application process in bytes",
    )
    process_resident_memory_bytes_gauge.set_function(get_current_rss_bytes)

if "app_process_memory_rss_megabytes" not in REGISTRY._names_to_collectors:
    process_resident_memory_mb_gauge = Gauge(
        "app_process_memory_rss_megabytes",
        "Current resident memory size (RSS) in Megabytes relative to 4GB RAM constraint",
    )
    process_resident_memory_mb_gauge.set_function(get_current_rss_mb)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup warm-up and shutdown cleanup."""
    logger.info("Initializing %s (v%s)...", settings.APP_NAME, settings.APP_VERSION)
    # Instantiate ModelService singleton (weights load on demand or warm-up)
    model_service = ModelService.get_instance()
    logger.info("ModelService singleton registered. Ready to serve requests.")

    yield

    logger.info("Shutting down application. Cleaning memory resources...")
    gc.collect()
    logger.info("Shutdown cleanup complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Operationalized NLP pipeline for academic paper summarization with strict memory safeguards.",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Initialize Prometheus Instrumentator
# ==============================================================================
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health"],
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])


@app.get("/", tags=["Root"])
def root():
    """Root discovery endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "summarize_endpoint": "/api/v1/summarize",
        "pdf_extract_endpoint": "/api/v1/extract-pdf",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Monitoring"],
    summary="Lightweight Kubernetes Liveness & Readiness probe",
)
def health_check() -> HealthResponse:
    """Ultra-lightweight health probe endpoint.

    Executes in < 5ms without triggering heavy model computation,
    ensuring Kubernetes readiness probes do not fail or timeout.
    """
    model_service = ModelService.get_instance()
    rss_mem = model_service.get_current_rss_memory_mb()

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_loaded=model_service.is_loaded,
        device=model_service.device_str,
        rss_memory_mb=rss_mem,
    )


@app.post(
    "/api/v1/extract-pdf",
    response_model=ExtractPdfResponse,
    status_code=status.HTTP_200_OK,
    tags=["Preprocessing"],
    summary="Extract plain text from uploaded academic PDF",
)
async def extract_pdf_text(file: UploadFile = File(...)) -> ExtractPdfResponse:
    """Extract and normalize plain text from an academic PDF file.

    - Validates file format and enforces strict 5MB file size limit
    - Reads pages stream-by-stream using lightweight pypdf (<5MB memory)
    - Normalizes line breaks and whitespace
    - Does NOT trigger summarizer inference
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF (.pdf) documents are supported.",
        )

    max_bytes = 5 * 1024 * 1024  # 5 MB
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of 5MB ({round(len(file_bytes) / (1024*1024), 2)}MB uploaded).",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is empty (0 bytes).",
        )

    try:
        reader = PdfReader(BytesIO(file_bytes))
        num_pages = len(reader.pages)
        if num_pages == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF document contains zero readable pages.",
            )

        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)

        raw_combined = "\n\n".join(extracted_pages)
        cleaned_text = normalize_pdf_text(raw_combined)

        if not cleaned_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No extractable text found in the PDF. The file may contain scanned images without OCR.",
            )

        return ExtractPdfResponse(
            extracted_text=cleaned_text,
            page_count=num_pages,
            filename=file.filename,
            char_count=len(cleaned_text),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error extracting text from PDF '%s': %s", file.filename, str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse PDF document: {str(exc)}",
        )
    finally:
        del file_bytes
        gc.collect()


@app.post(
    "/api/v1/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Summarize academic literature",
)
def summarize_text(payload: SummarizeRequest) -> SummarizeResponse:
    """Summarize academic text with strict RAM safeguards.

    - Input length strictly validated between 50 and 50,000 characters
    - Algorithmic Map-Reduce chunking executed on texts > 12,000 characters
    - Processed through singleton SciBERT tokenizer & LongT5 summarizer
    - Evaluates ROUGE-1/2/L and extracts scientific keywords
    - Reclaims intermediate tensors immediately post-inference
    """
    model_service = ModelService.get_instance()

    try:
        result = model_service.summarize(
            raw_text=payload.text,
            max_summary_tokens=payload.max_summary_tokens,
            min_summary_tokens=payload.min_summary_tokens,
        )
        return SummarizeResponse(**result)

    except ValueError as ve:
        logger.warning("Input validation error during summarization: %s", str(ve))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        logger.error("Unexpected error during summarization inference: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the summary. Memory safeguards engaged.",
        )
