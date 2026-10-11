"""Pydantic schemas and data validation models.

Strictly enforces length boundaries to prevent RAM spikes and memory exhaustion.
"""

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class TextStatsSchema(BaseModel):
    """Linguistic and structural statistics of processed text."""
    char_count: int = Field(..., description="Total character count")
    word_count: int = Field(..., description="Total word count")
    sentence_count: int = Field(..., description="Approximate sentence count")
    estimated_read_time_seconds: float = Field(..., description="Estimated human reading time in seconds")


class HealthResponse(BaseModel):
    """Ultra-lightweight health status schema for Kubernetes probes."""
    status: str = Field(default="healthy", description="Application operational status")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether model weights are loaded into memory")
    device: str = Field(default="cpu", description="Execution hardware device")
    rss_memory_mb: float = Field(..., description="Current resident memory footprint in MB")


class ExtractPdfResponse(BaseModel):
    """Output payload for PDF text extraction."""
    extracted_text: str = Field(..., description="Cleaned plain text extracted from the PDF document")
    page_count: int = Field(..., description="Total pages processed")
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    char_count: int = Field(..., description="Character count of extracted text")


class SummarizeRequest(BaseModel):
    """Input payload for academic text summarization.

    Enforces MIN_INPUT_CHARS=50 and MAX_INPUT_CHARS=50000 (Map-Reduce chunking active for >12,000 chars).
    """
    text: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Raw academic paper text or abstract (capped at 50,000 characters with Map-Reduce chunking)",
        examples=[
            "Transformer models have modernized natural language processing. In this paper, we explore "
            "sparse attention mechanisms to process long academic documents under restricted memory constraints."
        ]
    )
    max_summary_tokens: Optional[int] = Field(
        default=None,
        ge=20,
        le=512,
        description="Upper token limit for generated summary"
    )
    min_summary_tokens: Optional[int] = Field(
        default=None,
        ge=10,
        le=256,
        description="Lower token limit for generated summary"
    )

    @model_validator(mode="after")
    def validate_token_limits(self) -> "SummarizeRequest":
        """Ensure min_summary_tokens does not exceed max_summary_tokens."""
        if self.min_summary_tokens and self.max_summary_tokens:
            if self.min_summary_tokens > self.max_summary_tokens:
                raise ValueError("min_summary_tokens cannot be greater than max_summary_tokens")
        return self


class RougeMetric(BaseModel):
    """Individual ROUGE score metrics."""
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    fmeasure: float = Field(..., description="F1 measure")


class RougeScores(BaseModel):
    """Aggregate ROUGE-1, ROUGE-2, and ROUGE-L metrics."""
    rouge1: RougeMetric
    rouge2: RougeMetric
    rougeL: RougeMetric


class SummarizeResponse(BaseModel):
    """Output payload returned by the summarization endpoint."""
    summary: str = Field(..., description="Generated abstractive summary")
    cleaned_input_sample: str = Field(..., description="Preview snippet of cleaned input text")
    input_stats: TextStatsSchema = Field(..., description="Linguistic metrics of original input")
    summary_stats: TextStatsSchema = Field(..., description="Linguistic metrics of generated summary")
    compression_ratio: float = Field(..., description="Proportion of content condensed (0.0 to 1.0)")
    latency_seconds: float = Field(..., description="End-to-end inference execution time in seconds")
    rss_memory_mb: float = Field(..., description="Host process resident memory in MB after inference")
    model_identifier: str = Field(..., description="Name of the NLP model used")
    device: str = Field(default="cpu", description="Hardware device used for inference")
    cache_hit: bool = Field(default=False, description="Whether summary was served from zero-compute SQLite cache")
    rouge_scores: Optional[RougeScores] = Field(default=None, description="ROUGE-1, ROUGE-2, and ROUGE-L evaluation scores")
    keywords: list[str] = Field(default_factory=list, description="Top statistical scientific keywords extracted from input")

