"""AI Core Inference Service for Academic Literature Summarization.

Implements a thread-safe Singleton pattern for model management with strict
memory safeguards for low-RAM, CPU-constrained environments:
- SciBERT tokenizer (allenai/scibert_scivocab_uncased) for scientific tokenization
- LongT5 (google/long-t5-tglobal-base) for sequence-to-sequence summarization in high-accuracy FP32
- repetition_penalty=2.5 to strictly prevent token loops on citations and references
- PyTorch CPU Thread Tuning (torch.set_num_threads) to prevent WSL2 thread-thrashing
- Statistical Extractive Pre-Filtering to condense >12k char PDFs and run inference exactly ONCE
- Hardcoded torch.no_grad() context during inference
- Deterministic gc.collect() and tensor deletion in finally blocks
- Process memory tracking via psutil
- Zero-compute SQLite hash caching
"""

import gc
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

import psutil

from backend.app.config import get_settings
from backend.app.services.cache_service import get_cache_service
from backend.app.utils.text_cleaner import (
    clean_academic_text,
    compute_text_stats,
    extract_statistical_keywords,
    compute_rouge_scores,
)

# Configure module logger
logger = logging.getLogger("academic_summarizer.model_service")
logging.basicConfig(level=logging.INFO)

# ==============================================================================
# PyTorch CPU Thread Tuning (Prevent WSL2 / Container Thread-Thrashing)
# ==============================================================================
try:
    import torch
    _target_cpu_threads = min(4, max(1, os.cpu_count() or 1))
    torch.set_num_threads(_target_cpu_threads)
    torch.set_num_interop_threads(1)
    logger.info("Configured PyTorch CPU threads: %d (interop: 1)", _target_cpu_threads)
except ImportError:
    pass


class ModelService:
    """Thread-safe Singleton Model Service managing SciBERT and LongT5."""

    _instance: Optional["ModelService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "ModelService":
        """Ensure only one instance exists across all threads (Double-Checked Locking)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Convenience method to access the singleton instance."""
        return cls()

    def __init__(self) -> None:
        """Initialize service attributes once."""
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return

            self.settings = get_settings()
            self._model_lock = threading.Lock()
            self.is_loaded: bool = False

            # Model handles
            self.scibert_tokenizer: Any = None
            self.longt5_tokenizer: Any = None
            self.longt5_model: Any = None

            # Hardcode CPU execution
            self.device_str = "cpu"
            logger.info("ModelService initialized in CPU-only mode. Models will load on demand/startup.")
            self._initialized = True

    @staticmethod
    def get_current_rss_memory_mb() -> float:
        """Return the current process Resident Set Size (RSS) memory in Megabytes."""
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)

    def load_models(self) -> None:
        """Load SciBERT and LongT5 into memory once in a thread-safe manner (FP32 precision)."""
        if self.is_loaded:
            logger.info("Models already loaded in memory (Singleton). Skipping reload.")
            return

        with self._model_lock:
            if self.is_loaded:
                return

            mem_start = self.get_current_rss_memory_mb()
            logger.info("Loading models into memory... Current RSS Memory: %s MB", mem_start)

            if self.settings.USE_MOCK_MODEL:
                logger.warning("USE_MOCK_MODEL is TRUE. Loading lightweight mock model service (0 MB overhead).")
                self.is_loaded = True
                return

            # Dynamic import to prevent premature heavyweight memory allocation
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                # Enforce CPU-only PyTorch globally
                torch.set_grad_enabled(False)

                logger.info("Loading SciBERT Tokenizer: %s", self.settings.SCIBERT_MODEL_NAME)
                self.scibert_tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.SCIBERT_MODEL_NAME,
                    use_fast=True,
                )

                logger.info("Loading LongT5 Tokenizer: %s", self.settings.LONGT5_MODEL_NAME)
                self.longt5_tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.LONGT5_MODEL_NAME,
                    use_fast=True,
                )

                logger.info("Loading LongT5 Model weights: %s (torch.float32 on CPU)", self.settings.LONGT5_MODEL_NAME)
                # Load sequence-to-sequence model in evaluation mode
                self.longt5_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.settings.LONGT5_MODEL_NAME,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.longt5_model.eval()

                self.is_loaded = True
                mem_end = self.get_current_rss_memory_mb()
                logger.info("Models successfully loaded into memory. New RSS Memory: %s MB (Delta: +%s MB)",
                            mem_end, round(mem_end - mem_start, 2))

            except Exception as exc:
                logger.error("Failed to load models into memory: %s", str(exc), exc_info=True)
                raise RuntimeError(f"Model initialization failure: {str(exc)}") from exc
            finally:
                # Deterministic cleanup of temporary initialization objects
                gc.collect()

    @staticmethod
    def extract_salient_academic_text(
        text: str,
        max_chars: int = 12000,
        keywords: Optional[list[str]] = None,
    ) -> str:
        """Statistical extractive pre-filtering for large documents (>12,000 chars).

        Ranks sentences and paragraphs using scientific keyword frequency and structural
        position weights (prioritizing Abstract/Introduction and Results/Conclusion),
        condensing the corpus down to ~12,000 chars so LongT5 runs exactly once on CPU.
        """
        if len(text) <= max_chars:
            return text

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            raw_units = re.split(r"(?<=[.!?])\s+", text)
        else:
            raw_units = paragraphs

        units = [u.strip() for u in raw_units if len(u.strip()) > 30]
        if not units:
            return text[:max_chars]

        total_units = len(units)
        kw_set = {k.lower() for k in (keywords or [])}

        scored = []
        for i, unit in enumerate(units):
            u_lower = unit.lower()
            kw_hits = sum(1 for kw in kw_set if kw in u_lower)
            pos = i / max(total_units - 1, 1)

            # Structural weight: Abstract/Intro (first 20%) & Conclusion/Results (last 20%)
            if pos < 0.20:
                pos_weight = 2.0
            elif pos > 0.80:
                pos_weight = 1.8
            else:
                pos_weight = 1.0

            score = (kw_hits * 2.0 + 1.0) * pos_weight
            scored.append((score, i, unit))

        # Select highest ranking units
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        current_chars = 0
        for score, original_idx, unit in scored:
            if current_chars + len(unit) + 2 <= max_chars:
                selected.append((original_idx, unit))
                current_chars += len(unit) + 2
            if current_chars >= max_chars * 0.95:
                break

        # Restore original chronological reading flow
        selected.sort(key=lambda x: x[0])
        separator = "\n\n" if len(paragraphs) > 1 else " "
        return separator.join(unit for _, unit in selected)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 12000) -> list[str]:
        """Split text into sequential chunks of approximately chunk_size chars,
        safely respecting paragraph (\n\n) or sentence boundaries where possible.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        remaining = text.strip()
        while len(remaining) > chunk_size:
            slice_zone = remaining[:chunk_size]

            split_idx = slice_zone.rfind("\n\n")
            if split_idx != -1 and split_idx >= int(chunk_size * 0.5):
                split_point = split_idx + 2
            else:
                sentence_matches = [
                    slice_zone.rfind(". "),
                    slice_zone.rfind(".\n"),
                    slice_zone.rfind("? "),
                    slice_zone.rfind("! "),
                ]
                best_sentence = max(sentence_matches)
                if best_sentence != -1 and best_sentence >= int(chunk_size * 0.5):
                    split_point = best_sentence + 2
                else:
                    split_idx = slice_zone.rfind("\n")
                    if split_idx != -1 and split_idx >= int(chunk_size * 0.5):
                        split_point = split_idx + 1
                    else:
                        split_idx = slice_zone.rfind(" ")
                        if split_idx != -1 and split_idx >= int(chunk_size * 0.5):
                            split_point = split_idx + 1
                        else:
                            split_point = chunk_size

            chunk = remaining[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_point:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    def _generate_chunk_summary(
        self,
        chunk_text: str,
        max_tokens: int,
        min_tokens: int,
    ) -> str:
        """Run LongT5 generation on a single chunk with strict memory cleanup."""
        if self.settings.USE_MOCK_MODEL:
            time.sleep(0.05)
            sentences = [s.strip() for s in chunk_text.split(".") if len(s.strip()) > 20]
            summary_sentences = sentences[:2] if len(sentences) >= 2 else sentences
            return ". ".join(summary_sentences) + "." if summary_sentences else chunk_text[:200]

        import torch

        input_ids = None
        attention_mask = None
        summary_ids = None
        inputs = None

        try:
            prefixed_text = f"summarize: {chunk_text}"
            inputs = self.longt5_tokenizer(
                prefixed_text,
                max_length=self.settings.MAX_INPUT_TOKENS,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                summary_ids = self.longt5_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_tokens,
                    min_length=min_tokens,
                    num_beams=self.settings.NUM_BEAMS,
                    no_repeat_ngram_size=self.settings.NO_REPEAT_NGRAM_SIZE,
                    repetition_penalty=2.5,
                    early_stopping=True,
                    length_penalty=1.0,
                )

            summary_text = self.longt5_tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return summary_text

        finally:
            # Explicitly call del on intermediate tensors and run gc.collect()
            if input_ids is not None:
                del input_ids
            if attention_mask is not None:
                del attention_mask
            if summary_ids is not None:
                del summary_ids
            if inputs is not None:
                del inputs
            gc.collect()

    def summarize(
        self,
        raw_text: str,
        max_summary_tokens: Optional[int] = None,
        min_summary_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize academic text with strict memory safeguards.

        Enforces:
        - Input text preprocessing (cleaning LaTeX, citations, math)
        - Statistical extractive pre-filtering for large documents (>12,000 chars)
          to ensure LongT5 runs exactly ONCE on CPU instead of multi-pass delays
        - High-accuracy FP32 LongT5 inference with repetition penalty (2.5)
        - Hardcoded torch.no_grad() inference blocks
        - Deterministic gc.collect() and tensor deletion
        - Process memory tracking via psutil
        - Zero-compute SQLite hash caching
        """
        # Ensure models are loaded
        if not self.is_loaded:
            self.load_models()

        start_time = time.perf_counter()
        initial_mem = self.get_current_rss_memory_mb()

        # Step 1: Academic text cleaning
        cleaned_text = clean_academic_text(raw_text)
        input_stats = compute_text_stats(cleaned_text)

        if not cleaned_text:
            raise ValueError("Provided text contains no processable linguistic content after cleaning.")

        max_tokens = max_summary_tokens or self.settings.MAX_SUMMARY_TOKENS
        min_tokens = min_summary_tokens or self.settings.MIN_SUMMARY_TOKENS

        # Zero-Compute Hash Caching Check (SQLite)
        cache_service = get_cache_service()
        cache_key = cache_service.generate_cache_key(cleaned_text, max_tokens, min_tokens)
        cached_result = cache_service.get(cache_key)
        if cached_result:
            cached_result["rss_memory_mb"] = self.get_current_rss_memory_mb()
            return cached_result

        # Step 2: Extract Statistical Keywords
        keywords = extract_statistical_keywords(cleaned_text, top_k=5)

        # Step 3: Inference execution
        # For long documents (>12,000 chars), apply extractive pre-filtering so
        # the FP32 LongT5 model runs exactly ONCE on CPU, avoiding 20-minute Map-Reduce delays.
        if len(cleaned_text) > 12000:
            logger.info(
                "Input text length (%d chars) exceeds 12,000 chars. Applying statistical extractive pre-filtering...",
                len(cleaned_text),
            )
            distilled_input = self.extract_salient_academic_text(
                text=cleaned_text,
                max_chars=12000,
                keywords=keywords,
            )
            logger.info(
                "Extractive pre-filtering distilled text from %d to %d chars. Running LongT5 model once on CPU.",
                len(cleaned_text),
                len(distilled_input),
            )
            text_for_model = distilled_input
        else:
            text_for_model = cleaned_text

        # Run FP32 LongT5 model exactly ONCE
        summary_text = self._generate_chunk_summary(
            chunk_text=text_for_model,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

        elapsed = round(time.perf_counter() - start_time, 3)
        final_mem = self.get_current_rss_memory_mb()
        summary_stats = compute_text_stats(summary_text)

        compression_ratio = round(
            1.0 - (summary_stats["word_count"] / max(input_stats["word_count"], 1)),
            3,
        )

        if final_mem > self.settings.MEMORY_THRESHOLD_MB:
            logger.warning(
                "Memory consumption alert: Current RSS %s MB exceeds threshold %s MB",
                final_mem,
                self.settings.MEMORY_THRESHOLD_MB,
            )

        rouge_scores = compute_rouge_scores(reference=cleaned_text, hypothesis=summary_text)

        model_ident = (
            "mock-academic-summarizer"
            if self.settings.USE_MOCK_MODEL
            else self.settings.LONGT5_MODEL_NAME
        )

        result = {
            "summary": summary_text,
            "cleaned_input_sample": cleaned_text[:200] + ("..." if len(cleaned_text) > 200 else ""),
            "input_stats": input_stats,
            "summary_stats": summary_stats,
            "compression_ratio": max(0.0, compression_ratio),
            "latency_seconds": elapsed,
            "rss_memory_mb": final_mem,
            "memory_delta_mb": round(final_mem - initial_mem, 2),
            "model_identifier": model_ident,
            "device": self.device_str,
            "cache_hit": False,
            "rouge_scores": rouge_scores,
            "keywords": keywords,
        }

        # Save to SQLite zero-compute cache
        cache_service.set(cache_key, result)
        return result
