"""Zero-Compute SQLite Hash Cache Service.

Provides instant cache retrieval for duplicate academic literature inputs:
- Computes composite MD5 hash: md5(f"{cleaned_text}_{max_length}_{min_length}")
- Caches full inference result payload (including ROUGE scores & keywords) in lightweight SQLite database
- Guarantees zero CPU inference cycles and 0.0s latency on duplicate requests
- Resilient design: catches all DB locks/IO errors and gracefully falls back to live model
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, Optional

from backend.app.config import get_settings

logger = logging.getLogger("academic_summarizer.cache_service")


class CacheService:
    """Thread-safe SQLite hash cache manager for summarization requests."""

    _instance: Optional["CacheService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "CacheService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CacheService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return

            self.settings = get_settings()
            self.db_path = self.settings.CACHE_DB_PATH
            self._db_lock = threading.Lock()
            self._init_db()
            self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        """Create a resilient SQLite connection with timeout and WAL mode."""
        dirname = os.path.dirname(self.db_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite schema if not already present, with schema migration support."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS summary_cache (
                        cache_key TEXT PRIMARY KEY,
                        summary TEXT NOT NULL,
                        cleaned_input_sample TEXT,
                        input_stats JSON NOT NULL,
                        summary_stats JSON NOT NULL,
                        compression_ratio REAL NOT NULL,
                        model_identifier TEXT NOT NULL,
                        device TEXT NOT NULL,
                        rouge_scores JSON,
                        keywords JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_cache_key ON summary_cache(cache_key);")

                # Graceful schema migration for existing databases
                try:
                    conn.execute("ALTER TABLE summary_cache ADD COLUMN rouge_scores JSON;")
                except Exception:
                    pass
                try:
                    conn.execute("ALTER TABLE summary_cache ADD COLUMN keywords JSON;")
                except Exception:
                    pass

            logger.info("SQLite zero-compute hash cache initialized at: %s", self.db_path)
        except Exception as exc:
            logger.warning("Could not initialize SQLite cache database (%s): %s. Cache disabled.", self.db_path, str(exc))

    @staticmethod
    def generate_cache_key(cleaned_text: str, max_tokens: int, min_tokens: int) -> str:
        """Generate deterministic MD5 hash composite of text and length boundaries."""
        composite = f"{cleaned_text}_{max_tokens}_{min_tokens}"
        return hashlib.md5(composite.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached inference result by hash key. Returns None on cache miss or error."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT summary, cleaned_input_sample, input_stats, summary_stats, compression_ratio,
                           model_identifier, device, rouge_scores, keywords
                    FROM summary_cache WHERE cache_key = ?
                    """,
                    (cache_key,),
                )
                row = cursor.fetchone()
                if row:
                    logger.info("⚡ Zero-Compute Cache HIT for key: %s", cache_key[:8])
                    rouge_val = json.loads(row["rouge_scores"]) if row["rouge_scores"] else None
                    keywords_val = json.loads(row["keywords"]) if row["keywords"] else []

                    return {
                        "summary": row["summary"],
                        "cleaned_input_sample": row["cleaned_input_sample"],
                        "input_stats": json.loads(row["input_stats"]),
                        "summary_stats": json.loads(row["summary_stats"]),
                        "compression_ratio": row["compression_ratio"],
                        "latency_seconds": 0.0,  # Zero CPU inference time
                        "model_identifier": f"{row['model_identifier']} (Cached)",
                        "device": row["device"],
                        "cache_hit": True,
                        "rouge_scores": rouge_val,
                        "keywords": keywords_val,
                    }
        except Exception as exc:
            logger.warning("SQLite cache read error (%s): %s. Falling back to live inference.", cache_key[:8], str(exc))

        return None

    def set(self, cache_key: str, payload: Dict[str, Any]) -> None:
        """Store newly generated inference result in SQLite cache. Gracefully ignores write failures."""
        try:
            rouge_json = json.dumps(payload.get("rouge_scores")) if payload.get("rouge_scores") else None
            keywords_json = json.dumps(payload.get("keywords", []))

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO summary_cache (
                        cache_key, summary, cleaned_input_sample, input_stats, summary_stats,
                        compression_ratio, model_identifier, device, rouge_scores, keywords
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        payload["summary"],
                        payload.get("cleaned_input_sample", ""),
                        json.dumps(payload["input_stats"]),
                        json.dumps(payload["summary_stats"]),
                        payload["compression_ratio"],
                        payload["model_identifier"],
                        payload.get("device", "cpu"),
                        rouge_json,
                        keywords_json,
                    ),
                )
            logger.info("Saved inference summary to SQLite cache (key: %s)", cache_key[:8])
        except Exception as exc:
            logger.warning("SQLite cache write failed for key %s: %s", cache_key[:8], str(exc))


# Singleton access helper
def get_cache_service() -> CacheService:
    """Retrieve singleton CacheService instance."""
    return CacheService()
