"""
wavemind/config.py
------------------
All configuration is loaded from environment variables (or .env via python-dotenv).
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _positive_int(key: str, fallback: int) -> int:
    try:
        v = int(os.environ.get(key, ""))
        return v if v > 0 else fallback
    except (ValueError, TypeError):
        return fallback


def _nonneg_float(key: str, fallback: float) -> float:
    try:
        v = float(os.environ.get(key, ""))
        return v if v >= 0 else fallback
    except (ValueError, TypeError):
        return fallback


class Config:
    # Flask
    DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    PORT: int = _positive_int("PORT", 8000)

    # Ollama
    CHAT_PROVIDER: str = os.environ.get("CHAT_PROVIDER", "ollama").lower()
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "tinyllama")
    OLLAMA_KEEP_ALIVE: str = os.environ.get("OLLAMA_KEEP_ALIVE", "120m")   # keep model hot longer
    OLLAMA_TIMEOUT_MS: int = _positive_int("OLLAMA_TIMEOUT_MS", 15_000)    # 15s hard ceiling
    OLLAMA_NUM_CTX: int = _positive_int("OLLAMA_NUM_CTX", 512)             # 512 tokens = enough for Q+A
    OLLAMA_NUM_THREAD: int = _positive_int("OLLAMA_NUM_THREAD", 4)
    OLLAMA_NUM_PREDICT: int = _positive_int("OLLAMA_NUM_PREDICT", 120)     # 120 tokens ≈ 90 words
    OLLAMA_TEMPERATURE: float = _nonneg_float("OLLAMA_TEMPERATURE", 0.1)   # lower = faster, more focused

    # Reply quality controls
    # Continuations = extra round-trips = +10-15s each. Disabled.
    OLLAMA_MAX_CONTINUATIONS: int = 0
    OLLAMA_MAX_WORDS: int = _positive_int("OLLAMA_MAX_WORDS", 100)

    # Rate limiting
    CHAT_RATE_LIMIT_MAX: int = _positive_int("CHAT_RATE_LIMIT_MAX", 20)
    CHAT_RATE_LIMIT_WINDOW_MS: int = _positive_int("CHAT_RATE_LIMIT_WINDOW_MS", 60_000)
    CHAT_MAX_MESSAGE_LENGTH: int = _positive_int("CHAT_MAX_MESSAGE_LENGTH", 1200)
