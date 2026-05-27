from __future__ import annotations

import datetime
import json
import re
import sqlite3
import time
import threading
from typing import Generator

import requests

from wavemind.config import Config


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ChatServiceError(Exception):
    """Raised when the chat service cannot fulfil a request."""

    def __init__(self, code: str, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------
# Static reply strings
# ---------------------------------------------------------------------------

MEDICAL_SYSTEM_PROMPT = (
    "You are WaveMind, a medical assistant. "
    "Answer only medical or healthcare questions. "
    "Give general education only, never personal diagnosis or prescriptions. "
    "Never use numbered lists, bullet points, or headings. "
    "Write only in plain flowing sentences."
)

_PROMPT_TEMPLATE = (
    "Answer the following medical question in exactly 4 to 5 complete sentences. "
    "Do NOT use numbered lists, bullet points, or headings. "
    "Write only plain flowing sentences. "
    "Cover: what it is, common causes, what to do, and when to see a doctor. "
    "If it is an emergency (chest pain, stroke, seizure, heavy bleeding), say so in the first sentence.\n\n"
    "Question: {question}\n\n"
    "Answer in 4 to 5 sentences only:"
)

GREETING_REPLY = (
    "Hi, I'm WaveMind. I can help with general medical questions. "
    "Please don't share personal identifying medical details."
)

MEDICAL_ONLY_REPLY = (
    "I can only help with medical, healthcare, first-aid, injury-care, neurology, "
    "EEG, patient-report, appointment, or doctor-workflow questions. "
    "Please ask a medical or healthcare-related question."
)

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

MEDICAL_SCOPE_KEYWORDS: list[str] = [
    "medical", "health", "healthcare", "doctor", "clinic", "hospital",
    "appointment", "patient", "symptom", "diagnosis", "treatment",
    "medicine", "medication", "prescription", "dose", "dosage", "tablet",
    "capsule", "emergency", "first aid", "pain", "fever", "cough", "cold",
    "flu", "infection", "disease", "illness", "condition", "rash", "skin",
    "cut", "cuts", "wound", "deep cut", "injury", "injured", "bleed",
    "bleeds", "bleeding", "blood", "bandage", "dressing", "puncture",
    "laceration", "scrape", "scratch", "rust", "rusty", "iron rod", "metal",
    "tetanus", "leprosy", "hansen", "hansen's disease", "swelling",
    "breathing", "chest", "heart", "cardiac", "blood pressure", "hypertension",
    "cholesterol", "diabetes", "sugar", "thyroid", "kidney", "liver",
    "stomach", "abdomen", "gut", "urine", "pregnancy", "pregnant", "vaccine",
    "vaccination", "lab", "blood test", "scan", "xray", "x-ray", "mri", "ct",
    "surgery", "mental health", "anxiety", "depression", "therapy", "diet",
    "nutrition", "exercise", "lifestyle", "smoking", "alcohol", "weight",
    "brain", "neurology", "neurologist", "nerve", "nervous system", "eeg",
    "screening", "report", "seizure", "epilepsy", "stroke", "headache",
    "migraine", "dementia", "alzheimer", "parkinson", "tremor", "weakness",
    "numbness", "paralysis", "dizziness", "vertigo", "fainting", "memory",
    "cognitive", "sleep", "brainwave", "theta", "alpha", "risk",
    "follow-up", "follow up",
]

GENERAL_TOPIC_KEYWORDS: list[str] = [
    "algebra", "math", "mathematics", "equation", "geometry", "calculus",
    "physics", "history", "geography", "politics", "sports", "cricket",
    "football", "movie", "song", "music", "recipe", "cooking", "stock",
    "crypto", "finance", "investment", "coding", "programming", "javascript",
    "python", "java", "essay", "poem", "joke", "weather", "capital city",
    "translate", "grammar",
]

EXPLICIT_MEDICAL_CONTEXT: frozenset[str] = frozenset([
    "medical", "health", "healthcare", "patient", "doctor",
    "clinic", "hospital", "eeg", "neurology", "symptom", "treatment", "diagnosis",
])

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|namaste|thanks|thank you)[\s!.?]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# HTTP session — reuse TCP connection across requests (saves ~100-300ms/req)
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.strip().lower()


def _includes_keyword(message: str, keyword: str) -> bool:
    kw = keyword.lower()
    if " " in kw:
        return kw in message
    return bool(re.search(rf"\b{re.escape(kw)}\b", message))


def _is_greeting(message: str) -> bool:
    return bool(_GREETING_RE.match(message))


def _is_medical_related(message: str) -> bool:
    has_medical = any(_includes_keyword(message, kw) for kw in MEDICAL_SCOPE_KEYWORDS)
    if not has_medical:
        return False
    has_general = any(_includes_keyword(message, kw) for kw in GENERAL_TOPIC_KEYWORDS)
    if not has_general:
        return True
    return any(_includes_keyword(message, kw) for kw in EXPLICIT_MEDICAL_CONTEXT)


def counts_toward_daily_limit(message: str) -> bool:
    """
    Daily quota should only be consumed by medical-scope questions.
    Greetings and non-medical requests are always free.
    """
    normalized = _normalize(message)
    if not normalized:
        return False
    if _is_greeting(normalized):
        return False
    return _is_medical_related(normalized)


def _trim_to_last_sentence(text: str) -> str:
    """Cut text at the last complete sentence ending in . ! or ?"""
    match = re.search(r"^([\s\S]*[.!?])(?:\s|$)", text)
    return match.group(1).strip() if match else text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _ends_with_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text.strip()))


def _enforce_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    truncated = " ".join(words[:max_words])
    return _trim_to_last_sentence(truncated)


# ---------------------------------------------------------------------------
# Ollama integration
# ---------------------------------------------------------------------------

def _build_ollama_body(prompt: str, stream: bool) -> dict:
    return {
        "model": Config.OLLAMA_MODEL,
        "system": MEDICAL_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": Config.OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": Config.OLLAMA_TEMPERATURE,
            "num_predict": Config.OLLAMA_NUM_PREDICT,
            "num_ctx": Config.OLLAMA_NUM_CTX,
            "num_thread": Config.OLLAMA_NUM_THREAD,
        },
    }


def _ollama_generate(prompt: str) -> dict:
    """Call Ollama /api/generate (non-streaming). Uses persistent session."""
    timeout_s = Config.OLLAMA_TIMEOUT_MS / 1000.0
    url = f"{Config.OLLAMA_BASE_URL}/api/generate"

    try:
        resp = _session.post(url, json=_build_ollama_body(prompt, stream=False), timeout=timeout_s)
    except requests.Timeout:
        raise ChatServiceError(
            "PROVIDER_TIMEOUT",
            f"Ollama request timed out after {Config.OLLAMA_TIMEOUT_MS}ms",
            504,
        )
    except requests.ConnectionError:
        raise ChatServiceError(
            "PROVIDER_UNAVAILABLE",
            "Cannot connect to Ollama. Is it running?",
            503,
        )

    if not resp.ok:
        raise ChatServiceError("PROVIDER_ERROR", f"Ollama returned HTTP {resp.status_code}", 502)

    return resp.json()


def _ollama_stream(prompt: str) -> Generator[str, None, None]:
    """Stream tokens from Ollama /api/generate via persistent session."""
    url = f"{Config.OLLAMA_BASE_URL}/api/generate"
    timeout_s = Config.OLLAMA_TIMEOUT_MS / 1000.0

    try:
        with _session.post(
            url,
            json=_build_ollama_body(prompt, stream=True),
            timeout=timeout_s,
            stream=True,
        ) as resp:
            if not resp.ok:
                raise ChatServiceError("PROVIDER_ERROR", f"Ollama returned HTTP {resp.status_code}", 502)
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except requests.Timeout:
        raise ChatServiceError(
            "PROVIDER_TIMEOUT",
            f"Ollama request timed out after {Config.OLLAMA_TIMEOUT_MS}ms",
            504,
        )
    except requests.ConnectionError:
        raise ChatServiceError("PROVIDER_UNAVAILABLE", "Cannot connect to Ollama. Is it running?", 503)


def _get_ollama_reply(message: str) -> dict:
    prompt = _PROMPT_TEMPLATE.format(question=message)
    data = _ollama_generate(prompt)
    raw = (data.get("response") or "").strip()

    if not raw:
        raise ChatServiceError("PROVIDER_EMPTY_RESPONSE", "Ollama returned an empty reply", 502)

    reply_raw = re.sub(r"\s+", " ", raw).strip()
    reply = _enforce_word_limit(reply_raw, Config.OLLAMA_MAX_WORDS)

    if not _ends_with_sentence(reply):
        reply = _trim_to_last_sentence(reply)

    if not reply:
        raise ChatServiceError("PROVIDER_EMPTY_RESPONSE", "Ollama returned an empty reply", 502)

    return {"reply": reply, "matchedIntent": f"ollama:{Config.OLLAMA_MODEL}"}


# ---------------------------------------------------------------------------
# Per-minute rate limiting (burst protection — unchanged)
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_rate_buckets: dict[str, dict] = {}


def _check_rate_limit(client_key: str) -> bool:
    """Return True if the client exceeds the per-minute burst limit."""
    now = time.time() * 1000  # ms
    window_ms = Config.CHAT_RATE_LIMIT_WINDOW_MS
    max_req = Config.CHAT_RATE_LIMIT_MAX

    with _rate_lock:
        stale = [k for k, v in _rate_buckets.items() if v["reset_at"] <= now]
        for k in stale:
            del _rate_buckets[k]

        bucket = _rate_buckets.get(client_key)

        if not bucket or bucket["reset_at"] <= now:
            _rate_buckets[client_key] = {"count": 1, "reset_at": now + window_ms}
            return False

        bucket["count"] += 1
        return bucket["count"] > max_req


# ---------------------------------------------------------------------------
# Daily question limit
# ---------------------------------------------------------------------------

_daily_lock = threading.Lock()
_daily_buckets: dict[str, dict] = {}  # in-memory fallback: client_key -> {count, reset_at}
_daily_sqlite_init_lock = threading.Lock()
_daily_sqlite_initialized = False
_daily_sqlite_fallback_logged = False


def _next_midnight_ms() -> float:
    """Return Unix timestamp (ms) of the next midnight in server local time."""
    now = datetime.datetime.now()
    tomorrow = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return tomorrow.timestamp() * 1000


def _today_key() -> str:
    """Current date key in server local timezone."""
    return datetime.date.today().isoformat()


def _use_sqlite_daily_store() -> bool:
    return Config.CHAT_DAILY_STORE in {"sqlite", "sqlite3"}


def _open_daily_sqlite() -> sqlite3.Connection:
    timeout_s = Config.CHAT_DAILY_SQLITE_TIMEOUT_MS / 1000.0
    conn = sqlite3.connect(
        Config.CHAT_DAILY_SQLITE_PATH,
        timeout=timeout_s,
        isolation_level=None,  # autocommit mode; we manage explicit transactions.
        check_same_thread=False,
    )
    conn.execute(f"PRAGMA busy_timeout = {Config.CHAT_DAILY_SQLITE_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure_daily_sqlite() -> None:
    global _daily_sqlite_initialized

    if _daily_sqlite_initialized:
        return

    with _daily_sqlite_init_lock:
        if _daily_sqlite_initialized:
            return
        conn = _open_daily_sqlite()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_quota_usage (
                    date_key TEXT NOT NULL,
                    client_key TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (date_key, client_key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_quota_usage_updated_at
                ON daily_quota_usage(updated_at_ms)
                """
            )
            _daily_sqlite_initialized = True
        finally:
            conn.close()


def _check_daily_limit_sqlite(client_key: str) -> tuple[bool, int]:
    """
    Increment the shared daily counter in SQLite and return
    (is_limited, questions_remaining).
    """
    _ensure_daily_sqlite()

    daily_max = Config.CHAT_DAILY_LIMIT
    today = _today_key()
    now_ms = int(time.time() * 1000)

    conn = _open_daily_sqlite()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT count FROM daily_quota_usage WHERE date_key = ? AND client_key = ?",
            (today, client_key),
        ).fetchone()

        if row is None:
            new_count = 1
            conn.execute(
                """
                INSERT INTO daily_quota_usage (date_key, client_key, count, updated_at_ms)
                VALUES (?, ?, ?, ?)
                """,
                (today, client_key, new_count, now_ms),
            )
        else:
            current_count = int(row[0])
            if current_count >= daily_max:
                conn.commit()
                return True, 0

            new_count = current_count + 1
            conn.execute(
                """
                UPDATE daily_quota_usage
                SET count = ?, updated_at_ms = ?
                WHERE date_key = ? AND client_key = ?
                """,
                (new_count, now_ms, today, client_key),
            )

        # Keep table small by removing stale days.
        conn.execute("DELETE FROM daily_quota_usage WHERE date_key <> ?", (today,))
        conn.commit()
        remaining = daily_max - new_count
        return False, max(remaining, 0)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _get_daily_remaining_sqlite(client_key: str) -> int:
    """Read remaining daily quota from SQLite without consuming it."""
    _ensure_daily_sqlite()

    daily_max = Config.CHAT_DAILY_LIMIT
    today = _today_key()

    conn = _open_daily_sqlite()
    try:
        row = conn.execute(
            "SELECT count FROM daily_quota_usage WHERE date_key = ? AND client_key = ?",
            (today, client_key),
        ).fetchone()
        if not row:
            return daily_max
        used = int(row[0])
        return max(daily_max - used, 0)
    finally:
        conn.close()


def _check_daily_limit_memory(client_key: str) -> tuple[bool, int]:
    """
    In-memory daily limit fallback.
    Note: counters are per-process and reset on restart.
    """
    now = time.time() * 1000
    daily_max = Config.CHAT_DAILY_LIMIT

    with _daily_lock:
        stale = [k for k, v in _daily_buckets.items() if v["reset_at"] <= now]
        for k in stale:
            del _daily_buckets[k]

        bucket = _daily_buckets.get(client_key)
        if not bucket or bucket["reset_at"] <= now:
            bucket = {"count": 0, "reset_at": _next_midnight_ms()}
            _daily_buckets[client_key] = bucket

        if bucket["count"] >= daily_max:
            return True, 0

        bucket["count"] += 1
        remaining = daily_max - bucket["count"]
        return False, max(remaining, 0)


def _get_daily_remaining_memory(client_key: str) -> int:
    """In-memory daily remaining fallback."""
    now = time.time() * 1000
    daily_max = Config.CHAT_DAILY_LIMIT

    with _daily_lock:
        stale = [k for k, v in _daily_buckets.items() if v["reset_at"] <= now]
        for k in stale:
            del _daily_buckets[k]

        bucket = _daily_buckets.get(client_key)
        if not bucket:
            return daily_max
        return max(daily_max - bucket["count"], 0)


def _check_daily_limit(client_key: str) -> tuple[bool, int]:
    global _daily_sqlite_fallback_logged

    if _use_sqlite_daily_store():
        try:
            return _check_daily_limit_sqlite(client_key)
        except Exception as exc:
            if not _daily_sqlite_fallback_logged:
                print(f"[WaveMind] SQLite daily quota unavailable, falling back to memory: {exc}")
                _daily_sqlite_fallback_logged = True

    return _check_daily_limit_memory(client_key)


def _get_daily_remaining(client_key: str) -> int:
    global _daily_sqlite_fallback_logged

    if _use_sqlite_daily_store():
        try:
            return _get_daily_remaining_sqlite(client_key)
        except Exception as exc:
            if not _daily_sqlite_fallback_logged:
                print(f"[WaveMind] SQLite daily quota unavailable, falling back to memory: {exc}")
                _daily_sqlite_fallback_logged = True

    return _get_daily_remaining_memory(client_key)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chat_reply(message: str) -> dict:
    trimmed = message.strip()
    normalized = _normalize(trimmed)

    if not normalized:
        raise ChatServiceError("MESSAGE_REQUIRED", "message is required", 400)

    if _is_greeting(normalized):
        return {"reply": GREETING_REPLY, "matchedIntent": "greeting"}

    if not _is_medical_related(normalized):
        return {"reply": MEDICAL_ONLY_REPLY, "matchedIntent": "medical_scope_decline"}

    if Config.CHAT_PROVIDER == "ollama":
        try:
            return _get_ollama_reply(trimmed)
        except ChatServiceError:
            raise
        except Exception as exc:
            raise ChatServiceError("PROVIDER_UNAVAILABLE", "Chat provider unavailable", 503) from exc

    raise ChatServiceError("PROVIDER_NOT_CONFIGURED", "Only CHAT_PROVIDER=ollama is supported", 503)


def stream_chat_reply(message: str) -> Generator[dict, None, None]:
    """
    Stream response tokens. This is the recommended endpoint for the UI —
    the user sees the first words in ~1-2s while the rest generates.
    """
    trimmed = message.strip()
    normalized = _normalize(trimmed)

    if not normalized:
        raise ChatServiceError("MESSAGE_REQUIRED", "message is required", 400)

    if _is_greeting(normalized):
        yield {"type": "token", "content": GREETING_REPLY}
        yield {"type": "done", "matchedIntent": "greeting", "finalReply": GREETING_REPLY}
        return

    if not _is_medical_related(normalized):
        yield {"type": "token", "content": MEDICAL_ONLY_REPLY}
        yield {"type": "done", "matchedIntent": "medical_scope_decline", "finalReply": MEDICAL_ONLY_REPLY}
        return

    if Config.CHAT_PROVIDER == "ollama":
        try:
            accumulated = ""
            max_words = Config.OLLAMA_MAX_WORDS
            stream_prompt = _PROMPT_TEMPLATE.format(question=trimmed)

            for token in _ollama_stream(stream_prompt):
                accumulated += token
                current_words = _word_count(accumulated)

                if current_words >= max_words:
                    final = _enforce_word_limit(accumulated, max_words)
                    if not _ends_with_sentence(final):
                        final = _trim_to_last_sentence(final)
                    yield {"type": "token", "content": final}
                    yield {
                        "type": "done",
                        "matchedIntent": f"ollama:{Config.OLLAMA_MODEL}:word_limit",
                        "finalReply": final,
                    }
                    return
                else:
                    yield {"type": "token", "content": token}

            if accumulated and not _ends_with_sentence(accumulated):
                clean = _trim_to_last_sentence(accumulated)
                yield {"type": "token", "content": clean}
                accumulated = clean

            final_reply = re.sub(r"\s+", " ", accumulated).strip()
            if not final_reply:
                raise ChatServiceError("PROVIDER_EMPTY_RESPONSE", "Ollama returned an empty reply", 502)

            yield {
                "type": "done",
                "matchedIntent": f"ollama:{Config.OLLAMA_MODEL}",
                "finalReply": final_reply,
            }

        except ChatServiceError:
            raise
        except Exception as exc:
            raise ChatServiceError("PROVIDER_UNAVAILABLE", "Chat provider unavailable", 503) from exc
        return

    raise ChatServiceError("PROVIDER_NOT_CONFIGURED", "Only CHAT_PROVIDER=ollama is supported", 503)


def warmup_chat_provider() -> None:
    """Fire a silent warmup request so the model is pre-loaded into RAM."""
    if Config.CHAT_PROVIDER != "ollama":
        return
    try:
        _ollama_generate("Reply with one word only: ready.")
        print(f"[WaveMind] Ollama model warmed: {Config.OLLAMA_MODEL}")
    except Exception as exc:
        print(f"[WaveMind] Ollama warmup skipped: {exc}")
