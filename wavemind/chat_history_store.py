from __future__ import annotations

import datetime
from multiprocessing import Lock

from wavemind.config import Config

try:
    import psycopg2
except Exception:  # pragma: no cover - handled at runtime
    psycopg2 = None


_chat_history_init_lock = Lock()
_chat_history_initialized = False
_chat_history_disabled_reason_logged = False


def _chat_history_enabled() -> bool:
    return Config.CHAT_HISTORY_STORE in {"postgres", "postgresql", "pg"}


def _log_disabled_reason_once(reason: str) -> None:
    global _chat_history_disabled_reason_logged

    if _chat_history_disabled_reason_logged:
        return

    print(f"[WaveMind] Chat history storage disabled: {reason}")
    _chat_history_disabled_reason_logged = True


def _can_store_chat_history() -> bool:
    if not _chat_history_enabled():
        return False

    if not Config.CHAT_HISTORY_DATABASE_URL:
        _log_disabled_reason_once("CHAT_HISTORY_DATABASE_URL is not configured.")
        return False

    if psycopg2 is None:
        _log_disabled_reason_once("psycopg2 is not available in this environment.")
        return False

    return True


def _connect_postgres():
    return psycopg2.connect(  # type: ignore[union-attr]
        Config.CHAT_HISTORY_DATABASE_URL,
        connect_timeout=Config.CHAT_HISTORY_CONNECT_TIMEOUT_S,
    )


def _ensure_chat_history_table() -> None:
    global _chat_history_initialized

    if _chat_history_initialized:
        return

    with _chat_history_init_lock:
        if _chat_history_initialized:
            return

        conn = _connect_postgres()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_history (
                            id BIGSERIAL PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            question TEXT NOT NULL,
                            answer TEXT NOT NULL,
                            asked_at TIMESTAMPTZ NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_chat_history_user_asked_at
                        ON chat_history (user_id, asked_at DESC)
                        """
                    )
        finally:
            conn.close()

        _chat_history_initialized = True


def _normalize_asked_at(asked_at: datetime.datetime | None) -> datetime.datetime:
    if asked_at is None:
        return datetime.datetime.now(datetime.timezone.utc)
    if asked_at.tzinfo is None:
        return asked_at.replace(tzinfo=datetime.timezone.utc)
    return asked_at.astimezone(datetime.timezone.utc)


def store_chat_history(
    *,
    user_id: str,
    question: str,
    answer: str,
    asked_at: datetime.datetime | None = None,
) -> None:
    """
    Persist one chatbot interaction in Postgres.
    This function is best-effort and should not break request handling.
    """
    if not _can_store_chat_history():
        return

    normalized_user = (user_id or "").strip()
    normalized_question = (question or "").strip()
    normalized_answer = (answer or "").strip()
    normalized_asked_at = _normalize_asked_at(asked_at)

    if not normalized_user:
        normalized_user = "anonymous"
    if not normalized_question or not normalized_answer:
        return

    _ensure_chat_history_table()

    conn = _connect_postgres()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_history (user_id, question, answer, asked_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (normalized_user, normalized_question, normalized_answer, normalized_asked_at),
                )
    finally:
        conn.close()
