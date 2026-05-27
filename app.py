"""
app.py - WaveMind Chatbot API
------------------------------
All routes are defined here. No Blueprint. No routes.py.

GET  /api/chat           - health check
POST /api/chat           - single complete reply
POST /api/chat/stream    - NDJSON streaming reply

Daily limit: CHAT_DAILY_LIMIT medical questions per IP per day (resets at midnight).
Successful responses include `questionsRemaining` for the frontend counter.
"""

from __future__ import annotations

import datetime
import json
import threading

import requests as _requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from wavemind.chat_service import (
    ChatServiceError,
    _check_daily_limit,
    _check_rate_limit,
    _get_daily_remaining,
    counts_toward_daily_limit,
    get_chat_reply,
    stream_chat_reply,
    warmup_chat_provider,
)
from wavemind.chat_history_store import store_chat_history
from wavemind.config import Config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Warm up Ollama model in background so first request is fast
threading.Thread(target=warmup_chat_provider, daemon=True).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client_key() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _validate_message(message: str):
    """Return (error_dict, status) if invalid, else None."""
    if not message:
        return {"error": "message is required", "code": "MESSAGE_REQUIRED"}, 400
    if len(message) > Config.CHAT_MAX_MESSAGE_LENGTH:
        return {
            "error": f"Message must be {Config.CHAT_MAX_MESSAGE_LENGTH} characters or fewer.",
            "code": "MESSAGE_TOO_LONG",
        }, 413
    return None


def _normalize_user_id(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    return cleaned[:200]


def _get_user_id(body: dict, client_key: str) -> str:
    candidates = [
        body.get("userId"),
        body.get("user_id"),
        request.headers.get("X-User-Id", ""),
    ]
    for value in candidates:
        normalized = _normalize_user_id(value)
        if normalized:
            return normalized
    return f"anonymous:{client_key}"


# ---------------------------------------------------------------------------
# GET /api/chat - health check
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["GET"])
def health():
    """Confirms Flask is up and tests Ollama connectivity."""
    ollama_status = "unreachable"
    try:
        resp = _requests.get(Config.OLLAMA_BASE_URL, timeout=3)
        if resp.ok:
            ollama_status = "connected"
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "model": Config.OLLAMA_MODEL,
        "ollama": ollama_status,
        "provider": Config.CHAT_PROVIDER,
        "port": Config.PORT,
        "dailyLimit": Config.CHAT_DAILY_LIMIT,
    }), 200


# ---------------------------------------------------------------------------
# POST /api/chat - single complete reply
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    raw_message = body.get("text") or body.get("message", "")
    message = raw_message.strip() if isinstance(raw_message, str) else ""

    err = _validate_message(message)
    if err:
        return jsonify(err[0]), err[1]

    client_key = _get_client_key()
    user_id = _get_user_id(body, client_key)
    asked_at = datetime.datetime.now(datetime.timezone.utc)

    # 1. Per-minute burst check (unchanged)
    if _check_rate_limit(client_key):
        return jsonify({
            "error": "Too many chat requests. Please wait a moment and try again.",
            "code": "RATE_LIMITED",
        }), 429

    # 2. Daily question limit applies only to medical-scope messages.
    should_count = counts_toward_daily_limit(message)
    remaining = _get_daily_remaining(client_key)
    if should_count and remaining <= 0:
        return jsonify({
            "error": "You've used all your questions for today. Come back tomorrow!",
            "code": "DAILY_LIMIT_REACHED",
            "questionsRemaining": 0,
        }), 429

    try:
        result = get_chat_reply(message)
        if should_count and result.get("matchedIntent", "").startswith("ollama:"):
            is_limited, remaining = _check_daily_limit(client_key)
            if is_limited:
                remaining = 0

        try:
            store_chat_history(
                user_id=user_id,
                question=message,
                answer=result["reply"],
                asked_at=asked_at,
            )
        except Exception as exc:
            print(f"[WaveMind] Failed to store chat history: {exc}")

        return jsonify({
            "text": result["reply"],
            "matchedIntent": result["matchedIntent"],
            "questionsRemaining": remaining,
        }), 200
    except ChatServiceError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status


# ---------------------------------------------------------------------------
# POST /api/chat/stream - NDJSON streaming reply
# ---------------------------------------------------------------------------

@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    body = request.get_json(silent=True) or {}
    raw_message = body.get("text") or body.get("message", "")
    message = raw_message.strip() if isinstance(raw_message, str) else ""

    err = _validate_message(message)
    if err:
        return jsonify(err[0]), err[1]

    client_key = _get_client_key()
    user_id = _get_user_id(body, client_key)
    asked_at = datetime.datetime.now(datetime.timezone.utc)

    # 1. Per-minute burst check (unchanged)
    if _check_rate_limit(client_key):
        return jsonify({
            "error": "Too many chat requests. Please wait a moment and try again.",
            "code": "RATE_LIMITED",
        }), 429

    # 2. Daily question limit applies only to medical-scope messages.
    should_count = counts_toward_daily_limit(message)
    remaining = _get_daily_remaining(client_key)
    if should_count and remaining <= 0:
        return jsonify({
            "error": "You've used all your questions for today. Come back tomorrow!",
            "code": "DAILY_LIMIT_REACHED",
            "questionsRemaining": 0,
        }), 429

    def generate():
        try:
            for chunk in stream_chat_reply(message):
                if chunk.get("type") == "done":
                    final_reply = (chunk.pop("finalReply", "") or "").strip()
                    if should_count and chunk.get("matchedIntent", "").startswith("ollama:"):
                        is_limited, updated_remaining = _check_daily_limit(client_key)
                        chunk["questionsRemaining"] = 0 if is_limited else updated_remaining
                    else:
                        chunk["questionsRemaining"] = remaining

                    if final_reply:
                        try:
                            store_chat_history(
                                user_id=user_id,
                                question=message,
                                answer=final_reply,
                                asked_at=asked_at,
                            )
                        except Exception as exc:
                            print(f"[WaveMind] Failed to store chat history: {exc}")

                yield json.dumps(chunk) + "\n"
        except ChatServiceError as exc:
            yield json.dumps({"type": "error", "error": str(exc), "code": exc.code}) + "\n"
        except Exception:
            yield json.dumps({"type": "error", "error": "Internal server error", "code": "PROVIDER_ERROR"}) + "\n"

    return Response(
        stream_with_context(generate()),
        status=200,
        headers={
            "Content-Type": "application/x-ndjson; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=Config.PORT,
        debug=Config.DEBUG,
    )
