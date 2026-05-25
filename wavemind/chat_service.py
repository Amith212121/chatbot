from __future__ import annotations

import json
import re
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

# FIX: Tighter system prompt. Fewer tokens = faster first token time.
# "under 80 words" keeps tinyllama from rambling and hitting num_predict cutoff.
MEDICAL_SYSTEM_PROMPT = (
    "You are WaveMind, a medical assistant. "
    "Answer only medical or healthcare questions. "
    "Give general education only, never personal diagnosis or prescriptions. "
    "Never use numbered lists, bullet points, or headings. "
    "Write only in plain flowing sentences. "
    "Keep every final response between 4 and 6 complete sentences."
)

# Wrapping the user question in an explicit instruction template
# forces tinyllama to respect format constraints far more reliably
# than a system prompt alone.
_PROMPT_TEMPLATE = (
    "Answer the following medical question in 4 to 6 complete sentences. "
    "Do NOT use numbered lists, bullet points, or headings. "
    "Write only plain flowing sentences. "
    "Do not repeat, restate, or quote the question in the answer. "
    "Cover: what it is, common causes, what to do, and when to see a doctor. "
    "If it is an emergency (chest pain, stroke, seizure, heavy bleeding), say so in the first sentence.\n\n"
    "Question: {question}\n\n"
    "Answer in 4 to 6 sentences only:"
)
_PROMPT_TEMPLATE_STRICT = (
    "Answer the following medical question in exactly 4 complete sentences. "
    "Use only plain flowing sentences with proper punctuation. "
    "Do NOT use numbering, bullets, headings, labels, or list formatting. "
    "Do not repeat the question.\n\n"
    "Question: {question}\n\n"
    "Answer in exactly 4 complete sentences:"
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
    "follow-up", "follow up", "body"

 # General healthcare
    "physician", "surgeon", "consultation", "checkup", "health issue",
    "medical issue", "medical advice", "urgent care", "caregiver",
    "nurse", "paramedic", "ambulance", "icu", "opd", "ward",

    # Symptoms
    "vomit", "vomiting", "nausea", "fatigue", "tired", "weak",
    "body pain", "joint pain", "back pain", "neck pain",
    "burning sensation", "itching", "irritation", "allergy",
    "sneezing", "runny nose", "blocked nose", "congestion",
    "shortness of breath", "breathlessness", "palpitations",
    "loss of appetite", "weight loss", "weight gain",
    "dehydration", "chills", "night sweats", "blurred vision",
    "vision loss", "double vision", "hearing loss", "ringing ears",
    "ear pain", "tooth pain", "gum bleeding", "mouth ulcer",
    "sore throat", "hoarseness", "constipation", "diarrhea",
    "bloating", "acid reflux", "indigestion", "gastric",
    "heartburn", "vomiting blood", "bloody stool",

    # Emergency / trauma
    "burn", "fracture", "broken bone", "sprain", "dislocation",
    "head injury", "concussion", "poison", "poisoning",
    "snake bite", "dog bite", "animal bite", "electric shock",
    "heat stroke", "sunburn", "unconscious", "collapse",
    "shock", "severe pain", "trauma",

    # Infectious diseases
    "covid", "corona", "tuberculosis", "tb", "malaria",
    "dengue", "typhoid", "hepatitis", "hiv", "aids",
    "fungal infection", "viral infection", "bacterial infection",
    "food poisoning",

    # Chronic diseases
    "arthritis", "asthma", "copd", "obesity",
    "osteoporosis", "cancer", "tumor", "tumour",
    "autoimmune", "fibromyalgia", "anemia",

    # Neurology / EEG related
    "brain activity", "brain signal", "neural activity",
    "neurodegenerative", "neuro disorder", "attention",
    "focus", "concentration", "brain mapping",
    "signal quality", "artifact", "eeg artifact",
    "alpha wave", "beta wave", "gamma wave", "delta wave",
    "cognitive decline", "neurofeedback",

    # Mental health
    "stress", "panic attack", "panic", "mood disorder",
    "bipolar", "ocd", "ptsd", "insomnia",
    "suicidal", "hallucination", "confusion",

    # Women's health
    "period", "menstruation", "pcos", "pcod",
    "ovulation", "fertility", "miscarriage",
    "breast pain", "menopause",

    # Child health
    "pediatric", "child fever", "newborn", "infant",
    "vaccines", "growth", "developmental delay",

    # Diagnostics
    "ecg", "ekg", "ultrasound", "biopsy",
    "cbc", "lipid profile", "thyroid test",
    "glucose", "hb1ac", "oxygen level", "spo2",

    # Medications
    "antibiotic", "painkiller", "paracetamol",
    "ibuprofen", "insulin", "antidepressant",
    "side effects", "drug interaction",

    # Procedures
    "operation", "stitches", "scan report",
    "discharge summary", "medical report",
    "prescribed", "therapy session",

    # Conversational phrases
    "not feeling well", "feeling sick",
    "should i see a doctor", "is this serious",
    "what should i do", "how to treat",
    "home remedy", "medical emergency",
    "health concern", "symptoms of",
    "signs of", "side effect", "recovery time",

    # Common spelling variations
    "x ray", "cat scan", "bp", "sugar level",
    "high sugar", "low sugar", "high bp", "low bp",

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
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")

# ---------------------------------------------------------------------------
# HTTP session --" reuse TCP connection across requests (saves ~100-300ms/req)
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


def _trim_to_last_sentence(text: str) -> str:
    """Cut text at the last complete sentence ending in . ! or ?"""
    match = re.search(r"^([\s\S]*[.!?])(?:\s|$)", text)
    return match.group(1).strip() if match else text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _ends_with_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text.strip()))


def _enforce_max_sentences(text: str, max_sentences: int) -> str:
    # Convert inline numbered markers (" 3. ") into sentence boundaries first.
    normalized = re.sub(r"\s+\d+\.\s+(?=[A-Za-z])", ". ", text)
    sentences = [re.sub(r"\s+", " ", s).strip() for s in _SENTENCE_RE.findall(normalized)]
    sentences = [
        s for s in sentences
        if not re.fullmatch(r"\d+[.!?]?", re.sub(r"\s+", " ", s).strip())
    ]
    if not sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


def _count_complete_sentences(text: str) -> int:
    normalized = re.sub(r"\s+\d+\.\s+(?=[A-Za-z])", ". ", text)
    sentences = [re.sub(r"\s+", " ", s).strip() for s in _SENTENCE_RE.findall(normalized)]
    sentences = [
        s for s in sentences
        if not re.fullmatch(r"\d+[.!?]?", re.sub(r"\s+", " ", s).strip())
    ]
    return len(sentences)


def _sanitize_model_reply(text: str, question: str) -> str:
    """Remove prompt-echo artifacts like 'Question:' and 'Sentence 1:'."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    normalized_question = re.sub(r"\s+", " ", question).strip().rstrip(" .!?")

    for _ in range(2):
        cleaned = re.sub(r"^(?:answer|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        had_question_prefix = bool(
            re.match(r"^(?:question|user question|query)\s*:\s*", cleaned, flags=re.IGNORECASE)
        )
        cleaned = re.sub(r"^(?:question|user question|query)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()

        if normalized_question:
            cleaned = re.sub(
                rf"^[\"']?{re.escape(normalized_question)}[\"']?[.!?]*\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        if had_question_prefix and "?" in cleaned:
            first, rest = cleaned.split("?", 1)
            if len(first.split()) <= 30:
                cleaned = rest.strip()

    cleaned = re.sub(
        r"(?:(?<=^)|(?<=[.!?]\s))(?:sentence\s*)?\d+\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(?:(?<=^)|(?<=[.!?]\s))\d+\.\s+", "", cleaned)
    cleaned = re.sub(r"\s+\d+[.!?]?\s*$", "", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


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
    """
    Single-shot Ollama call. No continuations --" each continuation
    added 10-15s of latency. num_predict=120 + num_ctx=512 ensures
    the model finishes within budget without mid-sentence cuts.
    """
    prompt = _PROMPT_TEMPLATE.format(question=message)
    data = _ollama_generate(prompt)
    raw = (data.get("response") or "").strip()

    if not raw:
        raise ChatServiceError("PROVIDER_EMPTY_RESPONSE", "Ollama returned an empty reply", 502)

    def _finalize(raw_text: str) -> str:
        reply_raw = _sanitize_model_reply(raw_text, message)
        reply_clean = _enforce_word_limit(reply_raw, Config.OLLAMA_MAX_WORDS)
        if not _ends_with_sentence(reply_clean):
            reply_clean = _trim_to_last_sentence(reply_clean)
        return _enforce_max_sentences(reply_clean, 6)

    reply = _finalize(raw)
    if _count_complete_sentences(reply) < 4:
        strict_prompt = _PROMPT_TEMPLATE_STRICT.format(question=message)
        strict_data = _ollama_generate(strict_prompt)
        strict_raw = (strict_data.get("response") or "").strip()
        if strict_raw:
            reply = _finalize(strict_raw)

    if not reply:
        raise ChatServiceError("PROVIDER_EMPTY_RESPONSE", "Ollama returned an empty reply", 502)

    return {"reply": reply, "matchedIntent": f"ollama:{Config.OLLAMA_MODEL}"}


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_rate_buckets: dict[str, dict] = {}


def _check_rate_limit(client_key: str) -> bool:
    """Return True if the client is rate-limited."""
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
    Stream response tokens. This is the recommended endpoint for the UI --"
    the user sees the first words in ~1-2s while the rest generates.
    """
    trimmed = message.strip()
    normalized = _normalize(trimmed)

    if not normalized:
        raise ChatServiceError("MESSAGE_REQUIRED", "message is required", 400)

    if _is_greeting(normalized):
        yield {"type": "token", "content": GREETING_REPLY}
        yield {"type": "done", "matchedIntent": "greeting"}
        return

    if not _is_medical_related(normalized):
        yield {"type": "token", "content": MEDICAL_ONLY_REPLY}
        yield {"type": "done", "matchedIntent": "medical_scope_decline"}
        return

    if Config.CHAT_PROVIDER == "ollama":
        try:
            accumulated = ""
            max_words = Config.OLLAMA_MAX_WORDS
            emitted = ""
            stream_prompt = _PROMPT_TEMPLATE.format(question=trimmed)

            for token in _ollama_stream(stream_prompt):
                accumulated += token
                candidate = _sanitize_model_reply(accumulated, trimmed)
                candidate = _enforce_word_limit(candidate, max_words)
                current_words = _word_count(candidate)

                if current_words >= max_words:
                    final = candidate
                    if not _ends_with_sentence(final):
                        final = _trim_to_last_sentence(final)
                    final = _enforce_max_sentences(final, 6)
                    if final and final != emitted:
                        if final.startswith(emitted):
                            delta = final[len(emitted):]
                            if delta:
                                yield {"type": "token", "content": delta}
                        else:
                            yield {"type": "token", "content": final}
                    yield {"type": "done", "matchedIntent": f"ollama:{Config.OLLAMA_MODEL}:word_limit"}
                    return

                if candidate and candidate != emitted:
                    if candidate.startswith(emitted):
                        delta = candidate[len(emitted):]
                        if delta:
                            yield {"type": "token", "content": delta}
                    else:
                        # Rare fallback when cleanup rewrites earlier text.
                        yield {"type": "token", "content": candidate}
                    emitted = candidate

            final_clean = _sanitize_model_reply(accumulated, trimmed)
            final_clean = _enforce_word_limit(final_clean, max_words)
            if final_clean and not _ends_with_sentence(final_clean):
                final_clean = _trim_to_last_sentence(final_clean)
            final_clean = _enforce_max_sentences(final_clean, 6)
            if final_clean and final_clean != emitted:
                if final_clean.startswith(emitted):
                    delta = final_clean[len(emitted):]
                    if delta:
                        yield {"type": "token", "content": delta}
                else:
                    yield {"type": "token", "content": final_clean}

            yield {"type": "done", "matchedIntent": f"ollama:{Config.OLLAMA_MODEL}"}

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
