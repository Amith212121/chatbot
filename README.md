# WaveMind Chatbot API

Flask-based medical chatbot API backed by Ollama.

## What this project does
- Accepts user messages through REST endpoints.
- Replies only to medical and healthcare-related questions.
- Supports normal response mode and NDJSON streaming mode.
- Applies per-minute rate limiting.
- Applies per-day quota only for successful medical answers.

## API endpoints
- `GET /api/chat` - health check, provider status, model, and daily limit.
- `POST /api/chat` - non-streaming chat response.
- `POST /api/chat/stream` - streaming NDJSON response.

## Daily limit behavior (important)
- Daily quota is tracked per client IP.
- Quota increases only when a medical question is successfully answered by the model.
- Daily quota storage defaults to SQLite so limits are shared across Gunicorn workers.
- Quota does not increase for:
  - greetings (`hello`, `hi`, etc.)
  - non-medical rejects
  - provider failures/timeouts

### Daily quota storage config
- `CHAT_DAILY_STORE=sqlite` (default): shared daily quota across workers/processes.
- `CHAT_DAILY_STORE=memory`: per-process in-memory quota (not shared across workers).
- `CHAT_DAILY_SQLITE_PATH=./wavemind_daily_quota.sqlite3`: SQLite file path.
- `CHAT_DAILY_SQLITE_TIMEOUT_MS=5000`: SQLite lock wait timeout.

## Project structure
- `app.py` - all API routes and request flow.
- `wavemind/config.py` - all environment-based configuration.
- `wavemind/chat_service.py` - medical scope logic, Ollama integration, rate limiting, and daily limit logic.
- `docs/SERVER_DEPLOYMENT.md` - full production deployment guide.

## Quick local run (venv)
```bash
cd chatbot_optimized
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install Flask flask-cors requests python-dotenv gunicorn
cp .env.example .env
python app.py
```

Windows PowerShell:
```powershell
cd chatbot_optimized
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install Flask flask-cors requests python-dotenv gunicorn
Copy-Item .env.example .env
python app.py
```

## Quick test commands
```bash
curl http://127.0.0.1:8000/api/chat

curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hello"}'

curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"What causes fever?"}'
```

## Conda option
If you prefer Conda:
```bash
conda env create -f environment.yml
conda activate wavemind
python app.py
```

## Deploy on server
Use the complete production guide:
- [`docs/SERVER_DEPLOYMENT.md`](docs/SERVER_DEPLOYMENT.md)

That guide includes:
- server setup
- Ollama install and model pull
- systemd service setup
- Nginx reverse proxy
- SSL with Certbot
- logs, monitoring, troubleshooting, and update workflow
