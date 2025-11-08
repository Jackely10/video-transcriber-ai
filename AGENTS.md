# AGENTS.md Anleitung fuer Codex

## Projekt
Video Transcriber  Deploy auf Railway.

## Build & Tests
- Node/Next:
  - Build: `npm ci && npm run build`
  - Tests (falls vorhanden): `npm test -s || true`
- Python/FastAPI:
  - Install: `pip install -r requirements.txt`
  - Tests (falls vorhanden): `pytest -q || true`

## Start (Prod)
- Node/Next: `npm run start` (muss `$PORT` nutzen; Host `0.0.0.0`)
- FastAPI: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Healthcheck: `GET /healthz`  200

## Env Variablen (in Railway setzen)
- `OPENAI_API_KEY` (Pflicht)
- weitere: `REDIS_URL`, `BASIC_AUTH_USERNAME`, `BASIC_AUTH_PASSWORD`, `ANTHROPIC_API_KEY`, `STRIPE_SECRET_KEY`

## Deploy
- Railway: per `railway.toml` *oder* Dockerfile (siehe Repo).
- GitHub Action optional (`RAILWAY_TOKEN` erforderlich).

## Aufgaben fuer Codex
1. Stack erkennen und Start/Build fixen (PORT/0.0.0.0).
2. Healthcheck `/healthz` anlegen.
3. `railway.toml` oder Dockerfile erzeugen.
4. README aktualisieren (lokal & Railway).
5. PR mit Checkliste eroeffnen.

## Hinweise
- Keine Secrets committen.
- Fuehre vor PR Build + (optionale) Tests in Sandbox aus.
