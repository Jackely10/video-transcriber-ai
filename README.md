# Video Transcriber

Command line helper that downloads audio from a YouTube or Facebook video and generates a transcript with faster-whisper. Whisper kann entweder direkt ins Englische (Task `translate`) oder in der Originalsprache (`transcribe`) arbeiten; weitere Uebersetzungen sind derzeit deaktiviert.

## Requirements

- Python 3.9 oder neuer auf `PATH`
- [ffmpeg](https://ffmpeg.org/download.html) auf `PATH`
- Python packages:
  ```bash
  pip install yt-dlp faster-whisper transformers sentencepiece torch
  ```
- Optional fuer Web/Jobs: `pip install flask redis rq` sowie eine laufende Redis-Instanz (Standard: `redis://localhost:6379/0`).

## Usage

Standardausgabe ist die Whisper-Translate-Version (Englisch). Wenn der interaktive Prompt erscheint, tippe `source`, um das Original-Transkript ebenfalls auszugeben.

```bash
python video_transcriber.py "https://www.youtube.com/watch?v=VIDEO_ID" --show-console-output
```

Example (Originalsprache) mit Textdatei:

```bash
python video_transcriber.py "https://www.youtube.com/watch?v=VIDEO_ID" --whisper-task transcribe --text-output source.txt
```

Key flags:
- `--languages`: deprecated. Nur `source` wird akzeptiert, um das Original-Transkript zusaetzlich auszugeben (alle anderen Angaben werden ignoriert).
- `--model-size`: retained for backwards compatibility; die Laufzeit waehlt automatisch `medium` auf GPU und `tiny` auf CPU.
- `--device`: deprecated; Device-Erkennung geschieht automatisch.
- `--compute-type`: deprecated; das Preset bestimmt die Genauigkeit.
- `--source-language`: optionaler ISO-Hinweis (`en`, `ar`, ...). Whisper kann automatisch erkennen.
- `--output`: Pfad fuer eine JSON-Datei mit Metadaten und Transkript.
- `--text-output`: Pfad fuer eine UTF-8-Textdatei mit dem Transkript.
- `--whisper-task`: `translate` fuer englische Ausgabe, `transcribe` fuer Originalsprache.
- `--show-console-output`: erzwingt die Ausgabe im Terminal, auch wenn Dateien geschrieben werden.

## Web Interface (Optional)

A small Flask server powers a local web UI mit Formular fuer URL, Whisper-Task sowie erweiterte Optionen. Externe Uebersetzungen sind deaktiviert; der Fokus liegt auf Transkription.

### Install (once)

```bash
pip install flask
```

### Start the server

```bash
python server.py
```

Dann `http://127.0.0.1:5000/` im Browser oeffnen. Die Seite ermoeglicht:

- Einfuegen eines YouTube- oder Facebook-Links.
- Auswahl des Whisper-Tasks (`translate` fuer englische Ausgabe oder `transcribe` fuer Originalsprache) sowie optionaler Modell-/Compute-Optionen.
- Start des Jobs, Anzeige des Status und Download der Artefakte (TXT/SRT/VTT).

> Hinweis: `translate` nutzt die Whisper-eigene englische Ausgabe, `transcribe` behaelt die Ursprungssprache. Weitere Uebersetzungen sind deaktiviert.

## Asynchrone Jobs

1. Redis starten (z. B. `redis-server`).
2. Worker starten:
   ```bash
   python worker.py
   ```
3. Flask-App starten (`python server.py`) und Jobs ueber UI oder API einreichen.

### REST-Endpunkte

- `POST /jobs` - Body (JSON) zum Beispiel:
  ```json
  {
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "whisperTask": "translate",
    "includeSource": true,
    "summary": true,
    "profile": "fast"
  }
  ```
  Antwort: `{"job_id": "<id>"}` mit Status `202`.
- `GET /jobs/<id>` - liefert Status, Fortschritt, Meta-Daten, RTF und Ausgabe-Pfade.
- `GET /jobs/<id>/files/<path>` - stellt erzeugte TXT/SRT/VTT-Dateien bereit.
- `GET /jobs/<id>/summary` - gibt die bereitgestellte Zusammenfassung als `text/plain` zurueck (404, wenn keine Zusammenfassung angefordert wurde oder der Job noch laeuft).
- `GET /jobs/<id>/facts` - liefert bei aktiviertem Faktencheck (`FACT_CHECK=1`) strukturierte Bewertungsdaten als JSON.
- `GET /api/config` - gibt die Server-Defaults (`summary_default_on`, `fact_check_enabled`) als JSON zurueck.
- `GET /healthz` - liefert `{status, redis, worker_queue_len}` und ist fuer Healthchecks ohne Authentifizierung gedacht.
- `POST /selftest` - enqueued einen internen `ping`-Job auf der Default-Queue (Basic Auth erforderlich, falls gesetzt) und liefert `{"job_id": ...}`.
- `GET /job/<id>` - gibt den Job-Status sowie ggf. das Ergebnis zurueck; meldet `404`, wenn die ID unbekannt ist.

#### Zusammenfassung & Faktencheck

- Summaries sind jetzt standardmaessig aktiviert. Verwende das Feld `summary: false` (oder `add_summary: false`) im Request oder setze `SUMMARY_OFF=1`, um die automatische Zusammenfassung zu deaktivieren.
- Der API-Body akzeptiert sowohl `summary` als auch `add_summary`; fehlt beides, greift der Default aus `SUMMARY_OFF`.
- Setze `FACT_CHECK=1`, um strukturierte Faktencheck-Daten zu erzeugen und den Endpoint `/jobs/<id>/facts` zu aktivieren.
- Das Frontend liest die Einstellungen ueber `GET /api/config`; eigene Clients koennen denselben Endpoint wiederverwenden.
- Fuer AI-Summaries musst du `AI_PROVIDER` auf `openai` oder `anthropic` setzen und den passenden Key (`OPENAI_API_KEY` bzw. `ANTHROPIC_API_KEY`) hinterlegen.
- Sind diese Variablen nicht gesetzt bzw. der Key ungueltig, liefern `/jobs/<id>/summary` und `/jobs/<id>/facts` einen HTTP-422-Fehler mit JSON (`error`, `summary_provider`, `summary_provider_key_present`), sodass du die Deployment-Konfiguration direkt erkennen kannst.

Schnelltest per `curl`:

```bash
JOB=$(curl -s -X POST http://127.0.0.1:5000/jobs \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=VIDEO_ID","whisperTask":"translate"}' | jq -r .job_id)
curl http://127.0.0.1:5000/jobs/$JOB

# Healthcheck ohne Auth
curl http://127.0.0.1:5000/healthz

# Selftest mit Basic Auth
curl -u user:pass -X POST http://127.0.0.1:5000/selftest

# Job-Status
curl http://127.0.0.1:5000/job/$JOB
```

## Deploy zu Railway

Dieses Repository bringt `Procfile` und `nixpacks.toml` mit, sodass Railway via Nixpacks automatisch Web- und Worker-Prozesse erzeugt.

1. **Projekt importieren:** In Railway auf `New Project -> Deploy from GitHub` gehen und dieses Repo auswaehlen. Die Standard-Erkennung sollte automatisch die `nixpacks.toml` nutzen (Python 3.11, ffmpeg via nix).
2. **Variablen setzen:** Lege in Railway Shared Vars an, damit Web & Worker identisch konfiguriert werden:
   - `REDIS_URL` (Upstash: `rediss://...`, Railway-Redis: `redis://.../0`)
   - `NIXPACKS_CONFIG_PATH=./nixpacks.toml`
   - `NIXPACKS_USE_NIX=1`
   - `PIP_PREFER_BINARY=1`
   - Optional: `WHISPER_DEVICE=cpu`, `WHISPER_CPU_MODEL_ID=tiny`
   - Optional: `SUMMARY_OFF=1` (Summaries standardmaessig deaktivieren)
   - Optional: `FACT_CHECK=1` (Facts-Endpoint und JSON-Ausgabe aktivieren)
   - Optional (nur Web-Service): `BASIC_AUTH_USERNAME`, `BASIC_AUTH_PASSWORD`
3. **Services pruefen:** Railway legt ueber die `Procfile` zwei Prozesse an:
   - `web`: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - `worker`: `rq worker -u $REDIS_URL default`
4. **Redis bereitstellen:** In Railway `Add Plugin -> Redis` waehlen und die dortige URL als `REDIS_URL` eintragen.
5. **Rebuild ohne Cache:** Im Dashboard `Settings -> Deployments -> Clear Build Cache & Deploy` fuer Web und Worker ausloesen, damit Nixpacks mit der neuen Konfiguration baut. Im Log darf kein `apt-get install ffmpeg` mehr auftauchen.
6. **Smoke-Test:** `curl -sSf https://<WEB_URL>/healthz | jq` prueft den Healthcheck ohne Auth. Falls Basic Auth aktiv ist:
   ```bash
   curl -sSf -u USER:PASS -X POST https://<WEB_URL>/selftest | jq
   curl -sSf -u USER:PASS https://<WEB_URL>/job/<JOB_ID> | jq
   ```

## Health Check & Benchmark

- `curl http://127.0.0.1:5000/health` zeigt `gpu`, `device_name`, `device_profile`, `rtf_10s` sowie Benchmark-Zeiten.
- Beim Start loggt der Server CUDA-, GPU- und CPU-Infos.
- CLI-Benchmark:

  ```bash
  python bench.py --duration 10
  ```

  Mit `--audio pfad/zur/datei.wav` benchmarkst du reale Clips; das Skript meldet Segmentanzahl, Laufzeiten und Real-Time-Factor.

## Output

```json
{
  "metadata": {
    "transcript_text": "...",
    "detected_language": "ar",
    "language_probability": "0.78",
    "base_language": "en",
    "whisper_task": "translate"
  },
  "transcripts": [
    {"target_language": "english", "text": "..."}
  ]
}
```

## Notes

- Stelle sicher, dass `yt-dlp` Zugriff auf das Video hat (private oder geo-blockierte Inhalte schlagen fehl).
- Lass Server und Worker nach Moeglichkeit laufen, damit Whisper-Modelle im Cache bleiben und Folge-Jobs schneller sind.
- Verwende `--whisper-task transcribe` (und tippe `source`), wenn du das Original ohne Whisper-Translate moechtest.
- Falls `python` nicht aufrufbar ist, installiere es von https://www.python.org/downloads/ und oeffne eine neue Shell.
