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

Railway liest die Konfiguration aus `railway.toml` und startet den Web-Service mit `uvicorn asgi:app --host 0.0.0.0 --port $PORT`. Der integrierte Healthcheck unter `/healthz` wird automatisch überwacht.

1. **Projekt importieren:** In Railway auf `New Project -> Deploy from GitHub` gehen und dieses Repo ausw?hlen. Dank `railway.toml` erkennt Railway automatisch den Web-Dienst (uvicorn) und ?berwacht `/healthz`.
2. **Environment-Variablen setzen:** Als Shared Vars f?r Web & Worker anlegen:
   - Pflicht: `OPENAI_API_KEY`
   - Typisch: `REDIS_URL`, `ANTHROPIC_API_KEY`, `STRIPE_SECRET_KEY`
   - Optional (nur Web): `BASIC_AUTH_USERNAME`, `BASIC_AUTH_PASSWORD`
   - Weitere Tuning-Optionen: `SUMMARY_OFF=1`, `FACT_CHECK=1`, `WHISPER_DEVICE=cpu`, `WHISPER_CPU_MODEL_ID=tiny`, `NIXPACKS_CONFIG_PATH=./nixpacks.toml`, `NIXPACKS_USE_NIX=1`, `PIP_PREFER_BINARY=1`
3. **Redis bereitstellen:** ?ber `Add Plugin -> Redis` eine Instanz hinzuf?gen und die URL als `REDIS_URL` hinterlegen.
4. **Deploy/Rebuild:** Railway nutzt automatisch Nixpacks (Python 3.11 + ffmpeg). Bei Bedarf `Settings -> Deployments -> Clear Build Cache & Deploy` ausl?sen.
5. **Services pr?fen:** Das `Procfile` definiert weiterhin `web` und `worker` (`rq worker -u $REDIS_URL default`). In Railway kannst du beide Prozesse als getrennte Services laufen lassen.
6. **Smoke-Test:** `curl -sSf https://<WEB_URL>/healthz | jq` sollte `{"status":"ok","ok":true,...}` liefern. F?r gesch?tzte Endpunkte Basic Auth verwenden (`curl -u USER:PASS .../selftest`).



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

## Railway Resource Tips

- Kostenlose Railway-Container besitzen nur wenige hundert MB RAM. Setze deshalb `WHISPER_CPU_MODEL_ID=tiny` (oder `WHISPER_CPU_MODEL_ID_TRANSCRIBE`) sowie `WHISPER_CPU_MODEL_ID_TRANSLATE=Systran/faster-whisper-tiny.en`, solange kein GPU-Profil aktiv ist.
- Aktiviere optional `WHISPER_PRELOAD_MODEL=1`, damit der Worker das Whisper-Modell direkt beim Start herunterlaedt. Wenn der Download wegen Speichermangel scheitert, siehst du den Crash sofort und kannst Scale-/Model-Entscheidungen treffen, bevor echte Jobs laufen.
- Speicher- und CPU-Verbrauch findest du im Railway-Dashboard unter **Service → Metrics**. Wenn die Kurven dauerhaft beim Limit liegen, upgraden oder kleinere Modelle im `fast`-Profil verwenden.
