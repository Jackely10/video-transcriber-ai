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
    "profile": "fast"
  }
  ```
  Antwort: `{"job_id": "<id>"}` mit Status `202`.
- `GET /jobs/<id>` - liefert Status, Fortschritt, Meta-Daten, RTF und Ausgabe-Pfade.
- `GET /jobs/<id>/files/<path>` - stellt erzeugte TXT/SRT/VTT-Dateien bereit.
- `GET /healthz` - liefert `{status, redis, worker_queue_len}` und ist fuer Healthchecks ohne Authentifizierung gedacht.
- `POST /selftest` - enqueued einen internen `ping`-Job auf der Default-Queue (Basic Auth erforderlich, falls gesetzt) und liefert `{"job_id": ...}`.
- `GET /job/<id>` - gibt den Job-Status sowie ggf. das Ergebnis zurueck; meldet `404`, wenn die ID unbekannt ist.

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
2. **Variablen setzen (Web & Worker):**
   - `REDIS_URL` (z. B. Upstash mit `rediss://`)
   - `PIP_PREFER_BINARY=1`
   - `NIXPACKS_USE_NIX=1`
   - `NIXPACKS_CONFIG_PATH=./nixpacks.toml`
   - Optional: `WHISPER_DEVICE=cpu`, `WHISPER_CPU_MODEL_ID=tiny`
   - Optional: `BASIC_AUTH_USERNAME`, `BASIC_AUTH_PASSWORD`
   - Als Shared Vars gepflegt erhalten Web- und Worker-Service automatisch identische Einstellungen.
3. **Services pruefen:** Railway legt ueber die `Procfile` zwei Prozesse an:
   - `web`: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - `worker`: `rq worker default --url $REDIS_URL`
4. **Redis bereitstellen:** In Railway `Add Plugin -> Redis` waehlen und die dortige URL als `REDIS_URL` eintragen.
5. **Rebuild ohne Cache:** Im Dashboard `Settings -> Deployments -> Clear Build Cache & Deploy` fuer Web und Worker ausloesen, damit Nixpacks mit der neuen Konfiguration baut. Im Log darf kein `apt-get install ffmpeg` mehr auftauchen.
6. **Smoke-Test:** Nach erfolgreichem Deploy `/healthz` aufrufen, eine kurze Transkription anstossen (Web-Log zeigt Queueing, Worker-Log die Abarbeitung).

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

