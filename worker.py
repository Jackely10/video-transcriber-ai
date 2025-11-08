from __future__ import annotations

import logging
import os

from rq import SimpleWorker, Worker

from jobs import QUEUE_NAME, redis_connection
from video_transcriber import preload_default_whisper_model


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    preload_flag = os.getenv("WHISPER_PRELOAD_MODEL", "").strip().lower()
    if preload_flag in {"1", "true", "yes", "on"}:
        logging.info("Preloading Whisper model before starting worker...")
        try:
            profile = preload_default_whisper_model()
        except Exception:
            logging.exception("Failed to preload Whisper model; continuing without warm cache")
        else:
            logging.info("Whisper model ready: id=%s device=%s", profile.model_id, profile.device)
    if os.name == "nt":
        worker = SimpleWorker([QUEUE_NAME], connection=redis_connection)
    else:
        worker = Worker([QUEUE_NAME], connection=redis_connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
