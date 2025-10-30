from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Nur für lokale Entwicklung - Railway braucht das nicht
from dotenv import load_dotenv
if not os.environ.get("RAILWAY_ENVIRONMENT"):
    load_dotenv()

DEFAULT_SUMMARY_LANG = os.getenv("SUMMARY_UI_LANG", "auto")

import redis
from rq import Queue, get_current_job
from rq.job import Job
from rq.exceptions import NoSuchJobError

from logging_config import (
    log_job_complete,
    log_job_error,
    log_job_progress,
    log_job_start,
    setup_logging,
)
from users import UserManager
from video_transcriber import (
    SegmentResult,
    TranscriptResult,
    determine_base_language,
    download_audio,
    normalize_language_inputs,
    transcribe_once,
)
from ai_summary import SummaryGenerationError, generate_summary

LOGGER = setup_logging()

REDIS_URL = os.environ.get("REDIS_URL")
if not REDIS_URL:
    raise ValueError("❌ REDIS_URL environment variable is not set!")

QUEUE_NAME = os.getenv("RQ_QUEUE", "default")
JOB_STORAGE_ROOT = Path(os.getenv("JOB_STORAGE_ROOT", Path.cwd() / "jobs"))

redis_connection = redis.Redis.from_url(REDIS_URL)
job_queue = Queue(QUEUE_NAME, connection=redis_connection)
