from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from colorama import Fore, Style, init as colorama_init
except ImportError:  # pragma: no cover - optional dependency
    class _Fallback:
        RESET_ALL = ""
        CYAN = GREEN = YELLOW = RED = MAGENTA = ""

    Fore = Style = _Fallback()  # type: ignore

    def colorama_init(*_: Any, **__: Any) -> None:  # type: ignore
        return


LOGGER_NAME = "video_transcriber"


class ColourFormatter(logging.Formatter):
    LEVEL_COLOURS: Dict[int, str] = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelno, "")
        reset = Style.RESET_ALL if colour else ""
        base = super().format(record)
        return f"{colour}{base}{reset}"


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    colorama_init(autoreset=True)
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    log_path = Path(log_dir or os.getenv("LOG_DIR", "logs")).resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console_formatter = ColourFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        log_path / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Ensure root also emits, but avoid duplicate handlers.
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level)
        root.addHandler(console_handler)
        root.addHandler(file_handler)

    logger.debug("Logging initialised. Log directory: %s", log_path)
    return logger


def _kv(**kwargs: Any) -> str:
    return " ".join(f"{key}={value}" for key, value in kwargs.items() if value is not None)


def _logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_job_start(job_id: str, video_url: Optional[str], profile: Optional[str], device_profile: Optional[str]) -> None:
    _logger().info("job.start %s", _kv(job_id=job_id, profile=profile, device=device_profile, video=video_url or "-"))


def log_job_progress(job_id: str, status: str, progress: float, detail: Optional[str] = None) -> None:
    _logger().info(
        "job.progress %s",
        _kv(job_id=job_id, status=status, progress=f"{progress:.2%}", detail=detail),
    )


def log_job_complete(
    job_id: str,
    rtf: Optional[float],
    audio_seconds: Optional[float],
    transcription_seconds: Optional[float],
    total_seconds: Optional[float],
) -> None:
    _logger().info(
        "job.complete %s",
        _kv(
            job_id=job_id,
            rtf=f"{rtf:.3f}" if rtf is not None else None,
            audio=f"{audio_seconds:.2f}" if audio_seconds is not None else None,
            transcription=f"{transcription_seconds:.2f}" if transcription_seconds is not None else None,
            total=f"{total_seconds:.2f}" if total_seconds is not None else None,
        ),
    )


def log_job_error(job_id: str, error: Exception) -> None:
    _logger().exception("job.error %s", _kv(job_id=job_id, error=str(error)))
