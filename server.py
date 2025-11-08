from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from rq import Queue
from werkzeug.utils import secure_filename

from jobs import (
    JOB_STORAGE_ROOT,
    enqueue_ping_job,
    enqueue_transcription_job,
    ensure_summary_materialization_safe,
    get_job_status,
    summary_default_enabled,
)
from logging_config import log_job_error, log_job_progress, log_job_start, setup_logging
from payment import add_payment_routes
from users import UserManager, init_database
from video_transcriber import (
    SUPPORTED_LANGUAGES,
    get_runtime_device_info,
    normalize_language_inputs,
    run_rtf_benchmark,
    transcribe_video,
)

# Enhanced logging setup with file output
setup_logging()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_debug.log')
    ]
)

app = Flask(__name__, static_folder="static", static_url_path="")
init_database()
add_payment_routes(app)
logger = logging.getLogger("video_transcriber.server")

_BASIC_AUTH_USER_ENV = "BASIC_AUTH_USERNAME"
_BASIC_AUTH_PASS_ENV = "BASIC_AUTH_PASSWORD"
_UNPROTECTED_ENDPOINTS = {"api_health", "healthz", "healthz_legacy", "static"}
_UNPROTECTED_PREFIXES = ("/health", "/api/health", "/static", "/_static")
YTDLP_COOKIES_ENV = "YTDLP_COOKIES_B64"
YT_COOKIES_PRESENT = bool((os.getenv(YTDLP_COOKIES_ENV) or "").strip())


def _basic_auth_enabled() -> bool:
    return bool(os.getenv(_BASIC_AUTH_USER_ENV) and os.getenv(_BASIC_AUTH_PASS_ENV))


@app.before_request
def enforce_basic_auth() -> Optional[Response]:
    if not _basic_auth_enabled():
        return None
    path = request.path or "/"
    if path == "/" or any(path.startswith(prefix) for prefix in _UNPROTECTED_PREFIXES):
        return None
    endpoint = (request.endpoint or "").lower()
    if endpoint in _UNPROTECTED_ENDPOINTS:
        return None

    auth = request.authorization
    if not auth or auth.type.lower() != "basic":
        return Response("", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})

    expected_user = os.getenv(_BASIC_AUTH_USER_ENV, "")
    expected_pass = os.getenv(_BASIC_AUTH_PASS_ENV, "")
    if auth.username == expected_user and auth.password == expected_pass:
        return None

    return Response("", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})


def _get_queue_name() -> str:
    return os.getenv("RQ_QUEUE", "default")


def _get_redis_connection() -> Optional[redis.Redis]:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        return redis.from_url(redis_url)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Redis connection error: %s", exc)
        return None


def _build_health_payload() -> Dict[str, Any]:
    redis_ok = False
    queue_len = 0
    connection = _get_redis_connection()
    if connection is not None:
        try:
            redis_ok = bool(connection.ping())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Redis ping failed: %s", exc)
        else:
            try:
                queue = Queue(_get_queue_name(), connection=connection)
                queue_len = queue.count
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Queue length lookup failed: %s", exc)
    return {
        "status": "ok",
        "redis": redis_ok,
        "worker_queue_len": queue_len,
    }

logger.info("=" * 80)
logger.info("🚀 SERVER STARTING UP")
logger.info("=" * 80)
logger.info(f"📝 AI_PROVIDER: {os.getenv('AI_PROVIDER', 'not set')}")
logger.info(f"🔑 ANTHROPIC_API_KEY set: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
logger.info(f"🔑 OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
logger.info(f"📊 ANTHROPIC_MODEL: {os.getenv('ANTHROPIC_MODEL', 'not set')}")
logger.info(f"🌍 SUMMARY_UI_LANG: {os.getenv('SUMMARY_UI_LANG', 'not set')}")
logger.info("=" * 80)

runtime_info = get_runtime_device_info()
logger.info(
    "Runtime hardware -> CUDA: %s, GPU: %s, VRAM(bytes): %s, CPU threads: %s",
    runtime_info["cuda_available"],
    runtime_info["device_name"] or "n/a",
    runtime_info["total_vram_bytes"],
    runtime_info["cpu_threads"],
)

JOB_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_TRANSLATIONS = {"de", "en", "fr", "es", "ar"}
TRANSLATIONS_DIR = Path("translations")
_TRANSLATION_CACHE: Dict[str, Dict[str, Any]] = {}

_HEALTH_CACHE: Dict[str, Any] = {"timestamp": 0.0, "payload": None}
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
_CORS_ALLOWED_HEADERS = "Content-Type, Authorization"
_CORS_ALLOWED_METHODS = "GET, POST, OPTIONS"
_LOCALHOST_IPS = {"127.0.0.1", "::1"}
_DISABLE_CREDITS_CHECK = os.getenv("DISABLE_CREDITS_CHECK", "").strip().lower() in {"1", "true", "yes"}
_DISABLE_CREDITS_FOR_LOCALHOST = (
    os.getenv("DISABLE_CREDITS_FOR_LOCALHOST", "").strip().lower() in {"1", "true", "yes"}
)
_FLAG_TRUE_VALUES = {"1", "true", "yes", "on"}
_FLAG_FALSE_VALUES = {"0", "false", "no", "off"}
FACT_CHECK_ENABLED = (os.getenv("FACT_CHECK") or "").strip().lower() in _FLAG_TRUE_VALUES


def _parse_languages(payload: Dict[str, Any]) -> Tuple[List[str], bool, str]:
    languages_input = payload.get("languages") or []
    if isinstance(languages_input, str):
        languages_input = [languages_input]
    if not isinstance(languages_input, list):
        return [], False, "languages must be a list of language names."

    try:
        normalized, include_source = normalize_language_inputs(languages_input)
    except ValueError as exc:
        return [], False, str(exc)

    return normalized, include_source, ""


def _coerce_targets(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        separators = [",", ";"]
        for sep in separators:
            if sep in value:
                return [part.strip() for part in value.split(sep) if part.strip()]
        return value.split()
    return []


def _coerce_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_optional_bool(raw: Any) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in _FLAG_TRUE_VALUES:
        return True
    if text in _FLAG_FALSE_VALUES:
        return False
    return None


def _resolve_summary_flag(*values: Any) -> Optional[bool]:
    for value in values:
        result = _coerce_optional_bool(value)
        if result is not None:
            return result
    return None


def _safe_profile(value: str) -> str:
    return value if value in {"fast", "pro"} else "fast"


def _safe_task(value: str) -> str:
    return value if value in {"translate", "transcribe"} else "translate"


def _save_uploaded_file(file_storage) -> Path:
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        filename = f"upload_{uuid.uuid4().hex}"
    upload_dir = JOB_STORAGE_ROOT / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / filename
    file_storage.save(destination)
    return destination


def _apply_cors_headers(response):
    if not FRONTEND_ORIGIN:
        return response
    origin = request.headers.get("Origin")
    if origin == FRONTEND_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
        response.headers["Access-Control-Allow-Credentials"] = "true"
        vary = response.headers.get("Vary")
        if vary:
            if "Origin" not in vary:
                response.headers["Vary"] = f"{vary}, Origin"
        else:
            response.headers["Vary"] = "Origin"
    return response


@app.before_request
def handle_preflight() -> Optional[Any]:
    if request.method == "OPTIONS":
        response = app.make_response(("", 204))
        response.headers["Access-Control-Allow-Methods"] = _CORS_ALLOWED_METHODS
        response.headers["Access-Control-Allow-Headers"] = _CORS_ALLOWED_HEADERS
        response.headers["Access-Control-Max-Age"] = "86400"
        return _apply_cors_headers(response)
    return None


@app.after_request
def add_cors_headers(response):
    response = _apply_cors_headers(response)
    if request.method == "OPTIONS":
        response.headers.setdefault("Access-Control-Allow-Methods", _CORS_ALLOWED_METHODS)
        response.headers.setdefault("Access-Control-Allow-Headers", _CORS_ALLOWED_HEADERS)
    return response


@app.route("/", methods=["GET"])
def index() -> Any:
    video_url = (request.args.get("video_url") or request.args.get("url") or "").strip()
    if not video_url:
        return app.send_static_file("index.html")

    try:
        email = (request.args.get("email") or request.args.get("userEmail") or "").strip()
        if not email:
            raise ValueError("Email required")

        targets = _coerce_targets(request.args.get("targets") or request.args.get("languages"))
        include_source = _coerce_bool(request.args.get("includeSource") or request.args.get("include_source"), True)
        profile = _safe_profile((request.args.get("profile") or "fast").lower())
        source_language = (request.args.get("sourceLanguage") or request.args.get("source_language") or "").strip() or None
        whisper_task = _safe_task((request.args.get("whisperTask") or request.args.get("whisper_task") or "transcribe").lower())
        summary_pref = _resolve_summary_flag(
            request.args.get("summary"),
            request.args.get("add_summary"),
            request.args.get("addSummary"),
            request.args.get("addSummarySelected"),
        )
        add_summary_effective = summary_pref if summary_pref is not None else summary_default_enabled()
        summary_lang = (request.args.get("summary_lang") or os.getenv("SUMMARY_UI_LANG", "auto")).strip() or "auto"
        logger.info("Summary requested (GET /): %s", add_summary_effective)

        job = enqueue_transcription_job(
            video_url=video_url,
            targets=targets,
            profile=profile,
            include_source=include_source,
            source_language=source_language,
            whisper_task=whisper_task,
            upload_path=None,
            user_email=email,
            add_summary=summary_pref,
            summary_lang=summary_lang,
        )
        logger.info("Created job via GET /: job_id=%s video_url=%s", job.id, video_url)
        return (
            jsonify(
                {
                    "ok": True,
                    "job_id": job.id,
                }
            ),
            200,
        )
    except Exception as exc:
        logger.exception("Failed to create job via GET /")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "hint": "check Railway logs / job creator",
                }
            ),
            200,
        )


def _compute_health_payload() -> Dict[str, Any]:
    bench = run_rtf_benchmark(duration=10.0, task="transcribe")
    device_info = get_runtime_device_info()
    return {
        "gpu": bool(device_info["cuda_available"]),
        "device_name": device_info["device_name"] or ("cuda" if device_info["cuda_available"] else "cpu"),
        "rtf_10s": round(bench["rtf"], 4),
        "transcription_seconds": round(bench["elapsed_seconds"], 3),
        "audio_seconds": bench["audio_seconds"],
        "device_profile": bench["device_profile"],
    }


def _get_health_payload() -> Dict[str, Any]:
    now = time.time()
    cached = _HEALTH_CACHE["payload"]
    if cached and (now - _HEALTH_CACHE["timestamp"]) < 300:
        return cached
    payload = _compute_health_payload()
    _HEALTH_CACHE["payload"] = payload
    _HEALTH_CACHE["timestamp"] = now
    return payload


def _full_health_payload() -> Dict[str, Any]:
    payload = _build_health_payload()
    payload.update(_get_health_payload())
    return payload


@app.route("/health", methods=["GET"])
def api_health() -> Any:
    try:
        payload = _get_health_payload()
    except Exception as exc:  # pragma: no cover - safeguard
        log_job_error("health", exc)
        return jsonify({"error": str(exc)}), 500
    return jsonify(payload)


def _fetch_public_stats() -> Dict[str, Any]:
    payload = {"total_jobs": 0, "total_minutes": 0.0, "total_users": 0}
    try:
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM usage_history")
            row = cursor.fetchone()
            payload["total_jobs"] = int(row[0]) if row and row[0] else 0

            cursor.execute("SELECT COALESCE(SUM(video_duration), 0) FROM usage_history")
            row = cursor.fetchone()
            payload["total_minutes"] = float(row[0]) if row and row[0] is not None else 0.0

            cursor.execute("SELECT COUNT(*) FROM users")
            row = cursor.fetchone()
            payload["total_users"] = int(row[0]) if row and row[0] else 0
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load public stats: %s", exc)
    return payload


def _detect_language_from_headers() -> str:
    header = request.headers.get("Accept-Language", "")
    for part in header.split(","):
        code = part.split(";")[0].strip().lower()
        if not code:
            continue
        code = code.split("-")[0]
        if code in SUPPORTED_TRANSLATIONS:
            return code
    return "de"


def _load_translation_payload(lang: str) -> Dict[str, Any]:
    if lang in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[lang]
    candidate = TRANSLATIONS_DIR / f"{lang}.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Translation file not found for {lang}")
    with candidate.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    _TRANSLATION_CACHE[lang] = data
    return data


@app.route("/api/config", methods=["GET"])
def api_config() -> Any:
    return jsonify(
        {
            "summary_default_on": summary_default_enabled(),
            "fact_check_enabled": FACT_CHECK_ENABLED,
            "yt_cookies_present": YT_COOKIES_PRESENT,
        }
    )


@app.route("/api/translations/<lang>", methods=["GET"])
def api_translations(lang: str) -> Any:
    requested = (lang or "").lower()
    if requested == "auto":
        requested = _detect_language_from_headers()
    if requested not in SUPPORTED_TRANSLATIONS:
        requested = _detect_language_from_headers()
    if requested not in SUPPORTED_TRANSLATIONS:
        requested = "de"
    try:
        payload = _load_translation_payload(requested)
    except (FileNotFoundError, json.JSONDecodeError):
        requested = "de"
        payload = _load_translation_payload(requested)
    return jsonify({"lang": requested, "strings": payload})


@app.route("/translations/<lang>.json", methods=["GET"])
def get_translation_file(lang: str):
    requested = (lang or "").lower()
    if requested not in {"de", "en"}:
        requested = "de"
    return send_from_directory("static/translations", f"{requested}.json")


@app.route("/api/stats", methods=["GET"])
def get_public_stats() -> Any:
    return jsonify(_fetch_public_stats())


@app.route("/jobs/<job_id>/files/<path:filename>", methods=["GET"])
def get_job_file(job_id: str, filename: str) -> Any:
    """Serve files from completed jobs with detailed logging."""
    logger.info("=" * 80)
    logger.info("📁 FILE REQUEST")
    logger.info("=" * 80)
    logger.info(f"  Job ID: {job_id}")
    logger.info(f"  Filename: {filename}")

    # Security: validate job_id format
    if not job_id or '..' in job_id or '/' in job_id:
        logger.warning(f"❌ Invalid job_id format: {job_id}")
        abort(404)

    # Resolve paths
    base_dir = (JOB_STORAGE_ROOT / job_id / "files").resolve()
    logger.info(f"  Base directory: {base_dir}")
    logger.info(f"  Base exists: {base_dir.exists()}")

    if not base_dir.exists():
        logger.warning(f"❌ Job directory not found: {base_dir}")

        # List what's actually in JOB_STORAGE_ROOT
        logger.info(f"  Available jobs in {JOB_STORAGE_ROOT}:")
        try:
            for item in JOB_STORAGE_ROOT.iterdir():
                if item.is_dir():
                    logger.info(f"    - {item.name}")
        except Exception as e:
            logger.error(f"  Could not list jobs: {e}")

        abort(404)

    # List files in job directory
    logger.info(f"  Files in job directory:")
    try:
        for item in base_dir.iterdir():
            logger.info(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
    except Exception as e:
        logger.error(f"  Could not list files: {e}")

    target = (base_dir / filename).resolve()
    logger.info(f"  Target file: {target}")
    logger.info(f"  File exists: {target.exists()}")

    # Security check: ensure target is within base_dir
    try:
        target.relative_to(base_dir)
    except ValueError:
        logger.warning(f"❌ Path traversal attempt detected")
        logger.warning(f"  Target: {target}")
        logger.warning(f"  Base: {base_dir}")
        abort(404)

    if not target.exists():
        logger.warning(f"❌ File not found: {target}")
        abort(404)

    if not target.is_file():
        logger.warning(f"❌ Not a file: {target}")
        abort(404)

    logger.info(f"✅ Serving file: {filename}")
    logger.info("=" * 80)

    return send_from_directory(base_dir, filename, as_attachment=False)


@app.route("/jobs/<job_id>/summary", methods=["GET"])
def get_job_summary_text(job_id: str) -> Response:
    try:
        payload = get_job_status(job_id)
    except Exception as exc:
        logger.exception("Failed to load summary for job %s", job_id)
        return Response("Summary not available.\n", 404, {"Content-Type": "text/plain; charset=utf-8"})

    status = payload.get("status")
    result = payload.get("result") or {}
    metadata = result.get("metadata") or {}
    summary_requested = metadata.get("summary_requested")
    if summary_requested is None:
        summary_requested = bool(metadata.get("summary_file") or payload.get("summary_file"))

    if status != "finished":
        return Response("Summary not available yet.\n", 404, {"Content-Type": "text/plain; charset=utf-8"})
    if not summary_requested:
        return Response("Summary not requested for this job.\n", 404, {"Content-Type": "text/plain; charset=utf-8"})

    info = ensure_summary_materialization_safe(job_id, payload)
    summary_path_str = info.get("summary_path")
    if not summary_path_str:
        summary_rel = metadata.get("summary_file") or payload.get("summary_file")
        if summary_rel:
            summary_path = (JOB_STORAGE_ROOT / job_id / "files" / summary_rel).resolve()
            summary_path_str = str(summary_path)
    if not summary_path_str:
        return Response("Summary not available.\n", 404, {"Content-Type": "text/plain; charset=utf-8"})

    summary_path = Path(summary_path_str)
    if not summary_path.is_file():
        return Response("Summary not available.\n", 404, {"Content-Type": "text/plain; charset=utf-8"})
    try:
        content = summary_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to read summary for job %s: %s", job_id, exc)
        return Response("Summary not available.\n", 500, {"Content-Type": "text/plain; charset=utf-8"})

    response = Response(content, mimetype="text/plain; charset=utf-8")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/jobs/<job_id>/facts", methods=["GET"])
def get_job_facts(job_id: str) -> Any:
    if not FACT_CHECK_ENABLED:
        return jsonify({"error": "Fact-check endpoint disabled"}), 404
    try:
        payload = get_job_status(job_id)
    except Exception as exc:
        logger.exception("Failed to load fact data for job %s", job_id)
        return jsonify({"error": "Job not found"}), 404

    if payload.get("status") != "finished":
        return jsonify({"error": "Job not finished"}), 404

    result = payload.get("result") or {}
    metadata = result.get("metadata") or {}
    facts = metadata.get("summary_facts")
    if not facts:
        return jsonify({"error": "No fact-check data available"}), 404

    return jsonify({"job_id": job_id, "facts": facts})


@app.route("/jobs", methods=["POST"])
def create_job() -> Any:
    logger.info("=" * 80)
    logger.info("📥 NEW JOB REQUEST RECEIVED")
    logger.info("=" * 80)
    
    upload_path: Optional[Path] = None
    video_url: Optional[str] = None
    targets: List[str] = []
    include_source_flag = True
    profile = "fast"
    source_language: Optional[str] = None
    whisper_task = "transcribe"

    email: Optional[str] = None
    summary_pref: Optional[bool] = None
    data: Dict[str, Any] = {}

    try:
        logger.info(f"📋 Content-Type: {request.content_type}")
        logger.info(f"📍 Remote address: {request.remote_addr}")
        
        if request.content_type and request.content_type.startswith("multipart/form-data"):
            file = request.files.get("file")
            if file and file.filename:
                upload_path = _save_uploaded_file(file)

            form = request.form
            video_url = (form.get("video_url") or form.get("url") or "").strip() or None
            targets_input = form.get("targets") or form.get("languages")
            targets = _coerce_targets(targets_input)
            include_source_flag = _coerce_bool(form.get("includeSource"), True)
            profile = _safe_profile((form.get("profile") or "fast").lower())
            source_language = (form.get("sourceLanguage") or "").strip() or None
            whisper_task = _safe_task((form.get("whisperTask") or "transcribe").lower())
            email = (form.get("email") or form.get("userEmail") or form.get("user_email") or "").strip() or None
            summary_pref = _resolve_summary_flag(
                form.get("summary"),
                form.get("add_summary"),
                form.get("addSummary"),
                form.get("addSummarySelected"),
            )
        else:
            data = request.get_json(force=True, silent=True) or {}
            video_url = (data.get("video_url") or data.get("url") or "").strip() or None
            targets = _coerce_targets(data.get("targets") or data.get("languages"))
            include_source_flag = _coerce_bool(data.get("includeSource"), True)
            profile = _safe_profile((data.get("profile") or "fast").lower())
            source_language = (data.get("sourceLanguage") or "").strip() or None
            whisper_task = _safe_task((data.get("whisperTask") or "transcribe").lower())
            email = (data.get("email") or data.get("userEmail") or data.get("user_email") or "").strip() or None
            summary_pref = _resolve_summary_flag(
                data.get("summary"),
                data.get("add_summary"),
                data.get("addSummary"),
                data.get("addSummarySelected"),
            )

        if not video_url and not upload_path:
            logger.warning("❌ No video URL or upload provided")
            raise ValueError("Bitte gib eine Video-URL an oder lade eine Datei hoch.")

        if not email:
            logger.warning("❌ No email provided")
            raise ValueError("Email required")

        summary_requested = summary_pref if summary_pref is not None else summary_default_enabled()

        summary_lang = (
            (request.form.get("summary_lang") if (request.content_type and request.content_type.startswith("multipart/form-data")) else None)
            or (data.get("summary_lang") if data else None)
            or os.getenv("SUMMARY_UI_LANG", "auto")
        )
        
        logger.info("=" * 80)
        logger.info("📊 JOB REQUEST DETAILS:")
        logger.info(f"  📧 Email: {email}")
        logger.info(f"  🎥 Video URL: {video_url}")
        logger.info(f"  📁 Upload: {bool(upload_path)}")
        logger.info(f"  🎯 Targets: {targets}")
        logger.info(f"  🤖 Add Summary: {summary_requested}")
        logger.info(f"  🌍 Summary Lang: {summary_lang}")
        logger.info(f"  ⚙️ Profile: {profile}")
        logger.info(f"  📝 Whisper Task: {whisper_task}")
        logger.info("=" * 80)

        ip_address = request.remote_addr or "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip_address = forwarded_for.split(",")[0].strip() or ip_address

        if not UserManager.check_ip_abuse(email, ip_address):
            logger.warning("IP abuse detected: email=%s ip=%s", email, ip_address)
            raise PermissionError("Too many accounts from this IP")

        _, include_source_from_list, error = _parse_languages({"languages": targets})
        if error:
            raise ValueError(error)
        include_source_final = include_source_flag or include_source_from_list

        user = UserManager.get_or_create_user(email)

        if not UserManager.check_rate_limit(email):
            info = UserManager.get_rate_limit_info(email)
            logger.warning("Rate limit exceeded: email=%s ip=%s info=%s", email, ip_address, info)
            raise PermissionError(f"Rate limit exceeded: {info}")

        estimated_minutes = 5.0
        skip_credit_check = _DISABLE_CREDITS_CHECK or (
            _DISABLE_CREDITS_FOR_LOCALHOST and ip_address in _LOCALHOST_IPS
        )
        if skip_credit_check:
            logger.info("Skipping quota check for email=%s ip=%s", email, ip_address)
        if not skip_credit_check and not UserManager.check_quota(email, estimated_minutes):
            stats = UserManager.get_user_stats(email) or {}
            used = stats.get("minutes_used", 0)
            quota = stats.get("minutes_quota", user.get("minutes_quota") if user else 0)
            logger.warning("Insufficient credits: email=%s used=%s quota=%s", email, used, quota)
            raise PermissionError(f"Insufficient credits (used={used}, quota={quota})")

        UserManager.register_ip_usage(email, ip_address)
        logger.info("IP usage recorded: email=%s ip=%s", email, ip_address)

        logger.info("🎬 Enqueuing transcription job...")
        
        job = enqueue_transcription_job(
            video_url=video_url,
            targets=targets,
            profile=profile,
            include_source=include_source_final,
            source_language=source_language,
            whisper_task=whisper_task,
            upload_path=str(upload_path) if upload_path else None,
            user_email=email,
            add_summary=summary_pref,
            summary_lang=summary_lang,
        )

        logger.info("✅ Job enqueued successfully!")
        logger.info(
            "Job details: id=%s url=%s upload=%s profile=%s task=%s targets=%s include_source=%s add_summary=%s",
            job.id,
            video_url,
            bool(upload_path),
            profile,
            whisper_task,
            targets,
            include_source_final,
            summary_requested,
        )
        log_job_start(job.id, video_url, profile, "queued")
        log_job_progress(job.id, "queued", 0.0, detail="job enqueued via API")
        logger.info("=" * 80)
        logger.info(f"✅ JOB CREATED: {job.id}")
        logger.info("=" * 80)
        return (
            jsonify(
                {
                    "ok": True,
                    "job_id": job.id,
                }
            ),
            200,
        )
    except Exception as exc:
        logger.exception("Failed to create job")
        log_job_error("enqueue", exc)
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "hint": "check Railway logs / job creator",
                }
            ),
            200,
        )


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str) -> Any:
    try:
        payload = get_job_status(job_id)
    except Exception as exc:
        logger.exception("Failed to load job %s", job_id)
        return jsonify({"error": str(exc)}), 200

    info = ensure_summary_materialization_safe(job_id, payload)
    if info.get("error"):
        payload.setdefault("warnings", []).append("summary_materialization_failed")
        payload["summary_materialization_error"] = info["error"]
    return jsonify(payload)


@app.route("/api/user/stats", methods=["GET"])
def api_user_stats() -> Any:
    email = (request.args.get("email") or "").strip()
    if not email:
        return jsonify({"error": "Email required"}), 400

    stats = UserManager.get_user_stats(email)
    if not stats:
        return jsonify({"error": "User not found"}), 404
    return jsonify(stats)


@app.route("/api/user/rate-limit", methods=["GET"])
def api_user_rate_limit() -> Any:
    email = (request.args.get("email") or "").strip()
    if not email:
        return jsonify({"error": "Email required"}), 400
    info = UserManager.get_rate_limit_info(email)
    return jsonify(info)


@app.route("/api/admin/ip-stats", methods=["GET"])
def api_admin_ip_stats() -> Any:
    ip_address = (request.args.get("ip") or "").strip()
    if not ip_address:
        return jsonify({"error": "IP required"}), 400
    return jsonify(UserManager.get_ip_stats(ip_address))


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe() -> Any:
    data = request.get_json(force=True, silent=True) or {}

    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Bitte gib einen gueltigen YouTube- oder Facebook-Link an."}), 400

    languages, include_source_from_list, error = _parse_languages(data)
    if error:
        return jsonify({"error": error}), 400
    languages = []

    include_source = bool(data.get("includeSource", False)) or include_source_from_list
    whisper_task = (data.get("whisperTask") or "translate").lower()
    if whisper_task not in {"translate", "transcribe"}:
        return jsonify({"error": "whisperTask muss 'translate' oder 'transcribe' sein."}), 400

    model_size = (data.get("modelSize") or "small").lower()
    device = (data.get("device") or "cpu").lower()
    compute_type = (data.get("computeType") or "int8").lower()

    try:
        beam_size = int(data.get("beamSize", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "beamSize muss eine ganze Zahl sein."}), 400

    source_language = data.get("sourceLanguage")
    if source_language is not None:
        source_language = source_language.strip() or None

    try:
        metadata, base_result, translations, include_source_flag = transcribe_video(
            url=url,
            languages=languages,
            include_source=include_source,
            whisper_task=whisper_task,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            source_language=source_language,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - safeguard
        log_job_error("api_transcribe", exc)
        return jsonify({"error": f"Unerwarteter Fehler: {exc}"}), 500

    transcript_payload: List[Dict[str, str]] = []
    if include_source_flag:
        transcript_payload.append(base_result.__dict__)
    transcript_payload.extend(result.__dict__ for result in translations)

    return jsonify(
        {
            "metadata": metadata,
            "transcripts": transcript_payload,
            "supportedLanguages": sorted(SUPPORTED_LANGUAGES),
        }
    )


@app.route("/healthz", methods=["GET"])
def healthz() -> Any:
    """Lightweight healthcheck used by infrastructure probes."""
    payload = _full_health_payload()
    payload["ok"] = payload.get("status") == "ok"
    return jsonify(payload)




@app.route("/health", methods=["GET"])
def basic_health() -> Any:
    """Backward compatible alias for the health payload."""
    return jsonify(_full_health_payload())


@app.route("/api/healthz", methods=["GET"])
def healthz_legacy() -> Any:
    """Legacy API path returning the same health payload."""
    return jsonify(_full_health_payload())


@app.route("/selftest", methods=["POST"])
def api_selftest() -> Any:
    try:
        job_id = enqueue_ping_job()
    except Exception as exc:  # pragma: no cover - defensive logging
        log_job_error("selftest", exc)
        return jsonify({"error": "Failed to enqueue self-test job"}), 500
    return jsonify({"job_id": job_id}), 202


@app.route("/job/<job_id>", methods=["GET"])
def api_job_status(job_id: str) -> Any:
    try:
        status = get_job_status(job_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(status)


@app.route("/success", methods=["GET"])
def checkout_success() -> Any:
    return jsonify({"message": "Payment successful"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
