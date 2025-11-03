from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
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

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("RQ_QUEUE", "default")
JOB_STORAGE_ROOT = Path(os.getenv("JOB_STORAGE_ROOT", Path.cwd() / "jobs"))

redis_connection = redis.Redis.from_url(REDIS_URL)
job_queue = Queue(QUEUE_NAME, connection=redis_connection)

STEP_WEIGHTS: Dict[str, float] = {
    "download": 0.1,
    "extract": 0.15,
    "transcribe": 0.45,
    "translate": 0.2,
    "export": 0.1,
}


def ping_job() -> str:
    return "pong"


def _ensure_storage_root() -> None:
    JOB_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


def _ensure_job_dir(job_id: str) -> Path:
    _ensure_storage_root()
    job_dir = JOB_STORAGE_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _format_timestamp(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _write_txt(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_srt(path: Path, segments: Iterable[SegmentResult]) -> None:
    lines: List[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}")
        lines.append(segment.text.strip() or "(no speech)")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp(seconds).replace(",", ".")


def _write_vtt(path: Path, segments: Iterable[SegmentResult]) -> None:
    lines: List[str] = ["WEBVTT", ""]
    for segment in segments:
        lines.append(f"{_format_timestamp_vtt(segment.start)} --> {_format_timestamp_vtt(segment.end)}")
        lines.append(segment.text.strip() or "(no speech)")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _export_transcripts(
    job_dir: Path,
    transcripts: List[Dict[str, Any]],
    base_segments: List[SegmentResult],
    audio_seconds: float,
) -> Dict[str, Dict[str, str]]:
    outputs: Dict[str, Dict[str, str]] = {}
    for entry in transcripts:
        language = entry["target_language"]
        text = entry["text"].strip()
        is_base = entry.get("is_base", False)
        seg_override = entry.get("segments")

        language_dir = job_dir / language
        language_dir.mkdir(parents=True, exist_ok=True)

        txt_path = language_dir / f"{language}.txt"
        _write_txt(txt_path, text)

        if seg_override:
            segments_for_srt = [
                SegmentResult(start=item["start"], end=item["end"], text=item["text"])
                for item in seg_override
            ]
        elif is_base:
            segments_for_srt = base_segments
        else:
            segments_for_srt = [SegmentResult(start=0.0, end=audio_seconds, text=text)]

        srt_path = language_dir / f"{language}.srt"
        _write_srt(srt_path, segments_for_srt)
        vtt_path = language_dir / f"{language}.vtt"
        _write_vtt(vtt_path, segments_for_srt)

        outputs[language] = {
            "txt": txt_path.relative_to(job_dir).as_posix(),
            "srt": srt_path.relative_to(job_dir).as_posix(),
            "vtt": vtt_path.relative_to(job_dir).as_posix(),
        }

    return outputs


def _extract_audio(input_path: Path, output_path: Path) -> Path:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def _update_job_meta(status: str, progress: float, extra: Optional[Dict[str, Any]] = None) -> None:
    job = get_current_job()
    job_id = job.id if job else "local-run"
    bounded_progress = max(0.0, min(progress, 1.0))

    if job:
        job.meta.setdefault("steps", {})
        job.meta["status"] = status
        job.meta["progress"] = bounded_progress
        if extra:
            job.meta.update(extra)
        job.save_meta()

    detail = extra.get("status_detail") if extra else None
    log_job_progress(job_id, status, bounded_progress, detail=detail)


def _record_step(step_name: str, start_time: float, step_times: Dict[str, float]) -> None:
    step_times[step_name] = time.perf_counter() - start_time
    job = get_current_job()
    if job:
        job.meta.setdefault("steps", {})
        job.meta["steps"][step_name] = step_times[step_name]
        job.save_meta()


def process_job(
    video_url: Optional[str] = None,
    upload_path: Optional[str] = None,
    targets: Optional[List[str]] = None,
    profile: str = "fast",
    include_source: bool = True,
    source_language: Optional[str] = None,
    whisper_task: str = "translate",
    add_summary: bool = False,
    summary_lang: Optional[str] = None,
) -> Dict[str, Any]:
    job = get_current_job()
    job_id = job.id if job else "local-run"
    
    LOGGER.info("=" * 80)
    LOGGER.info("🎬 STARTING VIDEO JOB PROCESSING")
    LOGGER.info("=" * 80)
    LOGGER.info(f"  📋 Job ID: {job_id}")
    LOGGER.info(f"  🎥 Video URL: {video_url}")
    LOGGER.info(f"  📁 Upload path: {upload_path}")
    LOGGER.info(f"  🎯 Targets: {targets}")
    LOGGER.info(f"  ⚙️ Profile: {profile}")
    LOGGER.info(f"  🌍 Source language: {source_language}")
    LOGGER.info(f"  📝 Whisper task: {whisper_task}")
    LOGGER.info(f"  🤖 Add summary: {add_summary}")
    LOGGER.info(f"  🌍 Summary lang: {summary_lang}")
    LOGGER.info("=" * 80)
    
    job_meta = job.meta if job else {}
    summary_requested = bool(job_meta.get("add_summary")) if job_meta else bool(add_summary)
    if not summary_requested:
        summary_requested = bool(add_summary)
    if job:
        job.meta = job_meta or {}
        job.meta["add_summary"] = summary_requested
        job.save_meta()
    
    # Wunsch-Sprache in job.meta speichern
    if job:
        job.meta["summary_lang"] = summary_lang or DEFAULT_SUMMARY_LANG
        job.save_meta()
    if not video_url and not upload_path:
        raise ValueError("Either video_url or upload_path must be provided")

    step_times: Dict[str, float] = {}
    progress_accumulator = 0.0

    temp_audio_path: Optional[Path] = None
    extracted_path: Optional[Path] = None
    segments: List[SegmentResult] = []
    metadata: Dict[str, Any] = {}
    base_result: Optional[TranscriptResult] = None
    include_source_flag = include_source
    summary_rel_path: Optional[str] = None
    summary_error: Optional[str] = None

    job_id = job.id if job else "local-run"
    job_dir = _ensure_job_dir(job_id)
    log_job_start(job_id, video_url, profile, None)

    overall_start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            # Download or ingest upload
            start = time.perf_counter()
            if upload_path:
                _update_job_meta("ingesting", progress_accumulator, {"status_detail": "processing uploaded file"})
                source_file = Path(upload_path)
                if not source_file.exists():
                    raise ValueError("Uploaded file not found")
                temp_audio_path = temp_dir / source_file.name
                shutil.copy2(source_file, temp_audio_path)
            else:
                _update_job_meta("downloading", progress_accumulator, {"status_detail": "downloading source media"})
                temp_audio_path = download_audio(video_url, temp_dir)
            progress_accumulator += STEP_WEIGHTS["download"]
            _record_step("download", start, step_times)
            _update_job_meta("downloaded", progress_accumulator, {"status_detail": "media downloaded"})

            # Extract
            _update_job_meta("extracting", progress_accumulator, {"status_detail": "extracting audio track"})
            start = time.perf_counter()
            extracted_path = _extract_audio(temp_audio_path, temp_dir / "audio_16k.wav")
            progress_accumulator += STEP_WEIGHTS["extract"]
            _record_step("extract", start, step_times)
            _update_job_meta("extracted", progress_accumulator, {"status_detail": "audio extracted"})

            # Transcribe
            _update_job_meta("transcribing", progress_accumulator, {"status_detail": "running whisper transcription"})
            start = time.perf_counter()
            segments, info, elapsed, device_profile, rtf, audio_seconds = transcribe_once(
                extracted_path,
                task=whisper_task,
                language=source_language,
            )
            progress_accumulator += STEP_WEIGHTS["transcribe"]
            _record_step("transcribe", start, step_times)

            base_text = " ".join(segment.text for segment in segments).strip()
            if not base_text:
                raise ValueError("No speech detected in the audio.")

        metadata = {
            "transcript_text": base_text,
            "detected_language": info.language,
            "language_probability": f"{info.language_probability:.2f}",
            "device_profile": device_profile.key,
            "model_id": device_profile.model_id,
            "device": device_profile.device,
            "compute_type": device_profile.compute_type,
            "transcription_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "rtf": rtf,
            "segment_count": len(segments),
            "segments": [segment.__dict__ for segment in segments],
            "profile": profile,
            "summary_requested": summary_requested,
        }

        base_language = determine_base_language(whisper_task, source_language, metadata)
        base_result = TranscriptResult(target_language=base_language, text=base_text)
        metadata["base_language"] = base_language
        metadata["whisper_task"] = whisper_task
        
        # Effektive Sprache für Summary festlegen
        requested_lang = (job.meta.get("summary_lang") if job else None) or summary_lang or DEFAULT_SUMMARY_LANG
        summary_lang_effective = base_language if requested_lang in ("auto", "same", None, "") else requested_lang
        
        metadata["summary_lang_requested"] = requested_lang
        metadata["summary_lang_effective"] = summary_lang_effective
        _update_job_meta(
            "transcribed",
            progress_accumulator,
            {
                "metadata": metadata,
                "status_detail": f"transcription complete (device={device_profile.key}, rtf={metadata['rtf']:.3f})",
            },
        )
        # No translation fanout; account for step weight and move on.
        progress_accumulator += STEP_WEIGHTS["translate"]
        _update_job_meta("translated", progress_accumulator, {"status_detail": "translations processed"})
        transcript_entries: List[Dict[str, Any]] = []
        transcripts_payload: List[Dict[str, str]] = []

        if include_source_flag:
            transcript_entries.append(
                {
                    "target_language": base_result.target_language,
                    "text": base_result.text,
                    "is_base": True,
                    "segments": [segment.__dict__ for segment in segments],
                }
            )
            transcripts_payload.append(base_result.__dict__)

        # Export
        _update_job_meta("exporting", progress_accumulator, {"status_detail": "rendering transcript files"})
        start = time.perf_counter()
        outputs = _export_transcripts(job_dir, transcript_entries, segments, metadata["audio_seconds"])
        progress_accumulator = 1.0
        _record_step("export", start, step_times)

        files_dir = job_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        summary_llm_key_present: Optional[bool] = None
        summary_generation_success = False
        summary_lang_code: Optional[str] = None
        summary_content: Optional[str] = None

        if summary_requested:
            LOGGER.info("=" * 80)
            LOGGER.info("🤖 AI SUMMARY GENERATION REQUESTED")
            LOGGER.info("=" * 80)
            LOGGER.info(f"  📝 Base text length: {len(base_text)} chars")
            LOGGER.info(f"  🔢 Segments count: {len(segments)}")
            LOGGER.info(f"  🌍 Target language: {metadata.get('summary_lang_effective', 'auto')}")
            LOGGER.info(f"  🌍 Requested language: {metadata.get('summary_lang_requested', 'auto')}")
            LOGGER.info("=" * 80)
            summary_lang_code = (
                metadata.get("summary_lang_effective")
                or metadata.get("base_language")
                or metadata.get("detected_language")
                or "en"
            )
            summary_lang_code = (summary_lang_code or "en").strip().lower() or "en"
            LOGGER.info(
                "  🌍 Resolved summary language: %s (detected=%s, requested=%s)",
                summary_lang_code,
                base_language,
                requested_lang,
            )
            summary_llm_key_present = bool(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GROQ_API_KEY")
            )
            LOGGER.info(f"  🔑 LLM key present: {summary_llm_key_present}")
            summary_content = None

            if not summary_llm_key_present:
                summary_error = summary_error or "Missing API key for AI summary generation."
                summary_content = "AI summary not available (missing API key on server).\n"
                LOGGER.warning("Skipping AI summary generation because no LLM key is configured.")
            else:
                try:
                    target_summary_lang = (
                        summary_lang_code
                        or metadata.get("summary_lang_effective")
                        or base_language
                        or metadata.get("detected_language")
                        or "en"
                    )
                    metadata["summary_lang_effective"] = target_summary_lang
                    LOGGER.info("🚀 Calling generate_summary()...")
                    summary_text = generate_summary(
                        base_text,
                        segments=segments,
                        target_lang=target_summary_lang,
                    )
                    LOGGER.info("=" * 80)
                    LOGGER.info("✅ SUMMARY GENERATION SUCCESSFUL!")
                    LOGGER.info("=" * 80)
                    LOGGER.info(f"  📊 Summary length: {len(summary_text)} chars")
                    LOGGER.info(f"  📝 Preview: {summary_text[:200]}...")
                    LOGGER.info("=" * 80)

                    summary_content = summary_text.strip() + "\n"
                    summary_generation_success = True

                except SummaryGenerationError as err:
                    summary_error = str(err)
                    LOGGER.error("=" * 80)
                    LOGGER.error("❌ SUMMARY GENERATION ERROR")
                    LOGGER.error("=" * 80)
                    LOGGER.error(f"  🔴 Error type: SummaryGenerationError")
                    LOGGER.error(f"  💬 Error message: {err}")
                    LOGGER.error("=" * 80)
                    LOGGER.exception("Full traceback:")
                    LOGGER.error("=" * 80)
                except Exception as err:  # pragma: no cover - safeguard
                    summary_error = f"Summary generation failed: {err}"
                    LOGGER.error("=" * 80)
                    LOGGER.error("❌ UNEXPECTED SUMMARY ERROR")
                    LOGGER.error("=" * 80)
                    LOGGER.error(f"  🔴 Error type: {type(err).__name__}")
                    LOGGER.error(f"  💬 Error message: {err}")
                    LOGGER.error("=" * 80)
                    LOGGER.exception("Full traceback:")
                    LOGGER.error("=" * 80)
            summary_lang_dir = files_dir / summary_lang_code
            summary_lang_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_lang_dir / "summary.txt"

            if not summary_content:
                reason = summary_error or "Summary generator returned no content."
                summary_content = f"AI summary not available. Reason: {reason}\n"
            if not summary_content.endswith("\n"):
                summary_content += "\n"

            summary_path.write_text(summary_content, encoding="utf-8")
            summary_rel_path = summary_path.relative_to(files_dir).as_posix()
            metadata["summary_file"] = summary_rel_path

            LOGGER.info(f"📄 Summary saved to: {summary_path}")
            LOGGER.info(f"📋 Summary file in metadata: {metadata.get('summary_file')}")
            try:
                files_listing = [
                    str(path.relative_to(files_dir))
                    for path in files_dir.rglob("*")
                    if path.is_file()
                ]
            except Exception as listing_error:  # pragma: no cover - diagnostic logging
                LOGGER.warning("Failed to enumerate summary output files for job %s: %s", job_id, listing_error)
            else:
                LOGGER.info(f"📂 Files in output directory: {files_listing}")

        if summary_rel_path:
            metadata["summary_file"] = summary_rel_path
        if summary_error:
            metadata["summary_error"] = summary_error
        if summary_llm_key_present is not None:
            metadata["summary_llm_key_present"] = summary_llm_key_present
        metadata["summary_generation_success"] = summary_generation_success

        total_elapsed = time.perf_counter() - overall_start
        download_base = f"/jobs/{job_id}/files"
        result = {
            "job_id": job_id,
            "video_url": video_url,
            "upload_path": upload_path,
            "metadata": metadata,
            "transcripts": transcripts_payload,
            "outputs": outputs,
            "step_times": step_times,
            "total_seconds": total_elapsed,
            "device_profile": metadata["device_profile"],
            "rtf": metadata["rtf"],
            "download_base": download_base,
        }
        if summary_rel_path:
            result["summary_file"] = summary_rel_path
        if summary_error:
            result["summary_error"] = summary_error

        finished_meta: Dict[str, Any] = {
            "metadata": metadata,
            "outputs": outputs,
            "download_base": download_base,
            "status_detail": "job finished",
        }
        if summary_rel_path:
            finished_meta["summary_file"] = summary_rel_path
        if summary_error:
            finished_meta["summary_error"] = summary_error

        _update_job_meta("finished", progress_accumulator, finished_meta)
        if job:
            job.meta["step_times"] = step_times
            if summary_rel_path:
                job.meta["summary_file"] = summary_rel_path
            if summary_error:
                job.meta["summary_error"] = summary_error
            job.save_meta()
        if metadata.get("audio_seconds"):
            try:
                user_email = job.meta.get("user_email") if job else None
                if user_email:
                    actual_minutes = metadata["audio_seconds"] / 60.0
                    UserManager.deduct_minutes(user_email, actual_minutes, job_id)
            except Exception as e:  # pragma: no cover - safeguard
                LOGGER.error("Failed to deduct credits: %s", e, exc_info=True)
        log_job_complete(
            job_id,
            metadata.get("rtf"),
            metadata.get("audio_seconds"),
            metadata.get("transcription_seconds"),
            total_elapsed,
        )
        
        LOGGER.info("=" * 80)
        LOGGER.info(f"✅ JOB COMPLETED SUCCESSFULLY: {job_id}")
        LOGGER.info("=" * 80)
        LOGGER.info(f"  ⏱️ Total time: {total_elapsed:.2f}s")
        LOGGER.info(f"  🎯 RTF: {metadata.get('rtf', 'N/A')}")
        LOGGER.info(f"  📊 Audio duration: {metadata.get('audio_seconds', 'N/A')}s")
        if summary_rel_path:
            LOGGER.info(f"  📝 Summary: ✅ Generated")
        elif summary_error:
            LOGGER.info(f"  📝 Summary: ❌ Failed - {summary_error}")
        else:
            LOGGER.info(f"  📝 Summary: Not requested")
        LOGGER.info("=" * 80)
        
        return result
    except Exception as exc:
        LOGGER.error("=" * 80)
        LOGGER.error(f"❌ JOB FAILED: {job_id}")
        LOGGER.error("=" * 80)
        LOGGER.error(f"  🔴 Error type: {type(exc).__name__}")
        LOGGER.error(f"  💬 Error message: {str(exc)}")
        LOGGER.error("=" * 80)
        LOGGER.exception("Full traceback:")
        LOGGER.error("=" * 80)
        
        _update_job_meta("failed", progress_accumulator, {"error": str(exc), "status_detail": "job failed"})
        log_job_error(job_id, exc)
        raise


def enqueue_ping_job() -> str:
    job = job_queue.enqueue(ping_job)
    return job.id


def enqueue_transcription_job(
    video_url: Optional[str],
    targets: Optional[List[str]],
    profile: str = "fast",
    include_source: bool = True,
    source_language: Optional[str] = None,
    whisper_task: str = "translate",
    upload_path: Optional[str] = None,
    user_email: Optional[str] = None,
    add_summary: bool = False,
    summary_lang: Optional[str] = None,
) -> Job:
    _, include_source_override = normalize_language_inputs(targets or [])
    include_source_final = include_source or include_source_override
    job_timeout = int(os.getenv("JOB_TIMEOUT", "3600"))
    result_ttl = int(os.getenv("JOB_RESULT_TTL", "86400"))
    job = job_queue.enqueue(
        process_job,
        kwargs={
            "video_url": video_url,
            "upload_path": upload_path,
            "targets": targets or [],
            "profile": profile,
            "include_source": include_source_final,
            "source_language": source_language,
            "whisper_task": whisper_task,
            "add_summary": add_summary,
            "summary_lang": summary_lang or DEFAULT_SUMMARY_LANG,
        },
        job_timeout=job_timeout,
        result_ttl=result_ttl,
    )
    job.meta = job.meta or {}
    if user_email:
        job.meta["user_email"] = user_email
    job.meta["add_summary"] = add_summary
    job.meta["summary_lang"] = summary_lang or DEFAULT_SUMMARY_LANG
    job.save_meta()
    return job


def get_job_status(job_id: str) -> Dict[str, Any]:
    try:
        job = Job.fetch(job_id, connection=redis_connection)
    except NoSuchJobError as exc:
        raise ValueError(f"Job '{job_id}' not found") from exc
    status = job.get_status()
    meta = job.meta or {}
    response: Dict[str, Any] = {
        "job_id": job_id,
        "status": meta.get("status", status),
        "progress": meta.get("progress", 0.0),
        "step_times": meta.get("step_times") or meta.get("steps"),
        "rtf": meta.get("metadata", {}).get("rtf") if meta.get("metadata") else meta.get("rtf"),
        "device_profile": meta.get("metadata", {}).get("device_profile") if meta.get("metadata") else meta.get("device_profile"),
        "error": meta.get("error"),
    }
    if "cookie_file_present" in meta:
        response["cookie_file_present"] = meta["cookie_file_present"]
        response["cookie_file_path"] = meta.get("cookie_file_path")
    if "download_base" in meta:
        response["download_base"] = meta["download_base"]

    if status == "finished" and job.result:
        response["result"] = job.result
    elif status == "failed":
        response["error"] = response.get("error") or str(job.exc_info).splitlines()[-1] if job.exc_info else "Job failed"

    return response


def materialize_summary_file_safe(job: Dict[str, Any]) -> None:
    summary_path_str = None
    try:
        job_id = str(job.get("job_id") or "unknown")
        metadata: Dict[str, Any] = job.setdefault("metadata", {})
        provided_rel_path = (job.get("summary_file") or metadata.get("summary_file") or "").strip()
        summary_lang = (
            job.get("summary_lang")
            or metadata.get("summary_lang_effective")
            or metadata.get("base_language")
            or metadata.get("detected_language")
            or "en"
        )
        summary_lang = (summary_lang or "en").strip().lower() or "en"
        job["summary_lang"] = summary_lang

        files_dir = JOB_STORAGE_ROOT / job_id / "files"
        if provided_rel_path:
            summary_rel_path_path = Path(provided_rel_path.replace("\\", "/"))
            summary_path = files_dir / summary_rel_path_path
        else:
            summary_rel_path_path = Path(summary_lang) / "summary.txt"
            summary_path = files_dir / summary_rel_path_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path_str = str(summary_path)

        summary_error = job.get("summary_error") or metadata.get("summary_error")
        summary_content = job.get("summary_content")
        if not summary_content:
            reason = summary_error or "AI summary not available."
            summary_content = f"AI summary not available. Reason: {reason}\n"
        if not summary_content.endswith("\n"):
            summary_content += "\n"

        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(summary_content)

        relative_path = summary_path.relative_to(files_dir).as_posix()
        metadata["summary_file"] = relative_path
        job["summary_file"] = relative_path
        job["summary_path"] = summary_path_str

        LOGGER.info(f"📄 Summary saved to: {summary_path}")
        LOGGER.info(f"📋 Summary file in metadata: {metadata.get('summary_file')}")
        try:
            files_listing = [
                str(path.relative_to(files_dir))
                for path in files_dir.rglob("*")
                if path.is_file()
            ]
        except Exception as listing_error:  # pragma: no cover - diagnostic logging
            LOGGER.warning("Failed to enumerate summary output files for job %s: %s", job_id, listing_error)
        else:
            LOGGER.info(f"📂 Files in output directory: {files_listing}")

        LOGGER.info(
            "Summary materialized safely: job_id=%s summary_lang=%s summary_path=%s llm_key_available=%s error=%s",
            job_id,
            summary_lang,
            summary_path_str,
            job.get("summary_llm_key_present"),
            summary_error,
        )
    except Exception as exc:
        LOGGER.exception("could not write summary to %s", summary_path_str or "unknown path")
        job["summary_materialization_error"] = str(exc)
        if summary_path_str:
            job["summary_path"] = summary_path_str


def ensure_summary_materialization_safe(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "job_id": job_id,
        "summary_lang": None,
        "summary_path": None,
        "llm_key_available": None,
    }
    try:
        metadata = payload.get("metadata") or {}
        job_ctx: Dict[str, Any] = {
            "job_id": job_id,
            "metadata": metadata,
            "summary_lang": (
                metadata.get("summary_lang_effective")
                or metadata.get("base_language")
                or metadata.get("detected_language")
                or payload.get("summary_lang")
                or "en"
            ),
            "summary_error": payload.get("summary_error") or metadata.get("summary_error"),
            "summary_llm_key_present": metadata.get("summary_llm_key_present"),
            "summary_content": None,
            "summary_file": metadata.get("summary_file"),
        }
        materialize_summary_file_safe(job_ctx)
        if job_ctx.get("summary_file"):
            payload["summary_file"] = job_ctx["summary_file"]
            metadata["summary_file"] = job_ctx["summary_file"]
            info["summary_path"] = job_ctx.get("summary_path")
        info["summary_lang"] = job_ctx.get("summary_lang")
        info["llm_key_available"] = job_ctx.get("summary_llm_key_present")
        if job_ctx.get("summary_materialization_error"):
            info["error"] = job_ctx["summary_materialization_error"]
    except Exception as exc:
        LOGGER.warning("non-fatal summary materialization error for job %s: %s", job_id, exc)
        info["error"] = str(exc)
    return info
