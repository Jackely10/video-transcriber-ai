from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
import yt_dlp
from faster_whisper import WhisperModel
from yt_dlp.utils import DownloadError

SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
CPU_THREADS = os.cpu_count() or 4
MODEL_CACHE: Dict[str, WhisperModel] = {}
MODEL_LOCK = threading.Lock()
SILENCE_SAMPLE_PATH: Optional[Path] = None

SUPPORTED_LANGUAGES: set[str] = set()
LANGUAGE_ALIASES: Dict[str, str] = {"source": "source"}


logger = logging.getLogger(__name__)


def normalize_language_inputs(values: Iterable[str]) -> Tuple[List[str], bool]:
    include_source = False
    for value in values:
        if value.strip().lower() == "source":
            include_source = True
    return [], include_source


@dataclass
class SegmentResult:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    target_language: str
    text: str


@dataclass(frozen=True)
class DeviceProfile:
    key: str
    model_id: str
    device: str
    compute_type: str


def download_audio(url: str, temp_dir: Path) -> Path:
    audio_file = temp_dir / "%(id)s.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_file),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "extractor_args": {
            "youtube": {
                "skip": ["dash", "hls"],
                "player_skip": ["configs", "webpage"],
            },
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except DownloadError as exc:
            raise ValueError(f"Audio konnte nicht heruntergeladen werden: {exc}") from exc
        download_path = Path(ydl.prepare_filename(info))

    if not download_path.exists():
        raise FileNotFoundError(f"Downloaded audio file not found: {download_path}")

    return download_path


def _extract_audio(input_path: Path, output_path: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def _select_device_profile(task: str, source_language: Optional[str]) -> DeviceProfile:
    env_model = os.getenv("WHISPER_MODEL_ID")
    env_compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    default_cpu_transcribe = os.getenv("WHISPER_CPU_MODEL_ID_TRANSCRIBE")
    default_cpu_translate = os.getenv("WHISPER_CPU_MODEL_ID_TRANSLATE")
    default_cpu_general = os.getenv("WHISPER_CPU_MODEL_ID")

    if torch is not None and torch.cuda.is_available() and os.getenv("WHISPER_DEVICE", "auto") != "cpu":
        model_id = env_model or os.getenv("WHISPER_GPU_MODEL_ID", "medium")
        return DeviceProfile(key=f"cuda_{model_id}", model_id=model_id, device="cuda", compute_type="float16")

    # CPU fallback
    if env_model:
        model_id = env_model
    else:
        lang = (source_language or "").lower()
        if task == "transcribe":
            if default_cpu_general:
                model_id = default_cpu_general
            elif lang.startswith("en"):
                model_id = os.getenv("WHISPER_CPU_MODEL_ID_EN", "Systran/faster-whisper-tiny.en")
            else:
                model_id = default_cpu_transcribe or "tiny"
        else:
            if default_cpu_general:
                model_id = default_cpu_general
            else:
                model_id = default_cpu_translate or "Systran/faster-whisper-tiny.en"

    return DeviceProfile(key=f"cpu_{model_id}", model_id=model_id, device="cpu", compute_type=env_compute)


def _load_model(profile: DeviceProfile) -> WhisperModel:
    cache_key = f"{profile.model_id}_{profile.device}_{profile.compute_type}"
    with MODEL_LOCK:
        model = MODEL_CACHE.get(cache_key)
        if model is None:
            kwargs: Dict[str, Any] = {}
            if profile.device == "cpu":
                kwargs["cpu_threads"] = CPU_THREADS
                kwargs["num_workers"] = min(4, CPU_THREADS)
            model = WhisperModel(
                profile.model_id,
                device=profile.device,
                compute_type=profile.compute_type,
                **kwargs,
            )
            MODEL_CACHE[cache_key] = model
    return model


def preload_default_whisper_model() -> DeviceProfile:
    """
    Load the default Whisper model once on startup so Railway downloads the
    weights before jobs arrive. This helps avoid mid-job OOM kills.
    """
    profile = _select_device_profile(task=os.getenv("WHISPER_PRELOAD_TASK", "translate"), source_language=None)
    logger.info("Preloading Whisper model %s (%s)", profile.model_id, profile.device)
    _load_model(profile)
    return profile


def _ensure_silence_sample(duration: float = 10.0, sample_rate: int = SAMPLE_RATE) -> Path:
    global SILENCE_SAMPLE_PATH
    if SILENCE_SAMPLE_PATH and SILENCE_SAMPLE_PATH.exists():
        return SILENCE_SAMPLE_PATH

    tmp_path = Path(tempfile.gettempdir()) / f"video_transcriber_silence_{sample_rate}_{int(duration)}s.wav"
    if not tmp_path.exists():
        frames = int(duration * sample_rate)
        silence = array("h", [0] * frames)
        with wave.open(str(tmp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silence.tobytes())

    SILENCE_SAMPLE_PATH = tmp_path
    return tmp_path


def transcribe_once(
    audio_path: Path,
    task: str = "transcribe",
    language: Optional[str] = None,
) -> Tuple[List[SegmentResult], Any, float, DeviceProfile, float, float]:
    profile = _select_device_profile(task=task, source_language=language)
    model = _load_model(profile)

    start = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        beam_size=1,
        best_of=1,
        vad_filter=True,  # Aktiviert VAD - filtert Stille!
        word_timestamps=False,
        condition_on_previous_text=False,
        temperature=0.0,
        chunk_length=30,  # Groessere Chunks = schneller
    )
    segments = [SegmentResult(start=segment.start, end=segment.end, text=segment.text.strip()) for segment in segments_iter]
    elapsed = time.perf_counter() - start

    if segments and segments[-1].end and segments[-1].end > 0:
        audio_duration = float(segments[-1].end)
    else:
        audio_duration = float(info.duration or 0.0)
    if audio_duration <= 0:
        audio_duration = 1e-6
    rtf = elapsed / audio_duration

    return segments, info, elapsed, profile, rtf, audio_duration


def _transcription_metadata(
    base_text: str,
    info: Any,
    profile: DeviceProfile,
    elapsed: float,
    audio_seconds: float,
    segments: List[SegmentResult],
    task: str,
) -> Dict[str, Any]:
    return {
        "transcript_text": base_text,
        "detected_language": getattr(info, "language", None),
        "language_probability": f"{getattr(info, 'language_probability', 0.0):.2f}" if hasattr(info, "language_probability") else "0.00",
        "device_profile": profile.key,
        "model_id": profile.model_id,
        "device": profile.device,
        "compute_type": profile.compute_type,
        "transcription_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "rtf": elapsed / max(audio_seconds, 1e-6),
        "segment_count": len(segments),
        "segments": [segment.__dict__ for segment in segments],
        "whisper_task": task,
    }


def determine_base_language(task: str, source_language: Optional[str], metadata: Dict[str, Any]) -> str:
    if task == "translate":
        return "english"
    if source_language:
        return source_language.lower()
    detected = metadata.get("detected_language")
    return detected if detected else "unknown"


def transcribe_video(
    url: str,
    include_source: bool,
    whisper_task: str,
    model_size: str,
    device: str,
    compute_type: str,
    beam_size: int,
    source_language: Optional[str],
    status_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], TranscriptResult, List[TranscriptResult], bool]:
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        if status_callback:
            status_callback("Downloading audio...")
        audio_path = download_audio(url, temp_dir)
        if status_callback:
            status_callback("Extracting audio...")
        resampled_path = _extract_audio(audio_path, temp_dir / "audio.wav", sample_rate=SAMPLE_RATE)
        if status_callback:
            status_callback("Transcribing...")
        segments, info, elapsed, profile, rtf, audio_seconds = transcribe_once(
            resampled_path,
            task=whisper_task,
            language=source_language,
        )

    base_text = " ".join(segment.text for segment in segments).strip()
    if not base_text:
        raise ValueError("No speech detected in the audio.")

    metadata = _transcription_metadata(
        base_text=base_text,
        info=info,
        profile=profile,
        elapsed=elapsed,
        audio_seconds=audio_seconds,
        segments=segments,
        task=whisper_task,
    )

    base_language = determine_base_language(whisper_task, source_language, metadata)
    base_result = TranscriptResult(target_language=base_language, text=base_text)

    transcripts: List[TranscriptResult] = []
    include_source_flag = include_source or whisper_task == "translate"

    return metadata, base_result, transcripts, include_source_flag


def write_output(output_path: Path, metadata: Dict[str, Any], transcripts: List[TranscriptResult]) -> None:
    payload = {
        "metadata": metadata,
        "transcripts": [result.__dict__ for result in transcripts],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text_output(output_path: Path, metadata: Dict[str, Any], transcripts: List[TranscriptResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Detected source language: {metadata.get('detected_language')} (p={metadata.get('language_probability')})",
        "",
    ]
    for transcript in transcripts:
        lines.append(f"=== {transcript.target_language} ===")
        lines.append(transcript.text)
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_rtf_benchmark(duration: float = 10.0, task: str = "transcribe") -> Dict[str, float]:
    sample_path = _ensure_silence_sample(duration=duration)
    segments, info, elapsed, profile, rtf, audio_seconds = transcribe_once(sample_path, task=task, language=None)
    return {
        "rtf": rtf,
        "elapsed_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "device_profile": profile.key,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a video and extract speech as text.")
    parser.add_argument("url", help="YouTube or Facebook video URL.")
    parser.add_argument(
        "--source-language",
        default=None,
        help="Optional ISO language code hint for the source audio (e.g., 'en', 'ar').",
    )
    parser.add_argument(
        "--whisper-task",
        choices=("translate", "transcribe"),
        default="transcribe",
        help="Use 'transcribe' to keep the source language (default) or 'translate' for Whisper's English output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON output with transcripts.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional path to write plain text transcripts (UTF-8).",
    )
    parser.add_argument(
        "--show-console-output",
        action="store_true",
        help="Display the transcript in the console even when writing to files.",
    )
    parser.add_argument(
        "--include-source",
        dest="include_source",
        action="store_true",
        default=True,
        help="Include the source-language transcript (default).",
    )
    parser.add_argument(
        "--no-include-source",
        dest="include_source",
        action="store_false",
        help="Unterdrueckt das Originaltranskript.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        metadata, base_result, transcripts, include_source_flag = transcribe_video(
            url=args.url,
            include_source=args.include_source,
            whisper_task=args.whisper_task,
            model_size="auto",
            device="auto",
            compute_type="auto",
            beam_size=1,
            source_language=args.source_language,
            status_callback=lambda msg: print(msg, file=sys.stderr),
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    combined_results = ([base_result] if include_source_flag else []) + transcripts

    if args.show_console_output:
        for result in combined_results:
            print(f"\n=== Transcript ({result.target_language}) ===\n")
            print(result.text)

    if args.output:
        write_output(args.output, metadata, combined_results)
        print(f"Saved output to {args.output}", file=sys.stderr)

    if args.text_output:
        write_text_output(args.text_output, metadata, combined_results)
        print(f"Saved text transcripts to {args.text_output}", file=sys.stderr)

    if not args.show_console_output and not args.output and not args.text_output and combined_results:
        print(combined_results[0].text)

    return 0


if __name__ == "__main__":
    sys.exit(main())


def get_runtime_device_info() -> Dict[str, object]:
    cuda_available = torch.cuda.is_available() if torch is not None else False
    info: Dict[str, object] = {
        "cuda_available": cuda_available,
        "device_name": "cpu",
        "total_vram_bytes": None,
        "cpu_threads": os.cpu_count() or 1,
    }
    if cuda_available and torch is not None:
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_vram_bytes"] = int(props.total_memory)
    return info
