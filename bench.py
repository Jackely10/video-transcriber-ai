from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from video_transcriber import _ensure_silence_sample, transcribe_once


def benchmark(audio_path: Path, task: str = "transcribe", language: str | None = None) -> dict:
    overall_start = time.perf_counter()
    segments, info, elapsed, profile, rtf, audio_seconds = transcribe_once(
        audio_path=audio_path,
        task=task,
        language=language,
    )
    overall_elapsed = time.perf_counter() - overall_start
    return {
        "segment_count": len(segments),
        "audio_seconds": audio_seconds,
        "transcription_seconds": elapsed,
        "overall_seconds": overall_elapsed,
        "rtf": rtf,
        "device_profile": profile.key,
        "detected_language": info.language,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper transcription performance.")
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional path to an audio file to benchmark. If omitted, a synthetic 10s sample is used.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=("transcribe", "translate"),
        help="Whisper task to benchmark (defaults to transcribe).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint (ISO code) passed to Whisper.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration for the synthetic benchmark sample when --audio is not provided.",
    )
    args = parser.parse_args()

    if args.audio:
        audio_path = args.audio
    else:
        audio_path = _ensure_silence_sample(duration=args.duration)

    results = benchmark(audio_path, task=args.task, language=args.language)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
