import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import jobs  # noqa: E402


def _build_job(job_id: str, **overrides):
    base = {
        "job_id": job_id,
        "metadata": {},
    }
    base.update(overrides)
    return base


def test_materialize_writes_summary_content(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_STORAGE_ROOT", tmp_path)
    job = _build_job(
        "job-success",
        summary_content="Hallo Welt\n",
        metadata={
            "summary_generation_success": True,
            "summary_content": "Hallo Welt\n",
            "summary_lang_effective": "de",
        },
    )

    jobs.materialize_summary_file_safe(job)

    summary_path = tmp_path / "job-success" / "files" / "de" / "summary.txt"
    assert summary_path.read_text(encoding="utf-8") == "Hallo Welt\n"


def test_materialize_falls_back_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_STORAGE_ROOT", tmp_path)
    job = _build_job(
        "job-failed",
        metadata={
            "summary_generation_success": False,
            "summary_error": "Invalid API key",
        },
    )

    jobs.materialize_summary_file_safe(job)

    summary_path = tmp_path / "job-failed" / "files" / "en" / "summary.txt"
    content = summary_path.read_text(encoding="utf-8")
    assert "Invalid API key" in content


def test_materialize_keeps_existing_success_file(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_STORAGE_ROOT", tmp_path)
    summary_dir = tmp_path / "job-existing" / "files" / "de"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "summary.txt"
    summary_path.write_text("Bestehender Inhalt\n", encoding="utf-8")

    job = _build_job(
        "job-existing",
        metadata={
            "summary_generation_success": True,
            "summary_file": "de/summary.txt",
            "summary_lang_effective": "de",
        },
    )

    jobs.materialize_summary_file_safe(job)

    assert summary_path.read_text(encoding="utf-8") == "Bestehender Inhalt\n"
    assert job["summary_file"] == "de/summary.txt"
