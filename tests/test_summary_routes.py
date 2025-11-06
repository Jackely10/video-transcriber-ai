import sys
from pathlib import Path

import pytest

import server

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def client(monkeypatch):
    server.app.config.update(TESTING=True)
    return server.app.test_client()


def test_summary_endpoint_returns_json_error(monkeypatch, client):
    payload = {
        "status": "finished",
        "result": {
            "metadata": {
                "summary_requested": True,
                "summary_generation_success": False,
                "summary_error": "Summary provider 'openai' is not configured (missing API key).",
                "summary_provider": "openai",
                "summary_provider_key_present": False,
            }
        },
    }
    monkeypatch.setattr(server, "get_job_status", lambda _job_id: payload)

    resp = client.get("/jobs/demo/summary")

    assert resp.status_code == 422
    assert resp.json["summary_provider"] == "openai"
    assert resp.json["summary_provider_key_present"] is False


def test_summary_endpoint_serves_file(monkeypatch, client, tmp_path):
    job_id = "demo"
    summary_dir = tmp_path / job_id / "files" / "de"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "summary.txt"
    summary_path.write_text("Hallo Welt", encoding="utf-8")

    payload = {
        "status": "finished",
        "result": {
            "metadata": {
                "summary_requested": True,
                "summary_generation_success": True,
                "summary_file": "de/summary.txt",
            }
        },
    }

    monkeypatch.setattr(server, "JOB_STORAGE_ROOT", tmp_path)
    monkeypatch.setattr(server, "get_job_status", lambda _job_id: payload)
    monkeypatch.setattr(
        server,
        "ensure_summary_materialization_safe",
        lambda *_args, **_kwargs: {"summary_path": str(summary_path)},
    )

    resp = client.get(f"/jobs/{job_id}/summary")

    assert resp.status_code == 200
    assert "Hallo Welt" in resp.get_data(as_text=True)


def test_facts_endpoint_returns_summary_error(monkeypatch, client):
    monkeypatch.setattr(server, "FACT_CHECK_ENABLED", True)
    payload = {
        "status": "finished",
        "result": {
            "metadata": {
                "summary_requested": True,
                "summary_generation_success": False,
                "summary_error": "Missing API key",
                "summary_provider": "openai",
                "summary_provider_key_present": False,
            }
        },
    }
    monkeypatch.setattr(server, "get_job_status", lambda _job_id: payload)

    resp = client.get("/jobs/demo/facts")

    assert resp.status_code == 422
    assert resp.json["error"] == "Missing API key"
