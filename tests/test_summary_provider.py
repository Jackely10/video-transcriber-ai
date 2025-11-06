import importlib
import sys
import types
from pathlib import Path
from typing import Optional

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _reload_ai_summary(monkeypatch, provider: str, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
    for var in ("AI_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    if provider is not None:
        monkeypatch.setenv("AI_PROVIDER", provider)
    if openai_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    if anthropic_key is not None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    import ai_summary

    return importlib.reload(ai_summary)


def test_generate_summary_requires_configured_provider(monkeypatch):
    module = _reload_ai_summary(monkeypatch, provider="anthropic")

    with pytest.raises(module.SummaryGenerationError) as excinfo:
        module.generate_summary("Kurzer Testtext.")

    message = str(excinfo.value).lower()
    assert "anthropic" in message
    assert "missing api key" in message


def test_generate_summary_reports_invalid_key(monkeypatch):
    module = _reload_ai_summary(monkeypatch, provider="openai", openai_key="sk-invalid")

    class DummyAuthError(Exception):
        def __init__(self, message: str):
            super().__init__(message)
            self.status_code = 401

    class DummyCompletions:
        def create(self, *args, **kwargs):
            raise DummyAuthError("Invalid API key provided")

    module.openai_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=DummyCompletions())
    )

    with pytest.raises(module.SummaryGenerationError) as excinfo:
        module.generate_summary("Noch ein Testtext.")

    message = str(excinfo.value).lower()
    assert "openai" in message
    assert "401" in message or "status 401" in message
    assert "invalid api key" in message
