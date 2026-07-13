"""Tests for tulving.cli.eval_scoring — written BEFORE implementation.

All LLM calls are mocked via an injected `ScoreTransport` callable — no network in the suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tulving.cli.eval_scoring import (
    AnswerScorer,
    load_probes,
    score_probes,
)
from tulving.exceptions import MemoryStoreError


class _RecordingTransport:
    """Fake ScoreTransport recording every call; scripted responses by question."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
    ) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        question = payload["messages"][-1]["content"]
        system = payload["messages"][0]["content"]
        if "CONTEXT:\n(none)" in system:
            # The "none" condition: no context was supplied, so the model cannot know.
            answer = "unknown"
        else:
            answer = self.answers.get(question, "unknown")
        return {"choices": [{"message": {"content": answer}}]}


class _RaisingTransport:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __call__(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
    ) -> dict[str, Any]:
        raise self._exc


class _MalformedTransport:
    def __call__(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
    ) -> dict[str, Any]:
        return {"not": "the expected shape"}


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_load_probes_bad_shape_raises_memorystoreerror(self, tmp_path: Path) -> None:
        path = tmp_path / "probes.json"
        path.write_text(json.dumps([{"q": "only a question"}]), encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            load_probes(path)

    def test_load_probes_empty_list_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "probes.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            load_probes(path)

    def test_load_probes_not_a_list_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "probes.json"
        path.write_text(json.dumps({"q": "x", "expect": "y"}), encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            load_probes(path)

    def test_load_probes_corrupt_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "probes.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            load_probes(path)

    def test_malformed_json_response_treated_as_scoring_failure(self) -> None:
        scorer = AnswerScorer(
            "http://localhost:1234/v1/chat/completions", "m", 5, _MalformedTransport()
        )
        with pytest.raises(MemoryStoreError):
            scorer.ask("context", "question?")

    def test_scorer_propagates_transport_errors(self) -> None:
        import urllib.error

        transport = _RaisingTransport(urllib.error.URLError("connection refused"))
        scorer = AnswerScorer("http://localhost:1234/v1/chat/completions", "m", 5, transport)
        with pytest.raises(urllib.error.URLError):
            scorer.ask("context", "question?")


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_load_probes_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "probes.json"
        path.write_text(
            json.dumps([{"q": "Did we choose SQLite?", "expect": "SQLite"}]), encoding="utf-8"
        )
        probes = load_probes(path)
        assert probes == [{"q": "Did we choose SQLite?", "expect": "SQLite"}]

    def test_score_probes_empty_probe_list(self) -> None:
        scorer = AnswerScorer(
            "http://localhost:1234/v1/chat/completions", "m", 5, _RecordingTransport({})
        )
        result = score_probes(scorer, [], dump="", curate_fn=lambda _q: "")
        assert result == {"none": "0/0", "dump": "0/0", "curate": "0/0"}


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_score_probes_tallies_with_mock_transport(self) -> None:
        probes = [
            {"q": "What datastore?", "expect": "SQLite"},
            {"q": "Which auth?", "expect": "JWT"},
            {"q": "Token lifetime?", "expect": "15"},
        ]
        transport = _RecordingTransport(
            {
                "What datastore?": "SQLite is the datastore.",
                "Which auth?": "We use JWT.",
                "Token lifetime?": "15 minutes.",
            }
        )
        scorer = AnswerScorer("http://localhost:1234/v1/chat/completions", "m", 5, transport)
        result = score_probes(
            scorer,
            probes,
            dump="[fact] datastore SQLite; [fact] auth JWT; [fact] ttl 15",
            curate_fn=lambda q: "SQLite is the datastore. We use JWT. 15 minutes.",
        )
        assert result == {"none": "0/3", "dump": "3/3", "curate": "3/3"}
        # Every probe was asked under all three conditions.
        assert len(transport.calls) == 9

    def test_scorer_omits_auth_header_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TULVING_EVAL_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        transport = _RecordingTransport({"question?": "answer"})
        scorer = AnswerScorer("http://localhost:1234/v1/chat/completions", "m", 5, transport)
        scorer.ask("context", "question?")
        assert "Authorization" not in transport.calls[0]["headers"]

    def test_scorer_adds_bearer_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TULVING_EVAL_API_KEY", "sk-test-secret-value-123456")
        transport = _RecordingTransport({"question?": "answer"})
        scorer = AnswerScorer("http://localhost:1234/v1/chat/completions", "m", 5, transport)
        scorer.ask("context", "question?")
        auth = transport.calls[0]["headers"]["Authorization"]
        assert auth == "Bearer sk-test-secret-value-123456"

    def test_scorer_fallback_to_openai_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TULVING_EVAL_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback-secret-7890123456")
        transport = _RecordingTransport({"question?": "answer"})
        scorer = AnswerScorer("http://localhost:1234/v1/chat/completions", "m", 5, transport)
        scorer.ask("context", "question?")
        auth = transport.calls[0]["headers"]["Authorization"]
        assert auth == "Bearer sk-fallback-secret-7890123456"
