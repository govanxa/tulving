"""Tulving CLI — probe correctness scoring against an OpenAI-compatible endpoint.

Stdlib ``urllib`` only. The transport is injectable (:class:`ScoreTransport`) so tests
never touch the network. Credentials come from the environment ONLY (security req #3):
there is no inline-API-key parameter anywhere in this module.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

from tulving.exceptions import MemoryStoreError

API_KEY_ENV: Final[str] = "TULVING_EVAL_API_KEY"
API_KEY_ENV_FALLBACK: Final[str] = "OPENAI_API_KEY"

_SYSTEM_PROMPT: Final[str] = (
    "Answer ONLY from the provided context. If it is not there, say 'unknown'."
)


class ScoreTransport(Protocol):
    """Injectable transport: one HTTP round-trip to a chat-completions endpoint."""

    def __call__(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
    ) -> dict[str, Any]:
        """POST ``payload`` as JSON to ``url``; return the parsed JSON response."""
        ...


def _urllib_transport(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
) -> dict[str, Any]:
    """Default transport: stdlib ``urllib.request`` (localhost by default)."""
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, body, request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # local endpoint only
        parsed: dict[str, Any] = json.load(response)
        return parsed


class AnswerScorer:
    """One-turn question answering against an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        url: str,
        model: str,
        timeout: int,
        transport: ScoreTransport = _urllib_transport,
    ) -> None:
        """Store configuration (cheap; no network at construction, D8).

        Args:
            url: OpenAI-compatible chat-completions URL.
            model: Model name reported to the endpoint.
            timeout: Per-request timeout in seconds.
            transport: Injectable HTTP transport (tests supply a fake).
        """
        self._url = url
        self._model = model
        self._timeout = timeout
        self._transport = transport

    def ask(self, context: str, question: str) -> str:
        """One turn: answer ``question`` using only ``context``.

        Args:
            context: The context block (dump/curate text, or empty for "none").
            question: The probe question.

        Returns:
            The stripped model answer.

        Raises:
            MemoryStoreError: The response JSON is not shaped as expected.
            Exception: Whatever the transport raises (network errors) propagate
                unchanged so callers can degrade loudly.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": f"{_SYSTEM_PROMPT}\n\nCONTEXT:\n{context or '(none)'}",
                },
                {"role": "user", "content": question},
            ],
        }
        headers: dict[str, str] = {}
        api_key = os.environ.get(API_KEY_ENV) or os.environ.get(API_KEY_ENV_FALLBACK)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = self._transport(self._url, payload, headers, self._timeout)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise MemoryStoreError(f"malformed response from scoring endpoint: {exc}") from exc
        return str(content).strip()


def load_probes(path: Path) -> list[dict[str, str]]:
    """Load and validate a ``[{q, expect}]`` probe file.

    Args:
        path: Path to the probes JSON file.

    Returns:
        The validated probe list.

    Raises:
        MemoryStoreError: The file cannot be read/parsed, or is not a non-empty
            list of ``{"q": str, "expect": str}`` objects.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryStoreError(f"could not read probes file {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise MemoryStoreError(
            f"{path} must contain a non-empty JSON list of {{'q', 'expect'}} objects"
        )
    probes: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "q" not in item or "expect" not in item:
            raise MemoryStoreError(f"probe #{i} in {path} is malformed: expected 'q' and 'expect'")
        probes.append({"q": str(item["q"]), "expect": str(item["expect"])})
    return probes


def score_probes(
    scorer: AnswerScorer,
    probes: list[dict[str, str]],
    *,
    dump: str,
    curate_fn: Callable[[str], str],
) -> dict[str, str]:
    """Score every probe under three conditions: no memory / full dump / curate.

    Args:
        scorer: The configured :class:`AnswerScorer`.
        probes: Validated probes (:func:`load_probes`).
        dump: The full (redacted) dump text.
        curate_fn: Called with the probe question; returns curated (redacted) text.

    Returns:
        ``{"none": "n/total", "dump": "n/total", "curate": "n/total"}``.

    Raises:
        Exception: Whatever the scorer's transport raises propagates unchanged so
            the caller can degrade loudly (never silently) when the endpoint is
            unreachable or misconfigured.
    """
    tally = {"none": 0, "dump": 0, "curate": 0}
    total = len(probes)
    for probe in probes:
        question, needle = probe["q"], probe["expect"].lower()
        contexts = {"none": "", "dump": dump, "curate": curate_fn(question)}
        for condition, ctx in contexts.items():
            answer = scorer.ask(ctx, question).lower()
            if needle in answer:
                tally[condition] += 1
    return {key: f"{value}/{total}" for key, value in tally.items()}
