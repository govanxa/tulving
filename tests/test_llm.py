"""Tests for tulving.adapters.llm — written BEFORE implementation.

Constraints (blueprint-llm-adapter.md): no real network, ever. The
``anthropic`` package is NOT installed in the dev environment — every
``AnthropicAdapter`` is constructed with an injected fake client, or against
a fake ``anthropic`` module in ``sys.modules``. An autouse fixture deletes
``ANTHROPIC_API_KEY`` so an accidental real construction fails loudly.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, ClassVar

import pytest

from tulving.adapters.llm import (
    ANTHROPIC_ENV_VAR,
    DEFAULT_MODEL,
    AnthropicAdapter,
    CallBudget,
    LLMAdapter,
    from_kairos_adapter,
)
from tulving.exceptions import ConfigError, SecurityError

INLINE_KEY = "sk-live-inline123456789012345"

# ---------------------------------------------------------------------------
# Fakes — never the real anthropic SDK
# ---------------------------------------------------------------------------


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeThinkingBlock:
    def __init__(self) -> None:
        self.type = "thinking"
        self.thinking = "internal reasoning"


class FakeTypelessBlock:
    """A block with no ``type`` attribute at all — must be skipped, not crash."""


class FakeResponse:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class FakeMessages:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.raise_on_create: Exception | None = None
        self.response = FakeResponse([FakeTextBlock("ok")])

    def create(self, **kwargs: Any) -> FakeResponse:
        self.create_calls.append(dict(kwargs))
        if self.raise_on_create is not None:
            raise self.raise_on_create
        return self.response


class FakeAnthropicClient:
    """Mimics anthropic.Anthropic; records constructor args and requests."""

    instances: ClassVar[list[FakeAnthropicClient]] = []

    def __init__(self, api_key: str, timeout: float = 600.0, max_retries: int = 2) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.messages = FakeMessages()
        FakeAnthropicClient.instances.append(self)


class FakeAnthropicModule:
    Anthropic = FakeAnthropicClient


@pytest.fixture(autouse=True)
def _no_real_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any accidental env-path construction fails loudly instead of dialing out."""
    monkeypatch.delenv(ANTHROPIC_ENV_VAR, raising=False)


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[FakeAnthropicClient]:
    """Install a fake anthropic module + env key; reset recorded instances."""
    FakeAnthropicClient.instances = []
    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicModule())
    monkeypatch.setenv(ANTHROPIC_ENV_VAR, "env-key-from-monkeypatch")
    return FakeAnthropicClient


@pytest.fixture
def fake_client() -> FakeAnthropicClient:
    """A bare injected client (test seam) — skips env + import entirely."""
    return FakeAnthropicClient(api_key="seam-key-unused")


def _block_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Force ``import name`` to raise ImportError even if installed."""
    monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_missing_env_key_raises_config_error_naming_the_variable(
        self, fake_anthropic: type[FakeAnthropicClient], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ANTHROPIC_ENV_VAR, raising=False)
        with pytest.raises(ConfigError) as excinfo:
            AnthropicAdapter()
        assert ANTHROPIC_ENV_VAR in str(excinfo.value)
        assert "anthropic" in str(excinfo.value)
        assert fake_anthropic.instances == []  # no client constructed

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_env_key_raises_config_error(
        self,
        fake_anthropic: type[FakeAnthropicClient],
        monkeypatch: pytest.MonkeyPatch,
        blank: str,
    ) -> None:
        monkeypatch.setenv(ANTHROPIC_ENV_VAR, blank)
        with pytest.raises(ConfigError) as excinfo:
            AnthropicAdapter()
        assert ANTHROPIC_ENV_VAR in str(excinfo.value)
        assert fake_anthropic.instances == []

    def test_missing_extra_raises_config_error_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _block_module(monkeypatch, "anthropic")
        monkeypatch.setenv(ANTHROPIC_ENV_VAR, "env-key-set")
        with pytest.raises(ConfigError) as excinfo:
            AnthropicAdapter()
        assert "tulving[anthropic]" in str(excinfo.value)

    @pytest.mark.parametrize("max_calls", [0, -1])
    def test_invalid_max_calls_raises_config_error(
        self, fake_client: FakeAnthropicClient, max_calls: int
    ) -> None:
        with pytest.raises(ConfigError):
            AnthropicAdapter(client=fake_client, max_calls=max_calls)

    @pytest.mark.parametrize("value", [0, -5])
    def test_invalid_max_input_tokens_raises_config_error(
        self, fake_client: FakeAnthropicClient, value: int
    ) -> None:
        with pytest.raises(ConfigError):
            AnthropicAdapter(client=fake_client, max_input_tokens=value)

    @pytest.mark.parametrize("value", [0, -5])
    def test_invalid_max_output_tokens_raises_config_error(
        self, fake_client: FakeAnthropicClient, value: int
    ) -> None:
        with pytest.raises(ConfigError):
            AnthropicAdapter(client=fake_client, max_output_tokens=value)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_call_budget_limit_below_one_raises_config_error(self, limit: int) -> None:
        with pytest.raises(ConfigError):
            CallBudget(limit)

    def test_complete_async_raises_not_implemented_and_consumes_no_budget(
        self, fake_client: FakeAnthropicClient
    ) -> None:
        adapter = AnthropicAdapter(client=fake_client, max_calls=3)
        with pytest.raises(NotImplementedError):
            asyncio.run(adapter.complete_async("x"))
        assert adapter.calls_made == 0
        assert fake_client.messages.create_calls == []

    def test_sdk_exception_propagates_unwrapped(self, fake_client: FakeAnthropicClient) -> None:
        adapter = AnthropicAdapter(client=fake_client)
        boom = RuntimeError("rate limited")
        fake_client.messages.raise_on_create = boom
        with pytest.raises(RuntimeError) as excinfo:
            adapter.complete("prompt")
        assert excinfo.value is boom


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_inline_api_key_raises_security_error(self, fake_client: FakeAnthropicClient) -> None:
        with pytest.raises(SecurityError):
            AnthropicAdapter(api_key=INLINE_KEY, client=fake_client)

    def test_inline_key_rejection_message_never_echoes_key(self) -> None:
        with pytest.raises(SecurityError) as excinfo:
            AnthropicAdapter(api_key=INLINE_KEY)
        assert INLINE_KEY not in str(excinfo.value)

    def test_inline_key_rejected_before_import_and_env_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rejection is constructor statement 1: fires even with the anthropic
        package missing AND the env var set — SecurityError, never ConfigError."""
        _block_module(monkeypatch, "anthropic")
        monkeypatch.setenv(ANTHROPIC_ENV_VAR, "env-key-set")
        with pytest.raises(SecurityError):
            AnthropicAdapter(api_key=INLINE_KEY)

    def test_missing_env_error_never_echoes_values(
        self, fake_anthropic: type[FakeAnthropicClient], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ANTHROPIC_ENV_VAR, "   ")
        with pytest.raises(ConfigError) as excinfo:
            AnthropicAdapter(model="maybe-a-mispasted-credential")
        # neither the blank env value's whitespace framing nor the caller's
        # model string may be echoed
        assert "maybe-a-mispasted-credential" not in str(excinfo.value)

    def test_module_import_needs_no_anthropic_package(self) -> None:
        """No module-level import of anthropic — asserted on the SOURCE (ast),
        so the test cannot fail spuriously in environments where the real SDK
        happens to be installed and already imported in the session."""
        import ast
        import inspect

        import tulving.adapters.llm as llm_module

        assert "tulving.adapters.llm" in sys.modules  # core-only import succeeded
        tree = ast.parse(inspect.getsource(llm_module))
        for node in tree.body:  # top level only; the gated in-constructor import is fine
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                assert "anthropic" not in roots, "module-level import of anthropic"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root != "anthropic", "module-level import from anthropic"


# ---------------------------------------------------------------------------
# CallBudget — circuit-breaker semantics (audit-critical)
# ---------------------------------------------------------------------------


class TestCallBudget:
    def test_clean_stop_path(self) -> None:
        budget = CallBudget(3)
        assert [budget.try_acquire() for _ in range(3)] == [True, True, True]
        assert budget.try_acquire() is False
        assert budget.spent == 3
        assert budget.remaining == 0
        assert budget.exhausted is True

    def test_acquire_raises_config_error_when_exhausted(self) -> None:
        budget = CallBudget(1)
        budget.acquire()
        with pytest.raises(ConfigError):
            budget.acquire()

    def test_try_acquire_after_exhaustion_never_increments_past_limit(self) -> None:
        budget = CallBudget(2)
        for _ in range(5):
            budget.try_acquire()
        assert budget.spent == 2

    def test_fresh_budget_state(self) -> None:
        budget = CallBudget(4)
        assert budget.limit == 4
        assert budget.spent == 0
        assert budget.remaining == 4
        assert budget.exhausted is False

    def test_budget_isolation_between_instances(self) -> None:
        first = CallBudget(1)
        first.acquire()
        assert first.exhausted is True
        second = CallBudget(1)
        assert second.spent == 0
        assert second.try_acquire() is True

    def test_acquire_and_try_acquire_share_the_counter(self) -> None:
        budget = CallBudget(2)
        budget.acquire()
        assert budget.try_acquire() is True
        assert budget.exhausted is True
        assert budget.try_acquire() is False


class TestAdapterCircuitBreaker:
    def test_counter_increments_before_dispatch_on_failure(
        self, fake_client: FakeAnthropicClient
    ) -> None:
        """A call that raises still counts — a retry loop cannot spin free."""
        adapter = AnthropicAdapter(client=fake_client, max_calls=5)
        fake_client.messages.raise_on_create = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            adapter.complete("p")
        assert adapter.calls_made == 1
        with pytest.raises(RuntimeError):
            adapter.complete("p")
        assert adapter.calls_made == 2

    def test_lifetime_cap_raises_config_error_and_never_dispatches(
        self, fake_client: FakeAnthropicClient
    ) -> None:
        adapter = AnthropicAdapter(client=fake_client, max_calls=2)
        adapter.complete("one")
        adapter.complete("two")
        with pytest.raises(ConfigError):
            adapter.complete("three")
        assert len(fake_client.messages.create_calls) == 2  # third never dispatched

    def test_failed_calls_consume_the_lifetime_budget(
        self, fake_client: FakeAnthropicClient
    ) -> None:
        """Blueprint TDD item 6, second half: the lifetime budget (not just the
        ``calls_made`` counter) is spent BEFORE dispatch, so raising calls
        exhaust a capped adapter — a retry loop cannot spin free past the cap."""
        adapter = AnthropicAdapter(client=fake_client, max_calls=2)
        fake_client.messages.raise_on_create = RuntimeError("boom")
        for _ in range(2):
            with pytest.raises(RuntimeError):
                adapter.complete("p")
        with pytest.raises(ConfigError):  # budget spent by the failures alone
            adapter.complete("p")
        assert adapter.calls_made == 2  # the refused call never became a dispatch attempt
        assert len(fake_client.messages.create_calls) == 2

    def test_no_cap_by_default_but_calls_made_still_counts(
        self, fake_client: FakeAnthropicClient
    ) -> None:
        adapter = AnthropicAdapter(client=fake_client)
        for _ in range(12):
            adapter.complete("p")
        assert adapter.calls_made == 12
        assert len(fake_client.messages.create_calls) == 12


# ---------------------------------------------------------------------------
# Basic behavior — AnthropicAdapter (injected fake client)
# ---------------------------------------------------------------------------


class TestAnthropicAdapterBehavior:
    def test_complete_request_shape_defaults(self, fake_client: FakeAnthropicClient) -> None:
        adapter = AnthropicAdapter(client=fake_client)
        adapter.complete("hello")
        (call,) = fake_client.messages.create_calls
        assert call["model"] == DEFAULT_MODEL
        assert call["model"] == "claude-opus-4-8"
        assert call["max_tokens"] == 2048
        assert call["messages"] == [{"role": "user", "content": "hello"}]
        for forbidden in ("temperature", "top_p", "top_k", "thinking", "system"):
            assert forbidden not in call

    def test_system_passed_only_when_given(self, fake_client: FakeAnthropicClient) -> None:
        adapter = AnthropicAdapter(client=fake_client)
        adapter.complete("hello", system="You summarize.")
        (call,) = fake_client.messages.create_calls
        assert call["system"] == "You summarize."

    def test_response_extraction_concatenates_text_blocks_only(
        self, fake_client: FakeAnthropicClient
    ) -> None:
        fake_client.messages.response = FakeResponse(
            [FakeThinkingBlock(), FakeTextBlock("A"), FakeTypelessBlock(), FakeTextBlock("B")]
        )
        adapter = AnthropicAdapter(client=fake_client)
        assert adapter.complete("p") == "AB"

    def test_empty_content_returns_empty_string(self, fake_client: FakeAnthropicClient) -> None:
        fake_client.messages.response = FakeResponse([])
        adapter = AnthropicAdapter(client=fake_client)
        assert adapter.complete("p") == ""

    def test_model_override_passed_through_verbatim(self, fake_client: FakeAnthropicClient) -> None:
        adapter = AnthropicAdapter(model="claude-haiku-4-5", client=fake_client)
        adapter.complete("p")
        assert fake_client.messages.create_calls[0]["model"] == "claude-haiku-4-5"

    def test_max_output_tokens_override(self, fake_client: FakeAnthropicClient) -> None:
        adapter = AnthropicAdapter(client=fake_client, max_output_tokens=512)
        adapter.complete("p")
        assert fake_client.messages.create_calls[0]["max_tokens"] == 512

    def test_max_input_tokens_property(self, fake_client: FakeAnthropicClient) -> None:
        assert AnthropicAdapter(client=fake_client).max_input_tokens == 180_000
        assert AnthropicAdapter(client=fake_client, max_input_tokens=9).max_input_tokens == 9

    def test_runtime_protocol_conformance(self, fake_client: FakeAnthropicClient) -> None:
        assert isinstance(AnthropicAdapter(client=fake_client), LLMAdapter)

    def test_unknown_kwargs_ignored_not_forwarded(
        self, fake_client: FakeAnthropicClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = AnthropicAdapter(client=fake_client)
        with caplog.at_level(logging.DEBUG, logger="tulving.adapters.llm"):
            adapter.complete("p", temperature=0.7, thinking={"type": "adaptive"})
            adapter.complete("p", temperature=0.7)
        for call in fake_client.messages.create_calls:
            assert "temperature" not in call
            assert "thinking" not in call
        # one-time debug log, never an error; kwarg VALUES never logged
        debug_records = [r for r in caplog.records if "temperature" in r.getMessage()]
        assert len(debug_records) == 1
        assert all(record.levelno == logging.DEBUG for record in debug_records)
        assert "0.7" not in debug_records[0].getMessage()

    def test_client_constructed_from_env_with_timeout_and_retries(
        self, fake_anthropic: type[FakeAnthropicClient]
    ) -> None:
        adapter = AnthropicAdapter(timeout_seconds=30.0)
        (client,) = fake_anthropic.instances
        assert client.api_key == "env-key-from-monkeypatch"
        assert client.timeout == 30.0
        assert client.max_retries == 2
        assert adapter.complete("hello") == "ok"

    def test_constants(self) -> None:
        assert DEFAULT_MODEL == "claude-opus-4-8"
        assert ANTHROPIC_ENV_VAR == "ANTHROPIC_API_KEY"


# ---------------------------------------------------------------------------
# Kairos wrapper (D11) — duck-typed, zero kairos import
# ---------------------------------------------------------------------------


class FakeModelResponse:
    """Kairos adapter.call() returns a ModelResponse object, not a string."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:  # pragma: no cover - sentinel; must never be used
        return "WRONG-str-was-used"


class FakeKairosAdapter:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def call(self, prompt: str) -> FakeModelResponse:
        self.calls.append(prompt)
        return FakeModelResponse("summary")


class TestKairosWrapper:
    def test_complete_delegates_to_call_dot_text(self) -> None:
        fake = FakeKairosAdapter()
        wrapped = from_kairos_adapter(fake)
        assert wrapped.complete("p") == "summary"
        assert fake.calls == ["p"]  # positional prompt, not a kwargs dict

    def test_text_attribute_used_never_str(self) -> None:
        assert from_kairos_adapter(FakeKairosAdapter()).complete("p") != "WRONG-str-was-used"

    def test_object_without_callable_call_raises_at_wrap_time(self) -> None:
        with pytest.raises(ConfigError):
            from_kairos_adapter(object())

        class NotCallable:
            call = "a string, not a method"

        with pytest.raises(ConfigError):
            from_kairos_adapter(NotCallable())

    def test_complete_async_raises_not_implemented(self) -> None:
        wrapped = from_kairos_adapter(FakeKairosAdapter())
        with pytest.raises(NotImplementedError):
            asyncio.run(wrapped.complete_async("x"))

    def test_max_input_tokens_default_and_override(self) -> None:
        assert from_kairos_adapter(FakeKairosAdapter()).max_input_tokens == 180_000
        wrapped = from_kairos_adapter(FakeKairosAdapter(), max_input_tokens=42)
        assert wrapped.max_input_tokens == 42

    def test_runtime_protocol_conformance(self) -> None:
        assert isinstance(from_kairos_adapter(FakeKairosAdapter()), LLMAdapter)


# ---------------------------------------------------------------------------
# Package re-exports (blueprint file plan)
# ---------------------------------------------------------------------------


class TestAdapterPackageExports:
    def test_names_reexported_from_tulving_adapters(self) -> None:
        from tulving import adapters

        assert adapters.LLMAdapter is LLMAdapter
        assert adapters.AnthropicAdapter is AnthropicAdapter
        assert adapters.CallBudget is CallBudget
        assert adapters.from_kairos_adapter is from_kairos_adapter
