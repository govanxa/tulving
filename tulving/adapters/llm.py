"""LLM adapter protocol, the v0.1 Anthropic adapter, and call budgeting.

The LLM adapter is always optional (``Memory(llm_adapter=None)`` is the
default); every LLM-dependent behavior degrades to a deterministic fallback
with a logged warning. Credentials come from the environment only
(``ANTHROPIC_API_KEY``); inline keys raise ``SecurityError``.

Optional dependencies are gated inside constructors (D9): importing this
module on a core-only install always succeeds. Provider/runtime failures
propagate as-is — D6's hierarchy has no fitting bucket for a transient
provider error, and the summarizer's contract is "any exception = failed
call, budget already spent, stop the pass cleanly." This module raises only
``ConfigError`` and ``SecurityError`` (D6). The key, the prompt, and the
response text never appear in log records or exception messages.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Protocol, runtime_checkable

from tulving.exceptions import ConfigError
from tulving.security import credential_from_env, reject_inline_credential

logger = logging.getLogger("tulving.adapters.llm")

DEFAULT_MODEL: Final[str] = "claude-opus-4-8"
ANTHROPIC_ENV_VAR: Final[str] = "ANTHROPIC_API_KEY"
_DEFAULT_MAX_INPUT_TOKENS: Final[int] = 180_000


@runtime_checkable
class LLMAdapter(Protocol):
    """Structural interface every LLM backend satisfies (spec §2)."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt, return the text response. Sync is THE contract."""

    async def complete_async(self, prompt: str, **kwargs: Any) -> str:
        """OPTIONAL async variant.

        Implementations may raise ``NotImplementedError``; callers must not
        assume it exists.
        """

    @property
    def max_input_tokens(self) -> int:
        """Largest prompt this adapter accepts per call.

        Callers chunk to fit (minus ``LifecycleConfig.token_safety_margin``) —
        the adapter does NOT count tokens or truncate.
        """


class CallBudget:
    """Per-operation LLM circuit breaker (spec §2.1, mirroring Kairos).

    The counter increments BEFORE the underlying call is dispatched, so a
    call that raises still consumes budget — a retry loop cannot spin free.

    Two exhaustion styles:
      * ``try_acquire() -> bool`` — False when exhausted; the caller STOPS
        CLEANLY, reports what was done, and leaves the remainder for a later
        pass (the summarizer's contract).
      * ``acquire() -> None`` — raises ``ConfigError`` when exhausted, for
        callers that must never proceed past the limit.

    Not thread-safe; create one budget per operation/summarization pass.
    """

    def __init__(self, limit: int) -> None:
        """Create a budget of ``limit`` calls.

        Args:
            limit: Maximum number of calls this budget permits.

        Raises:
            ConfigError: If ``limit`` is below 1.
        """
        if limit < 1:
            raise ConfigError("CallBudget limit must be at least 1")
        self._limit = limit
        self._spent = 0

    def try_acquire(self) -> bool:
        """Consume one call if any remain.

        Returns:
            True when a call was consumed; False when the budget is
            exhausted (the counter never passes the limit).
        """
        if self._spent >= self._limit:
            return False
        self._spent += 1
        return True

    def acquire(self) -> None:
        """Consume one call or raise.

        Raises:
            ConfigError: If the budget is exhausted. The message carries
                counts only — never prompts or responses.
        """
        if not self.try_acquire():
            raise ConfigError(f"LLM call budget exhausted ({self._spent}/{self._limit} calls)")

    @property
    def limit(self) -> int:
        """The configured maximum number of calls."""
        return self._limit

    @property
    def spent(self) -> int:
        """Calls consumed so far (failed calls included)."""
        return self._spent

    @property
    def remaining(self) -> int:
        """Calls still available: ``max(limit - spent, 0)``."""
        return max(self._limit - self._spent, 0)

    @property
    def exhausted(self) -> bool:
        """True once ``spent`` has reached ``limit``."""
        return self._spent >= self._limit


class AnthropicAdapter:
    """v0.1 concrete adapter (ADR-016): Claude via the official ``anthropic`` SDK.

    Requires ``pip install "tulving[anthropic]"`` and the ``ANTHROPIC_API_KEY``
    environment variable. The constructor is cheap: no network I/O (D8) — the
    SDK client constructor is offline.

    SDK exceptions (rate limits, auth failures, timeouts) propagate
    unwrapped; the budget is already spent when they do. No retry loops are
    hand-rolled beyond the SDK's built-in ``max_retries=2`` — each SDK-internal
    retry is one budgeted call from Tulving's perspective, by design.

    The ``max_calls``/``calls_made`` counters are not synchronized; do not
    share a capped adapter across threads (relevant because the MCP server
    offloads to a thread pool, ADR-015).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_input_tokens: int = _DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = 2_048,
        max_calls: int | None = None,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Validate config and build the SDK client; open no connection (D8).

        Parameters are keyword-only so a credential accidentally passed
        positionally can never land in the model slot (and thence into an
        error message). Model IDs are opaque strings passed through verbatim —
        never constructed, suffixed, or validated against a hardcoded list.

        Args:
            model: Claude model ID, passed to the API unmodified.
            max_input_tokens: Advertised per-call prompt capacity; chunking
                to fit is the caller's job.
            max_output_tokens: ``max_tokens`` sent on every request.
            max_calls: Optional adapter-LIFETIME hard cap (circuit breaker,
                defense in depth). ``None`` = unlimited at the adapter layer;
                per-pass budgeting belongs to the caller's ``CallBudget``.
            timeout_seconds: Per-request timeout handed to the SDK client.
            api_key: MUST be None. Exists only to be rejected loudly —
                credentials come from the environment (security req #3).
            client: Test seam / advanced users bring their own configured SDK
                client; skips the import and env lookup. Env-only key policy
                applies to keys passed THROUGH Tulving — a user-configured
                client is the SDK's responsibility.

        Raises:
            SecurityError: If ``api_key`` is passed inline (checked FIRST).
            ConfigError: If a size/cap argument is invalid, the ``anthropic``
                package is not installed, or ``ANTHROPIC_API_KEY`` is
                unset/blank. Messages never echo caller-supplied values.
        """
        reject_inline_credential(api_key, adapter_name="anthropic")
        if max_input_tokens < 1:
            raise ConfigError("AnthropicAdapter max_input_tokens must be at least 1")
        if max_output_tokens < 1:
            raise ConfigError("AnthropicAdapter max_output_tokens must be at least 1")
        if max_calls is not None and max_calls < 1:
            raise ConfigError("AnthropicAdapter max_calls must be at least 1 (or None)")
        self._model = model
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._budget = CallBudget(max_calls) if max_calls is not None else None
        self._calls_made = 0
        self._unknown_kwargs_logged = False
        if client is not None:
            self._client: Any = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigError(
                "the Anthropic adapter requires the 'anthropic' package; "
                'install it with: pip install "tulving[anthropic]"'
            ) from exc
        key = credential_from_env(ANTHROPIC_ENV_VAR, adapter_name="anthropic")
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout_seconds, max_retries=2)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """One prompt -> one Messages API call -> concatenated text blocks.

        The call counter (and lifetime budget, when configured) increments
        BEFORE dispatch, so a raising call still counts. No sampling
        parameters (``temperature``/``top_p``/``top_k``) and no ``thinking``
        config are ever sent — current Claude models reject sampling params,
        and summarization needs neither. Unknown kwargs are ignored with a
        one-time debug log (names only — never values), never an error.

        Args:
            prompt: The user prompt, sent as a single user message.
            system: Optional system prompt; sent only when given.
            **kwargs: Accepted for protocol compatibility; only ``system``
                is honored in v0.1.

        Returns:
            The concatenation of all text blocks in the response (non-text
            blocks are skipped); ``""`` when the response has none.

        Raises:
            ConfigError: If the adapter-lifetime ``max_calls`` cap is spent.
            Exception: SDK exceptions propagate unwrapped (documented above).
        """
        if kwargs and not self._unknown_kwargs_logged:
            self._unknown_kwargs_logged = True
            logger.debug(
                "AnthropicAdapter.complete ignoring unsupported kwargs: %s",
                ", ".join(sorted(kwargs)),
            )
        if self._budget is not None:
            self._budget.acquire()  # increments BEFORE the call; ConfigError when spent
        self._calls_made += 1
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            request["system"] = system
        response = self._client.messages.create(**request)
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    async def complete_async(self, prompt: str, **kwargs: Any) -> str:
        """Async is deferred in v0.1 (matching Kairos); consumes no budget.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("AnthropicAdapter.complete_async is not available in v0.1")

    @property
    def max_input_tokens(self) -> int:
        """Advertised per-call prompt capacity (constructor argument)."""
        return self._max_input_tokens

    @property
    def calls_made(self) -> int:
        """Observability: dispatch attempts so far (failed calls included)."""
        return self._calls_made


class _KairosWrappedAdapter:
    """Duck-typed wrapper produced by :func:`from_kairos_adapter` (D11)."""

    def __init__(self, adapter: Any, *, max_input_tokens: int) -> None:
        self._adapter = adapter
        self._max_input_tokens = max_input_tokens

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Delegate to ``adapter.call(prompt).text``; extra kwargs are ignored.

        Args:
            prompt: The prompt, passed positionally to the Kairos adapter.
            **kwargs: Accepted for protocol compatibility; not forwarded.

        Returns:
            The ``.text`` of the Kairos ``ModelResponse`` — never
            ``str(response)``.
        """
        text: str = self._adapter.call(prompt).text
        return text

    async def complete_async(self, prompt: str, **kwargs: Any) -> str:
        """Async is deferred in v0.1.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Kairos-wrapped adapters do not support complete_async in v0.1")

    @property
    def max_input_tokens(self) -> int:
        """Advertised per-call prompt capacity (wrap-time argument)."""
        return self._max_input_tokens


def from_kairos_adapter(
    adapter: Any, *, max_input_tokens: int = _DEFAULT_MAX_INPUT_TOKENS
) -> LLMAdapter:
    """Wrap a Kairos ``ClaudeAdapter`` INSTANCE (spec §3.1 / D11).

    Delegates ``complete(prompt)`` to ``adapter.call(prompt).text`` — Kairos
    returns a ``ModelResponse``, not a string. Duck-typed: never imports
    kairos (ADR-011: sibling, never a dependency).

    Args:
        adapter: A constructed Kairos adapter instance exposing ``.call()``.
        max_input_tokens: Advertised per-call prompt capacity of the wrapper.

    Returns:
        An :class:`LLMAdapter`-conforming wrapper whose ``complete_async``
        raises ``NotImplementedError``.

    Raises:
        ConfigError: If ``adapter`` has no callable ``call`` attribute (fail
            at wrap time, not use time). The message never echoes the object.
    """
    if not callable(getattr(adapter, "call", None)):
        raise ConfigError(
            "from_kairos_adapter requires an adapter instance with a callable 'call' method"
        )
    return _KairosWrappedAdapter(adapter, max_input_tokens=max_input_tokens)
