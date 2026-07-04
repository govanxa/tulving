"""Tests for tulving.mcp.server — written BEFORE implementation (build step 14).

The Memory instance is always mocked (``create_autospec(Memory, instance=True)``)
and tools are invoked via the FastMCP in-memory interface (``server.call_tool``):
no stdio, no subprocess, no network, no real adapters. Coroutines run under
``asyncio.run`` (anyio's ``to_thread`` works on the running asyncio loop).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import anyio
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from tulving import Memory, MemoryType
from tulving.context.curator import CuratedContext
from tulving.entry import MemoryEntry, SourceInfo, utcnow
from tulving.enums import MatchType
from tulving.exceptions import ConfigError, MemoryStoreError, StorageError, TulvingError
from tulving.mcp import server as srv
from tulving.memory import SearchResult

# --------------------------------------------------------------------------- helpers

TOOL_NAMES = {
    "memory_store",
    "memory_get",
    "memory_search",
    "memory_curate",
    "memory_forget",
    "memory_list_keys",
}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _text(result: Any) -> str:
    """Extract the text of a successful ``call_tool`` return (content, structured)."""
    content = result[0]
    return str(content[0].text)


def make_settings(**kw: Any) -> srv.ServerSettings:
    defaults: dict[str, Any] = {
        "memory_path": Path("./mem"),
        "embedding": "local",
        "llm": "none",
        "llm_model": None,
        "default_token_budget": 4000,
        "read_only": False,
        "llm_configured": False,
    }
    defaults.update(kw)
    return srv.ServerSettings(**defaults)


def make_memory() -> Any:
    return create_autospec(Memory, instance=True)


def make_entry(
    *,
    entry_id: str = "id-1",
    content: str = "hello world",
    mtype: MemoryType = MemoryType.DECISION,
    key: str | None = "decision:auth",
    tags: list[str] | None = None,
    importance: float = 0.9,
) -> MemoryEntry:
    entry = MemoryEntry(
        id=entry_id,
        content=content,
        type=mtype,
        source=SourceInfo(agent_id="default"),
        key=key,
        tags=list(tags) if tags is not None else ["auth"],
        base_importance=0.5,
        created_at=utcnow(),
    )
    entry.importance = importance
    return entry


def build(memory: Any = None, settings: srv.ServerSettings | None = None) -> tuple[Any, Any, Any]:
    memory = memory if memory is not None else make_memory()
    settings = settings if settings is not None else make_settings()
    return srv.build_server(memory, settings), memory, settings


# =========================================================== 1-2 registration & schemas


class TestRegistrationAndSchemas:
    def test_exactly_six_tools_registered(self) -> None:
        server, _memory, _settings = build()
        tools = _run(server.list_tools())
        names = {t.name for t in tools}
        assert names == TOOL_NAMES  # no session/graph/summarize tools (ADR-016)

    def test_store_schema_requires_content_and_type_enum(self) -> None:
        server, _memory, _settings = build()
        tools = {t.name: t for t in _run(server.list_tools())}
        schema = tools["memory_store"].inputSchema
        assert set(schema["required"]) == {"content", "type"}
        assert schema["properties"]["type"]["enum"] == ["fact", "decision", "observation", "plan"]
        assert "summary" not in schema["properties"]["type"]["enum"]

    def test_curate_and_forget_schemas(self) -> None:
        server, _memory, _settings = build()
        tools = {t.name: t for t in _run(server.list_tools())}
        curate = tools["memory_curate"].inputSchema
        budget = curate["properties"]["token_budget"]
        # token_budget is `int | None` (defaults server-side) -> integer among anyOf.
        budget_types = {opt.get("type") for opt in budget.get("anyOf", [budget])}
        assert "integer" in budget_types
        assert curate["properties"]["mode"]["enum"] == ["query", "orient"]
        assert tools["memory_forget"].inputSchema.get("required", []) == []


# ===================================================================== 3-9 handler behavior


class TestMemoryStore:
    def test_store_maps_enum_defaults_tags_and_reports_id_key(self) -> None:
        server, memory, _settings = build()
        memory.store.return_value = make_entry(entry_id="abc123", key="decision:auth")
        text = _text(_run(server.call_tool("memory_store", {"content": "c", "type": "decision"})))
        kwargs = memory.store.call_args.kwargs
        assert kwargs["type"] is MemoryType.DECISION
        assert kwargs["tags"] == []
        assert "abc123" in text
        assert "decision:auth" in text

    def test_store_invalid_type_rejected_at_schema(self) -> None:
        server, memory, _settings = build()
        with pytest.raises(ToolError):
            _run(server.call_tool("memory_store", {"content": "c", "type": "summary"}))
        memory.store.assert_not_called()

    def test_importance_schema_carries_bounds(self) -> None:
        server, _memory, _settings = build()
        tools = {t.name: t for t in _run(server.list_tools())}
        importance = tools["memory_store"].inputSchema["properties"]["importance"]
        assert importance["minimum"] == 0.0
        assert importance["maximum"] == 1.0

    def test_out_of_range_importance_rejected_not_internal_error(self) -> None:
        server, memory, _settings = build()
        with pytest.raises(ToolError) as ei:
            _run(
                server.call_tool(
                    "memory_store", {"content": "c", "type": "fact", "importance": 5.0}
                )
            )
        assert "internal error" not in str(ei.value)  # schema-rejected, not a server fault
        memory.store.assert_not_called()


class TestMemoryGet:
    def test_get_hit_formats_entry(self) -> None:
        server, memory, _settings = build()
        memory.get.return_value = make_entry(content="the answer")
        text = _text(_run(server.call_tool("memory_get", {"key": "decision:auth"})))
        assert "the answer" in text
        assert "decision:auth" in text

    def test_get_miss_returns_message_no_exception(self) -> None:
        server, memory, _settings = build()
        memory.get.return_value = None
        text = _text(_run(server.call_tool("memory_get", {"key": "nope"})))
        assert "No memory found" in text
        assert "nope" in text


class TestMemorySearch:
    def test_search_formats_and_converts_types(self) -> None:
        server, memory, _settings = build()
        memory.search.return_value = [
            SearchResult(
                entry=make_entry(content="jwt chosen"), score=0.87, match_type=MatchType.SEMANTIC
            )
        ]
        text = _text(_run(server.call_tool("memory_search", {"query": "auth", "types": ["fact"]})))
        assert memory.search.call_args.kwargs["types"] == [MemoryType.FACT]
        assert "0.87" in text
        assert "semantic" in text
        assert "decision" in text
        assert "jwt chosen" in text

    def test_search_empty_returns_message(self) -> None:
        server, memory, _settings = build()
        memory.search.return_value = []
        text = _text(_run(server.call_tool("memory_search", {"query": "auth"})))
        assert "No matching memories." in text

    def test_search_unknown_type_is_actionable_error(self) -> None:
        server, memory, _settings = build()
        with pytest.raises(ToolError) as ei:
            _run(server.call_tool("memory_search", {"query": "q", "types": ["typo"]}))
        msg = str(ei.value)
        assert "unknown memory type 'typo'" in msg
        assert "fact, decision, observation, plan, summary" in msg
        assert "internal error" not in msg
        memory.search.assert_not_called()

    def test_search_accepts_summary_type_filter(self) -> None:
        server, memory, _settings = build()
        memory.search.return_value = []
        _run(server.call_tool("memory_search", {"query": "q", "types": ["summary"]}))
        assert memory.search.call_args.kwargs["types"] == [MemoryType.SUMMARY]


class TestMemoryCurate:
    def _ctx(self, content: str = "briefing body") -> CuratedContext:
        return CuratedContext(
            content=content,
            entries=[],
            token_count=120,
            budget_remaining=1114,
            sources_consulted=7,
        )

    def test_curate_default_budget_from_settings_and_footer(self) -> None:
        settings = make_settings(default_token_budget=1234)
        server, memory, _settings = build(settings=settings)
        memory.curate.return_value = self._ctx()
        text = _text(_run(server.call_tool("memory_curate", {"query": "auth"})))
        assert memory.curate.call_args.kwargs["token_budget"] == 1234
        assert "tokens: 120" in text
        assert "budget remaining: 1114" in text
        assert "sources consulted: 7" in text

    def test_orient_without_llm_is_loud(self) -> None:
        server, memory, _settings = build(settings=make_settings(llm_configured=False))
        memory.curate.return_value = self._ctx()
        text = _text(_run(server.call_tool("memory_curate", {"query": "x", "mode": "orient"})))
        assert srv.NO_LLM_NOTE in text

    def test_query_mode_never_appends_note(self) -> None:
        server, memory, _settings = build(settings=make_settings(llm_configured=False))
        memory.curate.return_value = self._ctx()
        text = _text(_run(server.call_tool("memory_curate", {"query": "x", "mode": "query"})))
        assert srv.NO_LLM_NOTE not in text

    def test_llm_configured_never_appends_note(self) -> None:
        server, memory, _settings = build(settings=make_settings(llm="claude", llm_configured=True))
        memory.curate.return_value = self._ctx()
        for mode in ("query", "orient"):
            memory.curate.return_value = self._ctx()
            text = _text(_run(server.call_tool("memory_curate", {"query": "x", "mode": mode})))
            assert srv.NO_LLM_NOTE not in text


class TestMemoryForget:
    def test_key_only_routes_to_forget(self) -> None:
        server, memory, _settings = build()
        memory.forget.return_value = True
        text = _text(
            _run(server.call_tool("memory_forget", {"key": "decision:auth", "hard": True}))
        )
        memory.forget.assert_called_once()
        assert memory.forget.call_args.kwargs["hard"] is True
        assert "decision:auth" in text

    def test_id_only_routes_to_forget_by_id(self) -> None:
        server, memory, _settings = build()
        memory.forget_by_id.return_value = True
        _run(server.call_tool("memory_forget", {"id": "id-9"}))
        memory.forget_by_id.assert_called_once()
        assert memory.forget_by_id.call_args.kwargs["hard"] is False

    def test_tags_only_routes_to_forget_by_tags(self) -> None:
        server, memory, _settings = build()
        memory.forget_by_tags.return_value = 3
        text = _text(_run(server.call_tool("memory_forget", {"tags": ["auth"]})))
        memory.forget_by_tags.assert_called_once()
        assert "3" in text

    def test_key_miss_returns_message_no_exception(self) -> None:
        server, memory, _settings = build()
        memory.forget.return_value = False
        text = _text(_run(server.call_tool("memory_forget", {"key": "gone"})))
        assert "No memory found" in text
        assert "gone" in text

    def test_id_miss_returns_message_no_exception(self) -> None:
        server, memory, _settings = build()
        memory.forget_by_id.return_value = False
        text = _text(_run(server.call_tool("memory_forget", {"id": "id-gone"})))
        assert "No memory found" in text
        assert "id-gone" in text

    def test_zero_selectors_is_tool_error_no_memory_call(self) -> None:
        server, memory, _settings = build()
        with pytest.raises(ToolError) as ei:
            _run(server.call_tool("memory_forget", {}))
        assert "exactly one of: key, id, tags" in str(ei.value)
        memory.forget.assert_not_called()
        memory.forget_by_id.assert_not_called()
        memory.forget_by_tags.assert_not_called()

    def test_two_selectors_is_tool_error_no_memory_call(self) -> None:
        server, memory, _settings = build()
        with pytest.raises(ToolError) as ei:
            _run(server.call_tool("memory_forget", {"key": "k", "id": "i"}))
        assert "exactly one of: key, id, tags" in str(ei.value)
        memory.forget.assert_not_called()
        memory.forget_by_id.assert_not_called()

    def test_empty_tags_list_is_not_a_selector(self) -> None:
        server, memory, _settings = build()
        with pytest.raises(ToolError) as ei:
            _run(server.call_tool("memory_forget", {"tags": []}))
        assert "exactly one of: key, id, tags" in str(ei.value)
        memory.forget_by_tags.assert_not_called()
        memory.forget.assert_not_called()

    def test_empty_string_key_is_not_a_selector(self) -> None:
        server, memory, _settings = build()
        with pytest.raises(ToolError) as ei:
            _run(server.call_tool("memory_forget", {"key": ""}))
        assert "exactly one of: key, id, tags" in str(ei.value)
        memory.forget.assert_not_called()


class TestMemoryListKeys:
    def test_prefix_passthrough(self) -> None:
        server, memory, _settings = build()
        memory.list_keys.return_value = ["decision:auth", "decision:db"]
        text = _text(_run(server.call_tool("memory_list_keys", {"prefix": "decision:"})))
        assert memory.list_keys.call_args.kwargs["prefix"] == "decision:"
        assert "decision:auth" in text

    def test_empty_returns_message(self) -> None:
        server, memory, _settings = build()
        memory.list_keys.return_value = []
        text = _text(_run(server.call_tool("memory_list_keys", {})))
        assert "No keys stored." in text


# ===================================================================== 10-11 redaction


SECRET_KEY = "sk-abcdefghijklmnopqrstuvwx"  # sk- + 24 chars
SECRET_PW = "password = hunter2secret"


class TestRedaction:
    def test_get_response_is_redacted(self) -> None:
        server, memory, _settings = build()
        memory.get.return_value = make_entry(content=f"{SECRET_KEY} and {SECRET_PW}")
        text = _text(_run(server.call_tool("memory_get", {"key": "k"})))
        assert "[REDACTED]" in text
        assert "sk-abcdefghijklmnopqrstuvwx" not in text
        assert "hunter2secret" not in text

    def test_search_response_is_redacted(self) -> None:
        server, memory, _settings = build()
        memory.search.return_value = [
            SearchResult(
                entry=make_entry(content=SECRET_KEY), score=0.5, match_type=MatchType.SEMANTIC
            )
        ]
        text = _text(_run(server.call_tool("memory_search", {"query": "q"})))
        assert "[REDACTED]" in text
        assert "sk-abcdefghijklmnopqrstuvwx" not in text

    def test_curate_response_is_redacted(self) -> None:
        server, memory, _settings = build()
        memory.curate.return_value = CuratedContext(
            content=f"{SECRET_KEY}",
            entries=[],
            token_count=1,
            budget_remaining=1,
            sources_consulted=1,
        )
        text = _text(_run(server.call_tool("memory_curate", {"query": "q"})))
        assert "[REDACTED]" in text
        assert "sk-abcdefghijklmnopqrstuvwx" not in text

    def test_tulving_error_is_redacted(self) -> None:
        server, memory, _settings = build()
        memory.forget.side_effect = MemoryStoreError("boom token=abcd1234efgh")
        with pytest.raises(ToolError) as ei:
            _run(server.call_tool("memory_forget", {"key": "k"}))
        msg = str(ei.value)
        assert "[REDACTED]" in msg
        assert "abcd1234efgh" not in msg

    def test_unexpected_error_is_generic_no_leak(self) -> None:
        server, memory, _settings = build()
        memory.get.side_effect = RuntimeError("secret internal path C:/x token=zzz")
        with pytest.raises(ToolError) as ei:
            _run(server.call_tool("memory_get", {"key": "k"}))
        msg = str(ei.value)
        assert "internal error" in msg
        assert "secret internal path" not in msg
        assert "zzz" not in msg


# Shape-bearing secrets stored AS the key itself must still redact (the bare-key
# format runs the key through the shape scan). Locks that interaction against
# future format changes.
KEY_SECRETS = ["sk-abcdefghijklmnopqrstuvwx", "AKIAIOSFODNN7EXAMPLE"]


class TestSecretStoredAsKeyIsRedacted:
    @pytest.mark.parametrize("secret", KEY_SECRETS)
    def test_get_redacts_secret_key(self, secret: str) -> None:
        server, memory, _settings = build()
        memory.get.return_value = make_entry(key=secret, content="benign body")
        text = _text(_run(server.call_tool("memory_get", {"key": secret})))
        assert "[REDACTED]" in text
        assert secret not in text

    @pytest.mark.parametrize("secret", KEY_SECRETS)
    def test_search_redacts_secret_key(self, secret: str) -> None:
        server, memory, _settings = build()
        memory.search.return_value = [
            SearchResult(
                entry=make_entry(key=secret, content="benign"),
                score=0.5,
                match_type=MatchType.SEMANTIC,
            )
        ]
        text = _text(_run(server.call_tool("memory_search", {"query": "q"})))
        assert "[REDACTED]" in text
        assert secret not in text

    @pytest.mark.parametrize("secret", KEY_SECRETS)
    def test_store_confirmation_redacts_secret_key(self, secret: str) -> None:
        server, memory, _settings = build()
        memory.store.return_value = make_entry(entry_id="id1", key=secret)
        text = _text(_run(server.call_tool("memory_store", {"content": "c", "type": "fact"})))
        assert "[REDACTED]" in text
        assert secret not in text

    @pytest.mark.parametrize("secret", KEY_SECRETS)
    def test_list_keys_redacts_secret_key(self, secret: str) -> None:
        server, memory, _settings = build()
        memory.list_keys.return_value = [secret, "decision:ok"]
        text = _text(_run(server.call_tool("memory_list_keys", {})))
        assert "[REDACTED]" in text
        assert secret not in text


# ============================================================= 12-13 concurrency & static guard


class TestConcurrencyAndStaticGuard:
    def test_every_handler_offloads_to_thread(self) -> None:
        server, memory, _settings = build()
        memory.store.return_value = make_entry()
        memory.get.return_value = make_entry()
        memory.search.return_value = []
        memory.curate.return_value = CuratedContext("b", [], 1, 1, 1)
        memory.forget.return_value = True
        memory.list_keys.return_value = []

        calls: list[Any] = []
        real = anyio.to_thread.run_sync

        async def spy(func: Any, *a: Any, **k: Any) -> Any:
            calls.append(func)
            return await real(func, *a, **k)

        invocations = [
            ("memory_store", {"content": "c", "type": "fact"}),
            ("memory_get", {"key": "k"}),
            ("memory_search", {"query": "q"}),
            ("memory_curate", {"query": "q"}),
            ("memory_forget", {"key": "k"}),
            ("memory_list_keys", {}),
        ]
        with patch("anyio.to_thread.run_sync", spy):
            for name, args in invocations:
                _run(server.call_tool(name, args))
        assert len(calls) == len(invocations)

    def test_memory_runs_off_the_event_loop_thread(self) -> None:
        server, memory, _settings = build()
        seen: dict[str, int] = {}

        def record(*_a: Any, **_k: Any) -> MemoryEntry:
            seen["worker"] = threading.get_ident()
            return make_entry()

        memory.get.side_effect = record
        loop_thread = threading.get_ident()
        _run(server.call_tool("memory_get", {"key": "k"}))
        assert seen["worker"] != loop_thread

    def test_no_network_transport_tokens_in_source(self) -> None:
        # Word-boundaried so incidental substrings (e.g. "dataclasses" -> "sse")
        # are not false positives; the guard targets real network transports.
        source = Path(srv.__file__).read_text(encoding="utf-8").lower()
        for pattern in (r"\bsse\b", r"\bwebsocket\b", r"host=", r"port=", r"\bbind\w*"):
            assert re.search(pattern, source) is None, f"forbidden network token {pattern!r}"


# ===================================================================== 14-20 main() lifecycle


class TestMainLifecycle:
    def test_lock_refused_returns_exit_locked(self, capsys: Any) -> None:
        with patch.object(srv, "create_memory", side_effect=StorageError("locked")):
            code = srv.main(["--memory-path", "./locked_store"])
        assert code == srv.EXIT_LOCKED
        err = capsys.readouterr().err
        assert "locked_store" in err
        assert "--read-only" in err

    def test_config_error_from_create_memory_returns_exit_config(self, capsys: Any) -> None:
        with patch.object(srv, "create_memory", side_effect=ConfigError("bad adapter")):
            code = srv.main(["--memory-path", "./m"])
        assert code == srv.EXIT_CONFIG
        assert "bad adapter" in capsys.readouterr().err

    def test_other_tulving_error_from_create_memory_returns_exit_config(self, capsys: Any) -> None:
        err = TulvingError("read-only refuses write")
        with patch.object(srv, "create_memory", side_effect=err):
            code = srv.main(["--memory-path", "./m"])
        assert code == srv.EXIT_CONFIG
        assert "read-only refuses write" in capsys.readouterr().err

    def test_missing_mcp_extra_returns_exit_config(self, capsys: Any) -> None:
        msg = 'the MCP server requires the [mcp] extra; install it with: pip install "tulving[mcp]"'
        with patch.object(srv, "_require_mcp", side_effect=ConfigError(msg)):
            code = srv.main(["--memory-path", "./m"])
        assert code == srv.EXIT_CONFIG
        assert 'pip install "tulving[mcp]"' in capsys.readouterr().err

    def test_read_only_constructs_memory_read_only(self) -> None:
        with (
            patch.object(srv, "Memory") as mem_cls,
            patch.object(srv, "LocalEmbedder"),
            patch.object(srv, "build_server") as build_mock,
        ):
            srv.main(["--memory-path", "./m", "--read-only"])
        assert mem_cls.call_args.kwargs["read_only"] is True
        build_mock.assert_called_once()

    def test_llm_none_warns(self, caplog: Any) -> None:
        with (
            patch.object(srv, "Memory"),
            patch.object(srv, "LocalEmbedder"),
            patch.object(srv, "AnthropicAdapter") as anthropic,
            patch.object(srv, "build_server"),
        ):
            with caplog.at_level(logging.WARNING, logger="tulving.mcp"):
                srv.main(["--memory-path", "./m", "--llm", "none"])
        assert any("no LLM adapter configured" in r.message for r in caplog.records)
        anthropic.assert_not_called()

    def test_llm_claude_constructs_adapter_no_warning(self, caplog: Any) -> None:
        with (
            patch.object(srv, "Memory"),
            patch.object(srv, "LocalEmbedder"),
            patch.object(srv, "AnthropicAdapter") as anthropic,
            patch.object(srv, "build_server"),
        ):
            with caplog.at_level(logging.WARNING, logger="tulving.mcp"):
                srv.main(["--memory-path", "./m", "--llm", "claude"])
        anthropic.assert_called_once()
        assert not any("no LLM adapter configured" in r.message for r in caplog.records)

    def test_flag_beats_env_for_memory_path(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("TULVING_MEMORY_PATH", "envpath")
        parser = srv._build_parser()
        args = parser.parse_args(["--memory-path", "flagpath"])
        settings = srv._resolve_settings(args)
        assert "flagpath" in str(settings.memory_path)
        assert "envpath" not in str(settings.memory_path)

    def test_env_used_when_flag_absent(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("TULVING_MEMORY_PATH", "envpath")
        parser = srv._build_parser()
        args = parser.parse_args([])
        settings = srv._resolve_settings(args)
        assert "envpath" in str(settings.memory_path)

    def test_bad_token_budget_returns_exit_config(self, capsys: Any) -> None:
        with patch.object(srv, "_require_mcp"):
            with patch.dict("os.environ", {"TULVING_DEFAULT_TOKEN_BUDGET": "not-an-int"}):
                code = srv.main(["--memory-path", "./m"])
        assert code == srv.EXIT_CONFIG

    def test_default_token_budget_env_plumbed(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("TULVING_DEFAULT_TOKEN_BUDGET", "9999")
        parser = srv._build_parser()
        settings = srv._resolve_settings(parser.parse_args([]))
        assert settings.default_token_budget == 9999

    def test_shutdown_always_closes_and_returns_ok(self) -> None:
        memory = make_memory()
        run_server = MagicMock()
        run_server.run.side_effect = KeyboardInterrupt
        memory.close.side_effect = RuntimeError("close boom")
        with (
            patch.object(srv, "create_memory", return_value=memory),
            patch.object(srv, "build_server", return_value=run_server),
        ):
            code = srv.main(["--memory-path", "./m"])
        assert code == srv.EXIT_OK
        memory.close.assert_called_once()

    def test_startup_called_once_and_nonfatal(self) -> None:
        memory = make_memory()
        memory.startup.side_effect = TulvingError("degraded")
        run_server = MagicMock()
        with (
            patch.object(srv, "create_memory", return_value=memory),
            patch.object(srv, "build_server", return_value=run_server),
        ):
            code = srv.main(["--memory-path", "./m"])
        assert code == srv.EXIT_OK
        memory.startup.assert_called_once()
        run_server.run.assert_called_once()
