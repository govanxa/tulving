# Tulving

> *The context-budget engine for AI agents.*

**Status: v0.1.1 released on PyPI.** `pip install tulving` installs the current stable release.
v0.2 is in development — see the [roadmap](#roadmap--coming-in-v02).

Tulving is a model-agnostic Python SDK that gives an AI agent persistent, structured,
searchable working memory — and, as its **headline capability**, curates that memory back
into a fixed token budget:

```python
ctx = memory.curate("resuming work on the auth refactor", token_budget=4000)
print(ctx.content)          # a prompt-ready block, already redacted, within budget
print(ctx.token_count, ctx.budget_remaining, ctx.sources_consulted)
```

`curate()` selects, ranks, decays, and trims your agent's memories into a block that fits
the budget you hand it — so you spend context tokens on what matters instead of dumping a
whole store into the prompt. `curate(mode="orient")` produces a cold-start briefing for a
fresh session.

Named after **Endel Tulving**, the psychologist who established that memory has types
(episodic vs semantic). Typed memories — facts, decisions, observations, plans, summaries,
each with its own lifecycle — are Tulving's core data model.

Zero infrastructure: one SQLite file, no server, no API key required. Vectors are optional.

> 📖 **New here? Read the [User Guide](https://github.com/govanxa/tulving/blob/main/GUIDE.md)** — using Tulving with LM Studio, Claude Code,
> and the MCP Inspector; verifying it works; and measuring the value it adds over no memory.

---

## Install

Requires **Python 3.11+**.

```bash
pip install tulving                 # core engine (stdlib + hnswlib only)
```

Optional extras:

```bash
pip install "tulving[local]"        # local embeddings (sentence-transformers)
pip install "tulving[openai]"       # OpenAI embeddings
pip install "tulving[anthropic]"    # Claude LLM for session summaries
pip install "tulving[mcp]"          # MCP server for Claude Code
```

> The core engine works with **no** embedder (key-exact lookup + importance/recency
> curation). Add `[local]` or `[openai]` to enable semantic search.

---

## Quickstart

```python
from tulving import Memory, MemoryType

memory = Memory("./agent_memory", agent_id="my-agent")

# Store typed memories. A repeated key supersedes the old entry (never raises).
memory.store(
    "We chose SQLite over Postgres for zero-infra local deployment.",
    type=MemoryType.DECISION,
    key="decision:datastore",
    tags=["architecture"],
)
memory.store("The auth token TTL is 15 minutes.", type=MemoryType.FACT)

# Exact recall by key.
entry = memory.get("decision:datastore")
print(entry.content)

# Curate into a token budget — the headline primitive.
ctx = memory.curate("what did we decide about storage?", token_budget=2000)
print(ctx.content)
```

Enable **semantic search** by passing an embedder:

```python
from tulving import Memory, LocalEmbedder      # pip install "tulving[local]"

memory = Memory("./agent_memory", embedding_adapter=LocalEmbedder())
results = memory.search("database choice", top_k=5)
for r in results:
    print(f"{r.score:.2f}  {r.match_type}  {r.entry.content}")
```

### Typed memories

| Type | Meaning | Decay |
|---|---|---|
| `FACT` | a discrete piece of information | decays over time |
| `DECISION` | a choice made and its reasoning | **never decays**, exempt from eviction |
| `OBSERVATION` | something noticed or analyzed | decays over time |
| `PLAN` | an intended future action | decays over time |
| `SUMMARY` | a system-generated digest | created by summarization, not by callers |

Importance is computed **lazily on read** (`base_importance * 0.5^(age / half_life[type])`).
Nothing is ever destroyed — low-importance entries are *archived*, not deleted, and
`DECISION`/pinned entries survive eviction.

---

## MCP server (Claude Code & other MCP clients)

Tulving ships a thin, **local-only** (stdio, no network) MCP server exposing six tools:
`memory_store`, `memory_get`, `memory_search`, `memory_curate`, `memory_forget`,
`memory_list_keys` (orient = `memory_curate(mode="orient")`).

```bash
pip install "tulving[mcp,local]"
tulving-mcp --memory-path ./agent_memory --embedding local
```

Register it with Claude Code (`.mcp.json` or your MCP client config):

```json
{
  "mcpServers": {
    "tulving": {
      "command": "tulving-mcp",
      "args": ["--memory-path", "./agent_memory", "--embedding", "local"]
    }
  }
}
```

Flags fall back to env vars (`TULVING_MEMORY_PATH`, `TULVING_EMBEDDING_ADAPTER`,
`TULVING_LLM_ADAPTER`, `TULVING_LLM_MODEL`, `TULVING_DEFAULT_TOKEN_BUDGET`). `--llm claude`
enables session summaries (needs `[anthropic]` + `ANTHROPIC_API_KEY`); with no LLM the
server degrades **loudly** to deterministic markers. A second process on the same path is
refused (single-writer safety, ADR-015).

---

## Export / import

Full JSON round-trip for backup, migration, or sharing. Exports are an **emission surface**:
redaction is **on by default** (secret-shaped tokens and sensitive keys are masked), with a
loud keyword-only opt-out for trusted backups.

```python
# By default the export file must live inside the memory directory; pass
# allowed_root to write elsewhere (leaf names are whitelisted, dirs contained).
memory.export_json("backup.json", allowed_root=".")                  # redacted, sharing-safe
memory.export_json("backup.json", allowed_root=".",
                   include_sensitive=True, include_archived=True)    # full plaintext backup

report = memory.import_json("backup.json", on_key_conflict="skip")   # or "supersede"
print(report.entries_imported, report.entries_reembedded, report.warnings)
```

Imported entries are re-embedded (unless the embedding model matches exactly) and their
IDs/reference graph are consistently remapped.

---

## What v0.1 ships

- **Core memory engine** — store / get / semantic search over typed memories; upsert-by-key
  supersede; zero infrastructure (one SQLite file, local vectors optional).
- **Token-budget curation** — `curate(query, token_budget)` + `orient` mode.
- **Lazy decay & eviction** — per-type importance decay; decisions/pinned never evicted;
  archive, never destroy.
- **Sessions** — per-agent session tracking, end-of-session summarization, abandoned-session
  recovery on startup.
- **MCP server** — six tools, single-writer safety, loud `llm=None` degradation.
- **Adapters & export** — pluggable embeddings (`local`/`openai`) and LLM (`anthropic`);
  JSON round-trip export/import with default-on redaction.

---

## Roadmap — coming in v0.2

Planned for the next release (**not yet available**):

- **`tulving eval` — a value report.** Measure whether Tulving is actually helping *your* store over
  time: how much `curate()` cuts context-token usage as your memory grows, plus answer correctness —
  rendered as a self-contained **HTML / PDF trend report**.
- **Torch-free offline MCP mode (`--embedding none`).** Run the MCP server with **no torch and no
  network** — key-exact recall + importance/recency curation — for users who want token-budget
  reduction without installing the local-embeddings stack. (Semantic "search by meaning" stays
  available via `--embedding local` / `openai`.)
- **`tulving maintenance` — housekeeping CLI.** `--inspect` / `--purge` / `--vacuum` / `--export`
  to keep a long-lived store tidy on disk.

Further out (ADR-016): knowledge graph, multi-agent machinery, semantic contradiction detection,
Postgres backend, and additional export formats.

---

## Development

```bash
git clone https://github.com/govanxa/tulving.git
cd tulving
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,mcp]"

pytest                                    # full suite
pytest --cov=tulving --cov-report=term-missing
mypy tulving/                             # strict
ruff check tulving/ tests/ && ruff format --check tulving/ tests/
```

CI runs these same gates plus a cross-platform test matrix (Linux + Windows ×
Python 3.11–3.13) on every push — see `.github/workflows/ci.yml`.

The [User Guide](https://github.com/govanxa/tulving/blob/main/GUIDE.md) covers real-world usage (LM Studio, Claude Code, MCP Inspector),
behavior verification, and how to measure Tulving against having no memory at all.

---

By the same author: [Kairos](https://github.com/govanxa/kairos) (`kairos-ai`) —
contract-enforced AI workflows — and its
[`kairos-ai-evidence`](https://pypi.org/project/kairos-ai-evidence/) plugin for
contract-validated evidence evaluation. A `kairos-ai-tulving` integration plugin is planned.

## License

[Apache 2.0](https://github.com/govanxa/tulving/blob/main/LICENSE)
