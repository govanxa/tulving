# Tulving

> *The context-budget engine for AI agents.*

**Status: pre-release (0.0.1) — name-holding release; v0.1 is in active development.**

Tulving is a model-agnostic Python SDK that gives an AI agent persistent, structured, searchable working memory — and, as its headline capability, curates that memory back into a fixed token budget:

```python
ctx = memory.curate("resuming work on the auth refactor", token_budget=4000)
```

Named after **Endel Tulving**, the psychologist who established that memory has types (episodic vs semantic). Typed memories — facts, decisions, observations, plans, summaries, each with its own lifecycle — are Tulving's core data model.

## What v0.1 will ship

- **Core memory engine** — store / get / semantic search over typed memories; zero infrastructure (one SQLite file, local vectors optional, no API key required)
- **Token-budget context curation** — `curate(query, token_budget)` selects, ranks, and trims memories into a prompt-ready block; `orient` mode for cold starts
- **Lazy decay & eviction** — importance fades per-type over time; decisions never decay; nothing is destroyed, only archived
- **Sessions** — per-agent session tracking, end-of-session summarization, abandoned-session recovery
- **MCP server** — six tools for Claude Code and other MCP clients, with single-writer safety
- **Adapters & export** — pluggable embeddings/LLM, JSON round-trip export

By the same author: [Kairos](https://github.com/govanxa/kairos) (`kairos-ai`) — contract-enforced AI workflows. A `kairos-ai-tulving` integration plugin is planned.

## License

TBD (MIT or Apache 2.0) — will be finalized with the v0.1 release.
