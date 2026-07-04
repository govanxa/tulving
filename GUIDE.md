# Tulving — User Guide

Installing Tulving, using it with a real model over MCP, verifying it behaves correctly, and
measuring the value it adds over having no memory at all. For the API-level quickstart see the
[README](README.md); for design rationale see the architecture docs.

**Contents**

1. [Installation](#1-installation)
2. [How it works — the mental model](#2-how-it-works--the-mental-model)
3. [Quickstart (Python)](#3-quickstart-python)
4. [Using Tulving over MCP](#4-using-tulving-over-mcp)
   - [LM Studio + a local model](#4a-lm-studio--a-local-model)
   - [Claude Code](#4b-claude-code)
   - [MCP Inspector — tool reference](#4c-mcp-inspector--tool-reference)
   - [Global vs. per-project memory, and tagging](#4d-global-vs-per-project-memory-and-tagging)
5. [Verifying it behaves correctly](#5-verifying-it-behaves-correctly)
6. [Measuring the value vs. no memory](#6-measuring-the-value-vs-no-memory)
7. [Storage model & platform support](#7-storage-model--platform-support)
8. [Running the test suite](#8-running-the-test-suite)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Installation

Requires **Python 3.11, 3.12, or 3.13**.

```bash
pip install tulving                 # core engine (standard library + hnswlib)
```

Optional extras:

```bash
pip install "tulving[local]"        # fully-offline embeddings (sentence-transformers)
pip install "tulving[openai]"       # OpenAI embeddings
pip install "tulving[anthropic]"    # Claude LLM for session summaries
pip install "tulving[mcp]"          # the MCP server (Claude Code, LM Studio, etc.)
```

The core engine works with **no** embedder (exact-key lookup and importance/recency curation).
Add `[local]` or `[openai]` to enable semantic search. A `hnswlib` wheel is installed
automatically; if your platform lacks a prebuilt wheel, install a C/C++ toolchain first
(Windows: Visual C++ Build Tools; Linux: `build-essential`; macOS: Xcode Command Line Tools).

---

## 2. How it works — the mental model

An LLM has no memory between turns beyond what you place in its context window, and that window
is finite and expensive. Tulving is **external, persistent, typed memory** that an agent reads
and writes through six tools. The model is the brain; Tulving is its notebook.

```
                 stores facts/decisions as it works
   ┌────────────┐  ────────────────────────────────▶  ┌─────────────────────────┐
   │  the model │                                      │  Tulving (one SQLite    │
   │ (LM Studio │  ◀────────────────────────────────   │  file + vector index)   │
   │  / Claude) │   curate() packs the relevant slice  │  survives every restart │
   └────────────┘   back into the prompt, within a     └─────────────────────────┘
                    token budget
```

The real-life loop:

1. **While working**, the agent calls `memory_store` to record decisions, facts, and plans
   ("we chose JWT over sessions", "auth TTL is 15 min"). These persist to disk.
2. **On a new task or a fresh session** — when the model's context has reset — the agent calls
   `memory_curate` (or `orient`) to pull *just the relevant memories* back into the prompt,
   ranked and trimmed to a token budget. That is continuity the model could not otherwise have.
3. Old, unreferenced memories **decay** in importance and are eventually archived; decisions
   never decay. Nothing is deleted silently.

**Why `curate` instead of dumping everything into the prompt?** Because the store grows without
bound and the context window does not. `curate` ranks by relevance + importance + recency and
trims to fit. [Section 6](#6-measuring-the-value-vs-no-memory) measures this: at 500 memories,
`curate` delivers the relevant slice in roughly **50× fewer tokens** than dumping the store —
and it keeps working after a full dump would overflow the window.

**Where does the `--llm` flag fit?** It powers *Tulving's own* summarization (end-of-session
rollups, the `orient` digest) — **not** the agent. A fully-local stack is your local model in
LM Studio doing the reasoning plus `tulving-mcp --llm none` (summaries degrade to deterministic
markers). Add `--llm claude` (with `ANTHROPIC_API_KEY`) only if you want LLM-written summaries.

---

## 3. Quickstart (Python)

```python
from tulving import Memory, MemoryType

memory = Memory("./agent_memory", agent_id="my-agent")

# Store typed memories. A repeated key supersedes the old entry (never raises).
memory.store(
    "We chose SQLite over Postgres for zero-infra local deployment.",
    type=MemoryType.DECISION, key="decision:datastore", tags=["architecture"],
)
memory.store("The auth token TTL is 15 minutes.", type=MemoryType.FACT)

memory.get("decision:datastore")                       # exact recall by key

ctx = memory.curate("what did we decide about storage?", token_budget=2000)
print(ctx.content)                                     # prompt-ready, within budget
```

Enable semantic search with an embedder:

```python
from tulving import Memory, LocalEmbedder            # pip install "tulving[local]"

memory = Memory("./agent_memory", embedding_adapter=LocalEmbedder())
for r in memory.search("database choice", top_k=5):
    print(f"{r.score:.2f}  {r.match_type}  {r.entry.content}")
```

**Typed memories:** `FACT`, `DECISION` (never decays), `OBSERVATION`, `PLAN`, and `SUMMARY`
(system-generated). Importance is computed lazily on read and decays per type; archived entries
are never destroyed.

---

## 4. Using Tulving over MCP

Tulving ships a thin, **local-only** (stdio, no network) MCP server exposing six tools:
`memory_store`, `memory_get`, `memory_search`, `memory_curate`, `memory_forget`,
`memory_list_keys` (orient = `memory_curate(mode="orient")`).

> The server's `--embedding` choices are `local` and `openai` (there is no zero-dependency
> embedder over MCP), so install `[mcp,local]` for a fully offline setup, or use
> `--embedding openai` with `OPENAI_API_KEY`.

### 4a. LM Studio + a local model

**Prerequisites**
- **LM Studio** with a **tool-calling-capable** model loaded (e.g. a Qwen2.5-Instruct or
  Llama-3.1-Instruct "tools"/"function-calling" model — a model without tool support cannot
  call MCP tools).
- `pip install "tulving[mcp,local]"` (fully offline; downloads a small embedding model on first
  run).

**Wire Tulving in.** LM Studio is an MCP host: it launches MCP servers and lets the loaded model
call their tools. Add Tulving to its MCP config (in the app, edit `mcp.json`; LM Studio uses the
same `mcpServers` shape as Claude Desktop):

```json
{
  "mcpServers": {
    "tulving": {
      "command": "tulving-mcp",
      "args": ["--memory-path", "/absolute/path/to/agent_memory", "--embedding", "local"]
    }
  }
}
```

Use an **absolute** `--memory-path`, save, and enable the `tulving` server. The model now sees
the six tools.

**Make the model use them.** A model won't use memory unless told to. Add to the system prompt:

```
You have a persistent memory via the `tulving` tools. Use it:
- When the user states a durable fact, decision, or plan, call memory_store
  (type = fact | decision | observation | plan; add a key like "decision:auth").
- At the START of a task, call memory_curate with the task description and a
  token_budget (e.g. 1500) to reload relevant context, or memory_curate with
  mode="orient" for a cold-start briefing.
- Prefer memory_get for an exact key you know; memory_search to find by meaning.
```

**Prove persistence with two chats.** In chat 1: "We decided to use SQLite over Postgres for
local deploys. Remember that." → the model calls `memory_store`. Close it; open chat 2 (empty
context): "What did we decide about the datastore?" → the model calls
`memory_get`/`memory_search`/`memory_curate` and answers correctly, recalling something it was
never told *in this chat*. That round-trip through a restart is the point.

> If the model doesn't call the tools, it's almost always a non-tool-calling model or a weak
> system prompt. Verify the tools independently with the [MCP Inspector](#4c-mcp-inspector--tool-reference)
> to isolate Tulving from the model's willingness to use it.

### 4b. Claude Code

```bash
pip install "tulving[mcp,local]"
claude mcp add tulving -- tulving-mcp --memory-path ./agent_memory --embedding local
claude mcp list        # confirm it's registered
```

Or commit a project `.mcp.json`:

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

The tools appear as `tulving:memory_store`, `tulving:memory_curate`, etc. Work normally and say
"remember this decision" or "what do we know about X"; for automatic use, add the memory-usage
guidance above to your project `CLAUDE.md`. To enable LLM-written summaries, run the server with
`--llm claude` and `ANTHROPIC_API_KEY` set (the server reads its own credentials).

### 4c. MCP Inspector — tool reference

The fastest way to see exactly what each tool does, with no model in the loop:

```bash
npx @modelcontextprotocol/inspector tulving-mcp --memory-path ./inspect_mem --embedding local
```

A UI lists the six tools; choose one, fill the JSON arguments, and call it. A full scripted
sequence with exact arguments and the exact responses Tulving returns:

| Step | Tool | Arguments | Response |
|---|---|---|---|
| 1 | `memory_store` | `{"content":"Chose JWT (RS256) over sessions.","type":"decision","key":"decision:auth","tags":["auth"]}` | `Stored memory <uuid> with key 'decision:auth'` |
| 2 | `memory_store` | `{"content":"Auth token TTL is 15 minutes.","type":"fact","key":"fact:ttl","tags":["auth"]}` | `Stored memory <uuid> with key 'fact:ttl'` |
| 3 | `memory_list_keys` | `{"prefix":"decision:"}` | `decision:auth` |
| 4 | `memory_get` | `{"key":"decision:auth"}` | a block: `[decision] decision:auth` / `id: …` / `tags: auth` / `importance: 0.50` / `created: …` / `---` / `Chose JWT (RS256) over sessions.` |
| 5 | `memory_search` | `{"query":"how does auth work","top_k":5}` | numbered lines like `1. [0.87 semantic \| decision \| decision:auth] Chose JWT…` |
| 6 | `memory_curate` | `{"query":"resuming the auth work","token_budget":800}` | prompt-ready text + footer `--- [tokens: N, budget remaining: M, sources consulted: K]` |
| 7 | `memory_curate` | `{"query":"","mode":"orient","token_budget":600}` | a cold-start briefing (Key Decisions, session history); with `--llm none`, ends with a "no LLM adapter configured" note |
| 8 | `memory_forget` | `{"key":"fact:ttl"}` | `Archived memory with key 'fact:ttl'` |
| 9 | `memory_get` | `{"key":"fact:ttl"}` | `No memory found for key 'fact:ttl'.` (archived ≠ retrievable) |

What each tool is for:

- **`memory_store`** — write a typed memory; storing to an existing key *supersedes* the old one
  (never errors). `importance` is 0–1 (schema-enforced).
- **`memory_get`** — exact recall by key; a miss returns a message, not an error.
- **`memory_search`** — find by meaning (semantic) plus exact key hits, ranked.
- **`memory_curate`** — the headline: the relevant slice packed into `token_budget`;
  `mode:"orient"` gives a cold-start briefing.
- **`memory_forget`** — remove by exactly one of key / id / tags; archives by default,
  `hard:true` deletes.
- **`memory_list_keys`** — enumerate keys, with an optional `prefix` filter.

### 4d. Global vs. per-project memory, and tagging

Tulving scopes memory **entirely by `--memory-path`** — that directory (one SQLite file + index)
is the whole boundary. There is no built-in "project" concept: point two sessions at the *same*
path and they share everything; point them at *different* paths and they are fully isolated.
Note the single-writer rule: only one *writable* server may hold a given path at a time (a second
is refused, or may open `--read-only`). Pick the model that fits how you work.

**Option 1 — Per-project memory (isolated).** Each project has its own store, its own writer lock,
and you can run many Claude Code instances in parallel. Cross-project facts must be stored in each
project separately. Either commit a project `.mcp.json`:

```json
{ "mcpServers": { "tulving": { "command": "tulving-mcp",
    "args": ["--memory-path", "./.tulving", "--embedding", "local"] } } }
```

…or install once at user scope with a **relative** path (it resolves against each project's
directory, so every project gets its own `./.tulving`):

```bash
claude mcp add --scope user tulving -- tulving-mcp --memory-path ./.tulving --embedding local
```

Add `./.tulving` to `.gitignore` unless you actually want to commit a project's memory.

**Option 2 — One global brain (shared across every project).** Install at user scope with a single
**absolute** path; every project reads and writes the same store:

```bash
claude mcp add --scope user tulving -- \
  tulving-mcp --memory-path /home/you/.tulving/global --embedding local
```

Trade-off: because of the single-writer rule, only **one** Claude Code instance can use it at a
time. That rules this out if you routinely run several sessions at once.

**Option 3 — Hybrid: per-project writable + a shared read-only knowledge base (recommended if you
run multiple instances).** This removes the "I have to repeat cross-project facts" downside without
the single-writer limitation. Register **two** servers — a per-project writable one, and a shared
one opened `--read-only` (read-only handles open concurrently, so all your parallel sessions can
read it at the same time):

```bash
# per-project, writable — isolated, one lock each, safe to run in parallel
claude mcp add --scope user tulving -- \
  tulving-mcp --memory-path ./.tulving --embedding local

# shared, READ-ONLY — universal knowledge every session can read at once
claude mcp add --scope user tulving-shared -- \
  tulving-mcp --memory-path /home/you/.tulving/global --embedding local --read-only
```

In a session the tools appear as `tulving:memory_*` (your project) and `tulving-shared:memory_*`
(the global knowledge base). `tulving-shared`'s `store`/`forget` tools return an error, which is
the read-only guarantee working as intended.

**Telling the agent where to save (local vs. shared).** The agent picks the store by the tool it
calls — `tulving:memory_store` writes this project's memory; `tulving-shared` is read-only, so it
*cannot* write there. In normal work the agent therefore saves everything locally, and simply
*reads* `tulving-shared` for cross-project facts. To **add** to the shared base, use one dedicated
"knowledge-base" project whose `.mcp.json` registers a **writable** server (e.g. `tulving-kb`) on
the same global path; open that project and tell the agent to store there, or populate it with a
short Python script. Only one writable server may hold the global path at a time, so keep writes to
that single knowledge-base session — every other project reads the result (picking up new entries
the next time its read-only session starts). Make the distinction explicit in each project's
`CLAUDE.md`; ready-to-paste blocks for all three setups are in
[`examples/claude-md-memory-snippet.md`](examples/claude-md-memory-snippet.md).

> Prefer writing shared knowledge on demand from *any* session? Then make the shared server
> **writable** instead of read-only — but only one Claude Code instance can hold it at a time, so
> this trades away parallel use. Read-only shared + a dedicated writer is what keeps many sessions
> working at once.

#### Tagging & key prefixes — organizing within a store

Tags and key-prefixes are how you structure a store and filter recall — essential for a shared
store, useful anywhere. **All tag filters are *any-of*:** an entry matches if it carries **at least
one** of the listed tags (and `exclude_tags` always wins).

**Add tags when storing** (the model does this via the MCP tool; you can do it directly in Python):

```jsonc
// MCP: memory_store arguments
{ "content": "Chose JWT (RS256) for auth.", "type": "decision",
  "key": "projA:decision:auth", "tags": ["projA", "auth"] }
```
```python
# Python
memory.store("Chose JWT (RS256) for auth.", type=MemoryType.DECISION,
             key="projA:decision:auth", tags=["projA", "auth"])
```

**Filter by tags / prefix when recalling:**

```jsonc
{ "query": "auth", "tags": ["projA"] }                       // memory_search: entries tagged projA
{ "query": "resuming auth", "include_tags": ["projA","shared"],
  "exclude_tags": ["scratch"] }                              // memory_curate: keep projA OR shared; drop scratch
{ "prefix": "projA:" }                                       // memory_list_keys: one project's keys
```

**Make the model tag automatically** — add to its system prompt (LM Studio) or `CLAUDE.md`:

```
When you store a memory, tag it with the project name (e.g. "projA") plus topic
tags (e.g. "auth", "billing"), and prefix its key with the project, like
"projA:decision:auth". Tag anything true across projects with "shared". When
recalling, call memory_curate with include_tags=["<project>", "shared"] so you
get both this project's memory and universal knowledge in one call.
```

That last convention is the tidy answer to sharing without mixing: keep a `shared` tag for
cross-project truths, and `curate(include_tags=[project, "shared"])` pulls the current project's
memory **and** the universal facts together — whether they live in one store or in the hybrid's
read-only global base.

---

## 5. Verifying it behaves correctly

Fast acceptance checks (via the Inspector or Python), each with a clear pass condition:

| Property | How to check | Pass condition |
|---|---|---|
| **Persistence across restarts** | store in one process, exit, reopen the same `--memory-path`, `memory_get` the key | the value comes back (it's on disk) |
| **Upsert / supersede** | `memory_store` twice with the same key, different content; `list_keys`; `get` | one key; `get` returns the *new* content; no error on the duplicate |
| **Budget adherence** | `memory_curate` with `token_budget=300` on a large store | footer `tokens:` ≤ 300 |
| **Relevance** | `memory_search`/`curate` for a topic you stored | the on-topic entries rank first |
| **Forget** | `memory_forget` a key, then `memory_get` it | "No memory found" (archived) |
| **Redaction (security)** | store content `my key is sk-ABCDEFGHIJKLMNOPQRSTUVWX`; `memory_get`/`curate` it | output shows `[REDACTED]`, never the token |

**The end-to-end continuity test** (the one that proves the value): in process #1 store two
decisions and exit; in a brand-new process #2 (empty model context) call
`memory_curate(mode="orient")` — the briefing contains those decisions. Memory outlived the
process. That is what an agent gains that a stateless prompt does not.

---

## 6. Measuring the value vs. no memory

Two things matter: **token cost** and **answer correctness**. Compare three conditions per
question — **(A)** no memory, **(B)** dump every memory into the prompt, **(C)** Tulving
`curate` (relevant slice within a budget).

### 6a. Token efficiency (deterministic — no model required)

Save `eval_tokens.py` and run it. It seeds a realistic project store and measures. Representative
output (with the `len//4` fallback estimator; a real tokenizer/embedder shifts the exact numbers,
not the shape):

```
SCALING  (fixed probe, budget=200): the dump grows with the store; curate stays flat
 store size   dump tokens   curate tokens   reduction
         20           367             155          2x
        100          1791             162         11x
        500          9107             166         54x
```

The shape is the argument: **`curate` is O(budget); the dump is O(store size).** At a small store
the win is modest; at 500 memories it is 54×, and beyond a few thousand the dump no longer fits
the context window while `curate` still returns a tight, relevant block.

```python
# eval_tokens.py  — runs anywhere; no model, no network.
import tempfile, os
from tulving import Memory, MemoryType, HashEmbedder
from tulving.context.curator import resolve_estimator

est = resolve_estimator()                       # tiktoken if installed, else len//4
toks = lambda s: est.estimate(s)

SEED = [
    ("Chose SQLite (WAL) over Postgres for zero-infra local deploy.", MemoryType.DECISION, "decision:datastore", ["arch","db"], 0.9),
    ("Chose JWT (RS256) over sessions for the auth layer.", MemoryType.DECISION, "decision:auth", ["auth"], 0.85),
    ("Auth tokens expire after 15 minutes; refresh after 7 days.", MemoryType.FACT, "fact:auth-ttl", ["auth"], 0.7),
    ("Chose Stripe over Paddle for billing (better API, EU support).", MemoryType.DECISION, "decision:billing", ["billing"], 0.8),
    ("Rate limit: 100 req/min per API key, 429 on exceed.", MemoryType.FACT, "fact:rate-limit", ["api"], 0.6),
]

print(f"{'store size':>11} {'dump tokens':>13} {'curate tokens':>15} {'reduction':>11}")
for n in (20, 100, 500):
    d = os.path.join(tempfile.mkdtemp(), "s")
    m = Memory(d, agent_id="dev", embedding_adapter=HashEmbedder())
    entries = [f"[{t.value}] {k} {c}" for c, t, k, tg, i in SEED]
    for c, t, k, tg, i in SEED:
        m.store(c, type=t, key=k, tags=tg, importance=i)
    for j in range(len(SEED), n):               # pad with realistic noise
        text = f"Misc note #{j}: routine log entry about module {j % 12}."
        m.store(text, type=MemoryType.OBSERVATION, key=f"note:{j}", tags=["log"], importance=0.2)
        entries.append(f"[observation] note:{j} {text}")
    dump = "\n".join(entries)
    ctok = m.curate("resuming the auth work", token_budget=200).token_count
    dtok = toks(dump)
    print(f"{n:>11} {dtok:>13} {ctok:>15} {f'{dtok // max(ctok,1)}x':>11}")
    m.close()
```

### 6b. Answer correctness with your real model

Token savings only matter if the answers stay right. This harness asks your **LM Studio** model
each question under all three conditions and scores against a known answer. LM Studio exposes an
OpenAI-compatible endpoint at `http://localhost:1234/v1` — no extra Python dependencies (standard
library `urllib`). Load a model in LM Studio, start its local server, then run it.

Expected result: **A (no memory) fails** the recall questions; **B (dump) and C (curate) both
answer correctly**, but **C sends far fewer tokens** — and C keeps working as the store grows,
while B eventually overflows the window. That gap is the case for using Tulving.

```python
# eval_model.py  — needs LM Studio running with a model loaded and its server started.
import json, os, tempfile, urllib.request
from tulving import Memory, MemoryType, HashEmbedder

LM_URL = "http://localhost:1234/v1/chat/completions"
MODEL  = os.environ.get("LM_MODEL", "local-model")   # any string; LM Studio ignores it

def ask(context, question):
    sys = "Answer ONLY from the provided context. If it is not there, say 'unknown'."
    body = json.dumps({"model": MODEL, "temperature": 0, "messages": [
        {"role": "system", "content": sys + "\n\nCONTEXT:\n" + (context or "(none)")},
        {"role": "user", "content": question},
    ]}).encode()
    req = urllib.request.Request(LM_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()

d = os.path.join(tempfile.mkdtemp(), "proj")
m = Memory(d, agent_id="dev", embedding_adapter=HashEmbedder())
SEED = [
    ("Chose JWT (RS256) over sessions for the auth layer.", MemoryType.DECISION, "decision:auth", ["auth"], 0.85),
    ("Auth tokens expire after 15 minutes.", MemoryType.FACT, "fact:ttl", ["auth"], 0.7),
    ("Chose Stripe over Paddle for billing.", MemoryType.DECISION, "decision:billing", ["billing"], 0.8),
]
for c, t, k, tg, i in SEED:
    m.store(c, type=t, key=k, tags=tg, importance=i)
dump = "\n".join(c for c, *_ in SEED)

PROBES = [  # (question, substring a CORRECT answer must contain)
    ("What auth scheme did we choose?", "JWT"),
    ("How long do auth tokens live?", "15"),
    ("Which billing provider did we pick?", "Stripe"),
]
score = {"none": 0, "dump": 0, "curate": 0}
for q, needle in PROBES:
    contexts = {"none": "", "dump": dump, "curate": m.curate(q, token_budget=400).content}
    for cond, ctx in contexts.items():
        ok = needle.lower() in ask(ctx, q).lower()
        score[cond] += ok
        print(f"{cond:7} {q:38} {'OK' if ok else 'MISS'}")
m.close()
print("\nCorrect:", {k: f"{v}/{len(PROBES)}" for k, v in score.items()})
```

Scale the seed to hundreds of entries and condition **B** starts failing (truncation, lost in the
middle, window overflow) while **C** holds. That is Tulving's value, quantified.

---

## 7. Storage model & platform support

Tulving uses a **two-layer, zero-infrastructure** storage model:

- **Structured layer** (memories, tags, sessions, metadata) → **SQLite in WAL mode**. SQLite is
  part of the Python standard library, so there is nothing to install.
- **Vector layer** (semantic search) → **hnswlib**, cosine distance. This is a rebuildable cache:
  embeddings persist as BLOBs in SQLite and the index file can always be regenerated. It ships as
  the one core dependency.

| Store | Role in Tulving | Install |
|---|---|---|
| **SQLite (WAL)** | the sole production backend for the structured layer | nothing (standard library) |
| **hnswlib (cosine)** | the vector index for semantic search | included with `pip install tulving` |
| **In-memory** | ephemeral backend used by the test suite | nothing |
| **PostgreSQL + pgvector** | *not part of this release* — the documented future upgrade path for multi-process deployments | not applicable |

Multi-process concurrency is out of scope by design: Tulving enforces a **single writer per
memory path**. A second writer (or a second `tulving-mcp`) on the same path is refused with a
clear error, or may open read-only; this prevents silent index corruption, which has no safe
cross-process coordination in the embedded index.

**Platform support.** Tulving is tested on **Linux and Windows** across Python 3.11–3.13 in CI on
every change. Do not place a memory path on OneDrive or a network-synced folder — SQLite WAL and
advisory locking are unreliable there, and Tulving emits a warning when it detects one.

**Security note.** Memories are stored **unencrypted** at rest in this release. Redaction masks
secret-shaped tokens on emission surfaces (curated context, MCP responses, exports), but do not
rely on it to store credentials you cannot afford on disk.

---

## 8. Running the test suite

For contributors and anyone verifying a build from source:

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

The suite runs in a few seconds and uses the zero-dependency `HashEmbedder` plus mocked network
adapters, so `[local]`/`[openai]`/`[anthropic]` are not required to run it (a small number of
optional-adapter tests self-skip when those extras are absent). These are exactly the gates CI
runs across the Linux + Windows × Python 3.11–3.13 matrix (`.github/workflows/ci.yml`).

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: mcp` when running `tulving-mcp` | Install the extra: `pip install "tulving[mcp]"`. |
| `tulving-mcp` exits with a `[local]`/`[openai]` hint | The chosen `--embedding` backend isn't installed or has no key. Install `[local]`, or set `OPENAI_API_KEY` and use `--embedding openai`. |
| The model connects but never calls the tools | Use a tool-calling-capable model and add explicit memory-usage guidance to its system prompt (§4a). Confirm the tools work with the Inspector (§4c). |
| `SecurityError: path resolves outside the allowed root` on export | The export path is outside the memory directory. Pass `allowed_root=...` or write inside the memory dir. |
| A second `tulving-mcp` on the same path exits immediately | Expected — single-writer safety. Close the other session, or start the second one with `--read-only`. |
| `hnswlib` fails to install | No prebuilt wheel for your platform/Python — install a C/C++ toolchain (§1) or use Python 3.11–3.13. |
| A `UserWarning` about synced/network storage | Your memory path is under OneDrive or a network share — move it to local disk. |

---

*Questions or issues: <https://github.com/govanxa/tulving/issues>. Licensed under Apache 2.0.*
