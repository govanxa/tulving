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
pip install "tulving[local]"        # fully-offline embeddings (sentence-transformers + torch, several hundred MB)
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

> **The MCP server requires an embedder to start** — `--embedding local` or `--embedding openai`
> (it defaults to `local`; there is **no** torch-free / zero-dependency embedder over MCP in v0.1):
> - **`--embedding local`** — fully offline, but installs `sentence-transformers` **+ `torch`
>   (several hundred MB)**. This is what local-LLM / LM Studio users want:
>   `pip install "tulving[mcp,local]"`.
> - **`--embedding openai`** — avoids torch, but sends your memory text to OpenAI to compute
>   embeddings (**not offline**; needs `OPENAI_API_KEY` + network):
>   `pip install "tulving[mcp,openai]"`.
>
> **This is independent of the MCP host.** Claude Code and LM Studio are both just *hosts* — they
> spawn the server and let their model call the tools; neither supplies embeddings. So the same
> choice, and the same torch requirement for an offline setup, applies to **both**. There is **no
> Anthropic/Claude embedder**, and `--llm claude` powers Tulving's *own* summaries (not
> embeddings) — it does **not** remove the torch requirement.

> **Coming in v0.2 — torch-free MCP.** If all you want is `curate`'s token-budget reduction and you
> don't need semantic search (find-by-meaning), note that v0.1 still forces a `local` (torch) or
> `openai` embedder just to *start* the server. v0.2 will add an **embedder-free mode**
> (`--embedding none`) — exact-key + importance/recency curation with **no torch and no network** —
> the right fit for token-reduction-only users on LM Studio or Claude Code. Until then, use
> `[mcp,local]` (offline, torch) or `[mcp,openai]` (cloud, no torch).

#### Do I have to tell the model "remember this" every time?

**No — you tell it once, as a policy, not per message.** Tulving is a passive tool provider: it
exposes the six tools and does nothing until the model calls one. There is no background process
watching the chat and auto-capturing "important" things (that machinery is deferred, ADR-016). So
whether a model stores autonomously is entirely a function of its **instructions**, not a Tulving
setting:

- **With no instruction**, a model rarely stores unprompted — you'd have to say "remember this"
  each time.
- **With a standing storage policy** in the system prompt (LM Studio) or `CLAUDE.md` (Claude Code),
  the model decides *what* is durable and stores it on its own — choosing a sensible `type`, `key`,
  and `tags` — without you narrating each save.
- **Explicit "remember this"** always works as a 100%-reliable override on top of either.

Give it **two** standing instructions and it runs the whole loop hands-off: a *storage* policy
("store durable decisions/facts/plans as they arise; skip chatter and secrets") and a *recall*
policy ("at the start of a task, call `memory_curate`, or `mode="orient"`, to reload context").
Ready-to-paste blocks are in [`examples/memory-snippet.md`](examples/memory-snippet.md); the
system-prompt snippet in [§4a](#4a-lm-studio--a-local-model) is the storage half.

Two caveats. **Reliability scales with model capability:** frontier models (Claude, GPT-4-class)
follow the policy well and fire the tools proactively; small local models understand the tools but
are less consistent at *proactively* storing mid-conversation, so they benefit from an occasional
nudge (recall tends to be steadier than proactive storage). And **over-storing is self-correcting:**
if an eager model saves marginal notes, lazy decay and eviction sink and archive them automatically
and `curate` only surfaces what's relevant — so it is safe to hand the model judgment rather than
gate every write yourself.

### 4a. LM Studio + a local model

**Prerequisites**
- **LM Studio** with a **tool-calling-capable** model loaded (e.g. a Qwen2.5-Instruct or
  Llama-3.1-Instruct "tools"/"function-calling" model — a model without tool support cannot
  call MCP tools).
- `pip install "tulving[mcp,local]"`. The MCP server needs an embedder; `[local]` runs fully
  offline but installs `sentence-transformers` **+ `torch` (several hundred MB)** and downloads a
  ~90 MB embedding model on first run. (To avoid torch, use `--embedding openai` instead — cloud,
  needs a key; see §4.)

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
  (type = fact | decision | observation | plan; add a key like
  "decision:api-scheme" or "fact:session-ttl").
- At the START of a task, call memory_curate with the task description and a
  token_budget (e.g. 1500) to reload relevant context, or memory_curate with
  mode="orient" for a cold-start briefing.
- Prefer memory_get for an exact key you know; memory_search to find by meaning.
- Do NOT put the words auth, token, secret, password, key, or credential in a
  key unless the value really is a secret — Tulving masks the whole content of
  a key that looks like it names one. Use neutral keys (e.g. "decision:login-flow",
  not "decision:auth"; "fact:session-ttl", not "fact:auth-ttl").
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
| 1 | `memory_store` | `{"content":"Chose JWT (RS256) over sessions.","type":"decision","key":"decision:api-scheme","tags":["auth"]}` | `Stored memory <uuid> with key 'decision:api-scheme'` |
| 2 | `memory_store` | `{"content":"Session tokens live for 15 minutes.","type":"fact","key":"fact:session-ttl","tags":["auth"]}` | `Stored memory <uuid> with key 'fact:session-ttl'` |
| 3 | `memory_list_keys` | `{"prefix":"decision:"}` | `decision:api-scheme` |
| 4 | `memory_get` | `{"key":"decision:api-scheme"}` | a block: `[decision] decision:api-scheme` / `id: …` / `tags: auth` / `importance: 0.50` / `created: …` / `---` / `Chose JWT (RS256) over sessions.` |
| 5 | `memory_search` | `{"query":"how does auth work","top_k":5}` | numbered lines like `1. [0.87 semantic \| decision \| decision:api-scheme] Chose JWT…` |
| 6 | `memory_curate` | `{"query":"resuming the auth work","token_budget":800}` | prompt-ready text + footer `--- [tokens: N, budget remaining: M, sources consulted: K]` |
| 7 | `memory_curate` | `{"query":"","mode":"orient","token_budget":600}` | a cold-start briefing (Key Decisions, session history); with `--llm none`, ends with a "no LLM adapter configured" note |
| 8 | `memory_forget` | `{"key":"fact:session-ttl"}` | `Archived memory with key 'fact:session-ttl'` |
| 9 | `memory_get` | `{"key":"fact:session-ttl"}` | `No memory found for key 'fact:session-ttl'.` (archived ≠ retrievable) |

> **Watch your key names.** The keys above deliberately avoid the words `auth`, `token`, `secret`,
> `password`, `key`, and `credential`. Tulving treats a key containing any of those as naming a
> secret and **masks the whole content as `[REDACTED]`** on every emission surface (get, curate,
> export) — a safety default, but a surprise if you keyed a non-secret `fact:auth-ttl`. Use neutral
> keys (`fact:session-ttl`, `decision:api-scheme`, `decision:login-flow`) for ordinary memories;
> the content itself is never redacted for containing the word "auth", only the *key* matters. This
> is a known sharp edge slated for softening in v0.2.

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
[`examples/memory-snippet.md`](examples/memory-snippet.md).

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
  "key": "projA:decision:api-scheme", "tags": ["projA", "auth"] }
```
```python
# Python
memory.store("Chose JWT (RS256) for auth.", type=MemoryType.DECISION,
             key="projA:decision:api-scheme", tags=["projA", "auth"])
```

(Note the key is `decision:api-scheme`, not `decision:auth` — a key containing `auth`/`token`/
`secret`/`password`/`key`/`credential` gets its content masked as `[REDACTED]`. Tags are never
redacted, so `tags: ["auth"]` is fine.)

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
"projA:decision:api-scheme". Do NOT put auth/token/secret/password/key/credential
in a key (Tulving masks such keys); keep keys neutral. Tag anything true across
projects with "shared". When recalling, call memory_curate with
include_tags=["<project>", "shared"] so you get both this project's memory and
universal knowledge in one call.
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
    ("Chose JWT (RS256) over sessions for the auth layer.", MemoryType.DECISION, "decision:api-scheme", ["auth"], 0.85),
    ("Auth tokens expire after 15 minutes; refresh after 7 days.", MemoryType.FACT, "fact:session-ttl", ["auth"], 0.7),
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
    ("Chose JWT (RS256) over sessions for the auth layer.", MemoryType.DECISION, "decision:api-scheme", ["auth"], 0.85),
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

### Inspecting the store

Everything is a plain SQLite database, so you can browse it with **DB Browser for SQLite**,
**JetBrains DataGrip**, or the `sqlite3` CLI. Open the file at `<your --memory-path>/tulving.db`:

- **DB Browser for SQLite:** *File → Open Database Read Only* → `tulving.db`.
- **DataGrip:** *New → Data Source → SQLite* → point at `tulving.db` (the SQLite driver is bundled;
  enable read-only in the options).
- **CLI:** `sqlite3 tulving.db` then `.tables`.

Files in the memory directory:

| File | What it is |
|---|---|
| `tulving.db` | the SQLite database — **open this** |
| `tulving.db-wal` / `-shm` | WAL sidecars; present only while Tulving is running (a DB tool reads them via `tulving.db` automatically) |
| `tulving.hnsw` | the vector index (a binary hnswlib cache, not SQLite) |
| `tulving.lock` | Tulving's advisory writer lock |

Tables: `memories`, `memory_tags` (`entry_id`, `tag`), `sessions`, `meta` (`schema_version`,
`embedding_model_id`, `dimension`, `distance_metric`), and `vector_labels` (the int-label↔UUID map
for the vector index). Two queries to start:

```sql
-- live memories (superseded/forgotten rows linger with archived = 1)
SELECT key, type, base_importance, tags, created_at
FROM memories WHERE archived = 0;

-- normalized tags
SELECT m.key, t.tag
FROM memories m JOIN memory_tags t ON t.entry_id = m.id
WHERE m.archived = 0;
```

Three things to keep in mind:

1. **Inspect read-only; do not edit while Tulving is running.** The `tulving.lock` file coordinates
   only other *Tulving* processes — an external DB tool ignores it. Reading concurrently is safe
   (WAL allows readers), but editing rows from outside can violate invariants and desync the
   `tulving.hnsw` index. If you must edit, do it with Tulving stopped, then **delete `tulving.hnsw`**
   so Tulving rebuilds it from the embedding BLOBs on the next `startup()` (the index is a
   rebuildable cache — the SQLite BLOBs are the source of truth).
2. The `embedding` column is a packed float32 **BLOB** — not human-readable; ignore it.
3. Only `base_importance` is stored; the effective (decayed) importance is computed lazily on read,
   so it does not appear in any column. Note `UNIQUE(key) WHERE archived = 0` — one live entry per
   key, with older copies kept as `archived = 1`.

### Maintenance & housekeeping — what a growing store needs

The design deliberately decouples **store size** from **prompt cost**, so for everyday use there
is almost nothing to maintain. The one thing that grows is the SQLite file on disk, and there's a
reclaim path for it.

**Automatic — you do nothing:**

- **Curation is token-budget-bounded.** `curate(query, token_budget)` returns the same-sized block
  whether the store holds 50 memories or 50,000 — what you feed the model never grows with the
  store (see [§6](#6-measuring-the-value-vs-no-memory)). This is the point of the whole design.
- **Lazy decay** demotes unused entries on read (`base * 0.5^(age / half_life)`); `DECISION` and
  pinned entries never decay, so stale notes sink below the recall cut on their own.
- **Eviction runs on every `startup()`** — each time the MCP server boots (i.e. each session), a
  time-boxed pass *archives* entries whose effective importance fell below the eviction threshold
  (pinned/`DECISION` are exempt). It only moves rows out of the active set; it never deletes.
- **WAL and the vector index self-manage:** the WAL is checkpoint-truncated on clean shutdown, and
  `tulving.hnsw` uses tombstone-and-rebuild compaction (and is rebuildable from the BLOBs anyway).

**Manual — only when the DB file itself gets large.** Tulving *archives, never destroys* (D2):
every supersede, forget, and eviction leaves an `archived = 1` row behind, so `tulving.db` grows
with your total write history even though the active set stays lean. For a per-project store this
is slow (tens of MB over a long time, not runaway). When you want to reclaim it — with Tulving
**stopped** (single-writer rule), from the SDK:

```python
from datetime import timedelta
from tulving import Memory

m = Memory("/path/to/agent_memory", agent_id="dev")
m.startup()

m.export_json("backup.json", allowed_root=".",              # 1. safety backup first
              include_archived=True, include_sensitive=True)
purged = m.purge_archived(older_than=timedelta(days=90))    # 2. drop old archived rows
print(f"purged {purged} archived rows")
m.close()                                                   # checkpoints the WAL
# 3. reclaim file size:  sqlite3 /path/to/agent_memory/tulving.db "VACUUM;"
```

`purge_archived` is reason-aware: it **refuses to delete summarization-source entries** unless you
name that reason explicitly, so compressing sessions into summaries never silently loses the
sources. `DECISION`/pinned entries are never eviction targets in the first place, so a purge of
archived rows can't remove a live decision.

> **v0.1 gap:** purge and vacuum are **SDK-only** — there is no MCP tool or `tulving-mcp` flag for
> them yet, so housekeeping means a short Python script or a scheduled job as above. A maintenance
> CLI (`tulving-mcp --inspect` / `--purge` / `--vacuum`) is planned for a later release; day to day,
> a per-project store rarely needs any of this.

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
