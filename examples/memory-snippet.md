# Memory snippet

Paste one of the blocks below into a project's `CLAUDE.md` (or an LM Studio system prompt) so
the agent uses Tulving's memory tools consistently. Pick the block that matches how you set up
the MCP server (see [`GUIDE.md` §4d](../GUIDE.md#4d-global-vs-per-project-memory-and-tagging)).

---

## 1. Basic — one store (per-project or a single global store)

Copy everything inside the fence into your `CLAUDE.md`:

```markdown
## Memory (Tulving)

You have persistent memory via the `tulving` MCP tools. Use it every session:

- **Start of a task:** call `memory_curate` with the task description and a
  `token_budget` (~1500) to reload relevant context, or `memory_curate` with
  `mode="orient"` for a cold-start briefing. Use `memory_get` for a key you
  already know; `memory_search` to find by meaning.
- **When the user states something durable** (a decision, fact, plan, or useful
  observation), call `memory_store`:
    - `type`: `decision` | `fact` | `plan` | `observation`
    - `key`: a stable, prefixed handle, e.g. `decision:auth`, `fact:rate-limit`
    - `tags`: the topic(s), e.g. `["auth"]`
- Storing to an existing `key` replaces the old value — reuse keys to update.
- Never store secrets, credentials, or throwaway chatter.
```

---

## 2. Tagged — a shared store you want to keep organized by project

Use this when several projects share one store and you want to keep them separable. Adds a
project tag and prefix convention on top of block 1:

```markdown
## Memory (Tulving)

You have persistent memory via the `tulving` MCP tools. This store is shared across
projects, so scope everything to the current project (PROJECT_TAG = `proj-acme`).

- **Start of a task:** call `memory_curate` with the task description, a
  `token_budget` (~1500), and `include_tags=["proj-acme", "shared"]` so you get
  both this project's memory and cross-project knowledge. Use `mode="orient"`
  for a cold-start briefing.
- **When the user states something durable**, call `memory_store`:
    - `type`: `decision` | `fact` | `plan` | `observation`
    - `key`: prefix with the project, e.g. `proj-acme:decision:auth`
    - `tags`: `["proj-acme", <topic>]`; add `"shared"` only if it is true for
      every project, not just this one.
- Recall within this project with `include_tags=["proj-acme", "shared"]`;
  list keys with `prefix="proj-acme:"`.
- Never store secrets, credentials, or throwaway chatter.
```

Replace `proj-acme` with your project's short name.

---

## 3. Hybrid — per-project writable `tulving` + read-only `tulving-shared`

Two servers are registered (see GUIDE §4d, Option 3): `tulving` is this project's writable
memory; `tulving-shared` is a global knowledge base opened **read-only**.

**In a normal project**, paste this — the agent writes locally and only *reads* the shared base:

```markdown
## Memory (Tulving)

You have TWO memory scopes:
- `tulving:*` — THIS project's memory. Writable. Store and recall here.
- `tulving-shared:*` — cross-project knowledge. READ-ONLY. Read it for shared
  facts; never try to write to it (its store/forget tools return an error).

- **Start of a task:** call `tulving:memory_curate` (task description,
  `token_budget` ~1500) for project context, AND `tulving-shared:memory_curate`
  (or `mode="orient"`) for relevant cross-project knowledge.
- **Store durable project facts** with `tulving:memory_store`
  (`type`, a prefixed `key`, and `tags`).
- If the user says something is a *cross-project* / *global* truth, tell them it
  must be added from the shared knowledge-base project — do NOT try to write it
  to `tulving-shared` here.
- Never store secrets, credentials, or throwaway chatter.
```

**In your dedicated "knowledge-base" project** (the one whose `.mcp.json` registers a *writable*
server — call it `tulving-kb` — on the global path), paste this instead:

```markdown
## Memory (Tulving — shared knowledge base)

`tulving-kb:*` writes to the SHARED, cross-project knowledge base. Use it only
for truths that apply to every project.

- **Store** with `tulving-kb:memory_store`: `type`, a `key`, and always include
  the tag `"shared"` (plus topic tags).
- Keep entries general and reusable — no project-specific details here.
- Never store secrets or credentials.
```

Run only ONE writable server on the shared path at a time (single-writer rule). Other projects
pick up newly-added shared knowledge the next time their read-only session starts.
