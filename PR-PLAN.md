# Upstream PR Plan — `dataO1/graphrag-rs:openai-compat` → `automataIA/graphrag-rs:main`

**Status as of last update**: pre-flight; no PRs filed yet.

This document is the source of truth for what we plan to upstream, why,
and how we'd split it. Update it whenever a commit is added/removed
from `openai-compat`, when an upstream maintainer responds, when a PR
is filed/merged/rejected, or when a design decision is made that
affects the upstream-facing surface.

`TODO.md` tracks **future work scope** (Phase E, F, …). This document
tracks **the state of upstream contributions**: what's ready, what's
filed, what landed.

## About these contributions

graphrag-rs is being used here as the discovery layer for local LLM
agents — the goal is a single MCP-shaped surface where coding/research
agents (Claude Code, opencode, crush) can ingest, index, and query a
personal knowledge base that lives entirely on the local machine. No
hosted APIs in the loop. The setup is:

- **Chat backend**: `llama-server` (`llama.cpp`'s OpenAI-compatible
  HTTP server) running a Qwen3-class GGUF locally. Same model is shared
  across multiple agent clients, which is why per-request controls like
  `chat_template_kwargs.enable_thinking=false` matter — Qwen3's
  reasoning would otherwise truncate JSON extraction output.
- **Embedding backend**: OpenVINO Model Server (OVMS) on the Intel NPU,
  serving the standard OpenAI-compat `/v1/embeddings` endpoint. mxbai-
  embed-large-v1 baked into a static-shape graph, ~350 ms per call,
  graph runs on dedicated NPU cores.
- **Vector store**: Qdrant running locally.
- **Agent surface**: `graphrag-mcp` (MCP-over-stdio bridge that proxies
  to the REST server), wired into MCP-aware editors so agents can
  search the user's notes inline during a coding session.

Graphrag-rs was the only project in the Rust ecosystem that hit the
right shape for this — full GraphRAG pipeline (chunking, entity
extraction, relationship graph, community detection, retrieval), MIT
licensed, embedded library plus REST server, and already built around
the right abstractions. The code is well-organized, the trait surface
is sensible, the configuration model is rich. Genuine respect for the
work that's gone in — the project's value to this use case is what
made the local-first agent setup viable in the first place. Thank you
for it.

The roadblocks that produced these patches were all in the
"non-OpenAI-non-Ollama backend" direction. The chat path was hardcoded
to Ollama protocol; the embedding side had a parsed-but-unused OpenAI
config branch; the request body had no escape hatch for server-specific
knobs (Qwen3 reasoning suppression, vLLM JSON mode). A few smaller
issues turned up alongside: the `/api/config` endpoint was unreachable
due to apistos scope shadowing, partial config posts were resetting
unset fields, and the agent UX surface (list_documents, delete by user
id, dedup, last-built timestamp) had visible rough edges once an LLM
client started exercising it end-to-end.

These PRs aim to land that "OpenAI-compatible local stack" path as a
first-class option in graphrag-rs, on parity with the existing Ollama
path — same feature-gate pattern, same config shape, same tests. The
goal is that anyone running a local OpenAI-compat server (vLLM,
llama.cpp, OVMS, OpenRouter, OpenAI proper) can drive graphrag-rs
without forking. None of the changes alter the existing Ollama or
hash-fallback paths.

We're aware these are non-trivial PRs against a project we don't
maintain. Happy to iterate on style, scope, or shape — split a PR
further, hold one back, change a feature-gate name, drop a commit
that doesn't fit the project's direction. Reasonable to say no to any
of them.

## PR writing style

When drafting PR bodies, the voice should be:

- Concise, informative, unambiguous. No filler.
- Structured for fast human reading: **(1) motivation**, **(2) goals**,
  **(3) what changed**, **(4) methodology** (testing + implementation
  approach). One paragraph or short bullet list per section is the
  default — only expand when the change genuinely needs it.
- No emojis. No marketing language. No "we hope" / "we believe"
  hedging — state facts and decisions.
- Tone: open-source colleague. Professional but not distanced.
  Acknowledge the maintainer's work where natural (once per PR is
  enough, not every section). Treat the maintainer as a human reviewer
  whose time you respect, not a process to be navigated.
- Open to suggestions. Make alternatives explicit ("happy to gate this
  behind X instead", "open to splitting this further") so the
  maintainer doesn't have to fight the framing to push back.
- Assume the maintainer doesn't know us. No personal introduction,
  no backstory beyond what's relevant to the change.
- The maintainer can say no. Don't pre-argue every objection — note
  the obvious alternative once, then let the discussion happen in
  review. Avoid sentences that start with "we strongly believe".
- Flag breaking-but-fix changes explicitly with a one-line "back-compat
  note" so reviewers don't have to dig.

## Upstream posture

- Repo: [`automataIA/graphrag-rs`](https://github.com/automataIA/graphrag-rs), MIT, not archived.
- `hasIssuesEnabled: true`. Two prior external PRs (#4, #5 by `joubertb`),
  both merged within ~2 weeks. Last upstream push 2026-03-18.
- No CI workflows in `.github/workflows/`; bar is "green local
  `cargo test` + green `cargo check` across feature combos." No
  automated PR gating.
- Default branch: `main`.

## Commits on `openai-compat` (`upstream/main..openai-compat`)

In topological order. PR-relevance grouping in the rightmost column.

| # | sha | commit | LOC | group |
|---|---|---|---|---|
| 1 | `878dfed` | qdrant-client: opt out of generate-snippets default feature | 1 | A — fixes |
| 2 | `15ad18e` | rename /api/config → /config to fix scope shadowing | 9 | A |
| 3 | `9b27417` | read OLLAMA_PORT from env | 9 | A |
| 4 | `5f53117` | merge doubled resource("") in /api/documents (was 405) | 3 | A |
| 5 | `7b05a06` | deep-merge POST /config bodies over defaults | 39 | A |
| 6 | `fee7490` | graphrag-server: OpenAI-compat embedding backend | 181 | B — feature |
| 7 | `9c129bd` | graphrag-core: OpenAI-compat chat + ChatClient enum | 475 | B |
| 8 | `898f5ae` | build_graph: route LLM gates through Config::chat_enabled() | 35 | B |
| 9 | `13572bf` | OpenAIConfig.extra_body for per-request body extras | 34 | B |
| 10 | `cbfe74e` | extraction max_tokens Option + active-backend reads | 45 | B |
| 11 | `0d303f3` | GET /api/embeddings/stats | 35 | C — observability |
| 12 | `4e33145` | TODO.md Phase E (CLI parity) | 157 | **internal — do NOT PR** |
| 13 | `662c86b` | TODO.md Phase F (Claude skill) | 129 | **internal — do NOT PR** |
| 14 | `ff39c84` | README: document OpenAI-compatible chat backend | 42 | B (folded into PR 2) |
| 15 | `79e3034` | feature-gate the openai-compat backend | 287 | B (folded into PR 2) |
| 16 | `0bd7018` | add PR-PLAN.md | 209 | **internal — do NOT PR** |
| 17 | `9135482` | graphrag-server: real list_documents, user-id resolution, content-hash dedup, last_built_at | 344 | D — UX |
| 18 | `82f271c` | PR-PLAN: motivation + writing-style sections | 84 | **internal — do NOT PR** |
| 19 | `f9bcfac` | graphrag-server: POST /api/graph/append for incremental updates | 164 | E — append |

(Anything added after this point — append rows here when committing to `openai-compat`.)

## PR split

### PR 1 — Server-side fixes & build cleanup
**Group A.** ~60 LOC across 5 commits. Boring, fast review, lands first
to clear PR 2.

**Cherry-pick**: `878dfed 15ad18e 9b27417 5f53117 7b05a06`.

**Title**: `Server-side fixes: scope shadowing, OLLAMA_PORT env, doubled resource, qdrant build, deep-merge /config`.

**Things to call out in the body**:
- `15ad18e` is functionally a **fix, not a rename** — upstream registers
  `/api/config` as a plain `web::scope` *after* the apistos `/api`
  scope, which means the apistos scope's prefix-match shadows it and
  every `/api/config` request 404s in stock builds. Renaming the plain
  scope to `/config` makes it reachable for the first time. Consider
  asking maintainer if a back-compat alias is desired.
- `7b05a06` is a behavior change for `POST /config`: partial bodies now
  preserve unset fields instead of resetting them to defaults. Flag
  clearly.

### PR 2 — OpenAI-compatible chat & embedding backends
**Group B.** ~775 LOC across 6 commits (including README + feature gate).

**Cherry-pick**: `fee7490 9c129bd 898f5ae 13572bf cbfe74e ff39c84 79e3034`.

**Title**: `Add OpenAI-compatible chat and embedding backends`.

**Sections to include in body**:
- *Chat (graphrag-core)*: new `OpenAIConfig` + `OpenAIClient` (ureq/spawn_blocking),
  `ChatClient` enum dispatcher. `OpenAIConfig` sits next to `OllamaConfig`
  on `Config`; `ChatClient::from_config` picks the active one.
- *Embeddings (graphrag-server)*: `EmbeddingService` got an OpenAI-compat
  branch (vLLM, llama.cpp `--embedding`, OVMS, OpenAI proper, OpenRouter, …).
- *extra_body*: per-request top-level fields merged into every
  `/chat/completions` body. Lets users pass server-specific knobs
  (`chat_template_kwargs.enable_thinking=false` for Qwen3 on llama.cpp,
  `response_format` for vLLM JSON mode) without growing OpenAIConfig
  for every backend quirk. Set fields beat extra_body collisions.
- *Token-cap rework*: extraction `max_tokens` now `Option<usize>` —
  `None` = "no cap", model stops at EOS. Useful for local LLMs where
  token cost is just compute time. Drive-by bug fix: `lib.rs::build_graph`
  was reading `ollama.max_tokens` even when `openai.enabled`,
  silently capping openai users at the ollama default. Now reads the
  active backend.
- *Feature gate*: `openai = ["ureq", "async"]` in graphrag-core,
  `openai = ["graphrag-core/openai"]` in graphrag-server. Mirrors the
  existing `ollama` feature. `OpenAIConfig` itself is unconditional so
  user configs round-trip through serde regardless. Without the
  feature, `ChatClient::from_config` falls through to ollama with a
  `tracing::warn!` explaining the build flag.
- *Tests*: 9 inline unit tests covering serde round-trip (incl.
  `extra_body` objects, `Option<max_tokens>`), body-shape construction,
  params override, max_tokens omission when uncapped, and
  `extra_body` merge precedence (set fields win on collisions). Run
  with `cargo test -p graphrag-core --lib --features openai openai::`.
- *README*: documents the `[openai]` chat block alongside `[ollama]`
  and adds an Option B to Quick Start.

**Pre-flight**: maintainer might ask for the openai-compat backend
gated behind a cargo feature flag. Already done — call it out explicitly.

### PR 3 — Embedding service introspection
**Group C.** 35 LOC, single commit.

**Cherry-pick**: `0d303f3`.

**Title**: `Add GET /api/embeddings/stats endpoint`.

### PR 5 — Append-only graph extraction (Group E)
**~140 LOC across 1 commit.**

**Cherry-pick**: row 19 (sha to be filled in once committed).

**Title**: `Add POST /api/graph/append for incremental graph updates`.

Mirrors Microsoft GraphRAG's `graphrag append` semantics: cheap call
agents/cron can fire after a batch of /api/documents to surface
newly-ingested content in queries, without a full rebuild.

**Implementation note** (called out in the commit body and the
endpoint's description): under the hood this currently delegates to
`GraphRAG::build_graph()` because graphrag-core's `incremental`
module isn't yet wired into the runtime pipeline. The LLM-call cache
makes repeat extraction near-free for unchanged chunks, so the cost
scales with new content rather than corpus size — but it's not a
true incremental update yet. A follow-up will route through
`graphrag-core::incremental::add_content`.

Two callable improvements regardless of internal wiring:

- **Fast no-op**: tracks `processed_chunk_count` after every build/
  append; returns immediately with `documentCount: 0` and a clear
  message when the live chunk count hasn't grown. Cron can call
  this every 30 min without paying LLM cost when nothing changed.
- **Distinct semantic for agents**: the MCP tool surface can expose
  `append` as the right tool to call after batch ingest, with
  `build_graph` reserved for cold-start ("graph empty but documents
  exist") or recovery scenarios.

Independent of PR 1–4. Touches only `main.rs`.

### PR 4 — Server UX quick wins (Group D)
**~250 LOC across 1 commit (server quick-wins).**

**Cherry-pick**: row 17 (sha to be filled in once committed).

**Title**: `Server UX: real list_documents, delete by user id, content-hash dedup, last_built_at`.

Four small fixes against issues that surface when an LLM agent
exercises the API end-to-end:

- `GET /api/documents` previously returned `[]` with a "not implemented"
  note. Now pages through Qdrant via scroll API; returns
  `{id, user_id, title, excerpt, added_at}` capped at 256 entries (use
  search to drill in beyond that).
- `POST /api/documents` accepts an optional caller-supplied `id` field
  (camelCase: `id` in the JSON body). Stored in `payload.user_id` so
  callers can later refer to documents by an id they remember.
- `DELETE /api/documents/{id}` resolves the path id as a `user_id`
  first, falling back to treating it as the Qdrant point UUID. Fixes
  the 500 error agents hit when deleting by their own id.
- `POST /api/documents` rejects exact-content duplicates. Computes
  SHA-256 of the sanitized content; if a Qdrant point with the same
  `content_hash` already exists, returns the existing id instead of
  inserting. Mirrors Microsoft GraphRAG's stable-id pattern (v0.5.0+,
  enables upsert-merge).
- `GET /api/graph/stats` now returns `last_built_at` (RFC 3339 timestamp
  of the last successful build, null pre-first-build). Lets agents
  decide whether the graph is fresh enough to query.

Independent of PR 2/3; can land in any order. Touches the same
`models.rs` / `qdrant_store.rs` / `main.rs` files but in
non-conflicting regions.

**Body**: reports the live `EmbeddingService.backend_name()` (openai /
ollama / hash-fallback), dimension, and per-source request counters.
Lets callers (e2e harness, monitoring) verify which path is actually
serving — separately from `/config` GET, which reflects graphrag-core's
internal embedding-generator config (a different layer that's not the
user-facing path). Plain Actix route below `.build()`, same OpenAPI-bypass
dance as `/config`.

Independent of PR 2; can land in any order.

## NOT for upstream

| Item | Why |
|---|---|
| `TODO.md` Phase E (CLI SetConfig parity) | Internal planning. Phase E itself is upstreamable when implemented; the planning doc is not. |
| `TODO.md` Phase F (Claude skill) | Same — skill content lives in user's agent config, not in graphrag-rs. |
| `graphrag-rs-nix/*` | Separate repo; Nix-specific packaging. |
| Dotfiles changes | Personal config. |
| `graphrag-e2e.sh` | Lives in graphrag-rs-nix. |

## API surface changes (vs upstream main)

### New endpoints
- `GET /api/embeddings/stats` — runtime EmbeddingService backend + counters.
- `POST /config` — formerly `POST /api/config`, now reachable (was 404 in
  upstream due to apistos scope shadowing).
- `GET /api/documents` — was a stub returning `[]`; now pages through
  Qdrant and returns real summaries.
- `POST /api/graph/append` — incremental-extraction analogue of
  `/api/graph/build`; cheap no-op when nothing new since last build.

### Changed behavior
- `POST /config` deep-merges over current config; previously partial
  bodies replaced wholesale (resetting unset fields to defaults).
- `EmbeddingService` now picks `openai` backend when `EMBEDDING_BACKEND=openai`
  + the new feature flag. Previously the openai branch existed in code
  but was never reachable.
- `Config.openai.max_tokens` is honored for entity extraction when
  `openai.enabled` (was always reading `Config.ollama.max_tokens`).
- `POST /api/documents` accepts optional `id` field; rejects exact-content
  duplicates by `content_hash`.
- `DELETE /api/documents/{id}` resolves user-supplied id → Qdrant UUID.
- `GET /api/graph/stats` includes `last_built_at`.

### New config fields
- `Config.openai` (always present, defaults to disabled). Fields:
  `enabled`, `base_url`, `chat_model`, `api_key`, `timeout_seconds`,
  `max_retries`, `max_tokens` (`Option<u32>`), `temperature`,
  `enable_caching`, `extra_body`.

### New cargo features
- `graphrag-core`: `openai = ["ureq", "async"]`. Added to `starter` bundle.
- `graphrag-server`: `openai = ["graphrag-core/openai"]`.

### Breaking-but-fix changes
- `/api/config` → `/config`. Was 404 in upstream so no real-world
  consumer is affected; technically still a path change. Maintainer
  may want a `/api/config` alias for symmetry — easy to add.

## Comparison to Microsoft GraphRAG

For maintainer/reviewer context — sets precedent for design choices.

| Concern | Microsoft GraphRAG | This fork | Notes |
|---|---|---|---|
| Multi-backend chat | OpenAI/Azure/Ollama via [llm.factory](https://github.com/microsoft/graphrag) | OpenAI-compat + Ollama via `ChatClient` enum | Microsoft uses a factory; we use a sum type. Both work; sum type is more rust-idiomatic. |
| Multi-backend embeddings | Same factory pattern | `EmbeddingService` reaches OpenAI-compat servers | Parity. |
| Per-request extras | Provider-specific `model_supports_json` etc. | Generic `extra_body: Option<serde_json::Value>` | Ours is more flexible; theirs is more validated. Trade-off: ours leaves validation to the LLM server. |
| Token caps | Provider-default; explicit cap in YAML | `Option<u32>` (None = uncapped) | Microsoft's path is "set a number"; ours adds the explicit "no cap" case for local LLMs. |
| Reasoning models (Qwen3, R1) | No special handling; `<think>` tags leak into responses | `extra_body.chat_template_kwargs.enable_thinking=false` is the recommended user path | This is the case `extra_body` was designed for. |
| Feature gating | All providers always compiled | `ollama` and `openai` are cargo features | Matches upstream graphrag-rs's pre-fork pattern; doesn't follow Microsoft's monorepo style. |

## Pre-flight checklist (run before opening each PR)

- [ ] Each PR branched off latest `upstream/main`, not `openai-compat`.
- [ ] `cargo test --workspace --features <relevant>` passes.
- [ ] `cargo clippy --workspace -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] Three feature combos compile (PR 2): `qdrant,ollama` / `qdrant,openai` / `qdrant,ollama,openai`.
- [ ] README updated where relevant (PR 2).
- [ ] PR body includes back-compat notes for behavior changes.

## Open questions to ask maintainer (in PR 2 body)

- Want a `/api/config` alias retained for back-compat, or is the
  rename acceptable? (Argue: it was 404 anyway.)
- Should the feature gate for openai be opt-in (current) or default-on?
  Mirrors the ollama choice; happy to flip if requested.
- Any preference for `extra_body` structure (`Option<Value>` vs typed
  per-server enum like `extra_body: BackendExtras`)?

## PR filing log

(append rows when filed/updated)

| PR | Date | Title | Status | Notes |
|---|---|---|---|---|
| _none filed yet_ | | | | |
