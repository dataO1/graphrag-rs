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
| 1 | `878dfed` | qdrant-client: opt out of generate-snippets default feature | 1 | **PR A** |
| 2 | `15ad18e` | rename /api/config → /config to fix scope shadowing | 9 | **PR A** |
| 3 | `9b27417` | read OLLAMA_PORT from env | 9 | **PR A** |
| 4 | `5f53117` | merge doubled resource("") in /api/documents (was 405) | 3 | **PR A** |
| 5 | `7b05a06` | deep-merge POST /config bodies over defaults | 39 | **PR A** |
| 6 | `fee7490` | graphrag-server: OpenAI-compat embedding backend | 181 | **PR B** |
| 7 | `9c129bd` | graphrag-core: OpenAI-compat chat + ChatClient enum | 475 | **PR B** |
| 8 | `898f5ae` | build_graph: route LLM gates through Config::chat_enabled() | 35 | **PR B** |
| 9 | `13572bf` | OpenAIConfig.extra_body for per-request body extras | 34 | **PR B** |
| 10 | `cbfe74e` | extraction max_tokens Option + active-backend reads | 45 | **PR B** |
| 11 | `0d303f3` | GET /api/embeddings/stats | 35 | **PR B** (folded — was its own PR pre-consolidation) |
| 12 | `4e33145` | TODO.md Phase E (CLI parity) | 157 | **internal — do NOT PR** |
| 13 | `662c86b` | TODO.md Phase F (Claude skill) | 129 | **internal — do NOT PR** |
| 14 | `ff39c84` | README: document OpenAI-compatible chat backend | 42 | **PR B** |
| 15 | `79e3034` | feature-gate the openai-compat backend | 287 | **PR B** |
| 16 | `0bd7018` | add PR-PLAN.md | 209 | **internal — do NOT PR** |
| 17 | `9135482` | graphrag-server: real list_documents, user-id resolution, content-hash dedup, last_built_at | 344 | **PR C** |
| 18 | `82f271c` | PR-PLAN: motivation + writing-style sections | 84 | **internal — do NOT PR** |
| 19 | `f9bcfac` | graphrag-server: POST /api/graph/append for incremental updates | 164 | **PR C** (folded — was its own PR pre-consolidation) |
| 20 | `c2f19b9` | PR-PLAN: row-19 sha fill | 1 | **internal — do NOT PR** |
| 21 | `4af92ae` | Cargo.lock: register sha2 (added in 9135482) | 1 | **PR C** — only needed by 9135482's sha2 dep; squash into the cherry-pick of 9135482 or carry as a trailing commit |

(Anything added after this point — append rows here when committing to `openai-compat`.)

## PR split

Three PRs, ordered by review burden (smallest first). Each tells one
coherent story with a single audience in mind.

### PR A — Server-side fixes & build cleanup
**~60 LOC across 5 commits.** Boring, fast review, lands first to
de-risk and establish contribution rapport.

**Cherry-pick**: `878dfed 15ad18e 9b27417 5f53117 7b05a06`.

**Title**: `Server-side fixes: scope shadowing, OLLAMA_PORT env, doubled resource, qdrant build, deep-merge /config`.

**Story for the maintainer**: five independent, low-risk fixes
against issues that show up when stock upstream is exercised in a
real deployment. None of them touch the feature surface.

**Things to call out in the body**:
- `15ad18e` is functionally a **fix, not a rename** — upstream
  registers `/api/config` as a plain `web::scope` *after* the apistos
  `/api` scope, which means the apistos scope's prefix-match shadows
  it and every `/api/config` request 404s in stock builds. Renaming
  the plain scope to `/config` makes it reachable for the first time.
  Open to adding a `/api/config` back-compat alias if preferred.
- `7b05a06` is a behavior change for `POST /config`: partial bodies
  now preserve unset fields instead of resetting them to defaults.
  Worth flagging because it changes the wire contract subtly even
  though no caller could have depended on the old shape (the endpoint
  was 404 anyway, see above).

**Audience**: reviewer focused on infra hygiene, not feature work.

### PR B — OpenAI-compatible backend (chat + embeddings + diagnostic)
**~810 LOC across 7 commits** (chat backend, embedding backend,
extra_body, max_tokens rework, feature gate, README, embedding-stats
endpoint, plus `Cargo.lock` registration).

**Cherry-pick**: `fee7490 9c129bd 898f5ae 13572bf cbfe74e ff39c84 79e3034 0d303f3 4af92ae`.

**Title**: `Add OpenAI-compatible chat and embedding backends`.

**Story for the maintainer**: enables running graphrag-rs against any
OpenAI-compatible server (vLLM, llama.cpp `--embedding` /
`llama-server`, OVMS, OpenRouter, OpenAI itself) — both for chat and
for embeddings. Mirrors the existing Ollama path: same feature-gate
pattern, same config shape, same trait surface. Folds in the
`/api/embeddings/stats` diagnostic endpoint that lets users verify
which path is actually serving — useful precisely when standing up a
new local OpenAI-compat backend.

**Sections to include in body**:
- *Chat (graphrag-core)*: new `OpenAIConfig` + `OpenAIClient`
  (ureq/spawn_blocking), `ChatClient` enum dispatcher.
  `OpenAIConfig` sits next to `OllamaConfig` on `Config`;
  `ChatClient::from_config` picks the active one.
- *Embeddings (graphrag-server)*: `EmbeddingService` got an
  OpenAI-compat branch.
- *extra_body*: per-request top-level fields merged into every
  `/chat/completions` body. Lets users pass server-specific knobs
  (`chat_template_kwargs.enable_thinking=false` for Qwen3 on
  llama.cpp, `response_format` for vLLM JSON mode) without growing
  `OpenAIConfig` for every backend quirk. Set fields beat extra_body
  collisions.
- *Token-cap rework*: extraction `max_tokens` is now `Option<usize>`
  — `None` = "no cap", model stops at EOS. Useful for local LLMs
  where token cost is just compute time. Drive-by bug fix:
  `lib.rs::build_graph` was reading `ollama.max_tokens` even when
  `openai.enabled`, silently capping openai users at the ollama
  default. Now reads the active backend.
- *Feature gate*: `openai = ["ureq", "async"]` in graphrag-core,
  `openai = ["graphrag-core/openai"]` in graphrag-server. Mirrors
  the existing `ollama` feature. `OpenAIConfig` itself is
  unconditional so user configs round-trip through serde regardless.
- *Diagnostic endpoint*: `GET /api/embeddings/stats` reports the
  live `EmbeddingService.backend_name()` (openai / ollama /
  hash-fallback), dimension, and per-source request counters. Plain
  Actix route below `.build()`, same OpenAPI-bypass dance as
  `/config`.
- *Tests*: 9 inline unit tests covering serde round-trip (incl.
  `extra_body` objects, `Option<max_tokens>`), body-shape
  construction, params override, max_tokens omission when uncapped,
  and `extra_body` merge precedence. Run with `cargo test -p
  graphrag-core --lib --features openai openai::`.
- *README*: documents the `[openai]` chat block alongside `[ollama]`
  and adds an Option B to Quick Start.

**Audience**: reviewer focused on feature design — chat protocol,
embedding flexibility, feature-gating choices.

**Pre-flight** to call out: feature gate is opt-in, mirroring
`ollama`. Happy to flip to default-on or rename if preferred.

### PR C — Agent-friendly UX
**~500 LOC across 2 commits** — server UX cluster + append endpoint.

**Cherry-pick**: `9135482 f9bcfac`.

**Title**: `Server UX: list_documents, user-id resolution, content-hash dedup, last_built_at, /api/graph/append`.

**Story for the maintainer**: five small fixes that all surface from
the same root cause — what an LLM agent (or any client driving the
API end-to-end without reading source) hits when exercising the
documents/graph endpoints. `last_built_at` and `/api/graph/append`
are the same conceptual unit: the timestamp gives agents/cron the
information they need to decide whether to call append. Filed
together so the contract makes sense as a whole.

**Sections to include in body**:
- *list_documents*: previously returned `[]` with a "not implemented"
  note. Now pages through Qdrant via the scroll API; returns
  `{id, user_id, title, excerpt, added_at}` capped at 256 entries
  with a "use search to drill in beyond that" message when
  truncated.
- *User-supplied IDs*: `POST /api/documents` accepts an optional
  `id` field, stored in `payload.user_id` alongside the
  auto-assigned UUID. `DELETE /api/documents/{id}` resolves the
  path id as a `user_id` first (one Qdrant scroll-with-filter call),
  falls back to treating it as a Qdrant point UUID. Fixes the 500
  agents hit when deleting by their own id.
- *Content-hash dedup*: `POST /api/documents` computes SHA-256 of
  the sanitized content. If a Qdrant point with the same
  `content_hash` already exists, returns the existing id without
  re-embedding. Mirrors Microsoft GraphRAG's stable-id pattern
  (v0.5.0+, enables upsert-merge); no behavioral change for new
  content.
- *last_built_at*: `GET /api/graph/stats` includes the RFC 3339
  timestamp of the last successful build (null pre-first-build).
  Lets agents/cron decide whether the graph is fresh enough relative
  to recent ingests.
- *Append endpoint*: `POST /api/graph/append` mirrors Microsoft's
  `graphrag append` semantics. Tracks `processed_chunk_count` after
  every build/append; returns immediately with `documentCount: 0`
  when the live chunk count hasn't grown. Cron can call it every
  30 min without paying LLM cost when nothing changed.

**Implementation note** (already in the endpoint description and
commit body): the append endpoint currently delegates to
`GraphRAG::build_graph()` because graphrag-core's `incremental`
module isn't yet wired into the runtime pipeline. The LLM-call
cache makes repeat extraction near-free for unchanged chunks, so
the cost scales with new content rather than corpus size — but
it's not a true incremental update yet. A follow-up will route
through `graphrag-core::incremental::add_content`. Worth flagging
to the maintainer in case the framing matters for review.

**Wire-format additions** (back-compat via `#[serde(default)]`):
- `DocumentMetadata.content_hash: Option<String>` (Qdrant payload).
- `DocumentMetadata.user_id: Option<String>` (Qdrant payload).
- `GraphStatsResponse.lastBuiltAt: Option<String>` (response field).
- `DocumentSummary.userId`, `excerpt`, `contentLength` (response
  field reshape; old-shape fields stay around).

**Audience**: reviewer focused on the API contract from a client's
point of view.

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
- [ ] Three feature combos compile (PR B): `qdrant,ollama` / `qdrant,openai` / `qdrant,ollama,openai`.
- [ ] README updated where relevant (PR B).
- [ ] PR body includes back-compat notes for behavior changes.

## Filing order

1. **PR A first** — lowest LOC, lowest risk, fastest review. Tests
   the maintainer's review style and engagement window before sinking
   time into prepping the bigger PRs.
2. **PR C second** — backend-agnostic UX work. Lands cleanly without
   needing PR A merged (no shared file regions).
3. **PR B last** — biggest, headline feature. By the time it lands,
   the maintainer has reviewed two of these PRs and seen the
   contribution style, which makes a 1000-LOC PR less of a cold open.

If the maintainer engages quickly on PR A and seems open to parallel
review, PRs B and C can be filed together. Default is sequential.

## Open questions to ask maintainer (in PR B body)

- Want a `/api/config` alias retained for back-compat, or is the
  rename acceptable? (Argue: it was 404 anyway.) — *applies to PR A*
- Should the feature gate for openai be opt-in (current) or
  default-on? Mirrors the ollama choice; happy to flip if requested.
- Any preference for `extra_body` structure (`Option<Value>` vs
  typed per-server enum like `extra_body: BackendExtras`)?

## PR filing log

(append rows when filed/updated)

| PR | Date | Title | Status | Notes |
|---|---|---|---|---|
| _none filed yet_ | | | | |
