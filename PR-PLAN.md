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
| 19 | `f9bcfac` | graphrag-server: POST /api/graph/append (full-rebuild stub) | 164 | **superseded by 9979a13** — do NOT cherry-pick |
| 20 | `c2f19b9` | PR-PLAN: row-19 sha fill | 1 | **internal — do NOT PR** |
| 21 | `4af92ae` | Cargo.lock: register sha2 (added in 9135482) | 1 | **PR C** — needed by 9135482's sha2 dep |
| 22 | `7f51cdc` | PR-PLAN: 5-PR → 3-PR consolidation | 167 | **internal — do NOT PR** |
| 23 | `82f271c` (`ecddf23`) | PR-PLAN: motivation + writing-style + drafted PR bodies | many | **internal — do NOT PR** |
| 24 | `9979a13` | graphrag-core: real incremental extend_graph; wire /api/graph/append to it | 904 | **PR C** — replaces f9bcfac entirely. Note the openai-compat-branch version uses ChatClient/chat_enabled; the cherry-pick onto pr/agent-ux uses Ollama-only primitives (`27a7c4e`) |
| 25 | `5464fa2` | TODO: Phase G/H — graph rehydration & persistence | 23 | **internal — do NOT PR** |
| 26 | `ca92f86` | graphrag-server: graph-aware /api/query (mode=ask\|explain\|reason) | 313 | **PR D** |
| 27 | `14e7f85` | graphrag-server: hydrate KnowledgeGraph from Qdrant on /config (Phase G) | 168 | **PR D** |
| 28 | `76daa04` | graphrag-server: persist entity graph across restarts (Phase H) | 469 | **PR D** |
| 29 | `c38542b` | graphrag-server: deprecate /api/graph/build for routine use | 16 | **PR D** |
| 30 | `143b9a3` | graphrag-core: KnowledgeGraph::add_entity/add_relationship dedupe by id | 249 | **PR C** (cherry-picked onto `pr/agent-ux` as `c17b5f6`; PR D inherits it via the stack — no separate cherry-pick needed onto `pr/graph-query-and-persistence` since that branch is rebased on `pr/agent-ux`) |
| 31 | `91c2125` | graphrag-server: embed entity/relationship descriptions on persist (Phase H+) | 164 | **PR D** (cherry-picked onto `pr/graph-query-and-persistence` as `cd45672`) |

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

### PR C — Agent-friendly UX + real incremental extend_graph
**~1.2k LOC across 3 commits** — server UX cluster + real
incremental graph extension (`extend_graph`).

**Cherry-pick onto `pr/agent-ux` (already prepared)**: `9135482`
+ a refactored version of `9979a13` (Ollama-only, since PR C must
land independently of PR B's `ChatClient` additions) + Cargo.lock.
The pre-prepared branch has commits `97a0e97 27a7c4e bbfff36`.

**Title**: `Server UX + real incremental graph extension (extend_graph)`

**Story for the maintainer**: five small fixes that all surface from
the same root cause — what an LLM agent (or any client driving the
API end-to-end without reading source) hits when exercising the
documents/graph endpoints. The marquee piece is a real
`extend_graph` that walks only the delta chunks, dedupes entities
by id (mentions of an existing entity extend the existing node's
`mentions` in place), and dedupes relationships by
`(source, target, relation_type)`. `last_built_at` and
`/api/graph/append` are the same conceptual unit: the timestamp
gives agents/cron the information they need to decide whether to
call append. Filed together so the contract makes sense as a whole.

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

### PR D — Graph-aware query API + cross-restart persistence
**~950 LOC across 3 commits.** This is the qualitative jump: until
PR D, `graphrag-server`'s `/api/query` is a thin Qdrant wrapper that
ignores the entity graph it builds, and the LLM-extracted graph
itself is wiped on every restart. PR D fixes both.

**Cherry-pick**: `ca92f86 14e7f85 76daa04 c38542b 91c2125` on top of
`pr/agent-ux` (branch `pr/graph-query-and-persistence`, pushed).

**Stack dependency**: PR D depends on PR C
(`GraphRAG::extend_graph` + `processed_chunks` tracking). The
cherry-pick branch is stacked on `pr/agent-ux`, so PR D should be
filed after PR C lands or rebased onto `upstream/main` once C
merges. The Phase H commit needed one tiny reconcile during the
cherry-pick: the `state.processed_chunk_count` AppState atomic
counter (which exists on `openai-compat` from an earlier,
since-superseded commit) doesn't exist on `pr/agent-ux`, so the
two mirror-into-AppState lines in `build_graph` and `set_config`
were dropped. Functionally a no-op — `graphrag.processed_chunk_count()`
is still readable directly where the values are actually used.

**Title**: `Graph-aware /api/query (ask/explain/reason) + cross-restart persistence`.

**Story for the maintainer**: graphrag-cli already exposes the four
query modes graphrag-core implements — `search` (vector-only), `ask`
(graph-aware + LLM answer), `explain` (`ask` + confidence + sources +
reasoning), `reason` (multi-hop decomposition). graphrag-server's
REST `/api/query` only ever did `search`. This PR ports the other
three modes onto the REST surface, gated by an optional `mode` field
on `QueryRequest` (default `search`, fully back-compatible). Then
because graph-aware modes are pointless if the graph is empty after
every restart, two follow-on commits add chunk hydration + entity/
relationship persistence to Qdrant sidecar collections.

The three commits split cleanly:

- `ca92f86` — `/api/query` learns `mode=ask|explain|reason`. Pure
  feature add over `QueryRequest` and `QueryResponse`. Calls into
  existing `GraphRAG::ask`, `ask_explained`, `ask_with_reasoning`.
- `14e7f85` — Phase G: hydrate `KnowledgeGraph` chunks from Qdrant
  on `POST /config`. Adds `GraphRAG::seed_processed_chunks` public
  helper to graphrag-core; `QdrantStore::list_full_documents` to
  graphrag-server. Without this, `/api/graph/build` after a restart
  only walks chunks added since restart — typically a tiny fraction
  of the corpus.
- `76daa04` — Phase H: persist + restore the LLM-extracted entity +
  relationship graph itself. Two new sidecar Qdrant collections
  (`{coll}-entities` / `{coll}-relationships`) carrying real
  description embeddings (mirrors MS GraphRAG's `description_embedding`
  convention), payload is the serde-serialized `Entity`/`Relationship`.
  Stable point ids via UUID5 over the natural identity. Persist
  runs at the end of every successful build/extend; restore runs at
  the end of `POST /config` after chunk hydration. After this
  commit, `/api/graph/build`'s LLM work genuinely survives restarts
  AND the entity store is searchable by vector — the substrate MS
  uses for `local_search`-style seed-point retrieval, ready for a
  follow-on PR to wire into `/api/query`.

**Why these belong together**: the modes in commit 1 are useful but
fragile without commits 2-3 (a server restart wipes the entity graph
they read from). Splitting commit 1 off as its own PR would land a
feature that silently regresses to vector-only retrieval after every
deploy. Splitting commit 3 off as a "future-PR" stranded persistence
scaffolding without a payoff. The three together tell one story:
"graphrag-server now uses the graph it builds, and keeps it across
restarts."

**Surface area on graphrag-core** (small, additive):
- `GraphRAG::seed_processed_chunks<I: IntoIterator<Item = ChunkId>>(self, chunk_ids)`

**Surface area on graphrag-server**:
- `QueryRequest.mode: Option<QueryMode>` (default `search`)
- `QueryResponse` gains optional `answer`, `confidence`,
  `key_entities`, `reasoning_steps`, `sources` (all
  `skip_serializing_if = "Option::is_none"` so search responses are
  byte-identical to before).
- `QdrantStore::list_full_documents`, `persist_graph`,
  `load_persisted_entities`, `load_persisted_relationships`,
  `clear_graph_collections`, `ensure_graph_collections`.
- New module `graph_persistence` glues `Entity`/`Relationship` to
  the wire envelopes.
- `POST /config` response gains a `hydrated: {documents, chunks,
  skipped, entities, relationships, relationships_skipped_orphan}`
  summary.

**No schema migrations needed** for existing Qdrant collections.
The two sidecar collections are auto-created on first persist;
older deploys without them work fine, just with empty restored
state.

**Audience**: reviewer focused on whether the REST API is using
graphrag-core's actual capabilities. Demos well: `curl /api/query
-d '{"query":"...","mode":"explain"}'` returns a typed answer with
source attribution.

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
- `POST /config` now triggers graph hydration from Qdrant
  (chunks + entities + relationships) so `/api/graph/build` and
  graph-aware query modes see the full corpus on first call after
  a restart, not just chunks ingested since restart.
- `EmbeddingService` now picks `openai` backend when `EMBEDDING_BACKEND=openai`
  + the new feature flag. Previously the openai branch existed in code
  but was never reachable.
- `Config.openai.max_tokens` is honored for entity extraction when
  `openai.enabled` (was always reading `Config.ollama.max_tokens`).
- `POST /api/documents` accepts optional `id` field; rejects exact-content
  duplicates by `content_hash`.
- `DELETE /api/documents/{id}` resolves user-supplied id → Qdrant UUID.
- `POST /api/query` accepts an optional `mode` field
  (`search`/`ask`/`explain`/`reason`); default `search` is
  byte-identical to the previous behavior.
- `POST /api/graph/build` and `POST /api/graph/append` now persist
  the resulting entity + relationship graph to Qdrant sidecar
  collections so the graph survives a restart instead of forcing
  full re-extraction at every boot.
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

## Drafted PR bodies (for user review before filing)

These are the bodies that would go into `gh pr create --body`. Wrap
in a HEREDOC at filing time. All three branches are prepared off
`upstream/main` (`c46e287`) and validated locally; nothing is filed.

### PR A body draft

**Title**: `Server-side fixes: scope shadowing, OLLAMA_PORT env, doubled resource, qdrant build, deep-merge /config`

**Branch**: `pr/server-fixes` (5 commits, ~60 LOC).

```markdown
Five small fixes against issues that surface when graphrag-server is
exercised in a real deployment. None of them touch the feature
surface; they all sit in graphrag-server (plus a one-line workspace
Cargo.toml change). Filing them together because each is too small
to justify its own PR overhead.

## Motivation

Hit each of these standing up a graphrag-rs deployment over a personal
Obsidian vault with Qdrant + Ollama on the server. Submitting upstream
because they all behave the same way for any deployment, not just
mine.

## Goals

- Fix `/api/config` so it's reachable.
- Fix `POST /api/documents` so it doesn't 405.
- Make `OLLAMA_PORT` actually configurable from the environment.
- Unbreak the qdrant-client build under restricted/sandboxed builds.
- Fix `POST /config` partial updates clobbering unset fields.

## Changes

Five commits, each independent:

1. **qdrant-client: opt out of generate-snippets default feature**
   The default `generate-snippets` feature panics in build.rs when
   network access is restricted (Nix sandbox, isolated CI). Disabling
   it doesn't affect runtime functionality — only the snippet
   generator that builds in offline-incompatible ways.

2. **graphrag-server: rename /api/config → /config to fix scope shadowing**
   `App` registers two services on the same prefix:

       .service(scope("/api") ...)                  // apistos
       ...
       .service(web::scope("/api/config") ...)      // plain actix

   actix-web matches services by registration order, prefix-first.
   The apistos `/api` scope claims any `/api/*` request that doesn't
   match an explicit sub-route — there's no `/api/config` inside that
   scope, so requests 404. The plain-actix block below `.build()` is
   dead code as written.

   Three constraints make a fix in place tricky:
   - `/api` can't move past `.build()` (apistos `scope` ≠ plain
     `web::scope`).
   - `config_endpoints::*` handlers can't move into the apistos `/api`
     scope without `#[api_operation]` macros (apistos's typed scope
     requires `PathItemDefinition`).
   - Plain `web::scope` can't be registered before `.build()`.

   Renaming `/api/config` → `/config` sidesteps all three: no overlap
   with `/api`, no shadowing, block stays plain actix post-`.build()`.
   The endpoint becomes reachable for the first time.

   **Back-compat note**: technically a path change. Since the old
   path 404'd in stock builds, no working caller could have depended
   on it. Happy to add a `/api/config` alias if preferred.

3. **graphrag-server: read OLLAMA_PORT from env (was hardcoded 11434)**
   Mirrors the existing `OLLAMA_URL` env var. One-liner.

4. **graphrag-server: merge doubled resource("") in /api/documents scope**
   Two `resource("")` registrations under `/api/documents`, one for
   `GET` (list) and one for `POST` (add). actix-web treats the second
   as duplicate-route and silently drops one — `POST` returned 405.
   Combine into a single `resource("")` with both methods chained.

5. **config: deep-merge POST /config bodies over defaults**
   Previously `POST /config` deserialized the body to `Config`,
   replacing the in-memory config wholesale. Partial bodies (very
   common — set just the openai or just the embeddings section) reset
   every unset field to its default. Now does a recursive deep merge
   over the existing config: only fields explicitly present in the
   body change.

   **Back-compat note**: behavior change for callers that were
   relying on the wholesale-replace semantics. Most callers I'd
   expect to want the new behavior — they were probably re-sending
   the entire config to avoid this — but worth flagging.

## Methodology

- Cherry-picked off `upstream/main` (c46e287).
- `cargo check -p graphrag-server --features qdrant,ollama` clean.
- `cargo test -p graphrag-server --lib` 12/12 pass.
- `cargo fmt --check` clean on touched files. Pre-existing fmt
  warnings in untouched upstream files left alone.
- `cargo clippy` introduces no new warnings; pre-existing warnings
  in upstream untouched.
```

### PR B body draft

**Title**: `Add OpenAI-compatible chat and embedding backends`

**Branch**: `pr/openai-backend` (8 commits, ~810 LOC).

```markdown
Adds an OpenAI-compatible backend on parity with the existing Ollama
path — for both chat (entity extraction, query, gleaning) and
embeddings. Lets users drive graphrag-rs against any server that
speaks `/v1/chat/completions` and `/v1/embeddings`: vLLM, llama.cpp's
`llama-server`, OpenVINO Model Server, OpenRouter, OpenAI itself,
self-hosted text-generation-inference, etc.

Includes the small diagnostic endpoint (`GET /api/embeddings/stats`)
that lets users verify which backend is actually serving — useful
specifically when standing up a new local OpenAI-compat stack.

## Motivation

Local LLM deployments increasingly run on OpenAI-compatible servers
(vLLM, llama.cpp, OVMS, etc.) rather than Ollama, partly because they
support more modern features (tool calling, structured output,
chat-template knobs) and partly because they integrate better with
existing OpenAI client tooling. graphrag-rs's chat path was hardcoded
to Ollama protocol and the embedding side had a parsed-but-unused
"openai" config branch that fell back to hash. This PR closes the gap.

## Goals

- Drive graphrag-rs against any OpenAI-compat chat server with no
  forking, on parity with the Ollama path.
- Same for embeddings.
- Per-request escape hatch for backend-specific knobs without
  growing the config struct for every quirk (motivating case:
  `chat_template_kwargs.enable_thinking=false` for Qwen3 on
  llama.cpp; `response_format` for vLLM JSON mode).
- Make uncapped extraction work for local LLMs (no token billing,
  reasoning models truncate JSON when capped).
- Feature-gate it the same way `ollama` is gated, to keep
  WASM/minimal builds slim.

## Changes

### Chat (graphrag-core)

- New `OpenAIConfig` struct alongside `OllamaConfig` on `Config`.
  Fields: `enabled`, `base_url`, `chat_model`, `api_key`,
  `timeout_seconds`, `max_retries`, `max_tokens` (`Option<u32>`),
  `temperature`, `enable_caching`, `extra_body`.
- New `OpenAIClient` (ureq + `tokio::task::spawn_blocking`, mirrors
  `OllamaClient`'s sync-wrapped-async pattern).
- New `ChatClient` enum dispatcher in `graphrag-core::chat`.
  `ChatClient::from_config` picks the active backend based on
  `openai.enabled` / `ollama.enabled`. Every consumer of chat —
  entity extraction, query planning, gleaning — now takes
  `ChatClient` instead of `OllamaClient` directly.
- `Config::chat_enabled()` helper that returns true when either
  backend is enabled. `build_graph` and friends gate on this so the
  graph build cleanly skips LLM extraction when no chat backend is
  available, instead of failing midway.

### Embeddings (graphrag-server)

- `EmbeddingService` got an OpenAI-compat branch alongside the
  existing Ollama path. Activated by `EMBEDDING_BACKEND=openai` plus
  `OPENAI_URL` / `OPENAI_EMBEDDING_MODEL` / `OPENAI_API_KEY` envs.
  Reqwest-based (already a non-optional dep), so the gate is purely
  a code-path toggle.

### Per-request extras (extra_body)

- New optional `OpenAIConfig.extra_body: Option<serde_json::Value>`
  field. Merged into every `/chat/completions` request body at the
  top level. Existing keys win — set fields on `OpenAIConfig`
  (model, max_tokens, temperature, stop, top_p) take precedence
  over `extra_body` collisions, so users can't accidentally
  overwrite a typed field with a raw JSON blob.
- Motivating cases (in the README):
  - `chat_template_kwargs.enable_thinking=false` for Qwen3 on
    llama.cpp's `--jinja` path (suppresses reasoning output that
    truncates JSON extraction within a token cap).
  - `response_format = { type = "json_object" }` for vLLM JSON mode.

### Token-cap rework

- `LLMEntityExtractor.max_tokens: usize` → `Option<usize>`. `None`
  means "no cap" — `num_predict` / `max_tokens` is omitted from the
  request body, server uses its own default (llama.cpp: -1 /
  unlimited up to ctx). Useful for local LLMs where token cost is
  just compute time and reasoning models truncate JSON when capped.
  Default stays at `Some(1500)`; existing call sites keep working.
- Drive-by bug fix: `lib.rs::build_graph` was reading
  `ollama.max_tokens` even when `openai.enabled` — silently capping
  openai extraction at the ollama default. Now reads the active
  backend's cap.

### Feature gate

- `graphrag-core`: `openai = ["ureq", "async"]`. Added to the
  `starter` bundle. Mirrors the existing `ollama` feature.
- `graphrag-server`: `openai = ["graphrag-core/openai"]`.
- `OpenAIConfig` itself stays unconditional so user configs round-
  trip through serde regardless. Without the feature,
  `ChatClient::from_config` falls through to ollama / None, with a
  `tracing::warn!` explaining how to enable it.

### Diagnostic endpoint

- `GET /api/embeddings/stats` reports the live
  `EmbeddingService.backend_name()` (openai / ollama / hash-fallback),
  dimension, and per-source request counters. Plain Actix route
  below `.build()`, same OpenAPI-bypass dance as `/config` — the
  handler returns `serde_json::Value` rather than an apistos-typed
  struct, which doesn't satisfy `PathItemDefinition`.

  Useful precisely when verifying a new OpenAI-compat backend is
  serving — separately from `/config`, which reflects graphrag-core's
  internal embedding-generator config (a different layer that's not
  the user-facing path).

### Documentation

- README: `[openai]` chat block alongside the existing `[ollama]`
  block. Quick Start gets an "Option B" path showing the
  `EMBEDDING_BACKEND=openai` flow against vLLM.

## Methodology

- Cherry-picked off `upstream/main` (c46e287).
- All three feature combos compile clean: `qdrant,ollama` /
  `qdrant,openai` / `qdrant,ollama,openai`.
- Nine inline unit tests in `graphrag-core/src/openai/mod.rs`:
  - serde round-trip with `extra_body` objects
  - `max_tokens=None` round-trip (skip-on-None)
  - `extra_body=None` round-trip
  - request body shape (model, messages, stream, defaults)
  - params override config (temperature, num_predict)
  - `max_tokens` omitted when uncapped
  - `extra_body` unique-key merge
  - **`extra_body` precedence rule** (set fields beat collisions)
  - defensive: non-object `extra_body` silently dropped
- Run with: `cargo test -p graphrag-core --lib --features openai openai::`. 9/9 pass.
- `cargo fmt --check` clean on touched files. Pre-existing fmt
  warnings in untouched upstream files left alone.

## Open questions

- `extra_body` is `Option<serde_json::Value>` for maximum flexibility.
  Considered a typed enum (e.g., `BackendExtras::LlamaCpp { ... } |
  BackendExtras::Vllm { ... }`) but settled on raw Value because the
  server-specific knobs change faster than this codebase's release
  cadence. Open to switching if you'd rather have validation.
- The feature gate is opt-in (mirrors `ollama`). Happy to flip to
  default-on or add to `default = [...]`.
- `/api/embeddings/stats` is folded in here because its primary use
  case is diagnosing the new OpenAI embedding backend. Happy to
  split into a follow-up PR if you'd prefer.
```

### PR C body draft

**Title**: `Server UX + real incremental graph extension (extend_graph)`

**Branch**: `pr/agent-ux` (3 commits, ~1.2k LOC). Split by concern;
happy to squash on merge.

- `97a0e97` — server UX (list_documents, dedup, user-id, last_built_at)
- `27a7c4e` — graphrag-core `extend_graph` + 4 inline tests + handler wiring
- `bbfff36` — Cargo.lock for the new sha2 dep

```markdown
Five small UX fixes plus a real incremental graph-extension API,
all clustered around the same root: what an LLM agent (or any
client driving the API end-to-end without reading source) hits
when actually exercising graphrag-server.

## Motivation

Driving graphrag-server from an MCP-bridged agent (Claude Code,
opencode, crush) over a personal knowledge base, several rough edges
showed up consistently:

- `list_documents` returns `[]` with a "not implemented" note —
  the agent can't discover what's indexed.
- Deleting by the id passed at ingest returns 500 — only the
  server-assigned UUID works, but the agent doesn't keep that.
- Ingesting the same content twice produces two Qdrant points with
  slightly different similarity scores in queries.
- `graph_stats` doesn't say when the graph was built, so agents
  can't tell if it's stale relative to recent ingests.
- Triggering entity extraction means a full `build_graph` even
  after one new ingest — Microsoft GraphRAG has the
  `graphrag append` pattern for exactly this case, but
  graphrag-server has no analogue.

These cluster naturally — `last_built_at` and `/api/graph/append`
are the same conceptual unit (the timestamp gives clients the
signal to call append). Filed together so the contract makes sense
as a whole.

## Goals

- Make `list_documents` actually return documents.
- Let clients refer to documents by the id they supplied at ingest.
- Stop duplicate-content ingest from creating duplicate vectors.
- Surface graph-build freshness through the stats endpoint.
- Add a **real** incremental extend endpoint — only walks the
  delta chunks since the last build/extend, dedupes entities by
  id, merges relationships. Not a wrapper around build_graph.

## Changes

### list_documents (was a stub)

`GET /api/documents` previously returned
`{documents: [], total: N, note: "Full document listing from Qdrant not implemented yet"}`.
Now pages through the collection via Qdrant's scroll API and returns
real summaries `{id, userId, title, excerpt (160 chars), addedAt}`.
Capped at 256 entries with a "use search to drill in beyond that"
note when truncated.

### User-supplied IDs

`POST /api/documents` accepts an optional `id` JSON field. Stored in
the Qdrant payload's new `user_id` field, alongside the UUID Qdrant
requires for the point id itself.

`DELETE /api/documents/{id}` resolves the path id as a `user_id`
first (one Qdrant scroll-with-filter call), falls back to treating
it as a UUID. Fixes the 500 callers hit when trying to delete by an
id they remembered handing in at ingest.

### Content-hash dedup

`POST /api/documents` computes SHA-256 of the sanitized content
before embedding. If a Qdrant point with the same `content_hash`
already exists, returns the existing id without re-embedding.
Mirrors Microsoft GraphRAG's stable-id pattern (v0.5.0+, enables
upsert-merge).

### last_built_at

`GET /api/graph/stats` includes `lastBuiltAt` (RFC 3339 timestamp
of the last successful `/api/graph/build`, null pre-first-build).
Set on every successful build/append.

### Real incremental graph extension

New `pub async fn GraphRAG::extend_graph(&mut self) -> Result<ExtendSummary>`
in graphrag-core. Mirrors Microsoft GraphRAG's `graphrag append`
semantics, properly:

- Tracks `processed_chunks: HashSet<ChunkId>` on `GraphRAG`.
  Populated at the end of `build_graph` (every chunk) and at the
  end of `extend_graph` (only the delta).
- `extend_graph` filters `knowledge_graph.chunks()` against
  `processed_chunks` and runs the same extractor `build_graph`
  would pick (gleaning / LLM single-pass / GLiNER /
  pattern-based) over **only the delta**.
- **Dedupes entities by id** before adding to the graph. If a
  delta chunk re-mentions an entity that already exists, the
  existing entity's `mentions` are extended in place (compared
  by `(chunk_id, start_offset)`); confidence is bumped to the
  max. No duplicate node. Mirrors Microsoft's stable-id pattern.
- **Dedupes relationships** by `(source, target, relation_type)`
  before adding. Skips edges already present.
- Returns `ExtendSummary { chunks_processed, new_entities,
  new_relationships, mentions_merged, total_entities,
  total_relationships }` so callers can tell whether the extend
  enriched existing nodes vs added new ones — useful for
  downstream community/PageRank recompute decisions, mirroring
  Microsoft's append heuristic.
- `clear_processed_chunks()` resets the tracking set so the next
  `extend_graph` re-walks every chunk. Useful after a config
  change (entity_types, prompts) where you want to re-extract
  without wiping the graph first.

`POST /api/graph/append` is a thin wrapper around `extend_graph`:
fast no-op when no delta, real incremental work when there is.

### KnowledgeGraph::add_entity / add_relationship dedupe by id

While `extend_graph` was working around the duplicate-node bug
via the private `merge_entity` / `merge_relationship` helpers, the
canonical `KnowledgeGraph::add_entity` / `add_relationship` methods
still appended a fresh petgraph node every time — so `build_graph`
(and any direct library user) still produced duplicate-id nodes
with orphaned mentions. This commit promotes the dedup logic
from the private helpers into the canonical public API, so the
two extraction paths agree on graph state.

Before:
```rust
pub fn add_entity(&mut self, entity: Entity) -> Result<NodeIndex> {
    let entity_id = entity.id.clone();
    let node_index = self.graph.add_node(entity);
    self.entity_index.insert(entity_id, node_index); // overwrites
    Ok(node_index)
}
```

After: checks `entity_index` first; if the id already exists, merges
mentions in place (dedupe by `(chunk_id, start_offset)`), bumps
confidence to `max(existing, new)`, and takes the new embedding
only if the existing was None. Returns the existing `NodeIndex`.

`add_relationship` similarly scans outgoing edges of the source
node for an identical `(target, relation_type)` pair and silently
returns `Ok(())` if found.

Symptom this fixes (from the maintainer's user perspective):
calling `build_graph` over a corpus where the LLM extracts the
same entity from 3 chunks previously produced 3 petgraph nodes.
`graph.entities().count()` returned 3; `entity_index` only mapped
the id to the most-recent node; the other 2 nodes' mentions were
unreachable. Any persistence layer keying on `entity.id` would
silently dedupe on the way out, hiding the in-memory bloat.

API surface impact: `add_entity` returns `Result<NodeIndex>` as
before; on a dedup-hit it returns the existing NodeIndex instead
of allocating a new one. No caller in the tree retains NodeIndex
across calls in a way that would break.

The private `merge_entity` / `merge_relationship` helpers in
extend_graph become thin wrappers — they only count metrics now,
since the underlying dedup happens inside the canonical add path.

Four new inline tests in `core::dedup_tests`:
- `add_entity_dedupes_by_id_and_merges_mentions`
- `add_relationship_dedupes_by_source_target_relation_type`
- `add_entity_takes_max_confidence_and_first_embedding`
- `add_relationship_returns_ok_on_dedup_not_err`

The four existing `extend_graph_*` tests still pass — the public
dedup matches what the private helpers were doing.

So with this PR: `build_graph` and `extend_graph` both produce
the same dedupe-correct in-memory graph, removing a long-standing
silent-correctness gap.

## Wire-format additions (back-compat)

All new fields use `#[serde(default, skip_serializing_if = "Option::is_none")]`
so older payloads parse cleanly and older clients see no change:

- `qdrant_store::DocumentMetadata.content_hash: Option<String>`.
- `qdrant_store::DocumentMetadata.user_id: Option<String>`.
- `models::GraphStatsResponse.lastBuiltAt: Option<String>`.
- `models::DocumentSummary.userId`, `excerpt`, `contentLength`
  (the Qdrant backend uses `excerpt`; the in-memory backend uses
  `contentLength`).
- `models::AddDocumentRequest.id: Option<String>` (request field).
- `GraphRAG::ExtendSummary` (new public type, returned by
  `extend_graph`).
- `GraphRAG::processed_chunk_count() -> usize`,
  `GraphRAG::clear_processed_chunks()` (new public methods).

## Methodology

- Cherry-picked off `upstream/main` (c46e287). Three commits,
  one per concern.
- `cargo check -p graphrag-server --features qdrant,ollama` clean.
- `cargo test -p graphrag-server --lib --features qdrant,ollama`
  12/12 pass.
- All three relevant feature combos compile clean:
  default features, `--features gliner`, and (for graphrag-server)
  `qdrant,ollama`.
- **Four new inline `extend_graph` tests** in
  `graphrag-core/src/lib.rs`, all using the pattern-based
  extractor (no LLM dependency, deterministic):
  - `extend_graph_no_new_chunks_is_a_fast_noop` — extend after a
    fresh build returns chunks_processed=0.
  - `extend_graph_processes_only_delta_chunks` — second doc gets
    a chunks_processed=1 extend (not 2).
  - `extend_graph_dedupes_entities_by_id` — entity re-mentioned
    in a delta chunk does NOT create a duplicate node;
    mentions are merged in place.
  - `extend_graph_after_clear_processed_re_extracts_everything`
    — `clear_processed_chunks()` resets the tracking set.
  Run with: `cargo test -p graphrag-core --lib extend_graph`.
  4/4 pass.
- **GLiNER incremental path is wired but untested**, matching
  build_graph's GLiNER branch which is also untested upstream
  (GLiNER needs a downloaded ONNX model and produces non-
  deterministic output, so neither extractor side has a
  deterministic test). Visual parity with build_graph's GLiNER
  branch + the shared `merge_entity` / `merge_relationship`
  helpers (which the four pattern-based tests exercise) are the
  evidence base. Marked **untested** in the
  `extend_with_gliner` doc comment.
- `cargo fmt --check` clean on touched files. Pre-existing fmt
  warnings in untouched upstream files left alone.
- New `sha2` dep is already a workspace dep used elsewhere; one
  Cargo.lock line added.

## Open questions

- Considered making `delete_document`'s user-id fallback configurable
  (some deployments might want strict UUID-only). Settled on
  always-try-user-id-first because it's the only reasonable default
  for clients that handed us an id. Open to making it opt-in.
- The three commits are split by concern (server UX / core
  incremental / Cargo.lock). Happy to squash on merge.
```

### PR D body draft

**Title**: `Graph-aware /api/query (ask/explain/reason) + cross-restart persistence`

**Branch**: `pr/graph-query-and-persistence` (5 commits, ~1.1k LOC).
Split by concern; happy to squash on merge.

- `ca92f86` — graphrag-server: graph-aware `/api/query`
- `14e7f85` — graphrag-server: hydrate `KnowledgeGraph` from Qdrant on `/config` (Phase G)
- `76daa04` — graphrag-server: persist entity graph across restarts (Phase H)
- `c38542b` — graphrag-server: deprecate `/api/graph/build` for routine use
- `91c2125` — graphrag-server: embed entity/relationship descriptions on persist (Phase H+)

```markdown
graphrag-cli already exposes the four query modes graphrag-core
implements: `search` (vector-only), `ask` (graph-aware + LLM
answer), `explain` (`ask` + confidence + sources + reasoning),
`reason` (multi-hop decomposition). graphrag-server's REST
`/api/query` only ever did `search`. This PR ports the other
three modes onto the REST surface, then makes the LLM-extracted
graph survive restarts so those modes have something to ground
against.

## Motivation

Driving graphrag-server through an MCP-bridged agent (Claude Code,
opencode, crush) the gap between the CLI surface and the REST
surface keeps showing up:

- `graphrag-cli /mode explain "..."` returns a typed answer with
  confidence, source attribution, and reasoning steps.
- `graphrag-server POST /api/query` returns vector-search excerpts
  and nothing else.

This is a server-implementation gap, not a graphrag-rs limitation —
the core has `GraphRAG::ask`, `ask_explained`, `ask_with_reasoning`
public APIs. The server just doesn't call them.

The second half of the gap is restart survival. Even if you wire
the modes through, they're useless when the server's in-memory
entity graph is empty — and today it always is, because nothing
persists the LLM-extracted graph. Every restart wipes ~minutes of
LLM extraction work. Graph-aware retrieval that has nothing to
retrieve from is worse than honest vector search.

So PR D is one coherent unit: REST `/api/query` learns the modes,
*and* the graph survives the server lifecycle.

## Goals

- `POST /api/query` accepts an optional `mode` field —
  `search` (default) / `ask` / `explain` / `reason`. Search stays
  byte-identical for back-compat.
- Graph-aware modes call into the existing graphrag-core
  `GraphRAG::ask*` APIs, surface the results through `QueryResponse`
  optional fields (`answer`, `confidence`, `key_entities`,
  `reasoning_steps`, `sources`).
- `KnowledgeGraph` chunks rehydrate from Qdrant on `POST /config`
  so `/api/graph/build` and the new modes see the full corpus.
- LLM-extracted entities + relationships persist to Qdrant sidecar
  collections at the end of every successful build/extend, restore
  on `POST /config`. Restart no longer wipes the graph.

## Changes

### Graph-aware /api/query (commit `ca92f86`)

`QueryRequest`:

```rust
pub struct QueryRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<QueryMode>,  // search | ask | explain | reason
}
```

`QueryResponse` gains optional fields populated per-mode:

- `answer: Option<String>` — `ask` / `explain` / `reason`
- `confidence: Option<f32>` — `explain`
- `key_entities: Option<Vec<String>>` — `explain`
- `reasoning_steps: Option<Vec<ReasoningStepDto>>` — `explain`
- `sources: Option<Vec<SourceReferenceDto>>` — `explain`
- `mode: String` — always set, echoes the mode used

Every field except `mode` is `skip_serializing_if = "Option::is_none"`,
so a `mode=search` (or no-mode) response is byte-identical to the
pre-PR shape. Existing clients break nothing.

Implementation: a single `graph_aware_query` helper handles
`ask`/`explain`/`reason`. It runs vector search in parallel so the
caller still gets `results` (source excerpts) alongside the LLM
answer — useful for UI rendering even when the user only wanted the
synthesized response.

`Mode != search` requires a configured chat backend; without one,
the handler returns 400 with a hint to `POST /config` first.

### Phase G — chunk hydration from Qdrant on /config (commit `14e7f85`)

graphrag-core gains one public helper:

```rust
impl GraphRAG {
    pub fn seed_processed_chunks<I: IntoIterator<Item = ChunkId>>(
        &mut self,
        chunk_ids: I,
    );
}
```

graphrag-server's `POST /config` handler now scrolls the Qdrant
collection, re-chunks each document through the configured
`TextProcessor`, pushes the chunks into the in-memory
`KnowledgeGraph`, and seeds `processed_chunks` with their ids.

Why: before this commit, after a server restart, the in-memory
chunk index started empty. `/api/graph/build` only walked chunks
ingested *since* restart — typically a tiny fraction of the corpus.
`/api/graph/append`'s no-op fast path was a lie: it claimed
"5 of 5 processed" while Qdrant held 45 docs that had never been
extracted.

`POST /config` response gains a `hydrated: {documents, chunks,
skipped, ...}` summary so deploys can verify hydration ran.

New API on `QdrantStore`:
```rust
pub async fn list_full_documents(&self, limit: u32)
    -> Result<Vec<(String, DocumentMetadata)>>;
```
Like `list_documents` but returns full payloads so callers can
rechunk for hydration.

### Phase H — persist entity graph across restarts (commit `76daa04`)

Two new sidecar Qdrant collections, suffixed off the main collection:
`{coll}-entities` and `{coll}-relationships`. One Qdrant point per
entity / relationship; payload is the serde-serialized
graphrag-core `Entity` / `Relationship`. Stable point ids:
UUID5 over the entity id (entities) or
`source|relation_type|target` (relationships).

Vectors carry real description embeddings, mirroring Microsoft
GraphRAG's `description_embedding` convention:
- Entities are embedded as `"{name} ({entity_type})"`.
- Relationships are embedded as `"{source_name} {relation_type} {target_name}"`.
- Vector dim matches the document collection's dim, so entity
  searches and document searches are in the same embedding space.
- Reuses `Entity.embedding` / `Relationship.embedding` if the
  extractor already populated it (today's extractors don't, but a
  future extractor PR could without changing this code path).
  Otherwise batches through the same `EmbeddingService` the
  document path uses.

This makes the sidecars a real vector index over the entity graph,
not just a key-value store — the substrate MS uses to power
`local_search`. A follow-on PR can wire entity-vector-search into
`/api/query`'s graph-aware modes for seed-point retrieval.

Wiring:
- `POST /api/graph/build` → after success, persist entire current
  graph (clear-and-repopulate so in-memory deletions propagate).
- `POST /api/graph/append` → same; the no-op fast path skips the
  persist call since the graph is unchanged.
- `POST /config` → after Phase G chunk hydration, restore entities
  first (so relationships have endpoints) and then relationships.
  Orphan relationship rows (whose source/target weren't restored)
  are logged and skipped, not fatal.

Schema versioning: each persisted row carries a `schema_version: u32`
field (currently `1`) for future incompatible migrations.

New API on `QdrantStore`:
```rust
pub async fn persist_graph(
    &self,
    entities: Vec<PersistedEntity>,
    relationships: Vec<PersistedRelationship>,
) -> Result<(usize, usize)>;
pub async fn load_persisted_entities(&self) -> Result<Vec<PersistedEntity>>;
pub async fn load_persisted_relationships(&self) -> Result<Vec<PersistedRelationship>>;
pub async fn ensure_graph_collections(&self) -> Result<()>;  // idempotent
pub async fn clear_graph_collections(&self) -> Result<()>;  // delete + recreate
pub fn entities_collection(&self) -> String;
pub fn relationships_collection(&self) -> String;
```

A new module `graphrag-server/src/graph_persistence.rs` glues
graphrag-core's `Entity`/`Relationship` to the wire envelopes.

## Methodology

- Cherry-picked off `upstream/main` (c46e287). Three commits, one
  per concern (modes / Phase G / Phase H).
- `cargo check -p graphrag-server --features qdrant` clean.
- 12 pre-existing test failures in graphrag-core
  (`normalize_name`, `boundary_detection`, etc.) are unrelated;
  they fail on `upstream/main` too.
- e2e suite (in graphrag-rs-nix) has new Tests 11 and 12 covering
  `mode=ask`/`mode=explain` plus the entity/relationship sidecar
  collections; tests pass against a real local LLM (Qwen3.6 27B
  via vLLM) and a real Qdrant.
- Workspace dep change: `uuid` gains the `v5` feature for
  deterministic point ids.

## Open questions

- **Entity / relationship description embeddings — done in this PR.**
  Earlier draft of this PR persisted with 1-D placeholder vectors;
  Microsoft GraphRAG embeds entity and relationship descriptions
  and uses those embeddings as the seed-point engine for its
  `local_search` mode. This PR follows MS's shape: each entity is
  embedded as `"{name} ({entity_type})"`, each relationship as
  `"{source_name} {relation_type} {target_name}"`, vectors live in
  the same dim as the document collection (so entity searches and
  document searches are directly comparable), reuses `Entity.embedding`
  / `Relationship.embedding` if the extractor populated it (today's
  extractors don't, but a future extractor PR could without
  changing this code path). One batch embed call per build/append
  for entities, one for relationships — not N×M. Unlocks future
  graph-aware retrieval work without a follow-up persistence
  refactor.
- **Hierarchical / community persistence is deliberately out of
  scope.** Restore order is "entities first, then relationships"
  because `add_relationship` validates endpoint existence —
  this is per-implementation, not a parallel to MS architecture
  (MS sidesteps the question with a batch parquet pipeline that
  doesn't have a long-running daemon to rehydrate). graphrag-core
  has a `relationship_hierarchy: Option<RelationshipHierarchy>`
  field on `KnowledgeGraph` and a graph-analytics module with
  Leiden community detection — both currently unpopulated by any
  code path. The closer mirror to MS would be Leiden communities
  + LLM-generated community reports persisted as a third sidecar;
  that's a follow-on PR (it requires the chat backend and a
  community-summary prompt). This PR establishes the persistence
  substrate so that work fits in cleanly later.
- **Persistence is "wipe and repopulate" on every build.** Simple
  but writes O(|entities| + |relationships|) per build. For a
  100K-entity graph this is bounded; not a blocker. A future
  delta-upsert path could track per-entity dirtiness (mirror of
  the `processed_chunks` set used by `extend_graph`) and write
  only changed rows. Out of scope here.
- Four commits split by concern (modes / chunk hydration / graph
  persistence / build_graph deprecation). Happy to squash on merge
  if that reads better.
```

## PR filing log

(append rows when filed/updated)

| PR | Date | Title | Status | Notes |
|---|---|---|---|---|
| _none filed yet_ | | | | |
