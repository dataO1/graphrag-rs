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

### Changed behavior
- `POST /config` deep-merges over current config; previously partial
  bodies replaced wholesale (resetting unset fields to defaults).
- `EmbeddingService` now picks `openai` backend when `EMBEDDING_BACKEND=openai`
  + the new feature flag. Previously the openai branch existed in code
  but was never reachable.
- `Config.openai.max_tokens` is honored for entity extraction when
  `openai.enabled` (was always reading `Config.ollama.max_tokens`).

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
