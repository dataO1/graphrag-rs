Phase D: Python Bindings (✅ COMPLETED)
✅ Created graphrag-py crate with uv + maturin
✅ Exposed PyGraphRAG struct via PyO3 0.21
✅ Implemented async methods: ask, ask_with_reasoning, add_document_from_text, build_graph
✅ Added comprehensive test suite (15 tests, 12 passing, 3 skipped)
✅ Created documentation and examples
⬜ Publish to PyPI (optional, ready when needed)

----

 Implementation Plan - Phase D: Python Bindings
Goal
Create Python bindings for the graphrag-core Rust crate to allow Python developers to use the GraphRAG system effortlessly. We will use uv for Python project management and maturin + pyo3 for building the extension module.

Technology Stack
Manager: uv (by Astral)
Build Backend: maturin
Bindings: pyo3
Async Runtime: tokio (handled via pyo3-asyncio or pyo3 v0.21+ async support)
User Review Required
IMPORTANT

This requires uv to be installed on the system. The python package will be named graphrag_rs (or graphrag_py pending preference, defaulting to graphrag_rs for consistency).

Proposed Changes
Directory Structure
We will create a new directory graphrag-py (or similar) alongside graphrag-core. It can be part of the Cargo workspace or standalone. For simplicity in bindings, often a standalone or workspace member crates/graphrag-py is good. Given the current structure, we'll put it in the root as graphrag-py.

1. Project Initialization
Run uv init --lib graphrag-py
Modify pyproject.toml to use build-system = { requires = ["maturin>=1.0"], build-backend = "maturin" }
2. Rust Dependencies (graphrag-py/Cargo.toml)
[package]
name = "graphrag-py"
version = "0.1.0"
edition = "2021"
[lib]
name = "graphrag_rs"
crate-type = ["cdylib"]
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module", "abi3-py39"] }
graphrag-core = { path = "../graphrag-core" }
tokio = { version = "1", features = ["full"] }
3. Binding Implementation (graphrag-py/src/lib.rs)
PyGraphRAG Class: Wrapper around std::sync::Arc<tokio::sync::Mutex<GraphRAG>> (or similar thread-safe wrapper).
__init__: Initialize the system (default local or custom).
ask
: Async method exposed to Python.
ask_with_reasoning
: Async method exposed to Python.
4. Verification Plan
Use uv add --dev pytest
Create tests/test_binding.py:
import pytest
from graphrag_rs import GraphRAG
@pytest.mark.asyncio
async def test_ask():
    rag = GraphRAG.default_local()
    answer = await rag.ask("Hello?")
    assert isinstance(answer, str)
Run uv run maturin develop then uv run pytest.

----

Phase E: CLI parity with openai-compat chat backend (⬜ NOT STARTED)
⬜ Add openai chat block to SetConfig schema
⬜ Wire SetConfig::to_graphrag_config() to populate config.openai
⬜ JSON5/TOML/JSON5 schema doc + at least one config template
⬜ Setup wizard support (graphrag-cli setup --backend openai)
⬜ Update CLI README and ../README.md "Configuration" section
⬜ Cross-format round-trip test (parse → to_graphrag_config → re-emit)

----

Implementation Plan — Phase E: CLI parity with the openai-compat chat backend

Goal

graphrag-cli is a standalone consumer of graphrag-core: it loads a
file via --config FILE, parses it as graphrag_core::config::SetConfig,
and projects to graphrag_core::Config via SetConfig::to_graphrag_config().
The openai-compat fork (Phase: openai-compat branch) added an OpenAI-
compat chat backend wired into the runtime Config (the `openai` field)
and into ChatClient dispatch. The server picks it up because callers
POST a runtime Config straight to /config — bypassing SetConfig.

The CLI has no /config endpoint analogue: every entry point loads via
SetConfig. SetConfig today exposes `embeddings.backend = "openai"` for
the embedding side but has no chat-side openai block; to_graphrag_config()
constructs only `config.ollama` and never touches `config.openai`. So
even a hand-written CLI config can't tell the CLI to talk to llama-server
/ vLLM / OpenAI for chat.

This phase closes the gap by extending SetConfig itself, not by
side-loading a runtime Config (a side-load would be a layering shortcut
that future SetConfig changes would silently break). The result is one
config schema that supports both backends symmetrically and a CLI that
can drive any OpenAI-compat chat server out of the box.

Non-Goals

- No refactor of graphrag-cli into a thin client of graphrag-server.
  CLI keeps its embedded GraphRAG instance.
- No deprecation of the ollama block. Both blocks coexist; routing is
  decided by the existing `openai.enabled` / `ollama.enabled` flags
  already honored by ChatClient::from_config.
- No change to /config POST or apply-config flow. Server path stays as
  is.

Proposed Changes

1. Extend SetConfig (graphrag-core/src/config/setconfig.rs)

   Add a new `openai` section parallel to the existing `ollama` block.
   Mirror the OpenAIConfig fields graphrag-core::openai::OpenAIConfig
   already exposes so that to_graphrag_config() is a straight copy:

       struct SetConfigOpenAI {
           enabled: bool,
           base_url: String,            // default: openai.com/v1
           chat_model: String,          // default: gpt-4o-mini
           api_key: String,             // env-var fallback handled in to_*
           timeout_seconds: u64,        // default: 60
           max_retries: u32,            // default: 3
           max_tokens: Option<u32>,     // None = uncapped (Phase pre-E)
           temperature: Option<f32>,
           enable_caching: bool,
           extra_body: Option<serde_json::Value>,  // Phase pre-E
       }

   Pick defaults that match graphrag-core::openai::OpenAIConfig::default()
   so a user setting `openai.enabled = true` with nothing else gets a
   sensible config aimed at OpenAI proper. Document in the schema doc
   (graphrag-core/src/config/schema/) what each field does and what
   "uncapped" means for max_tokens.

2. Wire it through to_graphrag_config()

   Append to the existing block at the bottom of to_graphrag_config()
   (lines around 1876–1891, where ollama is mapped). Construct
   `config.openai = OpenAIConfig { ... }` from `self.openai`. Do NOT
   gate this on `self.openai.enabled` — copy the values regardless,
   so /config GET round-trips correctly. The `enabled` flag is what
   ChatClient::from_config reads to pick the backend.

   Also: when `self.openai.enabled` is true and `self.ollama.enabled`
   is false (or unset), surface that distinction clearly in the
   `tracing::info!` lines at the end of CLI's load_config so users
   can confirm which backend they're driving.

3. Schema, templates, validation

   - Add the new section to graphrag-core/src/config/schema/graphrag-config.schema.json
     so JSON5 autocomplete keeps working.
   - Add a template at config/templates/semantic_openai.graphrag.json5
     (or extend semantic.graphrag.json5 with a commented-out openai
     block) demonstrating both modes.
   - graphrag-core/src/config/validation.rs: warn if both
     ollama.enabled and openai.enabled are true (ChatClient::from_config
     prefers openai today; users probably don't want a silent precedence
     rule). Hard-error if neither is enabled and entity extraction is
     configured to require an LLM.

4. Setup wizard (graphrag-cli/src/handlers/setup.rs or the equivalent
   module under `Commands::Setup`)

   Today: `graphrag-cli setup` walks the user through an Ollama-only
   path. Add a "Which chat backend?" prompt with options
   [ollama, openai-compat (vLLM/llama-server/OpenAI/...)]. The openai
   path should ask for base_url, chat_model, api_key (env-var hint),
   and optional extra_body (skip for v1 — wizard sticks to common knobs).
   Emit either ollama.enabled = true xor openai.enabled = true.

5. Documentation

   - graphrag-cli/README.md "Configuration File" section gets an openai
     block example.
   - Top-level README.md "Basic Configuration" section: replace the
     ollama-only TOML snippet with a tabbed/two-column "Ollama vs
     OpenAI-compat" example so newcomers see both.
   - Add a one-paragraph note on env-var fallback for api_key
     (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).

6. Verification Plan

   - Unit: parse all of {algorithmic, semantic, semantic_openai,
     hybrid, narrative_fiction, legal_documents}.graphrag.json5
     templates → SetConfig → to_graphrag_config(); assert
     config.openai.enabled and config.openai.base_url match the file.
   - Round-trip: SetConfig → to_graphrag_config() → serialize back to
     SetConfig (we don't have this today; add a Config::to_set_config
     helper or skip if the cost is high — call it out as a follow-up).
   - E2E: extend tests/e2e/configs with one openai-targeted config
     (point at a local llama-server or stub) and confirm /load + /query
     behave.
   - Migration: run an existing pure-ollama config through the new
     parser; assert no fields move and no warnings trip.

7. Upstream PR

   This change benefits any user pointing graphrag-cli at vLLM,
   llama-server, OpenRouter, or OpenAI itself — not just our setup.
   Open the PR against `automataIA/graphrag-rs:main` once the
   openai-compat branch's chat backend lands upstream (or in parallel,
   referencing the existing OpenAIConfig in graphrag-core). Frame the
   PR as "schema parity with the existing chat-side openai backend",
   not "support our use case".

Out of Scope (separate phases)

- Auto-discovery of the config file (XDG path, env var). The CLI
  staying explicit about --config is a deliberate design choice and
  any default-path behavior should be its own discussion.
- Wrapper script that pre-fills --config from XDG. That's packaging,
  not graphrag-rs proper.
- Refactoring graphrag-cli into a graphrag-server REST client. That's
  a much larger conversation and orthogonal to schema parity.

----

Phase F: Claude Code skill — "graphrag" (⬜ NOT STARTED)
⬜ Author SKILL.md describing WHEN to consult the graph
⬜ Concrete example questions / triggers (Obsidian-vault context)
⬜ Decision tree: "should I query the graph for this?"
⬜ Anti-patterns: don't query for code questions, ephemeral state, etc.
⬜ Pair with the existing graphrag-mcp tools (query, list_documents,
   graph_stats) — skill describes activation, MCP provides the call
⬜ Equivalent system-prompt addendum for opencode/crush (non-Claude
   clients can't load skills; render the same activation guidance into
   their custom_instructions)

----

Implementation Plan — Phase F: Claude Code "graphrag" skill

Goal

graphrag-mcp exposes the *capability* (six tools wired to the REST
server: query, graph_stats, list_documents, add_document,
delete_document, build_graph) to any MCP-aware agent. What's missing
is *activation* — a description of when an agent should reach for the
graph instead of just answering from its own context. Tool schema
descriptions in MCP are one-liners; they're enough for the model to
pick a tool when it's already decided to use one, but not enough to
prompt the decision in the first place.

A Claude Code skill at ~/.claude/skills/graphrag/SKILL.md provides
that activation layer. It's loaded into context only when the
description matches the user's request, so the cost is zero on
unrelated turns and high signal on relevant ones.

Non-Goals

- Replace the MCP server. Skill describes; MCP executes. Removing the
  MCP server would break opencode and crush, which can't load skills.
- Bundle the skill into the graphrag-rs repo. Skills are agent-specific
  artifacts and live with the user's agent config (dotfiles), not with
  the underlying tool repo. Tracked here only because the design
  choices (when to query, what counts as a "graph question") are
  intrinsic to graphrag-rs's value proposition.

Proposed Skill Content

1. Frontmatter

       ---
       name: graphrag
       description: Use when the user asks about content in their
         personal knowledge base (Obsidian vault) — questions like
         "what have I written about X", "what do my notes say about
         Y", "summarize my thinking on Z", or anything where the
         answer should come from prior personal notes rather than
         general knowledge or current code.
       ---

2. Body sections

   - **What this skill is for**: one paragraph framing graphrag as
     personal-knowledge retrieval over the user's Obsidian vault.
     Distinguish from: code search (use ripgrep), web search (use
     searxng MCP), and current-state lookups (use git log / file
     read).
   - **Activation triggers** (positive): list 6–8 concrete prompt
     shapes that should fire the skill. Examples:
       "what have I written about ..."
       "what notes do I have on ..."
       "summarize my thinking on ..."
       "what was my conclusion about ..."
       "find my notes that touch on ..."
   - **Anti-triggers**: list shapes that should NOT fire it. Examples:
       code questions, "how does X work" general explainers,
       greenfield brainstorming, anything in the current conversation
       context, ephemeral state (clipboard, git status).
   - **How to query**: invoke `mcp__graphrag__query` with the user's
     phrasing. Don't paraphrase aggressively — the embedding match is
     better with the user's own terms.
   - **When the graph is empty**: if `graph_stats` shows 0 entities,
     fall back to listing documents and reading a chunk directly via
     `list_documents`. Surface this gap to the user — it's a config
     issue, not a skill issue.
   - **Result handling**: the query response includes `documentId`,
     `title`, `similarity`, `excerpt`. Use the excerpt to ground the
     answer; cite the title; don't fabricate beyond what the excerpts
     contain.

3. Examples (literal Q→tool-call→A traces)

   - Q: "what did I write about LLM evaluation last quarter?"
     Tool: query(question="LLM evaluation 2026-Q1")
     Show how to thread top results into a synthesized answer.
   - Q: "summarize my notes on the Transformer paper"
     Tool: query(question="Transformer architecture self-attention")
     Show how to handle multiple matches with overlapping content.
   - Q (anti-trigger): "how does Rust's Send trait work?"
     → don't fire skill; this isn't personal-knowledge retrieval.

4. Cross-client parity

   For opencode and crush, render an equivalent block as
   custom_instructions in their JSON config. Same triggers, same
   anti-patterns, but reference the MCP server name as configured
   in each client (graphrag for both today). Source of truth: keep
   the SKILL.md as canonical and copy the activation/anti-pattern
   bullets into the JSON via a small generator if drift becomes a
   problem. For now, hand-sync.

Verification Plan

- Manual: ask Claude five "what have I written about X" variants,
  check the skill loads (visible via "Skill loaded: graphrag" in the
  trace) and a query MCP call follows.
- Manual: ask five anti-trigger variants (code questions, current-
  state lookups), check the skill does NOT load.
- Cross-client: same first round of tests against opencode and
  crush; verify their system prompt matches the SKILL.md content.

Out of Scope

- Skill bundling for sharing. If we ever want to publish this for
  other Obsidian + graphrag-rs users, that's a separate distribution
  story (probably a small companion repo). The skill itself is
  user-specific in its examples.
- Auto-generating SKILL.md content from graphrag-rs metadata
  (entity counts, document titles). Static text is fine and easier
  to reason about; introducing a generator adds maintenance burden
  for a small win.

----

Phase G: rehydrate graphrag-core's KnowledgeGraph from Qdrant on startup (✅ COMPLETED)
✅ On graphrag-server startup, scroll the Qdrant sidecar collections — done via `hydrate_in_memory_graph` (`graphrag-server/src/graph_persistence.rs:384`)
✅ Entities + relationships restored directly from sidecar payloads, not re-extracted; production boot log shows `🔄 Restored entity graph from Qdrant: 12383 entities, 26729 relationships (0 orphan rels skipped)` every restart
✅ Chunk surface hydrated lazily (counts only); chunks stay in Qdrant and `extend_graph_streaming` queries on demand
⬜ `GRAPHRAG_REHYDRATE_ON_STARTUP={true,false,timer}` env knob — punted; only the synchronous `true` path runs in production. Add the timer mode if a slow-start need ever materializes
⬜ `/health` field for rehydrate progress — currently logged only; punted
⬜ Integration test for documentCount post-restart — punted

----

Implementation Plan — Phase G: rehydrate KnowledgeGraph from Qdrant on startup

Goal

graphrag-server keeps two stores: Qdrant (persistent, vectors + payloads) and graphrag-core's in-memory KnowledgeGraph (chunks, documents, entities, relationships). Today only Qdrant survives across restarts. After a restart:

  /health                → documentCount: N    (Qdrant)
  /api/graph/stats       → documentCount: 0    (graphrag-core)
  Qdrant collection      → N points

The agent can ingest into both via `POST /api/documents`, so the disconnect is invisible until you query: `/api/graph/build` and `/api/graph/append` walk graphrag-core's chunks, which are zero post-restart, so extraction produces nothing — the graph stays empty until everything is re-ingested.

Phase G fixes this by replaying every Qdrant payload through `graphrag.add_document_from_text` on startup so graphrag-core's chunk store matches Qdrant. The user's content_hash dedup means re-issuing the same content via `add_document` is also safe — but on the rehydration path we go straight into graphrag-core, no re-embedding.

Non-Goals

- Persisting the entity/relationship graph itself — we re-extract from chunks via `extend_graph`. Entities / relationships are a derived view; persisting them is Phase H.
- Replacing Qdrant with a different store. Qdrant stays the source of truth for vectors + payloads.

Proposed Changes

1. New `graphrag-server/src/rehydrate.rs` module:

       pub async fn rehydrate_from_qdrant(
           qdrant: &QdrantStore,
           graphrag: &mut GraphRAG,
       ) -> Result<usize, RehydrateError>;

   Scrolls the collection in pages of 256 (matches list_documents cap). For each DocumentMetadata, calls graphrag.add_document_from_text(text). Returns total rehydrated count.

2. Wire into AppState::new. Three delivery modes via env:

   - GRAPHRAG_REHYDRATE_ON_STARTUP=true (default): block startup until rehydration completes. Server starts in a usable state. Slow on large corpora (~5ms per doc; 1k docs ≈ 5s; 100k ≈ 8min).
   - GRAPHRAG_REHYDRATE_ON_STARTUP=timer: spawn a background task that rehydrates after server bind. /health reports progress.
   - GRAPHRAG_REHYDRATE_ON_STARTUP=false: skip entirely, current behaviour.

3. Optional extend_graph call after rehydration. Without Phase H persistent entity storage, /api/graph/append re-extracts everything once.

4. /health surfaces rehydration state:

       {
         "status": "healthy",
         "documentCount": 39,
         "rehydration": {
           "completed": true | false,
           "processed": 39,
           "total": 39,
           "started_at": "RFC3339",
           "completed_at": "RFC3339" | null
         }
       }

5. Verification

   - Integration test: populate Qdrant; restart; assert `/api/graph/stats.documentCount` equals Qdrant's count after rehydration.
   - Manual: ingest 50 docs; restart; verify /health.rehydration.completed flips true; /api/graph/stats shows 50 chunks; /api/graph/build extracts entities normally.

Out of Scope (Phase H)

- Persistent entity/relationship storage so clean shutdowns don't lose the graph. Bigger refactor — serialization format for entities/relationships/mentions, a save_to_qdrant method writing to a sibling collection, a load_from_qdrant mirror. Punt until G ships.

  → 2026-05-08 update: this work also shipped. Entities and relationships
    persist to dedicated Qdrant sidecar collections via
    `persist_touched_snapshot` (`graphrag-server/src/graph_persistence.rs:274`),
    and the same path drives the 3-tier embed cache (in-process → qdrant
    fetch → OVMS). Phase G's rehydrate side is fully complementary.

----

# 2026-05-08 LightRAG audit — open work

The LightRAG runtime path is wired (dual-level keywords, relationship-description
retrieval, incremental updates, cross-restart hydrate, 3-tier embed cache,
per-request retry). Open items the audit surfaced — picked up after the
nginx-stall + OOM hardening lands and proves stable.

----

Phase H: cross-encoder re-ranking integration (⬜ IN PROGRESS)

Cross-encoder code is compiled in (`graphrag-core/src/reranking/cross_encoder.rs`,
Candle/BERT impl) but the runtime query path never calls it. The MCP
`recall` tool dispatches through `POST /api/query` →
`graphrag-server/src/main.rs:1014::graph_aware_query`, NOT a /api/ask
endpoint (no such route exists). All four MCP query modes — `default`,
`thorough`, `local`, `simple` — build a `Vec<QueryResult>` from
`version_aware_search`; the rerank fits right after that and before the
synthesis prompt (or before the response, in `simple` mode).

With top-k already returning ~30 candidates, cross-encoder rerank
typically buys +10–20% nDCG at ~50 ms/query.

⬜ Wire `CandleCrossEncoder::rerank(query, candidates)` after
   `version_aware_search` in `graphrag-server/src/main.rs::graph_aware_query`
   AND the `simple`-mode handler — both build the same `Vec<QueryResult>`
⬜ Add `enhancements.reranker.enabled` flag through home-manager so
   reranking is opt-in until the model size + cold-start cost is validated
⬜ Decide on model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (current default
   in code) is 22 MB; could swap to `BAAI/bge-reranker-base` (440 MB,
   stronger) — leave config-driven
⬜ Latency budget: emit a `rerank_ms` field on `QueryResponse` so we can
   observe the per-query overhead without enabling debug logs

----

Phase I: multi-turn chat memory (⬜ NOT STARTED)

`QueryRequest` is single-turn today. Follow-ups like "what about X?" lose
the previous context. Adds a small session-store layer.

⬜ Extend `QueryRequest` with `conversation_id: Option<Uuid>` and
   `messages: Option<Vec<{role, content}>>` (OpenAI-shaped)
⬜ New `conversations` table in the existing sqlite events store; ttl 24 h
⬜ Synthesis prompt includes the prior 4 turns (token-budget-aware)
⬜ Obsidian gateway plugin: keep a per-pane conversation_id and replay it
   on each ask
⬜ Decision: re-retrieve per turn vs cache the prior turn's seeds. Default
   to re-retrieve — cheap and avoids stale context

----

Phase J: hybrid BM25 + vector retrieval into the default path (⬜ NOT STARTED)

`graphrag-core/src/retrieval/hybrid.rs` is fully implemented (RRF / Weighted
/ CombSum / MaxScore fusion) but isn't called from `/api/ask`. Vector-only
retrieval misses rare exact-match terms (filenames, IDs, code symbols).

⬜ Build a tantivy or sled-based BM25 index alongside the qdrant chunk
   collection on first run; index updates piggyback on the existing
   `add_document` path
⬜ Wire `HybridRetriever::retrieve` into `graph_aware_query` behind
   `enhancements.hybrid_retrieval.enabled`; default to RRF fusion
⬜ Persist the BM25 index to disk so it survives restarts (matches qdrant)
⬜ Bench: how much recall does this buy on rare-term queries?

----

Phase K: native PDF / DOCX / HTML ingestion (⬜ NOT STARTED)

Today, `ingest_policy.rs` routes binary formats through an external
`INGEST_PREPROCESSOR_URL` (Nemotron-Omni or pandoc). Bundling a native
parser layer removes that external dep for the common cases.

⬜ Add `pdf-extract` (or `lopdf`) for `.pdf`
⬜ Add `docx-rs` for `.docx`
⬜ Add `scraper` (or `html2text`) for `.html` / `.htm`
⬜ Plumb through the `Preprocessor::extract_text` trait so the existing
   preprocessor can still kick in for unsupported MIME types
⬜ Update `DEFAULT_ALLOWED_EXTENSIONS` once parsers land
⬜ Decision: keep the external preprocessor as a feature flag for users
   who want OCR / table extraction the native parsers don't do

----

Phase L: cleanup MS-GraphRAG vestiges (✅ COMPLETED 2026-05-08)

Strict deletion. Git history is the archive.

✅ Deleted `graphrag-core/src/retrieval/symbolic_anchoring.rs` (CatRAG anchoring)
✅ Deleted `graphrag-core/src/retrieval/causal_analysis.rs`
✅ Deleted `graphrag-core/src/optimization/` (graph_weight_optimizer.rs, mod.rs — DW-GRPO)
✅ Deleted `graphrag-core/src/rograg/` entire directory (9 files: processor, decomposer, intent_classifier, logic_form, fuzzy_matcher, validator, quality_metrics, streaming, mod.rs, tests.rs)
✅ Removed `AdvancedFeaturesConfig` + 5 nested config structs (`SymbolicAnchoringConfig`, `DynamicWeightingConfig`, `CausalAnalysisConfig`, `HierarchicalClusteringConfig`, `WeightOptimizationConfig`, `ObjectiveWeightsConfig`) from `graphrag-core/src/config/mod.rs`
✅ Removed all corresponding `default_*` helper functions (anchor/causal/cluster/weight defaults)
✅ Removed `pub advanced_features: AdvancedFeaturesConfig` field from `Config` and its two `Default::default()` initializers
✅ Removed 5 ROGRAG `From` impls (LogicFormError, ProcessingError, MetricsError, StreamingError, FuzzyMatchError) from `graphrag-core/src/core/error.rs`
✅ Removed `pub mod symbolic_anchoring` and `pub mod causal_analysis` from `graphrag-core/src/retrieval/mod.rs`
✅ Removed `pub mod optimization` and `#[cfg(feature = "rograg")] pub mod rograg` from `graphrag-core/src/lib.rs`
✅ Removed `rograg` feature flag and its inclusion in the `research` bundle from `graphrag-core/Cargo.toml`
✅ Deleted `graphrag-core/tests/advanced_features_integration.rs` and `tests/dynamic_weighting_tests.rs`
✅ Deleted `graphrag-core/benches/advanced_features_benchmark.rs`
✅ Deleted `graphrag-core/ADVANCED_FEATURES.md`
✅ Deleted `graphrag-core/config-examples/advanced-features.toml`
✅ Stripped `[advanced_features.*]` blocks from `graphrag-core/config-examples/quick-start.toml`
✅ Stripped `[rograg]` blocks and `rograg_decomposition` flags from `config/templates/dynamic_universal.toml`
✅ Stripped "ROGRAG" mentions from header comments in `config/templates/{web_blog_content,legal_documents,technical_documentation,dynamic_universal}.toml`
✅ Removed all ROGRAG / vestige references from `README.md`, `graphrag-core/README.md`, `HOW_IT_WORKS.md`, `report.md`

Note: The leiden algorithm (`graph/leiden.rs`) was kept — it's used by `KnowledgeGraph::detect_hierarchical_communities`, the public API, and `graph/hierarchical_relationships.rs`. It is NOT a vestige.
