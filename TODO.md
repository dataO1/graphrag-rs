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
