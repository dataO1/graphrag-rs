//! `ChatClient` — runtime dispatcher between Ollama-protocol and
//! OpenAI-compatible chat backends.
//!
//! Every consumer of the chat LLM (entity extractors, query planner,
//! gleaning) used to take `OllamaClient` directly. Switching them to
//! `ChatClient` lets `Config.openai.enabled` route the same call sites
//! to either an Ollama server or any OpenAI-spec server (vLLM,
//! llama-server, real OpenAI, OpenRouter, ...) without per-callsite
//! branching.
//!
//! Method surface mirrors `OllamaClient` exactly (`generate`,
//! `generate_with_params`, `get_stats`, `clear_cache`, `cache_size`) so
//! the swap is a tree-wide find-replace. The canonical params type stays
//! `OllamaGenerationParams` — the OpenAI side ignores Ollama-only fields
//! (top_k, repeat_penalty, keep_alive, num_ctx).
//!
//! Adaptive concurrency: every `generate*` call acquires a permit from
//! an optional [`crate::llm_concurrency::AdaptiveSemaphore`] before
//! hitting the wire, classifies the response, and feeds success / failure
//! back to the AIMD controller. When no semaphore is attached (default),
//! calls are not gated. Hosts that want adaptive concurrency call
//! [`ChatClient::with_semaphore`] once at construction.

use crate::core::error::Result;
use crate::ollama::{OllamaClient, OllamaConfig, OllamaGenerationParams, OllamaUsageStats};
use crate::openai::OpenAIConfig;
#[cfg(feature = "openai")]
use crate::openai::OpenAIClient;

#[cfg(feature = "async")]
use std::sync::Arc;

/// Chat client. Dispatches to Ollama or OpenAI-compat backend, with an
/// optional adaptive concurrency semaphore gating every call.
#[derive(Clone, Debug)]
pub struct ChatClient {
    backend: Backend,
    /// Adaptive AIMD permit budget shared across all callers cloning
    /// this client. `None` means calls are not gated (default).
    #[cfg(feature = "async")]
    semaphore: Option<Arc<crate::llm_concurrency::AdaptiveSemaphore>>,
}

/// Internal backend dispatch. Private — consumers go through
/// [`ChatClient`]'s public surface.
#[derive(Clone, Debug)]
enum Backend {
    Ollama(OllamaClient),
    #[cfg(feature = "openai")]
    OpenAI(OpenAIClient),
}

impl ChatClient {
    /// Wire from the runtime config. OpenAI takes precedence over Ollama
    /// when both are enabled (so `openai.enabled = true` overrides any
    /// `ollama.enabled` setting). When the `openai` feature is not
    /// compiled in, `openai.enabled` is ignored and the dispatcher
    /// falls through to the Ollama branch (or `None`) — useful build-
    /// flag mistakes surface as a `tracing::warn!` if `tracing` is on.
    pub fn from_config(
        ollama: &OllamaConfig,
        openai: &OpenAIConfig,
    ) -> Option<Self> {
        #[cfg(feature = "openai")]
        if openai.enabled {
            return Some(Self::wrap(Backend::OpenAI(OpenAIClient::new(openai.clone()))));
        }
        #[cfg(all(not(feature = "openai"), feature = "tracing"))]
        if openai.enabled {
            tracing::warn!(
                "openai.enabled = true but graphrag-core was built without the `openai` feature; \
                 falling back to ollama. Rebuild with `--features openai` (or the `starter` bundle) to enable."
            );
        }
        let _ = openai; // keep the binding meaningful when no cfg-guarded read happens
        if ollama.enabled {
            Some(Self::wrap(Backend::Ollama(OllamaClient::new(ollama.clone()))))
        } else {
            None
        }
    }

    fn wrap(backend: Backend) -> Self {
        Self {
            backend,
            #[cfg(feature = "async")]
            semaphore: None,
        }
    }

    /// Construct directly from an `OllamaClient` (back-compat helper for
    /// call sites that build their own client).
    pub fn from_ollama(client: OllamaClient) -> Self {
        Self::wrap(Backend::Ollama(client))
    }

    /// Construct directly from an `OpenAIClient`. Only present with
    /// `feature = "openai"`.
    #[cfg(feature = "openai")]
    pub fn from_openai(client: OpenAIClient) -> Self {
        Self::wrap(Backend::OpenAI(client))
    }

    /// Attach an adaptive concurrency semaphore. Every subsequent
    /// `generate*` call acquires a permit before hitting the wire,
    /// classifies the response (transport error / 4xx 429 / 5xx →
    /// failure; any 2xx with parsed body → success; parse / repair
    /// failure that's NOT transport-level → neither), and feeds the
    /// signal back to the AIMD controller.
    ///
    /// Returns `self` for chaining: `ChatClient::from_config(...).with_semaphore(sem)`.
    #[cfg(feature = "async")]
    pub fn with_semaphore(
        mut self,
        semaphore: Arc<crate::llm_concurrency::AdaptiveSemaphore>,
    ) -> Self {
        self.semaphore = Some(semaphore);
        self
    }

    /// Whether either backend is available right now (mirrors
    /// `Config.{ollama,openai}.enabled` checks at call sites).
    pub fn is_enabled(&self) -> bool { true }

    /// Single-shot generate. Routes through the adaptive semaphore if
    /// one is attached.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        self.gated(self.dispatch_generate(prompt)).await
    }

    /// Generate with caller-supplied params. Routes through the adaptive
    /// semaphore if one is attached.
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
    ) -> Result<String> {
        self.gated(self.dispatch_generate_with_params(prompt, params)).await
    }

    /// Same as [`Self::generate_with_params`] but merges per-call `extras`
    /// (a JSON object) into the OpenAI request body. On the Ollama backend
    /// `extras` is ignored — Ollama does not support arbitrary extra body
    /// fields. Use this for vLLM-specific knobs such as `priority` that
    /// have no Ollama equivalent.
    pub async fn generate_with_extras(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
        extras: serde_json::Value,
    ) -> Result<String> {
        self.gated(self.dispatch_generate_with_extras(prompt, params, extras)).await
    }

    async fn dispatch_generate_with_extras(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
        extras: serde_json::Value,
    ) -> Result<String> {
        // Silence the unused-variable warning on builds without the openai feature:
        // extras is intentionally ignored on the Ollama backend (no wire field).
        let _ = &extras;
        match &self.backend {
            // Ollama has no extras mechanism — fall back to plain params.
            Backend::Ollama(c) => c.generate_with_params(prompt, params).await,
            #[cfg(feature = "openai")]
            Backend::OpenAI(c) => c.generate_with_extras(prompt, params, extras).await,
        }
    }

    async fn dispatch_generate(&self, prompt: &str) -> Result<String> {
        match &self.backend {
            Backend::Ollama(c) => c.generate(prompt).await,
            #[cfg(feature = "openai")]
            Backend::OpenAI(c) => c.generate(prompt).await,
        }
    }

    async fn dispatch_generate_with_params(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
    ) -> Result<String> {
        match &self.backend {
            Backend::Ollama(c) => c.generate_with_params(prompt, params).await,
            #[cfg(feature = "openai")]
            Backend::OpenAI(c) => c.generate_with_params(prompt, params).await,
        }
    }

    /// Generate a response constrained to a JSON Schema.
    ///
    /// On the OpenAI-compat backend with `guided_json = true`, attaches
    /// `response_format: { type: "json_schema", json_schema: { name,
    /// schema, strict: true } }` and routes through the per-call extras
    /// path so the upstream sampler (vLLM xgrammar / outlines, llama.cpp
    /// recent, real OpenAI strict mode) can only emit conforming JSON.
    ///
    /// On every other path (Ollama backend, or OpenAI with
    /// `guided_json = false`) this falls back to plain
    /// `generate_with_params` — the schema and name arguments are
    /// silently ignored. Caller's existing parse / repair pipeline then
    /// runs unchanged. Plumbing Ollama's native `format` field is left
    /// for a follow-up.
    pub async fn generate_for_structured_output(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
        schema: &serde_json::Value,
        name: &str,
    ) -> Result<String> {
        self.gated(self.dispatch_structured(prompt, params, schema, name)).await
    }

    async fn dispatch_structured(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
        schema: &serde_json::Value,
        name: &str,
    ) -> Result<String> {
        match &self.backend {
            #[cfg(feature = "openai")]
            Backend::OpenAI(c) if c.config().guided_json => {
                let extras = serde_json::json!({
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": name,
                            "schema": schema,
                            "strict": true,
                        }
                    }
                });
                c.generate_with_extras(prompt, params, extras).await
            }
            _ => {
                let _ = (schema, name);
                self.dispatch_generate_with_params(prompt, params).await
            }
        }
    }

    /// Wrap the inner generate future with semaphore acquire +
    /// success/failure classification. When no semaphore is attached,
    /// the inner future runs ungated and the result is returned as-is.
    #[cfg(feature = "async")]
    async fn gated<F>(&self, fut: F) -> Result<String>
    where
        F: std::future::Future<Output = Result<String>>,
    {
        match &self.semaphore {
            Some(sem) => {
                let permit = sem.acquire().await;
                let result = fut.await;
                if classify_failure(&result) {
                    permit.record_failure();
                } else {
                    permit.record_success();
                }
                result
            },
            None => fut.await,
        }
    }

    #[cfg(not(feature = "async"))]
    async fn gated<F>(&self, fut: F) -> Result<String>
    where
        F: std::future::Future<Output = Result<String>>,
    {
        fut.await
    }

    /// Stats handle (always returns the underlying client's stats).
    pub fn get_stats(&self) -> &OllamaUsageStats {
        match &self.backend {
            Backend::Ollama(c) => c.get_stats(),
            #[cfg(feature = "openai")]
            Backend::OpenAI(c) => c.get_stats(),
        }
    }

    /// `keep_alive` value when the active backend is Ollama; `None` for
    /// OpenAI (the field is Ollama-specific). Consumers used to read this
    /// off `ollama_client.config().keep_alive` directly; expose it here so
    /// the call site doesn't have to know which backend is live.
    pub fn keep_alive(&self) -> Option<String> {
        match &self.backend {
            Backend::Ollama(c) => c.config().keep_alive.clone(),
            #[cfg(feature = "openai")]
            Backend::OpenAI(_) => None,
        }
    }
}

/// Classify a `generate*` result as a transport-level failure that
/// should shrink the AIMD permit cap. JSON parse / schema-repair
/// failures are NOT transport-level — they reflect model output quality,
/// not capacity, and would over-shrink on a flaky model running below
/// the upstream's actual slot count.
///
/// `GraphRAGError::Generation` carries a string message in this codebase
/// (no typed kind), so we substring-match the prefixes the
/// `OpenAIClient::generate_with_params` uses:
///   - "OpenAI transport error" — connect/read timeout, reset, etc.
/// `OpenAIClient` reports HTTP status codes via "OpenAI HTTP <code>"
/// prefixes; we treat 429 and 5xx as failures, 4xx other as quality
/// issues that don't shrink.
#[cfg(feature = "async")]
fn classify_failure(result: &Result<String>) -> bool {
    let Err(e) = result else { return false };
    let msg = format!("{e}");
    if msg.contains("OpenAI transport error") || msg.contains("transport error") {
        return true;
    }
    if let Some(rest) = msg.split("OpenAI HTTP ").nth(1) {
        // Format: "OpenAI HTTP <code>: <body>"
        let code_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(code) = code_str.parse::<u16>() {
            if code == 429 || (500..600).contains(&code) {
                return true;
            }
        }
    }
    if msg.contains("OpenAI join error") {
        // task panicked / cancelled mid-call; treat as failure
        return true;
    }
    false
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use crate::core::error::GraphRAGError;

    fn err(msg: &str) -> Result<String> {
        Err(GraphRAGError::Generation { message: msg.to_string() })
    }

    #[test]
    fn classify_transport_is_failure() {
        assert!(classify_failure(&err("OpenAI transport error: read timeout")));
    }

    #[test]
    fn classify_429_is_failure() {
        assert!(classify_failure(&err("OpenAI HTTP 429: too many requests")));
    }

    #[test]
    fn classify_5xx_is_failure() {
        assert!(classify_failure(&err("OpenAI HTTP 503: upstream unavailable")));
        assert!(classify_failure(&err("OpenAI HTTP 500: internal")));
    }

    #[test]
    fn classify_4xx_other_is_not_failure() {
        // 400 Bad Request from a malformed prompt — capacity isn't the
        // problem, don't shrink.
        assert!(!classify_failure(&err("OpenAI HTTP 400: bad request")));
        assert!(!classify_failure(&err("OpenAI HTTP 404: model not found")));
    }

    #[test]
    fn classify_parse_error_is_not_failure() {
        // JSON repair failure — quality issue, not capacity.
        assert!(!classify_failure(&err("Failed to parse repaired JSON: missing field source")));
    }

    #[test]
    fn classify_ok_is_not_failure() {
        assert!(!classify_failure(&Ok("response".to_string())));
    }
}
