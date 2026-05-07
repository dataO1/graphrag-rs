//! OpenAI-compatible chat backend.
//!
//! Mirrors the surface of `crate::ollama::OllamaClient` so consumers can
//! talk to any OpenAI-spec server (vLLM, llama-server, real OpenAI,
//! OpenRouter, ...) without changing call sites. Pair with the
//! `ChatClient` enum dispatcher in `crate::chat` to route all chat-LLM
//! traffic — entity extraction, query planning, gleaning — between an
//! Ollama-protocol server and an OpenAI-compat one based on `Config`.
//!
//! Feature gating: `OpenAIConfig` is unconditional so user configs
//! round-trip through serde whether or not the `openai` feature is on.
//! `OpenAIClient` and the HTTP path are gated behind `feature = "openai"`
//! (which itself depends on `reqwest` + `async`). Without the feature,
//! `chat::ChatClient::from_config` silently treats `openai.enabled = true`
//! as "no openai available" and falls back to Ollama / None.

use serde::{Deserialize, Serialize};
#[cfg(feature = "openai")]
use crate::core::error::{GraphRAGError, Result};
#[cfg(feature = "openai")]
use crate::ollama::{OllamaGenerationParams, OllamaUsageStats};

/// OpenAI-compatible chat-backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// Whether to use the OpenAI-compat backend for chat. When true,
    /// takes precedence over `Config.ollama.enabled` for chat LLM use.
    #[serde(default)]
    pub enabled: bool,

    /// Full base URL including the version path. Examples:
    /// - `http://localhost:8000/v1` (vLLM)
    /// - `http://localhost:17171/v1` (llama-server)
    /// - `https://api.openai.com/v1` (real OpenAI)
    /// - `https://openrouter.ai/api/v1` (OpenRouter)
    #[serde(default = "default_base_url")]
    pub base_url: String,

    /// Chat model identifier. For self-hosted servers this is whatever
    /// the server reports under GET /models; for real OpenAI it's
    /// "gpt-4o-mini" / "gpt-4o" / "o1-preview" / etc.
    #[serde(default = "default_chat_model")]
    pub chat_model: String,

    /// Bearer token. Empty string disables the Authorization header
    /// (fine for self-hosted servers without auth).
    #[serde(default)]
    pub api_key: String,

    /// HTTP request timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Maximum retry attempts on transient errors.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Maximum tokens to generate per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Whether to cache responses by prompt hash. Stub (no cache yet).
    #[serde(default = "default_enable_caching")]
    pub enable_caching: bool,

    /// Extra top-level fields merged into every chat-completions request
    /// body. Used to pass non-standard knobs that specific OpenAI-compat
    /// servers accept — e.g. llama.cpp / vLLM honor
    /// `chat_template_kwargs: {"enable_thinking": false}` to suppress
    /// Qwen3-style reasoning without changing the server's CLI flags.
    /// Must be a JSON object; non-object values are ignored on merge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<serde_json::Value>,

    /// Attach a JSON Schema `response_format` to entity-extraction calls so
    /// the upstream sampler is constrained to emit conforming JSON. vLLM
    /// (xgrammar / outlines), llama.cpp (recent), and real OpenAI all
    /// honor `response_format: { type: "json_schema", ... }`. Backends
    /// without support typically 400 on the extra field — flip this off
    /// in that case. Default `false` to preserve existing behavior on
    /// upgrade. Only the entity-extractor path uses this; query planner
    /// and completion checks continue to send free-form requests.
    #[serde(default)]
    pub guided_json: bool,
}

fn default_base_url() -> String { "http://localhost:8000/v1".to_string() }
fn default_chat_model() -> String { "gpt-4o-mini".to_string() }
fn default_timeout() -> u64 { 60 }
fn default_max_retries() -> u32 { 3 }
fn default_enable_caching() -> bool { true }

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: default_base_url(),
            chat_model: default_chat_model(),
            api_key: String::new(),
            timeout_seconds: default_timeout(),
            max_retries: default_max_retries(),
            max_tokens: Some(2000),
            temperature: Some(0.7),
            enable_caching: default_enable_caching(),
            extra_body: None,
            guided_json: false,
        }
    }
}

/// OpenAI-compatible chat client.
///
/// Uses async `reqwest::Client` so callers don't pay the
/// `spawn_blocking + ureq` round-trip on every chat call. The blocking
/// thread pool is shared globally with embeddings, file I/O, etc.; with
/// `llm.max = 128` the OpenAI HTTP path used to starve that pool and cap
/// effective in-flight requests far below the configured concurrency.
/// One `reqwest::Client` is built per `OpenAIClient::new` and reused for
/// the lifetime of the client (connection pooling is automatic).
#[cfg(feature = "openai")]
#[derive(Clone)]
pub struct OpenAIClient {
    config: OpenAIConfig,
    http: reqwest::Client,
    stats: OllamaUsageStats,
}

#[cfg(feature = "openai")]
impl std::fmt::Debug for OpenAIClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIClient")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(feature = "openai")]
impl OpenAIClient {
    /// Build a new client. Network is not touched until the first
    /// `generate*` call. The reqwest client uses HTTP/1.1 keep-alive
    /// connections by default; one client instance pools them per host.
    pub fn new(config: OpenAIConfig) -> Self {
        let timeout = std::time::Duration::from_secs(config.timeout_seconds);
        // The reqwest builder rejects a client built without a runtime if
        // any cfg goes wrong; a `.build()` failure here would only happen
        // on a system without TLS support, which our rustls-tls feature
        // covers. Fall back to the default client (which will fail later
        // on TLS calls only) rather than panic at construction time.
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config,
            http,
            stats: OllamaUsageStats::new(),
        }
    }

    /// Read access to the configured chat model (used by adapters / logs).
    pub fn config(&self) -> &OpenAIConfig {
        &self.config
    }

    /// Stats handle (parity with `OllamaClient`).
    pub fn get_stats(&self) -> &OllamaUsageStats {
        &self.stats
    }

    /// Single-shot chat completion with default params.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let params = OllamaGenerationParams {
            num_predict: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: None,
            top_k: None,
            stop: None,
            repeat_penalty: None,
            num_ctx: None,
            keep_alive: None,
            context: None,
        };
        self.generate_with_params(prompt, params).await
    }

    /// Build the JSON body for a single /chat/completions request.
    /// Extracted from `generate_with_params` so unit tests can assert on
    /// the exact shape that goes over the wire (including extra_body
    /// merge precedence) without standing up an HTTP server.
    pub(crate) fn build_request_body(
        &self,
        prompt: &str,
        params: &OllamaGenerationParams,
    ) -> serde_json::Value {
        // Build request JSON. Single-turn chat (one user message) — same
        // shape consumers send to OllamaClient.generate{,_with_params}.
        let mut body = serde_json::json!({
            "model": self.config.chat_model,
            "messages": [
                { "role": "user", "content": prompt }
            ],
            "stream": false,
        });
        if let Some(t) = params.temperature.or(self.config.temperature) {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(n) = params.num_predict.or(self.config.max_tokens) {
            body["max_tokens"] = serde_json::json!(n);
        }
        if let Some(p) = params.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if let Some(ref stops) = params.stop {
            if !stops.is_empty() {
                body["stop"] = serde_json::json!(stops);
            }
        }
        // Merge config.extra_body keys at the top level. Lets callers pass
        // server-specific knobs (e.g. `chat_template_kwargs` on llama.cpp
        // to disable Qwen3 thinking) without expanding this struct for
        // every backend quirk. Existing keys win — set fields on this
        // struct take precedence over extra_body collisions.
        if let (Some(serde_json::Value::Object(extras)), Some(obj)) =
            (self.config.extra_body.as_ref(), body.as_object_mut())
        {
            for (k, v) in extras {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        let _ = prompt; // keep name in scope for the comment above
        body
    }

    /// Chat completion with caller-supplied params. The
    /// `OllamaGenerationParams` shape is kept as the canonical params
    /// type so consumers don't have to branch on backend; only the
    /// fields that map to OpenAI (max_tokens, temperature, top_p, stop)
    /// are honored. `top_k` / `repeat_penalty` / `keep_alive` /
    /// `num_ctx` / `context` are Ollama-specific and silently ignored.
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
    ) -> Result<String> {
        // Body construction extracted into build_request_body so unit
        // tests can assert on the wire shape (incl. extra_body merge
        // precedence) without standing up an HTTP server.
        let body = self.build_request_body(prompt, &params);
        self.send_chat_request(body).await
    }

    /// Same as [`Self::generate_with_params`] but merges per-call `extras`
    /// (a JSON object) into the request body on top of `config.extra_body`
    /// and the standard fields. Used by the entity extractor to attach
    /// `response_format: { type: "json_schema", ... }` so vLLM /
    /// llama.cpp / real-OpenAI sample tokens that conform to the
    /// extraction schema. Non-object `extras` are ignored.
    pub async fn generate_with_extras(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
        extras: serde_json::Value,
    ) -> Result<String> {
        let mut body = self.build_request_body(prompt, &params);
        if let (Some(obj), Some(extras_obj)) = (body.as_object_mut(), extras.as_object()) {
            for (k, v) in extras_obj {
                obj.insert(k.clone(), v.clone());
            }
        }
        self.send_chat_request(body).await
    }

    /// Send a fully-built chat-completions request body. Shared between
    /// `generate_with_params` and `generate_with_extras`. Async over
    /// reqwest — no spawn_blocking, the future yields on every I/O wait
    /// so the tokio reactor remains free to drive other futures (other
    /// in-flight LLM calls, embedding HTTP, qdrant writes, …).
    async fn send_chat_request(&self, body: serde_json::Value) -> Result<String> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        let mut req = self.http.post(&url).json(&body);
        if !self.config.api_key.is_empty() {
            req = req.bearer_auth(&self.config.api_key);
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                self.stats.record_failure();
                return Err(GraphRAGError::Generation {
                    message: format!("OpenAI transport error: {e}"),
                });
            },
        };

        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            self.stats.record_failure();
            return Err(GraphRAGError::Generation {
                message: format!("OpenAI HTTP {code}: {body}"),
            });
        }

        let parsed: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                self.stats.record_failure();
                return Err(GraphRAGError::Generation {
                    message: format!("OpenAI response parse: {e}"),
                });
            },
        };

        // OpenAI/llama-server/vLLM all return:
        //   { choices: [{ message: { role: "assistant", content: "..." } }],
        //     usage: { prompt_tokens, completion_tokens, total_tokens } }
        let content = parsed
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let total_tokens = parsed
            .pointer("/usage/total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        self.stats.record_success(total_tokens);

        Ok(content)
    }

    /// Stub for parity with OllamaClient.
    pub fn clear_cache(&self) {}

    /// Stub for parity.
    pub fn cache_size(&self) -> usize { 0 }
}

// ─────────────────────────────────────────────────────────────────────────
// Tests — exercise the public surface that's hardest to ship by hand:
// serde round-trip on the config (incl. Option<max_tokens> and the JSON
// blob in extra_body), and the request-body merge precedence rule
// (set fields on OpenAIConfig beat extra_body collisions).
// ─────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "openai"))]
mod tests {
    use super::*;

    fn cfg() -> OpenAIConfig {
        OpenAIConfig {
            enabled: true,
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            chat_model: "test-model".to_string(),
            api_key: String::new(),
            timeout_seconds: 60,
            max_retries: 3,
            max_tokens: Some(2000),
            temperature: Some(0.2),
            enable_caching: true,
            extra_body: None,
            guided_json: false,
        }
    }

    fn empty_params() -> OllamaGenerationParams {
        OllamaGenerationParams {
            num_predict: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            repeat_penalty: None,
            num_ctx: None,
            keep_alive: None,
            context: None,
        }
    }

    #[test]
    fn config_round_trips_through_serde_with_extra_body_object() {
        // Realistic case: chat_template_kwargs.enable_thinking=false for
        // Qwen3 on llama.cpp's --jinja path. The whole shape must come
        // back identical or we'd silently drop server-specific knobs.
        let mut c = cfg();
        c.extra_body = Some(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": false }
        }));
        let json = serde_json::to_string(&c).expect("serialize");
        let back: OpenAIConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.chat_model, c.chat_model);
        assert_eq!(back.extra_body, c.extra_body);
        assert_eq!(back.max_tokens, c.max_tokens);
    }

    #[test]
    fn config_round_trips_with_max_tokens_none() {
        // Local-LLM "no cap" mode: max_tokens = None must serialize to a
        // missing field (skip_serializing_if), not `null`, so the
        // OpenAI-spec server falls back to its own default.
        let mut c = cfg();
        c.max_tokens = None;
        let json = serde_json::to_string(&c).expect("serialize");
        assert!(
            !json.contains("max_tokens"),
            "max_tokens should be skipped when None, got: {json}"
        );
        let back: OpenAIConfig = serde_json::from_str(&json).expect("deserialize");
        assert!(back.max_tokens.is_none());
    }

    #[test]
    fn config_extra_body_omitted_when_none() {
        let c = cfg();
        let json = serde_json::to_string(&c).expect("serialize");
        assert!(!json.contains("extra_body"), "extra_body should be skipped when None: {json}");
    }

    /// f32 → JSON Number → f64 round-trip is lossy (0.2 ≠ 0.20000000298…),
    /// so float fields are compared with a tolerance. Tightened to 1e-5
    /// — coarse enough to absorb the f32 widening, tight enough to catch
    /// any genuine drift.
    fn approx_eq(v: &serde_json::Value, expected: f64) {
        let got = v.as_f64().unwrap_or_else(|| panic!("not a number: {v:?}"));
        assert!(
            (got - expected).abs() < 1e-5,
            "expected ~{expected}, got {got} (diff {})",
            (got - expected).abs()
        );
    }

    #[test]
    fn build_request_body_minimal_shape() {
        let client = OpenAIClient::new(cfg());
        let body = client.build_request_body("hello", &empty_params());
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
        // params override missing → fall back to config; config has these set
        approx_eq(&body["temperature"], 0.2);
        assert_eq!(body["max_tokens"], 2000);
    }

    #[test]
    fn build_request_body_uses_params_over_config() {
        let client = OpenAIClient::new(cfg());
        let mut p = empty_params();
        p.temperature = Some(0.9);
        p.num_predict = Some(50);
        let body = client.build_request_body("x", &p);
        approx_eq(&body["temperature"], 0.9);
        assert_eq!(body["max_tokens"], 50);
    }

    #[test]
    fn build_request_body_omits_max_tokens_when_uncapped() {
        // None on both config and params → no max_tokens in the body, so
        // llama.cpp / vLLM / OpenAI use their server-side default.
        let mut c = cfg();
        c.max_tokens = None;
        let client = OpenAIClient::new(c);
        let body = client.build_request_body("x", &empty_params());
        assert!(body.get("max_tokens").is_none(), "uncapped should omit max_tokens, got: {body}");
    }

    #[test]
    fn extra_body_merges_unique_keys() {
        let mut c = cfg();
        c.extra_body = Some(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": false },
            "response_format":      { "type": "json_object" },
        }));
        let client = OpenAIClient::new(c);
        let body = client.build_request_body("x", &empty_params());
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["response_format"]["type"], "json_object");
    }

    #[test]
    fn extra_body_loses_to_set_fields_on_collision() {
        // Precedence rule: anything we set via config or params (model,
        // max_tokens, temperature, stop, top_p) wins over extra_body
        // attempting the same key. Stops the user shooting themselves in
        // the foot by accidentally overwriting a typed field with a
        // raw JSON blob.
        let mut c = cfg();
        c.chat_model = "real-model".to_string();
        c.extra_body = Some(serde_json::json!({
            "model":       "should-not-win",
            "max_tokens":  9999,
            "temperature": 5.0,
        }));
        let client = OpenAIClient::new(c);
        let body = client.build_request_body("x", &empty_params());
        assert_eq!(body["model"], "real-model");
        assert_eq!(body["max_tokens"], 2000);
        approx_eq(&body["temperature"], 0.2);
    }

    #[test]
    fn extra_body_ignored_when_not_an_object() {
        // Defensive: a misconfigured extra_body (e.g. a bare string) must
        // not break the request — it's silently dropped, body proceeds.
        let mut c = cfg();
        c.extra_body = Some(serde_json::json!("not-an-object"));
        let client = OpenAIClient::new(c);
        let body = client.build_request_body("x", &empty_params());
        assert_eq!(body["model"], "test-model");
        assert!(body.get("not-an-object").is_none());
    }
}
