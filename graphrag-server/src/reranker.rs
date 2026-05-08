//! Cross-encoder reranking for the runtime retrieval path.
//!
//! Phase H wire-up: after `version_aware_search` returns the candidate
//! `Vec<QueryResult>`, an optional reranker re-scores `(query, excerpt)`
//! pairs and reorders the slice. With ~30 vector candidates and a
//! BGE-style reranker model, this typically buys +10–20% nDCG at
//! ~50 ms/query.
//!
//! Backend: HTTP-based, Cohere/vLLM-compatible POST /rerank. The user's
//! existing infrastructure can serve a reranker model on Spark's vLLM
//! (e.g. `BAAI/bge-reranker-base`, ~440 MB) or the embedding box, and
//! point this at it via the `reranker.endpoint` config field. The
//! advantage over a local Candle/BERT load is no on-host model files
//! and no cold-start cost on graphrag-server boot.
//!
//! Strict semantics:
//!   - When the config disables the reranker (`enabled = false` or
//!     `endpoint` empty), `RerankerService::from_config` returns None
//!     and the runtime path skips the rerank step entirely.
//!   - When enabled but the upstream is unreachable mid-query, the
//!     rerank call returns `Err` and the caller is expected to fall
//!     back to the original ordering — never silently corrupt the
//!     results list. (See `graph_aware_query` for the fallback shape.)

use serde::Deserialize;
use std::time::Duration;

use graphrag_core::config::RerankerConfig;

use crate::models::QueryResult;

/// Live reranker. One instance per process (or per `POST /config` swap).
/// Cheap to clone — internal state is just the reqwest::Client (which
/// owns its own connection pool) and the static config string fields.
#[derive(Clone)]
pub struct RerankerService {
    http: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: String,
    top_n: usize,
}

impl RerankerService {
    /// Build a reranker from the config. Returns `None` when the
    /// reranker is disabled or required fields are empty — the
    /// caller treats that as "skip rerank" rather than an error.
    /// Logs the reason when a misconfigured-but-enabled state slips
    /// through (e.g. enabled=true but endpoint="").
    pub fn from_config(cfg: &RerankerConfig) -> Option<Self> {
        if !cfg.enabled {
            return None;
        }
        if cfg.endpoint.is_empty() || cfg.model.is_empty() {
            tracing::warn!(
                "reranker: enabled=true but endpoint or model empty; skipping (endpoint={:?}, model={:?})",
                cfg.endpoint,
                cfg.model,
            );
            return None;
        }
        // Same timeout-hardening pattern as the chat / embed clients —
        // tcp_keepalive + connect_timeout catch dead upstreams long
        // before timeout_seconds runs out.
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_seconds.max(1)))
            .connect_timeout(Duration::from_secs(15))
            .tcp_keepalive(Duration::from_secs(60))
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .ok()?;
        Some(Self {
            http,
            endpoint: cfg.endpoint.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            top_n: cfg.top_n,
        })
    }

    /// Reorder `results` in-place by reranker score. Returns the
    /// per-query rerank latency (ms). Empty input is a no-op (returns 0).
    ///
    /// On HTTP failure: logs warn, leaves the slice in its original
    /// order, returns Err. Caller is expected to keep the original
    /// ordering and continue — never poison the result set with
    /// partial reranks.
    pub async fn rerank(
        &self,
        query: &str,
        results: &mut Vec<QueryResult>,
    ) -> Result<u128, RerankError> {
        if results.is_empty() {
            return Ok(0);
        }

        let started = std::time::Instant::now();
        let documents: Vec<String> =
            results.iter().map(|r| r.excerpt.clone()).collect();
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": if self.top_n == 0 { results.len() } else { self.top_n },
        });

        let url = format!("{}/rerank", self.endpoint);
        let mut req = self.http.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.map_err(|e| RerankError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(RerankError::Http {
                status: status.as_u16(),
                body: body_text,
            });
        }
        let parsed: RerankResponse =
            resp.json().await.map_err(|e| RerankError::Parse(e.to_string()))?;

        // Reorder `results` according to the upstream-supplied permutation
        // of (index, relevance_score). Indices outside the original range
        // are silently dropped (defensive against an upstream sending
        // bogus data). After the reorder, `similarity` carries the new
        // relevance score so downstream consumers can see how the rerank
        // moved things; we don't keep the original similarity around
        // because callers that care about original-vs-rerank can compare
        // to the unsorted vector.
        let n = results.len();
        let original = std::mem::take(results);
        let mut wrapped: Vec<Option<QueryResult>> =
            original.into_iter().map(Some).collect();
        for entry in &parsed.results {
            if entry.index < n {
                if let Some(mut r) = wrapped[entry.index].take() {
                    r.similarity = entry.relevance_score;
                    results.push(r);
                }
            }
        }
        // Truncate to top_n if the upstream returned more (some servers
        // do, some don't honor the cap server-side).
        if self.top_n > 0 && results.len() > self.top_n {
            results.truncate(self.top_n);
        }
        Ok(started.elapsed().as_millis())
    }
}

/// Wire-shape of the upstream response, Cohere/vLLM-compatible.
#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankEntry>,
}

#[derive(Debug, Deserialize)]
struct RerankEntry {
    index: usize,
    relevance_score: f32,
}

/// Rerank failure modes. All non-fatal at the call-site — the caller
/// keeps the original ordering when these fire.
#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    #[error("rerank transport error: {0}")]
    Transport(String),

    #[error("rerank HTTP {status}: {body}")]
    Http { status: u16, body: String },

    #[error("rerank response parse failed: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_disabled_returns_none() {
        let cfg = RerankerConfig::default();
        assert!(RerankerService::from_config(&cfg).is_none());
    }

    #[test]
    fn from_config_enabled_but_empty_endpoint_returns_none() {
        let cfg = RerankerConfig {
            enabled: true,
            endpoint: String::new(),
            model: "x".to_string(),
            ..Default::default()
        };
        assert!(RerankerService::from_config(&cfg).is_none());
    }

    #[test]
    fn from_config_enabled_with_endpoint_constructs() {
        let cfg = RerankerConfig {
            enabled: true,
            endpoint: "http://localhost:8080/v1".to_string(),
            model: "bge-reranker-base".to_string(),
            ..Default::default()
        };
        assert!(RerankerService::from_config(&cfg).is_some());
    }
}
