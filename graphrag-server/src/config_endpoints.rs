//! Configuration endpoints for GraphRAG Server
//!
//! These endpoints allow dynamic configuration of the GraphRAG pipeline via JSON REST API

use super::{config_handler, AppState};
use crate::embeddings::EmbeddingService;
use crate::models::ApiError;
use actix_web::web::{Data, Json};
use serde_json::json;
use std::sync::Arc;

/// GET /api/config - Get current configuration
pub async fn get_config(state: Data<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    if !state.config_manager.is_configured().await {
        return Err(ApiError::NotFound(
            "No configuration set. Use POST /api/config to initialize.".to_string(),
        ));
    }

    match state.config_manager.to_json().await {
        Ok(config_json) => {
            let config: serde_json::Value = serde_json::from_str(&config_json)
                .map_err(|e| ApiError::InternalError(e.to_string()))?;

            Ok(Json(json!({
                "success": true,
                "config": config,
                "graphrag_initialized": state.graphrag.load().is_some()
            })))
        },
        Err(e) => Err(ApiError::InternalError(e)),
    }
}

/// POST /api/config - Set configuration and initialize GraphRAG.
///
/// Atomically rebuilds the embedding subsystem to match the new config:
///
/// 1. Merge the posted patch into the active config (deep merge).
/// 2. Build a fresh `EmbeddingService` from the merged
///    `config.embeddings`. Probe-embed a known string and reject the
///    POST with HTTP 400 if the returned vector length doesn't equal
///    `config.embeddings.dimension`. This is the single chokepoint that
///    prevents silent Qdrant corruption — without it, a config that
///    claims dim=1024 but talks to a 768-D upstream only fails at
///    insert time, after we've already accepted documents.
/// 3. Hot-swap `state.embeddings` (lock-free via `ArcSwap`) and update
///    `state.config` so `/health`, `/config`, and `/embeddings/stats`
///    immediately reflect the new struct.
/// 4. Build a fresh `GraphRAG` instance with the new config and inject
///    the new embedder before `initialize()`.
pub async fn set_config(
    state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!("Received configuration request");

    // Parse the configuration from JSON
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {}", e)))?;

    // Set configuration via ConfigManager (handles deep-merge + validation)
    state
        .config_manager
        .set_from_json(&config_json)
        .await
        .map_err(ApiError::BadRequest)?;

    // Get the validated config
    let config = state
        .config_manager
        .get_config()
        .await
        .ok_or(ApiError::InternalError("Failed to get config".to_string()))?;

    // Build the new EmbeddingService from the merged embeddings block.
    // Failure here is a 400 — the caller's config is bad (unreachable
    // upstream, missing fields), not an internal server error.
    let new_embeddings = EmbeddingService::from_config(&config.embeddings)
        .await
        .map_err(|e| {
            ApiError::BadRequest(format!(
                "Embedding service rebuild failed for backend={}: {}",
                config.embeddings.backend, e
            ))
        })?;

    // Probe-embed to validate dimension end-to-end. Fails HERE rather
    // than at the next /api/documents POST, which would silently insert
    // wrong-dim vectors into a Qdrant collection sized to the configured
    // dimension and corrupt the index. Skip when backend=hash because
    // the hash generator is sized to `config.dimension` by construction.
    //
    // Also catches the upstream-misconfigured case: the EmbeddingService
    // silently falls through to hash when the real backend returns a
    // wrong-dim vector (so `generate_single` would return a hash vector
    // of the correct size and the dim check alone would pass). We
    // detect that by also asserting `backend_live()` — if the configured
    // backend isn't live but the user asked for it, that's also a 400.
    if config.embeddings.backend != "hash" {
        if !new_embeddings.backend_live() {
            return Err(ApiError::BadRequest(format!(
                "Embedding backend '{}' is not reachable at endpoint '{}'. \
                 Check the upstream server, or set embeddings.backend = \"hash\".",
                config.embeddings.backend,
                config.embeddings.api_endpoint.as_deref().unwrap_or("(none)")
            )));
        }

        let stats_before = new_embeddings.get_stats().await;
        let probe = new_embeddings.generate_single("graphrag dimension probe").await;
        let stats_after = new_embeddings.get_stats().await;

        match probe {
            Ok(v) if v.len() != config.embeddings.dimension => {
                return Err(ApiError::BadRequest(format!(
                    "Embedding dimension mismatch: config.embeddings.dimension={} but \
                     backend={} returned {}-D vectors. Update config.embeddings.dimension \
                     or change the model.",
                    config.embeddings.dimension,
                    config.embeddings.backend,
                    v.len()
                )));
            },
            Ok(_) if stats_after.fallback_used > stats_before.fallback_used => {
                // Real backend errored mid-call (e.g. dim-mismatch caught
                // inside generate_with_openai), fallback covered it. The
                // returned vector is hash-sized = config.dimension, so
                // the length check passed silently. Reject the POST.
                return Err(ApiError::BadRequest(format!(
                    "Embedding backend '{}' probe failed (fell through to hash fallback). \
                     Check server logs for the upstream error, or set embeddings.backend = \"hash\".",
                    config.embeddings.backend
                )));
            },
            Ok(_) => {},
            Err(e) => {
                return Err(ApiError::BadRequest(format!(
                    "Embedding probe failed during /config validation: {e}"
                )));
            },
        }
    }

    // Atomically swap the live embedder. Existing in-flight requests
    // that already grabbed an `Arc<EmbeddingService>` snapshot will
    // continue using the old one; new calls see the new service.
    let new_embeddings = Arc::new(new_embeddings);
    state.embeddings.store(new_embeddings.clone());

    // Same pattern for the reranker. Boot-time config has reranker
    // disabled (graphrag-core defaults); POST /config from
    // home-manager carries the user's actual reranker block. Without
    // this swap, the reranker stays None forever even after a valid
    // config arrives.
    {
        let new_reranker = crate::reranker::RerankerService::from_config(&config.reranker);
        if new_reranker.is_some() {
            tracing::info!(
                "reranker: enabled (endpoint={}, model={}, top_n={})",
                config.reranker.endpoint, config.reranker.model, config.reranker.top_n,
            );
        } else if config.reranker.enabled {
            tracing::info!("reranker: enabled in config but skipped (see prior warn)");
        } else {
            tracing::info!("reranker: disabled");
        }
        state.reranker.store(std::sync::Arc::new(new_reranker));
    }

    // Update the live config snapshot. AFTER the embedder swap so any
    // reader that wins the race sees old-config + old-embedder or
    // new-config + new-embedder, never new-config + old-embedder.
    state.config.store(std::sync::Arc::new(config.clone()));

    // Re-print the unified backend log line so users see the swap.
    crate::log_unified_embedding_line(&config.embeddings, new_embeddings.backend_live());

    // Initialize GraphRAG with the config
    tracing::info!("Initializing GraphRAG with custom configuration...");

    // Best-effort probe of the chat upstream's slot count. llama.cpp's
    // server exposes `GET /props` with `total_slots` (= --parallel); other
    // OpenAI-compat backends (vLLM, real OpenAI) typically don't, so a
    // 404 / non-JSON response is normal — we just fall back to the
    // configured `llm.initial`. The AIMD controller will discover the
    // actual capacity within a few minutes either way.
    //
    // The probe can only RAISE `config.llm.initial`, never lower it.
    // Reason: the configured value is a deliberate user/operator choice
    // (system flake `llm.initial = 128` for the Spark backend, etc).
    // We've seen probes return a buggy `total_slots=1` (e.g. nginx
    // routing chain returns a 502/HTML body that happens to deserialize
    // to `{ total_slots: 1 }`, or one of the round-robin upstreams
    // misreports), which would silently pin AIMD at 1 and force +1/10
    // ramp-up — never reaching the user's intended cap on a finite
    // workload. Trust the configured initial as a floor; let the probe
    // bump it higher when the upstream genuinely advertises more slots
    // than the user knew about.
    let mut config = config;
    let probed_slots = probe_upstream_slots(&config.openai.base_url).await;
    let configured_initial = config.llm.initial;
    let cap = config.llm.max;
    if let Some(slots) = probed_slots {
        let probe_clamped = slots.clamp(1, cap);
        if probe_clamped > configured_initial {
            tracing::info!(
                "llm.concurrency: probed total_slots={} from {}/props; raising initial={} → {} (cap={})",
                slots,
                config.openai.base_url.trim_end_matches('/'),
                configured_initial,
                probe_clamped,
                cap,
            );
            config.llm.initial = probe_clamped;
        } else {
            tracing::info!(
                "llm.concurrency: probed total_slots={} from {}/props; keeping configured initial={} (probe ≤ configured; cap={})",
                slots,
                config.openai.base_url.trim_end_matches('/'),
                configured_initial,
                cap,
            );
        }
    } else {
        tracing::info!(
            "llm.concurrency: no /props on upstream (or probe failed); using configured initial={} (cap={})",
            configured_initial,
            cap,
        );
    }

    // Synthesis prompt budget: when user left `max_input_chars=0`,
    // probe the chat upstream for its max context window and resolve.
    // Probe order: vLLM `/v1/models[].max_model_len` → llama.cpp
    // `/v1/models[].meta.n_ctx_train` → llama.cpp `/props`. Falls
    // back to a conservative 32 768-char cap (≈ 8 K tokens) if none
    // of those work, so the cap is never silently absent.
    if config.synthesis.max_input_chars == 0 {
        let probed_tokens = probe_upstream_max_model_len(&config.openai.base_url).await;
        let max_output_tokens = config.openai.max_tokens.unwrap_or(2_000) as usize;
        let chars_per_token = 4usize;
        let safety = 0.9f32;
        let resolved = match probed_tokens {
            Some(model_max) => {
                // (input_token_budget) × chars_per_token × safety
                let input_tokens = model_max.saturating_sub(max_output_tokens);
                let chars = ((input_tokens as f32) * (chars_per_token as f32) * safety) as usize;
                tracing::info!(
                    "synthesis: probed model max_model_len={} tokens; \
                     resolving max_input_chars={} (output_reserve={} tokens, \
                     ×{} chars/token, ×{} safety; chunks_budget={})",
                    model_max,
                    chars,
                    max_output_tokens,
                    chars_per_token,
                    safety,
                    chars.saturating_sub(config.synthesis.skeleton_reserve_chars),
                );
                chars
            },
            None => {
                let fallback = 32_768usize;
                tracing::warn!(
                    "synthesis: no max_model_len from upstream (vLLM /v1/models or \
                     llama.cpp /props); falling back to max_input_chars={} \
                     (≈{} tokens). Set `synthesis.max_input_chars` explicitly to \
                     match your model's context window.",
                    fallback,
                    fallback / chars_per_token,
                );
                fallback
            },
        };
        config.synthesis.max_input_chars = resolved;
    } else {
        tracing::info!(
            "synthesis: using configured max_input_chars={} (chunks_budget={})",
            config.synthesis.max_input_chars,
            config.synthesis.max_input_chars
                .saturating_sub(config.synthesis.skeleton_reserve_chars),
        );
    }

    let mut graphrag = graphrag_core::GraphRAG::new(config)
        .map_err(|e| ApiError::InternalError(format!("GraphRAG init failed: {}", e)))?;

    // Inject the freshly-built embedding service into graphrag-core
    // BEFORE initialize() runs. Single source of truth: graphrag-core's
    // retrieval system, semantic chunker, and entity-vector path all
    // route through this same service.
    graphrag.set_embedding_provider(new_embeddings.clone());

    graphrag
        .initialize()
        .map_err(|e| ApiError::InternalError(format!("GraphRAG initialization failed: {}", e)))?;

    // Phase 6: chunks are NOT loaded into in-memory state on hydrate
    // anymore. Qdrant is the source of truth for chunk content, chunk
    // metadata, and the `entities_extracted_at` dedup signal. We only
    // restore the entity + relationship petgraph from the sidecar
    // collections (small, ~tens of KB at current scale, load-bearing
    // for graph traversal in `ask_with_dual_seeds`).
    let mut hydration_summary = json!({
        "documents": 0,
        "entities": 0,
        "relationships": 0,
        "relationships_skipped_orphan": 0,
    });
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Document count for telemetry; we don't load text into memory.
        let doc_count = qdrant
            .list_full_documents(1_000_000)
            .await
            .map(|d| d.len())
            .unwrap_or(0);
        tracing::info!(
            "🔄 Hydrated chunk surface: {} documents in Qdrant (chunks not loaded into memory; extend_graph queries Qdrant for unextracted)",
            doc_count
        );
        state
            .processed_chunk_count
            .store(0, std::sync::atomic::Ordering::SeqCst);
        hydration_summary = json!({
            "documents": doc_count,
            "entities": 0,
            "relationships": 0,
            "relationships_skipped_orphan": 0,
        });
        // Restore the LLM-extracted entity + relationship graph from
        // its sidecar collections. Best-effort: a load failure (e.g.
        // missing sidecar on a fresh deploy) is normal — we just start
        // with an empty entity graph and the next /api/graph/append
        // populates it.
        match crate::graph_persistence::hydrate_in_memory_graph(&mut graphrag, qdrant).await {
            Ok((entities_restored, rels_restored, rels_skipped)) => {
                if entities_restored + rels_restored > 0 {
                    tracing::info!(
                        "🔄 Restored entity graph from Qdrant: {} entities, {} relationships ({} orphan rels skipped)",
                        entities_restored,
                        rels_restored,
                        rels_skipped,
                    );
                }
                if let Some(obj) = hydration_summary.as_object_mut() {
                    obj.insert("entities".into(), json!(entities_restored));
                    obj.insert("relationships".into(), json!(rels_restored));
                    obj.insert("relationships_skipped_orphan".into(), json!(rels_skipped));
                }
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "graph restore failed; starting with no entities (next build_graph will repopulate)"
                );
            },
        }
    }

    // Layer 4 (revised): publish the new graphrag to BOTH the
    // writer-owned master AND the reader snapshot. /api/graph/append
    // mutates the master in place and re-publishes the snapshot once
    // at the end of its cycle.
    let mut master = state.graphrag_writer.lock().await;
    state.graphrag.store(Some(std::sync::Arc::new(graphrag.clone())));
    *master = Some(graphrag);
    drop(master);

    tracing::info!("✅ GraphRAG initialized successfully with custom configuration");

    Ok(Json(json!({
        "success": true,
        "message": "GraphRAG initialized with custom configuration",
        "configured": true,
        "mode": "full_pipeline",
        "hydrated": hydration_summary,
    })))
}

/// GET /api/config/template - Get configuration template
pub async fn get_config_template() -> Json<config_handler::ConfigTemplateResponse> {
    Json(config_handler::get_config_templates())
}

/// GET /api/config/default - Get default configuration
pub async fn get_default_config() -> Json<serde_json::Value> {
    let default_json = config_handler::ConfigManager::default_config_json();
    let config: serde_json::Value = serde_json::from_str(&default_json).unwrap_or(json!({}));

    Json(json!({
        "config": config,
        "description": "Default GraphRAG configuration with sensible defaults"
    }))
}

/// Best-effort probe for the chat upstream's max context window in
/// **tokens**. Tries three endpoints in order, returns on the first
/// success:
///
/// 1. `GET <base_url>/models` → `data[0].max_model_len` (vLLM's
///    OpenAI-compat endpoint exposes this in many versions).
/// 2. `GET <base_url>/models` → `data[0].meta.n_ctx_train` (llama.cpp's
///    OpenAI-compat endpoint nests it under `meta`).
/// 3. `GET <base_url-without-/v1>/props` →
///    `default_generation_settings.n_ctx` or top-level `n_ctx_train`
///    (llama.cpp's properties endpoint).
///
/// Returns `None` for backends that don't advertise context size in
/// any of those shapes (real OpenAI, OpenRouter, …). Caller falls
/// back to a conservative default and logs.
async fn probe_upstream_max_model_len(base_url: &str) -> Option<usize> {
    let trimmed_v1 = base_url.trim_end_matches('/');
    let trimmed_root = trimmed_v1.trim_end_matches("/v1").trim_end_matches('/');
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;

    // (1) + (2) — /models is OpenAI-compat surface; both vLLM and
    // llama.cpp answer here.
    if let Ok(resp) = client.get(format!("{trimmed_v1}/models")).send().await {
        if resp.status().is_success() {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                let entry = body.get("data").and_then(|d| d.get(0));
                if let Some(e) = entry {
                    if let Some(n) = e.get("max_model_len").and_then(|v| v.as_u64()) {
                        return Some(n as usize);
                    }
                    if let Some(n) = e
                        .get("meta")
                        .and_then(|m| m.get("n_ctx_train"))
                        .and_then(|v| v.as_u64())
                    {
                        return Some(n as usize);
                    }
                }
            }
        }
    }

    // (3) — llama.cpp /props
    if let Ok(resp) = client.get(format!("{trimmed_root}/props")).send().await {
        if resp.status().is_success() {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                if let Some(n) = body
                    .get("default_generation_settings")
                    .and_then(|g| g.get("n_ctx"))
                    .and_then(|v| v.as_u64())
                {
                    return Some(n as usize);
                }
                if let Some(n) = body.get("n_ctx_train").and_then(|v| v.as_u64()) {
                    return Some(n as usize);
                }
            }
        }
    }

    None
}

/// Best-effort probe for the chat upstream's slot count. Hits
/// `<base_url-without-/v1>/props` with a 2s timeout — that's the
/// llama.cpp server's properties endpoint, which returns
/// `{ "total_slots": N, ... }` (matching `--parallel`).
///
/// Returns `Some(N)` only on a successful 2xx with a numeric
/// `total_slots` field. Any failure (404, timeout, non-llama backend,
/// nginx router that doesn't proxy `/props`, …) returns `None` and the
/// caller falls back to the configured `llm.initial`. The AIMD
/// controller discovers actual capacity at runtime regardless.
async fn probe_upstream_slots(base_url: &str) -> Option<usize> {
    let trimmed = base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    let url = format!("{trimmed}/props");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("total_slots")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .filter(|n| *n >= 1)
}

/// POST /api/config/validate - Validate configuration without applying
pub async fn validate_config(
    _state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {}", e)))?;

    // Try to parse as Config
    match serde_json::from_str::<graphrag_core::Config>(&config_json) {
        Ok(_) => Ok(Json(json!({
            "valid": true,
            "message": "Configuration is valid"
        }))),
        Err(e) => Ok(Json(json!({
            "valid": false,
            "errors": [format!("Parse error: {}", e)]
        }))),
    }
}
