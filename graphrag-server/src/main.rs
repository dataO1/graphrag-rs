//! GraphRAG REST API Server with Actix-web and Apistos OpenAPI
//!
//! Production-ready REST API for GraphRAG operations with automatic OpenAPI documentation.
//!
//! ## Features
//! - Automatic OpenAPI 3.0.3 documentation via Apistos
//! - Interactive Swagger UI at /swagger
//! - Qdrant vector database integration (optional)
//! - JWT and API key authentication (optional)
//! - Request validation and rate limiting
//!
//! ## Quick Start
//!
//! ```bash
//! # 1. Start Qdrant (Docker)
//! docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
//!
//! # 2. Start server with Qdrant
//! cargo run --bin graphrag-server --features qdrant
//!
//! # 3. Or without Qdrant (mock mode)
//! cargo run --bin graphrag-server --no-default-features
//!
//! # 4. View Swagger UI
//! # Browser: http://localhost:8080/swagger
//! ```

use actix_cors::Cors;
use actix_web::{
    web::{self, Data, Json, Path as WebPath},
    App, HttpServer, Responder,
};
use apistos::{
    api_operation,
    app::OpenApiWrapper,
    info::Info,
    spec::Spec,
    web::{delete, get, post, resource, scope},
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber;

mod models;
use models::*;

#[cfg(feature = "qdrant")]
mod qdrant_store;
#[cfg(feature = "qdrant")]
use qdrant_store::{DocumentMetadata, QdrantStore, SearchResult as QdrantSearchResult};

#[cfg(feature = "auth")]
mod auth;
#[cfg(feature = "auth")]
use auth::AuthState;

mod embeddings;
use embeddings::EmbeddingService;

mod validation;
use validation::{
    sanitize_string, validate_content, validate_query, validate_title, validate_top_k,
};

mod config_handler;
use config_handler::ConfigManager;

mod config_endpoints;

mod ingest_policy;
mod events_store;
mod stale_context;
use ingest_policy::{IngestPolicy, ResolvedPath};

#[cfg(feature = "qdrant")]
mod graph_persistence;

// Import full GraphRAG pipeline
use graphrag_core::GraphRAG;

/// Application state with optional Qdrant backend and full GraphRAG pipeline
#[derive(Clone)]
struct AppState {
    #[cfg(feature = "qdrant")]
    qdrant: Option<Arc<QdrantStore>>,

    /// Live embedding service. Wrapped in `ArcSwap` so `POST /config`
    /// can replace it atomically without touching every read site:
    /// readers `load()` an `Arc<EmbeddingService>` snapshot (lock-free),
    /// the writer `store()`s a fresh one built from the new config.
    embeddings: Arc<arc_swap::ArcSwap<EmbeddingService>>,

    /// Live `Config`. Single source of truth for `embeddings`, `graph`,
    /// `retrieval`, etc. — read by `/health`, `/config`, and
    /// `/embeddings/stats`; written by `POST /config`. Bootstrapped
    /// from `Config::default()` overlaid with env vars.
    config: Arc<RwLock<graphrag_core::Config>>,

    // Full GraphRAG pipeline (when configured via JSON)
    graphrag: Arc<RwLock<Option<GraphRAG>>>,

    // Configuration manager for JSON config
    config_manager: Arc<ConfigManager>,

    /// Path-based ingestion policy (sandbox roots, size caps, ext
    /// allow-list, optional preprocessor URL). Read once at boot from
    /// env vars; see `ingest_policy::IngestPolicy::from_env`. Cheap
    /// to clone (Arc'd). Empty `allowed_roots` ⇒ path-form `POST
    /// /api/documents` is rejected with 403.
    ingest_policy: Arc<IngestPolicy>,

    /// Coalescing signal for the in-server auto-append loop. Every
    /// successful new ingest calls `notify_one()`; the background task
    /// debounces by `APPEND_DEBOUNCE_SECS` of silence and then runs
    /// the same codepath `/api/graph/append` does. Replaces the
    /// previous home-manager 30-min cron — bursts collapse into one
    /// append, single-doc ingests become graph-queryable in
    /// `debounce_secs` rather than up to 30 min.
    auto_append_notify: Arc<tokio::sync::Notify>,

    /// Stale-context infrastructure (see graphrag-rs-nix/todo.md
    /// "Stale-context awareness for shared knowledge graph"):
    ///   - SQLite-backed event log + per-session lease table
    ///   - Live tokio broadcast bus for SSE fan-out
    /// Optional so deployments that don't enable the feature
    /// (or fail to open the SQLite file) still boot — the recall +
    /// ingest paths fall back to no-op when this is None.
    events_store: Option<Arc<events_store::EventsStore>>,
    /// Live event bus for SSE clients. Capacity 2048 — at 2 KB/event
    /// that's ~4 MB worst-case in memory, comfortably bounded for
    /// peak ingest bursts. Slow consumers see RecvError::Lagged and
    /// reconnect with Last-Event-ID for clean recovery.
    event_bus: Option<Arc<tokio::sync::broadcast::Sender<events_store::EventBody>>>,

    // Authentication state (optional)
    #[cfg(feature = "auth")]
    auth: Arc<AuthState>,

    // Fallback in-memory storage (used when Qdrant unavailable or simple mode)
    documents: Arc<RwLock<Vec<Document>>>,
    graph_built: Arc<RwLock<bool>>,
    /// RFC 3339 timestamp of the last successful /api/graph/build (None
    /// before the first build). Surfaced via /api/graph/stats so agents
    /// can decide whether the graph is fresh enough to query.
    last_built_at: Arc<RwLock<Option<String>>>,
    /// Number of chunks already passed through entity extraction. Set
    /// to the post-build chunk count after every /api/graph/build and
    /// /api/graph/append. Drives the no-op fast-path on /append: if
    /// the live chunk count hasn't grown since this counter, return
    /// early without re-running extraction.
    processed_chunk_count: Arc<std::sync::atomic::AtomicUsize>,
    query_count: Arc<RwLock<usize>>,
}

/// Overlay env-var bootstrap defaults onto a `Config.embeddings` block.
/// Lets deployments that only set env vars (the legacy path) keep working
/// without first posting `/config`. Once `POST /config` lands a complete
/// embeddings block, those values win on subsequent rebuilds.
///
/// Recognised env vars (all optional):
/// - `EMBEDDING_BACKEND` → `embeddings.backend` ("hash" / "openai" / "ollama")
/// - `EMBEDDING_DIM`     → `embeddings.dimension`
/// - `OPENAI_URL`        → `embeddings.api_endpoint` (when backend=openai)
/// - `OPENAI_EMBEDDING_MODEL` → `embeddings.model` (when backend=openai)
/// - `OPENAI_API_KEY`    → `embeddings.api_key`
/// - `OLLAMA_URL`+`OLLAMA_PORT` → `embeddings.api_endpoint` (joined "host:port")
/// - `OLLAMA_EMBEDDING_MODEL` → `embeddings.model` (when backend=ollama)
fn overlay_embedding_env_vars(emb: &mut graphrag_core::config::EmbeddingConfig) {
    if let Ok(b) = std::env::var("EMBEDDING_BACKEND") {
        emb.backend = b;
    }
    if let Ok(d) = std::env::var("EMBEDDING_DIM") {
        if let Ok(n) = d.parse::<usize>() {
            emb.dimension = n;
        }
    }
    match emb.backend.as_str() {
        "openai" => {
            if let Ok(u) = std::env::var("OPENAI_URL") {
                emb.api_endpoint = Some(u);
            }
            if let Ok(m) = std::env::var("OPENAI_EMBEDDING_MODEL") {
                emb.model = Some(m);
            }
            if let Ok(k) = std::env::var("OPENAI_API_KEY") {
                emb.api_key = Some(k);
            }
        },
        "ollama" => {
            // Combine OLLAMA_URL + OLLAMA_PORT into a single endpoint
            // string so the core `EmbeddingConfig` can stay backend-
            // agnostic. `EmbeddingService::from_config` parses it back
            // out via `parse_ollama_endpoint`.
            let host = std::env::var("OLLAMA_URL").ok();
            let port = std::env::var("OLLAMA_PORT").ok();
            if host.is_some() || port.is_some() {
                let h = host.unwrap_or_else(|| "http://localhost".to_string());
                let p = port.unwrap_or_else(|| "11434".to_string());
                emb.api_endpoint = Some(format!("{h}:{p}"));
            }
            if let Ok(m) = std::env::var("OLLAMA_EMBEDDING_MODEL") {
                emb.model = Some(m);
            }
        },
        _ => {},
    }
}

/// Single canonical log line that prints once after the embedding
/// service is built (boot or after `POST /config`). Replaces the two
/// older messages ("Initializing with backend: ..." printed before the
/// probe, and "Using hash-based fallback embeddings" printed regardless
/// of whether the upstream actually came up) so log readers can trust
/// what they see.
pub(crate) fn log_unified_embedding_line(
    cfg: &graphrag_core::config::EmbeddingConfig,
    live: bool,
) {
    let model = cfg.model.as_deref().unwrap_or("-");
    let endpoint = cfg.api_endpoint.as_deref().unwrap_or("-");
    let status = if live { "live" } else { "fallback=hash" };
    tracing::info!(
        "embeddings: backend={} model={} dim={} endpoint={} ({})",
        cfg.backend,
        model,
        cfg.dimension,
        endpoint,
        status
    );
}

impl AppState {
    async fn new() -> Self {
        // One source of truth for embeddings: a `graphrag_core::Config`
        // whose `embeddings` block is overlaid with env-var bootstrap
        // defaults so existing deployments that only set env vars still
        // work without posting `/config`. The same `config.embeddings`
        // is then handed to `EmbeddingService::from_config` and kept in
        // `state.config` for `/health`, `/config`, and
        // `/embeddings/stats` to read.
        let mut config = graphrag_core::Config::default();
        overlay_embedding_env_vars(&mut config.embeddings);

        let embeddings = match EmbeddingService::from_config(&config.embeddings).await {
            Ok(service) => Arc::new(arc_swap::ArcSwap::from_pointee(service)),
            Err(e) => {
                tracing::error!(
                    "❌ Failed to initialize embedding service: {}. Server cannot start.",
                    e
                );
                std::process::exit(1);
            },
        };
        log_unified_embedding_line(&config.embeddings, embeddings.load().backend_live());

        let config = Arc::new(RwLock::new(config));
        let embedding_dim = embeddings.load().dimension();
        let ingest_policy = IngestPolicy::from_env();
        let auto_append_notify = Arc::new(tokio::sync::Notify::new());

        // Stale-context infrastructure: SQLite-backed event log +
        // tokio broadcast bus. Best-effort — if the SQLite file
        // can't be opened, the server still boots; the SSE/recall
        // paths just no-op the event-emit side. STATE_DIR defaults
        // to ${XDG_STATE_HOME:-$HOME/.local/state}/graphrag-rs (set
        // by the home-manager module's StateDirectory).
        let stale_context_enabled = std::env::var("STALE_CONTEXT_ENABLE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let (events_store, event_bus) = if stale_context_enabled {
            let state_dir = std::env::var("STATE_DIR").ok().unwrap_or_else(|| {
                let base = std::env::var("XDG_STATE_HOME").ok().unwrap_or_else(|| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
                    format!("{home}/.local/state")
                });
                format!("{base}/graphrag-rs")
            });
            let db_path = format!("{state_dir}/state.sqlite");
            match events_store::EventsStore::open(
                &db_path,
                events_store::Settings::from_env(),
            )
            .await
            {
                Ok(store) => {
                    tracing::info!("📒 Stale-context events store opened at {}", db_path);
                    let (tx, _rx) = tokio::sync::broadcast::channel::<events_store::EventBody>(2048);
                    (Some(Arc::new(store)), Some(Arc::new(tx)))
                },
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %db_path,
                        "⚠️  Could not open events store; stale-context features disabled"
                    );
                    (None, None)
                },
            }
        } else {
            tracing::info!("📒 Stale-context events store disabled (STALE_CONTEXT_ENABLE=0)");
            (None, None)
        };

        #[cfg(feature = "qdrant")]
        {
            // Try to connect to Qdrant
            let qdrant_url =
                std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
            let collection_name =
                std::env::var("COLLECTION_NAME").unwrap_or_else(|_| "graphrag".to_string());

            match QdrantStore::new(&qdrant_url, &collection_name).await {
                Ok(store) => {
                    // Check if collection exists, create if not
                    if !store.collection_exists().await.unwrap_or(false) {
                        match store.create_collection(embedding_dim as u64).await {
                            Ok(_) => {
                                tracing::info!("✅ Created Qdrant collection: {}", collection_name);
                            },
                            Err(e) => {
                                tracing::warn!("⚠️  Could not create collection: {}", e);
                            },
                        }
                    } else {
                        tracing::info!(
                            "✅ Connected to existing Qdrant collection: {}",
                            collection_name
                        );
                    }

                    tracing::info!("🗄️  Using Qdrant at: {}", qdrant_url);

                    Self {
                        qdrant: Some(Arc::new(store)),
                        embeddings,
                        config,
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        ingest_policy: ingest_policy.clone(),
                        auto_append_notify: auto_append_notify.clone(),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                            |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                        ))),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        last_built_at: Arc::new(RwLock::new(None)),
                        processed_chunk_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                        query_count: Arc::new(RwLock::new(0)),
                        events_store: events_store.clone(),
                        event_bus: event_bus.clone(),
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        "⚠️  Could not connect to Qdrant: {}. Using in-memory storage.",
                        e
                    );
                    Self {
                        qdrant: None,
                        embeddings,
                        config,
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        ingest_policy: ingest_policy.clone(),
                        auto_append_notify: auto_append_notify.clone(),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                            |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                        ))),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        last_built_at: Arc::new(RwLock::new(None)),
                        processed_chunk_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                        query_count: Arc::new(RwLock::new(0)),
                        events_store: events_store.clone(),
                        event_bus: event_bus.clone(),
                    }
                },
            }
        }

        #[cfg(not(feature = "qdrant"))]
        {
            tracing::info!("📦 Using in-memory storage (Qdrant feature disabled)");
            Self {
                embeddings,
                config,
                graphrag: Arc::new(RwLock::new(None)),
                config_manager: Arc::new(ConfigManager::new()),
                ingest_policy: ingest_policy.clone(),
                auto_append_notify: auto_append_notify.clone(),
                #[cfg(feature = "auth")]
                auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                    |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                ))),
                documents: Arc::new(RwLock::new(Vec::new())),
                graph_built: Arc::new(RwLock::new(false)),
                last_built_at: Arc::new(RwLock::new(None)),
                processed_chunk_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                query_count: Arc::new(RwLock::new(0)),
                events_store: events_store.clone(),
                event_bus: event_bus.clone(),
            }
        }
    }

    /// Check if Qdrant is available
    fn has_qdrant(&self) -> bool {
        #[cfg(feature = "qdrant")]
        {
            self.qdrant.is_some()
        }
        #[cfg(not(feature = "qdrant"))]
        {
            false
        }
    }
}

// ============================================================================
// History-aware retrieval helper
// ============================================================================

/// Filter applied at every chunk-vector search call for `/api/query`.
/// Constructed from the request body's `as_of` and
/// `max_versions_per_doc` fields; defaults give the fast path
/// (`is_current = true` qdrant payload filter, no over-fetch).
#[derive(Debug, Clone)]
struct VersionFilter {
    as_of: Option<String>,
    max_versions_per_doc: u32,
}

impl VersionFilter {
    fn from_request(body: &QueryRequest) -> Self {
        Self {
            as_of: body.as_of.clone(),
            // 0 → 1 so callers can't accidentally turn off filtering.
            max_versions_per_doc: body.max_versions_per_doc.unwrap_or(1).max(1),
        }
    }

    fn is_default(&self) -> bool {
        self.as_of.is_none() && self.max_versions_per_doc == 1
    }
}

/// Vector search for chunks with the version filter applied.
///
/// Two strategies, chosen by `VersionFilter::is_default`:
///
/// 1. **Default path** (`as_of=None`, `max_versions_per_doc=1`):
///    pushes a Qdrant payload filter `is_current = true` into
///    `qdrant.search(...)`. Same cost as un-filtered search; only
///    current-version chunks come back; top-K never contaminated by
///    superseded versions.
///
/// 2. **History path** (any non-default filter): pulls back
///    `top_k * factor` candidates without a qdrant-side payload
///    filter, then in Rust:
///       a. drops chunks where `valid_from < as_of` (when `as_of` is
///          set; rfc3339 sorts lexicographically so a string compare
///          is correct);
///       b. groups by `user_id`, sorts each group by `version` desc,
///          keeps the top `max_versions_per_doc`;
///       c. re-sorts the union by similarity score and truncates to
///          `top_k`.
///    Over-fetch factor is `max(max_versions_per_doc, 3)`, capped
///    at 20× to keep wall-time bounded for absurd N. Legacy points
///    without `user_id` / `version` / `valid_from` are passed
///    through unchanged so old data still participates.
#[cfg(feature = "qdrant")]
async fn version_aware_search(
    qdrant: &QdrantStore,
    query_embedding: Vec<f32>,
    top_k: usize,
    filter: &VersionFilter,
) -> Result<Vec<QdrantSearchResult>, qdrant_store::QdrantError> {
    use qdrant_client::qdrant::{Condition, Filter as QFilter};

    if filter.is_default() {
        let f = QFilter::must([Condition::matches("is_current", true)]);
        return qdrant.search(query_embedding, top_k, Some(f)).await;
    }

    let over = filter.max_versions_per_doc.max(3) as usize;
    let fetch_k = top_k
        .saturating_mul(over)
        .min(top_k.saturating_mul(20))
        .max(top_k);

    let raw = qdrant.search(query_embedding, fetch_k, None).await?;

    // Stage 1: as_of (string compare on rfc3339).
    let staged: Vec<QdrantSearchResult> = if let Some(threshold) = filter.as_of.as_deref() {
        raw.into_iter()
            .filter(|r| {
                r.metadata
                    .valid_from
                    .as_deref()
                    .map(|vf| vf >= threshold)
                    .unwrap_or(true) // legacy point: keep
            })
            .collect()
    } else {
        raw
    };

    // Stage 2: per-user_id top-N by version desc.
    use std::collections::HashMap;
    let n = filter.max_versions_per_doc as usize;
    let mut grouped: HashMap<String, Vec<QdrantSearchResult>> = HashMap::new();
    let mut keyless: Vec<QdrantSearchResult> = Vec::new();
    for r in staged {
        match r.metadata.user_id.clone() {
            Some(uid) => grouped.entry(uid).or_default().push(r),
            None => keyless.push(r),
        }
    }
    let mut out: Vec<QdrantSearchResult> = keyless;
    for (_, mut group) in grouped {
        group.sort_by(|a, b| {
            b.metadata
                .version
                .unwrap_or(1)
                .cmp(&a.metadata.version.unwrap_or(1))
        });
        group.truncate(n);
        out.extend(group);
    }

    // Stage 3: best similarity first, truncate to top_k.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(top_k);
    Ok(out)
}

// ============================================================================
// API Handlers
// ============================================================================

/// Root endpoint - API information
#[api_operation(
    tag = "info",
    summary = "Get API information",
    description = "Returns basic information about the GraphRAG API, including version, status, and available endpoints"
)]
async fn root(state: Data<AppState>) -> impl Responder {
    Json(json!({
        "name": "GraphRAG REST API",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "backend": if state.has_qdrant() { "qdrant" } else { "memory" },
        "graphrag_configured": state.graphrag.read().await.is_some(),
        "documentation": "/swagger",
        "openapi_spec": "/openapi.json",
        "endpoints": {
            "health": "GET /health",
            "config": {
                "get": "GET /api/config - Get current configuration",
                "set": "POST /api/config - Set configuration and initialize GraphRAG",
                "template": "GET /api/config/template - Get configuration templates and examples",
                "default": "GET /api/config/default - Get default configuration",
                "validate": "POST /api/config/validate - Validate configuration without applying"
            },
            "query": {
                "endpoint": "POST /api/query",
                "modes": {
                    "search": "vector similarity over Qdrant (default; fast; no LLM)",
                    "ask": "graph-aware retrieval + LLM-composed answer",
                    "explain": "ask + confidence + source attribution + reasoning trace",
                    "reason": "query decomposition for multi-hop questions",
                    "local": "LightRAG `local` / MS GraphRAG `local_search`: vector-search entity sidecar → expand 1-hop → LLM answer (entity-centric)",
                    "global": "LightRAG `global`: high-level keywords → vector-search relationship sidecar → resolve endpoints → LLM answer (theme-centric)",
                    "hybrid": "LightRAG `hybrid`: dual-keyword extraction; both entity and relationship vector searches merged into one answer (best for mixed entity+theme questions)",
                    "mix": "LightRAG `mix`: hybrid + chunk-vector results merged; strongest recall, slightly slower"
                }
            },
            "documents": {
                "list": "GET /api/documents",
                "add": "POST /api/documents",
                "delete": "DELETE /api/documents/{id}"
            },
            "graph": {
                "build": "POST /api/graph/build",
                "append": "POST /api/graph/append",
                "stats": "GET /api/graph/stats"
            }
        }
    }))
}

/// Health check endpoint
#[api_operation(
    tag = "health",
    summary = "Health check",
    description = "Returns the current health status of the service, including document count, graph status, and total queries processed"
)]
async fn health(state: Data<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let doc_count;
    let graph_built;
    let query_count = *state.query_count.read().await;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                doc_count = count;
                graph_built = count > 0;
            },
            Err(_) => {
                doc_count = 0;
                graph_built = false;
            },
        }
    } else {
        doc_count = state.documents.read().await.len();
        graph_built = *state.graph_built.read().await;
    }

    #[cfg(not(feature = "qdrant"))]
    {
        doc_count = state.documents.read().await.len();
        graph_built = *state.graph_built.read().await;
    }

    let cfg = state.config.read().await;
    let svc = state.embeddings.load_full();
    let emb_block = HealthEmbeddings {
        backend: cfg.embeddings.backend.clone(),
        model: cfg.embeddings.model.clone().unwrap_or_default(),
        dimension: cfg.embeddings.dimension,
        endpoint: cfg.embeddings.api_endpoint.clone().unwrap_or_default(),
        live: svc.backend_live(),
    };
    drop(cfg);

    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        document_count: doc_count,
        graph_built,
        total_queries: query_count,
        backend: if state.has_qdrant() {
            "qdrant".to_string()
        } else {
            "memory".to_string()
        },
        embeddings: emb_block,
    }))
}

/// Query the knowledge graph
///
/// Routes by `mode`:
/// - `search` (default): Qdrant vector search; returns ranked excerpts.
///   ~350ms, no LLM call.
/// - `ask`: graph-aware retrieval + LLM-generated answer. Slower
///   (LLM round-trip) but produces a synthesized response, not just
///   excerpts.
/// - `explain`: same as `ask` plus confidence, source attribution
///   (chunks + entities + relationships), reasoning steps, and
///   key entities the answer relied on.
/// - `reason`: query decomposition for multi-hop questions; sub-queries
///   are answered and composed. Slowest but best for compound questions.
///
/// `ask`/`explain`/`reason` require a configured chat backend (POST /config
/// with `openai.enabled = true` or `ollama.enabled = true`). Without one
/// they return 400.
#[api_operation(
    tag = "query",
    summary = "Query the knowledge graph",
    description = "Search documents (mode=search, default) or ask the graph-aware engine for an LLM-composed answer (mode=ask|explain|reason).",
    error_code = 400,
    error_code = 500
)]
async fn query(
    state: Data<AppState>,
    body: Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    // Validate input
    if let Err(e) = validate_query(&body.query) {
        tracing::warn!(query = %body.query, error = %e.error, "Invalid query");
        return Err(ApiError::BadRequest(e.error));
    }

    if let Err(e) = validate_top_k(body.top_k) {
        tracing::warn!(top_k = body.top_k, error = %e.error, "Invalid top_k");
        return Err(ApiError::BadRequest(e.error));
    }

    let start = std::time::Instant::now();

    // Increment query count
    *state.query_count.write().await += 1;

    let mode = body.mode.unwrap_or_default();

    // Graph-aware modes: dispatch to graphrag-core. We always also attach
    // the vector-search hits as `results` so the caller still gets source
    // excerpts even when reading the LLM `answer`.
    if !matches!(mode, QueryMode::Search) {
        return graph_aware_query(&state, &body, mode, start).await;
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Real vector search with Qdrant using real embeddings
        let query_embedding = match state.embeddings.load_full().generate_single(&body.query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                tracing::error!("Failed to generate query embedding: {}", e);
                return Err(ApiError::InternalError(format!(
                    "Failed to generate embedding: {}",
                    e
                )));
            },
        };

        let vfilter = VersionFilter::from_request(&body);
        match version_aware_search(qdrant.as_ref(), query_embedding, body.top_k, &vfilter).await {
            Ok(search_results) => {
                let results: Vec<QueryResult> = search_results
                    .into_iter()
                    .map(|r| QueryResult {
                        document_id: r.id,
                        title: r.metadata.title,
                        similarity: r.score,
                        excerpt: if r.metadata.text.len() > 200 {
                            format!("{}...", &r.metadata.text[..200])
                        } else {
                            r.metadata.text
                        },
                        source: r.metadata.source,
                        line_start: r.metadata.line_start,
                        line_end: r.metadata.line_end,
                        heading_path: r.metadata.heading_path,
                        etag: r.metadata.block_hash.clone(),
                        block_id: r.metadata.block_id,
                    })
                    .collect();

                // Stale-context: record this session's lease entries
                // (block_id + etag) so the SSE stream and lease/check
                // can scope events to "blocks A actually retrieved".
                record_session_leases(&state, body.session_id.as_deref(), &results).await;

                let processing_time = start.elapsed().as_millis() as u64;

                return Ok(Json(QueryResponse {
                    query: body.query.clone(),
                    mode: mode.as_str().to_string(),
                    results,
                    answer: None,
                    confidence: None,
                    key_entities: None,
                    reasoning_steps: None,
                    sources: None,
                    processing_time_ms: processing_time,
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Qdrant search failed: {}",
                    e
                )));
            },
        }
    }

    // Fallback: in-memory search
    let documents = state.documents.read().await;

    if documents.is_empty() {
        return Err(ApiError::BadRequest(
            "No documents available. Add documents first.".to_string(),
        ));
    }

    // Simple keyword matching for demonstration
    let mut results: Vec<QueryResult> = documents
        .iter()
        .map(|doc| {
            let query_lower = body.query.to_lowercase();
            let content_lower = doc.content.to_lowercase();
            let title_lower = doc.title.to_lowercase();

            let similarity =
                if content_lower.contains(&query_lower) || title_lower.contains(&query_lower) {
                    0.85
                } else {
                    0.1
                };

            let excerpt = if doc.content.len() > 200 {
                format!("{}...", &doc.content[..200])
            } else {
                doc.content.clone()
            };

            QueryResult {
                document_id: doc.id.clone(),
                title: doc.title.clone(),
                similarity,
                excerpt,
                source: None,
                line_start: None,
                line_end: None,
                heading_path: Vec::new(),
                block_id: None,
                etag: None,
            }
        })
        .filter(|r| r.similarity > 0.5)
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.truncate(body.top_k);

    let processing_time = start.elapsed().as_millis() as u64;

    Ok(Json(QueryResponse {
        query: body.query.clone(),
        mode: mode.as_str().to_string(),
        results,
        answer: None,
        confidence: None,
        key_entities: None,
        reasoning_steps: None,
        sources: None,
        processing_time_ms: processing_time,
        backend: "memory".to_string(),
    }))
}

/// Graph-aware query path. Dispatches to `GraphRAG::ask`, `ask_explained`,
/// or `ask_with_reasoning` depending on `mode`. Always also runs a vector
/// search in parallel so the caller gets `results` (source excerpts) even
/// when the LLM call drives the `answer`.
async fn graph_aware_query(
    state: &AppState,
    body: &QueryRequest,
    mode: QueryMode,
    start: std::time::Instant,
) -> Result<Json<QueryResponse>, ApiError> {
    // Pre-compute vector hits (best-effort; failures don't block the
    // graph path because `answer` is the primary signal here).
    let vfilter = VersionFilter::from_request(body);
    let vector_results: Vec<QueryResult> = {
        #[cfg(feature = "qdrant")]
        if let Some(qdrant) = &state.qdrant {
            match state.embeddings.load_full().generate_single(&body.query).await {
                Ok(embedding) => match version_aware_search(
                    qdrant.as_ref(),
                    embedding,
                    body.top_k,
                    &vfilter,
                )
                .await
                {
                    Ok(results) => results
                        .into_iter()
                        .map(|r| QueryResult {
                            document_id: r.id,
                            title: r.metadata.title,
                            similarity: r.score,
                            excerpt: if r.metadata.text.len() > 200 {
                                format!("{}...", &r.metadata.text[..200])
                            } else {
                                r.metadata.text
                            },
                            source: r.metadata.source,
                            line_start: r.metadata.line_start,
                            line_end: r.metadata.line_end,
                            heading_path: r.metadata.heading_path,
                            etag: r.metadata.block_hash.clone(),
                            block_id: r.metadata.block_id,
                        })
                        .collect(),
                    Err(_) => Vec::new(),
                },
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        }
        #[cfg(not(feature = "qdrant"))]
        {
            Vec::new()
        }
    };

    // Stale-context: lease the vector hits for this session so the
    // SSE stream can scope events. graphrag_aware_query may take
    // longer than the search-only path, but the lease write is
    // off the critical path (cheap SQLite write).
    record_session_leases(state, body.session_id.as_deref(), &vector_results).await;

    let mut graphrag_guard = state.graphrag.write().await;
    let graphrag = graphrag_guard.as_mut().ok_or_else(|| {
        ApiError::BadRequest(
            "Mode requires a configured chat backend. POST /config with \
             openai.enabled=true or ollama.enabled=true first."
                .to_string(),
        )
    })?;

    match mode {
        QueryMode::Ask => {
            let answer = graphrag.ask(&body.query).await.map_err(|e| {
                tracing::error!(error = %e, "ask() failed");
                ApiError::InternalError(format!("ask() failed: {}", e))
            })?;
            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(answer),
                confidence: None,
                key_entities: None,
                reasoning_steps: None,
                sources: None,
                processing_time_ms: processing_time,
                backend: "graphrag".to_string(),
            }))
        },
        QueryMode::Explain => {
            let explained = graphrag.ask_explained(&body.query).await.map_err(|e| {
                tracing::error!(error = %e, "ask_explained() failed");
                ApiError::InternalError(format!("ask_explained() failed: {}", e))
            })?;
            let sources: Vec<SourceReferenceDto> = explained
                .sources
                .iter()
                .map(|s| SourceReferenceDto {
                    id: s.id.clone(),
                    kind: match s.source_type {
                        graphrag_core::retrieval::SourceType::TextChunk => SourceKind::TextChunk,
                        graphrag_core::retrieval::SourceType::Entity => SourceKind::Entity,
                        graphrag_core::retrieval::SourceType::Relationship => {
                            SourceKind::Relationship
                        },
                        graphrag_core::retrieval::SourceType::Summary => SourceKind::Summary,
                    },
                    excerpt: s.excerpt.clone(),
                    relevance: s.relevance_score,
                })
                .collect();
            let reasoning_steps: Vec<ReasoningStepDto> = explained
                .reasoning_steps
                .iter()
                .map(|s| ReasoningStepDto {
                    step: s.step_number,
                    description: s.description.clone(),
                    entities_used: s.entities_used.clone(),
                    evidence: s.evidence_snippet.clone(),
                    confidence: s.confidence,
                })
                .collect();
            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(explained.answer.clone()),
                confidence: Some(explained.confidence),
                key_entities: Some(explained.key_entities.clone()),
                reasoning_steps: Some(reasoning_steps),
                sources: Some(sources),
                processing_time_ms: processing_time,
                backend: "graphrag".to_string(),
            }))
        },
        QueryMode::Reason => {
            let answer = graphrag
                .ask_with_reasoning(&body.query)
                .await
                .map_err(|e| {
                    tracing::error!(error = %e, "ask_with_reasoning() failed");
                    ApiError::InternalError(format!("ask_with_reasoning() failed: {}", e))
                })?;
            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(answer),
                confidence: None,
                key_entities: None,
                reasoning_steps: None,
                sources: None,
                processing_time_ms: processing_time,
                backend: "graphrag".to_string(),
            }))
        },
        QueryMode::Local => {
            // Microsoft GraphRAG `local_search` shape:
            //   1. Embed the user query through the EmbeddingService
            //      (same one the document path uses).
            //   2. Vector-search the entity sidecar collection for
            //      top-K seed entities.
            //   3. Hand those entity ids to graphrag-core, which
            //      expands to 1-hop neighbors, gathers mentioning
            //      chunks, and feeds the assembly to the chat backend.
            //
            // If Qdrant or the entity sidecar is unavailable (cold
            // start, no build has run yet), top_k_ids is empty —
            // graphrag-core will return "no relevant information"
            // rather than fabricating an answer.
            let mut seed_ids: Vec<graphrag_core::core::EntityId> = Vec::new();

            #[cfg(feature = "qdrant")]
            if let Some(qdrant) = state.qdrant.as_ref() {
                match state.embeddings.load_full().generate_single(&body.query).await {
                    Ok(query_embedding) => {
                        match qdrant.search_entities(query_embedding, body.top_k.max(5)).await {
                            Ok(hits) => {
                                seed_ids = hits
                                    .into_iter()
                                    .map(|(id, _)| graphrag_core::core::EntityId::new(id))
                                    .collect();
                            },
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "search_entities failed; mode=local will run with no seeds"
                                );
                            },
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "query embedding failed; mode=local will run with no seeds"
                        );
                    },
                }
            }

            let max_neighbors_per_seed = 5usize;
            let explained = graphrag
                .ask_with_seed_entities(&body.query, &seed_ids, max_neighbors_per_seed)
                .await
                .map_err(|e| {
                    tracing::error!(error = %e, "ask_with_seed_entities() failed");
                    ApiError::InternalError(format!(
                        "ask_with_seed_entities() failed: {}",
                        e
                    ))
                })?;

            let sources: Vec<SourceReferenceDto> = explained
                .sources
                .iter()
                .map(|s| SourceReferenceDto {
                    id: s.id.clone(),
                    kind: match s.source_type {
                        graphrag_core::retrieval::SourceType::TextChunk => SourceKind::TextChunk,
                        graphrag_core::retrieval::SourceType::Entity => SourceKind::Entity,
                        graphrag_core::retrieval::SourceType::Relationship => {
                            SourceKind::Relationship
                        },
                        graphrag_core::retrieval::SourceType::Summary => SourceKind::Summary,
                    },
                    excerpt: s.excerpt.clone(),
                    relevance: s.relevance_score,
                })
                .collect();
            let reasoning_steps: Vec<ReasoningStepDto> = explained
                .reasoning_steps
                .iter()
                .map(|s| ReasoningStepDto {
                    step: s.step_number,
                    description: s.description.clone(),
                    entities_used: s.entities_used.clone(),
                    evidence: s.evidence_snippet.clone(),
                    confidence: s.confidence,
                })
                .collect();
            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(explained.answer.clone()),
                confidence: Some(explained.confidence),
                key_entities: Some(explained.key_entities.clone()),
                reasoning_steps: Some(reasoning_steps),
                sources: Some(sources),
                processing_time_ms: processing_time,
                backend: "graphrag-local-search".to_string(),
            }))
        },
        QueryMode::Global | QueryMode::Hybrid | QueryMode::Mix => {
            // LightRAG dual-level retrieval (arXiv:2410.05779).
            //
            // Pipeline:
            //   1. One LLM call extracts {low_level_keywords, high_level_keywords}
            //      from the user query.
            //   2. Each non-empty keyword set is joined into a single
            //      embed string, embedded once via OVMS/EmbeddingService,
            //      and used to vector-search the appropriate sidecar:
            //        - low-level  → graphrag-entities sidecar       (entity seeds)
            //        - high-level → graphrag-relationships sidecar  (relation seeds)
            //   3. For mode=mix, ALSO chunk-vector search with the
            //      original query → top-K chunk seeds.
            //   4. Hand the assembled `DualSeeds` to graphrag-core's
            //      ask_with_dual_seeds, which expands every seed
            //      (entities + relation endpoints), gathers mentioning
            //      chunks, and asks the chat backend for a synthesized
            //      answer.
            //
            // Mode-to-stream mapping:
            //   - global : relations only       (high-level keywords)
            //   - hybrid : entities + relations (both keyword sets)
            //   - mix    : entities + relations + chunk-vector
            let kw = graphrag.extract_query_keywords(&body.query).await.map_err(|e| {
                tracing::error!(error = %e, "extract_query_keywords() failed");
                ApiError::InternalError(format!("extract_query_keywords() failed: {}", e))
            })?;

            let mut seeds = graphrag_core::DualSeeds::default();

            #[cfg(feature = "qdrant")]
            if let Some(qdrant) = state.qdrant.as_ref() {
                // Low-level → entity sidecar (skip for global, which is
                // relation-only by definition).
                if !matches!(mode, QueryMode::Global) && !kw.low_level.is_empty() {
                    let low_text = kw.low_level.join(" ");
                    if let Ok(emb) = state.embeddings.load_full().generate_single(&low_text).await {
                        if let Ok(hits) = qdrant.search_entities(emb, body.top_k.max(5)).await {
                            seeds.entities = hits
                                .into_iter()
                                .map(|(id, _)| graphrag_core::core::EntityId::new(id))
                                .collect();
                        }
                    }
                }
                // High-level → relationship sidecar (driven by both
                // global and hybrid).
                if !kw.high_level.is_empty() {
                    let high_text = kw.high_level.join(" ");
                    if let Ok(emb) = state.embeddings.load_full().generate_single(&high_text).await {
                        if let Ok(hits) =
                            qdrant.search_relationships(emb, body.top_k.max(5)).await
                        {
                            seeds.relations = hits
                                .into_iter()
                                .map(|((s, t, r), _)| {
                                    (
                                        graphrag_core::core::EntityId::new(s),
                                        graphrag_core::core::EntityId::new(t),
                                        r,
                                    )
                                })
                                .collect();
                        }
                    }
                }
                // Mix mode also pulls a fresh chunk-vector pass.
                if matches!(mode, QueryMode::Mix) {
                    if let Ok(emb) = state.embeddings.load_full().generate_single(&body.query).await {
                        if let Ok(hits) = version_aware_search(
                            qdrant.as_ref(),
                            emb,
                            body.top_k.max(5),
                            &vfilter,
                        )
                        .await
                        {
                            // Caller-side chunk ids — Qdrant point ids
                            // for chunks ARE the document ids (one
                            // chunk per doc today; the mapping holds
                            // even if that changes).
                            seeds.chunks = hits
                                .into_iter()
                                .map(|r| graphrag_core::core::ChunkId::new(r.id))
                                .collect();
                        }
                    }
                }
            }

            let max_neighbors_per_seed = 5usize;
            let explained = graphrag
                .ask_with_dual_seeds(&body.query, &seeds, max_neighbors_per_seed)
                .await
                .map_err(|e| {
                    tracing::error!(error = %e, "ask_with_dual_seeds() failed");
                    ApiError::InternalError(format!("ask_with_dual_seeds() failed: {}", e))
                })?;

            // Prepend a reasoning step that documents the keyword
            // extraction itself so callers can audit which keywords
            // drove retrieval.
            let mut reasoning_steps: Vec<ReasoningStepDto> = vec![ReasoningStepDto {
                step: 0,
                description: format!(
                    "LightRAG dual-keyword extraction: low_level={:?}, high_level={:?}",
                    kw.low_level, kw.high_level
                ),
                entities_used: vec![],
                evidence: None,
                confidence: 1.0,
            }];
            reasoning_steps.extend(
                explained
                    .reasoning_steps
                    .iter()
                    .map(|s| ReasoningStepDto {
                        step: s.step_number,
                        description: s.description.clone(),
                        entities_used: s.entities_used.clone(),
                        evidence: s.evidence_snippet.clone(),
                        confidence: s.confidence,
                    }),
            );

            let sources: Vec<SourceReferenceDto> = explained
                .sources
                .iter()
                .map(|s| SourceReferenceDto {
                    id: s.id.clone(),
                    kind: match s.source_type {
                        graphrag_core::retrieval::SourceType::TextChunk => SourceKind::TextChunk,
                        graphrag_core::retrieval::SourceType::Entity => SourceKind::Entity,
                        graphrag_core::retrieval::SourceType::Relationship => {
                            SourceKind::Relationship
                        },
                        graphrag_core::retrieval::SourceType::Summary => SourceKind::Summary,
                    },
                    excerpt: s.excerpt.clone(),
                    relevance: s.relevance_score,
                })
                .collect();

            let backend_label = match mode {
                QueryMode::Global => "graphrag-lightrag-global",
                QueryMode::Hybrid => "graphrag-lightrag-hybrid",
                QueryMode::Mix => "graphrag-lightrag-mix",
                _ => "graphrag-lightrag",
            };

            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(explained.answer.clone()),
                confidence: Some(explained.confidence),
                key_entities: Some(explained.key_entities.clone()),
                reasoning_steps: Some(reasoning_steps),
                sources: Some(sources),
                processing_time_ms: processing_time,
                backend: backend_label.to_string(),
            }))
        },
        QueryMode::Search => unreachable!("search dispatched outside graph_aware_query"),
    }
}

/// Outcome of a single text-body ingest. Distinguishes a fresh add,
/// a re-ingest that bumped the version, and a no-op dedup so the
/// per-path response can label its `status` without parsing message
/// strings.
enum IngestOutcome {
    /// First-ever write for this `user_id` (or no `user_id` given,
    /// content not seen before). Carries the assigned UUID.
    Ingested(String),
    /// Re-ingest of an existing `user_id` whose content_hash changed.
    /// Old chunks were marked `is_current = false`; new chunks
    /// written at `version + 1`. Carries the new UUID.
    Updated { id: String, version: u32 },
    /// content_hash already present (either same `user_id` + same
    /// content, or no `user_id` and a global hash collision). No
    /// new write. Carries the pre-existing UUID.
    Duplicate(String),
}

/// Cheap URI-shape check for the `source` field. Accepts anything
/// matching `<scheme>:<rest>` where scheme is alpha+alnum+-./. and
/// rest is non-empty. Deliberately permissive — we want
/// file://, https://, obsidian://, arxiv:, doi:, urn:, custom: all
/// to pass. Reject only blatant garbage (no colon, empty rest).
fn is_uri_like(s: &str) -> bool {
    let Some((scheme, rest)) = s.split_once(':') else {
        return false;
    };
    if scheme.is_empty() || rest.is_empty() {
        return false;
    }
    scheme.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
}

#[derive(Debug)]
struct BlockIngestOutcome {
    user_id: String,
    added: usize,
    superseded: usize,
}

/// Block-aware ingest. The caller (typically the Obsidian plugin)
/// owns the diff state and sends only the blocks that actually
/// changed plus a list of `removed_block_ids`. We:
///   1. mark old (user_id, block_id) tuples superseded for both
///      removed and changed blocks (one server-side flip per id);
///   2. embed each changed block — the embedded text gets a
///      contextual prefix `[title > h1 > h2]` prepended at embed
///      time only (the stored text stays clean, so recall excerpts
///      don't show the prefix);
///   3. insert one Qdrant point per block with full chunk metadata
///      including `block_id`, `block_hash`, `heading_path`, line
///      range, and `source`;
///   4. mirror the FULL doc content into the graphrag pipeline so
///      entity extraction still has whole-doc context (block-level
///      extraction would lose cross-section coreferences).
async fn ingest_blocks(
    state: &AppState,
    title: String,
    full_content: String,
    source: String,
    user_id: String,
    blocks: Vec<crate::models::BlockInput>,
    removed_block_ids: Vec<String>,
    _file_hash: Option<String>,
) -> Result<BlockIngestOutcome, ApiError> {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let mut superseded: usize = 0;
    let mut added: usize = 0;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Step 1: supersede removed blocks. For each removed block,
        // read the prior content first so the stale-context event
        // can carry the deleted text for the agent to reason about
        // ("the paragraph you cited has been removed").
        for bid in &removed_block_ids {
            let prior = qdrant
                .find_current_block(&user_id, bid)
                .await
                .map_err(|e| ApiError::InternalError(format!("read prior {bid} failed: {e}")))?;
            qdrant
                .mark_block_superseded(&user_id, bid)
                .await
                .map_err(|e| ApiError::InternalError(format!("supersede removed block {bid} failed: {e}")))?;
            superseded += 1;
            if let Some(prior_md) = prior {
                emit_stale_context_event(
                    state,
                    &timestamp,
                    bid,
                    &source,
                    &user_id,
                    events_store::ChangeType::Removed,
                    prior_md.block_hash.as_deref(),
                    None,
                    Some(prior_md.text.as_str()),
                    None,
                )
                .await;
            }
        }

        // Step 2 + 3: for each changed block, supersede prior version,
        // then embed + insert. Sequential to keep the embedding service
        // honest (we don't want to spam concurrent embed requests for
        // a 200-block document during a bulk reindex; the user's 24 GB
        // VRAM laptop is the bottleneck here).
        for block in blocks {
            // Read prior content (if any) before flipping is_current
            // so we can emit a delta-bearing event.
            let prior = qdrant
                .find_current_block(&user_id, &block.id)
                .await
                .map_err(|e| ApiError::InternalError(format!("read prior {} failed: {e}", block.id)))?;
            qdrant
                .mark_block_superseded(&user_id, &block.id)
                .await
                .map_err(|e| ApiError::InternalError(format!("supersede prior block {} failed: {e}", block.id)))?;

            // Contextual prefix at embed time only.
            let prefix = build_context_prefix(&title, &block.heading_path);
            let embed_text = if prefix.is_empty() {
                block.content.clone()
            } else {
                format!("{prefix}\n\n{}", block.content)
            };
            let embedding = state
                .embeddings
                .load_full()
                .generate_single(&embed_text)
                .await
                .map_err(|e| {
                    ApiError::InternalError(format!("embed block {} failed: {e}", block.id))
                })?;

            let chunk_uuid = uuid::Uuid::new_v4().to_string();
            let metadata = qdrant_store::DocumentMetadata {
                id: chunk_uuid.clone(),
                title: title.clone(),
                text: block.content.clone(),
                chunk_index: 0,
                entities: Vec::new(),
                relationships: Vec::new(),
                timestamp: timestamp.clone(),
                content_hash: Some(block.hash.clone()),
                user_id: Some(user_id.clone()),
                version: Some(1),
                valid_from: Some(timestamp.clone()),
                is_current: Some(true),
                source: Some(source.clone()),
                block_id: Some(block.id.clone()),
                block_hash: Some(block.hash.clone()),
                heading_path: block.heading_path.clone(),
                line_start: block.line_start,
                line_end: block.line_end,
                custom: HashMap::new(),
            };
            qdrant
                .add_document(&chunk_uuid, embedding, metadata)
                .await
                .map_err(|e| ApiError::InternalError(format!("insert block {} failed: {e}", block.id)))?;
            added += 1;

            // Emit stale-context event AFTER the new chunk lands so
            // any client receiving the SSE notification can read the
            // new content. Best-effort — failure here logs but does
            // not abort the ingest.
            let (change_type, old_etag, old_text) = match prior {
                Some(p) => (
                    events_store::ChangeType::Updated,
                    p.block_hash,
                    Some(p.text),
                ),
                None => (events_store::ChangeType::Added, None, None),
            };
            emit_stale_context_event(
                state,
                &timestamp,
                &block.id,
                &source,
                &user_id,
                change_type,
                old_etag.as_deref(),
                Some(block.hash.as_str()),
                old_text.as_deref(),
                Some(block.content.as_str()),
            )
            .await;
        }

        // Step 4: pipeline feed (whole-doc text). Keep entity extraction
        // doc-level so cross-section relations stay visible.
        {
            let mut g = state.graphrag.write().await;
            if let Some(ref mut graphrag) = *g {
                if let Err(e) = graphrag.add_document_from_text(&full_content) {
                    tracing::warn!(error = %e, source = %source, "graphrag pipeline feed failed");
                } else {
                    *state.graph_built.write().await = false;
                }
            }
        }
        state.auto_append_notify.notify_one();

        return Ok(BlockIngestOutcome { user_id, added, superseded });
    }

    // Memory fallback: just push the whole content.
    let document = Document {
        id: user_id.clone(),
        title,
        content: full_content,
        added_at: timestamp,
    };
    state.documents.write().await.push(document);
    *state.graph_built.write().await = false;
    state.auto_append_notify.notify_one();
    Ok(BlockIngestOutcome { user_id, added: 1, superseded: 0 })
}

/// Record a session's lease entries on the server. Called after each
/// recall when the client supplied a `session_id`; populates the
/// SQLite lease table with `(block_id, etag, retrieved_at)` per hit.
/// FIFO eviction past `maxLeasesPerSession` happens inside
/// `EventsStore::add_leases`. Best-effort — failures log at warn,
/// don't break the recall response.
async fn record_session_leases(
    state: &AppState,
    session_id: Option<&str>,
    results: &[crate::models::QueryResult],
) {
    let Some(session_id) = session_id else { return };
    let Some(store) = state.events_store.as_ref() else { return };
    let now = chrono::Utc::now().to_rfc3339();
    let entries: Vec<events_store::LeaseEntry> = results
        .iter()
        .filter_map(|r| {
            let block_id = r.block_id.as_ref()?.clone();
            let etag = r.etag.as_ref()?.clone();
            Some(events_store::LeaseEntry {
                block_id,
                etag,
                retrieved_at: now.clone(),
            })
        })
        .collect();
    if entries.is_empty() {
        return;
    }
    if let Err(e) = store.add_leases(session_id.to_string(), entries).await {
        tracing::warn!(error = %e, %session_id, "stale-context: lease write failed");
    }
}

/// Persist + broadcast a stale-context event. No-op when the events
/// store / broadcast bus aren't initialized (deployment without the
/// stale-context layer enabled). All failures are logged at warn —
/// never propagated — because event emit is best-effort and must
/// not break ingest.
#[allow(clippy::too_many_arguments)]
async fn emit_stale_context_event(
    state: &AppState,
    ts: &str,
    block_id: &str,
    source: &str,
    user_id: &str,
    change_type: events_store::ChangeType,
    old_etag: Option<&str>,
    new_etag: Option<&str>,
    old_text: Option<&str>,
    new_text: Option<&str>,
) {
    let Some(store) = state.events_store.as_ref() else { return };
    let bus = state.event_bus.clone();

    // Cap excerpts; compute a unified diff for "updated" events so
    // the agent can see exactly what changed inline.
    let max_chars = store.settings().delta_excerpt_chars;
    let trim = |s: &str| -> String {
        if s.chars().count() <= max_chars {
            s.to_string()
        } else {
            let mut acc = String::with_capacity(max_chars + 1);
            for (i, c) in s.chars().enumerate() {
                if i >= max_chars { break; }
                acc.push(c);
            }
            acc.push('…');
            acc
        }
    };
    let old_excerpt = old_text.map(trim);
    let new_excerpt = new_text.map(trim);
    let unified_diff = match (old_text, new_text) {
        (Some(o), Some(n)) => {
            // similar's TextDiff handles multi-line input; small-block
            // size means line-level granularity is plenty.
            Some(
                similar::TextDiff::from_lines(o, n)
                    .unified_diff()
                    .header("old", "new")
                    .to_string(),
            )
        },
        _ => None,
    };
    let delta = if old_excerpt.is_some() || new_excerpt.is_some() || unified_diff.is_some() {
        Some(events_store::Delta {
            old_excerpt,
            new_excerpt,
            unified_diff,
        })
    } else {
        None
    };

    let pending = events_store::PendingEvent {
        ts: ts.to_string(),
        block_id: block_id.to_string(),
        source: source.to_string(),
        user_id: Some(user_id.to_string()),
        change_type: change_type.clone(),
        old_etag: old_etag.map(str::to_string),
        new_etag: new_etag.map(str::to_string),
        delta: delta.clone(),
    };

    let id = match store.append_event(pending).await {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!(error = %e, block_id, "stale-context: append_event failed");
            return;
        },
    };
    if let Some(bus) = bus {
        let body = events_store::EventBody {
            id,
            ts: ts.to_string(),
            block_id: block_id.to_string(),
            source: source.to_string(),
            user_id: Some(user_id.to_string()),
            change_type,
            old_etag: old_etag.map(str::to_string),
            new_etag: new_etag.map(str::to_string),
            delta,
        };
        // .send() returns Err only when there are zero subscribers —
        // expected and harmless. Drop silently.
        let _ = bus.send(body);
    }
}

/// Build the embedding-time contextual prefix. Format:
///   `[Title > Section > Subsection]`
/// Empty when both title and heading_path are empty.
fn build_context_prefix(title: &str, heading_path: &[String]) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if !title.is_empty() { parts.push(title); }
    for h in heading_path { if !h.is_empty() { parts.push(h); } }
    if parts.is_empty() { return String::new(); }
    format!("[{}]", parts.join(" > "))
}

/// Ingest one fully-prepared text body. Centralizes the embed →
/// dedup-check → qdrant-write → graphrag-feed pipeline that used to
/// live inline in `add_document`. Both legacy `content` requests and
/// path-form requests funnel through here so dedup, embedding, and
/// graphrag mirroring stay identical.
async fn ingest_one_text(
    state: &AppState,
    title: String,
    content: String,
    user_id: Option<String>,
) -> Result<IngestOutcome, ApiError> {
    let id = uuid::Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();
    let content_hash = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(content.as_bytes());
        format!("{:x}", h.finalize())
    };

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Upsert-by-user_id: when the caller supplies a stable id
        // (path-form ingest defaults this to the absolute path), we
        // scope the dedup/upsert decision to that user_id. Without a
        // user_id we fall back to the legacy global content_hash
        // dedup so legacy `{title, content}` callers behave exactly
        // as before.
        let prior = if let Some(uid) = user_id.as_deref() {
            qdrant
                .find_current_by_user_id(uid)
                .await
                .map_err(|e| ApiError::InternalError(format!("user_id lookup failed: {e}")))?
        } else {
            None
        };

        // Determine version for the new write.
        let new_version: u32 = match &prior {
            Some((_, md)) => match md.content_hash.as_deref() {
                Some(h) if h == content_hash => {
                    // Same user_id, same content — true no-op dedup.
                    let existing_id = prior.as_ref().unwrap().0.clone();
                    tracing::info!(
                        user_id = %user_id.as_deref().unwrap_or(""),
                        existing = %existing_id,
                        "ingest: content_hash matches current version; no-op"
                    );
                    return Ok(IngestOutcome::Duplicate(existing_id));
                },
                _ => md.version.unwrap_or(1).saturating_add(1),
            },
            None => {
                // No user_id-keyed prior. Fall back to the legacy
                // global content_hash check so re-ingest of the same
                // content via `add_document {content}` (no id) still
                // dedupes globally.
                if user_id.is_none() {
                    if let Ok(Some((existing_id, _))) =
                        qdrant.find_by_content_hash(&content_hash).await
                    {
                        tracing::info!(
                            existing = %existing_id,
                            "ingest: content_hash global match (legacy no-user_id path)"
                        );
                        return Ok(IngestOutcome::Duplicate(existing_id));
                    }
                }
                1
            },
        };

        // If this is a real upsert (prior with different content),
        // mark the old chunks superseded BEFORE we write the new
        // ones. set_payload is filter-targeted so it touches only
        // chunks tagged with this user_id + is_current=true.
        let is_upsert = prior.is_some();
        if is_upsert {
            if let Some(uid) = user_id.as_deref() {
                qdrant.mark_user_id_superseded(uid).await.map_err(|e| {
                    ApiError::InternalError(format!("supersede prior version failed: {e}"))
                })?;
            }
        }

        let embedding = state
            .embeddings
            .load_full()
            .generate_single(&content)
            .await
            .map_err(|e| {
                tracing::error!("Failed to generate document embedding: {}", e);
                ApiError::InternalError(format!("Failed to generate embedding: {}", e))
            })?;

        // Auto-derive source for path-form (user_id is the absolute
        // path string in that case) so every persisted point now
        // carries provenance — needed for the recall response to
        // include `source` uniformly.
        let derived_source = user_id.as_deref().and_then(|u| {
            if u.starts_with('/') { Some(format!("file://{u}")) } else { None }
        });
        let metadata = DocumentMetadata {
            id: id.clone(),
            title: title.clone(),
            text: content.clone(),
            chunk_index: 0,
            entities: Vec::new(),
            relationships: Vec::new(),
            timestamp: timestamp.clone(),
            content_hash: Some(content_hash.clone()),
            user_id: user_id.clone(),
            source: derived_source,
            block_id: None,
            block_hash: None,
            heading_path: Vec::new(),
            line_start: None,
            line_end: None,
            version: Some(new_version),
            valid_from: Some(timestamp.clone()),
            is_current: Some(true),
            custom: HashMap::new(),
        };

        qdrant
            .add_document(&id, embedding, metadata)
            .await
            .map_err(|e| ApiError::InternalError(format!("Failed to add document to Qdrant: {}", e)))?;

        if is_upsert {
            tracing::info!(
                user_id = %user_id.as_deref().unwrap_or(""),
                version = new_version,
                id = %id,
                "Updated document in Qdrant: {}",
                title
            );
        } else {
            tracing::info!("Added document to Qdrant: {} ({})", title, id);
        }

        {
            let mut g = state.graphrag.write().await;
            if let Some(ref mut graphrag) = *g {
                if let Err(e) = graphrag.add_document_from_text(&content) {
                    tracing::warn!(
                        error = %e,
                        "GraphRAG ingest failed; /api/graph/build will skip this doc"
                    );
                } else {
                    *state.graph_built.write().await = false;
                }
            }
        }

        // Wake the auto-append coalescer. notify_one stores at most
        // one permit, so a 200-file burst → many wakes → still one
        // append after the debounce window of silence.
        state.auto_append_notify.notify_one();
        if is_upsert {
            return Ok(IngestOutcome::Updated { id, version: new_version });
        } else {
            return Ok(IngestOutcome::Ingested(id));
        }
    }

    // Fallback: in-memory storage
    let document = Document {
        id: id.clone(),
        title: title.clone(),
        content: content.clone(),
        added_at: timestamp,
    };

    state.documents.write().await.push(document);
    *state.graph_built.write().await = false;

    tracing::info!("Added document to memory: {} ({})", title, id);
    state.auto_append_notify.notify_one();
    Ok(IngestOutcome::Ingested(id))
}

/// POST a non-text file path to the configured preprocessor service
/// and parse `{ "markdown": "...", "title"?: "..." }` from the
/// response. Used only when the file extension is not in the
/// allow-list AND `INGEST_PREPROCESSOR_URL` is set.
///
/// The preprocessor is expected to read the file itself (it sits on
/// the same machine) and return clean markdown — see
/// `graphrag-rs-nix/TODO.md` § "Multimodal preprocessor" for the full
/// contract and the planned Nemotron-Omni implementation.
async fn call_preprocessor(
    url: &str,
    absolute: &std::path::Path,
) -> Result<(Option<String>, String), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("preprocessor client build failed: {e}"))?;

    let req_body = serde_json::json!({
        "path": absolute.to_string_lossy(),
    });

    let resp = client
        .post(url)
        .json(&req_body)
        .send()
        .await
        .map_err(|e| format!("preprocessor request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("preprocessor returned {status}: {body}"));
    }

    let parsed: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("preprocessor returned non-JSON: {e}"))?;

    let markdown = parsed
        .get("markdown")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "preprocessor response missing `markdown` field".to_string())?
        .to_string();
    let title = parsed
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok((title, markdown))
}

/// Add document(s) to the knowledge graph.
///
/// Body is polymorphic — exactly one of `content`, `path`, `paths`,
/// or `pathsGlob` must be set:
///
/// * `content` (legacy) — inline body; requires `title`. Returns
///   `DocumentOperationResponse`.
/// * `path` — server reads one file off disk under
///   `INGEST_ALLOWED_ROOTS`. Returns `DocumentOperationResponse`.
/// * `paths` / `pathsGlob` — multi-file ingest; non-text files route
///   through `INGEST_PREPROCESSOR_URL` when set, otherwise skipped.
///   Returns `AddDocumentsResponse` (per-path results array).
#[api_operation(
    tag = "documents",
    summary = "Add a new document or batch of documents",
    description = "Ingest one document (legacy `content`/`title`), one file off disk (`path`), an explicit list of files (`paths`), or a glob expansion (`pathsGlob`). Path-form requests are sandboxed by INGEST_ALLOWED_ROOTS; non-text files route through INGEST_PREPROCESSOR_URL when set.",
    error_code = 400,
    error_code = 500
)]
async fn add_document(
    state: Data<AppState>,
    body: Json<AddDocumentRequest>,
) -> Result<Json<DocumentOperationResponse>, ApiError> {
    // Variant arbitration. Exactly one body flavor must be set; both
    // zero and >1 are user errors.
    let n_set = (body.content.is_some() as u8)
        + (body.path.is_some() as u8)
        + (body.paths.is_some() as u8)
        + (body.paths_glob.is_some() as u8);
    if n_set == 0 {
        return Err(ApiError::BadRequest(
            "POST /api/documents requires one of: content, path, paths, pathsGlob".into(),
        ));
    }
    if n_set > 1 {
        return Err(ApiError::BadRequest(
            "POST /api/documents accepts only one of: content, path, paths, pathsGlob".into(),
        ));
    }

    // ---- Branch A: content form (legacy whole-doc OR block-aware) ----
    if let Some(content) = body.content.as_ref() {
        let title = body.title.clone().ok_or_else(|| {
            ApiError::BadRequest("`title` is required when ingesting via `content`".into())
        })?;
        if let Err(e) = validate_title(&title) {
            tracing::warn!(title = %title, error = %e.error, "Invalid title");
            return Err(ApiError::BadRequest(e.error));
        }
        if let Err(e) = validate_content(content) {
            tracing::warn!(content_len = content.len(), error = %e.error, "Invalid content");
            return Err(ApiError::BadRequest(e.error));
        }
        // Phase B: `source` is required for `content`-form ingest. URI
        // form is up to the caller (https://, file://, obsidian://, etc.)
        // — we just check it parses as `<scheme>:<rest>` so provenance
        // is structured.
        let source = body.source.clone().ok_or_else(|| {
            ApiError::BadRequest(
                "`source` is required for content-form ingest (URI for provenance, e.g. https://… or obsidian://vault/…)".into(),
            )
        })?;
        if !is_uri_like(&source) {
            return Err(ApiError::BadRequest(format!(
                "`source` must be URI-shaped (got {:?})",
                source
            )));
        }

        // Block-aware path: when `blocks` is set, do surgical chunk
        // ingest in qdrant (one point per block) plus full-content
        // pipeline feed for entity extraction.
        if let Some(blocks) = body.blocks.clone() {
            let outcome = ingest_blocks(
                &state,
                sanitize_string(&title),
                sanitize_string(content),
                source.clone(),
                body.id.clone().unwrap_or_else(|| source.clone()),
                blocks,
                body.removed_block_ids.clone().unwrap_or_default(),
                body.file_hash.clone(),
            )
            .await?;
            let backend = if state.has_qdrant() { "qdrant" } else { "memory" };
            return Ok(Json(DocumentOperationResponse {
                success: true,
                document_id: Some(outcome.user_id),
                message: Some(format!(
                    "block-aware ingest: {} chunk(s) added, {} superseded",
                    outcome.added, outcome.superseded
                )),
                backend: backend.into(),
                results: None,
                ingested_count: Some(outcome.added),
                skipped_count: None,
            }));
        }

        let outcome = ingest_one_text(
            &state,
            sanitize_string(&title),
            sanitize_string(content),
            body.id.clone(),
        )
        .await?;
        let backend = if state.has_qdrant() { "qdrant" } else { "memory" };
        let resp = match outcome {
            IngestOutcome::Ingested(id) => DocumentOperationResponse {
                success: true,
                document_id: Some(id),
                message: Some(format!("Document added to {} successfully", backend)),
                backend: backend.into(),
                results: None,
                ingested_count: None,
                skipped_count: None,
            },
            IngestOutcome::Updated { id, version } => DocumentOperationResponse {
                success: true,
                document_id: Some(id),
                message: Some(format!(
                    "Document updated (now version {}); prior version chunks marked superseded",
                    version
                )),
                backend: backend.into(),
                results: None,
                ingested_count: None,
                skipped_count: None,
            },
            IngestOutcome::Duplicate(id) => DocumentOperationResponse {
                success: true,
                document_id: Some(id),
                message: Some("Document already indexed (content_hash match)".into()),
                backend: backend.into(),
                results: None,
                ingested_count: None,
                skipped_count: None,
            },
        };
        return Ok(Json(resp));
    }

    // ---- Path-form requests — require policy enabled ----
    if !state.ingest_policy.enabled() {
        return Err(ApiError::BadRequest(
            "path-based ingestion disabled (no INGEST_ALLOWED_ROOTS configured)".into(),
        ));
    }

    // Build the list of caller-input strings to resolve.
    let inputs: Vec<String> = if let Some(p) = body.path.as_ref() {
        vec![p.clone()]
    } else if let Some(ps) = body.paths.as_ref() {
        ps.clone()
    } else if let Some(pat) = body.paths_glob.as_ref() {
        match state
            .ingest_policy
            .expand_glob(pat, body.glob_root.as_deref())
        {
            Ok(v) => v,
            Err(e) => return Err(ApiError::BadRequest(e)),
        }
    } else {
        unreachable!("variant arbitration above")
    };

    let mut results: Vec<AddDocumentItemResult> = Vec::with_capacity(inputs.len());
    let mut ingested_count: usize = 0;
    let mut skipped_count: usize = 0;
    let backend_label = if state.has_qdrant() { "qdrant" } else { "memory" };

    for input in inputs {
        let resolved = state.ingest_policy.resolve(&input);

        // Fetch text content (either by reading the file, or by
        // calling the preprocessor). Per-item failures are recorded
        // and we move on; one bad path doesn't abort the batch.
        let (absolute, title_default, fetched): (
            std::path::PathBuf,
            String,
            Result<String, String>,
        ) = match resolved {
            ResolvedPath::Text { absolute } => {
                let title_default = absolute
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("untitled")
                    .to_string();
                let read = tokio::fs::read_to_string(&absolute)
                    .await
                    .map_err(|e| format!("read failed: {e}"));
                (absolute, title_default, read)
            },
            ResolvedPath::Preprocess { absolute, preprocessor_url } => {
                let title_default = absolute
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("untitled")
                    .to_string();
                let pre = call_preprocessor(&preprocessor_url, &absolute).await;
                match pre {
                    Ok((preprocessor_title, markdown)) => {
                        let td = preprocessor_title.unwrap_or(title_default);
                        (absolute, td, Ok(markdown))
                    },
                    Err(e) => (absolute, title_default, Err(e)),
                }
            },
            ResolvedPath::Unsupported { absolute, extension } => {
                results.push(AddDocumentItemResult {
                    path: absolute.to_string_lossy().into_owned(),
                    status: "unsupported".into(),
                    document_id: None,
                    title: None,
                    error: Some(format!(
                        ".{extension} is not in INGEST_ALLOWED_EXTENSIONS and no INGEST_PREPROCESSOR_URL is configured"
                    )),
                });
                skipped_count += 1;
                continue;
            },
            ResolvedPath::Rejected { input, reason } => {
                results.push(AddDocumentItemResult {
                    path: input,
                    status: "rejected".into(),
                    document_id: None,
                    title: None,
                    error: Some(reason),
                });
                skipped_count += 1;
                continue;
            },
        };

        let abs_str = absolute.to_string_lossy().into_owned();
        let raw = match fetched {
            Ok(t) => t,
            Err(e) => {
                results.push(AddDocumentItemResult {
                    path: abs_str,
                    status: "error".into(),
                    document_id: None,
                    title: Some(title_default),
                    error: Some(e),
                });
                skipped_count += 1;
                continue;
            },
        };

        let title_s = sanitize_string(&title_default);
        let content_s = sanitize_string(&raw);
        if let Err(e) = validate_content(&content_s) {
            results.push(AddDocumentItemResult {
                path: abs_str,
                status: "rejected".into(),
                document_id: None,
                title: Some(title_s),
                error: Some(e.error),
            });
            skipped_count += 1;
            continue;
        }

        // For multi-path requests, the optional caller `id` is
        // ambiguous (only one slot, many docs). Use the canonical
        // path string instead so each doc has a stable id; for
        // single `path` form, fall back to the caller id when given.
        let user_id = if body.path.is_some() {
            body.id.clone().or_else(|| Some(abs_str.clone()))
        } else {
            Some(abs_str.clone())
        };

        match ingest_one_text(&state, title_s.clone(), content_s, user_id).await {
            Ok(IngestOutcome::Ingested(doc_id)) => {
                ingested_count += 1;
                results.push(AddDocumentItemResult {
                    path: abs_str,
                    status: "ingested".into(),
                    document_id: Some(doc_id),
                    title: Some(title_s),
                    error: None,
                });
            },
            Ok(IngestOutcome::Updated { id: doc_id, version: _ }) => {
                ingested_count += 1;
                results.push(AddDocumentItemResult {
                    path: abs_str,
                    status: "updated".into(),
                    document_id: Some(doc_id),
                    title: Some(title_s),
                    error: None,
                });
            },
            Ok(IngestOutcome::Duplicate(doc_id)) => {
                skipped_count += 1;
                results.push(AddDocumentItemResult {
                    path: abs_str,
                    status: "duplicate".into(),
                    document_id: Some(doc_id),
                    title: Some(title_s),
                    error: None,
                });
            },
            Err(e) => {
                skipped_count += 1;
                results.push(AddDocumentItemResult {
                    path: abs_str,
                    status: "error".into(),
                    document_id: None,
                    title: Some(title_s),
                    error: Some(e.to_string()),
                });
            },
        }
    }

    // Single-path convenience: when caller asked for one `path`
    // (not `paths`/`pathsGlob`), return the legacy single-doc
    // response shape so simple clients don't have to dig into
    // `results[0]`. Multi-path requests always return the array form.
    if body.path.is_some() && results.len() == 1 {
        let r = &results[0];
        let resp = DocumentOperationResponse {
            success: matches!(r.status.as_str(), "ingested" | "updated" | "duplicate"),
            document_id: r.document_id.clone(),
            message: Some(match r.status.as_str() {
                "ingested" => format!("Document added to {} successfully", backend_label),
                "updated" => "Document updated; prior version chunks marked superseded".into(),
                "duplicate" => "Document already indexed (content_hash match)".into(),
                other => r
                    .error
                    .clone()
                    .unwrap_or_else(|| format!("ingest status: {other}")),
            }),
            backend: backend_label.into(),
            results: None,
            ingested_count: None,
            skipped_count: None,
        };
        return Ok(Json(resp));
    }

    let any_success = ingested_count > 0;
    let resp = DocumentOperationResponse {
        success: any_success,
        document_id: None,
        message: None,
        backend: backend_label.into(),
        results: Some(results),
        ingested_count: Some(ingested_count),
        skipped_count: Some(skipped_count),
    };
    Ok(Json(resp))
}

/// List all documents
#[api_operation(
    tag = "documents",
    summary = "List all documents",
    description = "Retrieve a list of all documents in the knowledge graph"
)]
async fn list_documents(state: Data<AppState>) -> Json<ListDocumentsResponse> {
    // Hard cap on the page size: ingesters can drive the corpus to
    // many thousands of points. 256 is plenty for an agent inspecting
    // what's indexed; deeper enumeration should use search.
    const LIST_LIMIT: u32 = 256;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Get total count separately — list_documents pages through
        // the collection but a separate /count is one cheap call.
        let total = match qdrant.stats().await {
            Ok((c, _)) => c,
            Err(e) => {
                tracing::warn!("Qdrant stats failed: {}", e);
                0
            },
        };
        match qdrant.list_documents(LIST_LIMIT).await {
            Ok(rows) => {
                let documents: Vec<DocumentSummary> = rows
                    .into_iter()
                    .map(|r| DocumentSummary {
                        id: r.id,
                        user_id: r.user_id,
                        title: r.title,
                        content_length: None,
                        excerpt: Some(r.excerpt),
                        added_at: r.timestamp,
                    })
                    .collect();
                let truncated = (documents.len() as u32) >= LIST_LIMIT && total > documents.len();
                return Json(ListDocumentsResponse {
                    documents,
                    total,
                    backend: "qdrant".to_string(),
                    note: if truncated {
                        Some(format!(
                            "Showing first {} of {} documents — use search to drill in",
                            LIST_LIMIT, total
                        ))
                    } else {
                        None
                    },
                });
            },
            Err(e) => {
                tracing::error!("Qdrant list_documents failed: {}", e);
            },
        }
    }

    // Fallback: in-memory storage
    let documents = state.documents.read().await;

    let doc_list: Vec<DocumentSummary> = documents
        .iter()
        .map(|doc| DocumentSummary {
            id: doc.id.clone(),
            user_id: None,
            title: doc.title.clone(),
            content_length: Some(doc.content.len()),
            excerpt: None,
            added_at: doc.added_at.clone(),
        })
        .collect();

    Json(ListDocumentsResponse {
        documents: doc_list.clone(),
        total: doc_list.len(),
        backend: "memory".to_string(),
        note: None,
    })
}

/// GET /embeddings/stats
/// Reports the configured embedding backend (sourced from
/// `state.config.embeddings` — single source of truth, can't disagree
/// with `GET /config`) plus runtime counters from the live
/// `EmbeddingService`. Plain Actix handler (no apistos
/// `#[api_operation]`) — registered below `.build()` to avoid the
/// `PathItemDefinition` trait bound (same workaround as `/config`).
async fn embeddings_stats(state: Data<AppState>) -> Json<serde_json::Value> {
    let svc = state.embeddings.load_full();
    let stats = svc.get_stats().await;
    let cfg = state.config.read().await;
    Json(json!({
        "backend": cfg.embeddings.backend,
        "model": cfg.embeddings.model.clone().unwrap_or_default(),
        "dimension": cfg.embeddings.dimension,
        "endpoint": cfg.embeddings.api_endpoint.clone().unwrap_or_default(),
        "live": svc.backend_live(),
        "stats": {
            "total_requests": stats.total_requests,
            "success": stats.backend_success,
            "failures": stats.backend_failures,
            "fallback_used": stats.fallback_used,
            "cache_hits": stats.cache_hits,
        },
    }))
}

/// Delete a document
#[api_operation(
    tag = "documents",
    summary = "Delete a document",
    description = "Remove a document from the knowledge graph by ID",
    error_code = 404,
    error_code = 500
)]
async fn delete_document(
    state: Data<AppState>,
    id: WebPath<String>,
) -> Result<Json<DocumentOperationResponse>, ApiError> {
    let supplied = id.into_inner();

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Two strategies:
        //  1. If the supplied id matches any payload's `user_id` we
        //     `delete_by_user_id` — purges every chunk under that id
        //     (current AND superseded historical versions). This is
        //     what the watcher's REMOVE handler relies on; without
        //     it, only the most-recent point would be removed and
        //     old superseded chunks would keep their `user_id` tag,
        //     so a future re-create of the same path would think
        //     it's an upsert against ghost history.
        //  2. Otherwise treat the supplied string as a raw Qdrant
        //     point UUID and delete just that one point. Back-compat
        //     for callers that delete by the auto-assigned UUID.
        let user_id_match = qdrant
            .find_id_by_user_id(&supplied)
            .await
            .ok()
            .flatten()
            .is_some();

        if user_id_match {
            match qdrant.delete_by_user_id(&supplied).await {
                Ok(_) => {
                    tracing::info!(
                        "Deleted document by user_id from Qdrant: {}",
                        supplied
                    );
                    return Ok(Json(DocumentOperationResponse {
                        success: true,
                        document_id: Some(supplied.clone()),
                        message: Some(format!(
                            "Deleted all chunks under user_id '{}' (current + superseded)",
                            supplied
                        )),
                        backend: "qdrant".to_string(),
                        results: None,
                        ingested_count: None,
                        skipped_count: None,
                    }));
                },
                Err(e) => {
                    return Err(ApiError::InternalError(format!(
                        "Failed to delete by user_id: {}",
                        e
                    )));
                },
            }
        }

        // Treat anything not a UUID as "not found" rather than
        // forwarding to qdrant where it would error with
        // `Unable to parse UUID`. The watcher fires DELETEs on every
        // REMOVE event, including paths that were never ingested
        // (e.g. .git/index in a watched repo, files filtered out by
        // `allowedExtensions`); without this guard, those crash the
        // request with 500 and produce noisy logs. 404 is the right
        // shape: the doc isn't there, ack it and move on.
        if uuid::Uuid::parse_str(&supplied).is_err() {
            tracing::info!("Delete: id '{}' not found (no user_id match, not a UUID)", supplied);
            return Err(ApiError::NotFound(format!(
                "Document with id '{}' not found",
                supplied
            )));
        }

        match qdrant.delete_document(&supplied).await {
            Ok(_) => {
                tracing::info!("Deleted document from Qdrant: {}", supplied);
                return Ok(Json(DocumentOperationResponse {
                    success: true,
                    document_id: Some(supplied.clone()),
                    message: Some(format!("Document {} deleted from Qdrant", supplied)),
                    backend: "qdrant".to_string(),
                    results: None,
                    ingested_count: None,
                    skipped_count: None,
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Failed to delete from Qdrant: {}",
                    e
                )));
            },
        }
    }

    // Fallback: in-memory storage
    let mut documents = state.documents.write().await;
    let original_len = documents.len();
    documents.retain(|doc| doc.id != supplied);

    if documents.len() == original_len {
        return Err(ApiError::NotFound(format!(
            "Document with id '{}' not found",
            supplied
        )));
    }

    *state.graph_built.write().await = false;
    tracing::info!("Deleted document from memory: {}", supplied);

    Ok(Json(DocumentOperationResponse {
        success: true,
        document_id: Some(supplied.clone()),
        message: Some(format!("Document {} deleted from memory", supplied)),
        backend: "memory".to_string(),
        results: None,
        ingested_count: None,
        skipped_count: None,
    }))
}

/// Build the knowledge graph (full re-extraction; deprecated for routine use)
///
/// **Deprecated for routine use.** The server persists the entity graph to
/// Qdrant on every successful build/append and rehydrates it on startup,
/// so a full rebuild is no longer needed across restarts. The 30-minute
/// `/api/graph/append` cron handles new ingests. Reserve this endpoint
/// for explicit user requests or recovery after a config change
/// (entity_types, prompts, chat model swap). The endpoint stays mounted
/// for those cases — it isn't going away — but agents should prefer
/// `/api/graph/append` for everything routine.
#[api_operation(
    tag = "graph",
    summary = "Build the knowledge graph (DEPRECATED for routine use — prefer /api/graph/append)",
    description = "Full LLM re-extraction over the entire corpus. DEPRECATED for routine use — the entity graph now persists to Qdrant and rehydrates on startup, so manual rebuilds are not needed in normal operation. Use /api/graph/append (which the cron timer also calls) for incremental updates. Reserve this endpoint for explicit user-requested rebuilds or recovery after a config change.",
    deprecated = true,
    error_code = 400,
    error_code = 500
)]
async fn build_graph(state: Data<AppState>) -> Result<Json<BuildGraphResponse>, ApiError> {
    let start = std::time::Instant::now();

    // Try the real GraphRAG pipeline first
    {
        let mut graphrag_guard = state.graphrag.write().await;
        if let Some(ref mut graphrag) = *graphrag_guard {
            // Use actual pipeline to build graph
            match graphrag.build_graph().await {
                Ok(_) => {
                    let (entities, relationships, chunk_count) = graphrag
                        .knowledge_graph()
                        .map(|kg| (
                            kg.entities().count(),
                            kg.relationships().count(),
                            kg.chunks().count(),
                        ))
                        .unwrap_or((0, 0, 0));

                    // Persist the freshly-built graph to Qdrant so it
                    // survives a server restart. Best-effort: a Qdrant
                    // failure logs and continues — the in-memory graph
                    // is still usable for the rest of the session.
                    #[cfg(feature = "qdrant")]
                    if let Some(qdrant) = state.qdrant.as_ref() {
                        match graph_persistence::persist_in_memory_graph(graphrag, qdrant, state.embeddings.load_full().as_ref()).await
                        {
                            Ok((e, r)) => tracing::info!(
                                "💾 Persisted graph to Qdrant: {} entities, {} relationships",
                                e, r
                            ),
                            Err(err) => tracing::warn!(
                                error = %err,
                                "graph persistence failed; in-memory build is still good but won't survive restart"
                            ),
                        }
                    }

                    let processing_time = start.elapsed().as_millis() as u64;

                    *state.graph_built.write().await = true;
                    *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());
                    state
                        .processed_chunk_count
                        .store(chunk_count, std::sync::atomic::Ordering::SeqCst);

                    tracing::info!(
                        "Built knowledge graph via pipeline in {}ms ({} entities, {} relationships)",
                        processing_time, entities, relationships
                    );

                    return Ok(Json(BuildGraphResponse {
                        success: true,
                        document_count: state.documents.read().await.len(),
                        processing_time_ms: processing_time,
                        message: format!(
                            "Knowledge graph built: {} entities, {} relationships",
                            entities, relationships
                        ),
                        backend: "graphrag-pipeline".to_string(),
                    }));
                },
                Err(e) => {
                    tracing::warn!("GraphRAG pipeline build failed, trying fallback: {}", e);
                    // Fall through to lower-priority backends
                },
            }
        }
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                if count == 0 {
                    return Err(ApiError::BadRequest(
                        "No documents in Qdrant. Add documents first.".to_string(),
                    ));
                }

                let processing_time = start.elapsed().as_millis() as u64;

                tracing::info!(
                    "Built knowledge graph from {} Qdrant documents in {}ms",
                    count,
                    processing_time
                );

                *state.graph_built.write().await = true;
                *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());

                return Ok(Json(BuildGraphResponse {
                    success: true,
                    document_count: count,
                    processing_time_ms: processing_time,
                    message: "Knowledge graph built from Qdrant successfully".to_string(),
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Failed to access Qdrant: {}",
                    e
                )));
            },
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();

    if doc_count == 0 {
        return Err(ApiError::BadRequest(
            "No documents to build graph from. Add documents first.".to_string(),
        ));
    }

    *state.graph_built.write().await = true;
    *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());
    let processing_time = start.elapsed().as_millis() as u64;

    tracing::info!(
        "Built knowledge graph from {} memory documents in {}ms",
        doc_count,
        processing_time
    );

    Ok(Json(BuildGraphResponse {
        success: true,
        document_count: doc_count,
        processing_time_ms: processing_time,
        message: "Knowledge graph built from memory successfully".to_string(),
        backend: "memory".to_string(),
    }))
}

/// Append-extract entities for chunks ingested since the last build.
///
/// Semantic-equivalent of Microsoft GraphRAG's `graphrag append`: run
/// after a batch of /api/documents calls so newly-ingested content
/// shows up in queries, without paying for a wholesale re-extraction
/// of everything that was already indexed.
///
/// **Implementation note** (today): under the hood this still calls
/// `GraphRAG::build_graph()` because graphrag-core's `incremental`
/// module isn't yet wired into the runtime pipeline. The LLM-call
/// cache (`enable_caching = true`) makes repeat extraction near-free
/// for unchanged chunks, so the cost scales with new content rather
/// than corpus size — but it's not a true incremental update yet.
/// A follow-up will route this through `graphrag-core::incremental::
/// add_content` for genuine incremental behavior.
///
/// Fast-paths: returns `{success: true, document_count: 0,
/// message: "no new chunks since last build"}` immediately when the
/// chunk count hasn't grown since the previous build/append. Cheap
/// for cron-driven callers that fire periodically regardless of
/// whether anything new was ingested.
///
/// Internally calls `GraphRAG::extend_graph` — a real incremental
/// pass that only walks the chunks ingested since the last build /
/// extend, dedupes entities by id (mentions of an existing entity
/// extend its `mentions` in place rather than creating a duplicate
/// node), and merges relationships keyed by (source, target,
/// relation_type). Cost scales with the size of the delta, not with
/// the total corpus.
#[api_operation(
    tag = "graph",
    summary = "Append new chunks to the knowledge graph",
    description = "Run entity extraction on chunks ingested since the last build. Walks only the delta (no full rebuild), dedupes entities by id, merges relationships. Cheap no-op when nothing new. Use after a batch of /api/documents calls; do NOT call once per document.",
    error_code = 500
)]
async fn append_graph(state: Data<AppState>) -> Result<Json<BuildGraphResponse>, ApiError> {
    do_append_graph(&state).await.map(Json)
}

/// In-server append coalescer. Waits on `state.auto_append_notify`,
/// then sleeps `debounce` seconds, restarting the sleep on every new
/// notification. When the window holds quiet, calls `do_append_graph`
/// in-process. Replaces the previous home-manager 30-min cron timer.
///
/// Single-flight: while an append is running, additional ingests
/// fire `notify_one()`, which stores a single permit. Once the
/// append finishes, the next loop iteration picks up that permit
/// immediately and starts a fresh debounce window.
///
/// Errors are logged, not propagated — a transient chat-backend
/// outage shouldn't crash the server, and the next ingest will wake
/// us to retry.
async fn auto_append_loop(state: AppState, debounce: std::time::Duration) {
    tracing::info!(
        debounce_secs = debounce.as_secs(),
        "auto-append loop running (in-server coalescer; replaces 30-min cron)"
    );
    loop {
        // Wait for the first ingest signal of this round.
        state.auto_append_notify.notified().await;

        // Drain: extend the deadline as long as new signals keep
        // arriving inside the window. The loop only breaks when
        // `debounce` seconds have passed without a new notification —
        // i.e. the user has stopped typing.
        loop {
            match tokio::time::timeout(debounce, state.auto_append_notify.notified()).await {
                Ok(_) => continue,
                Err(_) => break,
            }
        }

        match do_append_graph(&state).await {
            Ok(resp) => tracing::info!("auto-append: {}", resp.message),
            Err(e) => tracing::warn!(
                error = %e,
                "auto-append failed; will retry on next ingest"
            ),
        }
    }
}

/// Periodic cleanup of the stale-context state: drop events past
/// the event retention window, sessions past the TTL, truncate the
/// SQLite WAL so disk usage shrinks. Runs every `interval` for the
/// lifetime of the process.
async fn stale_context_cleanup_loop(state: AppState, interval: std::time::Duration) {
    tracing::info!(
        interval_secs = interval.as_secs(),
        "stale-context cleanup loop running"
    );
    let mut ticker = tokio::time::interval(interval);
    // First tick fires immediately; skip it so we don't run on boot
    // (the DB is fresh; nothing to clean) and instead run after the
    // configured interval.
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let Some(store) = state.events_store.as_ref() else { return };
        match store.cleanup().await {
            Ok(report) => tracing::info!(
                events_dropped = report.events_dropped,
                leases_dropped = report.leases_dropped,
                sessions_dropped = report.sessions_dropped,
                events_remaining = report.events_remaining,
                sessions_remaining = report.sessions_remaining,
                "🧹 stale-context cleanup: pruned old events + sessions"
            ),
            Err(e) => tracing::warn!(error = %e, "stale-context cleanup failed; will retry"),
        }
    }
}

/// Inner extend-graph routine, shared by `POST /api/graph/append` and
/// the in-server `auto_append_loop` coalescer. Same observable
/// behavior as the HTTP handler — fast-paths on no-delta, persists
/// to Qdrant on success, updates `state.graph_built` /
/// `state.last_built_at` / `state.processed_chunk_count`. Returns
/// `BuildGraphResponse` so the HTTP route can serialize it directly
/// and the coalescer can log a structured summary.
async fn do_append_graph(state: &AppState) -> Result<BuildGraphResponse, ApiError> {
    let start = std::time::Instant::now();

    // Phase 1 (under-lock): run extend_graph (which mutates the
    // in-memory graph), then snapshot the touched delta into owned
    // data so we can release the lock before paying for the embed +
    // qdrant write. Mirrors LightRAG's approach where embedding is
    // never done while holding any storage lock.
    let (summary, snapshot, processed_chunks_after) = {
        let mut graphrag_guard = state.graphrag.write().await;
        let Some(graphrag) = graphrag_guard.as_mut() else {
            return Err(ApiError::BadRequest(
                "GraphRAG not initialized. Call POST /config first.".to_string(),
            ));
        };
        let summary = graphrag.extend_graph().await.map_err(|e| {
            ApiError::InternalError(format!("Append failed: {}", e))
        })?;
        // Empty-delta fast path stays under-lock: no snapshot needed.
        if summary.chunks_processed == 0 {
            let processing_time = start.elapsed().as_millis() as u64;
            return Ok(BuildGraphResponse {
                success: true,
                document_count: 0,
                processing_time_ms: processing_time,
                message: format!(
                    "No new chunks since last build ({} processed). Nothing to append.",
                    graphrag.processed_chunk_count()
                ),
                backend: "graphrag-pipeline".to_string(),
            });
        }
        #[cfg(feature = "qdrant")]
        let snapshot = graph_persistence::snapshot_touched(
            graphrag,
            &summary.touched_entity_ids,
            &summary.touched_relationship_keys,
        );
        #[cfg(not(feature = "qdrant"))]
        let snapshot = ();
        let processed_chunks_after = graphrag.processed_chunk_count();
        (summary, snapshot, processed_chunks_after)
    }; // ← lock released here

    // Phase 2 (LOCK-FREE): embed + qdrant write of the touched delta.
    // Recall paths can take the GraphRAG write-lock during this window
    // without blocking on us. Best-effort: persist failure logs and
    // continues — the in-memory graph is still good for the rest of
    // the session.
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = state.qdrant.as_ref() {
        let entity_count = snapshot.entities.len();
        let rel_count = snapshot.relationships.len();
        match graph_persistence::persist_touched_snapshot(
            snapshot,
            qdrant,
            state.embeddings.load_full().as_ref(),
        )
        .await
        {
            Ok((e, r)) => tracing::info!(
                "💾 Persisted delta to Qdrant: {} entities, {} relationships (touched only; graph unchanged in size)",
                e, r
            ),
            Err(err) => tracing::warn!(
                error = %err,
                touched_entities = entity_count,
                touched_relationships = rel_count,
                "graph persistence failed; in-memory append is still good but won't survive restart"
            ),
        }
    }

    let processing_time = start.elapsed().as_millis() as u64;

    *state.graph_built.write().await = true;
    *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());
    // Mirror processed_chunks count into AppState for /health
    // and /embeddings/stats consumers.
    state
        .processed_chunk_count
        .store(processed_chunks_after, std::sync::atomic::Ordering::SeqCst);

    tracing::info!(
        "extend_graph: {} delta chunks, +{} entities, +{} rels, {} mentions merged, {} touched-entities, {} touched-rels ({}ms; graph: {} entities, {} rels)",
        summary.chunks_processed,
        summary.new_entities,
        summary.new_relationships,
        summary.mentions_merged,
        summary.touched_entity_ids.len(),
        summary.touched_relationship_keys.len(),
        processing_time,
        summary.total_entities,
        summary.total_relationships,
    );

    Ok(BuildGraphResponse {
        success: true,
        document_count: summary.chunks_processed,
        processing_time_ms: processing_time,
        message: format!(
            "Appended {} new chunks: +{} entities, +{} relationships, {} mentions merged ({} entities, {} relationships total)",
            summary.chunks_processed,
            summary.new_entities,
            summary.new_relationships,
            summary.mentions_merged,
            summary.total_entities,
            summary.total_relationships,
        ),
        backend: "graphrag-pipeline".to_string(),
    })
}

/// Get graph statistics
#[api_operation(
    tag = "graph",
    summary = "Get graph statistics",
    description = "Retrieve statistics about the knowledge graph, including document count, entity count, and relationship count"
)]
async fn graph_stats(state: Data<AppState>) -> Json<GraphStatsResponse> {
    // Read last_built_at once; same value for every branch below.
    let last_built_at = state.last_built_at.read().await.clone();

    // Try real GraphRAG pipeline stats first
    {
        let graphrag_guard = state.graphrag.read().await;
        if let Some(ref graphrag) = *graphrag_guard {
            if let Some(kg) = graphrag.knowledge_graph() {
                let entity_count = kg.entities().count();
                let relationship_count = kg.relationships().count();
                let doc_count = kg.documents().count();
                let chunk_count = kg.chunks().count();

                return Json(GraphStatsResponse {
                    document_count: doc_count,
                    entity_count,
                    relationship_count,
                    vector_count: chunk_count,
                    graph_built: true,
                    last_built_at,
                    backend: "graphrag-pipeline".to_string(),
                });
            }
        }
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, vectors)) => {
                return Json(GraphStatsResponse {
                    document_count: count,
                    entity_count: 0,
                    relationship_count: 0,
                    vector_count: vectors,
                    graph_built: count > 0,
                    last_built_at,
                    backend: "qdrant".to_string(),
                });
            },
            Err(e) => {
                tracing::error!("Failed to get Qdrant stats: {}", e);
            },
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();
    let graph_built = *state.graph_built.read().await;

    Json(GraphStatsResponse {
        document_count: doc_count,
        entity_count: 0,
        relationship_count: 0,
        vector_count: 0,
        graph_built,
        last_built_at,
        backend: "memory".to_string(),
    })
}

// ============================================================================
// Authentication Endpoints (feature-gated)
// ============================================================================

#[cfg(feature = "auth")]
#[api_operation(
    tag = "auth",
    summary = "User login",
    description = "Authenticate user and receive JWT token",
    error_code = 401,
    error_code = 500
)]
async fn login(
    state: Data<AppState>,
    body: Json<LoginRequest>,
) -> Result<Json<LoginResponse>, ApiError> {
    // TODO: Implement real user authentication against database
    // For now, accept any credentials for demo purposes
    tracing::info!("Login attempt for user: {}", body.username);

    let role = if body.username == "admin" {
        auth::UserRole::Admin
    } else {
        auth::UserRole::User
    };

    match state.auth.generate_token(&body.username, role.clone(), 24) {
        Ok(token) => {
            tracing::info!(
                "✅ Generated JWT token for user: {} (role: {:?})",
                body.username,
                role
            );
            Ok(Json(LoginResponse {
                success: true,
                token,
                user_id: body.username.clone(),
                role: format!("{:?}", role),
                expires_in_hours: 24,
                usage: "Add header: Authorization: Bearer <token>".to_string(),
            }))
        },
        Err(e) => {
            tracing::error!("❌ Failed to generate token: {}", e);
            Err(ApiError::InternalError(format!(
                "Token generation failed: {}",
                e
            )))
        },
    }
}

#[cfg(feature = "auth")]
#[api_operation(
    tag = "auth",
    summary = "Create API key",
    description = "Generate an API key for programmatic access",
    error_code = 500
)]
async fn create_api_key(
    state: Data<AppState>,
    body: Json<ApiKeyRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let role = body
        .role
        .as_deref()
        .and_then(|r| match r {
            "Admin" => Some(auth::UserRole::Admin),
            _ => Some(auth::UserRole::User),
        })
        .unwrap_or(auth::UserRole::User);

    match state
        .auth
        .create_api_key(&body.user_id, role.clone(), None)
        .await
    {
        Ok(api_key) => {
            tracing::info!(
                "✅ Created API key for user: {} (role: {:?})",
                body.user_id,
                role
            );
            Ok(Json(json!({
                "success": true,
                "api_key": api_key,
                "user_id": body.user_id,
                "role": format!("{:?}", role),
                "usage": "Add header: Authorization: ApiKey <key>",
                "rate_limit": {
                    "max_requests": 1000,
                    "window_seconds": 3600
                }
            })))
        },
        Err(e) => {
            tracing::error!("❌ Failed to create API key: {}", e);
            Err(ApiError::InternalError(format!(
                "API key creation failed: {}",
                e
            )))
        },
    }
}

// ============================================================================
// Main Server Configuration
// ============================================================================

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    // Create application state (connects to Qdrant if available)
    let state = AppState::new().await;
    let state_data = Data::new(state.clone());

    // In-server auto-append coalescer. Reads the debounce window
    // from APPEND_DEBOUNCE_SECS (default 60); 0 disables the loop
    // entirely (operators can fall back to manual / cron-driven
    // POST /api/graph/append). Spawned regardless of the chat-
    // backend wiring — it's a pull-driven loop that only acts when
    // an ingest fires a notification, so a misconfigured graphrag
    // instance just sits idle here.
    let debounce_secs: u64 = std::env::var("APPEND_DEBOUNCE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    if debounce_secs > 0 {
        tokio::spawn(auto_append_loop(
            state.clone(),
            std::time::Duration::from_secs(debounce_secs),
        ));
    } else {
        tracing::info!(
            "auto-append loop disabled (APPEND_DEBOUNCE_SECS=0); ingests will only enter the entity graph via manual POST /api/graph/append"
        );
    }

    // Stale-context cleanup loop. Runs every
    // `STALE_CONTEXT_CLEANUP_INTERVAL_HOURS` (default 6) when the
    // events store is enabled. No-op otherwise. Drops events older
    // than `eventRetentionDays`, sessions older than
    // `sessionTtlDays`, then truncates the WAL so disk usage
    // actually shrinks.
    if state.events_store.is_some() {
        let cleanup_interval_hours: u64 = std::env::var("STALE_CONTEXT_CLEANUP_INTERVAL_HOURS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        if cleanup_interval_hours > 0 {
            tokio::spawn(stale_context_cleanup_loop(
                state.clone(),
                std::time::Duration::from_secs(cleanup_interval_hours * 3600),
            ));
        } else {
            tracing::info!(
                "stale-context cleanup loop disabled (STALE_CONTEXT_CLEANUP_INTERVAL_HOURS=0)"
            );
        }
    }

    // Configure OpenAPI specification
    let spec = Spec {
        info: Info {
            title: "GraphRAG REST API".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: Some(concat!(
                "Production-ready REST API for GraphRAG operations with Qdrant vector database.\n\n",
                "## Features\n",
                "- Semantic search over documents\n",
                "- Knowledge graph construction\n",
                "- Real-time vector embeddings\n",
                "- Qdrant integration (optional)\n",
                "- JWT authentication (optional)\n\n",
                "## Getting Started\n",
                "1. Add documents via `POST /api/documents`\n",
                "2. Build graph via `POST /api/graph/build`\n",
                "3. Query via `POST /api/query`\n"
            ).to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    // JWT secret warning (item 3.4)
    #[cfg(feature = "auth")]
    if std::env::var("JWT_SECRET").is_err() {
        tracing::warn!(
            "⚠️  JWT_SECRET not set! Using insecure default. Set JWT_SECRET env var in production."
        );
    }

    // Bind from env vars so deployments can route around port
    // collisions without rebuilding. Defaults preserve the
    // historical 0.0.0.0:8080 behavior.
    let bind_host = std::env::var("GRAPHRAG_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let bind_port: u16 = std::env::var("GRAPHRAG_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let bind_addr = format!("{bind_host}:{bind_port}");

    tracing::info!("🚀 GraphRAG Server starting...");
    tracing::info!("📡 Listening on http://{bind_addr}");
    tracing::info!("📚 Swagger UI: http://{bind_addr}/swagger");
    tracing::info!("📄 OpenAPI spec: http://{bind_addr}/openapi.json");
    tracing::info!(
        "🗄️  Backend: {}",
        if state.has_qdrant() {
            "Qdrant"
        } else {
            "In-memory"
        }
    );

    HttpServer::new(move || {
        // Configure CORS for each app instance
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            // OpenAPI documentation
            .document(spec.clone())

            // Global middleware
            .wrap(cors)
            .wrap(actix_web::middleware::Logger::default())

            // Application state
            .app_data(state_data.clone())

            // Request body size limits (10MB for general payload, 10MB for JSON)
            .app_data(web::PayloadConfig::new(validation::MAX_BODY_SIZE))
            .app_data(web::JsonConfig::default().limit(validation::MAX_BODY_SIZE))

            // Public routes
            .service(resource("/").route(get().to(root)))
            .service(resource("/health").route(get().to(health)))

            // API routes
            .service(
                scope("/api")
                    // Documents endpoints
                    .service(
                        scope("/documents")
                            .service(resource("")
                                .route(get().to(list_documents))
                                .route(post().to(add_document)))
                            .service(resource("/{id}").route(delete().to(delete_document)))
                    )
                    // Query endpoints
                    .service(
                        scope("/query")
                            .service(resource("").route(post().to(query)))
                    )
                    // Graph endpoints
                    .service(
                        scope("/graph")
                            .service(resource("/build").route(post().to(build_graph)))
                            .service(resource("/append").route(post().to(append_graph)))
                            .service(resource("/stats").route(get().to(graph_stats)))
                    )
            )

            // Auth routes (temporarily disabled - feature "auth" is disabled)
            // #[cfg(feature = "auth")]
            // .service(
            //     scope("/auth")
            //         .service(resource("/login").route(post().to(login)))
            //         .service(resource("/api-key").route(post().to(create_api_key)))
            // )

            // Config endpoints (item 3.3): registered via plain Actix-web routes.
            // NOTE: To include them in OpenAPI spec, add #[api_operation] macros to
            //       each handler in config_endpoints.rs, then register via Apistos scope/resource.

            // Build OpenAPI spec endpoint
            .build("/openapi.json")

            // Config endpoints — exposed at /config (plain Actix-web routing).
            // NOTE: prefix is /config not /api/config because the apistos /api
            // scope above is registered first and matches /api/config (which
            // has no /config sub-route), shadowing this block. apistos's typed
            // scope/route requires handlers to implement PathItemDefinition
            // (i.e. carry #[api_operation]); plain web::scope can't be
            // registered before .build() either. Renaming to /config is the
            // simplest unblock and avoids both constraints.
            .service(
                web::scope("/config")
                    .route("", web::get().to(config_endpoints::get_config))
                    .route("", web::post().to(config_endpoints::set_config))
                    .route("/template", web::get().to(config_endpoints::get_config_template))
                    .route("/default", web::get().to(config_endpoints::get_default_config))
                    .route("/validate", web::post().to(config_endpoints::validate_config))
            )
            // Plain Actix scope (not apistos) — same OpenAPI-bypass reason
            // as /config above. Mounted at /embeddings (top-level, not
            // /api/embeddings) because the apistos /api scope above
            // matches /api/embeddings and shadows this block. Reports
            // the configured backend (from state.config.embeddings) and
            // runtime counters from the live EmbeddingService.
            .service(
                web::scope("/embeddings")
                    .route("/stats", web::get().to(embeddings_stats))
            )
            // Stale-context endpoints. Mounted at TOP-LEVEL (not
            // under `/api`) for two compounding reasons: (1) apistos's
            // `/api` scope is registered before `.build()` and would
            // shadow any sub-route on plain actix scopes; (2) the SSE
            // responder doesn't implement apistos's `PathItemDefinition`,
            // so it can't go inside the typed scope. Same pattern as
            // /config and /embeddings above.
            .service(
                web::scope("/recall")
                    .route("/revalidate", web::post().to(stale_context::recall_revalidate))
            )
            .service(
                web::scope("/lease")
                    .route("/check", web::get().to(stale_context::lease_check))
                    .route("/{session_id}", web::delete().to(stale_context::drop_session))
            )
            .service(
                web::scope("/events")
                    .route("/stream", web::get().to(stale_context::events_stream))
            )
    })
    .bind(&bind_addr)?
    .run()
    .await
}
