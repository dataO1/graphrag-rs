//! # GraphRAG Core
//!
//! Portable core library for GraphRAG - works on both native and WASM platforms.
//!
//! This is the foundational crate that provides:
//! - Knowledge graph construction and management
//! - Entity extraction and linking
//! - Vector embeddings and similarity search
//! - Graph algorithms (PageRank, community detection)
//! - Retrieval systems (semantic, keyword, hybrid)
//! - Caching and optimization
//!
//! ## Platform Support
//!
//! - **Native**: Full feature set with optional CUDA/Metal GPU acceleration
//! - **WASM**: Browser-compatible with Voy vector search and Candle embeddings
//!
//! ## Feature Flags
//!
//! - `wasm`: Enable WASM compatibility (uses Voy instead of HNSW)
//! - `cuda`: Enable NVIDIA GPU acceleration via Candle
//! - `metal`: Enable Apple Silicon GPU acceleration
//! - `webgpu`: Enable WebGPU acceleration for browser (via Burn)
//! - `pagerank`: Enable PageRank-based retrieval
//! - `lightrag`: Enable LightRAG optimizations (6000x token reduction)
//! - `caching`: Enable intelligent LLM response caching
//!
//! ## Quick Start
//!
//! ```rust
//! use graphrag_core::{GraphRAG, Config};
//!
//! # fn example() -> graphrag_core::Result<()> {
//! let config = Config::default();
//! let mut graphrag = GraphRAG::new(config)?;
//! graphrag.initialize()?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
// Note: WASM with wasm-bindgen DOES use std, so we don't disable it

// ================================
// MODULE DECLARATIONS
// ================================

// Core modules (always available)
/// Configuration management and loading
pub mod config;
/// Core traits and types
pub mod core;
/// Entity extraction and management
pub mod entity;
/// Text generation and LLM interactions (async feature only)
#[cfg(feature = "async")]
pub mod generation;
/// Graph data structures and algorithms
pub mod graph;
/// Retrieval strategies and implementations
pub mod retrieval;
/// Storage backends and persistence
#[cfg(any(
    feature = "memory-storage",
    feature = "persistent-storage",
    feature = "async"
))]
pub mod storage;
/// Text processing and chunking
pub mod text;
/// Vector operations and embeddings
pub mod vector;

/// Builder pattern implementations
pub mod builder;
/// Embedding generation and providers
pub mod embeddings;
/// Natural language processing utilities
pub mod nlp;
/// Ollama LLM integration
pub mod ollama;
pub mod openai;
pub mod chat;
/// Persistence layer for knowledge graphs (workspace management always available)
pub mod persistence;
/// Query processing and execution
pub mod query;
/// Text summarization capabilities
pub mod summarization;

// Pipeline modules
/// Data processing pipelines
pub mod pipeline;

// Advanced features (feature-gated)
#[cfg(feature = "parallel-processing")]
pub mod parallel;

#[cfg(feature = "lightrag")]
/// LightRAG dual-level retrieval optimization
pub mod lightrag;

/// Composable pipeline executor for build-graph operations

// Utility modules
/// Reranking utilities for improving search result quality
pub mod reranking;

/// Monitoring, benchmarking, and performance tracking
pub mod monitoring;

/// RAG answer evaluation and criticism
pub mod critic;

/// Evaluation framework for query results and pipeline validation
pub mod evaluation;

/// Graph optimization (weight optimization, DW-GRPO)
#[cfg(feature = "async")]
pub mod optimization;

/// API endpoints and handlers
#[cfg(feature = "api")]
pub mod api;

/// Inference module for model predictions
pub mod inference;

/// Multi-document corpus processing
#[cfg(feature = "corpus-processing")]
pub mod corpus;

// Feature-gated modules
#[cfg(feature = "async")]
/// Async GraphRAG implementation
pub mod async_graphrag;

#[cfg(feature = "async")]
/// Async processing pipelines
pub mod async_processing;

#[cfg(feature = "caching")]
/// Caching utilities for LLM responses
pub mod caching;

#[cfg(feature = "function-calling")]
/// Function calling capabilities for LLMs
pub mod function_calling;

#[cfg(feature = "incremental")]
/// Incremental graph updates
pub mod incremental;

#[cfg(feature = "rograg")]
/// ROGRAG (Robustly Optimized GraphRAG) implementation
pub mod rograg;

// Future utility modules (optional, not currently needed):
// pub mod automatic_entity_linking;  // Advanced entity linking
// pub mod phase_saver;               // Phase state persistence

// ================================
// PUBLIC API EXPORTS
// ================================

/// Prelude module containing the most commonly used types
///
/// Import everything you need with a single line:
/// ```rust
/// use graphrag_core::prelude::*;
/// ```
///
/// This includes:
/// - `GraphRAG` - The main orchestrator
/// - `Config` - Configuration management
/// - `GraphRAGBuilder` - Fluent configuration builder
/// - Core types: `Document`, `Entity`, `Relationship`, `TextChunk`
/// - Error handling: `Result`, `GraphRAGError`
pub mod prelude {
    // Main entry point
    pub use crate::GraphRAG;

    // Configuration & Builders
    pub use crate::builder::GraphRAGBuilder;
    pub use crate::builder::TypedBuilder;
    pub use crate::config::Config;

    // Error handling
    pub use crate::core::{GraphRAGError, Result};

    // Core data types
    pub use crate::core::{
        ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, KnowledgeGraph,
        Relationship, TextChunk,
    };

    // Search results and explained answers
    pub use crate::retrieval::SearchResult;
    pub use crate::retrieval::{ExplainedAnswer, ReasoningStep, SourceReference, SourceType};

    // Pipeline executor

    // Config deserialization helper
    pub use crate::config::setconfig::SetConfig;
}

// Re-export core types
pub use crate::config::Config;
pub use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, ErrorContext, ErrorSeverity,
    ErrorSuggestion, GraphRAGError, KnowledgeGraph, Relationship, Result, TextChunk,
};

// Re-export core traits (async feature only)
#[cfg(feature = "async")]
pub use crate::core::traits::{
    Embedder, EntityExtractor, GraphStore, LanguageModel, Retriever, Storage, VectorStore,
};

// Storage exports (when storage features are enabled)
#[cfg(feature = "memory-storage")]
pub use crate::storage::MemoryStorage;

// Re-export builder (GraphRAGBuilder exists, ConfigPreset and LLMProvider not yet implemented)
pub use crate::builder::GraphRAGBuilder;
// Note: GraphRAG struct is already public (defined at line 247)
// Note: builder::GraphRAG is a placeholder - the real implementation is the main GraphRAG struct

// Feature-gated exports
#[cfg(feature = "lightrag")]
pub use crate::lightrag::{
    DualLevelKeywords, DualLevelRetriever, DualRetrievalConfig, DualRetrievalResults,
    KeywordExtractor, KeywordExtractorConfig, MergeStrategy, SemanticSearcher,
};

#[cfg(feature = "pagerank")]
pub use crate::graph::pagerank::{PageRankConfig, PersonalizedPageRank};

#[cfg(feature = "leiden")]
pub use crate::graph::leiden::{HierarchicalCommunities, LeidenCommunityDetector, LeidenConfig};

#[cfg(feature = "cross-encoder")]
pub use crate::reranking::cross_encoder::{
    ConfidenceCrossEncoder, CrossEncoder, CrossEncoderConfig, RankedResult, RerankingStats,
};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::pagerank_retrieval::{PageRankRetrievalSystem, ScoredResult};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::hipporag_ppr::{Fact, HippoRAGConfig, HippoRAGRetriever};

// ================================
// MAIN GRAPHRAG SYSTEM
// ================================

/// Main GraphRAG system
///
/// This is the primary entry point for using GraphRAG. It orchestrates
/// all components: knowledge graph, retrieval, generation, and caching.
///
/// # Examples
///
/// ```rust
/// Phase 6: graphrag-core no longer owns chunk storage. Hosts (e.g.
/// graphrag-server) hold chunks in their own persistent store
/// (Qdrant) and pass them in to `extend_graph`. Recall paths
/// (`ask_with_seed_entities`, `ask_with_dual_seeds`) take a
/// pre-fetched `chunk_contents: HashMap<ChunkId, String>` parameter
/// — the caller looks the bytes up by id from its store before the
/// call. This keeps graphrag-core's in-memory state to entities +
/// relationships only.
///
/// Layer 4: GraphRAG implements Clone so the host can use copy-on-write
/// semantics around an `ArcSwap<Arc<GraphRAG>>` — readers do wait-free
/// pointer loads; writers `Arc::make_mut` (deep-clones only when the
/// snapshot is shared with a reader), mutate, atomic-swap. Recall and
/// ingestion never block each other.
#[derive(Clone)]
pub struct GraphRAG {
    config: Config,
    knowledge_graph: Option<KnowledgeGraph>,
    retrieval_system: Option<retrieval::RetrievalSystem>,
    query_planner: Option<query::planner::QueryPlanner>,
    critic: Option<critic::Critic>,
    /// Optional injected real embedding service (e.g. graphrag-server's
    /// mxbai-via-OVMS / Ollama / OpenAI-compat backend). Set via
    /// `set_embedding_provider`; propagated into `retrieval_system` so
    /// every internal embedding call uses the real provider instead of
    /// the hash-based dummy `EmbeddingGenerator`. `None` means tests /
    /// standalone use — fall back to dummy.
    embedding_provider: Option<core::traits::DynEmbedder>,
    #[cfg(feature = "parallel-processing")]
    #[allow(dead_code)]
    parallel_processor: Option<parallel::ParallelProcessor>,
}

/// Internal accumulator for `extend_graph`'s per-chunk pass. Tracks
/// what changed so the public `ExtendSummary` can report deltas
/// rather than raw graph totals.
///
/// `touched_entities` / `touched_relationships` collect the
/// invalidation set for the persistence layer: every entity id that
/// was newly inserted OR had its mentions extended in this pass, and
/// every (source, relation_type, target) tuple newly added. The
/// persistence layer then embeds + upserts ONLY these (LightRAG
/// `merge_nodes_and_edges` parity), instead of walking all
/// 1985 entities in the graph each cycle.
///
/// `LinkedHashSet`-style behavior is achieved via Vec + a parallel
/// HashSet for dedup: insertion order preserved so the persist
/// progress bar makes sense, dedup is O(1).
#[derive(Default)]
struct ExtractMetrics {
    new_entities: usize,
    new_relationships: usize,
    mentions_merged: usize,
    touched_entity_ids: Vec<String>,
    touched_entity_seen: std::collections::HashSet<String>,
    touched_relationship_keys: Vec<(String, String, String)>,
    touched_relationship_seen: std::collections::HashSet<(String, String, String)>,
}

impl ExtractMetrics {
    fn touch_entity(&mut self, id: &str) {
        if self.touched_entity_seen.insert(id.to_string()) {
            self.touched_entity_ids.push(id.to_string());
        }
    }
    fn touch_relationship(&mut self, source: &str, relation_type: &str, target: &str) {
        let key = (source.to_string(), relation_type.to_string(), target.to_string());
        if self.touched_relationship_seen.insert(key.clone()) {
            self.touched_relationship_keys.push(key);
        }
    }
}

/// LightRAG-style dual-level query keywords.
///
/// Reference: LightRAG paper (arXiv:2410.05779). At query time, one
/// LLM call extracts both keyword sets from the user query; each
/// drives a separate retrieval stream:
///   - `low_level` keywords are embedded and used to vector-search
///     the *entity* store for top-K seed entities (concrete things).
///   - `high_level` keywords are embedded and used to vector-search
///     the *relationship* store for top-K seed relations (themes,
///     abstract connections).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct QueryKeywords {
    /// Specific named entities, attributes, or details mentioned in
    /// the query (e.g. "PostgreSQL", "ACID compliance").
    pub low_level: Vec<String>,
    /// Overarching themes, concepts, or topics the query is about
    /// (e.g. "database transactions", "concurrency control").
    pub high_level: Vec<String>,
}

/// Wire envelope for parsing the dual-keyword JSON LLM output. Field
/// names match LightRAG's published prompt; not exposed publicly.
#[derive(Debug, serde::Deserialize)]
struct QueryKeywordsWire {
    high_level_keywords: Option<Vec<String>>,
    low_level_keywords: Option<Vec<String>>,
}

/// Caller-supplied seed populations for [`GraphRAG::ask_with_dual_seeds`].
/// Each LightRAG retrieval mode maps to a different non-empty subset:
/// local → entities; global → relations; hybrid → entities + relations;
/// mix → all three (entities, relations, chunks).
#[derive(Debug, Clone, Default)]
pub struct DualSeeds {
    /// Entity ids — typically the top-K result of vector-searching
    /// an entity-description embedding store with the embedded
    /// `low_level` keywords.
    pub entities: Vec<EntityId>,
    /// Relation triples `(source_entity_id, target_entity_id,
    /// relation_type)` — typically from vector-searching a
    /// relationship-description embedding store with the embedded
    /// `high_level` keywords.
    pub relations: Vec<(EntityId, EntityId, String)>,
    /// Direct chunk ids — for LightRAG's `mix` mode, where chunk-level
    /// vector search results are merged with the entity/relation
    /// expansion. Empty for local/global/hybrid.
    pub chunks: Vec<ChunkId>,
}

/// Summary returned by [`GraphRAG::extend_graph`] — what changed in
/// this incremental pass.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExtendSummary {
    /// Number of chunks that were extracted in this call. Zero on
    /// the no-op fast path (live chunk count hasn't grown since the
    /// last build/extend).
    pub chunks_processed: usize,
    /// New entities added to the graph (after dedup-by-id; existing
    /// entities re-mentioned in new chunks have their `mentions`
    /// extended in place rather than counted here).
    pub new_entities: usize,
    /// New relationships added to the graph in this call.
    pub new_relationships: usize,
    /// Existing entities whose `mentions` were extended because they
    /// were re-mentioned in a new chunk. Useful for assessing whether
    /// downstream community/PageRank recompute is warranted.
    pub mentions_merged: usize,
    /// Graph totals after the extend pass.
    pub total_entities: usize,
    pub total_relationships: usize,
    /// Entity ids touched by this pass — both genuinely new and
    /// existing-but-re-mentioned. Persistence layer uses this to
    /// embed + upsert ONLY these (not the whole graph), mirroring
    /// LightRAG's `merge_nodes_and_edges` scope. Stored as the
    /// underlying string id so callers don't need to re-import
    /// `EntityId`. Order: insertion (de-duped).
    pub touched_entity_ids: Vec<String>,
    /// Relationship keys touched by this pass: `(source_id,
    /// relation_type, target_id)`. Used to embed + upsert only the
    /// affected rows in the relationships sidecar.
    pub touched_relationship_keys: Vec<(String, String, String)>,
}

impl GraphRAG {
    /// Create a new GraphRAG instance with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            knowledge_graph: None,
            retrieval_system: None,
            query_planner: None,
            critic: None,
            embedding_provider: None,
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
        })
    }

    /// Inject a real embedding service. Call this once after
    /// `GraphRAG::new(config)` (or after `initialize()`) and before
    /// the first query. The provider is propagated into the retrieval
    /// system so every internal embedding call uses the real service.
    ///
    /// Hosts (e.g. graphrag-server) MUST inject a real provider before
    /// `initialize()` unless they are content with hash embeddings —
    /// `initialize()` will fail when `config.embeddings.backend != "hash"`
    /// and no provider has been wired in, since graphrag-core itself
    /// can't construct HTTP clients.
    pub fn set_embedding_provider(&mut self, provider: core::traits::DynEmbedder) {
        self.embedding_provider = Some(provider.clone());
        if let Some(rs) = self.retrieval_system.as_mut() {
            rs.set_embedding_provider(provider);
        }
    }

    /// Create a Zero-Config local GraphRAG instance
    /// Uses: Candle (MiniLM) for embeddings, Memory/LanceDB for storage, Ollama for LLM
    pub fn default_local() -> Result<Self> {
        let mut config = Config::default();
        // Configure for local use
        config.ollama.enabled = true;
        // config.storage.type = StorageType::LanceDB; // Future

        Self::new(config)
    }

    /// Create a builder for configuring GraphRAG
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::GraphRAG;
    ///
    /// # fn example() -> graphrag_core::Result<()> {
    /// let graphrag = GraphRAG::builder()
    ///     .with_output_dir("./workspace")
    ///     .with_chunk_size(512)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> crate::builder::GraphRAGBuilder {
        crate::builder::GraphRAGBuilder::new()
    }

    /// Initialize the GraphRAG system.
    ///
    /// When `auto_save.enabled = true` and a `base_dir` is configured, attempts to
    /// load an existing graph from the workspace on disk before starting fresh.
    /// This means a second run reuses the previously built graph automatically.
    pub fn initialize(&mut self) -> Result<()> {
        // Try to restore from workspace if persistent storage is configured
        let loaded = self.try_load_from_workspace();

        if !loaded {
            self.knowledge_graph = Some(KnowledgeGraph::new());
        }

        let mut rs = retrieval::RetrievalSystem::new(&self.config)?;
        // Propagate any provider that was set BEFORE initialize() into
        // the freshly-constructed retrieval system. Order-independence:
        // hosts can call set_embedding_provider before or after init.
        // If no provider was injected and the configured backend isn't
        // "hash", error out — graphrag-core itself can't construct HTTP
        // clients, so silently falling back to hash here would corrupt
        // any host that thinks it's running real embeddings.
        match self.embedding_provider.clone() {
            Some(provider) => rs.set_embedding_provider(provider),
            None if self.config.embeddings.backend == "hash" => {
                // RetrievalSystem::new already installs a HashEmbedder
                // sized to config.embeddings.dimension; nothing to do.
            },
            None => {
                return Err(GraphRAGError::Config {
                    message: format!(
                        "embeddings.backend is '{}' but no embedding provider has been injected. \
                         Call GraphRAG::set_embedding_provider before initialize(), or set \
                         embeddings.backend = \"hash\".",
                        self.config.embeddings.backend
                    ),
                });
            },
        }
        self.retrieval_system = Some(rs);

        if let Some(client) =
            chat::ChatClient::from_config(&self.config.ollama, &self.config.openai)
        {
            self.query_planner = Some(query::planner::QueryPlanner::new(client));
        }

        Ok(())
    }

    /// Attempt to load the knowledge graph from a workspace on disk.
    /// Returns `true` if the graph was loaded successfully, `false` otherwise.
    fn try_load_from_workspace(&mut self) -> bool {
        if !self.config.auto_save.enabled {
            return false;
        }
        let base_dir = match &self.config.auto_save.base_dir {
            Some(d) => d.clone(),
            None => return false,
        };
        let workspace_name = self
            .config
            .auto_save
            .workspace_name
            .as_deref()
            .unwrap_or("default");

        let manager = match persistence::WorkspaceManager::new(&base_dir) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Could not open workspace base dir '{}': {}", base_dir, e);
                return false;
            },
        };

        if !manager.workspace_exists(workspace_name) {
            return false;
        }

        match manager.load_graph(workspace_name) {
            Ok(graph) => {
                tracing::info!(
                    "Loaded graph from workspace '{}' ({} entities, {} relationships)",
                    workspace_name,
                    graph.entity_count(),
                    graph.relationship_count(),
                );
                self.knowledge_graph = Some(graph);
                true
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to load graph from workspace '{}': {}",
                    workspace_name,
                    e
                );
                false
            },
        }
    }

    /// Save the current knowledge graph to the configured workspace on disk.
    /// No-op when `auto_save.enabled = false` or `base_dir` is not set.
    pub fn save_to_workspace(&self) -> Result<()> {
        if !self.config.auto_save.enabled {
            return Ok(());
        }
        let base_dir = match &self.config.auto_save.base_dir {
            Some(d) => d,
            None => return Ok(()),
        };
        let workspace_name = self
            .config
            .auto_save
            .workspace_name
            .as_deref()
            .unwrap_or("default");

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let manager = persistence::WorkspaceManager::new(base_dir)?;
        manager.save_graph(graph, workspace_name)?;

        tracing::info!(
            "Saved graph to workspace '{}' in '{}' ({} entities, {} relationships)",
            workspace_name,
            base_dir,
            graph.entity_count(),
            graph.relationship_count(),
        );
        Ok(())
    }

    /// Clear all entities and relationships from the knowledge graph
    ///
    /// This method preserves documents and text chunks but removes all extracted entities and relationships.
    /// Useful for rebuilding the graph from scratch without reloading documents.
    pub fn clear_graph(&mut self) -> Result<()> {
        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        #[cfg(feature = "tracing")]
        tracing::info!("Clearing knowledge graph (preserving documents and chunks)");

        graph.clear_entities_and_relationships();
        Ok(())
    }

    /// Extract entities + relationships from the supplied chunks and
    /// merge them into the in-memory KnowledgeGraph. Replaces the old
    /// `extend_graph()` (which iterated `kg.chunks() - processed_chunks`)
    /// and `build_graph()` (which iterated `kg.chunks()` whole).
    ///
    /// Phase 6: graphrag-core no longer owns chunks. The caller
    /// (graphrag-server) is the source of truth for "which chunks need
    /// extracting" — it queries Qdrant for chunks where
    /// `entities_extracted_at IS NULL`, hands them in here, and on
    /// success sets the timestamp on those Qdrant payloads.
    ///
    /// Each input is `(qdrant_block_id, content)`. mention.chunk_id
    /// values produced by extraction reference these qdrant ids
    /// directly, so they resolve from Qdrant on subsequent recalls
    /// without any in-memory chunk universe.
    #[cfg(feature = "async")]
    pub async fn extend_graph(
        &mut self,
        input_chunks: &[(crate::core::ChunkId, String)],
    ) -> Result<ExtendSummary> {
        use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

        let total_entities_before = self
            .knowledge_graph
            .as_ref()
            .map(|g| g.entities().count())
            .unwrap_or(0);
        let total_relationships_before = self
            .knowledge_graph
            .as_ref()
            .map(|g| g.relationships().count())
            .unwrap_or(0);

        let total_delta = input_chunks.len();

        // Fast no-op path. Cron-callable without burning anything.
        if total_delta == 0 {
            return Ok(ExtendSummary {
                chunks_processed: 0,
                new_entities: 0,
                new_relationships: 0,
                mentions_merged: 0,
                total_entities: total_entities_before,
                total_relationships: total_relationships_before,
                touched_entity_ids: Vec::new(),
                touched_relationship_keys: Vec::new(),
            });
        }

        // Build transient TextChunks from caller input so the existing
        // extractor machinery (extend_with_*) can work unchanged. These
        // chunks live for the duration of this call only — they are NOT
        // added to KnowledgeGraph.chunks.
        let delta_chunks: Vec<crate::core::TextChunk> = input_chunks
            .iter()
            .map(|(id, content)| crate::core::TextChunk {
                id: id.clone(),
                document_id: crate::core::DocumentId::new("qdrant-block".to_string()),
                content: content.clone(),
                start_offset: 0,
                end_offset: content.len(),
                embedding: None,
                entities: Vec::new(),
                metadata: crate::core::ChunkMetadata::default(),
            })
            .collect();

        let suppress = self.config.suppress_progress_bars;
        let make_pb = move |total: u64, style: ProgressStyle| -> ProgressBar {
            let pb = ProgressBar::new(total).with_style(style);
            if suppress {
                pb.set_draw_target(ProgressDrawTarget::hidden());
            }
            pb
        };

        let mut metrics = ExtractMetrics::default();

        #[cfg(feature = "tracing")]
        tracing::info!(
            "extend_graph: processing {} chunks (approach='{}', use_gleaning={}, openai.enabled={}, ollama.enabled={})",
            total_delta,
            self.config.approach,
            self.config.entities.use_gleaning,
            self.config.openai.enabled,
            self.config.ollama.enabled,
        );

        if self.config.entities.use_gleaning && self.config.chat_enabled() {
            self.extend_with_gleaning(&delta_chunks, &mut metrics, &make_pb).await?;
        } else if self.config.chat_enabled() {
            self.extend_with_llm_single_pass(&delta_chunks, &mut metrics, &make_pb).await?;
        } else if self.config.gliner.enabled {
            #[cfg(feature = "gliner")]
            {
                self.extend_with_gliner(&delta_chunks, &mut metrics, &make_pb)
                    .await?;
            }
            #[cfg(not(feature = "gliner"))]
            return Err(GraphRAGError::Config {
                message: "extend_graph: gliner.enabled but crate compiled \
                          without --features gliner"
                    .to_string(),
            });
        } else {
            self.extend_with_pattern_extraction(&delta_chunks, &mut metrics, &make_pb)?;
        }

        // Final totals (re-read; extraction may have added entities).
        let (total_entities, total_relationships) = {
            let graph = self
                .knowledge_graph
                .as_ref()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Knowledge graph went away mid-extend (impossible)".to_string(),
                })?;
            (graph.entities().count(), graph.relationships().count())
        };

        // Persist (entity + relationship side; chunks are not part of
        // graphrag-core's persistence anymore — Qdrant owns them).
        self.save_to_workspace()?;

        Ok(ExtendSummary {
            chunks_processed: total_delta,
            new_entities: metrics.new_entities,
            new_relationships: metrics.new_relationships,
            mentions_merged: metrics.mentions_merged,
            total_entities,
            total_relationships,
            touched_entity_ids: metrics.touched_entity_ids,
            touched_relationship_keys: metrics.touched_relationship_keys,
        })
    }

    /// LLM single-pass extension path — mirrors the LLM single-pass
    /// branch in `build_graph` but operates on the supplied delta
    /// slice instead of every chunk in the graph, and dedupes
    /// entities by id on insert.
    #[cfg(feature = "async")]
    async fn extend_with_llm_single_pass(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::chat::ChatClient;
        use crate::entity::llm_extractor::LLMEntityExtractor;
        use indicatif::ProgressStyle;

        let client = ChatClient::from_config(&self.config.ollama, &self.config.openai)
            .ok_or_else(|| GraphRAGError::Config {
                message: "extend_graph: chat_enabled() but no ChatClient could be \
                          constructed; check config.{ollama,openai}.enabled"
                    .to_string(),
            })?;

        let entity_types = if self.config.entities.entity_types.is_empty() {
            vec![
                "PERSON".to_string(),
                "ORGANIZATION".to_string(),
                "LOCATION".to_string(),
            ]
        } else {
            self.config.entities.entity_types.clone()
        };

        let extraction_max_tokens: Option<usize> = if self.config.openai.enabled {
            self.config.openai.max_tokens.map(|n| n as usize)
        } else {
            self.config.ollama.max_tokens.map(|n| n as usize)
        };
        let extraction_temperature = if self.config.openai.enabled {
            self.config.openai.temperature.unwrap_or(0.1)
        } else {
            self.config.ollama.temperature.unwrap_or(0.1)
        };

        let extractor = std::sync::Arc::new(
            LLMEntityExtractor::new(client, entity_types)
                .with_temperature(extraction_temperature)
                .with_max_tokens_opt(extraction_max_tokens)
                .with_keep_alive(self.config.ollama.keep_alive.clone()),
        );

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} delta chunks ({eta})")
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (LLM single-pass)");

        // Concurrency knob. EXTRACTION_CONCURRENCY caps how many
        // chunk extractions can be in flight against the chat backend
        // simultaneously. Match it to llama-server's `--parallel`
        // (or vLLM's `--max-num-seqs`); going higher than the
        // backend's slot count just queues requests and adds latency
        // without throughput. Default 4 — comfortable for a
        // ctx=92K/parallel=4 llama-server slot layout (23K per slot,
        // ample for the ~4K an extraction call needs).
        let concurrency: usize = std::env::var("EXTRACTION_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n: &usize| *n >= 1)
            .unwrap_or(4);

        let total = delta_chunks.len();
        #[cfg(feature = "tracing")]
        tracing::info!(
            total = total,
            concurrency = concurrency,
            "extend_graph: starting LLM single-pass with bounded concurrency"
        );

        // Build a stream of (idx, future-of-result) and let
        // `buffer_unordered` keep `concurrency` futures in flight.
        // Each future is pure read-only (extractor + chunk → result);
        // graph mutation happens in the serial consumer below where
        // we hold &mut self.knowledge_graph exclusively.
        use futures::stream::{self, StreamExt};
        let mut stream = stream::iter(delta_chunks.iter().cloned().enumerate())
            .map(|(idx, chunk)| {
                let extractor = extractor.clone();
                async move {
                    let res = extractor.extract_from_chunk(&chunk).await;
                    (idx, chunk, res)
                }
            })
            .buffer_unordered(concurrency);

        let mut completed = 0usize;
        while let Some((idx, chunk, result)) = stream.next().await {
            completed += 1;
            #[cfg(feature = "tracing")]
            tracing::info!(
                "extend_graph: completed delta chunk {}/{} (idx={}, concurrency={}, LLM single-pass)",
                completed,
                total,
                idx,
                concurrency
            );
            pb.set_message(format!(
                "Delta chunk {}/{} (LLM single-pass, c={})",
                completed, total, concurrency
            ));

            match result {
                Ok((entities, relationships)) => {
                    let graph = self.knowledge_graph.as_mut().ok_or_else(|| {
                        GraphRAGError::Config {
                            message: "Knowledge graph went away mid-extract".to_string(),
                        }
                    })?;
                    for entity in entities {
                        Self::merge_entity(graph, entity, metrics)?;
                    }
                    for relationship in relationships {
                        Self::merge_relationship(graph, relationship, metrics);
                    }
                },
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        chunk_id = %chunk.id,
                        error = %e,
                        "extend_graph: LLM extraction failed for delta chunk; skipping"
                    );
                    let _ = e;
                },
            }
            pb.inc(1);
        }

        pb.finish_with_message("LLM single-pass extension complete");
        Ok(())
    }

    /// Gleaning extension path — same shape as the gleaning branch in
    /// `build_graph`, restricted to the delta slice with dedup-on-add.
    /// Triple-reflection validation is honored.
    #[cfg(feature = "async")]
    async fn extend_with_gleaning(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::chat::ChatClient;
        use crate::entity::GleaningEntityExtractor;
        use indicatif::ProgressStyle;

        let client = ChatClient::from_config(&self.config.ollama, &self.config.openai)
            .ok_or_else(|| GraphRAGError::Config {
                message: "extend_graph: chat_enabled() but no ChatClient could be \
                          constructed"
                    .to_string(),
            })?;

        let gleaning_config = crate::entity::GleaningConfig {
            max_gleaning_rounds: self.config.entities.max_gleaning_rounds,
            completion_threshold: 0.8,
            entity_confidence_threshold: self.config.entities.min_confidence as f64,
            use_llm_completion_check: true,
            entity_types: if self.config.entities.entity_types.is_empty() {
                vec![
                    "PERSON".to_string(),
                    "ORGANIZATION".to_string(),
                    "LOCATION".to_string(),
                ]
            } else {
                self.config.entities.entity_types.clone()
            },
            temperature: 0.1,
            max_tokens: 1500,
        };
        let extractor = GleaningEntityExtractor::new(client.clone(), gleaning_config);

        let rel_extractor = if self.config.entities.enable_triple_reflection {
            Some(crate::entity::LLMRelationshipExtractor::new(Some(
                &self.config.ollama,
            ))?)
        } else {
            None
        };

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} delta chunks ({eta})")
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (gleaning)");

        for (idx, chunk) in delta_chunks.iter().enumerate() {
            pb.set_message(format!(
                "Delta chunk {}/{} (gleaning, {} rounds)",
                idx + 1,
                delta_chunks.len(),
                self.config.entities.max_gleaning_rounds
            ));

            #[cfg(feature = "tracing")]
            tracing::info!(
                "extend_graph: processing delta chunk {}/{} (gleaning)",
                idx + 1,
                delta_chunks.len()
            );

            let (entities, relationships) = extractor.extract_with_gleaning(chunk).await?;

            let entity_map: std::collections::HashMap<_, _> = entities
                .iter()
                .map(|e| (e.id.clone(), e.name.clone()))
                .collect();

            // Add entities first so relationship merge can reference them.
            {
                let graph = self.knowledge_graph.as_mut().ok_or_else(|| {
                    GraphRAGError::Config {
                        message: "Knowledge graph went away mid-extract".to_string(),
                    }
                })?;
                for entity in entities {
                    Self::merge_entity(graph, entity, metrics)?;
                }
            }

            // Relationships, optionally validated.
            if let Some(ref validator) = rel_extractor {
                for relationship in relationships {
                    let (source_name, target_name) = {
                        let graph = self.knowledge_graph.as_ref().unwrap();
                        let source_name = entity_map
                            .get(&relationship.source)
                            .cloned()
                            .or_else(|| {
                                graph
                                    .entities()
                                    .find(|e| e.id == relationship.source)
                                    .map(|e| e.name.clone())
                            })
                            .unwrap_or_else(|| relationship.source.0.clone());
                        let target_name = entity_map
                            .get(&relationship.target)
                            .cloned()
                            .or_else(|| {
                                graph
                                    .entities()
                                    .find(|e| e.id == relationship.target)
                                    .map(|e| e.name.clone())
                            })
                            .unwrap_or_else(|| relationship.target.0.clone());
                        (source_name, target_name)
                    };
                    match validator
                        .validate_triple(
                            &source_name,
                            &relationship.relation_type,
                            &target_name,
                            &chunk.content,
                        )
                        .await
                    {
                        Ok(validation) => {
                            if validation.is_valid
                                && validation.confidence
                                    >= self.config.entities.validation_min_confidence
                            {
                                let graph = self.knowledge_graph.as_mut().unwrap();
                                Self::merge_relationship(graph, relationship, metrics);
                            } else {
                                #[cfg(feature = "tracing")]
                                tracing::debug!(
                                    "extend_graph: filtered relationship (valid={}, conf={:.2})",
                                    validation.is_valid,
                                    validation.confidence
                                );
                            }
                        },
                        Err(_e) => {
                            // Validation errored → add anyway (matches build_graph's behaviour).
                            let graph = self.knowledge_graph.as_mut().unwrap();
                            Self::merge_relationship(graph, relationship, metrics);
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                "extend_graph: validation error, adding relationship anyway: {}",
                                _e
                            );
                        },
                    }
                }
            } else {
                let graph = self.knowledge_graph.as_mut().unwrap();
                for relationship in relationships {
                    Self::merge_relationship(graph, relationship, metrics);
                }
            }

            pb.inc(1);
        }

        pb.finish_with_message("Gleaning extension complete");
        Ok(())
    }

    /// GLiNER joint NER + RE extension path — mirrors the GLiNER
    /// branch in `build_graph`, restricted to the delta slice with
    /// dedup-on-add via `merge_entity` / `merge_relationship`.
    ///
    /// `gline-rs` is synchronous (ONNX Runtime blocks the calling
    /// thread), so each per-chunk call runs inside
    /// `tokio::task::spawn_blocking`. The `Arc`-cloned extractor is
    /// cheaply shareable across blocking tasks.
    ///
    /// **Untested**: matches build_graph's GLiNER branch — upstream
    /// has no test coverage for that path either, since GLiNER needs
    /// a downloaded ONNX model at runtime and produces non-
    /// deterministic output. The four pattern-based `extend_graph_*`
    /// tests prove the dispatcher routes correctly; this path's
    /// correctness rests on visual parity with build_graph's GLiNER
    /// branch and on the shared `merge_entity` / `merge_relationship`
    /// dedup helpers (which are exercised by the pattern-based tests).
    #[cfg(feature = "gliner")]
    async fn extend_with_gliner(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::entity::GLiNERExtractor;
        use indicatif::ProgressStyle;
        use std::sync::Arc;

        // Lazy model load happens inside GLiNERExtractor::new; failures
        // surface here rather than on first chunk so the caller sees
        // them immediately.
        let extractor = Arc::new(GLiNERExtractor::new(self.config.gliner.clone()).map_err(
            |e| crate::core::error::GraphRAGError::EntityExtraction {
                message: format!("GLiNER init failed: {e}"),
            },
        )?);

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template("   [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} delta chunks ({eta})")
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (GLiNER-Relex)");

        for (idx, chunk) in delta_chunks.iter().enumerate() {
            pb.set_message(format!(
                "Delta chunk {}/{} (GLiNER-Relex)",
                idx + 1,
                delta_chunks.len()
            ));

            #[cfg(feature = "tracing")]
            tracing::info!(
                "extend_graph: processing delta chunk {}/{} (GLiNER)",
                idx + 1,
                delta_chunks.len()
            );

            let ext = Arc::clone(&extractor);
            let ch = chunk.clone();
            let result = tokio::task::spawn_blocking(move || ext.extract_from_chunk(&ch))
                .await
                .map_err(|e| crate::core::error::GraphRAGError::EntityExtraction {
                    message: format!("spawn_blocking join error: {e}"),
                })?;

            match result {
                Ok((entities, relationships)) => {
                    let graph = self.knowledge_graph.as_mut().ok_or_else(|| {
                        GraphRAGError::Config {
                            message: "Knowledge graph went away mid-extract".to_string(),
                        }
                    })?;
                    for entity in entities {
                        Self::merge_entity(graph, entity, metrics)?;
                    }
                    for rel in relationships {
                        Self::merge_relationship(graph, rel, metrics);
                    }
                },
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        chunk_id = %chunk.id,
                        error = %_e,
                        "extend_graph: GLiNER extraction failed for delta chunk; skipping"
                    );
                },
            }

            pb.inc(1);
        }

        pb.finish_with_message("GLiNER-Relex extension complete");
        Ok(())
    }

    /// Pattern-based extension path — regex/capitalization extraction
    /// over the delta slice. No LLM dependency, useful for setups that
    /// don't have a chat backend wired and for testing.
    fn extend_with_pattern_extraction(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::entity::EntityExtractor;
        use indicatif::ProgressStyle;

        let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template("   [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} delta chunks ({eta})")
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (pattern-based)");

        // Phase 1: entities
        for (idx, chunk) in delta_chunks.iter().enumerate() {
            pb.set_message(format!(
                "Delta chunk {}/{} (pattern entities)",
                idx + 1,
                delta_chunks.len()
            ));

            let entities = extractor.extract_from_chunk(chunk)?;
            let graph = self.knowledge_graph.as_mut().ok_or_else(|| {
                GraphRAGError::Config {
                    message: "Knowledge graph went away mid-extract".to_string(),
                }
            })?;
            for entity in entities {
                Self::merge_entity(graph, entity, metrics)?;
            }
            pb.inc(1);
        }
        pb.finish_with_message("Pattern entity extension complete");

        // Phase 2: relationships, only if config requests them.
        if self.config.graph.extract_relationships {
            let all_entities: Vec<_> = {
                let graph = self.knowledge_graph.as_ref().unwrap();
                graph.entities().cloned().collect()
            };

            let rel_pb = make_pb(
                delta_chunks.len() as u64,
                ProgressStyle::default_bar()
                    .template("   [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} delta chunks rels ({eta})")
                    .expect("Invalid progress bar template")
                    .progress_chars("=>-"),
            );
            rel_pb.set_message("Extending relationships (pattern-based)");

            for (idx, chunk) in delta_chunks.iter().enumerate() {
                rel_pb.set_message(format!(
                    "Delta chunk {}/{} (pattern relationships)",
                    idx + 1,
                    delta_chunks.len()
                ));

                let chunk_entities: Vec<_> = all_entities
                    .iter()
                    .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
                    .cloned()
                    .collect();
                if chunk_entities.len() < 2 {
                    rel_pb.inc(1);
                    continue;
                }

                let relationships = extractor.extract_relationships(&chunk_entities, chunk)?;
                let graph = self.knowledge_graph.as_mut().unwrap();
                for (source_id, target_id, relation_type) in relationships {
                    let relationship = Relationship {
                        source: source_id,
                        target: target_id,
                        relation_type,
                        confidence: self.config.graph.relationship_confidence_threshold,
                        context: vec![chunk.id.clone()],
                        embedding: None,
                        temporal_type: None,
                        temporal_range: None,
                        causal_strength: None,
                    };
                    Self::merge_relationship(graph, relationship, metrics);
                }
                rel_pb.inc(1);
            }
            rel_pb.finish_with_message("Pattern relationship extension complete");
        }

        Ok(())
    }

    /// Add `new_entity` to `graph`, merging into the existing entity
    /// if the id already exists. Used by the extend path; build_graph
    /// keeps its no-dedup behaviour for backward compatibility.
    ///
    /// Merge semantics: existing entity's `mentions` are extended with
    /// any not-already-present mentions from `new_entity` (compared by
    /// `(chunk_id, start_offset)`); `confidence` is bumped to the max
    /// of the two values.
    /// Thin metrics-tracking wrapper over `KnowledgeGraph::add_entity`.
    /// Since `add_entity` itself dedupes by id and merges mentions in
    /// place, this only counts "was this a new node vs a merge into
    /// an existing one" so callers (extend_graph) can report
    /// `new_entities` vs `mentions_merged` accurately.
    fn merge_entity(
        graph: &mut KnowledgeGraph,
        new_entity: Entity,
        metrics: &mut ExtractMetrics,
    ) -> Result<()> {
        let id = new_entity.id.clone();
        let was_existing = graph.get_entity(&id).is_some();
        let mentions_to_merge = if was_existing {
            // Count the mentions that ARE new vs the existing ones,
            // before add_entity does the merge. We only care about
            // the count here; the merge happens inside add_entity.
            let existing_mentions: Vec<_> = graph
                .get_entity(&id)
                .map(|e| {
                    e.mentions
                        .iter()
                        .map(|m| (m.chunk_id.clone(), m.start_offset))
                        .collect()
                })
                .unwrap_or_default();
            new_entity
                .mentions
                .iter()
                .filter(|m| {
                    !existing_mentions
                        .iter()
                        .any(|(c, off)| c == &m.chunk_id && *off == m.start_offset)
                })
                .count()
        } else {
            0
        };
        let id_str = id.0.clone();
        graph.add_entity(new_entity)?;
        if was_existing {
            metrics.mentions_merged += mentions_to_merge;
            // Existing-but-re-mentioned entity: still touched; persist
            // layer needs to refresh its embedding + mentions payload.
            // Skip the touch when zero new mentions actually merged
            // (re-extraction over the same chunk would otherwise mark
            // every existing entity touched on every pass).
            if mentions_to_merge > 0 {
                metrics.touch_entity(&id_str);
            }
        } else {
            metrics.new_entities += 1;
            metrics.touch_entity(&id_str);
        }
        Ok(())
    }

    /// Thin metrics-tracking wrapper over `KnowledgeGraph::add_relationship`.
    /// `add_relationship` itself dedupes by `(source, target,
    /// relation_type)` and silently ignores missing-endpoint errors
    /// (matching build_graph's existing behavior — the relationship's
    /// target may have been extracted from a chunk that hasn't been
    /// processed yet). This wrapper just counts how many were genuinely
    /// new for telemetry by scanning before the add.
    fn merge_relationship(
        graph: &mut KnowledgeGraph,
        relationship: Relationship,
        metrics: &mut ExtractMetrics,
    ) {
        let was_existing = graph.relationships().any(|r| {
            r.source == relationship.source
                && r.target == relationship.target
                && r.relation_type == relationship.relation_type
        });
        let src = relationship.source.0.clone();
        let tgt = relationship.target.0.clone();
        let rel_type = relationship.relation_type.clone();
        if graph.add_relationship(relationship).is_ok() && !was_existing {
            metrics.new_relationships += 1;
            metrics.touch_relationship(&src, &rel_type, &tgt);
        }
        // add_relationship error (missing endpoint) is intentionally
        // ignored — see method docs.
    }


    // =====================================================================
    // LightRAG-style dual-level retrieval
    //
    // Reference: "LightRAG: Simple and Fast Retrieval-Augmented Generation"
    //   (Guo, Wang, Lin, Hu, Bei, Chen, Liao, Lu, Zhang, Yan, Lu;
    //   arXiv:2410.05779, 2024)
    //
    // The paper's core idea: instead of MS GraphRAG's expensive
    // community-detection-and-LLM-summary index step, LightRAG keeps
    // index-time work to entity + relation extraction (with descriptions),
    // embeds both, and shifts intelligence to query time. Each query is
    // decomposed into TWO keyword sets via one LLM call:
    //
    //   - low_level_keywords: specific named entities, attributes,
    //     properties mentioned in the question
    //   - high_level_keywords: themes, concepts, abstract topics
    //
    // The two sets feed two retrieval streams:
    //   - low_level → vector-search the entity store → seed entities →
    //     1-hop expand → mentioning chunks
    //   - high_level → vector-search the relation store → seed
    //     relationships → resolve endpoints → mentioning chunks
    //
    // The four LightRAG modes are characterized by which streams run:
    //   - naive   : neither (chunk-vector RAG only — same as `ask`)
    //   - local   : low-level only (entities)  (== `ask_with_seed_entities`)
    //   - global  : high-level only (relations)
    //   - hybrid  : both streams, merged
    //   - mix     : hybrid + chunk-vector merged
    //
    // graphrag-core exposes one unified API for all of these via
    // `ask_with_dual_seeds` — the caller (e.g. graphrag-server) is
    // responsible for the vector-search step (it owns the embedding
    // service and the Qdrant sidecars); graphrag-core does the
    // graph expansion, context assembly, and LLM call.
    // =====================================================================

    /// LightRAG dual-level keyword extraction. Returns the two keyword
    /// sets the paper specifies, packaged for easy use as embedding
    /// inputs by the caller.
    #[cfg(feature = "async")]
    pub async fn extract_query_keywords(&self, query: &str) -> Result<QueryKeywords> {
        use crate::chat::ChatClient;

        let client = ChatClient::from_config(&self.config.ollama, &self.config.openai)
            .ok_or_else(|| GraphRAGError::Generation {
                message: "no chat backend enabled (config.ollama.enabled / config.openai.enabled both false)".to_string(),
            })?;

        let prompt = format!(
            "---Goal---\n\
             Given a user query, extract two sets of keywords:\n\
             - low_level_keywords: specific named entities, attributes, or details mentioned in the query (people, products, places, concrete things)\n\
             - high_level_keywords: overarching themes, concepts, or topics the query is about (abstractions, categories, processes)\n\n\
             ---Format---\n\
             Output ONLY a single valid JSON object, nothing else (no prose, no markdown, no explanation):\n\
             {{\"high_level_keywords\": [\"...\", \"...\"], \"low_level_keywords\": [\"...\", \"...\"]}}\n\n\
             Each list should have 1-5 keywords. Empty lists are allowed when nothing matches.\n\n\
             ---Query---\n\
             {}\n\n\
             ---Output---",
            query
        );

        let max_answer_tokens: u32 = 256;
        let prompt_tokens = (prompt.len() / 4) as u32;
        let total = prompt_tokens + max_answer_tokens;
        let with_margin = (total as f32 * 1.20) as u32;
        let num_ctx = (((with_margin + 1023) / 1024) * 1024).max(2048).min(16384);
        let params = crate::ollama::OllamaGenerationParams {
            num_predict: Some(max_answer_tokens),
            temperature: Some(0.1), // low temperature; we want deterministic JSON
            num_ctx: Some(num_ctx),
            keep_alive: self.config.ollama.keep_alive.clone(),
            ..Default::default()
        };

        let raw = client
            .generate_with_params(&prompt, params)
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("dual-keyword extraction LLM call failed: {}", e),
            })?;

        let stripped = Self::remove_thinking_tags(&raw);
        // Accept either a bare JSON object or one wrapped in ```json fences.
        let cleaned = stripped
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        // Find the first { and last } and parse just that — robust to
        // models that prepend explanatory prose despite our "ONLY JSON"
        // instruction.
        let json_slice = match (cleaned.find('{'), cleaned.rfind('}')) {
            (Some(a), Some(b)) if b > a => &cleaned[a..=b],
            _ => cleaned,
        };

        match serde_json::from_str::<QueryKeywordsWire>(json_slice) {
            Ok(wire) => Ok(QueryKeywords {
                low_level: wire.low_level_keywords.unwrap_or_default(),
                high_level: wire.high_level_keywords.unwrap_or_default(),
            }),
            Err(e) => {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    error = %e,
                    raw_response = %raw.chars().take(200).collect::<String>(),
                    "dual-keyword JSON parse failed; returning empty keyword sets so caller can fall back"
                );
                let _ = e;
                Ok(QueryKeywords::default())
            },
        }
    }

    /// LightRAG-style retrieval over an arbitrary mix of seed populations
    /// — entities, relationships, and/or text chunks — combined into one
    /// LLM-answered context.
    ///
    /// The four LightRAG retrieval modes are all expressible by varying
    /// which fields of `seeds` are non-empty:
    ///
    /// | LightRAG mode | seeds.entities | seeds.relations | seeds.chunks |
    /// |---|---|---|---|
    /// | local   | non-empty | empty | empty |
    /// | global  | empty | non-empty | empty |
    /// | hybrid  | non-empty | non-empty | empty |
    /// | mix     | non-empty | non-empty | non-empty |
    ///
    /// All four expand entity seeds to 1-hop neighbors (capped by
    /// `max_neighbors_per_seed`), resolve relation seeds to their
    /// source/target entities (and pull those endpoints' neighbors
    /// too), gather every mentioning chunk for every entity touched,
    /// merge the chunk seeds in, and feed the assembled
    /// ENTITIES / RELATIONSHIPS / SOURCE TEXT block to the chat
    /// backend — the same prompt skeleton `ask_with_seed_entities`
    /// uses, so output style stays consistent across modes.
    ///
    /// graphrag-core does not own the seeding step — the caller is
    /// responsible for vector-searching the entity / relationship /
    /// chunk stores and producing the seed sets. graphrag-server's
    /// Qdrant sidecars (`{coll}-entities`, `{coll}-relationships`,
    /// `{coll}` for chunks) are the canonical seeding surface.
    #[cfg(feature = "async")]
    /// Collect every chunk id that `ask_with_dual_seeds` would need
    /// content for — useful for the caller (graphrag-server) to
    /// pre-fetch chunk text from qdrant before invoking
    /// `ask_with_dual_seeds`. Mirrors the entity/relation walk inside
    /// `ask_with_dual_seeds` exactly so the resulting set covers every
    /// chunk that would land in the prompt's SOURCE TEXT block.
    pub fn collect_chunk_ids_for_dual_seeds(
        &self,
        seeds: &DualSeeds,
        max_neighbors_per_seed: usize,
    ) -> Vec<ChunkId> {
        use std::collections::HashSet;
        let kg = match self.knowledge_graph.as_ref() {
            Some(kg) => kg,
            None => return Vec::new(),
        };
        let mut chunk_ids: HashSet<ChunkId> = HashSet::new();
        let mut visited: HashSet<EntityId> = HashSet::new();

        for seed_id in &seeds.entities {
            if let Some(seed) = kg.get_entity(seed_id) {
                visited.insert(seed_id.clone());
                for m in &seed.mentions {
                    chunk_ids.insert(m.chunk_id.clone());
                }
                for (neighbor, _rel) in kg.get_neighbors(seed_id).into_iter().take(max_neighbors_per_seed) {
                    if visited.insert(neighbor.id.clone()) {
                        for m in &neighbor.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                    }
                }
            }
        }
        for (src_id, tgt_id, _relation_type) in &seeds.relations {
            for endpoint_id in [src_id, tgt_id] {
                if let Some(endpoint) = kg.get_entity(endpoint_id) {
                    if visited.insert(endpoint.id.clone()) {
                        for m in &endpoint.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                        for (neighbor, _rel) in kg.get_neighbors(endpoint_id).into_iter().take(max_neighbors_per_seed) {
                            if visited.insert(neighbor.id.clone()) {
                                for m in &neighbor.mentions {
                                    chunk_ids.insert(m.chunk_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        for cid in &seeds.chunks {
            chunk_ids.insert(cid.clone());
        }
        chunk_ids.into_iter().collect()
    }

    /// Collect every chunk id that `ask_with_seed_entities` would need
    /// content for. Walk: seed entity + 1-hop neighbors capped at
    /// `max_neighbors_per_seed`, gather mention chunk ids.
    pub fn collect_chunk_ids_for_seed_entities(
        &self,
        seed_entity_ids: &[EntityId],
        max_neighbors_per_seed: usize,
    ) -> Vec<ChunkId> {
        use std::collections::HashSet;
        let kg = match self.knowledge_graph.as_ref() {
            Some(kg) => kg,
            None => return Vec::new(),
        };
        let mut chunk_ids: HashSet<ChunkId> = HashSet::new();
        let mut visited: HashSet<EntityId> = HashSet::new();
        for seed_id in seed_entity_ids {
            if let Some(seed) = kg.get_entity(seed_id) {
                visited.insert(seed_id.clone());
                for m in &seed.mentions {
                    chunk_ids.insert(m.chunk_id.clone());
                }
                for (neighbor, _rel) in kg.get_neighbors(seed_id).into_iter().take(max_neighbors_per_seed) {
                    if visited.insert(neighbor.id.clone()) {
                        for m in &neighbor.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                    }
                }
            }
        }
        chunk_ids.into_iter().collect()
    }

    pub async fn ask_with_dual_seeds(
        &self,
        query: &str,
        seeds: &DualSeeds,
        max_neighbors_per_seed: usize,
        chunk_contents: &std::collections::HashMap<ChunkId, String>,
    ) -> Result<retrieval::ExplainedAnswer> {
        use crate::chat::ChatClient;
        use std::collections::{HashMap, HashSet};

        let kg = self.knowledge_graph.as_ref().ok_or_else(|| {
            GraphRAGError::Config { message: "Knowledge graph not initialized".to_string() }
        })?;

        let mut entity_set: HashMap<EntityId, Entity> = HashMap::new();
        let mut chunk_ids: HashSet<ChunkId> = HashSet::new();
        let mut bridge_rels: Vec<(String, String, String)> = Vec::new();

        // ---- Stream 1: entity seeds (low-level / local) ---------------
        for seed_id in &seeds.entities {
            if let Some(seed) = kg.get_entity(seed_id) {
                let seed_name = seed.name.clone();
                entity_set.insert(seed_id.clone(), seed.clone());
                for m in &seed.mentions {
                    chunk_ids.insert(m.chunk_id.clone());
                }
                let neighbors = kg.get_neighbors(seed_id);
                for (neighbor, rel) in neighbors.into_iter().take(max_neighbors_per_seed) {
                    bridge_rels.push((
                        seed_name.clone(),
                        rel.relation_type.clone(),
                        neighbor.name.clone(),
                    ));
                    if !entity_set.contains_key(&neighbor.id) {
                        entity_set.insert(neighbor.id.clone(), neighbor.clone());
                        for m in &neighbor.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                    }
                }
            }
        }

        // ---- Stream 2: relation seeds (high-level / global) -----------
        for (src_id, tgt_id, relation_type) in &seeds.relations {
            // Record the bridge triple even if endpoints are missing,
            // so the LLM sees the relation. Use stored ids as fallback names.
            let src_name = kg
                .get_entity(src_id)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| src_id.0.clone());
            let tgt_name = kg
                .get_entity(tgt_id)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| tgt_id.0.clone());
            bridge_rels.push((src_name, relation_type.clone(), tgt_name));

            for endpoint_id in [src_id, tgt_id] {
                if let Some(endpoint) = kg.get_entity(endpoint_id) {
                    if !entity_set.contains_key(&endpoint.id) {
                        entity_set.insert(endpoint.id.clone(), endpoint.clone());
                        for m in &endpoint.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                        // Also expand 1-hop from each relation endpoint.
                        let neighbors = kg.get_neighbors(endpoint_id);
                        for (neighbor, rel) in
                            neighbors.into_iter().take(max_neighbors_per_seed)
                        {
                            bridge_rels.push((
                                endpoint.name.clone(),
                                rel.relation_type.clone(),
                                neighbor.name.clone(),
                            ));
                            if !entity_set.contains_key(&neighbor.id) {
                                entity_set.insert(neighbor.id.clone(), neighbor.clone());
                                for m in &neighbor.mentions {
                                    chunk_ids.insert(m.chunk_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- Stream 3: direct chunk seeds (mix) -----------------------
        for cid in &seeds.chunks {
            chunk_ids.insert(cid.clone());
        }

        // ---- Build context block -------------------------------------
        let entities_block = if entity_set.is_empty() {
            "(no entities resolved in graph)".to_string()
        } else {
            entity_set
                .values()
                .map(|e| {
                    format!(
                        "- {} (type={}, mentioned_in={} chunks, confidence={:.2})",
                        e.name,
                        e.entity_type,
                        e.mentions.len(),
                        e.confidence
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Dedupe bridge_rels in case a triple appears via both streams.
        let mut seen_rels: HashSet<(String, String, String)> = HashSet::new();
        bridge_rels.retain(|t| seen_rels.insert(t.clone()));

        let relationships_block = if bridge_rels.is_empty() {
            "(no relationships gathered)".to_string()
        } else {
            bridge_rels
                .iter()
                .map(|(s, r, t)| format!("- {} --[{}]--> {}", s, r, t))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let chunks_block = chunk_ids
            .iter()
            .filter_map(|cid| chunk_contents.get(cid).map(|c| (cid, c)))
            .map(|(_, c)| format!("- {}", c))
            .collect::<Vec<_>>()
            .join("\n\n");

        let context = format!(
            "ENTITIES:\n{}\n\nRELATIONSHIPS:\n{}\n\nSOURCE TEXT:\n{}",
            entities_block, relationships_block, chunks_block,
        );

        // ---- LLM call -------------------------------------------------
        let client = ChatClient::from_config(&self.config.ollama, &self.config.openai)
            .ok_or_else(|| GraphRAGError::Generation {
                message: "no chat backend enabled (config.ollama.enabled / config.openai.enabled both false)".to_string(),
            })?;

        let prompt = format!(
            "You are a knowledgeable assistant answering questions grounded in a knowledge graph.\n\n\
             IMPORTANT INSTRUCTIONS:\n\
             - Answer ONLY using the provided entities, relationships, and source text below\n\
             - Synthesize across all three sections; relationships in particular often supply the connective tissue\n\
             - Provide direct, conversational, natural responses\n\
             - Do NOT show your reasoning process or use <think> tags\n\
             - If the context lacks sufficient information, clearly state: \"I don't have enough information to answer this question.\"\n\
             - Aim for a complete answer (3-6 sentences)\n\n\
             CONTEXT:\n\
             {}\n\n\
             QUESTION: {}\n\n\
             ANSWER (direct response only, no reasoning):",
            context, query
        );

        let max_answer_tokens: u32 = 800;
        let prompt_tokens = (prompt.len() / 4) as u32;
        let total = prompt_tokens + max_answer_tokens;
        let with_margin = (total as f32 * 1.20) as u32;
        let num_ctx = (((with_margin + 1023) / 1024) * 1024)
            .max(4096)
            .min(131_072);
        let params = crate::ollama::OllamaGenerationParams {
            num_predict: Some(max_answer_tokens),
            temperature: self.config.ollama.temperature,
            num_ctx: Some(num_ctx),
            keep_alive: self.config.ollama.keep_alive.clone(),
            ..Default::default()
        };

        let raw_answer = client
            .generate_with_params(&prompt, params)
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("LLM generation failed: {}", e),
            })?;
        let answer = Self::remove_thinking_tags(&raw_answer).trim().to_string();

        // ---- Pack ExplainedAnswer ------------------------------------
        let total_seeds =
            seeds.entities.len() + seeds.relations.len() + seeds.chunks.len();
        let confidence = if total_seeds == 0 {
            0.0
        } else {
            let chunk_ratio = (chunk_ids.len() as f32) / (total_seeds as f32 * 2.0);
            (chunk_ratio.min(1.0) * 0.7 + 0.3).min(1.0)
        };

        let mut sources: Vec<retrieval::SourceReference> = Vec::new();
        for cid in chunk_ids.iter().take(12) {
            if let Some(content) = chunk_contents.get(cid) {
                sources.push(retrieval::SourceReference {
                    id: cid.0.clone(),
                    source_type: retrieval::SourceType::TextChunk,
                    excerpt: content.chars().take(800).collect(),
                    relevance_score: confidence,
                });
            }
        }
        for (s, r, t) in bridge_rels.iter().take(10) {
            sources.push(retrieval::SourceReference {
                id: format!("{} --[{}]--> {}", s, r, t),
                source_type: retrieval::SourceType::Relationship,
                excerpt: format!("{} {} {}", s, r, t),
                relevance_score: 0.5,
            });
        }

        let key_entities: Vec<String> =
            entity_set.values().map(|e| e.name.clone()).collect();

        let mode_label = match (
            !seeds.entities.is_empty(),
            !seeds.relations.is_empty(),
            !seeds.chunks.is_empty(),
        ) {
            (true, false, false) => "local",
            (false, true, false) => "global",
            (true, true, false) => "hybrid",
            (true, true, true) | (false, _, true) | (true, false, true) => "mix",
            (false, false, false) => "empty",
        };

        let reasoning_steps = vec![
            retrieval::ReasoningStep {
                step_number: 1,
                description: format!(
                    "LightRAG {} mode: caller supplied {} entity seeds, {} relation seeds, {} chunk seeds",
                    mode_label,
                    seeds.entities.len(),
                    seeds.relations.len(),
                    seeds.chunks.len()
                ),
                entities_used: seeds.entities.iter().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 1.0,
            },
            retrieval::ReasoningStep {
                step_number: 2,
                description: format!(
                    "Expanded each seed to 1-hop neighbors (max {} per seed); gathered {} entities total, {} relationship bridges",
                    max_neighbors_per_seed,
                    entity_set.len(),
                    bridge_rels.len()
                ),
                entities_used: entity_set.keys().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 0.9,
            },
            retrieval::ReasoningStep {
                step_number: 3,
                description: format!(
                    "Collected {} mentioning chunks across all gathered entities and direct chunk seeds",
                    chunk_ids.len()
                ),
                entities_used: vec![],
                evidence_snippet: None,
                confidence: 0.85,
            },
            retrieval::ReasoningStep {
                step_number: 4,
                description: "Sent assembled entities + relationships + source text to chat backend".to_string(),
                entities_used: vec![],
                evidence_snippet: None,
                confidence,
            },
        ];

        Ok(retrieval::ExplainedAnswer {
            answer,
            confidence,
            sources,
            reasoning_steps,
            key_entities,
            query_analysis: None,
        })
    }

    /// Microsoft GraphRAG `local_search`-style query: answer the
    /// question by seeding retrieval from a caller-supplied set of
    /// entity ids (typically obtained by vector-searching an
    /// entity-description embedding store), expanding to their
    /// 1-hop neighbors via the relationship graph, gathering the
    /// chunks that mention any of those entities, and feeding the
    /// assembled context to the chat backend.
    ///
    /// Mirrors the seed-traverse-summarize shape of MS GraphRAG's
    /// `local_search` — but the seeding step is the caller's
    /// responsibility (graphrag-core does not own the entity
    /// vector store; graphrag-server's Qdrant sidecar is one such
    /// store).
    ///
    /// `max_neighbors_per_seed` caps fanout so a high-degree entity
    /// doesn't explode the context. Typical value: 3-5.
    ///
    /// Returns an `ExplainedAnswer` with:
    /// - `answer`: LLM-composed response
    /// - `key_entities`: the seed entities + their gathered neighbors
    /// - `sources`: text chunks (with relevance) and the relationship
    ///   triples used as bridges between entities
    /// - `confidence`: rough heuristic over the number of grounded
    ///   chunks vs. seed count
    /// - `reasoning_steps`: the seed-expand-gather pipeline stages
    #[cfg(feature = "async")]
    pub async fn ask_with_seed_entities(
        &self,
        query: &str,
        seed_entity_ids: &[EntityId],
        max_neighbors_per_seed: usize,
        chunk_contents: &std::collections::HashMap<ChunkId, String>,
    ) -> Result<retrieval::ExplainedAnswer> {
        use crate::chat::ChatClient;
        use std::collections::{HashMap, HashSet};

        let kg = self.knowledge_graph.as_ref().ok_or_else(|| {
            GraphRAGError::Config { message: "Knowledge graph not initialized".to_string() }
        })?;

        // 1. Seed expansion: collect seed entities + 1-hop neighbors,
        //    plus the relationship triples that bridge them.
        let mut entity_set: HashMap<EntityId, Entity> = HashMap::new();
        let mut chunk_ids: HashSet<ChunkId> = HashSet::new();
        let mut bridge_rels: Vec<(String, String, String)> = Vec::new(); // (src_name, relation, tgt_name)

        for seed_id in seed_entity_ids {
            if let Some(seed) = kg.get_entity(seed_id) {
                let seed_name = seed.name.clone();
                entity_set.insert(seed_id.clone(), seed.clone());
                for m in &seed.mentions {
                    chunk_ids.insert(m.chunk_id.clone());
                }

                // 1-hop neighbors via the relationship graph.
                let neighbors = kg.get_neighbors(seed_id);
                for (neighbor, rel) in neighbors.into_iter().take(max_neighbors_per_seed) {
                    bridge_rels.push((
                        seed_name.clone(),
                        rel.relation_type.clone(),
                        neighbor.name.clone(),
                    ));
                    if !entity_set.contains_key(&neighbor.id) {
                        entity_set.insert(neighbor.id.clone(), neighbor.clone());
                        for m in &neighbor.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                    }
                }
            }
        }

        // 2. Build the MS-style context block: entities table, relationships
        //    table, source-text section. Truncate aggressively if needed —
        //    the chat backend has its own token budget.
        let entities_block = if entity_set.is_empty() {
            "(no seed entities resolved in graph)".to_string()
        } else {
            entity_set
                .values()
                .map(|e| {
                    format!(
                        "- {} (type={}, mentioned_in={} chunks, confidence={:.2})",
                        e.name, e.entity_type, e.mentions.len(), e.confidence
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let relationships_block = if bridge_rels.is_empty() {
            "(no relationships gathered)".to_string()
        } else {
            bridge_rels
                .iter()
                .map(|(s, r, t)| format!("- {} --[{}]--> {}", s, r, t))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let chunks_block = chunk_ids
            .iter()
            .filter_map(|cid| chunk_contents.get(cid).map(|c| (cid, c)))
            .map(|(_, c)| format!("- {}", c))
            .collect::<Vec<_>>()
            .join("\n\n");

        let context = format!(
            "ENTITIES:\n{}\n\nRELATIONSHIPS:\n{}\n\nSOURCE TEXT:\n{}",
            entities_block, relationships_block, chunks_block,
        );

        // 3. LLM call. Same prompt skeleton + thinking-tag hygiene to
        //    keep output consistent across modes.
        let client = ChatClient::from_config(&self.config.ollama, &self.config.openai)
            .ok_or_else(|| GraphRAGError::Generation {
                message: "no chat backend enabled (config.ollama.enabled / config.openai.enabled both false)".to_string(),
            })?;

        let prompt = format!(
            "You are a knowledgeable assistant answering questions grounded in a knowledge graph.\n\n\
             IMPORTANT INSTRUCTIONS:\n\
             - Answer ONLY using the provided entities, relationships, and source text below\n\
             - Synthesize across all three sections; relationships in particular often supply the connective tissue\n\
             - Provide direct, conversational, natural responses\n\
             - Do NOT show your reasoning process or use <think> tags\n\
             - If the context lacks sufficient information, clearly state: \"I don't have enough information to answer this question.\"\n\
             - Aim for a complete answer (3-6 sentences)\n\n\
             CONTEXT:\n\
             {}\n\n\
             QUESTION: {}\n\n\
             ANSWER (direct response only, no reasoning):",
            context, query
        );

        let max_answer_tokens: u32 = 800;
        let prompt_tokens = (prompt.len() / 4) as u32;
        let total = prompt_tokens + max_answer_tokens;
        let with_margin = (total as f32 * 1.20) as u32;
        let num_ctx = (((with_margin + 1023) / 1024) * 1024)
            .max(4096)
            .min(131_072);

        let params = crate::ollama::OllamaGenerationParams {
            num_predict: Some(max_answer_tokens),
            temperature: self.config.ollama.temperature,
            num_ctx: Some(num_ctx),
            keep_alive: self.config.ollama.keep_alive.clone(),
            ..Default::default()
        };

        let raw_answer = client.generate_with_params(&prompt, params).await.map_err(|e| {
            GraphRAGError::Generation { message: format!("LLM generation failed: {}", e) }
        })?;
        let answer = Self::remove_thinking_tags(&raw_answer).trim().to_string();

        // 4. Pack ExplainedAnswer.
        let confidence = if seed_entity_ids.is_empty() {
            0.0
        } else {
            let chunk_ratio = (chunk_ids.len() as f32) / (seed_entity_ids.len() as f32 * 3.0);
            (chunk_ratio.min(1.0) * 0.7 + 0.3).min(1.0)
        };

        let mut sources: Vec<retrieval::SourceReference> = Vec::new();
        for cid in chunk_ids.iter().take(10) {
            if let Some(content) = chunk_contents.get(cid) {
                sources.push(retrieval::SourceReference {
                    id: cid.0.clone(),
                    source_type: retrieval::SourceType::TextChunk,
                    excerpt: content.chars().take(800).collect(),
                    relevance_score: confidence,
                });
            }
        }
        for (s, r, t) in bridge_rels.iter().take(8) {
            sources.push(retrieval::SourceReference {
                id: format!("{} --[{}]--> {}", s, r, t),
                source_type: retrieval::SourceType::Relationship,
                excerpt: format!("{} {} {}", s, r, t),
                relevance_score: 0.5,
            });
        }

        let key_entities: Vec<String> = entity_set.values().map(|e| e.name.clone()).collect();

        let reasoning_steps = vec![
            retrieval::ReasoningStep {
                step_number: 1,
                description: format!(
                    "Seeded local_search with {} entity ids supplied by caller (vector-search top-K)",
                    seed_entity_ids.len()
                ),
                entities_used: seed_entity_ids.iter().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 1.0,
            },
            retrieval::ReasoningStep {
                step_number: 2,
                description: format!(
                    "Expanded to 1-hop neighbors (max {} per seed); gathered {} entities total, {} relationship bridges",
                    max_neighbors_per_seed, entity_set.len(), bridge_rels.len()
                ),
                entities_used: entity_set.keys().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 0.9,
            },
            retrieval::ReasoningStep {
                step_number: 3,
                description: format!(
                    "Collected {} mentioning chunks across all gathered entities",
                    chunk_ids.len()
                ),
                entities_used: vec![],
                evidence_snippet: None,
                confidence: 0.85,
            },
            retrieval::ReasoningStep {
                step_number: 4,
                description: "Sent assembled entities + relationships + source text to chat backend".to_string(),
                entities_used: vec![],
                evidence_snippet: None,
                confidence,
            },
        ];

        Ok(retrieval::ExplainedAnswer {
            answer,
            confidence,
            sources,
            reasoning_steps,
            key_entities,
            query_analysis: None,
        })
    }


    /// Remove thinking tags from LLM output (for Qwen3 and similar models)
    ///
    /// Qwen3 often outputs <think>...</think> tags showing internal reasoning.
    /// This function removes all such tags and their content.
    #[cfg(feature = "async")]
    fn remove_thinking_tags(text: &str) -> String {
        // Remove all <think>...</think> blocks (including nested ones)
        // Use a simple approach: repeatedly remove until no more found
        let mut result = text.to_string();

        while let Some(start) = result.find("<think>") {
            // Find corresponding closing tag
            if let Some(end) = result[start..].find("</think>") {
                // Remove the entire block
                let end_pos = start + end + "</think>".len();
                result.replace_range(start..end_pos, "");
            } else {
                // No closing tag found, just remove opening tag
                result.replace_range(start..start + "<think>".len(), "");
                break;
            }
        }

        result.trim().to_string()
    }

    /// Get a reference to the current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Check if system is initialized
    pub fn is_initialized(&self) -> bool {
        self.knowledge_graph.is_some() && self.retrieval_system.is_some()
    }

    /// Check if documents have been added
    pub fn has_documents(&self) -> bool {
        if let Some(graph) = &self.knowledge_graph {
            graph.chunks().count() > 0
        } else {
            false
        }
    }

    /// Check if graph has been built
    pub fn has_graph(&self) -> bool {
        if let Some(graph) = &self.knowledge_graph {
            graph.entities().count() > 0
        } else {
            false
        }
    }

    /// Get a reference to the knowledge graph
    pub fn knowledge_graph(&self) -> Option<&KnowledgeGraph> {
        self.knowledge_graph.as_ref()
    }

    /// Get entity details by ID
    pub fn get_entity(&self, entity_id: &str) -> Option<&Entity> {
        if let Some(graph) = &self.knowledge_graph {
            graph.entities().find(|e| e.id.0 == entity_id)
        } else {
            None
        }
    }

    /// Get all relationships involving an entity
    pub fn get_entity_relationships(&self, entity_id: &str) -> Vec<&Relationship> {
        if let Some(graph) = &self.knowledge_graph {
            let entity_id_obj = EntityId::new(entity_id.to_string());
            graph
                .relationships()
                .filter(|r| r.source == entity_id_obj || r.target == entity_id_obj)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get chunk by ID
    pub fn get_chunk(&self, chunk_id: &str) -> Option<&TextChunk> {
        if let Some(graph) = &self.knowledge_graph {
            graph.chunks().find(|c| c.id.0 == chunk_id)
        } else {
            None
        }
    }


    /// Get a mutable reference to the knowledge graph
    pub fn knowledge_graph_mut(&mut self) -> Option<&mut KnowledgeGraph> {
        self.knowledge_graph.as_mut()
    }

    // ================================
    // CONVENIENCE CONSTRUCTORS
    // ================================

    /// Create GraphRAG from a JSON5 config file
    ///
    /// This is a convenience method that loads a JSON5 config file and creates a GraphRAG instance.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "json5-support")]
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// let graphrag = GraphRAG::from_json5_file("config/templates/symposium_zero_cost.graphrag.json5")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "json5-support")]
    pub fn from_json5_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use crate::config::json5_loader::load_json5_config;
        use crate::config::setconfig::SetConfig;

        let set_config = load_json5_config::<SetConfig, _>(path)?;
        let config = set_config.to_graphrag_config();
        Self::new(config)
    }

    /// Create GraphRAG from a config file (auto-detect format: TOML, JSON5, YAML, JSON)
    ///
    /// This method automatically detects the config file format based on the file extension
    /// and loads it appropriately.
    ///
    /// Supported formats:
    /// - `.toml` - TOML format
    /// - `.json5` - JSON5 format (requires `json5-support` feature)
    /// - `.yaml`, `.yml` - YAML format
    /// - `.json` - JSON format
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// // Auto-detect format from extension
    /// let graphrag = GraphRAG::from_config_file("config/templates/symposium_zero_cost.graphrag.json5")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_config_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use crate::config::setconfig::SetConfig;

        let set_config = SetConfig::from_file(path)?;
        let config = set_config.to_graphrag_config();
        Self::new(config)
    }


    /// Ensure system is initialized
    fn ensure_initialized(&mut self) -> Result<()> {
        if !self.is_initialized() {
            self.initialize()
        } else {
            Ok(())
        }
    }
}
