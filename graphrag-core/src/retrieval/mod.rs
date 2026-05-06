/// BM25 text retrieval implementation for keyword-based search
pub mod bm25;
/// Causal chain analysis for discovering cause-effect paths (Phase 2.3)
pub mod causal_analysis;
/// Enriched metadata-aware retrieval
pub mod enriched;
/// HippoRAG Personalized PageRank retrieval
#[cfg(feature = "pagerank")]
pub mod hipporag_ppr;
/// Hybrid retrieval combining multiple search strategies
pub mod hybrid;
pub mod pagerank_retrieval;
/// Symbolic anchoring for conceptual queries (Phase 2.1 - CatRAG)
pub mod symbolic_anchoring;

#[cfg(feature = "parallel-processing")]
use crate::parallel::ParallelProcessor;
use crate::{
    config::Config,
    core::{traits::DynEmbedder, ChunkId, EntityId, KnowledgeGraph},
    summarization::DocumentTree,
    vector::{HashEmbedder, VectorUtils},
    Result,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub use bm25::{BM25Result, BM25Retriever, Document as BM25Document};
pub use enriched::{EnrichedRetrievalConfig, EnrichedRetriever};
pub use hybrid::{FusionMethod, HybridConfig, HybridRetriever, HybridSearchResult};

#[cfg(feature = "pagerank")]
pub use pagerank_retrieval::{PageRankRetrievalSystem, ScoredResult};

#[cfg(feature = "pagerank")]
pub use hipporag_ppr::{Fact, HippoRAGConfig, HippoRAGRetriever};

use crate::vector::store::VectorStore;

/// Retrieval system for querying the knowledge graph.
///
/// Holds a single `embedder: DynEmbedder` — the only path through which
/// query-time and index-time embeddings are generated. Hosts (e.g.
/// graphrag-server) inject their real backend via
/// [`Self::set_embedding_provider`]; without injection, retrieval uses
/// a hash-based [`HashEmbedder`] sized to `config.embeddings.dimension`.
#[derive(Clone)]
pub struct RetrievalSystem {
    vector_store: std::sync::Arc<dyn VectorStore>,
    /// Single source of truth for embedding generation. Always populated.
    /// Swapped in place by [`Self::set_embedding_provider`].
    embedder: DynEmbedder,
    config: RetrievalConfig,
    #[cfg(feature = "parallel-processing")]
    parallel_processor: Option<ParallelProcessor>,
    // Phase 5 partial: pagerank_retriever and enriched_retriever
    // fields removed. Their owning modules are unreachable post-Phase 4
    // and never get populated. Removing them lets RetrievalSystem
    // derive Clone (Layer 4 needs it for ArcSwap copy-on-write).
    #[cfg(feature = "lazygraphrag")]
    concept_filtering_enabled: bool,
}

impl RetrievalSystem {
    /// Create a new retrieval system. The embedder is initialized to a
    /// hash-based [`HashEmbedder`] sized to `config.embeddings.dimension`;
    /// hosts that need a real backend should call
    /// [`Self::set_embedding_provider`] immediately after construction.
    pub fn new(config: &Config) -> Result<Self> {
        let retrieval_config = RetrievalConfig {
            top_k: config.retrieval.top_k,
            similarity_threshold: 0.35,
            max_expansion_depth: 2,
            entity_weight: 0.4,
            chunk_weight: 0.4,
            graph_weight: 0.2,
            #[cfg(feature = "lazygraphrag")]
            use_concept_filtering: false,
            #[cfg(feature = "lazygraphrag")]
            concept_top_k: 20,
        };

        // Default to MemoryVectorStore for now (mimics old behavior)
        // In the future, this will select based on Config (LanceDB, Qdrant, etc.)
        let vector_store =
            std::sync::Arc::new(crate::vector::memory_store::MemoryVectorStore::new());

        let embedder: DynEmbedder = Arc::new(HashEmbedder::new(config.embeddings.dimension));

        Ok(Self {
            vector_store,
            embedder,
            config: retrieval_config,
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
            #[cfg(feature = "lazygraphrag")]
            concept_filtering_enabled: false,
        })
    }

    /// Replace the active embedder. After this call, every internal
    /// embedding (query-time vector search, index-time chunk/entity
    /// embedding, relationship-similarity scoring) routes through the
    /// new provider.
    pub fn set_embedding_provider(&mut self, provider: DynEmbedder) {
        self.embedder = provider;
    }

    /// Embed a single text via the active embedder. The lone embedding
    /// path inside the retrieval system — every other site delegates here.
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder.embed(text).await
    }
}

/// Configuration parameters for the retrieval system
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum number of results to return
    pub top_k: usize,
    /// Minimum similarity score threshold for results (typically -1.0 to 1.0)
    pub similarity_threshold: f32,
    /// Maximum depth for graph relationship expansion
    pub max_expansion_depth: usize,
    /// Weight for entity-based results in scoring (0.0 to 1.0)
    pub entity_weight: f32,
    /// Weight for chunk-based results in scoring (0.0 to 1.0)
    pub chunk_weight: f32,
    /// Weight for graph-based results in scoring (0.0 to 1.0)
    pub graph_weight: f32,
    /// Enable concept-based chunk filtering (requires lazygraphrag feature)
    #[cfg(feature = "lazygraphrag")]
    pub use_concept_filtering: bool,
    /// Top-K concepts to select for filtering (requires lazygraphrag feature)
    #[cfg(feature = "lazygraphrag")]
    pub concept_top_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            similarity_threshold: 0.7,
            max_expansion_depth: 2,
            entity_weight: 0.4,
            chunk_weight: 0.4,
            graph_weight: 0.2,
            #[cfg(feature = "lazygraphrag")]
            use_concept_filtering: false,
            #[cfg(feature = "lazygraphrag")]
            concept_top_k: 20,
        }
    }
}

/// A search result containing relevant information
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unique identifier for this result
    pub id: String,
    /// Content or description of the result
    pub content: String,
    /// Relevance score (higher is better)
    pub score: f32,
    /// Type of result (entity, chunk, graph path, etc.)
    pub result_type: ResultType,
    /// Names of entities associated with this result
    pub entities: Vec<String>,
    /// IDs of source chunks this result is derived from
    pub source_chunks: Vec<String>,
}

/// Type of search result indicating the retrieval strategy used
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResultType {
    /// Result from entity-based retrieval
    Entity,
    /// Result from text chunk retrieval
    Chunk,
    /// Result from graph path traversal
    GraphPath,
    /// Result from hierarchical document summarization
    HierarchicalSummary,
    /// Result from combining multiple retrieval strategies
    Hybrid,
}

// ============================================================================
// EXPLAINED ANSWER - Structured answer with reasoning trace
// ============================================================================

/// An answer with detailed explanation of the reasoning process
///
/// This struct provides transparency into how the GraphRAG system
/// arrived at its answer, including confidence scores, source references,
/// and step-by-step reasoning.
///
/// # Example
/// ```no_run
/// use graphrag_core::prelude::*;
///
/// # async fn example() -> graphrag_core::Result<()> {
/// let mut graphrag = GraphRAG::quick_start("Your document").await?;
/// let explained = graphrag.ask_explained("What is the main topic?").await?;
///
/// println!("Answer: {}", explained.answer);
/// println!("Confidence: {:.0}%", explained.confidence * 100.0);
///
/// for step in &explained.reasoning_steps {
///     println!("Step {}: {} (confidence: {:.0}%)",
///         step.step_number, step.description, step.confidence * 100.0);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ExplainedAnswer {
    /// The answer text
    pub answer: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Sources used to generate the answer
    pub sources: Vec<SourceReference>,
    /// Step-by-step reasoning trace
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Entities that were key to the answer
    pub key_entities: Vec<String>,
    /// Query analysis that guided retrieval
    pub query_analysis: Option<QueryAnalysis>,
}

/// Reference to a source document or chunk used in the answer
#[derive(Debug, Clone)]
pub struct SourceReference {
    /// Identifier of the source (chunk ID, document ID, or entity ID)
    pub id: String,
    /// Type of source
    pub source_type: SourceType,
    /// Relevant excerpt from the source
    pub excerpt: String,
    /// Relevance score to the query
    pub relevance_score: f32,
}

/// Type of source reference
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceType {
    /// A text chunk from a document
    TextChunk,
    /// An entity in the knowledge graph
    Entity,
    /// A relationship between entities
    Relationship,
    /// A document-level summary
    Summary,
}

/// A single step in the reasoning process
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step number (1-indexed)
    pub step_number: u8,
    /// Description of what was done in this step
    pub description: String,
    /// IDs of entities involved in this step
    pub entities_used: Vec<String>,
    /// Evidence snippet that supports this step
    pub evidence_snippet: Option<String>,
    /// Confidence for this specific step
    pub confidence: f32,
}

impl ExplainedAnswer {
    /// Create a new explained answer from search results
    pub fn from_results(answer: String, search_results: &[SearchResult], query: &str) -> Self {
        // Calculate overall confidence from result scores
        let confidence = if search_results.is_empty() {
            0.0
        } else {
            let total_score: f32 = search_results.iter().map(|r| r.score).sum();
            let avg_score = total_score / search_results.len() as f32;
            // Normalize to 0-1 range (assuming scores are already somewhat normalized)
            (avg_score * 0.7 + 0.3).min(1.0).max(0.0)
        };

        // Build source references
        let sources: Vec<SourceReference> = search_results
            .iter()
            .take(5) // Top 5 sources
            .map(|r| SourceReference {
                id: r.id.clone(),
                source_type: match r.result_type {
                    ResultType::Entity => SourceType::Entity,
                    ResultType::Chunk => SourceType::TextChunk,
                    ResultType::GraphPath => SourceType::Relationship,
                    ResultType::HierarchicalSummary => SourceType::Summary,
                    ResultType::Hybrid => SourceType::TextChunk,
                },
                excerpt: if r.content.len() > 200 {
                    format!("{}...", &r.content[..200])
                } else {
                    r.content.clone()
                },
                relevance_score: r.score,
            })
            .collect();

        // Build reasoning steps
        let mut reasoning_steps = Vec::new();
        let mut step_num = 1u8;

        // Step 1: Query analysis
        reasoning_steps.push(ReasoningStep {
            step_number: step_num,
            description: format!("Analyzed query: \"{}\"", query),
            entities_used: vec![],
            evidence_snippet: None,
            confidence: 0.95,
        });
        step_num += 1;

        // Step 2: Entity retrieval
        let unique_entities: HashSet<_> = search_results
            .iter()
            .flat_map(|r| r.entities.iter().cloned())
            .collect();
        if !unique_entities.is_empty() {
            reasoning_steps.push(ReasoningStep {
                step_number: step_num,
                description: format!("Found {} relevant entities", unique_entities.len()),
                entities_used: unique_entities.iter().take(5).cloned().collect(),
                evidence_snippet: None,
                confidence: 0.85,
            });
            step_num += 1;
        }

        // Step 3: Chunk retrieval
        let chunk_count = search_results
            .iter()
            .filter(|r| r.result_type == ResultType::Chunk || r.result_type == ResultType::Hybrid)
            .count();
        if chunk_count > 0 {
            reasoning_steps.push(ReasoningStep {
                step_number: step_num,
                description: format!("Retrieved {} relevant text chunks", chunk_count),
                entities_used: vec![],
                evidence_snippet: search_results.first().map(|r| {
                    if r.content.len() > 100 {
                        format!("{}...", &r.content[..100])
                    } else {
                        r.content.clone()
                    }
                }),
                confidence,
            });
            step_num += 1;
        }

        // Step 4: Answer synthesis
        reasoning_steps.push(ReasoningStep {
            step_number: step_num,
            description: "Synthesized answer from retrieved information".to_string(),
            entities_used: unique_entities.into_iter().take(3).collect(),
            evidence_snippet: None,
            confidence,
        });

        // Collect key entities
        let key_entities: Vec<String> = search_results
            .iter()
            .flat_map(|r| r.entities.iter().cloned())
            .take(10)
            .collect();

        Self {
            answer,
            confidence,
            sources,
            reasoning_steps,
            key_entities,
            query_analysis: None,
        }
    }

    /// Format the explained answer for display
    pub fn format_display(&self) -> String {
        let mut output = String::new();

        // Answer
        output.push_str(&format!("**Answer:** {}\n\n", self.answer));

        // Confidence
        output.push_str(&format!(
            "**Confidence:** {:.0}%\n\n",
            self.confidence * 100.0
        ));

        // Reasoning steps
        if !self.reasoning_steps.is_empty() {
            output.push_str("**Reasoning:**\n");
            for step in &self.reasoning_steps {
                output.push_str(&format!(
                    "{}. {} (confidence: {:.0}%)\n",
                    step.step_number,
                    step.description,
                    step.confidence * 100.0
                ));
                if let Some(evidence) = &step.evidence_snippet {
                    output.push_str(&format!("   Evidence: \"{}\"\n", evidence));
                }
            }
            output.push('\n');
        }

        // Sources
        if !self.sources.is_empty() {
            output.push_str("**Sources:**\n");
            for (i, source) in self.sources.iter().enumerate() {
                output.push_str(&format!(
                    "{}. [{:?}] {} (relevance: {:.0}%)\n",
                    i + 1,
                    source.source_type,
                    source.id,
                    source.relevance_score * 100.0
                ));
            }
        }

        output
    }
}

// ============================================================================
// QUERY ANALYSIS - Adaptive retrieval strategy
// ============================================================================

/// Query analysis results to determine optimal retrieval strategy
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Type of query based on content analysis
    pub query_type: QueryType,
    /// Key entities detected in the query
    pub key_entities: Vec<String>,
    /// Conceptual terms extracted from the query
    pub concepts: Vec<String>,
    /// Inferred user intent from the query
    pub intent: QueryIntent,
    /// Query complexity score (0.0 to 1.0)
    pub complexity_score: f32,
}

/// Classification of query types for adaptive retrieval strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// Queries focused on specific entities
    EntityFocused,
    /// Abstract concept queries requiring broader context
    Conceptual,
    /// Specific fact retrieval queries
    Factual,
    /// Open-ended exploration queries
    Exploratory,
    /// Queries about relationships between entities
    Relationship,
}

/// User intent classification for result presentation
#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    /// User wants a high-level summary or overview
    Overview,
    /// User wants detailed, specific information
    Detailed,
    /// User wants to compare multiple items
    Comparative,
    /// User wants to understand cause-effect relationships
    Causal,
    /// User wants time-based or chronological information
    Temporal,
}

/// Query analysis result with additional metadata for adaptive retrieval
#[derive(Debug, Clone)]
pub struct QueryAnalysisResult {
    /// Detected query type
    pub query_type: QueryType,
    /// Confidence score for the detected query type (0.0 to 1.0)
    pub confidence: f32,
    /// Keywords extracted and matched from the query
    pub keywords_matched: Vec<String>,
    /// Recommended retrieval strategies based on analysis
    pub suggested_strategies: Vec<String>,
    /// Overall query complexity score (0.0 to 1.0)
    pub complexity_score: f32,
}

/// Query result with hierarchical summary
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Original query string
    pub query: String,
    /// List of search results
    pub results: Vec<SearchResult>,
    /// Optional generated summary of all results
    pub summary: Option<String>,
    /// Additional metadata about the query execution
    pub metadata: HashMap<String, String>,
}


/// Statistics about the retrieval system
#[derive(Debug)]
pub struct RetrievalStatistics {
    /// Number of vectors indexed in the system
    pub indexed_vectors: usize,
    /// Dimensionality of the vector embeddings
    pub vector_dimension: usize,
    /// Whether the vector index has been built
    pub index_built: bool,
    /// Current retrieval configuration
    pub config: RetrievalConfig,
}

impl RetrievalStatistics {
    /// Print retrieval statistics
    #[allow(dead_code)]
    pub fn print(&self) {
        tracing::info!("Retrieval System Statistics:");
        tracing::info!("  Indexed vectors: {}", self.indexed_vectors);
        tracing::info!("  Vector dimension: {}", self.vector_dimension);
        tracing::info!("  Index built: {}", self.index_built);
        tracing::info!("  Configuration:");
        tracing::info!("    Top K: {}", self.config.top_k);
        tracing::info!(
            "    Similarity threshold: {:.2}",
            self.config.similarity_threshold
        );
        tracing::info!(
            "    Max expansion depth: {}",
            self.config.max_expansion_depth
        );
        tracing::info!("    Entity weight: {:.2}", self.config.entity_weight);
        tracing::info!("    Chunk weight: {:.2}", self.config.chunk_weight);
        tracing::info!("    Graph weight: {:.2}", self.config.graph_weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, core::KnowledgeGraph};

    #[test]
    fn test_retrieval_system_creation() {
        let config = Config::default();
        let retrieval = RetrievalSystem::new(&config);
        assert!(retrieval.is_ok());
    }

    #[test]
    fn test_query_placeholder() {
        let config = Config::default();
        let retrieval = RetrievalSystem::new(&config).unwrap();

        let results = retrieval.query("test query");
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].contains("test query"));
    }

    #[tokio::test]
    async fn test_graph_indexing() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).unwrap();
        let graph = KnowledgeGraph::new();

        let result = retrieval.index_graph(&graph).await;
        assert!(result.is_ok());
    }

    // ============================================================================
    // ExplainedAnswer Tests
    // ============================================================================

    #[test]
    fn test_explained_answer_creation() {
        let search_results = vec![
            SearchResult {
                id: "chunk_1".to_string(),
                content: "This is the first relevant chunk about climate change.".to_string(),
                score: 0.85,
                result_type: ResultType::Chunk,
                entities: vec!["climate".to_string(), "environment".to_string()],
                source_chunks: vec!["doc1_chunk1".to_string()],
            },
            SearchResult {
                id: "chunk_2".to_string(),
                content: "Another chunk discussing environmental policies.".to_string(),
                score: 0.72,
                result_type: ResultType::Chunk,
                entities: vec!["policy".to_string(), "environment".to_string()],
                source_chunks: vec!["doc1_chunk2".to_string()],
            },
        ];

        let explained = ExplainedAnswer::from_results(
            "Climate change is a major environmental concern.".to_string(),
            &search_results,
            "What is climate change?",
        );

        assert!(!explained.answer.is_empty());
        assert!(explained.confidence > 0.0 && explained.confidence <= 1.0);
        assert!(!explained.sources.is_empty());
        assert!(!explained.reasoning_steps.is_empty());
    }

    #[test]
    fn test_explained_answer_empty_results() {
        let explained = ExplainedAnswer::from_results(
            "No relevant information found.".to_string(),
            &[],
            "What is something unknown?",
        );

        assert_eq!(explained.confidence, 0.0);
        assert!(explained.sources.is_empty());
        assert!(!explained.reasoning_steps.is_empty()); // Should still have query analysis step
    }

    #[test]
    fn test_explained_answer_format_display() {
        let search_results = vec![SearchResult {
            id: "test_chunk".to_string(),
            content: "Test content about technology.".to_string(),
            score: 0.9,
            result_type: ResultType::Chunk,
            entities: vec!["technology".to_string()],
            source_chunks: vec!["doc1_chunk1".to_string()],
        }];

        let explained = ExplainedAnswer::from_results(
            "Technology is important.".to_string(),
            &search_results,
            "Why is technology important?",
        );

        let formatted = explained.format_display();

        assert!(formatted.contains("**Answer:**"));
        assert!(formatted.contains("**Confidence:**"));
        assert!(formatted.contains("**Reasoning:**"));
        assert!(formatted.contains("**Sources:**"));
    }

    #[test]
    fn test_reasoning_steps_structure() {
        let search_results = vec![SearchResult {
            id: "entity_1".to_string(),
            content: "Entity description".to_string(),
            score: 0.8,
            result_type: ResultType::Entity,
            entities: vec!["person".to_string(), "organization".to_string()],
            source_chunks: vec![],
        }];

        let explained = ExplainedAnswer::from_results(
            "Answer text".to_string(),
            &search_results,
            "Who are the key people?",
        );

        // Check reasoning steps are numbered correctly
        for (i, step) in explained.reasoning_steps.iter().enumerate() {
            assert_eq!(step.step_number as usize, i + 1);
            assert!(!step.description.is_empty());
            assert!(step.confidence >= 0.0 && step.confidence <= 1.0);
        }
    }

    #[test]
    fn test_source_reference_types() {
        let search_results = vec![
            SearchResult {
                id: "chunk".to_string(),
                content: "Chunk content".to_string(),
                score: 0.7,
                result_type: ResultType::Chunk,
                entities: vec![],
                source_chunks: vec![],
            },
            SearchResult {
                id: "entity".to_string(),
                content: "Entity content".to_string(),
                score: 0.6,
                result_type: ResultType::Entity,
                entities: vec![],
                source_chunks: vec![],
            },
            SearchResult {
                id: "path".to_string(),
                content: "Graph path content".to_string(),
                score: 0.5,
                result_type: ResultType::GraphPath,
                entities: vec![],
                source_chunks: vec![],
            },
        ];

        let explained =
            ExplainedAnswer::from_results("Answer".to_string(), &search_results, "Query");

        let source_types: Vec<_> = explained.sources.iter().map(|s| &s.source_type).collect();
        assert!(source_types.contains(&&SourceType::TextChunk));
        assert!(source_types.contains(&&SourceType::Entity));
        assert!(source_types.contains(&&SourceType::Relationship));
    }
}
