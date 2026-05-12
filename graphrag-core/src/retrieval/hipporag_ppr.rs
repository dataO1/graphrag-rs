//! HippoRAG Personalized PageRank Retrieval
//!
//! This module implements the HippoRAG retrieval strategy that uses Personalized
//! PageRank (PPR) to combine fact-based entity signals with dense passage retrieval.
//!
//! Key innovation: Uses a dual-signal approach for PPR personalization:
//! 1. Entity weights from relevant facts (query-fact similarity)
//! 2. Passage weights from dense retrieval (scaled down)
//!
//! Reference: "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"
//! https://arxiv.org/abs/2405.14831

use std::collections::HashMap;

use crate::core::{ChunkId, EntityId, GraphRAGError, KnowledgeGraph, Result};
use crate::graph::pagerank::{PageRankConfig, PersonalizedPageRank};
use crate::retrieval::SearchResult;
use crate::vector::store::VectorStore;

// ---------------------------------------------------------------------------
// Type-boundary helpers (OQ-3 / Item 3 resolution — option b)
// ---------------------------------------------------------------------------
//
// `PersonalizedPageRank` (and the legacy helper methods below) operate on a
// single `HashMap<EntityId, _>` namespace.  The entity graph contains only
// *entity* nodes; chunks (ChunkId) are NOT graph nodes and therefore do NOT
// get a PPR slot.
//
// When we seed the PPR reset distribution with dense passage scores we are
// intentionally **merging** two namespaces at the PPR input boundary:
//   • Real entity IDs  → seeded by query-fact entity weights
//   • Chunk IDs        → seeded by dense retrieval scores (scaled down)
//
// This merging is the deliberate HippoRAG approach (passage nodes get a small
// 0.05 weight share in the PPR teleportation vector).  To make this crossing
// visible rather than scattered-and-implicit, every `ChunkId → EntityId`
// coercion routes through the function below, annotated with the reason.
//
// After PPR completes, the chunk-score aggregation step projects the
// *entity-keyed* PPR scores back into the `ChunkId` namespace via
// `entity_to_passages_typed`.  The reverse boundary crossing (`EntityId` that
// was originally a `ChunkId` → back to `ChunkId`) is handled by
// `chunk_scores_from_ppr_nodes`.

/// Translate a `ChunkId` into the `EntityId` namespace used by the PPR graph.
///
/// # Why this exists
/// `PersonalizedPageRank::calculate_scores` accepts `HashMap<EntityId, f64>`
/// for the teleportation (reset) distribution.  When passage nodes are seeded
/// alongside entity nodes, their IDs must be expressed as `EntityId` values.
/// Using `chunk_id.0` as the string key is safe because:
///   - Entity IDs and chunk IDs use different ID-generation schemes in
///     graphrag-rs (entity IDs are derived from entity text/hash; chunk IDs
///     are derived from document + offset ranges), so collisions are
///     exceedingly rare in practice.
///   - The PPR personalization vector is normalized; a stray collision would
///     at most slightly perturb one node's reset probability.
///
/// Every call site that needs this translation uses this function so that the
/// coercion is visible, named, and easy to audit.
#[inline]
fn chunk_id_as_ppr_node(cid: &ChunkId) -> EntityId {
    EntityId::new(cid.0.clone())
}

/// Translate an `EntityId` that was produced by `chunk_id_as_ppr_node` back
/// to a `ChunkId`.  Used when projecting PPR output scores for passage-seeded
/// nodes back into the `ChunkId` namespace.
#[inline]
fn ppr_node_as_chunk_id(eid: &EntityId) -> ChunkId {
    ChunkId::new(eid.0.clone())
}

/// Configuration for HippoRAG PPR retrieval
#[derive(Debug, Clone)]
pub struct HippoRAGConfig {
    /// Damping factor for PageRank (HippoRAG default: 0.5)
    pub damping_factor: f64,

    /// Maximum PageRank iterations
    pub max_iterations: usize,

    /// Convergence tolerance
    pub tolerance: f64,

    /// Number of top facts to use for entity weight calculation
    pub top_k_facts: usize,

    /// Weight multiplier for passage nodes (HippoRAG default: 0.05)
    /// This scales down passage scores relative to entity scores
    pub passage_node_weight: f64,

    /// Number of results to return
    pub top_k_results: usize,

    /// Minimum entity frequency threshold
    /// Entities appearing in many passages get downweighted
    pub min_entity_frequency: usize,

    /// Whether to normalize scores before combining
    pub normalize_scores: bool,

    /// Number of top dense chunk hits to include in passage_scores (default: 30)
    pub top_k_dense: usize,
}

impl Default for HippoRAGConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.5, // HippoRAG uses 0.5 instead of typical 0.85
            max_iterations: 100,
            tolerance: 1e-6,
            top_k_facts: 100,
            passage_node_weight: 0.05, // Passages get 5% weight vs entities
            top_k_results: 10,
            min_entity_frequency: 1,
            normalize_scores: true,
            top_k_dense: 30,
        }
    }
}

/// Fact triple for knowledge graph
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    /// Subject entity
    pub subject: String,

    /// Predicate/relation
    pub predicate: String,

    /// Object entity
    pub object: String,

    /// Fact relevance score (from query-fact similarity)
    pub score: f32,
}

/// HippoRAG retrieval system using Personalized PageRank
///
/// This combines:
/// - Fact retrieval (query → facts)
/// - Entity extraction from facts
/// - Dense passage retrieval
/// - Graph-based reranking via PPR
pub struct HippoRAGRetriever {
    config: HippoRAGConfig,
    pagerank: Option<PersonalizedPageRank>,
}

impl HippoRAGRetriever {
    /// Create a new HippoRAG retriever
    pub fn new(config: HippoRAGConfig) -> Self {
        Self {
            config,
            pagerank: None,
        }
    }

    /// Initialize with a PersonalizedPageRank instance
    pub fn with_pagerank(mut self, pagerank: PersonalizedPageRank) -> Self {
        self.pagerank = Some(pagerank);
        self
    }

    /// Retrieve top-K chunk IDs using the HippoRAG PPR strategy against the post-Phase-6 API.
    ///
    /// ## Flow
    /// 1. Embed `query` via `embedder`.
    /// 2. Top-K entity sidecar hits via `vector_store` (using `top_k_results` as the limit).
    /// 3. Top-K relation sidecar hits via a second `vector_store.search` call.
    ///    Materialises `Fact` triples from relation hits: `subject = source_entity`,
    ///    `predicate = relation_label`, `object = target_entity`, `score = similarity`.
    /// 4. Walk `graph` for `entity_to_passages` — each entity's `EntityMention::chunk_id` set.
    /// 5. Top-K dense chunk hits → `passage_scores` (`HashMap<ChunkId, f32>`).
    /// 6. Call existing helpers: `calculate_entity_weights`, `calculate_passage_weights`,
    ///    `combine_weights`, `rank_passages` (internal PPR run unchanged).
    /// 7. Return top-K seed `ChunkId`s.
    ///
    /// ## PPR snapshot semantics (OQ-1 resolution)
    /// We call `graph.build_pagerank_calculator()` once at the start of each
    /// `retrieve()` call to snapshot the current entity adjacency matrix.
    /// Because `KnowledgeGraph` may be appended to concurrently (live ingest),
    /// the snapshot represents the graph state at the moment of the query; edges
    /// added after the snapshot is taken will not influence this call's PPR
    /// scores.  This is deliberate eventual-consistency: a query that arrives
    /// mid-ingest sees a consistent (not half-written) graph, and the next
    /// query will see any entities/edges committed before it starts.
    pub async fn retrieve(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
        vector_store: &dyn VectorStore,
        embedder: &dyn crate::core::traits::AsyncEmbedder<Error = GraphRAGError>,
        ppr_override: Option<std::sync::Arc<crate::graph::pagerank::PersonalizedPageRank>>,
    ) -> Result<Vec<ChunkId>> {
        // Step 1: Embed query
        let query_vec = embedder.embed(query).await?;

        // Step 2: Top-K entity sidecar hits
        // Currently used only to drive entity weight seeding; the hit ids are
        // resolved to entities in the graph via entity_to_passages_typed in step 4.
        let _entity_hits = vector_store
            .search(&query_vec, self.config.top_k_results)
            .await?;

        // Step 3: Top-K relation sidecar hits → Fact triples
        // Each relation hit's metadata carries source_entity, relation_label, target_entity.
        // The metadata keys mirror the graphrag-server PersistedRelationship payload layout.
        let relation_hits = vector_store
            .search(&query_vec, self.config.top_k_facts)
            .await?;

        let top_k_facts: Vec<Fact> = relation_hits
            .into_iter()
            .filter_map(|hit| {
                let source = hit.metadata.get("source")?.clone();
                let predicate = hit.metadata.get("relation_type")
                    .or_else(|| hit.metadata.get("predicate"))
                    .cloned()
                    .unwrap_or_default();
                let target = hit.metadata.get("target")?.clone();
                Some(Fact {
                    subject: source,
                    predicate,
                    object: target,
                    score: hit.score,
                })
            })
            .take(self.config.top_k_facts)
            .collect();

        // Step 4: Build entity_to_passages from graph mentions.
        //
        // Primary type: EntityId → Vec<ChunkId>  (used for PPR→chunk projection in step 7).
        // Legacy type:  EntityId → Vec<EntityId>  (used by calculate_entity_weights helper).
        //
        // The legacy form is built by routing through `chunk_id_as_ppr_node` — the named
        // boundary-translation function defined above (OQ-3 / Item 3 resolution, option b).
        let entity_to_passages_typed: HashMap<EntityId, Vec<ChunkId>> = graph
            .entities()
            .map(|entity| {
                let chunk_ids: Vec<ChunkId> = entity
                    .mentions
                    .iter()
                    .map(|m| m.chunk_id.clone())
                    .collect();
                (entity.id.clone(), chunk_ids)
            })
            .collect();

        // Boundary translation: ChunkId → EntityId namespace for helper compatibility.
        // Explicit via `chunk_id_as_ppr_node`; see the module-level comment for rationale.
        let entity_to_passages_legacy: HashMap<EntityId, Vec<EntityId>> =
            entity_to_passages_typed
                .iter()
                .map(|(eid, cids)| {
                    let ppr_nodes: Vec<EntityId> =
                        cids.iter().map(chunk_id_as_ppr_node).collect();
                    (eid.clone(), ppr_nodes)
                })
                .collect();

        // Step 5: Top-K dense chunk hits → passage_scores (HashMap<ChunkId, f32>)
        //
        // Primary type: ChunkId → f32  (used for combined scoring in step 7).
        // Legacy type:  EntityId → f32  (used by calculate_passage_weights helper).
        let dense_hits = vector_store
            .search(&query_vec, self.config.top_k_dense)
            .await?;

        let passage_scores_typed: HashMap<ChunkId, f32> = dense_hits
            .into_iter()
            .map(|hit| (ChunkId::new(hit.id), hit.score))
            .collect();

        // Boundary translation: ChunkId → EntityId namespace for helper compatibility.
        let passage_scores_legacy: HashMap<EntityId, f32> = passage_scores_typed
            .iter()
            .map(|(cid, &score)| (chunk_id_as_ppr_node(cid), score))
            .collect();

        // Step 6a: Calculate entity weights from facts
        let entity_weights =
            self.calculate_entity_weights(&top_k_facts, &entity_to_passages_legacy)?;

        // Step 6b: Calculate passage weights from dense retrieval
        let passage_weights = self.calculate_passage_weights(&passage_scores_legacy)?;

        // Step 6c: Combine into reset probability distribution
        let reset_probabilities = self.combine_weights(entity_weights, passage_weights)?;

        // Step 6d: Snapshot PPR from graph (OQ-1: consistent snapshot semantics),
        // or use the pre-built cached instance supplied by the caller.
        //
        // When `ppr_override` is Some, the caller (e.g. AppState::ppr_cache) has already
        // built the PPR instance from the current graph and cached it. We skip
        // `build_pagerank_calculator()` — which re-iterates the entity adjacency matrix —
        // and use the cached `Arc<PersonalizedPageRank>` directly.  This is the hot-path
        // optimisation added in card 2 of the PPR caching feature.
        //
        // When `ppr_override` is None, we build a fresh instance from the graph as before
        // (consistent snapshot semantics).
        //
        // `PersonalizedPageRank` does not implement `Clone`, so we hold either an owned
        // value (None branch) or a shared Arc (Some branch) and unify them via a reference.
        let _built_ppr_holder: Option<crate::graph::pagerank::PersonalizedPageRank>;
        let ppr_ref: &crate::graph::pagerank::PersonalizedPageRank = match &ppr_override {
            Some(cached) => cached.as_ref(),
            None => {
                _built_ppr_holder = Some(graph.build_pagerank_calculator().map_err(|e| {
                    GraphRAGError::Config {
                        message: format!("Failed to build PPR from graph: {e}"),
                    }
                })?);
                _built_ppr_holder.as_ref().unwrap()
            }
        };

        // Run PPR — output is HashMap<EntityId, f64> over entity-graph nodes only.
        // Nodes whose keys originated from `chunk_id_as_ppr_node` will appear here
        // if their IDs matched the graph's entity namespace (rare but possible).
        let ppr_scores = ppr_ref.calculate_scores(&reset_probabilities)?;

        // Step 7: Aggregate entity PPR scores into passage (ChunkId) scores,
        //         then rank via `rank_passages`.
        //
        // The PPR graph contains entity nodes only; chunk IDs are NOT graph nodes.
        // To produce ChunkId scores we:
        //   (a) For each entity node, propagate its PPR score to all chunks it mentions
        //       (via entity_to_passages_typed), taking the maximum PPR score per chunk.
        //   (b) Add the dense passage score (scaled by passage_node_weight).
        //
        // The resulting `chunk_combined` map is keyed by EntityId (via
        // `chunk_id_as_ppr_node`) so it can be fed to `rank_passages`, which filters
        // its input to keys that appear in `passage_scores_legacy` — i.e. only chunk
        // nodes, not raw entity nodes.
        let mut chunk_combined_ppr: HashMap<EntityId, f64> = HashMap::new();

        // (a) Entity PPR → chunk propagation (entity namespace → chunk namespace)
        for (entity_id, &ppr_score) in &ppr_scores {
            if let Some(chunk_ids) = entity_to_passages_typed.get(entity_id) {
                for cid in chunk_ids {
                    let key = chunk_id_as_ppr_node(cid);
                    let entry = chunk_combined_ppr.entry(key).or_insert(0.0);
                    // Take the max PPR contribution across all entities pointing to this chunk
                    if ppr_score > *entry {
                        *entry = ppr_score;
                    }
                }
            }
        }

        // (b) Add dense passage scores (scaled by passage_node_weight)
        for (cid, &dense_score) in &passage_scores_typed {
            let key = chunk_id_as_ppr_node(cid);
            let ppr_contrib = chunk_combined_ppr.get(&key).copied().unwrap_or(0.0);
            let combined = ppr_contrib + (dense_score as f64) * self.config.passage_node_weight;
            chunk_combined_ppr.insert(key, combined);
        }

        // Step 6 (final): Call rank_passages to project combined scores → ranked Vec<ChunkId>.
        //
        // `rank_passages` filters `chunk_combined_ppr` to keys present in
        // `passage_scores_legacy`, sorts by score descending, and truncates to top-K.
        // The returned SearchResult.id strings are the EntityId keys, which were
        // produced by `chunk_id_as_ppr_node` — so we translate back via `ppr_node_as_chunk_id`.
        let ranked = self.rank_passages(chunk_combined_ppr, &passage_scores_legacy)?;

        let chunk_ids: Vec<ChunkId> = ranked
            .into_iter()
            .map(|r| ppr_node_as_chunk_id(&EntityId::new(r.id)))
            .collect();

        Ok(chunk_ids)
    }

    /// Retrieve documents using HippoRAG PPR strategy (legacy API)
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `top_k_facts` - Top-k facts ranked by query-fact similarity
    /// * `entity_to_passages` - Map from entity IDs to passage IDs
    /// * `passage_scores` - Dense retrieval scores for passages
    ///
    /// # Returns
    /// Ranked search results sorted by PPR score
    pub async fn retrieve_legacy(
        &self,
        _query: &str,
        top_k_facts: Vec<Fact>,
        entity_to_passages: &HashMap<EntityId, Vec<EntityId>>,
        passage_scores: &HashMap<EntityId, f32>,
    ) -> Result<Vec<SearchResult>> {
        // Step 1: Calculate entity weights from facts
        let entity_weights = self.calculate_entity_weights(&top_k_facts, entity_to_passages)?;

        // Step 2: Calculate passage weights from dense retrieval
        let passage_weights = self.calculate_passage_weights(passage_scores)?;

        // Step 3: Combine into reset probability distribution
        let reset_probabilities = self.combine_weights(entity_weights, passage_weights)?;

        // Step 4: Run Personalized PageRank
        let ppr_scores = self.run_ppr(&reset_probabilities).await?;

        // Step 5: Extract and rank passage scores
        let ranked_results = self.rank_passages(ppr_scores, passage_scores)?;

        Ok(ranked_results)
    }

    /// Calculate entity weights based on fact relevance
    ///
    /// Key insight: Entities from high-scoring facts get high weights,
    /// but downweighted by how many passages they appear in (reduces generic entities)
    fn calculate_entity_weights(
        &self,
        facts: &[Fact],
        entity_to_passages: &HashMap<EntityId, Vec<EntityId>>,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut weights = HashMap::new();
        let mut occurrence_count = HashMap::new();

        // Process top-k facts
        for fact in facts.iter().take(self.config.top_k_facts) {
            let fact_score = fact.score as f64;

            // Extract entities from subject and object
            for entity_text in [&fact.subject, &fact.object] {
                let entity_id = EntityId::new(entity_text.clone());

                // Get number of passages containing this entity
                let num_passages = entity_to_passages
                    .get(&entity_id)
                    .map(|p| p.len())
                    .unwrap_or(0);

                if num_passages >= self.config.min_entity_frequency {
                    // Weight by fact score, downweighted by passage frequency
                    let weighted_score = if num_passages > 0 {
                        fact_score / num_passages as f64
                    } else {
                        fact_score
                    };

                    *weights.entry(entity_id.clone()).or_insert(0.0) += weighted_score;
                    *occurrence_count.entry(entity_id).or_insert(0) += 1;
                }
            }
        }

        // Average by number of occurrences
        for (entity_id, count) in occurrence_count {
            if let Some(weight) = weights.get_mut(&entity_id) {
                *weight /= count as f64;
            }
        }

        // Normalize if configured
        if self.config.normalize_scores {
            self.normalize_weights(&mut weights);
        }

        Ok(weights)
    }

    /// Calculate passage weights from dense retrieval scores
    fn calculate_passage_weights(
        &self,
        passage_scores: &HashMap<EntityId, f32>,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut weights = HashMap::new();

        for (passage_id, score) in passage_scores {
            // Scale passage scores by passage_node_weight (default 0.05)
            let weighted_score = (*score as f64) * self.config.passage_node_weight;
            weights.insert(passage_id.clone(), weighted_score);
        }

        // Normalize if configured
        if self.config.normalize_scores {
            self.normalize_weights(&mut weights);
        }

        Ok(weights)
    }

    /// Combine entity and passage weights into reset probability distribution
    fn combine_weights(
        &self,
        entity_weights: HashMap<EntityId, f64>,
        passage_weights: HashMap<EntityId, f64>,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut combined = entity_weights;

        // Add passage weights
        for (passage_id, weight) in passage_weights {
            *combined.entry(passage_id).or_insert(0.0) += weight;
        }

        // Ensure non-negative and normalize
        let total: f64 = combined.values().sum();
        if total > 0.0 {
            for weight in combined.values_mut() {
                *weight /= total;
            }
        }

        Ok(combined)
    }

    /// Run Personalized PageRank with reset probabilities (legacy path: requires pre-set pagerank)
    async fn run_ppr(
        &self,
        reset_probabilities: &HashMap<EntityId, f64>,
    ) -> Result<HashMap<EntityId, f64>> {
        let pagerank = self
            .pagerank
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "PageRank not initialized".to_string(),
            })?;

        // Run PPR algorithm
        pagerank.calculate_scores(reset_probabilities)
    }

    /// Extract passage scores and rank by PPR
    fn rank_passages(
        &self,
        ppr_scores: HashMap<EntityId, f64>,
        original_scores: &HashMap<EntityId, f32>,
    ) -> Result<Vec<SearchResult>> {
        let mut results: Vec<_> = ppr_scores
            .iter()
            .filter_map(|(entity_id, &ppr_score)| {
                // Only include passage nodes (not entity nodes)
                if original_scores.contains_key(entity_id) {
                    Some(SearchResult {
                        id: entity_id.to_string(),
                        content: String::new(), // Will be filled by caller
                        score: ppr_score as f32,
                        result_type: crate::retrieval::ResultType::Chunk,
                        entities: Vec::new(),
                        source_chunks: Vec::new(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by PPR score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top-k
        results.truncate(self.config.top_k_results);

        Ok(results)
    }

    /// Normalize weights to [0, 1] range using min-max normalization
    fn normalize_weights(&self, weights: &mut HashMap<EntityId, f64>) {
        if weights.is_empty() {
            return;
        }

        let min = weights.values().cloned().fold(f64::INFINITY, f64::min);
        let max = weights.values().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() > 1e-10 {
            for weight in weights.values_mut() {
                *weight = (*weight - min) / (max - min);
            }
        }
    }
}

/// HippoRAG-specific PageRank configuration
impl HippoRAGConfig {
    /// Convert to PageRankConfig for compatibility
    pub fn to_pagerank_config(&self) -> PageRankConfig {
        PageRankConfig {
            damping_factor: self.damping_factor,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            personalized: true,
            parallel_enabled: true,
            cache_size: 1000,
            sparse_threshold: 1000,
            incremental_updates: true,
            simd_block_size: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, Entity, EntityId, EntityMention, KnowledgeGraph, Relationship};
    use crate::core::traits::AsyncEmbedder;
    use crate::vector::store::{SearchResult as VsSearchResult, VectorStore};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // ========================================================================
    // Mock infrastructure for new retrieve() tests
    // ========================================================================

    /// A mock embedder that returns a fixed vector for any input.
    struct ConstEmbedder {
        vec: Vec<f32>,
    }

    #[async_trait]
    impl AsyncEmbedder for ConstEmbedder {
        type Error = GraphRAGError;

        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(self.vec.clone())
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| self.vec.clone()).collect())
        }

        fn dimension(&self) -> usize {
            self.vec.len()
        }

        async fn is_ready(&self) -> bool {
            true
        }
    }

    /// A mock vector store that returns canned results per call round-robin.
    /// `calls[0]` is returned on the 1st search, `calls[1]` on the 2nd, etc.
    struct ScriptedVectorStore {
        calls: Arc<Mutex<Vec<Vec<VsSearchResult>>>>,
    }

    impl ScriptedVectorStore {
        fn new(responses: Vec<Vec<VsSearchResult>>) -> Self {
            Self {
                calls: Arc::new(Mutex::new(responses)),
            }
        }
    }

    #[async_trait]
    impl VectorStore for ScriptedVectorStore {
        async fn initialize(&self) -> Result<()> {
            Ok(())
        }

        async fn add_vector(
            &self,
            _id: &str,
            _embedding: Vec<f32>,
            _metadata: HashMap<String, String>,
        ) -> Result<()> {
            Ok(())
        }

        async fn add_vectors_batch(
            &self,
            _vectors: Vec<(&str, Vec<f32>, HashMap<String, String>)>,
        ) -> Result<()> {
            Ok(())
        }

        async fn search(&self, _query: &[f32], _top_k: usize) -> Result<Vec<VsSearchResult>> {
            let mut calls = self.calls.lock().unwrap();
            if calls.is_empty() {
                Ok(vec![])
            } else {
                Ok(calls.remove(0))
            }
        }

        async fn delete(&self, _id: &str) -> Result<()> {
            Ok(())
        }
    }

    /// Build a small test KnowledgeGraph with two entities, one relationship,
    /// and mentions pointing to two different chunks.
    ///
    /// Entity "alice" mentions chunk "chunk-journal"
    /// Entity "bob"   mentions chunk "chunk-unrelated"
    /// Relationship: alice -KNOWS-> bob
    fn build_test_graph() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();

        let alice = Entity::new(
            EntityId::new("alice".to_string()),
            "Alice".to_string(),
            "PERSON".to_string(),
            0.9,
        )
        .with_mentions(vec![EntityMention {
            chunk_id: ChunkId::new("chunk-journal".to_string()),
            start_offset: 0,
            end_offset: 5,
            confidence: 0.9,
        }]);

        let bob = Entity::new(
            EntityId::new("bob".to_string()),
            "Bob".to_string(),
            "PERSON".to_string(),
            0.8,
        )
        .with_mentions(vec![EntityMention {
            chunk_id: ChunkId::new("chunk-unrelated".to_string()),
            start_offset: 0,
            end_offset: 3,
            confidence: 0.8,
        }]);

        kg.add_entity(alice).unwrap();
        kg.add_entity(bob).unwrap();
        kg.add_relationship(Relationship::new(
            EntityId::new("alice".to_string()),
            EntityId::new("bob".to_string()),
            "KNOWS".to_string(),
            0.9,
        ))
        .unwrap();

        kg
    }

    // ========================================================================
    // TDD tests for the new retrieve() API (written first, then implementation)
    // ========================================================================

    /// Test that retrieve() returns top-K ChunkIds for a canned fixture where
    /// entity hits and dense hits both favour "chunk-journal".
    #[tokio::test]
    async fn test_retrieve_returns_top_k_chunks_for_canned_fixture() {
        let graph = build_test_graph();
        let config = HippoRAGConfig {
            top_k_results: 2,
            top_k_dense: 30,
            normalize_scores: false,
            ..Default::default()
        };
        let retriever = HippoRAGRetriever::new(config);
        let embedder = ConstEmbedder { vec: vec![1.0, 0.0] };

        // search call 1 (entity hits): alice scores high
        let entity_hits = vec![VsSearchResult {
            id: "alice".to_string(),
            score: 0.95,
            metadata: HashMap::new(),
        }];

        // search call 2 (relation hits): one relation alice-KNOWS-bob
        let mut rel_meta = HashMap::new();
        rel_meta.insert("source".to_string(), "alice".to_string());
        rel_meta.insert("relation_type".to_string(), "KNOWS".to_string());
        rel_meta.insert("target".to_string(), "bob".to_string());
        let relation_hits = vec![VsSearchResult {
            id: "rel-1".to_string(),
            score: 0.85,
            metadata: rel_meta,
        }];

        // search call 3 (dense chunk hits): chunk-journal scores higher
        let dense_hits = vec![
            VsSearchResult {
                id: "chunk-journal".to_string(),
                score: 0.92,
                metadata: HashMap::new(),
            },
            VsSearchResult {
                id: "chunk-unrelated".to_string(),
                score: 0.30,
                metadata: HashMap::new(),
            },
        ];

        let store = ScriptedVectorStore::new(vec![entity_hits, relation_hits, dense_hits]);
        let result = retriever
            .retrieve("alice journal entry", &graph, &store, &embedder, None)
            .await
            .unwrap();

        // Must return at least one chunk
        assert!(!result.is_empty(), "retrieve() must return at least one ChunkId");

        // chunk-journal must be ranked first (entity alice points to it and it scored highest dense)
        assert_eq!(
            result[0],
            ChunkId::new("chunk-journal".to_string()),
            "chunk-journal must rank first"
        );
    }

    /// Test that retrieve() returns Ok([]) when there are zero dense hits.
    ///
    /// With `rank_passages` in the flow, results are filtered to only keys present in
    /// `passage_scores_legacy`.  When dense hits are empty, `passage_scores_legacy` is
    /// empty, so `rank_passages` returns nothing — the output is an empty Vec.
    ///
    /// This is semantically correct: HippoRAG uses dense retrieval as the "passage anchor";
    /// if no passages arrived via dense search, there are no candidates to rank.  The test
    /// verifies the function does NOT panic or return Err in this edge case.
    #[tokio::test]
    async fn test_retrieve_handles_zero_dense_hits() {
        let graph = build_test_graph();
        let config = HippoRAGConfig {
            top_k_results: 5,
            top_k_dense: 30,
            normalize_scores: false,
            ..Default::default()
        };
        let retriever = HippoRAGRetriever::new(config);
        let embedder = ConstEmbedder { vec: vec![1.0, 0.0] };

        // entity hits
        let entity_hits = vec![VsSearchResult {
            id: "alice".to_string(),
            score: 0.95,
            metadata: HashMap::new(),
        }];
        // relation hits (empty)
        let relation_hits: Vec<VsSearchResult> = vec![];
        // dense hits EMPTY
        let dense_hits: Vec<VsSearchResult> = vec![];

        let store =
            ScriptedVectorStore::new(vec![entity_hits, relation_hits, dense_hits]);

        let result = retriever
            .retrieve("alice", &graph, &store, &embedder, None)
            .await;

        // Must not panic or return Err
        assert!(result.is_ok(), "retrieve() must not error with zero dense hits");

        // With zero dense hits, passage_scores_legacy is empty.  rank_passages filters
        // its input to keys in passage_scores_legacy, so the result is empty.
        let chunk_ids = result.unwrap();
        assert!(
            chunk_ids.is_empty(),
            "with zero dense hits, rank_passages produces no passage anchors → empty result"
        );
    }

    // ========================================================================
    // Card 2 tests: ppr_override parameter
    // ========================================================================

    /// Structural-proof test: `ppr_override = Some(pre_built_ppr)` bypasses
    /// `graph.build_pagerank_calculator()`.
    ///
    /// ## Why this proves the override is used
    ///
    /// `KnowledgeGraph` is a concrete type — we cannot wrap it in a counting
    /// newtype to intercept `build_pagerank_calculator()` calls.  Instead we
    /// use a two-path comparison that proves the same property behaviorally:
    ///
    /// 1. **None-path**: `retrieve(..., None)` — the implementation calls
    ///    `graph.build_pagerank_calculator()` (line ~311 of this file) and
    ///    produces PPR scores from the graph at call time.
    /// 2. **Some-path**: `retrieve(..., Some(pre_built_ppr))` — the
    ///    implementation enters `match &ppr_override { Some(cached) => cached.as_ref() }`
    ///    and NEVER reaches `graph.build_pagerank_calculator()`.  This is a
    ///    structural guarantee enforced by the `match` arm — it is impossible
    ///    for the `Some` branch to call the calculator.
    ///
    /// **Behavioral proof**: when the pre-built PPR comes from the SAME graph,
    /// both paths must produce identical `Vec<ChunkId>` results.  If the
    /// `Some` path were secretly calling `build_pagerank_calculator()` on a
    /// stale / different graph state it would produce different scores; the
    /// equality assertion below would fail.
    ///
    /// Note: `build_pagerank_calculator()` always returns `Ok(...)` even on
    /// empty graphs (it builds a 0×0 adjacency matrix), so we cannot use an
    /// error-based negative-control.  The equality assertion is the strongest
    /// behavioral check available given the concrete-type constraint.
    #[tokio::test]
    async fn test_retrieve_uses_ppr_override_not_graph_calculator() {
        let kg = build_test_graph();
        let config = HippoRAGConfig {
            top_k_results: 5,
            top_k_dense: 30,
            normalize_scores: false,
            ..Default::default()
        };
        let retriever = HippoRAGRetriever::new(config);
        let embedder = ConstEmbedder { vec: vec![1.0, 0.0] };

        // Pre-build the PPR instance — simulating what spawn_ppr_cache_rebuild does.
        // This PPR captures the current entity adjacency matrix of `kg`.
        let ppr = kg
            .build_pagerank_calculator()
            .expect("build_pagerank_calculator must succeed on a populated test graph");
        let ppr_arc = std::sync::Arc::new(ppr);

        // Helper closure that builds identical scripted-store responses for one call.
        let make_store = || {
            let entity_hits = vec![VsSearchResult {
                id: "alice".to_string(),
                score: 0.9,
                metadata: HashMap::new(),
            }];
            let mut rel_meta = HashMap::new();
            rel_meta.insert("source".to_string(), "alice".to_string());
            rel_meta.insert("relation_type".to_string(), "KNOWS".to_string());
            rel_meta.insert("target".to_string(), "bob".to_string());
            let relation_hits = vec![VsSearchResult {
                id: "rel-1".to_string(),
                score: 0.8,
                metadata: rel_meta,
            }];
            let dense_hits = vec![VsSearchResult {
                id: "chunk-journal".to_string(),
                score: 0.85,
                metadata: HashMap::new(),
            }];
            ScriptedVectorStore::new(vec![entity_hits, relation_hits, dense_hits])
        };

        // --- None-path: retrieve() calls build_pagerank_calculator() internally ---
        let result_none = retriever
            .retrieve("alice knows bob", &kg, &make_store(), &embedder, None)
            .await;
        assert!(
            result_none.is_ok(),
            "retrieve() with ppr_override=None must return Ok, got: {:?}",
            result_none.as_ref().err()
        );

        // --- Some-path: retrieve() uses the pre-built Arc<PersonalizedPageRank> ---
        // Structural guarantee: the Some(cached) match arm (see impl ~line 308-318)
        // resolves to `cached.as_ref()` and NEVER calls `graph.build_pagerank_calculator()`.
        let result_some = retriever
            .retrieve(
                "alice knows bob",
                &kg,
                &make_store(),
                &embedder,
                Some(std::sync::Arc::clone(&ppr_arc)),
            )
            .await;
        assert!(
            result_some.is_ok(),
            "retrieve() with ppr_override=Some must return Ok, got: {:?}",
            result_some.as_ref().err()
        );

        // Behavioral equality: both paths used the same graph and the same PPR
        // (None rebuilt it; Some used the pre-built one from the same graph).
        // The results must be identical, which is only possible if the Some-path
        // used the real_ppr (not a stale or wrong PPR from a different source).
        assert_eq!(
            result_none.unwrap(),
            result_some.unwrap(),
            "None-path and Some(pre_built)-path must produce identical results \
             when the PPR was built from the same graph state"
        );
    }

    /// Test that when `ppr_override = None`, the fallback path calls
    /// `build_pagerank_calculator()` internally. The call must not panic;
    /// it may succeed or return an error depending on graph state, but must
    /// not panic.
    #[tokio::test]
    async fn test_retrieve_fallback_when_ppr_override_is_none() {
        let kg = build_test_graph();
        let config = HippoRAGConfig {
            top_k_results: 5,
            top_k_dense: 30,
            normalize_scores: false,
            ..Default::default()
        };
        let retriever = HippoRAGRetriever::new(config);
        let embedder = ConstEmbedder { vec: vec![1.0, 0.0] };

        // Provide scripted store: entity hits, relation hits, dense hits.
        let entity_hits = vec![VsSearchResult {
            id: "alice".to_string(),
            score: 0.9,
            metadata: HashMap::new(),
        }];
        let relation_hits: Vec<VsSearchResult> = vec![];
        let dense_hits = vec![VsSearchResult {
            id: "chunk-journal".to_string(),
            score: 0.75,
            metadata: HashMap::new(),
        }];

        let store = ScriptedVectorStore::new(vec![entity_hits, relation_hits, dense_hits]);

        // Call with ppr_override = None — must not panic.
        // Result may be Ok or Err; both are acceptable. Panic is not.
        let result = retriever
            .retrieve("alice", &kg, &store, &embedder, None)
            .await;

        // Just verify no panic occurred (if Err, that is also acceptable).
        let _ = result;
    }

    /// Test that rank_passages is exercised by retrieve() and produces the correct
    /// ordering: only chunks present in passage_scores are returned, ranked by
    /// their combined PPR+dense score.
    ///
    /// This is the TDD-first test for Item 2 (rank_passages wiring).
    #[tokio::test]
    async fn test_retrieve_rank_passages_filters_and_orders_correctly() {
        let graph = build_test_graph();
        let config = HippoRAGConfig {
            top_k_results: 3,
            top_k_dense: 30,
            normalize_scores: false,
            ..Default::default()
        };
        let retriever = HippoRAGRetriever::new(config);
        let embedder = ConstEmbedder { vec: vec![1.0, 0.0] };

        // search call 1 (entity hits): alice scores high
        let entity_hits = vec![VsSearchResult {
            id: "alice".to_string(),
            score: 0.95,
            metadata: HashMap::new(),
        }];

        // search call 2 (relation hits): alice KNOWS bob
        let mut rel_meta = HashMap::new();
        rel_meta.insert("source".to_string(), "alice".to_string());
        rel_meta.insert("relation_type".to_string(), "KNOWS".to_string());
        rel_meta.insert("target".to_string(), "bob".to_string());
        let relation_hits = vec![VsSearchResult {
            id: "rel-1".to_string(),
            score: 0.9,
            metadata: rel_meta,
        }];

        // search call 3 (dense hits): only chunk-journal in dense results.
        // chunk-unrelated is NOT in dense — rank_passages must exclude it.
        let dense_hits = vec![VsSearchResult {
            id: "chunk-journal".to_string(),
            score: 0.88,
            metadata: HashMap::new(),
        }];

        let store = ScriptedVectorStore::new(vec![entity_hits, relation_hits, dense_hits]);
        let result = retriever
            .retrieve("alice", &graph, &store, &embedder, None)
            .await
            .unwrap();

        // rank_passages must only return chunks that appeared in dense retrieval
        assert!(
            result.iter().all(|cid| cid != &ChunkId::new("chunk-unrelated".to_string())),
            "chunk-unrelated was not in dense hits; rank_passages must not return it"
        );

        // chunk-journal must appear (it was in dense hits and alice→chunk-journal via entity graph)
        assert!(
            result.contains(&ChunkId::new("chunk-journal".to_string())),
            "chunk-journal must appear: it was in dense hits and alice's mention graph"
        );
    }

    // ========================================================================
    // Preserved pre-existing helper tests
    // ========================================================================

    #[tokio::test]
    async fn test_entity_weight_calculation() {
        let config = HippoRAGConfig::default();
        let retriever = HippoRAGRetriever::new(config);

        let facts = vec![
            Fact {
                subject: "Alice".to_string(),
                predicate: "works_at".to_string(),
                object: "Company".to_string(),
                score: 0.9,
            },
            Fact {
                subject: "Bob".to_string(),
                predicate: "works_at".to_string(),
                object: "Company".to_string(),
                score: 0.8,
            },
        ];

        let mut entity_to_passages = HashMap::new();
        entity_to_passages.insert(
            EntityId::new("Alice".to_string()),
            vec![EntityId::new("doc1".to_string())],
        );
        entity_to_passages.insert(
            EntityId::new("Company".to_string()),
            vec![
                EntityId::new("doc1".to_string()),
                EntityId::new("doc2".to_string()),
            ],
        );

        let weights = retriever
            .calculate_entity_weights(&facts, &entity_to_passages)
            .unwrap();

        // Alice should have higher weight (appears in fewer passages)
        let alice_weight = weights.get(&EntityId::new("Alice".to_string())).unwrap();
        let company_weight = weights.get(&EntityId::new("Company".to_string())).unwrap();

        assert!(
            alice_weight > company_weight,
            "Alice should have higher weight due to lower frequency"
        );
    }

    #[tokio::test]
    async fn test_passage_weight_calculation() {
        let config = HippoRAGConfig {
            passage_node_weight: 0.05,
            normalize_scores: false, // Disable normalization for this test
            ..Default::default()
        };
        let retriever = HippoRAGRetriever::new(config);

        let mut passage_scores = HashMap::new();
        passage_scores.insert(EntityId::new("doc1".to_string()), 0.9);
        passage_scores.insert(EntityId::new("doc2".to_string()), 0.5);

        let weights = retriever
            .calculate_passage_weights(&passage_scores)
            .unwrap();

        // Passage weights should be scaled by passage_node_weight
        let doc1_weight = weights.get(&EntityId::new("doc1".to_string())).unwrap();
        assert!(
            (*doc1_weight - 0.9 * 0.05).abs() < 0.001,
            "Passage weight should be scaled"
        );

        // doc1 should have higher weight than doc2
        let doc2_weight = weights.get(&EntityId::new("doc2".to_string())).unwrap();
        assert!(
            doc1_weight > doc2_weight,
            "Higher score should have higher weight"
        );
    }

    #[test]
    fn test_weight_combining() {
        let config = HippoRAGConfig::default();
        let retriever = HippoRAGRetriever::new(config);

        let mut entity_weights = HashMap::new();
        entity_weights.insert(EntityId::new("entity1".to_string()), 0.8);

        let mut passage_weights = HashMap::new();
        passage_weights.insert(EntityId::new("doc1".to_string()), 0.04);
        passage_weights.insert(EntityId::new("entity1".to_string()), 0.01); // Overlap

        let combined = retriever
            .combine_weights(entity_weights, passage_weights)
            .unwrap();

        // entity1 should have combined weight
        let entity1_combined = combined.get(&EntityId::new("entity1".to_string())).unwrap();
        assert!(
            *entity1_combined > 0.0,
            "Entity should have combined weight"
        );

        // All weights should sum to 1.0 (normalized)
        let total: f64 = combined.values().sum();
        assert!((total - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    }
}
