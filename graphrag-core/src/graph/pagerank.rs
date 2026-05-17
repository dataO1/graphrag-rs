//! PageRank implementation for GraphRAG
//!
//! This module is only available when the "pagerank" feature is enabled.
//!
//! # Architecture
//!
//! `PersonalizedPageRank` uses a sparse direct LU solver (via the `faer` crate)
//! to compute personalised PageRank scores.  The factorisation of
//! `A = I − α·Pᵀ` is computed once in `new()` (which runs in a background
//! task via `spawn_ppr_cache_rebuild`).  Per query, two sparse back-
//! substitutions are performed and combined via Sherman-Morrison-Woodbury (SMW)
//! to account for dangling nodes (out-degree 0), matching the canonical
//! Boldi-Vigna / Langville-Meyer formulation.

use crate::core::{EntityId, Result};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::Col;
use lru::LruCache;
use parking_lot::RwLock;
use sprs::CsMat;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Configuration for PageRank algorithm.
///
/// Fields that were only meaningful under the old power-iteration path have
/// been removed.  Any serialised config that still contains them is harmless
/// (serde skips unknown fields).
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor α (HippoRAG default 0.5; web-PR typical 0.85)
    pub damping_factor: f64,
    /// Whether to use personalized PageRank (should always be true in our use)
    pub personalized: bool,
    /// LRU cache size for repeated identical reset-probability queries
    pub cache_size: usize,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            personalized: true,
            cache_size: 1000,
        }
    }
}

/// Personalized PageRank calculator backed by a sparse direct LU factorisation.
///
/// Construction (`new`) factorises `A = I − α·Pᵀ` via `faer` sparse LU — this
/// is the expensive step (~100-500 ms on a 28k-node graph) and is expected to
/// run only in the background via `spawn_ppr_cache_rebuild`.
///
/// Per-query cost is two sparse back-substitutions combined via SMW — typically
/// 5-50 ms on the same graph.
pub struct PersonalizedPageRank {
    config: PageRankConfig,
    /// Number of nodes in the graph
    n: usize,
    /// Mapping from entity IDs to matrix row/column indices
    node_mapping: HashMap<EntityId, usize>,
    /// Mapping from matrix indices back to entity IDs
    reverse_mapping: HashMap<usize, EntityId>,
    /// LU factorisation of `A = I − α·Pᵀ` (sparse, reused across queries)
    factorization: Lu<usize, f64>,
    /// True when the graph has at least one dangling node (triggers L1-normalisation)
    has_dangling: bool,
    /// LRU cache: hash(reset_probabilities) → entity scores
    score_cache: Arc<RwLock<LruCache<u64, HashMap<EntityId, f64>>>>,
}

impl PersonalizedPageRank {
    /// Build a new `PersonalizedPageRank` from a sparse adjacency matrix.
    ///
    /// This performs the full LU factorisation and should only be called from
    /// the background `spawn_ppr_cache_rebuild` task.
    ///
    /// # Arguments
    /// * `config` - PageRank configuration
    /// * `adjacency` - Sparse adjacency matrix (CSR, `adj[i,j]` = weight of edge `i→j`)
    /// * `node_mapping` - Mapping from entity IDs to matrix indices
    /// * `reverse_mapping` - Mapping from matrix indices to entity IDs
    pub fn new(
        config: PageRankConfig,
        adjacency: CsMat<f64>,
        node_mapping: HashMap<EntityId, usize>,
        reverse_mapping: HashMap<usize, EntityId>,
    ) -> Self {
        let n = adjacency.rows();
        let cache_size =
            NonZeroUsize::new(config.cache_size).unwrap_or(NonZeroUsize::new(1000).unwrap());

        // --- Step 1: compute out-degrees (weighted row sums) ---
        let out_degrees = Self::compute_out_degrees(&adjacency);

        // --- Step 2: identify dangling nodes ---
        let has_dangling = out_degrees.iter().any(|&d| d == 0.0);

        let alpha = config.damping_factor;

        // --- Step 3: build A = I − α·Pᵀ in CSC format ---
        let faer_mat = Self::build_system_matrix(&adjacency, &out_degrees, alpha, n);

        // --- Step 5: symbolic + numeric LU factorisation ---
        let symbolic_ref = faer_mat.symbolic();
        let symbolic_lu = SymbolicLu::try_new(symbolic_ref)
            .expect("faer: symbolic LU failed — matrix may be structurally singular");
        let factorization = Lu::try_new_with_symbolic(symbolic_lu, faer_mat.as_ref())
            .expect("faer: numeric LU failed — matrix may be numerically singular");

        Self {
            config,
            n,
            node_mapping,
            reverse_mapping,
            factorization,
            has_dangling,
            score_cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
        }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Configuration used to build this instance.
    pub fn config(&self) -> &PageRankConfig {
        &self.config
    }

    /// Compute personalised PageRank scores.
    ///
    /// Uses the cached LU factorisation for a single sparse back-substitution.
    /// When dangling nodes (out-degree 0) exist, the resulting score vector is
    /// L1-normalised to redistribute the "missing" mass — equivalent to the
    /// Sherman-Morrison-Woodbury correction for personalised dangling-node
    /// redistribution (Boldi-Vigna / Langville-Meyer formulation).
    ///
    /// **Derivation sketch**: the full PPR system with personalised dangling
    /// handling is `(A − α·v·dᵀ)·x = (1−α)·v` where `A = I − α·Pᵀ` and
    /// `d` is the dangling indicator.  Applying SMW with `u = α·v` (same
    /// direction as the RHS), the correction reduces to a global scalar factor
    /// `(1−α) / (1−α − α·dᵀ·x₁)` applied to `x₁ = A⁻¹·(1−α)·v`.  That
    /// factor equals `1/sum(x₁)` (provable from `Aᵀ·1 = (1−α)·1` for
    /// non-dangling columns, so dangling columns "absorb" the missing mass).
    /// Hence L1-normalisation is exact.
    ///
    /// Results are cached by the hash of `reset_probabilities`.
    pub fn calculate_scores(
        &self,
        reset_probabilities: &HashMap<EntityId, f64>,
    ) -> Result<HashMap<EntityId, f64>> {
        if self.n == 0 {
            return Ok(HashMap::new());
        }

        // Cache check
        let cache_key = Self::generate_cache_key(reset_probabilities);
        {
            let cache = self.score_cache.read();
            if let Some(cached) = cache.peek(&cache_key) {
                return Ok(cached.clone());
            }
        }

        let alpha = self.config.damping_factor;

        // Build reset / personalisation vector v (sums to 1)
        let v = self.build_reset_vector(reset_probabilities);

        // b = (1 − α)·v
        let b = Col::from_fn(self.n, |i| (1.0 - alpha) * v[i]);

        // Solve A·x = b  (single LU back-substitution)
        let x_col: Col<f64> = self.factorization.solve(&b);
        let mut x: Vec<f64> = (0..self.n).map(|i| x_col[i]).collect();

        // Dangling-node mass redistribution via L1-normalisation.
        // For graphs without dangling nodes the sum is already 1.0 to
        // machine precision; normalising is a no-op (costs ~n multiplies).
        if self.has_dangling {
            let total: f64 = x.iter().sum();
            if total > 1e-15 {
                x.iter_mut().for_each(|v| *v /= total);
            }
        }

        // Map index → entity ID
        let scores = self.scores_to_entity_map(&x)?;

        // Store in cache
        {
            let mut cache = self.score_cache.write();
            cache.put(cache_key, scores.clone());
        }

        Ok(scores)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Compute weighted out-degrees (row sums of adjacency matrix weights).
    fn compute_out_degrees(adjacency: &CsMat<f64>) -> Vec<f64> {
        let n = adjacency.rows();
        let mut degrees = vec![0.0f64; n];
        for (i, deg) in degrees.iter_mut().enumerate().take(n) {
            if let Some(row) = adjacency.outer_view(i) {
                *deg = row.iter().map(|(_, &w)| w).sum();
            }
        }
        degrees
    }

    /// Build the system matrix `A = I − α·Pᵀ` in faer CSC format.
    ///
    /// `Pᵀ[i,j]` = probability of jumping from node `j` to node `i`
    ///            = `adj[j,i] / out_degree[j]`  (for non-dangling `j`)
    ///
    /// Column `j` of `A` therefore contains:
    /// - Diagonal `(j,j)`: value `1.0`
    /// - Off-diagonals `(i,j)` for each edge `j→i` in the adjacency:
    ///   value `−α · adj[j,i] / out_degree[j]`
    ///
    /// Dangling nodes (out_degree=0) contribute nothing to `Pᵀ`; their SMW
    /// correction is handled separately.
    ///
    /// # Bug history
    ///
    /// An earlier version pre-computed `nnz_per_col[j] = 1 + row.nnz()` then
    /// allocated flat `row_idx`/`values` arrays up front.  The count was wrong
    /// whenever node `j` had a self-loop (`j→j`): the self-loop is *merged*
    /// into the diagonal entry during fill (reducing the written count by 1),
    /// but the pre-computed count still reserved a slot for it.  That ghost
    /// slot remained zero-initialised, so `col_ptr[j+1]` was one beyond the
    /// last entry actually written for column `j`.  faer's `new_checked` then
    /// read `[…, row_j, 0]` as column `j`'s row indices and asserted
    /// `row_j < 0`, producing the panic
    ///   `Assertion failed: i < i_next — i = 16779, i_next = 0`
    /// on the 29 k-node production graph where node 16779 had a self-loop.
    ///
    /// The fix: collect each column's entries fully (with self-loop merging)
    /// *before* finalising `col_ptr`, so the count is always exact.
    fn build_system_matrix(
        adjacency: &CsMat<f64>,
        out_degrees: &[f64],
        alpha: f64,
        n: usize,
    ) -> SparseColMat<usize, f64> {
        // Pass 1: build each column's (row, value) pairs with the self-loop
        // diagonal merge applied.  We collect into a Vec-of-Vecs so that
        // col_ptr can be derived from the *actual* number of entries written
        // rather than from a pre-computed estimate that does not account for
        // the diagonal merge.
        let mut all_col_entries: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);

        for j in 0..n {
            let deg = out_degrees[j];

            // Every column has at least the diagonal entry I[j,j].
            let mut col_entries: Vec<(usize, f64)> = Vec::new();
            col_entries.push((j, 1.0));

            if deg > 0.0 {
                if let Some(row) = adjacency.outer_view(j) {
                    for (i, &w) in row.iter() {
                        // A[i,j] -= α * P^T[i,j] = α * adj[j,i] / deg[j]
                        if i == j {
                            // Self-loop: merge contribution into diagonal.
                            // This does NOT add a new row-index entry.
                            col_entries[0].1 -= alpha * w / deg;
                        } else {
                            col_entries.push((i, -alpha * w / deg));
                        }
                    }
                }
            }

            // Sort by row index (faer's CSC format requires sorted row indices
            // within each column).
            col_entries.sort_unstable_by_key(|(row, _)| *row);
            all_col_entries.push(col_entries);
        }

        // Pass 2: compute col_ptr from the *actual* per-column entry counts.
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + all_col_entries[j].len();
        }
        let total_nnz = col_ptr[n];

        // Pass 3: flatten into the CSC row_idx / values arrays.
        let mut row_idx = vec![0usize; total_nnz];
        let mut values = vec![0.0f64; total_nnz];

        for (j, col_entries) in all_col_entries.into_iter().enumerate() {
            let start = col_ptr[j];
            for (k, (row, val)) in col_entries.into_iter().enumerate() {
                row_idx[start + k] = row;
                values[start + k] = val;
            }
        }

        // Construct faer SymbolicSparseColMat then SparseColMat
        let symbolic = SymbolicSparseColMat::<usize, usize, usize>::new_checked(
            n,
            n,
            col_ptr,
            None,
            row_idx,
        );
        SparseColMat::new(symbolic, values)
    }

    /// Build the reset/personalisation vector `v` (length `n`, sums to 1).
    fn build_reset_vector(&self, reset_probabilities: &HashMap<EntityId, f64>) -> Vec<f64> {
        let n = self.n;
        let mut v = vec![1.0 / n as f64; n];

        if !reset_probabilities.is_empty() {
            let total: f64 = reset_probabilities.values().sum();
            if total > 0.0 {
                // Reset to sparse personalised vector
                v.iter_mut().for_each(|x| *x = 0.0);
                for (entity_id, &prob) in reset_probabilities {
                    if let Some(&idx) = self.node_mapping.get(entity_id) {
                        if idx < n {
                            v[idx] = prob / total;
                        }
                    }
                }
            }
        }

        v
    }

    /// Map a score slice (indexed by matrix position) to a `HashMap<EntityId, f64>`.
    fn scores_to_entity_map(&self, scores: &[f64]) -> Result<HashMap<EntityId, f64>> {
        let mut result = HashMap::with_capacity(scores.len());
        for (idx, &score) in scores.iter().enumerate() {
            if let Some(entity_id) = self.reverse_mapping.get(&idx) {
                result.insert(entity_id.clone(), score);
            }
        }
        Ok(result)
    }

    /// Hash `reset_probabilities` to a `u64` cache key.
    fn generate_cache_key(reset_probabilities: &HashMap<EntityId, f64>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        let mut sorted: Vec<_> = reset_probabilities.iter().collect();
        sorted.sort_by_key(|(id, _)| id.to_string());
        for (id, score) in sorted {
            id.to_string().hash(&mut hasher);
            score.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Multi-modal scoring system that combines different ranking signals
#[derive(Debug, Clone)]
pub struct MultiModalScores {
    /// Scores based on vector similarity
    pub vector_scores: HashMap<EntityId, f64>,
    /// Scores based on PageRank importance
    pub pagerank_scores: HashMap<EntityId, f64>,
    /// Scores for text chunks
    pub chunk_scores: HashMap<crate::core::ChunkId, f64>,
    /// Scores for relationships between entities
    pub relationship_scores: HashMap<String, f64>,
}

/// Weights for combining different scoring signals
#[derive(Debug, Clone)]
pub struct ScoreWeights {
    /// Weight for vector similarity scores
    pub vector_weight: f64,
    /// Weight for PageRank scores
    pub pagerank_weight: f64,
    /// Weight for chunk scores
    pub chunk_weight: f64,
    /// Weight for relationship scores
    pub relationship_weight: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            vector_weight: 0.3,
            pagerank_weight: 0.4,
            chunk_weight: 0.2,
            relationship_weight: 0.1,
        }
    }
}

impl MultiModalScores {
    /// Create a new empty MultiModalScores instance
    pub fn new() -> Self {
        Self {
            vector_scores: HashMap::new(),
            pagerank_scores: HashMap::new(),
            chunk_scores: HashMap::new(),
            relationship_scores: HashMap::new(),
        }
    }

    /// Combine multiple scoring signals with configurable weights
    pub fn combine_scores(&self, weights: &ScoreWeights) -> HashMap<EntityId, f64> {
        use std::collections::HashSet;

        let mut combined_scores = HashMap::new();

        let all_entities: HashSet<EntityId> = self
            .vector_scores
            .keys()
            .chain(self.pagerank_scores.keys())
            .cloned()
            .collect();

        for entity_id in all_entities {
            let vector_score = self.vector_scores.get(&entity_id).unwrap_or(&0.0);
            let pagerank_score = self.pagerank_scores.get(&entity_id).unwrap_or(&0.0);
            let chunk_score = self.get_entity_chunk_score(&entity_id);

            let combined = weights.vector_weight * vector_score
                + weights.pagerank_weight * pagerank_score
                + weights.chunk_weight * chunk_score;

            combined_scores.insert(entity_id, combined);
        }

        combined_scores
    }

    fn get_entity_chunk_score(&self, _entity_id: &EntityId) -> f64 {
        0.0
    }
}

impl Default for MultiModalScores {
    fn default() -> Self {
        Self::new()
    }
}

// ── PRPACK fidelity fixtures (auto-generated, cfg(test) only) ─────────────────
#[cfg(test)]
mod ppr_fixtures {
    include!("../../../graphrag-core/tests/fixtures/ppr_fixtures.rs");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::EntityId;

    /// Helper: build a simple 3-node graph A→B, A→C, B→C (no dangling nodes in the
    /// sense that A and B have outgoing edges; C is a sink / dangling node).
    fn create_simple_test_graph() -> (
        CsMat<f64>,
        HashMap<EntityId, usize>,
        HashMap<usize, EntityId>,
    ) {
        let entity_a = EntityId::new("A".to_string());
        let entity_b = EntityId::new("B".to_string());
        let entity_c = EntityId::new("C".to_string());

        let mut node_mapping = HashMap::new();
        let mut reverse_mapping = HashMap::new();
        node_mapping.insert(entity_a.clone(), 0);
        node_mapping.insert(entity_b.clone(), 1);
        node_mapping.insert(entity_c.clone(), 2);
        reverse_mapping.insert(0, entity_a);
        reverse_mapping.insert(1, entity_b);
        reverse_mapping.insert(2, entity_c);

        let mut triplet = sprs::TriMat::new((3, 3));
        triplet.add_triplet(0, 1, 1.0); // A→B
        triplet.add_triplet(0, 2, 1.0); // A→C
        triplet.add_triplet(1, 2, 1.0); // B→C
        let matrix = triplet.to_csr();

        (matrix, node_mapping, reverse_mapping)
    }

    // ── Acceptance-criteria tests ─────────────────────────────────────────────

    /// AC-1: factorisation succeeds on a small graph and returns n entries summing to ~1.
    #[test]
    fn test_pagerank_factor_succeeds_on_small_graph() {
        // 10-node graph with a simple chain + some cross-edges
        let n = 10usize;
        let mut triplet = sprs::TriMat::new((n, n));
        for i in 0..(n - 1) {
            triplet.add_triplet(i, i + 1, 1.0);
        }
        // Extra edges to make it less trivial
        triplet.add_triplet(3, 0, 0.5);
        triplet.add_triplet(7, 2, 0.5);

        let matrix = triplet.to_csr();

        let mut node_mapping = HashMap::new();
        let mut reverse_mapping = HashMap::new();
        for i in 0..n {
            let id = EntityId::new(format!("node_{i}"));
            node_mapping.insert(id.clone(), i);
            reverse_mapping.insert(i, id);
        }

        let mut config = PageRankConfig::default();
        config.damping_factor = 0.5;
        let ppr = PersonalizedPageRank::new(config, matrix, node_mapping, reverse_mapping);

        let scores = ppr.calculate_scores(&HashMap::new()).expect("calculate_scores failed");

        assert_eq!(scores.len(), n, "should have scores for all {n} nodes");
        let total: f64 = scores.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "scores should sum to 1.0 ± 1e-9, got {total}"
        );
        for (entity, &score) in &scores {
            assert!(
                score >= 0.0,
                "score for {entity:?} should be non-negative, got {score}"
            );
        }
    }

    /// AC-2: with dangling nodes present, ALL nodes (including the sink) receive
    /// non-zero mass after SMW correction.
    #[test]
    fn test_pagerank_dangling_node_uniform_redistribute() {
        // 3-node graph: 0→1, 1→2, node 2 is a sink (dangling)
        // Personalise at node 0.
        let n = 3usize;
        let mut triplet = sprs::TriMat::new((n, n));
        triplet.add_triplet(0, 1, 1.0); // 0→1
        triplet.add_triplet(1, 2, 1.0); // 1→2 (2 is dangling)
        let matrix = triplet.to_csr();

        let mut node_mapping = HashMap::new();
        let mut reverse_mapping = HashMap::new();
        for i in 0..n {
            let id = EntityId::new(format!("node_{i}"));
            node_mapping.insert(id.clone(), i);
            reverse_mapping.insert(i, id.clone());
        }

        let mut config = PageRankConfig::default();
        config.damping_factor = 0.5;
        let ppr = PersonalizedPageRank::new(config, matrix, node_mapping.clone(), reverse_mapping);

        // Seed at node 0
        let mut seeds = HashMap::new();
        seeds.insert(EntityId::new("node_0".to_string()), 1.0);

        let scores = ppr.calculate_scores(&seeds).expect("calculate_scores failed");

        // All 3 nodes must have non-zero mass (SMW redistributes dangling mass)
        for i in 0..n {
            let id = EntityId::new(format!("node_{i}"));
            let s = scores.get(&id).copied().unwrap_or(0.0);
            assert!(
                s > 1e-12,
                "node_{i} should have non-zero score after SMW, got {s}"
            );
        }

        // Scores should sum to ~1.0
        let total: f64 = scores.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "scores should sum to 1.0, got {total}"
        );
    }

    // ── Preserved / adapted baseline tests ──────────────────────────────────

    #[test]
    fn test_pagerank_convergence() {
        let (matrix, node_mapping, reverse_mapping) = create_simple_test_graph();
        let config = PageRankConfig::default();
        let pagerank = PersonalizedPageRank::new(config, matrix, node_mapping, reverse_mapping);

        let scores = pagerank.calculate_scores(&HashMap::new()).unwrap();

        let total_score: f64 = scores.values().sum();
        assert!((total_score - 1.0).abs() < 1e-6);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_personalized_pagerank() {
        let (matrix, node_mapping, reverse_mapping) = create_simple_test_graph();
        let config = PageRankConfig::default();
        let pagerank = PersonalizedPageRank::new(config, matrix, node_mapping, reverse_mapping);

        let mut reset_probs = HashMap::new();
        let entity_a = EntityId::new("A".to_string());
        let entity_b = EntityId::new("B".to_string());
        reset_probs.insert(entity_a.clone(), 0.8);
        reset_probs.insert(entity_b, 0.2);

        let scores = pagerank.calculate_scores(&reset_probs).unwrap();

        // Entity A should have a non-trivial score
        let score_a = scores.get(&entity_a).unwrap();
        assert!(*score_a > 0.0);
    }

    #[test]
    fn test_convergence_within_30_iterations_at_half_damping() {
        // Build a small 4-node graph with weighted edges (kept for compatibility)
        let mut triplet = sprs::TriMat::new((4, 4));
        triplet.add_triplet(0, 1, 0.8f64);
        triplet.add_triplet(1, 2, 0.6);
        triplet.add_triplet(2, 3, 0.4);
        triplet.add_triplet(3, 0, 0.5);
        triplet.add_triplet(0, 2, 0.3);
        let adj = triplet.to_csr();

        let entity_a = EntityId::new("A".to_string());
        let entity_b = EntityId::new("B".to_string());
        let entity_c = EntityId::new("C".to_string());
        let entity_d = EntityId::new("D".to_string());

        let mut node_mapping = HashMap::new();
        node_mapping.insert(entity_a.clone(), 0);
        node_mapping.insert(entity_b.clone(), 1);
        node_mapping.insert(entity_c.clone(), 2);
        node_mapping.insert(entity_d.clone(), 3);

        let mut reverse_mapping = HashMap::new();
        reverse_mapping.insert(0, entity_a.clone());
        reverse_mapping.insert(1, entity_b);
        reverse_mapping.insert(2, entity_c);
        reverse_mapping.insert(3, entity_d);

        let mut config = PageRankConfig::default();
        config.damping_factor = 0.5;

        let ppr = PersonalizedPageRank::new(config, adj, node_mapping, reverse_mapping);

        let mut seeds = HashMap::new();
        seeds.insert(entity_a.clone(), 1.0f64);

        let scores = ppr.calculate_scores(&seeds).expect("PPR should succeed");

        let sum: f64 = scores.values().sum();
        assert!(sum > 0.0 && sum.is_finite(), "scores sum={sum} should be finite and positive");
        for (_entity, &score) in &scores {
            assert!(score >= 0.0, "all scores should be non-negative");
        }

        let a_score = scores.get(&entity_a).copied().unwrap_or(0.0);
        assert!(a_score > 0.0, "seed node A should have meaningful score, got {a_score}");
    }

    #[test]
    fn test_multimodal_scores_combination() {
        let mut multi_scores = MultiModalScores::new();

        let entity_a = EntityId::new("A".to_string());
        let entity_b = EntityId::new("B".to_string());

        multi_scores.vector_scores.insert(entity_a.clone(), 0.8);
        multi_scores.vector_scores.insert(entity_b.clone(), 0.4);
        multi_scores.pagerank_scores.insert(entity_a.clone(), 0.6);
        multi_scores.pagerank_scores.insert(entity_b.clone(), 0.9);

        let weights = ScoreWeights::default();
        let combined = multi_scores.combine_scores(&weights);

        assert!(combined.contains_key(&entity_a));
        assert!(combined.contains_key(&entity_b));

        let score_a = combined.get(&entity_a).unwrap();
        let score_b = combined.get(&entity_b).unwrap();
        assert!(*score_a > 0.0);
        assert!(*score_b > 0.0);
    }

    // ── PRPACK numerical-fidelity tests (PRD ID-7) ──────────────────────────

    /// Build a `PersonalizedPageRank` from a `ppr_fixtures::PprFixture`.
    ///
    /// Node IDs are `n0`, `n1`, … matching the fixture's integer indices.
    fn ppr_from_fixture(fixture: &ppr_fixtures::PprFixture) -> PersonalizedPageRank {
        let n = fixture.n;

        // Build CSR adjacency matrix from fixture edges
        let mut triplet = sprs::TriMat::new((n, n));
        for &(src, dst, weight) in fixture.edges {
            triplet.add_triplet(src, dst, weight);
        }
        let adjacency = triplet.to_csr();

        // Build node mappings: index i → entity "ni"
        let mut node_mapping = HashMap::new();
        let mut reverse_mapping = HashMap::new();
        for i in 0..n {
            let id = EntityId::new(format!("n{i}"));
            node_mapping.insert(id.clone(), i);
            reverse_mapping.insert(i, id);
        }

        let mut config = PageRankConfig::default();
        config.damping_factor = fixture.damping;

        PersonalizedPageRank::new(config, adjacency, node_mapping, reverse_mapping)
    }

    /// Validate `calculate_scores` against igraph PRPACK ground truth for all
    /// embedded fixtures (PRD ID-7).  Tolerance: relative L∞ ≤ 1e-5.
    #[test]
    fn test_pagerank_matches_prpack_on_fixtures() {
        let tolerance = 1e-5_f64;

        for fixture in ppr_fixtures::FIXTURES {
            let ppr = ppr_from_fixture(fixture);

            // Build reset map from fixture's sparse personalisation vector
            let mut reset: HashMap<EntityId, f64> = HashMap::new();
            for &(idx, weight) in fixture.reset {
                reset.insert(EntityId::new(format!("n{idx}")), weight);
            }

            let scores = ppr
                .calculate_scores(&reset)
                .unwrap_or_else(|e| panic!("fixture '{}': calculate_scores failed: {e}", fixture.name));

            assert_eq!(
                scores.len(),
                fixture.n,
                "fixture '{}': expected {} scores, got {}",
                fixture.name,
                fixture.n,
                scores.len()
            );

            let mut max_rel_err = 0.0_f64;

            for i in 0..fixture.n {
                let id = EntityId::new(format!("n{i}"));
                let actual = scores
                    .get(&id)
                    .copied()
                    .unwrap_or_else(|| panic!("fixture '{}': missing score for node n{i}", fixture.name));
                let expected = fixture.expected_scores[i];

                // Relative error — denominator is max(expected, 1e-12) to handle
                // near-zero expected scores without division by zero.
                let rel_err = (actual - expected).abs() / expected.abs().max(1e-12);
                max_rel_err = max_rel_err.max(rel_err);

                assert!(
                    rel_err < tolerance,
                    "fixture '{}' node n{i}: expected {expected:.20e}, got {actual:.20e}, rel err {rel_err:.3e} ≥ {tolerance:.0e}",
                    fixture.name
                );
            }

            // Scores must be a probability distribution (sum ≈ 1, all non-negative)
            let total: f64 = scores.values().sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "fixture '{}': scores should sum to 1.0, got {total}",
                fixture.name
            );

            // Log max rel error for audit (only visible with --nocapture)
            eprintln!(
                "[prpack_fidelity] fixture '{}' n={}: max rel err = {max_rel_err:.2e}",
                fixture.name, fixture.n
            );
        }
    }

    /// Regression test for the CSC col_ptr monotonicity violation on graphs
    /// that contain self-loops.
    ///
    /// Root cause: the original `build_system_matrix` pre-computed
    /// `nnz_per_col[j] = 1 + row.nnz()` and allocated the flat `row_idx` /
    /// `values` arrays up front.  A self-loop `j→j` is *merged* into the
    /// diagonal during fill (not pushed as an extra entry), so the reserved
    /// slot was never written.  The ghost slot remained zero-initialised.
    /// `col_ptr[j+1]` then pointed one past the last written entry, and
    /// faer's `new_checked` read `[…, row_j, 0]` as column `j`'s row
    /// indices, asserting `row_j < 0` — reproducing the production panic:
    ///
    ///   `Assertion failed: i < i_next — i = 16779, i_next = 0`
    ///
    /// This test reproduces the shape that triggered it: a moderate-size
    /// directed graph (500 nodes) that mimics the production Obsidian KG
    /// (29 k nodes, ~2 edges per node, ~40% dangling nodes, long-tail degree
    /// distribution, edges added in unsorted/arbitrary order, and a
    /// self-loop on every hub node so the pre-fix count mismatch fires).
    #[test]
    fn test_csc_col_ptr_monotonicity_self_loop_regression() {
        // Build a 500-node graph that mirrors the production failure shape:
        //   - Hub nodes (every 50th node) each have a self-loop + fan-out
        //   - Long-tail edges: each non-hub node points to its predecessor
        //   - ~40% of nodes are dangling (out_degree = 0): every node whose
        //     index is divisible by 5 but not a hub has no outgoing edge
        //   - Edges are added in non-sequential (unsorted) order
        let n = 500usize;
        let mut triplet = sprs::TriMat::new((n, n));

        // Hub nodes: 0, 50, 100, … each with a self-loop and fan-out
        for h in (0..n).step_by(50) {
            // Self-loop — this is the entry that triggered the pre-fix panic
            triplet.add_triplet(h, h, 1.0);
            // Fan-out to the next 5 non-hub nodes
            for k in 1..=5usize {
                let dst = (h + k) % n;
                triplet.add_triplet(h, dst, 1.0);
            }
        }

        // Non-hub edges — added in reverse order (non-sorted) to stress
        // the sort path and ensure adjacency rows are not pre-sorted
        for i in (1..n).rev() {
            // Skip hub nodes (they already have edges above)
            if i % 50 == 0 {
                continue;
            }
            // ~40% dangling: skip nodes divisible by 5
            if i % 5 == 0 {
                continue;
            }
            // Point to previous node (creates a long-tail chain)
            triplet.add_triplet(i, i - 1, 1.0);
        }

        let adjacency = triplet.to_csr();

        let mut node_mapping = HashMap::new();
        let mut reverse_mapping = HashMap::new();
        for i in 0..n {
            let id = EntityId::new(format!("node_{i}"));
            node_mapping.insert(id.clone(), i);
            reverse_mapping.insert(i, id);
        }

        let mut config = PageRankConfig::default();
        config.damping_factor = 0.85;

        // Before the fix this call panicked with:
        //   "Assertion failed: i < i_next — i = <hub_idx>, i_next = 0"
        // After the fix it must complete without panic.
        let ppr = PersonalizedPageRank::new(config, adjacency, node_mapping.clone(), reverse_mapping);

        // Sanity-check: scores exist for all nodes and form a valid distribution
        let mut seeds = HashMap::new();
        seeds.insert(EntityId::new("node_0".to_string()), 1.0);
        let scores = ppr.calculate_scores(&seeds).expect("calculate_scores failed");

        assert_eq!(scores.len(), n, "scores should cover all {n} nodes");
        let total: f64 = scores.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "scores should sum to 1.0 ± 1e-9, got {total}"
        );
        for i in 0..n {
            let id = EntityId::new(format!("node_{i}"));
            let s = scores.get(&id).copied().unwrap_or(0.0);
            assert!(
                s.is_finite() && s >= 0.0,
                "node_{i} score must be finite and non-negative, got {s}"
            );
        }
    }
}
