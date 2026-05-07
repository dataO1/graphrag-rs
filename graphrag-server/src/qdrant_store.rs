//! Qdrant Vector Store Integration
//!
//! Provides integration with Qdrant vector database for production deployments.
//!
//! ## Features
//!
//! - Store document embeddings with JSON payload metadata
//! - Store entities and relationships as payload
//! - Advanced filtering and search
//! - Collection management
//! - Batch operations
//!
//! ## Usage
//!
//! ```rust
//! let store = QdrantStore::new("http://localhost:6334", "graphrag").await?;
//! store.create_collection(384).await?;
//! store.add_document("doc1", embedding, metadata).await?;
//! let results = store.search(query_embedding, 10, None).await?;
//! ```

use qdrant_client::{
    qdrant::{
        points_selector::PointsSelectorOneOf, Condition, CreateCollectionBuilder,
        DeletePointsBuilder, Distance, Filter, GetPointsBuilder, PointStruct, PointsIdsList,
        ScrollPointsBuilder, SearchPointsBuilder, SetPayloadPointsBuilder, UpsertPointsBuilder,
        Value as QdrantValue, VectorParamsBuilder,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Qdrant store errors
#[derive(Debug, thiserror::Error)]
pub enum QdrantError {
    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Collection error: {0}")]
    CollectionError(String),

    #[error("Operation error: {0}")]
    OperationError(String),

    #[error("Not found: {0}")]
    #[allow(dead_code)]
    NotFound(String),
}

/// Entity stored in Qdrant payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Relationship stored in Qdrant payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub source: String,
    pub relation: String,
    pub target: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Wire-format envelope for persisting a single graphrag-core Entity in
/// Qdrant. We store the whole serde-serialized Entity in `entity_json`
/// and surface a few flat fields (id/name/type) so basic Qdrant filters
/// stay possible without parsing JSON. Versioned to make future schema
/// migrations explicit (currently always 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEntity {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub entity_type: String,
    /// Full graphrag-core::Entity round-tripped through serde_json. Treated
    /// as opaque on the persistence side; loaders deserialize into the
    /// canonical Entity struct.
    pub entity_json: serde_json::Value,
}

/// Wire-format envelope for persisting a single graphrag-core Relationship
/// in Qdrant. Same pattern as PersistedEntity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedRelationship {
    pub schema_version: u32,
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub relationship_json: serde_json::Value,
}

/// Document metadata stored in Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: String,
    pub text: String,
    pub chunk_index: usize,
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
    pub timestamp: String,
    /// SHA-256 of the document content (lowercase hex). Used for
    /// dedup at ingest: if a point with the same hash already exists,
    /// `add_document` returns its id instead of inserting a duplicate.
    /// Optional so payloads written by older builds parse cleanly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Caller-supplied id. The Qdrant point id itself is a UUID
    /// (Qdrant requires UUID/u64 ids); we store the human-supplied
    /// id separately in the payload so callers can delete by it.
    /// Optional for back-compat with payloads written before this
    /// field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Monotonic version counter per `user_id`. Bumped on every
    /// upsert; first ingest of a given user_id gets `version = 1`.
    /// Old payloads without this field load as `None` and the
    /// retrieval/upsert paths treat them as the implicit current
    /// version (== 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    /// RFC 3339 timestamp of when this version landed. Distinct from
    /// `timestamp` (which is set on the original ingest and
    /// preserved across upserts of the same user_id). Used by
    /// `as_of` retrieval filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    /// True iff this is the most recent version for its `user_id`.
    /// Default qdrant payload filter is `is_current = true`, which
    /// keeps top-K clean of superseded versions. On upsert, the
    /// previous current point's `is_current` flips to `false`. Old
    /// payloads without this field load as `None`; treat as
    /// implicitly current (so legacy ingests keep showing up).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_current: Option<bool>,

    // ── Block-level / source provenance fields (Phase B) ──
    //
    // These travel with chunks emitted by block-aware ingest. Older
    // payloads without them load as None and the legacy doc-level
    // path keeps working unchanged.

    /// Source URI for provenance. `file://...` for path-based ingest,
    /// `obsidian://vault/<vault>/<path>` from the Obsidian gateway,
    /// `https://...` / `arxiv:...` for inline ingest with caller-
    /// supplied source. Required for `content`-form ingest now;
    /// optional on read for back-compat with pre-source payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Stable id within the source doc. For Obsidian-originated
    /// content this is `<heading-path>::<index>` or `^block-id`.
    /// Used to scope the (user_id, block_id) supersede tuple so a
    /// single-paragraph edit only invalidates one chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_id: Option<String>,

    /// sha256 of the block's normalized content. Used by the plugin
    /// (not the server) for diff; persisted so the server can
    /// optionally diff later for non-plugin clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_hash: Option<String>,

    /// Heading hierarchy ancestors, root → leaf. Joined into the
    /// embedding's contextual prefix at chunk-pack time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub heading_path: Vec<String>,

    /// 1-indexed inclusive line range in the source file as ingested.
    /// Snapshot at ingest time — may drift if the file changed since.
    /// Agent should treat as a navigation hint, not a stable id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_start: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_end: Option<u32>,

    /// Phase 6: unix-seconds timestamp of when LLM entity extraction
    /// last ran successfully against this chunk. `None` means "never
    /// extracted" — `extend_graph` queries qdrant with this filter to
    /// find work, so this field is the dedup signal that replaces the
    /// in-memory `processed_chunks` set. Set by graphrag-server after
    /// every successful entity-extraction batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entities_extracted_at: Option<i64>,

    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Lightweight summary returned by `list_documents` — title, ids,
/// and a content excerpt are enough for an agent to decide whether
/// to read the full doc, without paying the bandwidth of every
/// Qdrant payload field on a fleet-wide list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: String,
    pub user_id: Option<String>,
    pub title: String,
    pub timestamp: String,
    pub excerpt: String,
}

/// Render a `RetrievedPoint`'s id into a String. Qdrant ids are either
/// UUID strings or u64 numbers; both render to a String here so callers
/// don't have to branch.
fn point_id_to_string(point: qdrant_client::qdrant::RetrievedPoint) -> Option<String> {
    match point.id?.point_id_options? {
        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s) => Some(s),
        qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => Some(n.to_string()),
    }
}

/// Search result from Qdrant
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: DocumentMetadata,
}

/// Qdrant vector store
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
}

impl QdrantStore {
    /// Create a new Qdrant store
    ///
    /// # Arguments
    /// * `url` - Qdrant server URL (e.g., "http://localhost:6334")
    /// * `collection_name` - Collection name for this graph
    pub async fn new(url: &str, collection_name: &str) -> Result<Self, QdrantError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| QdrantError::ConnectionError(e.to_string()))?;

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    /// Create a collection with the specified dimension
    ///
    /// # Arguments
    /// * `dimension` - Embedding dimension (e.g., 384 for MiniLM, 768 for BERT)
    pub async fn create_collection(&self, dimension: u64) -> Result<(), QdrantError> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(VectorParamsBuilder::new(dimension, Distance::Cosine)),
            )
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        Ok(())
    }

    /// Check if collection exists
    pub async fn collection_exists(&self) -> Result<bool, QdrantError> {
        match self.client.collection_info(&self.collection_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Delete the collection
    #[allow(dead_code)]
    pub async fn delete_collection(&self) -> Result<(), QdrantError> {
        self.client
            .delete_collection(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        Ok(())
    }

    /// Add a document chunk with metadata
    ///
    /// # Arguments
    /// * `id` - Unique document ID
    /// * `embedding` - Embedding vector
    /// * `metadata` - Document metadata including entities and relationships
    pub async fn add_document(
        &self,
        id: &str,
        embedding: Vec<f32>,
        metadata: DocumentMetadata,
    ) -> Result<(), QdrantError> {
        let payload = serde_json::to_value(&metadata)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        use std::collections::HashMap;
        let point = PointStruct::new(
            id.to_string(),
            embedding,
            payload
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .map(|(k, v)| (k, QdrantValue::from(v)))
                .collect::<HashMap<String, QdrantValue>>(),
        );

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, vec![point]))
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Add multiple document chunks in batch
    #[allow(dead_code)]
    pub async fn add_documents_batch(
        &self,
        documents: Vec<(String, Vec<f32>, DocumentMetadata)>,
    ) -> Result<(), QdrantError> {
        let points: Vec<PointStruct> = documents
            .into_iter()
            .map(|(id, embedding, metadata)| {
                let payload = serde_json::to_value(&metadata).unwrap();
                PointStruct::new(
                    id,
                    embedding,
                    payload
                        .as_object()
                        .unwrap()
                        .clone()
                        .into_iter()
                        .map(|(k, v)| (k, QdrantValue::from(v)))
                        .collect::<HashMap<String, QdrantValue>>(),
                )
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Search for similar documents
    ///
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `limit` - Maximum number of results
    /// * `filter` - Optional filter on metadata fields
    ///
    /// # Returns
    /// Vector of search results with scores and metadata
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>, QdrantError> {
        let mut search_builder =
            SearchPointsBuilder::new(&self.collection_name, query_embedding, limit as u64)
                .with_payload(true);

        if let Some(f) = filter {
            search_builder = search_builder.filter(f);
        }

        let results = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        let search_results: Vec<SearchResult> = results
            .result
            .into_iter()
            .map(|point| {
                let payload_value = serde_json::to_value(&point.payload).unwrap();
                let metadata: DocumentMetadata = serde_json::from_value(payload_value).unwrap();

                // Extract ID from PointId enum
                let id_str = match point.id.unwrap() {
                    qdrant_client::qdrant::PointId {
                        point_id_options:
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)),
                    } => s,
                    qdrant_client::qdrant::PointId {
                        point_id_options:
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)),
                    } => n.to_string(),
                    _ => String::from("unknown"),
                };

                SearchResult {
                    id: id_str,
                    score: point.score,
                    metadata,
                }
            })
            .collect();

        Ok(search_results)
    }

    /// Delete a document by ID
    pub async fn delete_document(&self, id: &str) -> Result<(), QdrantError> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name).points(PointsIdsList {
                    ids: vec![id.to_string().into()],
                }),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Delete every chunk under a `user_id` — including superseded
    /// historical versions. Used by `DELETE /api/documents/<path>`
    /// when the caller-supplied id is recognized as a user_id, so a
    /// hard delete genuinely removes the doc and all its history
    /// (the watcher's REMOVE handler relies on this — without it,
    /// only the most-recent point gets removed and old superseded
    /// chunks linger in qdrant under that user_id).
    pub async fn delete_by_user_id(&self, user_id: &str) -> Result<(), QdrantError> {
        let filter = Filter::must([Condition::matches("user_id", user_id.to_string())]);
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsSelectorOneOf::Filter(filter)),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(())
    }

    /// Look up a Qdrant point id by the caller-supplied user_id. Returns
    /// the first match (user_id is treated as unique-per-document). Used
    /// by `delete_document` so callers can refer to documents by the id
    /// they handed us at ingest, not the internal UUID.
    pub async fn find_id_by_user_id(
        &self,
        user_id: &str,
    ) -> Result<Option<String>, QdrantError> {
        let filter = Filter::must([Condition::matches("user_id", user_id.to_string())]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(filter)
                    .with_payload(false)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(resp.result.into_iter().next().and_then(point_id_to_string))
    }

    /// Look up the *current* version of a doc by its caller-supplied
    /// `user_id`. Returns the Qdrant point id and its stored
    /// metadata. Used by the upsert path to decide whether to
    /// supersede an existing version or write fresh.
    ///
    /// "Current" = `is_current = true`. Payloads written before this
    /// field existed have `is_current = None`; we MATCH those too
    /// (treat as implicitly current) so legacy points still
    /// participate in upsert correctly.
    pub async fn find_current_by_user_id(
        &self,
        user_id: &str,
    ) -> Result<Option<(String, DocumentMetadata)>, QdrantError> {
        // Two-step: first try `is_current = true`. Only if that
        // misses do we fall back to the legacy match (no
        // is_current field). Avoids a heavier query in the hot
        // path while still upgrading old data lazily.
        let primary = Filter::must([
            Condition::matches("user_id", user_id.to_string()),
            Condition::matches("is_current", true),
        ]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(primary)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        if let Some(point) = resp.result.into_iter().next() {
            let id = match point_id_to_string(point.clone()) {
                Some(s) => s,
                None => return Ok(None),
            };
            let payload_value = serde_json::to_value(&point.payload)
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;
            let metadata: DocumentMetadata = serde_json::from_value(payload_value)
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;
            return Ok(Some((id, metadata)));
        }

        // Legacy fallback: any payload with this user_id and no
        // `is_current` field. Treats unversioned legacy points as
        // implicitly current. Returns the first match (legacy
        // points were 1:1 user_id-to-point).
        let legacy_filter = Filter::must([Condition::matches("user_id", user_id.to_string())]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(legacy_filter)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let Some(point) = resp.result.into_iter().next() else {
            return Ok(None);
        };
        let id = match point_id_to_string(point.clone()) {
            Some(s) => s,
            None => return Ok(None),
        };
        let payload_value = serde_json::to_value(&point.payload)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let metadata: DocumentMetadata = serde_json::from_value(payload_value)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(Some((id, metadata)))
    }

    /// Mark every chunk under `user_id` as superseded — sets
    /// `is_current = false` via Qdrant's filter-targeted set_payload.
    /// Used by the upsert path: just before writing the new version,
    /// we flip the flag on the previous current chunks so retrieval's
    /// default `is_current = true` filter starts skipping them.
    ///
    /// Look up the current chunk for a (user_id, block_id) tuple.
    /// Returns the stored DocumentMetadata if a live (is_current=true)
    /// chunk exists, otherwise None. Used by the stale-context event
    /// emit path so we can compute a delta between the prior
    /// content and the incoming block before flipping is_current.
    pub async fn find_current_block(
        &self,
        user_id: &str,
        block_id: &str,
    ) -> Result<Option<DocumentMetadata>, QdrantError> {
        let filter = Filter::must([
            Condition::matches("user_id", user_id.to_string()),
            Condition::matches("block_id", block_id.to_string()),
            Condition::matches("is_current", true),
        ]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(filter)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let Some(point) = resp.result.into_iter().next() else {
            return Ok(None);
        };
        let payload_value = serde_json::to_value(&point.payload)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let metadata: DocumentMetadata = serde_json::from_value(payload_value)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(Some(metadata))
    }

    /// Look up the current chunk for a block_id WITHOUT user_id
    /// scoping. Used by the revalidate endpoint, which receives a
    /// list of (block_id, etag) pairs without user_id (the agent
    /// holds opaque tags from prior recalls; user_id would force
    /// every client to track per-doc identity). Block ids are
    /// already namespaced under their source's heading-path so
    /// global uniqueness is OK in practice.
    pub async fn find_current_block_global(
        &self,
        block_id: &str,
    ) -> Result<Option<DocumentMetadata>, QdrantError> {
        let filter = Filter::must([
            Condition::matches("block_id", block_id.to_string()),
            Condition::matches("is_current", true),
        ]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(filter)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let Some(point) = resp.result.into_iter().next() else {
            return Ok(None);
        };
        let payload_value = serde_json::to_value(&point.payload)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let metadata: DocumentMetadata = serde_json::from_value(payload_value)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(Some(metadata))
    }

    /// Mark a SPECIFIC (user_id, block_id) tuple as superseded. Used
    /// by block-aware ingest: when a single block changes, only
    /// chunks tagged with that block_id flip to is_current=false,
    /// preserving sibling blocks of the same doc as current. The
    /// `removed_block_ids` path uses this with each removed id.
    pub async fn mark_block_superseded(
        &self,
        user_id: &str,
        block_id: &str,
    ) -> Result<(), QdrantError> {
        let filter = Filter::must([
            Condition::matches("user_id", user_id.to_string()),
            Condition::matches("block_id", block_id.to_string()),
            Condition::matches("is_current", true),
        ]);
        let mut payload: HashMap<String, QdrantValue> = HashMap::new();
        payload.insert("is_current".to_string(), QdrantValue::from(false));
        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection_name, payload)
                    .points_selector(PointsSelectorOneOf::Filter(filter)),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(())
    }

    /// Idempotent: calling on a user_id with no current chunks is a
    /// no-op (Qdrant set_payload with an empty match-set returns
    /// success without writing).
    pub async fn mark_user_id_superseded(&self, user_id: &str) -> Result<(), QdrantError> {
        let filter = Filter::must([
            Condition::matches("user_id", user_id.to_string()),
            Condition::matches("is_current", true),
        ]);
        let mut payload: HashMap<String, QdrantValue> = HashMap::new();
        payload.insert("is_current".to_string(), QdrantValue::from(false));

        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection_name, payload)
                    .points_selector(PointsSelectorOneOf::Filter(filter)),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(())
    }

    /// Look up an existing point by content hash. Returns the Qdrant
    /// point id (and stored DocumentMetadata) if a match exists. Drives
    /// dedup at ingest: same content → same point, no duplicate.
    pub async fn find_by_content_hash(
        &self,
        hash: &str,
    ) -> Result<Option<(String, DocumentMetadata)>, QdrantError> {
        let filter = Filter::must([Condition::matches("content_hash", hash.to_string())]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(filter)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let Some(point) = resp.result.into_iter().next() else {
            return Ok(None);
        };
        let id = match point_id_to_string(point.clone()) {
            Some(s) => s,
            None => return Ok(None),
        };
        let payload_value = serde_json::to_value(&point.payload)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let metadata: DocumentMetadata = serde_json::from_value(payload_value)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(Some((id, metadata)))
    }

    /// List documents stored in Qdrant. Pages through the collection
    /// using scroll; capped at `limit` to keep responses bounded.
    /// Returns lightweight summaries (id, title, timestamp, excerpt) —
    /// callers needing full text should query individual points.
    pub async fn list_documents(
        &self,
        limit: u32,
    ) -> Result<Vec<DocumentSummary>, QdrantError> {
        let mut summaries = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let page_size = limit.min(256).max(1);

        while summaries.len() < limit as usize {
            let mut builder = ScrollPointsBuilder::new(&self.collection_name)
                .with_payload(true)
                .with_vectors(false)
                .limit(page_size);
            if let Some(off) = offset.take() {
                builder = builder.offset(off);
            }
            let resp = self
                .client
                .scroll(builder)
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;

            if resp.result.is_empty() {
                break;
            }
            for point in resp.result {
                let id = match point_id_to_string(point.clone()) {
                    Some(s) => s,
                    None => continue,
                };
                let payload_value = match serde_json::to_value(&point.payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let md: DocumentMetadata = match serde_json::from_value(payload_value) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let excerpt = md.text.chars().take(160).collect::<String>();
                summaries.push(DocumentSummary {
                    id,
                    user_id: md.user_id,
                    title: md.title,
                    timestamp: md.timestamp,
                    excerpt,
                });
                if summaries.len() >= limit as usize {
                    break;
                }
            }
            offset = resp.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(summaries)
    }

    /// Vector-search the entity sidecar collection. Returns
    /// `(entity_id, score)` pairs where `entity_id` is the stable
    /// graphrag-core `EntityId` string (read out of the persisted
    /// payload, NOT the Qdrant point UUID). `limit` is the top-K.
    ///
    /// Returns an empty Vec if the collection doesn't exist (cold
    /// start) — callers should treat that as "no seeds, fall back
    /// to chunk-vector retrieval."
    ///
    /// This is the primitive behind MS GraphRAG-style local_search:
    /// the caller embeds the user query, calls this for top-K
    /// entity ids, then asks graphrag-core to expand from those
    /// seeds and synthesize an answer.
    pub async fn search_entities(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>, QdrantError> {
        let coll = self.entities_collection();
        if self.client.collection_info(&coll).await.is_err() {
            return Ok(Vec::new());
        }
        let resp = self
            .client
            .search_points(
                SearchPointsBuilder::new(&coll, query_embedding, limit as u64).with_payload(true),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        let mut out = Vec::new();
        for point in resp.result {
            let score = point.score;
            // Read the stable graphrag-core EntityId off the payload's
            // `id` field (the one PersistedEntity sets). The Qdrant
            // point's own UUID is a UUID5 hash of the entity id and
            // not directly useful to the caller.
            let payload_value = match serde_json::to_value(&point.payload) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let entity_id = match payload_value.get("id").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            out.push((entity_id, score));
        }
        Ok(out)
    }

    /// Vector-search the relationship sidecar collection. Returns
    /// `((source_entity_id, target_entity_id, relation_type), score)`
    /// triples — the stable graphrag-core identity for a relationship,
    /// read out of the PersistedRelationship payload (NOT the Qdrant
    /// point UUID).
    ///
    /// Returns an empty Vec if the collection doesn't exist (cold
    /// start). Callers should treat that as "no high-level seeds,
    /// fall back to entity-only retrieval or chunk-vector path."
    ///
    /// This is the primitive behind LightRAG's `global` retrieval:
    /// the caller embeds the user query (or its high-level keywords),
    /// calls this for top-K seed relations, then asks graphrag-core
    /// to expand from those seeds and synthesize an answer.
    pub async fn search_relationships(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<((String, String, String), f32)>, QdrantError> {
        let coll = self.relationships_collection();
        if self.client.collection_info(&coll).await.is_err() {
            return Ok(Vec::new());
        }
        let resp = self
            .client
            .search_points(
                SearchPointsBuilder::new(&coll, query_embedding, limit as u64).with_payload(true),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        let mut out = Vec::new();
        for point in resp.result {
            let score = point.score;
            let payload_value = match serde_json::to_value(&point.payload) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let source = match payload_value.get("source").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let target = match payload_value.get("target").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let relation_type = match payload_value
                .get("relation_type")
                .and_then(|v| v.as_str())
            {
                Some(s) => s.to_string(),
                None => continue,
            };
            out.push(((source, target, relation_type), score));
        }
        Ok(out)
    }

    /// Scroll through the collection returning full DocumentMetadata payloads.
    /// Unlike `list_documents` (which returns lightweight summaries with a
    /// 160-char excerpt), this returns the full text so callers can rechunk
    /// it for graph hydration on startup. `limit` caps the total returned;
    /// pass a generous value (e.g. 1_000_000) to drain the whole collection.
    pub async fn list_full_documents(
        &self,
        limit: u32,
    ) -> Result<Vec<(String, DocumentMetadata)>, QdrantError> {
        let mut docs = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let page_size = limit.min(256).max(1);

        while docs.len() < limit as usize {
            let mut builder = ScrollPointsBuilder::new(&self.collection_name)
                .with_payload(true)
                .with_vectors(false)
                .limit(page_size);
            if let Some(off) = offset.take() {
                builder = builder.offset(off);
            }
            let resp = self
                .client
                .scroll(builder)
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;

            if resp.result.is_empty() {
                break;
            }
            for point in resp.result {
                let id = match point_id_to_string(point.clone()) {
                    Some(s) => s,
                    None => continue,
                };
                let payload_value = match serde_json::to_value(&point.payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let md: DocumentMetadata = match serde_json::from_value(payload_value) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                docs.push((id, md));
                if docs.len() >= limit as usize {
                    break;
                }
            }
            offset = resp.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(docs)
    }

    /// Phase 6: scroll for chunks where `entities_extracted_at` is unset
    /// (i.e. LLM entity extraction hasn't run yet). This is the source-
    /// of-truth dedup signal — replaces the in-memory `processed_chunks`
    /// HashSet that the old graphrag-core flow kept. Filters
    /// `is_current = true` so superseded versions don't get re-extracted.
    pub async fn list_unextracted_chunks(
        &self,
        limit: u32,
    ) -> Result<Vec<(String, String)>, QdrantError> {
        self.list_unextracted_chunks_excluding(limit, &[]).await
    }

    /// Same as `list_unextracted_chunks` but excludes a caller-supplied
    /// set of point ids — used by the cross-page pipeline in
    /// `do_append_graph` to skip chunks that are *currently being
    /// processed* by a still-draining consumer task. Without the
    /// exclude, those in-flight chunks would re-appear in the next
    /// page (they're still NULL on `entities_extracted_at`) and double-
    /// extract.
    pub async fn list_unextracted_chunks_excluding(
        &self,
        limit: u32,
        exclude_ids: &[String],
    ) -> Result<Vec<(String, String)>, QdrantError> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let page_size = limit.min(256).max(1);

        // Filter: is_current=true AND entities_extracted_at IS NULL,
        // AND (when caller provided) NOT in the exclude_ids set.
        // Qdrant's `is_empty` matches "field absent or null", which is
        // exactly the "not yet extracted" signal we want. Combined with
        // is_current=true so we never re-extract superseded blocks.
        let mut filter = Filter {
            must: vec![
                Condition::matches("is_current", true),
                Condition::is_empty("entities_extracted_at"),
            ],
            ..Default::default()
        };
        if !exclude_ids.is_empty() {
            // Use must_not + has_id to drop the in-flight set. Our
            // chunk ids are UUID strings throughout.
            use qdrant_client::qdrant::PointId;
            let ids: Vec<PointId> = exclude_ids
                .iter()
                .map(|s| PointId {
                    point_id_options: Some(
                        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s.clone()),
                    ),
                })
                .collect();
            filter.must_not.push(Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::HasId(
                    qdrant_client::qdrant::HasIdCondition { has_id: ids },
                )),
            });
        }

        while out.len() < limit as usize {
            let mut builder = ScrollPointsBuilder::new(&self.collection_name)
                .filter(filter.clone())
                .with_payload(true)
                .with_vectors(false)
                .limit(page_size);
            if let Some(off) = offset.take() {
                builder = builder.offset(off);
            }
            let resp = self
                .client
                .scroll(builder)
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;

            if resp.result.is_empty() {
                break;
            }
            for point in resp.result {
                let id = match point_id_to_string(point.clone()) {
                    Some(s) => s,
                    None => continue,
                };
                let payload_value = match serde_json::to_value(&point.payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let md: DocumentMetadata = match serde_json::from_value(payload_value) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if md.text.is_empty() {
                    continue;
                }
                out.push((id, md.text));
                if out.len() >= limit as usize {
                    break;
                }
            }
            offset = resp.next_page_offset;
            if offset.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// Phase 6: mark chunks as having had successful LLM entity
    /// extraction. Sets `entities_extracted_at = ts` on every supplied
    /// point id. Empty list is a no-op.
    pub async fn mark_chunks_extracted(
        &self,
        point_ids: &[String],
        ts: i64,
    ) -> Result<(), QdrantError> {
        if point_ids.is_empty() {
            return Ok(());
        }
        let mut payload: HashMap<String, QdrantValue> = HashMap::new();
        payload.insert("entities_extracted_at".to_string(), QdrantValue::from(ts));

        let ids: Vec<qdrant_client::qdrant::PointId> = point_ids
            .iter()
            .map(|s| s.clone().into())
            .collect();

        if ids.is_empty() {
            return Ok(());
        }

        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection_name, payload)
                    .points_selector(PointsSelectorOneOf::Points(PointsIdsList { ids }))
                    .wait(true),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(())
    }

    /// Phase 6: batch-fetch chunk content by Qdrant point id.
    /// Used by recall path (graph_aware_query) to populate the
    /// chunk_contents map for `ask_with_dual_seeds` from the qdrant
    /// source-of-truth, since the in-memory KG no longer holds
    /// chunks. Missing ids are silently skipped — caller can detect
    /// gaps by comparing input vs returned-map size.
    pub async fn fetch_chunks_by_ids(
        &self,
        point_ids: &[String],
    ) -> Result<HashMap<String, String>, QdrantError> {
        let mut out: HashMap<String, String> = HashMap::new();
        if point_ids.is_empty() {
            return Ok(out);
        }
        let ids: Vec<qdrant_client::qdrant::PointId> = point_ids
            .iter()
            .map(|s| s.clone().into())
            .collect();
        if ids.is_empty() {
            return Ok(out);
        }
        let resp = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.collection_name, ids)
                    .with_payload(true)
                    .with_vectors(false),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        for point in resp.result {
            let id = match point_id_to_string(point.clone()) {
                Some(s) => s,
                None => continue,
            };
            let payload_value = match serde_json::to_value(&point.payload) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let md: DocumentMetadata = match serde_json::from_value(payload_value) {
                Ok(m) => m,
                Err(_) => continue,
            };
            out.insert(id, md.text);
        }
        Ok(out)
    }

    /// Clear all documents from collection
    #[allow(dead_code)]
    pub async fn clear(&self) -> Result<(), QdrantError> {
        // Delete and recreate collection
        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        let dimension = info
            .result
            .and_then(|c| c.config)
            .and_then(|cfg| cfg.params)
            .and_then(|p| p.vectors_config)
            .and_then(|v| v.config)
            .and_then(|cfg| match cfg {
                qdrant_client::qdrant::vectors_config::Config::Params(params) => Some(params.size),
                _ => None,
            })
            .ok_or_else(|| {
                QdrantError::OperationError("Could not get vector dimension".to_string())
            })?;

        self.delete_collection().await?;
        self.create_collection(dimension).await?;

        Ok(())
    }

    /// Name of the Qdrant collection that backs entity persistence.
    /// Suffixed off the main collection so a deployment with multiple
    /// graphrag-server instances against the same Qdrant cleanly isolates
    /// per-collection graphs (e.g. `graphrag` + `graphrag-entities`).
    pub fn entities_collection(&self) -> String {
        format!("{}-entities", self.collection_name)
    }

    /// Name of the Qdrant collection that backs relationship persistence.
    pub fn relationships_collection(&self) -> String {
        format!("{}-relationships", self.collection_name)
    }

    /// Create the entity + relationship sidecar collections if they don't
    /// already exist, with the supplied vector dimension. Idempotent —
    /// existing collections are left alone (even if their dim differs).
    /// To migrate an existing deploy from a different dim, call
    /// `clear_graph_collections` instead, which delete-and-recreates.
    ///
    /// Currently unused at the call sites: `clear_graph_collections`
    /// recreates from scratch on every persist, which already covers
    /// the bootstrap case. Kept public for a future incremental upsert
    /// path that wants to pre-warm the schema without dropping data.
    #[allow(dead_code)]
    pub async fn ensure_graph_collections(&self, dimension: u64) -> Result<(), QdrantError> {
        for name in [self.entities_collection(), self.relationships_collection()] {
            match self.client.collection_info(&name).await {
                Ok(_) => {},
                Err(_) => {
                    self.client
                        .create_collection(
                            CreateCollectionBuilder::new(&name).vectors_config(
                                VectorParamsBuilder::new(dimension, Distance::Cosine),
                            ),
                        )
                        .await
                        .map_err(|e| QdrantError::CollectionError(e.to_string()))?;
                },
            }
        }
        Ok(())
    }

    /// Wipe the entity + relationship sidecar collections and recreate
    /// them at the supplied dimension. Used by `persist_graph` so a full
    /// rebuild leaves no stale entries behind, and so a deploy that
    /// previously ran with placeholder 1-D vectors gets rebuilt with
    /// real-dimensional vectors on the next build.
    ///
    /// Robust against Qdrant's eventual-consistency on collection
    /// deletion: a `delete_collection` can return Ok before the
    /// namespace is actually freed, so a follow-up `create_collection`
    /// can fail with "already exists." This impl retries the delete +
    /// create once with a short sleep when create errors — observed
    /// in the wild leaving the entities collection wiped but never
    /// repopulated, which is silent data loss against the in-memory
    /// graph.
    pub async fn clear_graph_collections(&self, dimension: u64) -> Result<(), QdrantError> {
        for name in [self.entities_collection(), self.relationships_collection()] {
            self.recreate_collection(&name, dimension).await?;
        }
        Ok(())
    }

    async fn recreate_collection(&self, name: &str, dimension: u64) -> Result<(), QdrantError> {
        let _ = self.client.delete_collection(name).await;
        let first_attempt = self
            .client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorParamsBuilder::new(dimension, Distance::Cosine)),
            )
            .await;
        if first_attempt.is_ok() {
            return Ok(());
        }
        // Likely "already exists" — delete didn't propagate. Retry.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = self.client.delete_collection(name).await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorParamsBuilder::new(dimension, Distance::Cosine)),
            )
            .await
            .map_err(|e| QdrantError::CollectionError(format!("recreate {name}: {e}")))?;
        Ok(())
    }

    /// Persist (entities, relationships) to their sidecar collections,
    /// each row carrying a real description embedding so the collections
    /// double as a vector index over the entity / relationship graph.
    /// Mirrors Microsoft GraphRAG's `final_entities.parquet` +
    /// `description_embedding` column convention: enables semantic
    /// seed-point retrieval (find entities similar to query, walk the
    /// graph from there) — the engine behind MS's `local_search`.
    ///
    /// Strategy: clear-and-repopulate at the supplied dimension. The
    /// in-memory graph is the source of truth at the moment of this
    /// call; anything not present is removed from persistence.
    ///
    /// Each `(payload, embedding)` pair gets a deterministic UUID5
    /// point id derived from its stable identity (entity id, or
    /// `source|relation|target` for relationships) so a future
    /// incremental upsert path can target individual rows without
    /// reading the whole collection.
    pub async fn persist_graph(
        &self,
        entity_payloads: Vec<(PersistedEntity, Vec<f32>)>,
        relationship_payloads: Vec<(PersistedRelationship, Vec<f32>)>,
        dimension: u64,
    ) -> Result<(usize, usize), QdrantError> {
        self.clear_graph_collections(dimension).await?;

        if !entity_payloads.is_empty() {
            let points: Vec<PointStruct> = entity_payloads
                .iter()
                .map(|(e, embedding)| {
                    let pid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, e.id.as_bytes())
                        .to_string();
                    let payload_value = serde_json::to_value(e).unwrap_or(serde_json::json!({}));
                    let payload_map: HashMap<String, QdrantValue> = payload_value
                        .as_object()
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, v)| (k, QdrantValue::from(v)))
                        .collect();
                    PointStruct::new(pid, embedding.clone(), payload_map)
                })
                .collect();
            self.client
                .upsert_points(UpsertPointsBuilder::new(self.entities_collection(), points))
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        }

        if !relationship_payloads.is_empty() {
            let points: Vec<PointStruct> = relationship_payloads
                .iter()
                .map(|(r, embedding)| {
                    let stable = format!("{}|{}|{}", r.source, r.relation_type, r.target);
                    let pid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, stable.as_bytes())
                        .to_string();
                    let payload_value = serde_json::to_value(r).unwrap_or(serde_json::json!({}));
                    let payload_map: HashMap<String, QdrantValue> = payload_value
                        .as_object()
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, v)| (k, QdrantValue::from(v)))
                        .collect();
                    PointStruct::new(pid, embedding.clone(), payload_map)
                })
                .collect();
            self.client
                .upsert_points(UpsertPointsBuilder::new(
                    self.relationships_collection(),
                    points,
                ))
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        }

        Ok((entity_payloads.len(), relationship_payloads.len()))
    }

    /// Incremental upsert into the entities + relationships sidecar
    /// collections. Mirrors `persist_graph` but **without** clearing
    /// the collection first — only the supplied (entity, embedding) /
    /// (relationship, embedding) tuples are written, with the same
    /// deterministic UUID5 ids used by `persist_graph` so they overwrite
    /// the prior versions in place.
    ///
    /// Used by the LightRAG-parity append path: extend_graph emits the
    /// touched-only delta, the persistence layer embeds + upserts only
    /// those rows. A 1-chunk append touching 12 entities writes 12
    /// points instead of re-embedding all 1985.
    ///
    /// Idempotent: calling with empty payloads is a no-op (returns
    /// (0,0)), so this is safe to invoke unconditionally from the
    /// append handler.
    pub async fn persist_graph_delta(
        &self,
        entity_payloads: Vec<(PersistedEntity, Vec<f32>)>,
        relationship_payloads: Vec<(PersistedRelationship, Vec<f32>)>,
        dimension: u64,
    ) -> Result<(usize, usize), QdrantError> {
        // Make sure the sidecar collections exist at the right
        // dimension. ensure_graph_collections is a no-op if they
        // already exist; first-call-after-server-restart creates
        // them at the supplied dim. Re-using the same helper as
        // persist_graph for one source of truth.
        self.ensure_graph_collections(dimension).await?;

        // Build the entity + relationship PointStructs upfront. Cheap
        // (CPU + serde), so doing it before the Qdrant POST overlap
        // doesn't waste latency. Empty inputs short-circuit to a no-op
        // future so the join below stays the same shape regardless.
        let entity_points: Option<Vec<PointStruct>> = if entity_payloads.is_empty() {
            None
        } else {
            Some(
                entity_payloads
                    .iter()
                    .map(|(e, embedding)| {
                        let pid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, e.id.as_bytes())
                            .to_string();
                        let payload_value =
                            serde_json::to_value(e).unwrap_or(serde_json::json!({}));
                        let payload_map: HashMap<String, QdrantValue> = payload_value
                            .as_object()
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .map(|(k, v)| (k, QdrantValue::from(v)))
                            .collect();
                        PointStruct::new(pid, embedding.clone(), payload_map)
                    })
                    .collect(),
            )
        };

        let relationship_points: Option<Vec<PointStruct>> = if relationship_payloads.is_empty() {
            None
        } else {
            Some(
                relationship_payloads
                    .iter()
                    .map(|(r, embedding)| {
                        let stable =
                            format!("{}|{}|{}", r.source, r.relation_type, r.target);
                        let pid = uuid::Uuid::new_v5(
                            &uuid::Uuid::NAMESPACE_OID,
                            stable.as_bytes(),
                        )
                        .to_string();
                        let payload_value =
                            serde_json::to_value(r).unwrap_or(serde_json::json!({}));
                        let payload_map: HashMap<String, QdrantValue> = payload_value
                            .as_object()
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .map(|(k, v)| (k, QdrantValue::from(v)))
                            .collect();
                        PointStruct::new(pid, embedding.clone(), payload_map)
                    })
                    .collect(),
            )
        };

        // Run the two upserts concurrently — they target distinct
        // collections (entities sidecar + relationships sidecar), so
        // the qdrant server already serializes within a single
        // collection but doesn't between collections. Two HTTP/2
        // streams in flight halves the per-flush wall time. Was
        // previously two sequential awaits.
        let entities_collection = self.entities_collection();
        let relationships_collection = self.relationships_collection();
        let client = &self.client;

        let entities_fut = async move {
            if let Some(points) = entity_points {
                client
                    .upsert_points(UpsertPointsBuilder::new(entities_collection, points))
                    .await
                    .map_err(|e| QdrantError::OperationError(e.to_string()))?;
            }
            Ok::<(), QdrantError>(())
        };
        let relationships_fut = async move {
            if let Some(points) = relationship_points {
                client
                    .upsert_points(UpsertPointsBuilder::new(
                        relationships_collection,
                        points,
                    ))
                    .await
                    .map_err(|e| QdrantError::OperationError(e.to_string()))?;
            }
            Ok::<(), QdrantError>(())
        };

        tokio::try_join!(entities_fut, relationships_fut)?;

        Ok((entity_payloads.len(), relationship_payloads.len()))
    }

    /// Load all persisted entities by scrolling the entities sidecar
    /// collection. Returns the wire-format envelopes; callers
    /// (graphrag-server::config_endpoints::set_config) deserialize
    /// `entity_json` into graphrag-core::Entity. Tolerates a missing
    /// collection (returns empty vec) so first-run hydration doesn't
    /// require pre-creating the sidecar.
    pub async fn load_persisted_entities(&self) -> Result<Vec<PersistedEntity>, QdrantError> {
        let coll = self.entities_collection();
        if self.client.collection_info(&coll).await.is_err() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(&coll)
                .with_payload(true)
                .with_vectors(false)
                .limit(256u32);
            if let Some(off) = offset.take() {
                builder = builder.offset(off);
            }
            let resp = self
                .client
                .scroll(builder)
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;
            if resp.result.is_empty() {
                break;
            }
            for point in resp.result {
                if let Ok(v) = serde_json::to_value(&point.payload) {
                    if let Ok(p) = serde_json::from_value::<PersistedEntity>(v) {
                        out.push(p);
                    }
                }
            }
            offset = resp.next_page_offset;
            if offset.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// Load all persisted relationships. Mirror of `load_persisted_entities`.
    pub async fn load_persisted_relationships(
        &self,
    ) -> Result<Vec<PersistedRelationship>, QdrantError> {
        let coll = self.relationships_collection();
        if self.client.collection_info(&coll).await.is_err() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(&coll)
                .with_payload(true)
                .with_vectors(false)
                .limit(256u32);
            if let Some(off) = offset.take() {
                builder = builder.offset(off);
            }
            let resp = self
                .client
                .scroll(builder)
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;
            if resp.result.is_empty() {
                break;
            }
            for point in resp.result {
                if let Ok(v) = serde_json::to_value(&point.payload) {
                    if let Ok(p) = serde_json::from_value::<PersistedRelationship>(v) {
                        out.push(p);
                    }
                }
            }
            offset = resp.next_page_offset;
            if offset.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// Get collection statistics
    pub async fn stats(&self) -> Result<(usize, usize), QdrantError> {
        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        let count = info
            .result
            .as_ref()
            .and_then(|c| c.points_count)
            .unwrap_or(0) as usize;

        let vectors = info
            .result
            .as_ref()
            .and_then(|c| c.vectors_count)
            .unwrap_or(0) as usize;

        Ok((count, vectors))
    }

    /// Get the collection name
    #[allow(dead_code)]
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Qdrant server running
    async fn test_qdrant_store() {
        let store = QdrantStore::new("http://localhost:6334", "test-collection")
            .await
            .unwrap();
        store.create_collection(384).await.unwrap();

        let metadata = DocumentMetadata {
            id: "doc1".to_string(),
            title: "Test Document".to_string(),
            text: "This is a test document".to_string(),
            chunk_index: 0,
            entities: vec![],
            relationships: vec![],
            timestamp: chrono::Utc::now().to_rfc3339(),
            custom: HashMap::new(),
        };

        store
            .add_document("doc1", vec![0.1; 384], metadata)
            .await
            .unwrap();

        let results = store.search(vec![0.1; 384], 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");

        store.delete_collection().await.unwrap();
    }
}
