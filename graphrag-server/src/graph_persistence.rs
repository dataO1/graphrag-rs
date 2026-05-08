//! Conversions between graphrag-core's in-memory `Entity`/`Relationship`
//! types and the wire-format envelopes (`PersistedEntity`/
//! `PersistedRelationship`) we store in Qdrant for cross-restart
//! survival.
//!
//! These helpers exist so the call sites in `main.rs` (build_graph,
//! append_graph) and `config_endpoints.rs` (set_config hydration)
//! don't have to know the serde plumbing.

#[cfg(feature = "qdrant")]
use crate::qdrant_store::{PersistedEntity, PersistedRelationship, QdrantError, QdrantStore};
#[cfg(feature = "qdrant")]
use graphrag_core::core::{Entity, Relationship};

/// Current persistence schema version. Bump when the wire envelope
/// changes incompatibly so loaders can migrate or drop old payloads.
const SCHEMA_VERSION: u32 = 1;

/// Convert a graphrag-core `Entity` into the wire envelope.
#[cfg(feature = "qdrant")]
pub fn entity_to_persisted(entity: &Entity) -> PersistedEntity {
    PersistedEntity {
        schema_version: SCHEMA_VERSION,
        id: entity.id.0.clone(),
        name: entity.name.clone(),
        entity_type: entity.entity_type.clone(),
        entity_json: serde_json::to_value(entity).unwrap_or(serde_json::json!({})),
    }
}

/// Convert a graphrag-core `Relationship` into the wire envelope.
#[cfg(feature = "qdrant")]
pub fn relationship_to_persisted(rel: &Relationship) -> PersistedRelationship {
    PersistedRelationship {
        schema_version: SCHEMA_VERSION,
        source: rel.source.0.clone(),
        target: rel.target.0.clone(),
        relation_type: rel.relation_type.clone(),
        relationship_json: serde_json::to_value(rel).unwrap_or(serde_json::json!({})),
    }
}

/// Round-trip a wire envelope back into a graphrag-core `Entity`.
/// Returns `None` if the stored JSON is malformed (e.g. schema drift
/// across versions); callers should log and skip rather than fail
/// hydration outright.
#[cfg(feature = "qdrant")]
pub fn persisted_to_entity(p: &PersistedEntity) -> Option<Entity> {
    serde_json::from_value(p.entity_json.clone()).ok()
}

/// Round-trip a wire envelope back into a graphrag-core `Relationship`.
#[cfg(feature = "qdrant")]
pub fn persisted_to_relationship(p: &PersistedRelationship) -> Option<Relationship> {
    serde_json::from_value(p.relationship_json.clone()).ok()
}

/// Convenience: dump every entity + relationship currently in `graphrag`
/// to Qdrant, with a description embedding per row. Idempotent (the
/// underlying `QdrantStore::persist_graph` clear-and-repopulates).
/// Returns `(entities_persisted, relationships_persisted)` for telemetry.
///
/// Embedding strategy mirrors Microsoft GraphRAG's `description_embedding`
/// convention: each entity is embedded as `"{name} ({entity_type})"`;
/// each relationship as `"{source_name} {relation_type} {target_name}"`.
/// Reuses `Entity.embedding` / `Relationship.embedding` if already
/// populated by the extractor (saves a round-trip); otherwise batches
/// through the supplied `EmbeddingService` (the same service the
/// document path uses, so vectors live in one consistent space).
///
/// Safe to call from inside a write-locked GraphRAG: only does
/// Embedding + Qdrant I/O, doesn't mutate the in-memory state.
#[cfg(feature = "qdrant")]
pub async fn persist_in_memory_graph(
    graphrag: &graphrag_core::GraphRAG,
    qdrant: &QdrantStore,
    embeddings: &crate::embeddings::EmbeddingService,
) -> Result<(usize, usize), QdrantError> {
    let Some(kg) = graphrag.knowledge_graph() else {
        return Ok((0, 0));
    };

    let dim = embeddings.dimension() as u64;

    // ---- Entities ----------------------------------------------------
    let entities_with_text: Vec<(graphrag_core::core::Entity, String)> = kg
        .entities()
        .map(|e| {
            let text = format!("{} ({})", e.name, e.entity_type);
            (e.clone(), text)
        })
        .collect();

    // Entities already carrying an embedding from the extractor pass
    // skip the round-trip; collect the rest into a batch call.
    let (texts_to_embed, indices_to_embed): (Vec<String>, Vec<usize>) = entities_with_text
        .iter()
        .enumerate()
        .filter_map(|(i, (e, t))| {
            if e.embedding.is_some() {
                None
            } else {
                Some((t.clone(), i))
            }
        })
        .unzip();

    let mut entity_embeddings: Vec<Option<Vec<f32>>> =
        entities_with_text.iter().map(|(e, _)| e.embedding.clone()).collect();

    if !texts_to_embed.is_empty() {
        let refs: Vec<&str> = texts_to_embed.iter().map(String::as_str).collect();
        let computed = embeddings
            .generate(&refs)
            .await
            .map_err(|e| QdrantError::OperationError(format!("entity embed failed: {}", e)))?;
        for (i, vec) in indices_to_embed.iter().zip(computed) {
            entity_embeddings[*i] = Some(vec);
        }
    }

    let entity_payloads_with_vec: Vec<(PersistedEntity, Vec<f32>)> = entities_with_text
        .iter()
        .zip(entity_embeddings.iter())
        .map(|((e, _), emb)| {
            let vec = emb.clone().unwrap_or_else(|| vec![0.0_f32; dim as usize]);
            (entity_to_persisted(e), vec)
        })
        .collect();

    // ---- Relationships ----------------------------------------------
    let relationships_with_text: Vec<(graphrag_core::core::Relationship, String)> = kg
        .relationships()
        .map(|r| {
            let src_name = kg
                .get_entity(&r.source)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let tgt_name = kg
                .get_entity(&r.target)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let text = format!("{} {} {}", src_name, r.relation_type, tgt_name);
            (r.clone(), text)
        })
        .collect();

    let (rel_texts, rel_indices): (Vec<String>, Vec<usize>) = relationships_with_text
        .iter()
        .enumerate()
        .filter_map(|(i, (r, t))| {
            if r.embedding.is_some() {
                None
            } else {
                Some((t.clone(), i))
            }
        })
        .unzip();

    let mut rel_embeddings: Vec<Option<Vec<f32>>> = relationships_with_text
        .iter()
        .map(|(r, _)| r.embedding.clone())
        .collect();

    if !rel_texts.is_empty() {
        let refs: Vec<&str> = rel_texts.iter().map(String::as_str).collect();
        let computed = embeddings.generate(&refs).await.map_err(|e| {
            QdrantError::OperationError(format!("relationship embed failed: {}", e))
        })?;
        for (i, vec) in rel_indices.iter().zip(computed) {
            rel_embeddings[*i] = Some(vec);
        }
    }

    let rel_payloads_with_vec: Vec<(PersistedRelationship, Vec<f32>)> = relationships_with_text
        .iter()
        .zip(rel_embeddings.iter())
        .map(|((r, _), emb)| {
            let vec = emb.clone().unwrap_or_else(|| vec![0.0_f32; dim as usize]);
            (relationship_to_persisted(r), vec)
        })
        .collect();

    qdrant
        .persist_graph(entity_payloads_with_vec, rel_payloads_with_vec, dim)
        .await
}

/// Snapshot of the touched entities + relationships pulled from the
/// in-memory graph under-lock. Owned data so the caller can drop the
/// `RwLock` *before* embedding + persisting, leaving recall
/// uncontended. Mirrors LightRAG `merge_nodes_and_edges`'s practice
/// of only operating on the new chunk_results set.
#[cfg(feature = "qdrant")]
pub struct TouchedSnapshot {
    /// (Entity, embedding-text). Embedding-text is the same
    /// `"name (entity_type)"` string as the full-graph persist path,
    /// pre-computed under-lock so we don't need to peek at the entity
    /// again outside the lock.
    pub entities: Vec<(graphrag_core::core::Entity, String)>,
    /// (Relationship, embedding-text). Embedding-text is
    /// `"src_name relation_type tgt_name"` resolved under-lock against
    /// the live entity table — names are picked up at snapshot time so
    /// later concurrent appends can't change them out from under us.
    pub relationships: Vec<(graphrag_core::core::Relationship, String)>,
}

/// Snapshot the touched entities + relationships from the in-memory
/// graph. Call this WHILE holding the GraphRAG write-lock (since
/// extend_graph just released its mutating hold but the graph itself
/// is still under guard); then drop the lock, then call
/// `persist_touched_snapshot` on the returned data.
#[cfg(feature = "qdrant")]
pub fn snapshot_touched(
    graphrag: &graphrag_core::GraphRAG,
    touched_entity_ids: &[String],
    touched_relationship_keys: &[(String, String, String)],
) -> TouchedSnapshot {
    use graphrag_core::core::EntityId;
    let Some(kg) = graphrag.knowledge_graph() else {
        return TouchedSnapshot {
            entities: Vec::new(),
            relationships: Vec::new(),
        };
    };

    let entities: Vec<(graphrag_core::core::Entity, String)> = touched_entity_ids
        .iter()
        .filter_map(|id_str| {
            let eid = EntityId(id_str.clone());
            kg.get_entity(&eid).map(|e| {
                let text = format!("{} ({})", e.name, e.entity_type);
                (e.clone(), text)
            })
        })
        .collect();

    let relationships: Vec<(graphrag_core::core::Relationship, String)> = touched_relationship_keys
        .iter()
        .filter_map(|(src, rel_type, tgt)| {
            let src_eid = EntityId(src.clone());
            let tgt_eid = EntityId(tgt.clone());
            // Find the actual relationship instance in the graph
            // (we have its key, not the full record).
            let rel = kg
                .relationships()
                .find(|r| r.source == src_eid && r.target == tgt_eid && r.relation_type == *rel_type)
                .cloned()?;
            let src_name = kg
                .get_entity(&src_eid)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let tgt_name = kg
                .get_entity(&tgt_eid)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let text = format!("{} {} {}", src_name, rel_type, tgt_name);
            Some((rel, text))
        })
        .collect();

    TouchedSnapshot { entities, relationships }
}

/// Embed + persist the touched delta. Lock-free — no reference to
/// `GraphRAG` is needed; the snapshot is owned data. Call this AFTER
/// dropping the GraphRAG write-lock so concurrent recall isn't
/// blocked during embedding (which can take minutes for hundreds of
/// items via OVMS).
///
/// Returns `(entities_persisted, relationships_persisted)`. Empty
/// snapshot is a no-op (returns (0,0)).
#[cfg(feature = "qdrant")]
pub async fn persist_touched_snapshot(
    snapshot: TouchedSnapshot,
    qdrant: &QdrantStore,
    embeddings: &crate::embeddings::EmbeddingService,
) -> Result<(usize, usize), QdrantError> {
    let dim = embeddings.dimension() as u64;

    // 3-tier embedding lookup. Each tier handles cases the previous
    // missed; OVMS only fires for genuinely new entities/relationships.
    //
    //   Tier 1 — in-process text_cache         (µs, free, within-session)
    //   Tier 2 — qdrant fetch_*_vectors        (sub-ms batch RPC, cross-restart)
    //   Tier 3 — OVMS via generate_cached      (~600 ms per request)
    //
    // Tier 1 lives inside `generate_cached`. Tier 2 runs here, BEFORE
    // tier 1, by attaching the qdrant-found vectors directly to
    // Entity.embedding (which short-circuits the tier-1/tier-3 fallthrough
    // entirely) and seeding the text_cache so future within-session
    // requests for the same text bypass qdrant too.

    let mut snapshot_entities = snapshot.entities;
    let mut snapshot_relationships = snapshot.relationships;

    // ---- Tier 2 (entities): batch-fetch vectors for any entity that
    //      arrived without an embedding. On hit, attach to
    //      Entity.embedding AND seed the text cache. On miss, the
    //      entity stays as-is and the existing tier-1/tier-3 path picks
    //      it up below.
    {
        let lookup_ids: Vec<String> = snapshot_entities
            .iter()
            .filter(|(e, _)| e.embedding.is_none())
            .map(|(e, _)| e.id.0.clone())
            .collect();
        if !lookup_ids.is_empty() {
            match qdrant.fetch_entity_vectors(&lookup_ids).await {
                Ok(found) if !found.is_empty() => {
                    let hits = found.len();
                    let total = lookup_ids.len();
                    for (e, t) in snapshot_entities.iter_mut() {
                        if e.embedding.is_none() {
                            if let Some(v) = found.get(&e.id.0) {
                                e.embedding = Some(v.clone());
                                embeddings.seed_cache(t, v).await;
                            }
                        }
                    }
                    tracing::info!(
                        "persist: tier-2 (qdrant) hydrated {}/{} entity vectors",
                        hits, total
                    );
                },
                Ok(_) => {},
                Err(err) => {
                    // Non-fatal — entities without a tier-2 hit just
                    // fall through to tier-3 (re-embed via OVMS).
                    tracing::warn!(
                        "persist: tier-2 entity vector fetch failed ({}); falling back to OVMS",
                        err
                    );
                },
            }
        }
    }

    // ---- Tier 1+3 (entities): existing cache + OVMS path on whatever
    //      remains with embedding=None. ----
    let (texts_to_embed, indices_to_embed): (Vec<String>, Vec<usize>) = snapshot_entities
        .iter()
        .enumerate()
        .filter_map(|(i, (e, t))| {
            if e.embedding.is_some() {
                None
            } else {
                Some((t.clone(), i))
            }
        })
        .unzip();

    let mut entity_embeddings: Vec<Option<Vec<f32>>> =
        snapshot_entities.iter().map(|(e, _)| e.embedding.clone()).collect();

    if !texts_to_embed.is_empty() {
        let refs: Vec<&str> = texts_to_embed.iter().map(String::as_str).collect();
        // Cached path: same-text re-mentions (e.g. recurring novel
        // characters) are served from the in-process text→vector cache
        // and never round-trip OVMS again. The cache key is the embed
        // text itself ("<name> (<type>)"), which is deterministic per
        // entity identity, so cache hits are correct by construction.
        let computed = embeddings
            .generate_cached(&refs)
            .await
            .map_err(|e| QdrantError::OperationError(format!("delta entity embed failed: {}", e)))?;
        for (i, vec) in indices_to_embed.iter().zip(computed) {
            entity_embeddings[*i] = Some(vec);
        }
    }

    let entity_payloads_with_vec: Vec<(PersistedEntity, Vec<f32>)> = snapshot_entities
        .iter()
        .zip(entity_embeddings.iter())
        .map(|((e, _), emb)| {
            let vec = emb.clone().unwrap_or_else(|| vec![0.0_f32; dim as usize]);
            (entity_to_persisted(e), vec)
        })
        .collect();

    // ---- Tier 2 (relationships): same scheme keyed on
    //      (source, relation_type, target). ----
    {
        let lookup_keys: Vec<(String, String, String)> = snapshot_relationships
            .iter()
            .filter(|(r, _)| r.embedding.is_none())
            .map(|(r, _)| (r.source.0.clone(), r.relation_type.clone(), r.target.0.clone()))
            .collect();
        if !lookup_keys.is_empty() {
            match qdrant.fetch_relationship_vectors(&lookup_keys).await {
                Ok(found) if !found.is_empty() => {
                    let hits = found.len();
                    let total = lookup_keys.len();
                    for (r, t) in snapshot_relationships.iter_mut() {
                        if r.embedding.is_none() {
                            let k = (
                                r.source.0.clone(),
                                r.relation_type.clone(),
                                r.target.0.clone(),
                            );
                            if let Some(v) = found.get(&k) {
                                r.embedding = Some(v.clone());
                                embeddings.seed_cache(t, v).await;
                            }
                        }
                    }
                    tracing::info!(
                        "persist: tier-2 (qdrant) hydrated {}/{} relationship vectors",
                        hits, total
                    );
                },
                Ok(_) => {},
                Err(err) => {
                    tracing::warn!(
                        "persist: tier-2 relationship vector fetch failed ({}); falling back to OVMS",
                        err
                    );
                },
            }
        }
    }

    // ---- Tier 1+3 (relationships): cache + OVMS for whatever's still missing. ----
    let (rel_texts_to_embed, rel_indices_to_embed): (Vec<String>, Vec<usize>) =
        snapshot_relationships
            .iter()
            .enumerate()
            .filter_map(|(i, (r, t))| {
                if r.embedding.is_some() {
                    None
                } else {
                    Some((t.clone(), i))
                }
            })
            .unzip();

    let mut rel_embeddings: Vec<Option<Vec<f32>>> = snapshot_relationships
        .iter()
        .map(|(r, _)| r.embedding.clone())
        .collect();

    if !rel_texts_to_embed.is_empty() {
        let refs: Vec<&str> = rel_texts_to_embed.iter().map(String::as_str).collect();
        // Same cache as the entity branch above — relationship embed
        // text is `"<src_name> <relation_type> <tgt_name>"`, also
        // deterministic per relationship identity.
        let computed = embeddings.generate_cached(&refs).await.map_err(|e| {
            QdrantError::OperationError(format!("delta relationship embed failed: {}", e))
        })?;
        for (i, vec) in rel_indices_to_embed.iter().zip(computed) {
            rel_embeddings[*i] = Some(vec);
        }
    }

    let rel_payloads_with_vec: Vec<(PersistedRelationship, Vec<f32>)> = snapshot_relationships
        .iter()
        .zip(rel_embeddings.iter())
        .map(|((r, _), emb)| {
            let vec = emb.clone().unwrap_or_else(|| vec![0.0_f32; dim as usize]);
            (relationship_to_persisted(r), vec)
        })
        .collect();

    qdrant
        .persist_graph_delta(entity_payloads_with_vec, rel_payloads_with_vec, dim)
        .await
}

/// Hydrate the in-memory KnowledgeGraph from Qdrant's persisted entities
/// + relationships. Order matters: entities go in first so each
/// relationship's `add_relationship` call finds its source/target.
///
/// Returns `(entities_restored, relationships_restored,
/// relationships_skipped_orphan)` for telemetry. Orphan-skip happens
/// when a stored relationship references an entity id that isn't
/// in the persisted entity set (e.g. deleted between persist + restore);
/// we log and drop the row rather than fail hydration.
#[cfg(feature = "qdrant")]
pub async fn hydrate_in_memory_graph(
    graphrag: &mut graphrag_core::GraphRAG,
    qdrant: &QdrantStore,
) -> Result<(usize, usize, usize), QdrantError> {
    let entities = qdrant.load_persisted_entities().await?;
    let relationships = qdrant.load_persisted_relationships().await?;

    let Some(kg) = graphrag.knowledge_graph_mut() else {
        return Ok((0, 0, 0));
    };

    let mut entities_restored = 0usize;
    for p in &entities {
        if let Some(entity) = persisted_to_entity(p) {
            // add_entity ignores duplicates if the id is already present?
            // No — it always adds a new node. To stay idempotent across
            // re-hydrations, check first.
            if kg.get_entity(&entity.id).is_none() {
                if kg.add_entity(entity).is_ok() {
                    entities_restored += 1;
                }
            }
        }
    }

    let mut relationships_restored = 0usize;
    let mut relationships_skipped = 0usize;
    for p in &relationships {
        if let Some(rel) = persisted_to_relationship(p) {
            if kg.get_entity(&rel.source).is_none() || kg.get_entity(&rel.target).is_none() {
                relationships_skipped += 1;
                continue;
            }
            if kg.add_relationship(rel).is_ok() {
                relationships_restored += 1;
            }
        }
    }

    Ok((entities_restored, relationships_restored, relationships_skipped))
}
