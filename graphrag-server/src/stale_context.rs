//! HTTP handlers for the stale-context layer:
//!   - POST /api/recall/revalidate — synchronous "are these still
//!     current?" check. Pure SQL + qdrant lookup, no embeddings.
//!   - GET  /api/lease/check       — server-side equivalent that uses
//!     the session's stored lease table; no payload required.
//!   - DELETE /api/lease/{id}      — explicit teardown.
//!   - GET  /api/events/stream     — SSE stream filtered by the
//!     session's lease table; honors `Last-Event-ID` for resume.
//!
//! See `events_store` for the storage layer + types and
//! graphrag-rs-nix/todo.md "Stale-context awareness" for the full
//! design.

use crate::events_store::{self, EventBody, EventsStore, RevalidationResult};
use crate::AppState;
use actix_web::web::{Data, Json, Path, Query};
use actix_web::{HttpRequest, HttpResponse, Responder};
use actix_web_lab::sse::{self, Sse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RevalidateBody {
    pub entries: Vec<RevalidateEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RevalidateEntry {
    pub block_id: String,
    pub etag: String,
}

/// POST /api/recall/revalidate — body: `{ entries: [{blockId, etag}] }`.
/// Returns `{stale, current, missing}`. No session_id needed; the
/// caller knows which entries it's asking about. Use this when the
/// agent has a list of (block_id, etag) tuples it wants to spot-check
/// without going through the lease table.
pub async fn recall_revalidate(
    state: Data<AppState>,
    body: Json<RevalidateBody>,
) -> HttpResponse {
    let store = match state.events_store.as_ref() {
        Some(s) => s,
        None => return HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "error": "stale-context store not initialized"
        })),
    };

    let user_id_blocks = body
        .entries
        .iter()
        .map(|e| e.block_id.clone())
        .collect::<Vec<_>>();
    let live = match lookup_current_etags(state.get_ref(), &user_id_blocks).await {
        Ok(m) => m,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("etag lookup failed: {e}")
            }))
        },
    };

    let entries = body
        .entries
        .iter()
        .map(|e| (e.block_id.clone(), e.etag.clone()))
        .collect();
    match store.revalidate(entries, live).await {
        Ok(r) => HttpResponse::Ok().json(verdict_dto(r)),
        Err(e) => HttpResponse::InternalServerError()
            .json(serde_json::json!({ "error": format!("{e}") })),
    }
}

#[derive(Debug, Deserialize)]
pub struct LeaseCheckQuery {
    pub session_id: String,
}

/// GET /api/lease/check?session_id=X — server-side equivalent of
/// revalidate, but the entry list comes from the session's lease
/// table (server-stored on every recall). Used by clients on
/// session resume to do a single bulk check.
pub async fn lease_check(
    state: Data<AppState>,
    query: Query<LeaseCheckQuery>,
) -> HttpResponse {
    let store = match state.events_store.as_ref() {
        Some(s) => s,
        None => return HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "error": "stale-context store not initialized"
        })),
    };

    // Pull session's lease entries (block_id + etag pairs).
    let leases = match store
        .session_block_ids(query.session_id.clone())
        .await
    {
        Ok(ids) => ids,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("session lookup failed: {e}")
            }))
        },
    };
    if leases.is_empty() {
        return HttpResponse::Ok().json(verdict_dto(RevalidationResult::default()));
    }
    // We need the etag the session originally retrieved per block —
    // pull from leases table via a dedicated method (kept inline in
    // events_store as `session_leases`).
    let session_leases = match get_session_leases(store, &query.session_id).await {
        Ok(v) => v,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("session lease fetch failed: {e}")
            }))
        },
    };
    let block_ids: Vec<String> = session_leases.iter().map(|(b, _)| b.clone()).collect();
    let live = match lookup_current_etags(state.get_ref(), &block_ids).await {
        Ok(m) => m,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("etag lookup failed: {e}")
            }))
        },
    };
    match store.revalidate(session_leases, live).await {
        Ok(r) => HttpResponse::Ok().json(verdict_dto(r)),
        Err(e) => HttpResponse::InternalServerError()
            .json(serde_json::json!({ "error": format!("{e}") })),
    }
}

/// DELETE /api/lease/{session_id} — explicit session teardown.
pub async fn drop_session(state: Data<AppState>, path: Path<String>) -> HttpResponse {
    let session_id = path.into_inner();
    let store = match state.events_store.as_ref() {
        Some(s) => s,
        None => return HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "error": "stale-context store not initialized"
        })),
    };
    match store.drop_session(session_id.clone()).await {
        Ok(n) => HttpResponse::Ok().json(serde_json::json!({
            "sessionId": session_id,
            "leasesDropped": n,
        })),
        Err(e) => HttpResponse::InternalServerError()
            .json(serde_json::json!({ "error": format!("{e}") })),
    }
}

#[derive(Debug, Deserialize)]
pub struct EventsStreamQuery {
    pub session_id: String,
}

/// GET /api/events/stream?session_id=X — SSE stream of stale-context
/// events filtered by the session's lease table.
///
/// Honors the standard WHATWG `Last-Event-ID` request header for
/// resume: server replays missed events with id > last_seen, then
/// streams live. If `last_seen < compaction_watermark` the server
/// emits one `event: cursor-too-old` and closes the stream — the
/// client falls back to a `lease/check` for full re-sync.
pub async fn events_stream(
    req: HttpRequest,
    state: Data<AppState>,
    query: Query<EventsStreamQuery>,
) -> HttpResponse {
    let store = match state.events_store.as_ref() {
        Some(s) => s.clone(),
        None => return HttpResponse::ServiceUnavailable().body("stale-context disabled"),
    };
    let bus = match state.event_bus.as_ref() {
        Some(b) => b.clone(),
        None => return HttpResponse::ServiceUnavailable().body("stale-context disabled"),
    };

    // Cursor parsing: any non-NUL/CR/LF UTF-8 string per WHATWG
    // §9.2.4. We use monotonic integers; treat unparsable as 0
    // (full replay from start, capped by retention).
    let last_event_id: u64 = req
        .headers()
        .get("Last-Event-ID")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let session_id = query.session_id.clone();

    // Subscribe to the live broadcast BEFORE doing the historic
    // replay, so any event appended during replay is queued in the
    // broadcast buffer rather than lost. The live loop dedupes
    // against `cursor` to avoid double-delivering events that
    // straddle the boundary.
    let mut bus_rx = bus.subscribe();

    let (tx, rx) = mpsc::channel::<sse::Event>(64);

    actix_web::rt::spawn(async move {
        if let Err(e) = run_sse_pump(
            store,
            session_id.clone(),
            last_event_id,
            tx.clone(),
            &mut bus_rx,
        )
        .await
        {
            tracing::warn!(error = %e, %session_id, "SSE pump terminated with error");
        }
    });

    Sse::from_infallible_receiver(rx)
        .with_keep_alive(Duration::from_secs(15))
        .with_retry_duration(Duration::from_secs(3))
        .respond_to(&req)
}

async fn run_sse_pump(
    store: Arc<EventsStore>,
    session_id: String,
    last_event_id: u64,
    tx: mpsc::Sender<sse::Event>,
    bus_rx: &mut broadcast::Receiver<EventBody>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 0. cursor-too-old check: if the supplied last_event_id is
    //    older than the smallest event we still hold, replay isn't
    //    possible. Tell the client to re-sync via lease/check.
    let watermark = store.compaction_watermark().await?;
    if last_event_id > 0 && last_event_id < watermark {
        let _ = tx
            .send(
                sse::Data::new(serde_json::json!({
                    "reason": "cursor below compaction watermark",
                    "watermark": watermark,
                }).to_string())
                .event("cursor-too-old")
                .into(),
            )
            .await;
        return Ok(());
    }

    // 1. Resolve the session's lease set once. The SSE filter is
    //    static for the duration of this stream; if the agent
    //    leases new chunks, the next recall server-side adds them
    //    to the lease table and the events emit will broadcast to
    //    THIS session's subscriber automatically (the lease list is
    //    re-read in the live loop too — see the `live_check`).
    let lease_block_ids = store.session_block_ids(session_id.clone()).await?;

    // 2. Replay history (id > last_event_id, capped).
    let mut cursor = last_event_id;
    let history = store
        .events_since(cursor, lease_block_ids.clone(), 1024)
        .await?;
    for ev in history {
        cursor = ev.id;
        if tx
            .send(
                sse::Data::new(
                    serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into()),
                )
                .id(ev.id.to_string())
                .event(event_type_for(&ev.change_type))
                .into(),
            )
            .await
            .is_err()
        {
            return Ok(()); // client disconnected
        }
    }

    // 3. Live tail. Re-read the session lease set every Nth event
    //    so newly-leased blocks (added by other recall calls during
    //    the SSE session) get included without forcing a reconnect.
    let mut tick: u64 = 0;
    let mut current_leases: std::collections::HashSet<String> =
        lease_block_ids.into_iter().collect();
    loop {
        match bus_rx.recv().await {
            Ok(ev) => {
                if ev.id <= cursor {
                    continue;
                }
                tick += 1;
                if tick % 32 == 0 {
                    if let Ok(ids) = store.session_block_ids(session_id.clone()).await {
                        current_leases = ids.into_iter().collect();
                    }
                }
                if !current_leases.contains(&ev.block_id) {
                    continue;
                }
                cursor = ev.id;
                let s = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
                if tx
                    .send(
                        sse::Data::new(s)
                            .id(ev.id.to_string())
                            .event(event_type_for(&ev.change_type))
                            .into(),
                    )
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            },
            Err(broadcast::error::RecvError::Lagged(_)) => {
                // Slow consumer: send one final 'cursor-too-old'-style
                // marker and close. Client reconnects with
                // Last-Event-ID and the SQLite replay catches it up.
                let _ = tx
                    .send(
                        sse::Data::new(
                            serde_json::json!({
                                "reason": "broadcast lagged; reconnect to resync"
                            })
                            .to_string(),
                        )
                        .event("cursor-too-old")
                        .into(),
                    )
                    .await;
                return Ok(());
            },
            Err(broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

fn event_type_for(c: &events_store::ChangeType) -> &'static str {
    match c {
        events_store::ChangeType::Added => "added",
        events_store::ChangeType::Updated => "updated",
        events_store::ChangeType::Removed => "removed",
    }
}

/// JSON-camelCase wrapper for `RevalidationResult`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VerdictDto {
    current: Vec<EtagPair>,
    stale: Vec<EtagPair>,
    missing: Vec<String>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EtagPair {
    block_id: String,
    etag: String,
}
fn verdict_dto(r: RevalidationResult) -> VerdictDto {
    VerdictDto {
        current: r
            .current
            .into_iter()
            .map(|(b, e)| EtagPair { block_id: b, etag: e })
            .collect(),
        stale: r
            .stale
            .into_iter()
            .map(|(b, e)| EtagPair { block_id: b, etag: e })
            .collect(),
        missing: r.missing,
    }
}

/// Pull the (block_id, etag) lease pairs for a session. Lives here
/// rather than in events_store::EventsStore because it's a small DTO
/// concern only this file uses.
async fn get_session_leases(
    store: &EventsStore,
    session_id: &str,
) -> Result<Vec<(String, String)>, events_store::EventStoreError> {
    let session_id = session_id.to_string();
    let conn = store_inner(store);
    Ok(conn
        .call(move |c| {
            let mut stmt = c.prepare(
                "SELECT block_id, etag FROM leases WHERE session_id = ?1",
            )?;
            let rows = stmt
                .query_map(tokio_rusqlite::params![session_id], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
        .await?)
}

// We need direct access to the inner Connection to run a one-off
// query; events_store keeps it private but exposes a thin escape
// hatch via this helper. Defined as a free fn so we can use it
// from this module without making the field public.
fn store_inner(store: &EventsStore) -> &tokio_rusqlite::Connection {
    store.connection()
}

/// Look up the live etag for each of `block_ids` in the qdrant
/// store. Returns one (block_id, Option<etag>) per input — None
/// means the block has been removed (no current chunk).
async fn lookup_current_etags(
    state: &AppState,
    block_ids: &[String],
) -> Result<Vec<(String, Option<String>)>, String> {
    let mut out = Vec::with_capacity(block_ids.len());
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = state.qdrant.as_ref() {
        for bid in block_ids {
            // We need to find by block_id alone (not user_id+block_id)
            // because revalidate doesn't carry user_id. Use a
            // lightweight scroll filtered on (block_id, is_current=true)
            // and pull the block_hash.
            let etag = qdrant
                .find_current_block_global(bid)
                .await
                .map_err(|e| format!("{e}"))?
                .and_then(|md| md.block_hash);
            out.push((bid.clone(), etag));
        }
        return Ok(out);
    }
    // Memory fallback: no etags tracked in the in-memory path.
    for bid in block_ids {
        out.push((bid.clone(), None));
    }
    Ok(out)
}
