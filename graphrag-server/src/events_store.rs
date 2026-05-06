//! SQLite-backed event log + per-session lease table for the
//! stale-context awareness layer.
//!
//! Architecture summary (full proposal in graphrag-rs-nix/todo.md):
//!
//! - Every block insert / supersede / remove emits an event with a
//!   monotonic id, the block_id + new etag, and (for updates) the
//!   prior content / unified diff.
//! - Clients hold a per-session lease table — the set of (block_id,
//!   etag) pairs the agent has retrieved.
//! - The SSE endpoint (in main.rs) filters the event stream against
//!   the session's lease table so a session only gets notified about
//!   blocks it actually cares about.
//! - Periodic cleanup deletes events older than retention and
//!   sessions with stale `last_activity`.
//!
//! Storage choice: SQLite via `tokio-rusqlite`. Pure relational
//! workload (ORDER BY id, range scans on ts, transactional appends);
//! qdrant is wrong shape and adds gRPC latency per write. Single-
//! process embedded DB is the right answer for a single-binary
//! server with ~100MB-cap of state.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_rusqlite::{params, Connection};

/// Keys/keepers ─ schema version goes here so future migrations
/// can branch on `PRAGMA user_version`. Bump when adding/changing
/// columns; the v1 → v2 path is application code.
const SCHEMA_VERSION: i32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum EventStoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] tokio_rusqlite::Error),
    #[error("sqlite-direct: {0}")]
    Direct(#[from] rusqlite::Error),
    #[error("path: {0}")]
    Path(#[from] std::io::Error),
}

/// Wire-format event ─ what the SSE endpoint serializes per record.
/// Stored verbatim in `events.body_json` so the SSE layer doesn't
/// have to recompute anything per-frame; it just streams the bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBody {
    pub id: u64,
    pub ts: String,
    #[serde(rename = "blockId")]
    pub block_id: String,
    pub source: String,
    #[serde(rename = "userId")]
    pub user_id: Option<String>,
    #[serde(rename = "changeType")]
    pub change_type: ChangeType,
    #[serde(rename = "oldEtag", skip_serializing_if = "Option::is_none")]
    pub old_etag: Option<String>,
    #[serde(rename = "newEtag", skip_serializing_if = "Option::is_none")]
    pub new_etag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<Delta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChangeType {
    Added,
    Updated,
    Removed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(rename = "oldExcerpt", skip_serializing_if = "Option::is_none")]
    pub old_excerpt: Option<String>,
    #[serde(rename = "newExcerpt", skip_serializing_if = "Option::is_none")]
    pub new_excerpt: Option<String>,
    /// Unified-diff format from `similar::TextDiff`.
    #[serde(rename = "unifiedDiff", skip_serializing_if = "Option::is_none")]
    pub unified_diff: Option<String>,
}

/// Result of a revalidate / lease-check call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RevalidationResult {
    /// (block_id, current_etag) — leased and still matches the supplied etag
    pub current: Vec<(String, String)>,
    /// (block_id, current_etag) — leased but the live etag differs (stale)
    pub stale: Vec<(String, String)>,
    /// block_id — leased but no longer present in the graph
    pub missing: Vec<String>,
}

/// Tunables surfaced via env vars from the home-manager module.
/// All have sensible defaults so the server boots without explicit
/// configuration.
#[derive(Debug, Clone)]
pub struct Settings {
    pub event_retention_days: u32,
    pub session_ttl_days: u32,
    pub max_leases_per_session: usize,
    pub delta_excerpt_chars: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            event_retention_days: 7,
            session_ttl_days: 7,
            max_leases_per_session: 1000,
            delta_excerpt_chars: 500,
        }
    }
}

impl Settings {
    pub fn from_env() -> Self {
        let mut s = Self::default();
        if let Ok(v) = std::env::var("STALE_CONTEXT_EVENT_RETENTION_DAYS") {
            if let Ok(n) = v.parse() { s.event_retention_days = n; }
        }
        if let Ok(v) = std::env::var("STALE_CONTEXT_SESSION_TTL_DAYS") {
            if let Ok(n) = v.parse() { s.session_ttl_days = n; }
        }
        if let Ok(v) = std::env::var("STALE_CONTEXT_MAX_LEASES_PER_SESSION") {
            if let Ok(n) = v.parse() { s.max_leases_per_session = n; }
        }
        if let Ok(v) = std::env::var("STALE_CONTEXT_DELTA_EXCERPT_CHARS") {
            if let Ok(n) = v.parse() { s.delta_excerpt_chars = n; }
        }
        s
    }
}

/// Per-block lease entry — what session X retrieved at time Y.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseEntry {
    pub block_id: String,
    pub etag: String,
    pub retrieved_at: String,
}

#[derive(Clone)]
pub struct EventsStore {
    inner: Arc<Connection>,
    pub settings: Settings,
}

impl EventsStore {
    /// Open or create the events SQLite at `path`. Path's parent dir
    /// is created if missing. Schema is set up on first open via
    /// `CREATE TABLE IF NOT EXISTS` + `PRAGMA user_version`. Idempotent.
    pub async fn open<P: AsRef<Path>>(path: P, settings: Settings) -> Result<Self, EventStoreError> {
        let path: PathBuf = path.as_ref().to_owned();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).await?;
        conn.call(|c| {
            // WAL + NORMAL synchronous: durability for the event log
            // without paying full fsync cost on every append. The
            // event log isn't financial data — losing the last
            // half-second of events on a crash is acceptable.
            c.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=NORMAL;
                 PRAGMA auto_vacuum=INCREMENTAL;
                 PRAGMA foreign_keys=ON;",
            )?;

            let user_version: i32 = c.query_row("PRAGMA user_version", [], |r| r.get(0))?;
            if user_version < SCHEMA_VERSION {
                c.execute_batch(SCHEMA_V1)?;
                c.pragma_update(None, "user_version", SCHEMA_VERSION)?;
            }
            Ok(())
        })
        .await?;
        Ok(Self {
            inner: Arc::new(conn),
            settings,
        })
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Escape hatch for the SSE handler: it needs to issue a one-off
    /// query (per-session lease list with both block_id and etag)
    /// that doesn't fit neatly into the typed API on this struct.
    /// Kept narrow — only `stale_context.rs` uses this.
    pub fn connection(&self) -> &tokio_rusqlite::Connection {
        &self.inner
    }

    /// Append an event. Returns the auto-assigned monotonic id.
    pub async fn append_event(&self, ev: PendingEvent) -> Result<u64, EventStoreError> {
        let id = self
            .inner
            .call(move |c| {
                let body_json = serde_json::to_string(&ev.body_for_persist())
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                let change_type_str = match ev.change_type {
                    ChangeType::Added => "added",
                    ChangeType::Updated => "updated",
                    ChangeType::Removed => "removed",
                };
                c.execute(
                    "INSERT INTO events
                       (ts, block_id, source, user_id, change_type, old_etag, new_etag, body_json)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        ev.ts,
                        ev.block_id,
                        ev.source,
                        ev.user_id,
                        change_type_str,
                        ev.old_etag,
                        ev.new_etag,
                        body_json,
                    ],
                )?;
                let id = c.last_insert_rowid();
                Ok(id as u64)
            })
            .await?;
        Ok(id)
    }

    /// Replay all events with id > `since` whose `block_id` is in the
    /// given lease set. Used by the SSE endpoint on reconnect to
    /// catch the client up.
    pub async fn events_since(
        &self,
        since: u64,
        lease_block_ids: Vec<String>,
        limit: usize,
    ) -> Result<Vec<EventBody>, EventStoreError> {
        if lease_block_ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self
            .inner
            .call(move |c| {
                // SQLite doesn't have native list parameters; build
                // a positional placeholder string. block_ids come
                // from the server's own lease table, never from the
                // outside, so injection is not a concern — but we
                // still bind via params! to keep the path uniform.
                let placeholders: String = (0..lease_block_ids.len())
                    .map(|i| format!("?{}", i + 3))
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "SELECT id, body_json FROM events
                     WHERE id > ?1 AND block_id IN ({})
                     ORDER BY id ASC
                     LIMIT ?2",
                    placeholders
                );
                let mut stmt = c.prepare(&sql)?;
                let mut binds: Vec<rusqlite::types::Value> = Vec::with_capacity(2 + lease_block_ids.len());
                binds.push((since as i64).into());
                binds.push((limit as i64).into());
                for id in &lease_block_ids {
                    binds.push(id.clone().into());
                }
                let rows = stmt
                    .query_map(rusqlite::params_from_iter(binds.iter()), |r| {
                        let id: i64 = r.get(0)?;
                        let body_json: String = r.get(1)?;
                        Ok((id as u64, body_json))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?;
        let mut out: Vec<EventBody> = Vec::with_capacity(rows.len());
        for (id, json) in rows {
            if let Ok(mut ev) = serde_json::from_str::<EventBody>(&json) {
                ev.id = id;
                out.push(ev);
            }
        }
        Ok(out)
    }

    /// The smallest event id still in the log. Clients reconnecting
    /// with `Last-Event-ID < watermark` need a full re-sync.
    pub async fn compaction_watermark(&self) -> Result<u64, EventStoreError> {
        Ok(self
            .inner
            .call(|c| {
                let id: Option<i64> = c
                    .query_row("SELECT MIN(id) FROM events", [], |r| r.get(0))
                    .unwrap_or(None);
                Ok(id.unwrap_or(0) as u64)
            })
            .await?)
    }

    /// Add (or refresh) a lease entry for a session. Bumps
    /// `sessions.last_activity` so the cleanup task knows it's live.
    /// Enforces FIFO eviction at `max_leases_per_session`.
    pub async fn add_leases(
        &self,
        session_id: String,
        entries: Vec<LeaseEntry>,
    ) -> Result<(), EventStoreError> {
        if entries.is_empty() {
            return Ok(());
        }
        let max = self.settings.max_leases_per_session;
        let now = Utc::now().to_rfc3339();
        self.inner
            .call(move |c| {
                let tx = c.transaction()?;
                tx.execute(
                    "INSERT INTO sessions (id, last_activity)
                     VALUES (?1, ?2)
                     ON CONFLICT(id) DO UPDATE SET last_activity = excluded.last_activity",
                    params![session_id, now],
                )?;
                {
                    let mut up = tx.prepare(
                        "INSERT INTO leases (session_id, block_id, etag, retrieved_at)
                         VALUES (?1, ?2, ?3, ?4)
                         ON CONFLICT(session_id, block_id) DO UPDATE SET
                           etag = excluded.etag,
                           retrieved_at = excluded.retrieved_at",
                    )?;
                    for e in &entries {
                        up.execute(params![session_id, e.block_id, e.etag, e.retrieved_at])?;
                    }
                }
                // FIFO eviction: oldest retrieved_at goes first.
                let count: i64 = tx.query_row(
                    "SELECT COUNT(*) FROM leases WHERE session_id = ?1",
                    params![session_id],
                    |r| r.get(0),
                )?;
                if (count as usize) > max {
                    let to_drop = (count as usize) - max;
                    tx.execute(
                        "DELETE FROM leases
                         WHERE session_id = ?1
                           AND rowid IN (
                             SELECT rowid FROM leases
                              WHERE session_id = ?1
                              ORDER BY retrieved_at ASC
                              LIMIT ?2
                           )",
                        params![session_id, to_drop as i64],
                    )?;
                }
                tx.commit()?;
                Ok(())
            })
            .await?;
        Ok(())
    }

    /// Get all block_ids the session currently leases. Used by the
    /// SSE endpoint for filtering and by lease/check.
    pub async fn session_block_ids(
        &self,
        session_id: String,
    ) -> Result<Vec<String>, EventStoreError> {
        Ok(self
            .inner
            .call(move |c| {
                let mut stmt =
                    c.prepare("SELECT block_id FROM leases WHERE session_id = ?1")?;
                let rows = stmt
                    .query_map(params![session_id], |r| r.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?)
    }

    /// Compare the supplied (block_id, etag) tuples against the live
    /// state. `current_etag_lookup` is a closure provided by the
    /// caller (it queries qdrant — kept out of this module so the
    /// store stays storage-agnostic). Returns the staleness verdict.
    pub async fn revalidate(
        &self,
        entries: Vec<(String, String)>,
        current_etags: Vec<(String, Option<String>)>,
    ) -> Result<RevalidationResult, EventStoreError> {
        let mut out = RevalidationResult::default();
        let live: std::collections::HashMap<String, Option<String>> =
            current_etags.into_iter().collect();
        for (block_id, etag) in entries {
            match live.get(&block_id) {
                None => out.missing.push(block_id),
                Some(None) => out.missing.push(block_id),
                Some(Some(live_etag)) => {
                    if live_etag == &etag {
                        out.current.push((block_id, etag));
                    } else {
                        out.stale.push((block_id, live_etag.clone()));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Drop all lease entries for a session (explicit teardown).
    pub async fn drop_session(
        &self,
        session_id: String,
    ) -> Result<usize, EventStoreError> {
        Ok(self
            .inner
            .call(move |c| {
                let tx = c.transaction()?;
                let n = tx.execute(
                    "DELETE FROM leases WHERE session_id = ?1",
                    params![session_id],
                )?;
                tx.execute(
                    "DELETE FROM sessions WHERE id = ?1",
                    params![session_id],
                )?;
                tx.commit()?;
                Ok(n)
            })
            .await?)
    }

    /// Cleanup tick: drop expired events + sessions, checkpoint WAL,
    /// optionally vacuum. Returns counts for the log line.
    pub async fn cleanup(&self) -> Result<CleanupReport, EventStoreError> {
        let event_cutoff: DateTime<Utc> =
            Utc::now() - chrono::Duration::days(self.settings.event_retention_days as i64);
        let session_cutoff: DateTime<Utc> =
            Utc::now() - chrono::Duration::days(self.settings.session_ttl_days as i64);
        let event_cutoff_s = event_cutoff.to_rfc3339();
        let session_cutoff_s = session_cutoff.to_rfc3339();

        let report = self
            .inner
            .call(move |c| {
                let tx = c.transaction()?;
                let events_dropped = tx.execute(
                    "DELETE FROM events WHERE ts < ?1",
                    params![event_cutoff_s],
                )?;
                let leases_dropped = tx.execute(
                    "DELETE FROM leases
                     WHERE session_id IN (
                       SELECT id FROM sessions WHERE last_activity < ?1
                     )",
                    params![session_cutoff_s],
                )?;
                let sessions_dropped = tx.execute(
                    "DELETE FROM sessions WHERE last_activity < ?1",
                    params![session_cutoff_s],
                )?;
                tx.commit()?;

                // Truncate WAL so disk usage actually shrinks after
                // big DELETEs; without this WAL grows unbounded.
                let _ = c.pragma_update(None, "wal_checkpoint", "TRUNCATE");
                let _ = c.pragma_update(None, "incremental_vacuum", 1024i64);

                let event_count: i64 = c
                    .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
                    .unwrap_or(0);
                let session_count: i64 = c
                    .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
                    .unwrap_or(0);
                Ok(CleanupReport {
                    events_dropped: events_dropped as u64,
                    leases_dropped: leases_dropped as u64,
                    sessions_dropped: sessions_dropped as u64,
                    events_remaining: event_count as u64,
                    sessions_remaining: session_count as u64,
                })
            })
            .await?;
        Ok(report)
    }
}

#[derive(Debug, Default, Clone)]
pub struct CleanupReport {
    pub events_dropped: u64,
    pub leases_dropped: u64,
    pub sessions_dropped: u64,
    pub events_remaining: u64,
    pub sessions_remaining: u64,
}

/// Pre-persistence event payload. Caller-side helpers in main.rs
/// build this from the chunk metadata + diff computation, then call
/// `append_event` which assigns the monotonic id and serializes the
/// `EventBody` for SSE replay.
#[derive(Debug, Clone)]
pub struct PendingEvent {
    pub ts: String,
    pub block_id: String,
    pub source: String,
    pub user_id: Option<String>,
    pub change_type: ChangeType,
    pub old_etag: Option<String>,
    pub new_etag: Option<String>,
    pub delta: Option<Delta>,
}

impl PendingEvent {
    /// Serialize `body_json` with id=0; the real id is patched in by
    /// the SSE replay layer (which knows the row's `id` column) and
    /// by `append_event`'s broadcast emission.
    fn body_for_persist(&self) -> EventBody {
        EventBody {
            id: 0,
            ts: self.ts.clone(),
            block_id: self.block_id.clone(),
            source: self.source.clone(),
            user_id: self.user_id.clone(),
            change_type: self.change_type.clone(),
            old_etag: self.old_etag.clone(),
            new_etag: self.new_etag.clone(),
            delta: self.delta.clone(),
        }
    }
}

/// v1 schema. Idempotent (`CREATE TABLE IF NOT EXISTS`).
const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    block_id     TEXT NOT NULL,
    source       TEXT NOT NULL,
    user_id      TEXT,
    change_type  TEXT NOT NULL CHECK (change_type IN ('added','updated','removed')),
    old_etag     TEXT,
    new_etag     TEXT,
    body_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_block_id  ON events(block_id);

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    last_activity TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity);

CREATE TABLE IF NOT EXISTS leases (
    session_id    TEXT NOT NULL,
    block_id      TEXT NOT NULL,
    etag          TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, block_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_leases_block ON leases(block_id);
"#;
