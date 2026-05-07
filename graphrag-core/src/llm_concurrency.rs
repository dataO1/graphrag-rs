//! Adaptive concurrency control for LLM upstream calls.
//!
//! AIMD (additive-increase / multiplicative-decrease) — start at a
//! configured `initial` permit count, grow by 1 after `success_threshold`
//! consecutive successful calls, halve on transport-level failure.
//!
//! Floors at 1, caps at `max`. Backend-agnostic: the semaphore knows
//! nothing about which model or backend serves the request, only whether
//! the call returned a parsed body or hit a transport error / 429 / 5xx.
//!
//! Used by `chat::ChatClient` to gate every LLM upstream call so that the
//! permit budget shared across all extractions, gleaning, query planner
//! calls etc. self-tunes to whatever the active backend can handle. When
//! the local fallback is active (single-slot llama.cpp), the semaphore
//! shrinks to 1 within seconds of the first timeout. When Spark comes
//! back, it climbs to `max` over the next few minutes of successful calls.

#[cfg(feature = "async")]
use std::sync::Mutex;
#[cfg(feature = "async")]
use std::sync::Arc;
#[cfg(feature = "async")]
use std::time::{Duration, Instant};

#[cfg(feature = "async")]
use tokio::sync::Notify;

/// Static configuration for [`AdaptiveSemaphore`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveConfig {
    /// Initial permit count. Server probe (e.g. llama.cpp `/props`) can
    /// override this before construction; otherwise reflects the static
    /// default.
    pub initial: usize,
    /// Hard cap on permits. The semaphore never grows above this.
    pub max: usize,
    /// Number of consecutive successes before the permit count grows by 1.
    pub success_threshold: usize,
    /// Multiplicative decrease factor on failure. `0.5` halves the
    /// permit count; `0.25` quarters it. Floored at 1 permit.
    pub failure_decay: f32,
    /// Minimum gap between consecutive shrinks. Without a cooldown,
    /// a burst of N simultaneous failures (e.g. 30 in-flight requests
    /// all timing out when the upstream goes down) would halve N times
    /// and over-shrink. With a 500ms cooldown, only the first shrinks;
    /// the rest see they're inside the window and skip.
    pub shrink_cooldown_ms: u64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            initial: 64,
            max: 64,
            success_threshold: 10,
            failure_decay: 0.5,
            shrink_cooldown_ms: 500,
        }
    }
}

impl AdaptiveConfig {
    /// Clamp values into safe ranges. Called by [`AdaptiveSemaphore::new`]
    /// so callers don't have to validate ahead of time.
    pub(crate) fn sanitize(self) -> Self {
        let max = self.max.max(1);
        let initial = self.initial.clamp(1, max);
        let success_threshold = self.success_threshold.max(1);
        let failure_decay = self.failure_decay.clamp(0.05, 0.95);
        Self {
            initial,
            max,
            success_threshold,
            failure_decay,
            shrink_cooldown_ms: self.shrink_cooldown_ms,
        }
    }
}

/// Adaptive semaphore. Construct via [`AdaptiveSemaphore::new`]; clone
/// the returned `Arc<Self>` to share across tasks.
#[cfg(feature = "async")]
#[derive(Debug)]
pub struct AdaptiveSemaphore {
    config: AdaptiveConfig,
    state: Mutex<State>,
    notify: Notify,
}

#[cfg(feature = "async")]
#[derive(Debug)]
struct State {
    /// Current permit cap. Grows on N successes, halves on failure.
    permits_max: usize,
    /// Permits currently held by callers.
    in_flight: usize,
    /// Successes since last grow. Reset on grow or on shrink.
    successes_since_grow: usize,
    /// Last time we shrank — used for the shrink cooldown.
    last_shrink_at: Option<Instant>,
}

#[cfg(feature = "async")]
impl AdaptiveSemaphore {
    /// Build a new adaptive semaphore. Returns an `Arc` for sharing.
    pub fn new(config: AdaptiveConfig) -> Arc<Self> {
        let config = config.sanitize();
        Arc::new(Self {
            config,
            state: Mutex::new(State {
                permits_max: config.initial,
                in_flight: 0,
                successes_since_grow: 0,
                last_shrink_at: None,
            }),
            notify: Notify::new(),
        })
    }

    /// Permit cap right now. Snapshot — may change between calls.
    pub fn current_permits(&self) -> usize {
        self.state.lock().expect("poisoned").permits_max
    }

    /// In-flight count right now. For telemetry only.
    pub fn in_flight(&self) -> usize {
        self.state.lock().expect("poisoned").in_flight
    }

    /// Acquire a permit, waiting until one is available. Returns a
    /// guard that the caller must explicitly mark with
    /// [`AdaptivePermit::record_success`] or
    /// [`AdaptivePermit::record_failure`]. Dropping without marking
    /// releases the permit but does not adjust the cap (treat as
    /// telemetry-skipped — neither success nor failure).
    pub async fn acquire(self: &Arc<Self>) -> AdaptivePermit {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            {
                let mut state = self.state.lock().expect("poisoned");
                if state.in_flight < state.permits_max {
                    state.in_flight += 1;
                    return AdaptivePermit {
                        sem: Arc::clone(self),
                        released: false,
                    };
                }
            }
            notified.await;
        }
    }

    fn release_permit(&self) {
        {
            let mut state = self.state.lock().expect("poisoned");
            state.in_flight = state.in_flight.saturating_sub(1);
        }
        self.notify.notify_one();
    }

    fn record_success_inner(&self) {
        let mut state = self.state.lock().expect("poisoned");
        state.successes_since_grow = state.successes_since_grow.saturating_add(1);
        if state.successes_since_grow >= self.config.success_threshold
            && state.permits_max < self.config.max
        {
            let prev = state.permits_max;
            state.permits_max += 1;
            state.successes_since_grow = 0;
            let new = state.permits_max;
            drop(state);
            #[cfg(feature = "tracing")]
            tracing::info!(
                "llm.concurrency: {} → {} (grew: {} consecutive successes)",
                prev,
                new,
                self.config.success_threshold,
            );
            #[cfg(not(feature = "tracing"))]
            let _ = (prev, new);
            // Wake one waiter to let them re-check now that there's room.
            self.notify.notify_one();
        }
    }

    fn record_failure_inner(&self) {
        let mut state = self.state.lock().expect("poisoned");
        let now = Instant::now();
        if let Some(last) = state.last_shrink_at {
            if now.duration_since(last) < Duration::from_millis(self.config.shrink_cooldown_ms) {
                // Already shrank recently — skip to avoid over-correction
                // when many in-flight requests fail at once.
                return;
            }
        }
        let prev = state.permits_max;
        let scaled = (prev as f32 * self.config.failure_decay).floor() as usize;
        let new = scaled.max(1).min(self.config.max);
        if new < prev {
            state.permits_max = new;
            state.successes_since_grow = 0;
            state.last_shrink_at = Some(now);
            drop(state);
            #[cfg(feature = "tracing")]
            tracing::warn!(
                "llm.concurrency: {} → {} (shrunk: transport failure)",
                prev,
                new,
            );
            #[cfg(not(feature = "tracing"))]
            let _ = (prev, new);
        }
    }
}

/// RAII guard returned by [`AdaptiveSemaphore::acquire`]. Caller MUST
/// call [`record_success`] or [`record_failure`] to feed signals into
/// the AIMD controller; otherwise the permit is released silently
/// without adjusting the cap.
#[cfg(feature = "async")]
#[must_use = "AdaptivePermit must be marked with record_success or record_failure"]
pub struct AdaptivePermit {
    sem: Arc<AdaptiveSemaphore>,
    released: bool,
}

#[cfg(feature = "async")]
impl AdaptivePermit {
    /// Mark the gated operation as successful. Releases the permit and
    /// (if `success_threshold` consecutive successes have accumulated)
    /// grows the cap by 1.
    pub fn record_success(mut self) {
        self.released = true;
        self.sem.release_permit();
        self.sem.record_success_inner();
    }

    /// Mark the gated operation as failed (transport error / timeout /
    /// 429 / 5xx). Releases the permit and halves the cap (subject to
    /// shrink cooldown).
    pub fn record_failure(mut self) {
        self.released = true;
        self.sem.release_permit();
        self.sem.record_failure_inner();
    }
}

#[cfg(feature = "async")]
impl Drop for AdaptivePermit {
    fn drop(&mut self) {
        if !self.released {
            // Caller dropped the permit without marking — release the
            // slot so other callers aren't blocked, but don't move the
            // AIMD state in either direction.
            self.sem.release_permit();
        }
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;

    fn cfg() -> AdaptiveConfig {
        AdaptiveConfig {
            initial: 4,
            max: 8,
            success_threshold: 2,
            failure_decay: 0.5,
            shrink_cooldown_ms: 0, // no cooldown for synchronous tests
        }
    }

    #[tokio::test]
    async fn acquire_within_cap_does_not_block() {
        let sem = AdaptiveSemaphore::new(cfg());
        let p1 = sem.acquire().await;
        let p2 = sem.acquire().await;
        assert_eq!(sem.in_flight(), 2);
        p1.record_success();
        p2.record_success();
        assert_eq!(sem.in_flight(), 0);
    }

    #[tokio::test]
    async fn grows_after_n_consecutive_successes() {
        let sem = AdaptiveSemaphore::new(cfg());
        assert_eq!(sem.current_permits(), 4);
        // 2 successes → +1
        sem.acquire().await.record_success();
        sem.acquire().await.record_success();
        assert_eq!(sem.current_permits(), 5);
        // another 2 → +1
        sem.acquire().await.record_success();
        sem.acquire().await.record_success();
        assert_eq!(sem.current_permits(), 6);
    }

    #[tokio::test]
    async fn caps_at_max() {
        let sem = AdaptiveSemaphore::new(cfg()); // max=8
        for _ in 0..100 {
            sem.acquire().await.record_success();
        }
        assert_eq!(sem.current_permits(), 8);
    }

    #[tokio::test]
    async fn halves_on_failure() {
        let sem = AdaptiveSemaphore::new(cfg());
        // grow to 8 first
        for _ in 0..10 {
            sem.acquire().await.record_success();
        }
        assert_eq!(sem.current_permits(), 8);
        sem.acquire().await.record_failure();
        assert_eq!(sem.current_permits(), 4);
    }

    #[tokio::test]
    async fn floors_at_one() {
        let sem = AdaptiveSemaphore::new(cfg());
        for _ in 0..10 {
            sem.acquire().await.record_failure();
        }
        assert_eq!(sem.current_permits(), 1);
    }

    #[tokio::test]
    async fn cooldown_prevents_over_shrink() {
        let mut c = cfg();
        c.shrink_cooldown_ms = 10_000; // 10 seconds — well past test runtime
        let sem = AdaptiveSemaphore::new(c);
        // grow to 8
        for _ in 0..10 {
            sem.acquire().await.record_success();
        }
        assert_eq!(sem.current_permits(), 8);
        // first failure halves to 4
        sem.acquire().await.record_failure();
        assert_eq!(sem.current_permits(), 4);
        // second failure within cooldown — no shrink
        sem.acquire().await.record_failure();
        assert_eq!(sem.current_permits(), 4);
    }

    #[tokio::test]
    async fn drop_without_marking_releases_permit_but_no_aimd_change() {
        let sem = AdaptiveSemaphore::new(cfg());
        let before = sem.current_permits();
        {
            let _p = sem.acquire().await;
            assert_eq!(sem.in_flight(), 1);
        } // drop without record_*
        assert_eq!(sem.in_flight(), 0);
        assert_eq!(sem.current_permits(), before);
    }

    #[tokio::test]
    async fn waiters_unblock_on_release() {
        let mut c = cfg();
        c.initial = 1;
        c.max = 1;
        c.success_threshold = 1_000_000;
        let sem = AdaptiveSemaphore::new(c);
        let p1 = sem.acquire().await;
        let s2 = Arc::clone(&sem);
        let h = tokio::spawn(async move {
            let p2 = s2.acquire().await;
            p2.record_success();
        });
        // give the spawn a tick to start blocking
        tokio::task::yield_now().await;
        assert_eq!(sem.in_flight(), 1);
        p1.record_success();
        h.await.unwrap();
        assert_eq!(sem.in_flight(), 0);
    }

    #[test]
    fn config_sanitize_clamps() {
        let c = AdaptiveConfig {
            initial: 100,
            max: 10,
            success_threshold: 0,
            failure_decay: 0.0,
            shrink_cooldown_ms: 500,
        }
        .sanitize();
        assert_eq!(c.initial, 10);
        assert_eq!(c.max, 10);
        assert_eq!(c.success_threshold, 1);
        assert!(c.failure_decay >= 0.05);
    }
}
