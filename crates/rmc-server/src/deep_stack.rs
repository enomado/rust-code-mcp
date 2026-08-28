//! Running rust-analyzer work on a stack deep enough for it.
//!
//! Work that loads a workspace through rust-analyzer walks HIR/AST trees
//! recursively — the graph tools building a hypergraph, the audits, and the
//! interner sweep inside a garbage collection alike. That recursion is bounded
//! by the *shape of the analyzed source*, not by anything we control, and it
//! does not fit in the 2 MiB stack a tokio blocking-pool thread gets by
//! default: building the hypergraph for this very workspace aborted the
//! process with
//!
//! ```text
//! thread 'tokio-rt-worker' has overflowed its stack
//! fatal runtime error: stack overflow, aborting
//! ```
//!
//! A stack overflow is an `abort`, not a `panic` — it takes the whole MCP
//! server down rather than failing one tool call, so this is not something a
//! caller can guard against. The work therefore carries its own stack instead
//! of depending on whoever spawned it.

use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rmcp::ErrorData as McpError;

use crate::semantic::SemanticService;

/// Who may be inside rust-analyzer at the same time.
///
/// Ordinary analysis takes it shared; a garbage collection takes it
/// exclusively, because its interner sweep is process-global and `unsafe`.
///
/// # The defect this closes
///
/// `hir::collect_ty_garbage` marks live types by refcount, so anything another
/// `AnalysisHost` *stores* survives it. What it cannot see is a type held by a
/// query still in flight — computed, not yet recorded anywhere. The collection
/// was documented as safe because "every path into the semantic service goes
/// through one mutex", and that much is true; what it missed is that the
/// semantic service is not the only rust-analyzer in this process. The graph
/// tools, the five audits and the skeleton builder each load a workspace of
/// their own, outside that mutex, and the watchdog fires a collection on a
/// timer — so a hypergraph build and a sweep could, and did, overlap.
///
/// The symptom is not a wrong answer: it is `SIGSEGV`, observed in a parallel
/// test run of this crate (the same suite is green run single-threaded, and
/// green again on a rerun, which is what a race looks like).
///
/// Making the gate live at this door rather than at the call sites is the point
/// — [`run_analysis`] is already the one place all such work passes through, so
/// a new tool gets the guarantee without knowing it exists.
///
/// # What it costs
///
/// Analyses still run concurrently with each other; what changed is that a
/// collection waits for the ones already inside, and the ones arriving during
/// that wait queue behind it. The collection is short and fires on a timer
/// (`RMC_GC_INTERVAL_SECS`, 300s by default), so the added latency is bounded by
/// one sweep — against a crash that takes the whole daemon with it.
static ANALYSIS_GATE: RwLock<()> = RwLock::new(());

/// Enter the gate shared. A poisoned gate is stepped over rather than
/// propagated: it means some earlier analysis panicked, which says nothing
/// about whether it is safe to run this one, and refusing every analysis
/// afterwards would turn one failed tool call into a dead server.
fn enter_shared() -> RwLockReadGuard<'static, ()> {
    ANALYSIS_GATE.read().unwrap_or_else(|error| error.into_inner())
}

fn enter_exclusive() -> RwLockWriteGuard<'static, ()> {
    ANALYSIS_GATE.write().unwrap_or_else(|error| error.into_inner())
}

/// The same gate, for tests that drive a `SemanticService` directly instead of
/// through [`run_analysis`].
///
/// Those tests hold a loaded analysis for their whole body and the collection
/// tests call `collect_garbage` without going through
/// [`run_exclusive_analysis`] — so neither side takes this gate on its own, and
/// the sweep is free to land inside someone else's analysis. It does not fail an
/// assertion when it does; it takes a SIGSEGV and the whole test binary with it,
/// reported as "the suite crashed" with no clue which pair collided.
///
/// There used to be a second `RwLock` inside `semantic::tests` doing this job.
/// Two locks that must agree and cannot see each other agreed only by luck:
/// the heaviest analysis in the suite lives in `tools::graph::tests`, on the
/// other side of the divide, and the crash arrived the day something shifted the
/// schedule. One gate, the real one, or none.
#[cfg(test)]
pub(crate) fn test_holding_an_analysis() -> RwLockReadGuard<'static, ()> {
    enter_shared()
}

/// "I am the sweep" — see [`test_holding_an_analysis`].
#[cfg(test)]
pub(crate) fn test_sweeping_alone() -> RwLockWriteGuard<'static, ()> {
    enter_exclusive()
}

/// Either kind of gate guard, so one spawn path can hold whichever it took.
enum GateGuard {
    Shared(RwLockReadGuard<'static, ()>),
    Exclusive(RwLockWriteGuard<'static, ()>),
}

/// Stack for the analysis thread.
///
/// Measured, not guessed: the default 2 MiB aborts on this workspace, 32 MiB
/// completes it. 64 MiB doubles the headroom that measurement gives us, and
/// costs only address space — thread stacks are mapped lazily, so the pages a
/// shallower walk never touches are never backed by memory.
const ANALYSIS_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Run blocking rust-analyzer work on a dedicated thread with a deep stack, and
/// await its result.
///
/// Replaces `tokio::task::spawn_blocking` for this kind of work. The blocking
/// pool is otherwise the right tool — the point of the swap is solely the stack
/// size, which the pool does not let us set per task.
pub(crate) async fn run_analysis<T, F>(what: &'static str, work: F) -> Result<T, McpError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    spawn_gated(what, work, false).await
}

/// Run rust-analyzer work that must be the *only* such work in the process.
///
/// The one caller is the garbage collection; see [`ANALYSIS_GATE`] for why it
/// cannot share the process with an analysis in flight.
pub(crate) async fn run_exclusive_analysis<T, F>(what: &'static str, work: F) -> Result<T, McpError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    spawn_gated(what, work, true).await
}

async fn spawn_gated<T, F>(what: &'static str, work: F, exclusive: bool) -> Result<T, McpError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("rmc-analysis".to_string())
        .stack_size(ANALYSIS_STACK_BYTES)
        .spawn(move || {
            // Taken here rather than before the spawn: waiting for the gate is
            // the analysis thread's business, and doing it in the caller would
            // block a runtime worker on exactly what this module exists to keep
            // off one.
            let _guard = if exclusive {
                GateGuard::Exclusive(enter_exclusive())
            } else {
                GateGuard::Shared(enter_shared())
            };
            // A send error means the caller went away; nothing to report to.
            let _ = tx.send(work());
        })
        .map_err(|error| {
            McpError::internal_error(
                format!("{what}: failed to spawn analysis thread: {error}"),
                None,
            )
        })?;

    // The sender is dropped without sending only if the thread panicked, which
    // is the same failure `spawn_blocking` reported as a join error.
    rx.await
        .map_err(|_| McpError::internal_error(format!("{what}: analysis thread panicked"), None))
}

/// Run work against the shared [`SemanticService`] on the analysis thread.
///
/// Every semantic call loads or queries a rust-analyzer workspace, so every one
/// of them belongs on [`run_analysis`]'s stack for the reason in this module's
/// header. This is the single door to the service from an async context so that
/// the choice is made once rather than at each call site: taking the mutex
/// inline in an `async fn` gets the 2 MiB worker stack *and* blocks the runtime
/// for the duration of the analysis, and both are easy to reintroduce by
/// accident when the lock is one `.lock()` away.
///
/// The closure receives the locked service and reports its own failures, so a
/// caller keeps its own error wording rather than inheriting one from here. A
/// poisoned mutex is the only failure this adds.
pub(crate) async fn with_semantic<T, F>(
    semantic: &Arc<Mutex<SemanticService>>,
    what: &'static str,
    work: F,
) -> Result<T, McpError>
where
    F: FnOnce(&mut SemanticService) -> Result<T, McpError> + Send + 'static,
    T: Send + 'static,
{
    let semantic = Arc::clone(semantic);
    run_analysis(what, move || {
        let mut service = semantic.lock().map_err(|error| {
            McpError::internal_error(format!("Failed to acquire lock: {}", error), None)
        })?;
        work(&mut service)
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate, judged by the only thing that matters about it: exclusive work
    /// must not *begin* while shared work is still inside.
    ///
    /// The shared side reports that it entered and then waits to be released;
    /// the exclusive side stamps a flag the moment it starts. Between the two,
    /// the test waits a bounded while for that flag *not* to appear.
    ///
    /// That wait is one-sided, which is what makes it an oracle rather than a
    /// race: with the gate in place the flag can never be set before the
    /// release however long we wait, because a write lock cannot be taken while
    /// a read guard is held; without it, the flag appears as soon as the thread
    /// is scheduled. The bound therefore decides only how quickly a broken
    /// implementation is caught, never whether a correct one passes.
    ///
    /// The first version had no wait at all and passed under mutation — the
    /// exclusive thread simply had not been scheduled yet by the time the test
    /// released the shared side.
    #[tokio::test]
    async fn a_collection_waits_for_the_analysis_already_inside() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;

        static EXCLUSIVE_STARTED: AtomicBool = AtomicBool::new(false);
        EXCLUSIVE_STARTED.store(false, Ordering::SeqCst);

        // Entry is reported over a *tokio* channel and awaited, not received
        // with a blocking `recv`: `#[tokio::test]` gives a single-threaded
        // runtime, and a blocking wait on its thread would keep the very task
        // that reports entry from ever being polled. (Learned the hard way —
        // that version hung.) The release channel stays a blocking one, because
        // the side that waits on it is the analysis thread, which is allowed to
        // block and whose blocking is the point of the test.
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let shared = tokio::spawn(run_analysis("shared", move || {
            entered_tx.send(()).expect("report entry");
            // Held until the test says otherwise: this is the window during
            // which a collection must not run.
            release_rx.recv().expect("wait for release");
            EXCLUSIVE_STARTED.load(Ordering::SeqCst)
        }));

        entered_rx.await.expect("shared work entered the gate");

        let exclusive = tokio::spawn(run_exclusive_analysis("exclusive", || {
            EXCLUSIVE_STARTED.store(true, Ordering::SeqCst);
        }));

        let deadline =
            std::time::Instant::now() + std::time::Duration::from_millis(300);
        while std::time::Instant::now() < deadline && !EXCLUSIVE_STARTED.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        release_tx.send(()).expect("release the shared work");

        let started_before_release = shared.await.expect("shared task").expect("shared analysis");
        exclusive.await.expect("exclusive task").expect("exclusive analysis");

        assert!(
            !started_before_release,
            "the collection ran while an analysis was still inside rust-analyzer — that overlap \
             is the one the interner sweep cannot survive, and it shows up as a SIGSEGV rather \
             than as a failed call"
        );
        assert!(
            EXCLUSIVE_STARTED.load(Ordering::SeqCst),
            "positive control: the exclusive work never ran at all, so the assertion above \
             passed without proving anything"
        );
    }

    #[tokio::test]
    async fn a_result_travels_back_from_the_analysis_thread() {
        let value = run_analysis("test", || 6 * 7)
            .await
            .expect("analysis result");
        assert_eq!(value, 42);
    }

    #[tokio::test]
    async fn recursion_that_overflows_a_default_thread_completes_here() {
        // Positive control for the stack size itself rather than for the
        // plumbing: this frame chain needs far more than the 2 MiB a tokio
        // blocking thread would hand us, so the test fails by aborting if
        // `stack_size` ever stops being applied.
        fn burn(depth: usize, sink: &mut u64) -> u64 {
            // A big live frame, kept from being optimized away by feeding it
            // into the running sum.
            let block = [0xABu8; 8192];
            *sink = sink.wrapping_add(block[depth % block.len()] as u64);
            if depth == 0 {
                *sink
            } else {
                burn(depth - 1, sink)
            }
        }

        let mut sink = 0;
        // ~1000 frames × 8 KiB ≈ 8 MiB of live frames.
        let total = run_analysis("test", move || burn(1000, &mut sink))
            .await
            .expect("deep recursion completes");
        assert!(total > 0, "the recursion must actually have run");
    }

    fn test_semantic() -> Arc<Mutex<SemanticService>> {
        Arc::new(Mutex::new(SemanticService::new()))
    }

    #[tokio::test]
    async fn semantic_work_leaves_the_runtime_worker() {
        // The regression this guards is a call site taking the mutex inline in
        // an `async fn`: that runs rust-analyzer on the worker's 2 MiB stack,
        // which is what aborted the whole server. The thread name is how that
        // choice becomes observable from the closure.
        let semantic = test_semantic();
        let thread_name = with_semantic(&semantic, "test", |_| {
            Ok(std::thread::current().name().map(str::to_string))
        })
        .await
        .expect("semantic work runs");
        assert_eq!(thread_name.as_deref(), Some("rmc-analysis"));
    }

    #[tokio::test]
    async fn semantic_work_gets_the_deep_stack_too() {
        // Positive control end to end: the same frame chain that a worker
        // thread cannot hold must complete through the semantic door.
        fn burn(depth: usize, sink: &mut u64) -> u64 {
            let block = [0xCDu8; 8192];
            *sink = sink.wrapping_add(block[depth % block.len()] as u64);
            if depth == 0 {
                *sink
            } else {
                burn(depth - 1, sink)
            }
        }

        let semantic = test_semantic();
        let total = with_semantic(&semantic, "test", |_| {
            let mut sink = 0;
            Ok(burn(1000, &mut sink))
        })
        .await
        .expect("deep recursion completes");
        assert!(total > 0, "the recursion must actually have run");
    }

    #[tokio::test]
    async fn a_poisoned_semantic_mutex_is_reported_rather_than_panicking() {
        let semantic = test_semantic();
        let poisoner = Arc::clone(&semantic);
        // Poison the mutex the way a panicking analysis would.
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().expect("fresh mutex");
            panic!("poison the semantic mutex");
        })
        .join();

        let outcome: Result<(), McpError> = with_semantic(&semantic, "test", |_| Ok(())).await;
        let error = outcome.expect_err("a poisoned mutex must surface as an error");
        assert!(
            error.message.contains("Failed to acquire lock"),
            "unexpected error message: {}",
            error.message
        );
    }

    #[tokio::test]
    async fn a_panicking_analysis_is_reported_rather_than_hanging() {
        let outcome: Result<(), McpError> =
            run_analysis("test", || panic!("boom in analysis")).await;
        let error = outcome.expect_err("a panic must surface as an error");
        assert!(
            error.message.contains("panicked"),
            "unexpected error message: {}",
            error.message
        );
    }
}
