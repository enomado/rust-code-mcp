use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rmcp::schemars;
use serde::Serialize;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::deep_stack::try_run_exclusive_analysis;
use crate::semantic::{SemanticService, SemanticServiceStatus, UsesBatch};

/// Threads of the background builder of the tables of uses
/// ([`ServerRuntime::start_uses_builder`]); `0` — no builder, every library
/// reference query sweeps live.
///
/// One thread by default ([`DEFAULT_USES_THREADS`]). A refresh keeps ready
/// what an edit cannot re-resolve (`semantic::surface`): an edit inside a fn
/// body leaves one file to rebuild, a signature edit its crate and the crates
/// depending on it. The full build after a load is ≈ 13 min of a core on
/// ~8000 files (measured on 318 files, ×21); more threads finish sooner but
/// take cores from the queries of every client of the daemon, and the
/// transient memory of a pass grows with the threads.
pub const USES_THREADS_ENV: &str = "RMC_USES_THREADS";

/// [`USES_THREADS_ENV`] when unset; see there.
const DEFAULT_USES_THREADS: usize = 1;

/// Files each builder thread takes per batch: ≈ 100 ms a file, so a batch
/// holds the analysis gate for about a second — the longest a collection
/// that wants in waits for the builder.
const USES_FILES_PER_THREAD: usize = 8;
/// How often an idle builder looks for a project with pending files.
const USES_IDLE_POLL: Duration = Duration::from_secs(2);
/// Wait after every project with pending files was held by somebody.
const USES_BUSY_PAUSE: Duration = Duration::from_secs(1);
/// Wait after a refresh cut a batch, for the refresh to land.
const USES_STALE_PAUSE: Duration = Duration::from_millis(100);
/// A file of the table of uses slower than this is named in the log.
const USES_SLOW_FILE: Duration = Duration::from_secs(10);
/// Longest the builder yields to a collection the gate turned away. The
/// watchdog retries a deferred collection every 15 s (`IDLE_TICK`), so two
/// ticks and change; past it, what keeps the collection out is somebody
/// else's analysis, which the builder's waiting does not shorten.
const USES_COLLECTION_YIELD_MAX: Duration = Duration::from_secs(40);
const USES_COLLECTION_POLL: Duration = Duration::from_millis(250);

/// [`USES_THREADS_ENV`], [`DEFAULT_USES_THREADS`] when unset. A value that
/// is not a number stops the daemon at start rather than quietly running
/// without the builder.
pub fn uses_builder_threads() -> usize {
    match std::env::var(USES_THREADS_ENV) {
        Err(_) => DEFAULT_USES_THREADS,
        Ok(value) => value.trim().parse().unwrap_or_else(|_| {
            panic!("{USES_THREADS_ENV}={value:?} is not a number of threads")
        }),
    }
}

use super::memory::{MemoryRelease, release_and_measure, rss_kib};
use super::{
    SearchRuntimeCache, SearchRuntimeCacheStatus, SyncManager, SyncManagerStatus,
    WorkspaceLockRegistry,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, Serialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeClearScope {
    All,
    Workspace,
    SemanticOnly,
    SearchCacheOnly,
    SyncTrackingOnly,
}

impl Default for RuntimeClearScope {
    fn default() -> Self {
        Self::All
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeClearRequest {
    pub scope: RuntimeClearScope,
    pub workspace: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeClearReport {
    pub scope: RuntimeClearScope,
    pub workspace: Option<String>,
    pub search_cache_entries_cleared: usize,
    pub semantic_projects_cleared: usize,
    pub sync_directories_untracked: usize,
    /// What clearing actually did to the process's memory.
    ///
    /// Without this the report counted objects it had dropped and said nothing
    /// about the only quantity the operator came for. A run that clears one
    /// project and returns zero bytes — the normal case, see
    /// [`super::memory`] — has to be visible in the answer, not discovered
    /// later with `ps`.
    pub memory: MemoryRelease,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeSyncStatus {
    pub enabled: bool,
    pub tracked_count: usize,
    pub tracked_directories: Vec<String>,
}

impl RuntimeSyncStatus {
    fn disabled() -> Self {
        Self {
            enabled: false,
            tracked_count: 0,
            tracked_directories: Vec::new(),
        }
    }

    fn from_sync_manager(status: SyncManagerStatus) -> Self {
        Self {
            enabled: true,
            tracked_count: status.tracked_count,
            tracked_directories: status.tracked_directories,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackgroundSyncStatus {
    pub enabled: bool,
    pub running: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessStatus {
    pub pid: u32,
    pub rss_kib: Option<u64>,
}

/// The BM25 index of one sync-tracked directory, as the probe reads it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Bm25IndexStatus {
    pub directory: String,
    /// The probe's report, or why it could not read the index (no index yet
    /// is an `Err` too — the message says so).
    pub stats: Result<rmc_indexing::indexing::index_stats::Bm25IndexStats, String>,
}

/// What the keyword indexes of this daemon look like and cost right now.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Bm25Status {
    /// Threads of this process named `merge_thread_*` — tantivy's segment
    /// merge pools, one per live index writer (up to four threads each; the
    /// pool starts them as merges arrive, so an idle writer shows one). Several
    /// `merge_thread_0` at once = several writers alive. The only reading of merge
    /// activity that does not depend on a writer telling the truth: a writer
    /// dropped with merges in flight (the bug that kept three pools busy on a
    /// daemon for hours) leaves its threads here, and nowhere else.
    pub merge_threads: usize,
    pub indexes: Vec<Bm25IndexStatus>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuntimeStatus {
    pub sync: RuntimeSyncStatus,
    pub search_cache: SearchRuntimeCacheStatus,
    pub semantic: SemanticServiceStatus,
    pub background_sync: BackgroundSyncStatus,
    pub process: ProcessStatus,
    /// GPU embedding sessions by owner and the VRAM they were charged,
    /// against this process's and the card's meters.
    pub gpu: rmc_engine::embeddings::GpuLedgerReport,
    pub bm25: Bm25Status,
}

#[derive(Clone)]
pub struct RuntimeState {
    sync_manager: Option<Arc<SyncManager>>,
    workspace_locks: WorkspaceLockRegistry,
    search_cache: SearchRuntimeCache,
    semantic: Arc<SemanticService>,
    background_sync_enabled: Arc<AtomicBool>,
    background_sync_running: Arc<AtomicBool>,
}

impl RuntimeState {
    pub fn standalone() -> Self {
        Self {
            sync_manager: None,
            workspace_locks: WorkspaceLockRegistry::new(),
            search_cache: SearchRuntimeCache::new(),
            semantic: Arc::new(SemanticService::new()),
            background_sync_enabled: Arc::new(AtomicBool::new(false)),
            background_sync_running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_sync_manager(sync_manager: Arc<SyncManager>) -> Self {
        let workspace_locks = sync_manager.workspace_locks();
        Self {
            sync_manager: Some(sync_manager),
            workspace_locks,
            search_cache: SearchRuntimeCache::new(),
            semantic: Arc::new(SemanticService::new()),
            background_sync_enabled: Arc::new(AtomicBool::new(false)),
            background_sync_running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn sync_manager(&self) -> Option<&Arc<SyncManager>> {
        self.sync_manager.as_ref()
    }

    pub fn background_sync_manager(&self) -> Option<&Arc<SyncManager>> {
        if self.background_sync_enabled() {
            self.sync_manager.as_ref()
        } else {
            None
        }
    }

    pub fn workspace_locks(&self) -> &WorkspaceLockRegistry {
        &self.workspace_locks
    }

    pub fn search_cache(&self) -> &SearchRuntimeCache {
        &self.search_cache
    }

    pub(crate) fn semantic(&self) -> Arc<SemanticService> {
        Arc::clone(&self.semantic)
    }

    pub fn background_sync_enabled(&self) -> bool {
        self.background_sync_enabled.load(Ordering::SeqCst)
    }

    pub fn background_sync_running(&self) -> bool {
        self.background_sync_running.load(Ordering::SeqCst)
    }

    pub async fn status(&self) -> RuntimeStatus {
        let bm25 = match &self.sync_manager {
            Some(sync_manager) => {
                bm25_status(&sync_manager.get_tracked_directories().await)
            }
            None => bm25_status(&[]),
        };
        let sync = match &self.sync_manager {
            Some(sync_manager) => RuntimeSyncStatus::from_sync_manager(sync_manager.status().await),
            None => RuntimeSyncStatus::disabled(),
        };
        // Registry only: answers while any project is loading.
        let semantic = self.semantic.status();
        RuntimeStatus {
            sync,
            search_cache: self.search_cache.status(),
            semantic,
            background_sync: BackgroundSyncStatus {
                enabled: self.background_sync_enabled(),
                running: self.background_sync_running(),
            },
            process: current_process_status(),
            gpu: rmc_engine::embeddings::gpu_ledger_report(),
            bm25,
        }
    }

    /// Bump the revision of every loaded analysis and sweep the type interner.
    ///
    /// The reason this exists at all is in [`SemanticService::collect_garbage`]:
    /// salsa only evicts LRU-capped memos when the revision bumps, and a project
    /// nobody edits never bumps on its own. Called on a timer by the daemon's
    /// watchdog; returns the number of contexts collected, for its log line.
    ///
    /// # Two things about *how* it is called
    ///
    /// It runs on the analysis thread, not on a runtime worker: the interner
    /// sweep recurses over the shape of the analyzed types, and a 2 MiB tokio
    /// stack is what took the whole server down with an `abort` the last time
    /// rust-analyzer work was run on one.
    ///
    /// It takes that thread *exclusively*
    /// ([`crate::deep_stack::try_run_exclusive_analysis`]): the semantic
    /// service is not the only rust-analyzer in the process — the graph tools,
    /// the audits and the skeleton builder each load a workspace of their own,
    /// and this method runs on a timer that knows nothing about them. The gate
    /// is what makes "no query is in flight" true rather than assumed; without
    /// it the overlap is a `SIGSEGV`, which is how it was found.
    ///
    /// And only if it can have the gate *now*: `None` means analyses were
    /// inside and the sweep was skipped — the caller tries again on its next
    /// tick. Waiting instead would park every analysis that arrives behind the
    /// waiting sweep, on every project, for as long as the longest one inside
    /// runs (a load: minutes).
    ///
    /// It does *not* take the workspace locks that [`Self::clear`] takes. Those
    /// guard the search index and the sync manager, neither of which this
    /// touches; taking them on a timer would stall indexing for no reason.
    pub async fn collect_garbage(&self) -> Option<usize> {
        let semantic = Arc::clone(&self.semantic);
        match try_run_exclusive_analysis("collect_garbage", move || semantic.collect_garbage())
            .await
        {
            Ok(collected) => collected,
            Err(error) => {
                tracing::warn!("garbage collection did not run: {}", error.message);
                Some(0)
            }
        }
    }

    /// Stop loading rust-analyzer in this process, for good: the daemon retired.
    ///
    /// Every later semantic, graph-building, audit or skeleton call is refused
    /// with a message telling the session how to reach the successor; garbage
    /// collection and everything that loads no analysis keep working. Call it
    /// BEFORE unloading, so no load can slip in between the unload and the flag.
    ///
    /// ⚠ Process-wide rather than scoped to this `RuntimeState`, and the method
    /// takes `&self` only so a caller that holds the runtime finds it here —
    /// see `deep_stack::RETIRED` for why the flag cannot live in the state.
    pub fn refuse_new_analyses(&self) {
        crate::deep_stack::refuse_new_analyses();
    }

    pub async fn clear(&self, request: RuntimeClearRequest) -> RuntimeClearReport {
        let workspace = request.workspace.as_deref().map(normalize_workspace);
        if let Some(workspace) = &workspace {
            // No workspace lock: the search cache, the rust-analyzer projects and
            // the tracked directories each sit behind a lock of their own, and a
            // reader in flight keeps what it already holds until it answers. The
            // write side this took before made the clear wait for the longest
            // `search` of the workspace and parked every new one behind it.
            self.clear_locked(request.scope, Some(workspace)).await
        } else {
            let _workspace_lock = self.workspace_locks.lock_all().await;
            self.clear_locked(request.scope, None).await
        }
    }

    async fn clear_locked(
        &self,
        scope: RuntimeClearScope,
        workspace: Option<&Path>,
    ) -> RuntimeClearReport {
        let mut report = RuntimeClearReport {
            scope,
            workspace: workspace.map(|path| path.display().to_string()),
            search_cache_entries_cleared: 0,
            semantic_projects_cleared: 0,
            sync_directories_untracked: 0,
            memory: MemoryRelease::skipped(),
        };

        match (scope, workspace) {
            (RuntimeClearScope::All, Some(workspace))
            | (RuntimeClearScope::Workspace, Some(workspace)) => {
                report.search_cache_entries_cleared =
                    self.search_cache.invalidate_workspace(workspace);
                report.semantic_projects_cleared = self.semantic.clear_project(workspace);
                report.sync_directories_untracked = self.untrack_workspace(workspace).await;
            }
            (RuntimeClearScope::All, None) => {
                report.search_cache_entries_cleared = self.search_cache.invalidate_all();
                report.semantic_projects_cleared = self.semantic.clear_all();
                report.sync_directories_untracked = self.untrack_all().await;
            }
            (RuntimeClearScope::Workspace, None) => {}
            (RuntimeClearScope::SemanticOnly, Some(workspace)) => {
                report.semantic_projects_cleared = self.semantic.clear_project(workspace);
            }
            (RuntimeClearScope::SemanticOnly, None) => {
                report.semantic_projects_cleared = self.semantic.clear_all();
            }
            (RuntimeClearScope::SearchCacheOnly, Some(workspace)) => {
                report.search_cache_entries_cleared =
                    self.search_cache.invalidate_workspace(workspace);
            }
            (RuntimeClearScope::SearchCacheOnly, None) => {
                report.search_cache_entries_cleared = self.search_cache.invalidate_all();
            }
            (RuntimeClearScope::SyncTrackingOnly, Some(workspace)) => {
                report.sync_directories_untracked = self.untrack_workspace(workspace).await;
            }
            (RuntimeClearScope::SyncTrackingOnly, None) => {
                report.sync_directories_untracked = self.untrack_all().await;
            }
        }

        // After the drops, not before: the allocator can only hand back what the
        // clear has already released. Called unconditionally, including for the
        // no-op scopes, so the report always carries a real measurement rather
        // than sometimes a real one and sometimes a placeholder.
        report.memory = release_and_measure();

        report
    }

    async fn untrack_workspace(&self, workspace: &Path) -> usize {
        match &self.sync_manager {
            Some(sync_manager) if sync_manager.untrack_directory(workspace).await => 1,
            _ => 0,
        }
    }

    async fn untrack_all(&self) -> usize {
        match &self.sync_manager {
            Some(sync_manager) => sync_manager.untrack_all_directories().await,
            None => 0,
        }
    }
}

pub struct ServerRuntime {
    state: RuntimeState,
    shutdown_tx: watch::Sender<bool>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl ServerRuntime {
    pub fn new(sync_interval_secs: u64) -> Self {
        Self::with_sync_manager(Arc::new(SyncManager::with_defaults(sync_interval_secs)))
    }

    pub fn with_sync_manager(sync_manager: Arc<SyncManager>) -> Self {
        let (shutdown_tx, _) = watch::channel(false);
        Self {
            state: RuntimeState::with_sync_manager(sync_manager),
            shutdown_tx,
            tasks: Mutex::new(Vec::new()),
        }
    }

    pub fn state(&self) -> RuntimeState {
        self.state.clone()
    }

    pub fn start_background_sync(&self) {
        let Some(sync_manager) = self.state.sync_manager.clone() else {
            return;
        };
        if self
            .state
            .background_sync_enabled
            .swap(true, Ordering::SeqCst)
        {
            return;
        }

        self.state
            .background_sync_running
            .store(true, Ordering::SeqCst);

        let shutdown = self.shutdown_tx.subscribe();
        let running = Arc::clone(&self.state.background_sync_running);
        let handle = tokio::spawn(async move {
            sync_manager.run_until_shutdown(shutdown).await;
            running.store(false, Ordering::SeqCst);
        });

        self.tasks
            .lock()
            .expect("runtime task mutex poisoned")
            .push(handle);
    }

    /// Start the background builder of the tables of uses, if
    /// [`USES_THREADS_ENV`] asks for one (see [`uses_builder_threads`]).
    ///
    /// A loop of batches ([`SemanticService::build_uses_batch`]), each an
    /// analysis of its own through the usual door, so the gate is let go
    /// between two of them; before each it yields to a collection the gate
    /// turned away ([`crate::deep_stack::CollectionWanted`]) — without that a
    /// build of a large workspace, minutes of batch after batch, would keep
    /// every garbage collection out for as long as it runs. Stops on
    /// [`Self::request_shutdown`] and once the process refuses analyses.
    ///
    /// The same loop, before each batch, refreshes the loaded projects in the
    /// background and primes their symbol index
    /// ([`SemanticService::refresh_in_background`]), so the first query after
    /// other sessions' edits does not pay rust-analyzer's recomputation of them
    /// (plan item `post-refresh-cost`). With [`USES_THREADS_ENV`]`=0` the loop
    /// runs for that alone.
    ///
    /// Cost: the threads it is given, for as long as a loaded project has
    /// pending files; nothing (a registry look and a `read` of each loaded
    /// project's inotify fd every [`USES_IDLE_POLL`]) once none has.
    pub fn start_uses_builder(&self) {
        self.start_uses_builder_on(uses_builder_threads());
    }

    /// [`Self::start_uses_builder`] with the thread count given rather than
    /// read from the environment.
    fn start_uses_builder_on(&self, threads: usize) {
        if threads == 0 {
            tracing::info!(
                "table of uses: no background builder ({USES_THREADS_ENV}=0); library references \
                 are swept live; the loop refreshes projects in the background only"
            );
        } else {
            tracing::info!("table of uses: background builder on {threads} thread(s)");
        }
        let semantic = self.state.semantic();
        let mut shutdown = self.shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
            loop {
                if *shutdown.borrow() || crate::deep_stack::analyses_refused() {
                    break;
                }
                // A shutdown does not wait out the yield (up to 40 s).
                tokio::select! {
                    _ = crate::deep_stack::collection_wanted()
                        .yield_to(USES_COLLECTION_YIELD_MAX, USES_COLLECTION_POLL) => {}
                    // `Err`: the runtime is gone, nobody can stop this any more.
                    changed = shutdown.changed() => if changed.is_err() { break } else { continue },
                }
                let outcome = crate::deep_stack::with_semantic(&semantic, "uses builder", move |service| {
                    // Before the batch: a batch on the old revision would only
                    // be cancelled by the next query's refresh.
                    for done in service.refresh_in_background() {
                        tracing::debug!(
                            "background refresh of {}: primed {} in {:?}",
                            done.project.display(),
                            done.primed,
                            done.took
                        );
                    }
                    if threads == 0 {
                        return Ok(UsesBatch::Idle);
                    }
                    Ok(service.build_uses_batch(threads, USES_FILES_PER_THREAD))
                })
                .await;
                let pause = match outcome {
                    Ok(UsesBatch::Built {
                        project,
                        files,
                        stored,
                        refused,
                        interrupted,
                        slowest,
                        completed,
                        size: (ready, pending, rows),
                    }) => {
                        tracing::debug!(
                            "table of uses of {}: a batch of {files} file(s), {stored} stored, \
                             {refused} refused; {ready} ready, {pending} pending",
                            project.display()
                        );
                        // A file cut again and again is heavier than the time
                        // between two refreshes: worth a line, now and then.
                        for (path, cut) in &interrupted {
                            if [3, 10, 30, 100, 300].contains(cut) {
                                tracing::info!(
                                    "table of uses of {}: {path} cut by a refresh {cut} times; \
                                     {ready} ready, {pending} pending",
                                    project.display()
                                );
                            }
                        }
                        if let Some((path, took)) = &slowest
                            && *took >= USES_SLOW_FILE
                        {
                            tracing::info!(
                                "table of uses of {}: {path} took {:.1} s",
                                project.display(),
                                took.as_secs_f64()
                            );
                        }
                        if let Some(took) = completed {
                            tracing::info!(
                                "use table of {}: {ready} files, {rows} rows in {:.1} s",
                                project.display(),
                                took.as_secs_f64()
                            );
                        }
                        // Cut by a refresh, which is now waiting for the
                        // snapshots to go: let it land before the next batch.
                        if interrupted.is_empty() { Duration::ZERO } else { USES_STALE_PAUSE }
                    }
                    Ok(UsesBatch::Busy) => USES_BUSY_PAUSE,
                    Ok(UsesBatch::Idle) => USES_IDLE_POLL,
                    Err(error) => {
                        if crate::deep_stack::analyses_refused() {
                            break;
                        }
                        tracing::warn!("table of uses: a batch failed: {}", error.message);
                        USES_IDLE_POLL
                    }
                };
                if !pause.is_zero() {
                    tokio::select! {
                        _ = tokio::time::sleep(pause) => {}
                        changed = shutdown.changed() => if changed.is_err() { break },
                    }
                }
            }
        });
        self.tasks
            .lock()
            .expect("runtime task mutex poisoned")
            .push(handle);
    }

    /// Stop the background tasks — the background sync and the builder of the
    /// tables of uses. A retiring daemon calls this long before it exits, so a
    /// task added to this channel later must be one a draining daemon can do
    /// without.
    pub fn request_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    pub async fn wait_for_tasks(&self, timeout: Duration) -> RuntimeTaskReport {
        let mut handles = {
            let mut tasks = self.tasks.lock().expect("runtime task mutex poisoned");
            tasks.drain(..).collect::<Vec<_>>()
        };

        let mut report = RuntimeTaskReport {
            tasks_total: handles.len(),
            tasks_completed: 0,
            tasks_aborted: 0,
            join_errors: 0,
            timed_out: false,
        };

        let deadline = tokio::time::Instant::now() + timeout;
        while let Some(mut handle) = handles.pop() {
            if tokio::time::Instant::now() >= deadline {
                handle.abort();
                handles.push(handle);
                report.timed_out = true;
                break;
            }

            match tokio::time::timeout_at(deadline, &mut handle).await {
                Ok(Ok(())) => report.tasks_completed += 1,
                Ok(Err(_)) => report.join_errors += 1,
                Err(_) => {
                    handle.abort();
                    handles.push(handle);
                    report.timed_out = true;
                    break;
                }
            }
        }

        if report.timed_out {
            for handle in handles {
                if !handle.is_finished() {
                    handle.abort();
                    report.tasks_aborted += 1;
                }
            }
            self.state
                .background_sync_running
                .store(false, Ordering::SeqCst);
        }

        report
    }

    pub async fn shutdown_gracefully(&self, timeout: Duration) -> RuntimeTaskReport {
        self.request_shutdown();
        let report = self.wait_for_tasks(timeout).await;
        let _ = self
            .state
            .clear(RuntimeClearRequest {
                scope: RuntimeClearScope::All,
                workspace: None,
            })
            .await;
        report
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeTaskReport {
    pub tasks_total: usize,
    pub tasks_completed: usize,
    pub tasks_aborted: usize,
    pub join_errors: usize,
    pub timed_out: bool,
}

fn normalize_workspace(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The BM25 index of every directory background sync tracks, plus the merge
/// threads alive in this process. Reads `meta.json` and file sizes only (see
/// `index_stats`), so it is safe while the sync writes.
fn bm25_status(tracked: &[PathBuf]) -> Bm25Status {
    let indexes = tracked
        .iter()
        .map(|dir| {
            // The BM25 directory belongs to a collection: probe the one
            // search and background sync use for the automatic profile.
            let stats = super::project_paths::ProjectPaths::in_use(
                dir,
                &super::defaults::automatic_embedding_backend(),
            )
            .and_then(|paths| {
                rmc_indexing::indexing::index_stats::probe_bm25_index(&paths.tantivy_path)
                    .map_err(|e| format!("{e:#}"))
            });
            Bm25IndexStatus {
                directory: dir.display().to_string(),
                stats,
            }
        })
        .collect();
    Bm25Status {
        merge_threads: merge_thread_count(),
        indexes,
    }
}

/// Threads of this process whose name starts with `merge_thread` (Linux:
/// `/proc/self/task/*/comm`; elsewhere there is nothing to read and the
/// count is 0).
fn merge_thread_count() -> usize {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .filter_map(Result::ok)
        .filter(|task| {
            std::fs::read_to_string(task.path().join("comm"))
                .is_ok_and(|name| name.starts_with("merge_thread"))
        })
        .count()
}

fn current_process_status() -> ProcessStatus {
    ProcessStatus {
        pid: std::process::id(),
        rss_kib: rss_kib(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The merge-thread reading sees a live writer's pool. Other tests may
    /// hold writers of their own at the same time, so the check is on this
    /// writer's contribution, not on an absolute.
    #[test]
    #[cfg(target_os = "linux")]
    fn merge_thread_count_sees_a_live_writer() {
        let dir = tempfile::tempdir().unwrap();
        let index = tantivy::Index::create_in_dir(
            dir.path(),
            rmc_engine::schema::ChunkSchema::new().schema(),
        )
        .unwrap();
        let before = merge_thread_count();
        let writer: tantivy::IndexWriter = index.writer(15_000_000).unwrap();
        let with_writer = merge_thread_count();
        writer.wait_merging_threads().unwrap();
        assert!(with_writer > before, "{before} -> {with_writer}");
    }

    #[tokio::test]
    async fn runtime_status_reports_owned_empty_state() {
        let runtime = ServerRuntime::new(3600);
        let status = runtime.state().status().await;

        assert!(status.sync.enabled);
        assert_eq!(status.sync.tracked_count, 0);
        assert_eq!(status.search_cache.entry_count, 0);
        assert_eq!(status.semantic.project_count, 0);
        assert!(!status.background_sync.enabled);
        assert!(!status.background_sync.running);
        assert_eq!(status.process.pid, std::process::id());
    }

    #[tokio::test]
    async fn runtime_clear_workspace_clears_semantic_and_sync_tracking() {
        let runtime = ServerRuntime::new(3600);
        let state = runtime.state();
        let workspace = tempfile::tempdir().expect("create temp workspace");

        state
            .sync_manager()
            .expect("runtime has sync manager")
            .track_directory(workspace.path().join("."))
            .await;
        state
            .semantic()
            .insert_test_project_fast(workspace.path().join("."));

        let report = state
            .clear(RuntimeClearRequest {
                scope: RuntimeClearScope::Workspace,
                workspace: Some(workspace.path().to_path_buf()),
            })
            .await;

        assert_eq!(report.semantic_projects_cleared, 1);
        assert_eq!(report.sync_directories_untracked, 1);

        let status = state.status().await;
        assert_eq!(status.semantic.project_count, 0);
        assert_eq!(status.sync.tracked_count, 0);
    }

    /// `clear_runtime` of one workspace clears what has locks of its own and
    /// does not wait for the workspace's readers — before 2026-10-31 it took the
    /// operation lock for write, waited for the longest `search` in flight and
    /// parked every new one behind it. Mutant: the write lock back → timeout.
    #[tokio::test]
    async fn runtime_clear_of_a_workspace_does_not_wait_for_its_readers() {
        let runtime = ServerRuntime::new(3600);
        let state = runtime.state();
        let workspace = tempfile::tempdir().expect("create temp workspace");

        let _reader = state.workspace_locks().lock_shared(workspace.path()).await;
        let report = tokio::time::timeout(
            Duration::from_secs(5),
            state.clear(RuntimeClearRequest {
                scope: RuntimeClearScope::Workspace,
                workspace: Some(workspace.path().to_path_buf()),
            }),
        )
        .await
        .expect("clear_runtime of a workspace waited for a reader of it");
        assert_eq!(report.scope, RuntimeClearScope::Workspace);
    }

    #[tokio::test]
    async fn runtime_clear_all_waits_for_workspace_operations() {
        let runtime = ServerRuntime::new(3600);
        let state = runtime.state();
        let workspace = tempfile::tempdir().expect("create temp workspace");

        let guard = state
            .workspace_locks()
            .lock_index_writer(workspace.path())
            .await;
        let clear_state = state.clone();
        let clear_task = tokio::spawn(async move {
            clear_state
                .clear(RuntimeClearRequest {
                    scope: RuntimeClearScope::All,
                    workspace: None,
                })
                .await
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !clear_task.is_finished(),
            "global runtime clear should wait for in-flight workspace operations"
        );

        drop(guard);
        let report = tokio::time::timeout(Duration::from_secs(1), clear_task)
            .await
            .expect("runtime clear should complete after workspace operation finishes")
            .expect("clear task should not panic");
        assert_eq!(report.scope, RuntimeClearScope::All);
    }

    #[tokio::test]
    async fn runtime_background_sync_shutdown_stops_task() {
        let runtime = ServerRuntime::new(3600);
        runtime.start_background_sync();

        let running_status = runtime.state().status().await;
        assert!(running_status.background_sync.enabled);
        assert!(running_status.background_sync.running);

        let report = runtime.shutdown_gracefully(Duration::from_secs(1)).await;

        assert_eq!(report.tasks_total, 1);
        assert_eq!(report.tasks_completed, 1);
        assert!(!report.timed_out);

        let stopped_status = runtime.state().status().await;
        assert!(!stopped_status.background_sync.running);
    }

    /// The builder loop end to end: after a load the table of uses fills in
    /// the background — batch after batch through the analysis door — until a
    /// library query asks rust-analyzer nothing; a shutdown then ends the loop.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_uses_builder_fills_the_table_after_a_load_and_stops_on_shutdown() {
        // The queries below drive the service directly, not through the door.
        let _analysis = crate::deep_stack::test_holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create temp workspace");
        let root = workspace.path().to_path_buf();
        let files = 6;
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"spawning\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(root.join("src")).unwrap();
        let mods: String = (0..files).map(|i| format!("mod m{i};\n")).collect();
        std::fs::write(root.join("src/lib.rs"), mods).unwrap();
        for i in 0..files {
            std::fs::write(
                root.join(format!("src/m{i}.rs")),
                "pub fn f() {\n    std::thread::spawn(|| {}).join().unwrap();\n}\n",
            )
            .unwrap();
        }

        let runtime = ServerRuntime::new(3600);
        let semantic = runtime.state().semantic();
        let ask = {
            let root = root.clone();
            move || {
                let semantic = Arc::clone(&semantic);
                let root = root.clone();
                tokio::task::spawn_blocking(move || {
                    semantic
                        .find_references_within(&root, "std::thread::spawn", true, Duration::from_secs(3600))
                        .expect("references query")
                })
            }
        };
        // Loads the project: every file pending, swept live.
        let first = ask().await.unwrap();
        assert!(first.cost.ra_queries > 0, "{first:?}");

        runtime.start_uses_builder_on(2);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
        loop {
            let found = ask().await.unwrap();
            if found.cost.ra_queries == 0 {
                assert_eq!(found.cost.files_from_table, files, "{found:?}");
                assert_eq!(found.locations.len(), first.locations.len(), "{found:?}");
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the builder never completed the table: {found:?}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        // Shut down while the loop yields to a collection (up to 40 s): the
        // shutdown does not wait the yield out.
        crate::deep_stack::want_a_collection_for_test();
        tokio::time::sleep(USES_IDLE_POLL + Duration::from_millis(500)).await;
        let report = runtime.shutdown_gracefully(Duration::from_secs(10)).await;
        assert_eq!(report.tasks_total, 1);
        assert_eq!(report.tasks_completed, 1, "{report:?}");
        assert!(!report.timed_out);
    }

    /// The defect this closes: `clear` used to report only its own bookkeeping,
    /// so "one project cleared" and "not one byte returned to the kernel" — the
    /// normal outcome on glibc — were indistinguishable in the answer.
    #[tokio::test]
    async fn runtime_clear_reports_a_real_memory_measurement() {
        let runtime = ServerRuntime::new(3600);

        let report = runtime
            .state()
            .clear(RuntimeClearRequest {
                scope: RuntimeClearScope::SemanticOnly,
                workspace: None,
            })
            .await;

        assert!(
            report.memory.attempted,
            "a glibc or mimalloc build must actually attempt the release"
        );
        #[cfg(target_os = "linux")]
        assert!(
            report.memory.rss_kib_before.is_some_and(|rss| rss > 0),
            "RSS must be read, not guessed: {:?}",
            report.memory
        );
    }
}
