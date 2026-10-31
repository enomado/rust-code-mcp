//! Per-workspace operation locks for index/cache state.
//!
//! # Lock order
//!
//! `global (read) → workspace operation → workspace maintenance`. Every
//! acquisition below follows it; a new lock joins this list, not a call site.
//!
//! | Lock | Taken by | Protects |
//! |---|---|---|
//! | `global` read | every guard below | a whole-cache `lock_all` (directory-less `clear_cache`, shutdown) |
//! | `global` write | [`WorkspaceLockRegistry::lock_all`] | everything: waits for every guard below |
//! | operation read | [`WorkspaceLockRegistry::lock_shared`] | the workspace's index and caches against a writer, while readers run side by side |
//! | operation write | [`WorkspaceLockRegistry::lock_exclusive`], index writers | one writing operation on the workspace's index and caches, with no reader inside |
//! | maintenance | index writers, [`MaintenanceGuard`] | the vector store against two writers: an indexing run and a maintenance pass (compaction, index extension, prune) |
//!
//! An index writer ([`IndexWriterGuard`]) holds the operation lock for write and
//! the maintenance lock. When its run leaves vector-store maintenance to do, it
//! turns into a [`MaintenanceGuard`] ([`IndexWriterGuard::into_maintenance`]),
//! which lets the operation lock go and keeps the maintenance lock: readers of
//! the workspace (`lock_shared`) proceed while the pass runs — LanceDB gives
//! them a committed version — the next index writer waits for the pass, and
//! `lock_all` waits for it too.
//!
//! A read guard does not upgrade. An operation that learns under its read lock
//! that it has to write (`search` on a missing or corrupt index) lets it go,
//! takes [`WorkspaceLockRegistry::lock_index_writer`], and checks again under
//! that: another reader may have done the writing while it waited.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmc_indexing::indexing::VectorMaintenance;
use tokio::sync::{Mutex, OwnedMutexGuard, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

fn workspace_key(dir: &Path) -> PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// The two per-workspace locks, in acquisition order.
#[derive(Clone, Default)]
struct WorkspaceSlot {
    operation: Arc<RwLock<()>>,
    maintenance: Arc<Mutex<()>>,
}

/// Registry of async locks keyed by canonical workspace directory.
///
/// Levels (see the module docs for the order and the table):
/// - `global` is a read/write lock. Every per-workspace guard holds it for
///   READ, so operations on different workspaces run concurrently; `lock_all`
///   (whole-cache operations such as a directory-less `clear_cache`) takes it
///   for WRITE and so excludes everyone.
/// - per workspace, an operation read/write lock — readers side by side, a
///   writing operation alone — and a maintenance mutex serialising writers of
///   its vector store (indexing runs and maintenance passes).
///
/// The global level used to be a plain mutex taken by every workspace
/// operation. On a shared daemon that made the per-workspace mutex
/// meaningless: the first `search` on a not-yet-indexed checkout runs a
/// full index under its lock (an hour on a large repo), and every other
/// session's `search` on its own, already-indexed workspace queued behind
/// it with no log line at all.
#[derive(Clone, Default)]
pub struct WorkspaceLockRegistry {
    global: Arc<RwLock<()>>,
    slots: Arc<Mutex<HashMap<PathBuf, WorkspaceSlot>>>,
}

impl WorkspaceLockRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    async fn slot_for(&self, dir: &Path) -> (PathBuf, WorkspaceSlot) {
        let key = workspace_key(dir);
        let slot = {
            let mut slots = self.slots.lock().await;
            slots.entry(key.clone()).or_default().clone()
        };
        (key, slot)
    }

    /// Take an exclusive workspace lock.
    ///
    /// Excludes other operations on the same workspace and `lock_all`, but
    /// not operations on other workspaces, and not a maintenance pass of the
    /// workspace's vector store: an operation that writes that store must
    /// take [`Self::lock_index_writer`] instead. The global read guard is
    /// taken first, so a pending `lock_all` writer is never starved by a
    /// stream of workspace operations (tokio's `RwLock` is fair).
    pub async fn lock_exclusive(&self, dir: &Path) -> WorkspaceLockGuard {
        let global_guard = self.global.clone().read_owned().await;
        let (workspace, slot) = self.slot_for(dir).await;
        let guard = slot.operation.clone().write_owned().await;
        WorkspaceLockGuard {
            workspace,
            _global_guard: global_guard,
            _operation: OperationGuard::Write(guard),
        }
    }

    /// Take a read-side workspace lock: waits for a writing operation on the
    /// workspace, not for other readers, and NOT for a maintenance pass of its
    /// vector store.
    ///
    /// The holder must not write the workspace's index or caches. What the
    /// readers do share is guarded on its own: the search cache has its own
    /// mutex, a BM25 refill takes tantivy's writer lock (a second refill is
    /// refused, not queued, and that search reads vectors alone), and a reader
    /// that turns out to need a writer takes [`Self::lock_index_writer`] after
    /// letting this go.
    pub async fn lock_shared(&self, dir: &Path) -> WorkspaceLockGuard {
        let global_guard = self.global.clone().read_owned().await;
        let (workspace, slot) = self.slot_for(dir).await;
        let guard = slot.operation.clone().read_owned().await;
        WorkspaceLockGuard {
            workspace,
            _global_guard: global_guard,
            _operation: OperationGuard::Read(guard),
        }
    }

    /// Take the lock of an operation that writes the workspace's vector store
    /// (indexing, clearing it): the operation lock for write and then the
    /// maintenance lock, so it waits for a maintenance pass in flight and no
    /// pass starts under it.
    pub async fn lock_index_writer(&self, dir: &Path) -> IndexWriterGuard {
        let global_guard = self.global.clone().read_owned().await;
        let (workspace, slot) = self.slot_for(dir).await;
        let operation_guard = slot.operation.clone().write_owned().await;
        let maintenance_guard = slot.maintenance.clone().lock_owned().await;
        IndexWriterGuard {
            workspace,
            _global_guard: global_guard,
            operation_guard,
            maintenance_guard,
        }
    }

    /// Take the global operation lock: waits for every in-flight workspace
    /// operation and maintenance pass, and blocks new ones until dropped.
    pub async fn lock_all(&self) -> WorkspaceGlobalLockGuard {
        WorkspaceGlobalLockGuard {
            _global_guard: self.global.clone().write_owned().await,
        }
    }
}

/// Either side of a workspace's operation lock — held, never read.
enum OperationGuard {
    Read(#[allow(dead_code)] OwnedRwLockReadGuard<()>),
    Write(#[allow(dead_code)] OwnedRwLockWriteGuard<()>),
}

/// Held workspace operation lock, read or write.
pub struct WorkspaceLockGuard {
    workspace: PathBuf,
    _global_guard: OwnedRwLockReadGuard<()>,
    _operation: OperationGuard,
}

impl WorkspaceLockGuard {
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }
}

/// Held lock of a vector-store writer: global read, operation write,
/// maintenance.
pub struct IndexWriterGuard {
    workspace: PathBuf,
    _global_guard: OwnedRwLockReadGuard<()>,
    operation_guard: OwnedRwLockWriteGuard<()>,
    maintenance_guard: OwnedMutexGuard<()>,
}

impl IndexWriterGuard {
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Let the operation lock go and keep the rest: the writer is done
    /// indexing and only its vector-store maintenance remains. No other
    /// writer slips in between — they need the maintenance lock this keeps.
    pub fn into_maintenance(self) -> MaintenanceGuard {
        drop(self.operation_guard);
        MaintenanceGuard {
            workspace: self.workspace,
            _global_guard: self._global_guard,
            _maintenance_guard: self.maintenance_guard,
        }
    }
}

/// Held maintenance lock of a workspace (with the global read guard): a
/// vector-store maintenance pass runs under it, outside the operation lock.
pub struct MaintenanceGuard {
    workspace: PathBuf,
    _global_guard: OwnedRwLockReadGuard<()>,
    _maintenance_guard: OwnedMutexGuard<()>,
}

impl MaintenanceGuard {
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }
}

/// Run a writer's vector-store maintenance outside its operation lock.
///
/// The one place an index writer finishes: the guard turns into a
/// [`MaintenanceGuard`] BEFORE `maintenance` is polled, so the workspace's
/// readers do not wait for the pass (measured 2026-10-31: `search` waited
/// 413.9 s for a 425.2 s pass that ran under the operation lock), while the
/// next writer and `lock_all` still do.
pub async fn run_outside_operation_lock<F>(writer: IndexWriterGuard, maintenance: F) -> F::Output
where
    F: Future,
{
    let _maintenance_guard = writer.into_maintenance();
    maintenance.await
}

/// Finish an index writer that answers a tool call: release the operation
/// lock now and run the vector-store maintenance its run calls for (if any)
/// in the background, under the maintenance lock — the caller answers without
/// waiting for the pass, and so do the workspace's readers. A panic in the
/// pass is logged like a failed sync, not lost with the task.
pub fn finish_index_writer(writer: IndexWriterGuard, maintenance: Option<VectorMaintenance>) {
    let Some(pass) = maintenance else {
        return;
    };
    let workspace = writer.workspace().display().to_string();
    tokio::spawn(async move {
        let run = run_outside_operation_lock(writer, async move { pass.run().await });
        if let Err(message) = crate::panic_boundary::catch_panic(run).await {
            tracing::error!(
                target: "panic",
                "vector store maintenance of {workspace} panicked: {message}; the next write runs it again"
            );
        }
    });
}

/// Held global operation lock.
pub struct WorkspaceGlobalLockGuard {
    _global_guard: OwnedRwLockWriteGuard<()>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    #[tokio::test]
    async fn workspace_lock_blocks_same_workspace() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = temp_dir.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();

        let guard = registry.lock_exclusive(&workspace).await;
        let waiter_registry = registry.clone();
        let waiter_workspace = workspace.clone();
        let waiter = tokio::spawn(async move {
            let _guard = waiter_registry.lock_exclusive(&waiter_workspace).await;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());

        drop(guard);
        waiter.await.unwrap();
    }

    #[tokio::test]
    async fn global_lock_blocks_workspace_lock() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = temp_dir.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();

        let guard = registry.lock_all().await;
        let waiter_registry = registry.clone();
        let waiter_workspace = workspace.clone();
        let waiter = tokio::spawn(async move {
            let _guard = waiter_registry.lock_exclusive(&waiter_workspace).await;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());

        drop(guard);
        waiter.await.unwrap();
    }

    /// The regression this registry exists to prevent: a long operation on
    /// one workspace (first `search` on an unindexed checkout = full index)
    /// must not stall an operation on another workspace.
    #[tokio::test]
    async fn different_workspaces_do_not_block_each_other() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let a = temp_dir.path().join("a");
        let b = temp_dir.path().join("b");
        std::fs::create_dir(&a).unwrap();
        std::fs::create_dir(&b).unwrap();

        let _held_a = registry.lock_exclusive(&a).await;
        let other = tokio::time::timeout(Duration::from_secs(5), registry.lock_shared(&b)).await;
        assert!(other.is_ok(), "lock on workspace b waited for workspace a");
    }

    #[tokio::test]
    async fn global_lock_waits_for_workspace_lock() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = temp_dir.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();

        let guard = registry.lock_exclusive(&workspace).await;
        let waiter_registry = registry.clone();
        let waiter = tokio::spawn(async move {
            let _guard = waiter_registry.lock_all().await;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());

        drop(guard);
        waiter.await.unwrap();
    }

    #[tokio::test]
    async fn workspace_lock_reports_canonical_workspace() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = temp_dir.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();

        let guard = registry.lock_exclusive(&workspace.join(".")).await;

        assert_eq!(guard.workspace(), std::fs::canonicalize(&workspace).unwrap());
    }

    /// A maintenance pass that runs until the test lets it finish, through
    /// [`run_outside_operation_lock`] as the index writers run theirs.
    /// Returns the release switch and the task; the pass has started when
    /// this returns.
    async fn hold_maintenance(
        registry: &WorkspaceLockRegistry,
        workspace: &Path,
    ) -> (
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let writer = registry.lock_index_writer(workspace).await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(run_outside_operation_lock(writer, async move {
            started_tx.send(()).unwrap();
            release_rx.await.unwrap();
        }));
        started_rx.await.unwrap();
        (release_tx, task)
    }

    fn workspace_in(temp_dir: &TempDir) -> PathBuf {
        let workspace = temp_dir.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        workspace
    }

    /// The point of the maintenance lock: a reader of the workspace does not
    /// wait for the vector-store maintenance a sync left behind (it waited
    /// 413.9 s on 2026-10-31). Mutant: poll the pass before turning the
    /// guard into a `MaintenanceGuard` → this times out.
    #[tokio::test]
    async fn reader_passes_a_maintenance_pass_in_flight() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = workspace_in(&temp_dir);

        let (release, pass) = hold_maintenance(&registry, &workspace).await;
        let reader =
            tokio::time::timeout(Duration::from_secs(5), registry.lock_shared(&workspace)).await;
        assert!(reader.is_ok(), "lock_shared waited for the maintenance pass");
        drop(reader);

        release.send(()).unwrap();
        pass.await.unwrap();
    }

    /// Two writers of one vector store stay serialised: the next indexing run
    /// waits for the maintenance pass in flight.
    #[tokio::test]
    async fn index_writer_waits_for_a_maintenance_pass() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = workspace_in(&temp_dir);

        let (release, pass) = hold_maintenance(&registry, &workspace).await;
        let waiter_registry = registry.clone();
        let waiter_workspace = workspace.clone();
        let writer = tokio::spawn(async move {
            let _guard = waiter_registry.lock_index_writer(&waiter_workspace).await;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !writer.is_finished(),
            "an index writer started while a maintenance pass was running"
        );

        release.send(()).unwrap();
        pass.await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), writer)
            .await
            .expect("the index writer did not get the lock after the pass")
            .unwrap();
    }

    /// Readers of one workspace do not wait for each other: a long `search`
    /// (or `get_similar_code`, `codemap`) does not hold up the next one.
    /// Mutant (checked 2026-10-31): `lock_shared` taking the write side → this
    /// times out.
    #[tokio::test]
    async fn readers_of_one_workspace_do_not_wait_for_each_other() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = workspace_in(&temp_dir);

        let _first = registry.lock_shared(&workspace).await;
        let second =
            tokio::time::timeout(Duration::from_secs(5), registry.lock_shared(&workspace)).await;
        assert!(second.is_ok(), "a second reader waited for the first");
    }

    /// The other half of a read/write lock: a writing operation (a sync's
    /// index run, `index_codebase`, `clear_cache` of the workspace) waits for
    /// the readers inside, and a reader arriving meanwhile waits for it — a
    /// reader never sees an index mid-write.
    #[tokio::test]
    async fn a_writer_waits_for_readers_and_readers_wait_for_a_writer() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = workspace_in(&temp_dir);

        let reader = registry.lock_shared(&workspace).await;
        let writer = tokio::spawn({
            let registry = registry.clone();
            let workspace = workspace.clone();
            async move {
                let _guard = registry.lock_index_writer(&workspace).await;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!writer.is_finished(), "an index writer ran beside a reader");

        drop(reader);
        // The writer now holds the lock (or is about to): a reader waits.
        tokio::time::sleep(Duration::from_millis(10)).await;
        let late_reader = tokio::spawn({
            let registry = registry.clone();
            let workspace = workspace.clone();
            async move {
                let _guard = registry.lock_shared(&workspace).await;
            }
        });
        tokio::time::timeout(Duration::from_secs(5), writer)
            .await
            .expect("the writer never got the lock after the reader left")
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), late_reader)
            .await
            .expect("the reader never got the lock after the writer left")
            .unwrap();
    }

    /// `clear_cache` without a directory deletes stores wholesale: it must
    /// not run under a maintenance pass.
    #[tokio::test]
    async fn global_lock_waits_for_a_maintenance_pass() {
        let registry = WorkspaceLockRegistry::new();
        let temp_dir = TempDir::new().unwrap();
        let workspace = workspace_in(&temp_dir);

        let (release, pass) = hold_maintenance(&registry, &workspace).await;
        let waiter_registry = registry.clone();
        let global = tokio::spawn(async move {
            let _guard = waiter_registry.lock_all().await;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!global.is_finished(), "lock_all ran under a maintenance pass");

        release.send(()).unwrap();
        pass.await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), global)
            .await
            .expect("lock_all did not get the lock after the pass")
            .unwrap();
    }
}
