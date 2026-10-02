//! Per-workspace operation locks for index/cache state.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

fn workspace_key(dir: &Path) -> PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// Registry of async locks keyed by canonical workspace directory.
///
/// Two levels:
/// - `global` is a read/write lock. Every per-workspace operation holds it
///   for READ, so operations on different workspaces run concurrently;
///   `lock_all` (whole-cache operations such as a directory-less
///   `clear_cache`) takes it for WRITE and so excludes everyone.
/// - `locks` holds one mutex per canonical workspace, serialising
///   operations on the same workspace.
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
    locks: Arc<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>>,
}

impl WorkspaceLockRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    async fn lock_for(&self, dir: &Path) -> (PathBuf, Arc<Mutex<()>>) {
        let key = workspace_key(dir);
        let lock = {
            let mut locks = self.locks.lock().await;
            locks
                .entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        (key, lock)
    }

    /// Take an exclusive workspace lock.
    ///
    /// Excludes other operations on the same workspace and `lock_all`, but
    /// not operations on other workspaces. The global read guard is taken
    /// first, so a pending `lock_all` writer is never starved by a stream
    /// of workspace operations (tokio's `RwLock` is fair).
    pub async fn lock_exclusive(&self, dir: &Path) -> WorkspaceLockGuard {
        let global_guard = self.global.clone().read_owned().await;
        let (workspace, lock) = self.lock_for(dir).await;
        let guard = lock.lock_owned().await;
        WorkspaceLockGuard {
            workspace,
            _global_guard: global_guard,
            _guard: guard,
        }
    }

    /// Take a read-side workspace lock.
    ///
    /// This is intentionally backed by the same mutex as exclusive locks. It
    /// keeps the first implementation conservative while preserving a call
    /// shape that can move to a real read/write lock later.
    pub async fn lock_shared(&self, dir: &Path) -> WorkspaceLockGuard {
        self.lock_exclusive(dir).await
    }

    /// Take the global operation lock: waits for every in-flight workspace
    /// operation and blocks new ones until dropped.
    pub async fn lock_all(&self) -> WorkspaceGlobalLockGuard {
        WorkspaceGlobalLockGuard {
            _global_guard: self.global.clone().write_owned().await,
        }
    }
}

/// Held workspace operation lock.
pub struct WorkspaceLockGuard {
    workspace: PathBuf,
    _global_guard: OwnedRwLockReadGuard<()>,
    _guard: OwnedMutexGuard<()>,
}

impl WorkspaceLockGuard {
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }
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
}
