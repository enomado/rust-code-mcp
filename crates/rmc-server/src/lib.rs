//! `rmc-server` — MCP server cluster.
//!
//! Hosts the three modules that together implement the MCP server surface:
//! `tools` (the rmcp adapter endpoints), `mcp` (the `SyncManager` plus
//! project-path resolver), and `semantic` (the rust-analyzer IDE
//! wrapper used by analysis tools), plus `deep_stack`, the thread every one of
//! them has to borrow before it runs rust-analyzer.

// lancedb 0.29's async stack can push Send inference for the sync-manager
// background task past the default recursion limit when the runtime owner
// spawns it.
#![recursion_limit = "512"]
#![warn(unreachable_pub, dead_code)]

// The test binary of this crate allocates as the daemon does: through the heap
// census (`mcp::heap_census`), which the portrait's tests read, over the
// allocator the build selects. Without the mimalloc arm the feature would switch
// only the release call (`mi_collect` instead of `malloc_trim`) while every
// allocation still went to glibc, and a memory probe run "under mimalloc" would
// measure glibc with a no-op release.
#[cfg(all(test, feature = "mimalloc"))]
#[global_allocator]
static TEST_GLOBAL_ALLOCATOR: mcp::heap_census::CensusAllocator<mimalloc::MiMalloc> =
    mcp::heap_census::CensusAllocator {
        inner: mimalloc::MiMalloc,
        census: &mcp::heap_census::GLOBAL_CENSUS,
    };
#[cfg(all(test, not(feature = "mimalloc")))]
#[global_allocator]
static TEST_GLOBAL_ALLOCATOR: mcp::heap_census::CensusAllocator<std::alloc::System> =
    mcp::heap_census::CensusAllocator {
        inner: std::alloc::System,
        census: &mcp::heap_census::GLOBAL_CENSUS,
    };

/// The test binary keeps its data (graph snapshots, index caches, file-text
/// copies) in a directory of its own, removed at exit, not in the user's
/// `~/.local/share/search`.
///
/// # Why
///
/// The served data directory is shared with every live daemon, and the graph
/// tests build the snapshot of this very workspace — the one a daemon serving a
/// session in this repository builds and prunes too. One run failed on ENOENT
/// for the snapshot's `manifest.json`: removed under the test mid-write.
///
/// # How
///
/// Every data directory in the workspace resolves through `XDG_DATA_HOME`
/// (`directories::ProjectDirs`), so the variable is set — before `main`, from
/// `.init_array`, while the process has one thread and setting the environment
/// is sound. Cache directories (model weights, kernel caches) are left alone:
/// sharing them costs nothing and refetching them would cost a download.
#[cfg(all(test, target_os = "linux"))]
mod test_data_dir {
    use std::path::PathBuf;
    use std::sync::OnceLock;

    static DIR: OnceLock<PathBuf> = OnceLock::new();

    #[used]
    #[unsafe(link_section = ".init_array")]
    static SET_BEFORE_MAIN: extern "C" fn() = set_before_main;

    extern "C" fn set_before_main() {
        let dir = std::env::temp_dir().join(format!("rmc-server-test-data-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the test data directory");
        // SAFETY: runs from `.init_array`, before `main`: no other thread exists
        // yet to read the environment concurrently.
        unsafe { std::env::set_var("XDG_DATA_HOME", &dir) };
        DIR.set(dir).expect("set once");
        // SAFETY: `remove` is a plain `extern "C" fn()` with no captured state.
        unsafe { libc::atexit(remove) };
    }

    extern "C" fn remove() {
        if let Some(dir) = DIR.get() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

pub(crate) mod deep_stack;
pub mod panic_boundary;
pub mod tools;
pub mod mcp;
pub mod semantic;
