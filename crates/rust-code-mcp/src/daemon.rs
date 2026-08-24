//! One server per project: a daemon on a unix socket plus a proxy client.
//!
//! # Why
//!
//! The stdio transport ties server and client 1:1 by construction: one pipe,
//! one process. Every editor/agent session started its own `rust-code-mcp`, and
//! with it its own copy of `SemanticService` (the loaded RA context of the workspace,
//! about one and a half gigabytes per project) and its own ONNX/GPU context. Eight sessions on
//! one repository meant eight copies of the very same analysis.
//!
//! Meanwhile the server state is ALREADY shareable and already split per project:
//! `RuntimeState` is a set of `Arc`s, `SemanticService` caches contexts in
//! `HashMap<PathBuf, ProjectContext>`, and the lock is taken per workspace
//! (`WorkspaceLockRegistry`), not globally. Exactly one thing was missing: a transport
//! that can serve more than one client.
//!
//! This is where it appears: the daemon listens on a unix socket and for each connection starts
//! its own `SearchToolRouter` on top of the SHARED `RuntimeState`. The client is the same binary without
//! flags: it pumps stdin/stdout into the socket, and if there is no daemon, starts one itself.
//!
//! # The socket key is not just the project
//!
//! The key includes the cwd, the binary's size and mtime, and the env vars that change the process behavior
//! (embedding profile, background sync, EP census). Otherwise after `cargo build` or after
//! a profile change the client would silently attach to a daemon that computes something other than
//! what was asked, and it would look like the server lying, not like connecting to the wrong place.
//!
//! # A daemon failure never leaves the client without a server
//!
//! Any failure on the connect / start / wait path is `Ok(false)` from
//! [`run_client`], and the caller serves the session itself, in-process, exactly as before
//! this module existed. The daemon is a memory optimization, not a new point of failure.

use fs2::FileExt;
use rmc_server::mcp::{
    BACKGROUND_SYNC_ENV, EMBEDDING_PROFILE_ENV, EP_CENSUS_ENV, RuntimeState, ServerRuntime,
};
use rmc_server::tools::SearchTool;
use rmcp::ServiceExt;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncWriteExt, copy};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{SignalKind, signal};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Kill switch for the whole scheme: `RMC_DAEMON=0` (`off`/`false`/`no`) restores the behavior
/// where the server lives inside the client process.
pub const DAEMON_ENV: &str = "RMC_DAEMON";
/// Socket directory. Defaults to `$XDG_RUNTIME_DIR/rust-code-mcp`.
pub const DAEMON_DIR_ENV: &str = "RMC_DAEMON_DIR";
/// How long the daemon lives without a single connection, in seconds. `0` means forever.
pub const IDLE_ENV: &str = "RMC_DAEMON_IDLE_SECS";

/// Half an hour: enough to survive a pause between questions in a session, and short enough
/// that a closed editor does not hold one and a half gigabytes until the end of the day.
const DEFAULT_IDLE_SECS: u64 = 1800;
/// Idle check interval. Also the upper bound on the exit delay after the last client.
const IDLE_TICK: Duration = Duration::from_secs(15);
/// Upper bound on waiting for a starting daemon. Generous on purpose: with `RMC_EP_CENSUS=1`
/// startup blocks on the GPU probe. The wait is not blind: if the process dies
/// earlier, the wait ends with its exit code, not with a timeout.
const SPAWN_WAIT: Duration = Duration::from_secs(90);
const SPAWN_POLL: Duration = Duration::from_millis(50);

/// How the process was launched. Parsed BEFORE the heavy startup: the client needs neither
/// `ServerRuntime`, nor the EP probe, nor the background sync; it is a pipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Server inside this process over stdio: the behavior before the daemon existed.
    InProcess,
    /// Daemon: listens on the socket, serves many connections with one `RuntimeState`.
    Daemon { socket: PathBuf, idle: Duration },
    /// Client: stdin/stdout ↔ socket, starting the daemon if needed.
    Client { socket: PathBuf },
    /// `--print-socket`: print the socket path and exit (diagnostics).
    PrintSocket { socket: PathBuf },
    /// `--help`.
    Help,
}

/// Parses arguments and env. `args` excludes the program name.
pub fn resolve_mode(args: &[String]) -> Result<Mode, BoxError> {
    let mut socket: Option<PathBuf> = None;
    let mut idle: Option<Duration> = None;
    let mut explicit: Option<&str> = None;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(Mode::Help),
            "--daemon" | "--client" | "--in-process" | "--print-socket" => {
                if let Some(prev) = explicit {
                    return Err(format!("modes {prev} and {arg} are incompatible").into());
                }
                explicit = Some(match arg.as_str() {
                    "--daemon" => "--daemon",
                    "--client" => "--client",
                    "--in-process" => "--in-process",
                    _ => "--print-socket",
                });
            }
            "--socket" => {
                let value = it
                    .next()
                    .ok_or_else(|| BoxError::from("--socket requires a path"))?;
                socket = Some(PathBuf::from(value));
            }
            "--idle-secs" => {
                let value = it
                    .next()
                    .ok_or_else(|| BoxError::from("--idle-secs requires a number"))?;
                idle = Some(Duration::from_secs(value.parse::<u64>()?));
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let socket = match socket {
        Some(path) => path,
        None => default_socket_path()?,
    };
    let idle = idle.unwrap_or_else(idle_from_env);

    Ok(match explicit {
        Some("--daemon") => Mode::Daemon { socket, idle },
        Some("--client") => Mode::Client { socket },
        Some("--in-process") => Mode::InProcess,
        Some("--print-socket") => Mode::PrintSocket { socket },
        _ if daemon_disabled() => Mode::InProcess,
        _ => Mode::Client { socket },
    })
}

pub const USAGE: &str = "\
rust-code-mcp — an MCP server for Rust code.

Without arguments: client of this project's shared daemon (the daemon is started automatically).

  --client            the same, explicitly
  --daemon            become the daemon: listen on the socket, serve many clients
  --in-process        server inside this process over stdio (as before)
  --print-socket      print this project's socket path and exit
  --socket <PATH>     socket path instead of the one derived from the project
  --idle-secs <N>     the daemon exits after N seconds without connections (0 = never)

Env: RMC_DAEMON=0 — always in-process; RMC_DAEMON_DIR — socket directory;
     RMC_DAEMON_IDLE_SECS — same as --idle-secs.
";

fn daemon_disabled() -> bool {
    match std::env::var(DAEMON_ENV) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        Err(_) => false,
    }
}

fn idle_from_env() -> Duration {
    let secs = std::env::var(IDLE_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_IDLE_SECS);
    Duration::from_secs(secs)
}

/// Socket directory. `$XDG_RUNTIME_DIR` is preferred: it is private (0700),
/// on tmpfs, and is cleaned on logout together with orphaned sockets.
fn socket_dir() -> Result<PathBuf, BoxError> {
    if let Ok(dir) = std::env::var(DAEMON_DIR_ENV) {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !runtime_dir.is_empty() {
            return Ok(PathBuf::from(runtime_dir).join("rust-code-mcp"));
        }
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "shared".to_string());
    Ok(std::env::temp_dir().join(format!("rust-code-mcp-{user}")))
}

fn ensure_dir(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    // The socket is an entry point into analysis of someone's code: the directory is owner-only.
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
}

/// Daemon key: the project + everything that changes the meaning of the server's answers.
///
/// The binary is included by size and mtime, not by content hash: a rebuild must
/// yield a NEW daemon (otherwise a client of the new code attaches to the old server), and
/// there is no reason to read 60 megabytes on every startup for that.
fn workspace_key() -> Result<String, BoxError> {
    let cwd = std::env::current_dir()?;
    let cwd = fs::canonicalize(&cwd).unwrap_or(cwd);

    let exe = std::env::current_exe()?;
    let exe_meta = fs::metadata(&exe).ok();
    let exe_len = exe_meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let exe_mtime = exe_meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let env: Vec<(&str, String)> = KEYED_ENV
        .iter()
        .map(|key| {
            (
                *key,
                std::env::var(key).unwrap_or_else(|_| "<unset>".to_string()),
            )
        })
        .collect();

    Ok(key_from_parts(&cwd, &exe, exe_len, exe_mtime, &env))
}

/// Env vars that change the meaning of the server's answers, and therefore the daemon address.
const KEYED_ENV: [&str; 3] = [EMBEDDING_PROFILE_ENV, BACKGROUND_SYNC_ENV, EP_CENSUS_ENV];

/// The pure part of the key: everything that matters comes in as arguments.
///
/// Split out of [`workspace_key`] not for looks but for testability: checking
/// that the key diverges per profile via `set_var` means mutating the global env
/// in parallel with other tests and getting red results unrelated to the key.
fn key_from_parts(
    cwd: &Path,
    exe: &Path,
    exe_len: u64,
    exe_mtime: u128,
    env: &[(&str, String)],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(cwd.as_os_str().as_encoded_bytes());
    hasher.update([0]);
    hasher.update(exe.as_os_str().as_encoded_bytes());
    hasher.update(exe_len.to_le_bytes());
    hasher.update(exe_mtime.to_le_bytes());
    for (key, value) in env {
        hasher.update([0]);
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
    }

    let digest = hasher.finalize();
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

pub fn default_socket_path() -> Result<PathBuf, BoxError> {
    Ok(socket_dir()?.join(format!("{}.sock", workspace_key()?)))
}

fn lock_path(socket: &Path) -> PathBuf {
    socket.with_extension("lock")
}

fn log_path(socket: &Path) -> PathBuf {
    socket.with_extension("log")
}

/// File lock around check / remove stale / start / wait.
///
/// Without it two sessions started at the same time would both find no socket and both
/// start a daemon, i.e. exactly the extra copy of memory that
/// all of this was written to eliminate.
struct SpawnLock {
    _file: File,
}

impl SpawnLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        file.lock_exclusive()?;
        Ok(Self { _file: file })
    }
}

impl Drop for SpawnLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self._file);
    }
}

async fn try_connect(socket: &Path) -> Option<UnixStream> {
    UnixStream::connect(socket).await.ok()
}

/// Client: serve the session through the shared daemon.
///
/// `Ok(true)`: the session ran through the daemon and finished. `Ok(false)`: the daemon
/// could not be obtained; the caller must serve the session itself (in-process).
pub async fn run_client(socket: &Path) -> Result<bool, BoxError> {
    if let Some(stream) = try_connect(socket).await {
        tracing::info!("connected to shared daemon at {}", socket.display());
        proxy(stream).await?;
        return Ok(true);
    }

    if let Some(parent) = socket.parent() {
        if let Err(e) = ensure_dir(parent) {
            tracing::warn!("socket dir {} unusable: {e}", parent.display());
            return Ok(false);
        }
    }

    let lock = match SpawnLock::acquire(&lock_path(socket)) {
        Ok(lock) => lock,
        Err(e) => {
            tracing::warn!("spawn lock unavailable: {e}; serving in-process");
            return Ok(false);
        }
    };

    // Re-check under the lock: while we waited for the lock, the daemon may have started.
    let stream = match try_connect(socket).await {
        Some(stream) => Some(stream),
        None => {
            // The socket file exists but cannot be connected to ⇒ the daemon died without cleaning
            // up. Remove it ourselves: bind over a live file gives EADDRINUSE.
            if socket.exists() {
                let _ = fs::remove_file(socket);
            }
            match spawn_daemon(socket) {
                Ok(child) => wait_for_daemon(socket, child).await,
                Err(e) => {
                    tracing::warn!("failed to spawn daemon: {e}; serving in-process");
                    None
                }
            }
        }
    };
    drop(lock);

    match stream {
        Some(stream) => {
            proxy(stream).await?;
            Ok(true)
        }
        None => Ok(false),
    }
}

fn spawn_daemon(socket: &Path) -> io::Result<Child> {
    let exe = std::env::current_exe()?;
    let log = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_path(socket))?;

    let mut cmd = Command::new(exe);
    cmd.arg("--daemon")
        .arg("--socket")
        .arg(socket)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        // Daemon stderr goes to a file next to the socket: otherwise the shared
        // process's diagnostics are lost together with the session that spawned it.
        .stderr(Stdio::from(log))
        // Own process group: Ctrl-C in the client session must not take down a server
        // that other sessions are using.
        .process_group(0);
    cmd.spawn()
}

/// Wait until the daemon binds the socket. Ends early if the process dies:
/// otherwise a startup failure (e.g. a failed EP probe) would cost a minute and a half of silence.
async fn wait_for_daemon(socket: &Path, mut child: Child) -> Option<UnixStream> {
    let deadline = tokio::time::Instant::now() + SPAWN_WAIT;
    loop {
        if let Some(stream) = try_connect(socket).await {
            return Some(stream);
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                tracing::warn!(
                    "daemon exited before accepting connections ({status}); see {}",
                    log_path(socket).display()
                );
                return None;
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("cannot poll daemon process: {e}");
                return None;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            tracing::warn!("daemon did not come up in {:?}", SPAWN_WAIT);
            let _ = child.kill();
            return None;
        }
        tokio::time::sleep(SPAWN_POLL).await;
    }
}

/// stdin/stdout ↔ socket pipe.
///
/// `select`, not `join`: the daemon closes the connection, and waiting for EOF on
/// stdin at that point is pointless; it may never come.
async fn proxy(stream: UnixStream) -> io::Result<()> {
    let (mut from_daemon, mut to_daemon) = stream.into_split();
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();

    let upstream = async {
        copy(&mut stdin, &mut to_daemon).await?;
        to_daemon.shutdown().await
    };
    let downstream = async {
        copy(&mut from_daemon, &mut stdout).await?;
        stdout.flush().await
    };

    tokio::select! {
        result = upstream => result,
        result = downstream => result,
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Daemon: listen on the socket, serve connections with one shared `RuntimeState`.
pub async fn run_daemon(
    socket: &Path,
    idle: Duration,
    runtime: &ServerRuntime,
) -> Result<(), BoxError> {
    if let Some(parent) = socket.parent() {
        ensure_dir(parent)?;
    }
    let listener = UnixListener::bind(socket).map_err(|e| {
        BoxError::from(format!(
            "failed to bind {}: {e} (is a live daemon already holding the socket?)",
            socket.display()
        ))
    })?;
    tracing::info!(
        "daemon listening on {} (idle timeout {:?})",
        socket.display(),
        idle
    );

    let live = Arc::new(AtomicUsize::new(0));
    let idle_since = Arc::new(AtomicI64::new(now_secs()));

    // Signals must lead to the same exit as idleness: otherwise a daemon killed by `kill`
    // leaves its socket file behind, and the next client sees an address
    // nobody listens on. The client survives this (removes it and starts a new one), but
    // the diagnostics, `--print-socket` plus `ls`, start lying.
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    loop {
        let accepted = tokio::select! {
            accepted = listener.accept() => Some(accepted),
            _ = tokio::time::sleep(IDLE_TICK) => None,
            _ = sigterm.recv() => {
                tracing::info!("SIGTERM, shutting down");
                break;
            }
            _ = sigint.recv() => {
                tracing::info!("SIGINT, shutting down");
                break;
            }
        };

        match accepted {
            Some(Ok((stream, _addr))) => {
                let state = runtime.state();
                let live = Arc::clone(&live);
                let idle_since = Arc::clone(&idle_since);
                live.fetch_add(1, Ordering::SeqCst);
                tracing::info!("client connected ({} live)", live.load(Ordering::SeqCst));
                tokio::spawn(async move {
                    if let Err(e) = serve_connection(stream, state).await {
                        tracing::warn!("connection ended with error: {e}");
                    }
                    // The idle countdown starts when the LAST client leaves.
                    if live.fetch_sub(1, Ordering::SeqCst) == 1 {
                        idle_since.store(now_secs(), Ordering::SeqCst);
                    }
                });
            }
            Some(Err(e)) => {
                tracing::error!("accept failed: {e}");
                break;
            }
            None => {}
        }

        if !idle.is_zero()
            && live.load(Ordering::SeqCst) == 0
            && now_secs() - idle_since.load(Ordering::SeqCst) >= idle.as_secs() as i64
        {
            tracing::info!("no clients for {:?}, shutting down", idle);
            break;
        }
    }

    // Clean up after ourselves: otherwise the next client finds the file, gets a connection
    // refusal and spends a cycle removing the stale socket.
    let _ = fs::remove_file(socket);
    Ok(())
}

async fn serve_connection(stream: UnixStream, state: RuntimeState) -> Result<(), BoxError> {
    let (read_half, write_half) = stream.into_split();
    let service = SearchTool::with_runtime_state(state)
        .serve((read_half, write_half))
        .await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode_of(args: &[&str]) -> Mode {
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        resolve_mode(&owned).expect("mode")
    }

    #[test]
    fn explicit_socket_wins_over_computed_key() {
        let mode = mode_of(&["--daemon", "--socket", "/tmp/x.sock", "--idle-secs", "5"]);
        assert_eq!(
            mode,
            Mode::Daemon {
                socket: PathBuf::from("/tmp/x.sock"),
                idle: Duration::from_secs(5),
            }
        );
    }

    #[test]
    fn in_process_is_explicit_opt_out() {
        assert_eq!(
            mode_of(&["--in-process", "--socket", "/tmp/x.sock"]),
            Mode::InProcess
        );
    }

    #[test]
    fn two_modes_at_once_are_rejected() {
        let owned: Vec<String> = ["--daemon", "--client"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(resolve_mode(&owned).is_err());
    }

    #[test]
    fn unknown_argument_is_rejected() {
        let owned = vec!["--socks".to_string()];
        assert!(resolve_mode(&owned).is_err());
    }

    fn key(cwd: &str, exe: &str, len: u64, mtime: u128, profile: &str) -> String {
        key_from_parts(
            Path::new(cwd),
            Path::new(exe),
            len,
            mtime,
            &[(EMBEDDING_PROFILE_ENV, profile.to_string())],
        )
    }

    #[test]
    fn key_is_stable_for_same_inputs() {
        assert_eq!(
            key("/repo", "/bin/mcp", 10, 20, "gpu"),
            key("/repo", "/bin/mcp", 10, 20, "gpu")
        );
    }

    /// The key must diverge per profile: a daemon started under a different embedding
    /// profile computes something other than what the new client asks for.
    #[test]
    fn key_depends_on_keyed_env() {
        assert_ne!(
            key("/repo", "/bin/mcp", 10, 20, "gpu"),
            key("/repo", "/bin/mcp", 10, 20, "cpu")
        );
    }

    /// Different projects, different daemons; otherwise one per project turns into
    /// one for everything, and a neighboring repository's profile leaks in here.
    #[test]
    fn key_depends_on_project() {
        assert_ne!(
            key("/repo-a", "/bin/mcp", 10, 20, "gpu"),
            key("/repo-b", "/bin/mcp", 10, 20, "gpu")
        );
    }

    /// Rebuilding the binary must yield a new socket: otherwise a client of the new code
    /// is silently served by the old server.
    #[test]
    fn key_depends_on_binary_identity() {
        assert_ne!(
            key("/repo", "/bin/mcp", 10, 20, "gpu"),
            key("/repo", "/bin/mcp", 10, 21, "gpu"),
            "different binary mtime means a different daemon"
        );
        assert_ne!(
            key("/repo", "/bin/mcp", 10, 20, "gpu"),
            key("/repo", "/bin/mcp", 11, 20, "gpu"),
            "different binary size means a different daemon"
        );
    }
}
