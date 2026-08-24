//! Один сервер на проект: демон на unix-сокете плюс прокси-клиент.
//!
//! # Зачем
//!
//! Транспорт stdio связывает сервер с клиентом 1:1 по построению — одна труба,
//! один процесс. Каждая сессия редактора/агента поднимала свой `rust-code-mcp`, а
//! вместе с ним свою копию `SemanticService` (загруженный RA-контекст воркспейса,
//! порядка полутора гигабайт на проект) и свой контекст ONNX/GPU. Восемь сессий по
//! одному репозиторию — восемь копий одного и того же анализа.
//!
//! При этом состояние сервера УЖЕ разделяемо и уже разложено по проектам:
//! `RuntimeState` — набор `Arc`, `SemanticService` кэширует контексты в
//! `HashMap<PathBuf, ProjectContext>`, а лок берётся по воркспейсу
//! (`WorkspaceLockRegistry`), а не глобально. Не хватало ровно одного — транспорта,
//! который умеет больше одного клиента.
//!
//! Здесь он и появляется: демон слушает unix-сокет и на каждое подключение поднимает
//! свой `SearchToolRouter` поверх ОБЩЕГО `RuntimeState`. Клиент — тот же бинарь без
//! флагов: перекачивает stdin/stdout в сокет, а если демона нет — поднимает его сам.
//!
//! # Ключ сокета — не только проект
//!
//! В ключ входят cwd, размер и mtime бинаря, и те env, что меняют поведение процесса
//! (профиль эмбеддингов, фоновый синк, EP-перепись). Иначе после `cargo build` или со
//! сменой профиля клиент молча приклеился бы к демону, который считает не то, что
//! просили, — а выглядело бы это как «сервер врёт», не как «подключились не туда».
//!
//! # Отказ демона никогда не оставляет клиента без сервера
//!
//! Любой сбой на пути «подключиться / поднять / дождаться» — это `Ok(false)` из
//! [`run_client`], и вызывающий обслуживает сессию сам, in-process, ровно как до
//! появления этого модуля. Демон — оптимизация памяти, а не новая точка отказа.

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

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Выключатель всей схемы: `RMC_DAEMON=0` (`off`/`false`/`no`) возвращает поведение
/// «сервер живёт внутри процесса-клиента».
pub const DAEMON_ENV: &str = "RMC_DAEMON";
/// Каталог сокетов. По умолчанию `$XDG_RUNTIME_DIR/rust-code-mcp`.
pub const DAEMON_DIR_ENV: &str = "RMC_DAEMON_DIR";
/// Сколько демон живёт без единого подключения, секунды. `0` — вечно.
pub const IDLE_ENV: &str = "RMC_DAEMON_IDLE_SECS";

/// Полчаса: достаточно, чтобы пережить паузу между вопросами в сессии, и мало,
/// чтобы закрытый редактор не держал полтора гигабайта до конца дня.
const DEFAULT_IDLE_SECS: u64 = 1800;
/// Шаг проверки простоя. Он же потолок задержки выхода после последнего клиента.
const IDLE_TICK: Duration = Duration::from_secs(15);
/// Потолок ожидания поднимающегося демона. Щедрый намеренно: при `RMC_EP_CENSUS=1`
/// старт упирается в блокирующую GPU-пробу. Ждём не вслепую — если процесс умер
/// раньше, ожидание обрывается его кодом возврата, а не таймаутом.
const SPAWN_WAIT: Duration = Duration::from_secs(90);
const SPAWN_POLL: Duration = Duration::from_millis(50);

/// Как запущен процесс. Разбирается ДО тяжёлого старта: клиенту не нужны ни
/// `ServerRuntime`, ни EP-проба, ни фоновый синк — он труба.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Сервер внутри этого процесса поверх stdio — поведение до появления демона.
    InProcess,
    /// Демон: слушает сокет, обслуживает много подключений одним `RuntimeState`.
    Daemon { socket: PathBuf, idle: Duration },
    /// Клиент: stdin/stdout ↔ сокет, с подъёмом демона при необходимости.
    Client { socket: PathBuf },
    /// `--print-socket`: напечатать путь сокета и выйти (диагностика).
    PrintSocket { socket: PathBuf },
    /// `--help`.
    Help,
}

/// Разбор аргументов и env. `args` — без имени программы.
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
                    return Err(format!("режимы {prev} и {arg} несовместимы").into());
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
                    .ok_or_else(|| BoxError::from("--socket требует путь"))?;
                socket = Some(PathBuf::from(value));
            }
            "--idle-secs" => {
                let value = it
                    .next()
                    .ok_or_else(|| BoxError::from("--idle-secs требует число"))?;
                idle = Some(Duration::from_secs(value.parse::<u64>()?));
            }
            other => return Err(format!("неизвестный аргумент {other}").into()),
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
rust-code-mcp — MCP-сервер по Rust-коду.

Без аргументов: клиент общего демона этого проекта (демон поднимается сам).

  --client            то же явно
  --daemon            стать демоном: слушать сокет, обслуживать много клиентов
  --in-process        сервер внутри этого процесса поверх stdio (как было раньше)
  --print-socket      напечатать путь сокета этого проекта и выйти
  --socket <PATH>     путь сокета вместо вычисленного по проекту
  --idle-secs <N>     демон выходит после N секунд без подключений (0 — никогда)

Env: RMC_DAEMON=0 — всегда in-process; RMC_DAEMON_DIR — каталог сокетов;
     RMC_DAEMON_IDLE_SECS — то же, что --idle-secs.
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

/// Каталог сокетов. `$XDG_RUNTIME_DIR` предпочтителен: он приватный (0700),
/// на tmpfs и чистится при выходе из системы вместе с осиротевшими сокетами.
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
    // Сокет — точка входа в анализ чужого кода: каталог только владельцу.
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
}

/// Ключ демона: проект + всё, что меняет смысл ответов сервера.
///
/// Бинарь входит размером и mtime, а не хэшем содержимого: пересборка обязана
/// дать НОВЫЙ демон (иначе клиент нового кода приклеится к старому серверу), а
/// читать 60 мегабайт на каждом старте ради этого незачем.
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

/// Env, которые меняют смысл ответов сервера, а значит и адрес демона.
const KEYED_ENV: [&str; 3] = [EMBEDDING_PROFILE_ENV, BACKGROUND_SYNC_ENV, EP_CENSUS_ENV];

/// Чистая часть ключа: всё влияющее приходит аргументами.
///
/// Вынесено из [`workspace_key`] не ради красоты, а ради тестируемости: проверять
/// «ключ разъезжается по профилю» через `set_var` — значит гонять глобальный env
/// параллельно с другими тестами и получать красноту, не связанную с ключом.
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

/// Файловый лок вокруг «проверить / снести протухшее / поднять / дождаться».
///
/// Без него две сессии, стартовавшие одновременно, обе не найдут сокета и обе
/// поднимут демона — то есть ровно та лишняя копия памяти, ради устранения
/// которой всё это и написано.
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

/// Клиент: обслужить сессию через общий демон.
///
/// `Ok(true)` — сессия отработала через демон и завершилась. `Ok(false)` — демона
/// получить не удалось; вызывающий обязан обслужить сессию сам (in-process).
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

    // Повторная проверка под локом: пока мы ждали лок, демон мог подняться.
    let stream = match try_connect(socket).await {
        Some(stream) => Some(stream),
        None => {
            // Файл сокета есть, а подключиться нельзя ⇒ демон умер, не убрав за
            // собой. Снимаем сами: bind поверх живого файла даёт EADDRINUSE.
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
        // stderr демона — в файл рядом с сокетом: иначе диагностика общего
        // процесса теряется вместе с породившей его сессией.
        .stderr(Stdio::from(log))
        // Своя process group: Ctrl-C в сессии клиента не должен валить сервер,
        // которым пользуются другие сессии.
        .process_group(0);
    cmd.spawn()
}

/// Ждать, пока демон забиндит сокет. Обрывается досрочно, если процесс умер —
/// иначе отказ старта (например, провал EP-пробы) стоил бы полутора минут тишины.
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

/// Труба stdin/stdout ↔ сокет.
///
/// `select`, а не `join`: соединение закрывает демон, и ждать при этом EOF на
/// stdin бессмысленно — он может не прийти никогда.
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

/// Демон: слушать сокет, обслуживать подключения одним общим `RuntimeState`.
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
            "не удалось забиндить {}: {e} (живой демон уже держит сокет?)",
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

    loop {
        match tokio::time::timeout(IDLE_TICK, listener.accept()).await {
            Ok(Ok((stream, _addr))) => {
                let state = runtime.state();
                let live = Arc::clone(&live);
                let idle_since = Arc::clone(&idle_since);
                live.fetch_add(1, Ordering::SeqCst);
                tracing::info!("client connected ({} live)", live.load(Ordering::SeqCst));
                tokio::spawn(async move {
                    if let Err(e) = serve_connection(stream, state).await {
                        tracing::warn!("connection ended with error: {e}");
                    }
                    // Отсчёт простоя начинается с ухода ПОСЛЕДНЕГО клиента.
                    if live.fetch_sub(1, Ordering::SeqCst) == 1 {
                        idle_since.store(now_secs(), Ordering::SeqCst);
                    }
                });
            }
            Ok(Err(e)) => {
                tracing::error!("accept failed: {e}");
                break;
            }
            Err(_) => {}
        }

        if !idle.is_zero()
            && live.load(Ordering::SeqCst) == 0
            && now_secs() - idle_since.load(Ordering::SeqCst) >= idle.as_secs() as i64
        {
            tracing::info!("no clients for {:?}, shutting down", idle);
            break;
        }
    }

    // Убрать за собой: иначе следующий клиент найдёт файл, получит отказ в
    // подключении и потратит цикл на снятие протухшего сокета.
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

    /// Ключ обязан разъезжаться по профилю: демон, поднятый под другим профилем
    /// эмбеддингов, считает не то, что просит новый клиент.
    #[test]
    fn key_depends_on_keyed_env() {
        assert_ne!(
            key("/repo", "/bin/mcp", 10, 20, "gpu"),
            key("/repo", "/bin/mcp", 10, 20, "cpu")
        );
    }

    /// Разные проекты — разные демоны, иначе «один на проект» превращается в
    /// «один на всё» и профиль соседнего репозитория протекает сюда.
    #[test]
    fn key_depends_on_project() {
        assert_ne!(
            key("/repo-a", "/bin/mcp", 10, 20, "gpu"),
            key("/repo-b", "/bin/mcp", 10, 20, "gpu")
        );
    }

    /// Пересборка бинаря обязана дать новый сокет: иначе клиент нового кода
    /// молча обслуживается старым сервером.
    #[test]
    fn key_depends_on_binary_identity() {
        assert_ne!(
            key("/repo", "/bin/mcp", 10, 20, "gpu"),
            key("/repo", "/bin/mcp", 10, 21, "gpu"),
            "другой mtime бинаря — другой демон"
        );
        assert_ne!(
            key("/repo", "/bin/mcp", 10, 20, "gpu"),
            key("/repo", "/bin/mcp", 11, 20, "gpu"),
            "другой размер бинаря — другой демон"
        );
    }
}
