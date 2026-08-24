//! Оракул общего демона: сколько бы сессий ни подключилось к проекту, анализ
//! живёт в ОДНОМ процессе.
//!
//! Проверяется не «демон поднялся», а именно шаринг: `runtime_status` отдаёт pid
//! процесса, который реально обслуживает вызов. Два клиента, один pid — состояние
//! общее; два разных pid'а — каждая сессия снова тащит свою копию RA-контекста,
//! то есть ровно та регрессия, ради которой демон и появился.
//!
//! Позитивный контроль тут же, вторым тестом: с `RMC_DAEMON=0` pid обязан совпасть
//! с pid'ом самого клиента. Без него первый тест доказывал бы «два вызова вернули
//! одинаковое число», не отличая это от «оба посчитались где-то не там».

#![cfg(unix)]

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Клиент MCP: процесс-бинарь плюс труба к нему.
struct Session {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<std::io::Result<String>>,
    next_id: u64,
}

impl Session {
    /// `socket_dir` — свой на каждый тест: иначе прогон приклеился бы к демону
    /// живой рабочей сессии и проверял чужой процесс.
    fn start(socket_dir: &Path, shared: bool) -> Result<Self> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_rust-code-mcp"));
        command
            .env("RUST_LOG", "error")
            .env("RMC_DAEMON_DIR", socket_dir)
            // Демон не должен пережить прогон: тик простоя — 15 с, так что
            // осиротевший процесс уйдёт сам даже если тест упадёт до kill.
            .env("RMC_DAEMON_IDLE_SECS", "5")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if shared {
            command.env_remove("RMC_DAEMON");
        } else {
            command.env("RMC_DAEMON", "0");
        }

        let mut child = command.spawn().context("не удалось запустить MCP-клиент")?;
        let stdout = child.stdout.take().context("stdout не перехвачен")?;
        let stdin = child.stdin.take().context("stdin не перехвачен")?;

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let mut session = Self {
            child,
            stdin,
            rx,
            next_id: 1,
        };
        session.handshake()?;
        Ok(session)
    }

    fn handshake(&mut self) -> Result<()> {
        let id = self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "daemon-sharing-test", "version": "0.0.0" }
            }),
        )?;
        let response = self.read_response(id, Duration::from_secs(120))?;
        if response.get("error").is_some() {
            return Err(anyhow!("initialize failed: {response}"));
        }
        self.notify("notifications/initialized", json!({}))
    }

    fn request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        Ok(id)
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(json!({ "jsonrpc": "2.0", "method": method, "params": params }))
    }

    fn send(&mut self, message: Value) -> Result<()> {
        serde_json::to_writer(&mut self.stdin, &message)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_response(&self, id: u64, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(anyhow!("ответ на id {id} не пришёл"));
            }
            let line = match self.rx.recv_timeout(remaining) {
                Ok(line) => line?,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(anyhow!("ответ на id {id} не пришёл"));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(anyhow!("сервер закрыл stdout до ответа на id {id}"));
                }
            };
            let value: Value = serde_json::from_str(&line)
                .with_context(|| format!("не JSON-RPC строка на stdout: {line:?}"))?;
            if value.get("id").and_then(Value::as_u64) == Some(id) {
                return Ok(value);
            }
        }
    }

    /// pid процесса, который РЕАЛЬНО обслуживает вызовы этой сессии.
    fn serving_pid(&mut self) -> Result<u32> {
        let id = self.request(
            "tools/call",
            json!({ "name": "runtime_status", "arguments": {} }),
        )?;
        let response = self.read_response(id, Duration::from_secs(120))?;
        if response.get("error").is_some() {
            return Err(anyhow!("runtime_status failed: {response}"));
        }
        let text = response
            .pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("в ответе нет текста статуса: {response}"))?;
        let status: Value = serde_json::from_str(text)?;
        status
            .pointer("/process/pid")
            .and_then(Value::as_u64)
            .map(|pid| pid as u32)
            .ok_or_else(|| anyhow!("в статусе нет process.pid: {text}"))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn kill_pid(pid: u32) {
    let _ = Command::new("kill").arg(pid.to_string()).status();
}

#[test]
fn two_clients_share_one_server_process() -> Result<()> {
    let socket_dir = TempDir::new()?;

    let mut first = Session::start(socket_dir.path(), true)?;
    let first_pid = first.serving_pid()?;
    let mut second = Session::start(socket_dir.path(), true)?;
    let second_pid = second.serving_pid()?;

    assert_eq!(
        first_pid, second_pid,
        "две сессии обслужены разными процессами ⇒ каждая держит свою копию анализа"
    );
    assert_ne!(
        first_pid,
        first.child.id(),
        "вызов обслужен самим клиентом ⇒ демон не поднялся, шаринга нет"
    );
    assert_ne!(second_pid, second.child.id());

    drop(first);
    drop(second);
    kill_pid(first_pid);
    Ok(())
}

/// Убитый демон обязан убрать за собой файл сокета.
///
/// Клиент протухший сокет переживает — снимет и поднимет заново. Но пока файл
/// лежит, `--print-socket` плюс `ls` показывают адрес, по которому никого нет,
/// то есть диагностика врёт ровно в тот момент, когда за ней и приходят.
#[test]
fn killed_daemon_removes_its_socket() -> Result<()> {
    let dir = TempDir::new()?;
    let socket = dir.path().join("probe.sock");

    let mut daemon = Command::new(env!("CARGO_BIN_EXE_rust-code-mcp"))
        .arg("--daemon")
        .arg("--socket")
        .arg(&socket)
        .env("RUST_LOG", "error")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    wait_until(Duration::from_secs(60), || socket.exists())
        .ok_or_else(|| anyhow!("демон не забиндил сокет"))?;

    kill_pid(daemon.id());
    let gone = wait_until(Duration::from_secs(30), || !socket.exists());
    let _ = daemon.wait();
    gone.ok_or_else(|| anyhow!("после SIGTERM остался протухший {}", socket.display()))?;
    Ok(())
}

fn wait_until(timeout: Duration, mut done: impl FnMut() -> bool) -> Option<()> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if done() {
            return Some(());
        }
        thread::sleep(Duration::from_millis(20));
    }
    None
}

/// Позитивный контроль: выключатель обязан возвращать прежнее поведение.
#[test]
fn opt_out_serves_in_process() -> Result<()> {
    let socket_dir = TempDir::new()?;

    let mut session = Session::start(socket_dir.path(), false)?;
    let serving_pid = session.serving_pid()?;

    assert_eq!(
        serving_pid,
        session.child.id(),
        "с RMC_DAEMON=0 сессию обязан обслуживать сам процесс-клиент"
    );
    Ok(())
}
