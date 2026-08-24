// lancedb 0.29's async stack (lance_io::uring + moka::future) pushes the
// auto-trait Send check past the default 128-level recursion limit when
// the sync-manager future is spawned in main. Bump it locally; this is a
// compile-time inference budget, not a runtime cost.
#![recursion_limit = "512"]

// Один сервер на проект вместо одного на сессию: см. `daemon`. Unix-only —
// транспорт unix-сокетный, на остальных платформах остаётся прежний stdio.
#[cfg(unix)]
mod daemon;

use rmc_server::mcp::{
    BACKGROUND_SYNC_ENABLED_VALUES, BACKGROUND_SYNC_ENV, EP_CENSUS_ENV, ServerRuntime,
    automatic_embedding_profile_name, cuda_capable_features_compiled, parse_background_sync_env,
    probe_ep_census_on_startup,
};
use rmc_server::tools::SearchTool;
use rmcp::{ServiceExt, transport::stdio};
use std::time::Duration;
use tracing_subscriber::{self, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default to WARN for everything, INFO for our own crate. Users who want
    // RA's internal debug logs can set `RUST_LOG=ra_ap_hir=debug,...`.
    //
    // Why this matters: RA emits millions of `tracing::debug!` events during
    // name resolution. With Level::DEBUG enabled globally, the formatter +
    // socket-stderr write pipeline becomes the bottleneck — `build_hypergraph`
    // on a multi-crate workspace went from ~7s to 7+ minutes purely from log
    // formatting overhead. Keep this at WARN unless explicitly overridden.
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("warn,rust_code_mcp=info,rmc_server=info,rmc_indexing=info")
    });
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    // Режим разбирается ДО тяжёлого старта: клиенту общего демона не нужны ни
    // `ServerRuntime`, ни EP-проба, ни фоновый синк — он труба между stdio и сокетом.
    #[cfg(unix)]
    let mode = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match daemon::resolve_mode(&args) {
            Ok(mode) => mode,
            Err(e) => {
                eprintln!("{e}\n\n{}", daemon::USAGE);
                // Явная точка приведения: `main` отдаёт `Box<dyn Error>` без
                // `Send + Sync`, и без typed-let вывод уносит тип всего тела.
                let boxed: Box<dyn std::error::Error> = e;
                return Err(boxed);
            }
        }
    };

    #[cfg(unix)]
    match &mode {
        daemon::Mode::Help => {
            print!("{}", daemon::USAGE);
            return Ok(());
        }
        daemon::Mode::PrintSocket { socket } => {
            println!("{}", socket.display());
            return Ok(());
        }
        daemon::Mode::Client { socket } => {
            // Отказ демона не оставляет сессию без сервера: проваливаемся в
            // in-process ровно к прежнему поведению.
            match daemon::run_client(socket).await {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    tracing::info!("shared daemon unavailable; serving this session in-process")
                }
                Err(e) => tracing::warn!("shared daemon client failed: {e}; serving in-process"),
            }
        }
        daemon::Mode::Daemon { .. } | daemon::Mode::InProcess => {}
    }

    tracing::info!("Starting MCP Server...");

    // Syncs every 5 minutes (300 seconds).
    let runtime = ServerRuntime::new(300);
    tracing::info!("Created MCP server runtime (5-minute sync interval)");

    let background_sync_env = std::env::var(BACKGROUND_SYNC_ENV).ok();
    let background_sync_enabled = parse_background_sync_env(background_sync_env.as_deref());
    tracing::info!(
        "MCP startup defaults: background sync {} ({}='{}'; enabled only for {}, case-insensitive); automatic/background embedding profile default {}; CUDA-capable features compiled: {}",
        if background_sync_enabled {
            "enabled"
        } else {
            "disabled"
        },
        BACKGROUND_SYNC_ENV,
        background_sync_env.as_deref().unwrap_or("<unset>"),
        BACKGROUND_SYNC_ENABLED_VALUES,
        automatic_embedding_profile_name(),
        cuda_capable_features_compiled(),
    );

    // Проба «на чём реально считается граф» — по ручке RMC_EP_CENSUS.
    //
    // Блокирующая и делается ДО того, как поднят сервис: пока никто не
    // обслуживается, занять поток тут ничем не мешает, а вот получить ответ
    // после первого запроса было бы поздно.
    //
    // Отказ пробы валит старт намеренно: ручку взводят, чтобы узнать, работает
    // ли GPU, и «сервер поехал, но на CPU» — ровно тот исход, который она
    // обязана не пропустить (подробности у probe_ep_census_on_startup).
    match probe_ep_census_on_startup() {
        Ok(Some(census)) => tracing::info!("EP census on startup: {census}"),
        Ok(None) => tracing::info!(
            "EP census probe skipped; set {}=1 to check which provider runs the graph",
            EP_CENSUS_ENV
        ),
        Err(e) => {
            tracing::error!("{e}");
            let shutdown = runtime.shutdown_gracefully(Duration::from_secs(10)).await;
            tracing::info!("Runtime shutdown after EP census failure: {:?}", shutdown);
            return Err(e.into());
        }
    }

    if background_sync_enabled {
        runtime.start_background_sync();
        tracing::info!("Started background sync task");
    } else {
        tracing::info!(
            "Background sync task disabled; set {}=1 to enable",
            BACKGROUND_SYNC_ENV
        );
    }

    // Демон: тот же рантайм, но много подключений вместо одной трубы stdio.
    #[cfg(unix)]
    if let daemon::Mode::Daemon { socket, idle } = &mode {
        let result = daemon::run_daemon(socket, *idle, &runtime).await;
        let shutdown = runtime.shutdown_gracefully(Duration::from_secs(10)).await;
        tracing::info!("Runtime shutdown after daemon exit: {:?}", shutdown);
        return result.map_err(|e| -> Box<dyn std::error::Error> { e });
    }

    let service = match SearchTool::with_server_runtime(&runtime)
        .serve(stdio())
        .await
    {
        Ok(service) => service,
        Err(e) => {
            tracing::error!("serving error: {:?}", e);
            let shutdown = runtime.shutdown_gracefully(Duration::from_secs(10)).await;
            tracing::info!("Runtime shutdown after serve error: {:?}", shutdown);
            return Err(e.into());
        }
    };

    let service_result = service.waiting().await;
    if let Err(e) = &service_result {
        tracing::error!("service wait error: {:?}", e);
    }

    let shutdown = runtime.shutdown_gracefully(Duration::from_secs(10)).await;
    tracing::info!("Runtime shutdown complete: {:?}", shutdown);

    service_result?;
    Ok(())
}
