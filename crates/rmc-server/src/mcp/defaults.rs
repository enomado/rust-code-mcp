//! Operational defaults for MCP server startup and automatic work.

use rmc_engine::embeddings::{
    CPU_EP, DIRECTML_EP, EmbeddingBackend, EmbeddingProfile, EmbeddingRuntime, MIGRAPHX_EP,
    ProviderCensus, probe_provider_census, resolve_profile,
};
use std::path::PathBuf;
use std::sync::OnceLock;

pub const BACKGROUND_SYNC_ENV: &str = "RMC_BACKGROUND_SYNC";

pub const BACKGROUND_SYNC_ENABLED_VALUES: &str = "1/true/yes/on";

/// Ручка стартовой пробы «граф реально считается на execution provider'е»:
/// `RMC_EP_CENSUS=1`.
///
/// # Почему по ручке, а не всегда
/// Проба поднимает ОТДЕЛЬНУЮ сессию с профилированием (включить его после
/// сборки сессии нельзя), то есть повторно грузит модель, а на холодном кэше
/// ядер платит ещё и компиляцию MIGraphX (45–70 с). Платить это каждым стартом
/// сервера ради диагностики незачем.
pub const EP_CENSUS_ENV: &str = "RMC_EP_CENSUS";

/// Профиль, которым сервер считает эмбеддинги, когда вызывающий не назвал свой.
///
/// Дефолт — CPU: он собирается всегда и работает на любой машине. GPU-профиль
/// требует и фичи сборки (`--features migraphx`), и системного ONNX Runtime с
/// этим EP, поэтому включается ЯВНО, переменной [`EMBEDDING_PROFILE_ENV`], а не
/// угадыванием по тому, что доступно на машине.
pub const DEFAULT_AUTOMATIC_EMBEDDING_PROFILE: &str = "local-cpu-small";

/// Ручка выбора профиля по умолчанию: `RMC_EMBEDDING_PROFILE=local-gpu-bge`.
///
/// ⚠ Профиль входит в идентичность эмбеддера, а та — в путь коллекции: смена
/// профиля означает ДРУГОЙ индекс, который надо построить заново.
pub const EMBEDDING_PROFILE_ENV: &str = "RMC_EMBEDDING_PROFILE";

/// Разбор булевой ручки окружения: включено только явным словом из
/// [`BACKGROUND_SYNC_ENABLED_VALUES`].
///
/// Общий на все такие ручки намеренно: две переменные, включающиеся РАЗНЫМИ
/// словами, — источник «я же выставил, а не работает».
pub fn parse_enabled_env(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };

    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn parse_background_sync_env(value: Option<&str>) -> bool {
    parse_enabled_env(value)
}

/// Имя профиля по умолчанию: из [`EMBEDDING_PROFILE_ENV`], иначе
/// [`DEFAULT_AUTOMATIC_EMBEDDING_PROFILE`].
///
/// Читается ОДИН раз за процесс: профиль по умолчанию — свойство запуска
/// сервера, а не отдельного запроса, и меняться на ходу он не должен (иначе
/// половина индекса приехала бы одним эмбеддером, половина другим).
pub fn automatic_embedding_profile_name() -> &'static str {
    static PROFILE: OnceLock<String> = OnceLock::new();
    PROFILE
        .get_or_init(|| {
            let requested = resolve_automatic_profile_name(
                std::env::var(EMBEDDING_PROFILE_ENV).ok().as_deref(),
            );

            // Fail-fast: опечатка в имени профиля не должна тихо откатывать на
            // CPU-дефолт — иначе «GPU включён» окажется неправдой, а заметить
            // это можно будет только по скорости.
            //
            // Последний рубеж, а не первый: бинарь зовёт
            // [`validate_automatic_profile`] ДО диспетчера режимов и отказывает
            // кодом 2 с одной строкой. Паника остаётся для встраивания, где
            // такого входа нет, — и её текст читается хуже, поэтому доходить
            // сюда штатным путём не должно.
            if let Err(err) = resolve_startup_profile(&requested) {
                panic!(
                    "{EMBEDDING_PROFILE_ENV}='{requested}' is not a usable embedding profile: {err}"
                );
            }
            requested
        })
        .as_str()
}

/// The profile name a value of [`EMBEDDING_PROFILE_ENV`] asks for.
///
/// A blank value counts as unset: an empty value from a launch wrapper is a
/// slip, and taking it for a profile name would refuse to start instead of
/// falling back to a sensible default.
pub fn resolve_automatic_profile_name(env_value: Option<&str>) -> String {
    env_value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_AUTOMATIC_EMBEDDING_PROFILE)
        .to_string()
}

/// Каталог, относительно которого резолвится профиль ПО УМОЛЧАНИЮ.
///
/// Рабочая директория процесса, а не директория конкретного запроса: дефолтный
/// профиль — свойство запуска сервера, и сервер поднимают из корня проекта,
/// который он обслуживает.
fn startup_project_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Профиль по умолчанию, разрешённый ТЕМ ЖЕ резолвером, что и явно
/// запрошенный в вызове тула.
///
/// Именно резолвером, а не `from_profile_name`: тот знает только встроенный
/// список, а профиль вполне может быть ПРОЕКТНЫМ (`embedding_profiles.toml` в
/// корне проекта) — так на винде объявлен `local-qwen3-06b`, считающий
/// эмбеддинги на локальном llama-server. Со строгой проверкой по built-in
/// сервер с таким `RMC_EMBEDDING_PROFILE` паниковал на старте, то есть
/// проектные профили были недостижимы для дефолта — при том что для явного
/// параметра тула они работают.
fn resolve_startup_profile(name: &str) -> Result<EmbeddingProfile, String> {
    resolve_profile(name, &startup_project_root())
}

/// Check that `name` resolves to a usable profile, installing nothing.
///
/// Public because the refusal has to happen BEFORE a binary picks its mode. A
/// client of the shared daemon returns from `main` long before anything reads
/// the profile, so a typo used to travel through the daemon and exit 0 — while
/// the same typo with `RMC_DAEMON=0` refused to start. One input, two outcomes,
/// decided by a transport the caller never chose.
///
/// Resolved through [`resolve_startup_profile`], so a PROJECT profile from
/// `embedding_profiles.toml` counts as usable here exactly as it does for an
/// explicit tool parameter.
pub fn validate_automatic_profile(name: &str) -> Result<(), String> {
    resolve_startup_profile(name).map(|_| ())
}

pub(crate) fn automatic_embedding_backend() -> EmbeddingBackend {
    let profile = resolve_startup_profile(automatic_embedding_profile_name())
        .expect("automatic embedding profile is validated on first read");
    EmbeddingBackend::from_profile(profile)
}

/// The startup EP probe, if it was asked for through [`EP_CENSUS_ENV`].
///
/// Returns `Ok(None)` when the knob is not set, and `Ok(Some(census))` — the
/// per-provider node census, already written to the log.
///
/// # Why a failure rather than a warning
/// The knob is set with one question in mind: does the GPU really work? The
/// class this whole layer exists for — the EP came up, but the graph was
/// computed on the CPU — shows ONLY as speed, so a warning in a starting
/// server's log does not catch it: the server drives on, and "GPU is on"
/// stays a false conclusion. Hence a zero MIGraphX census on a GPU profile is
/// an `Err` rather than a log line: the caller is left with a value to decide
/// on.
///
/// ⚠ The startup caller no longer turns that value into a dead process — see
/// [`spawn_ep_census_on_startup`] for why. This function is left exactly as it
/// was so that the verdict can still be asked for on its own, away from a
/// startup path where it has someone to hold up.
///
/// On a CPU profile the probe asserts nothing and only prints the census: how
/// many nodes ran on the CPU is a fact, not a failure.
pub fn probe_ep_census_on_startup() -> Result<Option<String>, String> {
    if !parse_enabled_env(std::env::var(EP_CENSUS_ENV).ok().as_deref()) {
        return Ok(None);
    }

    let backend = automatic_embedding_backend();
    let profile = backend.profile.name();

    // Перепись узлов по провайдерам умеет ТОЛЬКО fastembed-ONNX-путь: она
    // строит сессию ORT и смотрит, кому достались узлы графа. У остальных
    // рантаймов графа тут нет вовсе — счёт идёт в чужом процессе (openrouter,
    // в т.ч. локальный llama-server) или в Candle (CUDA). Пропуск, а не отказ:
    // иначе взведённая ручка валит старт сервера на профиле, к которому она
    // просто не относится, и выглядит это как поломка конфигурации.
    if !matches!(
        backend.runtime,
        EmbeddingRuntime::LocalFastembedOnnxCpu
            | EmbeddingRuntime::LocalFastembedOnnxMigraphx
            | EmbeddingRuntime::LocalFastembedOnnxDirectml
    ) {
        tracing::info!(
            profile,
            runtime = ?backend.runtime,
            "{EP_CENSUS_ENV} is set, but this profile does not run an ONNX graph in-process — \
             nothing to census"
        );
        return Ok(None);
    }

    tracing::info!(
        profile,
        "{EP_CENSUS_ENV} is set: probing which execution provider actually runs the graph \
         (loads the model once more; a cold MIGraphX kernel cache costs 45-70s)"
    );

    let census = probe_provider_census(&backend)
        .map_err(|e| format!("EP census probe failed for profile `{profile}`: {e}"))?;
    tracing::info!(profile, census = %census, "EP census");

    ep_census_verdict(backend.runtime, profile, &census)?;
    Ok(Some(census.to_string()))
}

/// Start the EP census probe without waiting for it.
///
/// # Why it no longer blocks (2026-09-04)
/// The probe used to run between process start and the transport coming up,
/// and the argument for that — nobody is being served yet, so occupying this
/// thread costs nothing — held exactly as long as the server was a stdio one,
/// launched by the very client that was waiting for it anyway. A shared daemon
/// voids it: the client is ALREADY waiting on the socket under a handshake
/// deadline of its own (`MCP_TIMEOUT`, 30s by default in Claude Code).
///
/// Measured from the daemon log: 4.95s with a warm MIGraphX kernel cache and
/// 38.7s with a cold one — the probe itself warns about 45-70s. So the session
/// that had to start a cold daemon, and the daemon leaves on its idle timeout
/// every half hour, lost every tool this server offers — deterministically,
/// not now and then. A diagnostic does not get to decide whether the service
/// happens at all.
///
/// # What that costs
/// A failing probe no longer refuses the start. Refusing it AFTER the
/// transport is up would cut clients already being served, and since it is a
/// client that starts the daemon, a probe failing on every start would loop:
/// client starts daemon, daemon dies, client starts daemon. The
/// machine-readable verdict — the exit code — was a fiction in daemon mode
/// regardless: nobody reads the exit status of a process a client spawned into
/// the background. What remains is an `ERROR` in the log, and that is the
/// honest price of the server being reachable in the first place.
pub fn spawn_ep_census_on_startup() -> std::thread::JoinHandle<()> {
    spawn_startup_probe(probe_ep_census_on_startup)
}

/// The part of [`spawn_ep_census_on_startup`] that is worth an oracle: what has
/// to hold is that the call RETURNS rather than waiting for the probe, and a
/// stand-in probe lets that be judged without a GPU, a model, or a network.
fn spawn_startup_probe<F>(probe: F) -> std::thread::JoinHandle<()>
where
    F: FnOnce() -> Result<Option<String>, String> + Send + 'static,
{
    std::thread::Builder::new()
        .name("ep-census".to_string())
        // The stack the probe used to run on, main's. ORT graph initialisation
        // is not where one wants to find out empirically whether the default
        // of a fresh thread is enough.
        .stack_size(8 * 1024 * 1024)
        .spawn(move || match probe() {
            Ok(Some(census)) => tracing::info!("EP census on startup: {census}"),
            Ok(None) => tracing::info!(
                "EP census probe skipped; set {}=1 to check which provider runs the graph",
                EP_CENSUS_ENV
            ),
            Err(e) => tracing::error!("{e}"),
        })
        .expect("spawning a thread for the EP census probe")
}

/// Вердикт по переписи: приемлема ли она для профиля, который просили.
///
/// Отделено от пробы намеренно: сама проба требует карты, ORT с MIGraphX и
/// скачанной модели, то есть проверяется только на подходящей машине. Решение
/// же «это отказ или норма» — чистая функция, и гейтится обычной суитой.
pub(crate) fn ep_census_verdict(
    runtime: EmbeddingRuntime,
    profile: &str,
    census: &ProviderCensus,
) -> Result<(), String> {
    // Утверждение НЕ про долю: MIGraphX не берёт узлы поштучно, он вырезает
    // подграф и подставляет ОДИН фьюженный узел — на здоровом GPU-пути перепись
    // выглядит как `MIGraphXExecutionProvider=1`. Поэтому порог тут ровно один:
    // фьюженный узел есть или его нет.
    //
    // Порог одинаков для обоих GPU-рантаймов, но провайдер у каждого СВОЙ:
    // спрашивать MIGraphX на виндовом профиле — значит гарантированно получить
    // ноль и объявить отказ на здоровой машине.
    let (want_ep, ep_label) = match runtime {
        EmbeddingRuntime::LocalFastembedOnnxMigraphx => (MIGRAPHX_EP, "MIGraphX"),
        EmbeddingRuntime::LocalFastembedOnnxDirectml => (DIRECTML_EP, "DirectML"),
        EmbeddingRuntime::LocalQwen3CandleCuda
        | EmbeddingRuntime::LocalFastembedOnnxCpu
        | EmbeddingRuntime::OpenRouter => return Ok(()),
    };
    if census.nodes_on(want_ep) == 0 {
        return Err(format!(
            "profile `{profile}` asks for {ep_label}, but not a single graph node ran on it \
             (census: {census}) — the graph silently fell back to CPU"
        ));
    }
    Ok(())
}

pub fn cuda_capable_features_compiled() -> bool {
    rmc_engine::embeddings::CUDA_CAPABLE_FEATURES_COMPILED
}

/// GPU backends compiled into this binary, rendered for the startup line.
///
/// `"none"` means CPU-only for real, on every vendor — unlike the
/// CUDA-only flag it replaces.
pub fn gpu_backends_compiled() -> String {
    let backends = rmc_engine::embeddings::GPU_BACKENDS_COMPILED;
    if backends.is_empty() {
        "none".to_string()
    } else {
        backends.join(",")
    }
}

/// Собран с GPU-бэкендом, а автоматический профиль — CPU-шный.
///
/// Не ошибка: фича сборки не обещает, что в рантайме есть ORT с этим EP, и
/// CPU-дефолт честно работает везде — поэтому [`DEFAULT_AUTOMATIC_EMBEDDING_PROFILE`]
/// остаётся CPU, а GPU включают явно.
///
/// Но и молчать нельзя. Расхождение стоит ~80x на ОДНОЙ И ТОЙ ЖЕ модели
/// (2.6-3.3 чанка/с против 221-261 на migraphx), а в логах выглядит как
/// обычная работа: ни отказа, ни предупреждения — просто медленно. Ровно так
/// у codex-клиента без `env` месяцами считался фон, пока это не нашли по
/// 580% CPU. Забыть переменную в новом клиенте легко, поэтому забывчивость
/// должна быть ВИДНА на старте.
fn cpu_profile_on_gpu_build(compiled_backends: &[&str], automatic: &EmbeddingBackend) -> bool {
    !compiled_backends.is_empty()
        && matches!(automatic.runtime, EmbeddingRuntime::LocalFastembedOnnxCpu)
}

/// Текст стартового предупреждения для [`cpu_profile_on_gpu_build`], либо
/// `None`, когда предупреждать не о чем.
pub fn cpu_profile_on_gpu_build_warning() -> Option<String> {
    let automatic = automatic_embedding_backend();
    if !cpu_profile_on_gpu_build(rmc_engine::embeddings::GPU_BACKENDS_COMPILED, &automatic) {
        return None;
    }

    Some(format!(
        "This binary is built with GPU backends ({}) but the automatic/background embedding profile is {}, which runs on CPU: \
         background indexing will be roughly two orders of magnitude slower on the same model. \
         Set {}=local-gpu-bge for the GPU profile, or ignore this if the CPU profile is deliberate.",
        gpu_backends_compiled(),
        automatic.profile.name(),
        EMBEDDING_PROFILE_ENV,
    ))
}

/// Может ли этот бэкенд считать в ФОНЕ, без человека у клавиатуры.
///
/// # Что здесь решается
///
/// Фоновый sync — единственная работа сервера, которую никто не запускал
/// осознанно. Поэтому вопрос не «потянет ли машина», а «переживёт ли молчащий
/// автомат отказ этого рантайма». Отсюда две границы, и они разные:
///
/// - **ONNX-рантаймы (CPU и локальный GPU) — да.** Это одна и та же лёгкая
///   модель (bge-small, 384 измерения) на одном и том же графе, разница лишь
///   в execution provider'е. Локальный GPU пускается в фон с тех пор, как
///   не-финитные векторы ROCm EP закрыты гейтом в
///   `EmbeddingGenerator::guard_finite`: до гейта автомат мог месяцами тихо
///   писать в индекс NaN'ы, и это было бы неотличимо от «поиск стал хуже».
/// - **Qwen3 на CUDA — нет.** Это модель от 0.6B до 8B: автоматический старт
///   такой сессии каждые пять минут занимает VRAM у того, кто за машиной
///   сейчас работает. Ограничение здесь про РЕСУРС, а не про корректность,
///   поэтому гейт на NaN его не снимает; явные команды с этим профилем
///   работают как работали.
pub(crate) fn is_background_embedding_backend(backend: &EmbeddingBackend) -> bool {
    matches!(
        backend.runtime,
        EmbeddingRuntime::LocalFastembedOnnxCpu
            | EmbeddingRuntime::LocalFastembedOnnxMigraphx
            | EmbeddingRuntime::LocalFastembedOnnxDirectml
            | EmbeddingRuntime::OpenRouter
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the 2026-09-04 fix is about: the EP census does not get to hold up
    /// whoever started it. It used to block ahead of the transport coming up,
    /// and on a cold MIGraphX kernel cache (38.7s measured) it outlasted the
    /// client's handshake deadline of 30s — that session got no tools at all.
    ///
    /// The stand-in probe is deliberate: what is asserted is the return of
    /// control itself, and that has to be judged without a GPU, a model or a
    /// network. The mutant — calling the probe directly — fails the first
    /// assert; the second one is the positive control, that the probe is
    /// actually run rather than dropped on the floor.
    #[test]
    fn a_startup_probe_does_not_gate_the_caller() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let (tx, rx) = mpsc::channel();
        let started = Instant::now();
        let handle = spawn_startup_probe(move || {
            std::thread::sleep(Duration::from_millis(500));
            tx.send(()).expect("the oracle listens until it joins");
            Ok(None)
        });
        let returned_after = started.elapsed();

        assert!(
            returned_after < Duration::from_millis(200),
            "the spawn waited for the probe: returned after {returned_after:?}"
        );
        rx.recv_timeout(Duration::from_secs(10))
            .expect("the stand-in probe never ran — the spawn lost it");
        handle.join().expect("the probe thread panicked");
    }

    #[test]
    fn background_sync_env_is_disabled_by_default() {
        assert!(!parse_background_sync_env(None));
        assert!(!parse_background_sync_env(Some("")));
        assert!(!parse_background_sync_env(Some("0")));
        assert!(!parse_background_sync_env(Some("false")));
    }

    #[test]
    fn background_sync_env_accepts_explicit_true_values() {
        assert!(parse_background_sync_env(Some("1")));
        assert!(parse_background_sync_env(Some("true")));
        assert!(parse_background_sync_env(Some("YES")));
        assert!(parse_background_sync_env(Some(" on ")));
    }

    /// A GPU build quietly running the CPU profile is the failure this warns
    /// about; a CPU build doing the same is simply correct, and warning there
    /// would train the reader to skip the line.
    #[test]
    fn cpu_profile_is_only_worth_warning_about_on_a_gpu_build() {
        let cpu = EmbeddingBackend::from_profile_name("local-cpu-small").unwrap();
        let gpu = EmbeddingBackend::from_profile_name("local-gpu-bge").unwrap();

        assert!(cpu_profile_on_gpu_build(&["migraphx"], &cpu));
        assert!(!cpu_profile_on_gpu_build(&[], &cpu));
        assert!(!cpu_profile_on_gpu_build(&["migraphx"], &gpu));
        assert!(!cpu_profile_on_gpu_build(&[], &gpu));
    }

    #[test]
    fn profile_env_absent_or_blank_falls_back_to_cpu_default() {
        assert_eq!(resolve_automatic_profile_name(None), "local-cpu-small");
        assert_eq!(resolve_automatic_profile_name(Some("")), "local-cpu-small");
        assert_eq!(
            resolve_automatic_profile_name(Some("   ")),
            "local-cpu-small"
        );
    }

    #[test]
    fn profile_env_selects_the_named_profile() {
        assert_eq!(
            resolve_automatic_profile_name(Some("local-gpu-bge")),
            "local-gpu-bge"
        );
        // Обрамляющие пробелы — от обёрток запуска, а не часть имени.
        assert_eq!(
            resolve_automatic_profile_name(Some(" local-gpu-bge\n")),
            "local-gpu-bge"
        );
    }

    /// Дефолтный профиль обязан быть тем, что собирается всегда и годится для
    /// фоновой работы: сервер стартует с ним на любой машине.
    #[test]
    fn default_profile_is_a_cpu_background_capable_backend() {
        let backend = EmbeddingBackend::from_profile_name(DEFAULT_AUTOMATIC_EMBEDDING_PROFILE)
            .expect("default profile resolves");

        assert_eq!(backend.profile.name(), "local-cpu-small");
        assert!(is_background_embedding_backend(&backend));
    }

    /// Имя, которого нет среди профилей, должно ОТКАЗЫВАТЬ, а не тихо
    /// откатываться на CPU-дефолт.
    #[test]
    fn unknown_profile_name_is_rejected() {
        let requested = resolve_automatic_profile_name(Some("local-gpu-bge-typo"));

        assert!(
            EmbeddingBackend::from_profile_name(&requested).is_err(),
            "a typo in the profile name must not resolve"
        );
    }

    /// Профиль ORT в той форме, в какой его пишет рантайм: у узла три события.
    fn profile_json(nodes: &[(&str, &str)]) -> String {
        let events: Vec<serde_json::Value> = nodes
            .iter()
            .flat_map(|(name, provider)| {
                ["_fence_before", "_kernel_time", "_fence_after"]
                    .into_iter()
                    .map(move |phase| {
                        serde_json::json!({
                            "cat": "Node",
                            "name": format!("{name}{phase}"),
                            "dur": 7,
                            "args": {"provider": provider},
                        })
                    })
            })
            .collect();
        serde_json::Value::Array(events).to_string()
    }

    /// Здоровый GPU-путь: ОДИН фьюженный узел MIGraphX и ничего на CPU.
    #[test]
    fn ep_verdict_accepts_a_fused_migraphx_node() {
        let census =
            ProviderCensus::from_profile_json(&profile_json(&[("MIGraphX_0", MIGRAPHX_EP)]))
                .unwrap();

        assert!(
            ep_census_verdict(
                EmbeddingRuntime::LocalFastembedOnnxMigraphx,
                "local-gpu-bge",
                &census
            )
            .is_ok()
        );
    }

    /// Виндовый GPU-путь судится СВОИМ провайдером.
    #[test]
    fn ep_verdict_accepts_a_directml_node() {
        let census =
            ProviderCensus::from_profile_json(&profile_json(&[("MatMul_0", DIRECTML_EP)])).unwrap();

        assert!(
            ep_census_verdict(
                EmbeddingRuntime::LocalFastembedOnnxDirectml,
                "local-dml-bge",
                &census
            )
            .is_ok()
        );
    }

    /// 🚨 Гейт против самой вероятной ошибки этого слоя: судить виндовый
    /// профиль по MIGraphX. Перепись ЗДОРОВАЯ — весь граф на DirectML, — и
    /// вердикт, спрашивающий не тот провайдер, объявил бы отказ на исправной
    /// машине. Обратная пара тоже проверяется: MIGraphX-профиль с одними
    /// DirectML-узлами — отказ, а не «ну GPU же».
    #[test]
    fn ep_verdict_asks_the_provider_that_matches_the_runtime() {
        let dml_only =
            ProviderCensus::from_profile_json(&profile_json(&[("MatMul_0", DIRECTML_EP)])).unwrap();
        assert!(
            ep_census_verdict(
                EmbeddingRuntime::LocalFastembedOnnxMigraphx,
                "local-gpu-bge",
                &dml_only
            )
            .is_err(),
            "MIGraphX-профиль обязан отказать на переписи без узлов MIGraphX"
        );

        let migraphx_only =
            ProviderCensus::from_profile_json(&profile_json(&[("MIGraphX_0", MIGRAPHX_EP)]))
                .unwrap();
        assert!(
            ep_census_verdict(
                EmbeddingRuntime::LocalFastembedOnnxDirectml,
                "local-dml-bge",
                &migraphx_only
            )
            .is_err(),
            "DirectML-профиль обязан отказать на переписи без узлов DirectML"
        );
    }

    /// Тот самый класс, ради которого ручка заведена: EP зарегистрировался, а
    /// граф посчитан на CPU. Ошибок при этом нет НИ ОДНОЙ — только скорость.
    #[test]
    fn ep_verdict_rejects_a_silent_cpu_fallback() {
        let census = ProviderCensus::from_profile_json(&profile_json(&[
            ("Add_1", CPU_EP),
            ("MatMul_2", CPU_EP),
        ]))
        .unwrap();

        let verdict = ep_census_verdict(
            EmbeddingRuntime::LocalFastembedOnnxMigraphx,
            "local-gpu-bge",
            &census,
        );
        assert!(
            verdict.is_err(),
            "перепись без узлов MIGraphX обязана отказать"
        );
        assert!(verdict.unwrap_err().contains("fell back to CPU"));
    }

    /// На CPU-профиле та же перепись — норма, а не отказ: вердикт обязан
    /// различать «просили GPU и не получили» и «GPU не просили».
    #[test]
    fn ep_verdict_says_nothing_about_cpu_profiles() {
        let census =
            ProviderCensus::from_profile_json(&profile_json(&[("Add_1", CPU_EP)])).unwrap();

        assert!(
            ep_census_verdict(
                EmbeddingRuntime::LocalFastembedOnnxCpu,
                "local-cpu-small",
                &census
            )
            .is_ok()
        );
    }

    /// Ручка пробы включается тем же словарём, что и фоновый синк: две
    /// переменные с разными «включающими» словами — источник «я же выставил».
    #[test]
    fn ep_census_env_shares_the_enabled_vocabulary() {
        assert!(!parse_enabled_env(None));
        assert!(!parse_enabled_env(Some("0")));
        assert!(parse_enabled_env(Some("1")));
        assert!(parse_enabled_env(Some(" ON\n")));
    }
}
