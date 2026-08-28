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
            if let Err(err) = resolve_startup_profile(&requested) {
                panic!(
                    "{EMBEDDING_PROFILE_ENV}='{requested}' is not a usable embedding profile: {err}"
                );
            }
            requested
        })
        .as_str()
}

/// Разбор значения [`EMBEDDING_PROFILE_ENV`] в имя профиля.
///
/// Пустая строка и пробелы трактуются как «переменная не задана»: пустое
/// значение в обёртке запуска — обычная опечатка, и молча взять её за имя
/// профиля значило бы отказать в старте вместо разумного дефолта.
pub(crate) fn resolve_automatic_profile_name(env_value: Option<&str>) -> String {
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

pub(crate) fn automatic_embedding_backend() -> EmbeddingBackend {
    let profile = resolve_startup_profile(automatic_embedding_profile_name())
        .expect("automatic embedding profile is validated on first read");
    EmbeddingBackend::from_profile(profile)
}

/// Стартовая проба EP, если её попросили через [`EP_CENSUS_ENV`].
///
/// Возвращает `Ok(None)`, когда ручка не взведена, и `Ok(Some(census))` —
/// перепись узлов по провайдерам, уже записанную в лог.
///
/// # Почему отказ, а не предупреждение
/// Ручку взводят с одним вопросом: «GPU правда работает?». Класс, ради
/// которого весь этот слой существует — EP поднялся, но граф посчитан на CPU —
/// проявляется ТОЛЬКО скоростью, то есть предупреждение в логе стартующего
/// сервера его не ловит: сервер поедет, и «GPU включён» останется ложным
/// выводом. Поэтому на GPU-профиле нулевая перепись MIGraphX — ошибка старта:
/// вердикт машинно-читаем (код возврата), а не «видно на экране».
///
/// На CPU-профиле проба ничего не утверждает — только печатает перепись:
/// «сколько узлов на CPU» это не отказ, а факт.
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

/// Какие GPU-пути ВКОМПИЛИРОВАНЫ в этот бинарь — строкой для стартовой строки лога.
///
/// Печатается на старте потому, что «профиль просит GPU, а бинарь собран без
/// его фичи» — отказ, который иначе виден только скоростью: сервер поедет,
/// профиль резолвится, а считать будет CPU. Список именно бэкендов, а не одного
/// булева флага: их три (CUDA, MIGraphX, DirectML), и на разных машинах
/// компилируются разные.
pub fn gpu_backends_compiled() -> String {
    let backends: Vec<&str> = [
        ("cuda", rmc_engine::embeddings::CUDA_CAPABLE_FEATURES_COMPILED),
        (
            "migraphx",
            rmc_engine::embeddings::MIGRAPHX_FEATURES_COMPILED,
        ),
        (
            "directml",
            rmc_engine::embeddings::DIRECTML_FEATURES_COMPILED,
        ),
    ]
    .into_iter()
    .filter_map(|(name, compiled)| compiled.then_some(name))
    .collect();

    if backends.is_empty() {
        "none".to_string()
    } else {
        backends.join(", ")
    }
}

pub(crate) fn is_background_embedding_backend(backend: &EmbeddingBackend) -> bool {
    matches!(
        backend.runtime,
        EmbeddingRuntime::LocalFastembedOnnxCpu | EmbeddingRuntime::OpenRouter
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
