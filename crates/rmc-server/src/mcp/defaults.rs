//! Operational defaults for MCP server startup and automatic work.

use rmc_engine::embeddings::{EmbeddingBackend, EmbeddingRuntime};
use std::sync::OnceLock;

pub const BACKGROUND_SYNC_ENV: &str = "RMC_BACKGROUND_SYNC";
pub const BACKGROUND_SYNC_ENABLED_VALUES: &str = "1/true/yes/on";

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

pub fn parse_background_sync_env(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };

    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
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
            if let Err(err) = EmbeddingBackend::from_profile_name(&requested) {
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

pub(crate) fn automatic_embedding_backend() -> EmbeddingBackend {
    EmbeddingBackend::from_profile_name(automatic_embedding_profile_name())
        .expect("automatic embedding profile is validated on first read")
}

pub fn cuda_capable_features_compiled() -> bool {
    rmc_engine::embeddings::CUDA_CAPABLE_FEATURES_COMPILED
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
}
