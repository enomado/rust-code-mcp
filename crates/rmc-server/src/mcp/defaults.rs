//! Operational defaults for MCP server startup and automatic work.

use rmc_engine::embeddings::{EmbeddingBackend, EmbeddingRuntime};
use std::sync::OnceLock;

pub const BACKGROUND_SYNC_ENV: &str = "RMC_BACKGROUND_SYNC";
pub const BACKGROUND_SYNC_ENABLED_VALUES: &str = "1/true/yes/on";

/// The profile the server computes embeddings with when the caller did not name one.
///
/// The default is CPU: it always builds and works on any machine. The GPU profile
/// requires both a build feature (`--features migraphx`) and a system ONNX Runtime with
/// that EP, so it is enabled EXPLICITLY, via the [`EMBEDDING_PROFILE_ENV`] variable, rather than
/// by guessing from what is available on the machine.
pub const DEFAULT_AUTOMATIC_EMBEDDING_PROFILE: &str = "local-cpu-small";

/// Knob for choosing the default profile: `RMC_EMBEDDING_PROFILE=local-gpu-bge`.
///
/// ⚠ The profile is part of the embedder identity, and that is part of the collection path: changing
/// the profile means a DIFFERENT index that has to be rebuilt from scratch.
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

/// Default profile name: from [`EMBEDDING_PROFILE_ENV`], otherwise
/// [`DEFAULT_AUTOMATIC_EMBEDDING_PROFILE`].
///
/// Read ONCE per process: the default profile is a property of the server
/// launch, not of an individual request, and it must not change on the fly (otherwise
/// half the index would arrive from one embedder and half from another).
pub fn automatic_embedding_profile_name() -> &'static str {
    static PROFILE: OnceLock<String> = OnceLock::new();
    PROFILE
        .get_or_init(|| {
            let requested = resolve_automatic_profile_name(
                std::env::var(EMBEDDING_PROFILE_ENV).ok().as_deref(),
            );

            // Fail-fast: a typo in the profile name must not silently fall back to the
            // CPU default — otherwise 'GPU enabled' would turn out to be untrue, and one could
            // only notice it by the speed.
            if let Err(err) = EmbeddingBackend::from_profile_name(&requested) {
                panic!(
                    "{EMBEDDING_PROFILE_ENV}='{requested}' is not a usable embedding profile: {err}"
                );
            }
            requested
        })
        .as_str()
}

/// Parsing of the [`EMBEDDING_PROFILE_ENV`] value into a profile name.
///
/// An empty string and whitespace are treated as 'variable not set': an empty
/// value in a launch wrapper is a common typo, and silently taking it as the profile
/// name would mean refusing to start instead of using a sensible default.
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
        // Surrounding whitespace comes from launch wrappers, it is not part of the name.
        assert_eq!(
            resolve_automatic_profile_name(Some(" local-gpu-bge\n")),
            "local-gpu-bge"
        );
    }

    /// The default profile must be one that always builds and is suitable for
    /// background work: the server starts with it on any machine.
    #[test]
    fn default_profile_is_a_cpu_background_capable_backend() {
        let backend = EmbeddingBackend::from_profile_name(DEFAULT_AUTOMATIC_EMBEDDING_PROFILE)
            .expect("default profile resolves");

        assert_eq!(backend.profile.name(), "local-cpu-small");
        assert!(is_background_embedding_backend(&backend));
    }

    /// A name that is not among the profiles must be REJECTED, not silently
    /// fall back to the CPU default.
    #[test]
    fn unknown_profile_name_is_rejected() {
        let requested = resolve_automatic_profile_name(Some("local-gpu-bge-typo"));

        assert!(
            EmbeddingBackend::from_profile_name(&requested).is_err(),
            "a typo in the profile name must not resolve"
        );
    }
}
