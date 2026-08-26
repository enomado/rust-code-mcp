//! Operational defaults for MCP server startup and automatic work.

use rmc_engine::embeddings::{
    CPU_EP, DIRECTML_EP, EmbeddingBackend, EmbeddingRuntime, MIGRAPHX_EP, ProviderCensus,
    probe_provider_census,
};
use std::sync::OnceLock;

pub const BACKGROUND_SYNC_ENV: &str = "RMC_BACKGROUND_SYNC";

pub const BACKGROUND_SYNC_ENABLED_VALUES: &str = "1/true/yes/on";

/// Knob for the startup probe 'the graph is actually computed on the execution provider':
/// `RMC_EP_CENSUS=1`.
///
/// # Why behind a knob and not always
/// The probe brings up a SEPARATE session with profiling (it cannot be enabled after the
/// session is built), i.e. it loads the model again, and on a cold kernel
/// cache it also pays the MIGraphX compilation (45–70 s). There is no reason to pay that on every
/// server startup just for diagnostics.
pub const EP_CENSUS_ENV: &str = "RMC_EP_CENSUS";

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

/// Parsing of a boolean environment knob: enabled only by an explicit word from
/// [`BACKGROUND_SYNC_ENABLED_VALUES`].
///
/// Shared by all such knobs on purpose: two variables that are enabled by DIFFERENT
/// words are a source of 'but I did set it, and it does not work'.
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

/// Startup EP probe, if it was requested via [`EP_CENSUS_ENV`].
///
/// Returns `Ok(None)` when the knob is not set, and `Ok(Some(census))` —
/// the census of nodes per provider, already written to the log.
///
/// # Why a refusal and not a warning
/// The knob is set with one question in mind: 'does the GPU really work?'. The class for
/// which this whole layer exists — the EP came up, but the graph was computed on CPU —
/// shows up ONLY as speed, so a warning in the log of a starting
/// server does not catch it: the server will run, and 'GPU enabled' remains a false
/// conclusion. So on a GPU profile a zero MIGraphX census is a startup error:
/// the verdict is machine-readable (exit code), not 'visible on screen'.
///
/// On a CPU profile the probe asserts nothing — it only prints the census:
/// 'how many nodes are on CPU' is not a refusal but a fact.
pub fn probe_ep_census_on_startup() -> Result<Option<String>, String> {
    if !parse_enabled_env(std::env::var(EP_CENSUS_ENV).ok().as_deref()) {
        return Ok(None);
    }

    let backend = automatic_embedding_backend();
    let profile = backend.profile.name();
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

/// Verdict on the census: whether it is acceptable for the requested profile.
///
/// Separated from the probe on purpose: the probe itself requires a GPU, ORT with MIGraphX and
/// a downloaded model, i.e. it can only be checked on a suitable machine. The decision
/// 'is this a refusal or normal', on the other hand, is a pure function and is gated by the regular suite.
pub(crate) fn ep_census_verdict(
    runtime: EmbeddingRuntime,
    profile: &str,
    census: &ProviderCensus,
) -> Result<(), String> {
    // The assertion is NOT about a ratio: MIGraphX does not take nodes one at a time, it cuts out
    // a subgraph and substitutes ONE fused node — on a healthy GPU path the census
    // looks like `MIGraphXExecutionProvider=1`. So there is exactly one threshold here:
    // the fused node is either there or not.
    //
    // The threshold is the same for both GPU runtimes, but each has ITS OWN provider:
    // asking for MIGraphX on the Windows profile is guaranteed to yield
    // zero and declare a refusal on a healthy machine.
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

/// Built with a GPU backend, while the automatic profile is a CPU one.
///
/// Not an error: a build feature does not promise that ORT with this EP is present at runtime, and
/// the CPU default honestly works everywhere — so [`DEFAULT_AUTOMATIC_EMBEDDING_PROFILE`]
/// stays CPU, and GPU is enabled explicitly.
///
/// But staying silent is not acceptable either. The mismatch costs ~80x on THE SAME model
/// (2.6-3.3 chunks/s vs 221-261 on migraphx), and in the logs it looks like
/// normal operation: no refusal, no warning — just slow. This is exactly how
/// a codex client without `env` ran background work on CPU for months until it was found via
/// 580% CPU. Forgetting the variable in a new client is easy, so the forgetfulness
/// must be VISIBLE at startup.
fn cpu_profile_on_gpu_build(compiled_backends: &[&str], automatic: &EmbeddingBackend) -> bool {
    !compiled_backends.is_empty()
        && matches!(automatic.runtime, EmbeddingRuntime::LocalFastembedOnnxCpu)
}

/// Text of the startup warning for [`cpu_profile_on_gpu_build`], or
/// `None` when there is nothing to warn about.
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

/// Whether this backend can compute in the BACKGROUND, without a human at the keyboard.
///
/// # What is decided here
///
/// Background sync is the only server work that nobody started
/// deliberately. So the question is not 'can the machine handle it' but 'will a silent
/// automaton survive this runtime failing'. Hence two boundaries, and they differ:
///
/// - **ONNX runtimes (CPU and local GPU) — yes.** It is one and the same light
///   model (bge-small, 384 dimensions) on one and the same graph, the only difference
///   being the execution provider. The local GPU is allowed in the background since
///   non-finite vectors from the ROCm EP were closed off by the gate in
///   `EmbeddingGenerator::guard_finite`: before the gate the automaton could silently
///   write NaNs into the index for months, indistinguishable from 'search got worse'.
/// - **Qwen3 on CUDA — no.** It is a 0.6B to 8B model: automatically starting
///   such a session every five minutes takes VRAM away from whoever is working at the machine
///   right now. The restriction here is about RESOURCES, not correctness,
///   so the NaN gate does not lift it; explicit commands with this profile
///   work as before.
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

    /// The ORT profile in the form the runtime writes it: each node has three events.
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

    /// Healthy GPU path: ONE fused MIGraphX node and nothing on CPU.
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

    /// The Windows GPU path is judged by ITS OWN provider.
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

    /// 🚨 Gate against the most likely mistake in this layer: judging the Windows
    /// profile by MIGraphX. The census is HEALTHY — the whole graph is on DirectML — and
    /// a verdict asking about the wrong provider would declare a refusal on a working
    /// machine. The reverse pair is checked too: a MIGraphX profile with only
    /// DirectML nodes is a refusal, not 'well, it is a GPU anyway'.
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
            "a MIGraphX profile must refuse a census without MIGraphX nodes"
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
            "a DirectML profile must refuse a census without DirectML nodes"
        );
    }

    /// The very class the knob was introduced for: the EP registered, but
    /// the graph was computed on CPU. There is NOT A SINGLE error — only speed.
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
            "a census without MIGraphX nodes must be refused"
        );
        assert!(verdict.unwrap_err().contains("fell back to CPU"));
    }

    /// On a CPU profile the same census is normal, not a refusal: the verdict must
    /// distinguish 'asked for GPU and did not get it' from 'did not ask for GPU'.
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

    /// The probe knob is enabled by the same vocabulary as background sync: two
    /// variables with different 'enabling' words are a source of 'but I did set it'.
    #[test]
    fn ep_census_env_shares_the_enabled_vocabulary() {
        assert!(!parse_enabled_env(None));
        assert!(!parse_enabled_env(Some("0")));
        assert!(parse_enabled_env(Some("1")));
        assert!(parse_enabled_env(Some(" ON\n")));
    }
}
