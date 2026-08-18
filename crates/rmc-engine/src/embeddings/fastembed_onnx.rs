//! Text embedding backend backed by fastembed's ONNX path — CPU or AMD GPU.
//!
//! The two planes differ in EXACTLY one thing: the list of execution providers that
//! goes into `TextInitOptions::with_execution_providers`. fastembed itself knows
//! nothing about GPU EPs and should not have to — it accepts the EPs from outside.

use crate::embeddings::backend::{EmbeddingBackend, EmbeddingRuntime};
use crate::embeddings::profile::FastembedOnnxModel;
use crate::embeddings::{Embedding, EmbeddingError};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::sync::Mutex;

pub(super) struct FastembedOnnxEmbedder {
    inner: Mutex<TextEmbedding>,
    backend: EmbeddingBackend,
    dim: usize,
}

impl FastembedOnnxEmbedder {
    pub(super) fn new(backend: &EmbeddingBackend) -> Result<Self, EmbeddingError> {
        if !backend.is_fastembed_onnx() {
            return Err(EmbeddingError::model_init(format!(
                "embedding profile `{}` is not a fastembed ONNX profile",
                backend.profile.name()
            )));
        }
        let model = backend.require_fastembed_onnx_model()?;
        let on_gpu = backend.runtime == EmbeddingRuntime::LocalFastembedOnnxMigraphx;

        tracing::info!(
            target: "embeddings::fastembed_onnx",
            profile = backend.profile.name(),
            model = model.display_name(),
            max_len = backend.max_len,
            on_gpu,
            "loading fastembed ONNX model"
        );

        let mut options = TextInitOptions::new(to_fastembed_model(model))
            .with_max_length(backend.max_len)
            .with_show_download_progress(false);
        if on_gpu {
            options = options.with_execution_providers(migraphx_execution_providers()?);
        }
        let inner = TextEmbedding::try_new(options)
            .map_err(|e| EmbeddingError::model_init(e.to_string()))?;

        Ok(Self {
            inner: Mutex::new(inner),
            backend: backend.clone(),
            dim: backend.dim(),
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn embed_documents(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let mut model = self.inner.lock().unwrap();
        model
            .embed(texts, None)
            .map_err(|e| EmbeddingError::embed_failed(e.to_string()))
    }

    pub(super) fn embed_queries(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|text| self.backend.format_query(text))
            .collect();
        let refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        self.embed_documents(&refs)
    }
}

fn to_fastembed_model(model: FastembedOnnxModel) -> EmbeddingModel {
    match model {
        FastembedOnnxModel::BgeSmallEnV15Q => EmbeddingModel::BGESmallENV15Q,
        FastembedOnnxModel::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
    }
}

/// EP list for AMD GPU.
///
/// # Why only MIGraphX here
/// The ROCm EP is deprecated in ONNX Runtime, and the shipped builds physically
/// do not contain it: next to `libonnxruntime.so` there is only
/// `libonnxruntime_providers_migraphx.so`. Asking for the ROCm EP means getting
/// a silent fallback to CPU.
///
/// # Oracle against a silent fallback
/// `error_on_failure()` turns 'the EP did not come up' from silent degradation into
/// an initialization error. Without it the session is created, numbers get computed, and 'the GPU
/// works' becomes a false conclusion — exactly the class of error this code
/// must avoid.
#[cfg(feature = "embeddings-migraphx")]
fn migraphx_execution_providers()
-> Result<Vec<ort::execution_providers::ExecutionProviderDispatch>, EmbeddingError> {
    ensure_migraphx_kernel_cache()?;
    Ok(vec![
        ort::ep::migraphx::MIGraphX::default()
            .build()
            .error_on_failure(),
    ])
}

#[cfg(not(feature = "embeddings-migraphx"))]
fn migraphx_execution_providers()
-> Result<Vec<fastembed::ExecutionProviderDispatch>, EmbeddingError> {
    Err(EmbeddingError::model_init(
        "rmc-engine was built without the `embeddings-migraphx` feature; \
         rebuild with --features migraphx to use GPU embedding profiles",
    ))
}

/// Directory of the cache of compiled MIGraphX kernels (`.mxr`).
///
/// # Why this is not tuning but a precondition for working at all
/// ONNX Runtime 1.28 writes the compiled program EVEN when saving was not
/// requested, and takes the directory ONLY from `ORT_MIGRAPHX_MODEL_CACHE_PATH`. If
/// it is unset, the path is built from an empty string and the write fails — and what fails is not
/// initialization but the first `run`, already at runtime. That is why the variable
/// is set here, before the session is created, rather than left to the
/// caller's discretion.
///
/// Neighbouring knobs do NOT affect this path, checked one by one: neither the provider option
/// `migraphx_save_model_path` (ort passes it in the deprecated struct
/// `OrtMIGraphXProviderOptions`, and ORT 1.28 ignores it), nor the variable
/// `ORT_MIGRAPHX_CACHE_PATH`, which strings shows in the same library.
///
/// # Why each input shape gets its own directory
/// The `.mxr` name includes a graph hash, but when a foreign file is loaded the first `run`
/// after startup returns a result for the CACHED SHAPE rather than for the actual input —
/// silently, with no error. Programs of different shapes must not be mixed in one directory:
/// that silently corrupts the first batch. Compiling one shape costs 45–70 s and ~200 MB.
#[cfg(feature = "embeddings-migraphx")]
fn ensure_migraphx_kernel_cache() -> Result<std::path::PathBuf, EmbeddingError> {
    const CACHE_ENV: &str = "ORT_MIGRAPHX_MODEL_CACHE_PATH";

    if let Some(dir) = std::env::var_os(CACHE_ENV).filter(|v| !v.is_empty()) {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).map_err(|e| {
            EmbeddingError::model_init(format!(
                "cannot create MIGraphX kernel cache at {}: {e}",
                dir.display()
            ))
        })?;
        return Ok(dir);
    }
    let dir = directories::ProjectDirs::from("", "", "rust-code-mcp")
        .map(|d| d.cache_dir().join("migraphx"))
        .ok_or_else(|| {
            EmbeddingError::model_init("cannot resolve a cache directory for MIGraphX kernels")
        })?;
    std::fs::create_dir_all(&dir).map_err(|e| {
        EmbeddingError::model_init(format!(
            "cannot create MIGraphX kernel cache at {}: {e}",
            dir.display()
        ))
    })?;
    tracing::info!(
        target: "embeddings::fastembed_onnx",
        cache = %dir.display(),
        "MIGraphX kernel cache"
    );
    // SAFETY: called on the embedder initialization path, before the ORT session is created
    // and before any background threads exist that could read the environment.
    unsafe { std::env::set_var(CACHE_ENV, &dir) };
    Ok(dir)
}
