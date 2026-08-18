//! Text embedding backend backed by fastembed's ONNX path — CPU or AMD GPU.
//!
//! The two planes differ in EXACTLY one thing: the list of execution providers that
//! goes into `TextInitOptions::with_execution_providers`. fastembed itself knows
//! nothing about GPU EPs and should not have to — it accepts the EPs from outside.

use crate::embeddings::backend::{EmbeddingBackend, EmbeddingRuntime};
use crate::embeddings::profile::FastembedOnnxModel;
use crate::embeddings::{Embedding, EmbeddingError};
use fastembed::{EmbeddingModel, FixedBatchShape, TextEmbedding, TextInitOptions};
use std::sync::Mutex;

/// Batch height that the MIGraphX kernels are compiled for.
///
/// # Why the shape is constant
/// MIGraphX compiles kernels FOR THE INPUT SHAPE: every new pair
/// (rows × length) costs 45–70 s of compilation and ~145–200 MB in the `.mxr` cache.
/// The shape fastembed produces by default floats along both axes
/// (padding to the longest row IN THE BATCH + an incomplete last batch), and a single
/// indexing run over 40 files produced 4 shapes and 659 MB of cache. Fixing it leaves one.
///
/// # Why exactly 32
/// It is both the height at which the GPU path's ceiling was measured (242 seq/s vs 7.5 on CPU)
/// and the indexer's default `gpu_batch_size` — i.e. in a typical run
/// only the last chunk needs padding. The number is deliberately NOT derived from the
/// input: the point is that the shape does not depend on how many texts arrived.
const GPU_BATCH_ROWS: usize = 32;

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

        // The input shape is computed BEFORE the session is created: the kernel cache
        // directory depends on it and must be set before the EP takes control.
        let shape = FixedBatchShape {
            rows: GPU_BATCH_ROWS,
            seq_len: backend.max_len,
        };

        let mut options = TextInitOptions::new(to_fastembed_model(model))
            .with_max_length(backend.max_len)
            .with_show_download_progress(false);
        if on_gpu {
            options = options.with_execution_providers(migraphx_execution_providers(model, shape)?);
        }
        let mut inner = TextEmbedding::try_new(options)
            .map_err(|e| EmbeddingError::model_init(e.to_string()))?;
        if on_gpu {
            inner = inner
                .with_fixed_batch_shape(shape)
                .map_err(|e| EmbeddingError::model_init(e.to_string()))?;
            tracing::info!(
                target: "embeddings::fastembed_onnx",
                rows = shape.rows,
                seq_len = shape.seq_len,
                "fixed MIGraphX input shape"
            );
        }

        Ok(Self {
            inner: Mutex::new(inner),
            backend: backend.clone(),
            dim: backend.dim(),
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbeddingError> {
        let mut model = self.inner.lock().unwrap();
        model
            .embed(texts, None)
            .map_err(|e| EmbeddingError::embed_failed(e.to_string()))
    }

    pub(super) fn embed_queries(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbeddingError> {
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
fn migraphx_execution_providers(
    model: FastembedOnnxModel,
    shape: FixedBatchShape,
) -> Result<Vec<ort::execution_providers::ExecutionProviderDispatch>, EmbeddingError> {
    ensure_migraphx_kernel_cache(model, shape)?;
    Ok(vec![
        ort::ep::migraphx::MIGraphX::default()
            .build()
            .error_on_failure(),
    ])
}

#[cfg(not(feature = "embeddings-migraphx"))]
fn migraphx_execution_providers(
    _model: FastembedOnnxModel,
    _shape: FixedBatchShape,
) -> Result<Vec<fastembed::ExecutionProviderDispatch>, EmbeddingError> {
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
/// # Why the directory is addressed by input shape
/// The `.mxr` name includes a graph hash but does NOT distinguish shapes: when loading a file from
/// a DIFFERENT shape, the first `run` after startup returns a result for the CACHED shape rather than
/// for the actual input — silently, with no error (reproduced: input 16×512, output
/// `[32, 512, 384]`). So the directory is named by model and shape: programs
/// of different shapes never physically meet, and an old cache of mixed shapes is not
/// picked up. Compiling one shape costs 45–70 s and ~145–200 MB.
///
/// An explicit `ORT_MIGRAPHX_MODEL_CACHE_PATH` is respected but treated as a ROOT:
/// the shape subdirectory is appended to it as well — the invariant 'one directory = one
/// shape' must not depend on whether someone set the variable.
#[cfg(feature = "embeddings-migraphx")]
fn ensure_migraphx_kernel_cache(
    model: FastembedOnnxModel,
    shape: FixedBatchShape,
) -> Result<std::path::PathBuf, EmbeddingError> {
    const CACHE_ENV: &str = "ORT_MIGRAPHX_MODEL_CACHE_PATH";

    let shape_dir = format!("{}-{}x{}", model.display_name(), shape.rows, shape.seq_len);

    // The root is read from the environment ONCE per process and memoized: below we
    // ourselves write the shape subdirectory path into the same variable, and re-reading
    // would take our own answer for the root — directories would nest inside each
    // other with a second embedder in the same process.
    static CACHE_ROOT: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let root = CACHE_ROOT
        .get_or_init(|| {
            std::env::var_os(CACHE_ENV)
                .filter(|v| !v.is_empty())
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    directories::ProjectDirs::from("", "", "rust-code-mcp")
                        .map(|d| d.cache_dir().join("migraphx"))
                })
        })
        .clone()
        .ok_or_else(|| {
            EmbeddingError::model_init("cannot resolve a cache directory for MIGraphX kernels")
        })?;
    let dir = root.join(shape_dir);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Texts of different lengths: the batch must contain both short and
    /// long ones — on those `BatchLongest` and the constant shape produce DIFFERENT padding,
    /// and hence a different path through the model.
    fn corpus() -> Vec<String> {
        vec![
            "fn main() {}".to_string(),
            "pub struct ChunkId(pub u64);".to_string(),
            "async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Embedding>> { \
             let refs = texts.iter().map(String::as_str).collect::<Vec<_>>(); \
             self.inner.embed(&refs, None) }"
                .to_string(),
            "// comment".to_string(),
            "impl Display for EmbeddingError { fn fmt(&self, f: &mut Formatter) -> fmt::Result }"
                .to_string(),
        ]
    }

    /// The tests below pull THE SAME model file through hf-hub, which takes
    /// a file lock on the blob: two parallel tests fight over it and one
    /// fails with 'Lock acquisition failed'. Model loading is therefore
    /// serialized — this is about the HF cache, not about fastembed thread-safety.
    fn model_guard() -> std::sync::MutexGuard<'static, ()> {
        static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    /// Oracle for the constant input shape: it changes neither the NUMBERS nor the
    /// COUNT of embeddings.
    ///
    /// Both properties are checked because they break in different ways:
    /// - count — if padding rows leaked out (were not cut off by
    ///   `real_rows`); caught on an input whose length is NOT a multiple of the batch height;
    /// - numbers — if padding to the fixed length started affecting the result
    ///   (e.g. the attention mask stopped masking out the tail). The reference here is
    ///   the same fastembed without a fixed shape, so the comparison is fair:
    ///   exactly one thing changes.
    ///
    /// The test runs on CPU and so needs no GPU, but it needs a downloaded
    /// model and a full forward pass — hence `#[ignore]`; run it explicitly:
    /// `cargo test -p rmc-engine --features embeddings fixed_batch_shape -- --ignored`
    #[test]
    #[ignore = "downloads the model from HF and runs forward on CPU"]
    fn fixed_batch_shape_preserves_embeddings() {
        let texts = corpus();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let _guard = model_guard();
        let options = || {
            TextInitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_max_length(512)
                .with_show_download_progress(false)
        };

        let mut baseline = TextEmbedding::try_new(options()).unwrap();
        let expected = baseline.embed(&refs, None).unwrap();

        // rows=4 with 5 texts: two batches, the second one padded with three rows.
        // This non-multiple is exactly what makes the test a gate on cutting off the tail.
        let shape = FixedBatchShape {
            rows: 4,
            seq_len: 512,
        };
        let mut fixed = TextEmbedding::try_new(options())
            .unwrap()
            .with_fixed_batch_shape(shape)
            .unwrap();
        assert_eq!(fixed.fixed_batch_shape(), Some(shape));
        let actual = fixed.embed(&refs, None).unwrap();

        assert_eq!(
            actual.len(),
            texts.len(),
            "padding rows leaked out: more embeddings than texts"
        );
        for (idx, (want, got)) in expected.iter().zip(&actual).enumerate() {
            let sim = cosine(want, got);
            assert!(
                sim > 0.999,
                "text #{idx}: the constant shape changed the embedding (cosine {sim})"
            );
        }
    }

    /// Refusals that must be refusals, not a silent change of shape.
    #[test]
    #[ignore = "downloads the model from HF"]
    fn fixed_batch_shape_rejects_impossible_shapes() {
        let _guard = model_guard();
        let options = || {
            TextInitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_max_length(512)
                .with_show_download_progress(false)
        };

        // Sequence length beyond the truncation limit: the promised shape cannot
        // be obtained — the tokenizer would truncate anyway.
        let err = TextEmbedding::try_new(options())
            .unwrap()
            .with_fixed_batch_shape(FixedBatchShape {
                rows: 32,
                seq_len: 1024,
            })
            .err()
            .expect("seq_len beyond the truncation limit must be refused");
        assert!(err.to_string().contains("truncation limit"), "{err}");

        let err = TextEmbedding::try_new(options())
            .unwrap()
            .with_fixed_batch_shape(FixedBatchShape {
                rows: 0,
                seq_len: 512,
            })
            .err()
            .expect("a zero batch height must be refused");
        assert!(err.to_string().contains("non-zero"), "{err}");
    }
}
