//! Text embedding backend backed by fastembed's ONNX path — CPU или AMD GPU.
//!
//! Обе плоскости отличаются РОВНО одним: списком execution provider'ов, который
//! уезжает в `TextInitOptions::with_execution_providers`. Сам fastembed про
//! GPU-EP ничего не знает и знать не должен — он принимает EP снаружи.

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

/// Список EP для AMD GPU.
///
/// # Почему тут только MIGraphX
/// ROCm EP в ONNX Runtime депрекейтнут, и в поставляемых сборках его физически
/// нет: рядом с `libonnxruntime.so` лежит только
/// `libonnxruntime_providers_migraphx.so`. Просить ROCm EP — значит получить
/// тихий откат на CPU.
///
/// # Оракул против тихого отката
/// `error_on_failure()` превращает «EP не поднялся» из молчаливой деградации в
/// ошибку инициализации. Без него сессия создаётся, числа считаются, и «GPU
/// работает» становится ложным выводом — ровно тот класс, которого этот код
/// обязан избегать.
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

/// Каталог кэша скомпилированных MIGraphX-ядер (`.mxr`).
///
/// # Почему это не тюнинг, а условие работоспособности
/// ONNX Runtime 1.28 пишет скомпилированную программу ДАЖЕ когда сохранение не
/// запрошено, и берёт каталог ТОЛЬКО из `ORT_MIGRAPHX_MODEL_CACHE_PATH`. Если
/// её нет, путь собирается из пустой строки, запись падает — и падает не
/// инициализация, а первый `run`, уже в рантайме. Поэтому переменная
/// выставляется здесь, до создания сессии, а не оставляется на совесть
/// вызывающего.
///
/// Соседние ручки на этот путь НЕ влияют, проверено поимённо: ни provider-опция
/// `migraphx_save_model_path` (ort передаёт её в устаревшей структуре
/// `OrtMIGraphXProviderOptions`, и ORT 1.28 её игнорирует), ни переменная
/// `ORT_MIGRAPHX_CACHE_PATH`, которую strings показывает в той же библиотеке.
///
/// # Почему каталог свой у каждой формы входа
/// Имя `.mxr` включает хэш графа, но при загрузке чужого файла первый `run`
/// после старта отдаёт результат ФОРМЫ ИЗ КЭША, а не фактического входа —
/// молча, без ошибки. Смешивать в одном каталоге программы разных форм нельзя:
/// это тихая порча первого батча. Компиляция одной формы стоит 45–70 с и ~200 МБ.
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
    // SAFETY: вызывается на пути инициализации эмбеддера, до создания ORT-сессии
    // и до появления фоновых потоков, которые могли бы читать окружение.
    unsafe { std::env::set_var(CACHE_ENV, &dir) };
    Ok(dir)
}
