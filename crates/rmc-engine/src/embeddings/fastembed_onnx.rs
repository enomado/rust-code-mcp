//! Text embedding backend backed by fastembed's ONNX path — CPU или AMD GPU.
//!
//! Обе плоскости отличаются РОВНО одним: списком execution provider'ов, который
//! уезжает в `TextInitOptions::with_execution_providers`. Сам fastembed про
//! GPU-EP ничего не знает и знать не должен — он принимает EP снаружи.

use crate::embeddings::backend::{EmbeddingBackend, EmbeddingRuntime};
use crate::embeddings::profile::FastembedOnnxModel;
use crate::embeddings::{Embedding, EmbeddingError};
use fastembed::{EmbeddingModel, FixedBatchShape, TextEmbedding, TextInitOptions};
use std::sync::Mutex;

/// Высота батча, под которую компилируются MIGraphX-ядра.
///
/// # Почему форма постоянная
/// MIGraphX компилирует ядра ПОД ФОРМУ входа: каждая новая пара
/// (строк × длина) стоит 45–70 с компиляции и ~145–200 МБ в кэше `.mxr`.
/// Форма, которую fastembed отдаёт по умолчанию, плавает по обеим осям
/// (паддинг до самой длинной строки В БАТЧЕ + неполный последний батч), и одна
/// индексация 40 файлов породила 4 формы и 659 МБ кэша. Фиксация оставляет одну.
///
/// # Почему именно 32
/// Это и высота, на которой снят потолок GPU-пути (242 seq/s против 7.5 на CPU),
/// и дефолтный `gpu_batch_size` индексатора — то есть в типичном прогоне
/// добивать приходится только последний кусок. Число намеренно НЕ выводится из
/// входа: смысл в том, чтобы форма не зависела от того, сколько текстов пришло.
const GPU_BATCH_ROWS: usize = 32;

pub(super) struct FastembedOnnxEmbedder {
    inner: Mutex<TextEmbedding>,
    backend: EmbeddingBackend,
    dim: usize,
}

impl FastembedOnnxEmbedder {
    pub(super) fn new(backend: &EmbeddingBackend) -> Result<Self, EmbeddingError> {
        Self::new_inner(backend, None)
    }

    /// Тот же путь инициализации, но с включённым профилированием ORT.
    ///
    /// Профилирование НЕЛЬЗЯ включить после сборки сессии, поэтому и отдельный
    /// конструктор: тот же `backend`, тот же список EP, та же форма — иначе
    /// профиль описывал бы не ту сессию, которая работает в проде, и оракул
    /// гейтил бы собственную копию кода.
    ///
    /// `prefix` — префикс имени файла; ORT дописывает к нему отметку времени,
    /// фактический путь возвращает [`Self::end_profiling`].
    #[cfg(test)]
    pub(super) fn new_profiled(
        backend: &EmbeddingBackend,
        prefix: &std::path::Path,
    ) -> Result<Self, EmbeddingError> {
        Self::new_inner(backend, Some(prefix))
    }

    /// Закрыть профиль ORT и вернуть путь записанного файла.
    #[cfg(test)]
    pub(super) fn end_profiling(&self) -> Result<std::path::PathBuf, EmbeddingError> {
        let mut model = self.inner.lock().unwrap();
        model
            .end_profiling()
            .map(std::path::PathBuf::from)
            .map_err(|e| EmbeddingError::model_init(e.to_string()))
    }

    fn new_inner(
        backend: &EmbeddingBackend,
        profiling_prefix: Option<&std::path::Path>,
    ) -> Result<Self, EmbeddingError> {
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

        // Форма входа считается ДО создания сессии: от неё зависит каталог кэша
        // ядер, который надо выставить раньше, чем EP получит управление.
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
        if let Some(prefix) = profiling_prefix {
            options = options.with_profiling(prefix.to_path_buf());
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
/// # Почему каталог адресуется формой входа
/// Имя `.mxr` включает хэш графа, но НЕ различает формы: при загрузке файла от
/// ДРУГОЙ формы первый `run` после старта отдаёт результат формы ИЗ КЭША, а не
/// фактического входа — молча, без ошибки (воспроизведено: вход 16×512, выход
/// `[32, 512, 384]`). Поэтому каталог именуется моделью и формой: программы
/// разных форм физически не встречаются, и старый кэш смешанных форм не
/// подхватывается. Компиляция одной формы стоит 45–70 с и ~145–200 МБ.
///
/// Явный `ORT_MIGRAPHX_MODEL_CACHE_PATH` уважается, но трактуется как КОРЕНЬ:
/// подкаталог формы дописывается и к нему — инвариант «один каталог = одна
/// форма» не должен зависеть от того, задал ли кто-то переменную.
#[cfg(feature = "embeddings-migraphx")]
fn ensure_migraphx_kernel_cache(
    model: FastembedOnnxModel,
    shape: FixedBatchShape,
) -> Result<std::path::PathBuf, EmbeddingError> {
    const CACHE_ENV: &str = "ORT_MIGRAPHX_MODEL_CACHE_PATH";

    let shape_dir = format!("{}-{}x{}", model.display_name(), shape.rows, shape.seq_len);

    // Корень читается из окружения ОДИН раз за процесс и запоминается: ниже мы
    // сами пишем в ту же переменную путь подкаталога формы, и повторное чтение
    // приняло бы наш собственный ответ за корень — каталоги вложились бы друг в
    // друга при втором эмбеддере в том же процессе.
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
    // SAFETY: вызывается на пути инициализации эмбеддера, до создания ORT-сессии
    // и до появления фоновых потоков, которые могли бы читать окружение.
    unsafe { std::env::set_var(CACHE_ENV, &dir) };
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Тексты разной длины: важно, чтобы в батче встречались и короткие, и
    /// длинные — на них `BatchLongest` и постоянная форма дают РАЗНЫЙ паддинг,
    /// а значит разный путь через модель.
    fn corpus() -> Vec<String> {
        vec![
            "fn main() {}".to_string(),
            "pub struct ChunkId(pub u64);".to_string(),
            "async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Embedding>> { \
             let refs = texts.iter().map(String::as_str).collect::<Vec<_>>(); \
             self.inner.embed(&refs, None) }"
                .to_string(),
            "// комментарий".to_string(),
            "impl Display for EmbeddingError { fn fmt(&self, f: &mut Formatter) -> fmt::Result }"
                .to_string(),
        ]
    }

    /// Тесты ниже тянут ОДИН И ТОТ ЖЕ файл модели через hf-hub, а тот берёт
    /// файловый лок на блоб: два параллельных теста дерутся за него и один
    /// падает на «Lock acquisition failed». Загрузка модели поэтому
    /// сериализуется — это про кэш HF, а не про потокобезопасность fastembed.
    fn model_guard() -> std::sync::MutexGuard<'static, ()> {
        static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    /// Оракул постоянной формы входа: она не меняет ЧИСЛА и не меняет
    /// КОЛИЧЕСТВО эмбеддингов.
    ///
    /// Проверяются оба свойства, потому что ломаются они по-разному:
    /// - количество — если строки-добивки уехали наружу (не отрезаны по
    ///   `real_rows`); ловится на входе, чья длина НЕ кратна высоте батча;
    /// - числа — если паддинг до фиксированной длины начал влиять на результат
    ///   (например, attention-маска перестала гасить хвост). Эталон здесь —
    ///   тот же fastembed без фиксации формы, то есть сравнение честное:
    ///   меняется ровно одна вещь.
    ///
    /// Тест гоняется на CPU и потому не требует GPU, но требует скачанной
    /// модели и полноценного forward — отсюда `#[ignore]`, запускать явно:
    /// `cargo test -p rmc-engine --features embeddings fixed_batch_shape -- --ignored`
    #[test]
    #[ignore = "качает модель с HF и считает forward на CPU"]
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

        // rows=4 при 5 текстах: два батча, из них второй добит тремя строками.
        // Именно эта некратность и делает тест гейтом на отрезание хвоста.
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
            "строки-добивки уехали наружу: эмбеддингов больше, чем текстов"
        );
        for (idx, (want, got)) in expected.iter().zip(&actual).enumerate() {
            let sim = cosine(want, got);
            assert!(
                sim > 0.999,
                "текст #{idx}: постоянная форма изменила эмбеддинг (косинус {sim})"
            );
        }
    }

    /// Отказы, которые обязаны быть отказами, а не тихой сменой формы.
    #[test]
    #[ignore = "качает модель с HF"]
    fn fixed_batch_shape_rejects_impossible_shapes() {
        let _guard = model_guard();
        let options = || {
            TextInitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_max_length(512)
                .with_show_download_progress(false)
        };

        // Длина последовательности сверх предела усечения: обещанной формы не
        // получить — токенизатор всё равно обрежет.
        let err = TextEmbedding::try_new(options())
            .unwrap()
            .with_fixed_batch_shape(FixedBatchShape {
                rows: 32,
                seq_len: 1024,
            })
            .err()
            .expect("seq_len сверх предела усечения должен быть отказом");
        assert!(err.to_string().contains("truncation limit"), "{err}");

        let err = TextEmbedding::try_new(options())
            .unwrap()
            .with_fixed_batch_shape(FixedBatchShape {
                rows: 0,
                seq_len: 512,
            })
            .err()
            .expect("нулевая высота батча должна быть отказом");
        assert!(err.to_string().contains("non-zero"), "{err}");
    }

    /// Оракул GPU-пути: граф РЕАЛЬНО считается на MIGraphX, а не на CPU.
    ///
    /// # Что именно он ловит
    /// `error_on_failure()` на EP закрывает только «провайдер не поднялся».
    /// Класс «EP поднялся, но взял ноль узлов» проходит его насквозь: сессия
    /// жива, эмбеддинги считаются, отличается лишь скорость — то есть до этого
    /// теста деградация была видна только глазом и только в замере.
    ///
    /// # Почему утверждение про CPU-узлы, а не про долю
    /// MIGraphX не «берёт узлы по одному»: он вырезает подграф, компилирует его
    /// и подставляет ОДИН фьюженный узел. Замерено на этой сцене: здоровый
    /// GPU-путь даёт `MIGraphXExecutionProvider=1` и ноль CPU-узлов, тогда как
    /// тот же корпус на CPU-профиле — 365 узлов (см. позитивный контроль ниже).
    /// Доля тут поэтому не работает как метрика: одна единица «весит» весь
    /// граф. Утверждаем два свойства: фьюженный узел ЕСТЬ, и CPU не набрал
    /// заметного хвоста — то есть подграф не отгрызли по кусочку.
    ///
    /// Слак в 32 узла — не измеренная величина, а запас: шейповые операторы
    /// (Shape/Reshape/Cast) в принципе могут остаться снаружи подграфа, как это
    /// видно в python-плече на ROCm EP (там 4158 узлов на GPU и 48 на CPU — но
    /// ROCm EP не фьюзит, и картина узлов у него другая). Ниже 365 он на
    /// порядок, поэтому откат «граф вернулся на CPU» ловится с запасом.
    ///
    /// Требует карту, системный ORT с MIGraphX и скачанную модель, поэтому
    /// `#[ignore]`; первый прогон на холодном кэше ядер платит ~минуту
    /// компиляции. Запуск:
    /// `cargo test -p rmc-engine --features embeddings-migraphx migraphx_ep -- --ignored --nocapture`
    #[cfg(feature = "embeddings-migraphx")]
    #[test]
    #[ignore = "нужны AMD-карта, ORT с MIGraphX и скачанная модель"]
    fn migraphx_ep_actually_runs_the_graph() {
        use crate::embeddings::ep_census::{CPU_EP, MIGRAPHX_EP, ProviderCensus};

        let _guard = model_guard();
        let dir = tempfile::tempdir().unwrap();
        let backend = EmbeddingBackend::from_profile_name("local-gpu-bge").unwrap();

        let embedder =
            FastembedOnnxEmbedder::new_profiled(&backend, &dir.path().join("migraphx")).unwrap();

        // Профиль пуст, пока не было ни одного прогона: перепись должна
        // считаться по РАБОТЕ сессии, а не по факту её создания.
        let texts = corpus();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeddings = embedder.embed_documents(&refs).unwrap();
        assert_eq!(embeddings.len(), refs.len());

        let profile = embedder.end_profiling().unwrap();
        let census = ProviderCensus::from_profile_file(&profile).unwrap();
        eprintln!("узлы по провайдерам: {census}");

        assert!(
            census.nodes_on(MIGRAPHX_EP) > 0,
            "ни один узел не достался MIGraphX — тихий откат на CPU: {census}"
        );
        const CPU_TAIL_SLACK: usize = 32;
        assert!(
            census.nodes_on(CPU_EP) <= CPU_TAIL_SLACK,
            "на CPU осталось {} узлов (слак {CPU_TAIL_SLACK}) — подграф не ушёл на GPU целиком: {census}",
            census.nodes_on(CPU_EP),
        );
    }

    /// Позитивный контроль к оракулу выше: на CPU-профиле перепись обязана
    /// показать CPU и НОЛЬ узлов на MIGraphX.
    ///
    /// Без него «зелёный GPU-тест» ничего не доказывает: тест, который зелен и
    /// когда всё считается на GPU, и когда всё считается на CPU, не гейт, а
    /// украшение. Здесь та же машинерия (профиль → перепись) гоняется на
    /// заведомо CPU-сессии, и утверждение ровно обратное — так видно, что
    /// перепись РАЗЛИЧАЕТ два исхода, а не всегда говорит «да».
    ///
    /// GPU не нужен, нужна только скачанная модель — отсюда `#[ignore]`:
    /// `cargo test -p rmc-engine --features embeddings census_on_cpu -- --ignored --nocapture`
    #[test]
    #[ignore = "качает модель с HF и считает forward на CPU"]
    fn census_on_cpu_profile_sees_no_migraphx() {
        use crate::embeddings::ep_census::{CPU_EP, MIGRAPHX_EP, ProviderCensus};

        let _guard = model_guard();
        let dir = tempfile::tempdir().unwrap();
        let backend = EmbeddingBackend::from_profile_name("local-cpu-small").unwrap();

        let embedder =
            FastembedOnnxEmbedder::new_profiled(&backend, &dir.path().join("cpu")).unwrap();
        let texts = corpus();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        embedder.embed_documents(&refs).unwrap();

        let census = ProviderCensus::from_profile_file(&embedder.end_profiling().unwrap()).unwrap();
        eprintln!("узлы по провайдерам (CPU-профиль): {census}");
        assert_eq!(census.nodes_on(MIGRAPHX_EP), 0);
        assert!(census.nodes_on(CPU_EP) > 0, "{census}");
    }
}
