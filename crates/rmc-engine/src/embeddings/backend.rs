//! Embedding backend runtime wiring.
//!
//! [`EmbeddingBackend`] wraps an [`EmbeddingProfile`] with per-instance
//! runtime state (`max_len` override, `force_cpu`) and exposes the
//! stable identity string used in cache paths and `EMBEDDER_VERSION`.
//!
//! The profile data model itself — `EmbeddingProfile`, `QueryPolicy`,
//! `LocalLoaderSpec`, `FastembedOnnxModel`, `Qwen3Variant`, and the
//! built-in profile registry — lives in [`super::profile`].

use super::batching::{BatchRows, FixedInputShape};
use super::error::EmbeddingError;
use super::identity::EmbeddingIdentity;
use super::profile::{
    EmbeddingProfile, FastembedOnnxModel, LocalLoaderSpec, QueryPolicy, Qwen3Variant,
};
use super::util::arc;

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
/// Это высота, на которой снят потолок GPU-пути (242 seq/s против 7.5 на CPU).
/// Число намеренно НЕ выводится из входа: смысл в том, чтобы форма не зависела
/// от того, сколько текстов пришло.
///
/// Совпадение с дефолтным `gpu_batch_size` индексатора больше НЕ несёт нагрузки:
/// при постоянной форме индексатор берёт высоту отсюда, а свою настройку не
/// применяет вовсе.
const GPU_BATCH_ROWS: BatchRows = BatchRows(32);

/// Cross-crate embedding runtime boundary.
///
/// Indexing, graph, and server crates use this type when they need a concrete
/// embedding runtime choice, cache identity, or dimension contract. Backend
/// construction and identity parsing remain engine-owned so higher crates do
/// not duplicate runtime-specific policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingBackend {
    pub profile: EmbeddingProfile,
    pub runtime: EmbeddingRuntime,
    pub max_len: usize,
    /// Off by default. Set only for CI/benchmark runs. Enabling this
    /// emits a warn! on every construction.
    pub force_cpu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingRuntime {
    LocalQwen3CandleCuda,
    LocalFastembedOnnxCpu,
    /// Тот же ONNX-граф, что и у `LocalFastembedOnnxCpu`, но исполняемый на AMD
    /// GPU через MIGraphX EP.
    ///
    /// Отдельный вариант, а не флаг у CPU-рантайма, ПОТОМУ ЧТО он попадает в
    /// `EmbeddingIdentity` и, значит, разделяет индексы. Разделение здесь не
    /// перестраховка: GPU-профиль гоняет fp32-модель, а CPU-профиль —
    /// int8-квантованную, то есть векторы РАЗНЫЕ, и смешивать их в одном
    /// индексе нельзя.
    LocalFastembedOnnxMigraphx,
    /// Тот же ONNX-граф, исполняемый на GPU через **DirectML EP** — путь ВИНДЫ.
    ///
    /// Отдельный вариант по той же причине, что и MIGraphX (он в
    /// `EmbeddingIdentity` ⇒ разделяет индексы), и вдобавок потому, что это
    /// РАЗНЫЕ рантаймы с разной арифметикой: DirectML считает fp16 там, где
    /// MIGraphX считает fp32, так что векторы двух GPU-путей смешивать нельзя
    /// не меньше, чем векторы GPU и CPU.
    ///
    /// Почему не MIGraphX на винде: его там нет вообще — из ROCm под Windows
    /// портирован не весь стек, MIGraphX в портированное не входит.
    LocalFastembedOnnxDirectml,
    OpenRouter,
}

impl Default for EmbeddingBackend {
    fn default() -> Self {
        let profile = EmbeddingProfile::parse("local-gpu-small")
            .expect("built-in default embedding profile exists");
        Self::from_profile(profile)
    }
}

impl EmbeddingBackend {
    pub fn from_profile(profile: EmbeddingProfile) -> Self {
        Self {
            runtime: profile.runtime,
            max_len: profile.max_len,
            profile,
            force_cpu: false,
        }
    }

    pub fn from_profile_name(name: &str) -> Result<Self, EmbeddingError> {
        let profile = EmbeddingProfile::parse(name).map_err(EmbeddingError::model_init)?;
        Ok(Self::from_profile(profile))
    }

    pub fn from_qwen3_variant(variant: Qwen3Variant) -> Self {
        let profile = EmbeddingProfile::built_in_profiles()
            .iter()
            .find(|profile| profile.local_loader == Some(LocalLoaderSpec::Qwen3(variant)))
            .cloned()
            .expect("built-in Qwen3 embedding profile exists");
        Self::from_profile(profile)
    }

    pub fn dim(&self) -> usize {
        self.profile.dim
    }

    pub fn model_id(&self) -> &str {
        self.profile.model_id.as_ref()
    }

    pub fn tokenizer_model_id(&self) -> &str {
        self.profile
            .tokenizer_model_id
            .as_deref()
            .unwrap_or_else(|| self.model_id())
    }

    pub fn model_display_name(&self) -> &str {
        match self.profile.local_loader {
            Some(LocalLoaderSpec::Qwen3(variant)) => variant.display_name(),
            Some(LocalLoaderSpec::FastembedOnnx(model)) => model.display_name(),
            None => self.model_id(),
        }
    }

    pub fn qwen3_variant(&self) -> Option<Qwen3Variant> {
        match self.profile.local_loader {
            Some(LocalLoaderSpec::Qwen3(variant)) => Some(variant),
            _ => None,
        }
    }

    pub fn require_qwen3_variant(&self) -> Result<Qwen3Variant, EmbeddingError> {
        self.qwen3_variant().ok_or_else(|| {
            EmbeddingError::model_init(format!(
                "embedding profile `{}` does not use the local Qwen3 runtime",
                self.profile.name()
            ))
        })
    }

    pub fn fastembed_onnx_model(&self) -> Option<FastembedOnnxModel> {
        match self.profile.local_loader {
            Some(LocalLoaderSpec::FastembedOnnx(model)) => Some(model),
            _ => None,
        }
    }

    /// Идёт ли этот бэкенд через fastembed/ONNX — на CPU или на GPU.
    pub fn is_fastembed_onnx(&self) -> bool {
        matches!(
            self.runtime,
            EmbeddingRuntime::LocalFastembedOnnxCpu
                | EmbeddingRuntime::LocalFastembedOnnxMigraphx
                | EmbeddingRuntime::LocalFastembedOnnxDirectml
        )
    }

    pub fn require_fastembed_onnx_model(&self) -> Result<FastembedOnnxModel, EmbeddingError> {
        self.fastembed_onnx_model().ok_or_else(|| {
            EmbeddingError::model_init(format!(
                "embedding profile `{}` does not use the fastembed ONNX CPU runtime",
                self.profile.name()
            ))
        })
    }

    /// Постоянная форма входа, если рантайм этого профиля её ТРЕБУЕТ.
    ///
    /// # Единственный источник формы
    /// Форму читают двое: загрузчик модели (компилирует под неё ядра и
    /// адресует ею каталог кэша) и индексатор (режет ею входы). Пока каждый
    /// знал свою константу, их согласованность держалась на совпадении
    /// дефолтов — см. [`super::batching::BatchingPolicy`]. Теперь величину
    /// объявляет ОДНА сторона, вторая её принимает.
    ///
    /// `None` — не «форма неизвестна», а «рантайму форма не нужна»: на CPU и у
    /// API-моделей паддинг до самой длинной строки в батче бесплатен.
    pub fn fixed_input_shape(&self) -> Option<FixedInputShape> {
        match self.runtime {
            // MIGraphX компилирует ядра под форму; DirectML ядра не компилирует,
            // но так же перестраивает граф на каждую новую форму, а его
            // собственная дока прямо говорит, что EP работает лучше всего,
            // когда размеры входов известны на создании сессии. Обоим форму
            // ОБЪЯВЛЯЕМ — разница только в цене её нарушения.
            EmbeddingRuntime::LocalFastembedOnnxMigraphx
            | EmbeddingRuntime::LocalFastembedOnnxDirectml => Some(FixedInputShape {
                rows: GPU_BATCH_ROWS,
                seq_len: self.max_len,
            }),
            EmbeddingRuntime::LocalQwen3CandleCuda
            | EmbeddingRuntime::LocalFastembedOnnxCpu
            | EmbeddingRuntime::OpenRouter => None,
        }
    }

    /// Идёт ли этот бэкенд через fastembed/ONNX НА GPU.
    ///
    /// Собрано в одном месте нарочно: GPU-рантаймов теперь два (MIGraphX на
    /// линуксе, DirectML на винде), и любое `== LocalFastembedOnnxMigraphx` в
    /// смысле «это GPU» с появлением второго стало БАГОМ — молча отвечало бы
    /// «нет» на виндовом пути.
    pub fn is_fastembed_onnx_gpu(&self) -> bool {
        matches!(
            self.runtime,
            EmbeddingRuntime::LocalFastembedOnnxMigraphx
                | EmbeddingRuntime::LocalFastembedOnnxDirectml
        )
    }

    pub fn format_query(&self, text: &str) -> String {
        self.profile.query_policy.format_query(text)
    }

    /// Default cosine-similarity cutoff for the `semantic_overlaps`
    /// duplicate-detection audit.
    ///
    /// Cosine-similarity score distributions are model-specific, so a fixed
    /// cutoff is only meaningful relative to the model that produced the
    /// vectors. This derives the default from the active model instead of
    /// leaving a bare literal at the call site; callers may always pass an
    /// explicit threshold to override it.
    pub fn semantic_overlap_threshold(&self) -> f32 {
        match self.profile.local_loader {
            // Qwen3 code-embedding family: instruction-tuned, related code
            // clusters tightly at high cosine similarity.
            Some(LocalLoaderSpec::Qwen3(_)) => 0.85,
            // BGE general-purpose sentence embeddings sit on a lower
            // similarity scale than instruction-tuned code embeddings.
            Some(LocalLoaderSpec::FastembedOnnx(_)) => 0.80,
            // API models have no local loader. The built-in OpenRouter
            // Qwen3 model shares the Qwen3 scale; other API models (e.g.
            // OpenAI text-embedding-3, whose similarity range is markedly
            // compressed) should be given an explicit threshold until a
            // value is measured for them.
            None => 0.85,
        }
    }

    /// Stable string used in cache paths and EMBEDDER_VERSION.
    pub fn identity(&self) -> String {
        EmbeddingIdentity {
            runtime: self.runtime,
            model_id: self.model_id().to_string(),
            dim: self.dim(),
            max_len: self.max_len,
            query: self.profile.query_policy.encode_tag(),
        }
        .encode()
    }

    /// Parse an `identity()` string back into an `EmbeddingBackend`.
    ///
    /// Used to reconcile the embedder recorded in a vector store's
    /// `metadata.json` with the embedder a search-time caller wants.
    pub fn from_identity(s: &str) -> Result<Self, EmbeddingError> {
        if s.starts_with("emb;") {
            return Self::from_v2_identity(s);
        }

        Self::from_legacy_identity(s)
    }

    fn from_v2_identity(s: &str) -> Result<Self, EmbeddingError> {
        let identity = EmbeddingIdentity::decode(s).map_err(EmbeddingError::invalid_identity)?;
        let query_policy =
            QueryPolicy::decode_tag(&identity.query).map_err(EmbeddingError::invalid_identity)?;

        match identity.runtime {
            EmbeddingRuntime::OpenRouter => {
                let mut profile = EmbeddingProfile::built_in_api_for_identity(
                    EmbeddingRuntime::OpenRouter,
                    &identity.model_id,
                )
                .unwrap_or_else(|| EmbeddingProfile {
                    name: arc(&format!("openrouter:{}", identity.model_id)),
                    runtime: EmbeddingRuntime::OpenRouter,
                    model_id: arc(&identity.model_id),
                    tokenizer_model_id: None,
                    dim: identity.dim,
                    max_len: identity.max_len,
                    query_policy: query_policy.clone(),
                    chunk_target_tokens: 768,
                    chunk_hard_max_tokens: 1024,
                    local_loader: None,
                });
                profile.dim = identity.dim;
                profile.max_len = identity.max_len;
                profile.query_policy = query_policy;
                profile.local_loader = None;
                Ok(Self::from_profile(profile))
            }
            runtime => {
                let mut profile = EmbeddingProfile::built_in_local_for_identity(
                    runtime,
                    &identity.model_id,
                )
                .ok_or_else(|| {
                    EmbeddingError::invalid_identity(format!(
                        "local embedding identity references unknown built-in model `{}` for runtime {:?}; \
                         run `clear_cache` for this directory to discard the stale or foreign index",
                        identity.model_id, runtime
                    ))
                })?;
                if identity.dim != profile.dim {
                    return Err(EmbeddingError::invalid_identity(format!(
                        "dim `{}` does not match built-in profile `{}` (expected {}) in `{}`",
                        identity.dim,
                        profile.name(),
                        profile.dim,
                        s
                    )));
                }
                profile.query_policy = query_policy;
                let mut backend = Self::from_profile(profile);
                backend.max_len = identity.max_len;
                Ok(backend)
            }
        }
    }

    fn from_legacy_identity(s: &str) -> Result<Self, EmbeddingError> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 5 {
            return Err(EmbeddingError::invalid_identity(format!(
                "expected 5 colon-separated fields, got {}: `{}`",
                parts.len(),
                s
            )));
        }

        let mut backend = match parts[0] {
            "fastembed-candle" => {
                if parts[4] != "v2" {
                    return Err(EmbeddingError::invalid_identity(format!(
                        "unsupported identity schema version `{}` in `{}`",
                        parts[4], s
                    )));
                }
                let variant = match parts[1] {
                    "Qwen3-Embedding-0.6B" => Qwen3Variant::Embedding0_6B,
                    "Qwen3-Embedding-4B" => Qwen3Variant::Embedding4B,
                    "Qwen3-Embedding-8B" => Qwen3Variant::Embedding8B,
                    other => {
                        return Err(EmbeddingError::invalid_identity(format!(
                            "unknown Qwen3 variant `{}` in `{}`",
                            other, s
                        )));
                    }
                };
                Self::from_qwen3_variant(variant)
            }
            "fastembed-onnx-cpu" => {
                if parts[4] != "v1" {
                    return Err(EmbeddingError::invalid_identity(format!(
                        "unsupported identity schema version `{}` in `{}`",
                        parts[4], s
                    )));
                }
                match parts[1] {
                    "BGESmallENV15Q" => Self::from_profile_name("local-cpu-small")?,
                    other => {
                        return Err(EmbeddingError::invalid_identity(format!(
                            "unknown ONNX model `{}` in `{}`",
                            other, s
                        )));
                    }
                }
            }
            "openrouter" => {
                if parts[4] != "v1" {
                    return Err(EmbeddingError::invalid_identity(format!(
                        "unsupported identity schema version `{}` in `{}`",
                        parts[4], s
                    )));
                }
                match parts[1] {
                    "qwen/qwen3-embedding-8b" => Self::from_profile_name("openrouter-qwen3-8b")?,
                    other => {
                        return Err(EmbeddingError::invalid_identity(format!(
                            "unknown OpenRouter model `{}` in `{}`",
                            other, s
                        )));
                    }
                }
            }
            other => {
                return Err(EmbeddingError::invalid_identity(format!(
                    "unexpected backend prefix `{}` in `{}`",
                    other, s
                )));
            }
        };

        let dim_str = parts[2].strip_prefix("dim").ok_or_else(|| {
            EmbeddingError::invalid_identity(format!("missing `dim` prefix in `{}`", s))
        })?;
        let dim: usize = dim_str.parse().map_err(|e| {
            EmbeddingError::invalid_identity(format!(
                "failed to parse dim `{}` in `{}`: {}",
                dim_str, s, e
            ))
        })?;
        if dim != backend.dim() {
            return Err(EmbeddingError::invalid_identity(format!(
                "dim `{}` does not match profile `{}` (expected {}) in `{}`",
                dim,
                backend.profile.name(),
                backend.dim(),
                s
            )));
        }

        let max_str = parts[3].strip_prefix("max").ok_or_else(|| {
            EmbeddingError::invalid_identity(format!("missing `max` prefix in `{}`", s))
        })?;
        backend.max_len = max_str.parse().map_err(|e| {
            EmbeddingError::invalid_identity(format!(
                "failed to parse max_len `{}` in `{}`: {}",
                max_str, s, e
            ))
        })?;

        Ok(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::super::profile::QWEN3_CODE_QUERY_PREFIX;
    use super::*;

    fn profile(name: &str) -> EmbeddingProfile {
        EmbeddingProfile::parse(name).unwrap()
    }

    #[test]
    fn default_backend_dim_is_1024() {
        assert_eq!(EmbeddingBackend::default().dim(), 1024);
    }

    #[test]
    fn default_backend_identity_uses_v2_codec() {
        let identity = EmbeddingBackend::default().identity();
        let decoded = EmbeddingIdentity::decode(&identity).unwrap();

        assert!(identity.starts_with("emb;v=2;"));
        assert_eq!(decoded.runtime, EmbeddingRuntime::LocalQwen3CandleCuda);
        assert_eq!(decoded.model_id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(decoded.dim, 1024);
        assert_eq!(decoded.max_len, 1024);
    }

    #[test]
    fn profile_dimensions_match_expected_values() {
        assert_eq!(
            EmbeddingBackend::from_profile(profile("local-cpu-small")).dim(),
            384
        );
        assert_eq!(
            EmbeddingBackend::from_profile(profile("openrouter-qwen3-8b")).dim(),
            4096
        );
        assert_eq!(Qwen3Variant::Embedding4B.dim(), 2560);
        assert_eq!(Qwen3Variant::Embedding8B.dim(), 4096);
    }

    #[test]
    fn identities_are_unique_by_profile() {
        let mut identities = std::collections::HashSet::new();

        for profile in EmbeddingProfile::built_in_profiles().iter().cloned() {
            assert!(identities.insert(EmbeddingBackend::from_profile(profile).identity()));
        }
    }

    #[test]
    fn query_policy_is_profile_aware() {
        assert_eq!(
            EmbeddingBackend::from_profile(profile("local-gpu-small")).format_query("find parser"),
            "Instruct: Given a code search query, retrieve relevant code\nQuery: find parser"
        );
        assert_eq!(
            EmbeddingBackend::from_profile(profile("local-cpu-small")).format_query("find parser"),
            "Represent this sentence for searching relevant passages: find parser"
        );
    }

    #[test]
    fn from_identity_roundtrip_default() {
        let original = EmbeddingBackend::default();
        let parsed = EmbeddingBackend::from_identity(&original.identity()).unwrap();
        assert_eq!(parsed.profile, original.profile);
        assert_eq!(parsed.max_len, original.max_len);
    }

    #[test]
    fn from_identity_roundtrip_4b() {
        let original = EmbeddingBackend::from_qwen3_variant(Qwen3Variant::Embedding4B);
        let parsed = EmbeddingBackend::from_identity(&original.identity()).unwrap();
        assert_eq!(parsed.profile, original.profile);
        assert_eq!(parsed.max_len, original.max_len);
    }

    #[test]
    fn from_identity_roundtrip_cpu_profile() {
        let original = EmbeddingBackend::from_profile(profile("local-cpu-small"));
        let parsed = EmbeddingBackend::from_identity(&original.identity()).unwrap();
        assert_eq!(parsed.profile, original.profile);
        assert_eq!(parsed.max_len, original.max_len);
    }

    #[test]
    fn from_identity_roundtrip_openrouter_profile() {
        let original = EmbeddingBackend::from_profile(profile("openrouter-qwen3-8b"));
        let parsed = EmbeddingBackend::from_identity(&original.identity()).unwrap();
        assert_eq!(parsed.runtime, original.runtime);
        assert_eq!(parsed.model_id(), original.model_id());
        assert_eq!(parsed.dim(), original.dim());
        assert_eq!(parsed.max_len, original.max_len);
        assert_eq!(parsed.profile.query_policy, original.profile.query_policy);
    }

    #[test]
    fn from_identity_roundtrips_all_built_in_profiles() {
        for profile in EmbeddingProfile::built_in_profiles().iter().cloned() {
            let original = EmbeddingBackend::from_profile(profile);
            let parsed = EmbeddingBackend::from_identity(&original.identity()).unwrap();

            assert_eq!(parsed.runtime, original.runtime);
            assert_eq!(parsed.model_id(), original.model_id());
            assert_eq!(parsed.dim(), original.dim());
            assert_eq!(parsed.max_len, original.max_len);
            assert_eq!(parsed.profile.query_policy, original.profile.query_policy);
        }
    }

    #[test]
    fn from_identity_roundtrips_api_model_ids_with_reserved_chars() {
        let profile = EmbeddingProfile {
            name: arc("dynamic-openrouter"),
            runtime: EmbeddingRuntime::OpenRouter,
            model_id: arc("provider/model:revision"),
            tokenizer_model_id: None,
            dim: 1536,
            max_len: 8192,
            query_policy: QueryPolicy::InputType {
                document: arc("document=type;v1"),
                query: arc("query/type\nv1"),
            },
            chunk_target_tokens: 768,
            chunk_hard_max_tokens: 1024,
            local_loader: None,
        };
        let original = EmbeddingBackend::from_profile(profile);
        let parsed = EmbeddingBackend::from_identity(&original.identity()).unwrap();

        assert_eq!(parsed.runtime, EmbeddingRuntime::OpenRouter);
        assert_eq!(parsed.model_id(), "provider/model:revision");
        assert_eq!(parsed.dim(), 1536);
        assert_eq!(parsed.max_len, 8192);
        assert_eq!(parsed.profile.query_policy, original.profile.query_policy);
    }

    #[test]
    fn from_identity_accepts_legacy_identities() {
        let default = "fastembed-candle:Qwen3-Embedding-0.6B:dim1024:max1024:v2";
        let cpu = "fastembed-onnx-cpu:BGESmallENV15Q:dim384:max512:v1";
        let openrouter = "openrouter:qwen/qwen3-embedding-8b:dim4096:max32768:v1";

        assert_eq!(
            EmbeddingBackend::from_identity(default)
                .unwrap()
                .profile
                .name(),
            "local-gpu-small"
        );
        assert_eq!(
            EmbeddingBackend::from_identity(cpu).unwrap().profile.name(),
            "local-cpu-small"
        );
        assert_eq!(
            EmbeddingBackend::from_identity(openrouter)
                .unwrap()
                .profile
                .name(),
            "openrouter-qwen3-8b"
        );
    }

    #[test]
    fn from_identity_rejects_unknown_v2_local_model() {
        let identity = EmbeddingIdentity {
            runtime: EmbeddingRuntime::LocalQwen3CandleCuda,
            model_id: "Qwen/Unknown".to_string(),
            dim: 1024,
            max_len: 1024,
            query: QueryPolicy::InstructionPrefix(arc(QWEN3_CODE_QUERY_PREFIX)).encode_tag(),
        }
        .encode();
        let err = EmbeddingBackend::from_identity(&identity).unwrap_err();
        let text = err.to_string();

        assert!(text.contains("unknown built-in model"));
        assert!(text.contains("clear_cache"));
    }

    #[test]
    fn from_identity_rejects_garbage() {
        assert!(EmbeddingBackend::from_identity("garbage").is_err());
        assert!(
            EmbeddingBackend::from_identity(
                "fastembed-candle:Qwen3-Embedding-0.6B:dim999:max2048:v2"
            )
            .is_err()
        );
        assert!(
            EmbeddingBackend::from_identity(
                "fastembed-candle:Qwen3-Embedding-0.6B:dim1024:max1024:v1"
            )
            .is_err()
        );
    }

    /// Гейт пары «рантайм ⇄ постоянная форма»: форма есть РОВНО у тех
    /// рантаймов, что исполняют граф на GPU.
    ///
    /// Заведён потому, что половинки пары живут в разных файлах: загрузчик
    /// модели откажет, если рантайм GPU-шный, а формы нет, — но обратный
    /// перекос (форма объявлена рантайму, которому она не нужна) не заметил бы
    /// никто, кроме этого теста. Он же держит охват: новый рантайм обязан
    /// попасть в `match` явно, иначе `fixed_input_shape` не соберётся.
    #[test]
    fn fixed_shape_exists_exactly_for_shape_compiling_runtimes() {
        let profiles = EmbeddingProfile::built_in_profiles();
        let mut seen_gpu = false;
        for profile in profiles.iter() {
            let backend = EmbeddingBackend::from_profile(profile.clone());
            let shape = backend.fixed_input_shape();
            let wants_shape = backend.is_fastembed_onnx_gpu();
            assert_eq!(
                shape.is_some(),
                wants_shape,
                "профиль `{}` ({:?}): форма и рантайм разошлись",
                profile.name(),
                backend.runtime
            );
            if let Some(shape) = shape {
                seen_gpu = true;
                assert_eq!(shape.rows, GPU_BATCH_ROWS);
                // Длина последовательности — это `max_len` бэкенда, а не
                // константа: она переопределяется на инстансе, и разъезд с ней
                // означал бы компиляцию ядер под форму, которой не бывает.
                assert_eq!(shape.seq_len, backend.max_len);
            }
        }
        assert!(
            seen_gpu,
            "в реестре не осталось GPU-профиля — тест стал вакуумным"
        );
    }
}
