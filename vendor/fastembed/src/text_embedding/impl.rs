//! The definition of the main struct for text embeddings - [`TextEmbedding`].

#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    Embedding, EmbeddingModel, EmbeddingOutput, ModelInfo, OutputKey, QuantizationMode,
    SingleBatchOutput,
    common::load_tokenizer,
    models::{ModelTrait, text_embedding::models_list},
    pooling::Pooling,
};
#[cfg(feature = "hf-hub")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ndarray::Array;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use std::thread::available_parallelism;
use tokenizers::{PaddingStrategy, Tokenizer, TruncationParams};

#[cfg(feature = "hf-hub")]
use super::TextInitOptions;
use super::{
    DEFAULT_BATCH_SIZE, FixedBatchShape, InitOptionsUserDefined, TextEmbedding,
    UserDefinedEmbeddingModel, output,
};

impl TextEmbedding {
    fn builder_error(err: ort::Error<ort::session::builder::SessionBuilder>) -> anyhow::Error {
        anyhow::Error::msg(err.to_string())
    }

    /// Try to generate a new TextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: TextInitOptions) -> Result<Self> {
        let TextInitOptions {
            max_length,
            model_name,
            execution_providers,
            cache_dir,
            show_download_progress,
        } = options;
        let threads = available_parallelism()?.get();

        let model_repo = TextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = TextEmbedding::get_model_info(&model_name)?;
        let model_file_name = &model_info.model_file;
        let model_file_reference = model_repo
            .get(model_file_name)
            .context(format!("Failed to retrieve {}", model_file_name))?;

        if !model_info.additional_files.is_empty() {
            for file in &model_info.additional_files {
                model_repo
                    .get(file)
                    .context(format!("Failed to retrieve {}", file))?;
            }
        }

        // prioritise loading pooling config if available, if not (thanks qdrant!), look for it in hardcoded
        let post_processing = TextEmbedding::get_default_pooling_method(&model_name);

        #[cfg(feature = "directml")]
        let has_directml = execution_providers
            .iter()
            .any(|ep| ep.downcast_ref::<ort::ep::DirectML>().is_some());
        #[cfg(not(feature = "directml"))]
        let has_directml = false;

        let mut builder = Session::builder()?
            .with_execution_providers(execution_providers)
            .map_err(Self::builder_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(Self::builder_error)?
            .with_intra_threads(threads)
            .map_err(Self::builder_error)?;

        if has_directml {
            builder = builder
                .with_memory_pattern(false)
                .map_err(Self::builder_error)?
                .with_parallel_execution(false)
                .map_err(Self::builder_error)?;
        }

        let session = builder.commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            post_processing,
            TextEmbedding::get_quantization_mode(&model_name),
            model_info.output_key.clone(),
        ))
    }

    /// Create a TextEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedEmbeddingModel,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let InitOptionsUserDefined {
            execution_providers,
            max_length,
        } = options;

        let threads = available_parallelism()?.get();

        #[cfg(feature = "directml")]
        let has_directml = execution_providers
            .iter()
            .any(|ep| ep.downcast_ref::<ort::ep::DirectML>().is_some());
        #[cfg(not(feature = "directml"))]
        let has_directml = false;

        let session = {
            let mut session_builder = Session::builder()?
                .with_execution_providers(execution_providers)
                .map_err(Self::builder_error)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(Self::builder_error)?
                .with_intra_threads(threads)
                .map_err(Self::builder_error)?;

            if has_directml {
                session_builder = session_builder
                    .with_memory_pattern(false)
                    .map_err(Self::builder_error)?
                    .with_parallel_execution(false)
                    .map_err(Self::builder_error)?;
            }

            for external_initializer_file in model.external_initializers {
                session_builder = session_builder
                    .with_external_initializer_file_in_memory(
                        external_initializer_file.file_name,
                        external_initializer_file.buffer.into(),
                    )
                    .map_err(Self::builder_error)?;
            }

            session_builder.commit_from_memory(&model.onnx_file)?
        };

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            model.pooling,
            model.quantization,
            model.output_key,
        ))
    }

    /// Private method to return an instance
    fn new(
        tokenizer: Tokenizer,
        session: Session,
        post_process: Option<Pooling>,
        quantization: QuantizationMode,
        output_key: Option<OutputKey>,
    ) -> Self {
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Self {
            tokenizer,
            session,
            need_token_type_ids,
            pooling: post_process,
            quantization,
            output_key,
            fixed_shape: None,
        }
    }

    /// Зафиксировать форму входа модели: `shape.rows` × `shape.seq_len`.
    ///
    /// Меняет две вещи: токенизатор паддит РОВНО до `seq_len` (вместо «до самой
    /// длинной строки в батче»), а неполный последний батч добивается до
    /// `rows` строк. Наружу это не видно — добивки отрезаются по
    /// [`SingleBatchOutput::real_rows`], и на каждый входной текст приходится
    /// ровно один эмбеддинг.
    ///
    /// # Когда это нужно
    /// Компилирующим execution provider'ам (MIGraphX и родня): они компилируют
    /// ядра ПОД ФОРМУ, и каждая новая форма стоит десятков секунд и сотен
    /// мегабайт кэша. На CPU смысла нет — там форма бесплатна, а добивки просто
    /// сжигают такты.
    ///
    /// # Отказы
    /// - `QuantizationMode::Dynamic` — динамическая квантизация подгоняет
    ///   диапазон под КАЖДЫЙ батч, поэтому батчи там запрещены в принципе;
    ///   фиксировать высоту батча нечего.
    /// - `seq_len` больше предела усечения токенизатора: строка длиннее предела
    ///   всё равно будет обрезана, и обещанной формы не получится.
    pub fn with_fixed_batch_shape(mut self, shape: FixedBatchShape) -> Result<Self> {
        if shape.rows == 0 || shape.seq_len == 0 {
            return Err(anyhow::Error::msg(
                "Fixed batch shape requires non-zero rows and seq_len.",
            ));
        }
        if self.quantization == QuantizationMode::Dynamic {
            return Err(anyhow::Error::msg(
                "Fixed batch shape cannot be used with dynamic quantization: \
                 the data range is refitted per batch, so batching is disallowed \
                 for such models in the first place.",
            ));
        }

        // Усечение: форма достижима, только если токенизатор не отдаёт строк
        // длиннее seq_len. Предел ставит `load_tokenizer` (max_length, ужатый
        // до model_max_length модели), поэтому здесь его не поднимаем, а
        // ПРОВЕРЯЕМ — иначе тихо получили бы форму больше, чем модель умеет.
        let truncation_limit = self
            .tokenizer
            .get_truncation()
            .map(|params| params.max_length)
            .ok_or_else(|| {
                anyhow::Error::msg(
                    "Tokenizer has no truncation params; cannot fix the input shape.",
                )
            })?;
        if shape.seq_len > truncation_limit {
            return Err(anyhow::Error::msg(format!(
                "Fixed seq_len {} exceeds the tokenizer truncation limit {}.",
                shape.seq_len, truncation_limit
            )));
        }

        let mut padding = self.tokenizer.get_padding().cloned().ok_or_else(|| {
            anyhow::Error::msg("Tokenizer has no padding params; cannot fix the input shape.")
        })?;
        padding.strategy = PaddingStrategy::Fixed(shape.seq_len);
        self.tokenizer.with_padding(Some(padding));
        self.tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: shape.seq_len,
                ..Default::default()
            }))
            .map_err(anyhow::Error::msg)?;

        self.fixed_shape = Some(shape);
        Ok(self)
    }

    /// Постоянная форма входа, если она зафиксирована.
    pub fn fixed_batch_shape(&self) -> Option<FixedBatchShape> {
        self.fixed_shape
    }
    /// Return the TextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> anyhow::Result<ApiRepo> {
        use crate::common::pull_from_hf;

        let model_code = TextEmbedding::get_model_info(&model)?.model_code.clone();
        pull_from_hf(model_code, cache_dir, show_download_progress)
    }

    pub fn get_default_pooling_method(model_name: &EmbeddingModel) -> Option<Pooling> {
        match model_name {
            EmbeddingModel::AllMiniLML6V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML6V2Q => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2Q => Some(Pooling::Mean),

            EmbeddingModel::BGEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallZHV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeZHV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEM3 => Some(Pooling::Cls),

            EmbeddingModel::NomicEmbedTextV1 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15Q => Some(Pooling::Mean),

            EmbeddingModel::ParaphraseMLMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => Some(Pooling::Mean),
            EmbeddingModel::AllMpnetBaseV2 => Some(Pooling::Mean),

            EmbeddingModel::ModernBertEmbedLarge => Some(Pooling::Mean),

            EmbeddingModel::MultilingualE5Base => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Small => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Large => Some(Pooling::Mean),

            EmbeddingModel::MxbaiEmbedLargeV1 => Some(Pooling::Cls),
            EmbeddingModel::MxbaiEmbedLargeV1Q => Some(Pooling::Cls),

            EmbeddingModel::GTEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15Q => Some(Pooling::Cls),

            EmbeddingModel::ClipVitB32 => Some(Pooling::Mean),

            EmbeddingModel::JinaEmbeddingsV2BaseCode => Some(Pooling::Mean),
            EmbeddingModel::JinaEmbeddingsV2BaseEN => Some(Pooling::Mean),

            EmbeddingModel::EmbeddingGemma300M => Some(Pooling::Mean),

            EmbeddingModel::SnowflakeArcticEmbedXS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedXSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedM => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLong => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedL => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedLQ => Some(Pooling::Cls),
        }
    }

    /// Get the quantization mode of the model.
    ///
    /// Any models with a `Q` suffix in their name are quantized models.
    ///
    /// Currently only 6 supported models have dynamic quantization:
    /// - Alibaba-NLP/gte-base-en-v1.5
    /// - Alibaba-NLP/gte-large-en-v1.5
    /// - mixedbread-ai/mxbai-embed-large-v1
    /// - nomic-ai/nomic-embed-text-v1.5
    /// - Xenova/all-MiniLM-L12-v2
    /// - Xenova/all-MiniLM-L6-v2
    ///
    // TODO: Update this list when more models are added
    pub fn get_quantization_mode(model_name: &EmbeddingModel) -> QuantizationMode {
        match model_name {
            EmbeddingModel::AllMiniLML6V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::AllMiniLML12V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::BGEBaseENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGELargeENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGESmallENV15Q => QuantizationMode::Static,
            EmbeddingModel::NomicEmbedTextV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => QuantizationMode::Static,
            EmbeddingModel::MxbaiEmbedLargeV1Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTEBaseENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTELargeENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedXSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedLQ => QuantizationMode::Dynamic,
            _ => QuantizationMode::None,
        }
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<EmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from EmbeddingModel
    pub fn get_model_info(model: &EmbeddingModel) -> Result<&ModelInfo<EmbeddingModel>> {
        EmbeddingModel::get_model_info(model).ok_or_else(|| {
            anyhow::Error::msg(format!(
                "Model {model:?} not found. Please check if the model is supported \
                by the current version."
            ))
        })
    }

    /// Method to generate an [`ort::SessionOutputs`] wrapped in a [`EmbeddingOutput`]
    /// instance, which can be used to extract the embeddings with default or custom
    /// methods as well as output key precedence.
    ///
    /// Metadata that could be useful for creating the array transformer is
    /// returned alongside the [`EmbeddingOutput`] instance, such as pooling methods
    /// etc.
    ///
    /// # Note
    ///
    /// This is a lower level method than [`TextEmbedding::embed`], and is useful
    /// when you need to extract the session outputs in a custom way.
    ///
    /// If you want to extract the embeddings directly, use [`TextEmbedding::embed`].
    ///
    /// If you want to use the raw session outputs, use [`EmbeddingOutput::into_raw`]
    /// on the output of this method.
    ///
    /// If you want to choose a different export key or customize the way the batch
    /// arrays are aggregated, you can define your own array transformer
    /// and use it on [`EmbeddingOutput::export_with_transformer`] to extract the
    /// embeddings with your custom output type.
    pub fn transform<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<EmbeddingOutput> {
        let texts = texts.as_ref();
        // Determine the batch size according to the quantization method used.
        // Default if not specified
        let batch_size = match self.quantization {
            QuantizationMode::Dynamic => {
                if let Some(batch_size) = batch_size {
                    if batch_size < texts.len() {
                        Err(anyhow::Error::msg(
                            "Dynamic quantization cannot be used with batching. \
                            This is due to the dynamic quantization process adjusting \
                            the data range to fit each batch, making the embeddings \
                            incompatible across batches. Try specifying a batch size \
                            of `None`, or use a model with static or no quantization.",
                        ))
                    } else {
                        Ok(texts.len())
                    }
                } else {
                    Ok(texts.len())
                }
            }
            _ => Ok(batch_size.unwrap_or(DEFAULT_BATCH_SIZE)),
        }?;

        // Постоянная форма входа диктует высоту батча: резать надо РОВНО по
        // `rows`, иначе добивка не поможет — куски всё равно приедут разной
        // высоты. Запрошенный вызывающим batch_size при этом игнорируется
        // осознанно: форма — свойство модели, а не отдельного вызова.
        let fixed_shape = self.fixed_shape;
        let batch_size = fixed_shape.map_or(batch_size, |shape| shape.rows);

        let batches = texts
            .chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch.iter().map(|text| text.as_ref()).collect();
                let encodings = self.tokenizer.encode_batch(inputs, true).map_err(|e| {
                    anyhow::Error::msg(e.to_string()).context("Failed to encode the batch.")
                })?;

                // Extract the encoding length and batch size
                let encoding_length = encodings
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Tokenizer returned empty encodings"))?
                    .len();
                let real_rows = batch.len();
                // Высота тензора: при постоянной форме — всегда `rows`, иначе
                // столько, сколько текстов в куске.
                let batch_size = fixed_shape.map_or(real_rows, |shape| shape.rows);

                // Оракул на форму: длина последовательности должна быть ровно
                // такой, какую пообещал `with_fixed_batch_shape`. Если
                // токенизатор отдал другую (кто-то переставил padding-стратегию
                // снаружи — поле `tokenizer` публичное), лучше упасть здесь, чем
                // оплатить компиляцию ядер под неожиданную форму.
                if let Some(shape) = fixed_shape {
                    if encoding_length != shape.seq_len {
                        return Err(anyhow::anyhow!(
                            "Fixed batch shape promised seq_len {}, but the tokenizer \
                             produced {}; padding strategy was changed externally.",
                            shape.seq_len,
                            encoding_length
                        ));
                    }
                }

                let max_size = encoding_length * batch_size;

                // Preallocate arrays with the maximum size
                let mut ids_array = Vec::with_capacity(max_size);
                let mut mask_array = Vec::with_capacity(max_size);
                let mut type_ids_array = Vec::with_capacity(max_size);

                encodings.iter().for_each(|encoding| {
                    let ids = encoding.get_ids();
                    let mask = encoding.get_attention_mask();
                    let type_ids = encoding.get_type_ids();

                    ids_array.extend(ids.iter().map(|x| *x as i64));
                    mask_array.extend(mask.iter().map(|x| *x as i64));
                    type_ids_array.extend(type_ids.iter().map(|x| *x as i64));
                });

                // Добивка неполного куска до постоянной высоты. Строка-добивка —
                // КОПИЯ последней реальной строки, а не нули: у копии непустая
                // attention-маска, поэтому mean-пулинг по ней не делит на ноль.
                // Её эмбеддинг всё равно отбрасывается по `real_rows`.
                for _ in real_rows..batch_size {
                    let last = encodings
                        .last()
                        .ok_or_else(|| anyhow::anyhow!("Tokenizer returned empty encodings"))?;
                    ids_array.extend(last.get_ids().iter().map(|x| *x as i64));
                    mask_array.extend(last.get_attention_mask().iter().map(|x| *x as i64));
                    type_ids_array.extend(last.get_type_ids().iter().map(|x| *x as i64));
                }

                let inputs_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), ids_array)?;
                let attention_mask_array =
                    Array::from_shape_vec((batch_size, encoding_length), mask_array)?;
                let token_type_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), type_ids_array)?;

                let mut session_inputs = ort::inputs![
                    "input_ids" => Value::from_array(inputs_ids_array)?,
                    "attention_mask" => Value::from_array(attention_mask_array.clone())?,
                ];

                if self.need_token_type_ids {
                    session_inputs.push((
                        "token_type_ids".into(),
                        Value::from_array(token_type_ids_array)?.into(),
                    ));
                }

                let outputs_map = self
                    .session
                    .run(session_inputs)
                    .map_err(anyhow::Error::new)?
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect();
                Ok(SingleBatchOutput {
                    outputs: outputs_map,
                    attention_mask_array,
                    real_rows,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(EmbeddingOutput::new(batches))
    }

    /// Method to generate sentence embeddings for a collection of texts.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    ///
    /// The output is a [`Vec`] of [`Embedding`]s.
    ///
    /// # Note
    ///
    /// This method is a higher level method than [`TextEmbedding::transform`] by utilizing
    /// the default output precedence and array transformer for the [`TextEmbedding`] model.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batches = self.transform(texts.as_ref(), batch_size)?;
        if let Some(output_key) = &self.output_key {
            batches.export_with_transformer(output::transformer_with_precedence(
                output_key,
                self.pooling.clone(),
            ))
        } else {
            batches.export_with_transformer(output::transformer_with_precedence(
                output::OUTPUT_TYPE_PRECEDENCE,
                self.pooling.clone(),
            ))
        }
    }
}
