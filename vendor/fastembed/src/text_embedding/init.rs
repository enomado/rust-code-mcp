//! Initialization options for the text embedding models.
//!

use crate::{
    EmbeddingModel, OutputKey, QuantizationMode,
    common::TokenizerFiles,
    init::{HasMaxLength, InitOptionsWithLength},
    pooling::Pooling,
};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use tokenizers::Tokenizer;

use super::DEFAULT_MAX_LENGTH;

impl HasMaxLength for EmbeddingModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the TextEmbedding model
pub type TextInitOptions = InitOptionsWithLength<EmbeddingModel>;

/// Options for initializing UserDefinedEmbeddingModel
///
/// Model files are held by the UserDefinedEmbeddingModel struct
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
}

impl InitOptionsUserDefined {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

impl Default for InitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
        }
    }
}

/// Convert InitOptions to InitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<TextInitOptions> for InitOptionsUserDefined {
    fn from(options: TextInitOptions) -> Self {
        InitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub external_initializers: Vec<ExternalInitializerFile>,
    pub tokenizer_files: TokenizerFiles,
    pub pooling: Option<Pooling>,
    pub quantization: QuantizationMode,
    pub output_key: Option<OutputKey>,
}

/// Struct for adding external initializers to "bring your own" embedding models
///
/// The buffer is expecting the data of the external initializer and the file_name
/// must match the one referenced by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalInitializerFile {
    pub file_name: String,
    pub buffer: Vec<u8>,
}

impl UserDefinedEmbeddingModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_file,
            external_initializers: Vec::new(),
            tokenizer_files,
            quantization: QuantizationMode::None,
            pooling: None,
            output_key: None,
        }
    }

    pub fn with_quantization(mut self, quantization: QuantizationMode) -> Self {
        self.quantization = quantization;
        self
    }

    pub fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = Some(pooling);
        self
    }

    pub fn with_external_initializer(mut self, file_name: String, buffer: Vec<u8>) -> Self {
        self.external_initializers
            .push(ExternalInitializerFile { file_name, buffer });
        self
    }
}

/// Fixed model input shape: rows per batch × sequence length.
///
/// # Why
/// By default the input shape floats along BOTH axes: the tokenizer pads to the
/// longest sequence IN THE BATCH (`PaddingStrategy::BatchLongest`), and
/// the last batch is also partial. For CPU this does not matter, but compiling
/// runtimes (MIGraphX and other AOT EPs) compile KERNELS PER SHAPE: every new
/// shape = a separate compilation (tens of seconds) and a separate cache file (hundreds
/// of megabytes). Measured on indexing 40 files: 4 shapes, 659 MB of cache.
///
/// With the shape fixed there is exactly one compilation, and the price is the compute spent on
/// padding rows in the partial last batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedBatchShape {
    /// Batch height. Longer inputs are cut into chunks of `rows`; the last
    /// chunk is padded up to `rows` (the padding is discarded on output).
    pub rows: usize,
    /// Sequence length. The tokenizer pads EXACTLY to it, not to
    /// the longest row in the batch.
    pub seq_len: usize,
}

/// Rust representation of the TextEmbedding model
pub struct TextEmbedding {
    pub tokenizer: Tokenizer,
    pub(crate) pooling: Option<Pooling>,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) quantization: QuantizationMode,
    pub(crate) output_key: Option<OutputKey>,
    /// Fixed input shape, if requested via
    /// [`TextEmbedding::with_fixed_batch_shape`]. `None` keeps the old behavior
    /// (the shape floats along both axes).
    pub(crate) fixed_shape: Option<FixedBatchShape>,
}
