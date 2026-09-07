//! Indexing module - Unified pipeline for both Tantivy and vector store

pub(crate) mod backup;
mod consistency;
pub(crate) mod embedding_batcher;
pub mod error;
pub(crate) mod error_collection;
pub(crate) mod file_processor;
mod identity;
mod incremental;
pub mod incremental_service;
mod indexer_core;
mod merkle;
pub mod project_paths;
mod retry;
pub mod search;
mod tantivy_adapter;
mod traversal;
mod unified;
mod unified_parallel;

pub(crate) use error::IndexingError;
pub use incremental::{IncrementalIndexer, get_snapshot_path};
pub use incremental_service::{
    IncrementalIndexOutcome, IncrementalIndexRequest, index_project_incrementally,
};
pub use merkle::{ChangeSet, FileSystemMerkle};
pub use project_paths::{
    IndexedProfilePaths, IndexingProjectPaths, collection_prefix, dir_hash, read_embedder_identity,
};
pub use search::open_bm25_search;
pub use tantivy_adapter::TantivyAdapter;
// Наружу отдан ровно один обход дерева: rmc-server статит те же файлы, что
// индексатор, и «файл проекта» обе стороны обязаны понимать ОДИНАКОВО — иначе
// семантический кэш считает свежим то, что индекс уже переиндексировал.
// По той же причине наружу отдана и граница вложенного проекта: где кончается
// «этот проект», должны одинаково понимать все, кто спрашивает про его файлы.
pub use traversal::{NESTED_ROOT_MARKER, collect_project_rust_files, is_nested_project_root};
pub use unified::{IndexFileResult, IndexStats, UnifiedIndexer};
