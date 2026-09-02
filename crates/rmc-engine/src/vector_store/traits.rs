//! Vector store backend trait definition
//!
//! Defines the interface that all vector storage backends must implement.

use async_trait::async_trait;
use std::collections::HashSet;

use super::VectorSearchResult;
use super::VectorStoreMaintenance;
use super::error::VectorStoreError;
use crate::chunker::{ChunkId, CodeChunk};
use crate::embeddings::Embedding;

/// Trait for vector storage backends
///
/// Implementations must be Send + Sync for use with async runtimes.
/// All operations are async to support both embedded and remote backends.
#[async_trait]
pub trait VectorStoreBackend: Send + Sync {
    /// Insert or update chunks with their embeddings
    async fn upsert_chunks(
        &self,
        chunks_with_embeddings: Vec<(ChunkId, Embedding, CodeChunk)>,
    ) -> Result<(), VectorStoreError>;

    /// Search for similar chunks using a query vector
    async fn search(
        &self,
        query_vector: Embedding,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError>;

    /// Delete chunks by their IDs
    async fn delete_chunks(&self, chunk_ids: Vec<ChunkId>) -> Result<(), VectorStoreError>;

    /// Delete all chunks from a specific file path
    async fn delete_by_file_path(&self, file_path: &str) -> Result<(), VectorStoreError>;

    /// Get the total number of vectors in the store
    async fn count(&self) -> Result<usize, VectorStoreError>;

    /// Distinct source files that have at least one vector in the store.
    ///
    /// Coverage lives here and nowhere else: `count()` answers "how many
    /// vectors", which says nothing about whether a file was indexed at
    /// all. A run that dies halfway leaves both numbers plausible, and
    /// only the file set can be put next to the Merkle snapshot.
    async fn indexed_file_paths(&self) -> Result<HashSet<String>, VectorStoreError>;

    /// Clear all vectors (keep collection/table structure)
    async fn clear(&self) -> Result<(), VectorStoreError>;

    /// Reclaim disk space left behind by writes.
    ///
    /// The store is append-only: every upsert lands in a NEW fragment and
    /// every change publishes a NEW version, and neither the superseded
    /// rows nor the old manifests go away on their own. Nothing in the
    /// write path can do this incrementally — a fragment only becomes
    /// unreferenced once its rows have been rewritten elsewhere — so it is
    /// a separate pass, and without it on-disk size grows without bound
    /// against a row count that does not.
    ///
    /// `prune_keep` is the age below which versions are left alone: it is
    /// the margin for a reader that opened the table before this pass
    /// started and is still walking it.
    async fn maintain(
        &self,
        prune_keep: std::time::Duration,
    ) -> Result<VectorStoreMaintenance, VectorStoreError>;

    /// Check if the backend is healthy/connected
    async fn health_check(&self) -> Result<(), VectorStoreError>;
}
