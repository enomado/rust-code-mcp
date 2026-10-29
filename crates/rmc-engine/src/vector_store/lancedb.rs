//! LanceDB backend for vector storage
//!
//! Embedded vector database using Apache Arrow for columnar storage.
//! No external server required - direct file access.

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::DistanceType;
use lancedb::connect;
use lancedb::index::Index;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::optimize::Duration as ChronoDuration;
use lancedb::table::{CompactionOptions, OptimizeAction, OptimizeOptions};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::VectorSearchResult;
use super::VectorStoreMaintenance;
use super::error::VectorStoreError;
use super::traits::VectorStoreBackend;
use super::{VectorIndexCoverage, VectorStoreLayout};
use crate::chunker::{ChunkId, CodeChunk};
use crate::embeddings::Embedding;

const TABLE_NAME: &str = "vectors";
const METADATA_FILE: &str = "metadata.json";

/// Fragments below this many rows are what compaction merges.
///
/// Deliberately equal to the threshold `lancedb`'s `Table::stats()` uses for
/// `num_small_fragments` (a literal `100000` in its `table.rs`): the gate in
/// `maintain` counts with that number, and counting by one threshold while
/// compacting by another would make the gate fire on fragments the
/// compaction then leaves alone, or the reverse.
///
/// Why not lance's own default (1M rows): a project store is usually below
/// it, so its main fragment never "fills" and every compaction rewrites the
/// whole table. With this target a full fragment is left in place until more
/// than `materialize_deletions_threshold` (10%) of it has been deleted —
/// re-indexed files delete their old rows — and only then rewritten, once,
/// and split into fragments of this size.
const COMPACT_TARGET_ROWS: usize = 100_000;

/// How many small fragments must pile up before maintenance compacts.
///
/// Every `add` writes one fragment (deletes write none: they mark rows in
/// deletion files). A changed background sync appends once, so this is
/// "compact every eighth changed sync". It was not, while the sync wrote
/// file by file: 36 changed files were 36 fragments and a compaction on
/// every such sync (measured 2026-10-29). Search reads every fragment, so
/// the pile is not free either; eight keeps that cost small.
const COMPACT_MIN_SMALL_FRAGMENTS: usize = 8;

/// The scalar indices a table is created with: the columns lookups and
/// deletes filter on. `id` — `chunks_by_ids` and `delete_chunks`;
/// `file_path` — the per-sync delete. Nothing filters on `symbol_kind`, and
/// an index nothing reads is still one every maintenance pass extends, so
/// tables that have one lose it (see `maintain`).
const SCALAR_INDEX_COLUMNS: [&str; 2] = ["id", "file_path"];
const DROPPED_INDEX_COLUMNS: [&str; 1] = ["symbol_kind"];

/// The vector column and the name of its ANN index.
const VECTOR_COLUMN: &str = "vector";
const VECTOR_INDEX_NAME: &str = "vector_idx";

/// Rows from which `maintain` builds the ANN index (IVF_PQ, cosine).
///
/// Below it the exact scan is cheap and an index would be trained on too
/// few vectors to partition well. The scan costs about 2.2 CPU-s per query
/// at 0.93M rows of 384 floats (measured 2026-10-29), so ~0.24 at the
/// default; the index brings 0.93M rows to ~0.02. `0` turns the index off
/// (and keeps an existing one as it is).
const ANN_MIN_ROWS_ENV: &str = "RMC_VECTOR_ANN_MIN_ROWS";
const DEFAULT_ANN_MIN_ROWS: usize = 100_000;

// How the three ANN numbers below were chosen — measured 2026-10-29 on a copy
// of a 1.06M-row bge-small store (384 dims), 40 phrase and identifier queries
// of the kind agents send, recall against the exact scan of the same query
// vector (`probe_ann_recall_on_real_queries`). The exact scan: 2.3–2.65 CPU-s
// and 0.6–2 s per query.
//
// - PQ sub-vectors: lance's default (dim/16 = 24) tops out at recall@10 0.90
//   even re-ranking 1000 candidates; dim/8 = 48 reaches 0.995. dim/4 = 96 was
//   no better (0.975) and builds twice as slow. Build at 48: 75–160 s,
//   ~320 CPU-s, once per store.
// - nprobes: 20 leaves queries whose neighbours sit in unprobed partitions
//   (worst query 0/10 found); 40 lifts the worst to 8–9/10.
// - re-ranked candidates: at limit 100 (hybrid search's candidate count),
//   1000 → recall 0.956, 2000 → 0.970–0.976, 5000 → 0.972–0.978 at 1.5x the
//   CPU. 2000 it is. With all three defaults: recall@10 0.998 (worst query
//   9/10), recall@100 0.976, 0.054 CPU-s and ~0.1 s per query — about 45x
//   below the scan. IVF training is randomised: recall moved by ~0.01 and
//   the worst query by tens of ranks between builds of the same table.

/// PQ sub-vectors of the ANN index: an eighth of the dimension (see above).
/// lance needs them to divide the dimension; for a dimension that is not a
/// multiple of 8 lance's own default is used.
fn ann_sub_vectors(dim: usize) -> Option<u32> {
    (dim % 8 == 0).then_some((dim / 8) as u32)
}

/// IVF partitions a query searches. More is closer to the exact result and
/// slower; see the measurement above for the default.
const ANN_NPROBES_ENV: &str = "RMC_VECTOR_ANN_NPROBES";
const DEFAULT_ANN_NPROBES: usize = 40;

/// How many candidates are re-ranked by their full vectors, whatever the
/// limit: the refine factor is this divided by the limit. Without re-ranking
/// the order is by PQ-compressed distances, and on this model's embeddings
/// (neighbours within ~0.05 cosine of each other) that put recall at 0.25
/// whatever the nprobes. A budget rather than a factor because the factor
/// multiplies the limit: one factor is either too few candidates at limit 10
/// or too many at limit 100. See the measurement above for the default.
const ANN_RERANK_ENV: &str = "RMC_VECTOR_ANN_RERANK";
const DEFAULT_ANN_RERANK: usize = 2000;

/// The refine factor that re-ranks at least `budget` candidates for `limit`
/// results (and never fewer than the results themselves).
fn refine_factor(budget: usize, limit: usize) -> u32 {
    budget.div_ceil(limit.max(1)).max(1) as u32
}

/// A numeric knob from the environment: unset is the default, set to
/// something unparsable is an error rather than a silent default.
fn knob<T: std::str::FromStr>(name: &str, default: T) -> Result<T, VectorStoreError> {
    match std::env::var(name) {
        Err(_) => Ok(default),
        Ok(value) => value.trim().parse().map_err(|_| {
            VectorStoreError::backend(format!("{name}={value:?} is not a valid number"))
        }),
    }
}

fn compaction_options() -> CompactionOptions {
    CompactionOptions {
        target_rows_per_fragment: COMPACT_TARGET_ROWS,
        ..CompactionOptions::default()
    }
}

/// On-disk fingerprint stored next to the LanceDB table. Records which
/// embedder produced the vectors so that a switch is detected on reopen
/// instead of silently corrupting search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorStoreMetadata {
    embedder_version: String,
    created_at: String,
}

fn metadata_path(table_dir: &Path) -> PathBuf {
    table_dir.join(METADATA_FILE)
}

fn read_metadata(table_dir: &Path) -> Result<Option<VectorStoreMetadata>, VectorStoreError> {
    let path = metadata_path(table_dir);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).map_err(|e| {
        VectorStoreError::backend(format!(
            "failed to read vector store metadata at {}: {}",
            path.display(),
            e
        ))
    })?;
    let parsed: VectorStoreMetadata = serde_json::from_slice(&bytes).map_err(|e| {
        VectorStoreError::serialization(format!(
            "failed to parse vector store metadata at {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(Some(parsed))
}

fn write_metadata_if_missing(
    table_dir: &Path,
    embedder_identity: &str,
) -> Result<(), VectorStoreError> {
    let path = metadata_path(table_dir);
    if path.exists() {
        return Ok(());
    }
    // RFC 3339 timestamp using only std (avoid pulling in chrono just for
    // a stamp). Falls back to "0" on a wildly broken clock.
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| {
            // YYYY-MM-DDTHH:MM:SSZ isn't worth re-implementing — store
            // the unix timestamp as a decimal string instead. Good
            // enough to debug, doesn't add a dep.
            format!("unix:{}", d.as_secs())
        })
        .unwrap_or_else(|_| "unix:0".to_string());
    let meta = VectorStoreMetadata {
        embedder_version: embedder_identity.to_string(),
        created_at,
    };
    let bytes = serde_json::to_vec_pretty(&meta).map_err(|e| {
        VectorStoreError::serialization(format!("failed to serialize vector store metadata: {}", e))
    })?;
    std::fs::write(&path, bytes).map_err(|e| {
        VectorStoreError::backend(format!(
            "failed to write vector store metadata at {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(())
}

/// LanceDB backend for embedded vector storage
pub struct LanceDbBackend {
    db: lancedb::Connection,
    table_name: String,
    vector_dim: usize,
    /// Cached Arrow schema to avoid recreation on every batch
    schema: Arc<Schema>,
}

impl LanceDbBackend {
    /// Create a new LanceDB backend
    ///
    /// # Arguments
    /// * `path` - Directory for database storage
    /// * `vector_dim` - Embedding dimension (depends on active embedder)
    /// * `embedder_identity` - Stable identity string for the embedder
    ///   that produced (or will produce) vectors stored here. On first
    ///   create, the identity is recorded in `metadata.json` beside the
    ///   table directory. On reopen, the recorded identity is compared
    ///   against the configured one and a [`VectorStoreError::VersionMismatch`]
    ///   is returned on disagreement — callers are expected to refuse and
    ///   ask the user to run `clear_cache`.
    pub async fn new(
        path: PathBuf,
        vector_dim: usize,
        embedder_identity: &str,
    ) -> Result<Self, VectorStoreError> {
        // Ensure directory exists
        std::fs::create_dir_all(&path).map_err(|e| {
            VectorStoreError::connection(format!("Failed to create directory: {}", e))
        })?;

        let db = connect(path.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| VectorStoreError::connection(format!("Failed to connect: {}", e)))?;

        // Create and cache schema once
        let schema = Arc::new(Self::create_schema_for_dim(vector_dim));

        let backend = Self {
            db,
            table_name: TABLE_NAME.to_string(),
            vector_dim,
            schema,
        };

        // Ensure table exists, then reconcile the on-disk embedder
        // fingerprint with what the caller asked for. Order matters:
        // create the table first so a half-initialized directory cannot
        // leave a metadata file pointing at a model that never produced
        // any vectors.
        backend.ensure_table_exists().await?;
        match read_metadata(&path)? {
            Some(meta) if meta.embedder_version != embedder_identity => {
                return Err(VectorStoreError::version_mismatch(
                    meta.embedder_version,
                    embedder_identity,
                ));
            }
            Some(_) => {
                // Match — nothing to do.
            }
            None => {
                write_metadata_if_missing(&path, embedder_identity)?;
            }
        }

        Ok(backend)
    }

    /// Open an existing LanceDB backend without creating directories, tables,
    /// indexes, or metadata.
    pub async fn open_existing(
        path: PathBuf,
        vector_dim: usize,
        embedder_identity: &str,
    ) -> Result<Self, VectorStoreError> {
        if !path.is_dir() {
            return Err(VectorStoreError::not_found(format!(
                "vector store directory does not exist: {}",
                path.display()
            )));
        }

        let metadata = read_metadata(&path)?.ok_or_else(|| {
            VectorStoreError::not_found(format!(
                "vector store metadata does not exist: {}",
                metadata_path(&path).display()
            ))
        })?;
        if metadata.embedder_version != embedder_identity {
            return Err(VectorStoreError::version_mismatch(
                metadata.embedder_version,
                embedder_identity,
            ));
        }

        let db = connect(path.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| VectorStoreError::connection(format!("Failed to connect: {}", e)))?;

        let tables = db
            .table_names()
            .execute()
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to list tables: {}", e)))?;
        if !tables.iter().any(|name| name == TABLE_NAME) {
            return Err(VectorStoreError::not_found(format!(
                "vector table does not exist: {}",
                TABLE_NAME
            )));
        }

        db.open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| VectorStoreError::not_found(format!("Table not found: {}", e)))?;

        Ok(Self {
            db,
            table_name: TABLE_NAME.to_string(),
            vector_dim,
            schema: Arc::new(Self::create_schema_for_dim(vector_dim)),
        })
    }

    /// Create Arrow schema for vectors table (static version for initialization)
    fn create_schema_for_dim(vector_dim: usize) -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vector_dim as i32,
                ),
                false,
            ),
            Field::new("chunk_json", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("symbol_kind", DataType::Utf8, false),
            Field::new("module_path", DataType::Utf8, false),
        ])
    }

    /// Get cached Arrow schema for vectors table
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Ensure the vectors table exists
    async fn ensure_table_exists(&self) -> Result<(), VectorStoreError> {
        let tables = self
            .db
            .table_names()
            .execute()
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to list tables: {}", e)))?;

        if !tables.contains(&self.table_name) {
            // Create empty table with cached schema
            let schema = self.schema();

            // Create empty arrays for initial table
            let id_array = StringArray::from(Vec::<String>::new());
            let vector_array = self.create_empty_vector_array();
            let chunk_json_array = StringArray::from(Vec::<String>::new());
            let file_path_array = StringArray::from(Vec::<String>::new());
            let symbol_kind_array = StringArray::from(Vec::<String>::new());
            let module_path_array = StringArray::from(Vec::<String>::new());

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(id_array),
                    Arc::new(vector_array),
                    Arc::new(chunk_json_array),
                    Arc::new(file_path_array),
                    Arc::new(symbol_kind_array),
                    Arc::new(module_path_array),
                ],
            )
            .map_err(|e| VectorStoreError::backend(format!("Failed to create batch: {}", e)))?;

            let batches: Box<dyn RecordBatchReader + Send> =
                Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

            let table = self
                .db
                .create_table(&self.table_name, batches)
                .execute()
                .await
                .map_err(|e| VectorStoreError::backend(format!("Failed to create table: {}", e)))?;

            tracing::info!("Created LanceDB table: {}", self.table_name);

            // Created on the empty table, so they cover nothing yet: lance
            // never extends an index on write. `maintain` does, with
            // `OptimizeAction::Index` — without that these stayed at zero
            // indexed rows for the store's whole life (all three on a
            // 0.93M-row store, 2026-10-29), and every lookup was a scan.
            for column in SCALAR_INDEX_COLUMNS {
                table
                    .create_index(&[column], Index::BTree(BTreeIndexBuilder::default()))
                    .execute()
                    .await
                    .map_err(|e| {
                        VectorStoreError::backend(format!("Failed to create {column} index: {e}"))
                    })?;
            }

            tracing::info!("Created BTree indices on {:?}", SCALAR_INDEX_COLUMNS);
        }

        Ok(())
    }

    /// Create empty vector array for schema initialization
    fn create_empty_vector_array(&self) -> FixedSizeListArray {
        let values = Float32Array::from(Vec::<f32>::new());
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        FixedSizeListArray::try_new(field, self.vector_dim as i32, Arc::new(values), None).unwrap()
    }

    /// Convert chunks to Arrow RecordBatch
    ///
    /// Optimized single-pass implementation that:
    /// - Pre-allocates all vectors with known capacity
    /// - Iterates over chunks only once
    /// - Uses extend_from_slice for efficient vector flattening
    fn chunks_to_batch(
        &self,
        chunks: &[(ChunkId, Embedding, CodeChunk)],
    ) -> Result<RecordBatch, VectorStoreError> {
        let n = chunks.len();
        let schema = self.schema();

        // Pre-allocate all vectors with exact capacity
        let mut ids = Vec::with_capacity(n);
        let mut flat_vectors = Vec::with_capacity(n * self.vector_dim);
        let mut chunk_jsons = Vec::with_capacity(n);
        let mut file_paths = Vec::with_capacity(n);
        let mut symbol_kinds = Vec::with_capacity(n);
        let mut module_paths = Vec::with_capacity(n);

        // Single pass over all chunks
        for (id, embedding, chunk) in chunks {
            ids.push(id.to_string());
            flat_vectors.extend_from_slice(embedding);
            chunk_jsons.push(serde_json::to_string(chunk).map_err(|e| {
                VectorStoreError::serialization(format!("Failed to serialize chunk: {}", e))
            })?);
            file_paths.push(chunk.context.file_path.display().to_string());
            symbol_kinds.push(chunk.context.symbol_kind.clone());
            module_paths.push(chunk.context.module_path.join("::"));
        }

        // Build Arrow arrays from pre-allocated vectors
        let id_array = StringArray::from(ids);
        let values = Float32Array::from(flat_vectors);
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array =
            FixedSizeListArray::try_new(field, self.vector_dim as i32, Arc::new(values), None)
                .map_err(|e| {
                    VectorStoreError::serialization(format!("Failed to create vector array: {}", e))
                })?;
        let chunk_json_array = StringArray::from(chunk_jsons);
        let file_path_array = StringArray::from(file_paths);
        let symbol_kind_array = StringArray::from(symbol_kinds);
        let module_path_array = StringArray::from(module_paths);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(chunk_json_array),
                Arc::new(file_path_array),
                Arc::new(symbol_kind_array),
                Arc::new(module_path_array),
            ],
        )
        .map_err(|e| VectorStoreError::backend(format!("Failed to create batch: {}", e)))
    }

    /// Get table, returns error if not exists
    async fn get_table(&self) -> Result<lancedb::Table, VectorStoreError> {
        self.db
            .open_table(&self.table_name)
            .execute()
            .await
            .map_err(|e| VectorStoreError::not_found(format!("Table not found: {}", e)))
    }
}

/// The `(id, chunk)` rows of a batch projected to `id` and `chunk_json`.
///
/// The row id is the key the store is written and looked up by, and the
/// chunk carries its own copy of it; every writer passes the same value for
/// both (`(chunk.id, embedding, chunk)`). A row where they differ is a store
/// that contradicts itself, and is refused rather than resolved: an index
/// rebuilt from the chunk's id would point at a row lookups never find.
fn decode_chunk_rows(batch: &RecordBatch) -> Result<Vec<(ChunkId, CodeChunk)>, VectorStoreError> {
    let id_array = batch
        .column_by_name("id")
        .ok_or_else(|| VectorStoreError::query("Missing id column"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| VectorStoreError::query("Invalid id column type"))?;
    let chunk_json_array = batch
        .column_by_name("chunk_json")
        .ok_or_else(|| VectorStoreError::query("Missing chunk_json column"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| VectorStoreError::query("Invalid chunk_json column type"))?;

    let mut rows = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let id_str = id_array.value(i);
        let chunk_id = ChunkId::from_string(id_str)
            .map_err(|e| VectorStoreError::serialization(format!("Invalid chunk ID: {:?}", e)))?;
        let chunk: CodeChunk = serde_json::from_str(chunk_json_array.value(i)).map_err(|e| {
            VectorStoreError::serialization(format!("Failed to deserialize chunk: {}", e))
        })?;
        if chunk.id != chunk_id {
            return Err(VectorStoreError::serialization(format!(
                "row {id_str} holds the chunk of {}",
                chunk.id.to_string()
            )));
        }
        rows.push((chunk_id, chunk));
    }
    Ok(rows)
}

/// Build the ANN index if the table has none and holds at least `min_rows`
/// rows (`0`: never). Returns whether it built one.
///
/// Built by maintenance and not at table creation: IVF trains its
/// partitions on the vectors present, and an empty table has none. Once
/// built, the `OptimizeAction::Index` pass of `maintain` extends it with the
/// rows written since; the partitions themselves stay as trained.
async fn ensure_ann_index(
    table: &lancedb::Table,
    dim: usize,
    min_rows: usize,
) -> Result<bool, VectorStoreError> {
    if min_rows == 0 {
        return Ok(false);
    }
    let has_ann = table
        .list_indices()
        .await
        .map_err(|e| VectorStoreError::backend(format!("Failed to list indices: {}", e)))?
        .iter()
        .any(|config| config.columns.iter().any(|column| column == VECTOR_COLUMN));
    if has_ann {
        return Ok(false);
    }
    let rows = table
        .count_rows(None)
        .await
        .map_err(|e| VectorStoreError::query(format!("Failed to count rows: {}", e)))?;
    if rows < min_rows {
        return Ok(false);
    }
    let started = std::time::Instant::now();
    let mut builder = IvfPqIndexBuilder::default().distance_type(DistanceType::Cosine);
    if let Some(sub_vectors) = ann_sub_vectors(dim) {
        builder = builder.num_sub_vectors(sub_vectors);
    }
    table
        .create_index(&[VECTOR_COLUMN], Index::IvfPq(builder))
        .name(VECTOR_INDEX_NAME.to_string())
        .execute()
        .await
        .map_err(|e| VectorStoreError::backend(format!("Failed to build the ANN index: {}", e)))?;
    tracing::info!(
        "Built the ANN index (IVF_PQ, cosine) over {} rows in {:.1}s",
        rows,
        started.elapsed().as_secs_f64()
    );
    Ok(true)
}

#[async_trait]
impl VectorStoreBackend for LanceDbBackend {
    async fn insert_chunks(
        &self,
        chunks_with_embeddings: Vec<(ChunkId, Embedding, CodeChunk)>,
    ) -> Result<(), VectorStoreError> {
        if chunks_with_embeddings.is_empty() {
            return Ok(());
        }

        let num_chunks = chunks_with_embeddings.len();
        let table = self.get_table().await?;
        let batch = self.chunks_to_batch(&chunks_with_embeddings)?;

        // A plain append — the ids are fresh (see the trait). `merge_insert`
        // keyed on `id` joined every write against the whole table to match
        // nothing.
        table
            .add(batch)
            .execute()
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to insert chunks: {}", e)))?;

        tracing::debug!("Inserted {} chunks into LanceDB", num_chunks);
        Ok(())
    }

    async fn search(
        &self,
        query_vector: Embedding,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let table = self.get_table().await?;

        // Both knobs act only when the ANN index exists; without it the
        // search is the exact scan. Rows written since the index was last
        // extended are scanned exactly next to the indexed ones, so a fresh
        // file is found before any maintenance.
        let results = table
            .vector_search(query_vector)
            .map_err(|e| VectorStoreError::query(format!("Failed to create search: {}", e)))?
            .distance_type(DistanceType::Cosine)
            .nprobes(knob(ANN_NPROBES_ENV, DEFAULT_ANN_NPROBES)?)
            .refine_factor(refine_factor(
                knob(ANN_RERANK_ENV, DEFAULT_ANN_RERANK)?,
                limit,
            ))
            .limit(limit)
            .execute()
            .await
            .map_err(|e| VectorStoreError::query(format!("Search failed: {}", e)))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| VectorStoreError::query(format!("Failed to collect results: {}", e)))?;

        let mut search_results = Vec::new();

        for batch in batches {
            let id_col = batch
                .column_by_name("id")
                .ok_or_else(|| VectorStoreError::query("Missing id column"))?;
            let id_array = id_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| VectorStoreError::query("Invalid id column type"))?;

            let chunk_json_col = batch
                .column_by_name("chunk_json")
                .ok_or_else(|| VectorStoreError::query("Missing chunk_json column"))?;
            let chunk_json_array = chunk_json_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| VectorStoreError::query("Invalid chunk_json column type"))?;

            // LanceDB adds _distance column for search results
            let distance_col = batch
                .column_by_name("_distance")
                .ok_or_else(|| VectorStoreError::query("Missing _distance column"))?;
            let distance_array = distance_col
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| VectorStoreError::query("Invalid _distance column type"))?;

            for i in 0..batch.num_rows() {
                let id_str = id_array.value(i);
                let chunk_json = chunk_json_array.value(i);
                let distance = distance_array.value(i);

                let chunk_id = ChunkId::from_string(id_str).map_err(|e| {
                    VectorStoreError::serialization(format!("Invalid chunk ID: {:?}", e))
                })?;

                let chunk: CodeChunk = serde_json::from_str(chunk_json).map_err(|e| {
                    VectorStoreError::serialization(format!("Failed to deserialize chunk: {}", e))
                })?;

                // Convert cosine distance to similarity score
                // Cosine distance range is [0, 2], where 0 = identical, 2 = opposite
                // Convert to similarity [0, 1] (standard cosine similarity range)
                let score = 1.0 - (distance / 2.0);

                search_results.push(VectorSearchResult {
                    chunk_id,
                    score,
                    chunk,
                });
            }
        }

        Ok(search_results)
    }

    async fn chunks_by_ids(
        &self,
        chunk_ids: &[ChunkId],
    ) -> Result<HashMap<ChunkId, CodeChunk>, VectorStoreError> {
        if chunk_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let table = self.get_table().await?;

        // Ids are UUIDs: nothing in them needs quoting. The `id` column has a
        // BTree index (see `ensure_table_exists`), so this is a point lookup
        // per id, not a scan.
        let ids: Vec<String> = chunk_ids
            .iter()
            .map(|id| format!("'{}'", id.to_string()))
            .collect();
        let stream = table
            .query()
            .only_if(format!("id IN ({})", ids.join(", ")))
            .select(lancedb::query::Select::Columns(vec![
                "id".to_string(),
                "chunk_json".to_string(),
            ]))
            .execute()
            .await
            .map_err(|e| VectorStoreError::query(format!("Failed to look up chunks: {}", e)))?;
        let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(|e| {
            VectorStoreError::query(format!("Failed to collect chunk lookup: {}", e))
        })?;

        let mut chunks = HashMap::with_capacity(chunk_ids.len());
        for batch in &batches {
            chunks.extend(decode_chunk_rows(batch)?);
        }
        Ok(chunks)
    }

    async fn scan_chunks(
        &self,
        sink: &mut (dyn FnMut(Vec<CodeChunk>) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<usize> {
        let table = self.get_table().await?;

        // Only the two columns the chunk is made of: the vectors are most of
        // the table's bytes and none of the answer.
        let mut stream = table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "id".to_string(),
                "chunk_json".to_string(),
            ]))
            .execute()
            .await
            .map_err(|e| VectorStoreError::query(format!("Failed to scan chunks: {}", e)))?;

        let mut total = 0usize;
        while let Some(batch) = stream
            .try_next()
            .await
            .map_err(|e| VectorStoreError::query(format!("Failed to read chunk batch: {}", e)))?
        {
            let chunks: Vec<CodeChunk> = decode_chunk_rows(&batch)?
                .into_iter()
                .map(|(_, chunk)| chunk)
                .collect();
            total += chunks.len();
            sink(chunks)?;
        }
        Ok(total)
    }

    async fn delete_chunks(&self, chunk_ids: Vec<ChunkId>) -> Result<(), VectorStoreError> {
        if chunk_ids.is_empty() {
            return Ok(());
        }

        let table = self.get_table().await?;

        // Build filter for deletion
        let ids: Vec<String> = chunk_ids
            .iter()
            .map(|id| format!("'{}'", id.to_string()))
            .collect();
        let filter = format!("id IN ({})", ids.join(", "));

        table
            .delete(&filter)
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to delete chunks: {}", e)))?;

        tracing::debug!("Deleted {} chunks from LanceDB", chunk_ids.len());
        Ok(())
    }

    async fn delete_by_file_paths(&self, file_paths: &[String]) -> Result<(), VectorStoreError> {
        if file_paths.is_empty() {
            return Ok(());
        }
        let table = self.get_table().await?;

        // SQL string literals: a single quote is escaped by doubling it.
        let literals: Vec<String> = file_paths
            .iter()
            .map(|path| format!("'{}'", path.replace('\'', "''")))
            .collect();
        let filter = format!("file_path IN ({})", literals.join(", "));

        table
            .delete(&filter)
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to delete by path: {}", e)))?;

        tracing::debug!("Deleted chunks of {} file(s)", file_paths.len());
        Ok(())
    }

    async fn layout(&self) -> Result<VectorStoreLayout, VectorStoreError> {
        let table = self.get_table().await?;
        let backend = |what: &str, e: lancedb::Error| {
            VectorStoreError::backend(format!("Failed to read {what}: {e}"))
        };
        let version = table
            .version()
            .await
            .map_err(|e| backend("table version", e))?;
        let stats = table.stats().await.map_err(|e| backend("table stats", e))?;
        let mut indices = Vec::new();
        for config in table
            .list_indices()
            .await
            .map_err(|e| backend("indices", e))?
        {
            // `None`: the index was dropped since the listing — `maintain`
            // of another daemon on the same table can do that. Gone is gone.
            let Some(coverage) = table
                .index_stats(&config.name)
                .await
                .map_err(|e| backend("index stats", e))?
            else {
                continue;
            };
            indices.push(VectorIndexCoverage {
                name: config.name,
                columns: config.columns,
                indexed_rows: coverage.num_indexed_rows,
                unindexed_rows: coverage.num_unindexed_rows,
            });
        }
        Ok(VectorStoreLayout {
            version,
            fragments: stats.fragment_stats.num_fragments,
            rows: stats.num_rows,
            indices,
        })
    }

    async fn count(&self) -> Result<usize, VectorStoreError> {
        let table = self.get_table().await?;

        let count = table
            .count_rows(None)
            .await
            .map_err(|e| VectorStoreError::query(format!("Failed to count rows: {}", e)))?;

        Ok(count)
    }

    async fn indexed_file_paths(&self) -> Result<HashSet<String>, VectorStoreError> {
        let table = self.get_table().await?;

        // Column projection matters here: the table also holds the
        // vectors and the serialized chunk JSON, and pulling those to
        // count files would read hundreds of megabytes to answer a
        // question about a single string column.
        let stream = table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "file_path".to_string(),
            ]))
            .execute()
            .await
            .map_err(|e| {
                VectorStoreError::query(format!("Failed to scan file_path column: {}", e))
            })?;

        let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(|e| {
            VectorStoreError::query(format!("Failed to collect file_path batches: {}", e))
        })?;

        let mut paths = HashSet::new();
        for batch in batches {
            let column = batch
                .column_by_name("file_path")
                .ok_or_else(|| VectorStoreError::query("Missing file_path column"))?;
            let array = column
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| VectorStoreError::query("file_path column is not Utf8"))?;
            for i in 0..array.len() {
                if array.is_valid(i) {
                    paths.insert(array.value(i).to_string());
                }
            }
        }

        Ok(paths)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let table = self.get_table().await?;

        // Delete all rows
        table
            .delete("1=1")
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to clear table: {}", e)))?;

        tracing::info!("Cleared all vectors from LanceDB table");
        Ok(())
    }

    async fn maintain(
        &self,
        prune_keep: std::time::Duration,
    ) -> Result<VectorStoreMaintenance, VectorStoreError> {
        let table = self.get_table().await?;

        // Indices nothing reads. Dropped here rather than left in place:
        // the index pass below would otherwise extend them every time.
        let indices = table
            .list_indices()
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to list indices: {}", e)))?;
        for config in indices {
            if config
                .columns
                .iter()
                .any(|column| DROPPED_INDEX_COLUMNS.contains(&column.as_str()))
            {
                table.drop_index(&config.name).await.map_err(|e| {
                    VectorStoreError::backend(format!(
                        "Failed to drop index {}: {}",
                        config.name, e
                    ))
                })?;
                tracing::info!("Dropped unused vector-store index {}", config.name);
            }
        }

        // Compact only once enough small fragments piled up. Every write of
        // the background sync appends one small fragment, so compacting on
        // every pass means compacting every five minutes — and with lance's
        // default target (1M rows) the store's main fragment is "small" too,
        // so each pass rewrote the WHOLE table to absorb a handful of rows.
        // Measured 2026-10-19 on a large workspace's store (891 883 rows, 2.9 GB):
        // 8 appended rows -> the 2.9 GB fragment rewritten, 10.6 s, process
        // peak +2.4 GB, ~1.8 GB of heap left behind, on every changed sync.
        let small_fragments = table
            .stats()
            .await
            .map_err(|e| VectorStoreError::backend(format!("Failed to read table stats: {}", e)))?
            .fragment_stats
            .num_small_fragments;
        let compacted = if small_fragments >= COMPACT_MIN_SMALL_FRAGMENTS {
            // Compaction FIRST, and the order is not cosmetic: pruning is
            // what deletes files, but a fragment stays referenced by the
            // current version until compaction has rewritten its rows
            // elsewhere. Pruned-then-compacted would drop the manifests and
            // leave the data, which is the larger half.
            table
                .optimize(OptimizeAction::Compact {
                    options: compaction_options(),
                    remap_options: None,
                })
                .await
                .map_err(|e| {
                    VectorStoreError::backend(format!("Failed to compact vector table: {}", e))
                })?
                .compaction
        } else {
            None
        };

        // Extend every index over the rows written since the last pass.
        // After compaction, which moves rows and remaps the indices to their
        // new addresses: indexing first would index rows about to move.
        // Lance does not do this on write, so without it an index covers
        // what existed when it was created — for ours, an empty table — and
        // every delete by `file_path` and lookup by `id` scans the column.
        // Measured on a 0.93M-row copy (2026-10-29): the pass takes 0.7 s;
        // the per-file delete drops from 20 to 5 ms.
        table
            .optimize(OptimizeAction::Index(OptimizeOptions::default()))
            .await
            .map_err(|e| {
                VectorStoreError::backend(format!("Failed to update vector-store indices: {}", e))
            })?;

        ensure_ann_index(
            &table,
            self.vector_dim,
            knob(ANN_MIN_ROWS_ENV, DEFAULT_ANN_MIN_ROWS)?,
        )
        .await?;

        // `older_than` is the load-bearing argument, and it must be passed:
        // lance defaults it to SEVEN DAYS, and on a store written every few
        // minutes that keeps essentially every version there is.
        //
        // `delete_unverified` is deliberately left at lance's default.
        // Reading its implementation rather than its name: it governs only
        // files that NO manifest references — the debris of a write that
        // may still be in flight. Files held by a manifest we are dropping
        // are deleted whatever their age, so the flag buys us almost
        // nothing here while carrying the documented risk of deleting a
        // concurrent writer's in-progress files. Measured, not assumed:
        // flipping it changed neither the reclaimed bytes nor the version
        // count in `maintain_reclaims_versions_without_losing_rows`.
        let keep = ChronoDuration::from_std(prune_keep)
            .map_err(|e| VectorStoreError::backend(format!("prune window out of range: {}", e)))?;
        let pruned = table
            .optimize(OptimizeAction::Prune {
                older_than: Some(keep),
                delete_unverified: None,
                error_if_tagged_old_versions: None,
            })
            .await
            .map_err(|e| {
                VectorStoreError::backend(format!("Failed to prune vector table: {}", e))
            })?;

        let compaction = compacted.unwrap_or_default();
        let removal = pruned.prune;
        Ok(VectorStoreMaintenance {
            fragments_removed: compaction.fragments_removed,
            fragments_added: compaction.fragments_added,
            old_versions_removed: removal.as_ref().map(|s| s.old_versions).unwrap_or(0),
            bytes_removed: removal.as_ref().map(|s| s.bytes_removed).unwrap_or(0),
        })
    }

    async fn health_check(&self) -> Result<(), VectorStoreError> {
        // Try to list tables as a health check
        self.db
            .table_names()
            .execute()
            .await
            .map_err(|e| VectorStoreError::connection(format!("Health check failed: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::ChunkContext;
    use tempfile::TempDir;

    fn create_test_chunk(id: ChunkId, content: &str, file_path: &str) -> CodeChunk {
        CodeChunk {
            id,
            content: content.to_string(),
            context: ChunkContext {
                file_path: PathBuf::from(file_path),
                module_path: vec!["crate".to_string()],
                symbol_name: "test_func".to_string(),
                symbol_kind: "function".to_string(),
                docstring: None,
                imports: vec![],
                outgoing_calls: vec![],
                parent_symbol_name: None,
                split_part: None,
                split_total: None,
                line_start: 1,
                line_end: 10,
            },
            overlap_prev: None,
            overlap_next: None,
        }
    }

    #[tokio::test]
    async fn test_lancedb_creation() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(
            temp_dir.path().to_path_buf(),
            1024, // Qwen3-Embedding-0.6B
            "test-embedder:v1",
        )
        .await;
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_lancedb_open_existing_missing_path_does_not_create() {
        let temp_dir = TempDir::new().unwrap();
        let vector_path = temp_dir.path().join("missing");

        let result =
            LanceDbBackend::open_existing(vector_path.clone(), 4, "test-embedder:v1").await;

        assert!(matches!(result, Err(VectorStoreError::NotFound(_))));
        assert!(!vector_path.exists());
    }

    #[tokio::test]
    async fn test_lancedb_open_existing_missing_metadata_does_not_create_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let vector_path = temp_dir.path().join("vectors");
        std::fs::create_dir(&vector_path).unwrap();

        let result =
            LanceDbBackend::open_existing(vector_path.clone(), 4, "test-embedder:v1").await;

        assert!(matches!(result, Err(VectorStoreError::NotFound(_))));
        assert!(!vector_path.join(METADATA_FILE).exists());
    }

    #[tokio::test]
    async fn test_lancedb_open_existing_missing_table_does_not_create_table() {
        let temp_dir = TempDir::new().unwrap();
        let vector_path = temp_dir.path().join("vectors");
        std::fs::create_dir(&vector_path).unwrap();
        std::fs::write(
            vector_path.join(METADATA_FILE),
            serde_json::json!({
                "embedder_version": "test-embedder:v1",
                "created_at": "test",
            })
            .to_string(),
        )
        .unwrap();

        let result =
            LanceDbBackend::open_existing(vector_path.clone(), 4, "test-embedder:v1").await;

        assert!(matches!(result, Err(VectorStoreError::NotFound(_))));
        let db = connect(vector_path.to_string_lossy().as_ref())
            .execute()
            .await
            .unwrap();
        let tables = db.table_names().execute().await.unwrap();
        assert!(!tables.iter().any(|name| name == TABLE_NAME));
    }

    #[tokio::test]
    async fn test_lancedb_open_existing_valid_store() {
        let temp_dir = TempDir::new().unwrap();
        LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();

        let backend =
            LanceDbBackend::open_existing(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
                .await
                .unwrap();

        assert_eq!(backend.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_lancedb_upsert_and_count() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();

        let chunk_id = ChunkId::new();
        let chunk = create_test_chunk(chunk_id, "fn test() {}", "test.rs");
        let embedding = vec![0.1, 0.2, 0.3, 0.4];

        backend
            .insert_chunks(vec![(chunk_id, embedding, chunk)])
            .await
            .unwrap();

        let count = backend.count().await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_lancedb_delete() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();

        let chunk_id = ChunkId::new();
        let chunk = create_test_chunk(chunk_id, "fn test() {}", "test.rs");
        let embedding = vec![0.1, 0.2, 0.3, 0.4];

        backend
            .insert_chunks(vec![(chunk_id, embedding, chunk)])
            .await
            .unwrap();

        assert_eq!(backend.count().await.unwrap(), 1);

        backend.delete_chunks(vec![chunk_id]).await.unwrap();

        assert_eq!(backend.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_lancedb_delete_by_file_path() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();

        let chunk1_id = ChunkId::new();
        let chunk1 = create_test_chunk(chunk1_id, "fn test1() {}", "file1.rs");

        let chunk2_id = ChunkId::new();
        let chunk2 = create_test_chunk(chunk2_id, "fn test2() {}", "file2.rs");

        backend
            .insert_chunks(vec![
                (chunk1_id, vec![0.1, 0.2, 0.3, 0.4], chunk1),
                (chunk2_id, vec![0.5, 0.6, 0.7, 0.8], chunk2),
            ])
            .await
            .unwrap();

        assert_eq!(backend.count().await.unwrap(), 2);

        backend
            .delete_by_file_paths(&["file1.rs".to_string()])
            .await
            .unwrap();

        assert_eq!(backend.count().await.unwrap(), 1);
    }

    /// Several files go in ONE write — one version — and a quote in a path
    /// is a literal, not the end of the string. An empty list writes nothing.
    #[tokio::test]
    async fn deleting_several_files_is_one_version() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();
        let files = ["a.rs", "b's.rs", "c.rs", "kept.rs"];
        let rows = files
            .iter()
            .map(|file| {
                let id = ChunkId::new();
                (
                    id,
                    vec![0.1, 0.2, 0.3, 0.4],
                    create_test_chunk(id, "fn f() {}", file),
                )
            })
            .collect();
        backend.insert_chunks(rows).await.unwrap();

        let before = backend.layout().await.unwrap().version;
        backend.delete_by_file_paths(&[]).await.unwrap();
        assert_eq!(backend.layout().await.unwrap().version, before);

        let gone: Vec<String> = files[..3].iter().map(|f| f.to_string()).collect();
        backend.delete_by_file_paths(&gone).await.unwrap();
        assert_eq!(backend.layout().await.unwrap().version, before + 1);
        assert_eq!(
            backend.indexed_file_paths().await.unwrap(),
            HashSet::from(["kept.rs".to_string()])
        );
    }

    #[test]
    fn the_rerank_budget_becomes_a_refine_factor_per_limit() {
        assert_eq!(refine_factor(2000, 10), 200);
        assert_eq!(refine_factor(2000, 100), 20);
        assert_eq!(
            refine_factor(2000, 30),
            67,
            "rounded up: at least the budget"
        );
        assert_eq!(refine_factor(2000, 5000), 1, "never fewer than the results");
        assert_eq!(refine_factor(2000, 0), 2000);
        assert_eq!(ann_sub_vectors(384), Some(48));
        assert_eq!(ann_sub_vectors(1024), Some(128));
        assert_eq!(ann_sub_vectors(4), None);
    }

    /// Deterministic, spread-out unit vectors: row `i` of a `dim`-wide
    /// table, every `i` its own (a splitmix64 hash per coordinate).
    fn spread_vector(i: usize, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim)
            .map(|d| {
                let mut z = ((i * dim + d) as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.into_iter().map(|x| x / norm).collect()
    }

    /// The ANN index is built only past the row threshold, once; a search
    /// through it still finds rows written after it was built (the
    /// unindexed tail is scanned exactly) — a file synced a minute ago is
    /// not invisible until the next maintenance — and the next maintenance
    /// extends the index over them.
    #[tokio::test]
    async fn the_ann_index_is_built_past_the_threshold_and_misses_no_fresh_row() {
        const DIM: usize = 32;
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), DIM, "test-embedder:v1")
            .await
            .unwrap();
        let rows = (0..2000)
            .map(|i| {
                let id = ChunkId::new();
                (
                    id,
                    spread_vector(i, DIM),
                    create_test_chunk(id, "fn f() {}", "old.rs"),
                )
            })
            .collect();
        backend.insert_chunks(rows).await.unwrap();
        let table = backend.get_table().await.unwrap();

        assert!(
            !ensure_ann_index(&table, DIM, 2001).await.unwrap(),
            "below the threshold"
        );
        assert!(
            !ensure_ann_index(&table, DIM, 0).await.unwrap(),
            "0 turns it off"
        );
        assert!(ensure_ann_index(&table, DIM, 2000).await.unwrap());
        assert!(
            !ensure_ann_index(&table, DIM, 2000).await.unwrap(),
            "built once"
        );
        let ann = |layout: &VectorStoreLayout| {
            layout
                .indices
                .iter()
                .find(|index| index.name == VECTOR_INDEX_NAME)
                .cloned()
                .expect("the ANN index exists")
        };
        assert_eq!(ann(&backend.layout().await.unwrap()).indexed_rows, 2000);

        let fresh_id = ChunkId::new();
        let fresh_vector = spread_vector(5000, DIM);
        backend
            .insert_chunks(vec![(
                fresh_id,
                fresh_vector.clone(),
                create_test_chunk(fresh_id, "fn fresh() {}", "fresh.rs"),
            )])
            .await
            .unwrap();
        assert_eq!(ann(&backend.layout().await.unwrap()).unindexed_rows, 1);
        let hits = backend.search(fresh_vector.clone(), 5).await.unwrap();
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(fresh_id));

        backend
            .maintain(std::time::Duration::from_secs(3600))
            .await
            .unwrap();
        let after = ann(&backend.layout().await.unwrap());
        assert_eq!((after.indexed_rows, after.unindexed_rows), (2001, 0));
        let hits = backend.search(fresh_vector, 5).await.unwrap();
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(fresh_id));
    }

    /// The oracle for the index pass in `maintain`: indices created on the
    /// empty table end up covering every row, and the unused `symbol_kind`
    /// index of a table created before it was dropped is gone. The mutant
    /// (no `OptimizeAction::Index`) leaves `id` and `file_path` covering 0.
    #[tokio::test]
    async fn maintain_extends_the_indices_over_every_row() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();
        // A table as older builds created it.
        backend
            .get_table()
            .await
            .unwrap()
            .create_index(&["symbol_kind"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await
            .unwrap();
        for i in 0..3 {
            let rows = (0..20)
                .map(|j| {
                    let id = ChunkId::new();
                    let chunk = create_test_chunk(id, "fn f() {}", &format!("f{i}_{j}.rs"));
                    (id, vec![0.1, 0.2, 0.3, j as f32], chunk)
                })
                .collect();
            backend.insert_chunks(rows).await.unwrap();
        }
        let before = backend.layout().await.unwrap();
        assert!(
            before.indices.iter().all(|index| index.indexed_rows == 0),
            "{before:#?}"
        );

        backend
            .maintain(std::time::Duration::from_secs(3600))
            .await
            .unwrap();

        let after = backend.layout().await.unwrap();
        assert_eq!(after.rows, 60);
        let mut columns: Vec<&str> = after
            .indices
            .iter()
            .flat_map(|index| index.columns.iter().map(String::as_str))
            .collect();
        columns.sort();
        let mut expected = SCALAR_INDEX_COLUMNS.to_vec();
        expected.sort();
        assert_eq!(columns, expected, "{after:#?}");
        for index in &after.indices {
            assert_eq!(
                (index.indexed_rows, index.unindexed_rows),
                (60, 0),
                "{index:#?}"
            );
        }
    }

    #[tokio::test]
    async fn test_lancedb_clear() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();

        let chunk_id = ChunkId::new();
        let chunk = create_test_chunk(chunk_id, "fn test() {}", "test.rs");
        let embedding = vec![0.1, 0.2, 0.3, 0.4];

        backend
            .insert_chunks(vec![(chunk_id, embedding, chunk)])
            .await
            .unwrap();

        assert_eq!(backend.count().await.unwrap(), 1);

        backend.clear().await.unwrap();

        assert_eq!(backend.count().await.unwrap(), 0);
    }

    /// Write enough separate batches that the append-only store has real
    /// fragments and versions to reclaim, then check that maintenance
    /// reclaims them WITHOUT losing a row.
    ///
    /// Both halves matter: a pass that drops versions and also drops data
    /// would satisfy the first assert alone, and a pass that silently does
    /// nothing would satisfy the second alone.
    #[tokio::test]
    async fn maintain_reclaims_versions_without_losing_rows() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();

        // One upsert per call — batching them would produce a single
        // fragment and the scene would have nothing to compact.
        for i in 0..12 {
            let id = ChunkId::new();
            let chunk = create_test_chunk(id, "fn test() {}", &format!("f{}.rs", i));
            backend
                .insert_chunks(vec![(id, vec![0.1, 0.2, 0.3, i as f32], chunk)])
                .await
                .unwrap();
        }
        let rows_before = backend.count().await.unwrap();
        assert_eq!(rows_before, 12);

        // FIRST pass, with a keep window wider than the scene's whole
        // lifetime. Two things are judged here, and the second is the
        // positive control for the pass that follows.
        let first = backend
            .maintain(std::time::Duration::from_secs(3600))
            .await
            .unwrap();
        // Compaction is judged on THIS pass and only here: it is the pass
        // that finds the twelve one-row fragments. Asserting it on the
        // second pass would assert nothing — by then they are already
        // merged, and a build with compaction removed would look
        // identical.
        assert!(
            first.fragments_removed > first.fragments_added,
            "compaction must merge the per-write fragments, got {:?}",
            first
        );
        // The control: `older_than` is honoured, so a wide window protects
        // versions written seconds ago. Without this the prune assert
        // below would also pass on a build that ignores the window and
        // prunes unconditionally.
        assert_eq!(
            first.old_versions_removed, 0,
            "keep window of an hour must protect versions written seconds ago"
        );

        // SECOND pass, window closed: now the superseded versions — the
        // originals plus the ones compaction just replaced — must go.
        let stats = backend.maintain(std::time::Duration::ZERO).await.unwrap();
        assert!(
            stats.old_versions_removed > 0,
            "12 separate writes must leave prunable versions, got {:?}",
            stats
        );
        // Bytes, not just manifest entries: `old_versions_removed` counts
        // bookkeeping, and only this number answers the question the pass
        // exists for.
        assert!(
            stats.bytes_removed > 0,
            "pruning must free bytes on disk, not just manifest entries, got {:?}",
            stats
        );
        assert_eq!(
            backend.count().await.unwrap(),
            rows_before,
            "maintenance must not change what the store holds"
        );

        // And the store still answers: reclaiming must not orphan the
        // data files the current version points at.
        let hits = backend.search(vec![0.1, 0.2, 0.3, 0.0], 5).await.unwrap();
        assert!(!hits.is_empty(), "search must still work after maintenance");
    }

    /// `VmRSS` and `VmHWM` of this process, in MiB.
    fn rss_and_peak_mib() -> (u64, u64) {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let field = |name: &str| {
            status
                .lines()
                .find_map(|l| l.strip_prefix(name))
                .unwrap()
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse::<u64>()
                .unwrap()
                / 1024
        };
        (field("VmRSS:"), field("VmHWM:"))
    }

    /// Queries for the recall probe: searches of the kind agents run on the
    /// store under test, phrases and identifiers. One per line in the file
    /// named by `RMC_LANCE_PROBE_QUERIES`; blank lines and `#` comments are
    /// skipped. They belong to the store's project, so they live with it and
    /// not in this repository.
    fn probe_queries() -> Vec<String> {
        let path = std::env::var("RMC_LANCE_PROBE_QUERIES").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let queries: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_string)
            .collect();
        assert!(!queries.is_empty(), "{path} holds no queries");
        queries
    }

    /// Cumulative user+system CPU seconds of this process.
    fn cpu_secs() -> f64 {
        let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
        // Fields after the parenthesised command: utime is the 12th, stime the 13th.
        let after = &stat[stat.rfind(')').unwrap() + 2..];
        let fields: Vec<&str> = after.split(' ').collect();
        let ticks: f64 = fields[11].parse::<f64>().unwrap() + fields[12].parse::<f64>().unwrap();
        ticks / 100.0
    }

    /// Recall probe, not a test: is the ANN index close enough to the exact
    /// scan on queries people actually run? Point `RMC_LANCE_PROBE_DIR` at a
    /// COPY of a live vector store (it builds the index there) and
    /// `RMC_LANCE_PROBE_EMBEDDER` at the `embedder_version` in its
    /// `metadata.json`; dim 384 (bge-small).
    ///
    /// The queries are embedded with the CPU bge-small profile (quantised
    /// twin of the GPU one: a slightly different vector, the same kind of
    /// query). Recall compares ANN and exact search for the SAME vector, so
    /// that difference does not enter it. Prints recall@10, latency and
    /// CPU per query for each nprobes / refine_factor point.
    #[tokio::test]
    #[ignore]
    async fn probe_ann_recall_on_real_queries() {
        use crate::embeddings::{EmbeddingBackend, EmbeddingGenerator, EmbeddingOwner};
        use lancedb::query::Select;

        let dir = std::env::var("RMC_LANCE_PROBE_DIR").unwrap();
        let embedder = std::env::var("RMC_LANCE_PROBE_EMBEDDER").unwrap();
        let backend = LanceDbBackend::open_existing(PathBuf::from(dir), 384, &embedder)
            .await
            .unwrap();
        let table = backend.get_table().await.unwrap();
        let generator = EmbeddingGenerator::with_backend(
            EmbeddingBackend::from_profile_name("local-cpu-small").unwrap(),
            EmbeddingOwner::Tool,
        )
        .unwrap();
        let vectors = generator.embed_queries(probe_queries()).await.unwrap();
        eprintln!(
            "{} rows, {} queries",
            backend.count().await.unwrap(),
            vectors.len()
        );
        // `RMC_LANCE_PROBE_LIMIT`: results per query, 10 unless set. Hybrid
        // search asks the vector side for 100 candidates, and refine_factor
        // multiplies the limit, so the choice is measured at 100 too.
        let limit: usize = std::env::var("RMC_LANCE_PROBE_LIMIT")
            .map(|v| v.parse().unwrap())
            .unwrap_or(10);

        // (nprobes, refine_factor); `None` = the exact scan.
        let run = |point: Option<(usize, u32)>| {
            let table = table.clone();
            let vectors = vectors.clone();
            async move {
                // Per query: the ids found and their distances, nearest first.
                let mut results: Vec<(HashSet<String>, Vec<f32>)> = Vec::new();
                let mut latencies = Vec::new();
                let cpu0 = cpu_secs();
                for vector in vectors {
                    let mut query = table
                        .vector_search(vector)
                        .unwrap()
                        .distance_type(DistanceType::Cosine)
                        .select(Select::columns(&["id"]))
                        .limit(limit);
                    query = match point {
                        None => query.bypass_vector_index(),
                        Some((nprobes, refine)) => query.nprobes(nprobes).refine_factor(refine),
                    };
                    let started = std::time::Instant::now();
                    let batches: Vec<RecordBatch> =
                        query.execute().await.unwrap().try_collect().await.unwrap();
                    latencies.push(started.elapsed().as_secs_f64() * 1e3);
                    let mut ids = HashSet::new();
                    let mut distances = Vec::new();
                    for batch in &batches {
                        let column = batch.column_by_name("id").unwrap();
                        let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                        ids.extend((0..array.len()).map(|i| array.value(i).to_string()));
                        let column = batch.column_by_name("_distance").unwrap();
                        let array = column.as_any().downcast_ref::<Float32Array>().unwrap();
                        distances.extend(array.values().iter().copied());
                    }
                    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    results.push((ids, distances));
                }
                let cpu = (cpu_secs() - cpu0) / results.len() as f64;
                latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (results, latencies[latencies.len() / 2], cpu)
            }
        };

        let (exact, p50, cpu) = run(None).await;
        eprintln!("exact scan: p50 {p50:.0} ms, {cpu:.3} CPU-s/query");

        // `RMC_LANCE_PROBE_SUB_VECTORS`: PQ sub-vectors of the index built
        // here (unset — what `maintain` builds); an existing index is
        // replaced.
        let sub_vectors: Option<u32> = std::env::var("RMC_LANCE_PROBE_SUB_VECTORS")
            .ok()
            .map(|v| v.parse().unwrap())
            .or(ann_sub_vectors(384));
        let cpu0 = cpu_secs();
        let started = std::time::Instant::now();
        let mut builder = IvfPqIndexBuilder::default().distance_type(DistanceType::Cosine);
        if let Some(n) = sub_vectors {
            builder = builder.num_sub_vectors(n);
        }
        table
            .create_index(&[VECTOR_COLUMN], Index::IvfPq(builder))
            .name(VECTOR_INDEX_NAME.to_string())
            .replace(true)
            .execute()
            .await
            .unwrap();
        eprintln!(
            "IVF_PQ sub_vectors={sub_vectors:?} built in {:.1}s, {:.0} CPU-s",
            started.elapsed().as_secs_f64(),
            cpu_secs() - cpu0
        );

        // The defaults `search` runs with first, then neighbours of them.
        let defaults = (
            DEFAULT_ANN_NPROBES,
            refine_factor(DEFAULT_ANN_RERANK, limit),
        );
        for point in [
            defaults,
            (20, defaults.1),
            (60, defaults.1),
            (40, 10),
            (40, 50),
        ] {
            let (got, p50, cpu) = run(Some(point)).await;
            // By id, and by distance: a found row counts when it is no
            // farther than the exact 10th. The second does not punish a tie
            // — two rows with the same vector (duplicate chunks) are equally
            // right answers, but only one of their ids is in the exact set.
            let by_id: Vec<usize> = exact
                .iter()
                .zip(&got)
                .map(|((truth, _), (found, _))| truth.intersection(found).count())
                .collect();
            let by_distance: Vec<usize> = exact
                .iter()
                .zip(&got)
                .map(|((_, truth), (_, found))| {
                    let tenth = truth.last().copied().unwrap_or(f32::MAX);
                    found.iter().filter(|d| **d <= tenth + 1e-6).count()
                })
                .collect();
            let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / (limit * v.len()) as f64;
            eprintln!(
                "nprobes {:>2} refine {:>3}: recall@{limit} by id {:.3}, by distance {:.3} (worst {}/{limit}), p50 {p50:.1} ms, {cpu:.4} CPU-s/query",
                point.0,
                point.1,
                mean(&by_id),
                mean(&by_distance),
                by_distance.iter().min().unwrap(),
            );
        }
    }

    /// Memory probe, not a test: what ONE background-sync cycle's
    /// maintenance costs on a real store. Point `RMC_LANCE_PROBE_DIR` at a
    /// COPY of a live vector store (maintenance rewrites it) and
    /// `RMC_LANCE_PROBE_EMBEDDER` at the `embedder_version` in its
    /// `metadata.json`; dim is 384 (bge-small).
    ///
    /// Each round repeats the sync's own shape — a few new rows appended as
    /// a fresh fragment, then `maintain` — and prints RSS before, after and
    /// the process peak, so the gigabytes are attributed by measurement.
    #[tokio::test]
    #[ignore]
    async fn probe_maintain_memory_on_real_store() {
        let dir = std::env::var("RMC_LANCE_PROBE_DIR").unwrap();
        let embedder = std::env::var("RMC_LANCE_PROBE_EMBEDDER").unwrap();
        let backend = LanceDbBackend::open_existing(PathBuf::from(dir), 384, &embedder)
            .await
            .unwrap();
        let rows = backend.count().await.unwrap();
        let (rss, peak) = rss_and_peak_mib();
        eprintln!("opened: {rows} rows, RSS {rss} MiB, peak {peak} MiB");
        for round in 0..10 {
            let mut batch = Vec::new();
            for i in 0..8 {
                let id = ChunkId::new();
                let chunk =
                    create_test_chunk(id, "fn probe() {}", &format!("probe_{round}_{i}.rs"));
                batch.push((id, vec![0.01 * i as f32; 384], chunk));
            }
            backend.insert_chunks(batch).await.unwrap();
            let (before, _) = rss_and_peak_mib();
            let started = std::time::Instant::now();
            let stats = backend
                .maintain(std::time::Duration::from_secs(0))
                .await
                .unwrap();
            let (after, peak) = rss_and_peak_mib();
            eprintln!(
                "round {round}: {stats:?} in {:.1}s; RSS {before} -> {after} MiB, process peak {peak} MiB",
                started.elapsed().as_secs_f64()
            );
        }
    }

    #[tokio::test]
    async fn test_lancedb_health_check() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path().to_path_buf(), 4, "test-embedder:v1")
            .await
            .unwrap();

        let result = backend.health_check().await;
        assert!(result.is_ok());
    }
}
