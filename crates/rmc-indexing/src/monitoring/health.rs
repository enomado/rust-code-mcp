//! Health monitoring for production deployments
//!
//! Provides component-level health checks for:
//! - BM25 search (Tantivy)
//! - Vector search (LanceDB)
//! - Merkle tree snapshots
//!
//! Health states: Healthy, Degraded, Unhealthy

use crate::indexing::FileSystemMerkle;
use crate::metadata_cache::MetadataCache;
use rmc_engine::search::Bm25Search;
use rmc_engine::vector_store::VectorStore;
use serde::Serialize;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Overall system health status
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    /// Overall system status
    pub overall: Status,
    /// BM25 search component health
    pub bm25: ComponentHealth,
    /// Vector search component health
    pub vector: ComponentHealth,
    /// Merkle tree component health
    pub merkle: ComponentHealth,
    /// Index coverage: does the store actually hold what the indexer
    /// believes it already indexed?
    pub coverage: CoverageHealth,
}

/// Coverage report — the answer to "is this index complete?", which
/// none of the other three components can give.
///
/// The failure this exists for: an indexing run that dies halfway still
/// leaves a Merkle snapshot and a metadata cache behind. The next run
/// honestly considers those files unchanged (`Parser error: File
/// unchanged`) and skips them, so the missing vectors are never written
/// — and every component above reports `healthy`, because the store is
/// alive and the snapshot exists. Observed 2026-08-18 on `rust_app`:
/// 498 files indexed, 3437 skipped, 136 352 vectors instead of 465 350,
/// overall status `healthy`.
///
/// The verdict is deliberately built on `files_cached` vs
/// `files_with_vectors`, not on the Merkle count: a file with no
/// symbols legitimately produces no chunks and therefore no vectors,
/// and it never enters the metadata cache either (the cache is written
/// only after a successful upsert). So `stale_skips` has no false
/// positives by construction, while "tracked minus vectors" would have
/// them. Both numbers are reported; only one is a verdict.
#[derive(Debug, Clone, Serialize)]
pub struct CoverageHealth {
    /// Coverage status
    pub status: Status,
    /// Human-readable summary
    pub message: String,
    /// Files tracked by the Merkle snapshot (what the project contains)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_tracked: Option<usize>,
    /// Files the metadata cache calls indexed (what the next run will skip)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_cached: Option<usize>,
    /// Distinct files that actually have at least one vector in the store
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_with_vectors: Option<usize>,
    /// Files the indexer will skip as unchanged although the store holds
    /// nothing for them. Non-zero means the index is silently incomplete
    /// and will stay so until a forced reindex.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_skips: Option<usize>,
    /// A few example paths from `stale_skips`, for triage
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stale_skip_examples: Vec<String>,
}

impl CoverageHealth {
    /// Coverage could not be computed (missing snapshot, cache or store).
    ///
    /// Reported as degraded, never healthy: "we did not look" must not
    /// read as "we looked and it is fine".
    fn unknown(message: impl Into<String>) -> Self {
        Self {
            status: Status::Degraded,
            message: message.into(),
            files_tracked: None,
            files_cached: None,
            files_with_vectors: None,
            stale_skips: None,
            stale_skip_examples: Vec::new(),
        }
    }
}

/// Health status levels
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// All systems operational
    Healthy,
    /// Some systems degraded but functional
    Degraded,
    /// Critical systems failing
    Unhealthy,
}

/// Individual component health
#[derive(Debug, Clone, Serialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: Status,
    /// Status message
    pub message: String,
    /// Optional latency measurement in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

impl ComponentHealth {
    /// Create healthy component status
    pub fn healthy(message: impl Into<String>, latency_ms: Option<u64>) -> Self {
        Self {
            status: Status::Healthy,
            message: message.into(),
            latency_ms,
        }
    }

    /// Create degraded component status
    pub fn degraded(message: impl Into<String>) -> Self {
        Self {
            status: Status::Degraded,
            message: message.into(),
            latency_ms: None,
        }
    }

    /// Create unhealthy component status
    pub fn unhealthy(message: impl Into<String>) -> Self {
        Self {
            status: Status::Unhealthy,
            message: message.into(),
            latency_ms: None,
        }
    }
}

/// Read the metadata cache and return the file paths it holds for this
/// chunking identity.
///
/// Keys are `salt::path` (or a bare path when the salt is empty), and an
/// entry under a DIFFERENT salt belongs to another chunking config — it
/// says nothing about the index being checked, so it is dropped rather
/// than counted.
fn read_cached_paths(cache_path: &PathBuf, salt: &str) -> Result<HashSet<String>, String> {
    let cache = MetadataCache::new(cache_path).map_err(|e| e.to_string())?;
    let files = cache.list_files().map_err(|e| e.to_string())?;
    Ok(cached_paths_for_salt(files, salt))
}

/// Strip the salt prefix, keeping only keys that carry it.
fn cached_paths_for_salt(keys: Vec<String>, salt: &str) -> HashSet<String> {
    if salt.is_empty() {
        return keys.into_iter().collect();
    }
    let prefix = format!("{}::", salt);
    keys.into_iter()
        .filter_map(|key| key.strip_prefix(&prefix).map(str::to_string))
        .collect()
}

/// Health monitor for the search system
pub struct HealthMonitor {
    bm25: Option<Arc<Bm25Search>>,
    vector_store: Option<Arc<VectorStore>>,
    merkle_path: PathBuf,
    /// sled metadata cache of the project being checked, plus the salt
    /// its keys are prefixed with (the chunking identity). Absent for a
    /// system-wide probe, where there is no single project to speak of.
    metadata_cache: Option<(PathBuf, String)>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(
        bm25: Option<Arc<Bm25Search>>,
        vector_store: Option<Arc<VectorStore>>,
        merkle_path: PathBuf,
    ) -> Self {
        Self {
            bm25,
            vector_store,
            merkle_path,
            metadata_cache: None,
        }
    }

    /// Point the monitor at the project's metadata cache, enabling the
    /// coverage check.
    ///
    /// `cache_key_salt` must be the same chunking identity the indexer
    /// used (`IndexerCoreConfig::chunking_cache_salt`) — cache keys are
    /// `salt::path`, and reading them with the wrong salt would report
    /// an empty cache, i.e. a silent "nothing to compare".
    pub fn with_metadata_cache(mut self, cache_path: PathBuf, cache_key_salt: String) -> Self {
        self.metadata_cache = Some((cache_path, cache_key_salt));
        self
    }

    /// Perform comprehensive health check
    pub async fn check_health(&self) -> HealthStatus {
        // Run all checks in parallel
        let (bm25_health, vector_health, merkle_health, coverage) = tokio::join!(
            self.check_bm25(),
            self.check_vector(),
            self.check_merkle(),
            self.check_coverage()
        );

        // Determine overall status
        let overall =
            self.calculate_overall_status(&bm25_health, &vector_health, &merkle_health, &coverage);

        HealthStatus {
            overall,
            bm25: bm25_health,
            vector: vector_health,
            merkle: merkle_health,
            coverage,
        }
    }

    /// Check BM25 search health
    async fn check_bm25(&self) -> ComponentHealth {
        let Some(bm25) = &self.bm25 else {
            return ComponentHealth::degraded("BM25 search not configured");
        };

        let start = Instant::now();

        // Try a simple test query (BM25 search is synchronous)
        match bm25.search("__health_check__", 1) {
            Ok(_) => {
                let latency = start.elapsed().as_millis() as u64;
                ComponentHealth::healthy("BM25 search operational", Some(latency))
            }
            Err(e) => ComponentHealth::unhealthy(format!("BM25 search error: {:?}", e)),
        }
    }

    /// Check vector search health
    async fn check_vector(&self) -> ComponentHealth {
        let Some(vector_store) = &self.vector_store else {
            return ComponentHealth::degraded("Vector store not configured");
        };

        let start = Instant::now();

        // Check collection exists and is accessible
        match vector_store.count().await {
            Ok(count) => {
                let latency = start.elapsed().as_millis() as u64;
                ComponentHealth::healthy(
                    format!("Vector store operational ({} vectors)", count),
                    Some(latency),
                )
            }
            Err(e) => ComponentHealth::unhealthy(format!("Vector store error: {}", e)),
        }
    }

    /// Check Merkle tree snapshot health
    async fn check_merkle(&self) -> ComponentHealth {
        if self.merkle_path.exists() {
            match std::fs::metadata(&self.merkle_path) {
                Ok(metadata) => {
                    let size_bytes = metadata.len();
                    ComponentHealth::healthy(
                        format!("Merkle snapshot exists ({} bytes)", size_bytes),
                        None,
                    )
                }
                Err(e) => ComponentHealth::degraded(format!(
                    "Merkle snapshot exists but unreadable: {}",
                    e
                )),
            }
        } else {
            ComponentHealth::degraded("Merkle snapshot not found (first index pending)")
        }
    }

    /// Check index coverage (see [`CoverageHealth`])
    async fn check_coverage(&self) -> CoverageHealth {
        let Some(vector_store) = &self.vector_store else {
            return CoverageHealth::unknown(
                "Coverage unknown: vector store not open, nothing to compare against",
            );
        };
        let Some((cache_path, salt)) = &self.metadata_cache else {
            return CoverageHealth::unknown(
                "Coverage unknown: no project metadata cache (system-wide check; pass 'directory')",
            );
        };
        if !cache_path.exists() {
            return CoverageHealth::unknown(format!(
                "Coverage unknown: metadata cache missing at {} (project never indexed?)",
                cache_path.display()
            ));
        }

        // Files the indexer considers already done. Opened read-only in
        // spirit but sled has no such mode; a health probe must not fight
        // a running indexer over the lock, so a failure here is reported,
        // not retried.
        let cached_files = match read_cached_paths(cache_path, salt) {
            Ok(files) => files,
            Err(e) => {
                return CoverageHealth::unknown(format!(
                    "Coverage unknown: cannot read metadata cache at {}: {}",
                    cache_path.display(),
                    e
                ));
            }
        };

        let indexed_files = match vector_store.indexed_file_paths().await {
            Ok(files) => files,
            Err(e) => {
                return CoverageHealth::unknown(format!(
                    "Coverage unknown: cannot list indexed files in vector store: {}",
                    e
                ));
            }
        };

        // What the project actually contains, for context. Absent
        // snapshot is not fatal for the verdict — the verdict does not
        // depend on it.
        let files_tracked = FileSystemMerkle::load_snapshot(&self.merkle_path)
            .ok()
            .flatten()
            .map(|merkle| merkle.file_count());

        let mut stale: Vec<String> = cached_files
            .iter()
            .filter(|path| !indexed_files.contains(*path))
            .cloned()
            .collect();
        stale.sort();

        let files_cached = cached_files.len();
        let files_with_vectors = indexed_files.len();
        let stale_skips = stale.len();
        let examples: Vec<String> = stale.iter().take(10).cloned().collect();

        // Empty comparison is NOT a pass. The cache holds keys for one
        // embedder+chunking salt; a profile that was never indexed (or
        // whose salt changed, as happened when the embedder was added to
        // the salt) yields zero cached files, and "all 0 cached files
        // have vectors" is vacuously true — exactly the shape of verdict
        // this component exists to abolish. Caught on the live index
        // right after the salt change, where it printed `healthy` next
        // to `files_cached: 0`.
        //
        // "No cache entries" comes in TWO shapes that call for different
        // actions, so they get different messages. One text for both made
        // the cheap, frequent case (cache dropped or re-salted, vectors
        // intact) read exactly like the expensive, rare one (a run died
        // halfway) — whose printed remedy is a full `force_reindex`, an
        // hour of work this state does not call for. The shapes are
        // distinguishable by construction: the cache is written only
        // after a successful upsert, so a half-finished run leaves cache
        // entries behind; an empty cache next to a non-empty store means
        // the cache was lost, not the vectors.
        if files_cached == 0 {
            let message = if files_with_vectors == 0 {
                "Coverage unknown: nothing is indexed under this profile yet — \
                 neither cache entries nor vectors. Fix: a plain index_codebase run"
                    .to_string()
            } else {
                format!(
                    "Coverage unknown: store holds {} distinct files, but the metadata \
                     cache holds no entries for this profile (cleared, or written before \
                     the cache salt changed), so there is nothing to compare them against. \
                     This is lost bookkeeping, NOT evidence of a damaged index: a \
                     half-finished run would have left cache entries. Fix: a plain \
                     index_codebase run refills the cache for the files it touches; \
                     force_reindex is not indicated by this state",
                    files_with_vectors
                )
            };
            let mut unknown = CoverageHealth::unknown(message);
            unknown.files_tracked = files_tracked;
            unknown.files_cached = Some(0);
            unknown.files_with_vectors = Some(files_with_vectors);
            return unknown;
        }

        let (status, message) = if stale_skips == 0 {
            (
                Status::Healthy,
                format!(
                    "Coverage verified: all {} cached files have vectors ({} distinct files in store)",
                    files_cached, files_with_vectors
                ),
            )
        } else {
            (
                Status::Degraded,
                format!(
                    "Index incomplete: {} of {} files are cached as indexed but have NO vectors — \
                     the next run will skip them as unchanged. Fix: index_codebase with force_reindex: true",
                    stale_skips, files_cached
                ),
            )
        };

        CoverageHealth {
            status,
            message,
            files_tracked,
            files_cached: Some(files_cached),
            files_with_vectors: Some(files_with_vectors),
            stale_skips: Some(stale_skips),
            stale_skip_examples: examples,
        }
    }

    /// Calculate overall system status from component statuses
    fn calculate_overall_status(
        &self,
        bm25: &ComponentHealth,
        vector: &ComponentHealth,
        merkle: &ComponentHealth,
        coverage: &CoverageHealth,
    ) -> Status {
        // Critical: both search engines must work
        let search_unhealthy =
            bm25.status == Status::Unhealthy && vector.status == Status::Unhealthy;

        if search_unhealthy {
            return Status::Unhealthy;
        }

        // Degraded: one search engine down OR merkle issues OR an
        // index that is silently incomplete. The last one is the whole
        // point of the coverage check: without it a half-indexed store
        // answers "healthy" and quietly returns partial search results.
        let has_degraded = coverage.status != Status::Healthy
            || bm25.status == Status::Degraded
            || vector.status == Status::Degraded
            || merkle.status == Status::Degraded
            || bm25.status == Status::Unhealthy
            || vector.status == Status::Unhealthy;

        if has_degraded {
            return Status::Degraded;
        }

        // All healthy
        Status::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_component_health_constructors() {
        let healthy = ComponentHealth::healthy("All good", Some(50));
        assert_eq!(healthy.status, Status::Healthy);
        assert_eq!(healthy.latency_ms, Some(50));

        let degraded = ComponentHealth::degraded("Some issues");
        assert_eq!(degraded.status, Status::Degraded);
        assert_eq!(degraded.latency_ms, None);

        let unhealthy = ComponentHealth::unhealthy("Critical error");
        assert_eq!(unhealthy.status, Status::Unhealthy);
    }

    /// Coverage in the "checked and complete" state, for status-algebra tests.
    fn coverage_ok() -> CoverageHealth {
        CoverageHealth {
            status: Status::Healthy,
            message: "ok".to_string(),
            files_tracked: Some(3),
            files_cached: Some(3),
            files_with_vectors: Some(3),
            stale_skips: Some(0),
            stale_skip_examples: Vec::new(),
        }
    }

    #[test]
    fn test_overall_status_calculation() {
        let monitor = HealthMonitor {
            bm25: None,
            vector_store: None,
            merkle_path: PathBuf::from("/tmp/merkle.snapshot"),
            metadata_cache: None,
        };

        // All healthy
        let all_healthy = monitor.calculate_overall_status(
            &ComponentHealth::healthy("ok", None),
            &ComponentHealth::healthy("ok", None),
            &ComponentHealth::healthy("ok", None),
            &coverage_ok(),
        );
        assert_eq!(all_healthy, Status::Healthy);

        // One degraded
        let one_degraded = monitor.calculate_overall_status(
            &ComponentHealth::degraded("issues"),
            &ComponentHealth::healthy("ok", None),
            &ComponentHealth::healthy("ok", None),
            &coverage_ok(),
        );
        assert_eq!(one_degraded, Status::Degraded);

        // Both search engines down
        let both_down = monitor.calculate_overall_status(
            &ComponentHealth::unhealthy("down"),
            &ComponentHealth::unhealthy("down"),
            &ComponentHealth::healthy("ok", None),
            &coverage_ok(),
        );
        assert_eq!(both_down, Status::Unhealthy);

        // One search engine down (still degraded, not unhealthy)
        let one_down = monitor.calculate_overall_status(
            &ComponentHealth::unhealthy("down"),
            &ComponentHealth::healthy("ok", None),
            &ComponentHealth::healthy("ok", None),
            &coverage_ok(),
        );
        assert_eq!(one_down, Status::Degraded);
    }

    /// Build a store holding one vector for `file_with_vectors`, and a
    /// metadata cache claiming BOTH files are indexed. That is exactly
    /// the state a half-finished run leaves behind.
    async fn stale_skip_scene(temp: &TempDir, salt: &str) -> (Arc<VectorStore>, PathBuf) {
        use rmc_engine::chunker::{ChunkContext, ChunkId, CodeChunk};

        let store = VectorStore::new_embedded(temp.path().join("vectors"), 4, "test-embedder")
            .await
            .expect("open store");
        let chunk = CodeChunk {
            id: ChunkId::new(),
            content: "fn indexed() {}".to_string(),
            context: ChunkContext {
                file_path: PathBuf::from("/proj/indexed.rs"),
                module_path: vec!["crate".to_string()],
                symbol_name: "indexed".to_string(),
                symbol_kind: "function".to_string(),
                docstring: None,
                imports: vec![],
                outgoing_calls: vec![],
                parent_symbol_name: None,
                split_part: None,
                split_total: None,
                line_start: 1,
                line_end: 1,
            },
            overlap_prev: None,
            overlap_next: None,
        };
        store
            .upsert_chunks(vec![(chunk.id, vec![0.1, 0.2, 0.3, 0.4], chunk.clone())])
            .await
            .expect("upsert");

        let cache_path = temp.path().join("cache");
        let cache = MetadataCache::new(&cache_path).expect("cache");
        for path in ["/proj/indexed.rs", "/proj/skipped.rs"] {
            let key = if salt.is_empty() {
                path.to_string()
            } else {
                format!("{}::{}", salt, path)
            };
            cache
                .set(
                    &key,
                    &crate::metadata_cache::FileMetadata::from_content("x", 1, 1),
                )
                .expect("set metadata");
        }
        drop(cache);

        (Arc::new(store), cache_path)
    }

    #[tokio::test]
    async fn coverage_catches_file_cached_as_indexed_without_vectors() {
        let temp = TempDir::new().unwrap();
        let salt = "chunk-split:v1:target300:hard900";
        let (store, cache_path) = stale_skip_scene(&temp, salt).await;

        let monitor = HealthMonitor::new(None, Some(store), temp.path().join("missing.snapshot"))
            .with_metadata_cache(cache_path, salt.to_string());
        let health = monitor.check_health().await;

        assert_eq!(health.coverage.stale_skips, Some(1));
        assert_eq!(health.coverage.files_cached, Some(2));
        assert_eq!(health.coverage.files_with_vectors, Some(1));
        assert_eq!(
            health.coverage.stale_skip_examples,
            vec!["/proj/skipped.rs".to_string()]
        );
        assert_eq!(health.coverage.status, Status::Degraded);
        // The regression this whole component exists for: such an index
        // used to report `healthy`.
        assert_ne!(health.overall, Status::Healthy);
    }

    #[tokio::test]
    async fn coverage_is_healthy_when_every_cached_file_has_vectors() {
        let temp = TempDir::new().unwrap();
        let salt = "chunk-split:v1:target300:hard900";
        let (store, cache_path) = stale_skip_scene(&temp, salt).await;

        // Positive control: drop the bogus entry and the same scene must
        // flip to healthy — otherwise the assertion above would pass on a
        // component that is simply always degraded.
        let cache = MetadataCache::new(&cache_path).expect("cache");
        cache
            .remove(&format!("{}::{}", salt, "/proj/skipped.rs"))
            .expect("remove");
        drop(cache);

        let monitor = HealthMonitor::new(None, Some(store), temp.path().join("missing.snapshot"))
            .with_metadata_cache(cache_path, salt.to_string());
        let coverage = monitor.check_coverage().await;

        assert_eq!(coverage.stale_skips, Some(0));
        assert_eq!(coverage.status, Status::Healthy);
    }

    #[tokio::test]
    async fn coverage_with_wrong_salt_is_not_reported_as_verified() {
        let temp = TempDir::new().unwrap();
        let (store, cache_path) = stale_skip_scene(&temp, "salt-a").await;

        // Keys under another chunking identity must not be counted, and
        // an empty comparison must not read as "checked, all good".
        let monitor = HealthMonitor::new(None, Some(store), temp.path().join("missing.snapshot"))
            .with_metadata_cache(cache_path, "salt-b".to_string());
        let coverage = monitor.check_coverage().await;

        assert_eq!(coverage.files_cached, Some(0));
        // Vacuous "all 0 files are fine" must never read as healthy.
        assert_eq!(coverage.status, Status::Degraded);
        assert_eq!(coverage.stale_skips, None);
    }

    /// An empty cache next to a POPULATED store is lost bookkeeping, and
    /// the message must not push the reader towards `force_reindex` —
    /// that misread cost an hour of needless work on the live index
    /// (2026-08-19).
    #[tokio::test]
    async fn coverage_without_cache_but_with_vectors_does_not_advise_force_reindex() {
        let temp = TempDir::new().unwrap();
        let (store, cache_path) = stale_skip_scene(&temp, "salt-a").await;

        let monitor = HealthMonitor::new(None, Some(store), temp.path().join("missing.snapshot"))
            .with_metadata_cache(cache_path, "salt-b".to_string());
        let coverage = monitor.check_coverage().await;

        assert_eq!(coverage.files_cached, Some(0));
        assert_eq!(coverage.files_with_vectors, Some(1));
        assert!(
            coverage.message.contains("force_reindex is not indicated"),
            "message must say force_reindex is NOT the fix here: {}",
            coverage.message
        );
        assert!(
            !coverage.message.contains("force_reindex: true"),
            "message must not carry the force_reindex recipe: {}",
            coverage.message
        );
    }

    /// Positive control for the test above: the SAME empty cache over an
    /// EMPTY store must produce a different message. Without this, the
    /// assertion above would pass on a component that prints one text
    /// unconditionally — which is exactly the defect being fixed.
    #[tokio::test]
    async fn coverage_with_empty_cache_and_empty_store_says_nothing_is_indexed() {
        let temp = TempDir::new().unwrap();
        let store = VectorStore::new_embedded(temp.path().join("vectors"), 4, "test-embedder")
            .await
            .expect("open store");
        let cache_path = temp.path().join("cache");
        MetadataCache::new(&cache_path).expect("cache");

        let monitor = HealthMonitor::new(
            None,
            Some(Arc::new(store)),
            temp.path().join("missing.snapshot"),
        )
        .with_metadata_cache(cache_path, "salt-a".to_string());
        let coverage = monitor.check_coverage().await;

        assert_eq!(coverage.files_cached, Some(0));
        assert_eq!(coverage.files_with_vectors, Some(0));
        assert!(
            coverage
                .message
                .contains("nothing is indexed under this profile yet"),
            "empty store must read as never-indexed, not as lost bookkeeping: {}",
            coverage.message
        );
        assert_eq!(coverage.status, Status::Degraded);
    }

    #[test]
    fn coverage_without_metadata_cache_is_degraded_not_healthy() {
        let coverage = CoverageHealth::unknown("no cache");
        assert_eq!(coverage.status, Status::Degraded);
        assert_eq!(coverage.stale_skips, None);
    }

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus {
            overall: Status::Healthy,
            bm25: ComponentHealth::healthy("BM25 operational", Some(15)),
            vector: ComponentHealth::healthy("Vector operational", Some(42)),
            merkle: ComponentHealth::healthy("Merkle snapshot exists (2048 bytes)", None),
            coverage: coverage_ok(),
        };

        let json = serde_json::to_string_pretty(&status).unwrap();
        assert!(json.contains("\"overall\": \"healthy\""));
        assert!(json.contains("\"latency_ms\": 15"));
        assert!(json.contains("\"latency_ms\": 42"));
    }
}
