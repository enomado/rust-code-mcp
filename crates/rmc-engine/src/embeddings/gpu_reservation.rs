//! A run that needs the whole GPU: searching goes on, indexing waits.
//!
//! # Why
//!
//! Training a network wants all of the card, so the daemon must hold none of
//! it, and only a process that dies gives its VRAM back (the HIP context
//! outlives every session). Stopping the daemon was the easy half; keeping it
//! stopped was done by holding its spawn lock for the whole run. That took
//! everything down with the embedder: 2026-11-06, every session opened during
//! a training run queued on the lock, hit the client's connect timeout and
//! started with no tools at all — although `find_definition`, the graph tools
//! and the BM25 half of `search` never touch the GPU, and the query half does
//! not need it either.
//!
//! # The contract
//!
//! The trainer takes an exclusive `flock` on [`gpu_reservation_path`] for its
//! whole run, writes its pid into the file, THEN stops the daemons. A daemon
//! started under the reservation never opens a GPU session:
//!
//! - a query that someone is waiting on runs the same model on the CPU, so the
//!   vectors stay comparable with the stored ones ([`with_query_session`]);
//! - everything else that embeds is refused with
//!   [`EmbeddingError::GpuReserved`]; the background sync does not even start
//!   ([`reservation_pausing`]), so its changes wait for the first tick after
//!   the run.
//!
//! The kernel drops the lock when the trainer dies, so a crashed run reserves
//! nothing, and nothing has to be told that the run ended: the next check
//! finds the card free.
//!
//! The other party is the trainer, outside this repository: this file name
//! and this order of steps are the whole of what the two sides share.

use super::backend::EmbeddingBackend;
use super::error::EmbeddingError;
use super::fastembed_onnx::FastembedOnnxEmbedder;
use super::gpu_broker;
use super::vram_ledger::{EmbeddingOwner, VRAM_LOG_TARGET};
use fs2::FileExt;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The reservation file, next to `gpu-embed.lock`: one per user, because every
/// process of this user shares the card.
pub fn gpu_reservation_path() -> PathBuf {
    super::gpu_lock::gpu_lock_path().with_file_name("gpu-reservation")
}

/// Somebody holds the GPU for a whole run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuReservation {
    /// `None` between the holder's lock and its write of the pid.
    holder: Option<u32>,
}

impl fmt::Display for GpuReservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.holder {
            Some(pid) => write!(f, "the GPU is reserved by pid {pid}"),
            None => write!(f, "the GPU is reserved"),
        }
    }
}

/// The reservation at `path`, if any.
///
/// Asks by trying a shared lock on a descriptor of its own, which is released
/// at once. A shared lock, so that daemons checking at the same moment do not
/// look like a reservation to each other; a trainer colliding with a check
/// retries. No file means nobody ever reserved; any other failure to open it
/// is treated the same, because a broken note must not stop a search (a real
/// holder has the file open, so opening it works).
fn reservation_at(path: &Path) -> Option<GpuReservation> {
    let mut file = File::open(path).ok()?;
    if FileExt::try_lock_shared(&file).is_ok() {
        let _ = FileExt::unlock(&file);
        return None;
    }
    let mut text = String::new();
    let holder = file
        .read_to_string(&mut text)
        .ok()
        .and(text.trim().parse().ok());
    Some(GpuReservation { holder })
}

/// The reservation in force now.
pub fn current_gpu_reservation() -> Option<GpuReservation> {
    reservation_at(&gpu_reservation_path())
}

/// The reservation `backend` must wait for: `None` for a backend that does
/// not embed on the GPU, which goes on as usual. For callers that decide
/// before any work whether indexing can start.
pub fn reservation_pausing(backend: &EmbeddingBackend) -> Option<GpuReservation> {
    backend
        .runs_on_gpu()
        .then(current_gpu_reservation)
        .flatten()
}

/// Where a run of `owner` goes.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Route {
    /// As usual: the GPU broker, or a backend that never needed the GPU.
    Usual,
    /// The same model on the CPU.
    QueryOnCpu,
}

/// The decision of the module, apart from the file it reads.
pub(super) fn route(
    backend: &EmbeddingBackend,
    owner: EmbeddingOwner,
    reservation: Option<GpuReservation>,
) -> Result<Route, EmbeddingError> {
    let Some(reservation) = reservation.filter(|_| backend.runs_on_gpu()) else {
        return Ok(Route::Usual);
    };
    // Only the ONNX runtimes can move: the same graph runs on either provider.
    if backend.is_fastembed_onnx_gpu() && gpu_broker::is_interactive(owner) {
        Ok(Route::QueryOnCpu)
    } else {
        Err(EmbeddingError::GpuReserved(reservation))
    }
}

/// CPU sessions of GPU backends, one per backend, created by the first query
/// under a reservation and dropped by [`release_query_sessions_once_free`].
/// A session costs RAM only while a run lasts, and the run is hours long, so
/// loading the model per query is not an option.
static QUERY_SESSIONS: Mutex<Vec<(EmbeddingBackend, Arc<FastembedOnnxEmbedder>)>> =
    Mutex::new(Vec::new());

/// Run `f` on the CPU session of `backend`. Blocking.
pub(super) fn with_query_session<T>(
    backend: &EmbeddingBackend,
    owner: EmbeddingOwner,
    f: impl FnOnce(&FastembedOnnxEmbedder) -> Result<T, EmbeddingError>,
) -> Result<T, EmbeddingError> {
    let session = {
        let mut sessions = QUERY_SESSIONS.lock().unwrap();
        match sessions.iter().find(|(key, _)| key == backend) {
            Some((_, session)) => session.clone(),
            None => {
                tracing::info!(
                    target: VRAM_LOG_TARGET,
                    profile = backend.profile.name(),
                    "the GPU is reserved: answering queries on a CPU session of the same model"
                );
                let session = Arc::new(FastembedOnnxEmbedder::new_query_on_cpu(backend, owner)?);
                sessions.push((backend.clone(), session.clone()));
                session
            }
        }
    };
    f(&session)
}

/// Drop the CPU sessions once the reservation is over: queries are back on the
/// GPU, and the model should not stay in RAM twice. A watchdog tick.
pub fn release_query_sessions_once_free() {
    if current_gpu_reservation().is_some() {
        return;
    }
    let released = std::mem::take(&mut *QUERY_SESSIONS.lock().unwrap());
    if !released.is_empty() {
        tracing::info!(
            target: VRAM_LOG_TARGET,
            sessions = released.len(),
            "the GPU reservation is over: dropping the CPU query sessions"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;

    fn gpu_backend() -> EmbeddingBackend {
        EmbeddingBackend::from_profile_name("local-gpu-bge").unwrap()
    }

    fn cpu_backend() -> EmbeddingBackend {
        EmbeddingBackend::from_profile_name("local-cpu-small").unwrap()
    }

    const RESERVED: Option<GpuReservation> = Some(GpuReservation { holder: Some(7) });

    /// The holder is seen with its pid while it holds, and only then. A second
    /// descriptor stands in for the trainer: `flock` belongs to the open file,
    /// so it conflicts the way another process does.
    ///
    /// Mutation that must fail it: `try_lock_shared` turned into
    /// `File::open(path).is_ok()` — "the file exists" is not "it is held", and
    /// every search after the first run would stay on the CPU forever.
    #[test]
    fn a_reservation_lasts_as_long_as_its_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu-reservation");
        assert_eq!(reservation_at(&path), None, "no file, no reservation");

        let mut trainer = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        assert_eq!(reservation_at(&path), None, "a file nobody locks is free");

        trainer.lock_exclusive().unwrap();
        assert_eq!(
            reservation_at(&path),
            Some(GpuReservation { holder: None }),
            "locked, the pid not written yet"
        );
        write!(trainer, "4242").unwrap();
        trainer.flush().unwrap();
        assert_eq!(
            reservation_at(&path),
            Some(GpuReservation { holder: Some(4242) })
        );
        // Asking must leave no lock behind: a trainer's retry would then spin.
        assert_eq!(
            reservation_at(&path),
            Some(GpuReservation { holder: Some(4242) })
        );

        drop(trainer);
        assert_eq!(reservation_at(&path), None, "a dead holder reserves nothing");
    }

    /// A check does not keep the next trainer out: nothing stays locked.
    #[test]
    fn checking_leaves_the_file_lockable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu-reservation");
        File::create(&path).unwrap();
        reservation_at(&path);
        let trainer = OpenOptions::new().write(true).open(&path).unwrap();
        assert!(trainer.try_lock_exclusive().is_ok());
    }

    /// The whole point: under a reservation whoever waits on an answer gets
    /// one, and nobody else reaches the embedder.
    ///
    /// Mutation that must fail it: dropping the `is_interactive` condition —
    /// the indexer would then embed the project on the CPU, 80× slower, under
    /// a training run (measured 2026-08-26, 3 vs 260 chunks/s).
    #[test]
    fn a_reservation_moves_queries_and_stops_the_rest() {
        let gpu = gpu_backend();
        for owner in [EmbeddingOwner::Search, EmbeddingOwner::CodemapRerank] {
            assert_eq!(route(&gpu, owner, RESERVED).unwrap(), Route::QueryOnCpu);
        }
        for owner in [
            EmbeddingOwner::Index,
            EmbeddingOwner::Tool,
            EmbeddingOwner::GraphCache,
            EmbeddingOwner::ProviderCensus,
        ] {
            assert!(
                matches!(
                    route(&gpu, owner, RESERVED),
                    Err(EmbeddingError::GpuReserved(_))
                ),
                "{owner:?}"
            );
        }
    }

    /// Without a reservation nothing changes, and a backend off the GPU has
    /// nothing to wait for.
    #[test]
    fn only_a_gpu_backend_under_a_reservation_is_rerouted() {
        let (gpu, cpu) = (gpu_backend(), cpu_backend());
        for owner in [EmbeddingOwner::Search, EmbeddingOwner::Index] {
            assert_eq!(route(&gpu, owner, None).unwrap(), Route::Usual);
            assert_eq!(route(&cpu, owner, RESERVED).unwrap(), Route::Usual);
        }
    }

    /// Real `QueryOnCpu`: the GPU profile's model answers with no GPU in the
    /// process, in the GPU profile's dimension, and with a finite unit vector.
    ///
    /// `cargo test -p rmc-engine --features embeddings query_session -- --ignored`
    #[test]
    #[ignore = "needs the model of local-gpu-bge in the cache or a download"]
    fn the_query_session_runs_the_gpu_profile_on_the_cpu() {
        let backend = gpu_backend();
        let vector = with_query_session(&backend, EmbeddingOwner::Search, |session| {
            session.embed_queries(&["where is the spawn lock taken"])
        })
        .unwrap()
        .remove(0);
        assert_eq!(vector.len(), backend.dim());
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
    }

    /// What makes the CPU session usable at all: a query vector from it must
    /// land where the GPU session's would, because the store was filled by the
    /// GPU. Texts of different lengths, since only the GPU pads to a fixed shape.
    ///
    /// `cargo test --release -p rmc-engine --features embeddings-migraphx
    /// the_cpu_session_agrees -- --ignored --nocapture`
    #[cfg(feature = "embeddings-migraphx")]
    #[test]
    #[ignore = "needs the card free and the model in the cache"]
    fn the_cpu_session_agrees_with_the_gpu_session() {
        let backend = gpu_backend();
        let owner = EmbeddingOwner::Search;
        for text in [
            "fn main",
            "where is the spawn lock taken",
            "reciprocal rank fusion of vector and keyword hits with a weight per list",
            "größte gemeinsame Teilfolge zweier Zeichenketten",
        ] {
            let cpu = with_query_session(&backend, owner, |s| s.embed_queries(&[text]))
                .unwrap()
                .remove(0);
            let gpu = gpu_broker::with_session(&backend, owner, |s| s.embed_queries(&[text]))
                .unwrap()
                .remove(0);
            let cosine = cpu.iter().zip(&gpu).map(|(a, b)| a * b).sum::<f32>();
            eprintln!("{cosine:.6}  {text}");
            assert!(cosine > 0.999, "{cosine} for {text:?}");
        }
    }
}
