//! Admission of a project load: a load whose measured peak does not fit in
//! the machine's free memory is refused with the numbers, before it starts.
//!
//! # Why (2026-11-27)
//!
//! The night of 2026-11-26/27 ended in two hardware-watchdog resets. Between
//! 01:52 and 04:18 earlyoom killed the daemon seven times in a row, and every
//! time the clients' relays started a fresh one whose first tool loaded the
//! whole workspace again at 3.5–8 GB available — a load that peaks around
//! 8–11 GB. The watchdog of the new daemon saw nothing wrong until the load
//! had already taken the memory, and earlyoom killed it again. At 10:24 a tool
//! asked about one member of an already loaded workspace, and the load of that
//! member — the whole workspace a second time, 2439 crates — added 3.7 GB.
//!
//! Every watchdog reacts to memory already taken; only a check before the load
//! can keep the load from taking it. The check needs the load's cost before
//! it runs, and that cost is what the previous load of the same project
//! measured — kept on disk the way the price of a GPU session is
//! (`rmc_engine::embeddings::vram_ledger`), so the first load of a new daemon
//! is checked too.
//!
//! # The rule
//!
//! `available ≥ expected peak + reserved by loads in flight + floor`, where
//! `floor` is the daemon's machine floor (`RMC_MIN_AVAILABLE_MB`): the load
//! must leave the floor standing at its peak, or the watchdog unloads it on
//! its next tick and the memory was spent for nothing. Loads of different
//! projects run side by side (one slot per project), so each admitted load
//! reserves its expected peak until it ends and the next check counts it.
//!
//! A project never measured has no expected peak; it is admitted on the floor
//! alone. A member of a workspace whose root was measured is priced as the
//! root: loading a member loads the whole workspace (`cargo metadata` resolves
//! from it), which is the 10:24 case.
//!
//! # Off
//!
//! Only the daemon installs the gate ([`install`]); a test binary or a bare
//! server neither checks nor writes prices. A floor of `0` — the one switch for
//! "do not react to the rest of the machine" — admits every load and still
//! measures.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// What the daemon decides the gate with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateConfig {
    /// The daemon's machine floor in MB; `0` admits every load.
    pub floor_mb: u64,
    /// Where measured peaks are kept: one file per project directory.
    pub prices_dir: PathBuf,
}

impl GateConfig {
    /// The daemon's gate: prices under its data directory, beside the copies
    /// of file texts — so a daemon test with its own `XDG_DATA_HOME` keeps its
    /// own prices.
    pub fn for_daemon(floor_mb: u64) -> Self {
        Self {
            floor_mb,
            prices_dir: crate::mcp::project_paths::data_dir().join("load-peaks"),
        }
    }
}

static INSTALLED: OnceLock<GateConfig> = OnceLock::new();

/// Turn the gate on for every [`super::SemanticService`] of this process.
/// The first call wins; the daemon makes it once, at startup.
pub fn install(config: GateConfig) {
    let _ = INSTALLED.set(config);
}

/// The decision on one load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// The gate is off, the floor is `0`, or `MemAvailable` cannot be read:
    /// an unknown reading must not act like an alarming one.
    Unchecked,
    /// The load fits; `reserve_mb` is what it holds against later checks
    /// until it ends (its expected peak, `0` when never measured).
    Admit { reserve_mb: u64 },
    /// It does not: the error names every term of the sum.
    Refuse(Shortfall),
}

/// The terms of a refused admission, all in MB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shortfall {
    pub available_mb: u64,
    /// `None`: never measured, refused on the floor and reservations alone.
    pub expected_mb: Option<u64>,
    pub reserved_mb: u64,
    pub floor_mb: u64,
}

impl std::fmt::Display for Shortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let expected = match self.expected_mb {
            Some(mb) => format!("its expected peak {mb} MB (measured at its last load)"),
            None => "no measured peak".to_string(),
        };
        write!(
            f,
            "{expected} + {} MB reserved by loads in flight + the machine floor {} MB \
             is more than the {} MB available",
            self.reserved_mb, self.floor_mb, self.available_mb
        )
    }
}

/// The pure decision. `available_mb` is `MemAvailable`; `reserved_mb` the sum
/// of the expected peaks of the loads in flight.
pub fn admit(
    available_mb: Option<u64>,
    floor_mb: u64,
    expected_mb: Option<u64>,
    reserved_mb: u64,
) -> Admission {
    let Some(available_mb) = available_mb else {
        return Admission::Unchecked;
    };
    if floor_mb == 0 {
        return Admission::Unchecked;
    }
    let reserve_mb = expected_mb.unwrap_or(0);
    let needed = reserve_mb
        .saturating_add(reserved_mb)
        .saturating_add(floor_mb);
    if available_mb >= needed {
        Admission::Admit { reserve_mb }
    } else {
        Admission::Refuse(Shortfall {
            available_mb,
            expected_mb,
            reserved_mb,
            floor_mb,
        })
    }
}

/// The price of loading `canonical`: its own measured peak, or else that of
/// the nearest enclosing directory measured — the workspace a member belongs
/// to. `lookup` reads one directory's price.
pub fn price_of(canonical: &Path, lookup: impl Fn(&Path) -> Option<u64>) -> Option<u64> {
    canonical.ancestors().find_map(lookup)
}

/// How much a load raised the process's footprint at its peak, in KiB.
///
/// `start_kib` is the footprint (RSS + swap) when the load began, after the
/// context it replaces was dropped and trimmed; `sampled_peak_kib` the highest
/// footprint the sampler saw. The kernel's `VmHWM` catches a spike between two
/// samples, but only when it rose during the load — otherwise it is an older
/// peak of the process and says nothing about this load. Overlapping loads of
/// other projects make the rise an overestimate; that errs on the side of
/// refusing.
pub fn measured_rise_kib(
    start_kib: u64,
    sampled_peak_kib: u64,
    hwm_before_kib: Option<u64>,
    hwm_after_kib: Option<u64>,
) -> u64 {
    let hwm = match (hwm_before_kib, hwm_after_kib) {
        (Some(before), Some(after)) if after > before => after,
        _ => 0,
    };
    sampled_peak_kib.max(hwm).saturating_sub(start_kib)
}

/// Loads in flight and what each reserved. Owned by
/// [`super::SemanticService`]; a leaf lock, held for the check and the insert
/// or the removal only, never while taking another lock.
pub(crate) struct LoadGate {
    /// Set by a test; otherwise the process's [`install`]ed config is read.
    fixed: Option<GateConfig>,
    reserved: Mutex<Vec<(PathBuf, u64)>>,
}

/// A load's admission: releases its reservation when the load ends, however.
pub(crate) struct Admitted<'g> {
    gate: &'g LoadGate,
    project: PathBuf,
    config: Option<GateConfig>,
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        let mut reserved = self.gate.reserved();
        if let Some(at) = reserved.iter().position(|(path, _)| *path == self.project) {
            reserved.swap_remove(at);
        }
    }
}

impl LoadGate {
    pub(crate) fn from_install() -> Self {
        Self {
            fixed: None,
            reserved: Mutex::new(Vec::new()),
        }
    }

    #[cfg(test)]
    pub(crate) fn fixed(config: GateConfig) -> Self {
        Self {
            fixed: Some(config),
            reserved: Mutex::new(Vec::new()),
        }
    }

    fn config(&self) -> Option<&GateConfig> {
        self.fixed.as_ref().or_else(|| INSTALLED.get())
    }

    fn reserved(&self) -> std::sync::MutexGuard<'_, Vec<(PathBuf, u64)>> {
        self.reserved.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Check a load of `canonical` against the machine now and reserve its
    /// expected peak, or refuse it. The caller holds the project's slot, so
    /// one project has at most one load here at a time.
    pub(crate) fn admit(&self, canonical: &Path) -> anyhow::Result<Admitted<'_>> {
        self.admit_with(canonical, || {
            crate::mcp::memory::machine_memory().map(|machine| machine.available_mb)
        })
    }

    fn admit_with(
        &self,
        canonical: &Path,
        available_mb: impl FnOnce() -> Option<u64>,
    ) -> anyhow::Result<Admitted<'_>> {
        let config = self.config().cloned();
        let Some(config) = config else {
            return Ok(Admitted {
                gate: self,
                project: canonical.to_path_buf(),
                config: None,
            });
        };
        let expected_mb = price_of(canonical, |dir| read_price(&config.prices_dir, dir));
        // Check and reserve under one lock: two loads checked side by side
        // would each see the other's memory as still free.
        let mut reserved = self.reserved();
        let reserved_mb = reserved.iter().map(|(_, mb)| mb).sum();
        match admit(available_mb(), config.floor_mb, expected_mb, reserved_mb) {
            Admission::Refuse(shortfall) => {
                tracing::warn!(
                    "load of {} refused: {shortfall}",
                    canonical.display()
                );
                anyhow::bail!(
                    "not loading {} now, the machine has no room for it: {shortfall}. \
                     Retry when memory frees (a build or another analysis ends).",
                    canonical.display()
                )
            }
            Admission::Admit { reserve_mb } => {
                match expected_mb {
                    Some(mb) => tracing::info!(
                        "load of {} admitted: expected peak {mb} MB, {reserved_mb} MB reserved \
                         by loads in flight, floor {} MB",
                        canonical.display(),
                        config.floor_mb
                    ),
                    None => tracing::info!(
                        "load of {} admitted on the floor alone ({} MB): no measured peak yet",
                        canonical.display(),
                        config.floor_mb
                    ),
                }
                reserved.push((canonical.to_path_buf(), reserve_mb));
            }
            Admission::Unchecked => {}
        }
        drop(reserved);
        Ok(Admitted {
            gate: self,
            project: canonical.to_path_buf(),
            config: Some(config),
        })
    }
}

impl Admitted<'_> {
    /// Run the load under a footprint sampler and, when it succeeds, keep its
    /// measured peak as the project's price for the next check.
    pub(crate) fn measure<T>(self, load: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
        let Some(config) = self.config.clone() else {
            return load();
        };
        let (outcome, rise_kib) = sample_peak(load);
        if outcome.is_ok()
            && let Some(rise_kib) = rise_kib
        {
            let mb = rise_kib / 1024;
            tracing::info!(
                "load peak of {}: +{mb} MB over the footprint at its start; kept as its price",
                self.project.display()
            );
            write_price(&config.prices_dir, &self.project, mb);
        }
        outcome
    }
}

/// How often the sampler reads `/proc/self/status` during a load. A load
/// runs for tens of seconds and its footprint climbs over seconds; `VmHWM`
/// covers what falls between two samples when it sets a new process peak.
const SAMPLE_EVERY: Duration = Duration::from_millis(250);

/// Run `load` beside a thread that samples the process footprint; the rise
/// is `None` where `/proc/self/status` cannot be read.
fn sample_peak<T>(load: impl FnOnce() -> T) -> (T, Option<u64>) {
    let Some(start) = Footprint::now() else {
        return (load(), None);
    };
    let done = AtomicBool::new(false);
    let (outcome, sampled_peak_kib) = std::thread::scope(|scope| {
        let sampler = scope.spawn(|| {
            let mut peak = start.total_kib;
            while !done.load(Ordering::Relaxed) {
                std::thread::park_timeout(SAMPLE_EVERY);
                if let Some(now) = Footprint::now() {
                    peak = peak.max(now.total_kib);
                }
            }
            peak
        });
        let outcome = load();
        done.store(true, Ordering::Relaxed);
        sampler.thread().unpark();
        (outcome, sampler.join().unwrap())
    });
    let end_hwm = Footprint::now().and_then(|end| end.hwm_kib);
    (
        outcome,
        Some(measured_rise_kib(
            start.total_kib,
            sampled_peak_kib,
            start.hwm_kib,
            end_hwm,
        )),
    )
}

/// RSS + swap and the kernel's peak RSS of this process, in KiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Footprint {
    /// `VmRSS + VmSwap`: under swap pressure the load's pages are partly out,
    /// and RSS alone would price it low exactly when the price matters.
    total_kib: u64,
    hwm_kib: Option<u64>,
}

impl Footprint {
    fn now() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::parse(&std::fs::read_to_string("/proc/self/status").ok()?)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    fn parse(status: &str) -> Option<Self> {
        let field = |name: &str| {
            status.lines().find_map(|line| {
                line.strip_prefix(name)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
        };
        // No `VmSwap` line: a kernel without swap, so nothing is out.
        Some(Self {
            total_kib: field("VmRSS:")? + field("VmSwap:").unwrap_or(0),
            hwm_kib: field("VmHWM:"),
        })
    }
}

fn price_path(prices_dir: &Path, project: &Path) -> PathBuf {
    prices_dir.join(format!(
        "{}.peak_mb",
        crate::mcp::project_paths::dir_hash(project)
    ))
}

/// The file holds the MB on its first line and the directory on the second,
/// so a person reading the directory can tell which file is which; a file
/// whose second line names another directory (a hash collision) is not this
/// one's price.
pub(super) fn read_price(prices_dir: &Path, project: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(price_path(prices_dir, project)).ok()?;
    let mut lines = text.lines();
    let mb = lines.next()?.trim().parse().ok()?;
    (lines.next()? == project.to_string_lossy()).then_some(mb)
}

/// Keep the latest measurement, not the largest: a workspace shrinks too, and
/// a one-off overestimate (a load beside another project's) should not refuse
/// that project for ever. Zero is not written — it is what an unmeasurable
/// load reads, not a price. Best effort: an unwritable data directory costs
/// only the next check.
pub(super) fn write_price(prices_dir: &Path, project: &Path, mb: u64) {
    if mb == 0 {
        return;
    }
    let path = price_path(prices_dir, project);
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let written = std::fs::create_dir_all(prices_dir)
        .and_then(|()| std::fs::write(&tmp, format!("{mb}\n{}\n", project.to_string_lossy())))
        .and_then(|()| std::fs::rename(&tmp, &path));
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp);
        tracing::warn!(
            "could not keep the load peak of {} in {}: {error}",
            project.display(),
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule as a table, with the numbers of the night of 2026-11-26/27 on
    /// a 61 GB machine whose floor is 6343 MB: the workspace peaks at about
    /// 9000 MB over the daemon's footprint.
    #[test]
    fn admission_table() {
        const FLOOR: u64 = 6343;
        const BUR: u64 = 9000;
        let rows: &[(&str, Option<u64>, u64, Option<u64>, u64, Admission)] = &[
            (
                "a fresh daemon at 3.5 GB available after earlyoom (01:52-04:18 storm)",
                Some(3500),
                FLOOR,
                Some(BUR),
                0,
                Admission::Refuse(Shortfall {
                    available_mb: 3500,
                    expected_mb: Some(BUR),
                    reserved_mb: 0,
                    floor_mb: FLOOR,
                }),
            ),
            (
                "8 GB available: above the floor, not above floor + peak (10:55)",
                Some(8000),
                FLOOR,
                Some(BUR),
                0,
                Admission::Refuse(Shortfall {
                    available_mb: 8000,
                    expected_mb: Some(BUR),
                    reserved_mb: 0,
                    floor_mb: FLOOR,
                }),
            ),
            (
                "a working day, 30.9 GB available (10:24)",
                Some(30900),
                FLOOR,
                Some(BUR),
                0,
                Admission::Admit { reserve_mb: BUR },
            ),
            (
                "exactly peak + floor fits",
                Some(BUR + FLOOR),
                FLOOR,
                Some(BUR),
                0,
                Admission::Admit { reserve_mb: BUR },
            ),
            (
                "one MB short does not",
                Some(BUR + FLOOR - 1),
                FLOOR,
                Some(BUR),
                0,
                Admission::Refuse(Shortfall {
                    available_mb: BUR + FLOOR - 1,
                    expected_mb: Some(BUR),
                    reserved_mb: 0,
                    floor_mb: FLOOR,
                }),
            ),
            (
                "a second load beside one in flight: the reservations add up",
                Some(20000),
                FLOOR,
                Some(BUR),
                BUR,
                Admission::Refuse(Shortfall {
                    available_mb: 20000,
                    expected_mb: Some(BUR),
                    reserved_mb: BUR,
                    floor_mb: FLOOR,
                }),
            ),
            (
                "never measured, above the floor: admitted, reserves nothing",
                Some(8000),
                FLOOR,
                None,
                0,
                Admission::Admit { reserve_mb: 0 },
            ),
            (
                "never measured, under the floor: refused",
                Some(5000),
                FLOOR,
                None,
                0,
                Admission::Refuse(Shortfall {
                    available_mb: 5000,
                    expected_mb: None,
                    reserved_mb: 0,
                    floor_mb: FLOOR,
                }),
            ),
            (
                "floor 0 switches the machine watch off",
                Some(100),
                0,
                Some(BUR),
                0,
                Admission::Unchecked,
            ),
            (
                "MemAvailable unreadable",
                None,
                FLOOR,
                Some(BUR),
                0,
                Admission::Unchecked,
            ),
        ];
        for (case, available, floor, expected, reserved, want) in rows {
            assert_eq!(
                admit(*available, *floor, *expected, *reserved),
                *want,
                "{case}"
            );
        }
    }

    #[test]
    fn a_member_is_priced_as_its_workspace() {
        let prices = [
            (PathBuf::from("/w"), 9000),
            (PathBuf::from("/w/apps/b"), 400),
        ];
        let lookup = |dir: &Path| {
            prices
                .iter()
                .find(|(path, _)| path == dir)
                .map(|(_, mb)| *mb)
        };
        assert_eq!(price_of(Path::new("/w/apps/a/core"), lookup), Some(9000));
        assert_eq!(price_of(Path::new("/w/apps/b"), lookup), Some(400));
        assert_eq!(price_of(Path::new("/w"), lookup), Some(9000));
        assert_eq!(price_of(Path::new("/other"), lookup), None);
    }

    #[test]
    fn the_rise_counts_hwm_only_when_the_load_set_it() {
        // Sampler saw the peak.
        assert_eq!(measured_rise_kib(2_000, 11_000, Some(20_000), Some(20_000)), 9_000);
        // A spike between samples set a new process peak.
        assert_eq!(measured_rise_kib(2_000, 9_000, Some(5_000), Some(12_000)), 10_000);
        // An older, higher peak of the process is not this load's.
        assert_eq!(measured_rise_kib(2_000, 9_000, Some(25_000), Some(25_000)), 7_000);
        // A load that ended smaller than it began (a trim meanwhile).
        assert_eq!(measured_rise_kib(5_000, 5_000, None, None), 0);
    }

    #[test]
    fn footprint_adds_swap_to_rss() {
        let status = "VmHWM:\t  900 kB\nVmRSS:\t  700 kB\nVmSwap:\t  300 kB\n";
        assert_eq!(
            Footprint::parse(status),
            Some(Footprint {
                total_kib: 1000,
                hwm_kib: Some(900)
            })
        );
        assert_eq!(
            Footprint::parse("VmRSS:\t 700 kB\n").map(|f| f.total_kib),
            Some(700)
        );
    }

    /// The gate end to end without a workspace: a load's measured peak becomes
    /// the price that refuses the next one, a member reads its workspace's
    /// price, and an admitted load holds its reservation until it ends.
    #[test]
    fn a_measured_load_prices_the_next_admission() {
        let dir = tempfile::tempdir().unwrap();
        let config = GateConfig {
            floor_mb: 1000,
            prices_dir: dir.path().join("load-peaks"),
        };
        let gate = LoadGate::fixed(config.clone());
        let workspace = dir.path().join("w");
        write_price(&config.prices_dir, &workspace, 9000);
        assert_eq!(read_price(&config.prices_dir, &workspace), Some(9000));

        let refused = gate
            .admit_with(&workspace.join("member"), || Some(9999))
            .err()
            .expect("a member of a 9000 MB workspace does not fit in 9999 - 1000");
        assert!(
            refused.to_string().contains("9000 MB"),
            "the refusal names the price: {refused}"
        );

        let first = gate.admit_with(&workspace, || Some(20000)).unwrap();
        // The first load's 9000 MB is reserved: a second project does not fit.
        let other = dir.path().join("other");
        write_price(&config.prices_dir, &other, 5000);
        assert!(gate.admit_with(&other, || Some(14000)).is_err());
        drop(first);
        assert!(gate.admit_with(&other, || Some(14000)).is_ok());
    }

    #[test]
    fn a_price_file_of_another_directory_is_not_this_ones() {
        let dir = tempfile::tempdir().unwrap();
        let project = Path::new("/w");
        std::fs::write(price_path(dir.path(), project), "9000\n/elsewhere\n").unwrap();
        assert_eq!(read_price(dir.path(), project), None);
    }

    #[test]
    fn sampling_sees_a_peak_freed_before_the_end() {
        let (_, rise) = sample_peak(|| {
            let held = vec![1u8; 256 << 20];
            std::hint::black_box(&held);
            std::thread::sleep(SAMPLE_EVERY * 3);
            drop(held);
        });
        let Some(rise) = rise else { return };
        assert!(
            rise >= 200 << 10,
            "a 256 MiB allocation held for three samples must show: rose {rise} KiB"
        );
    }
}
