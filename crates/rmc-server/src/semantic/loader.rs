//! Project loading with rust-analyzer

use std::path::Path;
use ra_ap_load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use ra_ap_project_model::{CargoConfig, CargoFeatures, RustLibSource};
use ra_ap_ide::AnalysisHost;
use ra_ap_vfs::Vfs;
use anyhow::{Result, Context};

/// Load a Cargo project for semantic analysis
///
/// Uses no_deps=true for fast loading (~120ms).
/// Only local project code is analyzed.
pub(crate) fn load_project(path: &Path) -> Result<(AnalysisHost, Vfs)> {
    load_project_with_config(path, fast_project_cargo_config())
}

/// Load a Cargo project with full workspace dependency edges for rename.
pub(super) fn load_project_full(path: &Path) -> Result<(AnalysisHost, Vfs)> {
    load_project_with_config(path, full_workspace_cargo_config())
}

fn fast_project_cargo_config() -> CargoConfig {
    CargoConfig {
        sysroot: None,
        no_deps: true,
        ..Default::default()
    }
}

fn full_workspace_cargo_config() -> CargoConfig {
    CargoConfig {
        sysroot: Some(RustLibSource::Discover),
        no_deps: false,
        features: CargoFeatures::All,
        all_targets: true,
        set_test: true,
        ..Default::default()
    }
}

/// Off-switch for running build scripts: `RMC_BUILD_SCRIPTS=0` (`off`/`false`/
/// `no`) loads the workspace without them, as it was before.
pub(crate) const BUILD_SCRIPTS_ENV: &str = "RMC_BUILD_SCRIPTS";

/// Whether to discover `OUT_DIR` by running the workspace's build scripts.
///
/// # What is invisible without it
///
/// Code pulled in with `include!(concat!(env!("OUT_DIR"), …))` — prost message
/// types, generated tables — has no `OUT_DIR` to expand until build scripts
/// have run, so the generated module is not in the tree at all and every
/// reference *into* it resolves to nothing. Measured on `rust_app`: seven reads
/// of a proto field in `roundtrip_invariants.rs` counted as zero. That is the
/// same silent zero the rest of this file's switches produce — indistinguishable
/// from "nobody reads this".
///
/// # What it costs
///
/// One cargo invocation per project load, sharing the workspace's own `target/`
/// (so it reuses what is already built, and can block on the directory lock
/// while another cargo holds it). With cargo ≥ 1.89 rust-analyzer passes
/// `--compile-time-deps`, which builds *only* build scripts and proc macros
/// rather than checking the workspace; older cargo falls back to a full
/// `cargo check`, which is where the cost would actually bite.
///
/// `wrap_rustc_in_build_scripts` stays off deliberately: it points
/// `RUSTC_WRAPPER` at `current_exe()`, which is rust-analyzer's own binary in
/// rust-analyzer and *ours* here — we do not implement the `RA_RUSTC_WRAPPER`
/// protocol, so turning it on would wreck the build rather than speed it up.
fn build_scripts_enabled() -> bool {
    parse_build_scripts(std::env::var(BUILD_SCRIPTS_ENV).ok().as_deref())
}

/// Default is on: the switch exists to be turned off, not to be turned on.
fn parse_build_scripts(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return true;
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "off" | "false" | "no"
    )
}

fn load_project_with_config(path: &Path, cargo_config: CargoConfig) -> Result<(AnalysisHost, Vfs)> {
    let load_config = LoadCargoConfig {
        load_out_dirs_from_check: build_scripts_enabled(),
        with_proc_macro_server: ProcMacroServerChoice::None,
        prefill_caches: true,
        num_worker_threads: num_cpus::get_physical(),
        proc_macro_processes: 1,
    };

    let (db, vfs, _) = load_workspace_at(path, &cargo_config, &load_config, &|_| {})
        .context("Failed to load workspace")?;

    let host = AnalysisHost::with_database(db);

    Ok((host, vfs))
}

#[cfg(test)]
mod tests {
    use super::parse_build_scripts;

    /// The direction matters more than the spellings: an unset variable has to
    /// mean *on*, or the fix ships switched off for everyone who never heard of
    /// it.
    #[test]
    fn build_scripts_are_on_unless_switched_off() {
        assert!(parse_build_scripts(None), "unset must mean on");
        for off in ["0", "off", "false", "no", " OFF ", "No"] {
            assert!(
                !parse_build_scripts(Some(off)),
                "{off:?} must switch build scripts off"
            );
        }
        for on in ["1", "true", "yes", "", "please"] {
            assert!(
                parse_build_scripts(Some(on)),
                "{on:?} is not an off-switch and must leave build scripts on"
            );
        }
    }
}
