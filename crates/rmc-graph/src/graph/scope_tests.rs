//! A graph scoped to a group of workspace members: a directory without a
//! `Cargo.toml` of its own, optionally narrowed to named packages
//! (`BuildOptions::crates`), and the inputs above the requested directory that
//! decide its fingerprint.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use super::snapshot::{BuildOptions, BuildResult, build_and_persist, open_current};
use super::storage::{GraphEnvOptions, GraphPaths};

/// `group/alpha` (lib + bin + integration test + example + build script),
/// `group/beta` (lib calling alpha), and `other/gamma` (lib calling beta) —
/// a sibling outside `group/`.
fn write_grouped_fixture(root: &Path) {
    const FILES: &[(&str, &str)] = &[
        (
            "Cargo.toml",
            "[workspace]\nmembers = [\"group/alpha\", \"group/beta\", \"other/gamma\"]\nresolver = \"2\"\n",
        ),
        (
            "group/alpha/Cargo.toml",
            "[package]\nname = \"alpha\"\nversion = \"0.1.0\"\nedition = \"2021\"\nbuild = \"build.rs\"\n",
        ),
        ("group/alpha/build.rs", "fn main() {}\n"),
        ("group/alpha/src/lib.rs", "pub fn double(x: i32) -> i32 {\n    x * 2\n}\n"),
        ("group/alpha/src/bin/tool.rs", "fn main() {\n    println!(\"{}\", alpha::double(2));\n}\n"),
        ("group/alpha/tests/it.rs", "#[test]\nfn doubles() {\n    assert_eq!(alpha::double(2), 4);\n}\n"),
        ("group/alpha/examples/demo.rs", "fn main() {\n    alpha::double(3);\n}\n"),
        (
            "group/beta/Cargo.toml",
            "[package]\nname = \"beta\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nalpha = { path = \"../alpha\" }\n",
        ),
        ("group/beta/src/lib.rs", "pub fn quad(x: i32) -> i32 {\n    alpha::double(alpha::double(x))\n}\n"),
        (
            "other/gamma/Cargo.toml",
            "[package]\nname = \"gamma\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nbeta = { path = \"../../group/beta\" }\n",
        ),
        ("other/gamma/src/lib.rs", "pub fn g(x: i32) -> i32 {\n    beta::quad(x)\n}\n"),
    ];
    for (rel, text) in FILES {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
}

fn build(directory: &Path, data: &Path, crates: Option<&[&str]>) -> anyhow::Result<BuildResult> {
    build_and_persist(directory, BuildOptions {
        data_dir_override: Some(data.to_path_buf()),
        crates: crates.map(|c| c.iter().map(|s| s.to_string()).collect()),
        ..Default::default()
    })
}

fn names(result: &BuildResult) -> BTreeSet<&str> {
    result.crates.iter().map(String::as_str).collect()
}

/// Callers of `target` in the snapshot of `result`.
fn callers(data: &Path, result: &BuildResult, target: &str) -> BTreeSet<String> {
    let snap = open_current(&GraphPaths::for_workspace_in(data, &result.workspace_root), GraphEnvOptions::default())
        .unwrap()
        .unwrap();
    let (id, _) = snap.lookup_by_qualified_name(target).unwrap().unwrap_or_else(|| panic!("no node {target}"));
    snap.who_calls(id).unwrap().into_iter().filter_map(|c| c.caller_qualified_name).collect()
}

#[test]
fn a_directory_of_members_builds_their_lib_and_bin_targets_with_cross_crate_calls() {
    let ws = tempfile::tempdir().unwrap();
    write_grouped_fixture(ws.path());
    let data = tempfile::tempdir().unwrap();
    let result = build(&ws.path().join("group"), data.path(), None).unwrap();

    // Before the governing manifest was looked up, `cargo metadata` ran on
    // `group/Cargo.toml` (absent), failed, and the fallback let in every
    // target under `group/`: the integration test, the example, the build
    // script.
    assert_eq!(
        names(&result),
        BTreeSet::from(["tool[bin]@alpha/src/bin/tool.rs", "alpha[lib]@alpha/src/lib.rs", "beta[lib]@beta/src/lib.rs"]),
    );
    // The point of one graph over the group: calls across its crates are edges.
    assert_eq!(
        callers(data.path(), &result, "alpha::double"),
        BTreeSet::from(["beta::quad".to_string(), "tool::main".to_string()]),
    );
}

#[test]
fn a_crate_list_narrows_the_scope_and_names_the_snapshot() {
    let ws = tempfile::tempdir().unwrap();
    write_grouped_fixture(ws.path());
    let group = ws.path().join("group");
    let data = tempfile::tempdir().unwrap();

    let whole = build(&group, data.path(), None).unwrap();
    let alpha = build(&group, data.path(), Some(&["alpha"])).unwrap();
    // A package brings its bins.
    assert_eq!(names(&alpha), BTreeSet::from(["tool[bin]@alpha/src/bin/tool.rs", "alpha[lib]@alpha/src/lib.rs"]));
    assert!(!alpha.reused);
    assert_ne!(alpha.fingerprint, whole.fingerprint, "another crate set is another snapshot");

    let beta = build(&group, data.path(), Some(&["beta"])).unwrap();
    assert_eq!(names(&beta), BTreeSet::from(["beta[lib]@beta/src/lib.rs"]));
    let again = build(&group, data.path(), Some(&["beta"])).unwrap();
    assert!(again.reused, "same set, same tree: reused");

    // Back to the unscoped request: not mistaken for the narrowed snapshot.
    let whole_again = build(&group, data.path(), None).unwrap();
    assert_eq!(whole_again.fingerprint, whole.fingerprint);
    assert_eq!(names(&whole_again), names(&whole));
}

#[test]
fn a_crate_list_naming_nothing_buildable_is_refused() {
    let ws = tempfile::tempdir().unwrap();
    write_grouped_fixture(ws.path());
    let group = ws.path().join("group");
    let data = tempfile::tempdir().unwrap();

    // `gamma` is a member, but not under `group/`.
    let error = format!("{:#}", build(&group, data.path(), Some(&["alpha", "gamma"])).unwrap_err());
    assert!(error.contains("\"gamma\"") && error.contains("available there: {\"alpha\", \"beta\"}"), "{error}");
    let error = format!("{:#}", build(&group, data.path(), Some(&[])).unwrap_err());
    assert!(error.contains("empty crate list"), "{error}");
}

#[test]
fn inputs_above_the_requested_directory_are_part_of_its_fingerprint() {
    let ws = tempfile::tempdir().unwrap();
    write_grouped_fixture(ws.path());
    let beta = ws.path().join("group/beta");
    let data = tempfile::tempdir().unwrap();
    let first = build(&beta, data.path(), None).unwrap();
    assert!(build(&beta, data.path(), None).unwrap().reused);

    // The workspace manifest decides dependency versions and features of the
    // member; before, a graph of `group/beta` never noticed it change.
    let manifest = ws.path().join("Cargo.toml");
    let text = fs::read_to_string(&manifest).unwrap();
    fs::write(&manifest, format!("{text}\n[workspace.metadata.probe]\nedited = true\n")).unwrap();
    let after_manifest = build(&beta, data.path(), None).unwrap();
    assert!(!after_manifest.reused, "workspace Cargo.toml changed above the member");
    assert_ne!(after_manifest.fingerprint, first.fingerprint);

    // Cargo merges every ancestor's config.
    fs::create_dir_all(ws.path().join(".cargo")).unwrap();
    fs::write(ws.path().join(".cargo/config.toml"), "[build]\n").unwrap();
    let after_config = build(&beta, data.path(), None).unwrap();
    assert!(!after_config.reused, ".cargo/config.toml appeared above the member");
    assert_ne!(after_config.fingerprint, after_manifest.fingerprint);
}
