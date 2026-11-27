//! Oracles of plan item `no-brute-sweeps` (2026-11-27): a fuzzy name does not
//! buy a reference search per guess, and `rename_symbol` by name finds the
//! declarations `find_definition` finds — fields and `Type::method` paths
//! included — instead of a fuzzy index query cut at 50.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::SemanticService;
use super::position::Location;
use crate::deep_stack::test_holding_an_analysis as holding_an_analysis;

const AMPLE: Duration = Duration::from_secs(600);

fn write_file(path: &Path, contents: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

fn package(root: &Path, name: &str) {
    write_file(
        &root.join("Cargo.toml"),
        &format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"),
    );
}

/// 1-based line of the first line of `text` holding `needle`.
fn line_of(text: &str, needle: &str) -> u32 {
    text.lines()
        .position(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("fixture has no line with {needle:?}")) as u32
        + 1
}

fn reference_lines(locations: &[Location], file_suffix: &str) -> Vec<u32> {
    let mut lines: Vec<u32> = locations
        .iter()
        .filter(|l| l.name == "reference" && l.file_path.ends_with(file_suffix))
        .map(|l| l.line)
        .collect();
    lines.sort();
    lines
}

/// `target()` called once from `lib.rs`, and `decoys` functions the fuzzy
/// query for `target` also matches (`target_decoy0`, …), each called once
/// from `decoys.rs`.
fn fuzzy_decoys_workspace(root: &Path, decoys: usize) -> PathBuf {
    package(root, "fuzzy_probe");
    let mut lib = String::from("pub mod decoys;\n\npub fn target() {}\n\npub fn first() {\n    target();\n}\n");
    let mut calls = String::from("pub fn call_decoys() {\n");
    for n in 0..decoys {
        lib.push_str(&format!("\npub fn target_decoy{n}() {{}}\n"));
        calls.push_str(&format!("    crate::target_decoy{n}();\n"));
    }
    calls.push_str("}\n");
    write_file(&root.join("src/decoys.rs"), &calls);
    let lib_path = root.join("src/lib.rs");
    write_file(&lib_path, &lib);
    lib_path
}

/// A fuzzy `find_references` lists the fuzzy matches by their declarations and
/// searches the references of the full-name match alone: one `find_all_refs`
/// on 2 decoys and on 8, and no call site of a decoy in the answer.
///
/// Before, each of up to 50 fuzzy matches got its own `find_all_refs` — a
/// reference search per guess (on a large workspace a fuzzy `new` is fifty
/// searches of fifty `new`s). Mutant "search fuzzy matches too": 3 and 9
/// searches, and the decoys' call sites in `decoys.rs`.
#[test]
fn a_fuzzy_reference_query_searches_only_the_full_name_match() {
    let _analysis = holding_an_analysis();
    for decoys in [2, 8] {
        let workspace = tempfile::tempdir().unwrap();
        let root = workspace.path();
        let lib = fuzzy_decoys_workspace(root, decoys);
        let lib_text = std::fs::read_to_string(&lib).unwrap();
        let service = SemanticService::new();

        let fuzzy = service
            .find_references_within(root, "target", false, AMPLE)
            .expect("fuzzy references query");
        assert!(fuzzy.incomplete.is_none(), "{fuzzy:?}");
        assert_eq!(fuzzy.cost.declaration_searches, 1, "decoys={decoys}: {fuzzy:?}");
        assert_eq!(
            reference_lines(&fuzzy.locations, "src/lib.rs"),
            vec![line_of(&lib_text, "    target();")],
            "{fuzzy:?}"
        );
        assert_eq!(reference_lines(&fuzzy.locations, "src/decoys.rs"), Vec::<u32>::new(), "{fuzzy:?}");
        let listed = fuzzy
            .locations
            .iter()
            .filter(|l| l.name.starts_with("target_decoy") && !l.exact)
            .count();
        assert_eq!(listed, decoys, "every fuzzy match is listed by its declaration: {fuzzy:?}");

        // The full-name answer is the fuzzy one's exact part.
        let exact = service
            .find_references_within(root, "target", true, AMPLE)
            .expect("exact references query");
        let exact_part: Vec<_> = fuzzy.locations.iter().filter(|l| l.exact).cloned().collect();
        assert_eq!(
            exact.locations.iter().map(|l| (l.line, l.column)).collect::<Vec<_>>(),
            exact_part.iter().map(|l| (l.line, l.column)).collect::<Vec<_>>()
        );
    }
}

const RENAME_FIXTURE: &str = "\
pub struct Probe {
    pub radius_override: u32,
}

pub fn read(p: &Probe) -> u32 {
    p.radius_override
}

pub fn make() -> Probe {
    Probe { radius_override: 1 }
}

pub struct App;

impl App {
    pub fn new() -> Self {
        App
    }
}

pub struct Other;

impl Other {
    pub fn new() -> Self {
        Other
    }
}

pub fn build() {
    let _app = App::new();
    let _other = Other::new();
}

pub fn spawn() {}

pub fn run() {
    spawn();
}
";

/// `rename_symbol` by name, for what the fuzzy-query lookup could not name:
///
/// - a **field** (the index does not carry fields): was "No symbol found";
/// - a `Type::method` path, beside a same-named method of another type: was
///   "No exact match" (the path was compared with the leaf name);
/// - a name crowded out of a fuzzy query's first 50 by longer names holding
///   its letters (`a_s_p_a_w_n_0`, …, sorted ahead of `spawn`);
/// - and a bare `new`, which names two methods, is still refused as ambiguous.
#[test]
fn rename_by_name_finds_fields_paths_and_crowded_names() {
    let _analysis = holding_an_analysis();
    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path();
    package(root, "rename_probe");
    let mut text = String::from(RENAME_FIXTURE);
    for n in 0..60 {
        text.push_str(&format!("\npub fn a_s_p_a_w_n_{n}() {{}}\n"));
    }
    write_file(&root.join("src/lib.rs"), &text);
    let service = SemanticService::new();

    let lines_of = |preview: &super::rename::RenamePreview, new_text: &str| {
        let mut lines: Vec<u32> = preview
            .edits
            .iter()
            .inspect(|edit| assert_eq!(edit.new_text, new_text, "{preview:?}"))
            .map(|edit| edit.start_line)
            .collect();
        lines.sort();
        lines
    };

    let field = service
        .rename_by_name(root, "radius_override", "radius", AMPLE)
        .expect("rename a field by name");
    assert_eq!(
        lines_of(&field, "radius"),
        vec![
            line_of(&text, "pub radius_override"),
            line_of(&text, "p.radius_override"),
            line_of(&text, "Probe { radius_override"),
        ]
    );

    let path = service
        .rename_by_name(root, "App::new", "create", AMPLE)
        .expect("rename a Type::method path");
    assert_eq!(
        lines_of(&path, "create"),
        vec![line_of(&text, "pub fn new() -> Self {"), line_of(&text, "App::new()")],
        "only App's `new`, not Other's: {path:?}"
    );

    let crowded = service
        .rename_by_name(root, "spawn", "launch", AMPLE)
        .expect("rename a name fifty longer ones crowd");
    assert_eq!(
        lines_of(&crowded, "launch"),
        vec![line_of(&text, "pub fn spawn()"), line_of(&text, "    spawn();")]
    );

    let ambiguous = service
        .rename_by_name(root, "new", "create", AMPLE)
        .expect_err("two `new`s");
    assert!(ambiguous.to_string().contains("Ambiguous symbol 'new': 2 exact matches"), "{ambiguous}");
}
