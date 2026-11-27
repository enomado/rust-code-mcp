//! Probe, not a test: what one pass classifying every identifier of a project
//! costs on the database the daemon serves — the work a stored table of uses
//! (`library-refs-scope` of the plan) would do once per load instead of once
//! per query.
//!
//! Loads the project the way the daemon does (`load_project_full`: sysroot,
//! all targets, `cfg(test)`, primed), takes snapshots of the database the way
//! a background builder would (`raw_database().clone()`), and walks every
//! `IDENT` of every project `.rs` file through `Semantics` — descending into
//! macro calls like the graph's usages pass. Prints what the identifiers
//! resolve to (a library item, a local item, a local binding, nothing), the
//! number of distinct library items, the most-used ones, time and RSS: of a
//! cold pass, after the daemon's garbage collection and trim, of a second pass
//! after it, and of a pass split over `RMC_USE_CENSUS_THREADS` threads.
//!
//! `RMC_USE_CENSUS_PROJECT=<dir> cargo test -p rmc-server --lib -- --ignored --nocapture library_use_census`

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use ra_ap_hir::{Semantics, attach_db};
use ra_ap_ide_db::RootDatabase;
use ra_ap_ide_db::defs::{Definition, NameClass, NameRefClass};
use ra_ap_syntax::{AstNode, SyntaxKind, ast};
use ra_ap_vfs::FileId;

/// Resident set of this process, MiB, from `/proc/self/status`.
fn rss_mib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmRSS:")).unwrap();
    line.split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        / 1024
}

#[derive(Default, Debug)]
struct Counts {
    idents: usize,
    /// Identifiers inside a macro call: classified through their expansion.
    in_macro_calls: usize,
    /// Classified occurrences: one identifier may stand for several after
    /// descending into a macro.
    library_refs: usize,
    local_item_refs: usize,
    /// Locals, labels, generic parameters, `self` and the like.
    binding_refs: usize,
    /// Builtin types, tool modules, derive helpers.
    other_refs: usize,
    unresolved_name_refs: usize,
    declaring_names: usize,
}

struct Report {
    counts: Counts,
    distinct_library_items: usize,
    top: Vec<(usize, String)>,
    slowest: Vec<(Duration, FileId)>,
}

/// Classify every identifier of `files` on `db`.
fn pass(db: &RootDatabase, files: &[FileId]) -> Report {
    let mut counts = Counts::default();
    let mut slowest = Vec::new();
    attach_db(db, || {
        let sema = Semantics::new(db);
        let mut library: HashMap<Definition<'_>, usize> = HashMap::new();
        for &file in files {
            let file_at = Instant::now();
            let tree = sema.parse_guess_edition(file);
            let tokens: Vec<_> = tree
                .syntax()
                .descendants_with_tokens()
                .filter_map(|element| element.into_token())
                .filter(|token| token.kind() == SyntaxKind::IDENT)
                .collect();
            for token in tokens {
                counts.idents += 1;
                let Some(parent) = token.parent() else {
                    continue;
                };
                let in_macro = sema.is_inside_macro_call(ra_ap_hir::InFile::new(
                    sema.hir_file_for(&parent),
                    &token,
                ));
                let descended = if in_macro {
                    counts.in_macro_calls += 1;
                    sema.descend_into_macros_exact(token.clone())
                } else {
                    std::iter::once(token.clone()).collect()
                };
                for token in descended {
                    let Some(name_like) = token.parent().and_then(ast::NameLike::cast) else {
                        continue;
                    };
                    let def = match name_like {
                        ast::NameLike::NameRef(name_ref) => {
                            match NameRefClass::classify(&sema, &name_ref) {
                                Some(NameRefClass::Definition(def, _)) => def,
                                Some(NameRefClass::FieldShorthand { field_ref, .. }) => {
                                    Definition::Field(field_ref)
                                }
                                Some(NameRefClass::ExternCrateShorthand { .. }) => {
                                    counts.other_refs += 1;
                                    continue;
                                }
                                None => {
                                    counts.unresolved_name_refs += 1;
                                    continue;
                                }
                            }
                        }
                        ast::NameLike::Name(name) => match NameClass::classify(&sema, &name) {
                            Some(NameClass::ConstReference(def)) => def,
                            _ => {
                                counts.declaring_names += 1;
                                continue;
                            }
                        },
                        ast::NameLike::Lifetime(_) => continue,
                    };
                    match def {
                        Definition::Local(_)
                        | Definition::Label(_)
                        | Definition::GenericParam(_)
                        | Definition::SelfType(_)
                        | Definition::InlineAsmOperand(_)
                        | Definition::InlineAsmRegOrRegClass(_) => counts.binding_refs += 1,
                        _ => match def.krate(db) {
                            Some(krate) if !krate.origin(db).is_local() => {
                                counts.library_refs += 1;
                                *library.entry(def).or_default() += 1;
                            }
                            Some(_) => counts.local_item_refs += 1,
                            None => counts.other_refs += 1,
                        },
                    }
                }
            }
            slowest.push((file_at.elapsed(), file));
        }
        let mut top: Vec<_> = library.iter().map(|(def, n)| (*n, *def)).collect();
        top.sort_by(|a, b| b.0.cmp(&a.0));
        let top = top
            .iter()
            .take(15)
            .map(|(n, def)| {
                (
                    *n,
                    def.name(db)
                        .map(|name| name.as_str().to_owned())
                        .unwrap_or_default(),
                )
            })
            .collect();
        slowest.sort_by(|a, b| b.0.cmp(&a.0));
        slowest.truncate(5);
        Report {
            counts,
            distinct_library_items: library.len(),
            top,
            slowest,
        }
    })
}

#[test]
#[ignore = "probe: prints the cost of classifying every identifier of a project"]
fn library_use_census() {
    let project = std::env::var("RMC_USE_CENSUS_PROJECT")
        .expect("RMC_USE_CENSUS_PROJECT names the project to classify");
    let threads: usize = std::env::var("RMC_USE_CENSUS_THREADS")
        .map(|t| t.parse().unwrap())
        .unwrap_or(4);
    let root = Path::new(&project).canonicalize().unwrap();

    let rss_start = rss_mib();
    let at = Instant::now();
    let (mut host, vfs, _layout) = super::loader::load_project_full(&root).unwrap();
    let load = at.elapsed();
    let rss_loaded = rss_mib();
    let files = super::position::project_rust_files(&vfs, &root);

    let at = Instant::now();
    let report = pass(&host.raw_database().clone(), &files);
    let cold = at.elapsed();
    let rss_cold = rss_mib();

    host.trigger_garbage_collection();
    crate::mcp::memory::release_free_memory();
    let rss_collected = rss_mib();

    let at = Instant::now();
    pass(&host.raw_database().clone(), &files);
    let second = at.elapsed();
    let rss_second = rss_mib();

    host.trigger_garbage_collection();
    crate::mcp::memory::release_free_memory();
    let at = Instant::now();
    let chunk = files.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for part in files.chunks(chunk) {
            let db = host.raw_database().clone();
            scope.spawn(move || pass(&db, part));
        }
    });
    let parallel = at.elapsed();
    let rss_parallel = rss_mib();

    // The table's own pass (`uses::library_uses_in`, a `goto_definition` per
    // word in code), single-threaded, after a collection.
    host.trigger_garbage_collection();
    crate::mcp::memory::release_free_memory();
    let analysis = host.analysis();
    let at = Instant::now();
    let mut table = super::uses::UseTable::all_pending(files.iter().copied());
    for &file in &files {
        let found = super::uses::library_uses_in(&analysis, file).unwrap();
        table.set_ready(file, found);
    }
    let table_pass = at.elapsed();
    let (ready, _, rows) = table.size();
    let rss_table = rss_mib();

    let slowest: Vec<_> = report
        .slowest
        .iter()
        .map(|(time, file)| (*time, vfs.file_path(*file).to_string()))
        .collect();
    eprintln!(
        "use census of {} ({} project .rs files):\n  load {load:?}\n  \
         cold pass {cold:?} ({:?}/file), second pass after GC {second:?}, \
         {threads}-thread pass after GC {parallel:?}\n  \
         table pass (goto_definition per word) after GC {table_pass:?}: {ready} files, \
         {rows} rows, RSS after {rss_table} MiB\n  \
         RSS MiB: start {rss_start}, loaded {rss_loaded}, after cold pass {rss_cold}, \
         after GC+trim {rss_collected}, after second pass {rss_second}, after parallel \
         pass {rss_parallel}\n  {:#?}\n  distinct library items {}\n  top: {:?}\n  \
         slowest files: {slowest:?}",
        root.display(),
        files.len(),
        cold / files.len().max(1) as u32,
        report.counts,
        report.distinct_library_items,
        report.top,
    );
}
