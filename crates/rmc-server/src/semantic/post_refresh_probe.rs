//! Probe: what the first query after a refresh pays, part by part (plan item
//! `post-refresh-cost`). Measurement only.
//!
//! # What it separates
//!
//! On a live daemon the first `find_*` after a refresh with edits took 5–11 s
//! where the same query warm takes 0.1–0.4 s, and the refresh itself (our
//! `apply_tree_change`) was only 0.2–3 s of it. Three things can stand between
//! the refresh and a warm answer, and the daemon's log cannot tell them apart:
//!
//! - rust-analyzer honestly recomputing what the edits changed (item trees,
//!   def maps and the symbol index of the edited crates and of the crates
//!   depending on them, inference of the bodies the query reads);
//! - the garbage collection (every `RMC_GC_INTERVAL_SECS`), which was a
//!   synthetic write (`trigger_garbage_collection`): a revision bump at `LOW`,
//!   so the next query re-verified every memo it touched even with no edit;
//! - the builder of the table of uses, which starts a batch on the files a
//!   refresh made pending right beside the query.
//!
//! # What it found (2026-11-28, a ~9000-file workspace, load average ≈ 64 on 24 threads)
//!
//! The first: the symbol index 2.4–3.2 s after 3 edited files, 4.0–9.0 s after
//! 20, warm 0.02–0.06 s; `find_references` of a type a further +1.5–2.5 s
//! over warm. The synthetic write added 0.0–0.2 s, the builder nothing out of
//! the noise; our refresh 0.02–0.8 s. Hence the background refresh that primes
//! the symbol index (`SemanticService::refresh_in_background`), and the
//! collection became an eviction without the bump (`collect_garbage`).
//!
//! # How
//!
//! The workspace is loaded once, read-only, and its watch frozen, so what
//! others write to it meanwhile stays out. Each commit named is replayed in
//! memory: the modified `.rs` files get the texts the commit's parent had
//! (`git show <commit>~1:<path>`) and then back the texts the load read,
//! through the same `apply_tree_change` a refresh takes. Between the steps the
//! served queries run as the daemon runs them (`symbol_search_within`,
//! `find_references_within`, eviction after each query); `def:` queries also
//! time the symbol index alone first (`index`), which is the part a
//! `find_definition` shares with every other.
//!
//! ```text
//! RMC_POST_REFRESH_PROJECT=/path/to/workspace \
//! RMC_POST_REFRESH_COMMITS=<sha>,<sha> \
//! RMC_POST_REFRESH_QUERIES=def:load_project,refs:undo_to_position \
//!   cargo test --release -p rmc-server --lib -- --ignored --nocapture post_refresh_cost
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ra_ap_vfs::VfsPath;

use super::{SemanticService, TreeChange, UsesBatch, apply_tree_change_with, tree_watch};

/// The served paths' budget (`tools/endpoints/analysis.rs`).
const BUDGET: Duration = Duration::from_secs(30);

enum Query {
    Definition(String),
    References(String),
}

impl Query {
    fn parse(spec: &str) -> Self {
        match spec.split_once(':') {
            Some(("def", name)) => Query::Definition(name.to_owned()),
            Some(("refs", name)) => Query::References(name.to_owned()),
            _ => panic!("a query is def:<name> or refs:<name>, not {spec:?}"),
        }
    }

    fn label(&self) -> String {
        match self {
            Query::Definition(name) => format!("def {name}"),
            Query::References(name) => format!("refs {name}"),
        }
    }
}

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}; see the module header"))
}

fn git(root: &Path, args: &[&str]) -> String {
    let out = Command::new("git").arg("-C").arg(root).args(args).output().expect("git runs");
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).expect("git prints UTF-8")
}

/// The `.rs` files `commit` modified, with the texts its parent had.
fn parent_texts(root: &Path, commit: &str) -> HashMap<PathBuf, String> {
    let parent = format!("{commit}~1");
    git(root, &["diff", "--relative", "--name-only", "--diff-filter=M", &parent, commit, "--", "*.rs"])
        .lines()
        .map(|relative| {
            let text = git(root, &["show", &format!("{parent}:./{relative}")]);
            (root.join(relative), text)
        })
        .collect()
}

fn rss_mib() -> u64 {
    crate::mcp::memory::rss_kib().unwrap_or(0) / 1024
}

struct Probe {
    service: SemanticService,
    root: PathBuf,
    queries: Vec<Query>,
}

impl Probe {
    /// Runs every query once, printing a line per query under `step`.
    fn ask(&self, step: &str) {
        for query in &self.queries {
            if let Query::Definition(name) = query {
                let at = Instant::now();
                let hits = self.service.index_only_search(&self.root, name, 50).expect("index search");
                self.line(step, "index", &format!("def {name}"), at.elapsed(), hits.len(), "");
            }
            let at = Instant::now();
            let answer = match query {
                Query::Definition(name) => {
                    self.service.symbol_search_within(&self.root, name, 50, true, BUDGET)
                }
                Query::References(name) => {
                    self.service.find_references_within(&self.root, name, true, BUDGET)
                }
            }
            .expect("served query");
            let note = match &answer.incomplete {
                Some(why) => format!("INCOMPLETE {why}"),
                None => format!(
                    "{} read, {} RA, {} decl",
                    answer.cost.files_read, answer.cost.ra_queries, answer.cost.declaration_searches
                ),
            };
            self.line(step, "tool", &query.label(), at.elapsed(), answer.locations.len(), &note);
        }
    }

    fn line(&self, step: &str, part: &str, query: &str, took: Duration, found: usize, note: &str) {
        println!(
            "| {step} | {part} | {query} | {:.2} s | {found} | {note} | {} MiB |",
            took.as_secs_f64(),
            rss_mib()
        );
    }

    /// Sets the texts of `texts` the way a refresh would and returns the
    /// texts the database held before; prints how long our side took.
    fn edit(&self, step: &str, texts: &HashMap<PathBuf, String>) -> HashMap<PathBuf, String> {
        let slot = self.service.slot_for(&self.root);
        let mut guard = slot.lock();
        let ctx = guard.as_mut().expect("loaded");
        let held: HashMap<PathBuf, String> = {
            let vfs = super::read_vfs(&ctx.vfs);
            let analysis = ctx.host.analysis();
            texts
                .keys()
                .map(|path| {
                    let vfs_path = VfsPath::new_real_path(path.to_string_lossy().into_owned());
                    let (file_id, _) = vfs
                        .file_id(&vfs_path)
                        .unwrap_or_else(|| panic!("{} is not in the analysis", path.display()));
                    let text = analysis.file_text(file_id).expect("no write in flight");
                    (path.clone(), text.to_string())
                })
                .collect()
        };
        let change = TreeChange {
            edited: texts.keys().cloned().collect(),
            added: Vec::new(),
            removed: Vec::new(),
        };
        let at = Instant::now();
        let patched = apply_tree_change_with(ctx, &change, |path| Ok(texts[path].clone()))
            .expect("the edit applies");
        println!(
            "| {step} | refresh | {} files, {} surface roots, {} pending, {} widened | {:.2} s | | {} | {} MiB |",
            patched.edited,
            patched.surface_changes,
            patched.uses_pending,
            patched.uses_widened,
            at.elapsed().as_secs_f64(),
            patched.widening,
            rss_mib()
        );
        held
    }

    /// The daemon's garbage collection on this project, timed.
    fn collect_garbage(&self, step: &str) {
        let at = Instant::now();
        let collected = self.service.collect_garbage();
        assert_eq!(collected, 1, "one project loaded");
        println!("| {step} | gc | LRU eviction + type sweep | {:.2} s | | | {} MiB |", at.elapsed().as_secs_f64(), rss_mib());
    }

    /// Runs `ask` with the builder of the table of uses working beside it,
    /// one thread, as the daemon's default.
    fn ask_beside_the_builder(&self, step: &str) {
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let builder = scope.spawn(|| {
                let mut batches = 0usize;
                while !stop.load(Ordering::SeqCst) {
                    match self.service.build_uses_batch(1, 8) {
                        UsesBatch::Built { .. } => batches += 1,
                        UsesBatch::Busy => std::thread::sleep(Duration::from_millis(5)),
                        UsesBatch::Idle => break,
                    }
                }
                batches
            });
            // The builder's first batch is under way before the query, as after
            // a refresh in the daemon (it polls every 2 s; a refresh leaves
            // pending files).
            std::thread::sleep(Duration::from_millis(200));
            self.ask(step);
            stop.store(true, Ordering::SeqCst);
            let batches = builder.join().expect("builder");
            println!("| {step} | builder | {batches} batch(es) beside the queries | | | | |");
        });
    }
}

#[test]
#[ignore = "probe: needs a real workspace in RMC_POST_REFRESH_PROJECT and minutes to run"]
fn post_refresh_cost() {
    let _analysis = crate::deep_stack::test_holding_an_analysis();
    let root = PathBuf::from(required("RMC_POST_REFRESH_PROJECT")).canonicalize().unwrap();
    let commits: Vec<String> =
        required("RMC_POST_REFRESH_COMMITS").split(',').map(str::to_owned).collect();
    let queries: Vec<Query> =
        required("RMC_POST_REFRESH_QUERIES").split(',').map(Query::parse).collect();
    let probe = Probe { service: SemanticService::new(), root: root.clone(), queries };

    println!("| step | part | what | took | found | note | RSS |");
    println!("|---|---|---|---|---|---|---|");
    let at = Instant::now();
    probe.service.get_or_load_full(&root).expect("load");
    println!("| load | | | {:.1} s | | | {} MiB |", at.elapsed().as_secs_f64(), rss_mib());
    {
        let slot = probe.service.slot_for(&root);
        let mut guard = slot.lock();
        guard.as_mut().expect("loaded").watch = tree_watch::TreeWatch::frozen(&root);
    }

    probe.ask("cold");
    probe.ask("warm");
    probe.collect_garbage("gc");
    probe.ask("after gc");
    probe.ask("warm");

    for commit in &commits {
        let parent = parent_texts(&root, commit);
        let short = &commit[..commit.len().min(9)];
        let current = probe.edit(&format!("{short} revert"), &parent);
        probe.ask(&format!("{short} revert"));
        probe.ask("warm");
        probe.edit(&format!("{short} redo"), &current);
        probe.ask(&format!("{short} redo"));
        probe.ask("warm");
        probe.edit(&format!("{short} revert"), &parent);
        probe.ask_beside_the_builder(&format!("{short} revert + builder"));
        probe.ask("warm");
        probe.edit(&format!("{short} redo"), &current);
        probe.collect_garbage(&format!("{short} redo"));
        probe.ask(&format!("{short} redo + gc"));
        probe.ask("warm");
    }
}

/// Probe (plan item `manifest-reload-in-place`): what a reload in place costs
/// against a full load of the same workspace, and whether they answer alike.
///
/// The workspace is loaded once and its watch frozen; then reloaded in place
/// with nothing changed (the floor: `cargo metadata`, the read of every file,
/// the verification of every memo); then — with `RMC_RELOAD_PROBE_FEATURE_IN`
/// naming a package manifest of a *copy* (the probe writes it) — reloaded in
/// place after a feature is added to it, which changes that crate's cfg and
/// re-resolves its reverse dependencies. Last, that database is dropped and
/// the edited tree loaded afresh, and the queries asked again: the answers'
/// counts are printed beside each other.
///
/// ```text
/// RMC_POST_REFRESH_PROJECT=/path/to/copy RMC_POST_REFRESH_QUERIES=def:X,refs:Y \
/// RMC_RELOAD_PROBE_FEATURE_IN=/path/to/copy/crates/a/Cargo.toml \
///   cargo test --release -p rmc-server --lib -- --ignored --nocapture reload_in_place_cost
/// ```
#[test]
#[ignore = "probe: needs a real workspace in RMC_POST_REFRESH_PROJECT and minutes to run"]
fn reload_in_place_cost() {
    let _analysis = crate::deep_stack::test_holding_an_analysis();
    let root = PathBuf::from(required("RMC_POST_REFRESH_PROJECT")).canonicalize().unwrap();
    let queries: Vec<Query> =
        required("RMC_POST_REFRESH_QUERIES").split(',').map(Query::parse).collect();
    let feature_in = std::env::var("RMC_RELOAD_PROBE_FEATURE_IN").ok().map(PathBuf::from);
    let probe = Probe { service: SemanticService::new(), root: root.clone(), queries };

    println!("| step | part | what | took | found | note | RSS |");
    println!("|---|---|---|---|---|---|---|");
    let at = Instant::now();
    probe.service.get_or_load_full(&root).expect("load");
    println!("| load | | | {:.1} s | | | {} MiB |", at.elapsed().as_secs_f64(), rss_mib());
    let reload = |step: &str| {
        let slot = probe.service.slot_for(&root);
        let mut guard = slot.lock();
        let ctx = guard.as_mut().expect("loaded");
        ctx.watch = tree_watch::TreeWatch::frozen(&root);
        let at = Instant::now();
        let done = super::reload_in_place::reload_in_place(ctx, &root).expect("reload in place");
        println!(
            "| {step} | reload in place | metadata {:.1} s, change {:.1} s, prime {:.1} s | {:.1} s | | {done:?} | {} MiB |",
            done.workspace.as_secs_f64(),
            done.apply.as_secs_f64(),
            done.prime.as_secs_f64(),
            at.elapsed().as_secs_f64(),
            rss_mib()
        );
    };
    reload("frozen");
    probe.ask("warm");
    probe.ask("warm");
    reload("unchanged");
    probe.ask("after reload");

    let Some(manifest) = feature_in else {
        return;
    };
    let text = std::fs::read_to_string(&manifest).expect("read the manifest");
    let edited = match text.find("[features]\n") {
        Some(at) => {
            let end = at + "[features]\n".len();
            format!("{}rmc_reload_probe = []\n{}", &text[..end], &text[end..])
        }
        None => format!("{text}\n[features]\nrmc_reload_probe = []\n"),
    };
    std::fs::write(&manifest, edited).expect("write the manifest");
    reload("a feature added");
    probe.ask("after reload");
    probe.ask("warm");

    drop(probe.service);
    crate::mcp::memory::release_free_memory();
    let fresh = Probe {
        service: SemanticService::new(),
        root: root.clone(),
        queries: probe.queries,
    };
    let at = Instant::now();
    fresh.service.get_or_load_full(&root).expect("load");
    println!("| fresh load | | | {:.1} s | | | {} MiB |", at.elapsed().as_secs_f64(), rss_mib());
    fresh.ask("fresh");
    fresh.ask("warm");
}
