//! Probe, not a test: on a project's recent git history, how many files each
//! commit's surface changes make pending — by the crates (every file of the
//! changed packages and their reverse dependencies, what a refresh did up to
//! plan step `uses-narrow-widening`) and by the names (`widening::widen`).
//!
//! No rust-analyzer: packages and their dependencies come from `cargo metadata
//! --no-deps` (path dependencies between members, every kind), the other
//! files' texts from the working tree as it is now — a commit is judged
//! against today's tree, which is close enough for counts.
//!
//! `RMC_WIDENING_REPO=<dir> [RMC_WIDENING_SINCE="24 hours ago"]
//! [RMC_WIDENING_UNTIL=<date>] cargo test -p
//! rmc-server --lib -- --ignored --nocapture widening_census`

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ra_ap_vfs::FileId;

use ra_ap_syntax::{AstNode, SyntaxKind};

use super::units::{self, ModuleFiles, Reach, Side};
use super::widening::{self, Scope, Widening};
use super::word_index::WordIndex;

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .unwrap();
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Workspace packages: their directories, and by package the packages that
/// depend on it.
struct Packages {
    dirs: Vec<PathBuf>,
    dependents: Vec<Vec<usize>>,
}

impl Packages {
    fn of(repo: &Path) -> Self {
        let output = Command::new("cargo")
            .args([
                "metadata",
                "--no-deps",
                "--format-version",
                "1",
                "--offline",
            ])
            .current_dir(repo)
            .output()
            .unwrap();
        assert!(output.status.success(), "cargo metadata failed");
        let meta: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let packages = meta["packages"].as_array().unwrap();
        let dirs: Vec<PathBuf> = packages
            .iter()
            .map(|package| {
                Path::new(package["manifest_path"].as_str().unwrap())
                    .parent()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        let by_dir: HashMap<&Path, usize> = dirs
            .iter()
            .enumerate()
            .map(|(i, dir)| (dir.as_path(), i))
            .collect();
        let mut dependents = vec![Vec::new(); dirs.len()];
        for (i, package) in packages.iter().enumerate() {
            for dependency in package["dependencies"].as_array().unwrap() {
                if let Some(path) = dependency["path"].as_str() {
                    if let Some(&j) = by_dir.get(Path::new(path)) {
                        dependents[j].push(i);
                    }
                }
            }
        }
        Packages { dirs, dependents }
    }

    /// The package whose directory is the longest prefix of `path`.
    fn of_file(&self, path: &Path) -> Option<usize> {
        self.dirs
            .iter()
            .enumerate()
            .filter(|(_, dir)| path.starts_with(dir))
            .max_by_key(|(_, dir)| dir.as_os_str().len())
            .map(|(i, _)| i)
    }

    fn depending_on(&self, changed: &HashSet<usize>) -> HashSet<usize> {
        let mut seen = HashSet::new();
        let mut work: Vec<usize> = changed.iter().copied().collect();
        while let Some(package) = work.pop() {
            if seen.insert(package) {
                work.extend(self.dependents[package].iter().copied());
            }
        }
        seen
    }
}

#[derive(Default)]
struct Totals {
    commits: usize,
    with_rust: usize,
    no_surface_change: usize,
    opaque_changed: usize,
    opaque_in_closure: usize,
    past_budget: usize,
    by_names: usize,
    pending_by_crates: Vec<usize>,
    pending_by_names: Vec<usize>,
    /// Of the commits the names could tell — without those that went by the
    /// crates (mostly item macros the probe cannot expand).
    pending_names_told: Vec<usize>,
    reasons: BTreeMap<String, usize>,
}

fn quantiles(values: &[usize]) -> String {
    if values.is_empty() {
        return "-".to_owned();
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    format!(
        "p50 {} p90 {} max {} sum {}",
        at(0.5),
        at(0.9),
        sorted[sorted.len() - 1],
        sorted.iter().sum::<usize>()
    )
}

/// Reasons cut to their kind: a macro's name stays, the closure's word does not.
fn reason_kind(why: &str) -> String {
    match why.split_once(" mentions `") {
        Some((head, _)) => format!("closure: {head}"),
        None if why.starts_with("the names reach past") => {
            "closure: past the parse budget".to_owned()
        }
        None => why.to_owned(),
    }
}

#[test]
#[ignore = "probe: reads a project's git history; RMC_WIDENING_REPO=<dir>"]
fn widening_census() {
    let repo = PathBuf::from(std::env::var("RMC_WIDENING_REPO").expect("RMC_WIDENING_REPO=<dir>"));
    let since = std::env::var("RMC_WIDENING_SINCE").unwrap_or_else(|_| "24 hours ago".to_owned());
    // `empty`: an item macro taken as defining nothing either — the bound of
    // what expanding them can give (with analysis they define what they define).
    let trace_commit = std::env::var("RMC_WIDENING_TRACE").ok();
    let item_macros_empty = std::env::var("RMC_WIDENING_ITEM_MACROS").is_ok_and(|v| v == "empty");

    let started = Instant::now();
    let (paths, _) = rmc_indexing::indexing::collect_project_rust_files(&repo);
    let mut ids: HashMap<PathBuf, FileId> = HashMap::new();
    let mut words = WordIndex::default();
    for (i, path) in paths.iter().enumerate() {
        let id = FileId::from_raw(i as u32);
        ids.insert(path.clone(), id);
        words.set_file(id, Some(&std::fs::read_to_string(path).unwrap_or_default()));
    }
    // What cutting every file into units costs: the price of keeping an index
    // of surface mentions from the load on.
    let clock = Instant::now();
    let (mut unit_count, mut mention_count) = (0, 0);
    for path in &paths {
        let units = units::units_of(&std::fs::read_to_string(path).unwrap_or_default());
        unit_count += units.len();
        mention_count += units.iter().map(|unit| unit.mentions.len()).sum::<usize>();
    }
    eprintln!(
        "units of every file: {unit_count} units, {mention_count} mentions, {:.1} s",
        clock.elapsed().as_secs_f64()
    );
    let packages = Packages::of(&repo);
    let package_of: Vec<Option<usize>> = paths.iter().map(|path| packages.of_file(path)).collect();
    eprintln!(
        "{} files, {} packages, word index in {:.1} s",
        paths.len(),
        packages.dirs.len(),
        started.elapsed().as_secs_f64()
    );

    // Paths of `diff-tree` are from the top of the git tree, which may hold
    // the project in a subdirectory.
    let top = PathBuf::from(
        git(&repo, &["rev-parse", "--show-toplevel"])
            .unwrap()
            .trim(),
    );
    // A fixed window (`RMC_WIDENING_UNTIL`) keeps two runs comparable.
    let until = std::env::var("RMC_WIDENING_UNTIL").unwrap_or_else(|_| "now".to_owned());
    let commits = git(
        &repo,
        &[
            "log",
            "--since",
            &since,
            "--until",
            &until,
            "--format=%H",
            "--reverse",
            "--no-merges",
            "--",
            ".",
        ],
    )
    .unwrap();
    let mut totals = Totals::default();
    let mut rows: Vec<(usize, usize, String, String)> = Vec::new();
    let mut closure_secs = 0.0;
    // By commit: the closure's time (ms), files parsed, macros expanded.
    let mut closure_ms: Vec<usize> = Vec::new();
    let mut parsed_per_commit: Vec<usize> = Vec::new();
    let mut expanded_per_commit: Vec<usize> = Vec::new();
    let mut parsed_total = 0;
    let mut lexed_total = 0;
    let mut optimistic_total = 0;
    // By name as the widening shows it (`x@`, `x&T`, `T*`): the files holding
    // its bare word, summed over the commits — an upper bound of its share.
    let mut widest_names: HashMap<String, (usize, usize)> = HashMap::new();
    for commit in commits.lines() {
        totals.commits += 1;
        let changes = git(
            &repo,
            &[
                "diff-tree",
                "-r",
                "--no-commit-id",
                "--no-renames",
                "--name-status",
                commit,
            ],
        )
        .unwrap();
        let tracing_this = trace_commit
            .as_deref()
            .is_some_and(|prefix| commit.starts_with(prefix));
        if trace_commit.is_some() && !tracing_this {
            continue;
        }
        let trace = std::cell::RefCell::new(Vec::new());
        let mut changed: Vec<(FileId, Vec<(Side, units::Unit)>)> = Vec::new();
        // The texts of the commit's changed files, old and new: what their
        // macros are expanded on.
        let mut texts: HashMap<FileId, (Option<String>, Option<String>)> = HashMap::new();
        let mut changed_packages: HashSet<usize> = HashSet::new();
        let mut rust_files = 0;
        let mut extra_id = paths.len() as u32;
        for line in changes.lines() {
            let Some((status, path)) = line.split_once('\t') else {
                continue;
            };
            if !path.ends_with(".rs") {
                continue;
            }
            let absolute = top.join(path);
            if !absolute.starts_with(&repo) {
                continue;
            }
            // Only the files a load would hold: under the walk's boundaries.
            if !ids.contains_key(&absolute) && status != "D" {
                continue;
            }
            rust_files += 1;
            let old = (status != "A")
                .then(|| git(&repo, &["show", &format!("{commit}^:{path}")]))
                .flatten();
            let new = (status != "D")
                .then(|| git(&repo, &["show", &format!("{commit}:{path}")]))
                .flatten();
            let units = units::changed_units(old.as_deref(), new.as_deref());
            if units.is_empty() {
                continue;
            }
            let id = ids.get(&absolute).copied().unwrap_or_else(|| {
                extra_id += 1;
                FileId::from_raw(extra_id)
            });
            changed_packages.extend(packages.of_file(&absolute));
            texts.insert(id, (old, new));
            changed.push((id, units));
        }
        if rust_files == 0 {
            continue;
        }
        totals.with_rust += 1;
        if changed.is_empty() {
            totals.no_surface_change += 1;
            continue;
        }
        let reach = packages.depending_on(&changed_packages);
        let by_crates = package_of
            .iter()
            .filter(|p| p.is_some_and(|p| reach.contains(&p)))
            .count();
        let reachable = |file: FileId| {
            package_of
                .get(file.index() as usize)
                .copied()
                .flatten()
                .is_some_and(|p| reach.contains(&p))
        };
        let disk = |file: FileId| {
            paths
                .get(file.index() as usize)
                .map(|path| std::fs::read_to_string(path).unwrap_or_default())
                .unwrap_or_default()
        };
        let text = |file: FileId| match texts.get(&file) {
            Some((_, Some(new))) => new.clone(),
            _ => disk(file),
        };
        let module_files = |file: FileId, module: &ModuleFiles| -> Vec<FileId> {
            let Some(path) = paths.get(file.index() as usize) else {
                return Vec::new();
            };
            let (exact, dirs) = widening::module_paths(path, module);
            paths
                .iter()
                .filter(|candidate| {
                    exact.contains(candidate) || dirs.iter().any(|dir| candidate.starts_with(dir))
                })
                .map(|candidate| ids[candidate])
                .collect()
        };
        let module_names = |file: FileId| -> Vec<String> {
            paths
                .get(file.index() as usize)
                .map(|path| widening::module_names_by_path(path))
                .unwrap_or_default()
        };
        // No analysis here: an attribute is taken as written (as if no macro),
        // a body macro as expanding to nothing that escapes, an item macro as
        // unknown. Counted, to tell what the estimate leans on.
        let optimistic = std::cell::Cell::new(0usize);
        let trees: std::cell::RefCell<HashMap<(FileId, bool), ra_ap_syntax::SourceFile>> =
            Default::default();
        let expand = |side: Side,
                      file: FileId,
                      at: ra_ap_syntax::TextRange,
                      attr: bool,
                      module: &[String]|
         -> Result<Vec<units::Unit>, String> {
            let old = side == Side::Old;
            let root = trees
                .borrow_mut()
                .entry((file, old))
                .or_insert_with(|| {
                    let source = match (side, texts.get(&file)) {
                        (Side::Old, Some((Some(old), _))) => old.clone(),
                        _ => text(file),
                    };
                    ra_ap_syntax::SourceFile::parse(&source, ra_ap_syntax::Edition::CURRENT).tree()
                })
                .clone();
            if attr {
                optimistic.set(optimistic.get() + 1);
                let written =
                    units::as_written(root.syntax(), at, module).ok_or("no item at the range")?;
                Ok(written
                    .into_iter()
                    .filter(|unit| !matches!(unit.reach, Reach::Expand { .. }))
                    .collect())
            } else {
                let call =
                    units::macro_call_at(root.syntax(), at).ok_or("no macro call at the range")?;
                let in_items = call.syntax().parent().is_some_and(|parent| {
                    matches!(
                        parent.kind(),
                        SyntaxKind::SOURCE_FILE
                            | SyntaxKind::ITEM_LIST
                            | SyntaxKind::ASSOC_ITEM_LIST
                            | SyntaxKind::EXTERN_ITEM_LIST
                    )
                });
                if in_items && !item_macros_empty {
                    let name = call
                        .path()
                        .map(|path| path.syntax().text().to_string())
                        .unwrap_or_default();
                    return Err(format!("an item macro {name}! (no analysis in the probe)"));
                }
                optimistic.set(optimistic.get() + 1);
                Ok(Vec::new())
            }
        };
        let changed: Vec<(FileId, Vec<(Side, units::Unit)>)> = changed
            .into_iter()
            .map(|(file, units)| (file, widening::expand_old_side(file, units, &expand)))
            .collect();
        let scope = Scope {
            words: &words,
            reachable: &reachable,
            text: &text,
            module_files: &module_files,
            module_names: &module_names,
            expand: &expand,
            parse_budget: widening::PARSE_BUDGET,
            trace: tracing_this.then_some(&trace),
        };
        let clock = Instant::now();
        let widening = widening::widen(&changed, &scope);
        closure_ms.push(clock.elapsed().as_millis() as usize);
        if let Widening::Files {
            parsed, expanded, ..
        } = &widening
        {
            parsed_per_commit.push(*parsed);
            expanded_per_commit.push(*expanded);
        }
        if tracing_this {
            let lines = trace.borrow();
            eprintln!("trace of {commit}: {} names", lines.len());
            for line in lines.iter().take(80) {
                let file = |raw: &str| raw.to_owned();
                eprintln!("  {}", file(line));
            }
        }
        closure_secs += clock.elapsed().as_secs_f64();
        optimistic_total += optimistic.get();
        let edited = changed.len();
        let (by_names, what) = match &widening {
            Widening::Roots(why) => {
                let kind = reason_kind(why);
                if why.starts_with("changed: ") {
                    totals.opaque_changed += 1;
                } else if kind.contains("parse budget") {
                    totals.past_budget += 1;
                } else {
                    totals.opaque_in_closure += 1;
                }
                *totals.reasons.entry(kind).or_default() += 1;
                (by_crates, why.clone())
            }
            Widening::Files {
                files,
                names,
                widths,
                parsed,
                expanded,
                lexed,
            } => {
                lexed_total += lexed;
                totals.by_names += 1;
                parsed_total += parsed;
                // The widest names first: what the count is made of — the
                // files each made pending, and the files with its bare word.
                let mut by_width: Vec<(usize, &String)> =
                    names.iter().map(|name| (widths[name], name)).collect();
                by_width.sort_by(|a, b| b.0.cmp(&a.0));
                for (width, name) in &by_width {
                    // `x@`, `T*`, `x&T`, `x~Tr`, `T::x`, a `!` when untold.
                    let bare = name.trim_end_matches(['@', '*', '!']);
                    let bare = bare.rsplit("::").next().unwrap_or(bare);
                    let bare = bare.split(['&', '~']).next().unwrap_or(bare);
                    let entry = widest_names.entry((*name).clone()).or_default();
                    entry.0 += width;
                    entry.1 += words.files_with(bare).len();
                }
                let shown: Vec<String> = by_width
                    .iter()
                    .take(6)
                    .map(|(n, name)| format!("{name}:{n}"))
                    .collect();
                let more = names.len().saturating_sub(6);
                (
                    (files.len() + edited).min(by_crates.max(edited)),
                    format!(
                        "names [{}{}], parsed {parsed}, expanded {expanded}",
                        shown.join(", "),
                        if more > 0 {
                            format!(" +{more}")
                        } else {
                            String::new()
                        }
                    ),
                )
            }
        };
        totals.pending_by_crates.push(by_crates);
        totals.pending_by_names.push(by_names);
        if matches!(widening, Widening::Files { .. }) {
            totals.pending_names_told.push(by_names);
        }
        rows.push((by_crates, by_names, commit[..9].to_owned(), what));
    }

    eprintln!(
        "\n{} commits since {since:?}: {} touch loaded .rs files; {} change no surface; of the rest \
         {} by names, {} opaque among the changed units, {} opaque in the closure, {} past the budget",
        totals.commits,
        totals.with_rust,
        totals.no_surface_change,
        totals.by_names,
        totals.opaque_changed,
        totals.opaque_in_closure,
        totals.past_budget
    );
    eprintln!(
        "pending by crates: {}",
        quantiles(&totals.pending_by_crates)
    );
    eprintln!("pending by names:  {}", quantiles(&totals.pending_by_names));
    eprintln!(
        "  of {} told by names: {}",
        totals.pending_names_told.len(),
        quantiles(&totals.pending_names_told)
    );
    eprintln!(
        "closure: {parsed_total} files parsed, {lexed_total} lexed in {closure_secs:.1} s; {optimistic_total} macros taken optimistically (attribute as written, body macro empty)"
    );
    eprintln!(
        "by commit: closure ms {}; parsed {}; expanded {}",
        quantiles(&closure_ms),
        quantiles(&parsed_per_commit),
        quantiles(&expanded_per_commit)
    );
    eprintln!("\nwhy the crates (commits):");
    let mut reasons: Vec<(&String, &usize)> = totals.reasons.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1));
    for (why, count) in reasons.iter().take(40) {
        eprintln!("  {count:4}  {why}");
    }
    let mut widest: Vec<(&String, &(usize, usize))> = widest_names.iter().collect();
    widest.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    eprintln!(
        "\nwidest names (files made pending, and files with the bare word, summed over commits):"
    );
    for (name, (width, bare)) in widest.iter().take(30) {
        eprintln!("  {width:7} {bare:7}  {name}");
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    eprintln!("\nwidest by crates (crates → names):");
    for (crates, names, commit, what) in rows.iter().take(40) {
        eprintln!(
            "  {commit} {crates:5} → {names:5}  {}",
            what.chars().take(160).collect::<String>()
        );
    }
}
