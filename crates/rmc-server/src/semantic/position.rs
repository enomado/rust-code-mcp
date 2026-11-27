//! Position and coordinate utilities

use anyhow::{Context, Result};
use ra_ap_ide::{
    Analysis, AnalysisHost, FilePosition, LineCol, NavigationTarget, Query, RootDatabase,
    SymbolKind, TextSize,
};
use ra_ap_vfs::{Vfs, VfsPath};

use super::uses::UseTable;
use super::word_index::{WordIndex, code_occurrences, field_shape_occurrences};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A source code location
#[derive(Debug, Clone)]
pub(crate) struct Location {
    pub file_path: PathBuf,
    pub line: u32,   // 1-based
    pub column: u32, // 1-based
    pub name: String,
    pub exact: bool,
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{} ({}, exact={})",
            self.file_path.display(),
            self.line,
            self.column,
            self.name,
            self.exact
        )
    }
}

/// The references of a name, and whether the search was cut short.
#[derive(Debug)]
pub(crate) struct SweptAnswer {
    pub locations: Vec<Location>,
    /// `Some(why)` when a file sweep ran out of its time budget: the
    /// locations are what was found until then, and the answer must say so
    /// (a short list read as complete is read as "this code is dead").
    pub incomplete: Option<String>,
    /// What the file sweeps of this answer cost — the number that must not
    /// grow with the size of the workspace, only with the occurrences of the
    /// name.
    pub cost: SweepCost,
}

/// The work of the name sweeps (`collect_by_token_scan`,
/// `collect_library_refs`) for one answer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SweepCost {
    /// Candidate files whose text was read and lexed.
    pub files_read: usize,
    /// rust-analyzer queries asked about occurrences (`find_all_refs` /
    /// `goto_definition`).
    pub ra_queries: usize,
    /// Files whose uses of a library declaration came from the table of uses
    /// (`uses::UseTable`) rather than a sweep: no query asked about them.
    pub files_from_table: usize,
}

/// When a file sweep must stop, and what it was doing when it did.
///
/// # Why a budget
///
/// The sweeps (`collect_library_refs`, `collect_by_token_scan`) ask
/// rust-analyzer about every occurrence of a name in every project file, and
/// the per-match loop runs one `find_all_refs` for each of up to 50 fuzzy
/// matches. For a common name on a large workspace that is unbounded: on a
/// ~8000-file workspace a fuzzy `find_references` of `Widget::new` ran
/// for over 100 minutes (2026-11-26; which of the three loops it sat in was
/// not observable — the budget bounds all three). The last attempt of a
/// query keeps its project's slot for its whole run (`with_project`), so every
/// other call on the project waited behind it — 22 of them, none answered.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SweepBudget {
    started: std::time::Instant,
    deadline: Option<std::time::Instant>,
}

impl SweepBudget {
    pub(crate) fn unlimited() -> Self {
        Self {
            started: std::time::Instant::now(),
            deadline: None,
        }
    }

    pub(crate) fn of(limit: std::time::Duration) -> Self {
        let started = std::time::Instant::now();
        Self {
            started,
            deadline: Some(started + limit),
        }
    }

    fn exhausted(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
    }

    /// The note an answer carries when `sweep` stopped before file `done`
    /// of `total`.
    fn note(&self, sweep: &str, done: usize, total: usize) -> String {
        self.note_of(sweep, done, total, "project files searched")
    }

    fn note_of(&self, sweep: &str, done: usize, total: usize, unit: &str) -> String {
        format!(
            "time budget of {:.0} s exhausted in {sweep}: {done} of {total} {unit}; \
             the list above is partial (a full-name query, `exact=true`, or a full path narrows the search)",
            self.deadline
                .map_or(0.0, |deadline| (deadline - self.started).as_secs_f64()),
        )
    }
}

/// Convert file path to FileId
fn path_to_file_id(vfs: &Vfs, file_path: &Path) -> Result<ra_ap_vfs::FileId> {
    let abs_path = file_path
        .canonicalize()
        .context("Failed to canonicalize path")?;
    let vfs_path = VfsPath::new_real_path(abs_path.to_string_lossy().to_string());

    vfs.file_id(&vfs_path)
        .map(|(id, _)| id)
        .ok_or_else(|| anyhow::anyhow!("File not found in VFS: {}", file_path.display()))
}

/// Convert line/column to byte offset
fn to_offset(
    analysis: &Analysis,
    file_id: ra_ap_vfs::FileId,
    line: u32,
    column: u32,
) -> Result<TextSize> {
    let line_index = analysis
        .file_line_index(file_id)
        .context("Failed to get line index")?;

    // LineCol is 0-based, input is 1-based
    let line_col = LineCol {
        line: line.saturating_sub(1),
        col: column.saturating_sub(1),
    };

    line_index
        .offset(line_col)
        .ok_or_else(|| anyhow::anyhow!("Invalid position: line {}, col {}", line, column))
}

pub(crate) fn file_position(
    analysis: &Analysis,
    vfs: &Vfs,
    file_path: &Path,
    line: u32,
    column: u32,
) -> Result<FilePosition> {
    let file_id = path_to_file_id(vfs, file_path)?;
    let offset = to_offset(analysis, file_id, line, column)?;

    Ok(FilePosition { file_id, offset })
}

/// Convert NavigationTarget to Location
fn nav_target_to_location(
    vfs: &Vfs,
    analysis: &ra_ap_ide::Analysis,
    target: &NavigationTarget,
) -> Result<Location> {
    let vfs_path = vfs.file_path(target.file_id);
    let file_path: PathBuf = vfs_path
        .as_path()
        .ok_or_else(|| anyhow::anyhow!("Not a real path"))?
        .to_path_buf()
        .into();

    let line_index = analysis.file_line_index(target.file_id)?;
    let offset = target.focus_range.unwrap_or(target.full_range).start();
    let line_col = line_index.line_col(offset);

    Ok(Location {
        file_path,
        line: line_col.line + 1,
        column: line_col.col + 1,
        name: target.name.to_string(),
        exact: false,
    })
}

/// The one goto-definition configuration this module uses.
pub(super) fn goto_config() -> ra_ap_ide::GotoDefinitionConfig<'static> {
    ra_ap_ide::GotoDefinitionConfig {
        ra_fixture: ra_ap_ide_db::ra_fixture::RaFixtureConfig::default(),
    }
}

/// Where a navigation target points: its name if it has one, else its whole span.
fn nav_offset(nav: &NavigationTarget) -> TextSize {
    nav.focus_range.unwrap_or(nav.full_range).start()
}

/// Identity of a declaration for comparing answers of different queries: the
/// file and the offset `nav_offset` picks. Two navigation targets with the same
/// key are the same declaration.
pub(super) fn nav_key(nav: &NavigationTarget) -> (ra_ap_vfs::FileId, TextSize) {
    (nav.file_id, nav_offset(nav))
}

/// The item name a query asks about: the last segment of a path query
/// (`std::thread::spawn` → `spawn`), the query itself otherwise.
///
/// rust-analyzer's symbol index takes the path apart itself — the leading
/// segments pick the modules, the last one is matched against item names — so
/// a found symbol is named by the last segment only. Comparing that name with
/// the whole query marked every hit of a path query `exact=false`, and
/// `exact=true` then dropped all of them.
fn leaf_name(symbol_name: &str) -> &str {
    symbol_name.rsplit("::").next().unwrap()
}

/// Whether the query names a path (`Widget::new`, `std::thread::spawn`)
/// rather than a bare name.
///
/// A path query never runs the token sweep. The sweep is for what the symbol
/// index lacks — fields — and it looks for the LEAF: for an unresolved
/// `Widget::new` that is every `new` in the project, each asked
/// `find_all_refs` (an occurrence that is not a field is not marked covered),
/// which ran for over 100 minutes on a ~8000-file workspace (2026-11-26).
/// Before the leaf existed the sweep looked for the literal path, which no
/// identifier is, so a path query never swept; this keeps that. A field is
/// asked about by its bare name.
fn is_path_query(symbol_name: &str) -> bool {
    symbol_name.contains("::")
}

/// Goto definition at position
pub(crate) fn goto_definition(
    host: &AnalysisHost,
    vfs: &Vfs,
    file_path: &Path,
    line: u32,
    column: u32,
) -> Result<Vec<Location>> {
    let analysis = host.analysis();
    let position = file_position(&analysis, vfs, file_path, line, column)?;

    let result = analysis
        .goto_definition(position, &goto_config())
        .context("goto_definition query failed")?;

    match result {
        Some(nav_info) => nav_info
            .info
            .iter()
            .map(|target| nav_target_to_location(vfs, &analysis, target))
            .collect(),
        None => Ok(vec![]),
    }
}

/// The reference-search configuration every lookup in this module shares.
///
/// Both exclusions are off on purpose: a `use` line and a `#[cfg(test)]` call
/// site *are* references, and dropping either makes "who uses this" read lower
/// than the truth — the failure this module is built to avoid.
pub(super) fn all_refs_config() -> ra_ap_ide::FindAllRefsConfig<'static> {
    ra_ap_ide::FindAllRefsConfig {
        ra_fixture: ra_ap_ide_db::ra_fixture::RaFixtureConfig::default(),
        search_scope: None,
        exclude_imports: false,
        exclude_tests: false,
    }
}

/// What a position resolved to — enough to tell the name that was asked for
/// from something else spelled the same way.
struct ResolvedDecl {
    name: String,
    kind: Option<SymbolKind>,
}

/// Everything one `find_all_refs` call turned up, kept apart by role.
///
/// The halves are separated because the two callers want different ones: "who
/// uses this" wants both, "where is this declared" wants only the declarations
/// — and yet still needs the use sites, to know which text occurrences it no
/// longer has to ask about.
#[derive(Default)]
struct Harvest {
    /// Declaration sites of whatever the position resolved to.
    decls: Vec<Location>,
    /// Use sites of the same.
    refs: Vec<Location>,
}

impl Harvest {
    /// Every position this harvest accounts for, in dedup-key shape.
    fn keys(&self) -> impl Iterator<Item = (PathBuf, u32, u32)> + '_ {
        self.decls.iter().chain(&self.refs).map(location_key)
    }

    fn into_all(self) -> Vec<Location> {
        let mut all = self.decls;
        all.extend(self.refs);
        all
    }
}

/// Run `find_all_refs` at `position` and report what it found.
///
/// `None` means the position pointed at no symbol at all. That distinction is
/// the whole reason this returns an `Option`: `find_all_refs` reports "the
/// cursor is not on a name" and "this name has no references" identically, as
/// an empty result, and a caller that cannot tell them apart reports a missed
/// position as "nothing uses this".
fn harvest_refs_at(
    analysis: &Analysis,
    vfs: &Vfs,
    position: FilePosition,
    exact: bool,
) -> Result<Option<(ResolvedDecl, Harvest)>> {
    let Some(search_results) = analysis
        .find_all_refs(position, &all_refs_config())
        .context("find_all_refs query failed")?
    else {
        return Ok(None);
    };

    let mut resolved = None;
    let mut harvest = Harvest::default();

    for search_result in search_results {
        // Declaration.nav is NavigationTarget (not Option)
        if let Some(decl) = &search_result.declaration {
            resolved.get_or_insert_with(|| ResolvedDecl {
                name: decl.nav.name.to_string(),
                kind: decl.nav.kind,
            });
            let mut decl_location = nav_target_to_location(vfs, analysis, &decl.nav)?;
            decl_location.exact = exact;
            harvest.decls.push(decl_location);
        }

        // references is IntMap<FileId, Vec<(TextRange, ReferenceCategory)>>
        for (ref_file_id, refs) in &search_result.references {
            let ref_file_path = file_path_of(vfs, *ref_file_id)?;
            let ref_line_index = analysis.file_line_index(*ref_file_id)?;

            for (range, _category) in refs {
                let line_col = ref_line_index.line_col(range.start());
                harvest.refs.push(Location {
                    file_path: ref_file_path.clone(),
                    line: line_col.line + 1,
                    column: line_col.col + 1,
                    name: "reference".to_string(),
                    exact,
                });
            }
        }
    }

    Ok(resolved.map(|decl| (decl, harvest)))
}

/// Real filesystem path of a file the analysis knows.
fn file_path_of(vfs: &Vfs, file_id: ra_ap_vfs::FileId) -> Result<PathBuf> {
    Ok(vfs
        .file_path(file_id)
        .as_path()
        .ok_or_else(|| anyhow::anyhow!("Not a real path"))?
        .to_path_buf()
        .into())
}

/// Find all references at position
///
/// A position that points at no symbol is an **error**, not an empty list:
/// those two answers are the same string to a caller, and one of them means
/// "ask again, one character to the right".
pub(crate) fn find_references(
    host: &AnalysisHost,
    vfs: &Vfs,
    file_path: &Path,
    line: u32,
    column: u32,
) -> Result<Vec<Location>> {
    let analysis = host.analysis();
    let position = file_position(&analysis, vfs, file_path, line, column)?;

    let Some((_, harvest)) = harvest_refs_at(&analysis, vfs, position, false)? else {
        anyhow::bail!(
            "No symbol at {}:{}:{} — the position does not point at a name (try a column inside the identifier)",
            file_path.display(),
            line,
            column
        );
    };

    Ok(harvest.into_all())
}

/// Search for symbols by name, optionally retaining only full-name matches.
///
/// # The blind spot this covers
///
/// The symbol index carries module-scope declarations plus impl and trait
/// members — a **field** is not in it. So "where is this field declared"
/// answered *no definition found*, the same words a misspelled name gets, and
/// the same words that read as "there is no such thing". When the index
/// produces no full-name match, the sources are asked directly, exactly as
/// `find_references_by_name` does — same sweep, same guard against a
/// same-spelled local.
///
/// # What else the index does not carry: nothing, measured
///
/// One of every kind a caller might name was put to the index with the sweep
/// switched off (`which_kinds_the_symbol_index_carries`). Absent: a struct
/// field, a union field, a field of a struct-shaped enum variant — all three
/// `SymbolKind::Field`, all three already inside the sweep's accept list.
/// Present, some of them against expectation: enum variants, associated consts
/// and types, `macro_rules!` macros, and items a macro expansion produced.
///
/// So the blind spot is *fields*, and it is closed. A tuple struct's field has
/// no identifier to ask about — it is named `0` — so it is outside the question
/// rather than an answer to it.
pub(crate) fn symbol_search_with_exact(
    analysis: &Analysis,
    db: &RootDatabase,
    vfs: &Vfs,
    words: &WordIndex,
    symbol_name: &str,
    limit: usize,
    exact_only: bool,
) -> Result<Vec<Location>> {
    let answer = symbol_search_within(
        analysis,
        db,
        vfs,
        words,
        symbol_name,
        limit,
        exact_only,
        SweepBudget::unlimited(),
    )?;
    Ok(answer.locations)
}

/// [`symbol_search_with_exact`] whose token sweep stops at `budget`; a
/// stopped sweep is named in [`SweptAnswer::incomplete`].
pub(crate) fn symbol_search_within(
    analysis: &Analysis,
    db: &RootDatabase,
    vfs: &Vfs,
    words: &WordIndex,
    symbol_name: &str,
    limit: usize,
    exact_only: bool,
    budget: SweepBudget,
) -> Result<SweptAnswer> {
    let mut locations = index_matches(&analysis, vfs, symbol_name, limit, exact_only)?;
    // `Type::method` / `Trait::method`: the index reads leading segments as
    // modules and finds nothing (`assoc_item_navs`).
    if !locations.iter().any(|location| location.exact) {
        for target in assoc_item_navs(db, symbol_name) {
            let mut location = nav_target_to_location(vfs, analysis, &target)?;
            location.exact = true;
            locations.push(location);
        }
    }

    let mut incomplete = None;
    let mut cost = SweepCost::default();
    // A full-name index match may still leave a field of the same name out:
    // see `Occurrences`.
    if !is_path_query(symbol_name) {
        let scope = if locations.iter().any(|location| location.exact) {
            Occurrences::FieldShape
        } else {
            Occurrences::AllCode
        };
        incomplete = collect_by_token_scan(
            &analysis,
            vfs,
            words,
            leaf_name(symbol_name),
            Wants::DeclOnly,
            scope,
            budget,
            &mut cost,
            &mut locations,
        )?;
    }

    let mut ranked = rank_and_filter_exact(locations, exact_only);
    dedup_by_position(&mut ranked);
    // The index was asked for `limit` and the sweep may have added to that;
    // the caller's cap is on the answer, not on either source of it.
    ranked.truncate(limit);
    Ok(SweptAnswer {
        locations: ranked,
        incomplete,
        cost,
    })
}

/// What rust-analyzer's symbol index alone answers for a name.
///
/// Split out from the search so the index can be questioned on its own: the
/// sweep runs only when the index produced no exact match, which means the
/// search's answer conflates "the index carries this kind" with "the sweep
/// rescued it". Which kinds the index actually carries is a fact worth being
/// able to measure rather than guess — the guess about enum variants was half
/// wrong (see `symbol_search_with_exact`).
fn index_matches(
    analysis: &Analysis,
    vfs: &Vfs,
    symbol_name: &str,
    limit: usize,
    exact_only: bool,
) -> Result<Vec<Location>> {
    let name = leaf_name(symbol_name);
    index_navs(analysis, symbol_name, limit, exact_only)?
        .iter()
        .map(|target| {
            let mut location = nav_target_to_location(vfs, analysis, target)?;
            location.exact = location.name == name;
            Ok(location)
        })
        .collect()
}

/// The symbol index's declarations for a name: full-name matches first, then —
/// unless `exact_only` — fuzzy matches up to `limit`.
///
/// # Why two queries
///
/// A fuzzy query matches the name as a *subsequence* and comes back in the
/// index's lexicographic order, cut at `limit`. Any long name that happens to
/// contain the letters in order sorts ahead: on a 4000-file workspace the first
/// 50 fuzzy hits for `spawn` were test functions like
/// `a_batch_builds_one_record_with_a_channel_per_taken_row_or_none`, and the
/// three functions actually named `spawn` were cut off. Filtering for full
/// names after that cut then answered "no definition found" for a name declared
/// three times. The exact query cannot be crowded out that way; the fuzzy one
/// only fills what is left.
///
/// The exact query is case-insensitive in the index, so it is filtered to the
/// case-sensitive name when `exact_only` asks for full-name matches.
fn index_navs(
    analysis: &Analysis,
    symbol_name: &str,
    limit: usize,
    exact_only: bool,
) -> Result<Vec<NavigationTarget>> {
    let name = leaf_name(symbol_name);
    let mut exact_query = Query::new(symbol_name.to_string());
    exact_query.exact();
    let mut navs: Vec<NavigationTarget> = analysis
        .symbol_search(exact_query, limit)
        .context("symbol_search query failed")?
        .into_iter()
        .filter(|nav| !exact_only || nav.name.as_str() == name)
        .collect();
    navs.sort_by_key(|nav| nav.name.as_str() != name);

    if !exact_only && navs.len() < limit {
        let mut seen: HashSet<_> = navs.iter().map(nav_key).collect();
        let fuzzy = analysis
            .symbol_search(Query::new(symbol_name.to_string()), limit)
            .context("symbol_search query failed")?;
        for nav in fuzzy {
            if navs.len() == limit {
                break;
            }
            if seen.insert(nav_key(&nav)) {
                navs.push(nav);
            }
        }
    }
    Ok(navs)
}

/// The associated items a path `Owner::item` names when `Owner` is a type or a
/// trait: `WellLogApp::new`, `Vec::new`, `Option::map`, `Iterator::map`,
/// `crate_a::Widget::new`, `std::vec::Vec::new`.
///
/// # Why the index cannot
///
/// rust-analyzer's symbol index reads every leading segment of a path query
/// as a crate or a module (`resolve_path_to_modules`): for `App::new` it
/// looks for a module `App`, finds none and answers nothing — which the tools
/// reported as "No references found", read as "dead code"
/// (`probe_index_hits_for_path_queries`, 2026-11-26).
///
/// # How, and what it costs
///
/// 1. The owner: the index's full-name matches for the owner path (a type
///    anywhere in the project's crates, a path through modules); if there are
///    none, the owner path resolved from each local crate's root module the
///    way a `use` would — that is where the prelude lives (`Vec`, `Option`)
///    and where a dependency's name resolves. A local crate's def map is
///    already built by the load; one resolution is a few hash lookups — on
///    ~2000 crates, milliseconds.
/// 2. The item: the owner's inherent impls (`Type::iterate_assoc_items` —
///    the defining crate's impls only, already in the database); for a trait,
///    its own items. Only when the inherent impls hold no such item (`T::default`
///    from `Default`, `T::map` from a trait's provided method) the trait impls
///    for the type are asked (`Impl::all_for_type`): that walks the trait
///    impls of every crate — memoised per crate by salsa, so the first such
///    query of a revision pays for the crates it touches and the next ones only
///    the walk.
///
/// No file is read and no occurrence is resolved. An owner that is neither a
/// type nor a trait (a module, a function) contributes nothing: a module path
/// is the index's job, which it does.
pub(crate) fn assoc_item_navs(db: &RootDatabase, symbol_name: &str) -> Vec<NavigationTarget> {
    use ra_ap_hir::{AssocItem, Crate, Impl, ItemInNs, ModuleDef, Name, Semantics};
    use ra_ap_ide::TryToNav as _;

    let path = symbol_name.strip_prefix("::").unwrap_or(symbol_name);
    let segments: Vec<&str> = path.split("::").collect();
    let [owner_path @ .., item] = segments.as_slice() else {
        return Vec::new();
    };
    if owner_path.is_empty() || item.is_empty() || owner_path.iter().any(|s| s.is_empty()) {
        return Vec::new();
    }
    let owner_leaf = owner_path[owner_path.len() - 1];

    // The owner, from the index first.
    let mut owners: Vec<ModuleDef> = Vec::new();
    {
        let mut query = Query::new(owner_path.join("::"));
        query.exact();
        for symbol in ra_ap_ide_db::symbol_index::world_symbols(db, query) {
            if symbol.name.as_str() == owner_leaf && !owners.contains(&symbol.def) {
                owners.push(symbol.def);
            }
        }
    }
    // Then as a path from each local crate's root: the prelude, dependencies.
    if owners.is_empty() {
        let names: Vec<Name> = owner_path.iter().map(|s| Name::new_root(s)).collect();
        for krate in Crate::all(db)
            .into_iter()
            .filter(|k| k.origin(db).is_local())
        {
            let root = krate.root_module(db);
            let Some(items) = root.resolve_mod_path(db, names.iter().cloned()) else {
                continue;
            };
            for found in items {
                if let ItemInNs::Types(def) = found
                    && !owners.contains(&def)
                {
                    owners.push(def);
                }
            }
        }
    }

    let named = |assoc: &AssocItem| assoc.name(db).is_some_and(|n| n.as_str() == *item);
    let mut found: Vec<AssocItem> = Vec::new();
    let push = |assoc: AssocItem, found: &mut Vec<AssocItem>| {
        if !found.contains(&assoc) {
            found.push(assoc);
        }
    };
    for owner in owners {
        let ty = match owner {
            ModuleDef::Trait(trait_) => {
                for assoc in trait_.items(db).into_iter().filter(&named) {
                    push(assoc, &mut found);
                }
                continue;
            }
            ModuleDef::Adt(adt) => adt.ty(db),
            ModuleDef::TypeAlias(alias) => alias.ty(db),
            ModuleDef::BuiltinType(builtin) => builtin.ty(db),
            _ => continue,
        };
        let before = found.len();
        ty.iterate_assoc_items(db, |assoc| {
            if named(&assoc) {
                push(assoc, &mut found);
            }
            None::<()>
        });
        if found.len() > before {
            continue;
        }
        // Not inherent: a trait's, implemented for the type — the impl's own
        // item, or the trait's provided one the impl does not override.
        for imp in Impl::all_for_type(db, ty) {
            let Some(trait_) = imp.trait_(db) else {
                continue;
            };
            let own: Vec<AssocItem> = imp.items(db).into_iter().filter(&named).collect();
            if own.is_empty() {
                for assoc in trait_.items(db).into_iter().filter(&named) {
                    push(assoc, &mut found);
                }
            } else {
                for assoc in own {
                    push(assoc, &mut found);
                }
            }
        }
    }

    let sema = Semantics::new(db);
    found
        .into_iter()
        .filter_map(|assoc| assoc.try_to_nav(&sema))
        .map(|nav| nav.call_site())
        .collect()
}

/// The index's answer with the sweep deliberately not run — the mutant that
/// tells which kinds the index carries, kept in the source rather than applied
/// by hand to a working tree each time the question comes up.
#[cfg(test)]
pub(crate) fn index_only_search(
    analysis: &Analysis,
    vfs: &Vfs,
    symbol_name: &str,
    limit: usize,
) -> Result<Vec<Location>> {
    let matches = index_matches(&analysis, vfs, symbol_name, limit, true)?;
    Ok(rank_and_filter_exact(matches, true))
}

/// Collapse repeats of one position, keeping the first — which, after ranking,
/// is the exact match.
///
/// The sweep can arrive at a single declaration several times over. Its skip
/// list is built from the use sites `find_all_refs` reports, and a `Fast`
/// context has no dependency edges, so those stop at the crate boundary: an
/// occurrence in a consuming crate is not on the skip list, gets its own query,
/// and resolves to the same declaration again. Measured on a 4000-file
/// workspace — one field, the same file:line three times over.
fn dedup_by_position(locations: &mut Vec<Location>) {
    let mut seen = HashSet::new();
    locations.retain(|location| seen.insert(location_key(location)));
}

/// Find all references to symbols matching a name, optionally retaining only
/// full-name symbol matches before resolving references. No time budget: see
/// [`find_references_within`].
pub(crate) fn find_references_by_name_with_exact(
    analysis: &Analysis,
    db: &RootDatabase,
    vfs: &Vfs,
    words: &WordIndex,
    uses: &UseTable,
    symbol_name: &str,
    exact_only: bool,
) -> Result<Vec<Location>> {
    let answer = find_references_within(
        analysis,
        db,
        vfs,
        words,
        uses,
        symbol_name,
        exact_only,
        SweepBudget::unlimited(),
    )?;
    Ok(answer.locations)
}

/// [`find_references_by_name_with_exact`] whose file sweeps stop at `budget`;
/// a stopped sweep is named in [`SweptAnswer::incomplete`].
pub(crate) fn find_references_within(
    analysis: &Analysis,
    db: &RootDatabase,
    vfs: &Vfs,
    words: &WordIndex,
    uses: &UseTable,
    symbol_name: &str,
    exact_only: bool,
    budget: SweepBudget,
) -> Result<SweptAnswer> {
    let name = leaf_name(symbol_name);
    // First, find all symbols matching the name, full-name matches first.
    let mut symbols = index_navs(analysis, symbol_name, 50, exact_only)?;
    // `Type::method` / `Trait::method`: the index reads leading segments as
    // modules and finds nothing; the owner's associated items are the answer
    // (`assoc_item_navs`). Each is named `name`, so the loop below takes it as
    // a full-name match.
    if !symbols.iter().any(|symbol| symbol.name.as_str() == name) {
        symbols.extend(assoc_item_navs(db, symbol_name));
    }

    let mut all_locations = Vec::new();
    let mut resolved_the_name = false;
    let mut library_symbols = Vec::new();

    // For each symbol found, find its references. One `find_all_refs` per
    // match, up to 50 matches of a fuzzy query: bounded by the budget too.
    let mut incomplete = None;
    for (done, symbol) in symbols.iter().enumerate() {
        if budget.exhausted() {
            incomplete = Some(budget.note_of(
                "the per-match search",
                done,
                symbols.len(),
                "index matches resolved",
            ));
            break;
        }
        let symbol_exact = symbol.name.as_str() == name;
        if exact_only && !symbol_exact {
            continue;
        }
        resolved_the_name |= symbol_exact;

        if analysis
            .is_library_file(symbol.file_id)
            .context("is_library_file query failed")?
        {
            // Only a full-name match is asked about: a fuzzy library hit
            // (`add_spawn_hook` for `std::thread::spawn`) is not what
            // was asked, and sweeping the project for its name answered with
            // foreign call sites — slowly, as every occurrence of the name is
            // resolved (see `SweepBudget`).
            if symbol_exact {
                library_symbols.push((symbol, symbol_exact));
            }
            continue;
        }

        let position = FilePosition {
            file_id: symbol.file_id,
            offset: nav_offset(symbol),
        };
        if let Some((_, harvest)) = harvest_refs_at(&analysis, vfs, position, symbol_exact)? {
            all_locations.extend(harvest.into_all());
        }
    }

    let mut cost = SweepCost::default();
    if incomplete.is_none() {
        incomplete = collect_library_refs(
            analysis,
            vfs,
            words,
            uses,
            &library_symbols,
            budget,
            &mut cost,
            &mut all_locations,
        )?;
    }

    // The symbol index carries module-scope declarations plus impl and trait
    // members. It does NOT carry fields — a struct's, a union's, or a
    // struct-shaped variant's — so a field name matches nothing above and the
    // answer would be an empty list, which reads exactly like "nobody touches
    // this field". Ask the source instead. (Enum variants were expected to
    // share this fate and measurably do not; see `symbol_search_with_exact`.)
    // A name the index did resolve may still be a field's too: see
    // `Occurrences`.
    if incomplete.is_none() && !is_path_query(symbol_name) {
        let scope = if resolved_the_name {
            Occurrences::FieldShape
        } else {
            Occurrences::AllCode
        };
        incomplete = collect_by_token_scan(
            &analysis,
            vfs,
            words,
            name,
            Wants::DeclAndRefs,
            scope,
            budget,
            &mut cost,
            &mut all_locations,
        )?;
    }

    // Deduplicate locations (same file:line:col)
    all_locations.sort_by(|a, b| {
        (&a.file_path, a.line, a.column, !a.exact).cmp(&(&b.file_path, b.line, b.column, !b.exact))
    });
    all_locations
        .dedup_by(|a, b| a.file_path == b.file_path && a.line == b.line && a.column == b.column);
    all_locations.sort_by_key(|location| !location.exact);

    Ok(SweptAnswer {
        locations: all_locations,
        incomplete,
        cost,
    })
}

/// Which occurrences of the name the token sweep probes.
///
/// # Why a field beside an indexed name needs its own pass
///
/// The sweep used to run only when the symbol index resolved nothing of the
/// name. A field that shares its name with an indexed item — `rop` with a
/// `fn rop`, `lenght` with a method `lenght()`, very common for accessors —
/// was then never looked for: on a ~8000-file workspace both answered without
/// the field (2026-11-26), "fewer references than exist". Probing every
/// occurrence of an indexed name would ask rust-analyzer about every call of
/// it; a field's declaration always stands as `name:`, so that shape alone is
/// probed, in the files the word index has it in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Occurrences {
    /// Every occurrence in code: the index resolved nothing of the name.
    AllCode,
    /// Only `name:` — fields beside index matches of the same name.
    FieldShape,
}

/// What the caller wants out of each definition the sweep resolves.
///
/// The sweep itself is identical either way — the same query brings back the
/// declaration and its uses together — so the choice is only about what reaches
/// the answer. The uses are kept regardless, because they are what lets the
/// sweep skip the remaining occurrences of the name.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wants {
    /// Declaration and every use site — `find_references`.
    DeclAndRefs,
    /// The declaration alone — `find_definition`.
    DeclOnly,
}

/// Resolve `symbol_name` by finding the identifier in the project's own sources
/// and asking rust-analyzer what sits under it.
///
/// This is the path for names the symbol index does not carry — struct fields
/// and enum variants. It runs second because when the index *does* know a name
/// its answer is authoritative and costs one query instead of a file sweep.
///
/// # Why not every occurrence is queried
///
/// The first occurrence that resolves brings back the entire reference set for
/// that definition, which normally accounts for every other occurrence of the
/// name. Those are then skipped, so a field read twenty-one times costs one
/// query rather than twenty-one. What still costs a query each: an occurrence
/// belonging to a *different* definition (a same-named field on another
/// struct — both are wanted, so both are searched), and an occurrence in a
/// comment or string literal, which resolves to nothing.
///
/// # Scope
///
/// Only the project's own files are swept — the candidates the word index
/// ([`WordIndex`]) has for the name, i.e. the `.rs` files under the project
/// root whose code holds it. Under a full workspace load the VFS also holds
/// every dependency's sources, and a name that happens to exist in a registry
/// crate is not what the caller asked about.
///
/// # Cost
///
/// Reads and lexes the candidate files only — for a field, a handful of a
/// ~8000-file workspace, where this used to read all 148 MB — and asks
/// rust-analyzer only about occurrences in code, never about a comment or a
/// string (those used to be a query each, resolving to nothing).
#[allow(clippy::too_many_arguments)]
fn collect_by_token_scan(
    analysis: &Analysis,
    vfs: &Vfs,
    words: &WordIndex,
    symbol_name: &str,
    wants: Wants,
    scope: Occurrences,
    budget: SweepBudget,
    cost: &mut SweepCost,
    out: &mut Vec<Location>,
) -> Result<Option<String>> {
    // Positions already accounted for, so a definition reached through its
    // first occurrence is not re-queried through its remaining twenty.
    let mut covered: HashSet<(PathBuf, u32, u32)> = out.iter().map(location_key).collect();

    let files = match scope {
        Occurrences::AllCode => words.files_with(symbol_name),
        Occurrences::FieldShape => words.field_shape_files_with(symbol_name),
    };
    for (done, &file_id) in files.iter().enumerate() {
        if budget.exhausted() {
            return Ok(Some(budget.note("the token sweep", done, files.len())));
        }
        let text = analysis.file_text(file_id)?;
        let line_index = analysis.file_line_index(file_id)?;
        let file_path = file_path_of(vfs, file_id)?;
        cost.files_read += 1;

        let starts = match scope {
            Occurrences::AllCode => code_occurrences(&text, symbol_name),
            Occurrences::FieldShape => field_shape_occurrences(&text, symbol_name),
        };
        for start in starts {
            let line_col = line_index.line_col(TextSize::new(start as u32));
            let probe = (file_path.clone(), line_col.line + 1, line_col.col + 1);
            if !covered.insert(probe) {
                continue;
            }

            // One character INTO the identifier, not at its first byte:
            // rust-analyzer answers an empty list for a field probed at its
            // first character while answering in full one character later
            // (measured 2026-08-28). Single-character names have no inside.
            let inside = if symbol_name.len() > 1 {
                start + 1
            } else {
                start
            };
            let position = FilePosition {
                file_id,
                offset: TextSize::new(inside as u32),
            };

            cost.ra_queries += 1;
            let Some((resolved, harvest)) = harvest_refs_at(analysis, vfs, position, true)? else {
                continue;
            };
            if !is_the_wanted_name(&resolved, symbol_name) {
                // A comment, a string, a local binding spelled the same: the
                // occurrence looked right in text and is not this name.
                continue;
            }

            covered.extend(harvest.keys());
            match wants {
                Wants::DeclAndRefs => out.extend(harvest.into_all()),
                Wants::DeclOnly => out.extend(harvest.decls),
            }
        }
    }

    Ok(None)
}

/// The `.rs` files under `project_root` the analysis knows **in local source
/// roots**, sorted — what the word index and the table of uses are built over
/// (`build_word_index`, `UseTable::all_pending`); no query walks them.
///
/// # Why local roots only
///
/// A path under the project root is not enough: the `OUT_DIR` of a dependency
/// (`target/debug/build/<dep>/out/*.rs`, pulled in with `include!`) and a
/// vendored registry crate sit under it too, in a *library* root. They are not
/// the project's code — a reference found there is not one the project makes —
/// and they are the heaviest files there are: on a ~8000-file workspace 750 of
/// the 8834 files the analysis holds under the root were such, and the table's
/// builder spent 42–68 s on one each (`grammers-tl-types`
/// `generated_functions.rs`, `unicode-general-category` `category.rs`,
/// 2026-11-27). A local crate's own generated code is in a local root and stays.
///
/// Sorted, because the VFS hands files back in load order: two runs that probe
/// different occurrences first would otherwise answer in different order (the
/// index keeps its lists in the same `FileId` order).
///
/// Cost: one pass over the VFS plus two input reads per file under the root —
/// microseconds per file, once per load.
pub(super) fn project_rust_files(
    db: &ra_ap_ide::RootDatabase,
    vfs: &Vfs,
    project_root: &Path,
) -> Vec<ra_ap_vfs::FileId> {
    let mut files: Vec<ra_ap_vfs::FileId> = vfs
        .iter()
        .filter(|(file_id, path)| {
            path.as_path().is_some_and(|path| {
                let path: &Path = path.as_ref();
                path.extension().is_some_and(|ext| ext == "rs") && path.starts_with(project_root)
            }) && is_in_local_root(db, *file_id)
        })
        .map(|(file_id, _)| file_id)
        .collect();
    files.sort();
    files
}

/// Whether the file belongs to a local source root (the workspace's packages
/// and path dependencies), not to a library one (sysroot, registry, a
/// dependency's `OUT_DIR`). See [`project_rust_files`].
pub(super) fn is_in_local_root(db: &ra_ap_ide::RootDatabase, file_id: ra_ap_vfs::FileId) -> bool {
    use ra_ap_ide_db::base_db::SourceDatabase;
    let root = db.file_source_root(file_id).source_root_id(db);
    !db.source_root(root).source_root(db).is_library
}

/// References inside the project to declarations that live in a library — the
/// sysroot (`std::thread::spawn`) or a dependency — plus those declarations.
///
/// # Why not `find_all_refs` from the declaration
///
/// rust-analyzer scopes a search for a public item to every crate that depends
/// on the item's crate. For an item of `std` that is the entire crate graph: on
/// a workspace with ~2000 packages, every dependency's sources would be scanned
/// for the name and every candidate in them type-checked — in the daemon's
/// shared memory, for answers outside the project that are then thrown away.
///
/// So the question is turned around: each occurrence of the name in the
/// project's own files is asked where it goes (`goto_definition`), and kept if
/// it lands on the wanted declaration. The scope is the project by
/// construction, and the per-occurrence answer is exact — a method of the same
/// name on another type (`Builder::spawn` next to `thread::spawn`), a local
/// binding or a word in a comment goes elsewhere or nowhere.
///
/// A symbol the index returns can be a re-export (`pub use functions::spawn`
/// in `std::thread`): what is wanted is what it resolves to, so each symbol is
/// put through `goto_definition` too, and both it and its targets count as a
/// hit. The declarations reported are the resolved targets.
///
/// # Ready files come from the table of uses
///
/// A file the table of uses ([`UseTable`]) describes is not swept: its uses of
/// the wanted declarations are read from the table, which holds what this very
/// sweep would find there (`uses` module header). Only the *pending* files
/// among the candidates are swept live.
///
/// # Cost
///
/// With every file ready: the table's files for the wanted declarations, no
/// rust-analyzer query about any occurrence — the same for `new` as for
/// `spawn`. Per pending file among the word index's candidates for the names: a
/// `goto_definition` per occurrence of the name in code — cheap for `spawn`
/// (281 of ~8000 files), not for `new` (5194); the budget stops that while the
/// table is still being built.
#[allow(clippy::too_many_arguments)]
fn collect_library_refs(
    analysis: &Analysis,
    vfs: &Vfs,
    words: &WordIndex,
    uses: &UseTable,
    symbols: &[(&NavigationTarget, bool)],
    budget: SweepBudget,
    cost: &mut SweepCost,
    out: &mut Vec<Location>,
) -> Result<Option<String>> {
    if symbols.is_empty() {
        return Ok(None);
    }

    // Declaration key → whether a full-name match led to it.
    let mut wanted: HashMap<(ra_ap_vfs::FileId, TextSize), bool> = HashMap::new();
    let mut declarations: Vec<(NavigationTarget, bool)> = Vec::new();
    let mut names: BTreeSet<String> = BTreeSet::new();
    for &(symbol, exact) in symbols {
        names.insert(symbol.name.to_string());
        *wanted.entry(nav_key(symbol)).or_default() |= exact;
        let position = FilePosition {
            file_id: symbol.file_id,
            offset: nav_offset(symbol),
        };
        // `None` or empty: the symbol is itself the declaration.
        let targets = analysis
            .goto_definition(position, &goto_config())
            .context("goto_definition query failed")?
            .map(|info| info.info)
            .filter(|targets| !targets.is_empty())
            .unwrap_or_else(|| vec![symbol.clone()]);
        for target in targets {
            *wanted.entry(nav_key(&target)).or_default() |= exact;
            declarations.push((target, exact));
        }
    }

    // The final dedup of the caller keeps the first of a position after
    // ranking exact first, so a declaration reached twice is reported once.
    for (declaration, exact) in &declarations {
        let mut location = nav_target_to_location(vfs, analysis, declaration)?;
        location.exact = *exact;
        out.push(location);
    }

    // Ready files: the table's uses of each wanted declaration. A use reached
    // through two wanted keys (a re-export and its target) is one location
    // after the caller's dedup, exact if either was.
    let mut from_table: BTreeSet<ra_ap_vfs::FileId> = BTreeSet::new();
    for (&key, &exact) in &wanted {
        for (file_id, starts) in uses.uses(key) {
            let line_index = analysis.file_line_index(file_id)?;
            let file_path = file_path_of(vfs, file_id)?;
            from_table.insert(file_id);
            for start in starts {
                let line_col = line_index.line_col(start);
                out.push(Location {
                    file_path: file_path.clone(),
                    line: line_col.line + 1,
                    column: line_col.col + 1,
                    name: "reference".to_string(),
                    exact,
                });
            }
        }
    }
    cost.files_from_table += from_table.len();

    // Pending files holding one of the names, once, in `FileId` order: the
    // only ones swept.
    let files: BTreeSet<ra_ap_vfs::FileId> = names
        .iter()
        .flat_map(|name| words.files_with(name).iter().copied())
        .filter(|&file_id| uses.is_pending(file_id))
        .collect();
    for (done, &file_id) in files.iter().enumerate() {
        if budget.exhausted() {
            return Ok(Some(budget.note_of(
                "the library-item sweep",
                done,
                files.len(),
                &format!(
                    "candidate files searched (the table of uses still has {} of the project's files pending)",
                    uses.pending().len()
                ),
            )));
        }
        let text = analysis.file_text(file_id)?;
        let line_index = analysis.file_line_index(file_id)?;
        let file_path = file_path_of(vfs, file_id)?;
        cost.files_read += 1;
        for name in &names {
            for start in code_occurrences(&text, name) {
                // One character into the identifier, as in the token sweep.
                let inside = if name.len() > 1 { start + 1 } else { start };
                let position = FilePosition {
                    file_id,
                    offset: TextSize::new(inside as u32),
                };
                cost.ra_queries += 1;
                let Some(info) = analysis
                    .goto_definition(position, &goto_config())
                    .context("goto_definition query failed")?
                else {
                    continue;
                };
                let Some(&exact) = info.info.iter().find_map(|target| wanted.get(&nav_key(target)))
                else {
                    continue;
                };
                let line_col = line_index.line_col(TextSize::new(start as u32));
                out.push(Location {
                    file_path: file_path.clone(),
                    line: line_col.line + 1,
                    column: line_col.col + 1,
                    name: "reference".to_string(),
                    exact,
                });
            }
        }
    }
    Ok(None)
}

/// Whether a resolved definition is the thing the token sweep went looking for.
///
/// The name has to match, and so does the *kind*: the sweep exists for what
/// rust-analyzer's symbol index does not surface — every named **field**, be it
/// a struct's, a union's or a struct-shaped enum variant's; enum variants
/// themselves are a backstop, the index does carry them — so anything else
/// found under a matching
/// identifier is a different entity that happens to share a spelling. A local
/// `let radius_override = 7;` declares a name equal to the field's, and
/// counting it and its uses inflated a four-read field to six — silently, which
/// is the failure mode this whole path exists to end. As a *definition* the
/// same local is worse still: an answer pointing into an unrelated function
/// body, which reads as fact.
fn is_the_wanted_name(decl: &ResolvedDecl, symbol_name: &str) -> bool {
    decl.name == symbol_name && matches!(decl.kind, Some(SymbolKind::Field | SymbolKind::Variant))
}

/// Dedup key: one source position, however it was reached.
fn location_key(location: &Location) -> (PathBuf, u32, u32) {
    (location.file_path.clone(), location.line, location.column)
}

fn rank_and_filter_exact(mut locations: Vec<Location>, exact_only: bool) -> Vec<Location> {
    if exact_only {
        locations.retain(|location| location.exact);
    }
    locations.sort_by_key(|location| !location.exact);
    locations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(name: &str, exact: bool, line: u32) -> Location {
        Location {
            file_path: PathBuf::from(format!("{name}.rs")),
            line,
            column: 1,
            name: name.to_string(),
            exact,
        }
    }

    #[test]
    fn rank_and_filter_exact_keeps_substrings_by_default_but_ranks_exact_first() {
        let locations = vec![
            loc("VectorSearchResult", false, 2),
            loc("SearchResult", true, 1),
        ];
        let ranked = rank_and_filter_exact(locations, false);

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].name, "SearchResult");
        assert!(ranked[0].exact);
        assert_eq!(ranked[1].name, "VectorSearchResult");
        assert!(!ranked[1].exact);
    }

    /// One declaration reached through several text occurrences is still one
    /// declaration. Without this the sweep answered a single field with three
    /// identical lines on a real workspace.
    #[test]
    fn dedup_by_position_collapses_one_declaration_reached_twice() {
        let mut locations = vec![
            loc("radius_override", true, 280),
            loc("radius_override", true, 280),
            loc("radius_override", true, 60),
        ];
        dedup_by_position(&mut locations);

        assert_eq!(locations.len(), 2, "got {locations:?}");
        assert_eq!(locations[0].line, 280);
        assert_eq!(locations[1].line, 60);
    }

    #[test]
    fn rank_and_filter_exact_can_drop_substring_matches() {
        let locations = vec![
            loc("VectorSearchResult", false, 2),
            loc("SearchResult", true, 1),
        ];
        let ranked = rank_and_filter_exact(locations, true);

        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].name, "SearchResult");
        assert!(ranked[0].exact);
    }
}
