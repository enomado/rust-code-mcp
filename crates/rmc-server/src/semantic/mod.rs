//! Semantic code analysis using rust-analyzer

mod loader;
/// Pricing an LRU capacity at both ends — bytes handed back against the time the
/// next query pays for them. Measurement only, so it ships with the tests.
#[cfg(test)]
mod lru_sweep;
mod position;
mod rename;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use ra_ap_ide::AnalysisHost;
use ra_ap_ide_db::ChangeWithProcMacros;
use ra_ap_vfs::{FileExcluded, Vfs, VfsPath};
use anyhow::{Context as _, Result};
use rmc_indexing::indexing::collect_project_rust_files;
use serde::Serialize;

pub(crate) use position::Location;
pub(crate) use rename::RenamePreview;

/// What a `stat` can see about a file — enough to decide whether it is worth
/// reading, and cheap enough to collect for a whole workspace on every query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    modified: SystemTime,
    len: u64,
}

/// What the working tree did since a project context was built.
#[derive(Debug)]
enum Staleness {
    /// Same files, same stamps: the cached analysis still describes the code.
    Fresh,
    /// The same set of files, some of them edited. Their new text can be
    /// pushed into the existing database.
    Edited(Vec<PathBuf>),
    /// Files appeared or disappeared. A new module, a new crate or a deleted
    /// file changes the module tree and possibly the crate graph, and neither
    /// can be patched file-by-file — the project has to be loaded again.
    StructureChanged,
}

/// Cached project context
struct ProjectContext {
    host: AnalysisHost,
    vfs: Vfs,
    load_kind: LoadKind,
    /// The working tree as it was when this context was built or last
    /// refreshed. Without it the cache has no way to notice that the code
    /// moved on — see [`SemanticService::refresh_if_stale`].
    stamps: HashMap<PathBuf, FileStamp>,
    /// Value of [`SemanticService::clock`] when this context was last asked a
    /// question. Orders eviction — see [`SemanticService::evict_to_capacity`].
    ///
    /// A counter rather than a `SystemTime`: eviction needs the *order* of
    /// uses, and a wall clock can step backwards (NTP, suspend) and reorder
    /// them.
    last_used: u64,
}

fn stamp_of(path: &Path) -> Option<FileStamp> {
    let meta = std::fs::metadata(path).ok()?;
    Some(FileStamp {
        modified: meta.modified().ok()?,
        len: meta.len(),
    })
}

/// Stat every `*.rs` file of the project, using the same walker as the indexer
/// so both sides mean the same thing by "a project file".
///
/// Files that vanish between the walk and the stat are simply absent from the
/// result, which reads as a deletion — the correct conclusion.
fn collect_stamps(root: &Path) -> HashMap<PathBuf, FileStamp> {
    let (files, walk_errors) = collect_project_rust_files(root);
    if walk_errors > 0 {
        tracing::warn!(
            "{} unreadable entries while checking {} for staleness",
            walk_errors,
            root.display()
        );
    }
    files
        .into_iter()
        .filter_map(|path| stamp_of(&path).map(|stamp| (path, stamp)))
        .collect()
}

fn classify_staleness(
    old: &HashMap<PathBuf, FileStamp>,
    new: &HashMap<PathBuf, FileStamp>,
) -> Staleness {
    if old.len() != new.len() {
        return Staleness::StructureChanged;
    }

    let mut edited = Vec::new();
    for (path, stamp) in new {
        match old.get(path) {
            None => return Staleness::StructureChanged,
            Some(previous) if previous == stamp => {}
            Some(_) => edited.push(path.clone()),
        }
    }

    if edited.is_empty() {
        Staleness::Fresh
    } else {
        edited.sort();
        Staleness::Edited(edited)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoadKind {
    Fast,
    Full,
}

impl LoadKind {
    fn as_str(self) -> &'static str {
        match self {
            LoadKind::Fast => "fast",
            LoadKind::Full => "full",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticProjectStatus {
    pub path: String,
    pub load_kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticServiceStatus {
    pub project_count: usize,
    pub projects: Vec<SemanticProjectStatus>,
}

/// One salsa ingredient's line in a memory report — a query's memo table or an
/// interned/tracked struct's storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IngredientMemory {
    /// `query` or `struct`.
    pub family: &'static str,
    /// For a query this is the *query* name (`<SelfTy>::<fn>`), which is what
    /// an LRU capacity is addressed by. salsa keeps it in the map key and puts
    /// the result *type* in `debug_name`, so reading the values alone loses the
    /// only label that can be acted on.
    pub name: &'static str,
    /// The result type, beside a query name: two queries can return the same
    /// shape, and the shape is what hints at the heap behind it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub produces: Option<&'static str>,
    pub count: usize,
    pub stack_bytes: usize,
    pub metadata_bytes: usize,
    /// Present only where the query declares a `heap_size` function, and a
    /// lower bound where it does. Absent means "not measured", never "empty".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heap_bytes: Option<usize>,
    /// Page fill statistics, for the interned families that keep their slots in
    /// pages. Says how much of the reserved space is actually in use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<String>,
}

impl IngredientMemory {
    /// What this ingredient is known to cost. Unmeasured heap counts as zero
    /// rather than being guessed at, which is why a report prints this against
    /// RSS instead of presenting it as the whole.
    fn bytes(&self) -> usize {
        self.stack_bytes + self.metadata_bytes + self.heap_bytes.unwrap_or(0)
    }
}

/// What one loaded analysis is holding, ingredient by ingredient.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProjectMemoryReport {
    pub project: String,
    /// Resident set of the whole process — the daemon, not this project alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_kib: Option<u64>,
    pub ingredients: usize,
    /// How many of them report a heap size at all. The rest are counted, not
    /// weighed.
    pub ingredients_reporting_heap: usize,
    pub instances: usize,
    pub accounted_bytes: usize,
    /// Sorted by known bytes, largest first; truncated to the caller's `top`.
    pub rows: Vec<IngredientMemory>,
}

/// Flatten salsa's two families — interned/tracked structs and query memo
/// tables — into one list, sorted by what it can see.
fn ingredient_rows(host: &AnalysisHost) -> Vec<IngredientMemory> {
    use ra_ap_ide_db::base_db::salsa;

    // `memory_usage` is implemented on `dyn Database`, not on the trait, so the
    // unsize coercion here is required rather than stylistic.
    let db: &dyn salsa::Database = host.raw_database();
    let info = db.memory_usage();

    let mut rows: Vec<IngredientMemory> = info
        .structs
        .iter()
        .map(|ingredient| IngredientMemory {
            family: "struct",
            name: ingredient.debug_name(),
            produces: None,
            count: ingredient.count(),
            stack_bytes: ingredient.size_of_fields(),
            metadata_bytes: ingredient.size_of_metadata(),
            heap_bytes: ingredient.heap_size_of_fields(),
            page: ingredient.page_info().map(|page| format!("{page:?}")),
        })
        .chain(info.queries.iter().map(|(query, ingredient)| IngredientMemory {
            family: "query",
            name: query,
            produces: Some(ingredient.debug_name()),
            count: ingredient.count(),
            stack_bytes: ingredient.size_of_fields(),
            metadata_bytes: ingredient.size_of_metadata(),
            heap_bytes: ingredient.heap_size_of_fields(),
            page: None,
        }))
        .collect();
    rows.sort_by(|a, b| b.bytes().cmp(&a.bytes()).then(b.count.cmp(&a.count)));
    rows
}

/// Push the current contents of `paths` into an already-loaded analysis.
///
/// Fails rather than skips when a file cannot be mapped into the analysis: the
/// caller turns that into a full reload. Silently leaving one file at its old
/// revision would reproduce the very defect this module now guards against,
/// only harder to notice.
fn apply_edits(ctx: &mut ProjectContext, paths: &[PathBuf]) -> Result<()> {
    let mut updates = Vec::with_capacity(paths.len());

    // Read everything before mutating anything: a change applied halfway
    // would leave the database describing a mixture of two revisions.
    for path in paths {
        let vfs_path = VfsPath::new_real_path(path.to_string_lossy().into_owned());
        let (file_id, excluded) = ctx.vfs.file_id(&vfs_path).ok_or_else(|| {
            anyhow::anyhow!("{} is not part of the loaded analysis", path.display())
        })?;
        if excluded == FileExcluded::Yes {
            anyhow::bail!("{} is excluded from the loaded analysis", path.display());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        updates.push((vfs_path, file_id, text));
    }

    let mut change = ChangeWithProcMacros::default();
    for (vfs_path, file_id, text) in updates {
        ctx.vfs
            .set_file_contents(vfs_path, Some(text.clone().into_bytes()));
        change.change_file(file_id, Some(text));
    }
    ctx.host.apply_change(change);

    Ok(())
}

/// How many project contexts may stay loaded at once. `0` means unlimited.
///
/// Derived rather than guessed. A freshly started daemon costs ~2.3 GB before
/// it loads anything (ONNX runtime, embedding model, GPU probe); one
/// `Fast`-loaded workspace of ~4000 files adds ~3 GB. Three of them come to
/// ~11.3 GB, which is what the watchdog's soft limit is set to accommodate
/// (12288 MB); a fourth is memory that guard has already decided against.
///
/// It was 2 while the daemon key included the working directory, and 2 was the
/// right number then: each daemon served one directory, so a second project was
/// already unusual. One daemon now serves every directory, so the projects a
/// person moves between during a day all land in the same map — and evicting
/// one costs the next question in that project a full reload, measured in
/// minutes on a large workspace.
///
/// What this bounds is a *count*, which is only a proxy for memory — ten tiny
/// crates cost less than one workspace. The proxy is deliberate: it is the
/// cheap half of the job, and the watchdog remains the half that measures
/// actual RSS.
const DEFAULT_MAX_PROJECTS: usize = 3;

const MAX_PROJECTS_ENV: &str = "RMC_MAX_PROJECTS";

fn max_projects_from_env() -> usize {
    std::env::var(MAX_PROJECTS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_PROJECTS)
}

/// Service for semantic code queries
pub(crate) struct SemanticService {
    projects: HashMap<PathBuf, ProjectContext>,
    /// Upper bound on `projects`; `0` disables eviction entirely.
    ///
    /// # The defect this closes
    ///
    /// Before it, the map had no TTL, no LRU and no ceiling: every directory a
    /// session ever asked about kept its rust-analyzer context until the
    /// process died. A daemon outlives its clients by design, so nothing
    /// bounded that — measured 2026-08-27, one daemon sat on 14.6 GB, most of
    /// it contexts for directories nobody had touched in hours.
    max_projects: usize,
    /// Monotonic tick, handed out by [`Self::touch`]. Only its order matters.
    clock: u64,
}

impl SemanticService {
    pub(crate) fn new() -> Self {
        Self::with_max_projects(max_projects_from_env())
    }

    /// The cap as an argument rather than as an environment variable, so a test
    /// can state the capacity it is judging instead of arranging the process
    /// environment around it.
    pub(crate) fn with_max_projects(max_projects: usize) -> Self {
        Self {
            projects: HashMap::new(),
            max_projects,
            clock: 0,
        }
    }

    pub(crate) fn project_count(&self) -> usize {
        self.projects.len()
    }

    pub(crate) fn status(&self) -> SemanticServiceStatus {
        let mut projects = self
            .projects
            .iter()
            .map(|(path, ctx)| SemanticProjectStatus {
                path: path.display().to_string(),
                load_kind: ctx.load_kind.as_str().to_string(),
            })
            .collect::<Vec<_>>();
        projects.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then_with(|| a.load_kind.cmp(&b.load_kind))
        });
        SemanticServiceStatus {
            project_count: projects.len(),
            projects,
        }
    }

    pub(crate) fn clear_all(&mut self) -> usize {
        let count = self.projects.len();
        self.projects.clear();
        count
    }

    pub(crate) fn clear_project(&mut self, project_path: &Path) -> usize {
        let canonical =
            std::fs::canonicalize(project_path).unwrap_or_else(|_| project_path.to_path_buf());
        if self.projects.remove(&canonical).is_some() {
            1
        } else {
            0
        }
    }

    /// Give every loaded analysis a revision bump and sweep the type interner.
    ///
    /// # Why a daemon has to do this by hand
    ///
    /// salsa evicts an LRU-capped query's memos in exactly one place:
    /// `for_each_evicted`, reached from `reset_for_new_revision`. That is to
    /// say, **capacities only mean something at the moment the revision bumps**,
    /// and a revision bumps when a file changes. An IDE gets those for free on
    /// every keystroke; a project sitting in this daemon that nobody has edited
    /// for hours never bumps at all, so its capacity — whatever it is set to —
    /// evicts nothing, ever. Those are precisely the cold contexts that made a
    /// daemon 14.6 GB. This call is where the bump comes from instead.
    ///
    /// `AnalysisHost::trigger_garbage_collection` is a synthetic write plus a
    /// mark-and-sweep over the interned type storage, and rust-analyzer's own
    /// main loop calls it the same way when its worker pools go quiet.
    ///
    /// # Why calling it with several hosts loaded is sound
    ///
    /// The sweep is process-global — the type interner is shared by every
    /// `AnalysisHost` here — but its roots are refcounts (`Arc::strong_count(item) > 1`
    /// marks alive), so a type another host holds is a root and survives. What
    /// the `unsafe` really demands is that no query be *in flight*, holding a
    /// type it has not recorded anywhere yet. Every path into this service goes
    /// through one `Mutex<SemanticService>`, and this method needs `&mut self`,
    /// so holding that lock means no analysis is running *in this service*.
    ///
    /// That is not the whole process, though — the graph tools, the audits and
    /// the skeleton builder each load a rust-analyzer workspace outside this
    /// service. What covers them is the gate in [`crate::deep_stack`], which
    /// the caller enters exclusively for this call; see
    /// `RuntimeState::collect_garbage`.
    ///
    /// # Cost
    ///
    /// The sweep runs once per host, so N loaded projects pay N sweeps; the
    /// caller keeps this off the hot path (see the watchdog's interval) rather
    /// than the method trying to be clever about it. The bump itself is paid
    /// later, by the next query re-validating its memos.
    ///
    /// Returns the number of contexts collected, for the log line.
    pub(crate) fn collect_garbage(&mut self) -> usize {
        for (path, ctx) in self.projects.iter_mut() {
            tracing::debug!("collecting garbage in {}", path.display());
            ctx.host.trigger_garbage_collection();
        }
        self.projects.len()
    }

    /// Per-ingredient memory of one *already loaded* analysis.
    ///
    /// # Why it refuses to load the project
    ///
    /// The question this answers is "what is this daemon holding", and a daemon
    /// worth asking about has been alive for hours. Loading the project here to
    /// answer would build a fresh database and report *that* — a floor, not a
    /// portrait, and one indistinguishable from the real thing in the output.
    /// So an unloaded path is an error naming what is loaded instead.
    ///
    /// # How to read the numbers
    ///
    /// `heap_bytes` is present only for the queries whose fork declares a
    /// `heap_size` function, and it is a lower bound there (see
    /// `syntax::heap_size`). Everything else reports `count`, `stack_bytes` and
    /// `metadata_bytes` only — enough to say which memos are numerous, never
    /// how many bytes they are worth. `accounted_bytes` against RSS is printed
    /// precisely so the gap cannot be mistaken for a full budget.
    pub(crate) fn memory_breakdown(&self, project: &Path, top: usize) -> Result<ProjectMemoryReport> {
        let canonical = project
            .canonicalize()
            .with_context(|| format!("resolving {}", project.display()))?;
        let ctx = self.projects.get(&canonical).ok_or_else(|| {
            let loaded: Vec<String> = self
                .projects
                .keys()
                .map(|path| path.display().to_string())
                .collect();
            anyhow::anyhow!(
                "no analysis is loaded for {}; this reports what a live daemon is holding and \
                 will not load a project to answer, because a database built for the question \
                 measures a cold start rather than the daemon. Loaded: {}",
                canonical.display(),
                if loaded.is_empty() { "nothing".to_owned() } else { loaded.join(", ") }
            )
        })?;

        let mut rows = ingredient_rows(&ctx.host);
        let instances = rows.iter().map(|row| row.count).sum();
        let accounted_bytes = rows.iter().map(IngredientMemory::bytes).sum();
        let ingredients = rows.len();
        let ingredients_reporting_heap = rows.iter().filter(|row| row.heap_bytes.is_some()).count();
        if top > 0 {
            rows.truncate(top);
        }

        Ok(ProjectMemoryReport {
            project: canonical.display().to_string(),
            rss_kib: crate::mcp::memory::rss_kib(),
            ingredients,
            ingredients_reporting_heap,
            instances,
            accounted_bytes,
            rows,
        })
    }

    /// Mark `canonical` as the most recently used context.
    fn touch(&mut self, canonical: &Path) {
        self.clock += 1;
        let clock = self.clock;
        if let Some(ctx) = self.projects.get_mut(canonical) {
            ctx.last_used = clock;
        }
    }

    /// Drop the least recently used contexts until [`Self::max_projects`] holds.
    ///
    /// `keep` is never evicted whatever its age: it is the project the caller is
    /// about to answer a question about, and evicting it would mean reloading it
    /// on the very next line. With a cap of 1 that is the whole rule — the
    /// working project stays, everything else goes.
    ///
    /// Returns how many contexts were dropped.
    fn evict_to_capacity(&mut self, keep: &Path) -> usize {
        if self.max_projects == 0 {
            return 0;
        }

        let mut evicted = 0;
        while self.projects.len() > self.max_projects {
            let victim = self
                .projects
                .iter()
                .filter(|(path, _)| path.as_path() != keep)
                .min_by_key(|(_, ctx)| ctx.last_used)
                .map(|(path, _)| path.clone());

            // Only `keep` is left: the cap is smaller than one project, and one
            // is the least we can hold and still answer.
            let Some(victim) = victim else { break };

            tracing::info!(
                "unloading {} to stay within {}={} loaded project(s)",
                victim.display(),
                MAX_PROJECTS_ENV,
                self.max_projects
            );
            self.projects.remove(&victim);
            evicted += 1;
        }
        evicted
    }

    /// Get or load project with full workspace dependency edges.
    ///
    /// The only entry point left: `find_definition` was the last served path on
    /// a `Fast` context and moved off it, because "which context am I on"
    /// decided whether it could see `#[cfg(test)]` and that depended on call
    /// order. `LoadKind::Fast` stays for the upgrade machinery it is the source
    /// half of — nothing requests it any more.
    fn get_or_load_full(&mut self, project_path: &Path) -> Result<()> {
        self.get_or_load_kind(project_path, LoadKind::Full)
    }

    fn get_or_load_kind(&mut self, project_path: &Path, requested: LoadKind) -> Result<()> {
        let canonical = project_path.canonicalize()?;

        let needs_load = match self.projects.get(&canonical) {
            Some(ctx) => requested == LoadKind::Full && ctx.load_kind == LoadKind::Fast,
            None => true,
        };

        let outcome = if needs_load {
            self.load_kind_into_cache(&canonical, requested)
        } else {
            // Cached — but "cached" said nothing about "current" until this call
            // existed. See [`Self::refresh_if_stale`].
            self.refresh_if_stale(&canonical, requested)
        };
        outcome?;

        // Both branches, and only after they succeeded: a load that failed left
        // no context to order, and a query that never happened is not a use.
        self.touch(&canonical);
        self.evict_to_capacity(&canonical);
        Ok(())
    }

    fn load_kind_into_cache(&mut self, canonical: &Path, requested: LoadKind) -> Result<()> {
        tracing::info!(
            "Loading {:?} IDE for project: {}",
            requested,
            canonical.display()
        );
        // Stamps are taken BEFORE the load, not after: a file edited while
        // rust-analyzer was reading the workspace must come out stale on the
        // next query, not silently accepted as already analysed.
        let stamps = collect_stamps(canonical);
        let (host, vfs) = match requested {
            LoadKind::Fast => loader::load_project(canonical)?,
            LoadKind::Full => loader::load_project_full(canonical)?,
        };
        self.clock += 1;
        self.projects.insert(
            canonical.to_path_buf(),
            ProjectContext {
                host,
                vfs,
                load_kind: requested,
                stamps,
                last_used: self.clock,
            },
        );
        tracing::info!("IDE loaded successfully");
        Ok(())
    }

    /// Bring a cached project up to date with the working tree.
    ///
    /// # The defect this closes
    ///
    /// A `ProjectContext` is an `AnalysisHost` built from the files as they
    /// were at load time. Nothing ever invalidated it: the only reload
    /// condition was upgrading a `Fast` context to `Full`. So every answer
    /// after the first edit described code that no longer existed —
    /// `find_references` reporting 2 call sites where the file on disk had 7,
    /// with no error and no warning anywhere. The only cure was knowing to
    /// call `clear_runtime scope=semantic_only`, which requires already
    /// suspecting the answer, which is exactly what a confident wrong answer
    /// prevents.
    ///
    /// # Why two repair paths
    ///
    /// Edits to existing files are pushed straight into the salsa database
    /// (`apply_change`), which invalidates precisely the derived queries that
    /// depended on those files — milliseconds, and it is what a real IDE does
    /// on every keystroke. Added or deleted files are not patchable that way:
    /// they change the module tree, and a new crate changes the crate graph,
    /// so those force a reload. Reloading costs seconds (`Fast`, `no_deps`) to
    /// minutes (`Full`), which is why it is reserved for the case that needs
    /// it rather than used for everything.
    ///
    /// # Known limit
    ///
    /// Only `*.rs` files are watched. A `Cargo.toml` edited on its own —
    /// dropping a dependency, renaming a feature — is invisible here. Adding a
    /// crate is not: its sources are new files, which trips the structural
    /// path anyway.
    fn refresh_if_stale(&mut self, canonical: &Path, requested: LoadKind) -> Result<()> {
        let current = collect_stamps(canonical);

        let staleness = {
            let ctx = self
                .projects
                .get(canonical)
                .ok_or_else(|| anyhow::anyhow!("Project not loaded"))?;
            classify_staleness(&ctx.stamps, &current)
        };

        match staleness {
            Staleness::Fresh => Ok(()),
            Staleness::StructureChanged => {
                tracing::info!(
                    "Project {} gained or lost files since it was analysed; reloading",
                    canonical.display()
                );
                self.load_kind_into_cache(canonical, requested)
            }
            Staleness::Edited(paths) => {
                let patched = {
                    let ctx = self
                        .projects
                        .get_mut(canonical)
                        .ok_or_else(|| anyhow::anyhow!("Project not loaded"))?;
                    apply_edits(ctx, &paths).map(|()| ctx.stamps = current)
                };
                match patched {
                    Ok(()) => {
                        tracing::info!(
                            "Refreshed {} edited file(s) in {}",
                            paths.len(),
                            canonical.display()
                        );
                        Ok(())
                    }
                    Err(e) => {
                        // A file the analysis does not know (excluded from the
                        // build, unreadable, non-UTF-8) cannot be patched in.
                        // Falling back to a reload is slow but honest; keeping
                        // the stale context would be the original defect.
                        tracing::info!(
                            "Cannot patch {} incrementally ({e}); reloading",
                            canonical.display()
                        );
                        self.load_kind_into_cache(canonical, requested)
                    }
                }
            }
        }
    }

    /// Search for symbols by name (for find_definition)
    pub(crate) fn symbol_search(
        &mut self,
        project_path: &Path,
        symbol_name: &str,
        limit: usize,
    ) -> Result<Vec<Location>> {
        self.symbol_search_with_exact(project_path, symbol_name, limit, false)
    }

    /// Search for symbols by name with optional full-name filtering.
    ///
    /// # Why this needs the same `Full` context references do
    ///
    /// A declaration lives in exactly one crate, so this one never needed the
    /// dependency edges — and it ran on `Fast` for that reason. What that
    /// overlooked is that `Fast` also has `set_test` off, which takes every
    /// `#[cfg(test)]` module out of the tree: "where is this test declared"
    /// answered *no definition found* for a test that exists.
    ///
    /// And it did so **only sometimes**. `get_or_load` does not downgrade a
    /// context already loaded as `Full`, so after any `find_references` call in
    /// the same session this one silently got the full tree and started finding
    /// what it had just denied. Measured 2026-08-28 on a 4000-file workspace:
    /// the same query, the same session, answered "no definition found" and then
    /// the file and line. An answer that depends on which questions came before
    /// it is worse than a blind spot — a blind spot at least reproduces.
    ///
    /// The load is shared, not doubled: `find_references` pays for the same
    /// upgrade, and whichever runs first pays it once.
    pub(crate) fn symbol_search_with_exact(
        &mut self,
        project_path: &Path,
        symbol_name: &str,
        limit: usize,
        exact: bool,
    ) -> Result<Vec<Location>> {
        self.get_or_load_full(project_path)?;

        let canonical = project_path.canonicalize()?;
        let ctx = self.projects.get(&canonical)
            .ok_or_else(|| anyhow::anyhow!("Project not loaded"))?;

        position::symbol_search_with_exact(
            &ctx.host,
            &ctx.vfs,
            &canonical,
            symbol_name,
            limit,
            exact,
        )
    }

    /// The symbol index's own answer, with the source sweep not run.
    ///
    /// Only the survey of "which kinds does the index carry" uses this; the
    /// served path always keeps the sweep behind it. The context is the same
    /// `Full` one the served path loads, so the survey measures the index the
    /// tool actually queries.
    #[cfg(test)]
    pub(crate) fn index_only_search(
        &mut self,
        project_path: &Path,
        symbol_name: &str,
        limit: usize,
    ) -> Result<Vec<Location>> {
        self.get_or_load_full(project_path)?;

        let canonical = project_path.canonicalize()?;
        let ctx = self
            .projects
            .get(&canonical)
            .ok_or_else(|| anyhow::anyhow!("Project not loaded"))?;

        position::index_only_search(&ctx.host, &ctx.vfs, symbol_name, limit)
    }

    /// Find all references to symbols matching a name
    /// First finds all symbols matching the name, then finds references for each
    pub(crate) fn find_references_by_name(
        &mut self,
        project_path: &Path,
        symbol_name: &str,
    ) -> Result<Vec<Location>> {
        self.find_references_by_name_with_exact(project_path, symbol_name, false)
    }

    /// Find all references to symbols matching a name with optional exact filtering.
    ///
    /// # Why this needs a `Full` context
    ///
    /// A `Fast` context is loaded with `no_deps: true`, and `cargo metadata
    /// --no-deps` omits the resolve graph entirely — so rust-analyzer gets the
    /// workspace's packages with **no dependency edges between them**, and a
    /// call from one member crate into another is not a reference it can see.
    /// `set_test` is off there too, which takes every `#[cfg(test)]` call site
    /// out of the module tree along with it.
    ///
    /// Measured on a 4000-file workspace (2026-08-28): a `pub fn` with six call
    /// sites answered three — the cross-crate one and both test ones missing,
    /// with no indication that the answer was partial. "Fewer references than
    /// exist" is the one wrong answer this tool must not give, because it is
    /// read as "this code is dead".
    ///
    /// The cost lands once per project: `get_or_load_full` upgrades a cached
    /// `Fast` context in place rather than holding a second one, and `rename`
    /// already pays the same price for the same reason.
    pub(crate) fn find_references_by_name_with_exact(
        &mut self,
        project_path: &Path,
        symbol_name: &str,
        exact: bool,
    ) -> Result<Vec<Location>> {
        self.get_or_load_full(project_path)?;

        let canonical = project_path.canonicalize()?;
        let ctx = self.projects.get(&canonical)
            .ok_or_else(|| anyhow::anyhow!("Project not loaded"))?;

        position::find_references_by_name_with_exact(
            &ctx.host,
            &ctx.vfs,
            &canonical,
            symbol_name,
            exact,
        )
    }

    /// Preview rename of a symbol by name. Does not modify any files.
    pub(crate) fn rename_by_name(
        &mut self,
        project_path: &Path,
        symbol_name: &str,
        new_name: &str,
    ) -> Result<RenamePreview> {
        self.get_or_load_full(project_path)?;

        let canonical = project_path.canonicalize()?;
        let ctx = self.projects.get(&canonical)
            .ok_or_else(|| anyhow::anyhow!("Project not loaded"))?;

        rename::rename_by_name(&ctx.host, &ctx.vfs, symbol_name, new_name)
    }

    /// Preview rename of a symbol at a concrete file position. Does not modify any files.
    pub(crate) fn rename_by_position(
        &mut self,
        project_path: &Path,
        file_path: &Path,
        line: u32,
        column: u32,
        symbol_name: &str,
        new_name: &str,
    ) -> Result<RenamePreview> {
        self.get_or_load_full(project_path)?;

        let canonical = project_path.canonicalize()?;
        let ctx = self.projects.get(&canonical)
            .ok_or_else(|| anyhow::anyhow!("Project not loaded"))?;

        rename::rename_by_position(
            &ctx.host,
            &ctx.vfs,
            file_path,
            line,
            column,
            symbol_name,
            new_name,
        )
    }

    #[cfg(test)]
    pub(crate) fn insert_test_project_fast(&mut self, project_path: PathBuf) {
        let canonical =
            std::fs::canonicalize(&project_path).unwrap_or(project_path);
        self.clock += 1;
        self.projects.insert(
            canonical,
            ProjectContext {
                host: AnalysisHost::new(None),
                vfs: Vfs::default(),
                load_kind: LoadKind::Fast,
                stamps: HashMap::new(),
                last_used: self.clock,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// "I am holding a loaded analysis — do not sweep under me."
    ///
    /// This is the *production* gate, not a lock of its own: see
    /// [`crate::deep_stack::test_holding_an_analysis`] for why a second one
    /// beside it was the bug rather than the protection.
    use crate::deep_stack::test_holding_an_analysis as holding_an_analysis;

    /// "I am the sweep."
    use crate::deep_stack::test_sweeping_alone as sweeping_alone;

    #[test]
    fn rename_preview_includes_workspace_reverse_dependencies() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let workspace_path = workspace.path();

        write_file(
            &workspace_path.join("Cargo.toml"),
            r#"
[workspace]
members = ["engine_sdk", "engine_consumer"]
resolver = "2"
"#,
        );
        write_file(
            &workspace_path.join("engine_sdk/Cargo.toml"),
            r#"
[package]
name = "engine_sdk"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
"#,
        );
        let sdk_lib = workspace_path.join("engine_sdk/src/lib.rs");
        write_file(
            &sdk_lib,
            r#"pub trait Engine {
    fn tick(&self);
}
"#,
        );
        write_file(
            &workspace_path.join("engine_consumer/Cargo.toml"),
            r#"
[package]
name = "engine_consumer"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[dependencies]
engine_sdk = { path = "../engine_sdk" }
"#,
        );
        write_file(
            &workspace_path.join("engine_consumer/src/lib.rs"),
            r#"use engine_sdk::Engine;

pub struct Candle;

impl Engine for Candle {
    fn tick(&self) {}
}

pub fn run(engine: &dyn Engine) {
    engine.tick();
}
"#,
        );

        let mut service = SemanticService::new();
        let preview = service
            .rename_by_position(
                workspace_path,
                &sdk_lib,
                1,
                11,
                "Engine",
                "RenamedEngine",
            )
            .expect("rename preview");

        assert!(
            preview
                .edits
                .iter()
                .any(|edit| edit.file_path.ends_with("engine_sdk/src/lib.rs")),
            "expected declaration edit in engine_sdk, got {:?}",
            preview.edits
        );
        assert!(
            preview
                .edits
                .iter()
                .any(|edit| edit.file_path.ends_with("engine_consumer/src/lib.rs")),
            "expected downstream edit in engine_consumer, got {:?}",
            preview.edits
        );
    }

    #[test]
    fn runtime_semantic_status_and_clear_are_workspace_scoped() {
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let mut service = SemanticService::new();
        service.insert_test_project_fast(workspace.path().join("."));

        let status = service.status();
        assert_eq!(status.project_count, 1);
        assert_eq!(status.projects[0].load_kind, "fast");

        assert_eq!(service.clear_project(workspace.path()), 1);
        assert_eq!(service.project_count(), 0);
        assert_eq!(service.clear_all(), 0);
    }

    /// Call sites only. `find_references_by_name` returns the declaration
    /// alongside them (tagged `"target"` rather than `"reference"`), and
    /// counting both would make the assertions read one higher than the
    /// number they are actually about.
    fn call_sites(locations: &[Location]) -> usize {
        locations
            .iter()
            .filter(|location| location.name == "reference")
            .count()
    }

    /// A single-crate workspace with `target()` and one call site.
    fn one_crate_workspace(root: &Path) -> PathBuf {
        write_file(
            &root.join("Cargo.toml"),
            r#"
[package]
name = "staleness_probe"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
"#,
        );
        let lib = root.join("src/lib.rs");
        write_file(
            &lib,
            r#"pub fn target() {}

pub fn first() {
    target();
}
"#,
        );
        lib
    }

    /// The defect: a cached rust-analyzer context was never invalidated, so
    /// every answer after the first edit described the code as it was at load
    /// time. Here the second call site is added AFTER the first query, and the
    /// same service must see it.
    #[test]
    fn references_see_an_edit_made_after_the_project_was_cached() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        let lib = one_crate_workspace(root);

        let mut service = SemanticService::new();
        let before = service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("first query");
        assert_eq!(
            call_sites(&before),
            1,
            "positive control: the fixture must start with exactly one call site, got {before:?}"
        );

        write_file(
            &lib,
            r#"pub fn target() {}

pub fn first() {
    target();
}

pub fn second() {
    target();
}
"#,
        );

        let after = service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("query after edit");
        assert_eq!(
            call_sites(&after),
            2,
            "the edit must be visible without clearing the cache, got {after:?}"
        );
    }

    /// A new file changes the module tree, which cannot be patched into the
    /// database file-by-file — the refresh has to notice and reload.
    #[test]
    fn references_see_a_call_site_added_in_a_new_file() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        let lib = one_crate_workspace(root);

        let mut service = SemanticService::new();
        let before = service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("first query");
        assert_eq!(call_sites(&before), 1, "positive control: {before:?}");

        write_file(
            &root.join("src/extra.rs"),
            r#"pub fn third() {
    crate::target();
}
"#,
        );
        write_file(
            &lib,
            r#"pub mod extra;

pub fn target() {}

pub fn first() {
    target();
}
"#,
        );

        let after = service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("query after new file");
        assert_eq!(
            call_sites(&after),
            2,
            "a call site in a file added after load must be visible, got {after:?}"
        );
    }

    /// Call sites landing in one file, named by the tail of its path.
    fn refs_in(locations: &[Location], file_suffix: &str) -> usize {
        locations
            .iter()
            .filter(|location| {
                location.name == "reference" && location.file_path.ends_with(file_suffix)
            })
            .count()
    }

    /// Two crates and a known answer, built so that each way the search used to
    /// come up short is one countable reference.
    ///
    /// `probed()` is called three times: once in ordinary code, once from a
    /// `#[cfg(test)]` module, once from the other crate. `radius_override` is a
    /// **struct field** — the symbol index does not carry those at all — read
    /// four times across the same three places.
    ///
    /// Nothing here uses `use`: every cross-crate mention is written out as
    /// `oracle_lib::…`, so an import edge cannot pad the counts and make a
    /// missing call site look present.
    fn two_crate_oracle_workspace(root: &Path) {
        write_file(
            &root.join("Cargo.toml"),
            r#"
[workspace]
members = ["oracle_lib", "oracle_consumer"]
resolver = "2"
"#,
        );
        write_file(
            &root.join("oracle_lib/Cargo.toml"),
            r#"
[package]
name = "oracle_lib"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
"#,
        );
        write_file(
            &root.join("oracle_lib/src/lib.rs"),
            r#"
pub struct Probe {
    pub radius_override: u32,
}

impl Probe {
    pub fn read_here(&self) -> u32 {
        self.radius_override
    }
}

pub fn probed() {}

pub fn calls_here() {
    probed();
}

/// The other kind `SymbolCollector` drops on the floor, next to struct fields.
pub enum Facing {
    Downhole,
}

#[cfg(test)]
mod tests {
    #[test]
    fn touches_both() {
        let probe = super::Probe { radius_override: 1 };
        assert_eq!(probe.radius_override, 1);
        super::probed();
    }
}
"#,
        );
        write_file(
            &root.join("oracle_consumer/Cargo.toml"),
            r#"
[package]
name = "oracle_consumer"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[dependencies]
oracle_lib = { path = "../oracle_lib" }
"#,
        );
        write_file(
            &root.join("oracle_consumer/src/lib.rs"),
            r#"
pub fn read_across(probe: &oracle_lib::Probe) -> u32 {
    probe.radius_override
}

pub fn call_across() {
    oracle_lib::probed();
}
"#,
        );
    }

    /// The regression net for the two ways a `fn` search came up short: a
    /// workspace loaded without dependency edges cannot see the call from the
    /// other crate, and one loaded without `cfg(test)` cannot see the call from
    /// the test module. Both used to be invisible *and* silent — the answer was
    /// a shorter list, not an error.
    #[test]
    fn references_to_a_fn_include_the_other_crate_and_the_test_module() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        two_crate_oracle_workspace(root);

        let mut service = SemanticService::new();
        let found = service
            .find_references_by_name_with_exact(root, "probed", true)
            .expect("query references");

        assert_eq!(
            refs_in(&found, "oracle_consumer/src/lib.rs"),
            1,
            "the call from the other crate must be visible, got {found:?}"
        );
        assert_eq!(
            refs_in(&found, "oracle_lib/src/lib.rs"),
            2,
            "both the plain call and the #[cfg(test)] one must be visible, got {found:?}"
        );
        assert_eq!(
            call_sites(&found),
            3,
            "three call sites exist and three must be reported, got {found:?}"
        );
    }

    /// A struct field reaches rust-analyzer's symbol index not at all, so the
    /// name resolved to nothing and the tool answered "no references" — the
    /// same words it uses for a field nobody reads. Four reads exist here.
    #[test]
    fn references_to_a_struct_field_are_found_at_all() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        two_crate_oracle_workspace(root);

        let mut service = SemanticService::new();
        let found = service
            .find_references_by_name_with_exact(root, "radius_override", true)
            .expect("query references");

        assert_eq!(
            refs_in(&found, "oracle_consumer/src/lib.rs"),
            1,
            "the read from the other crate must be visible, got {found:?}"
        );
        assert_eq!(
            refs_in(&found, "oracle_lib/src/lib.rs"),
            3,
            "the impl read plus both test reads must be visible, got {found:?}"
        );
        assert!(
            found.iter().any(|location| location.name != "reference"),
            "the declaration itself must come back alongside the reads, got {found:?}"
        );
    }

    /// The fallback sweeps source text for the identifier, so it must not report
    /// a same-spelled *local* as a field read. Negative control for the sweep:
    /// without the "what did this actually resolve to" check, this reads 2.
    #[test]
    fn a_local_binding_spelled_like_the_field_is_not_counted() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        two_crate_oracle_workspace(root);
        write_file(
            &root.join("oracle_consumer/src/lib.rs"),
            r#"
pub fn read_across(probe: &oracle_lib::Probe) -> u32 {
    probe.radius_override
}

pub fn a_local_of_the_same_name() -> u32 {
    // Same spelling, unrelated binding: not a read of the field.
    let radius_override = 7;
    radius_override
}
"#,
        );

        let mut service = SemanticService::new();
        let found = service
            .find_references_by_name_with_exact(root, "radius_override", true)
            .expect("query references");

        assert_eq!(
            refs_in(&found, "oracle_consumer/src/lib.rs"),
            1,
            "only the field read counts, not the local spelled like it, got {found:?}"
        );
    }

    /// `find_definition` resolves through the same symbol index that carries no
    /// struct fields, so "where is this field declared" answered *no definition
    /// found* — the words a misspelled name gets, and the words that read as
    /// "there is no such field". The declaration is on line 2 of the lib
    /// (`write_file` trims the fixture's leading newline).
    #[test]
    fn definition_of_a_struct_field_is_found_at_all() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        two_crate_oracle_workspace(root);

        let mut service = SemanticService::new();
        let found = service
            .symbol_search_with_exact(root, "radius_override", 8, true)
            .expect("query definition");

        assert_eq!(
            found.len(),
            1,
            "the field is declared exactly once, got {found:?}"
        );
        assert!(
            found[0].file_path.ends_with("oracle_lib/src/lib.rs"),
            "the declaring crate's lib must be the answer, got {found:?}"
        );
        assert_eq!(
            found[0].line, 2,
            "the declaration line, not one of the reads, got {found:?}"
        );
    }

    /// Enum variants were read as sharing the field's fate — absent from the
    /// symbol index. They do not: disabling the sweep leaves this test green,
    /// so the index carries them and the index is what answers. Kept as the
    /// regression that says which half of the guess held, and as the net under
    /// the ranking and dedup the field fix put in this path.
    #[test]
    fn definition_of_an_enum_variant_is_found_at_all() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        two_crate_oracle_workspace(root);

        let mut service = SemanticService::new();
        let found = service
            .symbol_search_with_exact(root, "Downhole", 8, true)
            .expect("query definition");

        assert_eq!(
            found.len(),
            1,
            "the variant is declared exactly once, got {found:?}"
        );
        assert!(
            found[0].file_path.ends_with("oracle_lib/src/lib.rs"),
            "the declaring crate's lib must be the answer, got {found:?}"
        );
    }

    /// One crate declaring one of every kind a caller might reasonably type
    /// into `find_definition`, each under a name used nowhere else.
    ///
    /// Every surveyed candidate lives at module scope and outside
    /// `#[cfg(test)]`, so that none of them can come back absent for the
    /// cfg-gating reason instead of the one under study. The one item that *is*
    /// behind `#[cfg(test)]` is there to test exactly that gating, and the
    /// survey does not ask about it.
    /// A crate whose only declaration of `generated_field` is written by a build
    /// script into `OUT_DIR` and pulled in with `include!` — the shape prost
    /// uses, and the shape `ss_project_proto` uses in `rust_app`.
    ///
    /// The name also appears inside a string literal in `build.rs`. That is
    /// deliberate: the sweep walks source text, so a fixture where the only
    /// other occurrence is a literal proves it is resolving rather than
    /// grepping.
    fn generated_code_workspace(root: &Path) {
        write_file(
            &root.join("Cargo.toml"),
            r#"
[package]
name = "generated_probe"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
"#,
        );
        write_file(
            &root.join("build.rs"),
            r#"
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("generated.rs");
    std::fs::write(&out, "pub struct Generated { pub generated_field: u32 }\n").unwrap();
}
"#,
        );
        write_file(
            &root.join("src/lib.rs"),
            r#"
include!(concat!(env!("OUT_DIR"), "/generated.rs"));

pub fn read_generated(value: &Generated) -> u32 {
    value.generated_field
}
"#,
        );
    }

    /// Generated code was invisible: without build scripts there is no `OUT_DIR`
    /// to expand, so the included module is not in the tree and every reference
    /// into it resolves to nothing. Measured on `rust_app` before the fix —
    /// seven reads of a proto field counted as zero, in the same words used for
    /// a field nobody reads.
    ///
    /// The declaration's path is asserted too, and that is the non-vacuity
    /// check: it can only sit under `OUT_DIR` if resolution actually went
    /// through the generated file.
    #[test]
    fn references_into_generated_code_are_found() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        generated_code_workspace(root);

        let mut service = SemanticService::new();
        let found = service
            .find_references_by_name_with_exact(root, "generated_field", true)
            .expect("query references");

        assert_eq!(
            refs_in(&found, "src/lib.rs"),
            1,
            "the read of the generated field must be visible, got {found:?}"
        );
        assert!(
            found.iter().any(|location| {
                location.name != "reference" && location.file_path.ends_with("out/generated.rs")
            }),
            "the declaration must resolve into the build script's output, not to \
             a text match elsewhere, got {found:?}"
        );
    }

    fn kind_survey_workspace(root: &Path) {
        write_file(
            &root.join("Cargo.toml"),
            r#"
[package]
name = "kind_survey"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
"#,
        );
        write_file(
            &root.join("src/lib.rs"),
            r#"
pub const KIND_CONST: u32 = 1;
pub static KIND_STATIC: u32 = 2;
pub type KindAlias = u32;

pub struct KindStruct {
    pub kind_field: u32,
}

pub struct KindTuple(pub u32);

pub union KindUnion {
    pub kind_union_field: u32,
}

pub enum KindEnum {
    KindVariant,
    KindStructVariant { kind_variant_field: u32 },
}

pub trait KindTrait {
    const KIND_TRAIT_CONST: u32;
    type KindAssocType;
    fn kind_trait_method(&self) -> u32;
}

impl KindStruct {
    pub const KIND_ASSOC_CONST: u32 = 3;

    pub fn kind_impl_method(&self) -> u32 {
        self.kind_field
    }
}

impl KindTrait for KindStruct {
    const KIND_TRAIT_CONST: u32 = 4;
    type KindAssocType = u32;

    fn kind_trait_method(&self) -> u32 {
        self.kind_field
    }
}

pub mod kind_module {
    pub fn kind_module_fn() {}
}

macro_rules! kind_macro {
    () => {
        pub fn kind_generated() {}
    };
}

kind_macro!();

#[cfg(test)]
mod tests {
    #[test]
    fn kind_test_fn() {}
}
"#,
        );
    }

    /// What the symbol index carries and what it does not — measured, one row
    /// per kind, rather than inferred from the two kinds that were looked at.
    ///
    /// This is the mutant "turn the sweep off and ask for the definition" made
    /// permanent: `index_only_search` is the served path with the rescue removed,
    /// so a `false` here is a kind that answers *no definition found* whenever
    /// the sweep does not cover it. The sweep's accept list is `Field | Variant`,
    /// so every `false` row that is neither is a live blind spot, not a covered
    /// one.
    ///
    /// The guess this replaces held for exactly one of its two halves: fields are
    /// absent, enum variants are not.
    #[test]
    fn which_kinds_the_symbol_index_carries() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        kind_survey_workspace(root);

        // Sorted by name so the table reads as a table; the value is "the index
        // answered with at least one exact match".
        let candidates = [
            ("KIND_ASSOC_CONST", "associated const, inherent impl"),
            ("KIND_CONST", "module-scope const"),
            ("KIND_STATIC", "module-scope static"),
            ("KIND_TRAIT_CONST", "associated const, trait"),
            ("KindAlias", "type alias"),
            ("KindAssocType", "associated type"),
            ("KindEnum", "enum"),
            ("KindStruct", "struct"),
            ("KindStructVariant", "struct-shaped enum variant"),
            ("KindTrait", "trait"),
            ("KindTuple", "tuple struct"),
            ("KindUnion", "union"),
            ("KindVariant", "unit enum variant"),
            ("kind_field", "struct field"),
            ("kind_generated", "fn produced by macro expansion"),
            ("kind_impl_method", "inherent method"),
            ("kind_macro", "macro_rules! macro"),
            ("kind_module", "module"),
            ("kind_module_fn", "fn inside a submodule"),
            ("kind_trait_method", "trait method declaration"),
            ("kind_union_field", "union field"),
            ("kind_variant_field", "field of a struct-shaped variant"),
        ];

        let mut service = SemanticService::new();
        let measured: Vec<(&str, bool)> = candidates
            .iter()
            .map(|(name, _what)| {
                let found = service
                    .index_only_search(root, name, 8)
                    .expect("query the index");
                (*name, !found.is_empty())
            })
            .collect();

        let expected: Vec<(&str, bool)> = vec![
            ("KIND_ASSOC_CONST", true),
            ("KIND_CONST", true),
            ("KIND_STATIC", true),
            ("KIND_TRAIT_CONST", true),
            ("KindAlias", true),
            ("KindAssocType", true),
            ("KindEnum", true),
            ("KindStruct", true),
            ("KindStructVariant", true),
            ("KindTrait", true),
            ("KindTuple", true),
            ("KindUnion", true),
            ("KindVariant", true),
            ("kind_field", false),
            ("kind_generated", true),
            ("kind_impl_method", true),
            ("kind_macro", true),
            ("kind_module", true),
            ("kind_module_fn", true),
            ("kind_trait_method", true),
            ("kind_union_field", false),
            ("kind_variant_field", false),
        ];

        assert_eq!(
            measured, expected,
            "the index's coverage changed — a row flipping to false is a new \
             blind spot, and one flipping to true means the sweep is carrying \
             weight it no longer needs to"
        );
    }

    /// A definition behind `#[cfg(test)]` must be found by the first question
    /// asked, not only by one that follows a `find_references`.
    ///
    /// `find_definition` used to run on a `Fast` context, where `set_test` is
    /// off and a test module is not in the tree — so it answered *no definition
    /// found* for a test that exists. What made that a bug rather than a limit:
    /// `get_or_load` does not downgrade an already-`Full` context, so the very
    /// same query answered correctly once anything else in the session had asked
    /// for references. Measured on a real workspace, both answers minutes apart.
    ///
    /// Hence the shape of this test — one service, nothing asked before, the
    /// query cold. Run it after a reference lookup and it passes either way.
    #[test]
    fn a_definition_behind_cfg_test_is_found_by_the_first_question() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        kind_survey_workspace(root);

        let mut service = SemanticService::new();
        let found = service
            .symbol_search_with_exact(root, "kind_test_fn", 8, true)
            .expect("query definition");

        assert_eq!(
            found.len(),
            1,
            "the test fn is declared once and must be found on a cold service, \
             got {found:?}"
        );
    }

    /// Every kind the index does not carry must be one the sweep does, or the
    /// tool still answers *no definition found* for something that exists.
    ///
    /// The survey above names three: a struct field, a union field, and a field
    /// of a struct-shaped enum variant. All three are `SymbolKind::Field`, which
    /// is what makes one accept list enough — but "rust-analyzer labels a union
    /// field a field" is an assumption about someone else's code, so it is
    /// checked here rather than reasoned about.
    ///
    /// A tuple struct's field is deliberately absent from both lists: it is
    /// named `0`, so there is no identifier to ask about and no query that could
    /// come up short.
    #[test]
    fn the_sweep_covers_every_kind_the_index_misses() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        kind_survey_workspace(root);

        let mut service = SemanticService::new();
        for name in ["kind_field", "kind_union_field", "kind_variant_field"] {
            let found = service
                .symbol_search_with_exact(root, name, 8, true)
                .expect("query definition");

            assert_eq!(
                found.len(),
                1,
                "`{name}` is declared exactly once and the served path must say \
                 so, got {found:?}"
            );
        }
    }

    /// Negative control for the definition sweep, the twin of the one guarding
    /// references: a `let radius_override = 7;` declares that exact name too.
    /// Accepting it would answer "the field is declared here" pointing into an
    /// unrelated function body — a wrong location reads worse than none.
    #[test]
    fn a_local_binding_spelled_like_the_field_is_not_a_definition_of_it() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        two_crate_oracle_workspace(root);
        write_file(
            &root.join("oracle_consumer/src/lib.rs"),
            r#"
pub fn a_local_of_the_same_name() -> u32 {
    // Same spelling, unrelated binding: not a declaration of the field.
    let radius_override = 7;
    radius_override
}
"#,
        );

        let mut service = SemanticService::new();
        let found = service
            .symbol_search_with_exact(root, "radius_override", 8, true)
            .expect("query definition");

        assert_eq!(
            found.len(),
            1,
            "only the field is a definition of this name, got {found:?}"
        );
        assert!(
            found[0].file_path.ends_with("oracle_lib/src/lib.rs"),
            "the local must not be offered as the declaration, got {found:?}"
        );
    }

    /// Untouched code must not pay for the check: same stamps, no reload, and
    /// crucially the same answer.
    #[test]
    fn an_untouched_project_is_not_reloaded() {
        let _analysis = holding_an_analysis();
        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        one_crate_workspace(root);

        let mut service = SemanticService::new();
        service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("first query");

        let canonical = root.canonicalize().expect("canonicalize");
        let stamps_before = service.projects[&canonical].stamps.clone();

        service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("second query");

        assert!(
            matches!(
                classify_staleness(&stamps_before, &service.projects[&canonical].stamps),
                Staleness::Fresh
            ),
            "an untouched tree must classify as fresh"
        );
    }

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, contents.trim_start()).expect("write fixture file");
    }

    /// Which projects the service is currently holding, canonicalized, so an
    /// assertion can name them instead of counting them.
    fn loaded(service: &SemanticService) -> Vec<PathBuf> {
        let mut paths: Vec<_> = service.projects.keys().cloned().collect();
        paths.sort();
        paths
    }

    /// The cap has to evict by *use*, not by arrival. A queue that dropped the
    /// oldest arrival would throw away the project the session actually works
    /// in, which is the one it will ask about next.
    #[test]
    fn a_third_project_evicts_the_least_recently_used_not_the_oldest() {
        let _analysis = holding_an_analysis();
        let (first, second, third) = (
            tempfile::tempdir().expect("tempdir a"),
            tempfile::tempdir().expect("tempdir b"),
            tempfile::tempdir().expect("tempdir c"),
        );
        for dir in [&first, &second, &third] {
            one_crate_workspace(dir.path());
        }

        let mut service = SemanticService::with_max_projects(2);
        for dir in [&first, &second] {
            service
                .symbol_search(dir.path(), "target", 8)
                .expect("load project");
        }

        // The point of the test: `first` arrived first but is used last, so it
        // is `second` that must go when `third` needs the room.
        service
            .symbol_search(first.path(), "target", 8)
            .expect("re-query the first project");
        service
            .symbol_search(third.path(), "target", 8)
            .expect("load the third project");

        let mut expected = vec![
            first.path().canonicalize().expect("canonicalize first"),
            third.path().canonicalize().expect("canonicalize third"),
        ];
        expected.sort();
        assert_eq!(
            loaded(&service),
            expected,
            "the cap must keep the two most recently used projects"
        );
    }

    /// A cap of one is the interesting edge: every load evicts everything else,
    /// and the one thing that must survive is the project being asked about.
    #[test]
    fn the_project_being_used_is_never_the_victim() {
        let _analysis = holding_an_analysis();
        let (first, second) = (
            tempfile::tempdir().expect("tempdir a"),
            tempfile::tempdir().expect("tempdir b"),
        );
        for dir in [&first, &second] {
            one_crate_workspace(dir.path());
        }

        let mut service = SemanticService::with_max_projects(1);
        service
            .symbol_search(first.path(), "target", 8)
            .expect("load the first project");
        let found = service
            .symbol_search(second.path(), "target", 8)
            .expect("load the second project");

        assert!(
            !found.is_empty(),
            "positive control: the surviving project must still answer, got {found:?}"
        );
        assert_eq!(
            loaded(&service),
            vec![second.path().canonicalize().expect("canonicalize second")],
            "the project just queried is the one that stays"
        );
    }

    /// `0` means unlimited, the same as it does for the daemon's other knobs.
    #[test]
    fn a_cap_of_zero_evicts_nothing() {
        let dirs: Vec<_> = (0..3)
            .map(|_| tempfile::tempdir().expect("tempdir"))
            .collect();

        let mut service = SemanticService::with_max_projects(0);
        for dir in &dirs {
            service.insert_test_project_fast(dir.path().to_path_buf());
        }
        let kept = service.projects.keys().next().cloned().expect("a project");
        assert_eq!(service.evict_to_capacity(&kept), 0);
        assert_eq!(
            service.project_count(),
            3,
            "an unlimited cap holds everything"
        );
    }

    /// Two things at once, because the collection can fail in two directions.
    ///
    /// *That it happened*: the whole point is the revision bump — without one,
    /// salsa never reaches `for_each_evicted` and an LRU capacity evicts
    /// nothing. A collection that returned a count without bumping would look
    /// exactly like a working one from the outside, so the revision is read
    /// directly.
    ///
    /// *That it was safe*: the sweep is `unsafe` and process-global. Had it
    /// freed something still in use, the symptom would be a wrong answer or a
    /// crash on the next query — hence the same question on both sides of it.
    #[test]
    fn a_collection_bumps_the_revision_and_keeps_the_answers() {
        let _sweep = sweeping_alone();
        use ra_ap_ide_db::base_db::SourceDatabase;

        let workspace = tempfile::tempdir().expect("create workspace tempdir");
        let root = workspace.path();
        one_crate_workspace(root);

        let mut service = SemanticService::new();
        let before = service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("query before collection");
        assert_eq!(
            call_sites(&before),
            1,
            "positive control: the fixture must start with one call site, got {before:?}"
        );

        let canonical = root.canonicalize().expect("canonicalize");
        let revision_before = service.projects[&canonical]
            .host
            .raw_database()
            .nonce_and_revision();

        assert_eq!(
            service.collect_garbage(),
            1,
            "the collection must visit the loaded project"
        );

        let revision_after = service.projects[&canonical]
            .host
            .raw_database()
            .nonce_and_revision();
        assert_ne!(
            revision_before, revision_after,
            "without a revision bump the LRU capacities never evict anything"
        );

        let after = service
            .find_references_by_name_with_exact(root, "target", true)
            .expect("query after collection");
        assert_eq!(
            call_sites(&after),
            call_sites(&before),
            "collecting garbage must not change what the analysis knows, got {after:?}"
        );
    }

    /// Does rust-analyzer *leak*, or does the allocator merely keep what it
    /// freed? Both look identical from outside — RSS goes up and stays up — and
    /// they need opposite fixes, so this measures rather than assumes.
    ///
    /// # How it tells them apart
    ///
    /// Each cycle loads the workspace, drops the context, and asks the
    /// allocator to hand memory back. The number that matters is RSS *after*
    /// that release, cycle over cycle:
    ///
    /// - flat across cycles → nothing leaks; the resident memory is
    ///   fragmentation the allocator is sitting on, and the cure is the
    ///   allocator (the `mimalloc` feature), not rust-analyzer;
    /// - rising by roughly the same amount every cycle → a genuine leak, and
    ///   the per-cycle step is its size. rust-analyzer interns symbols in
    ///   process-global tables that are never freed, so a small constant step
    ///   is expected; a step near the cost of a whole load is not.
    ///
    /// The first cycle is excluded from the verdict: it pays one-time costs
    /// (proc-macro server, lazily built tables) that never recur.
    ///
    /// # Running it
    ///
    /// ```text
    /// RMC_RSS_CYCLE_PROJECT=/home/sc/t/bur/rust_app \
    ///   cargo test -p rmc-server --features migraphx -- --ignored --nocapture rss_across
    /// ```
    ///
    /// `RMC_RSS_CYCLE_COUNT` (default 4) and `RMC_RSS_CYCLE_TOLERANCE_MB`
    /// (default 512) tune length and verdict.
    #[test]
    #[ignore = "needs a real workspace in RMC_RSS_CYCLE_PROJECT and minutes to run"]
    fn rss_across_load_clear_cycles_separates_a_leak_from_fragmentation() {
        // Exclusive: this one sweeps too, and it measures RSS — a peer holding
        // a workspace would be counted as this test's own memory.
        let _sweep = sweeping_alone();
        let Ok(project) = std::env::var("RMC_RSS_CYCLE_PROJECT") else {
            panic!(
                "set RMC_RSS_CYCLE_PROJECT to a cargo workspace root; without one this test \
                 would measure nothing and pass, which is worse than not running"
            );
        };
        let project = PathBuf::from(project);
        let cycles = env_usize("RMC_RSS_CYCLE_COUNT", 4);
        let tolerance_mb = env_usize("RMC_RSS_CYCLE_TOLERANCE_MB", 512) as u64;
        assert!(cycles >= 2, "a verdict needs at least two cycles");

        let mut after_release = Vec::with_capacity(cycles);
        for cycle in 0..cycles {
            let mut service = SemanticService::new();
            // A real query, not a bare load: this is what fills the salsa
            // database in production, and an empty database would understate
            // both fragmentation and any leak.
            let found = service
                .symbol_search(&project, "main", 16)
                .expect("symbol search on the probe workspace");
            let loaded = crate::mcp::memory::rss_kib().expect("RSS readable on linux");

            service.clear_all();
            drop(service);
            let release = crate::mcp::memory::release_and_measure();
            let settled = release.rss_kib_after.expect("RSS readable on linux");
            after_release.push(settled);

            println!(
                "cycle {cycle}: {} symbol(s), loaded {} MB, after clear+release {} MB (released {:?} KiB)",
                found.len(),
                loaded / 1024,
                settled / 1024,
                release.released_kib,
            );
        }

        // Compare the second cycle with the last: the first is warm-up.
        let baseline = after_release[1];
        let final_rss = *after_release.last().expect("cycles >= 2");
        let growth_kib = final_rss.saturating_sub(baseline);
        let per_cycle_kib = growth_kib / (cycles as u64 - 1).max(1);

        println!(
            "verdict: grew {} MB over {} cycles after warm-up ({} MB per cycle); tolerance {} MB",
            growth_kib / 1024,
            cycles - 1,
            per_cycle_kib / 1024,
            tolerance_mb,
        );

        assert!(
            growth_kib / 1024 <= tolerance_mb,
            "RSS after clear+release grew {} MB across {} cycles ({} MB per cycle) — that is a \
             leak, not fragmentation, since every cycle drops its context and trims. Baseline \
             {} MB, final {} MB, all cycles: {:?} MB",
            growth_kib / 1024,
            cycles - 1,
            per_cycle_kib / 1024,
            baseline / 1024,
            final_rss / 1024,
            after_release.iter().map(|kib| kib / 1024).collect::<Vec<_>>(),
        );
    }

    /// Where does a loaded rust-analyzer database actually keep its bytes?
    ///
    /// The RSS cycle test above answers "does it leak"; this one answers "what
    /// is it holding". It asks salsa itself — `Database::memory_usage()`, built
    /// because `base-db` enables the `salsa_unstable` feature — for a per
    /// ingredient breakdown, and prints it sorted by bytes.
    ///
    /// # 🚨 What this measurement can and cannot see
    ///
    /// Salsa reports three numbers per ingredient: the number of instances, the
    /// *stack* size of their fields, and its own metadata. Heap behind those
    /// fields is counted only where the query declares `heap_size = <fn>`, and
    /// every heavy memo in rust-analyzer is a `Vec`/`Arc`/`Box` behind a small
    /// struct — so the heap is exactly where the gigabytes are.
    ///
    /// Upstream declares it nowhere (zero occurrences as of the 2026-08-26
    /// tree, which is what made this table explain 4.2% of RSS). Our fork
    /// declares it on the families that were the biggest by count *and* own
    /// buffers: the parse query, bodies with their source maps, the signature
    /// queries and macro expansion. That took the same table to 42.7%. Def
    /// maps, item trees and symbol indices followed, then type inference — the
    /// one capacity the fork chose for itself, and so the one that most needed
    /// bytes to argue from.
    ///
    /// The rest of the rows are still counts and shapes, not bytes, and a
    /// number that is present is a LOWER bound (see `syntax::heap_size`). Read
    /// the printed "accounted for" percentage as the honest measure of how much
    /// of the process this table explains — never as a budget. Turning another
    /// suspect into a number takes one `heap_size = <fn>` attribute on its
    /// query in the fork; picking which suspect is what this output is for.
    ///
    /// # Running it
    ///
    /// ```text
    /// RMC_SALSA_MEMORY_PROJECT=/home/sc/t/bur/rust_app \
    ///   cargo test -p rmc-server --features migraphx -- --ignored --nocapture salsa_ingredient
    /// ```
    ///
    /// `RMC_SALSA_MEMORY_TOP` (default 25) caps how many rows are printed.
    ///
    /// `RMC_SALSA_MEMORY_PROBE` picks what fills the database first:
    /// `symbols` (default, a symbol search — cheap, and what every earlier run
    /// in this track used) or `references` (chase the uses of
    /// `RMC_SALSA_MEMORY_SYMBOL`, which then has to be set). Only the second
    /// runs type inference, so only under it does the `InferenceResult` row mean
    /// anything.
    /// The probe decides every number in the table: two runs are comparable only
    /// if they probed alike.
    #[test]
    #[ignore = "needs a real workspace in RMC_SALSA_MEMORY_PROJECT and minutes to run"]
    fn salsa_ingredient_memory_breakdown_names_the_suspects() {
        // Exclusive for the same two reasons as the RSS cycle test: it sweeps,
        // and it attributes memory.
        let _sweep = sweeping_alone();
        // Still needed below, where the test asks the database itself which
        // queries have a tunable capacity; the breakdown no longer needs it.
        use ra_ap_ide_db::base_db::salsa;

        let Ok(project) = std::env::var("RMC_SALSA_MEMORY_PROJECT") else {
            panic!(
                "set RMC_SALSA_MEMORY_PROJECT to a cargo workspace root; without one this test \
                 would measure nothing and pass, which is worse than not running"
            );
        };
        let project = PathBuf::from(project);
        let top = env_usize("RMC_SALSA_MEMORY_TOP", 25);
        // Which query fills the database before it is weighed. `symbols` is the
        // default because it is what every earlier run in this track used, and
        // the probe decides every number in the table — two runs are comparable
        // only if they probed the same way.
        //
        // `references` exists because `symbols` never runs type inference:
        // measured on a 4000-file workspace, it leaves `InferenceResult` at
        // thirteen memos, and a byte count over thirteen memos is noise. Chasing
        // a name's use sites infers every body that mentions it, which is also
        // what the `find_references` endpoint does in production.
        let probe = std::env::var("RMC_SALSA_MEMORY_PROBE")
            .unwrap_or_else(|_| "symbols".to_string());
        // Deliberately without a default. The name decides both what the probe
        // costs and whether it proves anything, and one guessed here would do
        // neither: `new` looked like the obvious choice and matches up to fifty
        // declarations, each of which sweeps the whole tree for its own uses —
        // measured at 10 GB and still climbing after two minutes on a 4000-file
        // workspace. What works is a name with ONE declaration and many uses.
        let symbol = std::env::var("RMC_SALSA_MEMORY_SYMBOL").unwrap_or_default();

        // The breakdown itself is production code — the `analysis_memory` tool
        // serves the same rows — so this test exercises what ships rather than
        // a copy that could drift from it.
        type Row = IngredientMemory;
        let breakdown = ingredient_rows;

        let mut service = SemanticService::new();
        // A real query, not a bare load: an empty database has nothing memoized
        // and every row would read as zero.
        let probed = match probe.as_str() {
            "symbols" => {
                let found = service
                    .symbol_search(&project, "main", 16)
                    .expect("symbol search on the probe workspace");
                format!("probe `symbols`: {} symbol(s) named `main`", found.len())
            }
            "references" => {
                assert!(
                    !symbol.is_empty(),
                    "the `references` probe needs RMC_SALSA_MEMORY_SYMBOL: a name with one \
                     declaration and many use sites in the probe workspace. See the comment on \
                     `symbol` for why there is no default"
                );
                let found = service
                    .find_references_by_name_with_exact(&project, &symbol, true)
                    .expect("reference search on the probe workspace");
                format!(
                    "probe `references`: {} location(s) for `{symbol}`",
                    found.len()
                )
            }
            other => panic!(
                "RMC_SALSA_MEMORY_PROBE={other:?} names no probe; use `symbols` (cheap, runs no \
                 type inference) or `references` (infers every body mentioning \
                 RMC_SALSA_MEMORY_SYMBOL, default `new`)"
            ),
        };
        let rss_kib = crate::mcp::memory::rss_kib().expect("RSS readable on linux");

        let canonical = project
            .canonicalize()
            .expect("canonicalize the probe project");
        // Borrowed inline, not held: `collect_garbage` below needs `&mut` on the
        // service, so no borrow of a host may outlive a single measurement.
        let rows = breakdown(
            &service
                .projects
                .get(&canonical)
                .expect("the query above must have cached a context for this path")
                .host,
        );

        let accounted: usize = rows.iter().map(Row::bytes).sum();
        let with_heap = rows.iter().filter(|row| row.heap_bytes.is_some()).count();
        let instances: usize = rows.iter().map(|row| row.count).sum();

        println!(
            "{probed}; RSS {} MB; {} ingredients, {} instances total",
            rss_kib / 1024,
            rows.len(),
            instances,
        );
        println!(
            "accounted for {} MB = {:.1}% of RSS; {} of {} ingredients report heap size",
            accounted / (1024 * 1024),
            100.0 * accounted as f64 / (rss_kib as f64 * 1024.0),
            with_heap,
            rows.len(),
        );
        println!(
            "{:<8} {:<44} {:>10} {:>10} {:>10} {:>10}  {}",
            "family", "query / struct", "count", "stack KiB", "meta KiB", "heap KiB", "produces"
        );
        for row in rows.iter().take(top) {
            println!(
                "{:<8} {:<44} {:>10} {:>10} {:>10} {:>10}  {}",
                row.family,
                row.name,
                row.count,
                row.stack_bytes / 1024,
                row.metadata_bytes / 1024,
                row.heap_bytes
                    .map_or("-".to_string(), |heap| (heap / 1024).to_string()),
                row.produces.or(row.page.as_deref()).unwrap_or(""),
            );
        }

        assert!(
            !rows.is_empty(),
            "a database that answered a query must have memoized something; an empty breakdown \
             means the measurement, not the database, is broken"
        );

        // The salsa `debug_name` of the type inference query. The spelling is
        // the macro's, lifetime and spaces included, and it cannot be guessed —
        // it is what `Database::lru_capacity_names` prints below.
        const INFER_QUERY: &str = "InferenceResult < 'db >::for_body_";
        let row_of = |rows: &[Row], query: &str| {
            rows.iter()
                .find(|row| row.family == "query" && row.name == query)
                .map(|row| (row.count, row.heap_bytes))
        };
        if probe == "references" {
            // A floor on "did inference run at all", not on "is the sample
            // large": the `symbols` probe leaves this query at thirteen memos,
            // so anything in that neighbourhood means the probe missed its
            // target rather than found a small answer. How big the sample is,
            // the reader judges from the per-memo average printed beside it.
            const INFER_FLOOR: usize = 50;
            let (inferred, infer_heap) = row_of(&rows, INFER_QUERY).unwrap_or((0, None));
            println!(
                "type inference: {inferred} memo(s), heap {} KiB ({} bytes each on average)",
                infer_heap.unwrap_or(0) / 1024,
                infer_heap.unwrap_or(0).checked_div(inferred).unwrap_or(0),
            );
            assert!(
                inferred > INFER_FLOOR,
                "the reference probe inferred only {inferred} bodies (floor {INFER_FLOOR}) — \
                 `{symbol}` is used too little in this workspace to make type inference measurable, \
                 so any heap number for `{INFER_QUERY}` above is noise. Point \
                 RMC_SALSA_MEMORY_SYMBOL at a name called from many bodies"
            );
        }
        // The gap between this table and RSS is the finding, not a defect: it is
        // the heap that salsa cannot see without `heap_size`. If the table ever
        // covers the process, this assert fires and the conclusion above — "read
        // counts, not bytes" — has to be revisited.
        assert!(
            accounted < rss_kib as usize * 1024,
            "the breakdown accounts for {} MB of a {} MB process — salsa is not supposed to be \
             able to see that much without `heap_size` on the queries; re-check the arithmetic \
             before trusting it",
            accounted / (1024 * 1024),
            rss_kib / 1024,
        );

        // ---- the end-to-end positive control for the whole LRU + GC chain ----
        //
        // A capacity is declared on this query (512, restated in
        // `update_base_query_lru_capacities`) but salsa only evicts inside
        // `reset_for_new_revision`, so a database nobody writes to keeps every
        // memo it ever made however small the capacity is. That is the entire
        // reason `collect_garbage` exists; this asserts it in *counts*, which —
        // unlike RSS — cannot be hidden by glibc holding freed pages.
        // Ask the database which queries actually have a tunable capacity rather
        // than trusting the names the fork hardcodes: salsa suffixes the
        // generated ingredient of an *associated* tracked fn with `_`, so a
        // hand-written `Type::method` matches nothing and only logs a warning
        // nobody reads. This line is what makes that visible.
        {
            let db: &dyn salsa::Database = service
                .projects
                .get(&canonical)
                .expect("context still cached")
                .host
                .raw_database();
            let mut tunable = salsa::Database::lru_capacity_names(db);
            tunable.sort_unstable();
            println!("queries with a tunable LRU capacity: {tunable:?}");
        }
        {
            // And the other half: that rust-analyzer's own configuration reaches
            // those queries. It reports a miss only through `tracing::warn!`, and
            // a warning in a daemon log is not an oracle — this turns the whole
            // set of names into one red assertion.
            let missed = service
                .projects
                .get_mut(&canonical)
                .expect("context still cached")
                .host
                .raw_database_mut()
                .update_base_query_lru_capacities(None);
            assert_eq!(
                missed, 0,
                "rust-analyzer asked for {missed} LRU capacities that match no query in this \
                 database; the knob looks alive and retunes nothing (see the printed list of \
                 tunable names just above for what it should have said)"
            );
        }

        // The salsa `debug_name` of a tracked query, which is what the capacity
        // is addressed by. Associated tracked fns carry a trailing `_` — the
        // macro's own name for the generated ingredient — so the readable
        // `Body::with_source_map` matches nothing at all.
        const CAPPED_QUERY: &str = "Body::with_source_map_";
        const DECLARED_CAP: usize = 512;
        let (before, heap_before) = row_of(&rows, CAPPED_QUERY).unwrap_or_else(|| {
            panic!(
                "no query named `{CAPPED_QUERY}` in the breakdown — upstream renamed it, and \
                 `update_base_query_lru_capacities` is now setting a capacity on nothing"
            )
        });
        assert!(
            before > DECLARED_CAP,
            "the probe workspace memoized only {before} bodies, at or under the {DECLARED_CAP} \
             capacity — eviction has nothing to do here, so a pass below would prove nothing; \
             point RMC_SALSA_MEMORY_PROJECT at a bigger workspace"
        );

        service.collect_garbage();

        let after_rows = breakdown(
            &service
                .projects
                .get(&canonical)
                .expect("collecting garbage must not drop the context")
                .host,
        );
        let (after, heap_after) = row_of(&after_rows, CAPPED_QUERY).unwrap_or((0, None));
        println!(
            "after collect_garbage: `{CAPPED_QUERY}` {before} -> {after} memo slot(s), heap \
             {heap_before:?} -> {heap_after:?} (cap {DECLARED_CAP}); RSS {} MB",
            crate::mcp::memory::rss_kib().unwrap_or(0) / 1024,
        );

        // What the collection actually freed, per query. The table above says
        // where the memory *is*; this says how much of it a collection can take
        // back, which is the other half of arguing about a capacity — and it is
        // not derivable from the first, because only the LRU-capped queries
        // shed anything at all.
        let mut freed: Vec<(&str, usize)> = rows
            .iter()
            .filter_map(|row| {
                let before = row.heap_bytes?;
                let after = row_of(&after_rows, row.name).map(|(_, heap)| heap)??;
                before.checked_sub(after).filter(|freed| *freed > 0).map(|freed| (row.name, freed))
            })
            .collect();
        freed.sort_by_key(|&(_, freed)| std::cmp::Reverse(freed));
        println!("freed by the collection:");
        for (query, freed) in &freed {
            println!("  {:<50} {:>9} KiB", query, freed / 1024);
        }
        if freed.is_empty() {
            println!("  (nothing — every instrumented ingredient kept every byte)");
        }
        // Why the *slot count* cannot be the oracle: evicting a memo sets its
        // value to `None` and keeps the slot (`function/memo.rs`, `map_memo`), so
        // the count is unchanged whether eviction ran or not. What does move is
        // the reported heap, and it is the only allocator-independent signal
        // available — RSS cannot serve, because glibc need not return the pages.
        //
        // The shape of that signal depends on the fork declaring `heap_size` on
        // this query. Before it did, a live memo reported `None` and an evicted
        // one `Some(0)`, so the whole test could say was "at least one value went
        // away". With the option in place the ingredient reports real bytes, and
        // the same question is answered by a strict drop — with the size of the
        // drop printed above, which the old encoding could never show.
        let describe_missing_option = || {
            format!(
                "`{CAPPED_QUERY}` reports no heap size, so this test cannot tell an evicted memo \
                 from a live one. The `heap_size` option on that query is gone — most likely an \
                 upstream rebase dropped it from the rust-analyzer fork (see \
                 `hir-def/src/expr_store/heap_size.rs`)"
            )
        };
        let heap_before = heap_before.unwrap_or_else(|| panic!("{}", describe_missing_option()));
        let heap_after = heap_after.unwrap_or_else(|| panic!("{}", describe_missing_option()));
        assert!(
            heap_after < heap_before,
            "`{CAPPED_QUERY}` holds {after} memo slots against a capacity of {DECLARED_CAP}, and \
             after a garbage collection it still owns every byte it did before ({heap_before}): \
             the capacity is declared but nothing enforces it, which is exactly the defect this \
             whole chain exists to fix"
        );
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(default)
    }
}
