//! A changed manifest, lockfile or toolchain file reloads the workspace into
//! the database it already has — plan item `manifest-reload-in-place`.
//!
//! # What a full load cost
//!
//! A load builds a new database and primes it: on a ~8000-file workspace with
//! ≈ 2400 crates, 36–47 s on a quiet machine (twice that under memory
//! pressure), of which `cargo metadata`, build scripts and reading the files
//! are ≈ 6 s and priming the rest. A fleet editing manifests and the lockfile
//! made it five loads in three hours, each a call of 76–100 s, and each left
//! the table of uses to be built again from nothing (≈ 9000 pending files) —
//! so the first queries after it swept live and came back `INCOMPLETE`.
//!
//! # What this does instead
//!
//! What rust-analyzer's own main loop does on a workspace change
//! (`GlobalState::switch_workspaces`): the new workspace from `cargo metadata`,
//! its crate graph and source roots written into the **same** database, its
//! files into the **same** `Vfs`. So:
//!
//! - file ids stay: a file the old load had keeps its id, and `set_file_contents`
//!   drops a text whose hash is unchanged — only changed, added and removed
//!   files become text writes;
//! - crates stay: `CrateGraphBuilder::set_in_db` finds an old crate by its root
//!   file and writes only the inputs that differ (salsa does not compare
//!   inputs itself), adds the new ones and leaves the gone ones out of
//!   `all_crates`;
//! - source roots stay: each set of the new partition takes the id of the old
//!   root most of its files were in, so an unchanged root is not written at all
//!   and its files keep their `file_source_root`. Numbering by the new
//!   partition's order (`set_roots`, what a fresh load does) would shift every
//!   root after a new package and rewrite the root of every file — every def
//!   map would then recompute. New roots get ids no root had; roots gone are
//!   emptied and retired.
//!
//! Memos of unchanged crates are then verified, not recomputed; the priming
//! pass that follows ([`super::loader::prime`]) does the verifying and the
//! recomputation of what did change before a query asks.
//!
//! # Why the answer is a fresh load's
//!
//! The inputs end up what a fresh load writes, up to the numbering of ids:
//! the same crate graph (same builder, same `cargo metadata`), the same files
//! (the same loader entries, read through the same loader; a file no entry
//! holds and no crate root names is removed, as a fresh load would never have
//! read it), each in the set the same partition classifies it into, the same
//! proc-macro results (no server: every one "disabled", as the load writes
//! it). Nothing derived survives that is not recomputed from them. The test
//! oracle is exactly that comparison (`a_manifest_change_reloads_in_place_*`).
//!
//! The table of uses keeps a ready file only where nothing it may resolve
//! through changed: files of the crates whose inputs changed, that are new,
//! whose root had a file changed, added or removed — and of all their
//! transitive reverse dependencies — are pending, as a surface change makes
//! them (`roots_depending_on`); so are the changed, added files themselves.
//! A library declaration's key (its file id and offset) stays valid for every
//! file left ready: the library's files did not change, or its crate changed
//! and its users are pending.
//!
//! # Failure
//!
//! `cargo metadata` and the build scripts run first, before anything is
//! written: their failure leaves the context as it was, and the caller falls
//! back to a full load (which reports the same error if it is real). Nothing
//! after the first write can fail.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use ra_ap_ide_db::base_db::salsa::{Durability, Setter as _};
use ra_ap_ide_db::base_db::{
    Crate, FileId, FileSet, LibraryRoots, LocalRoots, ProcMacroLoadingError, SourceDatabase,
    SourceRoot, SourceRootId, all_crates,
};
use ra_ap_ide_db::ChangeWithProcMacros;
use ra_ap_load_cargo::ProjectFolders;
use ra_ap_vfs::loader::{self as vfs_loader, Handle as _, LoadingProgress};
use ra_ap_vfs::{AbsPath, Change, FileExcluded, VfsPath};

use super::{
    LoadKind, ProjectContext, exposure_by_root, loader, position, read_vfs, roots_depending_on,
};

/// What one reload in place did, for its log line.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct Reloaded {
    /// `cargo metadata` and the build scripts.
    pub(super) workspace: Duration,
    /// Reading the files and writing the change.
    pub(super) apply: Duration,
    /// Priming every crate after the change.
    pub(super) prime: Duration,
    pub(super) crates_new: usize,
    pub(super) crates_changed: usize,
    pub(super) crates_gone: usize,
    pub(super) roots_new: usize,
    pub(super) roots_rewritten: usize,
    pub(super) roots_retired: usize,
    /// Text writes: files changed or added, and files removed.
    pub(super) texts_written: usize,
    pub(super) texts_removed: usize,
    /// Ready files of the table of uses made pending, and pending after.
    pub(super) uses_widened: usize,
    pub(super) uses_pending: usize,
}

/// The inputs of `krate` a reload may change, owned, comparable: what
/// `set_in_db` compares one by one before writing.
fn crate_inputs(db: &ra_ap_ide::RootDatabase, krate: Crate) -> impl PartialEq + use<> {
    (
        krate.data(db).clone(),
        krate.extra_data(db).clone(),
        krate.workspace_data(db).clone(),
        krate.cfg_options(db).clone(),
        krate.env(db).clone(),
    )
}

/// Reload the workspace of `ctx` at `canonical` into its own database; see the
/// module header. On error nothing is changed.
///
/// # What it costs
///
/// `cargo metadata` (and the build scripts, cached by cargo: on the ~8000-file
/// workspace ≈ 6 s together with the file read below — what a full load
/// spends before priming); one read and hash of every file the loader entries
/// hold (the same read a load makes, ≈ 49 000 files); a pass over the crate
/// graph and the roots' file sets (milliseconds); then the priming pass, which
/// verifies the memos of unchanged crates and recomputes those of changed ones.
/// No second database is ever alive: the peak is the one database plus what
/// the change recomputes.
pub(super) fn reload_in_place(ctx: &mut ProjectContext, canonical: &Path) -> Result<Reloaded> {
    let mut reloaded = Reloaded::default();
    let started = Instant::now();
    let cargo_config = loader::cargo_config_of(ctx.load_kind == LoadKind::Full)?;
    let load_config = loader::load_config();
    // The only step that can fail; queries in flight go on meanwhile.
    let workspace = loader::project_workspace(canonical, &cargo_config, &load_config)?;
    reloaded.workspace = started.elapsed();
    let applying = Instant::now();
    let manifests = loader::package_manifests(&workspace);
    let folders = ProjectFolders::new(std::slice::from_ref(&workspace), &[], None);

    // Before: the crates and their inputs, the project files, and the root of
    // every file the database has a root for.
    let (old_crates, old_project_files, old_root_of) = {
        let db = ctx.host.raw_database();
        let old_crates: HashMap<Crate, _> = all_crates(db)
            .iter()
            .map(|&krate| (krate, crate_inputs(db, krate)))
            .collect();
        let old_project_files: HashSet<FileId> =
            position::project_rust_files(db, &read_vfs(&ctx.vfs), canonical)
                .into_iter()
                .collect();
        let mut old_root_of: HashMap<FileId, SourceRootId> = HashMap::new();
        for &id in &ctx.layout.root_ids {
            for file in db.source_root(id).source_root(db).iter() {
                old_root_of.insert(file, id);
            }
        }
        (old_crates, old_project_files, old_root_of)
    };

    // Queries may still run on snapshots, reading the `Vfs` beside them:
    // cancelled first, as a refresh does (`apply_tree_change`). The slot is
    // held, so no new one starts until this is done.
    ctx.host.trigger_cancellation();

    // The files, read as a fresh load reads them (`load_workspace_into_db`):
    // the crate roots the graph names, synchronously, then every file of the
    // loader entries.
    let mut vfs = ctx.vfs.write().unwrap_or_else(|poisoned| poisoned.into_inner());
    // Nothing should be queued (a refresh takes its own); anything that were
    // would be written again below as if read now.
    drop(vfs.take_changes());
    let (sender, receiver) = crossbeam_channel::unbounded();
    let mut file_loader = ra_ap_vfs_notify::NotifyHandle::spawn(sender);
    let mut read: HashSet<FileId> = HashSet::new();
    let (crate_graph, proc_macro_paths) = workspace.to_crate_graph(
        &mut |path: &AbsPath| {
            let contents = file_loader.load_sync(path);
            let path = VfsPath::from(path.to_path_buf());
            vfs.set_file_contents(path.clone(), contents);
            let (file_id, excluded) = vfs.file_id(&path)?;
            read.insert(file_id);
            (excluded == FileExcluded::No).then_some(file_id)
        },
        &cargo_config.extra_env,
    );
    file_loader.set_config(vfs_loader::Config {
        load: folders.load.clone(),
        watch: vec![],
        version: 0,
    });
    for message in &receiver {
        match message {
            vfs_loader::Message::Progress { n_done: LoadingProgress::Finished, .. } => break,
            vfs_loader::Message::Progress { .. } => {}
            vfs_loader::Message::Loaded { files } | vfs_loader::Message::Changed { files } => {
                for (path, contents) in files {
                    let present = contents.is_some();
                    let path = VfsPath::from(path);
                    vfs.set_file_contents(path.clone(), contents);
                    if present && let Some((file_id, _)) = vfs.file_id(&path) {
                        read.insert(file_id);
                    }
                }
            }
        }
    }
    drop(file_loader);
    // A file of the old load that nothing read now: no entry holds it, no crate
    // root names it — a fresh load would not have it.
    let unread: Vec<VfsPath> = vfs
        .iter()
        .filter(|(file_id, _)| !read.contains(file_id))
        .map(|(_, path)| path.clone())
        .collect();
    for path in unread {
        vfs.set_file_contents(path, None);
    }
    let mut texts: Vec<(FileId, Option<String>)> = Vec::new();
    for (_, file) in vfs.take_changes() {
        let text = match file.change {
            // A load leaves a non-UTF-8 file without a text; the closest a
            // database that had one can come is an empty one.
            Change::Create(bytes, _) | Change::Modify(bytes, _) => String::from_utf8(bytes).ok(),
            Change::Delete => None,
        };
        if text.is_none() {
            reloaded.texts_removed += 1;
        } else {
            reloaded.texts_written += 1;
        }
        texts.push((file.file_id, text));
    }
    let new_roots: Vec<SourceRoot> = folders.source_root_config.partition(&vfs);
    drop(vfs);

    // The ids of the new partition's sets: the catch-all keeps the catch-all's;
    // every other set the old root most of its files were in, the largest
    // share first, each old id once; the rest new ids.
    let catch_all = new_roots.len() - 1;
    let old_catch_all = *ctx
        .layout
        .root_ids
        .last()
        .expect("a layout has at least the catch-all set");
    let mut votes: Vec<(usize, usize, SourceRootId)> = Vec::new();
    for (index, root) in new_roots.iter().enumerate().take(catch_all) {
        let mut count: HashMap<SourceRootId, usize> = HashMap::new();
        for file in root.iter() {
            if let Some(&old) = old_root_of.get(&file)
                && old != old_catch_all
            {
                *count.entry(old).or_default() += 1;
            }
        }
        votes.extend(count.into_iter().map(|(old, files)| (files, index, old)));
    }
    votes.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.0.cmp(&b.2.0)));
    // Not matched yet: a set no old root has taken.
    let mut matched: Vec<Option<SourceRootId>> = vec![None; new_roots.len()];
    matched[catch_all] = Some(old_catch_all);
    let mut claimed: HashSet<SourceRootId> = HashSet::from([old_catch_all]);
    for (_, index, old) in votes {
        if matched[index].is_none() && !claimed.contains(&old) {
            claimed.insert(old);
            matched[index] = Some(old);
        }
    }
    // A set no file voted for (an empty one: a package root with no file the
    // loader reads) takes a free old id whose root is equal to it — else every
    // reload, changed or not, moved it to a new id and rewrote the root sets
    // (seen on a large workspace: one such set). Cost: the unmatched sets
    // against the free old ids, a handful each.
    {
        let db = ctx.host.raw_database();
        for (index, root) in new_roots.iter().enumerate() {
            if matched[index].is_some() {
                continue;
            }
            let equal = ctx.layout.root_ids.iter().copied().find(|old| {
                !claimed.contains(old) && *db.source_root(*old).source_root(db) == *root
            });
            if let Some(old) = equal {
                claimed.insert(old);
                matched[index] = Some(old);
            }
        }
    }
    let first_new_id = ctx.layout.next_root_id;
    let mut next_root_id = first_new_id;
    let root_ids: Vec<SourceRootId> = matched
        .into_iter()
        .map(|id| {
            id.unwrap_or_else(|| {
                let id = SourceRootId(next_root_id);
                next_root_id += 1;
                id
            })
        })
        .collect();
    let retired: Vec<SourceRootId> = ctx
        .layout
        .root_ids
        .iter()
        .copied()
        .filter(|id| !claimed.contains(id))
        .collect();

    // Roots before texts: storing a text asks the file's root (see
    // `apply_tree_change`). Only what differs is written.
    let db = ctx.host.raw_database_mut();
    // Of the inputs' own set type, which they are compared with and written to.
    let mut local = LocalRoots::get(db).roots(db).clone();
    local.clear();
    let mut library = LibraryRoots::get(db).roots(db).clone();
    library.clear();
    // The sets of the roots whose files changed, for the table of uses below.
    let mut changed_roots: BTreeSet<SourceRootId> = BTreeSet::new();
    for (root, &id) in new_roots.into_iter().zip(&root_ids) {
        let durability = if root.is_library { Durability::MEDIUM } else { Durability::LOW };
        if root.is_library {
            library.insert(id);
        } else {
            local.insert(id);
        }
        let new = id.0 >= first_new_id;
        if new {
            reloaded.roots_new += 1;
        } else if *db.source_root(id).source_root(db) == root {
            continue;
        } else {
            reloaded.roots_rewritten += 1;
        }
        changed_roots.insert(id);
        for file in root.iter() {
            if old_root_of.get(&file) != Some(&id) {
                db.set_file_source_root_with_durability(file, id, durability);
            }
        }
        db.set_source_root_with_durability(id, triomphe::Arc::new(root), durability);
    }
    for &id in &retired {
        let old = db.source_root(id).source_root(db);
        if old.iter().next().is_some() {
            let (empty, durability) = if old.is_library {
                (SourceRoot::new_library(FileSet::default()), Durability::MEDIUM)
            } else {
                (SourceRoot::new_local(FileSet::default()), Durability::LOW)
            };
            db.set_source_root_with_durability(id, triomphe::Arc::new(empty), durability);
        }
    }
    reloaded.roots_retired = retired.len();
    if *LocalRoots::get(db).roots(db) != local {
        LocalRoots::get(db).set_roots(db).to(local);
    }
    if *LibraryRoots::get(db).roots(db) != library {
        LibraryRoots::get(db).set_roots(db).to(library);
    }

    // The proc macros as a load without a server writes them
    // (`load_workspace_into_db`, `ProcMacroServerChoice::None`): every crate's
    // path, found or not, ends in the server's "disabled" error.
    let disabled = ProcMacroLoadingError::ProcMacroSrvError(
        ProcMacroLoadingError::Disabled.to_string().into_boxed_str(),
    );
    let proc_macros: ra_ap_hir::ProcMacrosBuilder = proc_macro_paths
        .into_iter()
        .map(|(krate, path)| (krate, path.and_then(|_| Err(disabled.clone()))))
        .collect();
    let mut change = ChangeWithProcMacros::default();
    let text_files: Vec<(FileId, bool)> =
        texts.iter().map(|(file, text)| (*file, text.is_some())).collect();
    for (file, text) in texts {
        change.change_file(file, text);
    }
    change.set_crate_graph(crate_graph);
    change.set_proc_macros(proc_macros);
    ctx.host.apply_change(change);

    // Crates new or with a changed input, and gone.
    let db = ctx.host.raw_database();
    let crates = all_crates(db);
    for &krate in crates.iter() {
        let changed = match old_crates.get(&krate) {
            None => {
                reloaded.crates_new += 1;
                true
            }
            Some(old) if *old != crate_inputs(db, krate) => {
                reloaded.crates_changed += 1;
                true
            }
            Some(_) => false,
        };
        if changed {
            changed_roots.insert(db.file_source_root(krate.data(db).root_file_id).source_root_id(db));
        }
    }
    let now: HashSet<Crate> = crates.iter().copied().collect();
    reloaded.crates_gone = old_crates.keys().filter(|krate| !now.contains(krate)).count();
    // A text write: its root. A removed file's root is the old one, which is
    // where its crate's other files still are (or a retired root).
    for &(file, _) in &text_files {
        changed_roots.insert(db.file_source_root(file).source_root_id(db));
    }

    // The word index and the table of uses follow the project files.
    let vfs = read_vfs(&ctx.vfs);
    let project_files: HashSet<FileId> =
        position::project_rust_files(db, &vfs, canonical).into_iter().collect();
    let written: HashSet<FileId> = text_files.iter().map(|&(file, _)| file).collect();
    let stale_roots = roots_depending_on(db, &changed_roots);
    {
        let mut words = ctx.words.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut uses = ctx.uses.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        for &file in old_project_files.difference(&project_files) {
            words.set_file(file, None);
            uses.forget(file);
        }
        reloaded.uses_widened = uses.mark_pending_where(|file| {
            stale_roots.contains(&db.file_source_root(file).source_root_id(db))
        });
        for &file in &project_files {
            if written.contains(&file) || !old_project_files.contains(&file) {
                words.set_file(file, Some(&db.file_text(file).text(db)));
                uses.mark_pending(file);
            }
        }
        reloaded.uses_pending = uses.size().1;
    }
    drop(vfs);
    ctx.exposure = exposure_by_root(db);
    ctx.layout = loader::SourceLayout::reloaded(
        folders.source_root_config,
        folders.load,
        canonical,
        manifests,
        root_ids,
        next_root_id,
    );
    reloaded.apply = applying.elapsed();

    let priming = Instant::now();
    loader::prime(ctx.host.raw_database_mut());
    reloaded.prime = priming.elapsed();
    Ok(reloaded)
}
