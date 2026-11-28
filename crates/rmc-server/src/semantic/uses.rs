//! Where the project uses each library declaration — a stored answer to "the
//! references of `std::thread::spawn`", so that the query does not resolve
//! every occurrence of the name across the workspace (`library-refs-scope` of
//! the plan).
//!
//! # What a row is
//!
//! A use is an occurrence of a word in code of a project file (the word index's
//! occurrences, [`code_words`]) whose `goto_definition` lands in a library file;
//! the row is filed under each such target's key (`position::nav_key`: the
//! library file and the offset of the declaration's name). That is exactly what
//! the live sweep of `position::collect_library_refs` computes per occurrence of
//! the wanted name, done once for every name — so the rows of a file answer the
//! same as sweeping it, by construction.
//!
//! Library files do not change while a context lives, but for a manifest or
//! lockfile change — reloaded into the same database (`super::reload_in_place`),
//! which keeps a file ready only if no crate it may resolve through changed:
//! files of a changed library's users are pending. So a key of a ready file
//! stays valid for the life of the table.
//!
//! # Ready and pending
//!
//! Every project file is either **ready** — its rows describe its text as the
//! database holds it, *and* what that text resolves to — or **pending**: the
//! table says nothing about it and a query sweeps it live. A load starts with
//! every file pending; a file becomes ready when [`library_uses_in`] has been
//! run on it and the result stored ([`UseTable::set_ready`]).
//!
//! What an occurrence resolves to depends on more than its own file: `.map` on
//! a value whose type comes from another crate's signature. So a refresh does
//! not get to keep a file ready just because its text is unchanged. What it
//! does (`apply_tree_change`): an edited file is pending; if its **surface**
//! changed (`super::surface` — the file without its private fn bodies), or a
//! file was added or removed (the module tree may change), the files its
//! changed items reach are pending too ([`UseTable::mark_pending_where`]):
//! those naming them, or holding a value of a type whose impl or alias
//! changed (`super::widening`), within the crates whose root is in that file's
//! source root and all their transitive reverse dependencies — and all of
//! those when the names cannot tell (a changed macro definition, a public
//! glob). An edit inside a body therefore costs one file; a new function in a
//! shared crate costs the files that call it.
//!
//! # Cost
//!
//! Building: one `goto_definition` per word in code — ≈ 100 ms a file on a
//! 317-file workspace measured with a plain classification pass
//! (`use_census`), ≈ 11 min on one core for a ~8000-file workspace; done once
//! per load, never per query, by the background builder
//! (`SemanticService::build_uses_batch`) in batches of
//! [`library_uses_of_files`]. Size: about a quarter of the identifiers are
//! uses of a library item (63.7 k of 252 k there), 8 bytes a row plus a
//! sorted list of files per key — ≈ 1.3 M rows, ≈ 16 MB on ~8000 files.
//! Query: the ready files holding the key, read from the table, plus the live
//! sweep of the pending ones only.

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ra_ap_ide::{Analysis, FilePosition, TextSize};
use ra_ap_ide_db::base_db::salsa::Cancelled;
use ra_ap_vfs::FileId;

use super::position::{goto_config, nav_key};
use super::word_index::code_words;

/// A library declaration: its file and the offset of its name.
pub(crate) type DeclKey = (FileId, TextSize);

/// Library declaration → the ready project files using it → where; see the
/// module header.
#[derive(Default)]
pub(crate) struct UseTable {
    ids: HashMap<DeclKey, u32>,
    /// By key id: the ready files with a use of it; sorted, no duplicates.
    files_of: Vec<Vec<FileId>>,
    /// By ready file: its uses as (key id, word start), in text order. A ready
    /// file without uses is here with an empty list.
    uses_of: HashMap<FileId, Vec<(u32, TextSize)>>,
    /// Project files the table does not describe.
    pending: BTreeSet<FileId>,
    /// Bumped by every invalidation ([`Self::mark_pending`],
    /// [`Self::mark_pending_where`], [`Self::forget`]). A builder computes a
    /// batch on a snapshot taken at one generation; see [`Self::marked_at`].
    generation: u64,
    /// By file: the generation at which it was last made pending (a load: 0)
    /// — every pending file has one. A batch computed at generation `g` is
    /// stored file by file ([`Self::store_batch`]): a file marked after `g`
    /// is refused — a refresh between the snapshot and the store changed its
    /// text or what it may resolve to, and its rows describe the old ones.
    /// A file the refresh did not mark is stored: that refresh leaves its
    /// resolution as it was, by the same rule that keeps a ready file ready
    /// through it (module header). Refusing the whole batch instead made a
    /// batch longer than the time between two refreshes — a heavy file, on a
    /// workspace edited every minute — never stored, and taken again and again.
    marked_at: HashMap<FileId, u64>,
    /// By file: how many batches a refresh cut while this file was being
    /// built. The builder takes such a file after the others of its exposure
    /// ([`Self::next_pending`]), so that one file heavier than the time
    /// between two refreshes does not hold up the rest; the memos it got
    /// through stay in the database, so each new attempt goes further.
    /// Dropped once the file is ready.
    interruptions: HashMap<FileId, u32>,
    /// When the table last went from "every file ready" (or a load) to having
    /// pending files: what the builder's "built in T s" line measures from.
    /// `None` while nothing is pending.
    building_since: Option<Instant>,
}

/// How many local source roots can make a file pending by a surface change:
/// the roots holding the crates its crates transitively depend on, its own
/// root included — what `apply_tree_change` widens a surface change to, read
/// from the other side (`roots_depending_on`).
///
/// The builder takes the least exposed files first. A ready file survives
/// every surface change outside its exposure, so building the leaves of the
/// crate graph first keeps what is built: on a busy workspace a surface
/// change lands every few minutes somewhere, and top-level application crates
/// depend on nearly every root — built first (`FileId` order puts them first
/// on one workspace measured), they were made pending again by almost every
/// refresh, and the table never grew (plan item `uses-bur`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Exposure(pub u32);

/// What [`UseTable::store_batch`] did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Stored {
    /// Files made ready.
    pub(crate) stored: usize,
    /// Files refused: made pending (or forgotten) since the batch's snapshot;
    /// they stay pending (or gone).
    pub(crate) refused: usize,
    /// The batch made the last pending file ready: how long the table took
    /// since it last had pending files.
    pub(crate) completed: Option<Duration>,
}

impl UseTable {
    /// A table describing none of `files` yet — what a load starts with.
    pub(crate) fn all_pending(files: impl IntoIterator<Item = FileId>) -> Self {
        let pending: BTreeSet<FileId> = files.into_iter().collect();
        Self {
            building_since: (!pending.is_empty()).then(Instant::now),
            marked_at: pending.iter().map(|&file| (file, 0)).collect(),
            pending,
            ..Self::default()
        }
    }

    /// See [`Self::generation`](UseTable#structfield.generation).
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Up to `count` pending files, least exposed first; among equals the less
    /// often interrupted ([`Self::interruptions`]), then by `FileId`: the
    /// builder's next batch. `exposure` — of each pending file's root
    /// ([`Exposure`]).
    ///
    /// Cost: `exposure` once per pending file and a partial sort of them —
    /// ≈ 9000 on a ~8000-file workspace just loaded, well under a millisecond
    /// against a batch of ≈ 1 s.
    pub(crate) fn next_pending(
        &self,
        count: usize,
        mut exposure: impl FnMut(FileId) -> Exposure,
    ) -> Vec<FileId> {
        let mut ranked: Vec<(Exposure, u32, FileId)> = self
            .pending
            .iter()
            .map(|&file| {
                let cut = self.interruptions.get(&file).copied().unwrap_or(0);
                (exposure(file), cut, file)
            })
            .collect();
        if count < ranked.len() {
            ranked.select_nth_unstable(count);
            ranked.truncate(count);
        }
        ranked.sort_unstable();
        ranked.into_iter().map(|(_, _, file)| file).collect()
    }

    /// Make ready the files of a batch whose uses were computed at
    /// `generation` and that nothing made pending since; see
    /// [`Self::marked_at`](UseTable#structfield.marked_at).
    pub(crate) fn store_batch(
        &mut self,
        generation: u64,
        batch: Vec<(FileId, Vec<(DeclKey, TextSize)>)>,
    ) -> Stored {
        let had_pending = !self.pending.is_empty();
        let mut stored = 0;
        let mut refused = 0;
        for (file, uses) in batch {
            // Not pending: forgotten since (a ready file is not in a batch).
            let current = self.pending.contains(&file)
                && self.marked_at[&file] <= generation;
            if current {
                self.set_ready(file, uses);
                stored += 1;
            } else {
                refused += 1;
            }
        }
        let completed = if had_pending && self.pending.is_empty() {
            self.building_since.take().map(|since| since.elapsed())
        } else {
            None
        };
        Stored { stored, refused, completed }
    }

    /// Note that a refresh cut a batch while `file` was being built; returns
    /// how many times that has happened to it. See
    /// [`Self::interruptions`](UseTable#structfield.interruptions).
    pub(crate) fn note_interrupted(&mut self, file: FileId) -> u32 {
        let cut = self.interruptions.entry(file).or_default();
        *cut += 1;
        *cut
    }

    /// Note that a file became pending (or might have); see `building_since`.
    fn invalidated(&mut self) {
        self.generation += 1;
        if self.pending.is_empty() {
            self.building_since = None;
        } else if self.building_since.is_none() {
            self.building_since = Some(Instant::now());
        }
    }

    /// Whether a query has to sweep `file` live.
    pub(crate) fn is_pending(&self, file: FileId) -> bool {
        self.pending.contains(&file)
    }

    /// The project files a query has to sweep live, in `FileId` order.
    pub(crate) fn pending(&self) -> &BTreeSet<FileId> {
        &self.pending
    }

    /// Store the uses of `file` — [`library_uses_in`] on the text the database
    /// holds — and make it ready.
    pub(crate) fn set_ready(&mut self, file: FileId, uses: Vec<(DeclKey, TextSize)>) {
        self.drop_rows(file);
        self.pending.remove(&file);
        self.interruptions.remove(&file);
        let mut rows: Vec<(u32, TextSize)> = uses
            .into_iter()
            .map(|(key, start)| (self.intern(key), start))
            .collect();
        rows.sort_unstable_by_key(|&(id, start)| (start, id));
        rows.dedup();
        let mut ids: Vec<u32> = rows.iter().map(|&(id, _)| id).collect();
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let files = &mut self.files_of[id as usize];
            if let Err(at) = files.binary_search(&file) {
                files.insert(at, file);
            }
        }
        self.uses_of.insert(file, rows);
    }

    /// Make `file` pending: it changed, or what it resolves to may have.
    pub(crate) fn mark_pending(&mut self, file: FileId) {
        self.mark(file);
        self.invalidated();
    }

    /// Make `file` pending as of the generation the next [`Self::invalidated`]
    /// moves to — pending already or not: a batch in flight may be building
    /// it on what it was before.
    fn mark(&mut self, file: FileId) {
        self.drop_rows(file);
        self.pending.insert(file);
        self.marked_at.insert(file, self.generation + 1);
    }

    /// Make every file pending — what a refresh did before it could tell which
    /// ready files a change leaves describing the truth; the tests' blunt
    /// invalidation now.
    ///
    /// Bumps the generation even when every file already was pending: a batch
    /// in flight was computed on the texts before the change.
    #[cfg(test)]
    pub(crate) fn mark_all_pending(&mut self) {
        self.mark_pending_where(|_| true);
    }

    /// Make pending every file `stale` picks — the files a surface change may
    /// have re-resolved (`apply_tree_change`). Pending files are asked too: a
    /// batch in flight may be building one of them on what it resolved to
    /// before ([`Self::marked_at`](UseTable#structfield.marked_at)). One
    /// generation bump for the lot.
    ///
    /// Cost: `stale` once per file of the table (≈ 9000 on a ~8000-file
    /// workspace, a root lookup each — under a millisecond). Returns how many
    /// ready files it made pending.
    pub(crate) fn mark_pending_where(&mut self, mut stale: impl FnMut(FileId) -> bool) -> usize {
        let ready: Vec<FileId> = self.uses_of.keys().copied().filter(|&file| stale(file)).collect();
        let pending: Vec<FileId> = self.pending.iter().copied().filter(|&file| stale(file)).collect();
        let count = ready.len();
        for file in ready.into_iter().chain(pending) {
            self.mark(file);
        }
        self.invalidated();
        count
    }

    /// Every row of the ready files, as (file, word start, key), sorted — what
    /// a test compares with a table built from scratch.
    #[cfg(test)]
    pub(crate) fn rows(&self) -> Vec<(FileId, TextSize, DeclKey)> {
        let keys: HashMap<u32, DeclKey> = self.ids.iter().map(|(&key, &id)| (id, key)).collect();
        let mut rows: Vec<_> = self
            .uses_of
            .iter()
            .flat_map(|(&file, uses)| uses.iter().map(move |&(id, start)| (file, start, id)))
            .map(|(file, start, id)| (file, start, keys[&id]))
            .collect();
        rows.sort();
        rows
    }

    /// `file` left the analysis: neither ready nor pending.
    pub(crate) fn forget(&mut self, file: FileId) {
        self.drop_rows(file);
        self.pending.remove(&file);
        self.marked_at.remove(&file);
        self.interruptions.remove(&file);
        self.invalidated();
    }

    /// The uses of `key` in ready files, by file in `FileId` order, each file's
    /// word starts in text order.
    pub(crate) fn uses(&self, key: DeclKey) -> Vec<(FileId, Vec<TextSize>)> {
        let Some(&id) = self.ids.get(&key) else {
            return Vec::new();
        };
        self.files_of[id as usize]
            .iter()
            .map(|file| {
                let starts = self.uses_of[file]
                    .iter()
                    .filter(|&&(of, _)| of == id)
                    .map(|&(_, start)| start)
                    .collect();
                (*file, starts)
            })
            .collect()
    }

    /// Ready files, pending files and rows — for log lines.
    pub(crate) fn size(&self) -> (usize, usize, usize) {
        (
            self.uses_of.len(),
            self.pending.len(),
            self.uses_of.values().map(Vec::len).sum(),
        )
    }

    fn drop_rows(&mut self, file: FileId) {
        let Some(rows) = self.uses_of.remove(&file) else {
            return;
        };
        for (id, _) in rows {
            let files = &mut self.files_of[id as usize];
            if let Ok(at) = files.binary_search(&file) {
                files.remove(at);
            }
        }
    }

    fn intern(&mut self, key: DeclKey) -> u32 {
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = u32::try_from(self.files_of.len()).expect("fewer than 2^32 declarations");
        self.ids.insert(key, id);
        self.files_of.push(Vec::new());
        id
    }
}

/// The uses of library declarations in project file `file`: for every word in
/// code, the keys of the library declarations its `goto_definition` lands on.
///
/// Probes each word one character into it, as the live sweep does (an empty
/// answer at a field's first byte, see `collect_by_token_scan`).
///
/// Cost: one `goto_definition` per word in code of the file, plus one
/// `is_library_file` per target found.
pub(crate) fn library_uses_in(
    analysis: &Analysis,
    file: FileId,
) -> Result<Vec<(DeclKey, TextSize)>> {
    let text = analysis.file_text(file)?;
    let mut uses = Vec::new();
    for (start, word) in code_words(&text) {
        let inside = if word.len() > 1 { start + 1 } else { start };
        let position = FilePosition {
            file_id: file,
            offset: TextSize::new(inside as u32),
        };
        let Some(info) = analysis
            .goto_definition(position, &goto_config())
            .context("goto_definition query failed")?
        else {
            continue;
        };
        for target in info.info {
            if analysis
                .is_library_file(target.file_id)
                .context("is_library_file query failed")?
            {
                uses.push((nav_key(&target), TextSize::new(start as u32)));
            }
        }
    }
    Ok(uses)
}

/// What one batch of [`library_uses_of_files`] got through.
#[derive(Debug, Default)]
pub(crate) struct BatchUses {
    /// The files built, with their uses.
    pub(crate) done: Vec<(FileId, Vec<(DeclKey, TextSize)>)>,
    /// The files a cancellation cut — at most one a thread; their work is
    /// lost (the memos they completed are not) and they stay pending.
    pub(crate) interrupted: Vec<FileId>,
    /// The slowest file built and how long it took — for the log.
    pub(crate) slowest: Option<(FileId, Duration)>,
}

/// The uses of one batch of files: [`library_uses_in`] on each, the files dealt
/// round-robin to one thread per snapshot in `analyses` (all of one database).
///
/// A write to the database (a refresh, a reload) cancels the snapshots: each
/// thread stops at the file it is on ([`BatchUses::interrupted`]) and lets go
/// of its snapshot at once, which is what the write waits for; the files it
/// built before are returned, for [`UseTable::store_batch`] to sort out which
/// the write left current. The files it never reached stay pending.
///
/// Threads get `stack_bytes` each, the analysis thread's stack: inference on
/// a body recurses over its shape, as on the thread that called this.
///
/// Cost: the sum of [`library_uses_in`] over `files`, divided among the
/// threads; on a ~8000-file workspace ≈ 100 ms a file on one core on
/// average, but one file can take minutes.
pub(crate) fn library_uses_of_files(
    analyses: Vec<Analysis>,
    files: &[FileId],
    stack_bytes: usize,
) -> BatchUses {
    let threads = analyses.len();
    assert!(threads > 0, "a batch needs at least one snapshot");
    // Thread `t` takes files `t`, `t + threads`, … — neighbours in `FileId`
    // order are often neighbours in a crate, and a crate's first file pays its
    // def map for the rest; dealing them out spreads that cost.
    let parts: Vec<BatchUses> = std::thread::scope(|scope| {
        let handles: Vec<_> = analyses
            .into_iter()
            .enumerate()
            .map(|(thread, analysis)| {
                std::thread::Builder::new()
                    .name("rmc-uses".to_string())
                    .stack_size(stack_bytes)
                    .spawn_scoped(scope, move || {
                        let mut mine = BatchUses::default();
                        for &file in files.iter().skip(thread).step_by(threads) {
                            let started = Instant::now();
                            let outcome = Cancelled::catch(std::panic::AssertUnwindSafe(|| {
                                library_uses_in(&analysis, file)
                            }));
                            match outcome {
                                Ok(Ok(uses)) => {
                                    let took = started.elapsed();
                                    if mine.slowest.is_none_or(|(_, slowest)| took > slowest) {
                                        mine.slowest = Some((file, took));
                                    }
                                    mine.done.push((file, uses));
                                }
                                Ok(Err(error)) => {
                                    // Every query in it is `Cancellable`: an
                                    // error is a cancellation carried by `?`.
                                    if !super::is_cancellation(&error) {
                                        tracing::warn!("table of uses: {error:#}");
                                    }
                                    mine.interrupted.push(file);
                                    return mine;
                                }
                                Err(_cancelled) => {
                                    mine.interrupted.push(file);
                                    return mine;
                                }
                            }
                        }
                        mine
                    })
                    .expect("spawn a thread of the uses builder")
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(found) => found,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    });
    let mut batch = BatchUses::default();
    for part in parts {
        batch.done.extend(part.done);
        batch.interrupted.extend(part.interrupted);
        if let Some((file, took)) = part.slowest
            && batch.slowest.is_none_or(|(_, slowest)| took > slowest)
        {
            batch.slowest = Some((file, took));
        }
    }
    batch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(id: u32) -> FileId {
        FileId::from_raw(id)
    }

    fn key(file_id: u32, at: u32) -> DeclKey {
        (file(file_id), TextSize::new(at))
    }

    #[test]
    fn a_file_is_pending_until_its_uses_are_set_and_again_once_marked() {
        let mut table = UseTable::all_pending([file(1), file(2)]);
        assert!(table.is_pending(file(1)) && table.is_pending(file(2)));
        assert!(table.uses(key(9, 4)).is_empty());

        table.set_ready(
            file(2),
            vec![
                (key(9, 4), TextSize::new(30)),
                (key(9, 4), TextSize::new(10)),
            ],
        );
        table.set_ready(
            file(1),
            vec![(key(9, 4), TextSize::new(5)), (key(9, 8), TextSize::new(7))],
        );
        assert!(table.pending().is_empty());
        assert_eq!(
            table.uses(key(9, 4)),
            vec![
                (file(1), vec![TextSize::new(5)]),
                (file(2), vec![TextSize::new(10), TextSize::new(30)]),
            ]
        );
        assert_eq!(table.size(), (2, 0, 4));

        table.mark_pending(file(2));
        assert!(table.is_pending(file(2)));
        assert_eq!(
            table.uses(key(9, 4)),
            vec![(file(1), vec![TextSize::new(5)])]
        );

        // Setting a file again replaces its rows rather than adding to them.
        table.set_ready(file(1), vec![(key(9, 8), TextSize::new(7))]);
        assert!(table.uses(key(9, 4)).is_empty());
        assert_eq!(
            table.uses(key(9, 8)),
            vec![(file(1), vec![TextSize::new(7)])]
        );

        table.mark_all_pending();
        assert_eq!(table.size(), (0, 2, 0));
        table.forget(file(1));
        assert_eq!(
            table.pending().iter().copied().collect::<Vec<_>>(),
            vec![file(2)]
        );
    }

    /// The builder's batch is the least exposed pending files, `FileId` among
    /// equals — whatever the `FileId` order of the exposures.
    #[test]
    fn the_next_batch_is_the_least_exposed_pending_files() {
        let table = UseTable::all_pending((1..=6).map(file));
        let exposure = |f: FileId| Exposure([0, 5, 1, 5, 1, 0, 9][f.index() as usize]);
        assert_eq!(table.next_pending(3, exposure), vec![file(5), file(2), file(4)]);
        assert_eq!(
            table.next_pending(10, exposure),
            vec![file(5), file(2), file(4), file(1), file(3), file(6)]
        );
        assert_eq!(table.next_pending(0, exposure), Vec::<FileId>::new());
    }

    /// The files of a batch computed before an invalidation that the
    /// invalidation made pending are not stored: their rows describe texts (or
    /// what they resolved to) that the database no longer holds. They stay
    /// pending and the next batch takes them again — an edit during the build
    /// loses no file. A file the invalidation left alone is stored: that
    /// refresh does not change what it resolves to. A pending file a surface
    /// change reaches counts as made pending, though it already was: the batch
    /// may have been building it on the old resolution.
    #[test]
    fn a_batch_from_before_an_invalidation_keeps_only_the_files_it_left_alone() {
        let mut table = UseTable::all_pending([file(1), file(2), file(3)]);
        let row = || vec![(key(9, 4), TextSize::new(5))];
        let before = table.generation();
        table.mark_all_pending();
        assert_eq!(
            table.store_batch(before, vec![(file(1), row())]),
            Stored { stored: 0, refused: 1, completed: None }
        );
        assert!(table.is_pending(file(1)), "the refused file stays pending");
        assert!(table.uses(key(9, 4)).is_empty());

        let now = table.generation();
        // An edit of file 1 and a surface change reaching file 3, both pending,
        // land while the batch builds all three.
        table.mark_pending(file(1));
        table.mark_pending_where(|f| f == file(3));
        assert_eq!(
            table.store_batch(now, vec![(file(1), row()), (file(2), row()), (file(3), row())]),
            Stored { stored: 1, refused: 2, completed: None },
            "only the file nothing touched is stored"
        );
        assert!(!table.is_pending(file(2)));
        assert!(table.is_pending(file(1)) && table.is_pending(file(3)));

        // A forgotten file is refused, whatever the generation.
        let now = table.generation();
        table.forget(file(3));
        assert_eq!(
            table.store_batch(now, vec![(file(3), row())]),
            Stored { stored: 0, refused: 1, completed: None }
        );

        let now = table.generation();
        let stored = table.store_batch(now, vec![(file(1), row())]);
        assert_eq!((stored.stored, stored.refused), (1, 0));
        assert!(stored.completed.is_some(), "the last pending file made the table complete");
        assert!(table.pending().is_empty());
        assert_eq!(
            table.uses(key(9, 4)),
            vec![(file(1), vec![TextSize::new(5)]), (file(2), vec![TextSize::new(5)])]
        );
    }

    /// A file a refresh cut goes behind the other files of its exposure, not
    /// behind more exposed ones, and is back in place once ready.
    #[test]
    fn an_interrupted_file_goes_behind_the_others_of_its_exposure() {
        let mut table = UseTable::all_pending((1..=4).map(file));
        let exposure = |f: FileId| Exposure([0, 1, 1, 1, 2][f.index() as usize]);
        assert_eq!(table.next_pending(4, exposure), vec![file(1), file(2), file(3), file(4)]);
        assert_eq!(table.note_interrupted(file(1)), 1);
        assert_eq!(table.note_interrupted(file(1)), 2);
        assert_eq!(table.note_interrupted(file(2)), 1);
        assert_eq!(table.next_pending(4, exposure), vec![file(3), file(2), file(1), file(4)]);

        table.set_ready(file(1), Vec::new());
        table.mark_pending(file(1));
        assert_eq!(table.next_pending(2, exposure), vec![file(1), file(3)]);
    }
}
