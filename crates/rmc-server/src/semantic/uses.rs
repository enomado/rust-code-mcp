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
//! Library files do not change while a context lives (a manifest or lockfile
//! change reloads it, and a reload builds a new table), so a key stays valid
//! for the life of the table.
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
//! file was added or removed (the module tree may change), every file of the
//! crates whose root is in that file's source root, and of all their
//! transitive reverse dependencies, is pending too
//! ([`UseTable::mark_pending_where`]). An edit inside a body therefore costs
//! one file; an edit of a signature in a leaf crate costs that crate.
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
    /// [`Self::mark_all_pending`], [`Self::forget`]). A builder computes a
    /// batch on a snapshot taken at one generation and stores it only if the
    /// table is still at that generation ([`Self::store_batch`]): otherwise a
    /// refresh landed in between, and rows of the old texts — or of what they
    /// resolved to before the change — would be filed as ready.
    generation: u64,
    /// When the table last went from "every file ready" (or a load) to having
    /// pending files: what the builder's "built in T s" line measures from.
    /// `None` while nothing is pending.
    building_since: Option<Instant>,
}

/// What [`UseTable::store_batch`] did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Stored {
    /// The table moved on since the batch's snapshot; nothing was stored and
    /// the batch's files stay pending.
    Stale,
    /// The files are ready. `completed` — the batch made the last pending
    /// file ready; how long the table took since it last had pending files.
    Ready { completed: Option<Duration> },
}

impl UseTable {
    /// A table describing none of `files` yet — what a load starts with.
    pub(crate) fn all_pending(files: impl IntoIterator<Item = FileId>) -> Self {
        let pending: BTreeSet<FileId> = files.into_iter().collect();
        Self {
            building_since: (!pending.is_empty()).then(Instant::now),
            pending,
            ..Self::default()
        }
    }

    /// See [`Self::generation`](UseTable#structfield.generation).
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Up to `count` pending files, in `FileId` order: the builder's next batch.
    pub(crate) fn next_pending(&self, count: usize) -> Vec<FileId> {
        self.pending.iter().copied().take(count).collect()
    }

    /// Make the files of a batch ready — if the table is still at `generation`,
    /// the one their uses were computed at; see
    /// [`Self::generation`](UseTable#structfield.generation).
    pub(crate) fn store_batch(
        &mut self,
        generation: u64,
        batch: Vec<(FileId, Vec<(DeclKey, TextSize)>)>,
    ) -> Stored {
        if generation != self.generation {
            return Stored::Stale;
        }
        let had_pending = !self.pending.is_empty();
        for (file, uses) in batch {
            // A file forgotten since cannot be here: forgetting bumps the
            // generation. So every file of the batch is still pending.
            self.set_ready(file, uses);
        }
        let completed = if had_pending && self.pending.is_empty() {
            self.building_since.take().map(|since| since.elapsed())
        } else {
            None
        };
        Stored::Ready { completed }
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
        self.drop_rows(file);
        self.pending.insert(file);
        self.invalidated();
    }

    /// Make every file pending — what a refresh did before it could tell which
    /// ready files a change leaves describing the truth; the tests' blunt
    /// invalidation now.
    ///
    /// Bumps the generation even when every file already was pending: a batch
    /// in flight was computed on the texts before the change.
    #[cfg(test)]
    pub(crate) fn mark_all_pending(&mut self) {
        let ready: Vec<FileId> = self.uses_of.keys().copied().collect();
        for file in ready {
            self.drop_rows(file);
            self.pending.insert(file);
        }
        self.invalidated();
    }

    /// Make pending every ready file `stale` picks — the files a surface change
    /// may have re-resolved (`apply_tree_change`). One generation bump for the
    /// lot, even when it picks none: the caller marks the changed files right
    /// after, and a batch in flight is stale either way.
    ///
    /// Cost: `stale` once per ready file. Returns how many it made pending.
    pub(crate) fn mark_pending_where(&mut self, mut stale: impl FnMut(FileId) -> bool) -> usize {
        let picked: Vec<FileId> = self.uses_of.keys().copied().filter(|&file| stale(file)).collect();
        let count = picked.len();
        for file in picked {
            self.drop_rows(file);
            self.pending.insert(file);
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

/// The uses of one batch of files: [`library_uses_in`] on each, the files dealt
/// round-robin to one thread per snapshot in `analyses` (all of one database).
///
/// `None` — a write to the database (a refresh, a reload) cancelled the
/// snapshots: none of the batch is kept, its files stay pending, and the
/// threads let go of their snapshots at once, which is what the write waits
/// for.
///
/// Threads get `stack_bytes` each, the analysis thread's stack: inference on
/// a body recurses over its shape, as on the thread that called this.
///
/// Cost: the sum of [`library_uses_in`] over `files`, divided among the
/// threads; on a ~8000-file workspace ≈ 100 ms a file on one core.
pub(crate) fn library_uses_of_files(
    analyses: Vec<Analysis>,
    files: &[FileId],
    stack_bytes: usize,
) -> Option<Vec<(FileId, Vec<(DeclKey, TextSize)>)>> {
    let threads = analyses.len();
    assert!(threads > 0, "a batch needs at least one snapshot");
    // Thread `t` takes files `t`, `t + threads`, … — neighbours in `FileId`
    // order are often neighbours in a crate, and a crate's first file pays its
    // def map for the rest; dealing them out spreads that cost.
    let found: Vec<Option<Vec<(FileId, Vec<(DeclKey, TextSize)>)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = analyses
            .into_iter()
            .enumerate()
            .map(|(thread, analysis)| {
                std::thread::Builder::new()
                    .name("rmc-uses".to_string())
                    .stack_size(stack_bytes)
                    .spawn_scoped(scope, move || {
                        let mut mine = Vec::new();
                        for &file in files.iter().skip(thread).step_by(threads) {
                            let outcome = Cancelled::catch(std::panic::AssertUnwindSafe(|| {
                                library_uses_in(&analysis, file)
                            }));
                            match outcome {
                                Ok(Ok(uses)) => mine.push((file, uses)),
                                Ok(Err(error)) => {
                                    // Every query in it is `Cancellable`: an
                                    // error is a cancellation carried by `?`.
                                    if !super::is_cancellation(&error) {
                                        tracing::warn!("table of uses: {error:#}");
                                    }
                                    return None;
                                }
                                Err(_cancelled) => return None,
                            }
                        }
                        Some(mine)
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
    let mut batch = Vec::with_capacity(files.len());
    for part in found {
        batch.extend(part?);
    }
    Some(batch)
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

    /// A batch computed before an invalidation is not stored: its rows describe
    /// texts (or what they resolved to) that the database no longer holds. The
    /// file stays pending and the next batch takes it again — an edit during
    /// the build loses no file. Even `mark_all_pending` with nothing ready
    /// moves the table on: what was in flight is just as stale.
    #[test]
    fn a_batch_from_before_an_invalidation_is_not_stored() {
        let mut table = UseTable::all_pending([file(1), file(2)]);
        assert_eq!(table.next_pending(1), vec![file(1)]);

        let before = table.generation();
        table.mark_all_pending();
        assert_eq!(
            table.store_batch(before, vec![(file(1), vec![(key(9, 4), TextSize::new(5))])]),
            Stored::Stale
        );
        assert!(table.is_pending(file(1)), "the stale batch's file stays pending");
        assert!(table.uses(key(9, 4)).is_empty());

        let now = table.generation();
        assert_eq!(
            table.store_batch(now, vec![(file(1), vec![(key(9, 4), TextSize::new(5))])]),
            Stored::Ready { completed: None },
            "one of two files: not complete yet"
        );
        table.mark_pending(file(1));
        assert!(matches!(
            table.store_batch(now, vec![(file(2), Vec::new())]),
            Stored::Stale
        ));
        let now = table.generation();
        let Stored::Ready { completed } = table.store_batch(
            now,
            vec![(file(1), vec![(key(9, 4), TextSize::new(5))]), (file(2), Vec::new())],
        ) else {
            panic!("a current batch is stored");
        };
        assert!(completed.is_some(), "the last pending file made the table complete");
        assert!(table.pending().is_empty());
        assert_eq!(table.uses(key(9, 4)), vec![(file(1), vec![TextSize::new(5)])]);
    }
}
