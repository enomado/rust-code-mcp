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
//! not get to keep a file ready just because its text is unchanged — for now
//! every change makes every file pending again ([`UseTable::mark_all_pending`]);
//! narrowing that to the changed file and what depends on its surface is the
//! plan's `uses-freshness`.
//!
//! # Cost
//!
//! Building: one `goto_definition` per word in code — ≈ 100 ms a file on a
//! 317-file workspace measured with a plain classification pass
//! (`use_census`), ≈ 11 min on one core for a ~8000-file workspace; done once
//! per load, never per query. Size: about a quarter of the identifiers are
//! uses of a library item (63.7 k of 252 k there), 8 bytes a row plus a
//! sorted list of files per key — ≈ 1.3 M rows, ≈ 16 MB on ~8000 files.
//! Query: the ready files holding the key, read from the table, plus the live
//! sweep of the pending ones only.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Context, Result};
use ra_ap_ide::{Analysis, FilePosition, TextSize};
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
}

impl UseTable {
    /// A table describing none of `files` yet — what a load starts with.
    pub(crate) fn all_pending(files: impl IntoIterator<Item = FileId>) -> Self {
        Self {
            pending: files.into_iter().collect(),
            ..Self::default()
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
    }

    /// Make every file pending — what a refresh does until the table can tell
    /// which ready files a change leaves describing the truth (module header).
    pub(crate) fn mark_all_pending(&mut self) {
        let ready: Vec<FileId> = self.uses_of.keys().copied().collect();
        for file in ready {
            self.mark_pending(file);
        }
    }

    /// `file` left the analysis: neither ready nor pending.
    pub(crate) fn forget(&mut self, file: FileId) {
        self.drop_rows(file);
        self.pending.remove(&file);
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
// Served code calls it once the background builder lands (`uses-builder` of
// the plan); until then only tests fill the table.
#[cfg_attr(not(test), allow(dead_code))]
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
}
