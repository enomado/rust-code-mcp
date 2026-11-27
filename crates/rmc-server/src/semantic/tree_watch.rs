//! Staleness without a walk: what changed in a project's tree since the last
//! look, from the kernel's change notifications (plan item `stamps-watcher`).
//!
//! # Why
//!
//! Every query used to take the project's stamps afresh ([`super::collect_stamps`]:
//! a walk of the tree and a `stat` of every `*.rs` and manifest) under the
//! project's slot lock, so ~30 clients of one large workspace queued behind
//! each other's walks — p50 0.15 s, p90 0.55 s per call on a ~8000-file
//! workspace before any analysis ran.
//!
//! # How
//!
//! The walk is done once, at load: [`TreeWatch::start`] puts an inotify watch
//! on every directory the walker enters — before the directory is listed
//! ([`rmc_indexing::indexing::walk_project_files`]), so a file created there is
//! in the listing or raises an event — and stamps the files it finds. A look
//! ([`TreeWatch::current`]) then reads the queued events without blocking and
//! re-stamps only what they name: a file event re-stamps the file, an event
//! that can change which files the walker sees (a directory created, removed or
//! moved, a nested-root marker added or removed) walks that directory again.
//! No event: nothing changed, and the look is one `read` returning `EAGAIN`.
//!
//! # Why reading the fd in the query is enough
//!
//! fsnotify queues an event inside the `write`/`rename`/`unlink` that causes
//! it, before the call returns. So every change that completed before the query
//! began is already in the inotify queue when the query reads it: an edit made
//! just before a query is in that query's answer, as with the walk. A reader
//! thread forwarding events over a channel (the `notify` crate) would be a race
//! between the two; this has none.
//!
//! # When it walks the whole tree again
//!
//! The kernel's queue overflowed (`IN_Q_OVERFLOW`, `max_queued_events` —
//! 16384 here — events between two looks), the root itself was moved or
//! deleted, or reading the queue failed: the watcher starts over with a fresh
//! inotify instance and a full walk, and logs why. If watches cannot be set up
//! at all (not Linux, `max_user_watches` spent, a directory that refuses a
//! watch), every look walks, as before the watcher — logged once.
//!
//! # What it does not see
//!
//! Changes no inotify event reports: a file changed through a second hard link
//! in another directory (the event goes to the directory of the path written),
//! a file system shared with another machine (NFS, sshfs), writes through a
//! shared `mmap`. None is how source trees are edited.
//!
//! # Cost on an ~8000-file workspace with ~30 clients
//!
//! Start: the one walk the first look used to do anyway, plus one
//! `inotify_add_watch` per directory (~10 000 directories, a few ms) and their
//! kernel memory (~1 KiB a watch). A look with nothing changed: one `read`
//! syscall. A look after edits: one `stat` per named file, one walk per
//! created or moved directory, and a copy of the stamp map (~9000 entries, ~1
//! ms) for the caller to compare.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::{FileStamp, is_stamped_file};

/// The stamps of a project's tree, kept current by inotify where it can.
pub(super) struct TreeWatch {
    root: PathBuf,
    mode: Mode,
}

enum Mode {
    #[cfg(target_os = "linux")]
    Watching(linux::Watcher),
    /// Every look walks the tree: off Linux, for a context built by hand in a
    /// test, or after the watcher could not be set up (logged once).
    Walking,
}

impl TreeWatch {
    /// Watch `root` and stamp its files, watches first (module header).
    pub(super) fn start(root: &Path) -> (Self, HashMap<PathBuf, FileStamp>) {
        #[cfg(target_os = "linux")]
        match linux::Watcher::start(root) {
            Ok((watcher, stamps)) => {
                tracing::info!(
                    "watching {} for changes: {} directories",
                    root.display(),
                    watcher.directory_count()
                );
                return (
                    Self {
                        root: root.to_path_buf(),
                        mode: Mode::Watching(watcher),
                    },
                    stamps,
                );
            }
            Err(reason) => tracing::warn!(
                "cannot watch {} for changes ({reason}); every query walks the tree",
                root.display()
            ),
        }
        (Self::walking(root), super::collect_stamps(root))
    }

    /// A watch that walks on every look — what a test's hand-built context gets.
    pub(super) fn walking(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            mode: Mode::Walking,
        }
    }

    /// The stamps a walk of the tree would take now, given those of the last
    /// look; `None` when nothing a walk sees has changed since (`stamps` are
    /// current — no copy is made).
    pub(super) fn current(
        &mut self,
        stamps: &HashMap<PathBuf, FileStamp>,
    ) -> Option<HashMap<PathBuf, FileStamp>> {
        match &mut self.mode {
            Mode::Walking => Some(super::collect_stamps(&self.root)),
            #[cfg(target_os = "linux")]
            Mode::Watching(watcher) => match watcher.current(&self.root, stamps) {
                Ok(current) => current,
                Err(reason) => {
                    tracing::info!(
                        "watch of {} lost track ({reason}); walking it again",
                        self.root.display()
                    );
                    let (fresh, stamps) = Self::start(&self.root);
                    *self = fresh;
                    Some(stamps)
                }
            },
        }
    }
}

/// The stamp a walk would record for `path`: a regular file — the walker does
/// not follow symbolic links — that [`is_stamped_file`] keeps.
#[cfg(target_os = "linux")]
fn walked_stamp(path: &Path) -> Option<FileStamp> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if !meta.is_file() || !is_stamped_file(path) {
        return None;
    }
    Some(FileStamp {
        modified: meta.modified().ok()?,
        len: meta.len(),
    })
}

/// Whether the walker of `root` enters `dir`: every directory from `dir` up to
/// (not including) `root` is a real directory with a name the walker does not
/// skip and no nested-root marker. A few `lstat`s.
#[cfg(target_os = "linux")]
fn is_walked_dir(root: &Path, dir: &Path) -> bool {
    use rmc_indexing::indexing::{is_nested_project_root, is_skipped_dir_name};
    if !dir.starts_with(root) {
        return false;
    }
    let mut current = dir;
    while current != root {
        let is_real_dir = std::fs::symlink_metadata(current).is_ok_and(|meta| meta.is_dir());
        let skipped = current
            .file_name()
            .is_some_and(|name| is_skipped_dir_name(&name.to_string_lossy()));
        if !is_real_dir || skipped || is_nested_project_root(current) {
            return false;
        }
        current = current.parent().expect("a path below root has a parent");
    }
    std::fs::symlink_metadata(root).is_ok_and(|meta| meta.is_dir())
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use inotify::{EventMask, Inotify, WatchDescriptor, WatchMask};
    use rmc_indexing::indexing::{NESTED_ROOT_MARKER, WalkedDir, walk_project_files};
    use std::ffi::OsString;

    /// What a directory watch reports. `DONT_FOLLOW`/`ONLYDIR`: the walker does
    /// not follow links either. `EXCL_UNLINK`: nothing from a file once it is
    /// unlinked. No `ACCESS`/`OPEN`/`CLOSE_*` — a stamp is mtime and length,
    /// which only `MODIFY` (write, truncate) and `ATTRIB` (utimes) move.
    fn mask() -> WatchMask {
        WatchMask::MODIFY
            | WatchMask::ATTRIB
            | WatchMask::CREATE
            | WatchMask::DELETE
            | WatchMask::MOVED_FROM
            | WatchMask::MOVED_TO
            | WatchMask::DELETE_SELF
            | WatchMask::MOVE_SELF
            | WatchMask::DONT_FOLLOW
            | WatchMask::ONLYDIR
            | WatchMask::EXCL_UNLINK
    }

    /// One event, copied out of the read buffer.
    struct Seen {
        wd: WatchDescriptor,
        mask: EventMask,
        name: Option<OsString>,
    }

    pub(super) struct Watcher {
        inotify: Inotify,
        /// Each watched directory by its watch, and the other way round. A
        /// watch follows an inode, not a path: a directory moved keeps its
        /// watch, so a path here is only trusted until an event says the
        /// directory (or one above it) moved — then the watches under it are
        /// dropped and set up again by a walk ([`Self::current`]).
        by_wd: HashMap<WatchDescriptor, PathBuf>,
        by_path: HashMap<PathBuf, WatchDescriptor>,
        /// Watches on nested project roots the walk stops at: only their
        /// marker is news — removed, it lets the walk in (those the load found,
        /// or one a marker made since).
        boundaries: HashSet<WatchDescriptor>,
        buffer: Vec<u8>,
    }

    impl Watcher {
        pub(super) fn start(root: &Path) -> Result<(Self, HashMap<PathBuf, FileStamp>), String> {
            let inotify = Inotify::init().map_err(|e| format!("inotify_init1: {e}"))?;
            let mut watcher = Self {
                inotify,
                by_wd: HashMap::new(),
                by_path: HashMap::new(),
                boundaries: HashSet::new(),
                // Room for ~1000 events of short names a read.
                buffer: vec![0; 64 * 1024],
            };
            let stamps = watcher.walk(root, root)?.into_iter().collect();
            Ok((watcher, stamps))
        }

        pub(super) fn directory_count(&self) -> usize {
            self.by_wd.len()
        }

        /// Walk `subtree` of `root` like [`super::super::collect_stamps`],
        /// watching each directory before it is listed, and each nested root
        /// it stops at. A watch that cannot be set up fails the walk: the
        /// directory would go unobserved.
        fn walk(
            &mut self,
            root: &Path,
            subtree: &Path,
        ) -> Result<Vec<(PathBuf, FileStamp)>, String> {
            let mut failure: Option<String> = None;
            let walked = walk_project_files(root, subtree, is_stamped_file, |dir, kind| {
                if failure.is_some() {
                    return;
                }
                match self.inotify.watches().add(dir, mask()) {
                    Ok(wd) => {
                        // Re-adding an inode already watched returns its watch:
                        // the old path of that watch, if any, is no longer it.
                        if let Some(old) = self.by_wd.insert(wd.clone(), dir.to_path_buf()) {
                            if self.by_path.get(&old) == Some(&wd) {
                                self.by_path.remove(&old);
                            }
                        }
                        self.by_path.insert(dir.to_path_buf(), wd.clone());
                        match kind {
                            WalkedDir::Entered => self.boundaries.remove(&wd),
                            WalkedDir::NestedRoot => self.boundaries.insert(wd),
                        };
                    }
                    Err(e) => failure = Some(format!("inotify_add_watch {}: {e}", dir.display())),
                }
            });
            if let Some(failure) = failure {
                return Err(failure);
            }
            let (files, walk_errors) = walked;
            if walk_errors > 0 {
                tracing::warn!(
                    "{walk_errors} unreadable entries while checking {} for staleness",
                    subtree.display()
                );
            }
            Ok(files
                .into_iter()
                .filter_map(|path| super::super::stamp_of(&path).map(|stamp| (path, stamp)))
                .collect())
        }

        /// Every event queued so far, without blocking.
        fn drain(&mut self) -> Result<Vec<Seen>, String> {
            let mut seen = Vec::new();
            loop {
                match self.inotify.read_events(&mut self.buffer) {
                    Ok(events) => {
                        let before = seen.len();
                        seen.extend(events.map(|event| Seen {
                            wd: event.wd,
                            mask: event.mask,
                            name: event.name.map(|name| name.to_os_string()),
                        }));
                        if seen.len() == before {
                            return Ok(seen);
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(seen),
                    Err(e) => return Err(format!("reading inotify events: {e}")),
                }
            }
        }

        /// [`super::TreeWatch::current`]; `Err` when the events cannot be
        /// trusted to name every change — the caller starts over.
        pub(super) fn current(
            &mut self,
            root: &Path,
            stamps: &HashMap<PathBuf, FileStamp>,
        ) -> Result<Option<HashMap<PathBuf, FileStamp>>, String> {
            let mut files: HashSet<PathBuf> = HashSet::new();
            // Directories whose walk may have changed: walked again below.
            let mut dirs: HashSet<PathBuf> = HashSet::new();
            for event in self.drain()? {
                if event.mask.contains(EventMask::Q_OVERFLOW) {
                    return Err("the kernel's event queue overflowed".to_string());
                }
                // A watch dropped below and set up again under a new
                // descriptor: its queued events describe paths the walk that
                // dropped it has already seen.
                let Some(dir) = self.by_wd.get(&event.wd).cloned() else {
                    continue;
                };
                if event.mask.contains(EventMask::IGNORED) {
                    // The kernel dropped the watch (directory deleted, or
                    // `inotify_rm_watch`).
                    self.by_wd.remove(&event.wd);
                    self.boundaries.remove(&event.wd);
                    if self.by_path.get(&dir) == Some(&event.wd) {
                        self.by_path.remove(&dir);
                    }
                    continue;
                }
                if self.boundaries.contains(&event.wd) {
                    // A nested root: its marker is the only news. Its own
                    // move or removal reaches the watch of its parent.
                    if event.name.as_deref() == Some(std::ffi::OsStr::new(NESTED_ROOT_MARKER)) {
                        dirs.insert(dir);
                    }
                    continue;
                }
                if event
                    .mask
                    .intersects(EventMask::DELETE_SELF | EventMask::MOVE_SELF)
                {
                    if dir == root {
                        return Err("the project root was moved or deleted".to_string());
                    }
                    dirs.insert(dir);
                    continue;
                }
                let Some(name) = event.name else { continue };
                let path = dir.join(&name);
                if event.mask.contains(EventMask::ISDIR) {
                    // A directory's own `MODIFY`/`ATTRIB` changes no file the
                    // walk lists; its arrival, removal or move does.
                    if event.mask.intersects(
                        EventMask::CREATE
                            | EventMask::DELETE
                            | EventMask::MOVED_FROM
                            | EventMask::MOVED_TO,
                    ) {
                        dirs.insert(path);
                    }
                } else if name == NESTED_ROOT_MARKER {
                    // The marker at the root itself does not bound its walk.
                    if dir != root {
                        dirs.insert(dir);
                    }
                } else if is_stamped_file(&path) {
                    files.insert(path);
                }
            }
            if files.is_empty() && dirs.is_empty() {
                return Ok(None);
            }

            // Only the outermost of nested dirty directories: its walk covers
            // the rest.
            let mut outermost: Vec<PathBuf> = dirs.into_iter().collect();
            outermost.sort();
            let mut kept: Vec<PathBuf> = Vec::with_capacity(outermost.len());
            for dir in outermost {
                if !kept.last().is_some_and(|last| dir.starts_with(last)) {
                    kept.push(dir);
                }
            }
            if kept.iter().any(|dir| dir == root) {
                // No event names the root as a child; should one, its walk is
                // the whole tree, which starting over does with fresh watches.
                return Err("the project root itself changed".to_string());
            }
            let under_kept = |path: &Path| kept.iter().any(|dir| path.starts_with(dir));

            let mut next = stamps.clone();
            // Forget everything under the dirty directories — stamps and
            // watches — before walking any of them again. The order matters: a
            // directory moved from `a` to `b` keeps its watch; walked first, `b`
            // would take that watch over, and forgetting `a` after would then
            // remove the watch `b` now holds.
            next.retain(|path, _| !under_kept(path));
            let dropped: Vec<(PathBuf, WatchDescriptor)> = self
                .by_path
                .iter()
                .filter(|(path, _)| under_kept(path))
                .map(|(path, wd)| (path.clone(), wd.clone()))
                .collect();
            for (path, wd) in dropped {
                self.by_path.remove(&path);
                self.by_wd.remove(&wd);
                self.boundaries.remove(&wd);
                // Fails for a watch the kernel already dropped (deleted
                // directory): nothing left to remove.
                let _ = self.inotify.watches().remove(wd);
            }
            for dir in &kept {
                // Judged by its parent: the walk of `dir` itself decides
                // whether it is entered, a nested root, or skipped.
                let parent_walked = dir
                    .parent()
                    .is_some_and(|parent| is_walked_dir(root, parent));
                let is_real_dir = std::fs::symlink_metadata(dir).is_ok_and(|meta| meta.is_dir());
                if parent_walked && is_real_dir {
                    next.extend(self.walk(root, dir)?);
                }
            }
            for file in files {
                if under_kept(&file) {
                    continue;
                }
                match walked_stamp(&file) {
                    Some(stamp) => next.insert(file, stamp),
                    None => next.remove(&file),
                };
            }
            Ok(Some(next))
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::fs;

    /// After every step the watcher's stamps equal a fresh walk's — the
    /// staleness check answers as it did when it walked on every query. Each
    /// step is a way the set of walked files or their stamps can change.
    #[test]
    fn after_every_change_the_watched_stamps_equal_a_walk() {
        let tree = tempfile::TempDir::new().unwrap();
        let root = fs::canonicalize(tree.path()).unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let write = |rel: &str, text: &str| {
            let path = root.join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        };
        write("Cargo.toml", "[workspace]\n");
        write("a/src/lib.rs", "mod m;\n");
        write("a/src/m.rs", "fn f() {}\n");
        write("b/src/lib.rs", "\n");
        write("b/src/deep/x/y.rs", "\n");
        write("target/debug/build/out.rs", "\n");
        // A nested root from the start, as a large workspace has several.
        write("nested/.rmcroot", "");
        write("nested/src/lib.rs", "\n");

        let (mut watch, mut stamps) = TreeWatch::start(&root);
        assert!(matches!(watch.mode, Mode::Watching(_)), "inotify available");
        assert_eq!(stamps, crate::semantic::collect_stamps(&root));
        assert!(watch.current(&stamps).is_none(), "nothing changed yet");
        // A file outside what is stamped: no look result at all.
        write("a/notes.txt", "x");
        assert!(watch.current(&stamps).is_none(), "a .txt is not stamped");
        write("target/debug/build/out.rs", "changed");
        assert!(watch.current(&stamps).is_none(), "target/ is not walked");

        let check =
            |step: &str, watch: &mut TreeWatch, stamps: &mut HashMap<PathBuf, FileStamp>| {
                let next = watch.current(stamps).unwrap_or_else(|| stamps.clone());
                assert_eq!(
                    next,
                    crate::semantic::collect_stamps(&root),
                    "after {step}: the watched stamps are not a walk's"
                );
                *stamps = next;
            };

        write("a/src/m.rs", "fn f() { g() }\n");
        check("an edit", &mut watch, &mut stamps);
        let file = fs::File::options()
            .write(true)
            .open(root.join("a/src/m.rs"))
            .unwrap();
        file.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(7))
            .unwrap();
        drop(file);
        check("a touch to an old time", &mut watch, &mut stamps);
        write("a/src/n.rs", "\n");
        check("a new file", &mut watch, &mut stamps);
        fs::rename(root.join("a/src/n.rs"), root.join("a/src/o.rs")).unwrap();
        check("a file renamed", &mut watch, &mut stamps);
        fs::rename(root.join("a/src/o.rs"), root.join("a/src/o.txt")).unwrap();
        check("a file renamed out of *.rs", &mut watch, &mut stamps);
        fs::remove_file(root.join("a/src/m.rs")).unwrap();
        check("a file removed", &mut watch, &mut stamps);
        write("a/Cargo.toml", "[package]\n");
        check("a manifest", &mut watch, &mut stamps);
        std::os::unix::fs::symlink(root.join("b/src/lib.rs"), root.join("a/src/link.rs")).unwrap();
        check(
            "a symbolic link (the walker does not follow)",
            &mut watch,
            &mut stamps,
        );

        write("c/src/x/y/z.rs", "\n");
        check("a new tree of directories", &mut watch, &mut stamps);
        fs::create_dir_all(outside.path().join("moved/src")).unwrap();
        fs::write(outside.path().join("moved/src/lib.rs"), "\n").unwrap();
        fs::rename(outside.path().join("moved"), root.join("moved")).unwrap();
        check("a tree moved in", &mut watch, &mut stamps);
        // Moved within: the watches under `b` follow the inodes to `b2`.
        fs::rename(root.join("b"), root.join("b2")).unwrap();
        check("a tree moved within", &mut watch, &mut stamps);
        write("b2/src/deep/x/y.rs", "after the move\n");
        write("b2/src/deep/x/new.rs", "\n");
        check("edits under the moved tree", &mut watch, &mut stamps);
        // Both in one look: the move, then an edit through the moved watch.
        fs::rename(root.join("b2"), root.join("b3")).unwrap();
        write("b3/src/deep/x/y.rs", "after the second move\n");
        check(
            "a move and an edit under it in one look",
            &mut watch,
            &mut stamps,
        );
        fs::rename(root.join("moved"), outside.path().join("gone")).unwrap();
        check("a tree moved out", &mut watch, &mut stamps);
        write("gone_check.rs", "\n");
        fs::write(outside.path().join("gone/src/lib.rs"), "not ours\n").unwrap();
        check("an edit in the tree moved out", &mut watch, &mut stamps);
        fs::remove_dir_all(root.join("c")).unwrap();
        check("a tree removed", &mut watch, &mut stamps);

        write("b3/src/deep/.rmcroot", "");
        check("a nested-root marker added", &mut watch, &mut stamps);
        write("b3/src/deep/x/hidden.rs", "\n");
        check("an edit under the new nested root", &mut watch, &mut stamps);
        fs::remove_file(root.join("b3/src/deep/.rmcroot")).unwrap();
        check("the marker removed", &mut watch, &mut stamps);
        write("vendor/v/src/lib.rs", "\n");
        check("a skipped directory", &mut watch, &mut stamps);
        write(".rmcroot", "");
        check("the marker at the root", &mut watch, &mut stamps);
        write("nested/src/lib.rs", "edited while nested\n");
        assert!(
            watch.current(&stamps).is_none(),
            "inside a nested root is not walked"
        );
        fs::remove_file(root.join("nested/.rmcroot")).unwrap();
        check(
            "the marker of a nested root found at the start removed",
            &mut watch,
            &mut stamps,
        );
        write("nested/src/lib.rs", "edited after\n");
        check("an edit in the former nested root", &mut watch, &mut stamps);
        write("nested/.rmcroot", "");
        check("the marker back", &mut watch, &mut stamps);
    }

    /// The oracle above on a live tree others edit: every `RMC_WATCH_PROBE_EVERY`
    /// seconds (10) for `RMC_WATCH_PROBE_SECS` (120) the watched stamps are
    /// compared with a walk, with the time of both. No rust-analyzer.
    /// `RMC_WATCH_PROBE_PROJECT=<dir> cargo test -p rmc-server --lib
    /// probe_watch_against_walks -- --ignored --nocapture`
    #[test]
    #[ignore = "probe: needs RMC_WATCH_PROBE_PROJECT"]
    fn probe_watch_against_walks() {
        let env_secs = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .map_or(default, |text| text.parse().unwrap())
        };
        let root = fs::canonicalize(std::env::var("RMC_WATCH_PROBE_PROJECT").unwrap()).unwrap();
        let (secs, every) = (
            env_secs("RMC_WATCH_PROBE_SECS", 120),
            env_secs("RMC_WATCH_PROBE_EVERY", 10),
        );
        let started = std::time::Instant::now();
        let (mut watch, mut stamps) = TreeWatch::start(&root);
        let directories = match &watch.mode {
            Mode::Watching(watcher) => watcher.directory_count(),
            Mode::Walking => panic!("no watcher on {}", root.display()),
        };
        println!(
            "start: {directories} directories, {} files in {:?}",
            stamps.len(),
            started.elapsed()
        );
        let mut mismatches = 0;
        for round in 0..secs / every {
            std::thread::sleep(std::time::Duration::from_secs(every));
            let look = std::time::Instant::now();
            let next = watch.current(&stamps);
            let look = look.elapsed();
            let walk = std::time::Instant::now();
            let walked = crate::semantic::collect_stamps(&root);
            let walk = walk.elapsed();
            let changed = next.is_some();
            let next = next.unwrap_or_else(|| stamps.clone());
            let differ: Vec<&PathBuf> = next
                .keys()
                .chain(walked.keys())
                .filter(|path| next.get(*path) != walked.get(*path))
                .collect();
            // A file written between the look and the walk differs for a
            // moment; the next round must agree on it.
            println!(
                "round {round}: look {look:?} ({}), walk {walk:?}, {} differ {:?}",
                if changed { "changes" } else { "nothing" },
                differ.len(),
                differ.iter().take(3).collect::<Vec<_>>()
            );
            if !differ.is_empty() {
                mismatches += 1;
            }
            stamps = next;
        }
        println!("{mismatches} round(s) with a difference");
    }

    /// More events than the kernel queues between two looks: the watcher walks
    /// again rather than trusting a partial list.
    #[test]
    fn an_overflowed_queue_walks_the_tree_again() {
        let tree = tempfile::TempDir::new().unwrap();
        let root = fs::canonicalize(tree.path()).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "\n").unwrap();
        let (mut watch, stamps) = TreeWatch::start(&root);

        let limit: usize = fs::read_to_string("/proc/sys/fs/inotify/max_queued_events")
            .map(|text| text.trim().parse().unwrap())
            .unwrap_or(16384);
        // Alternating names: the kernel merges an event equal to the last one
        // queued, so writes to one file would not fill the queue.
        for i in 0..limit + 10 {
            fs::write(root.join(["a.txt", "b.txt"][i % 2]), [b'x'; 1]).unwrap();
        }
        fs::write(root.join("src/new.rs"), "\n").unwrap();

        let next = watch.current(&stamps).expect("an overflow is a change");
        assert_eq!(next, crate::semantic::collect_stamps(&root));
        assert!(
            matches!(watch.mode, Mode::Watching(_)),
            "watching again after the walk"
        );
        assert!(watch.current(&next).is_none());
    }
}
