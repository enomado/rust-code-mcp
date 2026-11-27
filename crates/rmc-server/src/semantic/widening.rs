//! Which files a surface change can make re-resolve — the table of uses' files
//! to make pending (plan step `uses-narrow-widening`).
//!
//! # Why not the crates
//!
//! A surface change in a file used to make pending every file of its crate
//! and of all the crates depending on it, transitively. On a ~9000-file
//! workspace whose crates nearly all depend on a few shared ones, one new
//! function in a shared crate made pending a third to a half of the table
//! (3111 files after one commit to the base crate, 2026-11-27), and the next
//! commit came before the ≈ 8 min rebuild ended: the table was never whole.
//!
//! # What instead
//!
//! The changed units of the edited files (`units::changed_units`) say whom they
//! reach (`units` module header): the files naming an item, and — for a trait
//! impl, a tuple field, an alias — the files that may hold a value of a type.
//! The second is a closure: the files naming the type, and those naming what
//! hands its values out (a unit naming the type in a signature, a field, a
//! variant — its `holders`), and through tuple structs, aliases and trait
//! impls' associated types the holders of other types in turn. The files to
//! make pending are those whose code holds a name reached (the word index),
//! within the crates' roots — both bounds are sound, so is their intersection
//! — plus the module files of changed `mod` declarations and private globs.
//!
//! Some names count only in some of their files ([`Narrow`]): a method `get`
//! of `S` where a value of `S` may be held (the holders of `S` — its own
//! closure), a trait's item `k` where the trait may be in scope (the holders
//! of the trait, and files with a glob import), an associated function `new`
//! of `S` where `S` or an alias of it is named. That goes for what a holder
//! hands out too, not only for the changed units: a type `S` whose values
//! matter makes pending the callers of `S::new`, not every file naming a
//! `new`. Those nested closures share the parsed files and the second half of
//! the budget; one that does not end leaves its name to the files naming it
//! as a member can be named (`x.f`, `T::f`, a literal naming `T`, a glob —
//! [`Closer::shaped`]), which bounds every narrowed name in any case. A field
//! or method name in at most [`HOLDERS_ABOVE`] such files takes them without
//! the closure (step `uses-narrow-untold`: a god-type's holders are thousands
//! of files, a rare field's closure cost more than it spared).
//!
//! A private item reaches the files below its module that name it after a
//! path to the module (`super::x`, `crate::x`, `m::x`) or hold a glob import
//! and name it other than declaring it (`word_index::path_uses`).
//!
//! A unit that is a macro (`Reach::Expand`) stands for what rust-analyzer
//! expands it to; one that cannot tell (`Reach::Opaque`, a changed macro
//! definition) ends the closure: the answer is the crates' roots, as before.
//!
//! # Cost
//!
//! Names cost a word-index lookup each; a private item, a lexing of each file
//! below its module naming it (≈ 0.2 ms, at most [`LEX_BUDGET`] a change).
//! A held type costs its candidate files
//! (the word index), parsed once each — ≈ 1 ms a file — and the macro calls
//! among their units that name it, expanded once each. At most [`PARSE_BUDGET`]
//! files and expansions together: past it the closure is about as wide as the
//! crates anyway, and the answer is the crates' roots.

use std::collections::{HashMap, HashSet};

use ra_ap_syntax::TextRange;
use ra_ap_vfs::FileId;

use super::units::{self, ModuleFiles, Reach, Side, Unit, Witness};
use super::word_index::{WordIndex, path_uses};

/// Files parsed and macros expanded by one closure before it gives up for the
/// crates' roots.
pub(crate) const PARSE_BUDGET: usize = 2000;

/// Files lexed by one change to tell how a private item is named below its
/// module; past it the rest are kept as they are.
const LEX_BUDGET: usize = 4 * PARSE_BUDGET;

/// A field's, method's, variant's or trait item's name is narrowed by the
/// holders of its type only when the files naming it in its shape are more
/// than this ([`Closer::narrowed`]).
const HOLDERS_ABOVE: usize = 32;

/// The files a surface change can make re-resolve.
#[derive(Debug)]
pub(crate) enum Widening {
    /// Every file of the roots of the changed crates and of their reverse
    /// dependencies; why the names could not say.
    Roots(String),
    /// These files (within those roots), by these names.
    Files {
        files: HashSet<FileId>,
        names: Vec<String>,
        /// By name as `names` shows it, the files it makes pending (they
        /// overlap) — what the count is made of, for the probe.
        widths: HashMap<String, usize>,
        /// Files parsed and macros expanded to close the names, files lexed
        /// for the paths to private items.
        parsed: usize,
        expanded: usize,
        lexed: usize,
    },
}

/// The names the module of the file at `declaring` is known by below it, by
/// its path: the stem of `a/b.rs`, the directory of `a/mod.rs`, none for
/// `lib.rs` and `main.rs`. A `#[path]` module is named by its declaration,
/// which only rust-analyzer knows; a crate root elsewhere (`src/bin/x.rs`)
/// gets its stem, which only adds files.
pub(crate) fn module_names_by_path(declaring: &std::path::Path) -> Vec<String> {
    let name = |path: Option<&std::ffi::OsStr>| -> Vec<String> {
        path.map(|name| name.to_string_lossy().into_owned())
            .into_iter()
            .collect()
    };
    match declaring.file_stem().and_then(|stem| stem.to_str()) {
        Some("lib" | "main") => Vec::new(),
        Some("mod") => name(declaring.parent().and_then(|dir| dir.file_name())),
        _ => name(declaring.file_stem()),
    }
}

/// What a closure reads.
pub(crate) struct Scope<'a> {
    pub(crate) words: &'a WordIndex,
    /// Whether a file is in a root the change can reach — the roots of the
    /// changed crates and of their reverse dependencies.
    pub(crate) reachable: &'a dyn Fn(FileId) -> bool,
    /// A file's text as the change leaves it.
    pub(crate) text: &'a dyn Fn(FileId) -> String,
    /// The files `module` stands for, declared in `file`.
    pub(crate) module_files: &'a dyn Fn(FileId, &ModuleFiles) -> Vec<FileId>,
    /// The names the module of a file is known by to the files below it —
    /// none for a crate root, which they reach as `crate`.
    pub(crate) module_names: &'a dyn Fn(FileId) -> Vec<String>,
    /// The units of what the macro at `at` in `file` expands to on `side` (see
    /// `Reach::Expand`), macros within the expansion expanded in their turn —
    /// none of the units is an `Expand`; `Err`, why it could not tell.
    pub(crate) expand:
        &'a dyn Fn(Side, FileId, TextRange, bool, &[String]) -> Result<Vec<Unit>, String>,
    pub(crate) parse_budget: usize,
    /// When set, a line per name taken in: the name, and what took it in.
    pub(crate) trace: Option<&'a std::cell::RefCell<Vec<String>>>,
}

/// Where the files of `module`, declared in the file at `declaring`, are: the
/// file paths it names and the directories every file under which belongs to
/// it. A file's own modules live in the directory of its name (`a/b.rs` →
/// `a/b/`), a `lib.rs`/`main.rs`/`mod.rs` in its own directory.
pub(crate) fn module_paths(
    declaring: &std::path::Path,
    module: &ModuleFiles,
) -> (Vec<std::path::PathBuf>, Vec<std::path::PathBuf>) {
    let stem = declaring
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned());
    let base = || match stem.as_deref() {
        Some("lib" | "main" | "mod") => declaring
            .parent()
            .map(|dir| dir.to_owned())
            .unwrap_or_default(),
        _ => declaring.with_extension(""),
    };
    let below = |inline: &[String]| {
        let mut dir = base();
        dir.extend(inline);
        dir
    };
    match module {
        ModuleFiles::Declared(path) => {
            let (last, inline) = path.split_last().expect("a declared module has a name");
            let dir = below(inline);
            (vec![dir.join(format!("{last}.rs"))], vec![dir.join(last)])
        }
        ModuleFiles::AtPath(inline, at) => {
            // A `#[path]` outside inline modules is relative to the declaring
            // file's directory, inside them to their directory under it.
            let mut dir = if inline.is_empty() {
                declaring
                    .parent()
                    .map(|dir| dir.to_owned())
                    .unwrap_or_default()
            } else {
                below(inline)
            };
            dir.push(at);
            let under = dir.with_extension("");
            (vec![dir], vec![under])
        }
        ModuleFiles::Below(inline) => {
            let dir = below(inline);
            // A file's own inline modules are in the file itself: the caller
            // makes it pending as a changed file.
            (Vec::new(), vec![dir])
        }
    }
}

/// A witness taken in, in the terms a closure follows.
enum Taken {
    /// Files naming `names` are pending, and those `narrow` says; values of
    /// the `held` types matter.
    Plain {
        names: Vec<String>,
        held: Vec<String>,
        narrow: Vec<Narrow>,
    },
    /// Files naming `.0` in the module `.1` of the declaring file and below.
    Scoped(String, Vec<String>),
}

/// A name among the files of something else — never wider than the files
/// naming it, and those files when the something else cannot be told.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Narrow {
    /// `.0` among the files that may hold a value of the type `.1`
    /// ([`Witness::Both`]).
    Holding(String, String),
    /// `.0` among the files that may have the trait `.1` in scope
    /// ([`Witness::InScope`]).
    InScope(String, String),
    /// `.0` among the files naming the type `.1` or an alias of it
    /// ([`Witness::Either`]).
    Through(String, String),
}

impl Narrow {
    fn name(&self) -> &str {
        match self {
            Narrow::Holding(name, _) | Narrow::InScope(name, _) | Narrow::Through(name, _) => name,
        }
    }

    /// As the log line shows it.
    fn shown(&self) -> String {
        match self {
            Narrow::Holding(name, ty) => format!("{name}&{ty}"),
            Narrow::InScope(name, tr) => format!("{name}~{tr}"),
            Narrow::Through(name, ty) => format!("{ty}::{name}"),
        }
    }
}

fn resolve(witness: &Witness, words: &WordIndex) -> Taken {
    let plain = |names: Vec<String>, held: Vec<String>| Taken::Plain {
        names,
        held,
        narrow: Vec::new(),
    };
    let narrow = |narrow: Narrow| Taken::Plain {
        names: Vec::new(),
        held: Vec::new(),
        narrow: vec![narrow],
    };
    match witness {
        Witness::Name(name) => plain(vec![name.clone()], Vec::new()),
        Witness::InScope(item, tr) => narrow(Narrow::InScope(item.clone(), tr.clone())),
        Witness::Either(ty, item) => narrow(Narrow::Through(item.clone(), ty.clone())),
        Witness::Holds(types) => plain(Vec::new(), types.clone()),
        Witness::Both(name, ty) => narrow(Narrow::Holding(name.clone(), ty.clone())),
        Witness::Scoped(name, module) => Taken::Scoped(name.clone(), module.clone()),
        Witness::TraitImpl {
            self_type,
            trait_ref,
            own,
        } => {
            let local = |name: &String| words.is_declared(name) && !units::is_primitive(name);
            let self_local: Vec<String> = self_type
                .iter()
                .filter(|name| local(name))
                .cloned()
                .collect();
            if !self_local.is_empty() {
                return plain(Vec::new(), self_local);
            }
            // The trait's own name is the last of `own`.
            if let Some((trait_name, items)) = own.split_last()
                && local(trait_name)
            {
                // The project's trait on a foreign type: its users name it, or
                // its items with it in scope.
                return Taken::Plain {
                    names: vec![trait_name.clone()],
                    held: Vec::new(),
                    narrow: items
                        .iter()
                        .map(|item| Narrow::InScope(item.clone(), trait_name.clone()))
                        .collect(),
                };
            }
            let trait_local: Vec<String> = trait_ref
                .iter()
                .filter(|name| local(name))
                .cloned()
                .collect();
            if !trait_local.is_empty() {
                // `impl From<Foo> for String`: what a `Foo` turns into.
                return plain(Vec::new(), trait_local);
            }
            // No name the project declares as a type: one a macro made, its
            // values then.
            let made: Vec<String> = self_type
                .iter()
                .filter(|name| !units::is_primitive(name))
                .cloned()
                .collect();
            plain(own.clone(), made)
        }
    }
}

/// What one closure reached: files naming `names` or `held` are pending, and
/// those `narrow` says.
#[derive(Default)]
struct Reached {
    names: HashSet<String>,
    /// Types whose values matter.
    held: HashSet<String>,
    narrow: HashSet<Narrow>,
}

/// Closures over one change: the parsed candidate files and the budget are
/// shared by all of them.
struct Closer<'s, 'a> {
    scope: &'s Scope<'a>,
    parsed: HashMap<FileId, Vec<Unit>>,
    expanded: usize,
    /// Files parsed and macros expanded together past which a closure gives
    /// up: `Scope::parse_budget` for the change's own closures, twice that
    /// for the nested ones ([`Self::holders_of`]) — the budget's second half
    /// is theirs.
    limit: usize,
    /// By type or trait, the files that may hold a value of it
    /// ([`Self::holders_of`]); `None` while its closure runs or when it did
    /// not end.
    holders: HashMap<String, Option<HashSet<FileId>>>,
    /// Narrowed names that fell back on their bare names, for the log.
    bare: HashSet<Narrow>,
    /// What the names taken in now come from, for `Scope::trace`.
    cause: String,
}

impl Closer<'_, '_> {
    fn note(&self, what: &str) {
        if let Some(trace) = self.scope.trace {
            trace.borrow_mut().push(format!("{what} <- {}", self.cause));
        }
    }

    fn expansion(
        &mut self,
        side: Side,
        file: FileId,
        at: TextRange,
        attr: bool,
        module: &[String],
    ) -> Result<Vec<Unit>, String> {
        self.expanded += 1;
        (self.scope.expand)(side, file, at, attr, module)
    }

    /// The closure from the types `roots` over the holders of their values —
    /// see the module header. Holders hand out names and held types only.
    fn close(&mut self, roots: Vec<String>) -> Result<Reached, String> {
        let mut reached = Reached::default();
        let mut queue = Vec::new();
        for root in roots {
            if reached.held.insert(root.clone()) {
                queue.push(root);
            }
        }
        while let Some(held) = queue.pop() {
            for &file in self.scope.words.files_with(&held) {
                if !(self.scope.reachable)(file) {
                    continue;
                }
                if !self.parsed.contains_key(&file) {
                    if self.parsed.len() + self.expanded >= self.limit {
                        return Err(format!(
                            "the held types reach past {} files and expansions ({} types, at `{held}`)",
                            self.scope.parse_budget,
                            reached.held.len()
                        ));
                    }
                    let units = units::units_of(&(self.scope.text)(file));
                    self.parsed.insert(file, units);
                }
                let naming: Vec<Unit> = self.parsed[&file]
                    .iter()
                    .filter(|unit| unit.mentions.binary_search(&held).is_ok())
                    .cloned()
                    .collect();
                for unit in &naming {
                    if self.scope.trace.is_some() {
                        self.cause = format!("`{held}` in {file:?}");
                    }
                    let witnesses = self
                        .handed_out(unit, file, &held)
                        .map_err(|why| format!("{why} names `{held}`"))?;
                    for witness in witnesses {
                        let Taken::Plain {
                            names,
                            held,
                            narrow,
                        } = resolve(&witness, self.scope.words)
                        else {
                            unreachable!("holders hand out no private names")
                        };
                        for narrow in narrow {
                            if self.scope.trace.is_some() && !reached.narrow.contains(&narrow) {
                                self.note(&narrow.shown());
                            }
                            reached.narrow.insert(narrow);
                        }
                        for name in names {
                            if reached.names.insert(name.clone()) {
                                self.note(&name);
                            }
                        }
                        for name in held {
                            if reached.held.insert(name.clone()) {
                                self.note(&format!("holds {name}"));
                                queue.push(name);
                            }
                        }
                    }
                }
            }
        }
        Ok(reached)
    }

    /// The reachable files naming `name`.
    fn files_with(&self, name: &str) -> impl Iterator<Item = FileId> + '_ {
        self.scope
            .words
            .files_with(name)
            .iter()
            .copied()
            .filter(|&file| (self.scope.reachable)(file))
    }

    /// The files `reached` makes pending: those naming its names and held
    /// types, and those its narrowed names say.
    fn files_of(&mut self, reached: &Reached) -> HashSet<FileId> {
        let mut files: HashSet<FileId> = reached
            .names
            .iter()
            .chain(&reached.held)
            .flat_map(|name| self.files_with(name))
            .collect();
        for narrow in &reached.narrow {
            files.extend(self.narrowed(narrow));
        }
        files
    }

    /// The files a narrowed name makes pending — see [`Narrow`] — among those
    /// naming it in a shape it can be reached by ([`Self::shaped`]). Only that
    /// shape when the other side cannot be told.
    fn narrowed(&mut self, narrow: &Narrow) -> Vec<FileId> {
        let shaped = self.shaped(narrow);
        let name = narrow.name().to_owned();
        let shaped_files: Vec<FileId> = self
            .files_with(&name)
            .filter(|file| shaped.contains(file))
            .collect();
        // A closure of holders parses its candidates under the refresh (a few
        // hundred to thousands of files on a god-type): for a field's or a
        // method's name already in few files it costs more than the files it
        // could spare. Not for a trait's item: a method call `x.k()` is in
        // the member shape on every `k` of the workspace, the trait's holders
        // are what tells them apart.
        if shaped_files.len() <= HOLDERS_ABOVE && matches!(narrow, Narrow::Holding(..)) {
            return shaped_files;
        }
        let among: Option<HashSet<FileId>> = match narrow {
            Narrow::Through(_, ty) => Some(
                self.scope
                    .words
                    .names_for(ty)
                    .iter()
                    .flat_map(|name| self.files_with(name))
                    .collect(),
            ),
            // A trait's items resolve without it in scope on `dyn Tr` and on
            // a parameter bounded by it or a subtrait — the files holding its
            // values — and with it imported by a glob.
            Narrow::Holding(_, ty) | Narrow::InScope(_, ty) => self.holders_of(ty),
        };
        let with_glob = matches!(narrow, Narrow::InScope(..));
        match among {
            Some(among) => shaped_files
                .into_iter()
                .filter(|file| {
                    among.contains(file) || (with_glob && self.scope.words.has_glob(*file))
                })
                .collect(),
            None => {
                self.bare.insert(narrow.clone());
                shaped_files
            }
        }
    }

    /// The files that can reach a narrowed name at all, by how it is written:
    /// a field, a method, a variant, an associated or a trait's item is named
    /// as a member (`x.f`, `x.m()`, `T::v`, `Self::new`, `<T as Tr>::k`, in a
    /// `use` item: `WordIndex::member_files_with`). A field and a variant also
    /// bare in a literal or a pattern naming the type (`S { f, .. }`, `Self {
    /// f }` in an `impl` naming `S`, or an alias of it) and a variant after a
    /// glob import (`use E::*`; `Some`, a variant of std, is not the
    /// project's); a trait's item is defined bare in an `impl` naming the
    /// trait. An associated function of an inherent `impl` (`Through`) is
    /// never bare: no `use S::new`, no glob of a type. The rest of the files
    /// with the bare word declare it (`fn f`, a binding `let f`, a field `f:`
    /// of another struct) or name something else: on a day of a ~9000-file
    /// workspace, a god-type's field `error` with untold holders stood in
    /// 23.7 k files summed over the commits, 2.9 k of them in this shape. Both bounds are
    /// sound, so is their intersection with the told side's.
    ///
    /// Cost: the word index's member list and the files of the type's names —
    /// lookups, no lexing.
    fn shaped(&self, narrow: &Narrow) -> HashSet<FileId> {
        let words = self.scope.words;
        let (name, owner) = match narrow {
            Narrow::Through(name, _) => {
                return words.member_files_with(name).iter().copied().collect();
            }
            Narrow::Holding(name, owner) | Narrow::InScope(name, owner) => (name, owner),
        };
        let mut files: HashSet<FileId> = words.member_files_with(name).iter().copied().collect();
        files.extend(
            words
                .names_for(owner)
                .iter()
                .flat_map(|owner| words.files_with(owner))
                .copied(),
        );
        files.extend(
            words
                .files_with(name)
                .iter()
                .copied()
                .filter(|&file| words.has_glob(file)),
        );
        files
    }

    /// The files that may hold a value of the type (or trait) `ty`: its
    /// closure as a held type, the narrowed names it hands out resolved in
    /// turn — memoised; one met again while its closure runs counts as untold
    /// (`None`), as does one whose closure did not end within the nested half
    /// of the budget (see [`Self::limit`]).
    ///
    /// Cost: what the closure parses and expands, shared with every other
    /// closure of the change through `parsed`.
    fn holders_of(&mut self, ty: &str) -> Option<HashSet<FileId>> {
        if let Some(known) = self.holders.get(ty) {
            return known.clone();
        }
        self.holders.insert(ty.to_owned(), None);
        let saved = (self.limit, std::mem::take(&mut self.cause));
        self.limit = 2 * self.scope.parse_budget;
        if self.scope.trace.is_some() {
            self.cause = format!("holders of `{ty}`");
        }
        let reached = self.close(vec![ty.to_owned()]);
        let files = match reached {
            Ok(reached) => Some(self.files_of(&reached)),
            Err(why) => {
                self.note(&format!("{ty}: untold ({why})"));
                None
            }
        };
        (self.limit, self.cause) = saved;
        self.holders.insert(ty.to_owned(), files.clone());
        files
    }

    /// What `unit` of `file`, which names the held type `held`, hands out of
    /// its values.
    fn handed_out(
        &mut self,
        unit: &Unit,
        file: FileId,
        held: &str,
    ) -> Result<Vec<Witness>, String> {
        match &unit.reach {
            Reach::Opaque(why) => Err(why.clone()),
            Reach::Expand { at, attr, module } => {
                // What the call expands to, whatever of it names the type: an
                // argument may come out as anything.
                let mut out = Vec::new();
                for unit in &self.expansion(Side::New, file, *at, *attr, module)? {
                    if let Reach::Opaque(why) = &unit.reach {
                        return Err(why.clone());
                    }
                    out.extend(unit.holders.iter().map(|holder| holder.witness.clone()));
                }
                Ok(out)
            }
            _ => Ok(unit
                .holders
                .iter()
                .filter(|holder| {
                    holder
                        .mentions
                        .binary_search_by(|name| name.as_str().cmp(held))
                        .is_ok()
                })
                .map(|holder| holder.witness.clone())
                .collect()),
        }
    }

    /// The witnesses of a change of `unit` of `file`, macros expanded, with
    /// the module files it reaches.
    fn changed(
        &mut self,
        side: Side,
        unit: &Unit,
        file: FileId,
        out: &mut Vec<(FileId, Witness)>,
        files: &mut HashSet<FileId>,
    ) -> Result<(), String> {
        match &unit.reach {
            Reach::Opaque(why) => Err(why.clone()),
            Reach::MacroDef(name) => Err(format!("a macro definition {name}!")),
            Reach::Expand { at, attr, module } => {
                for unit in &self.expansion(side, file, *at, *attr, module)? {
                    self.changed(side, unit, file, out, files)?;
                }
                Ok(())
            }
            Reach::Named { witnesses, module } => {
                out.extend(witnesses.iter().map(|witness| (file, witness.clone())));
                if let Some(module) = module {
                    files.extend((self.scope.module_files)(file, module));
                }
                Ok(())
            }
        }
    }
}

/// The old side's macro units of `units` replaced by what they expand to —
/// done while the database still holds the old texts; a failed expansion
/// stands as an opaque unit.
pub(crate) fn expand_old_side(
    file: FileId,
    units: Vec<(Side, Unit)>,
    expand: &dyn Fn(Side, FileId, TextRange, bool, &[String]) -> Result<Vec<Unit>, String>,
) -> Vec<(Side, Unit)> {
    let mut out = Vec::with_capacity(units.len());
    for (side, unit) in units {
        match (&side, &unit.reach) {
            (Side::Old, Reach::Expand { at, attr, module }) => {
                match expand(side, file, *at, *attr, module) {
                    Ok(expansion) => out.extend(expansion.into_iter().map(|unit| (side, unit))),
                    Err(why) => out.push((side, Unit::opaque(why))),
                }
            }
            _ => out.push((side, unit)),
        }
    }
    out
}

/// See the module header. `changed`: by edited, added or removed file, its
/// changed units — an old side's macros already expanded (the database holds
/// the new texts by now), the new side's as they are.
pub(crate) fn widen(changed: &[(FileId, Vec<(Side, Unit)>)], scope: &Scope<'_>) -> Widening {
    match widen_or_why(changed, scope) {
        Ok(widening) => widening,
        Err(why) => Widening::Roots(why),
    }
}

fn widen_or_why(
    changed: &[(FileId, Vec<(Side, Unit)>)],
    scope: &Scope<'_>,
) -> Result<Widening, String> {
    let mut closer = Closer {
        scope,
        parsed: HashMap::new(),
        expanded: 0,
        limit: scope.parse_budget,
        holders: HashMap::new(),
        bare: HashSet::new(),
        cause: String::new(),
    };
    let mut witnesses = Vec::new();
    let mut files: HashSet<FileId> = HashSet::new();
    for (file, units) in changed {
        for (side, unit) in units {
            closer
                .changed(*side, unit, *file, &mut witnesses, &mut files)
                .map_err(|why| format!("changed: {why}"))?;
        }
    }
    let reachable_files_with = |name: &str| -> Vec<FileId> {
        scope
            .words
            .files_with(name)
            .iter()
            .copied()
            .filter(|&file| (scope.reachable)(file))
            .collect()
    };
    let mut names: Vec<String> = Vec::new();
    let mut roots: Vec<String> = Vec::new();
    let mut narrow: Vec<Narrow> = Vec::new();
    let mut lexed = 0;
    let mut widths: HashMap<String, usize> = HashMap::new();
    for (file, witness) in &witnesses {
        match resolve(witness, scope.words) {
            Taken::Plain {
                names: plain,
                held,
                narrow: narrowed,
            } => {
                names.extend(plain);
                roots.extend(held);
                narrow.extend(narrowed);
            }
            Taken::Scoped(name, module) => {
                let below: HashSet<FileId> =
                    (scope.module_files)(*file, &ModuleFiles::Below(module.clone()))
                        .into_iter()
                        .collect();
                // A file below the module reaches its private item through a
                // path to the module or a glob import (glob or not, a name
                // in a file's code alone resolves in its own scope). The
                // module's own file is pending as a changed file.
                let mut prefixes: Vec<String> =
                    ["super", "self", "crate"].map(str::to_owned).into();
                match module.last() {
                    Some(inline) => prefixes.push(inline.clone()),
                    None => prefixes.extend((scope.module_names)(*file)),
                }
                let prefixes: Vec<&str> = prefixes.iter().map(String::as_str).collect();
                let width = widths.entry(format!("{name}@")).or_default();
                for below_file in reachable_files_with(&name)
                    .into_iter()
                    .filter(|file| below.contains(file))
                {
                    let keep = lexed >= LEX_BUDGET || {
                        lexed += 1;
                        let found = path_uses(&(scope.text)(below_file), &name, &prefixes);
                        found.after_prefix
                            || (found.undeclared && scope.words.has_glob(below_file))
                    };
                    if keep {
                        files.insert(below_file);
                        *width += 1;
                    }
                }
                names.push(format!("{name}@"));
            }
        }
    }
    if scope.trace.is_some() {
        closer.cause = "changed units".to_owned();
    }
    // The change's own closure: past its budget, the crates' roots. The
    // narrowed names it and the changed units give are told after it, each
    // other side's closure once, within the budget's second half — past it a
    // name's bare files.
    let mut main = closer.close(roots)?;
    main.names
        .extend(names.iter().filter(|name| !name.ends_with('@')).cloned());
    main.narrow.extend(narrow);
    // `Closer::files_of`, name by name to count each one's files.
    for (name, shown) in main
        .names
        .iter()
        .map(|name| (name, name.clone()))
        .chain(main.held.iter().map(|name| (name, format!("{name}*"))))
    {
        let of_name: Vec<FileId> = closer.files_with(name).collect();
        *widths.entry(shown).or_default() += of_name.len();
        files.extend(of_name);
    }
    for narrow in &main.narrow {
        let of_name = closer.narrowed(narrow);
        let shown = if closer.bare.contains(narrow) {
            format!("{}!", narrow.shown())
        } else {
            narrow.shown()
        };
        *widths.entry(shown).or_default() += of_name.len();
        files.extend(of_name);
    }
    let mut shown: Vec<String> = widths.keys().cloned().collect();
    shown.sort();
    Ok(Widening::Files {
        files,
        names: shown,
        widths,
        parsed: closer.parsed.len(),
        expanded: closer.expanded,
        lexed,
    })
}
