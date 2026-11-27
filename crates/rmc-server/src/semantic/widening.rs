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
//! A unit that is a macro (`Reach::Expand`) stands for what rust-analyzer
//! expands it to; one that cannot tell (`Reach::Opaque`, a changed macro
//! definition) ends the closure: the answer is the crates' roots, as before.
//!
//! # Cost
//!
//! Names cost a word-index lookup each. A held type costs its candidate files
//! (the word index), parsed once each — ≈ 1 ms a file — and the macro calls
//! among their units that name it, expanded once each. At most [`PARSE_BUDGET`]
//! files and expansions together: past it the closure is about as wide as the
//! crates anyway, and the answer is the crates' roots.

use std::collections::{HashMap, HashSet};

use ra_ap_syntax::TextRange;
use ra_ap_vfs::FileId;

use super::units::{self, ModuleFiles, Reach, Side, Unit, Witness};
use super::word_index::WordIndex;

/// Files parsed and macros expanded by one closure before it gives up for the
/// crates' roots.
pub(crate) const PARSE_BUDGET: usize = 2000;

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
        /// Files parsed and macros expanded to close the names.
        parsed: usize,
        expanded: usize,
    },
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
    /// Files naming these are pending; values of those types matter.
    Plain {
        names: Vec<String>,
        held: Vec<String>,
    },
    /// Files naming `.0` among those that may hold a `.1`.
    Both(String, String),
    /// Files naming `.0` in the module `.1` of the declaring file and below.
    Scoped(String, Vec<String>),
}

fn resolve(witness: &Witness, words: &WordIndex) -> Taken {
    let plain = |names: Vec<String>, held: Vec<String>| Taken::Plain { names, held };
    match witness {
        Witness::Name(name) => plain(vec![name.clone()], Vec::new()),
        Witness::Either(a, b) => {
            let pick = if words.files_with(a).len() <= words.files_with(b).len() {
                a
            } else {
                b
            };
            plain(vec![pick.clone()], Vec::new())
        }
        Witness::Holds(types) => plain(Vec::new(), types.clone()),
        Witness::Both(name, ty) => Taken::Both(name.clone(), ty.clone()),
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
            if own.last().is_some_and(local) {
                // The project's trait on a foreign type: its users name it or
                // its items.
                return plain(own.clone(), Vec::new());
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

/// What one closure reached: files naming `names` or `held` are pending.
#[derive(Default)]
struct Reached {
    names: HashSet<String>,
    /// Types whose values matter.
    held: HashSet<String>,
}

/// Closures over one change: the parsed candidate files and the budget are
/// shared by all of them.
struct Closer<'s, 'a> {
    scope: &'s Scope<'a>,
    parsed: HashMap<FileId, Vec<Unit>>,
    expanded: usize,
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
                    if self.parsed.len() + self.expanded >= self.scope.parse_budget {
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
                        let Taken::Plain { names, held } = resolve(&witness, self.scope.words)
                        else {
                            unreachable!("holders hand out names and held types only");
                        };
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
    let mut both: Vec<(String, String)> = Vec::new();
    for (file, witness) in &witnesses {
        match resolve(witness, scope.words) {
            Taken::Plain { names: plain, held } => {
                names.extend(plain);
                roots.extend(held);
            }
            Taken::Both(name, ty) => both.push((name, ty)),
            Taken::Scoped(name, module) => {
                let below: HashSet<FileId> =
                    (scope.module_files)(*file, &ModuleFiles::Below(module))
                        .into_iter()
                        .collect();
                files.extend(
                    reachable_files_with(&name)
                        .into_iter()
                        .filter(|file| below.contains(file)),
                );
                names.push(format!("{name}@"));
            }
        }
    }
    if scope.trace.is_some() {
        closer.cause = "changed units".to_owned();
    }
    let main = closer.close(roots)?;
    for name in main
        .names
        .iter()
        .chain(&main.held)
        .chain(names.iter().filter(|name| !name.ends_with('@')))
    {
        files.extend(reachable_files_with(name));
    }
    // A name among the files that may hold its type: each type's closure once.
    let mut holding: HashMap<String, HashSet<FileId>> = HashMap::new();
    for (name, ty) in &both {
        if !holding.contains_key(ty) {
            if scope.trace.is_some() {
                closer.cause = format!("holders of `{ty}`");
            }
            let reached = closer.close(vec![ty.clone()])?;
            let holders: HashSet<FileId> = reached
                .names
                .iter()
                .chain(&reached.held)
                .flat_map(|name| reachable_files_with(name))
                .collect();
            holding.insert(ty.clone(), holders);
        }
        let holders = &holding[ty];
        files.extend(
            reachable_files_with(name)
                .into_iter()
                .filter(|file| holders.contains(file)),
        );
    }
    let mut shown: Vec<String> = names
        .into_iter()
        .chain(main.names)
        .chain(main.held.into_iter().map(|name| format!("{name}*")))
        .chain(both.iter().map(|(name, ty)| format!("{name}&{ty}")))
        .collect();
    shown.sort();
    shown.dedup();
    Ok(Widening::Files {
        files,
        names: shown,
        parsed: closer.parsed.len(),
        expanded: closer.expanded,
    })
}
