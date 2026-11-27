//! Which project files mention a word **in code** — the candidates a name
//! search asks rust-analyzer about, so that no query walks every file.
//!
//! # Why it exists
//!
//! The sweeps of `position` (a struct field's references, a library item's
//! uses) used to read every `.rs` file of the project and probe every textual
//! occurrence of the name — comments and strings included — on every call: on
//! a ~8000-file workspace that is 148 MB of text per query before the first
//! rust-analyzer call. This is the "index of words" of IntelliJ-style engines:
//! a word maps to the files whose code holds it, built once per load and
//! patched per changed file, and a query reads only those files.
//!
//! # What counts as "in code"
//!
//! Every `IDENT` token of rust-analyzer's lexer — inside macro token trees too
//! (`vec![radius_override]` is code to the macro), a raw `r#type` as `type` —
//! plus the implicit captures of format strings (`"{PI}"` reads the constant
//! `PI`: rust-analyzer resolves it, so it is a use). Comments and other string
//! contents are not: an occurrence there resolves to nothing, and it was
//! exactly those that the old sweep sent to rust-analyzer one by one.
//!
//! Files are lexed as edition 2015, where `async`, `await`, `dyn`, `try` and
//! `gen` are still identifiers: a file of a newer edition then indexes a few
//! keywords as words too, which only adds candidates. The other way round, a
//! file that names something `gen` would lose it.
//!
//! # Numbers (a ~8000-file workspace, 2026-11-26)
//!
//! 148 MB of text, 5.3 M identifiers in code, 272 k distinct words, 1.16 M
//! (word, file) pairs; built in ≈ 1.4 s on one core during a 44 s load. Without
//! offsets the index is ≈ 10 MB; occurrences are found by lexing the candidate
//! files only, when asked (≈ 0.2 ms a file). Candidates per word: a field a
//! handful of files, `spawn` 281, `new` 5194 — common names stay expensive and
//! need the graph (`library-refs-scope` of the plan).

use std::collections::HashMap;

use ra_ap_parser::{Edition, LexedStr, SyntaxKind};
use ra_ap_vfs::FileId;

/// Word → files whose code holds it, and file → its words (to patch a file).
///
/// Word ids are never retired: a word whose last file went away keeps an id
/// with an empty list, which costs its spelling and nothing else.
#[derive(Default)]
pub(crate) struct WordIndex {
    ids: HashMap<Box<str>, u32>,
    /// By word id: its spelling.
    spelling_of: Vec<Box<str>>,
    /// By word id; sorted, no duplicates.
    files_of: Vec<Vec<FileId>>,
    /// By word id: the files where the word stands as `word:` — the shape of
    /// every named field's declaration (and of struct literal fields, typed
    /// parameters and bindings, which only add candidates). Sorted.
    field_files_of: Vec<Vec<FileId>>,
    /// By word id: the files where the word stands as a member — right after
    /// `.` or `::`, or anywhere in a `use` item (`Shape::member`). Sorted.
    member_files_of: Vec<Vec<FileId>>,
    /// By word id: how many files declare a type of that name — `struct`,
    /// `enum`, `union`, `type`, `trait` right before it, or `as` (an import
    /// alias; a cast `as u32` counts too, which only keeps a name). Whether a
    /// name in a trait impl's header can be a project type (`surface::Witness`).
    declaring_files: Vec<u32>,
    /// By file: its words, those of them in field shape, those it declares
    /// as types; sorted, no duplicates. A file without words and globs is
    /// absent (an alias has words).
    words_of: HashMap<FileId, FileWords>,
    /// The files with a glob import not of std ([`Facts::glob`]); sorted.
    glob_files: Vec<FileId>,
    /// By target word id: the names it is also known by ([`Facts::aliases`]),
    /// with how many files say so.
    aliases_of: HashMap<u32, HashMap<u32, u32>>,
}

#[derive(Default)]
struct FileWords {
    words: Vec<u32>,
    fields: Vec<u32>,
    members: Vec<u32>,
    declared: Vec<u32>,
    glob: bool,
    /// (alias, target); sorted, no duplicates.
    aliases: Vec<(u32, u32)>,
}

impl WordIndex {
    /// Make `file` hold the words of `text`, or none (`None`: the file left the
    /// analysis).
    ///
    /// Cost: the file's text lexed once, plus one sorted insert per distinct
    /// word. A file indexed in increasing `FileId` order — as a build goes —
    /// appends to every list.
    pub(crate) fn set_file(&mut self, file: FileId, text: Option<&str>) {
        if let Some(old) = self.words_of.remove(&file) {
            for word in old.words {
                remove_sorted(&mut self.files_of[word as usize], file);
            }
            for word in old.fields {
                remove_sorted(&mut self.field_files_of[word as usize], file);
            }
            for word in old.members {
                remove_sorted(&mut self.member_files_of[word as usize], file);
            }
            for word in old.declared {
                self.declaring_files[word as usize] -= 1;
            }
            if old.glob {
                remove_sorted(&mut self.glob_files, file);
            }
            for (alias, target) in old.aliases {
                let known = self.aliases_of.get_mut(&target).expect("counted when set");
                let count = known.get_mut(&alias).expect("counted when set");
                *count -= 1;
                if *count == 0 {
                    known.remove(&alias);
                }
            }
        }
        let Some(text) = text else {
            return;
        };
        let mut new = FileWords::default();
        let facts = for_each_code_word(text, |_, word, shape| {
            let id = self.intern(word);
            new.words.push(id);
            if shape.field {
                new.fields.push(id);
            }
            if shape.member {
                new.members.push(id);
            }
            if shape.declared {
                new.declared.push(id);
            }
        });
        new.glob = facts.glob;
        if new.glob {
            insert_sorted(&mut self.glob_files, file);
        }
        new.aliases = facts
            .aliases
            .iter()
            .map(|(alias, target)| (self.intern(alias), self.intern(target)))
            .filter(|(alias, target)| alias != target)
            .collect();
        new.aliases.sort_unstable();
        new.aliases.dedup();
        for &(alias, target) in &new.aliases {
            *self
                .aliases_of
                .entry(target)
                .or_default()
                .entry(alias)
                .or_default() += 1;
        }
        for list in [
            &mut new.words,
            &mut new.fields,
            &mut new.members,
            &mut new.declared,
        ] {
            list.sort_unstable();
            list.dedup();
        }
        for &word in &new.words {
            insert_sorted(&mut self.files_of[word as usize], file);
        }
        for &word in &new.fields {
            insert_sorted(&mut self.field_files_of[word as usize], file);
        }
        for &word in &new.members {
            insert_sorted(&mut self.member_files_of[word as usize], file);
        }
        for &word in &new.declared {
            self.declaring_files[word as usize] += 1;
        }
        // `use super::*;` has no word but a glob to drop when patched.
        if !new.words.is_empty() || new.glob {
            self.words_of.insert(file, new);
        }
    }

    /// Whether some project file declares a type named `word` (see
    /// `declaring_files`).
    pub(crate) fn is_declared(&self, word: &str) -> bool {
        let word = word.strip_prefix("r#").unwrap_or(word);
        self.ids
            .get(word)
            .is_some_and(|&id| self.declaring_files[id as usize] > 0)
    }

    /// The files whose code holds `word` (a raw `r#x` is asked as `x`), in
    /// `FileId` order.
    pub(crate) fn files_with(&self, word: &str) -> &[FileId] {
        let word = word.strip_prefix("r#").unwrap_or(word);
        match self.ids.get(word) {
            Some(&id) => &self.files_of[id as usize],
            None => &[],
        }
    }

    /// The files where `word` stands in field shape (`word:`) — a subset of
    /// [`Self::files_with`] holding every declaration of a named field `word`.
    pub(crate) fn field_shape_files_with(&self, word: &str) -> &[FileId] {
        let word = word.strip_prefix("r#").unwrap_or(word);
        match self.ids.get(word) {
            Some(&id) => &self.field_files_of[id as usize],
            None => &[],
        }
    }

    /// The files where `word` stands as a member (`x.word`, `T::word`, in a
    /// `use` item) — a subset of [`Self::files_with`]. How a file names a field,
    /// a method, a variant or an associated item other than in a literal or a
    /// pattern naming its type, or through a glob import (`widening`).
    pub(crate) fn member_files_with(&self, word: &str) -> &[FileId] {
        let word = word.strip_prefix("r#").unwrap_or(word);
        match self.ids.get(word) {
            Some(&id) => &self.member_files_of[id as usize],
            None => &[],
        }
    }

    /// Whether `file` has a glob import not of std (`use a::*`, `use
    /// super::*`) — the names it brings are not written, a trait's among them.
    pub(crate) fn has_glob(&self, file: FileId) -> bool {
        self.glob_files.binary_search(&file).is_ok()
    }

    /// `name` and the names it is known by through aliases, transitively —
    /// `type A = m::S<u8>;`, `use m::S as A;` make `A` a name of `S`. What a
    /// file says `S::new()` with: it names `new` and one of these (or `Self`
    /// inside an `impl` that names one).
    ///
    /// Cost: one lookup per name found; aliases of a type are a handful.
    pub(crate) fn names_for(&self, name: &str) -> Vec<String> {
        let name = name.strip_prefix("r#").unwrap_or(name);
        let mut names = vec![name.to_owned()];
        let Some(&first) = self.ids.get(name) else {
            return names;
        };
        let mut seen = vec![first];
        let mut at = 0;
        while at < seen.len() {
            let target = seen[at];
            at += 1;
            for &alias in self
                .aliases_of
                .get(&target)
                .into_iter()
                .flat_map(|known| known.keys())
            {
                if !seen.contains(&alias) {
                    seen.push(alias);
                }
            }
        }
        names.extend(
            seen[1..]
                .iter()
                .map(|&id| self.spelling_of[id as usize].to_string()),
        );
        names
    }

    /// Distinct words, and (word, file) pairs — for the load's log line.
    pub(crate) fn size(&self) -> (usize, usize) {
        (
            self.ids.len(),
            self.words_of.values().map(|file| file.words.len()).sum(),
        )
    }

    fn intern(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.ids.get(word) {
            return id;
        }
        let id = u32::try_from(self.files_of.len()).expect("fewer than 2^32 distinct words");
        self.ids.insert(word.into(), id);
        self.spelling_of.push(word.into());
        self.files_of.push(Vec::new());
        self.field_files_of.push(Vec::new());
        self.member_files_of.push(Vec::new());
        self.declaring_files.push(0);
        id
    }
}

/// A raw identifier `r#x` as `x`.
fn unraw(word: &str) -> &str {
    word.strip_prefix("r#").unwrap_or(word)
}

fn insert_sorted(files: &mut Vec<FileId>, file: FileId) {
    if let Err(at) = files.binary_search(&file) {
        files.insert(at, file);
    }
}

fn remove_sorted(files: &mut Vec<FileId>, file: FileId) {
    if let Ok(at) = files.binary_search(&file) {
        files.remove(at);
    }
}

/// Byte offsets of `word` in the code of `text` — the start of the identifier
/// proper (after `r#`), in text order. What a sweep probes in a candidate file.
pub(crate) fn code_occurrences(text: &str, word: &str) -> Vec<usize> {
    occurrences(text, word, false)
}

/// [`code_occurrences`] in field shape only (`word:`, not `word::`): where a
/// named field `word` can be declared.
pub(crate) fn field_shape_occurrences(text: &str, word: &str) -> Vec<usize> {
    occurrences(text, word, true)
}

/// Every word in code of `text` with its byte offset, in text order — the
/// occurrences [`code_occurrences`] picks from, for all words at once.
#[cfg_attr(not(test), allow(dead_code))] // see `uses::library_uses_in`
pub(crate) fn code_words(text: &str) -> Vec<(usize, &str)> {
    let mut found = Vec::new();
    for_each_code_word(text, |start, word, _| found.push((start, word)));
    found
}

fn occurrences(text: &str, word: &str, field_shape_only: bool) -> Vec<usize> {
    let word = word.strip_prefix("r#").unwrap_or(word);
    let mut found = Vec::new();
    if word.is_empty() {
        return found;
    }
    for_each_code_word(text, |start, seen, shape| {
        if seen == word && (shape.field || !field_shape_only) {
            found.push(start);
        }
    });
    found
}

/// Where a word in code stands, by its neighbouring tokens.
#[derive(Clone, Copy, Default)]
struct Shape {
    /// The next token past whitespace and comments is a single `:` (a `::`
    /// path separator is two adjacent `:` tokens).
    field: bool,
    /// The previous one is `.` or `::` (a field or method, a path's segment
    /// past the first), or the word is in a `use` item (a group's member, an
    /// alias). A `..word` range counts too, which only adds a file.
    member: bool,
    /// The previous one is `struct`, `enum`, `union`, `type`, `trait` or `as`.
    declared: bool,
}

/// What a file's tokens tell beyond its words.
#[derive(Default)]
struct Facts<'t> {
    /// (alias, target): `type A<…> = path::S<…>;` (the target is the last
    /// segment of the path the alias starts with — `Box<dyn S>` is `Box`, an
    /// associated type `type Item = S;` counts too) and `use path::S as A;`.
    aliases: Vec<(&'t str, &'t str)>,
    /// A glob import that may bring in names from outside the file: a `use`
    /// with a `*`, but not one whose path starts with `std`, `core` or
    /// `alloc`, and not `use super::*` inside an inline module (it brings the
    /// enclosing module's names, which are in the file — or came by a glob of
    /// that module, counted by itself). Lexed, not parsed: a `*` repetition
    /// after a `use` in a macro pattern counts too, which only adds a file.
    glob: bool,
}

/// How a word in code stands for [`path_uses`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PathUse {
    /// Somewhere as the segment right after one of the prefixes, or one of
    /// the prefixes is renamed — see [`path_uses`].
    pub(crate) after_prefix: bool,
    /// Somewhere not as the name an item or a `let` declares (`mod tests`,
    /// `fn new`): a use that a glob import may resolve.
    pub(crate) undeclared: bool,
}

/// How `word` stands in `text`: [`PathUse::after_prefix`] — as the next
/// segment after one of `prefixes` (`super::word`, `crate::word`,
/// `m::word`), in a use group too (`use super::{a, word}`), or one of
/// `prefixes` is renamed (`use super as up;`: `up::word` reaches it unseen; a
/// cast `m as u8` counts too, which only keeps a file); and
/// [`PathUse::undeclared`]. How a file below a module names a private item of
/// it (`widening`).
///
/// Cost: one lexing of `text` (≈ 0.2 ms a file).
pub(crate) fn path_uses(text: &str, word: &str, prefixes: &[&str]) -> PathUse {
    let word = word.strip_prefix("r#").unwrap_or(word);
    let mut found = PathUse::default();
    let lexed = LexedStr::new(Edition::Edition2015, text);
    let significant: Vec<usize> = (0..lexed.len())
        .filter(|&token| {
            !matches!(
                lexed.kind(token),
                SyntaxKind::WHITESPACE | SyntaxKind::COMMENT
            )
        })
        .collect();
    let kind = |at: usize| lexed.kind(significant[at]);
    let spelled = |at: usize| {
        let text = lexed.text(significant[at]);
        text.strip_prefix("r#").unwrap_or(text)
    };
    let is_prefix = |at: usize| prefixes.contains(&spelled(at));
    // The segment before a `::` ending right before `at`, if one does.
    let qualifier = |at: usize| -> Option<usize> {
        if at >= 1 && kind(at - 1) == SyntaxKind::COLON2 {
            return (at >= 2).then(|| at - 2);
        }
        let colons = at >= 2
            && kind(at - 1) == SyntaxKind::COLON
            && kind(at - 2) == SyntaxKind::COLON
            && lexed.text_range(significant[at - 2]).end
                == lexed.text_range(significant[at - 1]).start;
        (colons && at >= 3).then(|| at - 3)
    };
    // Whether the word at `at` is right after a prefix and `::`, or in a
    // group whose `{` is.
    let after_prefix = |at: usize| -> bool {
        if qualifier(at).is_some_and(is_prefix) {
            return true;
        }
        // In a group: back to the group's unmatched `{`, then its qualifier.
        if at == 0 || !matches!(kind(at - 1), SyntaxKind::COMMA | SyntaxKind::L_CURLY) {
            return false;
        }
        let mut depth = 0usize;
        let mut back = at;
        while back > 0 {
            back -= 1;
            match kind(back) {
                SyntaxKind::R_CURLY | SyntaxKind::R_PAREN | SyntaxKind::R_BRACK => depth += 1,
                SyntaxKind::L_PAREN | SyntaxKind::L_BRACK if depth == 0 => return false,
                SyntaxKind::L_CURLY if depth == 0 => return qualifier(back).is_some_and(is_prefix),
                SyntaxKind::L_CURLY | SyntaxKind::L_PAREN | SyntaxKind::L_BRACK => depth -= 1,
                SyntaxKind::SEMICOLON if depth == 0 => return false,
                _ => {}
            }
        }
        false
    };
    let declares = |at: usize| {
        at >= 1
            && (matches!(
                kind(at - 1),
                SyntaxKind::MOD_KW
                    | SyntaxKind::FN_KW
                    | SyntaxKind::STRUCT_KW
                    | SyntaxKind::ENUM_KW
                    | SyntaxKind::TRAIT_KW
                    | SyntaxKind::TYPE_KW
                    | SyntaxKind::CONST_KW
                    | SyntaxKind::STATIC_KW
                    | SyntaxKind::LET_KW
            ) || (kind(at - 1) == SyntaxKind::IDENT && spelled(at - 1) == "union"))
    };
    for at in 0..significant.len() {
        if kind(at) == SyntaxKind::AS_KW && at >= 1 && is_prefix(at - 1) {
            found.after_prefix = true;
        }
        if kind(at) != SyntaxKind::IDENT || spelled(at) != word {
            continue;
        }
        found.after_prefix |= after_prefix(at);
        found.undeclared |= !declares(at);
        if found.after_prefix && found.undeclared {
            break;
        }
    }
    found
}

/// Call `visit(offset, word, shape)` for every word in code; see the module
/// header.
fn for_each_code_word<'t>(
    text: &'t str,
    mut visit: impl FnMut(usize, &'t str, Shape),
) -> Facts<'t> {
    let lexed = LexedStr::new(Edition::Edition2015, text);
    let mut facts = Facts::default();
    // A type alias being lexed: its name, the depth of `<` in its generics,
    // whether past its `=`, and the last segment of the target's path so far
    // (with whether the previous token was that segment).
    struct Alias<'t> {
        name: &'t str,
        angles: usize,
        target: Option<(&'t str, bool)>,
        past_eq: bool,
    }
    let mut alias: Option<Alias<'t>> = None;
    let mut after_type_kw = false;
    // The significant tokens of the `use` item being lexed, `None` outside one.
    let mut in_use: Option<Vec<(SyntaxKind, &str)>> = None;
    // `true` for each open `{` that opens an inline module; the last two
    // significant tokens, to tell it (`mod name {`).
    let mut braces: Vec<bool> = Vec::new();
    let mut last_two = [SyntaxKind::EOF, SyntaxKind::EOF];
    let trivia = |token: usize| {
        matches!(
            lexed.kind(token),
            SyntaxKind::WHITESPACE | SyntaxKind::COMMENT
        )
    };
    let shape = |token: usize| {
        let mut next = token + 1;
        while next < lexed.len() && trivia(next) {
            next += 1;
        }
        let field = next < lexed.len()
            && lexed.kind(next) == SyntaxKind::COLON
            && (next + 1 >= lexed.len() || lexed.kind(next + 1) != SyntaxKind::COLON);
        let mut previous = token;
        while previous > 0 && trivia(previous - 1) {
            previous -= 1;
        }
        // `union` is a word in edition 2015, a contextual keyword after it.
        let declared = previous > 0
            && (matches!(
                lexed.kind(previous - 1),
                SyntaxKind::STRUCT_KW
                    | SyntaxKind::ENUM_KW
                    | SyntaxKind::TYPE_KW
                    | SyntaxKind::TRAIT_KW
                    | SyntaxKind::AS_KW
            ) || (lexed.kind(previous - 1) == SyntaxKind::IDENT
                && lexed.text(previous - 1) == "union"));
        // `::` is one `COLON2` or two adjacent `COLON`s, as the lexer goes.
        let member = previous > 0
            && match lexed.kind(previous - 1) {
                SyntaxKind::DOT | SyntaxKind::COLON2 => true,
                SyntaxKind::COLON => {
                    previous > 1
                        && lexed.kind(previous - 2) == SyntaxKind::COLON
                        && lexed.text_range(previous - 2).end
                            == lexed.text_range(previous - 1).start
                }
                _ => false,
            };
        Shape {
            field,
            member,
            declared,
        }
    };
    for token in 0..lexed.len() {
        // Sliced from `text` itself, so the words outlive the lexer.
        let range = lexed.text_range(token);
        let start = range.start;
        let token_text = &text[range];
        let kind = lexed.kind(token);
        if !trivia(token) {
            match kind {
                SyntaxKind::L_CURLY => braces.push(
                    last_two[0] == SyntaxKind::MOD_KW && last_two[1] == SyntaxKind::IDENT,
                ),
                SyntaxKind::R_CURLY => {
                    braces.pop();
                }
                _ => {}
            }
            match (&mut in_use, kind) {
                (_, SyntaxKind::USE_KW) => in_use = Some(Vec::new()),
                (Some(seen), SyntaxKind::SEMICOLON) => {
                    let seen = std::mem::take(seen);
                    in_use = None;
                    let mut path = seen.iter().map(|(kind, _)| *kind).skip_while(|kind| {
                        matches!(kind, SyntaxKind::COLON | SyntaxKind::COLON2)
                    });
                    let first_is_std = seen
                        .iter()
                        .find(|(kind, _)| !matches!(kind, SyntaxKind::COLON | SyntaxKind::COLON2))
                        .is_some_and(|(kind, text)| {
                            *kind == SyntaxKind::IDENT && matches!(*text, "std" | "core" | "alloc")
                        });
                    let star = seen.iter().any(|(kind, _)| *kind == SyntaxKind::STAR);
                    let parents_glob = path.next() == Some(SyntaxKind::SUPER_KW)
                        && path
                            .filter(|kind| *kind != SyntaxKind::COLON && *kind != SyntaxKind::COLON2)
                            .eq([SyntaxKind::STAR]);
                    let in_inline_module = braces.iter().any(|&module| module);
                    facts.glob |= star && !first_is_std && !(parents_glob && in_inline_module);
                }
                (Some(seen), _) => {
                    if let [.., (SyntaxKind::IDENT, target), (SyntaxKind::AS_KW, _)] = seen[..]
                        && kind == SyntaxKind::IDENT
                    {
                        facts.aliases.push((unraw(token_text), unraw(target)));
                    }
                    seen.push((kind, token_text));
                }
                (None, _) => {}
            }
            let mut ended = false;
            if let Some(state) = &mut alias {
                if !state.past_eq {
                    match kind {
                        SyntaxKind::L_ANGLE => state.angles += 1,
                        SyntaxKind::R_ANGLE => state.angles = state.angles.saturating_sub(1),
                        SyntaxKind::EQ if state.angles == 0 => state.past_eq = true,
                        // `type Item;`, `type Item: Bound;` in a trait.
                        SyntaxKind::SEMICOLON | SyntaxKind::L_CURLY | SyntaxKind::R_CURLY => {
                            ended = true
                        }
                        _ => {}
                    }
                } else {
                    let separator = matches!(
                        kind,
                        SyntaxKind::COLON
                            | SyntaxKind::COLON2
                            | SyntaxKind::CRATE_KW
                            | SyntaxKind::SELF_KW
                            | SyntaxKind::SUPER_KW
                    );
                    match (kind, state.target) {
                        // `dyn` is a word in edition 2015.
                        (SyntaxKind::IDENT, None) if token_text == "dyn" => {}
                        (SyntaxKind::IDENT, None | Some((_, false))) => {
                            state.target = Some((unraw(token_text), true))
                        }
                        (_, Some((target, _))) if separator => {
                            state.target = Some((target, false))
                        }
                        (_, None) if separator => {}
                        _ => {
                            if let Some((target, _)) = state.target {
                                facts.aliases.push((state.name, target));
                            }
                            ended = true;
                        }
                    }
                }
            }
            if ended {
                alias = None;
            }
            if after_type_kw && kind == SyntaxKind::IDENT {
                alias = Some(Alias {
                    name: unraw(token_text),
                    angles: 0,
                    target: None,
                    past_eq: false,
                });
            }
            after_type_kw = kind == SyntaxKind::TYPE_KW;
            last_two = [last_two[1], kind];
        }
        match kind {
            SyntaxKind::IDENT => {
                let mut shape = shape(token);
                shape.member |= in_use.is_some();
                match token_text.strip_prefix("r#") {
                    Some(raw) => visit(start + 2, raw, shape),
                    None => visit(start, token_text, shape),
                }
            }
            SyntaxKind::STRING => {
                for (offset, word) in format_captures(token_text) {
                    visit(start + offset, word, Shape::default());
                }
            }
            _ => {}
        }
    }
    facts
}

/// The implicit captures of a format string: `name` in `{name}`, `{name:?}`,
/// `{name:>width$}` — an identifier right after an unescaped `{`. Offsets are
/// into `literal`. Any string is read this way, format string or not: a
/// `{word}` in an ordinary string is one extra candidate, which rust-analyzer
/// then resolves to nothing.
fn format_captures(literal: &str) -> Vec<(usize, &str)> {
    let bytes = literal.as_bytes();
    let mut captures = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] != b'{' {
            at += 1;
            continue;
        }
        if bytes.get(at + 1) == Some(&b'{') {
            at += 2;
            continue;
        }
        let begin = at + 1;
        let end = literal[begin..]
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .map_or(literal.len(), |len| begin + len);
        let word = &literal[begin..end];
        let starts_like_a_name = word
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_');
        if starts_like_a_name && matches!(bytes.get(end), Some(b'}' | b':')) {
            captures.push((begin, word));
        }
        at = end.max(begin);
    }
    captures
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(id: u32) -> FileId {
        FileId::from_raw(id)
    }

    /// Code words, and nothing that only looks like one: comments, doc
    /// comments, ordinary string contents, char and number literals.
    #[test]
    fn only_words_in_code_are_occurrences() {
        let text = r##"
/// radius_override in a doc comment
// radius_override in a comment
/* radius_override in a block */
fn f(p: &Probe) -> u32 {
    let s = "radius_override in a string";
    let raw = r#"radius_override in a raw string"#;
    vec![p.radius_override];
    println!("{radius_override}");
    p.r#radius_override + p.radius_override
}
"##;
        let found = code_occurrences(text, "radius_override");
        let lines: Vec<usize> = found
            .iter()
            .map(|&at| text[..at].matches('\n').count())
            .collect();
        // The macro argument, the format capture, the raw identifier and the
        // plain field read.
        assert_eq!(lines, vec![7, 8, 9, 9], "{found:?}");
        for &at in &found {
            assert!(
                text[at..].starts_with("radius_override"),
                "offset {at} is not the word"
            );
        }
        assert!(code_occurrences(text, "comment").is_empty());
        assert!(code_occurrences(text, "string").is_empty());
    }

    /// Field shape is `word:` — a declaration, a struct literal field, a typed
    /// parameter — and not a call, a method, a path segment (`word::`), or the
    /// word in a comment.
    #[test]
    fn field_shape_is_a_word_before_a_single_colon() {
        let text = "struct Bit { pub rop: u32 }\n\
                    fn rop(rop : u32) {}\n\
                    fn f(b: Bit) { b.rop; rop(1); rop::x(); Bit { rop: 2 }; } // rop: no\n";
        let lines: Vec<usize> = field_shape_occurrences(text, "rop")
            .iter()
            .map(|&at| text[..at].matches('\n').count())
            .collect();
        assert_eq!(lines, [0, 1, 2], "declaration, parameter, literal");
        let mut index = WordIndex::default();
        index.set_file(file(1), Some(text));
        index.set_file(file(2), Some("fn g() { rop(); x.rop; }"));
        assert_eq!(index.files_with("rop"), [file(1), file(2)]);
        assert_eq!(index.field_shape_files_with("rop"), [file(1)]);
        index.set_file(file(1), None);
        assert!(index.field_shape_files_with("rop").is_empty());
    }

    /// A member: after `.` or `::` (one `COLON2` token or two `:`), or in a
    /// `use` item; not a declaration, a binding, a call, a field in a literal.
    #[test]
    fn a_member_is_a_word_after_a_dot_or_a_path_separator_or_in_a_use() {
        let member = |text: &str| {
            let mut index = WordIndex::default();
            index.set_file(file(1), Some(text));
            !index.member_files_with("err").is_empty()
        };
        for text in [
            "fn f(x: S) { x.err; }",
            "fn f(x: S) { x . err(); }",
            "fn f() { S::err(); }",
            "fn f() { Self :: err; }",
            "fn f() { <S as T>::err(); }",
            "use a::{b, err};",
            "use a::b as err;",
            "fn f() { x.0.err; }",
        ] {
            assert!(member(text), "{text}");
        }
        for text in [
            "fn err() {}",
            "fn f(err: u8) { let err = 1; err(); }",
            "fn f() { S { err: 1 }; }",
            "struct S { err: u8 }",
            "// x.err\nfn f() { \"x.err\"; }",
        ] {
            assert!(!member(text), "{text}");
        }
        let mut index = WordIndex::default();
        index.set_file(file(1), Some("fn f(x: S) { x.err; }"));
        index.set_file(file(1), Some("fn err() {}"));
        assert!(index.member_files_with("err").is_empty(), "patched away");
        assert_eq!(index.files_with("err"), [file(1)]);
    }

    /// A glob that may bring names from outside the file counts; one of std,
    /// and `use super::*` inside an inline module (the file's own names), do
    /// not — a top-level `use super::*` does.
    #[test]
    fn a_glob_from_outside_the_file_is_flagged() {
        let flagged = |text: &str| {
            let mut index = WordIndex::default();
            index.set_file(file(1), Some(text));
            index.has_glob(file(1))
        };
        assert!(flagged("use crate::prelude::*;"));
        assert!(flagged("use super::*;"));
        assert!(flagged("use a::{b::*, c};"));
        assert!(flagged("mod tests { use crate::x::*; }"));
        assert!(flagged("mod tests { use super::inner::*; }"));
        assert!(!flagged("use std::io::prelude::*;"));
        assert!(!flagged("use ::core::ops::*;"));
        assert!(!flagged("mod tests { use super::*; fn t() {} }"));
        assert!(!flagged("fn f() { let x = 2 * 3; }"));
        assert!(!flagged("use a::b; fn f() -> u8 { 2 * 3 }"));
        // After the inline module closes, a top-level glob counts again.
        assert!(flagged("mod tests { use super::*; }\nuse super::*;"));
        let mut index = WordIndex::default();
        index.set_file(file(1), Some("use super::*;"));
        index.set_file(file(1), Some("use super::x;"));
        assert!(!index.has_glob(file(1)), "a patched file drops its glob");
    }

    /// How a private item of a module is named below it: after a path to the
    /// module (a group too, a rename of the module), and whether anywhere not
    /// as a declaration.
    #[test]
    fn path_uses_tell_a_path_to_the_module_from_an_own_name() {
        let prefixes = ["super", "self", "crate", "m"];
        let uses = |text: &str| {
            let found = path_uses(text, "t", &prefixes);
            (found.after_prefix, found.undeclared)
        };
        assert_eq!(uses("fn f() { super::t(); }"), (true, true));
        assert_eq!(uses("use crate::t;"), (true, true));
        assert_eq!(uses("use super::{a, t};"), (true, true));
        assert_eq!(uses("use super::{a::{b, t}};"), (false, true), "`a::t`");
        assert_eq!(uses("fn f() { m::t(); }"), (true, true));
        assert_eq!(uses("use super as up; fn f() { up::t(); }"), (true, true));
        // Its own: imported from elsewhere, declared, a struct literal field.
        assert_eq!(uses("use rust_i18n::t; fn f() { t!(\"k\"); }"), (false, true));
        assert_eq!(uses("mod t;"), (false, false));
        assert_eq!(uses("fn t() {}"), (false, false));
        assert_eq!(uses("fn f() { S { a, t }; g(a, t); }"), (false, true));
        assert_eq!(uses("fn f() {} // super::t"), (false, false));
    }

    /// Aliases: a `type` whose target's path ends in the name, a renaming
    /// import — transitively, patched with their file.
    #[test]
    fn a_name_is_known_by_its_aliases() {
        let mut index = WordIndex::default();
        index.set_file(file(1), Some("pub struct S; impl S { fn new() -> S { S } }"));
        index.set_file(
            file(2),
            Some("pub type A = crate::m::S;\npub type G<T = u8> = S<T>;\ntype Boxed = Box<dyn S>;"),
        );
        index.set_file(file(3), Some("use other::{A as B, C};\ntype Fp = fn() -> S;"));
        let mut names = index.names_for("S");
        names.sort();
        assert_eq!(names, ["A", "B", "G", "S"]);
        assert_eq!(index.names_for("Box"), ["Box", "Boxed"]);
        index.set_file(file(3), None);
        let mut names = index.names_for("S");
        names.sort();
        assert_eq!(names, ["A", "G", "S"], "a removed file takes its aliases");
        assert_eq!(index.names_for("Nothing"), ["Nothing"]);
    }

    #[test]
    fn format_captures_skip_escapes_positions_and_expressions() {
        let literal = r#""{a} {b:?} {c:>w$} {{d}} {0} {} {e.f} {_g}""#;
        let captured: Vec<&str> = format_captures(literal)
            .into_iter()
            .map(|(_, word)| word)
            .collect();
        assert_eq!(captured, ["a", "b", "c", "_g"]);
    }

    /// A file's words follow its text: replaced on edit, gone on removal, and
    /// no other file's lists are touched.
    #[test]
    fn a_patched_file_changes_only_its_own_words() {
        let mut index = WordIndex::default();
        index.set_file(file(1), Some("fn alpha() { beta(); }"));
        index.set_file(file(2), Some("fn beta() {}"));
        assert_eq!(index.files_with("beta"), [file(1), file(2)]);
        assert_eq!(index.files_with("alpha"), [file(1)]);

        index.set_file(file(1), Some("fn gamma() {} // beta"));
        assert_eq!(
            index.files_with("beta"),
            [file(2)],
            "a word only in a comment is not code"
        );
        assert!(index.files_with("alpha").is_empty());
        assert_eq!(index.files_with("gamma"), [file(1)]);

        index.set_file(file(2), None);
        assert!(index.files_with("beta").is_empty());
        // `fn` is a keyword, not a word; ids stay for words gone from every file.
        assert_eq!(
            index.size(),
            (3, 1),
            "alpha, beta, gamma; file 1 holds gamma alone"
        );
        assert_eq!(index.files_with("r#gamma"), [file(1)]);
    }

    /// Oracle (a) of the plan on real sources — this crate's own: for each
    /// word, every file where the parsed syntax tree has the word as an
    /// identifier token is a candidate. The tree is the parser's, a different
    /// path from the index's lexer walk, and it keeps macro token trees.
    #[test]
    fn candidates_cover_every_file_with_the_word_as_an_identifier() {
        use ra_ap_syntax::{AstNode, SourceFile, SyntaxKind as Kind};

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let (paths, _) = rmc_indexing::indexing::collect_project_rust_files(&root);
        assert!(
            paths.len() > 20,
            "too few sources to judge: {}",
            paths.len()
        );
        let texts: Vec<String> = paths
            .iter()
            .map(|path| std::fs::read_to_string(path).expect("read a source"))
            .collect();
        let mut index = WordIndex::default();
        for (id, text) in texts.iter().enumerate() {
            index.set_file(file(id as u32), Some(text));
        }

        let words = [
            "new",
            "map",
            "clone",
            "spawn",
            "Location",
            "budget",
            "vfs",
            "SweepBudget",
        ];
        let mut judged = 0;
        for word in words {
            let mut in_tree = Vec::new();
            for (id, text) in texts.iter().enumerate() {
                let tree = SourceFile::parse(text, ra_ap_syntax::Edition::Edition2024).tree();
                let has = tree
                    .syntax()
                    .descendants_with_tokens()
                    .filter_map(|element| element.into_token())
                    .any(|token| token.kind() == Kind::IDENT && token.text() == word);
                if has {
                    in_tree.push(file(id as u32));
                }
            }
            let candidates = index.files_with(word);
            for wanted in &in_tree {
                assert!(
                    candidates.contains(wanted),
                    "`{word}` is an identifier in {} but not a candidate",
                    paths[wanted.index() as usize].display()
                );
            }
            judged += in_tree.len();
        }
        assert!(
            judged > 50,
            "the words barely occur ({judged} files): the oracle judged nothing"
        );
    }
}
