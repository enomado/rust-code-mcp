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
    /// By word id; sorted, no duplicates.
    files_of: Vec<Vec<FileId>>,
    /// By word id: the files where the word stands as `word:` — the shape of
    /// every named field's declaration (and of struct literal fields, typed
    /// parameters and bindings, which only add candidates). Sorted.
    field_files_of: Vec<Vec<FileId>>,
    /// By word id: how many files declare a type of that name — `struct`,
    /// `enum`, `union`, `type`, `trait` right before it, or `as` (an import
    /// alias; a cast `as u32` counts too, which only keeps a name). Whether a
    /// name in a trait impl's header can be a project type (`surface::Witness`).
    declaring_files: Vec<u32>,
    /// By file: its words, those of them in field shape, those it declares
    /// as types; sorted, no duplicates. A file without words is absent.
    words_of: HashMap<FileId, FileWords>,
}

#[derive(Default)]
struct FileWords {
    words: Vec<u32>,
    fields: Vec<u32>,
    declared: Vec<u32>,
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
            for word in old.declared {
                self.declaring_files[word as usize] -= 1;
            }
        }
        let Some(text) = text else {
            return;
        };
        let mut new = FileWords::default();
        for_each_code_word(text, |_, word, shape| {
            let id = self.intern(word);
            new.words.push(id);
            if shape.field {
                new.fields.push(id);
            }
            if shape.declared {
                new.declared.push(id);
            }
        });
        for list in [&mut new.words, &mut new.fields, &mut new.declared] {
            list.sort_unstable();
            list.dedup();
        }
        for &word in &new.words {
            insert_sorted(&mut self.files_of[word as usize], file);
        }
        for &word in &new.fields {
            insert_sorted(&mut self.field_files_of[word as usize], file);
        }
        for &word in &new.declared {
            self.declaring_files[word as usize] += 1;
        }
        if !new.words.is_empty() {
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
        self.files_of.push(Vec::new());
        self.field_files_of.push(Vec::new());
        self.declaring_files.push(0);
        id
    }
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
    /// The previous one is `struct`, `enum`, `union`, `type`, `trait` or `as`.
    declared: bool,
}

/// Call `visit(offset, word, shape)` for every word in code; see the module
/// header.
fn for_each_code_word<'t>(text: &'t str, mut visit: impl FnMut(usize, &'t str, Shape)) {
    let lexed = LexedStr::new(Edition::Edition2015, text);
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
        Shape { field, declared }
    };
    for token in 0..lexed.len() {
        // Sliced from `text` itself, so the words outlive the lexer.
        let range = lexed.text_range(token);
        let start = range.start;
        let token_text = &text[range];
        match lexed.kind(token) {
            SyntaxKind::IDENT => match token_text.strip_prefix("r#") {
                Some(raw) => visit(start + 2, raw, shape(token)),
                None => visit(start, token_text, shape(token)),
            },
            SyntaxKind::STRING => {
                for (offset, word) in format_captures(token_text) {
                    visit(start + offset, word, Shape::default());
                }
            }
            _ => {}
        }
    }
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
