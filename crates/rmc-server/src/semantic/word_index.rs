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
    /// By file; sorted, no duplicates. A file without words is absent.
    words_of: HashMap<FileId, Vec<u32>>,
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
            for word in old {
                let files = &mut self.files_of[word as usize];
                if let Ok(at) = files.binary_search(&file) {
                    files.remove(at);
                }
            }
        }
        let Some(text) = text else {
            return;
        };
        let mut words: Vec<u32> = Vec::new();
        for_each_code_word(text, |_, word| words.push(self.intern(word)));
        words.sort_unstable();
        words.dedup();
        for &word in &words {
            let files = &mut self.files_of[word as usize];
            if let Err(at) = files.binary_search(&file) {
                files.insert(at, file);
            }
        }
        if !words.is_empty() {
            self.words_of.insert(file, words);
        }
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

    /// Distinct words, and (word, file) pairs — for the load's log line.
    pub(crate) fn size(&self) -> (usize, usize) {
        (self.ids.len(), self.words_of.values().map(Vec::len).sum())
    }

    fn intern(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.ids.get(word) {
            return id;
        }
        let id = u32::try_from(self.files_of.len()).expect("fewer than 2^32 distinct words");
        self.ids.insert(word.into(), id);
        self.files_of.push(Vec::new());
        id
    }
}

/// Byte offsets of `word` in the code of `text` — the start of the identifier
/// proper (after `r#`), in text order. What a sweep probes in a candidate file.
pub(crate) fn code_occurrences(text: &str, word: &str) -> Vec<usize> {
    let word = word.strip_prefix("r#").unwrap_or(word);
    let mut found = Vec::new();
    if word.is_empty() {
        return found;
    }
    for_each_code_word(text, |start, seen| {
        if seen == word {
            found.push(start);
        }
    });
    found
}

/// Call `visit(offset, word)` for every word in code; see the module header.
fn for_each_code_word<'t>(text: &'t str, mut visit: impl FnMut(usize, &'t str)) {
    let lexed = LexedStr::new(Edition::Edition2015, text);
    for token in 0..lexed.len() {
        // Sliced from `text` itself, so the words outlive the lexer.
        let range = lexed.text_range(token);
        let start = range.start;
        let token_text = &text[range];
        match lexed.kind(token) {
            SyntaxKind::IDENT => match token_text.strip_prefix("r#") {
                Some(raw) => visit(start + 2, raw),
                None => visit(start, token_text),
            },
            SyntaxKind::STRING => {
                for (offset, word) in format_captures(token_text) {
                    visit(start + offset, word);
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
