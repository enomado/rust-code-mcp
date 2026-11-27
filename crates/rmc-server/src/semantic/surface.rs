//! A file's **surface**: what of its text other files' analysis can depend on.
//! A refresh compares the surface of an edited file before and after, and only
//! when it changed does the table of uses stop vouching for other files
//! (`uses` module header, plan step `uses-freshness`).
//!
//! # What is surface and what is not
//!
//! The surface is the file's tokens (no whitespace, no comments) with the body
//! of every plain `fn` left out — the body is private to it: name resolution
//! elsewhere sees the module's items, type inference elsewhere sees
//! signatures. What a body holds that *is* visible elsewhere is kept, in text
//! order, in place of the body:
//!
//! - `impl` blocks, at any depth: an `impl` inside a body adds inherent methods
//!   or trait implementations to a type for the whole crate graph;
//! - `macro_rules!` / `macro` definitions (`#[macro_export]` ones are
//!   crate-visible; kept all, it costs nothing to be careful);
//! - macro calls, except the [`EXPRESSION_MACROS`] of std, `log`/`tracing` and
//!   `anyhow` (an unknown macro may expand to an `impl`); kept as their text.
//!
//! Other items in a body (`fn`, `struct`, `use`, `const`, …) are visible only
//! inside that block; they are looked into for the three kinds above but are
//! not surface themselves.
//!
//! A body stays surface whole when it can leak out of its `fn`:
//!
//! - `-> impl Trait` (anywhere in the return type): the hidden type and its
//!   auto traits come from the body;
//! - `async fn` / `gen fn`: the same, through the future the body becomes;
//! - `const fn`: a const evaluation in another signature (`[u8; f()]`) runs it.
//!
//! An item with an attribute other than the inert built-ins ([`INERT_ATTRS`])
//! is surface whole: an attribute macro (`#[async_trait]`, any proc macro)
//! may rewrite bodies into items. `#[cfg(...)]`, `#[test]`, `#[allow]` and
//! the like leave the item as written.
//!
//! # What this does not see
//!
//! A user macro shadowing a std expression macro (`macro_rules! assert_eq`)
//! and expanding to an `impl` — the call would count as private. A file
//! `include!`d by another is surface whole only if the including file's own
//! surface changes; its edits are seen as its own. The edition the text is
//! parsed with is the latest: an edition-2015 file using `async` as a name
//! parses differently, but both the before and after text alike.
//!
//! # Cost
//!
//! Parsing the edited file's old and new text — the files of one refresh,
//! usually one or two, a millisecond or so each — and nothing else.

use ra_ap_syntax::ast::{self, HasAttrs};
use ra_ap_syntax::{AstNode, Edition, NodeOrToken, SourceFile, SyntaxKind, SyntaxNode};

/// Macros of std, `log`/`tracing` and `anyhow` that expand to an expression or
/// to block-local statements — never to an item visible outside the block —
/// matched by the last segment of the call's path.
const EXPRESSION_MACROS: &[&str] = &[
    // std / core / alloc
    "assert",
    "assert_eq",
    "assert_ne",
    "cfg",
    "column",
    "compile_error",
    "concat",
    "dbg",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
    "env",
    "eprint",
    "eprintln",
    "file",
    "format",
    "format_args",
    "include_bytes",
    "include_str",
    "line",
    "matches",
    "module_path",
    "option_env",
    "panic",
    "print",
    "println",
    "stringify",
    "todo",
    "unimplemented",
    "unreachable",
    "vec",
    "write",
    "writeln",
    // log / tracing
    "debug",
    "debug_span",
    "error",
    "error_span",
    "event",
    "info",
    "info_span",
    "span",
    "trace",
    "trace_span",
    "warn",
    "warn_span",
    // anyhow
    "anyhow",
    "bail",
    "ensure",
    "format_err",
];

/// Built-in attributes that leave an item as written (matched by the first
/// segment of the attribute's path: `clippy::…`, `rustfmt::…` too).
const INERT_ATTRS: &[&str] = &[
    "allow",
    "bench",
    "cfg",
    "clippy",
    "cold",
    "deny",
    "deprecated",
    "doc",
    "expect",
    "forbid",
    "ignore",
    "inline",
    "must_use",
    "no_mangle",
    "rustfmt",
    "should_panic",
    "test",
    "track_caller",
    "warn",
];

/// See the module header. Two surfaces are compared, nothing else is done
/// with one.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Surface(Vec<String>);

/// The surface of a file's text; see the module header.
pub(crate) fn surface_of(text: &str) -> Surface {
    let file = SourceFile::parse(text, Edition::CURRENT).tree();
    let mut tokens = Vec::new();
    emit(file.syntax(), false, &mut tokens);
    Surface(tokens)
}

/// The tokens of `node`, fn bodies left out unless `whole` or the body can
/// leak (`keeps_body`); `whole` — inside an item an attribute may rewrite.
fn emit(node: &SyntaxNode, whole: bool, out: &mut Vec<String>) {
    let whole = whole || ast::Item::cast(node.clone()).is_some_and(|item| may_be_rewritten(&item));
    for element in node.children_with_tokens() {
        match element {
            NodeOrToken::Token(token) => {
                // Doc comments are tokens of their own kinds in this parser,
                // not trivia; their text resolves nothing.
                let silent = token.kind().is_trivia()
                    || matches!(
                        token.kind(),
                        SyntaxKind::INNER_DOC_COMMENT | SyntaxKind::OUTER_DOC_COMMENT
                    );
                if !silent {
                    out.push(token.text().to_owned());
                }
            }
            NodeOrToken::Node(child) => {
                if child.kind() == SyntaxKind::DOC_COMMENT {
                    continue;
                }
                if !whole && is_private_body(&child) {
                    out.push("{".to_owned());
                    emit_visible_in_body(&child, out);
                    out.push("}".to_owned());
                } else {
                    emit(&child, whole, out);
                }
            }
        }
    }
}

/// Of a private body: the `impl`s, macro definitions and macro calls in it, at
/// any depth, each as its own surface.
fn emit_visible_in_body(node: &SyntaxNode, out: &mut Vec<String>) {
    for child in node.children() {
        let visible = match child.kind() {
            SyntaxKind::IMPL | SyntaxKind::MACRO_RULES | SyntaxKind::MACRO_DEF => true,
            SyntaxKind::MACRO_CALL => !ast::MacroCall::cast(child.clone())
                .and_then(|call| call.path())
                .and_then(|path| path.segment())
                .and_then(|segment| segment.name_ref())
                .is_some_and(|name| EXPRESSION_MACROS.contains(&&*name.text())),
            _ => false,
        };
        if visible {
            emit(&child, false, out);
        } else {
            emit_visible_in_body(&child, out);
        }
    }
}

/// Whether `node` is the body of a `fn` that nothing outside the `fn` sees.
fn is_private_body(node: &SyntaxNode) -> bool {
    let Some(block) = ast::BlockExpr::cast(node.clone()) else {
        return false;
    };
    let Some(function) = node.parent().and_then(ast::Fn::cast) else {
        return false;
    };
    function.body().as_ref() == Some(&block)
        && function.const_token().is_none()
        && function.async_token().is_none()
        && function.gen_token().is_none()
        && !function.ret_type().is_some_and(|ret| {
            ret.syntax()
                .descendants_with_tokens()
                .any(|element| element.kind() == SyntaxKind::IMPL_KW)
        })
}

/// Whether an attribute of `item` may be a macro that rewrites it. `cfg(…)`
/// is inert; `cfg_attr(pred, a, b)` is judged by `a` and `b`, whatever the
/// predicate (this parser has a meta kind for each, without a path).
fn may_be_rewritten(item: &ast::Item) -> bool {
    item.attrs().any(|attr| {
        let Some(meta) = attr.meta() else {
            return true;
        };
        meta.skip_cfg_attrs().into_iter().any(|meta| {
            if matches!(meta, ast::Meta::CfgMeta(_)) {
                return false;
            }
            let first = meta
                .path()
                .and_then(|path| path.first_segment())
                .and_then(|segment| segment.name_ref());
            !first.is_some_and(|name| INERT_ATTRS.contains(&&*name.text()))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(before: &str, after: &str) -> bool {
        surface_of(before) == surface_of(after)
    }

    #[test]
    fn a_body_edit_keeps_the_surface_and_a_signature_edit_does_not() {
        let base = "/// doc\npub fn f(x: u8) -> Option<u8> {\n    Some(x)\n}\n";
        assert!(same(base, "/// other doc\npub fn f(x: u8) -> Option<u8> { let y = x; Some(y) }\n"));
        assert!(same(base, "pub fn f(x: u8) -> Option<u8> {\n    assert_eq!(x, 1);\n    None\n}\n"));
        assert!(!same(base, "pub fn f(x: u8) -> Result<u8, ()> {\n    Ok(x)\n}\n"));
        assert!(!same(base, "pub fn f(x: u16) -> Option<u8> {\n    None\n}\n"));
        // An item beside it, a field, a `mod` declaration are surface.
        assert!(!same(base, &format!("{base}mod more;\n")));
        assert!(!same("struct S { a: u8 }", "struct S { a: u16 }"));
        // Methods in an impl are stripped the same way.
        assert!(same("impl S { fn m(&self) { 1; } }", "impl S { fn m(&self) { 2; } }"));
    }

    #[test]
    fn an_impl_or_an_unknown_macro_in_a_body_is_surface() {
        let base = "fn f() {\n    let x = 1;\n}\n";
        assert!(!same(base, "fn f() {\n    let x = 1;\n    impl S { fn m() {} }\n}\n"));
        assert!(!same(base, "fn f() {\n    let x = 1;\n    make_impls!(S);\n}\n"));
        assert!(!same(base, "fn f() {\n    macro_rules! m { () => {} }\n}\n"));
        // The body of a fn in an impl in a body is private again.
        assert!(same(
            "fn f() { impl S { fn m() { 1; } } }",
            "fn f() { impl S { fn m() { 2; } } }"
        ));
        assert!(!same(
            "fn f() { impl S { fn m() { 1; } } }",
            "fn f() { impl S { fn m(&self) { 1; } } }"
        ));
        // Block-local items are not surface; what they hold may be.
        assert!(same(base, "fn f() {\n    use std::fmt;\n    struct L;\n}\n"));
        assert!(!same(base, "fn f() {\n    fn g() { impl S {} }\n}\n"));
        // A known expression macro, a nested std::-pathed one, a log line.
        assert!(same(base, "fn f() {\n    println!(\"{}\", 1);\n    std::assert!(true);\n    tracing::info!(\"x\");\n}\n"));
    }

    #[test]
    fn a_body_that_can_leak_is_surface() {
        for (before, after) in [
            ("fn f() -> impl Fn() { || {} }", "fn f() -> impl Fn() { move || {} }"),
            ("fn f() -> Box<impl Fn()> { 1 }", "fn f() -> Box<impl Fn()> { 2 }"),
            ("async fn f() { 1; }", "async fn f() { 2; }"),
            ("const fn f() -> usize { 1 }", "const fn f() -> usize { 2 }"),
            ("#[async_trait] impl T for S { fn f() { 1; } }", "#[async_trait] impl T for S { fn f() { 2; } }"),
            ("#[my::attr] fn f() { 1; }", "#[my::attr] fn f() { 2; }"),
            (
                "#[cfg_attr(test, my_attr)] fn f() { 1; }",
                "#[cfg_attr(test, my_attr)] fn f() { 2; }",
            ),
        ] {
            assert!(!same(before, after), "{before:?} → {after:?} is a surface change");
        }
        // `impl Trait` in an argument does not leak the body.
        assert!(same("fn f(g: impl Fn()) { 1; }", "fn f(g: impl Fn()) { 2; }"));
        // Inert attributes leave the item as written.
        assert!(same(
            "#[cfg(test)]\nmod tests {\n    #[test]\n    #[allow(unused)]\n    fn t() { 1; }\n}\n",
            "#[cfg(test)]\nmod tests {\n    #[test]\n    #[allow(unused)]\n    fn t() { 2; }\n}\n"
        ));
        assert!(same(
            "#[cfg_attr(test, allow(unused), inline)] fn f() { 1; }",
            "#[cfg_attr(test, allow(unused), inline)] fn f() { 2; }"
        ));
        assert!(same("//! inner doc\nfn f() {}", "//! other inner doc\nfn f() {}"));
    }
}
