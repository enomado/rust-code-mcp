//! A file's surface cut into **units** — an item, or one associated item of an
//! inherent `impl` — each saying which files a change of it can reach (plan
//! step `uses-narrow-widening`).
//!
//! # Why
//!
//! A surface change (`surface` module) used to make pending every file of the
//! changed crate and of all its reverse dependencies. Most surface changes are
//! one new or changed item, and the files whose resolution can follow it are
//! few: those that name it, and — for a few kinds of change — those that hold a
//! value of its type.
//!
//! # Two kinds of witness
//!
//! A file's uses resolve through the names it holds and through the types of
//! its expressions.
//!
//! - [`Witness::Name`]: a dependent names the item. A call names its function,
//!   a field access its field, a pattern its variant, a method call its method,
//!   a path its type. Changing a function's signature, a named field's type, a
//!   variant, a method reaches those files and no others: the new types flow
//!   into a body only through the expression that names the item.
//! - [`Witness::Holds`]: a dependent holds a value of the type, by whatever
//!   path. A trait `impl` changes what an operator, a `for`, a `?`, an autoderef
//!   or a blanket impl's method picks on any value of its self type; a tuple
//!   field is reached as `.0`; a type alias stands for its target wherever an
//!   item names it. The files holding a value of `T` are those naming `T` and
//!   those naming an item that hands out `T` values — a function or method
//!   whose signature names `T`, a field or variant of that type — and, through
//!   a tuple struct, an alias or a trait impl's associated type, the values of
//!   another type in turn. Each unit says what it hands out ([`Holder`]); the
//!   `widening` module closes over them.
//!
//! # Macros
//!
//! What a macro call or an attribute macro defines is not in the text. Such a
//! unit is [`Reach::Expand`]: the refresh asks rust-analyzer for that one
//! expansion and takes the units of what it produced ([`expansion_units`]) — an
//! expression macro (`t!`, `json!`) produces none, `#[tokio::main]` the `fn` as
//! written, `#[pymethods]` the impls it generates. Only the calls a change or
//! the closure reaches are expanded, never all of them.
//!
//! # What still widens to the crates
//!
//! [`Reach::Opaque`] and a changed [`Reach::MacroDef`]: a changed macro
//! definition (every call site's expansion follows it), a public glob import or
//! `use … as _` (the names they bring are not written), a non-inert crate
//! attribute, text that does not parse.
//!
//! # What this does not see
//!
//! Items a derive generates under new names (`#[derive(EnumIter)]` makes
//! `FooIter`): a file naming only that is not reached by a change of `Foo`'s
//! variants. A local type named like a built-in one (`struct u8`) in a trait
//! impl's header.

use std::collections::HashMap;

use ra_ap_syntax::ast::{self, HasAttrs, HasGenericParams, HasModuleItem, HasName, HasVisibility};
use ra_ap_syntax::{AstNode, Edition, NodeOrToken, SourceFile, SyntaxKind, SyntaxNode, TextRange};

use super::surface::{INERT_ATTRS, Token, emit, emit_without_bodies, is_expression_macro};

/// Built-in types: never a type the project declares, so never a witness of a
/// trait impl's header.
const PRIMITIVES: &[&str] = &[
    "bool", "char", "str", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
    "i128", "isize", "f16", "f32", "f64", "f128",
];

/// Whether `name` is a built-in type ([`PRIMITIVES`]).
pub(crate) fn is_primitive(name: &str) -> bool {
    PRIMITIVES.contains(&name)
}

/// Which files a change reaches — see the module header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Witness {
    /// The files that name it.
    Name(String),
    /// The files that name either: an associated item without `self` of an
    /// inherent `impl` is reached only as `Type::item`, or `Self::item` inside
    /// an `impl Type` — a dependent names both, so either is sound, and the
    /// refresh takes the one fewer files hold.
    Either(String, String),
    /// The files that may hold a value of one of these types.
    Holds(Vec<String>),
    /// The files that name `.0` and may hold a value of the type `.1`: a
    /// field (`x.field`, or a pattern or literal naming the type), a variant
    /// (a pattern on a value, or a path naming the enum), a method (`x.m()`).
    Both(String, String),
    /// The files of the module of the inline path `.1` in the declaring file
    /// and of the modules below it that name `.0`: a private `use`.
    Scoped(String, Vec<String>),
    /// A trait `impl`: the files holding a value of its self type — of the
    /// header's types the project declares (`impl From<Foo> for String` is
    /// `Foo`). When the self type holds no such name (`impl Ext for Vec<u8>`,
    /// a blanket `impl<T> Ext for T`) the trait is the project's (orphan
    /// rule), and a dependent names it or the items it calls.
    TraitImpl {
        /// The types the self type names, the impl's own parameters left out.
        self_type: Vec<String>,
        /// The types the trait's path names, its generic arguments included.
        trait_ref: Vec<String>,
        /// The names of the impl's associated items, then the trait's name.
        own: Vec<String>,
    },
}

/// What a unit hands out to a file holding a value of a type it names: when
/// `mentions` names a type whose values matter, so does `witness`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Holder {
    pub(crate) witness: Witness,
    /// Sorted, deduplicated.
    pub(crate) mentions: Vec<String>,
}

/// Files a unit reaches that need not name it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModuleFiles {
    /// An out-of-line `mod name;`: the module's own file and those under its
    /// directory (their place in the crate follows the declaration). The path
    /// of names from the declaring file's module, inline modules first.
    Declared(Vec<String>),
    /// `#[path = "…"] mod name;`: the file the attribute names, relative to
    /// the declaring file's directory (past the inline modules, `.0`), and
    /// those under its directory.
    AtPath(Vec<String>, String),
    /// A private glob import or `use … as _`: what it brings in is visible in
    /// its module (the inline path `.0` in the declaring file) and the
    /// modules under it; a `use super::*` there sees it too.
    Below(Vec<String>),
}

/// What a change of a unit reaches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Reach {
    Named {
        witnesses: Vec<Witness>,
        module: Option<ModuleFiles>,
    },
    /// What the macro at `at` (in the tree the unit was cut from) expands to:
    /// a macro call, or an item with an attribute that may be a macro
    /// (`attr`); `module`, the inline modules around it.
    Expand {
        at: TextRange,
        attr: bool,
        module: Vec<String>,
    },
    /// A macro definition named so: a changed one reaches what every call
    /// site's expansion defines — the crates.
    MacroDef(String),
    /// No telling who depends on it; why, for the log.
    Opaque(String),
}

/// See the module header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unit {
    /// Its surface tokens, prefixed by the headers of the inline modules (and
    /// of the inherent `impl`) it is in: two texts' units are matched by these.
    tokens: Vec<Token>,
    /// The names it refers to (not those it declares), sorted, deduplicated;
    /// an inherent associated item's `Self` counts as the self type's name.
    pub(crate) mentions: Vec<String>,
    pub(crate) reach: Reach,
    pub(crate) holders: Vec<Holder>,
}

impl Unit {
    /// A unit that stands for a failed expansion.
    pub(crate) fn opaque(why: String) -> Unit {
        Unit {
            tokens: Vec::new(),
            mentions: Vec::new(),
            reach: Reach::Opaque(why),
            holders: Vec::new(),
        }
    }
}

/// Whether a changed unit is of the text before the change or after it — which
/// revision of the database expands it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Side {
    Old,
    New,
}

/// The surface of `text` cut into [`Unit`]s, in text order.
///
/// Cost: one parse and one walk of the file — on a ~8000-file workspace all
/// files take ≈ 7.7 s on one core (213 k units, 824 k mentions).
pub(crate) fn units_of(text: &str) -> Vec<Unit> {
    let parse = SourceFile::parse(text, Edition::CURRENT);
    let file = parse.tree();
    let mut units = Vec::new();
    if !parse.errors().is_empty() {
        let mut tokens = Vec::new();
        emit(file.syntax(), true, &mut tokens);
        units.push(opaque(tokens, "text that does not parse".to_owned()));
        return units;
    }
    let crate_attrs: Vec<String> = file
        .attrs()
        .filter_map(|attr| rewriting_attr(&attr, false))
        .collect();
    if !crate_attrs.is_empty() {
        let mut tokens = Vec::new();
        for attr in file.attrs() {
            emit(attr.syntax(), true, &mut tokens);
        }
        let why = format!("a crate attribute #![{}]", crate_attrs.join(", "));
        units.push(opaque(tokens, why));
    }
    let context = Context::default();
    for item in file.items() {
        item_units(&item, &context, false, &mut units);
    }
    units
}

/// The units of `old` not in `new` and those of `new` not in `old`, matched as
/// multisets by their tokens (a moved item is no change), with their side.
/// `None`: no file — an added file is all new, a removed one all old.
pub(crate) fn changed_units(old: Option<&str>, new: Option<&str>) -> Vec<(Side, Unit)> {
    let old = old.map(units_of).unwrap_or_default();
    let new = new.map(units_of).unwrap_or_default();
    let mut count: HashMap<&[Token], isize> = HashMap::new();
    for unit in &old {
        *count.entry(&unit.tokens).or_default() += 1;
    }
    for unit in &new {
        *count.entry(&unit.tokens).or_default() -= 1;
    }
    let mut changed = Vec::new();
    // Each surplus copy of a token sequence, on the side that has it.
    for (units, side, sign) in [(&old, Side::Old, 1), (&new, Side::New, -1)] {
        for unit in units {
            let left = count.get_mut(&*unit.tokens).expect("counted above");
            if *left * sign > 0 {
                *left -= sign;
                changed.push((side, unit.clone()));
            }
        }
    }
    changed
}

/// The units of a macro's expansion `root` (a node of the expansion's own
/// tree): items for an expansion in item position (`MACRO_ITEMS`), and of any
/// other only what escapes a body — `impl`s, macro definitions, calls of
/// macros that may expand to them.
pub(crate) fn expansion_units(root: &SyntaxNode, module: &[String]) -> Vec<Unit> {
    let context = Context {
        prefix: Vec::new(),
        module: module.to_vec(),
    };
    let mut units = Vec::new();
    if root.kind() == SyntaxKind::MACRO_ITEMS {
        for child in root.children() {
            match ast::Item::cast(child.clone()) {
                Some(item) => item_units(&item, &context, false, &mut units),
                None => nested_units(&child, &context, &mut units),
            }
        }
    } else {
        nested_units(root, &context, &mut units);
    }
    units
}

/// The units of the item at `at` in `root` as written, its attributes taken
/// as inert — what an attribute that is no macro leaves (`expand_attr_macro`
/// finds none). `None` when no item is there.
pub(crate) fn as_written(root: &SyntaxNode, at: TextRange, module: &[String]) -> Option<Vec<Unit>> {
    let item = item_at(root, at)?;
    let context = Context {
        prefix: Vec::new(),
        module: module.to_vec(),
    };
    let mut units = Vec::new();
    // An associated item of an inherent impl is reached by its own name either
    // way; as a free item it says so too.
    item_units(&item, &context, true, &mut units);
    Some(units)
}

/// The item spanning exactly `at` in `root` — an associated item too.
pub(crate) fn item_at(root: &SyntaxNode, at: TextRange) -> Option<ast::Item> {
    let node = match root.covering_element(at) {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(token) => token.parent()?,
    };
    node.ancestors()
        .take_while(|node| node.text_range().contains_range(at))
        .filter(|node| node.text_range() == at)
        .find_map(ast::Item::cast)
}

/// The macro call spanning exactly `at` in `root`.
pub(crate) fn macro_call_at(root: &SyntaxNode, at: TextRange) -> Option<ast::MacroCall> {
    let node = match root.covering_element(at) {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(token) => token.parent()?,
    };
    node.ancestors()
        .take_while(|node| node.text_range().contains_range(at))
        .filter(|node| node.text_range() == at)
        .find_map(ast::MacroCall::cast)
}

/// Where an item sits: the tokens of the headers around it, and the names of
/// the inline modules it is in.
#[derive(Default)]
struct Context {
    prefix: Vec<Token>,
    module: Vec<String>,
}

fn opaque(tokens: Vec<Token>, why: String) -> Unit {
    unit(tokens, Reach::Opaque(why), Vec::new())
}

fn unit(tokens: Vec<Token>, reach: Reach, holders: Vec<Holder>) -> Unit {
    let mentions = refs(&tokens);
    Unit {
        tokens,
        mentions,
        reach,
        holders,
    }
}

fn named(tokens: Vec<Token>, witnesses: Vec<Witness>, holders: Vec<Holder>) -> Unit {
    unit(
        tokens,
        Reach::Named {
            witnesses,
            module: None,
        },
        holders,
    )
}

/// A unit asking for the expansion of the macro call or attribute macro on `node`.
fn expand(node: &SyntaxNode, context: &Context, attr: bool) -> Unit {
    let mut tokens = context.prefix.clone();
    emit(node, true, &mut tokens);
    unit(
        tokens,
        Reach::Expand {
            at: node.text_range(),
            attr,
            module: context.module.clone(),
        },
        Vec::new(),
    )
}

/// The names `tokens` refer to (identifiers that do not declare), sorted,
/// deduplicated, raw `r#x` as `x`.
fn refs(tokens: &[Token]) -> Vec<String> {
    let mut names: Vec<String> = tokens
        .iter()
        .filter(|(kind, _)| *kind == SyntaxKind::IDENT)
        .map(|(_, text)| plain(text))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// The names the surface of `node` refers to.
fn refs_of(node: &SyntaxNode) -> Vec<String> {
    let mut tokens = Vec::new();
    emit_without_bodies(node, &mut tokens);
    refs(&tokens)
}

fn holder(witness: Witness, mentions: Vec<String>) -> Holder {
    Holder { witness, mentions }
}

fn plain(name: &str) -> String {
    name.strip_prefix("r#").unwrap_or(name).to_owned()
}

fn name_of(node: &impl HasName) -> Option<String> {
    node.name().map(|name| plain(&name.text()))
}

/// The surface tokens of `node`, after `prefix`.
fn tokens_of(node: &SyntaxNode, prefix: &[Token]) -> Vec<Token> {
    let mut tokens = prefix.to_vec();
    emit_without_bodies(node, &mut tokens);
    tokens
}

/// The surface tokens of `node` without its child of kind `skip` (a module's
/// or an impl's list of items), after `prefix`.
fn header_tokens(node: &SyntaxNode, skip: SyntaxKind, prefix: &[Token]) -> Vec<Token> {
    let mut tokens = prefix.to_vec();
    for element in node.children_with_tokens() {
        match element {
            NodeOrToken::Token(token) => {
                let silent = token.kind().is_trivia()
                    || matches!(
                        token.kind(),
                        SyntaxKind::INNER_DOC_COMMENT | SyntaxKind::OUTER_DOC_COMMENT
                    );
                if !silent {
                    tokens.push((token.kind(), token.text().to_owned()));
                }
            }
            NodeOrToken::Node(child)
                if child.kind() == skip || child.kind() == SyntaxKind::DOC_COMMENT => {}
            NodeOrToken::Node(child) => emit_without_bodies(&child, &mut tokens),
        }
    }
    tokens
}

/// The attribute's path if it may be a macro rewriting its item, `None` for
/// an inert one: the `INERT_ATTRS`, `cfg`, `derive` (it adds impls of the
/// item's own type), and on an ADT with a derive (`adt_with_derive`) a
/// one-segment attribute with arguments — a derive helper (`#[serde(…)]`,
/// `#[ts(export)]`). Whether it really is a macro rust-analyzer tells, when
/// asked to expand it.
fn rewriting_attr(attr: &ast::Attr, adt_with_derive: bool) -> Option<String> {
    let Some(meta) = attr.meta() else {
        return Some("?".to_owned());
    };
    let rewriting: Vec<String> = meta
        .skip_cfg_attrs()
        .into_iter()
        .filter(|meta| !matches!(meta, ast::Meta::CfgMeta(_)))
        .filter_map(|meta| {
            let path = meta.path()?;
            let first = path
                .first_segment()
                .and_then(|segment| segment.name_ref())
                .map(|name| name.text().to_string());
            if first
                .as_deref()
                .is_some_and(|name| INERT_ATTRS.contains(&name) || name == "derive")
            {
                return None;
            }
            let helper = adt_with_derive
                && path.qualifier().is_none()
                && matches!(
                    meta,
                    ast::Meta::TokenTreeMeta(_) | ast::Meta::KeyValueMeta(_)
                );
            (!helper).then(|| path.syntax().text().to_string())
        })
        .collect();
    (!rewriting.is_empty()).then(|| rewriting.join(", "))
}

/// Whether an outer attribute of `item` may be a macro rewriting it.
fn has_rewriting_attr(item: &impl HasAttrs) -> bool {
    let adt_with_derive = matches!(
        item.syntax().kind(),
        SyntaxKind::STRUCT | SyntaxKind::ENUM | SyntaxKind::UNION
    ) && item.attrs().any(|attr| {
        attr.meta().is_some_and(|meta| {
            meta.skip_cfg_attrs().into_iter().any(|meta| {
                meta.path()
                    .and_then(|path| path.first_segment())
                    .and_then(|segment| segment.name_ref())
                    .is_some_and(|name| name.text() == "derive")
            })
        })
    });
    item.attrs()
        .filter(|attr| attr.excl_token().is_none())
        .any(|attr| rewriting_attr(&attr, adt_with_derive).is_some())
}

/// Whether the use or glob is private to its module (`pub(self)` too).
fn is_private(vis: Option<ast::Visibility>) -> bool {
    vis.is_none_or(|vis| vis.syntax().text() == "pub(self)")
}

/// A struct's or union's reach and holders: its name and each named field's
/// (a field is reached as `x.field` and in patterns and literals); a tuple
/// field is reached as `x.0` on any value of the type.
fn fields_reach(
    name: Option<String>,
    fields: Option<ast::FieldList>,
) -> (Vec<Witness>, Vec<Holder>) {
    let mut reach: Vec<Witness> = name.clone().into_iter().map(Witness::Name).collect();
    let mut holders = Vec::new();
    match fields {
        Some(ast::FieldList::RecordFieldList(list)) => {
            for field in list.fields() {
                if let Some(field_name) = name_of(&field) {
                    reach.push(match &name {
                        Some(owner) => Witness::Both(field_name.clone(), owner.clone()),
                        None => Witness::Name(field_name.clone()),
                    });
                    holders.push(holder(Witness::Name(field_name), refs_of(field.syntax())));
                }
            }
        }
        Some(ast::FieldList::TupleFieldList(list)) => {
            let held: Vec<String> = name.into_iter().collect();
            reach.push(Witness::Holds(held.clone()));
            for field in list.fields() {
                holders.push(holder(
                    Witness::Holds(held.clone()),
                    refs_of(field.syntax()),
                ));
            }
        }
        None => {}
    }
    (reach, holders)
}

/// The units of one item at `context`, nested ones included; `as_written`:
/// its attributes taken as inert.
fn item_units(item: &ast::Item, context: &Context, as_written: bool, out: &mut Vec<Unit>) {
    if !as_written && has_rewriting_attr(item) {
        return out.push(expand(item.syntax(), context, true));
    }
    let tokens = || tokens_of(item.syntax(), &context.prefix);
    let mentions = || refs_of(item.syntax());
    // What a plain named item reaches and hands out: its name.
    let by_name = |name: Option<String>| -> (Vec<Witness>, Vec<Holder>) {
        let reach: Vec<Witness> = name.into_iter().map(Witness::Name).collect();
        let holders = reach
            .iter()
            .map(|witness| holder(witness.clone(), mentions()))
            .collect();
        (reach, holders)
    };
    let (reach, holders) = match item {
        ast::Item::Impl(imp) => return impl_units(imp, context, out),
        ast::Item::Module(module) => return module_units(module, context, out),
        ast::Item::MacroCall(_) => return out.push(expand(item.syntax(), context, false)),
        ast::Item::MacroRules(rules) => return out.push(macro_def(tokens(), name_of(rules))),
        ast::Item::MacroDef(def) => return out.push(macro_def(tokens(), name_of(def))),
        ast::Item::AsmExpr(_) => return out.push(opaque(tokens(), "global_asm!".to_owned())),
        ast::Item::Use(import) => match use_names(import) {
            Ok(names) => {
                let mut reach = Vec::new();
                let mut holders = Vec::new();
                let private = is_private(import.visibility());
                for (name, renamed) in names {
                    // An alias stands for its target where it is named.
                    if renamed {
                        holders.push(holder(Witness::Holds(vec![name.clone()]), mentions()));
                    }
                    // A private import is seen in its module and below only.
                    reach.push(if private {
                        Witness::Scoped(name, context.module.clone())
                    } else {
                        Witness::Name(name)
                    });
                }
                (reach, holders)
            }
            // Visible in its module and below only: those files.
            Err(_) if is_private(import.visibility()) => {
                return out.push(unit(
                    tokens(),
                    Reach::Named {
                        witnesses: Vec::new(),
                        module: Some(ModuleFiles::Below(context.module.clone())),
                    },
                    Vec::new(),
                ));
            }
            Err(why) => return out.push(opaque(tokens(), format!("a public {why}"))),
        },
        ast::Item::ExternBlock(block) => {
            let mut reach = Vec::new();
            let mut holders = Vec::new();
            for extern_item in block
                .extern_item_list()
                .into_iter()
                .flat_map(|list| list.extern_items())
            {
                let name = match &extern_item {
                    ast::ExternItem::Fn(f) => name_of(f),
                    ast::ExternItem::Static(s) => name_of(s),
                    ast::ExternItem::TypeAlias(t) => name_of(t),
                    ast::ExternItem::MacroCall(call) => {
                        out.push(expand(call.syntax(), context, false));
                        None
                    }
                };
                if let Some(name) = name {
                    reach.push(Witness::Name(name.clone()));
                    holders.push(holder(Witness::Name(name), refs_of(extern_item.syntax())));
                }
            }
            (reach, holders)
        }
        ast::Item::ExternCrate(krate) => {
            let alias = krate
                .rename()
                .and_then(|rename| rename.name())
                .map(|name| plain(&name.text()));
            by_name(alias.or_else(|| krate.name_ref().map(|name| plain(&name.text()))))
        }
        ast::Item::Trait(tr) => {
            let name = name_of(tr);
            let mut reach: Vec<Witness> = name.clone().into_iter().map(Witness::Name).collect();
            // A supertrait or bound naming a type: a value of the trait's
            // objects derefs or iterates to it.
            let header = header_tokens(tr.syntax(), SyntaxKind::ASSOC_ITEM_LIST, &[]);
            let mut holders = vec![holder(
                Witness::Holds(name.into_iter().collect()),
                refs(&header),
            )];
            for assoc in tr
                .assoc_item_list()
                .into_iter()
                .flat_map(|list| list.assoc_items())
            {
                match &assoc {
                    ast::AssocItem::MacroCall(call) => {
                        out.push(expand(call.syntax(), context, false))
                    }
                    _ => {
                        if let Some(assoc_name) = assoc_name(&assoc) {
                            reach.push(Witness::Name(assoc_name.clone()));
                            holders
                                .push(holder(Witness::Name(assoc_name), refs_of(assoc.syntax())));
                        }
                    }
                }
            }
            (reach, holders)
        }
        ast::Item::Fn(f) => by_name(name_of(f)),
        ast::Item::Struct(s) => fields_reach(name_of(s), s.field_list()),
        ast::Item::Union(u) => fields_reach(
            name_of(u),
            u.record_field_list().map(ast::FieldList::RecordFieldList),
        ),
        ast::Item::Enum(e) => {
            let enum_name = name_of(e);
            let mut reach: Vec<Witness> =
                enum_name.clone().into_iter().map(Witness::Name).collect();
            let mut holders = Vec::new();
            for variant in e
                .variant_list()
                .into_iter()
                .flat_map(|list| list.variants())
            {
                if let Some(variant_name) = name_of(&variant) {
                    reach.push(match &enum_name {
                        Some(owner) => Witness::Both(variant_name.clone(), owner.clone()),
                        None => Witness::Name(variant_name.clone()),
                    });
                    holders.push(holder(
                        Witness::Name(variant_name),
                        refs_of(variant.syntax()),
                    ));
                }
            }
            (reach, holders)
        }
        ast::Item::TypeAlias(t) => {
            // Where an item names the alias, its target is meant.
            let name = name_of(t);
            let held: Vec<String> = name.clone().into_iter().collect();
            let mut reach: Vec<Witness> = name.into_iter().map(Witness::Name).collect();
            reach.push(Witness::Holds(held.clone()));
            (reach, vec![holder(Witness::Holds(held), mentions())])
        }
        ast::Item::Const(c) => by_name(name_of(c)),
        ast::Item::Static(s) => by_name(name_of(s)),
    };
    // A private item is seen in its module and below only.
    let reach = if is_private(visibility_of(item.syntax())) {
        reach
            .into_iter()
            .map(|witness| scope_name(witness, context))
            .collect()
    } else {
        reach
    };
    out.push(named(tokens(), reach, holders));
    nested_units(item.syntax(), context, out);
}

/// The visibility written on an item, if any.
fn visibility_of(node: &SyntaxNode) -> Option<ast::Visibility> {
    node.children().find_map(ast::Visibility::cast)
}

/// A name witness of a private item, scoped to its module and below; other
/// witnesses as they are (a field or a variant is reached through values).
fn scope_name(witness: Witness, context: &Context) -> Witness {
    match witness {
        Witness::Name(name) => Witness::Scoped(name, context.module.clone()),
        other => other,
    }
}

/// A macro definition's unit: its call sites name it, and when it names a type
/// whose values matter, what they expand to may hand those out.
fn macro_def(tokens: Vec<Token>, name: Option<String>) -> Unit {
    let name = name.unwrap_or_default();
    let mentions = refs(&tokens);
    let holders = vec![holder(Witness::Holds(vec![name.clone()]), mentions)];
    unit(tokens, Reach::MacroDef(name), holders)
}

/// What an associated item is named, if it is named.
fn assoc_name(assoc: &ast::AssocItem) -> Option<String> {
    match assoc {
        ast::AssocItem::Fn(f) => name_of(f),
        ast::AssocItem::Const(c) => name_of(c),
        ast::AssocItem::TypeAlias(t) => name_of(t),
        ast::AssocItem::MacroCall(_) => None,
    }
}

/// The names a `use` brings into scope, each with whether it is an alias
/// (`as x`), or what keeps them unwritten: a glob import, an import `as _` (a
/// trait's methods, by names it does not show).
fn use_names(import: &ast::Use) -> Result<Vec<(String, bool)>, String> {
    fn walk(
        tree: &ast::UseTree,
        last: Option<String>,
        names: &mut Vec<(String, bool)>,
    ) -> Result<(), String> {
        if tree.star_token().is_some() {
            return Err("glob import".to_owned());
        }
        // The last segment of this tree's own path, `self` as its parent's.
        let segment = tree
            .path()
            .and_then(|path| path.segment())
            .map(|segment| match segment.name_ref() {
                Some(name) if name.self_token().is_none() => plain(&name.text()),
                _ => last.clone().unwrap_or_default(),
            })
            .or(last);
        if let Some(list) = tree.use_tree_list() {
            for inner in list.use_trees() {
                walk(&inner, segment.clone(), names)?;
            }
            return Ok(());
        }
        match tree.rename() {
            Some(rename) if rename.underscore_token().is_some() => Err("import as _".to_owned()),
            Some(rename) => {
                names.extend(rename.name().map(|name| (plain(&name.text()), true)));
                Ok(())
            }
            None => {
                names.extend(
                    segment
                        .filter(|name| !name.is_empty())
                        .map(|name| (name, false)),
                );
                Ok(())
            }
        }
    }
    let mut names = Vec::new();
    if let Some(tree) = import.use_tree() {
        walk(&tree, None, &mut names)?;
    }
    Ok(names)
}

/// The string of a `#[path = "…"]` attribute on `module`, if it has one.
fn path_attr(module: &ast::Module) -> Option<String> {
    module.attrs().find_map(|attr| {
        let ast::Meta::KeyValueMeta(meta) = attr.meta()? else {
            return None;
        };
        let path = meta.path()?;
        if path.syntax().text() != "path" {
            return None;
        }
        let value = meta.expr()?.syntax().text().to_string();
        Some(value.trim_matches('"').to_owned())
    })
}

/// An inline module: a unit of its header, its items with the header as their
/// prefix. An out-of-line `mod name;`: one unit that also reaches the
/// module's files.
fn module_units(module: &ast::Module, context: &Context, out: &mut Vec<Unit>) {
    let name = name_of(module).unwrap_or_default();
    let private = is_private(module.visibility());
    let mut path = context.module.clone();
    path.push(name.clone());
    let Some(list) = module.item_list() else {
        let files = match path_attr(module) {
            Some(at) => ModuleFiles::AtPath(context.module.clone(), at),
            None => ModuleFiles::Declared(path),
        };
        let witness = Witness::Name(name);
        let witness = if private {
            scope_name(witness, context)
        } else {
            witness
        };
        return out.push(unit(
            tokens_of(module.syntax(), &context.prefix),
            Reach::Named {
                witnesses: vec![witness],
                module: Some(files),
            },
            Vec::new(),
        ));
    };
    let mut prefix = header_tokens(module.syntax(), SyntaxKind::ITEM_LIST, &context.prefix);
    let witness = Witness::Name(name);
    let witness = if private {
        scope_name(witness, context)
    } else {
        witness
    };
    out.push(named(prefix.clone(), vec![witness], Vec::new()));
    prefix.push((SyntaxKind::L_CURLY, "{".to_owned()));
    let inner = Context {
        prefix,
        module: path,
    };
    for item in list.items() {
        item_units(&item, &inner, false, out);
    }
}

/// The names of an impl's type and const parameters.
fn generic_names(list: Option<ast::GenericParamList>) -> Vec<String> {
    let Some(list) = list else {
        return Vec::new();
    };
    list.generic_params()
        .filter_map(|param| match param {
            ast::GenericParam::TypeParam(t) => name_of(&t),
            ast::GenericParam::ConstParam(c) => name_of(&c),
            ast::GenericParam::LifetimeParam(_) => None,
        })
        .collect()
}

/// A trait impl: one unit. An inherent impl: a unit per associated item, its
/// header the prefix (an empty one reaches nothing). A macro call among the
/// associated items: a unit of its expansion.
fn impl_units(imp: &ast::Impl, context: &Context, out: &mut Vec<Unit>) {
    let generics = generic_names(imp.generic_param_list());
    // The types a header names: the last segment of every path that is not
    // the qualifier of another (`fmt::Display<Foo>` is `Display` and `Foo`),
    // the impl's own parameters and the std roots left out.
    let types_of = |node: Option<&SyntaxNode>| -> Vec<String> {
        let mut names: Vec<String> = node
            .into_iter()
            .flat_map(|node| node.descendants())
            .filter_map(ast::Path::cast)
            .filter(|path| {
                path.syntax()
                    .parent()
                    .is_none_or(|parent| parent.kind() != SyntaxKind::PATH)
            })
            .filter_map(|path| path.segment()?.name_ref())
            .map(|name| plain(&name.text()))
            .filter(|name| {
                !generics.contains(name) && !matches!(name.as_str(), "std" | "core" | "alloc")
            })
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    };
    let items = || {
        imp.assoc_item_list()
            .into_iter()
            .flat_map(|list| list.assoc_items())
    };
    if let Some(trait_ref) = imp.trait_() {
        let mut own: Vec<String> = items().filter_map(|assoc| assoc_name(&assoc)).collect();
        let trait_path = match &trait_ref {
            ast::Type::PathType(path) => path.path(),
            _ => None,
        };
        own.extend(
            trait_path
                .and_then(|path| path.segment())
                .and_then(|segment| segment.name_ref())
                .map(|name| plain(&name.text())),
        );
        let witness = Witness::TraitImpl {
            self_type: types_of(imp.self_ty().as_ref().map(|ty| ty.syntax())),
            trait_ref: types_of(Some(trait_ref.syntax())),
            own,
        };
        // A method hands out what its signature names to its callers; an
        // associated type or constant, to every value of the self type
        // (`for`, operators, `?`, autoderef).
        let mut holders = Vec::new();
        for assoc in items() {
            match &assoc {
                ast::AssocItem::MacroCall(call) => out.push(expand(call.syntax(), context, false)),
                ast::AssocItem::Fn(f) => {
                    if let Some(name) = name_of(f) {
                        holders.push(holder(Witness::Name(name), refs_of(f.syntax())));
                    }
                }
                _ => holders.push(holder(witness.clone(), refs_of(assoc.syntax()))),
            }
        }
        out.push(named(
            tokens_of(imp.syntax(), &context.prefix),
            vec![witness],
            holders,
        ));
        nested_units(imp.syntax(), context, out);
        return;
    }
    // Inherent: the type's own name, for `Type::item` and `Self`.
    let head = imp.self_ty().and_then(|ty| match ty {
        ast::Type::PathType(path) => path
            .path()
            .and_then(|path| path.segment())
            .and_then(|segment| segment.name_ref())
            .map(|name| plain(&name.text())),
        _ => None,
    });
    let mut prefix = header_tokens(imp.syntax(), SyntaxKind::ASSOC_ITEM_LIST, &context.prefix);
    prefix.push((SyntaxKind::L_CURLY, "{".to_owned()));
    let header_mentions: Vec<String> = refs(&prefix)
        .into_iter()
        .filter(|name| Some(name) != head.as_ref())
        .collect();
    for assoc in items() {
        let inner = Context {
            prefix: prefix.clone(),
            module: context.module.clone(),
        };
        if matches!(assoc, ast::AssocItem::MacroCall(_)) {
            out.push(expand(assoc.syntax(), &inner, false));
            continue;
        }
        if has_rewriting_attr(&assoc) {
            out.push(expand(assoc.syntax(), &inner, true));
            continue;
        }
        let mut tokens = prefix.clone();
        emit_without_bodies(assoc.syntax(), &mut tokens);
        let is_method = matches!(&assoc, ast::AssocItem::Fn(f)
            if f.param_list().and_then(|params| params.self_param()).is_some());
        // A method is called on a value of the type; another item is named
        // through the type or `Self`.
        let witness = assoc_name(&assoc).map(|name| match &head {
            Some(head) if is_method => Witness::Both(name, head.clone()),
            Some(head) => Witness::Either(head.clone(), name),
            None => Witness::Name(name),
        });
        // What it hands out goes to whoever calls it by name.
        let handed = assoc_name(&assoc).map(Witness::Name);
        let own = &tokens[prefix.len()..];
        let mut mentions = refs(own);
        mentions.extend(header_mentions.iter().cloned());
        if own
            .iter()
            .any(|(kind, _)| *kind == SyntaxKind::SELF_TYPE_KW)
        {
            mentions.extend(head.clone());
        }
        mentions.sort_unstable();
        mentions.dedup();
        let holders = handed
            .into_iter()
            .map(|witness| holder(witness, mentions.clone()))
            .collect();
        out.push(Unit {
            tokens,
            mentions,
            reach: Reach::Named {
                witnesses: witness.into_iter().collect(),
                module: None,
            },
            holders,
        });
        nested_units(assoc.syntax(), &inner, out);
    }
}

/// Units of what inside `node` is visible beyond it: `impl`s at any depth,
/// macro definitions, macro calls that may expand to items (not the
/// expression macros of std and the logging crates, not one in a type or a
/// pattern). An `impl` stops the walk: its own items are its units' business.
fn nested_units(node: &SyntaxNode, context: &Context, out: &mut Vec<Unit>) {
    for child in node.children() {
        match child.kind() {
            SyntaxKind::IMPL => {
                let imp = ast::Impl::cast(child).expect("an IMPL node is an impl");
                if has_rewriting_attr(&imp) {
                    out.push(expand(imp.syntax(), context, true));
                } else {
                    impl_units(&imp, context, out);
                }
            }
            SyntaxKind::MACRO_RULES | SyntaxKind::MACRO_DEF => {
                let name = ast::Macro::cast(child.clone())
                    .and_then(|def| def.name())
                    .map(|name| plain(&name.text()));
                out.push(macro_def(tokens_of(&child, &context.prefix), name));
            }
            SyntaxKind::MACRO_CALL => {
                let in_type_or_pattern = child.parent().is_some_and(|parent| {
                    matches!(
                        parent.kind(),
                        SyntaxKind::MACRO_TYPE | SyntaxKind::MACRO_PAT
                    )
                });
                if !in_type_or_pattern && !is_expression_macro(&child) {
                    out.push(expand(&child, context, false));
                }
            }
            _ => nested_units(&child, context, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness_text(witness: &Witness) -> String {
        match witness {
            Witness::Name(name) => name.clone(),
            Witness::Either(a, b) => format!("{a}|{b}"),
            Witness::Holds(types) => format!("holds({})", types.join(",")),
            Witness::Both(name, ty) => format!("{name}&{ty}"),
            Witness::Scoped(name, module) => format!("{name}@{}", module.join("::")),
            Witness::TraitImpl {
                self_type,
                trait_ref,
                own,
            } => format!(
                "impl[{}; {}; {}]",
                self_type.join(","),
                trait_ref.join(","),
                own.join(",")
            ),
        }
    }

    /// What the changed units reach, `Either` as `a|b`, sorted.
    fn reach(old: &str, new: &str) -> Vec<String> {
        let mut found: Vec<String> = changed_units(Some(old), Some(new))
            .into_iter()
            .flat_map(|(_, unit)| match unit.reach {
                Reach::Named { witnesses, module } => {
                    let mut names: Vec<String> = witnesses.iter().map(witness_text).collect();
                    names.extend(module.map(|module| format!("{module:?}")));
                    names
                }
                Reach::Expand { attr, .. } => vec![format!("expand(attr={attr})")],
                Reach::MacroDef(name) => vec![format!("macro {name}")],
                Reach::Opaque(why) => vec![format!("opaque: {why}")],
            })
            .collect();
        found.sort();
        found.dedup();
        found
    }

    /// What the units of `text` hand out to a holder of a `held` value.
    fn handed_out(text: &str, held: &str) -> Vec<String> {
        let mut found: Vec<String> = units_of(text)
            .iter()
            .flat_map(|unit| &unit.holders)
            .filter(|holder| holder.mentions.iter().any(|name| name == held))
            .map(|holder| witness_text(&holder.witness))
            .collect();
        found.sort();
        found
    }

    #[test]
    fn a_changed_item_is_reached_by_its_names() {
        assert_eq!(reach("fn a() {}", "fn a() {}\npub fn b() {}"), ["b"]);
        assert_eq!(reach("pub fn a(x: u8) {}", "pub fn a(x: u16) {}"), ["a"]);
        // A private item: its module and those below.
        assert_eq!(
            reach("mod m { fn a(x: u8) {} }", "mod m { fn a(x: u16) {} }"),
            ["a@m"]
        );
        // A struct by its name and fields; a tuple struct by its values too.
        assert_eq!(
            reach("pub struct S { a: u8 }", "pub struct S { a: u16 }"),
            ["S", "a&S"]
        );
        assert_eq!(
            reach("pub struct S(u8);", "pub struct S(u16);"),
            ["S", "holds(S)"]
        );
        assert_eq!(
            reach("pub enum E { A }", "pub enum E { A, B(u8) }"),
            ["A&E", "B&E", "E"]
        );
        assert_eq!(
            reach("pub type A = u8;", "pub type A = u16;"),
            ["A", "holds(A)"]
        );
        // A body edit changes no unit, an `async fn`'s neither; moving an
        // item is no change.
        assert!(reach("fn a() { 1; }\nfn b() {}", "fn b() {}\nfn a() { 2; }").is_empty());
        assert!(reach("async fn a() -> u8 { 1 }", "async fn a() -> u8 { 2 }").is_empty());
        assert_eq!(
            reach(
                "pub const fn a() -> u8 { 1 }",
                "pub const fn a() -> u8 { 2 }"
            ),
            ["a"]
        );
        // A trait: its name and items (a glob-imported trait's methods are
        // called by name only).
        assert_eq!(
            reach("pub trait T {}", "pub trait T { fn m(&self); }"),
            ["T", "m"]
        );
        assert_eq!(
            reach("", "pub use a::{b::{self, C}, D as E};"),
            ["C", "E", "b"]
        );
        assert_eq!(reach("", "mod m { use a::C; }"), ["C@m", "m@"]);
    }

    #[test]
    fn an_inherent_item_is_reached_by_its_own_name_or_its_type() {
        let base = "impl S { fn m(&self) {} }";
        // A method: its name (`x.m()`); an associated fn: the type or the name.
        assert_eq!(
            reach(base, "impl S { fn m(&self) {} fn k(&self) {} }"),
            ["k&S"]
        );
        assert_eq!(
            reach(base, "impl S { fn m(&self) {} fn new() -> Self { S } }"),
            ["S|new"]
        );
        // The header changed: every item of it.
        assert_eq!(reach(base, "impl<T> S<T> { fn m(&self) {} }"), ["m&S"]);
        let units = units_of("impl S { fn a(&self) -> Self { S } fn b(&self) {} }");
        assert_eq!(units.len(), 2);
        assert!(
            units[0].mentions.contains(&"S".to_owned()),
            "`Self` is the type"
        );
        assert!(
            !units[1].mentions.contains(&"S".to_owned()),
            "the header's type is not"
        );
    }

    #[test]
    fn a_trait_impl_is_reached_through_its_header() {
        assert_eq!(
            reach(
                "",
                "impl<T: X> fmt::From<Foo<T>> for Bar { fn from(_: Foo<T>) -> Self { todo!() } }"
            ),
            ["impl[Bar; Foo,From; from,From]"]
        );
    }

    /// Who hands out a value of `T`: what names it in a signature, a field, a
    /// variant — by name; a tuple field, an alias, a trait impl's associated
    /// type — by values of their own type. Declarations are not references.
    #[test]
    fn what_hands_out_a_value_is_what_names_its_type() {
        let text = "fn make() -> T { todo!() }\n\
                    fn take(t: u8) {}\n\
                    struct S { t: T, other: u8 }\n\
                    struct W(T);\n\
                    enum E { V(T), U }\n\
                    type A = Vec<T>;\n\
                    impl Iterator for It { type Item = T; fn next(&mut self) -> Option<T> { None } }\n\
                    impl Q { fn get(&self) -> &T { todo!() } fn put(&self) {} }\n\
                    trait Tr { fn out(&self) -> T; fn other(&self); }\n\
                    use crate::T as Alias;\n";
        assert_eq!(
            handed_out(text, "T"),
            [
                "V",
                "get",
                "holds(A)",
                "holds(Alias)",
                "holds(W)",
                "impl[It; Iterator; Item,next,Iterator]",
                "make",
                "next",
                "out",
                "t"
            ]
        );
        // A field's own name is a declaration, not a reference.
        assert!(handed_out("struct S { t: u8 }", "t").is_empty());
    }

    #[test]
    fn macros_attributes_and_globs_say_how_they_reach() {
        // A body macro that may define an impl: ask for its expansion; a std
        // expression macro: no change at all. The `fn` itself is unchanged.
        assert_eq!(
            reach("fn f() {}", "fn f() { t!(\"k\"); }"),
            ["expand(attr=false)"]
        );
        assert!(reach("fn f() {}", "fn f() { println!(\"k\"); }").is_empty());
        assert_eq!(
            reach("", "#[tokio::main] async fn main() {}"),
            ["expand(attr=true)"]
        );
        assert_eq!(reach("", "my_items!(Foo);"), ["expand(attr=false)"]);
        assert_eq!(reach("", "macro_rules! m { () => {} }"), ["macro m"]);
        // Derives and their helpers are as written; `#[path]` too.
        assert_eq!(
            reach(
                "",
                "#[derive(Serialize)]\n#[serde(rename_all = \"x\")]\npub struct S;"
            ),
            ["S"]
        );
        assert_eq!(
            reach("", "#[path = \"x/y.rs\"] mod m;"),
            ["AtPath([], \"x/y.rs\")", "m@"]
        );
        assert_eq!(
            reach("", "mod a { mod b; }"),
            ["Declared([\"a\", \"b\"])", "a@", "b@a"]
        );
        // A private glob reaches its module and below; a public one, anyone.
        assert_eq!(
            reach("", "mod tests { use super::*; }"),
            ["Below([\"tests\"])", "tests@"]
        );
        assert_eq!(reach("", "pub use a::*;"), ["opaque: a public glob import"]);
        assert_eq!(
            reach("", "#![feature(x)]"),
            ["opaque: a crate attribute #![feature]"]
        );
        assert_eq!(reach("", "fn f( {"), ["opaque: text that does not parse"]);
    }

    #[test]
    fn an_expansion_in_a_body_keeps_only_what_escapes() {
        let body = SourceFile::parse(
            "fn f() { struct L; impl L { fn m(&self) {} } }",
            Edition::CURRENT,
        )
        .tree();
        let block = body
            .syntax()
            .descendants()
            .find(|node| node.kind() == SyntaxKind::BLOCK_EXPR)
            .unwrap();
        let escaping = expansion_units(&block, &[]);
        assert_eq!(escaping.len(), 1);
        assert_eq!(
            escaping[0].reach,
            Reach::Named {
                witnesses: vec![Witness::Both("m".into(), "L".into())],
                module: None
            }
        );
    }
}
