//! Position and coordinate utilities

use anyhow::{Context, Result};
use ra_ap_ide::{
    Analysis, AnalysisHost, FilePosition, LineCol, NavigationTarget, Query, SymbolKind, TextSize,
};
use ra_ap_vfs::{Vfs, VfsPath};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// A source code location
#[derive(Debug, Clone)]
pub(crate) struct Location {
    pub file_path: PathBuf,
    pub line: u32,   // 1-based
    pub column: u32, // 1-based
    pub name: String,
    pub exact: bool,
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{} ({}, exact={})",
            self.file_path.display(),
            self.line,
            self.column,
            self.name,
            self.exact
        )
    }
}

/// Convert file path to FileId
fn path_to_file_id(vfs: &Vfs, file_path: &Path) -> Result<ra_ap_vfs::FileId> {
    let abs_path = file_path
        .canonicalize()
        .context("Failed to canonicalize path")?;
    let vfs_path = VfsPath::new_real_path(abs_path.to_string_lossy().to_string());

    vfs.file_id(&vfs_path)
        .map(|(id, _)| id)
        .ok_or_else(|| anyhow::anyhow!("File not found in VFS: {}", file_path.display()))
}

/// Convert line/column to byte offset
fn to_offset(
    analysis: &Analysis,
    file_id: ra_ap_vfs::FileId,
    line: u32,
    column: u32,
) -> Result<TextSize> {
    let line_index = analysis
        .file_line_index(file_id)
        .context("Failed to get line index")?;

    // LineCol is 0-based, input is 1-based
    let line_col = LineCol {
        line: line.saturating_sub(1),
        col: column.saturating_sub(1),
    };

    line_index
        .offset(line_col)
        .ok_or_else(|| anyhow::anyhow!("Invalid position: line {}, col {}", line, column))
}

pub(crate) fn file_position(
    analysis: &Analysis,
    vfs: &Vfs,
    file_path: &Path,
    line: u32,
    column: u32,
) -> Result<FilePosition> {
    let file_id = path_to_file_id(vfs, file_path)?;
    let offset = to_offset(analysis, file_id, line, column)?;

    Ok(FilePosition { file_id, offset })
}

/// Convert NavigationTarget to Location
fn nav_target_to_location(
    vfs: &Vfs,
    analysis: &ra_ap_ide::Analysis,
    target: &NavigationTarget,
) -> Result<Location> {
    let vfs_path = vfs.file_path(target.file_id);
    let file_path: PathBuf = vfs_path
        .as_path()
        .ok_or_else(|| anyhow::anyhow!("Not a real path"))?
        .to_path_buf()
        .into();

    let line_index = analysis.file_line_index(target.file_id)?;
    let offset = target.focus_range.unwrap_or(target.full_range).start();
    let line_col = line_index.line_col(offset);

    Ok(Location {
        file_path,
        line: line_col.line + 1,
        column: line_col.col + 1,
        name: target.name.to_string(),
        exact: false,
    })
}

/// Goto definition at position
pub(crate) fn goto_definition(
    host: &AnalysisHost,
    vfs: &Vfs,
    file_path: &Path,
    line: u32,
    column: u32,
) -> Result<Vec<Location>> {
    let analysis = host.analysis();
    let position = file_position(&analysis, vfs, file_path, line, column)?;
    let config = ra_ap_ide::GotoDefinitionConfig {
        ra_fixture: ra_ap_ide_db::ra_fixture::RaFixtureConfig::default(),
    };

    let result = analysis
        .goto_definition(position, &config)
        .context("goto_definition query failed")?;

    match result {
        Some(nav_info) => nav_info
            .info
            .iter()
            .map(|target| nav_target_to_location(vfs, &analysis, target))
            .collect(),
        None => Ok(vec![]),
    }
}

/// The reference-search configuration every lookup in this module shares.
///
/// Both exclusions are off on purpose: a `use` line and a `#[cfg(test)]` call
/// site *are* references, and dropping either makes "who uses this" read lower
/// than the truth — the failure this module is built to avoid.
fn all_refs_config() -> ra_ap_ide::FindAllRefsConfig<'static> {
    ra_ap_ide::FindAllRefsConfig {
        ra_fixture: ra_ap_ide_db::ra_fixture::RaFixtureConfig::default(),
        search_scope: None,
        exclude_imports: false,
        exclude_tests: false,
    }
}

/// What a position resolved to — enough to tell the name that was asked for
/// from something else spelled the same way.
struct ResolvedDecl {
    name: String,
    kind: Option<SymbolKind>,
}

/// Run `find_all_refs` at `position` and append everything it found to `out`.
///
/// Returns the definition the position resolved to, or `None` when the position
/// pointed at no symbol at all. That distinction is the whole reason this
/// returns anything: `find_all_refs` reports "the cursor is not on a name" and
/// "this name has no references" identically, as an empty result, and a caller
/// that cannot tell them apart reports a missed position as "nothing uses this".
fn harvest_refs_at(
    analysis: &Analysis,
    vfs: &Vfs,
    position: FilePosition,
    exact: bool,
    out: &mut Vec<Location>,
) -> Result<Option<ResolvedDecl>> {
    let Some(search_results) = analysis
        .find_all_refs(position, &all_refs_config())
        .context("find_all_refs query failed")?
    else {
        return Ok(None);
    };

    let mut resolved_name = None;

    for search_result in search_results {
        // Declaration.nav is NavigationTarget (not Option)
        if let Some(decl) = &search_result.declaration {
            resolved_name.get_or_insert_with(|| ResolvedDecl {
                name: decl.nav.name.to_string(),
                kind: decl.nav.kind,
            });
            let mut decl_location = nav_target_to_location(vfs, analysis, &decl.nav)?;
            decl_location.exact = exact;
            out.push(decl_location);
        }

        // references is IntMap<FileId, Vec<(TextRange, ReferenceCategory)>>
        for (ref_file_id, refs) in &search_result.references {
            let ref_file_path = file_path_of(vfs, *ref_file_id)?;
            let ref_line_index = analysis.file_line_index(*ref_file_id)?;

            for (range, _category) in refs {
                let line_col = ref_line_index.line_col(range.start());
                out.push(Location {
                    file_path: ref_file_path.clone(),
                    line: line_col.line + 1,
                    column: line_col.col + 1,
                    name: "reference".to_string(),
                    exact,
                });
            }
        }
    }

    Ok(resolved_name)
}

/// Real filesystem path of a file the analysis knows.
fn file_path_of(vfs: &Vfs, file_id: ra_ap_vfs::FileId) -> Result<PathBuf> {
    Ok(vfs
        .file_path(file_id)
        .as_path()
        .ok_or_else(|| anyhow::anyhow!("Not a real path"))?
        .to_path_buf()
        .into())
}

/// Find all references at position
///
/// A position that points at no symbol is an **error**, not an empty list:
/// those two answers are the same string to a caller, and one of them means
/// "ask again, one character to the right".
pub(crate) fn find_references(
    host: &AnalysisHost,
    vfs: &Vfs,
    file_path: &Path,
    line: u32,
    column: u32,
) -> Result<Vec<Location>> {
    let analysis = host.analysis();
    let position = file_position(&analysis, vfs, file_path, line, column)?;

    let mut locations = Vec::new();
    if harvest_refs_at(&analysis, vfs, position, false, &mut locations)?.is_none() {
        anyhow::bail!(
            "No symbol at {}:{}:{} — the position does not point at a name (try a column inside the identifier)",
            file_path.display(),
            line,
            column
        );
    }

    Ok(locations)
}

/// Search for symbols by name
pub(crate) fn symbol_search(
    host: &AnalysisHost,
    vfs: &Vfs,
    symbol_name: &str,
    limit: usize,
) -> Result<Vec<Location>> {
    symbol_search_with_exact(host, vfs, symbol_name, limit, false)
}

/// Search for symbols by name, optionally retaining only full-name matches.
pub(crate) fn symbol_search_with_exact(
    host: &AnalysisHost,
    vfs: &Vfs,
    symbol_name: &str,
    limit: usize,
    exact_only: bool,
) -> Result<Vec<Location>> {
    let analysis = host.analysis();

    let query = Query::new(symbol_name.to_string());
    let results = analysis
        .symbol_search(query, limit)
        .context("symbol_search query failed")?;

    let locations = results
        .iter()
        .map(|target| {
            let mut location = nav_target_to_location(vfs, &analysis, target)?;
            location.exact = location.name == symbol_name;
            Ok(location)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(rank_and_filter_exact(locations, exact_only))
}

/// Find all references to symbols matching a name
/// First finds all symbols with that name, then finds references for each
pub(crate) fn find_references_by_name(
    host: &AnalysisHost,
    vfs: &Vfs,
    project_root: &Path,
    symbol_name: &str,
) -> Result<Vec<Location>> {
    find_references_by_name_with_exact(host, vfs, project_root, symbol_name, false)
}

/// Find all references to symbols matching a name, optionally retaining only
/// full-name symbol matches before resolving references.
pub(crate) fn find_references_by_name_with_exact(
    host: &AnalysisHost,
    vfs: &Vfs,
    project_root: &Path,
    symbol_name: &str,
    exact_only: bool,
) -> Result<Vec<Location>> {
    let analysis = host.analysis();

    // First, find all symbols matching the name
    let query = Query::new(symbol_name.to_string());
    let mut symbols = analysis
        .symbol_search(query, 50)
        .context("symbol_search query failed")?;
    symbols.sort_by_key(|symbol| symbol.name.to_string() != symbol_name);

    let mut all_locations = Vec::new();
    let mut resolved_the_name = false;

    // For each symbol found, find its references
    for symbol in &symbols {
        let symbol_exact = symbol.name.to_string() == symbol_name;
        if exact_only && !symbol_exact {
            continue;
        }
        resolved_the_name |= symbol_exact;

        // Get the position of this symbol definition
        let file_id = symbol.file_id;
        let offset = symbol.focus_range.unwrap_or(symbol.full_range).start();
        let position = FilePosition { file_id, offset };

        harvest_refs_at(&analysis, vfs, position, symbol_exact, &mut all_locations)?;
    }

    // The symbol index carries module-scope declarations plus impl and trait
    // members. It does NOT carry struct fields, and `SymbolCollector` skips
    // enum variants outright — so a field name matches nothing above and the
    // answer would be an empty list, which reads exactly like "nobody touches
    // this field". Ask the source instead.
    if !resolved_the_name {
        collect_by_token_scan(
            &analysis,
            vfs,
            project_root,
            symbol_name,
            &mut all_locations,
        )?;
    }

    // Deduplicate locations (same file:line:col)
    all_locations.sort_by(|a, b| {
        (&a.file_path, a.line, a.column, !a.exact).cmp(&(&b.file_path, b.line, b.column, !b.exact))
    });
    all_locations
        .dedup_by(|a, b| a.file_path == b.file_path && a.line == b.line && a.column == b.column);
    all_locations.sort_by_key(|location| !location.exact);

    Ok(all_locations)
}

/// Resolve `symbol_name` by finding the identifier in the project's own sources
/// and asking rust-analyzer what sits under it.
///
/// This is the path for names the symbol index does not carry — struct fields
/// and enum variants. It runs second because when the index *does* know a name
/// its answer is authoritative and costs one query instead of a file sweep.
///
/// # Why not every occurrence is queried
///
/// The first occurrence that resolves brings back the entire reference set for
/// that definition, which normally accounts for every other occurrence of the
/// name. Those are then skipped, so a field read twenty-one times costs one
/// query rather than twenty-one. What still costs a query each: an occurrence
/// belonging to a *different* definition (a same-named field on another
/// struct — both are wanted, so both are searched), and an occurrence in a
/// comment or string literal, which resolves to nothing.
///
/// # Scope
///
/// Only `.rs` files under `project_root` are swept. Under a full workspace load
/// the VFS also holds every dependency's sources, and a name that happens to
/// exist in a registry crate is not what the caller asked about.
fn collect_by_token_scan(
    analysis: &Analysis,
    vfs: &Vfs,
    project_root: &Path,
    symbol_name: &str,
    out: &mut Vec<Location>,
) -> Result<()> {
    // Sorted, because the VFS hands files back in load order: two runs that
    // probe different occurrences first would otherwise answer in different
    // order.
    let mut files: Vec<ra_ap_vfs::FileId> = vfs
        .iter()
        .filter(|(_, path)| {
            path.as_path().is_some_and(|path| {
                let path: &Path = path.as_ref();
                path.extension().is_some_and(|ext| ext == "rs") && path.starts_with(project_root)
            })
        })
        .map(|(file_id, _)| file_id)
        .collect();
    files.sort();

    // Positions already accounted for, so a definition reached through its
    // first occurrence is not re-queried through its remaining twenty.
    let mut covered: HashSet<(PathBuf, u32, u32)> = out.iter().map(location_key).collect();

    for file_id in files {
        let text = analysis.file_text(file_id)?;
        let line_index = analysis.file_line_index(file_id)?;
        let file_path = file_path_of(vfs, file_id)?;

        for start in identifier_occurrences(&text, symbol_name) {
            let line_col = line_index.line_col(TextSize::new(start as u32));
            let probe = (file_path.clone(), line_col.line + 1, line_col.col + 1);
            if !covered.insert(probe) {
                continue;
            }

            // One character INTO the identifier, not at its first byte:
            // rust-analyzer answers an empty list for a field probed at its
            // first character while answering in full one character later
            // (measured 2026-08-28). Single-character names have no inside.
            let inside = if symbol_name.len() > 1 {
                start + 1
            } else {
                start
            };
            let position = FilePosition {
                file_id,
                offset: TextSize::new(inside as u32),
            };

            let before = out.len();
            let resolved = harvest_refs_at(analysis, vfs, position, true, out)?;
            if !resolved.is_some_and(|decl| is_the_wanted_name(&decl, symbol_name)) {
                // A comment, a string, a local binding spelled the same: the
                // occurrence looked right in text and is not this name. Drop
                // whatever it dragged in.
                out.truncate(before);
                continue;
            }
            covered.extend(out[before..].iter().map(location_key));
        }
    }

    Ok(())
}

/// Whether a resolved definition is the thing the token sweep went looking for.
///
/// The name has to match, and so does the *kind*: the sweep exists only for the
/// two kinds rust-analyzer's symbol index leaves out, so anything else found
/// under a matching identifier is a different entity that happens to share a
/// spelling. A local `let radius_override = 7;` declares a name equal to the
/// field's, and counting it and its uses inflated a four-read field to six —
/// silently, which is the failure mode this whole path exists to end.
fn is_the_wanted_name(decl: &ResolvedDecl, symbol_name: &str) -> bool {
    decl.name == symbol_name && matches!(decl.kind, Some(SymbolKind::Field | SymbolKind::Variant))
}

/// Dedup key: one source position, however it was reached.
fn location_key(location: &Location) -> (PathBuf, u32, u32) {
    (location.file_path.clone(), location.line, location.column)
}

/// Byte offsets at which `name` appears as a whole identifier.
///
/// The boundary test is on `char`, not on bytes, so that a name butting against
/// a multi-byte character in a comment is not mistaken for a longer identifier.
fn identifier_occurrences(text: &str, name: &str) -> Vec<usize> {
    if name.is_empty() {
        return Vec::new();
    }
    let is_ident_char = |c: char| c.is_alphanumeric() || c == '_';

    text.match_indices(name)
        .filter(|(start, _)| {
            let before_ok = text[..*start]
                .chars()
                .next_back()
                .is_none_or(|c| !is_ident_char(c));
            let after_ok = text[start + name.len()..]
                .chars()
                .next()
                .is_none_or(|c| !is_ident_char(c));
            before_ok && after_ok
        })
        .map(|(start, _)| start)
        .collect()
}

fn rank_and_filter_exact(mut locations: Vec<Location>, exact_only: bool) -> Vec<Location> {
    if exact_only {
        locations.retain(|location| location.exact);
    }
    locations.sort_by_key(|location| !location.exact);
    locations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(name: &str, exact: bool, line: u32) -> Location {
        Location {
            file_path: PathBuf::from(format!("{name}.rs")),
            line,
            column: 1,
            name: name.to_string(),
            exact,
        }
    }

    #[test]
    fn rank_and_filter_exact_keeps_substrings_by_default_but_ranks_exact_first() {
        let locations = vec![
            loc("VectorSearchResult", false, 2),
            loc("SearchResult", true, 1),
        ];
        let ranked = rank_and_filter_exact(locations, false);

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].name, "SearchResult");
        assert!(ranked[0].exact);
        assert_eq!(ranked[1].name, "VectorSearchResult");
        assert!(!ranked[1].exact);
    }

    #[test]
    fn rank_and_filter_exact_can_drop_substring_matches() {
        let locations = vec![
            loc("VectorSearchResult", false, 2),
            loc("SearchResult", true, 1),
        ];
        let ranked = rank_and_filter_exact(locations, true);

        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].name, "SearchResult");
        assert!(ranked[0].exact);
    }
}
