//! Indexing primitive parameter structs.

use rmcp::schemars;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub(crate) struct BuildHypergraphParams {
    #[schemars(description = "Graph scope: a workspace root, one member's directory, or any directory inside a workspace (no Cargo.toml of its own needed) — the graph holds every member crate under it, lib and bin targets, with the edges between them. Graph tools then take this same directory.")]
    pub directory: String,
    #[schemars(description = "Force a rebuild even if a snapshot for the current fingerprint already exists")]
    pub force_rebuild: Option<bool>,
    #[schemars(description = "Optional: only these member packages under `directory` (e.g. [\"store_core\", \"tsdb_engine\"]); a package's bins come with it. Part of the fingerprint: another set is a full rebuild. An unknown name is an error listing the available ones.")]
    pub crates: Option<Vec<String>>,
}
