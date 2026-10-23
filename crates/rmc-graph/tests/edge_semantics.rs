//! Independent selected-site oracle, authored before extractor fixes.
use std::{fs, path::Path};
use rmc_graph::graph::{self, BuildOptions, GraphEnvOptions, GraphPaths, Usage};

fn copy_tree(source: &Path, target: &Path) {
    fs::create_dir_all(target).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() == "target" { continue; }
        let dest = target.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() { copy_tree(&entry.path(), &dest); }
        else { fs::copy(entry.path(), dest).unwrap(); }
    }
}

fn marked_site(root: &Path, site: &str, token: &str) -> (String, u32, u32) {
    for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if entry.path().extension().is_none_or(|e| e != "rs") || entry.path().components().any(|p| p.as_os_str() == "target") { continue; }
        let text = fs::read_to_string(entry.path()).unwrap();
        if let Some(start) = text.find(&format!("/*@{site}*/")) {
            let after_marker = start + site.len() + 5;
            let offset = after_marker + text[after_marker..].find(token).unwrap();
            return (entry.path().strip_prefix(root).unwrap().to_str().unwrap().to_owned(), offset as u32, (offset + token.len()) as u32);
        }
    }
    panic!("missing marker {site}");
}

#[tokio::test]
async fn handwritten_edges_and_call_projections() {
    let work = tempfile::tempdir().unwrap();
    copy_tree(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/graph_truth"), work.path());
    let data = tempfile::tempdir().unwrap();
    let built = graph::build_and_persist(work.path(), BuildOptions { data_dir_override: Some(data.path().to_owned()), ..Default::default() }).unwrap();
    let paths = GraphPaths::for_workspace_in(data.path(), &built.workspace_root);
    let snap = graph::open_current(&paths, GraphEnvOptions::default()).unwrap().unwrap();
    let txn = snap.read_txn().unwrap();
    let usages: Vec<Usage> = snap.dbs.usages_by_id.iter(&txn).unwrap().map(|r| r.unwrap().1).collect();
    let name = |id| snap.node_by_id(&txn, id).unwrap().unwrap().qualified_name;
    let expected = fs::read_to_string(work.path().join("expected.tsv")).unwrap();
    let mut checked = 0;
    let mut excluded = 0;
    for row in expected.lines().skip(1) {
        let cols: Vec<_> = row.split('\t').collect();
        let (site, token, relation, target, owner, coverage) = (cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]);
        let (file, start, end) = marked_site(work.path(), site, token);
        let hits: Vec<_> = usages.iter().filter(|u| u.file == file && u.start <= start && u.end >= end).collect();
        if relation == "Import" {
            assert!(hits.is_empty(), "import also became usage: {site}");
            continue;
        }
        if coverage != "included" {
            if relation == "UnresolvedCall" {
                let caller = snap.lookup_by_qualified_name(owner).unwrap().unwrap().0;
                let unresolved = snap.unresolved_calls_from(caller).unwrap();
                assert!(unresolved.iter().any(|u| u.file == file && u.start <= start && u.end >= end), "unknown lost: {site}");
            } else { assert!(hits.is_empty(), "inactive/excluded edge: {site}: {hits:?}"); excluded += 1; }
            continue;
        }
        // Cargo-target disambiguation is the next task. Still check the edge,
        // relation and owner name here, without claiming target identity.
        let target = target.replace("[lib]", "").replace("[bin]", "");
        let owner = owner.replace("[lib]", "").replace("[bin]", "");
        assert!(hits.iter().any(|u| name(u.target) == target
            && u.consumer_item.is_some_and(|id| name(id) == owner)
            && format!("{:?}", u.relation) == relation), "missing/wrong {site}: expected {owner} -{relation}-> {target}, got {hits:?}");
        checked += 1;
    }
    assert_eq!(checked, 42);
    assert_eq!(excluded, 7);
    drop(txn);
    let caller = snap.lookup_by_qualified_name("truth_app::references_only").unwrap().unwrap().0;
    let calls = snap.calls_from(caller).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].callee_qualified_name, "truth_dep::register");
    let leaf = snap.lookup_by_qualified_name("truth_dep::leaf").unwrap().unwrap().0;
    assert!(!snap.who_calls(leaf).unwrap().iter().any(|u| u.caller_qualified_name.as_deref() == Some("truth_app::references_only")));
    let seeds = vec!["truth_app::references_only".to_owned()];
    let cm = graph::codemap::build_codemap(&snap, None, Some(&seeds), None, &Default::default(), &[]).await.unwrap();
    use graph::codemap::EdgeKind;
    assert!(cm.edges.iter().any(|e| e.from == caller && e.to == leaf && e.kind == EdgeKind::FunctionReference));
    assert!(!cm.edges.iter().any(|e| e.from == caller && e.to == leaf && e.kind == EdgeKind::Calls));
}

#[test]
fn syntax_roles_disabled_fields_and_macro_field_targets() {
    let work = tempfile::tempdir().unwrap();
    fs::write(work.path().join("Cargo.toml"), "[package]\nname='roles'\nversion='0.1.0'\nedition='2021'\n[workspace]\n").unwrap();
    fs::create_dir(work.path().join("src")).unwrap();
    let source = r#"
pub struct Hidden;
pub struct Data {
    pub inner: Inner,
    #[cfg(any())] pub disabled: Hidden,
}
pub struct Inner { pub value: u32 }
pub fn leaf() {}
pub fn roles(d: &mut Data) {
    let _f = leaf;
    (leaf)();
    d.inner.value = 1;
    let f: fn() = leaf;
    (f)();
}
pub fn signature(_: Hidden) -> Hidden { Hidden }
macro_rules! make { () => { pub struct Generated { pub field: Hidden } } }
make!();
pub fn read_generated(g: &Generated) { let _ = &g.field; }
pub trait Contract<const N: usize> { fn act(&self); }
impl Contract<1> for Data { fn act(&self) { leaf(); } }
pub fn generic<T: Contract<1>>(t: &T) { t.act(); }
"#;
    fs::write(work.path().join("src/lib.rs"), source).unwrap();
    let loaded = graph::load(work.path()).unwrap();
    let model = graph::extract_workspace_model(&loaded);
    let named = |name: &str| *model.nodes.iter().find(|(_, n)| n.qualified_name == name).unwrap().0;
    let leaf = named("roles::leaf");
    let roles = named("roles::roles");
    let mut relations: Vec<_> = model.usages.iter().filter(|u| u.target == leaf && u.consumer_function == Some(roles))
        .map(|u| format!("{:?}", u.relation)).collect();
    relations.sort();
    assert_eq!(relations, ["Call", "FunctionReference", "FunctionReference"]);
    assert!(model.unresolved_calls.iter().any(|c| c.expression == "(f)" && c.consumer_item == Some(roles)));
    let inner = named("roles::Data::inner");
    let value = named("roles::Inner::value");
    assert!(model.usages.iter().any(|u| u.target == inner && format!("{:?}", u.relation) == "Read"));
    assert!(model.usages.iter().any(|u| u.target == value && format!("{:?}", u.relation) == "Write"));
    let disabled = source.find("disabled: Hidden").unwrap() + "disabled: ".len();
    assert!(!model.usages.iter().any(|u| u.start as usize == disabled));
    let field = named("roles::Generated::field");
    assert!(model.usages.iter().any(|u| u.target == field));
    let sig = named("roles::signature");
    let type_sites: Vec<_> = model.usages.iter().filter(|u| u.consumer_item == Some(sig) && format!("{:?}", u.relation) == "TypeRef").collect();
    assert_eq!(type_sites.len(), 2);
    assert!(type_sites.iter().all(|u| u.consumer_function.is_none()));
    let generic = named("roles::generic");
    assert!(model.usages.iter().any(|u| u.consumer_function == Some(generic) && format!("{:?}", u.relation) == "TraitDispatch"));
    assert!(model.usages.iter().all(|u| model.nodes.contains_key(&u.target)), "dangling usage target");
}
