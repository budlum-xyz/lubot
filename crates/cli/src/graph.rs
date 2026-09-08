//! # graph - the repository, as a map
//!
//! The manifest map, encoded for this tree: read the workspace
//! `Cargo.toml` files, and produce one table of crates with their internal
//! (path) dependencies, external dependency counts and source size. Nothing
//! here interprets semantics - it counts what the manifests say, and the
//! manifest is the only source, so the map cannot drift from the build.

use std::path::Path;

/// One crate's row in the map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateRow {
    pub name: String,
    pub src_files: usize,
    pub loc: usize,
    pub internal_deps: Vec<String>,
    pub external_deps: usize,
}

/// Parse the package name out of a manifest's `[package]` block.
fn package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = line.strip_prefix("name") {
                let rest = rest.trim_start();
                if let Some(name) = rest.strip_prefix('=') {
                    let name = name.trim().trim_matches('"');
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Collect the dependency lines of the `[dependencies]` block: the names,
/// and whether each is an internal (path) dependency.
fn dependencies(manifest: &str) -> Vec<(String, bool)> {
    let mut in_deps = false;
    let mut out = Vec::new();
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_deps = line.starts_with("[dependencies");
            continue;
        }
        if in_deps && !line.is_empty() && !line.starts_with('#') {
            if let Some(name) = line.split('=').next() {
                let name = name.trim().trim_matches('"').to_string();
                if !name.is_empty() {
                    let internal = line.contains("path = \"") && line.contains("../");
                    out.push((name, internal));
                }
            }
        }
    }
    out
}

fn count_src(dir: &Path) -> (usize, usize) {
    let mut files = 0usize;
    let mut loc = 0usize;
    walk(dir, &mut files, &mut loc);
    (files, loc)
}

fn walk(dir: &Path, files: &mut usize, loc: &mut usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, files, loc);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            *files += 1;
            if let Ok(body) = std::fs::read_to_string(&path) {
                *loc += body.lines().count();
            }
        }
    }
}

/// Build the repository map. Only `crates/*/Cargo.toml` are read; the
/// returned document is a Markdown table, one row per crate, stable order.
///
/// # Errors
/// When the workspace has no `crates` directory, or a manifest cannot be
/// parsed as a crate (no package name).
pub fn repo_graph(root: &Path) -> Result<String, String> {
    let crates_dir = root.join("crates");
    let entries =
        std::fs::read_dir(&crates_dir).map_err(|e| format!("{}: {e}", crates_dir.display()))?;
    let mut rows: Vec<CrateRow> = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        let manifest_path = dir.join("Cargo.toml");
        if !manifest_path.is_file() {
            continue;
        }
        let manifest = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        let Some(name) = package_name(&manifest) else {
            return Err(format!("{}: no package name", manifest_path.display()));
        };
        let deps = dependencies(&manifest);
        let (src_files, loc) = count_src(&dir.join("src"));
        rows.push(CrateRow {
            internal_deps: deps
                .iter()
                .filter(|(_, internal)| *internal)
                .map(|(name, _)| name.clone())
                .collect(),
            external_deps: deps.iter().filter(|(_, internal)| !*internal).count(),
            name,
            src_files,
            loc,
        });
    }
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    if rows.is_empty() {
        return Err(format!("{}: no crates found", crates_dir.display()));
    }
    let total_files: usize = rows.iter().map(|r| r.src_files).sum();
    let total_loc: usize = rows.iter().map(|r| r.loc).sum();
    let mut doc = String::from("# Graf\n\n| crate | src files | LOC | internal deps | external deps |\n|---|---|---|---|---|\n");
    for row in &rows {
        let internal = row.internal_deps.join(", ");
        doc.push_str(&format!(
            "| {name} | {files} | {loc} | {internal} | {external} |\n",
            name = row.name,
            files = row.src_files,
            loc = row.loc,
            internal = if internal.is_empty() { "-" } else { &internal },
            external = row.external_deps,
        ));
    }
    doc.push_str(&format!(
        "\n{} crates, {total_files} src files, {total_loc} LOC\n",
        rows.len()
    ));
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_give_names_deps_and_counts() {
        let manifest = r#"
[package]
name = "lubot-tools"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
lubot-read = { path = "../read" }
lubot-grant = { path = "../grant" }
"#;
        assert_eq!(package_name(manifest).as_deref(), Some("lubot-tools"));
        let deps = dependencies(manifest);
        assert_eq!(deps.len(), 3);
        assert_eq!(deps[0], ("serde".to_string(), false));
        assert_eq!(deps[1], ("lubot-read".to_string(), true));
    }

    #[test]
    fn a_missing_package_name_refuses_the_whole_map() {
        assert_eq!(package_name("[workspace]\nmembers = []"), None);
    }

    #[test]
    fn the_repo_map_passes_the_schema() {
        let doc = repo_graph(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .as_path(),
        )
        .expect("repo graph builds");
        lubot_read::output_schema::validate_markdown_output(doc.as_bytes())
            .expect("repo graph passes the Markdown schema");
    }
}
