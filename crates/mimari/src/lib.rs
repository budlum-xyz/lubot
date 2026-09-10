//! An architecture table that cannot lie about the code.
//!
//! Module documentation rots in a specific, repeatable way: a function is
//! renamed, its entry point moves, a stage is deleted, and the table describing
//! the pipeline keeps naming the thing that no longer exists. Nothing fails. A
//! reader trusts the table, greps for a symbol, finds nothing, and concludes
//! they are the confused one.
//!
//! # The rule
//!
//! Every symbol named in the table must exist in the file the table says it
//! lives in. That is all this checks, and it is enough: it turns documentation
//! into a claim the tree can answer.
//!
//! ```text
//! | module | entry point | input -> output |
//! ```
//!
//! Three columns, first backticked token in columns one and two, and a non-empty
//! third column. A row whose contract is empty is a violation, not a formatting
//! choice: a table full of names with no claims about them is documentation that
//! reads like a specification and constrains nothing.
//!
//! # What it does not claim
//!
//! It cannot tell you the contract is *true* - only that the thing it names is
//! there. A signature that exists and behaves differently is a different class
//! of failure, and the honest answer to that class is a test, which is why this
//! crate is used from one: `tests/table_matches_tree.rs` in the workspace reads
//! the real sources and feeds them here.

use std::collections::BTreeMap;

/// One row of the table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    module: String,
    symbol: String,
    contract: String,
    line: usize,
}

impl Row {
    /// The file the row claims the symbol lives in.
    #[must_use]
    pub fn module(&self) -> &str {
        &self.module
    }

    /// The entry point named.
    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// The claim about it.
    #[must_use]
    pub fn contract(&self) -> &str {
        &self.contract
    }

    /// Where the row was found, for the message.
    #[must_use]
    pub fn line(&self) -> usize {
        self.line
    }
}

/// Why a table could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// A pipe row that does not have the three columns the format promises.
    WrongColumnCount {
        /// Line number, one-based.
        line: usize,
        /// Columns found.
        found: usize,
    },
    /// A row with no module named.
    MissingModule {
        /// Line number.
        line: usize,
    },
    /// A row with no entry point named.
    MissingSymbol {
        /// Line number.
        line: usize,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongColumnCount { line, found } => {
                write!(f, "line {line}: {found} columns, expected 3")
            }
            Self::MissingModule { line } => write!(f, "line {line}: no module in column 1"),
            Self::MissingSymbol { line } => write!(f, "line {line}: no entry point in column 2"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Why a table drifts from the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Drift {
    /// The table says the symbol lives here; it does not.
    Missing {
        /// The file.
        module: String,
        /// The symbol.
        symbol: String,
        /// The row's line, so the fix is findable.
        line: usize,
    },
    /// The row names a module this tree does not have a file for.
    UnknownModule {
        /// The file.
        module: String,
        /// The row's line.
        line: usize,
    },
    /// The row names something and says nothing about it.
    Contractless {
        /// The symbol.
        symbol: String,
        /// The row's line.
        line: usize,
    },
    /// A table with no rows checks nothing and passes, which is the worst thing
    /// a gate can do.
    Empty,
    /// More rows than the caller is willing to accept as "a real table": a
    /// pipeline described in one row is a slogan, not an architecture.
    Vacuous {
        /// Rows found.
        found: usize,
        /// Rows required.
        floor: usize,
    },
}

impl std::fmt::Display for Drift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing {
                module,
                symbol,
                line,
            } => write!(
                f,
                "line {line}: the table names `{symbol}` in `{module}` and no such item is \
                 declared there; the row is describing a symbol that has been renamed or removed"
            ),
            Self::UnknownModule { module, line } => {
                write!(f, "line {line}: `{module}` is not in the tree")
            }
            Self::Contractless { symbol, line } => write!(
                f,
                "line {line}: the row for `{symbol}` has an empty contract; a name with no \
                 claim about it constrains nothing"
            ),
            Self::Empty => write!(
                f,
                "the table has no rows; a gate that passes because it was handed nothing is \
                 not a gate"
            ),
            Self::Vacuous { found, floor } => write!(
                f,
                "the table has {found} rows, below the floor of {floor}: one row is a slogan"
            ),
        }
    }
}

impl std::error::Error for Drift {}

/// The parsed table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    rows: Vec<Row>,
}

impl Table {
    /// Reads markdown pipe rows, skipping the header separator row.
    ///
    /// Lines that are not pipe rows are ignored, so prose around the table is
    /// not a parse failure.
    ///
    /// # Errors
    ///
    /// The first [`ParseError`] found.
    pub fn parse(markdown: &str) -> Result<Self, ParseError> {
        let mut rows = Vec::new();
        for (idx, raw) in markdown.lines().enumerate() {
            let line = raw.trim();
            if !line.starts_with('|') {
                continue;
            }
            let cells: Vec<&str> = line
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect();
            if cells.len() < 3 {
                return Err(ParseError::WrongColumnCount {
                    line: idx + 1,
                    found: cells.len(),
                });
            }
            // `|---|---|---|` and its variants: a separator, and the row above
            // it was a header, not a claim about the code.
            if !cells.is_empty()
                && cells
                    .iter()
                    .all(|c| c.chars().all(|ch| matches!(ch, '-' | ':' | ' ')))
            {
                if let Some(last) = rows.last() {
                    if last.line + 1 == idx + 1 {
                        rows.pop();
                    }
                }
                continue;
            }
            let module = cell_text(cells[0]).ok_or(ParseError::MissingModule { line: idx + 1 })?;
            let symbol = first_token(cells[1]).ok_or(ParseError::MissingSymbol { line: idx + 1 })?;
            let contract = cells[2].trim_matches('`').trim().to_string();
            rows.push(Row {
                module,
                symbol,
                contract,
                line: idx + 1,
            });
        }
        Ok(Self { rows })
    }

    /// The rows, in document order.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// The tree the table is checked against: file contents, keyed by path.
#[derive(Debug, Clone, Default)]
pub struct SourceTree {
    files: BTreeMap<String, String>,
}

impl SourceTree {
    /// A tree from pairs of `(path, contents)`.
    #[must_use]
    pub fn from_pairs(pairs: &[(String, String)]) -> Self {
        Self {
            files: pairs.iter().cloned().collect(),
        }
    }

    /// True when the file exists here.
    #[must_use]
    pub fn has_file(&self, module: &str) -> bool {
        self.files.contains_key(module) || self.files.contains_key(&normalize(module))
    }

    /// True when `symbol` is *declared* in `module`.
    ///
    /// Declaration keywords are matched per language so the same table can be
    /// checked in a Rust crate and in a Python module without a parser.
    #[must_use]
    pub fn declares(&self, module: &str, symbol: &str) -> bool {
        let key = if self.files.contains_key(module) {
            module.to_string()
        } else {
            normalize(module)
        };
        let Some(text) = self.files.get(&key) else {
            return false;
        };
        for prefix in [
            "fn ", "pub fn ", "async fn ", "struct ", "enum ", "trait ", "type ", "const ",
            "static ", "mod ", "def ", "class ", "async def ",
        ] {
            let needle = format!("{prefix}{symbol}");
            if text.contains(&needle) {
                return true;
            }
        }
        false
    }
}

/// Strips the leading `./` a path may carry.
fn normalize(module: &str) -> String {
    module.trim_start_matches("./").to_string()
}

/// The plain content of a cell, backticks removed.
///
/// Used for the module column, where the whole path is the name.
fn cell_text(cell: &str) -> Option<String> {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return None;
    }
    let inner = if let Some(rest) = trimmed.strip_prefix('`') {
        rest.split('`').next().unwrap_or("")
    } else {
        trimmed
    };
    let text = inner.trim();
    if text.is_empty() || text == "-" || text == "\u{2014}" {
        return None;
    }
    Some(text.to_string())
}

/// The first backticked token in a cell, or the bare word if it is unquoted.
///
/// `entry.py::run`, `` `run(x, y)` `` and `run` all name `run`: arguments and
/// paths are decoration, and a check that demanded an exact signature would be
/// rewritten the first time someone typed a parameter name.
fn first_token(cell: &str) -> Option<String> {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = if let Some(rest) = trimmed.strip_prefix('`') {
        rest.split('`').next().unwrap_or("")
    } else {
        trimmed
    };
    let last = candidate
        .rsplit("::")
        .next()
        .unwrap_or(candidate)
        .rsplit('/')
        .next()
        .unwrap_or(candidate);
    let name = last
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
        .collect::<String>();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// What a check found, counted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Report {
    /// Rows examined.
    pub checked: usize,
    /// Rows that passed.
    pub matched: usize,
    /// The failures, in row order.
    pub findings: Vec<String>,
}

impl Report {
    /// True when nothing failed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    /// The report as a table body, for a log or a CI summary.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_clean() {
            return format!("{} rows checked, all of them name a real symbol", self.checked);
        }
        let mut out = format!(
            "{} of {} rows do not match the tree:\n",
            self.findings.len(),
            self.checked
        );
        for line in &self.findings {
            out.push_str("  - ");
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

/// Checks a table against a tree.
///
/// `floor` is the smallest row count that counts as a real table; a call with a
/// floor of zero only checks the non-empty rule.
///
/// # Errors
///
/// The first [`Drift`] found, so the fix is one row at a time and unambiguous.
pub fn check(table: &Table, tree: &SourceTree, floor: usize) -> Result<Report, Drift> {
    if table.is_empty() {
        return Err(Drift::Empty);
    }
    if table.len() < floor {
        return Err(Drift::Vacuous {
            found: table.len(),
            floor,
        });
    }
    let mut report = Report {
        checked: table.len(),
        matched: 0,
        findings: Vec::new(),
    };
    for row in table.rows() {
        if row.contract.trim().is_empty() || row.contract == "-" || row.contract == "—" {
            return Err(Drift::Contractless {
                symbol: row.symbol().to_string(),
                line: row.line(),
            });
        }
        if !tree.has_file(row.module()) {
            return Err(Drift::UnknownModule {
                module: row.module().to_string(),
                line: row.line(),
            });
        }
        if !tree.declares(row.module(), row.symbol()) {
            return Err(Drift::Missing {
                module: row.module().to_string(),
                symbol: row.symbol().to_string(),
                line: row.line(),
            });
        }
        report.matched += 1;
    }
    Ok(report)
}

/// The failure form of [`check`], for a CI gate that only needs a yes or no.
///
/// # Errors
///
/// The first [`Drift`] found.
pub fn verify(table: &Table, tree: &SourceTree, floor: usize) -> Result<(), Drift> {
    check(table, tree, floor).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> SourceTree {
        let files = vec![
            (
                "crates/core/src/detect.rs".to_string(),
                "pub fn detect(root: &str) -> Summary {}\nstruct Summary {}".to_string(),
            ),
            (
                "crates/core/src/build.rs".to_string(),
                "pub fn build(items: usize) -> Graph {}\nfn helper() {}".to_string(),
            ),
        ];
        SourceTree::from_pairs(&files)
    }

    const GOOD: &str = "| module | entry point | input -> output |\n\
                         |---|---|---|\n\
                         | `crates/core/src/detect.rs` | `detect(root)` | directory -> summary |\n\
                         | `crates/core/src/build.rs` | `build` | list -> graph |\n";

    #[test]
    fn a_table_naming_real_symbols_passes() {
        let table = Table::parse(GOOD).expect("parses");
        assert_eq!(table.len(), 2);
        let report = check(&table, &tree(), 2).expect("clean");
        assert!(report.is_clean());
        assert_eq!(report.matched, 2);
        assert!(report.summary().contains("all of them"));
    }

    #[test]
    fn a_renamed_symbol_fails_with_the_line_that_says_so() {
        let markdown = GOOD.replace("`detect(root)`", "`detect_all(root)`");
        let table = Table::parse(&markdown).expect("parses");
        assert_eq!(
            verify(&table, &tree(), 2),
            Err(Drift::Missing {
                module: "crates/core/src/detect.rs".to_string(),
                symbol: "detect_all".to_string(),
                line: 3
            })
        );
    }

    #[test]
    fn an_empty_contract_is_a_violation_not_a_style_choice() {
        let markdown = GOOD.replace("directory -> summary", "   ");
        let table = Table::parse(&markdown).expect("parses");
        assert_eq!(
            verify(&table, &tree(), 2),
            Err(Drift::Contractless {
                symbol: "detect".to_string(),
                line: 3
            })
        );
    }

    #[test]
    fn an_empty_table_checks_nothing_and_must_not_pass() {
        let table = Table::parse("no rows here\n").expect("parses");
        assert!(table.is_empty());
        assert_eq!(verify(&table, &tree(), 0), Err(Drift::Empty));
    }

    #[test]
    fn one_row_is_a_slogan() {
        let only_one = "| module | entry point | contract |\n|---|---|---|\n\
                         | `crates/core/src/detect.rs` | `detect` | directory -> summary |\n";
        let table = Table::parse(only_one).expect("parses");
        assert_eq!(
            verify(&table, &tree(), 2),
            Err(Drift::Vacuous { found: 1, floor: 2 })
        );
    }

    #[test]
    fn a_module_that_is_not_in_the_tree_is_its_own_failure() {
        let markdown = GOOD.replace("crates/core/src/build.rs", "crates/core/src/gone.rs");
        let table = Table::parse(&markdown).expect("parses");
        assert_eq!(
            verify(&table, &tree(), 2),
            Err(Drift::UnknownModule {
                module: "crates/core/src/gone.rs".to_string(),
                line: 4
            })
        );
    }

    #[test]
    fn paths_and_signatures_are_decoration_around_the_name() {
        assert_eq!(first_token("`crates/a/b.py`").as_deref(), Some("b.py"));
        assert_eq!(first_token("`run(x, y)`").as_deref(), Some("run"));
        assert_eq!(first_token("pkg::core::stage").as_deref(), Some("stage"));
        assert_eq!(first_token(""), None);
        assert_eq!(first_token("`-`"), None);
    }

    #[test]
    fn a_row_with_too_few_columns_is_a_parse_failure_not_a_skip() {
        let broken = "| a | b |\n";
        assert_eq!(
            Table::parse(broken),
            Err(ParseError::WrongColumnCount { line: 1, found: 2 })
        );
    }

    #[test]
    fn declaration_keywords_cover_more_than_one_language() {
        let t = tree();
        assert!(t.declares("crates/core/src/build.rs", "build"));
        assert!(t.declares("crates/core/src/detect.rs", "Summary"));
        assert!(!t.declares("crates/core/src/build.rs", "helper_missing"));
        let py = SourceTree::from_pairs(&[(
            "lib/cluster.py".to_string(),
                "def cluster(g):\n    return {}\n\nclass Graph:\n    pass\n".to_string(),
        )]);
        assert!(py.declares("lib/cluster.py", "cluster"));
        assert!(py.declares("lib/cluster.py", "Graph"));
    }

    #[test]
    fn the_leading_dot_slash_is_not_a_second_file() {
        let t = tree();
        assert!(t.has_file("./crates/core/src/build.rs"));
        assert!(t.declares("./crates/core/src/build.rs", "build"));
    }
}
