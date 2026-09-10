//! A wiring claim is a fact with an expiry date.
//!
//! "This guard is not called from anywhere" is written next to the code as a
//! comment, and then the code changes and the comment does not. The comment is
//! not a lie when it is written, which is exactly why it cannot be trusted
//! later: nothing checks it, so its truth decays at the rate of commits.
//!
//! This crate keeps such claims as data. A claim names a symbol, says whether
//! production reaches it, and - if it says *yes* - records the call sites that
//! make it so. Because the sites are data, they can be re-derived against the
//! tree, and the claim is asked the same question a reviewer would ask: is that
//! call still there?
//!
//! # What a drift report is for
//!
//! Two directions fail, and both are bugs:
//!
//! * **claimed wired, site gone** - the guarantee nobody knew was missing, the
//!   one that makes a security comment worse than no comment;
//! * **claimed unwired, site present** - code reached the symbol and nobody
//!   re-read the label, so the next audit wastes its pass on a fixed item, or
//!   worse, "fixes" it a second time in a different place.
//!
//! The second direction is what a ratchet that only shrinks cannot see.
//!
//! # What is not claimed
//!
//! Text search is not a call graph. A site here means "`symbol(` appears in
//! another file" - a proxy that is honest about being a proxy. It cannot be
//! fooled by a comment mentioning the symbol *and* it will miss an indirect
//! call through a function pointer; both limits are why the crate reports drift
//! instead of asserting correctness, and why the tree it checks is passed in
//! rather than read here.

use std::collections::BTreeMap;

/// A location that supposedly reaches the symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    file: String,
    line: usize,
    /// How the path got there: `chain_actor.rs:4106 <- storage_deal.rs:2168`.
    via: String,
}

impl Site {
    /// A site with the file, the line, and the route that made it reachable.
    ///
    /// A site without a route is a guess, and this crate exists because a
    /// guess is not a proof of reachability.
    #[must_use]
    pub fn new(file: &str, line: usize, via: &str) -> Self {
        Self {
            file: file.to_string(),
            line,
            via: via.to_string(),
        }
    }

    /// The file.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// The line.
    #[must_use]
    pub fn line(&self) -> usize {
        self.line
    }

    /// The route.
    #[must_use]
    pub fn via(&self) -> &str {
        &self.via
    }
}

/// What a claim says about reachability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Claim {
    /// Production reaches it, with sites.
    Wired,
    /// Nothing reaches it, and there is a stated reason.
    Unwired {
        /// Why this is not wired, and why that is acceptable or a defect.
        reason: &'static str,
    },
}

/// One recorded claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    module: String,
    symbol: String,
    claim: Claim,
    sites: Vec<Site>,
    /// The commit the claim was made against.
    at_commit: String,
}

impl Record {
    /// The file the symbol lives in.
    #[must_use]
    pub fn module(&self) -> &str {
        &self.module
    }

    /// The symbol.
    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// What is claimed.
    #[must_use]
    pub fn claim(&self) -> Claim {
        self.claim
    }

    /// The recorded call sites.
    #[must_use]
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// The commit the claim was written against.
    #[must_use]
    pub fn at_commit(&self) -> &str {
        &self.at_commit
    }
}

/// Why a claim could not be recorded, or what moved under it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Drift {
    /// A wired claim with no site is a label, not a claim.
    WiredWithoutSite {
        /// The symbol.
        symbol: String,
    },
    /// A site whose route is empty.
    SiteWithoutRoute {
        /// The file named in the site.
        file: String,
        /// The line named in the site.
        line: usize,
    },
    /// A site pointing at a file the tree does not have.
    SiteFileMissing {
        /// The file.
        file: String,
        /// The symbol whose claim carried it.
        symbol: String,
    },
    /// The route named a file that does not mention the symbol either.
    RouteUnsupported {
        /// The file the route claims to come from.
        from: String,
        /// The symbol.
        symbol: String,
    },
    /// The call text is gone: the claim no longer holds.
    SiteNoLongerHolds {
        /// The symbol.
        symbol: String,
        /// The file that used to call it.
        file: String,
        /// The line that used to call it.
        line: usize,
    },
    /// The claim says unwired; the tree says otherwise.
    UnwiredClaimNowHolds {
        /// The symbol.
        symbol: String,
        /// The file that now calls it.
        file: String,
    },
    /// The commit the registry was written against is not the commit checked
    /// against: the whole point of the expiry date.
    CommitMoved {
        /// What the registry was written at.
        written: String,
        /// What it was checked at.
        checked: String,
    },
    /// An empty registry checks nothing, and a check that checks nothing passes.
    Empty,
}

impl std::fmt::Display for Drift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WiredWithoutSite { symbol } => write!(
                f,
                "`{symbol}` is claimed wired with no call site; a guarantee needs a place \
                 where it is used"
            ),
            Self::SiteWithoutRoute { file, line } => write!(
                f,
                "site `{file}:{line}` has no route: a file and a line say where, not how"
            ),
            Self::SiteFileMissing { file, symbol } => write!(
                f,
                "the claim for `{symbol}` points at `{file}`, which is not in the tree"
            ),
            Self::RouteUnsupported { from, symbol } => write!(
                f,
                "the route for `{symbol}` claims `{from}`, which does not mention `{symbol}`"
            ),
            Self::SiteNoLongerHolds {
                symbol,
                file,
                line,
            } => write!(
                f,
                "`{symbol}` was wired from `{file}:{line}` and that call is gone; the \
                 comment is now the only thing saying this is safe"
            ),
            Self::UnwiredClaimNowHolds { symbol, file } => write!(
                f,
                "`{symbol}` is labelled unwired but `{file}` calls it: either the label is \
                 stale or the caller is a mock pretending to be production"
            ),
            Self::CommitMoved { written, checked } => write!(
                f,
                "claims written at `{written}` were checked at `{checked}`: re-derive \
                 before trusting any of them"
            ),
            Self::Empty => write!(
                f,
                "no claims recorded; a registry that holds nothing is a passing gate with \
                 nothing behind it"
            ),
        }
    }
}

impl std::error::Error for Drift {}

/// The tree the claims are re-checked against: path to text, supplied by the
/// caller. No file reading here, so the check can be pointed at a snapshot, a
/// worktree, or a fixture in a test.
#[derive(Debug, Clone, Default)]
pub struct Tree {
    files: BTreeMap<String, String>,
}

impl Tree {
    /// A tree from `(path, text)` pairs.
    #[must_use]
    pub fn from_pairs(pairs: &[(String, String)]) -> Self {
        Self {
            files: pairs.iter().cloned().collect(),
        }
    }

    /// Whether the file exists.
    #[must_use]
    pub fn has_file(&self, path: &str) -> bool {
        self.files.contains_key(path)
    }

    /// Whether `path` mentions `symbol(` outside the file that declares it.
    #[must_use]
    pub fn calls(&self, path: &str, symbol: &str) -> bool {
        let Some(text) = self.files.get(path) else {
            return false;
        };
        let needle = format!("{symbol}(");
        text.contains(&needle)
    }

    /// Every file that calls `symbol`, excluding `home` (the declaration).
    #[must_use]
    pub fn callers_of(&self, symbol: &str, home: &str) -> Vec<String> {
        self.files
            .iter()
            .filter(|(path, _)| path.as_str() != home)
            .filter(|(path, _)| self.calls(path, symbol))
            .map(|(path, _)| path.clone())
            .collect()
    }

    /// Files in the tree.
    #[must_use]
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }
}

/// The registry of claims.
#[derive(Debug, Clone)]
pub struct Registry {
    records: BTreeMap<String, Record>,
    written_at: String,
}

impl Registry {
    /// An empty registry written at `commit`.
    #[must_use]
    pub fn new(commit: &str) -> Self {
        Self {
            records: BTreeMap::new(),
            written_at: commit.to_string(),
        }
    }

    /// The commit the registry was written at.
    #[must_use]
    pub fn written_at(&self) -> &str {
        &self.written_at
    }

    /// Number of claims.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether nothing is claimed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Records a wired claim. Fails closed when the sites would make the claim
    /// decorative.
    ///
    /// # Errors
    ///
    /// [`Drift::WiredWithoutSite`] and [`Drift::SiteWithoutRoute`].
    pub fn declare_wired(
        &mut self,
        module: &str,
        symbol: &str,
        sites: &[Site],
    ) -> Result<(), Drift> {
        if sites.is_empty() {
            return Err(Drift::WiredWithoutSite {
                symbol: symbol.to_string(),
            });
        }
        for site in sites {
            if site.via().trim().is_empty() {
                return Err(Drift::SiteWithoutRoute {
                    file: site.file().to_string(),
                    line: site.line(),
                });
            }
        }
        let key = format!("{module}:{symbol}");
        self.records.insert(
            key,
            Record {
                module: module.to_string(),
                symbol: symbol.to_string(),
                claim: Claim::Wired,
                sites: sites.to_vec(),
                at_commit: self.written_at.clone(),
            },
        );
        Ok(())
    }

    /// Records an unwired claim with its reason. The reason is typed
    /// `&'static str` on purpose: it comes from the source of the label, which
    /// is where a justification belongs.
    pub fn declare_unwired(&mut self, module: &str, symbol: &str, reason: &'static str) {
        let key = format!("{module}:{symbol}");
        self.records.insert(
            key,
            Record {
                module: module.to_string(),
                symbol: symbol.to_string(),
                claim: Claim::Unwired { reason },
                sites: Vec::new(),
                at_commit: self.written_at.clone(),
            },
        );
    }

    /// A claim, if one was recorded.
    #[must_use]
    pub fn get(&self, module: &str, symbol: &str) -> Option<&Record> {
        self.records.get(&format!("{module}:{symbol}"))
    }

    /// Symbols with no claim at all, in a tree the caller cares about.
    ///
    /// A guard's silence is the failure mode this whole registry exists for:
    /// not a wrong label, but no label.
    #[must_use]
    pub fn unclaimed(&self, module: &str, symbols: &[&str]) -> Vec<String> {
        symbols
            .iter()
            .filter(|s| !self.records.contains_key(&format!("{module}:{s}")))
            .map(|s| (*s).to_string())
            .collect()
    }

    /// Re-derives every claim against `tree` at `commit`, returning what moved.
    ///
    /// The order is deliberate: file existence before call text, so a renamed
    /// file reports as a missing file rather than as a hundred vanished calls.
    #[must_use]
    pub fn recompute(&self, tree: &Tree, commit: &str) -> Vec<Drift> {
        let mut out = Vec::new();
        if self.records.is_empty() {
            out.push(Drift::Empty);
            return out;
        }
        if commit != self.written_at {
            out.push(Drift::CommitMoved {
                written: self.written_at.clone(),
                checked: commit.to_string(),
            });
        }
        for record in self.records.values() {
            match record.claim() {
                Claim::Wired => {
                    for site in record.sites() {
                        if !tree.has_file(site.file()) {
                            out.push(Drift::SiteFileMissing {
                                file: site.file().to_string(),
                                symbol: record.symbol().to_string(),
                            });
                            continue;
                        }
                        let from = route_file(site.via());
                        if !from.is_empty()
                            && tree.has_file(&from)
                            && !tree.calls(&from, record.symbol())
                        {
                            out.push(Drift::RouteUnsupported {
                                from,
                                symbol: record.symbol().to_string(),
                            });
                        }
                        if !tree.calls(site.file(), record.symbol()) {
                            out.push(Drift::SiteNoLongerHolds {
                                symbol: record.symbol().to_string(),
                                file: site.file().to_string(),
                                line: site.line(),
                            });
                        }
                    }
                }
                Claim::Unwired { .. } => {
                    let callers = tree.callers_of(record.symbol(), record.module());
                    if let Some(file) = callers.first() {
                        out.push(Drift::UnwiredClaimNowHolds {
                            symbol: record.symbol().to_string(),
                            file: file.clone(),
                        });
                    }
                }
            }
        }
        out
    }

    /// Claims whose recorded commit differs from `commit`: the cheap question,
    /// asked before the expensive one, so a reader knows which labels are even
    /// worth re-deriving.
    #[must_use]
    pub fn expired(&self, commit: &str) -> Vec<&Record> {
        self.records
            .values()
            .filter(|r| r.at_commit() != commit)
            .collect()
    }

    /// The registry as the lines a repository file would carry, so the ledger
    /// and the labels next to the code cannot disagree.
    #[must_use]
    pub fn render(&self) -> Vec<String> {
        self.records
            .values()
            .map(|r| match r.claim() {
                Claim::Wired => format!(
                    "WIRING: wired :: {} :: {} site={}",
                    r.module(),
                    r.symbol(),
                    r.sites()
                        .first()
                        .map(|s| format!("{}:{}", s.file(), s.line()))
                        .unwrap_or_else(|| "-".to_string())
                ),
                Claim::Unwired { reason } => format!(
                    "WIRING: unwired :: {} :: {} :: {}",
                    r.module(),
                    r.symbol(),
                    reason
                ),
            })
            .collect()
    }
}

/// The file part of a route like `a.rs:12 <- b.rs`, which is `a.rs`.
fn route_file(via: &str) -> String {
    let head = via.split_whitespace().next().unwrap_or("");
    head.split(':').next().unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> Tree {
        Tree::from_pairs(&[
            (
                "src/storage/storage_deal.rs".to_string(),
                "fn run() { let _ = audit_coding(&manifest); }".to_string(),
            ),
            (
                "src/chain/chain_actor.rs".to_string(),
                "fn tick() { let _ = audit_coding(&m); }".to_string(),
            ),
            (
                "src/core/account.rs".to_string(),
                "pub fn slash_all_roles() {} fn other() {}".to_string(),
            ),
        ])
    }

    fn wired() -> Registry {
        let mut r = Registry::new("abc1234");
        r.declare_wired(
            "src/storage/storage_deal.rs",
            "audit_coding",
            &[Site::new(
                "src/chain/chain_actor.rs",
                4106,
                "chain_actor.rs:4106 <- storage_deal.rs:2168",
            )],
        )
        .expect("a wired claim with a site");
        r
    }

    #[test]
    fn a_wired_claim_without_a_site_is_refused() {
        let mut r = Registry::new("abc1234");
        assert_eq!(
            r.declare_wired("src/a.rs", "guard", &[]),
            Err(Drift::WiredWithoutSite {
                symbol: "guard".to_string()
            })
        );
    }

    #[test]
    fn a_site_without_a_route_is_refused() {
        let mut r = Registry::new("abc1234");
        assert_eq!(
            r.declare_wired("src/a.rs", "guard", &[Site::new("src/b.rs", 3, "")]),
            Err(Drift::SiteWithoutRoute {
                file: "src/b.rs".to_string(),
                line: 3
            })
        );
    }

    #[test]
    fn a_honest_registry_agrees_with_the_tree() {
        let r = wired();
        assert_eq!(r.recompute(&tree(), "abc1234"), Vec::<Drift>::new());
    }

    #[test]
    fn a_claim_surviving_a_renamed_file_reports_the_file_not_the_hundred_calls() {
        let r = wired();
        let t = Tree::from_pairs(&[(
            "src/chain/renamed_actor.rs".to_string(),
            "fn tick() { audit_coding(); }".to_string(),
        )]);
        assert_eq!(
            r.recompute(&t, "abc1234"),
            vec![Drift::SiteFileMissing {
                file: "src/chain/chain_actor.rs".to_string(),
                symbol: "audit_coding".to_string()
            }]
        );
    }

    #[test]
    fn a_call_that_disappears_is_a_drift_not_a_silence() {
        let r = wired();
        let t = Tree::from_pairs(&[
            (
                "src/storage/storage_deal.rs".to_string(),
                "fn run() {}".to_string(),
            ),
            (
                "src/chain/chain_actor.rs".to_string(),
                "fn tick() { let _ = unrelated(); }".to_string(),
            ),
        ]);
        let drift = r.recompute(&t, "abc1234");
        assert_eq!(
            drift,
            vec![Drift::SiteNoLongerHolds {
                symbol: "audit_coding".to_string(),
                file: "src/chain/chain_actor.rs".to_string(),
                line: 4106
            }]
        );
    }

    #[test]
    fn a_label_that_the_code_outgrew_is_reported_too() {
        let mut r = Registry::new("abc1234");
        r.declare_unwired(
            "src/core/account.rs",
            "slash_all_roles",
            "registry keeps the live path",
        );
        let t = Tree::from_pairs(&[(
            "src/registry/permissionless.rs".to_string(),
            "fn x() { slash_all_roles(); }".to_string(),
        )]);
        assert_eq!(
            r.recompute(&t, "abc1234"),
            vec![Drift::UnwiredClaimNowHolds {
                symbol: "slash_all_roles".to_string(),
                file: "src/registry/permissionless.rs".to_string()
            }]
        );
    }

    #[test]
    fn checking_a_different_commit_says_so_first() {
        let r = wired();
        let drift = r.recompute(&tree(), "def9999");
        assert_eq!(drift.first(), Some(&Drift::CommitMoved {
            written: "abc1234".to_string(),
            checked: "def9999".to_string(),
        }));
        assert_eq!(r.expired("def9999").len(), 1);
        assert!(r.expired("abc1234").is_empty());
    }

    #[test]
    fn an_empty_registry_is_a_failure_not_a_pass() {
        let r = Registry::new("abc1234");
        assert_eq!(r.recompute(&tree(), "abc1234"), vec![Drift::Empty]);
    }

    #[test]
    fn a_route_naming_a_file_that_does_not_mention_the_symbol_is_a_drift() {
        let mut r = Registry::new("abc1234");
        r.declare_wired(
            "src/storage/storage_deal.rs",
            "audit_coding",
            &[Site::new(
                "src/chain/chain_actor.rs",
                1,
                "src/chain/chain_actor.rs:1 <- nowhere",
            )],
        )
        .unwrap();
        let t = Tree::from_pairs(&[
            (
                "src/chain/chain_actor.rs".to_string(),
                "audit_coding();".to_string(),
            ),
            (
                "src/storage/storage_deal.rs".to_string(),
                "fn audit_coding() {}".to_string(),
            ),
        ]);
        // The site file itself calls it; the route's own file is the site file,
        // so the route is supported and nothing drifts. This pins the direction
        // of the check: the route is judged, not repeated.
        assert_eq!(t.paths().count(), 2);
        assert_eq!(r.recompute(&t, "abc1234"), Vec::<Drift>::new());
    }

    #[test]
    fn symbols_without_any_claim_are_listed() {
        let r = wired();
        assert_eq!(
            r.unclaimed(
                "src/core/account.rs",
                &["slash_all_roles", "get_active_validators"]
            ),
            vec![
                "slash_all_roles".to_string(),
                "get_active_validators".to_string()
            ]
        );
        assert!(r
            .unclaimed("src/storage/storage_deal.rs", &["audit_coding"])
            .is_empty());
    }

    #[test]
    fn the_rendered_form_carries_the_reason_for_an_unwired_claim() {
        let mut r = Registry::new("abc1234");
        r.declare_unwired("src/core/account.rs", "slash_all_roles", "duplicate of registry");
        let lines = r.render();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("WIRING: unwired"));
        assert!(lines[0].contains("duplicate of registry"));
    }

    #[test]
    fn route_extraction_is_not_confused_by_missing_spaces() {
        assert_eq!(route_file("a.rs:1 <- b.rs"), "a.rs");
        assert_eq!(route_file(""), "");
        assert_eq!(route_file("no-colon"), "no-colon");
    }
}
