//! Isolation boundary for a Lubot agent session.
//!
//! Adapted (as an approach, not a runtime) from sandbox isolation: the
//! isolation boundary *is* the environment. A session opens from an **empty
//! workspace** under a **session-scoped identity**, with **exactly one
//! contract** in force (a single field, so the type itself enforces "one"), and
//! results leave only as **copies** (the workspace is discarded at session end).
//!
//! This crate models that boundary as a *checkable contract* — what it accepts,
//! it can check — rather than running a sandbox. The filesystem binding lives in
//! the node; here the boundary is a pure, deterministic invariant.

use std::collections::BTreeSet;

/// Why a session boundary fails an isolation invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsolationError {
    /// The workspace is not empty at session start (it must begin clean).
    NonEmptyWorkspace { count: usize },
    /// No contract is in force (a session runs exactly one, non-empty contract).
    NoContract,
    /// The session identity is empty (it must be freshly minted, not reused).
    EmptyIdentity,
}

impl std::fmt::Display for IsolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonEmptyWorkspace { count } => {
                write!(f, "workspace not empty at session start ({count} entries)")
            }
            Self::NoContract => write!(f, "no contract is in force"),
            Self::EmptyIdentity => write!(f, "session identity is empty"),
        }
    }
}

impl std::error::Error for IsolationError {}

/// A session-scoped identity: it exists only for the session and is not
/// persisted or reused across sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionIdentity {
    /// A non-empty, session-unique token.
    pub token: String,
}

/// The entries present in the workspace at session start, modelled as an
/// abstract set so the boundary is a pure, checkable contract (no filesystem).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Workspace {
    entries: BTreeSet<String>,
}

impl Workspace {
    /// An empty workspace — the required session-start state.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            entries: BTreeSet::new(),
        }
    }

    /// A workspace pre-populated with the given entries (a violated boundary).
    #[must_use]
    pub fn with_entries<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            entries: entries.into_iter().map(Into::into).collect(),
        }
    }

    /// Whether the workspace is clean (no pre-existing entries).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The number of pre-existing entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The isolation boundary for one session: an empty workspace, a session-scoped
/// identity, and exactly one contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionBoundary {
    /// The session-scoped identity.
    pub identity: SessionIdentity,
    /// The single in-force contract (one field enforces "exactly one").
    pub contract: String,
    /// The (empty) workspace the session opened from.
    pub workspace: Workspace,
}

impl SessionBoundary {
    /// Open a session boundary, enforcing every isolation invariant.
    ///
    /// # Errors
    ///
    /// [`IsolationError::EmptyIdentity`] for a missing identity,
    /// [`IsolationError::NonEmptyWorkspace`] when the workspace is not empty, or
    /// [`IsolationError::NoContract`] when the contract is empty.
    pub fn open(
        identity: SessionIdentity,
        contract: String,
        workspace: Workspace,
    ) -> Result<Self, IsolationError> {
        if identity.token.is_empty() {
            return Err(IsolationError::EmptyIdentity);
        }
        if !workspace.is_empty() {
            return Err(IsolationError::NonEmptyWorkspace {
                count: workspace.len(),
            });
        }
        if contract.is_empty() {
            return Err(IsolationError::NoContract);
        }
        Ok(Self {
            identity,
            contract,
            workspace,
        })
    }

    /// Re-verify the invariants (a canary target for the repository gate).
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn verify(&self) -> Result<(), IsolationError> {
        if self.identity.token.is_empty() {
            return Err(IsolationError::EmptyIdentity);
        }
        if !self.workspace.is_empty() {
            return Err(IsolationError::NonEmptyWorkspace {
                count: self.workspace.len(),
            });
        }
        if self.contract.is_empty() {
            return Err(IsolationError::NoContract);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> SessionIdentity {
        SessionIdentity {
            token: "sess-01".to_owned(),
        }
    }

    #[test]
    fn opens_from_clean_environment() {
        let b = SessionBoundary::open(id(), "contract-a".to_owned(), Workspace::empty())
            .expect("clean session opens");
        b.verify().expect("boundary holds");
        assert_eq!(b.contract, "contract-a");
    }

    #[test]
    fn refuses_non_empty_workspace() {
        let ws = Workspace::with_entries(["pre-existing.txt"]);
        let err = SessionBoundary::open(id(), "contract-a".to_owned(), ws).unwrap_err();
        assert_eq!(err, IsolationError::NonEmptyWorkspace { count: 1 });
    }

    #[test]
    fn refuses_empty_contract_and_identity() {
        assert_eq!(
            SessionBoundary::open(id(), String::new(), Workspace::empty()),
            Err(IsolationError::NoContract)
        );
        let empty_id = SessionIdentity {
            token: String::new(),
        };
        assert_eq!(
            SessionBoundary::open(empty_id, "contract-a".to_owned(), Workspace::empty()),
            Err(IsolationError::EmptyIdentity)
        );
    }

    /// Canary: a mutation of the boundary after `open` must be caught by
    /// `verify`.
    #[test]
    fn verify_catches_mutation() {
        let mut b =
            SessionBoundary::open(id(), "contract-a".to_owned(), Workspace::empty()).expect("open");
        b.workspace = Workspace::with_entries(["leaked.txt"]);
        assert_eq!(
            b.verify(),
            Err(IsolationError::NonEmptyWorkspace { count: 1 })
        );
    }
}
