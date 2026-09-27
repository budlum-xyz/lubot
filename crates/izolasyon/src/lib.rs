//! The isolation boundary for one Lubot agent session.
//!
//! # What this models
//!
//! The isolation boundary *is* the environment, not a check performed on it. A
//! session opens from an **empty workspace**, under a **session-scoped identity**
//! that has never been used before, with **exactly one contract** in force, and
//! its results leave only as **copies** - the workspace itself is discarded when
//! the session ends.
//!
//! This crate does not run a sandbox. The filesystem binding lives in the node.
//! What is here is the boundary as a **checkable contract**: the invariants, the
//! order they must hold in, and the refusals when they do not. Modelling the
//! boundary separately from enforcing it is deliberate - a rule that only exists
//! inside a runtime cannot be tested without the runtime, and a rule nobody has
//! tested is a rule nobody has checked.
//!
//! # The four invariants, and what breaks without each
//!
//! **1. The workspace is empty when the session opens.** A session that inherits
//! files inherits whatever the previous occupant left, including its secrets and
//! its assumptions. "Empty" is checked, not assumed.
//!
//! **2. The identity is fresh.** This is the one most often relaxed, because a
//! stable identity is convenient. A reused identity means two sessions share
//! state that neither of them chose to share - which is not a leak in the usual
//! sense and is therefore much harder to notice. Freshness is a property of the
//! identity, checked against a register of identities already issued, not a
//! property of the caller's claim.
//!
//! **3. Exactly one contract is in force.** The type enforces this: [`Session`]
//! holds a single [`Contract`] field, not a collection. A session running two
//! contracts has to decide which one governs a given action, and every such
//! decision is a place for the stricter one to lose.
//!
//! **4. Results leave as copies and the workspace is discarded.** A session that
//! hands out references to its own workspace keeps the workspace alive, and with
//! it everything the session touched.
//!
//! # What this cannot guarantee
//!
//! Stated rather than papered over: a checkable contract cannot stop a session
//! from doing something the contract does not describe. It can refuse to open a
//! session that starts dirty, refuse to run one with no contract, and refuse to
//! resume one that has ended. It cannot observe what happens between those
//! points. That observation is the sandbox's job, and this crate would be
//! lying if it claimed otherwise.

use std::collections::BTreeSet;

/// Why a session boundary refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsolationError {
    /// The workspace is not empty when the session opens.
    NonEmptyWorkspace { count: usize },
    /// The identity has been issued before. A reused identity means two sessions
    /// share state neither chose to share.
    IdentityReused { identity: String },
    /// The identity is empty. It must be minted, not left blank and filled in
    /// later.
    EmptyIdentity,
    /// A closed session was asked to continue.
    SessionClosed { identity: String },
    /// An output was requested that the contract does not declare.
    OutputNotDeclared { name: String },
    /// The contract is empty. A session runs exactly one, non-empty contract.
    EmptyContract,
    /// A resource bound was exceeded.
    BoundExceeded {
        what: &'static str,
        limit: u64,
        got: u64,
    },
}

impl std::fmt::Display for IsolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonEmptyWorkspace { count } => {
                write!(
                    f,
                    "the workspace is not empty at session start ({count} entries)"
                )
            }
            Self::IdentityReused { identity } => {
                write!(
                    f,
                    "the identity {identity:?} has been issued before; a session needs a fresh one"
                )
            }
            Self::EmptyIdentity => write!(f, "the session identity is empty"),
            Self::SessionClosed { identity } => {
                write!(
                    f,
                    "the session {identity:?} has ended and cannot be resumed"
                )
            }
            Self::OutputNotDeclared { name } => {
                write!(
                    f,
                    "{name:?} is not a declared output of this session's contract"
                )
            }
            Self::EmptyContract => write!(
                f,
                "the contract is empty; a session runs exactly one, non-empty contract"
            ),
            Self::BoundExceeded { what, limit, got } => {
                write!(f, "{what} reached {got}, the bound is {limit}")
            }
        }
    }
}

/// The one contract in force for a session.
///
/// A single field holding a set of declared outputs, rather than a list of
/// contracts, so that the type itself enforces "exactly one". A session that
/// could hold two would have to decide which governs each action, and every such
/// decision is a place for the stricter one to lose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contract {
    /// What the session is allowed to hand back. Anything else is refused at
    /// [`Session::release`].
    pub declared_outputs: BTreeSet<String>,
    /// How many outputs at most. Bounded because an unbounded output set is an
    /// unbounded exfiltration channel with a declaration step bolted onto it.
    pub max_outputs: u64,
    /// How many bytes at most may leave. Same reasoning.
    pub max_release_bytes: u64,
}

impl Contract {
    /// A contract with `declared_outputs`.
    ///
    /// # Errors
    ///
    /// [`IsolationError::EmptyContract`] when the set is empty.
    pub fn new(
        declared_outputs: &[&str],
        max_outputs: u64,
        max_release_bytes: u64,
    ) -> Result<Self, IsolationError> {
        if declared_outputs.is_empty() {
            return Err(IsolationError::EmptyContract);
        }
        let mut set = BTreeSet::new();
        for name in declared_outputs {
            set.insert((*name).to_string());
        }
        Ok(Self {
            declared_outputs: set,
            max_outputs,
            max_release_bytes,
        })
    }
}

/// Where a session is in its lifecycle. Ordered: it only moves forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// Constructed but not opened. Nothing has been checked yet.
    Constructed,
    /// Open: the workspace was verified empty, the identity was verified fresh,
    /// and the contract was verified non-empty.
    Open,
    /// Ended. The workspace is discarded and the identity is spent.
    Closed,
}

/// One session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub identity: String,
    /// The single contract. Not a collection, on purpose.
    pub contract: Contract,
    pub phase: Phase,
    /// How many entries the workspace held at open. Recorded so a session that
    /// opened dirty is auditable even after the fact.
    pub opened_with_entries: usize,
    /// Bytes released so far.
    pub released_bytes: u64,
    /// Outputs released so far.
    pub released_outputs: BTreeSet<String>,
}

impl Session {
    /// Opens a session.
    ///
    /// Every check runs before anything is constructed, so a refused open leaves
    /// no half-built session behind.
    ///
    /// # Errors
    ///
    /// [`IsolationError::NonEmptyWorkspace`], [`IsolationError::EmptyIdentity`],
    /// [`IsolationError::IdentityReused`].
    pub fn open(
        identity: &str,
        contract: Contract,
        workspace_entries: usize,
        issued_identities: &BTreeSet<String>,
    ) -> Result<Self, IsolationError> {
        if identity.is_empty() {
            return Err(IsolationError::EmptyIdentity);
        }
        if workspace_entries > 0 {
            return Err(IsolationError::NonEmptyWorkspace {
                count: workspace_entries,
            });
        }
        if issued_identities.contains(identity) {
            return Err(IsolationError::IdentityReused {
                identity: identity.to_string(),
            });
        }
        Ok(Self {
            identity: identity.to_string(),
            contract,
            phase: Phase::Open,
            opened_with_entries: workspace_entries,
            released_bytes: 0,
            released_outputs: BTreeSet::new(),
        })
    }

    /// Releases one output as a copy.
    ///
    /// Returns the bytes rather than a reference into the workspace, which is the
    /// whole point: a session that hands out references keeps its workspace
    /// alive, and with it everything the session touched.
    ///
    /// # Errors
    ///
    /// [`IsolationError::SessionClosed`], [`IsolationError::OutputNotDeclared`],
    /// or [`IsolationError::BoundExceeded`].
    pub fn release(&mut self, name: &str, bytes: &[u8]) -> Result<Vec<u8>, IsolationError> {
        if self.phase == Phase::Closed {
            return Err(IsolationError::SessionClosed {
                identity: self.identity.clone(),
            });
        }
        if !self.contract.declared_outputs.contains(name) {
            return Err(IsolationError::OutputNotDeclared {
                name: name.to_string(),
            });
        }
        let total = self.released_bytes.saturating_add(bytes.len() as u64);
        if total > self.contract.max_release_bytes {
            return Err(IsolationError::BoundExceeded {
                what: "released bytes",
                limit: self.contract.max_release_bytes,
                got: total,
            });
        }
        let outputs = self.released_outputs.len().saturating_add(1) as u64;
        if outputs > self.contract.max_outputs {
            return Err(IsolationError::BoundExceeded {
                what: "released outputs",
                limit: self.contract.max_outputs,
                got: outputs,
            });
        }
        self.released_bytes = total;
        self.released_outputs.insert(name.to_string());
        Ok(bytes.to_vec())
    }

    /// Ends the session. Idempotent: ending twice is not an error, because the
    /// caller cannot always tell whether it already ended, and refusing the
    /// second call would push that uncertainty into the cleanup path where it is
    /// hardest to handle.
    pub fn close(&mut self) {
        self.phase = Phase::Closed;
        // The workspace is discarded. There is nothing to discard in this model -
        // the session never held one - but the phase is what the node checks
        // before touching the directory again.
        self.opened_with_entries = 0;
    }

    /// Whether the session may still act.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.phase == Phase::Open
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> Contract {
        Contract::new(&["answer", "sources"], 4, 65_536).expect("contract")
    }

    #[test]
    fn a_session_opens_from_an_empty_workspace() {
        let s = Session::open("s-001", contract(), 0, &BTreeSet::new()).expect("open");
        assert!(s.is_open());
        assert_eq!(s.opened_with_entries, 0);
    }

    #[test]
    fn a_dirty_workspace_refuses_the_open() {
        // A session that inherits files inherits the previous occupant's secrets
        // and assumptions.
        assert_eq!(
            Session::open("s-001", contract(), 3, &BTreeSet::new()),
            Err(IsolationError::NonEmptyWorkspace { count: 3 })
        );
    }

    #[test]
    fn a_reused_identity_is_refused() {
        // The one most often relaxed, because a stable identity is convenient. A
        // reused identity means two sessions share state neither chose to share.
        let mut issued = BTreeSet::new();
        issued.insert("s-001".to_string());
        assert_eq!(
            Session::open("s-001", contract(), 0, &issued),
            Err(IsolationError::IdentityReused {
                identity: "s-001".to_string()
            })
        );
    }

    #[test]
    fn an_empty_identity_is_refused() {
        // It must be minted, not left blank and filled in later.
        assert_eq!(
            Session::open("", contract(), 0, &BTreeSet::new()),
            Err(IsolationError::EmptyIdentity)
        );
    }

    #[test]
    fn an_empty_contract_is_refused() {
        assert_eq!(
            Contract::new(&[], 4, 1024),
            Err(IsolationError::EmptyContract)
        );
    }

    #[test]
    fn only_declared_outputs_leave() {
        let mut s = Session::open("s-001", contract(), 0, &BTreeSet::new()).expect("open");
        assert!(s.release("answer", b"42").is_ok());
        assert_eq!(
            s.release("private_notes", b"x"),
            Err(IsolationError::OutputNotDeclared {
                name: "private_notes".to_string()
            }),
            "an undeclared output left the session"
        );
    }

    #[test]
    fn the_release_bound_is_enforced() {
        // An unbounded output set is an unbounded exfiltration channel with a
        // declaration step bolted onto it.
        let c = Contract::new(&["answer"], 4, 8).expect("contract");
        let mut s = Session::open("s-001", c, 0, &BTreeSet::new()).expect("open");
        assert!(s.release("answer", b"12345678").is_ok());
        assert!(matches!(
            s.release("answer", b"9"),
            Err(IsolationError::BoundExceeded {
                what: "released bytes",
                ..
            })
        ));
    }

    #[test]
    fn a_closed_session_cannot_release() {
        let mut s = Session::open("s-001", contract(), 0, &BTreeSet::new()).expect("open");
        s.close();
        assert!(!s.is_open());
        assert_eq!(
            s.release("answer", b"42"),
            Err(IsolationError::SessionClosed {
                identity: "s-001".to_string()
            })
        );
    }

    #[test]
    fn closing_twice_is_not_an_error() {
        // The caller cannot always tell whether it already ended, and refusing
        // the second call would push that uncertainty into the cleanup path where
        // it is hardest to handle.
        let mut s = Session::open("s-001", contract(), 0, &BTreeSet::new()).expect("open");
        s.close();
        s.close();
        assert_eq!(s.phase, Phase::Closed);
    }

    #[test]
    fn a_release_returns_a_copy_not_a_reference_into_the_workspace() {
        // A session that hands out references keeps its workspace alive, and with
        // it everything the session touched.
        let mut s = Session::open("s-001", contract(), 0, &BTreeSet::new()).expect("open");
        let out = s.release("answer", b"42").expect("release");
        assert_eq!(out, b"42".to_vec());
        // The returned value owns its bytes: mutating it cannot reach back.
        let mut owned = out;
        owned.push(b'!');
        assert_eq!(
            s.released_bytes, 2,
            "the session's accounting moved with the copy"
        );
    }

    #[test]
    fn the_phase_only_moves_forward() {
        assert!(Phase::Constructed < Phase::Open);
        assert!(Phase::Open < Phase::Closed);
    }

    #[test]
    fn a_refused_open_leaves_no_session_behind() {
        // Every check runs before anything is constructed.
        let result = Session::open("s-001", contract(), 5, &BTreeSet::new());
        assert!(result.is_err());
    }
}
