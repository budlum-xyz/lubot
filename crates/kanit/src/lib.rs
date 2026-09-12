//! Proof records: what was claimed, what proves it, and whether that still holds.
//!
//! # The three things a proof registry gets wrong
//!
//! **1. Treating a proof as a fact rather than a fact about a state.** A proof is
//! made against a state root. The state moves. A proof verified against root `R`
//! is not a proof about root `R'`, and a registry that stores "this claim is
//! proven" without the root it was proven against will happily answer for a
//! state the proof never covered. Every record here carries the root, and
//! [`Ledger::accept`] refuses one presented against a different root.
//!
//! **2. Letting a status regress silently.** A proof that was verified and is
//! now merely "pending" again has lost information, and nobody can tell whether
//! it was re-opened deliberately or overwritten by a bug. Status here only moves
//! along recorded transitions, and every transition carries a reason. There is no
//! path from [`Status::Verified`] back to [`Status::Pending`] at all - not a
//! guarded one, not a logged one. A proof that stops holding **expires**, which
//! is a different state with a different meaning.
//!
//! **3. Conflating expiry with failure.** An expired proof is not a bad proof.
//! It was checked, it held, and the state it described has since moved. Treating
//! the two alike means either retrying proofs that were fine or discarding
//! verdicts that were real. [`Status::Expired`] and [`Status::Rejected`] are
//! separate states and are reported separately.
//!
//! # What this crate does not do
//!
//! It does not verify proofs. It records verdicts that a verifier produced, and
//! answers what the registry believes now. Keeping the two apart means the
//! registry's rules can be tested without a prover, which is the only way they
//! get tested at all.

use std::collections::BTreeMap;

/// Where a proof record stands.
///
/// The transitions are deliberately one-directional. See the module note on why
/// there is no path back to [`Self::Pending`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    /// Recorded, not yet checked.
    Pending,
    /// Checked and held, against the state root the record carries.
    Verified,
    /// Checked and did not hold.
    Rejected,
    /// Held when it was checked, and the state it described has since moved.
    /// Not a failure and not a pending check.
    Expired,
}

impl Status {
    /// Whether this status is a verdict.
    #[must_use]
    pub fn is_verdict(self) -> bool {
        matches!(self, Self::Verified | Self::Rejected | Self::Expired)
    }

    /// A stable label, for logs and for the transition history.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Verified => "verified",
            Self::Rejected => "rejected",
            Self::Expired => "expired",
        }
    }
}

/// Why the registry refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    /// No record with this id.
    UnknownProof { id: u64 },
    /// The proof was presented against a state root other than the one it was
    /// made for.
    WrongStateRoot { id: u64 },
    /// The proof digest presented does not match the recorded one. A different
    /// proof for the same claim is a different proof.
    DigestMismatch { id: u64 },
    /// The record already carries a verdict and cannot be re-opened. See the
    /// module note: there is no path back to pending.
    AlreadyDecided { id: u64, status: Status },
    /// The record has expired and cannot accept a new verdict. Re-claim it as a
    /// new proof instead.
    Expired { id: u64 },
    /// The id was already used. Ids are not reused, because a reused id makes
    /// every earlier statement about that id ambiguous.
    DuplicateId { id: u64 },
    /// A transition with no reason. Every transition is recorded, and a
    /// transition nobody can explain afterwards is indistinguishable from one
    /// that happened by accident.
    NoReason,
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownProof { id } => write!(f, "there is no proof record {id}"),
            Self::WrongStateRoot { id } => write!(
                f,
                "proof {id} was presented against a different state root than the one it was made for"
            ),
            Self::DigestMismatch { id } => {
                write!(f, "the proof presented for {id} is not the proof that was recorded")
            }
            Self::AlreadyDecided { id, status } => {
                write!(f, "proof {id} already carries the verdict {status} and cannot be re-opened")
            }
            Self::Expired { id } => {
                write!(f, "proof {id} has expired; re-claim it as a new proof rather than reviving it")
            }
            Self::DuplicateId { id } => write!(f, "proof id {id} is already in use"),
            Self::NoReason => write!(f, "a status transition without a reason is not recorded"),
        }
    }
}

/// One recorded status transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub from: Status,
    pub to: Status,
    pub at_height: u64,
    /// Why. Never empty - see [`ProofError::NoReason`].
    pub reason: String,
}

/// One proof record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofRecord {
    pub id: u64,
    /// What the proof claims. Opaque here; the registry does not interpret it and
    /// therefore cannot be fooled by it.
    pub claim: String,
    /// The proof's own digest. A different digest for the same claim is a
    /// different proof and is refused.
    pub proof_digest: [u8; 32],
    /// The state root the proof was made against. The field that makes a proof a
    /// fact *about something* rather than a free-floating assertion.
    pub state_root: [u8; 32],
    /// The height at which the proof stops describing the current state. Zero
    /// means it does not expire by height, which is a choice the caller makes
    /// explicitly rather than one that happens by default.
    pub expires_at_height: u64,
    pub status: Status,
    pub transitions: Vec<Transition>,
}

/// The registry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Ledger {
    records: BTreeMap<u64, ProofRecord>,
}

impl Ledger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a new proof, pending.
    ///
    /// # Errors
    ///
    /// [`ProofError::DuplicateId`].
    pub fn claim(
        &mut self,
        id: u64,
        claim: &str,
        proof_digest: [u8; 32],
        state_root: [u8; 32],
        expires_at_height: u64,
        at_height: u64,
    ) -> Result<(), ProofError> {
        if self.records.contains_key(&id) {
            return Err(ProofError::DuplicateId { id });
        }
        self.records.insert(
            id,
            ProofRecord {
                id,
                claim: claim.to_string(),
                proof_digest,
                state_root,
                expires_at_height,
                status: Status::Pending,
                transitions: vec![Transition {
                    from: Status::Pending,
                    to: Status::Pending,
                    at_height,
                    reason: "claimed".to_string(),
                }],
            },
        );
        Ok(())
    }

    /// Records a verdict.
    ///
    /// Checks, in order: the record exists, it is still pending, the proof
    /// presented is the proof that was recorded, and it is presented against the
    /// state root it was made for. The last two are what stop a caller from
    /// laundering a verdict: a valid proof for a different claim, or a valid
    /// claim proven against a stale state.
    ///
    /// # Errors
    ///
    /// Any [`ProofError`] that applies.
    pub fn accept(
        &mut self,
        id: u64,
        presented_digest: [u8; 32],
        presented_root: [u8; 32],
        verified: bool,
        reason: &str,
        at_height: u64,
    ) -> Result<Status, ProofError> {
        if reason.is_empty() {
            return Err(ProofError::NoReason);
        }
        let Some(record) = self.records.get(&id) else {
            return Err(ProofError::UnknownProof { id });
        };
        if record.status != Status::Pending {
            return Err(ProofError::AlreadyDecided {
                id,
                status: record.status,
            });
        }
        if record.proof_digest != presented_digest {
            return Err(ProofError::DigestMismatch { id });
        }
        if record.state_root != presented_root {
            return Err(ProofError::WrongStateRoot { id });
        }
        let to = if verified { Status::Verified } else { Status::Rejected };
        self.apply(id, to, reason, at_height)
    }

    /// Expires every record whose `expires_at_height` has passed.
    ///
    /// Only verified records expire. A pending record that passes its expiry was
    /// never checked, and marking it expired would turn "nobody looked" into
    /// "it held once", which is the exact confusion this crate exists to prevent.
    /// Such a record is rejected instead, with the reason saying so.
    pub fn expire(&mut self, now_height: u64, reason: &str) -> usize {
        let due: Vec<u64> = self
            .records
            .iter()
            .filter(|(_, r)| {
                r.expires_at_height > 0
                    && now_height >= r.expires_at_height
                    && matches!(r.status, Status::Pending | Status::Verified)
            })
            .map(|(id, _)| *id)
            .collect();
        let mut changed = 0;
        for id in due {
            let to = if self.records.get(&id).is_some_and(|r| r.status == Status::Verified) {
                Status::Expired
            } else {
                // Never checked and now stale. Rejected, not expired, because
                // expiry would claim a verdict that was never made.
                Status::Rejected
            };
            let text = if to == Status::Expired {
                reason.to_string()
            } else {
                format!("{reason} (never verified)");
            };
            if self.apply(id, to, &text, now_height).is_ok() {
                changed += 1;
            }
        }
        changed
    }

    /// Applies a transition. Private: every entry point that reaches it has
    /// already checked the preconditions, and a public version would be a way to
    /// skip them.
    fn apply(&mut self, id: u64, to: Status, reason: &str, at_height: u64) -> Result<Status, ProofError> {
        if reason.is_empty() {
            return Err(ProofError::NoReason);
        }
        let Some(record) = self.records.get_mut(&id) else {
            return Err(ProofError::UnknownProof { id });
        };
        if record.status == Status::Expired {
            return Err(ProofError::Expired { id });
        }
        record.transitions.push(Transition {
            from: record.status,
            to,
            at_height,
            reason: reason.to_string(),
        });
        record.status = to;
        Ok(to)
    }

    /// Reads a record.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&ProofRecord> {
        self.records.get(&id)
    }

    /// How many records carry each status. Reported as counts per status rather
    /// than a single number, because "twelve records" does not say how many of
    /// them are verdicts.
    #[must_use]
    pub fn status_counts(&self) -> [(Status, u64); 4] {
        let mut counts = [
            (Status::Pending, 0u64),
            (Status::Verified, 0),
            (Status::Rejected, 0),
            (Status::Expired, 0),
        ];
        for record in self.records.values() {
            if let Some(slot) = counts.iter_mut().find(|(s, _)| *s == record.status) {
                slot.1 = slot.1.saturating_add(1);
            }
        }
        counts
    }

    /// The number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the ledger holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT_A: [u8; 32] = [0xaa; 32];
    const ROOT_B: [u8; 32] = [0xbb; 32];
    const PROOF: [u8; 32] = [0x11; 32];

    fn ledger_with_one() -> Ledger {
        let mut l = Ledger::new();
        l.claim(1, "the batch settles", PROOF, ROOT_A, 100, 10).expect("claim");
        l
    }

    #[test]
    fn a_claimed_proof_starts_pending() {
        let l = ledger_with_one();
        assert_eq!(l.get(1).map(|r| r.status), Some(Status::Pending));
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn an_id_is_never_reused() {
        // A reused id makes every earlier statement about that id ambiguous.
        let mut l = ledger_with_one();
        assert_eq!(
            l.claim(1, "something else", PROOF, ROOT_A, 100, 11),
            Err(ProofError::DuplicateId { id: 1 })
        );
    }

    #[test]
    fn a_proof_presented_against_a_different_state_root_is_refused() {
        // A proof made against root A is not a proof about root B, and a registry
        // that stores "this claim is proven" without the root will answer for a
        // state the proof never covered.
        let mut l = ledger_with_one();
        assert_eq!(
            l.accept(1, PROOF, ROOT_B, true, "checked", 20),
            Err(ProofError::WrongStateRoot { id: 1 })
        );
        assert_eq!(l.get(1).map(|r| r.status), Some(Status::Pending), "a refused verdict changed the status");
    }

    #[test]
    fn a_different_proof_for_the_same_claim_is_refused() {
        let mut l = ledger_with_one();
        assert_eq!(
            l.accept(1, [0x22; 32], ROOT_A, true, "checked", 20),
            Err(ProofError::DigestMismatch { id: 1 })
        );
    }

    #[test]
    fn a_verdict_cannot_be_re_opened() {
        // A proof that was verified and is now merely pending again has lost
        // information, and nobody can tell whether it was re-opened deliberately
        // or overwritten by a bug.
        let mut l = ledger_with_one();
        assert_eq!(l.accept(1, PROOF, ROOT_A, true, "checked", 20), Ok(Status::Verified));
        assert_eq!(
            l.accept(1, PROOF, ROOT_A, false, "changed my mind", 21),
            Err(ProofError::AlreadyDecided {
                id: 1,
                status: Status::Verified
            })
        );
    }

    #[test]
    fn expiry_is_not_failure() {
        // An expired proof was checked and held; the state has since moved. A
        // rejected one was checked and did not hold. The two are different facts.
        let mut l = ledger_with_one();
        l.accept(1, PROOF, ROOT_A, true, "checked", 20).expect("verify");
        assert_eq!(l.expire(150, "state moved"), 1);
        assert_eq!(l.get(1).map(|r| r.status), Some(Status::Expired));
        let counts = l.status_counts();
        let expired = counts.iter().find(|(s, _)| *s == Status::Expired).map(|(_, n)| *n);
        let rejected = counts.iter().find(|(s, _)| *s == Status::Rejected).map(|(_, n)| *n);
        assert_eq!(expired, Some(1));
        assert_eq!(rejected, Some(0));
    }

    #[test]
    fn a_pending_proof_that_goes_stale_is_rejected_not_expired() {
        // Marking an unchecked record expired would turn "nobody looked" into "it
        // held once", which is the exact confusion this crate exists to prevent.
        let mut l = ledger_with_one();
        assert_eq!(l.expire(150, "state moved"), 1);
        assert_eq!(l.get(1).map(|r| r.status), Some(Status::Rejected));
        let reason = l.get(1).and_then(|r| r.transitions.last().map(|t| t.reason.clone()));
        assert!(
            reason.is_some_and(|r| r.contains("never verified")),
            "the rejection does not say it was never checked: {reason:?}"
        );
    }

    #[test]
    fn an_expired_record_cannot_accept_a_new_verdict() {
        // It is re-claimed as a new proof instead. Reviving it would make the
        // earlier expiry meaningless.
        let mut l = ledger_with_one();
        l.accept(1, PROOF, ROOT_A, true, "checked", 20).expect("verify");
        l.expire(150, "state moved");
        assert!(matches!(
            l.accept(1, PROOF, ROOT_A, true, "again", 160),
            Err(ProofError::AlreadyDecided { .. })
        ));
    }

    #[test]
    fn a_transition_without_a_reason_is_refused() {
        // A transition nobody can explain afterwards is indistinguishable from
        // one that happened by accident.
        let mut l = ledger_with_one();
        assert_eq!(l.accept(1, PROOF, ROOT_A, true, "", 20), Err(ProofError::NoReason));
        assert_eq!(l.get(1).map(|r| r.status), Some(Status::Pending));
    }

    #[test]
    fn every_transition_is_recorded_in_order() {
        let mut l = ledger_with_one();
        l.accept(1, PROOF, ROOT_A, true, "checked", 20).expect("verify");
        l.expire(150, "state moved");
        let record = l.get(1).expect("record");
        assert_eq!(record.transitions.len(), 3, "claim, verdict, expiry");
        assert_eq!(record.transitions.first().map(|t| t.to), Some(Status::Pending));
        assert_eq!(record.transitions.get(1).map(|t| t.to), Some(Status::Verified));
        assert_eq!(record.transitions.get(2).map(|t| t.to), Some(Status::Expired));
    }

    #[test]
    fn a_proof_that_never_expires_says_so_explicitly() {
        // Zero means "does not expire by height", and it is a choice the caller
        // makes rather than something that happens by default.
        let mut l = Ledger::new();
        l.claim(1, "permanent", PROOF, ROOT_A, 0, 10).expect("claim");
        l.accept(1, PROOF, ROOT_A, true, "checked", 20).expect("verify");
        assert_eq!(l.expire(u64::MAX, "state moved"), 0, "a non-expiring proof expired");
        assert_eq!(l.get(1).map(|r| r.status), Some(Status::Verified));
    }

    #[test]
    fn status_counts_report_each_status_separately() {
        // "Twelve records" does not say how many of them are verdicts.
        let mut l = Ledger::new();
        l.claim(1, "a", [1; 32], ROOT_A, 0, 10).expect("claim");
        l.claim(2, "b", [2; 32], ROOT_A, 0, 10).expect("claim");
        l.accept(1, [1; 32], ROOT_A, true, "ok", 11).expect("verify");
        l.accept(2, [2; 32], ROOT_A, false, "bad", 11).expect("reject");
        let counts: BTreeMap<Status, u64> = l.status_counts().into_iter().collect();
        assert_eq!(counts.get(&Status::Verified), Some(&1));
        assert_eq!(counts.get(&Status::Rejected), Some(&1));
        assert_eq!(counts.get(&Status::Pending), Some(&0));
    }

    #[test]
    fn only_a_verdict_counts_as_a_verdict() {
        assert!(!Status::Pending.is_verdict());
        assert!(Status::Verified.is_verdict());
        assert!(Status::Rejected.is_verdict());
        assert!(Status::Expired.is_verdict(), "expiry is a verdict about a past state");
        assert_eq!(Status::Expired.label(), "expired");
    }
}
