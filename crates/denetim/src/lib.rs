//! The audit trail: what happened, who did it, and whether the record still holds.
//!
//! # The property this crate exists to provide
//!
//! **An audit log that can be edited is not an audit log.** That is not a slogan,
//! it is a design constraint, and the way it is enforced here is by absence: this
//! type has no update method and no delete method. Not a guarded one, not one
//! that requires authority - none. A caller that needs to correct a record
//! appends a correction, and the correction is itself an audited entry with an
//! actor and a reason. The original stays.
//!
//! # What the chain does and does not prove
//!
//! Every entry commits to the one before it, so an edit to any existing entry
//! changes that entry's link and every link after it. [`Trail::verify`] walks the
//! chain and reports the **first** broken entry, because an auditor holding a
//! thousand entries and "the log is wrong" has a long afternoon.
//!
//! The honest limitation, stated rather than hidden: **a hash chain does not
//! detect truncation.** Removing the last entry leaves every remaining link
//! valid. Detecting that requires an anchor outside the log - a published head,
//! a commitment in a block, anything the person doing the truncating cannot also
//! rewrite. [`Trail::head`] exists so a caller can publish it, and
//! [`Trail::verify_against_anchor`] is the check that uses it. A trail whose head
//! has never been anchored is a trail that can be shortened silently, and this
//! crate says so instead of implying otherwise.
//!
//! # Gaps
//!
//! Entries carry a sequence number and the sequence must be contiguous. **A gap
//! is a finding, not a silence.** A missing number is reported as a hole with the
//! numbers on either side, because the alternative - a log that quietly renumbers
//! - is a log that has already been edited.

use lubot_muhur::Sealer;
use std::collections::BTreeSet;

/// Why an audit operation refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditError {
    /// An entry with no actor. An unattributable entry is not evidence of
    /// anything, and accepting one would let a caller add findings nobody has to
    /// answer for.
    NoActor,
    /// An entry with no reason. Same reasoning: an action nobody can explain
    /// afterwards is indistinguishable from an accident.
    NoReason,
    /// The sequence number is not the next one. Refused rather than renumbered,
    /// because a log that renumbers has already been edited.
    SequenceGap { expected: u64, got: u64 },
    /// The entry at `index` does not match its recorded link.
    EntryTampered { index: usize, expected: String, got: String },
    /// The log is shorter than the anchored head says it should be. This is the
    /// truncation the chain itself cannot see.
    Truncated { anchored_length: usize, got: usize },
    /// The head does not match the anchor.
    HeadMismatch { anchored: String, got: String },
}

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActor => write!(f, "the entry has no actor and is therefore not attributable"),
            Self::NoReason => write!(f, "the entry has no reason"),
            Self::SequenceGap { expected, got } => {
                write!(f, "the entry carries sequence {got}, the next one is {expected}; the log does not renumber")
            }
            Self::EntryTampered {
                index,
                expected,
                got,
            } => write!(
                f,
                "entry {index} does not match its recorded link: expected {expected}, these entries reach {got}"
            ),
            Self::Truncated {
                anchored_length,
                got,
            } => write!(
                f,
                "the log holds {got} entries but the anchored head says {anchored_length}; entries were removed from the end"
            ),
            Self::HeadMismatch { anchored, got } => {
                write!(f, "the head does not match the anchor: {anchored} != {got}")
            }
        }
    }
}

/// One audit entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub sequence: u64,
    /// Who. Never empty.
    pub actor: String,
    /// What happened. A short, stable label - not prose, because prose gets
    /// improved and an improved entry is a different entry.
    pub kind: &'static str,
    /// Why. Never empty.
    pub reason: String,
    /// The height the action was recorded at.
    pub at_height: u64,
    /// What the entry was about, if anything. Opaque: the trail records that an
    /// action referenced it, not what it says.
    pub subject: String,
}

impl Entry {
    /// The canonical text of the entry, which is what gets chained.
    ///
    /// Written out field by field rather than delegated to a serializer, because
    /// a serializer whose output format may change would make every existing
    /// chain unverifiable the day it did.
    #[must_use]
    pub fn canonical(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.sequence, self.actor, self.kind, self.reason, self.at_height, self.subject
        )
    }
}

/// An append-only audit trail.
///
/// No update method, no delete method. See the module note.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Trail {
    entries: Vec<Entry>,
    links: Vec<String>,
}

impl Trail {
    /// An empty trail.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an entry.
    ///
    /// The sequence number is assigned by the trail, not by the caller, because a
    /// caller-chosen sequence is a caller-chosen position in the record.
    ///
    /// # Errors
    ///
    /// [`AuditError::NoActor`] or [`AuditError::NoReason`].
    pub fn append(
        &mut self,
        actor: &str,
        kind: &'static str,
        reason: &str,
        at_height: u64,
        subject: &str,
    ) -> Result<u64, AuditError> {
        if actor.is_empty() {
            return Err(AuditError::NoActor);
        }
        if reason.is_empty() {
            return Err(AuditError::NoReason);
        }
        let sequence = self.entries.len() as u64;
        let entry = Entry {
            sequence,
            actor: actor.to_string(),
            kind,
            reason: reason.to_string(),
            at_height,
            subject: subject.to_string(),
        };
        let previous = self.links.last().cloned().unwrap_or_else(Sealer::genesis);
        let link = Sealer::link(&previous, &entry.canonical());
        self.entries.push(entry);
        self.links.push(link.clone());
        Ok(sequence)
    }

    /// The current head: the link after the last entry.
    ///
    /// Publish this somewhere the person who would truncate the log cannot also
    /// rewrite. Without an anchor, [`Self::verify`] cannot detect a shortened
    /// tail - see the module note.
    #[must_use]
    pub fn head(&self) -> String {
        self.links.last().cloned().unwrap_or_else(Sealer::genesis)
    }

    /// How many entries the trail holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the trail holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reads an entry.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index)
    }

    /// The entries, in order. Returned as an iterator over references: the trail
    /// does not hand out its vector, because a caller holding it could edit it
    /// and the type would have no way to notice.
    pub fn iter(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter()
    }

    /// Verifies the chain and reports the first broken entry.
    ///
    /// Cannot detect truncation. Use [`Self::verify_against_anchor`] for that.
    ///
    /// # Errors
    ///
    /// [`AuditError::EntryTampered`] naming the first index that disagrees.
    pub fn verify(&self) -> Result<(), AuditError> {
        let mut running = Sealer::genesis();
        for (index, entry) in self.entries.iter().enumerate() {
            running = Sealer::link(&running, &entry.canonical());
            let Some(expected) = self.links.get(index) else {
                return Err(AuditError::EntryTampered {
                    index,
                    expected: String::new(),
                    got: running,
                });
            };
            if expected != &running {
                return Err(AuditError::EntryTampered {
                    index,
                    expected: expected.clone(),
                    got: running,
                });
            }
        }
        Ok(())
    }

    /// Verifies the chain against a published anchor.
    ///
    /// This is the check that catches truncation: the anchor says how long the
    /// log was and what its head was, and a shortened log fails on length even
    /// though every remaining link is valid.
    ///
    /// # Errors
    ///
    /// [`AuditError::Truncated`], [`AuditError::HeadMismatch`], or anything
    /// [`Self::verify`] returns.
    pub fn verify_against_anchor(&self, anchored_head: &str, anchored_length: usize) -> Result<(), AuditError> {
        if self.entries.len() < anchored_length {
            return Err(AuditError::Truncated {
                anchored_length,
                got: self.entries.len(),
            });
        }
        self.verify()?;
        if self.head() != anchored_head {
            return Err(AuditError::HeadMismatch {
                anchored: anchored_head.to_string(),
                got: self.head(),
            });
        }
        Ok(())
    }

    /// Finds gaps in the sequence.
    ///
    /// There should be none, because [`Self::append`] assigns the numbers. The
    /// check exists for a trail rebuilt from storage, where the numbers come back
    /// from somewhere else and a missing one means an entry was removed. **A gap
    /// is a finding, not a silence**: the alternative is a log that quietly
    /// renumbers, which is a log that has already been edited.
    #[must_use]
    pub fn gaps(&self) -> Vec<(u64, u64)> {
        let present: BTreeSet<u64> = self.entries.iter().map(|e| e.sequence).collect();
        let mut holes = Vec::new();
        let mut missing_start: Option<u64> = None;
        let highest = present.iter().max().copied().unwrap_or(0);
        for n in 0..=highest {
            if present.contains(&n) {
                if let Some(start) = missing_start.take() {
                    holes.push((start, n.saturating_sub(1)));
                }
            } else if missing_start.is_none() {
                missing_start = Some(n);
            }
        }
        if let Some(start) = missing_start {
            holes.push((start, highest));
        }
        holes
    }

    /// The entries by one actor. Provided because "what did this identity do" is
    /// the first question an audit asks, and answering it by scanning the whole
    /// trail at every call site is how the question stops being asked.
    #[must_use]
    pub fn by_actor(&self, actor: &str) -> Vec<&Entry> {
        self.entries.iter().filter(|e| e.actor == actor).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trail_with_three() -> Trail {
        let mut t = Trail::new();
        t.append("alice", "batch-opened", "scheduled payout", 10, "batch-1").expect("1");
        t.append("bob", "batch-signed", "quorum reached", 11, "batch-1").expect("2");
        t.append("alice", "batch-closed", "window elapsed", 12, "batch-1").expect("3");
        t
    }

    #[test]
    fn entries_append_in_order_with_assigned_sequences() {
        // The sequence is assigned by the trail, not the caller, because a
        // caller-chosen sequence is a caller-chosen position in the record.
        let t = trail_with_three();
        assert_eq!(t.len(), 3);
        let seqs: Vec<u64> = t.iter().map(|e| e.sequence).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        assert!(t.verify().is_ok());
    }

    #[test]
    fn an_entry_with_no_actor_is_refused() {
        // An unattributable entry is not evidence of anything.
        let mut t = Trail::new();
        assert_eq!(
            t.append("", "batch-opened", "reason", 10, "x"),
            Err(AuditError::NoActor)
        );
        assert!(t.is_empty(), "a refused entry was appended");
    }

    #[test]
    fn an_entry_with_no_reason_is_refused() {
        let mut t = Trail::new();
        assert_eq!(
            t.append("alice", "batch-opened", "", 10, "x"),
            Err(AuditError::NoReason)
        );
        assert!(t.is_empty());
    }

    #[test]
    fn an_edited_entry_is_detected_and_the_index_is_named() {
        let mut t = trail_with_three();
        if let Some(entry) = t.entries.get_mut(1) {
            entry.reason = "quorum reached by someone else".to_string();
        }
        let err = t.verify().unwrap_err();
        assert!(
            matches!(err, AuditError::EntryTampered { index: 1, .. }),
            "the tamper was not localised to entry 1: {err:?}"
        );
    }

    #[test]
    fn truncation_is_invisible_to_the_chain_alone() {
        // Stated rather than hidden: removing the last entry leaves every
        // remaining link valid. This test documents the limitation by exhibiting
        // it, so that nobody later "fixes" verify and believes truncation is
        // covered.
        let mut t = trail_with_three();
        let anchor_head = t.head();
        let anchor_len = t.len();
        t.entries.pop();
        t.links.pop();
        assert!(t.verify().is_ok(), "the chain detected a truncation it cannot detect");
        // The anchor is what catches it.
        assert_eq!(
            t.verify_against_anchor(&anchor_head, anchor_len),
            Err(AuditError::Truncated {
                anchored_length: 3,
                got: 2
            })
        );
    }

    #[test]
    fn an_anchor_catches_a_rewritten_tail_too() {
        let t = trail_with_three();
        let anchor_head = t.head();
        let anchor_len = t.len();
        let mut other = Trail::new();
        other.append("alice", "batch-opened", "scheduled payout", 10, "batch-1").expect("1");
        other.append("mallory", "batch-signed", "forged", 11, "batch-1").expect("2");
        other.append("alice", "batch-closed", "window elapsed", 12, "batch-1").expect("3");
        assert!(other.verify().is_ok(), "the rewritten chain is internally consistent");
        assert!(matches!(
            other.verify_against_anchor(&anchor_head, anchor_len),
            Err(AuditError::HeadMismatch { .. })
        ));
    }

    #[test]
    fn an_unanchored_trail_is_not_a_verified_trail() {
        // The point of the anchor: without one, a shortened log passes verify.
        let mut t = trail_with_three();
        t.entries.pop();
        t.links.pop();
        assert!(t.verify().is_ok());
    }

    #[test]
    fn a_gap_in_the_sequence_is_reported_as_a_finding() {
        // A gap is a finding, not a silence. The alternative is a log that
        // quietly renumbers, which is a log that has already been edited.
        let mut t = trail_with_three();
        t.entries.remove(1);
        assert_eq!(t.gaps(), vec![(1, 1)]);
    }

    #[test]
    fn a_contiguous_trail_has_no_gaps() {
        assert!(trail_with_three().gaps().is_empty());
        assert!(Trail::new().gaps().is_empty());
    }

    #[test]
    fn a_correction_is_appended_not_applied() {
        // There is no update method and no delete method. A caller that needs to
        // correct a record appends a correction, and the correction is itself an
        // audited entry. The original stays.
        let mut t = trail_with_three();
        let before = t.len();
        t.append("alice", "entry-corrected", "the earlier reason was wrong", 13, "batch-1")
            .expect("correction");
        assert_eq!(t.len(), before + 1);
        assert_eq!(t.get(1).map(|e| e.reason.as_str()), Some("quorum reached"));
        assert!(t.verify().is_ok());
    }

    #[test]
    fn the_canonical_form_is_field_separated_and_unambiguous() {
        // Written out field by field rather than delegated to a serializer,
        // because a serializer whose output may change would make every existing
        // chain unverifiable the day it did.
        let e = Entry {
            sequence: 7,
            actor: "alice".to_string(),
            kind: "batch-opened",
            reason: "scheduled".to_string(),
            at_height: 10,
            subject: "batch-1".to_string(),
        };
        assert_eq!(e.canonical(), "7\talice\tbatch-opened\tscheduled\t10\tbatch-1");
        // Two entries differing in any field differ canonically.
        let mut other = e.clone();
        other.at_height = 11;
        assert_ne!(e.canonical(), other.canonical());
    }

    #[test]
    fn entries_can_be_read_by_actor() {
        let t = trail_with_three();
        assert_eq!(t.by_actor("alice").len(), 2);
        assert_eq!(t.by_actor("bob").len(), 1);
        assert!(t.by_actor("nobody").is_empty());
    }

    #[test]
    fn an_empty_trail_has_the_genesis_head() {
        let t = Trail::new();
        assert_eq!(t.head(), Sealer::genesis());
        assert!(t.is_empty());
        assert!(t.verify().is_ok());
    }
}
