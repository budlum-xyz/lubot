//! Binding a run to the exact corpus, grant epoch and policy it will use.
//!
//! # Why activation is a separate step
//!
//! A run answers questions over a corpus, under a grant book, inside a tool
//! policy. All three are facts about the run that have to be settled *before*
//! the first question, and settled in one place, because afterwards they are the
//! only way to say what an answer was produced from.
//!
//! Without an activation the same three facts are read ad hoc: the corpus digest
//! when a record is loaded, the grant state when a permission is checked, the
//! policy when a tool is called. Each read is correct at the moment it happens
//! and the run as a whole describes nothing, because nothing records that all
//! three were the same throughout.
//!
//! # The rules
//!
//! **1. The corpus digest is recorded once, at activation.** Reading it per
//! query would let a corpus swapped mid-run produce answers attributed to the
//! corpus that was there at the start, and nothing would notice.
//!
//! **2. A grant issued against one corpus does not authorize another.**
//! [`ActivationLedger::activate`] refuses with
//! [`ActivationError::GrantBoundElsewhere`] when the grant epoch was issued for a
//! different digest. The alternative - a grant that follows the reader rather
//! than the corpus - means replacing the corpus silently widens every grant ever
//! issued.
//!
//! **3. The policy comes from the activation, never from the request.** A caller
//! that could supply the limits it will run under could supply wider ones.
//!
//! **4. A run is not re-activated in place.** Pointing the same activation at a
//! different corpus mints a new one and closes the old, so the record for the
//! first run keeps describing the first run.
//!
//! **5. An activation expires.** A standing activation is a standing
//! authorization nobody re-approved, and [`ActivationLedger::authorize`] refuses
//! an expired one rather than quietly extending it.

use std::collections::BTreeMap;

/// Seconds, matching [`lubot_grant::Seconds`].
pub type Seconds = u64;

/// The limits a run operates inside.
///
/// Set by whoever mints the activation. Nothing in this module reads a policy
/// out of a request, because a caller that supplies its own limits supplies
/// wider ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// How many questions one activation may answer.
    pub question_budget: u64,
    /// How many restricted items one activation may open.
    pub restricted_open_ceiling: u64,
    /// How long the activation stays valid.
    pub lifetime: Seconds,
}

impl Policy {
    /// A policy with no room to do anything. Used as the floor when a caller
    /// supplies something nonsensical, so a bad value cannot mean "unlimited".
    #[must_use]
    pub const fn none() -> Self {
        Self {
            question_budget: 0,
            restricted_open_ceiling: 0,
            lifetime: 0,
        }
    }
}

/// Why an activation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationError {
    /// No activation with this id.
    Unknown { id: u64 },
    /// The corpus carries no digest, so nothing can be bound to it.
    NoCorpusDigest,
    /// The grant epoch was issued against a different corpus.
    GrantBoundElsewhere {
        epoch: u64,
        issued_for: String,
        requested: String,
    },
    /// The activation's lifetime has passed.
    Expired { id: u64, expires_at: Seconds, now: Seconds },
    /// The activation was closed, by supersession or by hand.
    Closed { id: u64, reason: String },
    /// The policy has no room for another question.
    QuestionBudgetSpent { id: u64, budget: u64 },
    /// The policy has no room for another restricted open.
    RestrictedCeilingReached { id: u64, ceiling: u64 },
}

impl std::fmt::Display for ActivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { id } => write!(f, "there is no activation {id}"),
            Self::NoCorpusDigest => write!(
                f,
                "the corpus carries no digest, so a run cannot be bound to it"
            ),
            Self::GrantBoundElsewhere {
                epoch,
                issued_for,
                requested,
            } => write!(
                f,
                "grant epoch {epoch} was issued for corpus {issued_for} and cannot authorize {requested}; a grant that follows the reader instead of the corpus widens itself the day the corpus is replaced"
            ),
            Self::Expired {
                id,
                expires_at,
                now,
            } => write!(
                f,
                "activation {id} expired at {expires_at} and it is now {now}; it is not quietly extended"
            ),
            Self::Closed { id, reason } => {
                write!(f, "activation {id} is closed: {reason}")
            }
            Self::QuestionBudgetSpent { id, budget } => {
                write!(f, "activation {id} has spent its budget of {budget} questions")
            }
            Self::RestrictedCeilingReached { id, ceiling } => write!(
                f,
                "activation {id} has reached its ceiling of {ceiling} restricted opens"
            ),
        }
    }
}

/// A run that has been bound to a corpus, a grant epoch and a policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activation {
    pub id: u64,
    /// The corpus this run reads. Recorded here and never re-read.
    pub corpus_digest: String,
    /// The grant epoch this run consults.
    pub grant_epoch: u64,
    /// Who is running it.
    pub reader: String,
    pub policy: Policy,
    pub activated_at: Seconds,
    pub expires_at: Seconds,
    pub questions_used: u64,
    pub restricted_opens_used: u64,
    /// Set once the activation stops being the current one.
    pub closed_reason: String,
}

impl Activation {
    /// Whether this activation is still the current one.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.closed_reason.is_empty()
    }

    /// The canonical text of the activation.
    ///
    /// Written out field by field rather than delegated to a serializer, because
    /// a serializer whose format may change would make every recorded activation
    /// unverifiable the day it did.
    #[must_use]
    pub fn canonical(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.id,
            self.corpus_digest,
            self.grant_epoch,
            self.reader,
            self.policy.question_budget,
            self.policy.restricted_open_ceiling,
            self.policy.lifetime,
            self.activated_at,
            self.expires_at
        )
    }
}

/// The ledger of activations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ActivationLedger {
    activations: BTreeMap<u64, Activation>,
    /// Which grant epoch was issued for which corpus. Populated by
    /// [`Self::bind_epoch`]; an epoch the ledger has never seen is refused rather
    /// than assumed to match.
    epoch_bindings: BTreeMap<u64, String>,
    next_id: u64,
    /// Refusals by kind. Counted because a caller repeatedly asking for an
    /// activation it cannot have is worth seeing.
    pub refused_stale_grant: u64,
    pub refused_expired: u64,
}

impl ActivationLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that a grant epoch was issued for a corpus.
    ///
    /// Called when grants are issued, not when a run starts. That ordering is the
    /// point: the binding has to exist before anything can be checked against it.
    pub fn bind_epoch(&mut self, epoch: u64, corpus_digest: &str) {
        self.epoch_bindings
            .insert(epoch, corpus_digest.to_string());
    }

    /// Activates a run.
    ///
    /// Closes any previous open activation for the same reader, because a reader
    /// runs one thing at a time and a ledger showing two open activations for one
    /// reader cannot say which produced an answer.
    ///
    /// # Errors
    ///
    /// Any [`ActivationError`] that applies.
    pub fn activate(
        &mut self,
        reader: &str,
        corpus_digest: &str,
        grant_epoch: u64,
        policy: Policy,
        now: Seconds,
    ) -> Result<u64, ActivationError> {
        if corpus_digest.is_empty() {
            return Err(ActivationError::NoCorpusDigest);
        }
        // An epoch the ledger has never seen is not assumed to match. Assuming it
        // would mean a grant issued somewhere else authorizes a corpus it was
        // never issued for.
        match self.epoch_bindings.get(&grant_epoch) {
            Some(issued_for) if issued_for == corpus_digest => {}
            Some(issued_for) => {
                self.refused_stale_grant = self.refused_stale_grant.saturating_add(1);
                return Err(ActivationError::GrantBoundElsewhere {
                    epoch: grant_epoch,
                    issued_for: issued_for.clone(),
                    requested: corpus_digest.to_string(),
                });
            }
            None => {
                self.refused_stale_grant = self.refused_stale_grant.saturating_add(1);
                return Err(ActivationError::GrantBoundElsewhere {
                    epoch: grant_epoch,
                    issued_for: "an unrecorded epoch".to_string(),
                    requested: corpus_digest.to_string(),
                });
            }
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        // A reader runs one thing at a time.
        for previous in self.activations.values_mut() {
            if previous.reader == reader && previous.is_open() {
                previous.closed_reason = format!("superseded by activation {id}");
            }
        }
        self.activations.insert(
            id,
            Activation {
                id,
                corpus_digest: corpus_digest.to_string(),
                grant_epoch,
                reader: reader.to_string(),
                policy,
                activated_at: now,
                expires_at: now.saturating_add(policy.lifetime),
                questions_used: 0,
                restricted_opens_used: 0,
                closed_reason: String::new(),
            },
        );
        Ok(id)
    }

    /// Checks that an activation may answer a question, and spends one.
    ///
    /// # Errors
    ///
    /// Any [`ActivationError`] that applies.
    pub fn authorize(&mut self, id: u64, now: Seconds) -> Result<(), ActivationError> {
        let Some(activation) = self.activations.get(&id) else {
            return Err(ActivationError::Unknown { id });
        };
        if !activation.closed_reason.is_empty() {
            return Err(ActivationError::Closed {
                id,
                reason: activation.closed_reason.clone(),
            });
        }
        if now >= activation.expires_at {
            self.refused_expired = self.refused_expired.saturating_add(1);
            return Err(ActivationError::Expired {
                id,
                expires_at: activation.expires_at,
                now,
            });
        }
        if activation.questions_used >= activation.policy.question_budget {
            return Err(ActivationError::QuestionBudgetSpent {
                id,
                budget: activation.policy.question_budget,
            });
        }
        if let Some(activation) = self.activations.get_mut(&id) {
            activation.questions_used = activation.questions_used.saturating_add(1);
        }
        Ok(())
    }

    /// Checks that an activation may open a restricted item, and spends one.
    ///
    /// Separate from [`Self::authorize`] because the two budgets are different
    /// limits: a run that is allowed many questions is not thereby allowed many
    /// restricted opens.
    ///
    /// # Errors
    ///
    /// Any [`ActivationError`] that applies.
    pub fn authorize_restricted(&mut self, id: u64, now: Seconds) -> Result<(), ActivationError> {
        self.authorize_open(id, now)?;
        let Some(activation) = self.activations.get(&id) else {
            return Err(ActivationError::Unknown { id });
        };
        if activation.restricted_opens_used >= activation.policy.restricted_open_ceiling {
            return Err(ActivationError::RestrictedCeilingReached {
                id,
                ceiling: activation.policy.restricted_open_ceiling,
            });
        }
        if let Some(activation) = self.activations.get_mut(&id) {
            activation.restricted_opens_used = activation.restricted_opens_used.saturating_add(1);
        }
        Ok(())
    }

    /// The shared part of the two authorization checks.
    fn authorize_open(&self, id: u64, now: Seconds) -> Result<(), ActivationError> {
        let Some(activation) = self.activations.get(&id) else {
            return Err(ActivationError::Unknown { id });
        };
        if !activation.closed_reason.is_empty() {
            return Err(ActivationError::Closed {
                id,
                reason: activation.closed_reason.clone(),
            });
        }
        if now >= activation.expires_at {
            return Err(ActivationError::Expired {
                id,
                expires_at: activation.expires_at,
                now,
            });
        }
        Ok(())
    }

    /// Closes an activation deliberately.
    ///
    /// # Errors
    ///
    /// [`ActivationError::Unknown`].
    pub fn close(&mut self, id: u64, reason: &str) -> Result<(), ActivationError> {
        let Some(activation) = self.activations.get_mut(&id) else {
            return Err(ActivationError::Unknown { id });
        };
        if activation.closed_reason.is_empty() {
            activation.closed_reason = reason.to_string();
        }
        Ok(())
    }

    /// Reads an activation.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Activation> {
        self.activations.get(&id)
    }

    /// The open activation for a reader, if there is one.
    ///
    /// At most one exists, which [`Self::activate`] maintains.
    #[must_use]
    pub fn open_for(&self, reader: &str) -> Option<&Activation> {
        self.activations
            .values()
            .find(|a| a.reader == reader && a.is_open())
    }

    /// How many activations exist, open or closed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Whether the ledger is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "9f2c4a1b";
    const OTHER: &str = "7d05e3ff";

    fn policy() -> Policy {
        Policy {
            question_budget: 3,
            restricted_open_ceiling: 1,
            lifetime: 100,
        }
    }

    fn ledger() -> ActivationLedger {
        let mut ledger = ActivationLedger::new();
        ledger.bind_epoch(1, CORPUS);
        ledger
    }

    #[test]
    fn a_run_is_bound_to_the_digest_recorded_at_activation() {
        // Read once here, never per query. A corpus swapped mid-run would
        // otherwise produce answers attributed to the wrong corpus.
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        assert_eq!(ledger.get(id).map(|a| a.corpus_digest.clone()), Some(CORPUS.to_string()));
    }

    #[test]
    fn a_grant_epoch_issued_for_another_corpus_is_refused() {
        // A grant that follows the reader rather than the corpus widens itself
        // the day the corpus is replaced.
        let mut ledger = ledger();
        assert_eq!(
            ledger.activate("alice", OTHER, 1, policy(), 0),
            Err(ActivationError::GrantBoundElsewhere {
                epoch: 1,
                issued_for: CORPUS.to_string(),
                requested: OTHER.to_string(),
            })
        );
        assert_eq!(ledger.refused_stale_grant, 1);
    }

    #[test]
    fn an_epoch_the_ledger_has_never_seen_is_refused_not_assumed() {
        // Assuming a match would let a grant issued somewhere else authorize a
        // corpus it was never issued for.
        let mut ledger = ledger();
        assert!(matches!(
            ledger.activate("alice", CORPUS, 99, policy(), 0),
            Err(ActivationError::GrantBoundElsewhere { epoch: 99, .. })
        ));
    }

    #[test]
    fn a_corpus_without_a_digest_cannot_be_bound() {
        let mut ledger = ledger();
        assert_eq!(
            ledger.activate("alice", "", 1, policy(), 0),
            Err(ActivationError::NoCorpusDigest)
        );
    }

    #[test]
    fn activating_again_supersedes_the_previous_activation() {
        // A reader runs one thing at a time. Two open activations for one reader
        // cannot say which produced an answer.
        let mut ledger = ledger();
        let first = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        let second = ledger
            .activate("alice", CORPUS, 1, policy(), 10)
            .expect("activate");
        assert_ne!(first, second);
        assert!(ledger.get(first).is_some_and(|a| !a.is_open()));
        assert!(ledger.get(second).is_some_and(Activation::is_open));
        assert_eq!(ledger.open_for("alice").map(|a| a.id), Some(second));
    }

    #[test]
    fn the_superseded_record_keeps_describing_its_own_run() {
        // Re-activating in place would make the first run's record describe the
        // second run.
        let mut ledger = ledger();
        let first = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        ledger.authorize(first, 1).expect("authorize");
        ledger
            .activate("alice", CORPUS, 1, policy(), 10)
            .expect("activate");
        let record = ledger.get(first).expect("record");
        assert_eq!(record.questions_used, 1, "the old record lost its own count");
        assert_eq!(record.activated_at, 0);
        assert!(record.closed_reason.contains("superseded"));
    }

    #[test]
    fn an_expired_activation_is_refused_not_extended() {
        // A standing activation is a standing authorization nobody re-approved.
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        assert_eq!(
            ledger.authorize(id, 100),
            Err(ActivationError::Expired {
                id,
                expires_at: 100,
                now: 100
            })
        );
        assert_eq!(ledger.refused_expired, 1);
        assert_eq!(
            ledger.get(id).map(|a| a.questions_used),
            Some(0),
            "a refused question was still spent"
        );
    }

    #[test]
    fn the_question_budget_is_spent_and_enforced() {
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        for moment in 1..=3 {
            ledger.authorize(id, moment).expect("authorize");
        }
        assert_eq!(
            ledger.authorize(id, 4),
            Err(ActivationError::QuestionBudgetSpent {
                id,
                budget: 3
            })
        );
    }

    #[test]
    fn the_restricted_ceiling_is_a_separate_limit() {
        // Many questions is not thereby many restricted opens.
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        ledger.authorize_restricted(id, 1).expect("first open");
        assert_eq!(
            ledger.authorize_restricted(id, 2),
            Err(ActivationError::RestrictedCeilingReached { id, ceiling: 1 })
        );
        // The question budget is untouched by the restricted refusal.
        ledger.authorize(id, 3).expect("a question is still allowed");
    }

    #[test]
    fn a_closed_activation_refuses_both_kinds_of_request() {
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        ledger.close(id, "the reader signed off").expect("close");
        assert!(matches!(
            ledger.authorize(id, 1),
            Err(ActivationError::Closed { .. })
        ));
        assert!(matches!(
            ledger.authorize_restricted(id, 1),
            Err(ActivationError::Closed { .. })
        ));
    }

    #[test]
    fn closing_twice_keeps_the_first_reason() {
        // The reason a run stopped is part of its record, and the second close is
        // not a better account of it.
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        ledger.close(id, "first").expect("close");
        ledger.close(id, "second").expect("close again");
        assert_eq!(
            ledger.get(id).map(|a| a.closed_reason.clone()),
            Some("first".to_string())
        );
    }

    #[test]
    fn authorizing_an_unknown_activation_says_so() {
        let mut ledger = ledger();
        assert_eq!(
            ledger.authorize(7, 0),
            Err(ActivationError::Unknown { id: 7 })
        );
        assert_eq!(
            ledger.authorize_restricted(7, 0),
            Err(ActivationError::Unknown { id: 7 })
        );
        assert_eq!(
            ledger.close(7, "reason"),
            Err(ActivationError::Unknown { id: 7 })
        );
    }

    #[test]
    fn a_policy_with_no_room_cannot_do_anything() {
        // The floor for a nonsensical policy is "nothing", never "unlimited".
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, Policy::none(), 0)
            .expect("activate");
        assert_eq!(
            ledger.authorize(id, 0),
            Err(ActivationError::QuestionBudgetSpent { id, budget: 0 })
        );
    }

    #[test]
    fn the_canonical_text_covers_every_binding_field() {
        // This is what makes "the answer came from corpus X under grants Y" a
        // checkable claim rather than a memory.
        let mut ledger = ledger();
        let id = ledger
            .activate("alice", CORPUS, 1, policy(), 5)
            .expect("activate");
        let canonical = ledger.get(id).map(|a| a.canonical());
        let Some(text) = canonical else {
            panic!("the activation was not recorded");
        };
        let fields: Vec<&str> = text.split('\t').collect();
        assert_eq!(fields.len(), 9);
        assert_eq!(fields[1], CORPUS);
        assert_eq!(fields[2], "1");
        assert_eq!(fields[3], "alice");
        assert_eq!(fields[8], "105", "expiry is activation plus lifetime");
    }

    #[test]
    fn two_readers_hold_independent_activations() {
        let mut ledger = ledger();
        let alice = ledger
            .activate("alice", CORPUS, 1, policy(), 0)
            .expect("activate");
        let bob = ledger
            .activate("bob", CORPUS, 1, policy(), 0)
            .expect("activate");
        ledger.authorize(alice, 1).expect("alice");
        assert_eq!(ledger.get(bob).map(|a| a.questions_used), Some(0));
        assert!(ledger.get(bob).is_some_and(Activation::is_open));
        assert_eq!(ledger.len(), 2);
    }

    #[test]
    fn an_empty_ledger_has_no_activations() {
        let ledger = ActivationLedger::new();
        assert!(ledger.is_empty());
        assert_eq!(ledger.len(), 0);
        assert_eq!(ledger.open_for("alice"), None);
    }
}
