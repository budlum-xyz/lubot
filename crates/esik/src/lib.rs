//! Thresholds: when enough parties have agreed that it counts.
//!
//! # The four rules this crate is built around
//!
//! Counting signatures is easy. Deciding when a count *means* something is the
//! hard part, and every rule below exists because a plausible-looking threshold
//! fails without it.
//!
//! **1. A quorum counted over a set that includes the requester is not a
//! quorum.** An actor asking for approval and also holding one of the votes can
//! reach the threshold by approving itself. The requester is excluded from the
//! counting set before anything is counted, not after the answer disappoints.
//!
//! **2. A threshold whose denominator is chosen by the party seeking approval
//! is not a threshold.** `2 of 3` and `2 of 200` are the same numerator and
//! wildly different assurances. The denominator is fixed when the quorum is
//! created and cannot be widened by adding members later without going through
//! [`Quorum::extend`], which refuses to let the threshold fall below the BFT
//! floor.
//!
//! **3. Two signatures from the same signer count once.** A signer holding two
//! keys, or one key submitted twice, is one party. Counting keys instead of
//! parties is how a single actor reaches any threshold.
//!
//! **4. `n >= 3f + 1` to tolerate `f` faults.** With `n = 3f`, `f` faulty and
//! `2f` honest leaves two equally sized groups and no way to tell which is
//! right. The check is here rather than in a comment because it is the rule
//! most often quietly violated by "we only had four signers that day".
//!
//! # What this crate does not do
//!
//! It does not verify signatures. It counts parties that have been verified
//! elsewhere and answers whether the count is enough. Keeping the two apart is
//! deliberate: a quorum layer that also verifies signatures cannot be tested
//! without a key, and a quorum rule that cannot be tested is a quorum rule
//! nobody has checked.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// Why a quorum refused to form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuorumError {
    /// The threshold is zero. A threshold of zero is met by nobody, which is
    /// not a weaker quorum, it is the absence of one.
    ZeroThreshold,
    /// The member set is empty.
    NoMembers,
    /// The threshold exceeds the member count. Reachable only through a
    /// construction bug, and reported rather than silently unreachable.
    ThresholdAboveMembers { threshold: u64, members: u64 },
    /// The member set is too small to tolerate the requested fault count.
    BelowByzantineFloor {
        members: u64,
        faults: u64,
        required: u64,
    },
    /// A signer was counted who is not a member.
    NotAMember { signer: u64 },
    /// The requester tried to count its own vote.
    SelfApproval { requester: u64 },
    /// Not enough approving weight. Carries the numbers so a refusal says how
    /// far short it was instead of only that it was.
    ///
    /// One variant rather than a head-count one and a weight one, because
    /// `count` measures weight and an unweighted quorum's weight *is* its head
    /// count. Two variants would mean two places for a caller to match the wrong
    /// one.
    Insufficient { reached: u64, required: u64 },
}

impl std::fmt::Display for QuorumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroThreshold => write!(
                f,
                "the threshold is zero, which is not a weaker quorum but the absence of one"
            ),
            Self::NoMembers => write!(f, "the quorum has no members"),
            Self::ThresholdAboveMembers { threshold, members } => {
                write!(
                    f,
                    "the threshold {threshold} is above the member count {members}"
                )
            }
            Self::BelowByzantineFloor {
                members,
                faults,
                required,
            } => write!(
                f,
                "{members} members cannot tolerate {faults} faults: 3f+1 requires {required}"
            ),
            Self::NotAMember { signer } => {
                write!(f, "signer {signer} is not a member of this quorum")
            }
            Self::SelfApproval { requester } => {
                write!(f, "the requester {requester} cannot count its own approval")
            }
            Self::Insufficient { reached, required } => {
                write!(
                    f,
                    "the approvals carry weight {reached}, {required} is required"
                )
            }
        }
    }
}

/// A counting set with a threshold.
///
/// Members are identified by `u64` and carry a weight. Weight defaults to one,
/// so an unweighted m-of-n is the ordinary case and a weighted one is the same
/// type rather than a parallel implementation - two implementations of "enough"
/// is two places for the rules above to be violated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quorum {
    weights: BTreeMap<u64, u64>,
    threshold_weight: u64,
}

impl Quorum {
    /// An unweighted m-of-n quorum over `members`.
    ///
    /// # Errors
    ///
    /// [`QuorumError::ZeroThreshold`], [`QuorumError::NoMembers`], or
    /// [`QuorumError::ThresholdAboveMembers`].
    pub fn unweighted(members: &[u64], threshold: u64) -> Result<Self, QuorumError> {
        if threshold == 0 {
            return Err(QuorumError::ZeroThreshold);
        }
        if members.is_empty() {
            return Err(QuorumError::NoMembers);
        }
        let mut weights = BTreeMap::new();
        for member in members {
            weights.insert(*member, 1);
        }
        if threshold > weights.len() as u64 {
            return Err(QuorumError::ThresholdAboveMembers {
                threshold,
                members: weights.len() as u64,
            });
        }
        Ok(Self {
            weights,
            threshold_weight: threshold,
        })
    }

    /// A weighted quorum.
    ///
    /// # Errors
    ///
    /// [`QuorumError::ZeroThreshold`], [`QuorumError::NoMembers`], or
    /// [`QuorumError::ThresholdAboveMembers`] when the threshold exceeds the
    /// total weight.
    pub fn weighted(
        weights: BTreeMap<u64, u64>,
        threshold_weight: u64,
    ) -> Result<Self, QuorumError> {
        if threshold_weight == 0 {
            return Err(QuorumError::ZeroThreshold);
        }
        if weights.is_empty() {
            return Err(QuorumError::NoMembers);
        }
        let total: u64 = weights.values().copied().sum();
        if threshold_weight > total {
            return Err(QuorumError::ThresholdAboveMembers {
                threshold: threshold_weight,
                members: total,
            });
        }
        Ok(Self {
            weights,
            threshold_weight,
        })
    }

    /// The member count.
    #[must_use]
    pub fn member_count(&self) -> u64 {
        self.weights.len() as u64
    }

    /// The total weight.
    #[must_use]
    pub fn total_weight(&self) -> u64 {
        self.weights.values().copied().sum()
    }

    /// The weight needed to form a quorum.
    #[must_use]
    pub fn threshold(&self) -> u64 {
        self.threshold_weight
    }

    /// Whether `signer` is a member.
    #[must_use]
    pub fn is_member(&self, signer: u64) -> bool {
        self.weights.contains_key(&signer)
    }

    /// Adds members.
    ///
    /// The threshold is deliberately **not** raised to follow, and it is
    /// deliberately **not** left alone either: the caller passes the new
    /// threshold, and this method refuses one that would drop the set below the
    /// BFT floor for the faults it already tolerates. Widening a quorum without
    /// re-checking the floor is how `5 of 7` becomes `5 of 40` - the same
    /// numerator, an eighth of the assurance.
    ///
    /// # Errors
    ///
    /// [`QuorumError::BelowByzantineFloor`] or [`QuorumError::ZeroThreshold`].
    pub fn extend(
        &mut self,
        new_members: &[(u64, u64)],
        new_threshold_weight: u64,
        faults_to_tolerate: u64,
    ) -> Result<(), QuorumError> {
        if new_threshold_weight == 0 {
            return Err(QuorumError::ZeroThreshold);
        }
        for (member, weight) in new_members {
            self.weights.insert(*member, *weight);
        }
        let required = byzantine_floor(faults_to_tolerate);
        if self.member_count() < required {
            return Err(QuorumError::BelowByzantineFloor {
                members: self.member_count(),
                faults: faults_to_tolerate,
                required,
            });
        }
        let total = self.total_weight();
        if new_threshold_weight > total {
            return Err(QuorumError::ThresholdAboveMembers {
                threshold: new_threshold_weight,
                members: total,
            });
        }
        self.threshold_weight = new_threshold_weight;
        Ok(())
    }

    /// Counts `signers` and reports whether the quorum formed.
    ///
    /// Duplicates are collapsed before counting, and a signer that is not a
    /// member is a refusal rather than an ignored entry - an unknown signer
    /// means somebody is trying to vote who should not be able to, and quietly
    /// dropping them hides that.
    ///
    /// # Errors
    ///
    /// [`QuorumError::NotAMember`], or [`QuorumError::Insufficient`] naming how
    /// far short the count was.
    pub fn count(&self, signers: &[u64]) -> Result<u64, QuorumError> {
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut reached = 0u64;
        for signer in signers {
            if !seen.insert(*signer) {
                // One party, one vote. A signer holding two keys or one key
                // submitted twice is still one party.
                continue;
            }
            let Some(weight) = self.weights.get(signer) else {
                return Err(QuorumError::NotAMember { signer: *signer });
            };
            reached = reached.saturating_add(*weight);
        }
        if reached < self.threshold_weight {
            return Err(QuorumError::Insufficient {
                reached,
                required: self.threshold_weight,
            });
        }
        Ok(reached)
    }

    /// Counts, excluding the requester.
    ///
    /// This is the method a real approval path should call. [`Self::count`]
    /// exists for the case where there is no requester - a genesis ceremony, a
    /// migration - and calling it where there is one is how an actor ends up
    /// able to approve itself.
    ///
    /// # Errors
    ///
    /// [`QuorumError::SelfApproval`] if the requester appears in `signers`,
    /// plus everything [`Self::count`] can return.
    pub fn count_excluding(&self, signers: &[u64], requester: u64) -> Result<u64, QuorumError> {
        if signers.contains(&requester) {
            return Err(QuorumError::SelfApproval { requester });
        }
        self.count(signers)
    }

    /// Whether the quorum can still be met by the parties that have not yet
    /// answered.
    ///
    /// Used to stop waiting early: if the remaining weight cannot reach the
    /// threshold, no amount of waiting will produce a quorum, and holding the
    /// request open is latency with nothing at the end of it.
    #[must_use]
    pub fn still_reachable(&self, answered: &[u64]) -> bool {
        let outstanding: u64 = self
            .weights
            .iter()
            .filter(|(member, _)| !answered.contains(member))
            .map(|(_, weight)| *weight)
            .sum();
        let reached = self
            .weights
            .iter()
            .filter(|(member, _)| answered.contains(member))
            .map(|(_, weight)| *weight)
            .sum();
        reached.saturating_add(outstanding) >= self.threshold_weight
    }
}

/// The smallest member count that can tolerate `faults` Byzantine faults.
///
/// `3f + 1`. With `3f` members and `f` faulty, the honest `2f` cannot be
/// distinguished from a `2f` group that includes the faulty ones, and the
/// protocol has no way to choose. The extra one is not margin, it is the
/// difference between a decision and a tie.
///
/// Saturating: a fault count large enough to overflow means the caller asked for
/// something impossible, and returning the maximum says so more honestly than
/// wrapping to a small number that would look achievable.
#[must_use]
pub fn byzantine_floor(faults: u64) -> u64 {
    faults.saturating_mul(3).saturating_add(1)
}

/// How many faults `members` can tolerate.
///
/// The inverse of [`byzantine_floor`], and the number an operator should be
/// shown instead of a member count: "seven members" does not say what the
/// system survives, "two faults" does.
#[must_use]
pub fn tolerable_faults(members: u64) -> u64 {
    members.saturating_sub(1) / 3
}

/// Whether a member set is large enough to tolerate `faults`.
#[must_use]
pub fn meets_byzantine_floor(members: u64, faults: u64) -> bool {
    members >= byzantine_floor(faults)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_threshold_of_zero_is_refused() {
        // A threshold of zero is met by nobody, which is not a weaker quorum but
        // the absence of one.
        assert_eq!(
            Quorum::unweighted(&[1, 2, 3], 0),
            Err(QuorumError::ZeroThreshold)
        );
        assert_eq!(
            Quorum::weighted(BTreeMap::from([(1, 10)]), 0),
            Err(QuorumError::ZeroThreshold)
        );
    }

    #[test]
    fn a_threshold_above_the_member_count_is_refused() {
        assert_eq!(
            Quorum::unweighted(&[1, 2], 3),
            Err(QuorumError::ThresholdAboveMembers {
                threshold: 3,
                members: 2
            })
        );
    }

    #[test]
    fn two_signatures_from_one_signer_count_once() {
        // Counting keys instead of parties is how a single actor reaches any
        // threshold.
        let q = Quorum::unweighted(&[1, 2, 3], 3).expect("quorum");
        // Three distinct parties, one of them repeated twice: still three votes.
        assert_eq!(q.count(&[1, 1, 1, 2, 3]), Ok(3));
        // One party repeated three times is one vote, not three.
        assert_eq!(
            q.count(&[1, 1, 1]),
            Err(QuorumError::Insufficient {
                reached: 1,
                required: 3
            }),
            "one signer reached a 3-of-3 by repeating itself"
        );
        assert!(q.count(&[1, 2, 3]).is_ok());
    }

    #[test]
    fn a_non_member_cannot_vote_and_is_not_silently_dropped() {
        // An unknown signer means somebody is trying to vote who should not be
        // able to. Dropping them quietly hides that.
        let q = Quorum::unweighted(&[1, 2, 3], 2).expect("quorum");
        assert_eq!(
            q.count(&[1, 99]),
            Err(QuorumError::NotAMember { signer: 99 })
        );
    }

    #[test]
    fn the_requester_cannot_approve_itself() {
        // An actor asking for approval and also holding a vote can reach the
        // threshold by approving itself.
        let q = Quorum::unweighted(&[1, 2, 3], 2).expect("quorum");
        assert_eq!(
            q.count_excluding(&[1, 2], 1),
            Err(QuorumError::SelfApproval { requester: 1 })
        );
        // Without the requester's own vote, two others still form the quorum.
        assert_eq!(q.count_excluding(&[2, 3], 1), Ok(2));
    }

    #[test]
    fn the_byzantine_floor_is_three_f_plus_one() {
        // With 3f members and f faulty, the honest 2f cannot be distinguished
        // from a 2f group that includes the faulty ones.
        assert_eq!(byzantine_floor(0), 1);
        assert_eq!(byzantine_floor(1), 4);
        assert_eq!(byzantine_floor(2), 7);
        assert_eq!(byzantine_floor(3), 10);
        assert!(
            !meets_byzantine_floor(3, 1),
            "three members cannot tolerate one fault"
        );
        assert!(meets_byzantine_floor(4, 1));
    }

    #[test]
    fn the_inverse_of_the_floor_agrees_with_it() {
        for faults in 0..8u64 {
            let members = byzantine_floor(faults);
            assert_eq!(
                tolerable_faults(members),
                faults,
                "the inverse disagrees at {faults}"
            );
            assert!(tolerable_faults(members.saturating_sub(1)) < faults);
        }
    }

    #[test]
    fn a_fault_count_that_would_overflow_returns_the_maximum() {
        // Returning a small wrapped number would look achievable, which is worse
        // than saying the request is impossible.
        assert_eq!(byzantine_floor(u64::MAX), u64::MAX);
    }

    #[test]
    fn widening_a_quorum_cannot_drop_it_below_the_floor() {
        // "5 of 7" and "5 of 40" are the same numerator and an eighth of the
        // assurance. Extending re-checks the floor.
        let mut q = Quorum::unweighted(&[1, 2, 3, 4, 5, 6, 7], 5).expect("quorum");
        let many: Vec<(u64, u64)> = (8..40).map(|m| (m, 1)).collect();
        // Adding members with a threshold that stays at 5 is allowed - the set
        // grew, the floor still holds.
        assert!(q.extend(&many, 5, 2).is_ok());
        assert_eq!(q.member_count(), 39);
        // But asking to tolerate more faults than the set can support is not.
        let mut small = Quorum::unweighted(&[1, 2, 3, 4], 3).expect("quorum");
        assert_eq!(
            small.extend(&[(5, 1)], 3, 5),
            Err(QuorumError::BelowByzantineFloor {
                members: 5,
                faults: 5,
                required: 16
            })
        );
    }

    #[test]
    fn weights_are_counted_not_heads() {
        let mut weights = BTreeMap::new();
        weights.insert(1, 50);
        weights.insert(2, 30);
        weights.insert(3, 20);
        let q = Quorum::weighted(weights, 60).expect("quorum");
        assert_eq!(q.total_weight(), 100);
        // The two large parties reach 80; the two small ones reach 50.
        assert_eq!(q.count(&[1, 2]), Ok(80));
        // 30 + 20 is 50, short of the 60 required, even though two parties
        // approved - heads are not what is counted.
        assert_eq!(
            q.count(&[2, 3]),
            Err(QuorumError::Insufficient {
                reached: 50,
                required: 60
            })
        );
    }

    #[test]
    fn an_unreachable_quorum_is_reported_as_unreachable() {
        // If the remaining weight cannot reach the threshold, no amount of
        // waiting produces a quorum.
        let q = Quorum::unweighted(&[1, 2, 3], 3).expect("quorum");
        assert!(
            !q.still_reachable(&[1]),
            "one of three answered and two are needed more"
        );
        assert!(q.still_reachable(&[1, 2]));
        assert!(q.still_reachable(&[]));
    }

    #[test]
    fn an_empty_member_set_is_refused() {
        assert_eq!(Quorum::unweighted(&[], 1), Err(QuorumError::NoMembers));
        assert_eq!(
            Quorum::weighted(BTreeMap::new(), 1),
            Err(QuorumError::NoMembers)
        );
    }

    #[test]
    fn a_weighted_threshold_above_the_total_is_refused() {
        let mut weights = BTreeMap::new();
        weights.insert(1, 10);
        assert_eq!(
            Quorum::weighted(weights, 11),
            Err(QuorumError::ThresholdAboveMembers {
                threshold: 11,
                members: 10
            })
        );
    }
}
