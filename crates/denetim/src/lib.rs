//! Security review discipline for a change set.
//!
//! Models the review cycle (scan, then validate, then fix) as a *checkable
//! ledger* rather than as a scanner: findings are recorded, and every
//! finding must end in an evidenced disposition. Nothing closes without
//! evidence, a serious finding cannot be waived without an attestation, and
//! a finding that changes on re-scan loses any stale closure it carried.
//!
//! The ledger is a pure, deterministic invariant: what it accepts, it can
//! check. The scanning tooling that feeds it lives outside this crate.

use std::collections::BTreeMap;
use std::fmt;

/// How serious a finding is. The order is part of the discipline: the
/// attestation floor sits at [`Severity::High`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// A finding as recorded from a scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Stable, non-empty identifier (how the ledger addresses the finding).
    pub id: String,
    /// Seriousness; re-recording with a different severity is a *changed*
    /// finding.
    pub severity: Severity,
    /// What the finding says; non-empty, and part of the fingerprint.
    pub detail: String,
}

/// Evidence that a finding was actually fixed: the reference to the fix and
/// the reference to the check that proves it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixEvidence {
    /// The reference of the verifying check (a test or gate name).
    pub verification: String,
    /// The reference of the fix itself (a commit).
    pub commit: String,
}

/// The case for why a finding does not need fixing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Waiver {
    /// Why the finding does not apply; always non-empty.
    pub reason: String,
    /// Who stands behind the waiver; required from [`Severity::High`] up.
    pub attester: String,
}

/// How a finding was closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposition {
    /// Fixed, with the evidence that proves it.
    Fixed(FixEvidence),
    /// Rejected after review, with the waiver carrying the case.
    Rejected(Waiver),
}

impl Disposition {
    /// Whether the closure is a fix (as opposed to a waiver).
    #[must_use]
    pub fn is_fix(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }
}

/// The state of one finding in the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Recorded, not yet dispositioned.
    Open,
    /// Closed by a [`Disposition`].
    Closed,
}

/// Why a ledger operation is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewError {
    /// A finding was recorded with an empty id or an empty detail.
    EmptyFinding,
    /// A disposition was requested for an id the ledger does not know.
    UnknownFinding,
    /// A fix was claimed with an empty verification or an empty commit.
    EmptyEvidence,
    /// A waiver was given with an empty reason.
    EmptyWaiver,
    /// A finding at the attestation floor or above was waived without an
    /// attester.
    AttesterRequired,
}

impl fmt::Display for ReviewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFinding => write!(f, "a finding needs a non-empty id and detail"),
            Self::UnknownFinding => write!(f, "the ledger does not know this finding"),
            Self::EmptyEvidence => write!(f, "a fix needs both a verification and a commit"),
            Self::EmptyWaiver => write!(f, "a waiver needs a reason"),
            Self::AttesterRequired => {
                write!(f, "a serious waiver needs an attester")
            }
        }
    }
}

impl std::error::Error for ReviewError {}

/// Findings that were recorded but never dispositioned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    /// The open finding ids, in ledger order.
    pub open: Vec<String>,
}

impl fmt::Display for Pending {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "review incomplete: {} finding(s) still open",
            self.open.len()
        )
    }
}

impl std::error::Error for Pending {}

/// The finished ledger: every finding dispositioned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// Findings closed as fixed.
    pub fixed: usize,
    /// Findings closed as waived.
    pub waived: usize,
    /// The ids of waived findings at the attestation floor or above — the
    /// auditable residue a reviewer re-reads first.
    pub attested_waivers: Vec<String>,
}

/// The review ledger: scan -> validate -> fix as state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Review {
    /// Every finding recorded so far, keyed by id.
    pub findings: BTreeMap<String, Finding>,
    /// The closure of the findings that were closed, keyed by id. A recorded
    /// finding without an entry here is open.
    pub dispositions: BTreeMap<String, Disposition>,
}

impl Review {
    /// An empty ledger: nothing scanned yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            findings: BTreeMap::new(),
            dispositions: BTreeMap::new(),
        }
    }

    /// Record a finding (the scan step).
    ///
    /// Re-recording an unchanged finding is a no-op. Re-recording a finding
    /// whose severity or detail changed *voids* any closure it carried and
    /// returns the voided disposition: the changed thing was never reviewed,
    /// so its old closure must not be inherited.
    ///
    /// # Errors
    ///
    /// [`ReviewError::EmptyFinding`] for an empty id or an empty detail.
    pub fn record(&mut self, finding: Finding) -> Result<Option<Disposition>, ReviewError> {
        if finding.id.is_empty() || finding.detail.is_empty() {
            return Err(ReviewError::EmptyFinding);
        }
        let known = self
            .findings
            .get(&finding.id)
            .map(|old| old.severity == finding.severity && old.detail == finding.detail)
            .unwrap_or(false);
        if !known {
            // New or changed: a changed finding loses any stale closure.
            let voided = self.dispositions.remove(&finding.id);
            self.findings.insert(finding.id.clone(), finding);
            return Ok(voided);
        }
        Ok(None)
    }

    /// Close a finding as fixed (the fix step), with the evidence that
    /// proves it.
    ///
    /// # Errors
    ///
    /// [`ReviewError::UnknownFinding`] for an unrecorded id, or
    /// [`ReviewError::EmptyEvidence`] when either evidence field is empty.
    pub fn fix(&mut self, id: &str, evidence: FixEvidence) -> Result<(), ReviewError> {
        if !self.findings.contains_key(id) {
            return Err(ReviewError::UnknownFinding);
        }
        if evidence.verification.is_empty() || evidence.commit.is_empty() {
            return Err(ReviewError::EmptyEvidence);
        }
        self.dispositions
            .insert(id.to_owned(), Disposition::Fixed(evidence));
        Ok(())
    }

    /// Close a finding as waived (the validate step), with the case.
    ///
    /// # Errors
    ///
    /// [`ReviewError::UnknownFinding`] for an unrecorded id,
    /// [`ReviewError::EmptyWaiver`] for an empty reason, or
    /// [`ReviewError::AttesterRequired`] when a finding at
    /// [`Severity::High`] or above is waived without an attester.
    pub fn reject(&mut self, id: &str, waiver: Waiver) -> Result<(), ReviewError> {
        let Some(finding) = self.findings.get(id) else {
            return Err(ReviewError::UnknownFinding);
        };
        if waiver.reason.is_empty() {
            return Err(ReviewError::EmptyWaiver);
        }
        if finding.severity >= Severity::High && waiver.attester.is_empty() {
            return Err(ReviewError::AttesterRequired);
        }
        self.dispositions
            .insert(id.to_owned(), Disposition::Rejected(waiver));
        Ok(())
    }

    /// The current status of one finding.
    #[must_use]
    pub fn status(&self, id: &str) -> Option<Status> {
        self.findings.get(id).map(|_| {
            if self.dispositions.contains_key(id) {
                Status::Closed
            } else {
                Status::Open
            }
        })
    }

    /// The ids of every finding that is still open, in ledger order.
    #[must_use]
    pub fn open_ids(&self) -> Vec<String> {
        self.findings
            .keys()
            .filter(|id| !self.dispositions.contains_key(*id))
            .cloned()
            .collect()
    }

    /// Finish the review (the gate step).
    ///
    /// # Errors
    ///
    /// [`Pending`] naming every finding that is still open — the ledger
    /// does not finish with anything left in the air.
    pub fn complete(&self) -> Result<Report, Pending> {
        let open = self.open_ids();
        if !open.is_empty() {
            return Err(Pending { open });
        }
        let mut report = Report::default();
        for (id, disposition) in &self.dispositions {
            match disposition {
                Disposition::Fixed(_) => report.fixed += 1,
                Disposition::Rejected(_) => {
                    report.waived += 1;
                    if self.findings[id].severity >= Severity::High {
                        report.attested_waivers.push(id.clone());
                    }
                }
            }
        }
        Ok(report)
    }

    /// Re-verify the structural invariants of the ledger (a canary target
    /// for the repository gate).
    ///
    /// # Errors
    ///
    /// The first violated invariant: a fix with an empty evidence field, a
    /// waiver with an empty reason, or a serious waiver without an attester.
    pub fn verify(&self) -> Result<(), ReviewError> {
        for (id, disposition) in &self.dispositions {
            let severity = self
                .findings
                .get(id)
                .map(|finding| finding.severity)
                .unwrap_or(Severity::Critical);
            match disposition {
                Disposition::Fixed(evidence) => {
                    if evidence.verification.is_empty() || evidence.commit.is_empty() {
                        return Err(ReviewError::EmptyEvidence);
                    }
                }
                Disposition::Rejected(waiver) => {
                    if waiver.reason.is_empty() {
                        return Err(ReviewError::EmptyWaiver);
                    }
                    if severity >= Severity::High && waiver.attester.is_empty() {
                        return Err(ReviewError::AttesterRequired);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(id: &str, severity: Severity, detail: &str) -> Finding {
        Finding {
            id: id.to_owned(),
            severity,
            detail: detail.to_owned(),
        }
    }

    fn evidence() -> FixEvidence {
        FixEvidence {
            verification: "gate-secrets".to_owned(),
            commit: "c0ffee".to_owned(),
        }
    }

    #[test]
    fn record_requires_non_empty_finding() {
        let mut review = Review::new();
        assert_eq!(
            review.record(finding("", Severity::Low, "detail")),
            Err(ReviewError::EmptyFinding)
        );
        assert_eq!(
            review.record(finding("f-1", Severity::Low, "")),
            Err(ReviewError::EmptyFinding)
        );
        assert!(review.findings.is_empty());
    }

    #[test]
    fn idempotent_rescan_keeps_disposition() {
        let mut review = Review::new();
        review
            .record(finding("f-1", Severity::Low, "leak in log"))
            .unwrap();
        review.fix("f-1", evidence()).unwrap();

        // The same scan runs again, unchanged: nothing moves.
        assert_eq!(
            review.record(finding("f-1", Severity::Low, "leak in log")),
            Ok(None)
        );
        assert_eq!(review.status("f-1"), Some(Status::Closed));
        assert!(matches!(review.dispositions["f-1"], Disposition::Fixed(_)));
    }

    #[test]
    fn changed_finding_voids_stale_closure() {
        let mut review = Review::new();
        review
            .record(finding("f-1", Severity::Medium, "first form"))
            .unwrap();
        review.fix("f-1", evidence()).unwrap();

        // The scan sees the same finding changed: the old closure is voided
        // and handed back, and the finding is open again.
        let voided = review
            .record(finding("f-1", Severity::High, "same leak, worse path"))
            .unwrap();
        assert!(voided.is_some_and(Disposition::is_fix));
        assert_eq!(review.status("f-1"), Some(Status::Open));
        let err = review.complete().unwrap_err();
        assert_eq!(err.open, vec!["f-1".to_owned()]);
    }

    #[test]
    fn fix_requires_evidence() {
        let mut review = Review::new();
        review
            .record(finding("f-1", Severity::Low, "leak in log"))
            .unwrap();
        assert_eq!(
            review.fix(
                "f-1",
                FixEvidence {
                    verification: String::new(),
                    commit: "c0ffee".to_owned(),
                }
            ),
            Err(ReviewError::EmptyEvidence)
        );
        assert_eq!(
            review.fix(
                "f-1",
                FixEvidence {
                    verification: "gate-secrets".to_owned(),
                    commit: String::new(),
                }
            ),
            Err(ReviewError::EmptyEvidence)
        );
        assert_eq!(review.status("f-1"), Some(Status::Open));
    }

    #[test]
    fn serious_rejection_requires_attestation() {
        let mut review = Review::new();
        review
            .record(finding("low", Severity::Low, "style nit"))
            .unwrap();
        review
            .record(finding("high", Severity::High, "injection path"))
            .unwrap();
        review
            .record(finding("crit", Severity::Critical, "root compromise"))
            .unwrap();

        // Low: the floor is not reached, no attestation needed.
        review
            .reject(
                "low",
                Waiver {
                    reason: "not on the read path".to_owned(),
                    attester: String::new(),
                },
            )
            .unwrap();
        // High: an unattested waiver is refused and the finding stays open.
        assert_eq!(
            review.reject(
                "high",
                Waiver {
                    reason: "no attacker can reach it".to_owned(),
                    attester: String::new(),
                }
            ),
            Err(ReviewError::AttesterRequired)
        );
        assert_eq!(review.status("high"), Some(Status::Open));
        // High with an attester: accepted.
        review
            .reject(
                "high",
                Waiver {
                    reason: "no attacker can reach it".to_owned(),
                    attester: "s-01".to_owned(),
                },
            )
            .unwrap();
        assert_eq!(
            review.reject(
                "crit",
                Waiver {
                    reason: "theoretical".to_owned(),
                    attester: String::new(),
                }
            ),
            Err(ReviewError::AttesterRequired)
        );
    }

    #[test]
    fn complete_lists_every_open_finding() {
        let mut review = Review::new();
        review.record(finding("b", Severity::Low, "x")).unwrap();
        review.record(finding("a", Severity::Low, "y")).unwrap();
        review.record(finding("c", Severity::Medium, "z")).unwrap();
        review.fix("b", evidence()).unwrap();

        let err = review.complete().unwrap_err();
        assert_eq!(err.open, vec!["a".to_owned(), "c".to_owned()]);
    }

    #[test]
    fn complete_reports_attested_waivers() {
        let mut review = Review::new();
        review
            .record(finding("f-1", Severity::Low, "leak in log"))
            .unwrap();
        review
            .record(finding("f-2", Severity::Critical, "root compromise"))
            .unwrap();
        review.fix("f-1", evidence()).unwrap();
        review
            .reject(
                "f-2",
                Waiver {
                    reason: "the path is unreachable on this protocol".to_owned(),
                    attester: "s-02".to_owned(),
                },
            )
            .unwrap();

        let report = review.complete().unwrap();
        assert_eq!(report.fixed, 1);
        assert_eq!(report.waived, 1);
        assert_eq!(report.attested_waivers, vec!["f-2".to_owned()]);
    }

    #[test]
    fn unknown_finding_id_is_an_error() {
        let mut review = Review::new();
        assert_eq!(
            review.fix("ghost", evidence()),
            Err(ReviewError::UnknownFinding)
        );
        assert_eq!(
            review.reject(
                "ghost",
                Waiver {
                    reason: "n/a".to_owned(),
                    attester: "s-01".to_owned(),
                }
            ),
            Err(ReviewError::UnknownFinding)
        );
        assert_eq!(review.status("ghost"), None);
    }

    /// Canary: a disposition mutated after the fact must be caught by
    /// `verify`, not trusted.
    #[test]
    fn verify_catches_mutation() {
        let mut review = Review::new();
        review
            .record(finding("f-1", Severity::Low, "leak in log"))
            .unwrap();
        review.fix("f-1", evidence()).unwrap();
        review.verify().unwrap();

        if let Disposition::Fixed(evidence) = &mut review.dispositions["f-1"] {
            evidence.commit.clear();
        } else {
            panic!("expected a fix");
        }
        assert_eq!(review.verify(), Err(ReviewError::EmptyEvidence));
    }
}
