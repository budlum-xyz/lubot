//! Evidence, then a finding, then a path - and nothing closes without all three.
//!
//! A review that ends in a sentence is a review that ended somewhere else. The
//! failure this crate exists for is not a wrong conclusion; it is a conclusion
//! that *reads* correctly, gets written into a report, and is never checked
//! against the code it describes. The way that happens is short: someone
//! describes a hole, someone else writes the description into the file, and the
//! description is mistaken for the fix.
//!
//! # The chain
//!
//! ```text
//! observe  ->  claim  ->  support  ->  path  ->  run  ->  close
//! (bytes)    (a finding) (links)     (an act)  (output)  (rules)
//! ```
//!
//! Each arrow is a rule here rather than a convention:
//!
//! * **observe** produces evidence only about a target the session was
//!   authorized for, and stores a digest of what was seen, not a paraphrase.
//! * **support** links evidence to a finding; out-of-scope evidence is refused
//!   instead of being quietly useful.
//! * **path** records the check that would settle the claim *before* it runs,
//!   so "the output looked fine" cannot be decided afterwards.
//! * **close** requires at least one piece of evidence that can close a claim
//!   about behaviour. [`EvidenceKind::Narrative`] cannot, which is the whole
//!   point of keeping it in the enum: prose is recorded, counted, and refused.
//!
//! # Why the scope gate is first
//!
//! An agent that can look at anything will look at anything. The chain therefore
//! opens from a [`Scope`], and a plan must be recorded before any observation is
//! admissible: evidence gathered before the plan exists is a walk through a
//! repository, not an audit of one.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// A cheap, deterministic content digest: FNV-1a over the bytes, 64 bits.
///
/// It is not a security primitive. It is a way to say "the bytes that were
/// produced" without storing them, and to compare two claims about output
/// without re-reading either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest(
    /// The 64 bits, little-endian, in the order FNV produced them.
    pub [u8; 8],
);

impl Digest {
    /// The digest of some text.
    #[must_use]
    pub fn of(text: &str) -> Self {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in text.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self(hash.to_le_bytes())
    }

    /// Lowercase hex, for messages and for the table in a report.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut out = String::with_capacity(16);
        for byte in self.0 {
            out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
            out.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
        }
        out
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hex())
    }
}

/// What was seen. The kinds differ in one respect that matters: whether the
/// observation can settle a claim about behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceKind {
    /// A test or check that ran, with its output.
    ObservedRun,
    /// A build or compile step, with its exit status.
    ObservedBuild,
    /// A gate or scan whose output was read.
    ObservedGate,
    /// A number measured from the tree or the run.
    MeasuredNumber,
    /// A description. Real, useful, and never a closure.
    Narrative,
}

impl EvidenceKind {
    /// Can this kind close a claim about what the code does?
    #[must_use]
    pub fn closes(self) -> bool {
        !matches!(self, Self::Narrative)
    }
}

/// Why a scope refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeError {
    /// The target was never authorized.
    NotAuthorized {
        /// What was asked for.
        target: String,
    },
    /// No plan has been recorded, so nothing observed counts as an audit.
    NoPlan,
    /// The scope was closed after the target was authorized.
    Closed,
}

impl std::fmt::Display for ScopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAuthorized { target } => write!(f, "`{target}` is outside the scope"),
            Self::NoPlan => write!(
                f,
                "no plan is recorded; observing before deciding what to look for is a \
                 walk through a repository"
            ),
            Self::Closed => write!(f, "the scope is closed"),
        }
    }
}

impl std::error::Error for ScopeError {}

/// Which targets a session may look at, and whether it has said what it is
/// looking for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scope {
    authorized: BTreeSet<String>,
    plan: Option<String>,
    closed: bool,
}

impl Scope {
    /// An empty scope: nothing authorized, no plan.
    #[must_use]
    pub fn open() -> Self {
        Self::default()
    }

    /// Adds a target.
    pub fn authorize(&mut self, target: &str) {
        if !self.closed {
            self.authorized.insert(target.to_string());
        }
    }

    /// Records the plan. Returns false when one was already recorded, because a
    /// plan rewritten after the first observation is not a plan.
    pub fn record_plan(&mut self, plan: &str) -> bool {
        if self.closed || self.plan.is_some() {
            return false;
        }
        self.plan = Some(plan.to_string());
        true
    }

    /// True when `target` is inside the authorized set.
    #[must_use]
    pub fn is_authorized(&self, target: &str) -> bool {
        !self.closed && self.authorized.iter().any(|t| target.starts_with(t.as_str()))
    }

    /// True when the session may act.
    #[must_use]
    pub fn may_act(&self) -> bool {
        !self.closed && self.plan.is_some() && !self.authorized.is_empty()
    }

    /// Closes the scope: no further observation is admissible.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Checks one act against the scope.
    ///
    /// # Errors
    ///
    /// The first [`ScopeError`] found.
    pub fn check(&self, target: &str) -> Result<(), ScopeError> {
        if self.closed {
            return Err(ScopeError::Closed);
        }
        if self.plan.is_none() {
            return Err(ScopeError::NoPlan);
        }
        if !self.is_authorized(target) {
            return Err(ScopeError::NotAuthorized {
                target: target.to_string(),
            });
        }
        Ok(())
    }
}

/// A piece of evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    id: u32,
    kind: EvidenceKind,
    subject: String,
    digest: Digest,
    inside_scope: bool,
}

impl Evidence {
    /// Its identity inside one chain.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// What kind of observation it is.
    #[must_use]
    pub fn kind(&self) -> EvidenceKind {
        self.kind
    }

    /// What it was about.
    #[must_use]
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// The bytes it saw, as a digest.
    #[must_use]
    pub fn digest(&self) -> Digest {
        self.digest
    }

    /// Whether it was gathered inside the scope that admitted it.
    #[must_use]
    pub fn inside_scope(&self) -> bool {
        self.inside_scope
    }
}

/// A claim about the code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    id: u32,
    claim: String,
    supports: Vec<u32>,
    outcome: Option<Outcome>,
    expectation: Expectation,
    action: Option<String>,
}

/// What a path promises to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expectation {
    /// Any output at all: the claim is about existence, not value.
    AnyRun,
    /// The exact bytes, by digest: the claim was decided before the run.
    DigestIs(Digest),
}

/// What a check produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// It said what it was going to say.
    Matched,
    /// It said something else.
    Mismatched {
        /// Digest actually produced.
        got: Digest,
    },
}

/// Where a finding stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Claimed, not supported.
    Claimed,
    /// Supported by evidence that cannot close it.
    Described,
    /// Supported and awaiting its check.
    AwaitingCheck,
    /// Closed: supported by observation, checked, inside scope.
    Closed,
    /// The check ran and disagreed.
    Refuted,
}

/// Why the chain refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// No such finding.
    UnknownFinding(u32),
    /// No such evidence.
    UnknownEvidence(u32),
    /// The evidence was gathered outside the scope.
    OutOfScope {
        /// The evidence in question.
        evidence: u32,
        /// What it was about.
        subject: String,
    },
    /// The finding's only support is a description.
    NarrativeOnly {
        /// The finding.
        finding: u32,
    },
    /// The path was opened but never run.
    CheckNotRun(u32),
    /// The path ran and the output disagreed with the promise.
    CheckMismatch {
        /// The finding.
        finding: u32,
        /// What was produced.
        got: Digest,
    },
    /// Already closed; reopening is a different act than continuing.
    AlreadyClosed(u32),
    /// No action is admissible before the plan exists.
    ActWithoutPlan,
    /// The target is outside the scope.
    OutsideScope(String),
    /// A closed finding lost the properties that closed it.
    ClosureInvalid {
        /// The finding.
        finding: u32,
        /// Which rule.
        rule: &'static str,
    },
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFinding(id) => write!(f, "no finding {id}"),
            Self::UnknownEvidence(id) => write!(f, "no evidence {id}"),
            Self::OutOfScope { evidence, subject } => {
                write!(f, "evidence {evidence} is about `{subject}`, outside the scope")
            }
            Self::NarrativeOnly { finding } => write!(
                f,
                "finding {finding} is supported only by prose; a description of a hole is \
                 not the code that closes it"
            ),
            Self::CheckNotRun(id) => {
                write!(f, "finding {id} promised a check that never ran")
            }
            Self::CheckMismatch { finding, got } => {
                write!(f, "finding {finding}: the check produced {got}, not the promised digest")
            }
            Self::AlreadyClosed(id) => write!(f, "finding {id} is already closed"),
            Self::ActWithoutPlan => write!(
                f,
                "nothing may be observed before the plan is recorded: an audit without a \
                 question is a tour"
            ),
            Self::OutsideScope(target) => write!(f, "`{target}` is not a target of this scope"),
            Self::ClosureInvalid { finding, rule } => {
                write!(f, "closure of finding {finding} no longer holds: {rule}")
            }
        }
    }
}

impl std::error::Error for ChainError {}

/// The chain itself.
#[derive(Debug, Clone)]
pub struct Chain {
    scope: Scope,
    evidence: BTreeMap<u32, Evidence>,
    findings: BTreeMap<u32, Finding>,
    closed: BTreeSet<u32>,
    next_evidence: u32,
    next_finding: u32,
}

impl Chain {
    /// A chain over a scope.
    #[must_use]
    pub fn new(scope: Scope) -> Self {
        Self {
            scope,
            evidence: BTreeMap::new(),
            findings: BTreeMap::new(),
            closed: BTreeSet::new(),
            next_evidence: 1,
            next_finding: 1,
        }
    }

    /// The scope in force.
    #[must_use]
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Records the plan. Must happen before the first observation.
    pub fn plan(&mut self, plan: &str) -> bool {
        self.scope.record_plan(plan)
    }

    /// Authorizes a target.
    pub fn authorize(&mut self, target: &str) {
        self.scope.authorize(target);
    }

    /// Observes bytes about a subject.
    ///
    /// # Errors
    ///
    /// [`ChainError::ActWithoutPlan`] before the plan exists, and
    /// [`ChainError::OutsideScope`] for an unauthorized subject.
    pub fn observe(
        &mut self,
        kind: EvidenceKind,
        subject: &str,
        text: &str,
    ) -> Result<u32, ChainError> {
        self.scope.check(subject).map_err(|e| match e {
            ScopeError::NoPlan => ChainError::ActWithoutPlan,
            ScopeError::NotAuthorized { target } => ChainError::OutsideScope(target),
            ScopeError::Closed => ChainError::OutsideScope("<scope closed>".to_string()),
        })?;
        let id = self.next_evidence;
        self.next_evidence += 1;
        self.evidence.insert(
            id,
            Evidence {
                id,
                kind,
                subject: subject.to_string(),
                digest: Digest::of(text),
                inside_scope: true,
            },
        );
        Ok(id)
    }

    /// Claims a finding.
    #[must_use]
    pub fn claim(&mut self, claim: &str) -> u32 {
        let id = self.next_finding;
        self.next_finding += 1;
        self.findings.insert(
            id,
            Finding {
                id,
                claim: claim.to_string(),
                supports: Vec::new(),
                outcome: None,
                expectation: Expectation::AnyRun,
                action: None,
            },
        );
        id
    }

    /// A finding's text.
    #[must_use]
    pub fn claim_of(&self, id: u32) -> Option<&str> {
        self.findings.get(&id).map(|f| f.claim.as_str())
    }

    /// Links evidence to a finding, refusing support gathered outside the scope.
    ///
    /// # Errors
    ///
    /// [`ChainError::UnknownFinding`], [`ChainError::UnknownEvidence`],
    /// [`ChainError::OutOfScope`].
    pub fn support(&mut self, finding: u32, evidence: u32) -> Result<(), ChainError> {
        if !self.findings.contains_key(&finding) {
            return Err(ChainError::UnknownFinding(finding));
        }
        let item = self
            .evidence
            .get(&evidence)
            .ok_or(ChainError::UnknownEvidence(evidence))?;
        if !item.inside_scope() {
            return Err(ChainError::OutOfScope {
                evidence,
                subject: item.subject().to_string(),
            });
        }
        if let Some(record) = self.findings.get_mut(&finding) {
            record.supports.push(evidence);
        }
        Ok(())
    }

    /// Opens the path that would settle the claim: the act, and the check that
    /// decides it, recorded now rather than later.
    ///
    /// # Errors
    ///
    /// [`ChainError::UnknownFinding`] or [`ChainError::AlreadyClosed`].
    pub fn open_path(
        &mut self,
        finding: u32,
        action: &str,
        expectation: Expectation,
    ) -> Result<(), ChainError> {
        let record = self
            .findings
            .get_mut(&finding)
            .ok_or(ChainError::UnknownFinding(finding))?;
        if self.closed.contains(&finding) {
            return Err(ChainError::AlreadyClosed(finding));
        }
        record.action = Some(action.to_string());
        record.expectation = expectation;
        record.outcome = None;
        Ok(())
    }

    /// Runs the promised check over some output.
    ///
    /// # Errors
    ///
    /// [`ChainError::UnknownFinding`] or [`ChainError::CheckNotRun`] when no
    /// path was opened.
    pub fn run(&mut self, finding: u32, output: &str) -> Result<Outcome, ChainError> {
        let record = self
            .findings
            .get(&finding)
            .ok_or(ChainError::UnknownFinding(finding))?;
        if record.action.is_none() {
            return Err(ChainError::CheckNotRun(finding));
        }
        let got = Digest::of(output);
        let outcome = match &record.expectation {
            Expectation::AnyRun => Outcome::Matched,
            Expectation::DigestIs(want) => {
                if *want == got {
                    Outcome::Matched
                } else {
                    Outcome::Mismatched { got }
                }
            }
        };
        if let Some(record) = self.findings.get_mut(&finding) {
            record.outcome = Some(outcome);
        }
        Ok(outcome)
    }

    /// Closes a finding, if the rules are satisfied.
    ///
    /// # Errors
    ///
    /// [`ChainError::UnknownFinding`], [`ChainError::CheckNotRun`],
    /// [`ChainError::CheckMismatch`], [`ChainError::NarrativeOnly`],
    /// [`ChainError::AlreadyClosed`].
    pub fn close(&mut self, finding: u32) -> Result<(), ChainError> {
        let record = self
            .findings
            .get(&finding)
            .ok_or(ChainError::UnknownFinding(finding))?;
        if self.closed.contains(&finding) {
            return Err(ChainError::AlreadyClosed(finding));
        }
        let closing = record
            .supports
            .iter()
            .filter_map(|id| self.evidence.get(id))
            .any(|item| item.kind().closes());
        if !closing {
            return Err(ChainError::NarrativeOnly { finding });
        }
        if record.action.is_none() {
            return Err(ChainError::CheckNotRun(finding));
        }
        match record.outcome {
            Some(Outcome::Matched) => {}
            Some(Outcome::Mismatched { got }) => {
                return Err(ChainError::CheckMismatch { finding, got })
            }
            None => return Err(ChainError::CheckNotRun(finding)),
        }
        self.closed.insert(finding);
        Ok(())
    }

    /// A finding's status, derived rather than stored.
    #[must_use]
    pub fn status(&self, finding: u32) -> Option<Status> {
        let record = self.findings.get(&finding)?;
        if self.closed.contains(&finding) {
            return Some(Status::Closed);
        }
        if matches!(record.outcome, Some(Outcome::Mismatched { .. })) {
            return Some(Status::Refuted);
        }
        if record.outcome == Some(Outcome::Matched) {
            return Some(Status::AwaitingCheck);
        }
        if record
            .supports
            .iter()
            .filter_map(|id| self.evidence.get(id))
            .any(|item| item.kind().closes())
        {
            return Some(Status::AwaitingCheck);
        }
        if record.supports.is_empty() {
            return Some(Status::Claimed);
        }
        Some(Status::Described)
    }

    /// True when the finding's only support is prose. Reported separately,
    /// because "described, not shown" is the finding's own finding.
    #[must_use]
    pub fn prose_only(&self, finding: u32) -> bool {
        let Some(record) = self.findings.get(&finding) else {
            return false;
        };
        !record.supports.is_empty()
            && !record
                .supports
                .iter()
                .filter_map(|id| self.evidence.get(id))
                .any(|item| item.kind().closes())
    }

    /// Findings closed without prose-only support, for a report.
    #[must_use]
    pub fn closed(&self) -> Vec<u32> {
        self.closed.iter().copied().collect()
    }

    /// Findings that are described but not shown.
    #[must_use]
    pub fn described_only(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for id in self.findings.keys() {
            if self.prose_only(*id) && !self.closed.contains(id) {
                out.push(*id);
            }
        }
        out
    }

    /// Re-checks every closure against the rules, so a closure cannot survive
    /// the removal of what justified it.
    ///
    /// # Errors
    ///
    /// The first [`ChainError::ClosureInvalid`] found.
    pub fn verify(&self) -> Result<(), ChainError> {
        for id in &self.closed {
            let Some(record) = self.findings.get(id) else {
                return Err(ChainError::UnknownFinding(*id));
            };
            let closing = record
                .supports
                .iter()
                .filter_map(|e| self.evidence.get(e))
                .any(|item| item.kind().closes() && item.inside_scope());
            if !closing {
                return Err(ChainError::ClosureInvalid {
                    finding: *id,
                    rule: "at least one piece of closing evidence",
                });
            }
            if record.outcome != Some(Outcome::Matched) {
                return Err(ChainError::ClosureInvalid {
                    finding: *id,
                    rule: "the promised check ran and matched",
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scoped() -> Chain {
        let mut scope = Scope::open();
        scope.authorize("src/storage");
        let mut chain = Chain::new(scope);
        assert!(chain.plan("check the repair path"));
        chain
    }

    #[test]
    fn nothing_is_observed_before_the_plan_exists() {
        let mut scope = Scope::open();
        scope.authorize("src");
        let mut chain = Chain::new(scope);
        assert_eq!(
            chain.observe(EvidenceKind::ObservedRun, "src/a.rs", "ok"),
            Err(ChainError::ActWithoutPlan)
        );
        assert!(chain.plan("anything"));
        assert!(chain.observe(EvidenceKind::ObservedRun, "src/a.rs", "ok").is_ok());
    }

    #[test]
    fn a_target_outside_the_scope_is_refused_not_useful() {
        let mut chain = scoped();
        assert_eq!(
            chain.observe(EvidenceKind::ObservedGate, "src/consensus/poa.rs", "ok"),
            Err(ChainError::OutsideScope("src/consensus/poa.rs".to_string()))
        );
    }

    #[test]
    fn a_plan_is_recorded_once() {
        let mut chain = scoped();
        assert!(!chain.plan("rewritten after the first look"));
    }

    #[test]
    fn prose_cannot_close_a_claim() {
        let mut chain = scoped();
        let note = chain
            .observe(EvidenceKind::Narrative, "src/storage/deal.rs", "looks wrong")
            .unwrap();
        let finding = chain.claim("repair never fires");
        chain.support(finding, note).unwrap();
        assert!(chain.prose_only(finding));
        assert_eq!(chain.status(finding), Some(Status::Described));
        chain
            .open_path(finding, "run the repair test", Expectation::AnyRun)
            .unwrap();
        chain.run(finding, "1 passed").unwrap();
        assert_eq!(
            chain.close(finding),
            Err(ChainError::NarrativeOnly { finding })
        );
        assert_eq!(chain.described_only(), vec![finding]);
    }

    #[test]
    fn an_observation_and_a_matching_check_close_it() {
        let mut chain = scoped();
        let run = chain
            .observe(EvidenceKind::ObservedRun, "src/storage/deal.rs", "0 passed")
            .unwrap();
        let finding = chain.claim("repair never fires");
        chain.support(finding, run).unwrap();
        chain
            .open_path(
                finding,
                "add the ticket and re-run",
                Expectation::DigestIs(Digest::of("1 passed")),
            )
            .unwrap();
        assert_eq!(
            chain.run(finding, "1 passed").unwrap(),
            Outcome::Matched
        );
        chain.close(finding).unwrap();
        assert_eq!(chain.status(finding), Some(Status::Closed));
        assert_eq!(chain.verify(), Ok(()));
    }

    #[test]
    fn a_check_that_says_something_else_does_not_close_it() {
        let mut chain = scoped();
        let run = chain
            .observe(EvidenceKind::ObservedBuild, "src/storage/deal.rs", "built")
            .unwrap();
        let finding = chain.claim("the fix compiles");
        chain.support(finding, run).unwrap();
        chain
            .open_path(
                finding,
                "build",
                Expectation::DigestIs(Digest::of("built clean")),
            )
            .unwrap();
        let outcome = chain.run(finding, "error[E0308]").unwrap();
        assert!(matches!(outcome, Outcome::Mismatched { .. }));
        assert!(matches!(
            chain.close(finding),
            Err(ChainError::CheckMismatch { .. })
        ));
    }

    #[test]
    fn a_promise_of_a_check_is_not_a_check() {
        let mut chain = scoped();
        let gate = chain
            .observe(EvidenceKind::ObservedGate, "src/storage/deal.rs", "clean")
            .unwrap();
        let finding = chain.claim("gate passes after the change");
        chain.support(finding, gate).unwrap();
        assert_eq!(chain.close(finding), Err(ChainError::CheckNotRun(finding)));
    }

    #[test]
    fn evidence_from_outside_the_scope_cannot_support_a_finding() {
        let mut chain = scoped();
        let outside = Evidence {
            id: 99,
            kind: EvidenceKind::ObservedRun,
            subject: "src/consensus/poa.rs".to_string(),
            digest: Digest::of("x"),
            inside_scope: false,
        };
        chain.evidence.insert(99, outside);
        let finding = chain.claim("borrowed evidence");
        assert_eq!(
            chain.support(finding, 99),
            Err(ChainError::OutOfScope {
                evidence: 99,
                subject: "src/consensus/poa.rs".to_string()
            })
        );
    }

    #[test]
    fn closing_twice_is_a_different_act_than_continuing() {
        let mut chain = scoped();
        let run = chain
            .observe(EvidenceKind::MeasuredNumber, "src/storage/deal.rs", "42")
            .unwrap();
        let finding = chain.claim("the count is 42");
        chain.support(finding, run).unwrap();
        chain
            .open_path(finding, "count", Expectation::AnyRun)
            .unwrap();
        chain.run(finding, "42").unwrap();
        chain.close(finding).unwrap();
        assert_eq!(chain.close(finding), Err(ChainError::AlreadyClosed(finding)));
        assert!(matches!(
            chain.open_path(finding, "again", Expectation::AnyRun),
            Err(ChainError::AlreadyClosed(_))
        ));
    }

    #[test]
    fn a_closure_that_loses_its_evidence_stops_being_a_closure() {
        let mut chain = scoped();
        let run = chain
            .observe(EvidenceKind::ObservedRun, "src/storage/deal.rs", "ok")
            .unwrap();
        let finding = chain.claim("it works");
        chain.support(finding, run).unwrap();
        chain
            .open_path(finding, "run", Expectation::AnyRun)
            .unwrap();
        chain.run(finding, "ok").unwrap();
        chain.close(finding).unwrap();
        assert_eq!(chain.verify(), Ok(()));
        let mut broken = chain.clone();
        broken
            .findings
            .get_mut(&finding)
            .expect("present")
            .supports
            .clear();
        assert!(matches!(
            broken.verify(),
            Err(ChainError::ClosureInvalid { .. })
        ));
    }

    #[test]
    fn digests_are_stable_and_distinguish_content() {
        assert_eq!(Digest::of("same"), Digest::of("same"));
        assert_ne!(Digest::of("same"), Digest::of("same "));
        assert_eq!(Digest::of("abc").hex().len(), 16);
    }

    #[test]
    fn unknown_ids_are_refused_rather_than_ignored() {
        let mut chain = scoped();
        assert_eq!(chain.support(1234, 1), Err(ChainError::UnknownFinding(1234)));
        let finding = chain.claim("x");
        assert_eq!(chain.support(finding, 4321), Err(ChainError::UnknownEvidence(4321)));
    }
}
