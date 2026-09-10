//! A skill is a route, not a paragraph.
//!
//! Lubot's skill ledger holds the method cards the agent applies when it
//! audits or changes code. A card is only useful if three things are checkable:
//! *when* it applies, *what it needs* to be run, and *what it must leave behind*
//! when it is done. Each of those is a type here, so a card that claims to
//! apply everywhere, or that promises evidence it never records, fails at the
//! ledger boundary instead of in a reader's surprise.
//!
//! # Why a router instead of a prompt
//!
//! Free text describing "when to use this" cannot be sorted, counted or
//! refused. The same condition written as a [`Trigger`] list can: `route`
//! returns the most specific applicable card first, refuses to invent one when
//! nothing applies, and defers a card whose [`Capability`] the session does not
//! have rather than running it halfway. A deferred run looks exactly like a
//! completed one in a log, which is the failure this prevents.
//!
//! # Name hygiene
//!
//! A card carries the method, and a method that arrived from somewhere else
//! carries an obligation to *not* smuggle that place's name into our public
//! surface: a name reads like an integration, a reader starts reasoning about
//! upgrades to a thing we never link, and the design stops being ours to
//! change. [`Ledger::admit`] therefore refuses any card whose public name or
//! step text contains a token the ledger was told to reject.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// A card's stable identity. Ordering on this value is what makes routing
/// deterministic when two cards are equally specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SkillId(pub u32);

/// What a session must be able to do before a card can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// Read the tree under audit.
    ReadTree,
    /// Write to it.
    WriteTree,
    /// Execute the project's own checks.
    RunChecks,
    /// Reach a network resource.
    Network,
    /// Hold more than one file of context at once.
    WideContext,
}

impl Capability {
    /// Human-readable name, for messages only.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::ReadTree => "read-tree",
            Self::WriteTree => "write-tree",
            Self::RunChecks => "run-checks",
            Self::Network => "network",
            Self::WideContext => "wide-context",
        }
    }
}

/// What a finished card must have produced.
///
/// `Narrative` is a kind on purpose: a written conclusion *is* an artefact, it
/// is just not one that can close a claim about behaviour. Keeping it in the
/// enum, instead of rejecting it at the type level, is what lets the ledger
/// record "the only evidence was prose" as a finding of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceKind {
    /// A test run, with its outcome.
    TestRun,
    /// A build or compile step, with its exit status.
    BuildRun,
    /// A gate or scan whose output was read, not summarised.
    GateRun,
    /// A script that reproduces the claim.
    ReproScript,
    /// Prose. Counts as a note, never as a closure.
    Narrative,
}

impl EvidenceKind {
    /// Can this kind close a claim about what the code does?
    ///
    /// Only observation can. A description of an observation is not one, which
    /// is the distinction the audit trail keeps needing.
    #[must_use]
    pub fn closes(self) -> bool {
        !matches!(self, Self::Narrative)
    }
}

/// One applicability condition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Trigger {
    /// The subject path contains this fragment.
    PathContains(&'static str),
    /// A symbol in the subject starts with this fragment.
    SymbolPrefixed(&'static str),
    /// The subject belongs to this domain.
    DomainIs(&'static str),
    /// The session already holds this capability.
    HasCapability(Capability),
    /// The session already holds evidence of this kind.
    HasEvidence(EvidenceKind),
    /// Applies to anything. Specificity weight zero, so a card with no other
    /// trigger is always routed last - the fallback that exists must be seen
    /// to be a fallback.
    Always,
}

impl Trigger {
    /// Weight used to order cards. A wildcard is worth nothing precisely so
    /// that "we matched it" cannot be inflated into "it applies here".
    #[must_use]
    pub fn weight(&self) -> usize {
        match self {
            Self::Always => 0,
            Self::PathContains(_) | Self::SymbolPrefixed(_) | Self::DomainIs(_) => 2,
            Self::HasCapability(_) | Self::HasEvidence(_) => 1,
        }
    }
}

/// What a card is being asked about right now.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Context {
    path: String,
    domain: String,
    symbols: Vec<String>,
    capabilities: BTreeSet<Capability>,
    evidence: BTreeSet<EvidenceKind>,
}

impl Context {
    /// A context naming one file and the domain it belongs to.
    #[must_use]
    pub fn at(path: &str, domain: &str) -> Self {
        Self {
            path: path.to_string(),
            domain: domain.to_string(),
            symbols: Vec::new(),
            capabilities: BTreeSet::new(),
            evidence: BTreeSet::new(),
        }
    }

    /// Adds a symbol present in the subject.
    #[must_use]
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbols.push(symbol.to_string());
        self
    }

    /// Grants a capability to the session.
    #[must_use]
    pub fn with_capability(mut self, capability: Capability) -> Self {
        self.capabilities.insert(capability);
        self
    }

    /// Records that evidence of this kind already exists.
    #[must_use]
    pub fn with_evidence(mut self, kind: EvidenceKind) -> Self {
        self.evidence.insert(kind);
        self
    }

    /// True when every symbol of the subject is known; cards that inspect
    /// symbols refuse rather than run against an empty list and report success.
    #[must_use]
    pub fn has_symbols(&self) -> bool {
        !self.symbols.is_empty()
    }

    /// The subject path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    fn holds(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }
}

/// A method card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCard {
    id: SkillId,
    name: &'static str,
    applies_when: Vec<Trigger>,
    requires: Vec<Capability>,
    steps: Vec<&'static str>,
    exit_evidence: EvidenceKind,
}

impl SkillCard {
    /// A card with no steps and no requirement; add to those with the builder
    /// methods below.
    #[must_use]
    pub fn new(id: SkillId, name: &'static str, exit_evidence: EvidenceKind) -> Self {
        Self {
            id,
            name,
            applies_when: Vec::new(),
            requires: Vec::new(),
            steps: Vec::new(),
            exit_evidence,
        }
    }

    /// Adds an applicability condition.
    #[must_use]
    pub fn applies_when(mut self, trigger: Trigger) -> Self {
        self.applies_when.push(trigger);
        self
    }

    /// Adds a capability the session must hold.
    #[must_use]
    pub fn requires(mut self, capability: Capability) -> Self {
        if !self.requires.contains(&capability) {
            self.requires.push(capability);
        }
        self
    }

    /// Adds one step of the method.
    #[must_use]
    pub fn step(mut self, text: &'static str) -> Self {
        self.steps.push(text);
        self
    }

    /// The card's identity.
    #[must_use]
    pub fn id(&self) -> SkillId {
        self.id
    }

    /// The card's public name. Nothing else in the tree may refer to it by a
    /// different string, which is why routing reports ids, not names.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// What closing this card must leave behind.
    #[must_use]
    pub fn exit_evidence(&self) -> EvidenceKind {
        self.exit_evidence
    }

    /// The steps, in order.
    #[must_use]
    pub fn steps(&self) -> &[&'static str] {
        &self.steps
    }

    /// The applicability conditions.
    #[must_use]
    pub fn triggers(&self) -> &[Trigger] {
        &self.applies_when
    }

    /// Specificity of this card against `ctx`: the summed weight of the
    /// conditions it satisfies. Zero means it does not apply at all.
    #[must_use]
    pub fn matches(&self, ctx: &Context) -> usize {
        let mut score = 0usize;
        for trigger in &self.applies_when {
            let hit = match trigger {
                Trigger::Always => true,
                Trigger::PathContains(needle) => ctx.path.contains(needle),
                Trigger::SymbolPrefixed(prefix) => {
                    ctx.has_symbols() && ctx.symbols.iter().any(|s| s.starts_with(prefix))
                }
                Trigger::DomainIs(domain) => ctx.domain == *domain,
                Trigger::HasCapability(capability) => ctx.holds(*capability),
                Trigger::HasEvidence(kind) => ctx.evidence.contains(kind),
            };
            if hit {
                score += trigger.weight();
            }
        }
        if self.applies_when.is_empty() {
            return 0;
        }
        score
    }

    /// Capabilities the card needs that `ctx` does not have.
    #[must_use]
    pub fn unmet(&self, ctx: &Context) -> Vec<Capability> {
        self.requires
            .iter()
            .copied()
            .filter(|c| !ctx.holds(*c))
            .collect()
    }
}

/// Why a card cannot be admitted to a ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmitError {
    /// The card names a project we read for method and did not adopt.
    BorrowedName { token: String, field: &'static str },
    /// The card has no applicability condition, so it would match everything.
    Unconditional,
    /// The card closes on prose alone: nothing observable ends it.
    ClosesOnNarrative,
    /// The card has no steps, so running it is indistinguishable from not
    /// running it.
    Stepless,
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BorrowedName { token, field } => write!(
                f,
                "`{token}` appears in the card's {field}; a researched product's name \
                 is not part of our public surface"
            ),
            Self::Unconditional => write!(f, "a card with no trigger matches everything"),
            Self::ClosesOnNarrative => {
                write!(f, "a card that closes on a description closes on nothing")
            }
            Self::Stepless => {
                write!(f, "a card with no steps cannot be distinguished from no card")
            }
        }
    }
}

impl std::error::Error for AdmitError {}

/// Why a recorded piece of evidence was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    /// Nothing is registered under this id.
    Unknown(SkillId),
    /// A promotion threshold of zero would promote on registration.
    ZeroThreshold,
    /// `verify` found a trusted card with no evidence behind it.
    TrustedWithoutEvidence(SkillId),
    /// `verify` found a card still trusted after a contradiction.
    ContradictedWhileTrusted(SkillId),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "no card registered as {id:?}"),
            Self::ZeroThreshold => write!(f, "promotion threshold must be at least one"),
            Self::TrustedWithoutEvidence(id) => {
                write!(f, "{id:?} is trusted but holds no closing evidence")
            }
            Self::ContradictedWhileTrusted(id) => {
                write!(f, "{id:?} is contradicted and must not stay trusted")
            }
        }
    }
}

impl std::error::Error for LedgerError {}

/// Where a card stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    /// Registered, not yet earned.
    Draft,
    /// Enough closing evidence, no open contradiction.
    Trusted,
    /// Contradicted. It stays in the ledger as a negative result: a method that
    /// was measured and failed is knowledge, and deleting it invites a retry.
    Rejected,
}

/// One card's standing in the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    card: SkillCard,
    status: Status,
    closing: u32,
    contradictions: u32,
}

impl Record {
    /// The card itself.
    #[must_use]
    pub fn card(&self) -> &SkillCard {
        &self.card
    }

    /// Its standing.
    #[must_use]
    pub fn status(&self) -> Status {
        self.status
    }

    /// Evidence count that closed a claim.
    #[must_use]
    pub fn closing(&self) -> u32 {
        self.closing
    }

    /// Contradictions recorded against it.
    #[must_use]
    pub fn contradictions(&self) -> u32 {
        self.contradictions
    }
}

/// The ledger: cards, their standing, and the rules that move one to the other.
#[derive(Debug, Clone)]
pub struct Ledger {
    promotion_after: u32,
    banned_tokens: Vec<&'static str>,
    records: BTreeMap<SkillId, Record>,
}

impl Ledger {
    /// A ledger that promotes after `promotion_after` closing evidences.
    ///
    /// A threshold of zero is refused by [`Self::set_promotion_after`]'s rule
    /// applied at construction: it would let a card be trusted before it ran.
    #[must_use]
    pub fn new(promotion_after: u32) -> Self {
        Self {
            promotion_after: promotion_after.max(1),
            banned_tokens: Vec::new(),
            records: BTreeMap::new(),
        }
    }

    /// Refuses cards whose public surface names a project read for method.
    #[must_use]
    pub fn reject_names(mut self, tokens: &[&'static str]) -> Self {
        self.banned_tokens = tokens.to_vec();
        self
    }

    /// The promotion threshold in force.
    #[must_use]
    pub fn promotion_after(&self) -> u32 {
        self.promotion_after
    }

    /// Number of registered cards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True when nothing is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Checks a card against the admission rules and inserts it.
    ///
    /// # Errors
    ///
    /// Returns the first [`AdmitError`] found.
    pub fn admit(&mut self, card: SkillCard) -> Result<(), AdmitError> {
        for token in &self.banned_tokens {
            let lower = token.to_lowercase();
            if card.name.to_lowercase().contains(&lower) {
                return Err(AdmitError::BorrowedName {
                    token: (*token).to_string(),
                    field: "name",
                });
            }
            for step in card.steps() {
                if step.to_lowercase().contains(&lower) {
                    return Err(AdmitError::BorrowedName {
                        token: (*token).to_string(),
                        field: "a step",
                    });
                }
            }
        }
        if card.triggers().is_empty() {
            return Err(AdmitError::Unconditional);
        }
        if !card.exit_evidence().closes() {
            return Err(AdmitError::ClosesOnNarrative);
        }
        if card.steps().is_empty() {
            return Err(AdmitError::Stepless);
        }
        let id = card.id();
        self.records.insert(
            id,
            Record {
                card,
                status: Status::Draft,
                closing: 0,
                contradictions: 0,
            },
        );
        Ok(())
    }

    /// Records an outcome for a card.
    ///
    /// `contradicted` counts against the card whatever its kind: a test that
    /// ran and refuted the method is still closing evidence for the *opposite*
    /// claim, and the card must not keep its standing over it.
    ///
    /// # Errors
    ///
    /// [`LedgerError::Unknown`] when the id was never admitted.
    pub fn record(
        &mut self,
        id: SkillId,
        kind: EvidenceKind,
        contradicted: bool,
    ) -> Result<Status, LedgerError> {
        let threshold = self.promotion_after;
        let record = self
            .records
            .get_mut(&id)
            .ok_or(LedgerError::Unknown(id))?;
        if contradicted {
            record.contradictions += 1;
            record.status = Status::Rejected;
            return Ok(record.status);
        }
        if kind.closes() {
            record.closing += 1;
            if record.status == Status::Draft && record.closing >= threshold {
                record.status = Status::Trusted;
            }
        }
        Ok(record.status)
    }

    /// A card's record.
    #[must_use]
    pub fn record_of(&self, id: SkillId) -> Option<&Record> {
        self.records.get(&id)
    }

    /// Cards a session may run: everything not `Rejected`.
    #[must_use]
    pub fn runnable(&self) -> Vec<SkillId> {
        self.records
            .values()
            .filter(|r| r.status != Status::Rejected)
            .map(|r| r.card.id())
            .collect()
    }

    /// Routes a context: applicable cards first by specificity, then by id.
    ///
    /// Cards whose capabilities the context lacks are returned as deferrals by
    /// [`Self::route_with_deferrals`]; in this list they simply do not appear,
    /// because a route is a plan, not a wish.
    #[must_use]
    pub fn route(&self, ctx: &Context) -> Vec<SkillId> {
        self.route_with_deferrals(ctx).0
    }

    /// The same routing, split into what can run and what is waiting on a
    /// capability.
    #[must_use]
    pub fn route_with_deferrals(
        &self,
        ctx: &Context,
    ) -> (Vec<SkillId>, Vec<(SkillId, Vec<Capability>)>) {
        let mut applied: Vec<(usize, SkillId)> = Vec::new();
        let mut deferred: Vec<(SkillId, Vec<Capability>)> = Vec::new();
        for record in self.records.values() {
            let score = record.card.matches(ctx);
            if score == 0 {
                continue;
            }
            let missing = record.card.unmet(ctx);
            if missing.is_empty() {
                applied.push((score, record.card.id()));
            } else {
                deferred.push((record.card.id(), missing));
            }
        }
        applied.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        (
            applied.into_iter().map(|(_, id)| id).collect(),
            deferred,
        )
    }

    /// Checks the ledger's own invariants, for a CI canary.
    ///
    /// # Errors
    ///
    /// The first [`LedgerError`] found.
    pub fn verify(&self) -> Result<(), LedgerError> {
        if self.promotion_after == 0 {
            return Err(LedgerError::ZeroThreshold);
        }
        for record in self.records.values() {
            let id = record.card.id();
            if record.status == Status::Trusted && record.closing == 0 {
                return Err(LedgerError::TrustedWithoutEvidence(id));
            }
            if record.contradictions > 0 && record.status == Status::Trusted {
                return Err(LedgerError::ContradictedWhileTrusted(id));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(id: u32, name: &'static str) -> SkillCard {
        SkillCard::new(SkillId(id), name, EvidenceKind::TestRun)
            .applies_when(Trigger::PathContains("src/storage"))
            .step("read the production call site")
            .step("run the narrow check, do not summarise it")
    }

    fn ledger() -> Ledger {
        let mut l = Ledger::new(2).reject_names(&["borrowedproject", "othershop"]);
        l.admit(card(1, "call-site audit")).expect("admits");
        l
    }

    #[test]
    fn a_card_applies_only_where_it_says() {
        let l = ledger();
        let hit = Context::at("src/storage/erasure.rs", "storage");
        let miss = Context::at("src/consensus/poa.rs", "consensus");
        assert_eq!(l.route(&hit), vec![SkillId(1)]);
        assert!(l.route(&miss).is_empty(), "a route that matches everything is not a route");
    }

    #[test]
    fn a_card_with_no_trigger_is_refused() {
        let mut l = Ledger::new(1);
        let bare = SkillCard::new(SkillId(9), "bare", EvidenceKind::GateRun)
            .step("do the thing");
        assert_eq!(l.admit(bare), Err(AdmitError::Unconditional));
    }

    #[test]
    fn a_card_that_closes_on_prose_is_refused() {
        let mut l = Ledger::new(1);
        let soft = SkillCard::new(SkillId(8), "soft", EvidenceKind::Narrative)
            .applies_when(Trigger::DomainIs("storage"))
            .step("write a paragraph about it");
        assert_eq!(l.admit(soft), Err(AdmitError::ClosesOnNarrative));
    }

    #[test]
    fn a_card_with_no_steps_is_refused() {
        let mut l = Ledger::new(1);
        let empty = SkillCard::new(SkillId(7), "empty", EvidenceKind::TestRun)
            .applies_when(Trigger::DomainIs("storage"));
        assert_eq!(l.admit(empty), Err(AdmitError::Stepless));
    }

    #[test]
    fn promotion_needs_the_threshold_of_closing_evidences() {
        let mut l = ledger();
        let id = SkillId(1);
        assert_eq!(l.record(id, EvidenceKind::TestRun, false).unwrap(), Status::Draft);
        assert_eq!(l.record(id, EvidenceKind::TestRun, false).unwrap(), Status::Trusted);
        assert_eq!(l.verify(), Ok(()));
    }

    #[test]
    fn narrative_evidence_does_not_count_toward_promotion() {
        let mut l = ledger();
        let id = SkillId(1);
        for _ in 0..4 {
            assert_eq!(l.record(id, EvidenceKind::Narrative, false).unwrap(), Status::Draft);
        }
        assert_eq!(l.record_of(id).unwrap().closing(), 0);
    }

    #[test]
    fn a_contradiction_demotes_and_is_not_forgotten() {
        let mut l = ledger();
        let id = SkillId(1);
        l.record(id, EvidenceKind::TestRun, false).unwrap();
        l.record(id, EvidenceKind::TestRun, false).unwrap();
        assert_eq!(l.record_of(id).unwrap().status(), Status::Trusted);
        assert_eq!(l.record(id, EvidenceKind::TestRun, true).unwrap(), Status::Rejected);
        assert!(l.runnable().is_empty(), "a refuted method must not keep running");
        assert_eq!(l.record_of(id).unwrap().contradictions(), 1);
    }

    #[test]
    fn borrowed_names_are_refused_in_the_public_surface() {
        let mut l = Ledger::new(1).reject_names(&["borrowedproject"]);
        let bad = SkillCard::new(SkillId(3), "borrowedproject port", EvidenceKind::TestRun)
            .applies_when(Trigger::DomainIs("x"))
            .step("follow borrowedproject");
        assert!(matches!(
            l.admit(bad),
            Err(AdmitError::BorrowedName { field: "name", .. })
        ));
    }

    #[test]
    fn specificity_orders_the_route() {
        let mut l = Ledger::new(1);
        l.admit(
            SkillCard::new(SkillId(10), "fallback", EvidenceKind::GateRun)
                .applies_when(Trigger::Always)
                .step("orient"),
        )
        .unwrap();
        l.admit(
            SkillCard::new(SkillId(11), "narrow", EvidenceKind::GateRun)
                .applies_when(Trigger::PathContains("storage"))
                .applies_when(Trigger::DomainIs("storage"))
                .step("measure"),
        )
        .unwrap();
        let ctx = Context::at("src/storage/deal.rs", "storage");
        assert_eq!(l.route(&ctx), vec![SkillId(11), SkillId(10)]);
    }

    #[test]
    fn a_missing_capability_defers_instead_of_running_half() {
        let mut l = Ledger::new(1);
        l.admit(
            SkillCard::new(SkillId(12), "repair", EvidenceKind::BuildRun)
                .applies_when(Trigger::DomainIs("storage"))
                .requires(Capability::WriteTree)
                .step("change the placement"),
        )
        .unwrap();
        let ctx = Context::at("a.rs", "storage");
        let (applied, deferred) = l.route_with_deferrals(&ctx);
        assert!(applied.is_empty());
        assert_eq!(deferred, vec![(SkillId(12), vec![Capability::WriteTree])]);
        let allowed = ctx.with_capability(Capability::WriteTree);
        assert_eq!(l.route(&allowed), vec![SkillId(12)]);
    }

    #[test]
    fn symbol_triggers_refuse_an_unread_subject() {
        let card = SkillCard::new(SkillId(13), "symbols", EvidenceKind::TestRun)
            .applies_when(Trigger::SymbolPrefixed("assign_"))
            .step("list them");
        let no_symbols = Context::at("a.rs", "storage");
        let read = no_symbols.clone().with_symbol("assign_object");
        assert_eq!(card.matches(&no_symbols), 0, "an empty symbol list is not a non-match");
        assert!(card.matches(&read) > 0);
    }

    #[test]
    fn verify_catches_a_trusted_card_with_nothing_behind_it() {
        let mut l = ledger();
        // Reach into the record through the public path: promote, then strip
        // the evidence by re-admitting the same id (registration resets it).
        l.record(SkillId(1), EvidenceKind::TestRun, false).unwrap();
        l.record(SkillId(1), EvidenceKind::TestRun, false).unwrap();
        assert_eq!(l.verify(), Ok(()));
        let mut broken = l.clone();
        broken.records.get_mut(&SkillId(1)).unwrap().closing = 0;
        assert_eq!(broken.verify(), Err(LedgerError::TrustedWithoutEvidence(SkillId(1))));
    }

    #[test]
    fn zero_threshold_is_clamped_not_trusted_by_default() {
        let l = Ledger::new(0);
        assert_eq!(l.promotion_after(), 1, "zero would trust a card on arrival");
    }
}
