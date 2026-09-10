//! Behavior changes have an activation epoch, and the epoch is a fact.
//!
//! Budlum's consensus surface changes by name: `BDLM_MAINTENANCE_PLACEMENT_V1`
//! and its siblings are flags that flip at an epoch, so a node that upgrades
//! early and a node that upgrades late disagree about the same block only if the
//! disagreement was not scheduled. The scheduling is what this crate keeps. It is
//! not a feature-flag system for a web app: the whole value is that "the new
//! placement rule is live from epoch 4410" is *in the ledger*, checkable, and not
//! a comment three lines above the `if`.
//!
//! The shape came out of a real deferral in this codebase. Opening a repair
//! ticket for an under-replicated shard is implemented, and pricing that ticket
//! is not, because the price change moves the state root - which means it needs
//! an activation epoch of its own, agreed before the code ships. A deferral with
//! no epoch in it is a TODO: this crate is what makes the deferral auditable
//! instead. It does not claim the flag exists in the node; it says what a record
//! of such a flag must never be able to look like.
//!
//! # The rules
//!
//! - **Nothing activates at genesis.** `activates_at == 0` is refused
//!   [`RegistryError::Immediate`]: a change live before any operator opted in
//!   cannot be avoided by waiting, and "we'll see what happens" is not a plan.
//! - **The comparison is never local.** [`Registry::is_active`] is the only
//!   question the ledger answers, and it is `>=`. Every site that reimplements
//!   the comparison is a place where `>` and `>=` can disagree forever.
//! - **A plan groups changes and fixes their epoch.** A change cannot carry an
//!   epoch its plan does not name ([`RegistryError::EpochDrift`]), and it cannot
//!   bring a flag the plan never promised ([`RegistryError::UnlistedFlag`]).
//! - **Nothing waits forever by accident.** A pending change whose epoch has
//!   passed is [`LedgerViolation::MissedActivation`] - the flag was never thrown
//!   and the code is quietly behaving like the old rule.
//! - **Retirement is a state, not a deletion.** A change that never activated
//!   cannot be retired ([`LedgerViolation::NeverActivated`]), because retiring
//!   something that never ran is how a failed rollout disappears from the record.
//! - **Epochs do not collide by accident.** Two live plans may not open at the
//!   same epoch; one plan may contain any number of changes, because they were
//!   agreed together.
//!
//! # What it is not
//!
//! No clock. Epochs arrive as parameters, so "is it live now" is a question a
//! test can answer for any `now`. No hashes and no signatures: this ledger says
//! what the schedule was, not that somebody authorized it - that belongs to the
//! consensus layer whose state root this is trying not to surprise.

/// A consensus-relevant height. Kept as a plain alias: the node has its own
/// epoch type and this crate must not pretend to own it.
pub type Epoch = u64;

/// What a change does once it is live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// Nodes that disagree now produce different state: every peer must flip at
    /// the same epoch or the chain splits.
    Consensus,
    /// Local only: a node may run either side without splitting anything.
    Local,
    /// Deprecation of an old path: reading is still allowed, writing is not.
    Freeze,
}

impl Effect {
    /// Label for the schedule table.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Consensus => "consensus",
            Self::Local => "local",
            Self::Freeze => "freeze",
        }
    }

    /// Whether peers must agree on the epoch, not merely on the code.
    #[must_use]
    pub fn requires_quorum(self) -> bool {
        matches!(self, Self::Consensus)
    }
}

/// The state a change is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Scheduled, epoch not reached.
    Pending,
    /// The epoch arrived and nothing on record says the flag went in: the chain
    /// is running the old rule while the schedule promises the new one.
    Unratified,
    /// Scheduled, epoch reached or passed.
    Active,
    /// Withdrawn from the epoch it was retired at.
    Retired,
}

impl Status {
    /// Label for the schedule table.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Unratified => "unratified",
            Self::Active => "active",
            Self::Retired => "retired",
        }
    }
}

/// A scheduled behavior change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    flag: String,
    activates_at: Epoch,
    effect: Effect,
    plan: String,
    why: String,
    /// The epoch the operators recorded the flag in. `None` means the schedule
    /// promises the change and nothing has put it on a block yet, which is a
    /// different thing from "not yet due".
    ratified_at: Option<Epoch>,
    retired_at: Option<Epoch>,
}

impl Change {
    /// Declare a change. The name is the flag the node will read, so it is not
    /// free text: see [`Registry::admit`] for what is refused.
    #[must_use]
    pub fn new(flag: &str, activates_at: Epoch, effect: Effect, plan: &str) -> Self {
        Self {
            flag: flag.to_string(),
            activates_at,
            effect,
            plan: plan.to_string(),
            why: String::new(),
            ratified_at: None,
            retired_at: None,
        }
    }

    /// The flag name.
    #[must_use]
    pub fn flag(&self) -> &str {
        &self.flag
    }

    /// The epoch it becomes live at, inclusive.
    #[must_use]
    pub fn activates_at(&self) -> Epoch {
        self.activates_at
    }

    /// How much the network cares.
    #[must_use]
    pub fn effect(&self) -> Effect {
        self.effect
    }

    /// The plan it was agreed under.
    #[must_use]
    pub fn plan(&self) -> &str {
        &self.plan
    }

    /// The sentence that says why, if one was written.
    #[must_use]
    pub fn why(&self) -> &str {
        &self.why
    }

    /// Record the reason. Required for a consensus change: see
    /// [`Registry::admit`].
    #[must_use]
    pub fn with_why(mut self, why: &str) -> Self {
        self.why = why.to_string();
        self
    }

    /// The epoch the flag was actually recorded in, if it was.
    #[must_use]
    pub fn ratified_at(&self) -> Option<Epoch> {
        self.ratified_at
    }

    /// The epoch it was withdrawn at, if any.
    #[must_use]
    pub fn retired_at(&self) -> Option<Epoch> {
        self.retired_at
    }

    /// Where it stands at `at`.
    #[must_use]
    pub fn status_at(&self, at: Epoch) -> Status {
        if let Some(retired) = self.retired_at {
            if at >= retired {
                return Status::Retired;
            }
        }
        if at < self.activates_at {
            return Status::Pending;
        }
        if self.ratified_at.is_none() {
            return Status::Unratified;
        }
        Status::Active
    }

    /// Whether the change is live at `at`. Delegated to [`Self::status_at`] on
    /// purpose: two implementations of "is it live" is where `>` and `>=` part
    /// ways.
    #[must_use]
    pub fn is_live_at(&self, at: Epoch) -> bool {
        self.status_at(at) == Status::Active
    }
}

/// A group of changes agreed together, at one epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    name: String,
    epoch: Epoch,
    flags: Vec<String>,
    note: String,
}

impl Plan {
    /// Declare a plan. The epoch is fixed here and only here: a change carries
    /// the epoch of its plan rather than one of its own, which is what makes a
    /// mismatch a ledger violation instead of a typo nobody noticed.
    #[must_use]
    pub fn new(name: &str, epoch: Epoch) -> Self {
        Self {
            name: name.to_string(),
            epoch,
            flags: Vec::new(),
            note: String::new(),
        }
    }

    /// The plan name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The single epoch the plan opens at.
    #[must_use]
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Flags that belong to it.
    #[must_use]
    pub fn flags(&self) -> &[String] {
        &self.flags
    }

    /// A sentence of context, if there is one.
    #[must_use]
    pub fn note(&self) -> &str {
        &self.note
    }

    /// Set the note.
    #[must_use]
    pub fn with_note(mut self, note: &str) -> Self {
        self.note = note.to_string();
        self
    }

    /// Add a flag to the list the plan was agreed with. The list is closed at
    /// [`Registry::declare`]: a plan that can gain a flag afterwards is a plan
    /// whose scope was decided by whoever got to the code first.
    #[must_use]
    pub fn with_flag(mut self, flag: &str) -> Self {
        let flag = flag.to_string();
        if !self.flags.contains(&flag) {
            self.flags.push(flag);
        }
        self
    }
}

/// Why something cannot enter the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Epoch 0: live before anybody could opt out.
    Immediate {
        /// The flag.
        flag: String,
    },
    /// A flag name that is not a flag: spaces, lowercase, punctuation.
    MalformedFlag {
        /// What was offered.
        flag: String,
        /// The rule it broke.
        rule: &'static str,
    },
    /// An empty ledger is not a plan.
    EmptyPlan {
        /// The plan.
        plan: String,
    },
    /// A consensus change with no reason recorded.
    ReasonlessConsensusChange {
        /// The flag.
        flag: String,
    },
    /// The same flag twice, with different terms.
    FlagConflict {
        /// The flag.
        flag: String,
        /// The epoch already recorded.
        recorded: Epoch,
        /// The epoch being offered.
        offered: Epoch,
    },
    /// Two live plans opening at the same epoch.
    PlanCollision {
        /// The epoch.
        epoch: Epoch,
        /// The plan already there.
        first: String,
        /// The plan that would join it.
        second: String,
    },
    /// The plan is not declared, so the change has no agreed epoch.
    UndeclaredPlan {
        /// The plan named by the change.
        plan: String,
    },
    /// The plan listed the flags it was agreed with, and this one is not there.
    UnlistedFlag {
        /// The plan.
        plan: String,
        /// The flag the plan never promised.
        flag: String,
    },
    /// The change carries an epoch its plan does not name.
    EpochDrift {
        /// The flag.
        flag: String,
        /// What the change asked for.
        change_epoch: Epoch,
        /// What was agreed.
        plan_epoch: Epoch,
    },
    /// The ledger has no such flag.
    UnknownFlag {
        /// What was asked for.
        flag: String,
    },
    /// The plan name is already taken.
    DuplicatePlan {
        /// The plan.
        plan: String,
    },
    /// The ledger already records a different epoch for the same event.
    RewriteOfRecord {
        /// The flag.
        flag: String,
        /// What is on record.
        recorded: Epoch,
        /// What was offered instead.
        offered: Epoch,
    },
    /// A retired change cannot be ratified from at or after its retirement.
    AfterRetirement {
        /// The flag.
        flag: String,
        /// The retirement on record.
        retired_at: Epoch,
    },
    /// A change may not be retired before it ever ran.
    NeverActivated {
        /// The flag.
        flag: String,
    },
    /// A retirement epoch cannot be earlier than the activation it undoes.
    TimeTravelling {
        /// The flag.
        flag: String,
        /// Activation epoch.
        activates_at: Epoch,
        /// Retirement epoch offered.
        retired_at: Epoch,
    },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Immediate { flag } => write!(
                f,
                "`{flag}` activates at epoch 0: a change nobody could decline is not a \
                 rollout, it is a new default"
            ),
            Self::MalformedFlag { flag, rule } => {
                write!(f, "`{flag}` is not a flag name: {rule}")
            }
            Self::EmptyPlan { plan } => write!(
                f,
                "plan `{plan}` has no changes: an empty plan is how a version bump with no \
                 behavior in it gets remembered as a behavior change"
            ),
            Self::ReasonlessConsensusChange { flag } => write!(
                f,
                "`{flag}` splits state if peers disagree and carries no reason: a \
                 consensus-gated change has to say what it changes"
            ),
            Self::FlagConflict {
                flag,
                recorded,
                offered,
            } => write!(
                f,
                "`{flag}` is already scheduled at {recorded}; a second entry at {offered} \
                 means two nodes can pick either and both look correct"
            ),
            Self::PlanCollision {
                epoch,
                first,
                second,
            } => write!(
                f,
                "plans `{first}` and `{second}` both open at {epoch}: agreed separately, \
                 deployed together - that is one agreement, and it has two authors"
            ),
            Self::UndeclaredPlan { plan } => write!(
                f,
                "change belongs to `{plan}`, which was never declared: the epoch would be \
                 whatever the code says it is"
            ),
            Self::UnlistedFlag { plan, flag } => write!(
                f,
                "`{flag}` is not among the flags `{plan}` was agreed with: the list is the \
                 agreement, and a change added to it afterwards is a different change"
            ),
            Self::EpochDrift {
                flag,
                change_epoch,
                plan_epoch,
            } => write!(
                f,
                "`{flag}` asks for {change_epoch} while its plan names {plan_epoch}: the plan is \
                 the thing that was agreed, so fix the change"
            ),
            Self::UnknownFlag { flag } => write!(
                f,
                "the ledger has no `{flag}`: retiring a name that was never scheduled is how a \
                 typo turns into a cancelled rollout"
            ),
            Self::DuplicatePlan { plan } => write!(
                f,
                "plan `{plan}` is already declared: redeclaring it hides which of the two the \
                 operators agreed to"
            ),
            Self::RewriteOfRecord {
                flag,
                recorded,
                offered,
            } => write!(
                f,
                "the ledger already records {recorded} for `{flag}`; {offered} is a rewrite of \
                 what happened, not an update to it"
            ),
            Self::AfterRetirement { flag, retired_at } => write!(
                f,
                "`{flag}` is retired from {retired_at}: a ratification at or after that does not \
                 un-retire it, it contradicts the schedule"
            ),
            Self::NeverActivated { flag } => write!(
                f,
                "`{flag}` never reached its epoch, so there is nothing to retire: say it was \
                 cancelled instead of making the failed rollout disappear"
            ),
            Self::TimeTravelling {
                flag,
                activates_at,
                retired_at,
            } => write!(
                f,
                "`{flag}` activates at {activates_at} and retires at {retired_at}: a change \
                 cannot be withdrawn before it ran"
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Something inconsistent about a ledger that is already built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerViolation {
    /// A pending change whose epoch has passed.
    MissedActivation {
        /// The flag.
        flag: String,
        /// The epoch it was supposed to open at.
        activates_at: Epoch,
        /// The epoch it was checked at.
        now: Epoch,
    },
    /// A plan declared and never used.
    UnfilledPlan {
        /// The plan.
        plan: String,
    },
    /// Two live changes under one flag name after the fact.
    DuplicateFlag {
        /// The flag.
        flag: String,
    },
}

impl std::fmt::Display for LedgerViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissedActivation {
                flag,
                activates_at,
                now,
            } => write!(
                f,
                "`{flag}` was due at {activates_at}, it is {now}, and it is still pending: \
                 the code is running the old rule while the schedule promises the new one"
            ),
            Self::UnfilledPlan { plan } => write!(
                f,
                "plan `{plan}` was declared and no change carries it: an unfilled plan is a \
                 promise with no code behind it"
            ),
            Self::DuplicateFlag { flag } => write!(
                f,
                "`{flag}` appears twice in the ledger"
            ),
        }
    }
}

impl std::error::Error for LedgerViolation {}

/// The answer to "is this flag live?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Live {
    /// Scheduled and the epoch has arrived.
    On,
    /// Scheduled and the epoch has not arrived, or it was retired.
    Off,
    /// The epoch arrived and no ratification exists. Not `Off`: the schedule
    /// says the new rule should already be running.
    Unratified,
    /// Not in the ledger at all. Not the same as [`Live::Off`]: `Off` is a
    /// decision, this is a typo or a change that was never agreed.
    NotScheduled,
}

impl Live {
    /// Whether code should take the new path.
    #[must_use]
    pub fn is_on(self) -> bool {
        self == Self::On
    }

    /// Label for the report.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
            Self::Unratified => "unratified",
            Self::NotScheduled => "not-scheduled",
        }
    }
}

/// How the ledger stands at one epoch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Live at this epoch.
    pub active: usize,
    /// Due and unratified at this epoch.
    pub unratified: usize,
    /// Scheduled for later.
    pub pending: usize,
    /// Withdrawn at or before this epoch.
    pub retired: usize,
}

impl Counts {
    /// The line a reader wants first.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} active, {} pending, {} unratified, {} retired",
            self.active, self.pending, self.unratified, self.retired
        )
    }
}

/// The schedule: plans, the changes inside them, and the answer to the only
/// question consensus asks of this file.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    plans: Vec<Plan>,
    changes: Vec<Change>,
}

impl Registry {
    /// An empty ledger. Legitimate: a node with nothing scheduled has nothing to
    /// disagree about, and an empty ledger that verifies clean is what makes a
    /// non-empty one meaningful.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declared plans, in the order they were added.
    #[must_use]
    pub fn plans(&self) -> &[Plan] {
        &self.plans
    }

    /// Every change, including retired ones. Nothing is deleted from a schedule.
    #[must_use]
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    /// One plan by name.
    #[must_use]
    pub fn plan(&self, name: &str) -> Option<&Plan> {
        self.plans.iter().find(|p| p.name() == name)
    }

    /// One change by flag.
    #[must_use]
    pub fn change(&self, flag: &str) -> Option<&Change> {
        self.changes.iter().find(|c| c.flag() == flag)
    }

    /// Declare a plan. A plan with no flags in it is refused: the flags are part
    /// of what was agreed, not something the code decides later.
    pub fn declare(&mut self, plan: Plan) -> Result<(), RegistryError> {
        if plan.flags().is_empty() {
            return Err(RegistryError::EmptyPlan {
                plan: plan.name().to_string(),
            });
        }
        for other in &self.plans {
            if other.name() == plan.name() {
                return Err(RegistryError::DuplicatePlan {
                    plan: plan.name().to_string(),
                });
            }
        }
        for other in &self.plans {
            if other.epoch() == plan.epoch() && self.plan_is_live(other) {
                return Err(RegistryError::PlanCollision {
                    epoch: plan.epoch(),
                    first: other.name().to_string(),
                    second: plan.name().to_string(),
                });
            }
        }
        self.plans.push(plan);
        Ok(())
    }

    /// Add a change. Every rule that can be checked without knowing "now" is
    /// checked here, because a rejection at admission time names the author's
    /// mistake, and the same violation found later names only the symptom.
    pub fn admit(&mut self, change: Change) -> Result<(), RegistryError> {
        let flag = change.flag().to_string();
        check_flag_shape(&flag)?;
        if change.activates_at() == 0 {
            return Err(RegistryError::Immediate { flag });
        }
        if change.effect() == Effect::Consensus && change.why().trim().is_empty() {
            return Err(RegistryError::ReasonlessConsensusChange { flag });
        }
        let Some(plan) = self.plans.iter().find(|p| p.name() == change.plan()) else {
            return Err(RegistryError::UndeclaredPlan {
                plan: change.plan().to_string(),
            });
        };
        if plan.epoch() != change.activates_at() {
            return Err(RegistryError::EpochDrift {
                flag: flag.clone(),
                change_epoch: change.activates_at(),
                plan_epoch: plan.epoch(),
            });
        }
        if !plan.flags().iter().any(|listed| *listed == flag) {
            return Err(RegistryError::UnlistedFlag {
                plan: plan.name().to_string(),
                flag,
            });
        }
        for other in &self.changes {
            if other.flag() == change.flag() {
                if other.activates_at() != change.activates_at() || other.plan() != change.plan() {
                    return Err(RegistryError::FlagConflict {
                        flag: flag.clone(),
                        recorded: other.activates_at(),
                        offered: change.activates_at(),
                    });
                }
            }
        }
        // Admission can create a collision by making a retired plan live again,
        // so the epoch rule is re-checked - and rolled back, because a ledger with
        // a half-applied change in it is worse than a refusal.
        let before = self.changes.len();
        self.changes.push(change);
        if let Err(why) = self.check_collisions() {
            self.changes.truncate(before);
            return Err(why);
        }
        Ok(())
    }

    /// Record that the flag went into the schedule at `at`. Early ratification is
    /// the normal path - the flag is agreed before its epoch - so no comparison
    /// against `activates_at` is made here. Ratification is a fact, so it is
    /// written once.
    pub fn ratify(&mut self, flag: &str, at: Epoch) -> Result<(), RegistryError> {
        let Some(index) = self.changes.iter().position(|c| c.flag() == flag) else {
            return Err(RegistryError::UnknownFlag {
                flag: flag.to_string(),
            });
        };
        if let Some(retired) = self.changes[index].retired_at() {
            if at >= retired {
                return Err(RegistryError::AfterRetirement {
                    flag: flag.to_string(),
                    retired_at: retired,
                });
            }
        }
        if let Some(existing) = self.changes[index].ratified_at() {
            if existing != at {
                return Err(RegistryError::RewriteOfRecord {
                    flag: flag.to_string(),
                    recorded: existing,
                    offered: at,
                });
            }
            return Ok(());
        }
        self.changes[index].ratified_at = Some(at);
        Ok(())
    }

    /// Withdraw a change from `at`. It stays in the ledger: a schedule that can
    /// forget is a schedule that cannot be audited.
    pub fn retire(&mut self, flag: &str, at: Epoch) -> Result<(), RegistryError> {
        let Some(index) = self.changes.iter().position(|c| c.flag() == flag) else {
            return Err(RegistryError::UnknownFlag {
                flag: flag.to_string(),
            });
        };
        let activates_at = self.changes[index].activates_at();
        if at < activates_at {
            return Err(RegistryError::TimeTravelling {
                flag: flag.to_string(),
                activates_at,
                retired_at: at,
            });
        }
        if at == activates_at {
            return Err(RegistryError::NeverActivated {
                flag: flag.to_string(),
            });
        }
        if let Some(retired) = self.changes[index].retired_at() {
            if retired != at {
                return Err(RegistryError::RewriteOfRecord {
                    flag: flag.to_string(),
                    recorded: retired,
                    offered: at,
                });
            }
            return Ok(());
        }
        self.changes[index].retired_at = Some(at);
        if let Err(why) = self.check_collisions() {
            self.changes[index].retired_at = None;
            return Err(why);
        }
        Ok(())
    }

    /// Two live plans may not open at the same epoch. Re-checked after every
    /// mutation, because retirement changes which plans are live and can retire
    /// a collision as easily as it can create one.
    fn check_collisions(&self) -> Result<(), RegistryError> {
        for (index, first) in self.plans.iter().enumerate() {
            if !self.plan_is_live(first) {
                continue;
            }
            for second in &self.plans[index + 1..] {
                if second.epoch() == first.epoch() && self.plan_is_live(second) {
                    return Err(RegistryError::PlanCollision {
                        epoch: first.epoch(),
                        first: first.name().to_string(),
                        second: second.name().to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// A plan with no changes is live in the sense that it is an open promise; a
    /// plan whose every change was retired at or before `now` cannot collide with
    /// anything.
    fn plan_is_live(&self, plan: &Plan) -> bool {
        let mut any = false;
        for change in &self.changes {
            if change.plan() != plan.name() {
                continue;
            }
            if change.retired_at().is_none() {
                return true;
            }
            any = true;
        }
        // Declared but not yet filled: still an open promise.
        !any
    }

    /// The only question the ledger answers.
    #[must_use]
    pub fn live(&self, flag: &str, at: Epoch) -> Live {
        let Some(change) = self.change(flag) else {
            return Live::NotScheduled;
        };
        match change.status_at(at) {
            Status::Active => Live::On,
            Status::Unratified => Live::Unratified,
            Status::Pending | Status::Retired => Live::Off,
        }
    }

    /// The same answer for every flag at once.
    #[must_use]
    pub fn counts_at(&self, at: Epoch) -> Counts {
        let mut out = Counts::default();
        for change in &self.changes {
            match change.status_at(at) {
                Status::Active => out.active += 1,
                Status::Pending => out.pending += 1,
                Status::Unratified => out.unratified += 1,
                Status::Retired => out.retired += 1,
            }
        }
        out
    }

    /// Flags live at `at`, in admission order.
    #[must_use]
    pub fn live_at(&self, at: Epoch) -> Vec<&Change> {
        self.changes
            .iter()
            .filter(|c| c.status_at(at) == Status::Active)
            .collect()
    }

    /// Flags scheduled after `at`.
    #[must_use]
    pub fn pending_at(&self, at: Epoch) -> Vec<&Change> {
        self.changes
            .iter()
            .filter(|c| c.status_at(at) == Status::Pending)
            .collect()
    }

    /// The earliest epoch a new change may open at: never genesis, and never on
    /// top of another live plan's epoch.
    #[must_use]
    pub fn earliest_open(&self, now: Epoch) -> Epoch {
        let mut candidate = if now == 0 { 1 } else { now };
        loop {
            let taken = self
                .plans
                .iter()
                .any(|plan| plan.epoch() == candidate && self.plan_is_live(plan));
            if !taken {
                return candidate;
            }
            candidate += 1;
        }
    }

    /// Check the ledger against itself, at `now`.
    ///
    /// # Errors
    ///
    /// [`LedgerViolation::MissedActivation`] when a pending change is past due,
    /// [`LedgerViolation::UnfilledPlan`] when a declared plan carries no change,
    /// [`LedgerViolation::DuplicateFlag`] when one flag appears twice.
    pub fn verify(&self, now: Epoch) -> Result<(), LedgerViolation> {
        let mut seen: Vec<&str> = Vec::new();
        for change in &self.changes {
            if !seen.iter().any(|flag| *flag == change.flag()) {
                seen.push(change.flag());
            } else {
                return Err(LedgerViolation::DuplicateFlag {
                    flag: change.flag().to_string(),
                });
            }
            if change.status_at(now) == Status::Unratified {
                return Err(LedgerViolation::MissedActivation {
                    flag: change.flag().to_string(),
                    activates_at: change.activates_at(),
                    now,
                });
            }
        }
        for plan in &self.plans {
            if !self.changes.iter().any(|c| c.plan() == plan.name()) {
                return Err(LedgerViolation::UnfilledPlan {
                    plan: plan.name().to_string(),
                });
            }
        }
        Ok(())
    }

    /// The schedule as a table, with anything overdue spelled out.
    #[must_use]
    pub fn render(&self, now: Epoch) -> String {
        let mut out = String::new();
        out.push_str(&format!("schedule at epoch {now}\n"));
        for plan in &self.plans {
            let filled = self
                .changes
                .iter()
                .filter(|c| c.plan() == plan.name())
                .count();
            out.push_str(&format!(
                "  plan {} @{} ({} of {} flags admitted)\n",
                plan.name(),
                plan.epoch(),
                filled,
                plan.flags().len()
            ));
            if !plan.note().is_empty() {
                out.push_str(&format!("    note: {}\n", plan.note()));
            }
        }
        for change in &self.changes {
            out.push_str(&format!(
                "  {:<38} {:<9} {:<8} {}\n",
                change.flag(),
                change.effect().label(),
                change.status_at(now).label(),
                format!("epoch {}", change.activates_at())
            ));
            if let Some(retired) = change.retired_at() {
                out.push_str(&format!("    retired at {retired}\n"));
            }
            if !change.why().is_empty() {
                out.push_str(&format!("    why: {}\n", change.why()));
            }
            if change.status_at(now) == Status::Unratified {
                out.push_str(&format!(
                    "    OVERDUE: due at {}, it is {now}, and nothing on record ratified it\n",
                    change.activates_at()
                ));
            }
        }
        out.push_str(&format!("  {}\n", self.counts_at(now).summary()));
        out
    }
}

/// A flag name is a name: upper snake case, long enough to search for, and with
/// a prefix that says who owns it. A ledger full of `enable_new_thing` cannot be
/// grepped from a runbook, and a runbook is where these names are actually read.
fn check_flag_shape(flag: &str) -> Result<(), RegistryError> {
    let malformed = |rule: &'static str| RegistryError::MalformedFlag {
        flag: flag.to_string(),
        rule,
    };
    if flag.len() < 5 {
        return Err(malformed("at least five characters, so the name is searchable"));
    }
    if !flag.contains('_') {
        return Err(malformed("an underscore between the owner prefix and the meaning"));
    }
    let mut chars = flag.chars();
    let Some(first) = chars.next() else {
        return Err(malformed("at least five characters, so the name is searchable"));
    };
    if !first.is_ascii_uppercase() {
        return Err(malformed("the first character is an ASCII uppercase letter"));
    }
    for ch in chars {
        if !(ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_') {
            return Err(malformed("letters are ASCII uppercase, digits and underscore only"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAG: &str = "BDLM_MAINTENANCE_PLACEMENT_V1";
    const AT: Epoch = 4410;

    fn plan(name: &str, epoch: Epoch, flags: &[&str]) -> Plan {
        let mut out = Plan::new(name, epoch);
        for flag in flags {
            out = out.with_flag(flag);
        }
        out
    }

    fn placement() -> (Change, Plan) {
        (
            Change::new(FLAG, AT, Effect::Consensus, "placement-v1")
                .with_why("price demand-driven repair tickets instead of logging them"),
            plan("placement-v1", AT, &[FLAG]),
        )
    }

    /// Declared and scheduled, but not ratified: the state a deferral is in
    /// before anybody throws the flag.
    fn scheduled() -> Registry {
        let (change, plan) = placement();
        let mut reg = Registry::new();
        reg.declare(plan)
            .unwrap_or_else(|why| panic!("declare: {why}"));
        reg.admit(change)
            .unwrap_or_else(|why| panic!("admit: {why}"));
        reg
    }

    /// The same, with the ratification recorded ahead of its epoch, which is what
    /// an agreed upgrade looks like on the day it ships.
    fn shipped() -> Registry {
        let mut reg = scheduled();
        reg.ratify(FLAG, AT - 10)
            .unwrap_or_else(|why| panic!("ratify: {why}"));
        reg
    }

    #[test]
    fn genesis_activation_is_refused() {
        let mut reg = scheduled();
        reg.declare(plan("zero-v1", 0, &[FLAG]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        let bad = Change::new(FLAG, 0, Effect::Local, "zero-v1");
        assert_eq!(
            reg.admit(bad),
            Err(RegistryError::Immediate {
                flag: FLAG.to_string()
            })
        );
    }

    #[test]
    fn a_flag_name_has_to_be_a_flag_name() {
        assert!(check_flag_shape("BDLM_X").is_ok());
        assert!(check_flag_shape(FLAG).is_ok());
        for bad in ["ok", "bdlm_x", "BDLMX", "BDLM X", "BDLM-x", "1BDLM_X"] {
            let err = check_flag_shape(bad);
            assert!(
                matches!(err, Err(RegistryError::MalformedFlag { .. })),
                "`{bad}` should not be admissible: {err:?}"
            );
        }
    }

    #[test]
    fn the_epoch_belongs_to_the_plan() {
        let mut reg = scheduled();
        let drift = Change::new(FLAG, AT + 1, Effect::Local, "placement-v1");
        assert_eq!(
            reg.admit(drift),
            Err(RegistryError::EpochDrift {
                flag: FLAG.to_string(),
                change_epoch: AT + 1,
                plan_epoch: AT,
            })
        );
    }

    #[test]
    fn a_change_needs_a_plan_and_a_place_on_its_list() {
        let mut reg = Registry::new();
        let orphan = Change::new("BDLM_ORPHAN", 10, Effect::Local, "nope");
        assert_eq!(
            reg.admit(orphan),
            Err(RegistryError::UndeclaredPlan {
                plan: "nope".to_string()
            })
        );
        reg.declare(plan("placement-v1", 10, &["BDLM_LISTED"]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        let uninvited = Change::new("BDLM_UNINVITED", 10, Effect::Local, "placement-v1");
        assert_eq!(
            reg.admit(uninvited),
            Err(RegistryError::UnlistedFlag {
                plan: "placement-v1".to_string(),
                flag: "BDLM_UNINVITED".to_string(),
            })
        );
    }

    #[test]
    fn a_plan_declares_its_flags_up_front() {
        let mut reg = Registry::new();
        assert_eq!(
            reg.declare(Plan::new("empty-v1", 12)),
            Err(RegistryError::EmptyPlan {
                plan: "empty-v1".to_string()
            })
        );
        reg.declare(plan("twice-v1", 12, &[FLAG]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        assert_eq!(
            reg.declare(plan("twice-v1", 13, &[FLAG])),
            Err(RegistryError::DuplicatePlan {
                plan: "twice-v1".to_string()
            })
        );
        // The refused declaration left the first plan's epoch alone.
        assert_eq!(reg.plan("twice-v1").map(|p| p.epoch()), Some(12));
    }

    #[test]
    fn two_live_plans_cannot_share_an_epoch() {
        let mut reg = Registry::new();
        reg.declare(plan("first-v1", 20, &["BDLM_FIRST"]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        reg.admit(Change::new("BDLM_FIRST", 20, Effect::Local, "first-v1"))
            .unwrap_or_else(|why| panic!("admit: {why}"));
        let second = plan("second-v1", 20, &["BDLM_SECOND"]);
        assert_eq!(
            reg.declare(second),
            Err(RegistryError::PlanCollision {
                epoch: 20,
                first: "first-v1".to_string(),
                second: "second-v1".to_string(),
            })
        );
        assert_eq!(reg.plans().len(), 1, "a refused declare changes nothing");
    }

    #[test]
    fn a_fully_retired_plan_frees_its_epoch() {
        let mut reg = Registry::new();
        reg.declare(plan("first-v1", 20, &["BDLM_FIRST"]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        reg.admit(Change::new("BDLM_FIRST", 20, Effect::Local, "first-v1"))
            .unwrap_or_else(|why| panic!("admit: {why}"));
        assert!(reg
            .declare(plan("second-v1", 20, &["BDLM_SECOND"]))
            .is_err());
        reg.retire("BDLM_FIRST", 25)
            .unwrap_or_else(|why| panic!("retire: {why}"));
        reg.declare(plan("second-v1", 20, &["BDLM_SECOND"]))
            .unwrap_or_else(|why| panic!("a retired plan should not hold an epoch: {why}"));
    }

    #[test]
    fn a_consensus_change_has_to_say_what_it_changes() {
        let mut reg = Registry::new();
        reg.declare(plan("placement-v1", AT, &[FLAG]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        assert_eq!(
            reg.admit(Change::new(FLAG, AT, Effect::Consensus, "placement-v1")),
            Err(RegistryError::ReasonlessConsensusChange {
                flag: FLAG.to_string()
            })
        );
        assert!(reg
            .admit(Change::new(FLAG, AT, Effect::Local, "placement-v1"))
            .is_ok());
    }

    #[test]
    fn ratification_is_what_turns_a_schedule_into_a_fact() {
        let mut reg = scheduled();
        assert_eq!(reg.live(FLAG, AT), Live::Unratified);
        assert_eq!(reg.live(FLAG, AT - 1), Live::Off);
        reg.ratify(FLAG, AT - 10)
            .unwrap_or_else(|why| panic!("ratify: {why}"));
        assert_eq!(reg.live(FLAG, AT - 1), Live::Off, "early, but not due");
        assert_eq!(reg.live(FLAG, AT), Live::On);
        assert_eq!(reg.live(FLAG, AT + 1), Live::On);
        assert_eq!(
            reg.ratify(FLAG, AT - 9),
            Err(RegistryError::RewriteOfRecord {
                flag: FLAG.to_string(),
                recorded: AT - 10,
                offered: AT - 9,
            })
        );
        assert_eq!(
            reg.ratify(FLAG, AT - 10),
            Ok(()),
            "the same fact twice is not a rewrite"
        );
        assert_eq!(
            reg.ratify("BDLM_NOPE", AT),
            Err(RegistryError::UnknownFlag {
                flag: "BDLM_NOPE".to_string()
            })
        );
        reg.retire(FLAG, AT + 5)
            .unwrap_or_else(|why| panic!("retire: {why}"));
        assert_eq!(
            reg.ratify(FLAG, AT + 5),
            Err(RegistryError::AfterRetirement {
                flag: FLAG.to_string(),
                retired_at: AT + 5,
            })
        );
    }

    #[test]
    fn the_comparison_is_inclusive_and_lives_in_one_place() {
        let reg = shipped();
        let change = reg
            .change(FLAG)
            .unwrap_or_else(|| panic!("in the ledger"));
        assert!(change.is_live_at(AT));
        assert!(!change.is_live_at(AT - 1));
        assert_eq!(reg.counts_at(AT - 1).pending, 1);
        assert_eq!(reg.counts_at(AT).active, 1);
        assert_eq!(reg.counts_at(AT).unratified, 0);
        assert!(change.effect().requires_quorum());
    }

    #[test]
    fn an_unknown_flag_is_not_the_same_as_off() {
        let reg = shipped();
        assert_eq!(reg.live("BDLM_NEVER_AGREED", AT), Live::NotScheduled);
        assert!(!reg.live("BDLM_NEVER_AGREED", AT).is_on());
        assert_eq!(Live::NotScheduled.label(), "not-scheduled");
        assert_eq!(Live::Unratified.label(), "unratified");
    }

    #[test]
    fn a_due_change_that_never_opened_is_a_violation() {
        let reg = scheduled();
        reg.verify(AT - 1)
            .unwrap_or_else(|why| panic!("nothing is due yet: {why}"));
        assert_eq!(
            reg.verify(AT),
            Err(LedgerViolation::MissedActivation {
                flag: FLAG.to_string(),
                activates_at: AT,
                now: AT,
            })
        );
        assert_eq!(
            reg.change(FLAG).map(|c| c.status_at(AT)),
            Some(Status::Unratified)
        );
        let fixed = shipped();
        fixed
            .verify(AT)
            .unwrap_or_else(|why| panic!("ratified, so live: {why}"));
    }

    #[test]
    fn an_unfilled_plan_is_a_violation() {
        let mut reg = Registry::new();
        reg.declare(plan("placement-v1", AT, &[FLAG, "BDLM_SECOND"]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        reg.admit(Change::new(FLAG, AT, Effect::Local, "placement-v1"))
            .unwrap_or_else(|why| panic!("admit: {why}"));
        reg.verify(AT - 1)
            .unwrap_or_else(|why| panic!("the plan is filled: {why}"));
        let mut empty = Registry::new();
        empty
            .declare(plan("nothing-v1", AT, &[FLAG]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        assert_eq!(
            empty.verify(AT),
            Err(LedgerViolation::UnfilledPlan {
                plan: "nothing-v1".to_string()
            })
        );
    }

    #[test]
    fn the_same_change_admitted_twice_is_found() {
        let mut reg = shipped();
        let (change, _) = placement();
        reg.admit(change)
            .unwrap_or_else(|why| panic!("an identical re-admission is not a conflict: {why}"));
        assert_eq!(
            reg.verify(AT).err(),
            Some(LedgerViolation::DuplicateFlag {
                flag: FLAG.to_string()
            })
        );
    }

    #[test]
    fn retirement_needs_a_lifetime() {
        let mut reg = shipped();
        assert!(matches!(
            reg.retire(FLAG, AT - 1),
            Err(RegistryError::TimeTravelling { .. })
        ));
        assert_eq!(
            reg.retire(FLAG, AT),
            Err(RegistryError::NeverActivated {
                flag: FLAG.to_string()
            })
        );
        assert_eq!(
            reg.retire("BDLM_NOT_THERE", AT + 1),
            Err(RegistryError::UnknownFlag {
                flag: "BDLM_NOT_THERE".to_string()
            })
        );
        reg.retire(FLAG, AT + 3)
            .unwrap_or_else(|why| panic!("retire: {why}"));
        assert_eq!(
            reg.retire(FLAG, AT + 9),
            Err(RegistryError::RewriteOfRecord {
                flag: FLAG.to_string(),
                recorded: AT + 3,
                offered: AT + 9,
            })
        );
        reg.retire(FLAG, AT + 3)
            .unwrap_or_else(|why| panic!("the same retirement twice: {why}"));
        assert_eq!(reg.live(FLAG, AT + 4), Live::Off);
        assert_eq!(reg.live(FLAG, AT), Live::On);
        assert_eq!(reg.changes().len(), 1, "a schedule does not forget");
        assert_eq!(
            reg.change(FLAG).map(|c| c.retired_at()),
            Some(Some(AT + 3)),
            "and the retirement is still readable"
        );
        assert_eq!(reg.change(FLAG).map(|c| c.ratified_at()), Some(Some(AT - 10)));
    }

    #[test]
    fn the_suggested_epoch_skips_other_peoples_slots() {
        let reg = shipped();
        assert_eq!(reg.earliest_open(0), 1, "genesis is never offered");
        assert_eq!(reg.earliest_open(AT), AT + 1, "a live plan owns its epoch");
        assert_eq!(reg.earliest_open(AT - 1), AT - 1);
        let mut crowded = Registry::new();
        crowded
            .declare(plan("other-v1", 50, &["BDLM_OTHER"]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        crowded
            .admit(Change::new("BDLM_OTHER", 50, Effect::Local, "other-v1"))
            .unwrap_or_else(|why| panic!("admit: {why}"));
        assert_eq!(crowded.earliest_open(50), 51);
        assert_eq!(crowded.earliest_open(49), 49);
    }

    #[test]
    fn counts_partition_the_ledger_at_every_epoch() {
        let mut reg = shipped();
        reg.declare(plan("freeze-v1", AT + 100, &["BDLM_FREEZE_OLD"]))
            .unwrap_or_else(|why| panic!("declare: {why}"));
        reg.admit(Change::new(
            "BDLM_FREEZE_OLD",
            AT + 100,
            Effect::Freeze,
            "freeze-v1",
        ))
        .unwrap_or_else(|why| panic!("admit: {why}"));
        reg.retire(FLAG, AT + 50)
            .unwrap_or_else(|why| panic!("retire: {why}"));
        for now in [0, AT - 1, AT, AT + 49, AT + 50, AT + 100, AT + 500] {
            let counts = reg.counts_at(now);
            assert_eq!(
                counts.active + counts.pending + counts.unratified + counts.retired,
                reg.changes().len(),
                "epoch {now}: {counts:?} does not cover the ledger"
            );
            assert_eq!(counts.active, reg.live_at(now).len(), "epoch {now}");
            assert_eq!(counts.pending, reg.pending_at(now).len(), "epoch {now}");
        }
        assert!(reg.counts_at(AT + 500).summary().contains("retired"));
        assert!(reg.counts_at(AT + 100).summary().contains("1 unratified"));
    }

    #[test]
    fn an_empty_ledger_verifies_clean_and_says_so() {
        let reg = Registry::new();
        reg.verify(AT)
            .unwrap_or_else(|why| panic!("nothing scheduled is not a violation: {why}"));
        assert_eq!(reg.counts_at(AT), Counts::default());
        assert!(reg.render(AT).contains("0 active, 0 pending, 0 unratified, 0 retired"));
        assert_eq!(reg.changes().len(), 0);
    }

    #[test]
    fn effects_and_statuses_are_named_for_the_report() {
        assert!(Effect::Consensus.requires_quorum());
        assert!(!Effect::Local.requires_quorum());
        assert!(!Effect::Freeze.requires_quorum());
        assert_eq!(Effect::Freeze.label(), "freeze");
        assert_eq!(Status::Unratified.label(), "unratified");
        assert_eq!(Status::Retired.label(), "retired");
    }

    #[test]
    fn render_shows_the_schedule_and_the_overdue_line() {
        let ok = shipped().render(AT);
        for needle in ["schedule at epoch", FLAG, "consensus", "placement-v1", "why:"] {
            assert!(ok.contains(needle), "missing `{needle}` in:\n{ok}");
        }
        assert!(!ok.contains("OVERDUE"));
        assert!(ok.contains("1 active, 0 pending, 0 unratified, 0 retired"));
        let late = scheduled().render(AT + 5);
        assert!(late.contains("OVERDUE: due at 4410, it is 4415, and nothing on record"));
        assert!(late.contains("unratified"));
        let note = shipped();
        let planned = note.render(AT - 10);
        assert!(planned.contains("pending"), "not due yet:\n{planned}");
    }
}
