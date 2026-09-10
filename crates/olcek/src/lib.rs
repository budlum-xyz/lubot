//! A limit that does not report itself is a feature switch.
//!
//! Every agent eventually meets a size: context, output, work per session. The
//! common response is to truncate and continue. That is not a limit, it is a
//! silent downgrade, and in a log it is indistinguishable from success: the run
//! finishes and nobody can tell that half the subject was never looked at.
//! Lubot's rule is the opposite - a ceiling may refuse, but it may never be
//! quiet.
//!
//! # The shape
//!
//! Three numbers, two pools, one invariant:
//!
//! ```text
//! 0 < reserved_floor < hard_ceiling      soft_watermark <= hard_ceiling - reserved_floor
//! ```
//!
//! * **ordinary work** may fill up to `hard_ceiling - reserved_floor`; between
//!   the watermark and that limit it needs [`Priority::Planned`] or above, and
//!   the admission is counted as pressure.
//! * **closing work** - the test run, the gate output, the diff that settles a
//!   claim - may spend the reserve too. That is the entire point of reserving
//!   it: a session must not be able to read everything and then report that
//!   there was no room left to verify anything.
//! * anything that does not fit is **refused with numbers attached**, and the
//!   refusal is a counter a report has to state.
//!
//! Eviction is counted the same way. `ease_to_watermark` removes only
//! [`Priority::Optional`] items, and every removal lands in a table with its
//! cost. An item that leaves the ledger without a record is exactly what
//! [`Ledger::verify`] exists to catch.

use std::collections::BTreeMap;

/// Why an item wants in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Nice to have. The only class eviction may remove.
    Optional,
    /// Part of the plan; may enter the pressure band.
    Planned,
    /// Required to close a claim: a test run, a gate, the diff itself.
    Required,
}

impl Priority {
    /// May this priority cross the watermark?
    #[must_use]
    pub fn may_apply_pressure(self) -> bool {
        !matches!(self, Self::Optional)
    }
}

/// What a run asks the ledger for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Reading, notes, background: bounded by everything except the reserve.
    Ordinary,
    /// A check that closes a claim: may also spend the reserved floor.
    Closing,
}

/// What happened when an item asked for room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admitted {
    /// Room was found below the watermark.
    Granted,
    /// Room was found by pressing into the band above the watermark.
    GrantedUnderPressure,
    /// Room was found inside the reserved floor, which is reported separately
    /// so a run that survives only on its reserve is visible as one.
    GrantedFromReserve {
        /// Reserve consumed by this admission.
        taken: usize,
        /// Reserve left afterwards.
        left: usize,
    },
}

impl Admitted {
    /// True for every outcome that put the item in the ledger.
    #[must_use]
    pub fn is_granted(&self) -> bool {
        matches!(
            self,
            Self::Granted | Self::GrantedUnderPressure | Self::GrantedFromReserve { .. }
        )
    }
}

/// Why a ledger is misconfigured, or why it refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// A ceiling of zero admits nothing and reports nothing.
    ZeroCeiling,
    /// A reserve equal to the ceiling leaves ordinary work nowhere to go.
    ReserveIsTheCeiling {
        /// The ceiling.
        ceiling: usize,
        /// The reserve.
        floor: usize,
    },
    /// A watermark above the ordinary limit defines an empty pressure band and
    /// silently disables the band's rule.
    WatermarkOverOrdinary {
        /// The watermark.
        watermark: usize,
        /// `ceiling - floor`.
        ordinary_limit: usize,
    },
    /// The item does not fit, and the numbers say by how much.
    OverCeiling {
        /// What the item asked for.
        wanted: usize,
        /// What was free in the pool it asked about.
        free: usize,
        /// Which pool.
        kind: Kind,
    },
    /// A closing check outgrew everything, reserve included.
    FloorSpent {
        /// What it needed.
        needed: usize,
        /// Free ordinary room plus free reserve.
        available: usize,
    },
    /// Something left the ledger without a drop record.
    UnrecordedDrop {
        /// The item, or a placeholder when it is already gone.
        label: String,
        /// The unexplained cost.
        cost: usize,
    },
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCeiling => write!(f, "a ceiling of zero is not a limit, it is an off switch"),
            Self::ReserveIsTheCeiling { ceiling, floor } => write!(
                f,
                "reserve {floor} equals the ceiling {ceiling}: ordinary work has no room \
                 and the reserve has nothing left to protect"
            ),
            Self::WatermarkOverOrdinary {
                watermark,
                ordinary_limit,
            } => write!(
                f,
                "watermark {watermark} is past the ordinary limit {ordinary_limit}, which \
                 turns the pressure rule off by configuration rather than by decision"
            ),
            Self::OverCeiling { wanted, free, kind } => write!(
                f,
                "asked for {wanted} in the {kind:?} pool with {free} free; this run is \
                 truncated and the report must say so"
            ),
            Self::FloorSpent { needed, available } => write!(
                f,
                "a closing check needed {needed} and {available} was available including \
                 the reserve; reading everything and verifying nothing is a failure with \
                 better optics"
            ),
            Self::UnrecordedDrop { label, cost } => write!(
                f,
                "`{label}` ({cost}) left the ledger uncounted; a silent drop is how a \
                 ceiling becomes a feature switch"
            ),
        }
    }
}

impl std::error::Error for BudgetError {}

/// The three numbers, checked once, at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    hard_ceiling: usize,
    soft_watermark: usize,
    reserved_floor: usize,
}

impl Budget {
    /// Builds a budget, refusing an ordering that would make one of the three
    /// numbers decorative.
    ///
    /// # Errors
    ///
    /// The first [`BudgetError`] found.
    pub fn new(
        hard_ceiling: usize,
        soft_watermark: usize,
        reserved_floor: usize,
    ) -> Result<Self, BudgetError> {
        if hard_ceiling == 0 {
            return Err(BudgetError::ZeroCeiling);
        }
        let ordinary_limit = hard_ceiling.saturating_sub(reserved_floor);
        if reserved_floor >= hard_ceiling {
            return Err(BudgetError::ReserveIsTheCeiling {
                ceiling: hard_ceiling,
                floor: reserved_floor,
            });
        }
        if soft_watermark > ordinary_limit {
            return Err(BudgetError::WatermarkOverOrdinary {
                watermark: soft_watermark,
                ordinary_limit,
            });
        }
        Ok(Self {
            hard_ceiling,
            soft_watermark,
            reserved_floor,
        })
    }

    /// The hard ceiling, reserve included.
    #[must_use]
    pub fn hard_ceiling(&self) -> usize {
        self.hard_ceiling
    }

    /// The line above which admission needs a claim, not a wish.
    #[must_use]
    pub fn soft_watermark(&self) -> usize {
        self.soft_watermark
    }

    /// Space only closing work may spend.
    #[must_use]
    pub fn reserved_floor(&self) -> usize {
        self.reserved_floor
    }

    /// What ordinary work may reach.
    #[must_use]
    pub fn ordinary_limit(&self) -> usize {
        self.hard_ceiling.saturating_sub(self.reserved_floor)
    }
}

/// One held item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    label: String,
    cost: usize,
    priority: Priority,
    kind: Kind,
}

impl Item {
    /// An item with its cost in the ledger's own unit (tokens, files, lines).
    #[must_use]
    pub fn new(label: &str, cost: usize, priority: Priority, kind: Kind) -> Self {
        Self {
            label: label.to_string(),
            cost,
            priority,
            kind,
        }
    }

    /// Its label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Its cost.
    #[must_use]
    pub fn cost(&self) -> usize {
        self.cost
    }

    /// Its claim.
    #[must_use]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Which pool it belongs to.
    #[must_use]
    pub fn kind(&self) -> Kind {
        self.kind
    }
}

/// Where a run stands, counted rather than felt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Band {
    /// Below the watermark.
    Under,
    /// Between the watermark and the ordinary limit: legal, loud.
    Pressure,
    /// Ordinary work is at its limit; only the reserve remains.
    ReservedOnly,
    /// Everything, reserve included, is spent.
    Truncated,
}

/// A counted admission log plus its two running totals.
#[derive(Debug, Clone)]
pub struct Ledger {
    budget: Budget,
    ordinary: usize,
    closing: usize,
    pressure: usize,
    refusals: usize,
    drops: BTreeMap<String, usize>,
    items: Vec<Item>,
}

impl Ledger {
    /// An empty ledger for `budget`.
    #[must_use]
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            ordinary: 0,
            closing: 0,
            pressure: 0,
            refusals: 0,
            drops: BTreeMap::new(),
            items: Vec::new(),
        }
    }

    /// The budget in force.
    #[must_use]
    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Total cost held.
    #[must_use]
    pub fn used(&self) -> usize {
        self.ordinary.saturating_add(self.closing)
    }

    /// Cost held by ordinary work.
    #[must_use]
    pub fn ordinary_used(&self) -> usize {
        self.ordinary
    }

    /// Cost held by closing work.
    #[must_use]
    pub fn closing_used(&self) -> usize {
        self.closing
    }

    /// Headroom left under the hard ceiling.
    #[must_use]
    pub fn headroom(&self) -> usize {
        self.budget.hard_ceiling().saturating_sub(self.used())
    }

    /// Reserve left for closing work.
    #[must_use]
    pub fn reserve_left(&self) -> usize {
        self.budget
            .reserved_floor()
            .saturating_sub(self.used().saturating_sub(self.budget.ordinary_limit()))
    }

    /// The current band.
    #[must_use]
    pub fn band(&self) -> Band {
        let used = self.used();
        let ordinary_limit = self.budget.ordinary_limit();
        if used >= self.budget.hard_ceiling() {
            Band::Truncated
        } else if self.ordinary >= ordinary_limit {
            Band::ReservedOnly
        } else if self.ordinary > self.budget.soft_watermark() {
            Band::Pressure
        } else {
            Band::Under
        }
    }

    /// Admissions that had to press past the watermark.
    #[must_use]
    pub fn pressure_admissions(&self) -> usize {
        self.pressure
    }

    /// Refusals counted.
    #[must_use]
    pub fn refusals(&self) -> usize {
        self.refusals
    }

    /// Items held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True when nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The held items, in admission order.
    #[must_use]
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// Admits an item or refuses it with numbers.
    ///
    /// # Errors
    ///
    /// [`BudgetError::OverCeiling`] when the pool it asked about is full, and
    /// [`BudgetError::FloorSpent`] when even the reserve cannot hold a closing
    /// check. The two cases a silent truncate would hide.
    pub fn admit(&mut self, item: Item) -> Result<Admitted, BudgetError> {
        let wanted = item.cost();
        let projected_total = self.used().saturating_add(wanted);
        if item.kind() == Kind::Closing {
            if projected_total > self.budget.hard_ceiling() {
                self.refusals += 1;
                return Err(BudgetError::FloorSpent {
                    needed: wanted,
                    available: self
                        .budget
                        .hard_ceiling()
                        .saturating_sub(self.used())
                        .max(self.reserve_left()),
                });
            }
            let from_reserve =
                self.ordinary.saturating_add(self.closing).saturating_add(wanted)
                    > self.budget.ordinary_limit();
            self.closing = self.closing.saturating_add(wanted);
            self.items.push(item);
            return Ok(if from_reserve {
                Admitted::GrantedFromReserve {
                    taken: wanted,
                    left: self.reserve_left(),
                }
            } else if self.used() > self.budget.soft_watermark() {
                self.pressure += 1;
                Admitted::GrantedUnderPressure
            } else {
                Admitted::Granted
            });
        }

        let projected_ordinary = self.ordinary.saturating_add(wanted);
        if projected_ordinary > self.budget.ordinary_limit() {
            self.refusals += 1;
            return Err(BudgetError::OverCeiling {
                wanted,
                free: self.budget.ordinary_limit().saturating_sub(self.ordinary),
                kind: item.kind(),
            });
        }
        if projected_ordinary > self.budget.soft_watermark()
            && !item.priority().may_apply_pressure()
        {
            self.refusals += 1;
            return Err(BudgetError::OverCeiling {
                wanted,
                free: self.budget.soft_watermark().saturating_sub(self.ordinary),
                kind: Kind::Ordinary,
            });
        }
        let outcome = if projected_ordinary > self.budget.soft_watermark() {
            self.pressure += 1;
            Admitted::GrantedUnderPressure
        } else {
            Admitted::Granted
        };
        self.ordinary = projected_ordinary;
        self.items.push(item);
        Ok(outcome)
    }

    /// Drops the cheapest optional items until the ledger is back under the
    /// watermark, counting every one.
    #[must_use]
    pub fn ease_to_watermark(&mut self) -> Vec<String> {
        let mut dropped = Vec::new();
        while self.band() == Band::Pressure
            || self.band() == Band::ReservedOnly
            || self.band() == Band::Truncated
        {
            let Some(idx) = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, i)| i.priority() == Priority::Optional)
                .min_by_key(|(_, i)| (i.cost(), i.label().to_string()))
                .map(|(idx, _)| idx)
            else {
                break;
            };
            let item = self.items.remove(idx);
            if item.kind() == Kind::Closing {
                self.closing = self.closing.saturating_sub(item.cost());
            } else {
                self.ordinary = self.ordinary.saturating_sub(item.cost());
            }
            self.drops
                .entry(item.label().to_string())
                .and_modify(|c| *c += 1)
                .or_insert(item.cost());
            dropped.push(item.label().to_string());
        }
        dropped
    }

    /// Evicted cost by label.
    #[must_use]
    pub fn drops(&self) -> &BTreeMap<String, usize> {
        &self.drops
    }

    /// True when the run lost work: evicted, refused, or both.
    #[must_use]
    pub fn was_truncated(&self) -> bool {
        !self.drops.is_empty() || self.refusals > 0
    }

    /// Recomputes the totals from the held items and refuses to accept a
    /// ledger whose arithmetic does not close.
    ///
    /// # Errors
    ///
    /// [`BudgetError::UnrecordedDrop`] naming the gap.
    pub fn verify(&self) -> Result<(), BudgetError> {
        let mut ordinary = 0usize;
        let mut closing = 0usize;
        for item in &self.items {
            match item.kind() {
                Kind::Ordinary => ordinary += item.cost(),
                Kind::Closing => closing += item.cost(),
            }
        }
        if ordinary != self.ordinary || closing != self.closing {
            let cost = self
                .used()
                .saturating_sub(ordinary.saturating_add(closing));
            let label = self
                .drops
                .keys()
                .next()
                .cloned()
                .unwrap_or_else(|| "(gone without a drop record)".to_string());
            return Err(BudgetError::UnrecordedDrop {
                label,
                cost: if cost == 0 { 1 } else { cost },
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> Budget {
        Budget::new(100, 60, 20).expect("valid")
    }

    fn read(label: &str, cost: usize, priority: Priority) -> Item {
        Item::new(label, cost, priority, Kind::Ordinary)
    }

    #[test]
    fn misordered_numbers_are_refused_at_construction() {
        assert_eq!(Budget::new(0, 0, 0), Err(BudgetError::ZeroCeiling));
        assert_eq!(
            Budget::new(100, 40, 100),
            Err(BudgetError::ReserveIsTheCeiling {
                ceiling: 100,
                floor: 100
            })
        );
        // ordinary limit is 100-20 = 80; a watermark above it disables the band.
        assert_eq!(
            Budget::new(100, 90, 20),
            Err(BudgetError::WatermarkOverOrdinary {
                watermark: 90,
                ordinary_limit: 80
            })
        );
    }

    #[test]
    fn work_below_the_watermark_is_admitted_quietly() {
        let mut l = Ledger::new(budget());
        assert_eq!(l.admit(read("a", 30, Priority::Planned)).unwrap(), Admitted::Granted);
        assert_eq!(l.band(), Band::Under);
        assert_eq!(l.pressure_admissions(), 0);
    }

    #[test]
    fn the_pressure_band_needs_a_claim_and_counts_that_it_had_one() {
        let mut l = Ledger::new(budget());
        l.admit(read("a", 50, Priority::Planned)).unwrap();
        let optional = l.admit(read("b", 20, Priority::Optional));
        assert!(
            matches!(optional, Err(BudgetError::OverCeiling { .. })),
            "optional work may not cross the watermark"
        );
        let planned = l.admit(read("c", 15, Priority::Planned)).unwrap();
        assert_eq!(planned, Admitted::GrantedUnderPressure);
        assert_eq!(l.band(), Band::Pressure);
        assert_eq!(l.pressure_admissions(), 1);
        assert_eq!(l.refusals(), 1, "the refusal above was counted, not swallowed");
    }

    #[test]
    fn ordinary_work_stops_at_the_reserve_and_says_so() {
        let mut l = Ledger::new(budget());
        l.admit(read("a", 80, Priority::Required)).unwrap();
        assert_eq!(l.band(), Band::ReservedOnly);
        let err = l
            .admit(read("b", 1, Priority::Required))
            .expect_err("the reserve is not for reading");
        assert_eq!(
            err,
            BudgetError::OverCeiling {
                wanted: 1,
                free: 0,
                kind: Kind::Ordinary
            }
        );
    }

    #[test]
    fn a_closing_check_spends_the_reserve_and_reports_the_balance() {
        let mut l = Ledger::new(budget());
        l.admit(read("a", 80, Priority::Required)).unwrap();
        let gate = Item::new("run the gate", 15, Priority::Required, Kind::Closing);
        assert_eq!(
            l.admit(gate).unwrap(),
            Admitted::GrantedFromReserve {
                taken: 15,
                left: 5
            }
        );
        assert_eq!(l.closing_used(), 15);
        assert_eq!(l.reserve_left(), 5);
    }

    #[test]
    fn a_closing_check_that_outgrows_everything_fails_loudly() {
        let mut l = Ledger::new(budget());
        l.admit(read("a", 80, Priority::Required)).unwrap();
        l.admit(Item::new("first gate", 15, Priority::Required, Kind::Closing))
            .unwrap();
        let gate = Item::new("run the gate", 20, Priority::Required, Kind::Closing);
        assert_eq!(
            l.admit(gate),
            Err(BudgetError::FloorSpent {
                needed: 20,
                available: 5
            })
        );
    }

    #[test]
    fn eviction_removes_optional_work_and_never_the_closing_artefact() {
        let mut l = Ledger::new(budget());
        l.admit(read("scratch", 30, Priority::Optional)).unwrap();
        l.admit(read("notes", 25, Priority::Optional)).unwrap();
        l.admit(read("context", 25, Priority::Required)).unwrap();
        l.admit(Item::new("gate", 20, Priority::Required, Kind::Closing)).unwrap();
        assert_eq!(l.band(), Band::Truncated);
        let dropped = l.ease_to_watermark();
        assert_eq!(
            dropped,
            vec!["notes".to_string()],
            "the cheapest optional item goes first, and one is enough"
        );
        assert_eq!(l.band(), Band::Under);
        assert!(l.items().iter().any(|i| i.label() == "gate"));
        assert_eq!(*l.drops().get("notes").unwrap(), 25);
        assert_eq!(l.verify(), Ok(()), "every removal was counted");
    }

    #[test]
    fn verify_catches_cost_that_left_the_ledger_unaccounted() {
        let mut l = Ledger::new(budget());
        l.admit(read("a", 40, Priority::Optional)).unwrap();
        assert_eq!(l.verify(), Ok(()));
        l.items.clear();
        assert!(matches!(l.verify(), Err(BudgetError::UnrecordedDrop { .. })));
    }

    #[test]
    fn truncation_shows_in_the_report_when_the_run_lost_work() {
        let mut l = Ledger::new(budget());
        l.admit(read("only", 80, Priority::Required)).unwrap();
        assert!(!l.was_truncated(), "a full ordinary pool is not a truncated run");
        assert!(l.admit(read("more", 1, Priority::Required)).is_err());
        assert!(l.was_truncated());
    }

    #[test]
    fn headroom_counts_both_pools() {
        let mut l = Ledger::new(budget());
        l.admit(read("a", 30, Priority::Planned)).unwrap();
        l.admit(Item::new("gate", 10, Priority::Required, Kind::Closing)).unwrap();
        assert_eq!(l.used(), 40);
        assert_eq!(l.headroom(), 60);
    }
}
