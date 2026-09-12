//! Load measurement and scaling decisions that cannot oscillate.
//!
//! # The two rules that keep a scaling system from flapping
//!
//! **1. The scale-up and scale-down thresholds must differ.** If the system adds
//! capacity at the same load at which it removes capacity, it will alternate
//! forever: load crosses the line, capacity is added, load drops, capacity is
//! removed, load rises. [`Policy`] refuses to be built with the two thresholds
//! equal or inverted, because a policy that permits flapping is a policy that
//! will flap.
//!
//! **2. No new action within the cooldown.** After capacity changes, the load
//! has not yet reflected the change; acting again inside that window means
//! reacting to your own action as though it were new information.
//! [`Policy::decide`] returns [`Decision::Hold`] during the cooldown and says
//! how much time is left.
//!
//! # What is measured
//!
//! Scaling on a single instantaneous sample scales on noise. [`Load`] carries a
//! window and reports the *sustained* value - the fraction of samples above the
//! threshold - so a one-second spike does not buy capacity that is then paid for
//! all day.
//!
//! # Flapping is visible
//!
//! A controller that scales up and down repeatedly has thresholds that are wrong,
//! and that is a fact about the policy rather than about the traffic. Every
//! oscillation is counted in [`Controller::flaps`] and the history of decisions
//! is kept, because "the system is unstable" is not actionable and "it has
//! reversed direction eleven times since Tuesday" is.

use std::collections::VecDeque;

/// What a scaling decision can be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// No change. `reason` says why, because "hold" covers both "load is fine"
    /// and "cooldown has not elapsed", and those need different responses.
    Hold { reason: &'static str },
    /// Add capacity.
    ScaleUp { from: u32, to: u32 },
    /// Remove capacity, but never to zero - see [`Decision::ScaleDown`].
    ScaleDown { from: u32, to: u32 },
}

impl Decision {
    /// Whether this decision changes anything.
    #[must_use]
    pub fn is_change(self) -> bool {
        matches!(self, Self::ScaleUp { .. } | Self::ScaleDown { .. })
    }
}

/// Why a policy could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// The thresholds are equal or inverted, which permits flapping.
    NoHysteresis { up: f64, down: f64 },
    /// A threshold outside 0..=1 is not a fraction of capacity.
    ThresholdOutOfRange { value: f64 },
    /// The bounds are inverted or zero.
    BadBounds { min: u32, max: u32 },
    /// The step size is zero, so scaling can never finish.
    ZeroStep,
    /// A `NaN` threshold can never be compared reliably.
    NotANumber { which: &'static str },
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoHysteresis { up, down } => write!(
                f,
                "scale-up at {up} and scale-down at {down} leaves no gap, and a controller with no gap alternates forever"
            ),
            Self::ThresholdOutOfRange { value } => {
                write!(f, "the threshold {value} is not a fraction of capacity (0..=1)")
            }
            Self::BadBounds { min, max } => write!(
                f,
                "the replica bounds {min}..={max} are inverted or zero, and either way no decision is possible"
            ),
            Self::ZeroStep => write!(f, "a step of zero means scaling can never finish"),
            Self::NotANumber { which } => write!(
                f,
                "the {which} threshold is NaN, which compares false against everything"
            ),
        }
    }
}

/// The policy. Immutable once built, because a policy that changes under a
/// running controller makes its history unreadable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    /// The load fraction at which capacity is added.
    pub scale_up_at: f64,
    /// The load fraction at which capacity is removed. Strictly below
    /// `scale_up_at`; the gap between them is what stops oscillation.
    pub scale_down_at: f64,
    /// Intervals after a change during which no new change is made.
    pub cooldown_intervals: u32,
    pub min_replicas: u32,
    pub max_replicas: u32,
    /// How much of the current replica count one step changes.
    pub step_fraction: f64,
}

impl Policy {
    /// Builds a policy, refusing one that could oscillate.
    ///
    /// # Errors
    ///
    /// Any [`PolicyError`] that applies.
    pub fn new(
        scale_up_at: f64,
        scale_down_at: f64,
        cooldown_intervals: u32,
        min_replicas: u32,
        max_replicas: u32,
        step_fraction: f64,
    ) -> Result<Self, PolicyError> {
        if scale_up_at.is_nan() {
            return Err(PolicyError::NotANumber { which: "scale-up" });
        }
        if scale_down_at.is_nan() {
            return Err(PolicyError::NotANumber { which: "scale-down" });
        }
        for value in [scale_up_at, scale_down_at] {
            if !(0.0..=1.0).contains(&value) {
                return Err(PolicyError::ThresholdOutOfRange { value });
            }
        }
        // The gap is the point. Equal thresholds mean the system adds and removes
        // capacity at the same load, and inverted thresholds mean it removes
        // capacity exactly when it should be adding it.
        if scale_down_at >= scale_up_at {
            return Err(PolicyError::NoHysteresis {
                up: scale_up_at,
                down: scale_down_at,
            });
        }
        if min_replicas == 0 || max_replicas < min_replicas {
            return Err(PolicyError::BadBounds {
                min: min_replicas,
                max: max_replicas,
            });
        }
        if step_fraction <= 0.0 {
            return Err(PolicyError::ZeroStep);
        }
        Ok(Self {
            scale_up_at,
            scale_down_at,
            cooldown_intervals,
            min_replicas,
            max_replicas,
            step_fraction,
        })
    }

    /// Decides, given the sustained load and the current state.
    ///
    /// `intervals_since_last_change` is how many intervals have passed since the
    /// controller last acted; a controller that has never acted passes a value
    /// large enough to be past any cooldown.
    #[must_use]
    pub fn decide(&self, sustained_load: f64, replicas: u32, intervals_since_last_change: u32) -> Decision {
        // Cooldown first. Acting inside the window means reacting to the previous
        // action as though it were new information.
        if intervals_since_last_change < self.cooldown_intervals {
            return Decision::Hold {
                reason: "cooldown has not elapsed since the last change",
            };
        }
        if sustained_load >= self.scale_up_at {
            if replicas >= self.max_replicas {
                return Decision::Hold { reason: "already at the maximum replica count" };
            }
            let to = self.step_up(replicas).min(self.max_replicas);
            return Decision::ScaleUp { from: replicas, to };
        }
        if sustained_load <= self.scale_down_at {
            if replicas <= self.min_replicas {
                return Decision::Hold { reason: "already at the minimum replica count" };
            }
            let to = self.step_down(replicas).max(self.min_replicas);
            return Decision::ScaleDown { from: replicas, to };
        }
        Decision::Hold { reason: "load is between the two thresholds" }
    }

    /// One step up. Always at least one replica, or a controller at the top of
    /// its range with a small step fraction would compute a step of zero and
    /// never move.
    fn step_up(&self, replicas: u32) -> u32 {
        let step = ((f64::from(replicas) * self.step_fraction).ceil()) as u32;
        replicas.saturating_add(step.max(1))
    }

    /// One step down. At least one replica removed, and the caller clamps to the
    /// minimum, so this never returns the input unchanged.
    fn step_down(&self, replicas: u32) -> u32 {
        let step = ((f64::from(replicas) * self.step_fraction).floor()) as u32;
        replicas.saturating_sub(step.max(1))
    }
}

/// A window of load samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Load {
    window: usize,
    samples: VecDeque<f64>,
}

impl Load {
    /// A window of `window` samples.
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            samples: VecDeque::new(),
        }
    }

    /// Records a sample.
    pub fn record(&mut self, load: f64) {
        if self.samples.len() >= self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(load);
    }

    /// How many samples are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The fraction of the window's samples at or above `threshold`.
    ///
    /// Sustained rather than instantaneous: a single spike should not buy
    /// capacity that is then paid for all day.
    #[must_use]
    pub fn sustained_above(&self, threshold: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let above = self
            .samples
            .iter()
            .filter(|v| **v >= threshold)
            .count();
        above as f64 / self.samples.len() as f64
    }

    /// The mean of the window. Reported alongside the sustained fraction,
    /// because a mean hides a spike and a sustained fraction hides its size.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }
}

/// The controller: the policy plus the state that makes history readable.
#[derive(Debug, Clone, PartialEq)]
pub struct Controller {
    policy: Policy,
    pub replicas: u32,
    /// Intervals since the last change.
    intervals_since_change: u32,
    /// Direction reversals. A controller that keeps reversing has thresholds
    /// that are wrong, and that is a fact about the policy rather than about the
    /// traffic.
    pub flaps: u64,
    last_direction: Option<bool>,
    /// The decisions taken, most recent last, bounded.
    history: VecDeque<Decision>,
}

impl Controller {
    /// A controller at `initial` replicas.
    ///
    /// # Errors
    ///
    /// Any [`PolicyError`] that applies.
    pub fn new(policy: Policy, initial: u32) -> Result<Self, PolicyError> {
        if initial < policy.min_replicas || initial > policy.max_replicas {
            return Err(PolicyError::BadBounds {
                min: policy.min_replicas,
                max: policy.max_replicas,
            });
        }
        Ok(Self {
            policy,
            replicas: initial,
            // Start past the cooldown so the first decision is not blocked by a
            // change that never happened.
            intervals_since_change: u32::MAX,
            flaps: 0,
            last_direction: None,
            history: VecDeque::new(),
        })
    }

    /// Takes one interval's decision and applies it if it is a change.
    #[must_use]
    pub fn tick(&mut self, load: &Load) -> Decision {
        // The load that drives the decision is the sustained fraction above the
        // scale-up threshold, which is what the policy's thresholds are
        // expressed against.
        let sustained = load.sustained_above(self.policy.scale_down_at);
        let decision = self
            .policy
            .decide(sustained, self.replicas, self.intervals_since_change);
        match decision {
            Decision::ScaleUp { to, .. } => {
                self.replicas = to;
                self.intervals_since_change = 0;
                self.record(true);
            }
            Decision::ScaleDown { to, .. } => {
                self.replicas = to;
                self.intervals_since_change = 0;
                self.record(false);
            }
            Decision::Hold { .. } => {
                self.intervals_since_change = self.intervals_since_change.saturating_add(1);
            }
        }
        if self.history.len() >= 64 {
            self.history.pop_front();
        }
        self.history.push_back(decision);
        decision
    }

    /// Records a direction and counts a reversal.
    fn record(&mut self, up: bool) {
        if let Some(previous) = self.last_direction {
            if previous != up {
                self.flaps = self.flaps.saturating_add(1);
            }
        }
        self.last_direction = Some(up);
    }

    /// The recent decisions, most recent last.
    ///
    /// Collected rather than borrowed: a [`VecDeque`] is two slices once it has
    /// wrapped, and returning only the back half would silently report a short
    /// history for exactly the controllers that have been running longest.
    #[must_use]
    pub fn history(&self) -> Vec<Decision> {
        self.history.iter().copied().collect()
    }

    /// The policy in force.
    #[must_use]
    pub fn policy(&self) -> Policy {
        self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy::new(0.8, 0.4, 3, 1, 10, 0.5).expect("policy")
    }

    #[test]
    fn equal_thresholds_are_refused() {
        // A controller with no gap adds and removes capacity at the same load and
        // alternates forever.
        assert!(matches!(
            Policy::new(0.7, 0.7, 3, 1, 10, 0.5),
            Err(PolicyError::NoHysteresis { up: 0.7, down: 0.7 })
        ));
    }

    #[test]
    fn inverted_thresholds_are_refused() {
        // Removing capacity exactly when it should be added is worse than
        // flapping.
        assert!(matches!(
            Policy::new(0.4, 0.8, 3, 1, 10, 0.5),
            Err(PolicyError::NoHysteresis { .. })
        ));
    }

    #[test]
    fn a_nan_threshold_is_refused() {
        // NaN compares false against everything, so every comparison would be
        // silently wrong rather than loudly wrong.
        assert!(matches!(
            Policy::new(f64::NAN, 0.4, 3, 1, 10, 0.5),
            Err(PolicyError::NotANumber { which: "scale-up" })
        ));
    }

    #[test]
    fn an_out_of_range_threshold_is_refused() {
        assert!(matches!(
            Policy::new(1.5, 0.4, 3, 1, 10, 0.5),
            Err(PolicyError::ThresholdOutOfRange { value: 1.5 })
        ));
    }

    #[test]
    fn inverted_bounds_are_refused() {
        assert!(matches!(
            Policy::new(0.8, 0.4, 3, 5, 2, 0.5),
            Err(PolicyError::BadBounds { min: 5, max: 2 })
        ));
        assert!(matches!(
            Policy::new(0.8, 0.4, 3, 0, 10, 0.5),
            Err(PolicyError::BadBounds { min: 0, max: 10 })
        ));
    }

    #[test]
    fn a_zero_step_is_refused() {
        // Scaling could never finish.
        assert!(matches!(
            Policy::new(0.8, 0.4, 3, 1, 10, 0.0),
            Err(PolicyError::ZeroStep)
        ));
    }

    #[test]
    fn a_controller_starting_outside_its_bounds_is_refused() {
        assert!(Controller::new(policy(), 20).is_err());
        assert!(Controller::new(policy(), 4).is_ok());
    }

    #[test]
    fn high_load_scales_up() {
        let mut load = Load::new(4);
        for _ in 0..4 {
            load.record(1.0);
        }
        let mut c = Controller::new(policy(), 4).expect("controller");
        match c.tick(&load) {
            Decision::ScaleUp { from, to } => {
                assert_eq!((from, to), (4, 6), "a 0.5 step on 4 replicas is 2");
            }
            other => panic!("expected a scale-up, got {other:?}"),
        }
    }

    #[test]
    fn the_cooldown_blocks_the_next_decision() {
        // Acting inside the window means reacting to the previous action as
        // though it were new information.
        let mut load = Load::new(4);
        for _ in 0..4 {
            load.record(1.0);
        }
        let mut c = Controller::new(policy(), 4).expect("controller");
        assert!(c.tick(&load).is_change());
        assert_eq!(
            c.tick(&load),
            Decision::Hold {
                reason: "cooldown has not elapsed since the last change"
            }
        );
        // Two more intervals pass the cooldown of three.
        assert!(!c.tick(&load).is_change());
        assert!(c.tick(&load).is_change());
    }

    #[test]
    fn the_first_decision_is_not_blocked_by_a_change_that_never_happened() {
        let mut load = Load::new(4);
        for _ in 0..4 {
            load.record(1.0);
        }
        let mut c = Controller::new(policy(), 4).expect("controller");
        assert!(c.tick(&load).is_change());
    }

    #[test]
    fn a_single_spike_does_not_buy_capacity() {
        // Sustained rather than instantaneous: one sample above the threshold out
        // of four is not a reason to pay for capacity all day.
        let mut load = Load::new(4);
        load.record(1.0);
        for _ in 0..3 {
            load.record(0.1);
        }
        assert_eq!(load.sustained_above(0.4), 0.25);
        let mut c = Controller::new(policy(), 4).expect("controller");
        assert_eq!(
            c.tick(&load),
            Decision::Hold {
                reason: "load is between the two thresholds"
            }
        );
    }

    #[test]
    fn a_step_is_never_zero() {
        // A controller at the top of its range with a small step fraction would
        // compute a step of zero and never move.
        let small_step = Policy::new(0.8, 0.4, 3, 1, 10, 0.01).expect("policy");
        let mut load = Load::new(4);
        for _ in 0..4 {
            load.record(1.0);
        }
        let mut c = Controller::new(small_step, 1).expect("controller");
        match c.tick(&load) {
            Decision::ScaleUp { from, to } => assert!(to > from, "the step was zero"),
            other => panic!("expected a scale-up, got {other:?}"),
        }
    }

    #[test]
    fn the_maximum_is_not_exceeded() {
        let mut load = Load::new(4);
        for _ in 0..4 {
            load.record(1.0);
        }
        let mut c = Controller::new(policy(), 9).expect("controller");
        assert_eq!(
            c.tick(&load),
            Decision::ScaleUp { from: 9, to: 10 },
            "the step must be clamped to the maximum"
        );
        assert_eq!(c.replicas, 10);
        // Past the cooldown, at the maximum, it holds rather than growing.
        for _ in 0..3 {
            c.tick(&load);
        }
        assert_eq!(
            c.tick(&load),
            Decision::Hold {
                reason: "already at the maximum replica count"
            }
        );
    }

    #[test]
    fn the_minimum_is_not_gone_below() {
        // Scaling to zero is a different decision from scaling down, and this
        // policy never makes it.
        let mut load = Load::new(4);
        for _ in 0..4 {
            load.record(0.0);
        }
        let mut c = Controller::new(policy(), 2).expect("controller");
        assert_eq!(c.tick(&load), Decision::ScaleDown { from: 2, to: 1 });
        for _ in 0..3 {
            c.tick(&load);
        }
        assert_eq!(
            c.tick(&load),
            Decision::Hold {
                reason: "already at the minimum replica count"
            }
        );
    }

    #[test]
    fn direction_reversals_are_counted() {
        // A controller that keeps reversing has thresholds that are wrong. That
        // is a fact about the policy, not about the traffic, and it must be
        // visible.
        let mut c = Controller::new(policy(), 4).expect("controller");
        let high = {
            let mut l = Load::new(4);
            for _ in 0..4 {
                l.record(1.0);
            }
            l
        };
        let low = {
            let mut l = Load::new(4);
            for _ in 0..4 {
                l.record(0.0);
            }
            l
        };
        c.tick(&high);
        for _ in 0..3 {
            c.tick(&high);
        }
        c.tick(&low);
        assert_eq!(c.flaps, 1, "one reversal was not counted");
        for _ in 0..3 {
            c.tick(&low);
        }
        c.tick(&high);
        assert_eq!(c.flaps, 2);
    }

    #[test]
    fn the_window_drops_old_samples() {
        let mut load = Load::new(2);
        load.record(1.0);
        load.record(0.0);
        load.record(0.0);
        assert_eq!(load.len(), 2);
        assert_eq!(load.sustained_above(0.4), 0.0);
    }

    #[test]
    fn an_empty_window_reports_no_load() {
        let load = Load::new(4);
        assert_eq!(load.sustained_above(0.4), 0.0);
        assert_eq!(load.mean(), 0.0);
    }

    #[test]
    fn the_history_is_bounded() {
        let mut load = Load::new(1);
        load.record(0.6);
        let mut c = Controller::new(policy(), 4).expect("controller");
        for _ in 0..200 {
            c.tick(&load);
        }
        assert!(c.history().len() <= 64);
    }
}
