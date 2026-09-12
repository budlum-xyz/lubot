//! Classification that can decline to answer, and that can be shown to be
//! honest about its own confidence.
//!
//! # Abstention is an answer
//!
//! A classifier forced to pick a category always picks one, including when the
//! evidence supports nothing. Everything downstream then treats that pick as a
//! finding. [`Classifier::classify`] returns [`Verdict::Abstain`] in three
//! distinct situations, and the reason is part of the value because they need
//! different responses:
//!
//! - **no signal supports any category** - the evidence is empty for this
//!   classifier, which is usually a wiring problem rather than an ambiguous item;
//! - **the evidence is contradictory** - the top two categories score within
//!   [`TIE_EPSILON`] of each other. Picking the alphabetically first one would
//!   make the answer depend on category spelling;
//! - **the best score is below the floor** - there is a leading category, it is
//!   just not leading enough to act on.
//!
//! # A tie is not resolved by ordering
//!
//! [`TIE_EPSILON`] is explicit rather than a bare `==` on two sums of floats. Two
//! categories that score the same are genuinely tied, and an implementation that
//! breaks the tie by map order returns an answer whose value depends on how the
//! categories happen to be spelled.
//!
//! # Calibration, not confidence
//!
//! A classifier that says 0.9 and is right half the time is worse than one that
//! says 0.6 and is right 0.6 of the time: the first one's confidence cannot be
//! used to decide anything. [`Calibration`] records outcomes in confidence
//! buckets and reports the gap between the confidence claimed in a bucket and the
//! accuracy actually observed there. The gap is the number that says whether the
//! confidence is usable, and it is not visible from any single classification.
//!
//! [`TIE_EPSILON`]: self::TIE_EPSILON

use std::collections::BTreeMap;

/// Scores closer than this are a tie.
///
/// Explicit rather than `==`: the scores are sums of the same weights, so exact
/// equality is meaningful, but an epsilon makes the tie rule robust to the order
/// in which the weights were summed.
pub const TIE_EPSILON: f64 = 1e-12;

/// The number of confidence buckets. Buckets are 0.1 wide.
pub const CALIBRATION_BUCKETS: usize = 10;

/// One observed feature.
#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    pub name: String,
    /// How strongly this signal is present. Usually 1.0; a partially present
    /// signal carries less.
    pub strength: f64,
}

/// The observations for one item.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Evidence {
    pub signals: Vec<Signal>,
}

impl Evidence {
    /// An empty evidence set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a signal.
    #[must_use]
    pub fn with(mut self, name: &str, strength: f64) -> Self {
        self.signals.push(Signal {
            name: name.to_string(),
            strength,
        });
        self
    }

    /// Whether nothing was observed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// How many signals were observed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// A fingerprint of the evidence, so two runs over the same item can be shown
    /// to have seen the same thing. Signals are sorted by name and strength so
    /// the order they were collected in does not change the fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut parts: Vec<String> = self
            .signals
            .iter()
            .map(|s| format!("{}\u{1f}{}", s.name, s.strength.to_bits()))
            .collect();
        parts.sort();
        lubot_read::sha256_hex(parts.join("\u{1e}").as_bytes())
    }
}

/// Why the classifier declined.
#[derive(Debug, Clone, PartialEq)]
pub enum Abstention {
    /// No signal scored above zero for any category.
    NoSupport,
    /// The top two categories are within [`TIE_EPSILON`].
    Contradictory {
        first: String,
        second: String,
        score: f64,
    },
    /// There is a leader, but not by enough to act on.
    BelowFloor {
        category: String,
        confidence: f64,
        floor: f64,
    },
}

impl std::fmt::Display for Abstention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSupport => write!(
                f,
                "no signal supports any category; the evidence is empty for this classifier, which is usually a wiring problem"
            ),
            Self::Contradictory {
                first,
                second,
                score,
            } => write!(
                f,
                "{first:?} and {second:?} both score {score}; picking one would make the answer depend on category spelling"
            ),
            Self::BelowFloor {
                category,
                confidence,
                floor,
            } => write!(
                f,
                "{category:?} leads at {confidence}, below the floor of {floor}"
            ),
        }
    }
}

/// What a classification produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// A category, with the confidence and the signals that produced it.
    Category {
        category: String,
        confidence: f64,
        /// The contributing signals and their weights, so the answer is
        /// explainable from the evidence rather than from the model.
        reasons: Vec<(String, f64)>,
    },
    /// No answer. See [`Abstention`].
    Abstain { reason: Abstention },
}

impl Verdict {
    /// The category, if there is one.
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        match self {
            Self::Category { category, .. } => Some(category),
            Self::Abstain { .. } => None,
        }
    }

    /// Whether this verdict named a category.
    #[must_use]
    pub fn is_answer(&self) -> bool {
        matches!(self, Self::Category { .. })
    }
}

/// Why a classifier could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassifierError {
    /// The floor is not a probability.
    FloorOutOfRange { floor: f64 },
    /// No categories were declared, so nothing can ever be answered.
    NoCategories,
}

impl std::fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FloorOutOfRange { floor } => {
                write!(f, "the confidence floor {floor} is not within 0..=1")
            }
            Self::NoCategories => {
                write!(
                    f,
                    "the classifier declares no categories, so it can never answer"
                )
            }
        }
    }
}

/// The classifier: a weight per (category, signal), plus a floor.
#[derive(Debug, Clone, PartialEq)]
pub struct Classifier {
    weights: BTreeMap<String, BTreeMap<String, f64>>,
    /// Below this confidence the classifier abstains rather than guessing.
    pub floor: f64,
}

impl Classifier {
    /// Builds a classifier.
    ///
    /// # Errors
    ///
    /// Any [`ClassifierError`] that applies.
    pub fn new(
        weights: BTreeMap<String, BTreeMap<String, f64>>,
        floor: f64,
    ) -> Result<Self, ClassifierError> {
        if !(0.0..=1.0).contains(&floor) {
            return Err(ClassifierError::FloorOutOfRange { floor });
        }
        if weights.is_empty() {
            return Err(ClassifierError::NoCategories);
        }
        Ok(Self { weights, floor })
    }

    /// The declared categories, in order.
    #[must_use]
    pub fn categories(&self) -> Vec<&str> {
        self.weights.keys().map(String::as_str).collect()
    }

    /// Classifies, or abstains.
    #[must_use]
    pub fn classify(&self, evidence: &Evidence) -> Verdict {
        // Score every category over the observed signals. A weight the classifier
        // does not declare for a signal contributes nothing, which is what makes
        // "no support" reachable.
        let mut scores: Vec<(String, f64)> = Vec::with_capacity(self.weights.len());
        for (category, table) in &self.weights {
            let score: f64 = evidence
                .signals
                .iter()
                .filter_map(|signal| {
                    table
                        .get(&signal.name)
                        .map(|weight| weight * signal.strength)
                })
                .sum();
            scores.push((category.clone(), score));
        }
        // Sort by score descending, then by name, so the order is deterministic
        // even though the tie rule below does not depend on it.
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let (best_name, best_score) = scores[0].clone();
        let positive_total: f64 = scores
            .iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(_, s)| *s)
            .sum();
        if positive_total <= 0.0 {
            return Verdict::Abstain {
                reason: Abstention::NoSupport,
            };
        }
        let runner_up = scores[1].clone();
        if (best_score - runner_up.1).abs() < TIE_EPSILON {
            return Verdict::Abstain {
                reason: Abstention::Contradictory {
                    first: best_name,
                    second: runner_up.0,
                    score: best_score,
                },
            };
        }
        let confidence = best_score / positive_total;
        if confidence < self.floor {
            return Verdict::Abstain {
                reason: Abstention::BelowFloor {
                    category: best_name,
                    confidence,
                    floor: self.floor,
                },
            };
        }
        let reasons: Vec<(String, f64)> = evidence
            .signals
            .iter()
            .filter_map(|signal| {
                self.weights
                    .get(&best_name)
                    .and_then(|table| table.get(&signal.name))
                    .map(|weight| (signal.name.clone(), *weight))
            })
            .collect();
        Verdict::Category {
            category: best_name,
            confidence,
            reasons,
        }
    }
}

/// One recorded outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outcome {
    /// The confidence the classifier claimed.
    pub confidence: f64,
    /// Whether the answer was in fact right.
    pub correct: bool,
}

/// Calibration across confidence buckets.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Calibration {
    buckets: Vec<(f64, u64)>,
}

impl Calibration {
    /// An empty record.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: vec![(0.0, 0); CALIBRATION_BUCKETS],
        }
    }

    /// Records an outcome.
    ///
    /// A confidence outside 0..=1 is not recorded, because there is no bucket for
    /// it and silently clamping it would put a nonsense claim in a real bucket.
    pub fn record(&mut self, outcome: Outcome) {
        if !(0.0..=1.0).contains(&outcome.confidence) {
            return;
        }
        let index = ((outcome.confidence * CALIBRATION_BUCKETS as f64) as usize)
            .min(CALIBRATION_BUCKETS - 1);
        let slot = &mut self.buckets[index];
        slot.0 += if outcome.correct { 1.0 } else { 0.0 };
        slot.1 = slot.1.saturating_add(1);
    }

    /// How many outcomes were recorded.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.buckets.iter().map(|(_, n)| *n).sum()
    }

    /// The gap between the confidence claimed in a bucket and the accuracy
    /// observed there.
    ///
    /// This is the number that says whether the confidence is usable. A
    /// classifier claiming 0.95 in a bucket where it is right 0.5 of the time is
    /// off by 0.45, and no single classification reveals that.
    #[must_use]
    pub fn gaps(&self) -> Vec<BucketReport> {
        self.buckets
            .iter()
            .enumerate()
            .map(|(index, (correct, total))| {
                let low = index as f64 / CALIBRATION_BUCKETS as f64;
                let claimed = (low + (index as f64 + 1.0) / CALIBRATION_BUCKETS as f64) / 2.0;
                let observed = if *total == 0 {
                    0.0
                } else {
                    correct / *total as f64
                };
                BucketReport {
                    low,
                    claimed,
                    observed,
                    gap: claimed - observed,
                    samples: *total,
                }
            })
            .collect()
    }

    /// The largest absolute gap over buckets that have samples.
    ///
    /// `None` when nothing has been recorded, which is not the same as a gap of
    /// zero: an unmeasured classifier is not a calibrated one.
    #[must_use]
    pub fn worst_gap(&self) -> Option<BucketReport> {
        self.gaps()
            .into_iter()
            .filter(|report| report.samples > 0)
            .max_by(|a, b| {
                a.gap
                    .abs()
                    .partial_cmp(&b.gap.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// One calibration bucket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BucketReport {
    /// The bottom of the bucket.
    pub low: f64,
    /// The confidence the bucket stands for: its midpoint.
    pub claimed: f64,
    /// How often answers in this bucket were right.
    pub observed: f64,
    /// `claimed - observed`. Positive means overconfident.
    pub gap: f64,
    /// How many outcomes landed here.
    pub samples: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classifier() -> Classifier {
        let mut weights = BTreeMap::new();
        let mut bug = BTreeMap::new();
        bug.insert("stack_trace".to_string(), 3.0);
        bug.insert("crash".to_string(), 2.0);
        bug.insert("feature_request".to_string(), -2.0);
        weights.insert("bug".to_string(), bug);
        let mut request = BTreeMap::new();
        request.insert("feature_request".to_string(), 3.0);
        request.insert("would_be_nice".to_string(), 1.0);
        weights.insert("request".to_string(), request);
        Classifier::new(weights, 0.6).expect("classifier")
    }

    #[test]
    fn a_clear_case_is_answered_with_its_reasons() {
        // Explainable from the evidence rather than from the model.
        let evidence = Evidence::new().with("stack_trace", 1.0).with("crash", 1.0);
        match classifier().classify(&evidence) {
            Verdict::Category {
                category,
                confidence,
                reasons,
            } => {
                assert_eq!(category, "bug");
                assert!(confidence > 0.9, "confidence was {confidence}");
                assert_eq!(reasons.len(), 2);
            }
            other => panic!("expected a category, got {other:?}"),
        }
    }

    #[test]
    fn a_tie_is_not_resolved_by_category_spelling() {
        // Picking the alphabetically first would make the answer depend on how
        // the categories happen to be named.
        let mut weights = BTreeMap::new();
        let mut a = BTreeMap::new();
        a.insert("shared".to_string(), 1.0);
        weights.insert("alpha".to_string(), a);
        let mut b = BTreeMap::new();
        b.insert("shared".to_string(), 1.0);
        weights.insert("beta".to_string(), b);
        let c = Classifier::new(weights, 0.1).expect("classifier");
        match c.classify(&Evidence::new().with("shared", 1.0)) {
            Verdict::Abstain { reason } => assert!(matches!(
                reason,
                Abstention::Contradictory { score, .. } if (score - 1.0).abs() < TIE_EPSILON
            )),
            other => panic!("a tie was answered: {other:?}"),
        }
    }

    #[test]
    fn unsupported_evidence_abstains_as_a_wiring_problem() {
        // Distinct from ambiguous: nothing scored for any category at all.
        let evidence = Evidence::new().with("unrelated", 1.0);
        match classifier().classify(&evidence) {
            Verdict::Abstain { reason } => assert_eq!(reason, Abstention::NoSupport),
            other => panic!("expected an abstention, got {other:?}"),
        }
    }

    #[test]
    fn empty_evidence_abstains() {
        assert_eq!(
            classifier().classify(&Evidence::new()),
            Verdict::Abstain {
                reason: Abstention::NoSupport
            }
        );
    }

    #[test]
    fn a_weak_leader_abstains_below_the_floor() {
        // There is a leader; it is just not leading enough to act on. The reason
        // says which, because "ambiguous" and "weak" need different responses.
        let mut weights = BTreeMap::new();
        let mut a = BTreeMap::new();
        a.insert("one".to_string(), 1.1);
        weights.insert("alpha".to_string(), a);
        let mut b = BTreeMap::new();
        b.insert("two".to_string(), 1.0);
        weights.insert("beta".to_string(), b);
        let c = Classifier::new(weights, 0.6).expect("classifier");
        let evidence = Evidence::new().with("one", 1.0).with("two", 1.0);
        match c.classify(&evidence) {
            Verdict::Abstain { reason } => assert!(
                matches!(reason, Abstention::BelowFloor { .. }),
                "expected a floor abstention, got {reason:?}"
            ),
            other => panic!("expected an abstention, got {other:?}"),
        }
    }

    #[test]
    fn a_negative_weight_suppresses_a_category() {
        // "feature_request" weighs against "bug", so a report carrying both leans
        // the other way.
        let evidence = Evidence::new()
            .with("stack_trace", 1.0)
            .with("feature_request", 1.0)
            .with("would_be_nice", 1.0);
        match classifier().classify(&evidence) {
            Verdict::Category { category, .. } => assert_eq!(category, "request"),
            other => panic!("expected a category, got {other:?}"),
        }
    }

    #[test]
    fn an_out_of_range_floor_is_refused() {
        let weights = BTreeMap::from([("a".to_string(), BTreeMap::new())]);
        assert_eq!(
            Classifier::new(weights.clone(), 1.5),
            Err(ClassifierError::FloorOutOfRange { floor: 1.5 })
        );
        assert_eq!(
            Classifier::new(weights, -0.1),
            Err(ClassifierError::FloorOutOfRange { floor: -0.1 })
        );
    }

    #[test]
    fn a_classifier_with_no_categories_is_refused() {
        // It could never answer, and every call would look like an ambiguous item.
        assert_eq!(
            Classifier::new(BTreeMap::new(), 0.5),
            Err(ClassifierError::NoCategories)
        );
    }

    #[test]
    fn partial_strength_scales_the_signal() {
        // Both categories have support, so the strength actually moves the
        // confidence rather than rescaling a single-candidate score.
        let c = classifier();
        let strong = Evidence::new()
            .with("stack_trace", 1.0)
            .with("feature_request", 1.0);
        let weak = Evidence::new()
            .with("stack_trace", 0.1)
            .with("feature_request", 1.0);
        let high = match c.classify(&strong) {
            Verdict::Category {
                category,
                confidence,
                ..
            } => {
                assert_eq!(category, "bug");
                confidence
            }
            other => panic!("expected a category, got {other:?}"),
        };
        let low = match c.classify(&weak) {
            Verdict::Category {
                category,
                confidence,
                ..
            } => {
                assert_eq!(
                    category, "request",
                    "a tenth-strength signal should not win"
                );
                confidence
            }
            other => panic!("expected a category, got {other:?}"),
        };
        assert!(
            (high - low).abs() > 0.3,
            "strength did not move the confidence: {high} vs {low}"
        );
    }

    #[test]
    fn the_fingerprint_ignores_collection_order() {
        // Two runs over the same item must be shown to have seen the same thing.
        let a = Evidence::new().with("one", 1.0).with("two", 2.0);
        let b = Evidence::new().with("two", 2.0).with("one", 1.0);
        assert_eq!(a.fingerprint(), b.fingerprint());
        let c = Evidence::new().with("one", 1.0).with("two", 3.0);
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn an_honest_classifier_shows_a_small_gap() {
        let mut calibration = Calibration::new();
        // Ten outcomes at 0.95, nine of them right: observed 0.9 against a
        // claimed 0.95.
        for i in 0..10 {
            calibration.record(Outcome {
                confidence: 0.95,
                correct: i < 9,
            });
        }
        assert_eq!(calibration.total(), 10);
        let worst = calibration.worst_gap().expect("worst");
        assert!((worst.claimed - 0.95).abs() < 0.06);
        assert!((worst.observed - 0.9).abs() < 1e-9);
        assert!(worst.gap.abs() < 0.1, "gap was {}", worst.gap);
    }

    #[test]
    fn an_overconfident_classifier_shows_a_large_gap() {
        // Claiming 0.95 and being right half the time is worse than claiming 0.6:
        // the first one's confidence cannot be used to decide anything.
        let mut calibration = Calibration::new();
        for i in 0..10 {
            calibration.record(Outcome {
                confidence: 0.95,
                correct: i % 2 == 0,
            });
        }
        let worst = calibration.worst_gap().expect("worst");
        assert!(worst.gap > 0.4, "gap was {}", worst.gap);
    }

    #[test]
    fn an_unmeasured_classifier_is_not_a_calibrated_one() {
        // None rather than a gap of zero: nothing has been observed.
        let calibration = Calibration::new();
        assert_eq!(calibration.total(), 0);
        assert_eq!(calibration.worst_gap(), None);
    }

    #[test]
    fn a_confidence_outside_the_range_is_not_recorded() {
        // Silently clamping it would put a nonsense claim in a real bucket.
        let mut calibration = Calibration::new();
        calibration.record(Outcome {
            confidence: 1.7,
            correct: true,
        });
        calibration.record(Outcome {
            confidence: -0.2,
            correct: true,
        });
        assert_eq!(calibration.total(), 0);
    }

    #[test]
    fn a_confidence_of_exactly_one_falls_in_the_last_bucket() {
        let mut calibration = Calibration::new();
        calibration.record(Outcome {
            confidence: 1.0,
            correct: true,
        });
        assert_eq!(calibration.total(), 1);
        let worst = calibration.worst_gap().expect("worst");
        assert!((worst.low - 0.9).abs() < 1e-9);
    }

    #[test]
    fn the_declared_categories_are_listed_in_order() {
        assert_eq!(classifier().categories(), vec!["bug", "request"]);
    }
}
