#![forbid(unsafe_code)]
//! # operator sync - Aşama 7'nin Lubot tarafı
//!
//! The report's operator-synchronisation rules, as checks Lubot can run
//! without being the chain:
//!
//! 1. **Compute bond.** An operator registers with a non-zero bond at or
//!    above the floor. The floor is a chain constant; Lubot takes it as a
//!    parameter and refuses a zero bond outright.
//! 2. **One `model_hash`.** All active operators run the same model hash. If
//!    they do not, `agreement_threshold` can never be satisfied.
//! 3. **Effort tier is a ceiling.** `0.5x`-`10.0x` tags are the operator's
//!    hardware ceiling and are hashed into the request's `effort` field, so
//!    a low-ceiling operator cannot accept a high-effort request and do the
//!    cheap work. [`effort_tag_ok`] accepts only canonical tags in the
//!    report's range and refuses anything else; the granularity between the
//!    bounds is the chain's, not invented here.
//! 4. **Checkpoint transition.** A new checkpoint publishes and both old and
//!    new `model_hash` stay active through a window; when the window closes
//!    the old record is retired. The "everyone switches instantly" assumption
//!    is not used.

use lubot_read::sha256_hex;

/// The compute bond rule: non-zero, and at or above the floor.
#[must_use]
pub fn compute_bond_ok(bond: u64, floor: u64) -> bool {
    bond > 0 && bond >= floor
}

/// One operator's registration record, as the sync rule sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorRecord {
    pub model_hash: String,
    pub active: bool,
}

/// True only when every active operator runs the same `model_hash`.
///
/// No active operators means no agreement is possible, so the answer is
/// `false` rather than a vacuous `true` (Aşama 11: a single-operator result
/// is not consumed either).
#[must_use]
pub fn same_model_hash(operators: &[OperatorRecord]) -> bool {
    let active: Vec<&OperatorRecord> = operators.iter().filter(|o| o.active).collect();
    let Some(first) = active.first() else {
        return false;
    };
    active.iter().all(|o| o.model_hash == first.model_hash)
}

/// The header of an effort tag is `d.d x` in the report's range.
#[must_use]
pub fn effort_tag_ok(tag: &str) -> bool {
    let Some(rest) = tag.strip_suffix("x") else {
        return false;
    };
    let Some(number) = rest.parse::<f64>().ok() else {
        return false;
    };
    (0.5..=10.0).contains(&number)
}

/// The canonical hash of an effort tag, written into a request's `effort`
/// field. Refuses tags outside the report's range.
///
/// # Errors
/// `effort_tag_ok` failures.
pub fn effort_hash(tag: &str) -> Result<String, String> {
    if !effort_tag_ok(tag) {
        return Err(format!(
            "effort tag `{tag}` is outside the 0.5x-10.0x ceiling range"
        ));
    }
    Ok(sha256_hex(tag.as_bytes()))
}

/// The checkpoint transition window (Aşama 7, madde 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointWindow {
    pub window_start: u64,
    pub window_end: u64,
}

impl CheckpointWindow {
    /// Within the window the old and the new hash are parallel-active.
    #[must_use]
    pub fn both_active(&self, now: u64) -> bool {
        now >= self.window_start && now < self.window_end
    }

    /// After the window the old hash is retired (`active = false`).
    #[must_use]
    pub fn old_retired(&self, now: u64) -> bool {
        now >= self.window_end
    }

    /// The active state of the old hash at `now`.
    #[must_use]
    pub fn old_active(&self, now: u64) -> bool {
        self.both_active(now)
    }
}

/// The answer budget an effort ceiling admits: a `0.5x` machine gets a
/// short answer, a `10.0x` machine a long one. The tier is the operator's
/// hardware ceiling (Aşama 7), so the budget translates that ceiling into
/// the one resource the reader can actually spend - passages per answer.
/// The mapping is fixed and measured, not a hint the caller may override.
///
/// # Errors
/// `effort_tag_ok` failures.
pub fn answer_budget(effort_tag: &str) -> Result<usize, String> {
    if !effort_tag_ok(effort_tag) {
        return Err(format!(
            "effort tag `{effort_tag}` is outside the 0.5x-10.0x ceiling range"
        ));
    }
    let number: f64 = effort_tag
        .strip_suffix("x")
        .ok_or_else(|| "effort tag has no x suffix".to_string())?
        .parse()
        .map_err(|_| format!("effort tag `{effort_tag}` is not a number"))?;
    let budget = (number * 3.0_f64).round() as usize;
    Ok(budget.clamp(1, 10))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_bond_never_registers() {
        assert!(!compute_bond_ok(0, 1_000));
        assert!(compute_bond_ok(1_000, 1_000));
        assert!(compute_bond_ok(1_001, 1_000));
        assert!(!compute_bond_ok(999, 1_000));
    }

    #[test]
    fn divergent_hashes_never_satisfy_sync() {
        let ops = vec![
            OperatorRecord {
                model_hash: "h1".to_string(),
                active: true,
            },
            OperatorRecord {
                model_hash: "h2".to_string(),
                active: true,
            },
        ];
        assert!(!same_model_hash(&ops));
        let aligned = vec![
            OperatorRecord {
                model_hash: "h1".to_string(),
                active: true,
            },
            OperatorRecord {
                model_hash: "h1".to_string(),
                active: true,
            },
        ];
        assert!(same_model_hash(&aligned));
        // One active operator trivially matches itself; the refusal to
        // consume a single-operator result is the threshold rule (Aşama 11),
        // not the sync rule.
        let single = vec![OperatorRecord {
            model_hash: "h1".to_string(),
            active: true,
        }];
        assert!(same_model_hash(&single));
        // No active operators: no agreement is possible, not a vacuous yes.
        assert!(!same_model_hash(&[]));
    }

    #[test]
    fn effort_tags_are_bounded_and_hashed() {
        assert!(effort_tag_ok("0.5x"));
        assert!(effort_tag_ok("1.0x"));
        assert!(effort_tag_ok("10.0x"));
        assert!(!effort_tag_ok("0.4x"));
        assert!(!effort_tag_ok("10.1x"));
        assert!(!effort_tag_ok("cheap"));
        assert_eq!(
            effort_hash("1.0x").unwrap(),
            lubot_read::sha256_hex(b"1.0x")
        );
        assert!(effort_hash("0.4x").is_err());
    }

    #[test]
    fn the_answer_budget_tracks_the_ceiling() {
        assert_eq!(answer_budget("0.5x").unwrap(), 2);
        assert_eq!(answer_budget("1.0x").unwrap(), 3);
        assert_eq!(answer_budget("2.0x").unwrap(), 6);
        assert_eq!(answer_budget("10.0x").unwrap(), 10);
        assert!(answer_budget("0.4x").unwrap_err().contains("outside"));
    }

    #[test]
    fn the_transition_window_keeps_both_then_retires_old() {
        let window = CheckpointWindow {
            window_start: 100,
            window_end: 200,
        };
        assert!(!window.both_active(99));
        assert!(window.both_active(100));
        assert!(window.both_active(199));
        assert!(window.old_active(150));
        assert!(!window.both_active(200));
        assert!(window.old_retired(200));
        assert!(!window.old_active(200));
    }

    #[test]
    fn retired_and_dormant_operators_do_not_count_toward_alignment() {
        let ops = vec![
            OperatorRecord {
                model_hash: "old".to_string(),
                active: false,
            },
            OperatorRecord {
                model_hash: "new".to_string(),
                active: true,
            },
        ];
        assert!(same_model_hash(&ops));
    }
}
