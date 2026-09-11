#![forbid(unsafe_code)]
//! # operator sync - Aşama 7'nin Lubot tarafı: tavan
//!
//! The report's operator-synchronisation rules split by who can check them.
//! **Effort tier is a ceiling:** `0.5x`-`10.0x` tags are the operator's
//! hardware ceiling and are hashed into the request's `effort` field, so a
//! low-ceiling operator cannot accept a high-effort request and do the cheap
//! work. [`effort_tag_ok`] accepts only canonical tags in the report's range
//! and refuses anything else; the granularity between the bounds is the
//! chain's, not invented here. [`answer_budget`] translates the ceiling into
//! passages per answer - fixed and measured, not a hint to override.
//!
//! The other three rules - non-zero compute bond above the floor, one
//! `model_hash` across active operators, the checkpoint transition window -
//! described records Lubot never holds. There is no registration file, no
//! operator set, no window producer in this tree; README's own scope line
//! assigns registration and bond to the chain's AI inference layer "in the
//! node". They were deleted with their accessors rather than kept as a
//! library for a caller that does not exist, and they return, callers in
//! the same patch, if Lubot ever takes a side of the registry.
use lubot_read::sha256_hex;

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

}
