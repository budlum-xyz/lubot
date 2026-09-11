//! The activation ledger, as the inventory reads it.
//!
//! `lubot-esik` keeps the schedule of behavior changes - flags that flip at an
//! agreed epoch - and this module is its reader inside this repository: it parses
//! `training/activation.jsonl` into an [`esik::Registry`] and renders the line the
//! inventory wants, "what does the schedule say right now, and what is the next
//! epoch a change may open at". It is deliberately thin: no fs surprises (a
//! missing ledger is an empty ledger, a corrupt one is an error), no clock (the
//! reporting epoch arrives as a parameter, exactly like the crate it reads), and
//! no re-implemented comparison - `live(flag, at)` and `is_live_at(at)` are asked,
//! and their agreement is checked rather than assumed.
//!
//! The record format is one JSON object per line:
//!
//! ```text
//! {"plan":"placement","plan_epoch":4400,"note":"agreed 2026-08","flag":"BDLM_X","activates_at":4410,"effect":"consensus","why":"moves the state root","ratified_at":4410}
//! ```
//!
//! `why` and `ratified_at` are optional; every other field is required. One line
//! describes one change; a plan may be named by many lines and must agree on
//! `plan_epoch` and `note` across them (the Registry refuses drift anyway -
//! this parser only refuses earlier and with a line number).

use std::collections::BTreeMap;
use std::path::Path;

use lubot_esik::{Change, Effect, Epoch, Plan, Registry, Status};
use serde_json::Value;

/// Where the inventory looks for the ledger, unless told otherwise.
pub const DEFAULT_LEDGER: &str = "training/activation.jsonl";

fn effect_from(name: &str) -> Result<Effect, String> {
    match name {
        "consensus" => Ok(Effect::Consensus),
        "local" => Ok(Effect::Local),
        "freeze" => Ok(Effect::Freeze),
        other => Err(format!("unknown effect `{other}`")),
    }
}

fn str_field(rec: &Value, key: &str) -> Result<String, String> {
    rec.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing string field `{key}`"))
}

fn u64_field(rec: &Value, key: &str) -> Result<u64, String> {
    rec.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing integer field `{key}`"))
}

fn opt_u64_field(rec: &Value, key: &str) -> Result<Option<u64>, String> {
    match rec.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("field `{key}` is not an integer")),
    }
}

/// Parse the ledger text. Plan declarations are collected first so every change
/// can be admitted against a plan that already lists it; ratifications are applied
/// last, through `Registry::ratify`, which is the only writer of that record.
///
/// # Errors
///
/// A malformed or refused record fails with its line number; a partial file is
/// never reported as a shorter schedule.
pub fn parse_ledger(text: &str) -> Result<Registry, String> {
    struct Line {
        line_no: usize,
        plan_name: String,
        flag: String,
        activates_at: Epoch,
        effect: Effect,
        why: Option<String>,
        ratified_at: Option<Epoch>,
    }
    let mut plans: BTreeMap<String, (Epoch, String, Vec<String>, usize)> = BTreeMap::new();
    let mut lines: Vec<Line> = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        if raw.trim().is_empty() {
            continue;
        }
        let rec: Value =
            serde_json::from_str(raw).map_err(|e| format!("line {line_no}: not JSON: {e}"))?;
        let plan_name = str_field(&rec, "plan").map_err(|e| format!("line {line_no}: {e}"))?;
        let plan_epoch =
            u64_field(&rec, "plan_epoch").map_err(|e| format!("line {line_no}: {e}"))?;
        let note = str_field(&rec, "note").map_err(|e| format!("line {line_no}: {e}"))?;
        let flag = str_field(&rec, "flag").map_err(|e| format!("line {line_no}: {e}"))?;
        let activates_at =
            u64_field(&rec, "activates_at").map_err(|e| format!("line {line_no}: {e}"))?;
        let effect_raw = str_field(&rec, "effect").map_err(|e| format!("line {line_no}: {e}"))?;
        let effect =
            effect_from(&effect_raw).map_err(|e| format!("line {line_no}: {e}"))?;
        let why = match rec.get("why") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("line {line_no}: `why` is not a string"))?,
            ),
        };
        let ratified_at =
            opt_u64_field(&rec, "ratified_at").map_err(|e| format!("line {line_no}: {e}"))?;

        match plans.get(&plan_name) {
            Some((epoch, known_note, _, _)) if *epoch != plan_epoch || *known_note != note => {
                return Err(format!(
                    "line {line_no}: plan `{plan_name}` re-declared with different \
                     epoch or note"
                ));
            }
            Some((_, _, flags, _)) => {
                if !flags.contains(&flag) {
                    flags.push(flag.clone());
                }
            }
            None => {
                plans.insert(
                    plan_name.clone(),
                    (plan_epoch, note.clone(), vec![flag.clone()], line_no),
                );
            }
        }
        lines.push(Line {
            line_no,
            plan_name,
            flag,
            activates_at,
            effect,
            why,
            ratified_at,
        });
    }

    let mut reg = Registry::new();
    for (name, (epoch, note, flags, line_no)) in &plans {
        let mut plan = Plan::new(name, *epoch).with_note(note);
        for f in flags {
            plan = plan.with_flag(f);
        }
        reg.declare(plan)
            .map_err(|e| format!("line {line_no}: plan `{name}` refused: {e}"))?;
    }
    for l in &lines {
        let mut change = Change::new(&l.flag, l.activates_at, l.effect, &l.plan_name);
        if let Some(why) = &l.why {
            change = change.with_why(why);
        }
        reg.admit(change)
            .map_err(|e| format!("line {}: change `{}` refused: {e}", l.line_no, l.flag))?;
    }
    for l in &lines {
        if let Some(at) = l.ratified_at {
            reg.ratify(&l.flag, at)
                .map_err(|e| format!("line {}: ratify `{}`: {e}", l.line_no, l.flag))?;
        }
    }
    Ok(reg)
}

/// The line the inventory prints, at the given epoch. If the ledger does not
/// verify at `now`, that is what the line says - a schedule with a hole is not
/// summarized as if it were sound.
#[must_use]
pub fn report(reg: &Registry, now: Epoch) -> String {
    if let Err(violation) = reg.verify(now) {
        return format!("ledger invalid at {now}: {violation}");
    }
    let pending = reg.pending_at(now);
    let live = reg.live_at(now);
    let unratified = reg
        .changes()
        .iter()
        .filter(|c| c.status_at(now) == Status::Unratified)
        .count();
    let quorum_due = live
        .iter()
        .chain(pending.iter())
        .filter(|c| c.effect().requires_quorum() && !c.is_live_at(now))
        .count();
    let mut checked_on = 0usize;
    for c in live.iter().chain(pending.iter()) {
        let says = reg.live(c.flag(), now).is_on();
        if says != c.is_live_at(now) {
            return format!(
                "ledger inconsistent: `{}` live check disagrees at {now}",
                c.flag()
            );
        }
        checked_on += usize::from(says);
    }
    let earliest = reg.earliest_open(now);
    format!(
        "{} plan(s); {checked_on} on at {now}, {} pending, {unratified} unratified, \
         {quorum_due} awaiting quorum; earliest open epoch {earliest}",
        reg.plans().len(),
        pending.len()
    )
}

/// Read the ledger file. Absent is empty, not an error: an un-scheduled node has
/// nothing to disagree about, and the inventory says so on the same line.
///
/// # Errors
///
/// Unreadable-but-present files and malformed records fail with detail.
pub fn load(path: &Path) -> Result<Registry, String> {
    if !path.exists() {
        return Ok(Registry::new());
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("activation: read {}: {e}", path.display()))?;
    parse_ledger(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAG: &str = "BDLM_MAINTENANCE_PLACEMENT_V1";

    fn one_line(extra: &str) -> String {
        format!(
            "{{\"plan\":\"placement\",\"plan_epoch\":4400,\"note\":\"agreed\",\
             \"flag\":\"{FLAG}\",\"activates_at\":4410,\"effect\":\"consensus\",{extra}}}\n"
        )
    }

    #[test]
    fn a_pending_schedule_reports_pending_not_broken() {
        let reg = parse_ledger(&one_line("\"why\":\"moves the state root\"")).expect("parses");
        let line = report(&reg, 4409);
        assert!(line.contains("1 pending"), "{line}");
        assert!(!line.contains("invalid"), "{line}");
        assert!(line.contains("awaiting quorum"), "{line}");
    }

    #[test]
    fn a_ratified_change_is_on_and_not_unratified() {
        let reg = parse_ledger(&one_line(
            "\"why\":\"moves the state root\",\"ratified_at\":4410",
        ))
        .expect("parses");
        assert!(reg.change(FLAG).expect("present").is_live_at(4410));
        let line = report(&reg, 4410);
        assert!(line.contains("0 unratified"), "{line}");
        assert!(line.contains("1 on at 4410"), "{line}");
    }

    #[test]
    fn a_missing_plan_or_a_bad_effect_fails_with_the_line_number() {
        let bad_effect = format!(
            "{{\"plan\":\"p\",\"plan_epoch\":1,\"note\":\"n\",\"flag\":\"F\",\
             \"activates_at\":2,\"effect\":\"maybe\"}}\n"
        );
        let err = parse_ledger(&bad_effect).unwrap_err();
        assert!(err.starts_with("line 1:"), "{err}");
    }
}
