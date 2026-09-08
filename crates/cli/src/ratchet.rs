//! # ratchet - measured baselines may only rise
//!
//! The ratchet power, in code: every regression-prone number this repository
//! claims (tests, gates, corpus records) has a baseline in
//! `training/ratchet.json`, and the checks refuse a run where a number fell
//! below its baseline. One number is inverted by nature: the clippy pedantic
//! warning count may only fall, never rise - the baseline is a ceiling.
//!
//! Lowering a baseline is possible but never silent: `--set` rewrites the
//! file to exactly what was measured, and the rewrite is a commit like any
//! other. A baseline that was never touched is a claim the run must live up
//! to.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// One baseline: the best known value for a measured number.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Baseline {
    /// `cargo test --workspace` passing tests.
    pub tests: u64,
    /// Gates in `gates/check.py`.
    pub gates: u64,
    /// `clippy -W pedantic` warnings: may only fall.
    pub pedantic: u64,
    /// Records in `corpus/*.jsonl.gz`.
    pub corpus: u64,
}

/// Load the baseline file.
///
/// # Errors
/// Missing or unparsable files refuse - a run without a baseline is a run
/// without a claim.
pub fn load(path: &Path) -> Result<Baseline, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write the baseline file.
///
/// # Errors
/// File creation or serialization failures.
pub fn save(path: &Path, baseline: &Baseline) -> Result<(), String> {
    let text = serde_json::to_string_pretty(baseline).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// One comparison result, named so the report reads both directions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diff {
    pub field: &'static str,
    pub baseline: u64,
    pub measured: u64,
    /// True when the measurement regressed (moved away from the goal).
    pub regressed: bool,
}

/// Compare a measurement to the baseline. `pedantic` counts down (warnings
/// may only fall); everything else counts up (may only rise).
#[must_use]
pub fn compare(measured: &Baseline, baseline: &Baseline) -> Vec<Diff> {
    let rising = [
        ("tests", baseline.tests, measured.tests),
        ("gates", baseline.gates, measured.gates),
        ("corpus", baseline.corpus, measured.corpus),
    ];
    let mut diffs: Vec<Diff> = rising
        .iter()
        .map(|(field, base, value)| Diff {
            field,
            baseline: *base,
            measured: *value,
            regressed: *value < *base,
        })
        .collect();
    diffs.push(Diff {
        field: "pedantic",
        baseline: baseline.pedantic,
        measured: measured.pedantic,
        regressed: measured.pedantic > baseline.pedantic,
    });
    diffs
}

/// The four baseline-bearing counts, measured by talking to the same
/// commands the gates run - the numbers are not re-derived from prose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Measured {
    pub tests: u64,
    pub gates: u64,
    pub pedantic: u64,
    pub corpus: u64,
}

impl Measured {
    #[must_use]
    pub fn as_baseline(&self) -> Baseline {
        Baseline {
            tests: self.tests,
            gates: self.gates,
            pedantic: self.pedantic,
            corpus: self.corpus,
        }
    }
}

/// Run `cargo test --workspace` and sum the passing counts.
///
/// # Errors
/// When the suite itself fails - a falling suite must never be measured as a
/// claim.
pub fn measure_tests(exe: &str) -> Result<u64, String> {
    let out = run_capture(exe, &["test", "--workspace"])?;
    let mut total = 0u64;
    for line in out.lines() {
        if let Some(rest) = line.strip_prefix("test result: ok.") {
            if let Some(count) = rest.split_whitespace().next() {
                total += count
                    .parse::<u64>()
                    .map_err(|_| format!("test count: `{count}`"))?;
            }
        }
    }
    if total == 0 {
        return Err("measure: cargo test reported no passing tests".to_string());
    }
    Ok(total)
}

/// Count `clippy -W pedantic` warnings on stderr. Two meta lines are not
/// code warnings and are excluded: the "generated N warnings" summaries, and
/// the deprecation notice for the `pedantic` lint group itself.
///
/// # Errors
/// When clippy fails to run; the lint level is advisory, so lint findings
/// do not fail this measurement - the ratchet does.
pub fn measure_pedantic(exe: &str) -> Result<u64, String> {
    let out = run_capture_stderr(
        exe,
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-W",
            "pedantic",
        ],
    )?;
    Ok(out
        .lines()
        .filter(|l| {
            l.starts_with("warning:") && !l.contains("generated") && !l.contains("is deprecated")
        })
        .count() as u64)
}

/// Count the gates by listing them with the gate script itself.
///
/// # Errors
/// When the script cannot be run.
pub fn measure_gates(python: &str, script: &str) -> Result<u64, String> {
    let out = run_capture(python, &[script, "--list"])?;
    let count = out.lines().filter(|l| !l.trim().is_empty()).count() as u64;
    if count == 0 {
        return Err("measure: gate list is empty".to_string());
    }
    Ok(count)
}

/// Count records across `corpus/*.jsonl.gz`. Digest verification is the
/// loader's job; a baseline is a count, and counting reads lines only.
///
/// # Errors
/// A corpus file that cannot be opened refuses the count.
pub fn measure_corpus(dir: &Path) -> Result<u64, String> {
    use std::io::BufRead;
    let mut total = 0u64;
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut found = false;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().map(|e| e == "gz").unwrap_or(false)
            && path
                .file_name()
                .map(|n| n.to_string_lossy().starts_with("knowledge-"))
                .unwrap_or(false)
        {
            found = true;
            let file =
                std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let reader = std::io::BufReader::new(flate2::read::GzDecoder::new(file));
            total += reader.lines().count() as u64;
        }
    }
    if !found {
        return Err(format!(
            "{}: no knowledge-*.jsonl.gz corpus files",
            dir.display()
        ));
    }
    Ok(total)
}

fn run_capture(exe: &str, args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new(exe)
        .args(args)
        .output()
        .map_err(|e| format!("{exe}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{exe} {}: exited {}: {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn run_capture_stderr(exe: &str, args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new(exe)
        .args(args)
        .output()
        .map_err(|e| format!("{exe}: {e}"))?;
    Ok(String::from_utf8_lossy(&out.stderr).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> Baseline {
        Baseline {
            tests: 138,
            gates: 24,
            pedantic: 17,
            corpus: 793,
        }
    }

    #[test]
    fn an_unchanged_measurement_never_regresses() {
        let measured = baseline();
        let diffs = compare(&measured, &baseline());
        assert!(diffs.iter().all(|d| !d.regressed), "{diffs:?}");
    }

    #[test]
    fn a_falling_test_count_is_a_regression() {
        let mut measured = baseline();
        measured.tests -= 1;
        let diffs = compare(&measured, &baseline());
        let tests = diffs.iter().find(|d| d.field == "tests").unwrap();
        assert!(tests.regressed);
        assert_eq!(tests.measured, 137);
    }

    #[test]
    fn a_rising_pedantic_count_is_a_regression_but_falling_is_not() {
        let mut measured = baseline();
        measured.pedantic += 1;
        assert!(compare(&measured, &baseline())
            .iter()
            .any(|d| d.field == "pedantic" && d.regressed));
        measured.pedantic -= 2;
        assert!(!compare(&measured, &baseline())
            .iter()
            .any(|d| d.field == "pedantic" && d.regressed));
    }

    #[test]
    fn the_baseline_round_trips() {
        let dir = std::env::temp_dir().join(format!("lubot-ratchet-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ratchet.json");
        save(&path, &baseline()).unwrap();
        assert_eq!(load(&path).unwrap(), baseline());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn measured_snaps_to_a_baseline() {
        let m = Measured {
            tests: 1,
            gates: 2,
            pedantic: 3,
            corpus: 4,
        };
        let b = m.as_baseline();
        assert_eq!(b.tests, 1);
        assert_eq!(b.pedantic, 3);
    }
}
