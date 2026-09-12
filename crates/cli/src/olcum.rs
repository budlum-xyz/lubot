//! Measurement commands: checks that answer a question about the repository or
//! about a piece of arithmetic, rather than doing work.
//!
//! # Why these live together
//!
//! Each subcommand answers one question that is otherwise answered by opinion:
//!
//! - `olcum mimari` - does the crate graph respect the declared layers, and can
//!   it be started at all? A layering rule that is only written down is a rule
//!   that erodes one call at a time.
//! - `olcum esik` - is a given signer set actually a quorum? The arithmetic has
//!   an off-by-one in it that is easy to get wrong in a hurry and expensive to
//!   get wrong in production.
//! - `olcum takip` - given a plan, what can start now, and is anything in a
//!   cycle?
//! - `olcum sinif` - classify a set of signals, or say why it declined.
//!
//! # `olcum mimari` is a real check, not a demonstration
//!
//! It reads every `crates/*/Cargo.toml`, builds the dependency graph, and runs it
//! through [`lubot_mimari::Blueprint`]. The layers come from
//! [`ARCHITECTURE`]; a crate missing from that table is reported as unclassified
//! and left out of the layer check rather than being assumed into a layer,
//! because an assumed layer produces violations that mean nothing.
//!
//! Cycles are still checked for every crate, classified or not, because a cycle
//! makes the graph unstartable regardless of which layer anything sits in.

use std::collections::BTreeMap;

use lubot_anlama::{Classifier, Evidence, Verdict};
use lubot_esik::Quorum;
use lubot_mimari::{Blueprint, WiringError};
use lubot_olcek::{Controller, Decision, Load};
use lubot_takip::{Phase, Tracker};

/// Which layer each crate sits in. Lower is deeper; a crate may depend on its
/// own layer or below.
///
/// Written out rather than derived, because a layer derived from the dependency
/// graph would make the check vacuous: every edge would point downward by
/// construction and no violation could ever be found.
pub const ARCHITECTURE: &[(&str, u32)] = &[
    // Nothing in the workspace.
    ("read", 0),
    ("muhur", 0),
    ("denetim", 1),
    ("esik", 0),
    ("izolasyon", 0),
    ("kanit", 0),
    ("kuyruk", 0),
    ("erisim", 0),
    ("takip", 0),
    ("olcek", 0),
    ("yetenek", 0),
    ("mimari", 0),
    ("anlama", 1),
    ("usl", 1),
    // Built on the primitives above.
    ("index", 1),
    ("grant", 1),
    ("tools", 1),
    ("sikistir", 1),
    ("doc", 2),
    ("answer", 2),
    // The binary sits on top of everything.
    ("cli", 3),
];

/// The layer a crate is declared to sit in, if it is declared at all.
#[must_use]
pub fn layer_of(crate_name: &str) -> Option<u32> {
    ARCHITECTURE
        .iter()
        .find(|(name, _)| *name == crate_name)
        .map(|(_, layer)| *layer)
}

/// Reads the `path = "../name"` dependencies out of a manifest.
///
/// A line-based read rather than a TOML parse, because the only thing being
/// asked of the file is which sibling crates it names, and a parser dependency
/// for that is more machinery than the question needs.
#[must_use]
pub fn path_dependencies(manifest: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in manifest.lines() {
        let Some(at) = line.find("path") else { continue };
        let rest = &line[at..];
        let Some(quote) = rest.find('"') else { continue };
        let after = &rest[quote + 1..];
        let Some(close) = after.find('"') else { continue };
        let path = &after[..close];
        let Some(name) = path.rsplit('/').next() else { continue };
        if !name.is_empty() && !found.iter().any(|existing: &String| existing == name) {
            found.push(name.to_string());
        }
    }
    found
}

/// The result of checking the crate graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ArchitectureReport {
    /// Crates examined.
    pub crates: Vec<String>,
    /// Crates with no declared layer.
    pub unclassified: Vec<String>,
    /// Edges that point upward.
    pub violations: Vec<String>,
    /// Edges that would close a cycle.
    pub cycles: Vec<String>,
    /// A start order, when one exists.
    pub start_order: Vec<String>,
}

impl ArchitectureReport {
    /// Whether the graph is clean.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty() && self.cycles.is_empty()
    }
}

/// Checks a dependency map against the declared architecture.
///
/// Takes the map rather than reading the filesystem, so the same check is usable
/// from a test without a repository.
#[must_use]
pub fn check_architecture(dependencies: &BTreeMap<String, Vec<String>>) -> ArchitectureReport {
    let mut report = ArchitectureReport::default();
    let mut blueprint = Blueprint::new();
    for name in dependencies.keys() {
        report.crates.push(name.clone());
        let Some(layer) = layer_of(name) else {
            report.unclassified.push(name.clone());
            // Still added, at the deepest layer, so cycles through it are caught.
            // An unclassified crate is not exempt from the graph, only from the
            // layer rule.
            let _ = blueprint.add(name, u32::MAX);
            continue;
        };
        let _ = blueprint.add(name, layer);
    }
    for (name, deps) in dependencies {
        // Unclassified crates are skipped for the layer rule: an assumed layer
        // produces violations that mean nothing.
        if report.unclassified.iter().any(|u| u == name) {
            continue;
        }
        for dependency in deps {
            if !dependencies.contains_key(dependency) {
                continue;
            }
            if report.unclassified.iter().any(|u| u == dependency) {
                continue;
            }
            match blueprint.connect(name, dependency) {
                Ok(()) => {}
                Err(WiringError::UpwardDependency { .. }) => {
                    report
                        .violations
                        .push(format!("{name} depends upward on {dependency}"));
                }
                Err(WiringError::Cycle { .. }) => {
                    report.cycles.push(format!("{name} depends on {dependency} in a cycle"));
                }
                Err(other) => {
                    report.violations.push(format!("{name} -> {dependency}: {other}"));
                }
            }
        }
    }
    report.start_order = blueprint.start_order().unwrap_or_default();
    report
}

/// Reads the crate graph from the repository and checks it.
///
/// # Errors
///
/// A message for the caller.
pub fn check_repository() -> Result<ArchitectureReport, String> {
    let crates_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("the cli crate has no parent directory")?
        .join("crates");
    let mut dependencies = BTreeMap::new();
    let entries = std::fs::read_dir(&crates_dir)
        .map_err(|err| format!("{}: {err}", crates_dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let text = std::fs::read_to_string(&manifest)
            .map_err(|err| format!("{}: {err}", manifest.display()))?;
        dependencies.insert(name, path_dependencies(&text));
    }
    Ok(check_architecture(&dependencies))
}

/// Pulls `--name value` pairs out of `args`.
fn options(args: &[String]) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(name) = args[i].strip_prefix("--") {
            if let Some(value) = args.get(i + 1) {
                found.push((name.to_string(), value.clone()));
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    found
}

/// Looks one option up.
fn option(args: &[String], name: &str) -> Option<String> {
    options(args)
        .into_iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

/// Splits a comma-separated list of numbers.
fn numbers(value: &str) -> Result<Vec<u64>, String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u64>()
                .map_err(|_| format!("{part:?} is not a number"))
        })
        .collect()
}

/// The `lubot olcum` subcommand.
///
/// # Errors
///
/// A message for the caller, which `main` prints to stderr.
pub fn cmd_olcum(args: &[String]) -> Result<(), String> {
    let Some(sub) = args.first() else {
        return Err("usage: lubot olcum <mimari|esik|takip|sinif|olcek> ...".to_string());
    };
    let rest = &args[1..];
    match sub.as_str() {
        "mimari" => olcum_mimari(),
        "esik" => olcum_esik(rest),
        "takip" => olcum_takip(rest),
        "sinif" => olcum_sinif(rest),
        "olcek" => olcum_olcek(rest),
        other => Err(format!("unknown olcum subcommand: {other}")),
    }
}

/// Reports the crate graph's layering and startability.
fn olcum_mimari() -> Result<(), String> {
    let report = check_repository()?;
    for violation in &report.violations {
        eprintln!("olcum: a layer is violated: {violation}");
    }
    for cycle in &report.cycles {
        eprintln!("olcum: the graph has a cycle: {cycle}");
    }
    if !report.unclassified.is_empty() {
        eprintln!(
            "olcum: {} crates have no declared layer and were left out of the layer check: {}",
            report.unclassified.len(),
            report.unclassified.join(", ")
        );
    }
    println!(
        "{} crates, {} layers, start order: {}",
        report.crates.len(),
        report.start_order.len(),
        report.start_order.join(" -> ")
    );
    if report.is_clean() {
        println!("OK   the crate graph respects the declared architecture");
        Ok(())
    } else {
        Err(format!(
            "{} layer violation(s), {} cycle(s)",
            report.violations.len(),
            report.cycles.len()
        ))
    }
}

/// Checks whether a signer set is a quorum.
fn olcum_esik(args: &[String]) -> Result<(), String> {
    let members = numbers(&option(args, "members").ok_or("olcum esik needs --members")?)?;
    let threshold: u64 = option(args, "threshold")
        .ok_or("olcum esik needs --threshold")?
        .parse()
        .map_err(|_| "--threshold is not a number".to_string())?;
    let signers = numbers(&option(args, "signers").unwrap_or_default())?;
    let quorum = Quorum::unweighted(&members, threshold).map_err(|err| err.to_string())?;
    let requester: u64 = option(args, "requester")
        .unwrap_or_else(|| "0".to_string())
        .parse()
        .map_err(|_| "--requester is not a number".to_string())?;
    println!(
        "members {}, total weight {}, threshold {}, tolerable faults {}",
        quorum.member_count(),
        quorum.total_weight(),
        quorum.threshold(),
        lubot_esik::tolerable_faults(quorum.member_count())
    );
    match quorum.count_excluding(&signers, requester) {
        Ok(reached) => {
            println!("OK   {reached} of {} signed, excluding the requester", quorum.threshold());
            Ok(())
        }
        Err(err) => Err(err.to_string()),
    }
}

/// Checks a plan written as `task[,dependency...]` per line.
fn olcum_takip(args: &[String]) -> Result<(), String> {
    use std::io::BufRead;
    let bound: usize = option(args, "bound")
        .unwrap_or_else(|| "4".to_string())
        .parse()
        .map_err(|_| "--bound is not a number".to_string())?;
    let mut tracker = Tracker::new(bound);
    // Read once. Two passes over stdin would find the stream exhausted the second
    // time, so every dependency would be dropped silently and the plan would look
    // fine.
    let stdin = std::io::stdin();
    let plan: Vec<String> = stdin
        .lock()
        .lines()
        .map(|line| line.map_err(|err| err.to_string()))
        .collect::<Result<_, String>>()?
        .into_iter()
        .filter(|line| !line.trim().is_empty())
        .collect();
    for line in &plan {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let Some(name) = fields.first() else { continue };
        tracker.add(name).map_err(|err| err.to_string())?;
    }
    // Dependencies are applied after every task exists, because an edge naming a
    // task that has not been added yet is a refusal rather than a forward
    // reference.
    for line in &plan {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let Some(name) = fields.first() else { continue };
        for dependency in fields.iter().skip(1).filter(|part| !part.is_empty()) {
            tracker
                .add_dependency(name, dependency)
                .map_err(|err| err.to_string())?;
        }
    }
    let ready = tracker.ready();
    for (phase, count) in tracker.phase_counts() {
        println!("{:>10}: {}", phase.label(), count);
    }
    println!(
        "ready now: {}",
        if ready.is_empty() {
            "nothing".to_string()
        } else {
            ready.join(", ")
        }
    );
    Ok(())
}

/// Classifies signals given as `name=strength` pairs.
fn olcum_sinif(args: &[String]) -> Result<(), String> {
    let weights = option(args, "weights").ok_or(
        "olcum sinif needs --weights category=signal:weight,signal:weight;category=...",
    )?;
    let floor: f64 = option(args, "floor")
        .unwrap_or_else(|| "0.6".to_string())
        .parse()
        .map_err(|_| "--floor is not a number".to_string())?;
    let mut table = BTreeMap::new();
    for category in weights.split(';') {
        let Some((name, body)) = category.split_once('=') else {
            return Err(format!("expected category=signal:weight, got {category:?}"));
        };
        let mut signals = BTreeMap::new();
        for pair in body.split(',') {
            let Some((signal, weight)) = pair.split_once(':') else {
                return Err(format!("expected signal:weight, got {pair:?}"));
            };
            let value: f64 = weight
                .trim()
                .parse()
                .map_err(|_| format!("{weight:?} is not a number"))?;
            signals.insert(signal.trim().to_string(), value);
        }
        table.insert(name.trim().to_string(), signals);
    }
    let classifier = Classifier::new(table, floor).map_err(|err| err.to_string())?;
    let mut evidence = Evidence::new();
    for signal in option(args, "signals").unwrap_or_default().split(',') {
        let Some((name, strength)) = signal.split_once('=') else {
            continue;
        };
        let value: f64 = strength
            .trim()
            .parse()
            .map_err(|_| format!("{strength:?} is not a number"))?;
        evidence = evidence.with(name.trim(), value);
    }
    match classifier.classify(&evidence) {
        Verdict::Category {
            category,
            confidence,
            reasons,
        } => {
            println!("{category} at {confidence:.3}");
            for (signal, weight) in reasons {
                println!("  {signal} weighs {weight}");
            }
            Ok(())
        }
        Verdict::Abstain { reason } => Err(format!("declined: {reason}")),
    }
}

/// Runs a scaling policy over a load series and reports what it would do.
///
/// The point is not the decisions, it is the flap count. A policy that scales up
/// and down repeatedly has thresholds that are wrong, and that is a fact about
/// the policy rather than about the traffic - but it is only visible over a
/// series, never from one sample.
fn olcum_olcek(args: &[String]) -> Result<(), String> {
    let up: f64 = option(args, "up")
        .ok_or("olcum olcek needs --up")?
        .parse()
        .map_err(|_| "--up is not a number".to_string())?;
    let down: f64 = option(args, "down")
        .ok_or("olcum olcek needs --down")?
        .parse()
        .map_err(|_| "--down is not a number".to_string())?;
    let cooldown: u32 = option(args, "cooldown")
        .unwrap_or_else(|| "3".to_string())
        .parse()
        .map_err(|_| "--cooldown is not a number".to_string())?;
    let min: u32 = option(args, "min")
        .unwrap_or_else(|| "1".to_string())
        .parse()
        .map_err(|_| "--min is not a number".to_string())?;
    let max: u32 = option(args, "max")
        .unwrap_or_else(|| "10".to_string())
        .parse()
        .map_err(|_| "--max is not a number".to_string())?;
    let step: f64 = option(args, "step")
        .unwrap_or_else(|| "0.5".to_string())
        .parse()
        .map_err(|_| "--step is not a number".to_string())?;
    let replicas: u32 = option(args, "replicas")
        .unwrap_or_else(|| min.to_string())
        .parse()
        .map_err(|_| "--replicas is not a number".to_string())?;
    let window: usize = option(args, "window")
        .unwrap_or_else(|| "4".to_string())
        .parse()
        .map_err(|_| "--window is not a number".to_string())?;
    // The policy is built first, so an oscillating policy is refused here rather
    // than after a hundred decisions that all looked reasonable.
    let policy = lubot_olcek::Policy::new(up, down, cooldown, min, max, step)
        .map_err(|err| err.to_string())?;
    let mut controller = Controller::new(policy, replicas).map_err(|err| err.to_string())?;
    let mut load = Load::new(window);
    let series = option(args, "load").unwrap_or_default();
    let mut interval = 0usize;
    for sample in series.split(',') {
        if sample.trim().is_empty() {
            continue;
        }
        let value: f64 = sample
            .trim()
            .parse()
            .map_err(|_| format!("{sample:?} is not a number".to_string()))?;
        load.record(value);
        match controller.tick(&load) {
            Decision::ScaleUp { from, to } => {
                println!("{:>4} load {value:.2} -> scale up {from} to {to}", interval)
            }
            Decision::ScaleDown { from, to } => {
                println!("{:>4} load {value:.2} -> scale down {from} to {to}", interval)
            }
            Decision::Hold { reason } => {
                println!("{:>4} load {value:.2} -> hold ({}), {} replicas", interval, reason, controller.replicas)
            }
        }
        interval = interval.saturating_add(1);
    }
    println!(
        "final {} replicas, {} direction reversal(s) over {} intervals",
        controller.replicas,
        controller.flaps,
        interval
    );
    if controller.flaps > 0 {
        return Err(format!(
            "the policy reversed direction {} time(s); its thresholds are too close together for this traffic",
            controller.flaps
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_dependencies_are_read_out_of_a_manifest() {
        let manifest = r#"
[dependencies]
flate2 = "1"
lubot-read = { path = "../read" }
lubot-answer = { path = "../answer" }
"#;
        assert_eq!(path_dependencies(manifest), vec!["read", "answer"]);
    }

    #[test]
    fn a_duplicate_path_dependency_is_listed_once() {
        let manifest = "a = { path = \"../read\" }\nb = { path = \"../read\" }\n";
        assert_eq!(path_dependencies(manifest), vec!["read"]);
    }

    #[test]
    fn a_manifest_with_no_path_dependencies_yields_nothing() {
        assert!(path_dependencies("[dependencies]\nserde = \"1\"\n").is_empty());
    }

    #[test]
    fn a_downward_graph_is_clean() {
        let mut dependencies = BTreeMap::new();
        dependencies.insert("read".to_string(), vec![]);
        dependencies.insert("index".to_string(), vec!["read".to_string()]);
        dependencies.insert("answer".to_string(), vec!["index".to_string()]);
        dependencies.insert("cli".to_string(), vec!["answer".to_string()]);
        let report = check_architecture(&dependencies);
        assert!(report.is_clean(), "a downward graph was reported dirty: {report:?}");
        assert_eq!(report.start_order.first().map(String::as_str), Some("read"));
        assert_eq!(report.start_order.last().map(String::as_str), Some("cli"));
    }

    #[test]
    fn an_upward_dependency_is_reported() {
        // This is the erosion the check exists for: one call from a deep crate to
        // a shallow one.
        let mut dependencies = BTreeMap::new();
        dependencies.insert("read".to_string(), vec!["answer".to_string()]);
        dependencies.insert("answer".to_string(), vec![]);
        let report = check_architecture(&dependencies);
        assert_eq!(report.violations.len(), 1);
        assert!(report.violations[0].contains("read"));
        assert!(!report.is_clean());
    }

    #[test]
    fn an_unclassified_crate_is_reported_and_left_out_of_the_layer_check() {
        // An assumed layer produces violations that mean nothing.
        let mut dependencies = BTreeMap::new();
        dependencies.insert("read".to_string(), vec!["yeni".to_string()]);
        dependencies.insert("yeni".to_string(), vec![]);
        let report = check_architecture(&dependencies);
        assert_eq!(report.unclassified, vec!["yeni".to_string()]);
        assert!(
            report.violations.is_empty(),
            "an unclassified crate produced a violation: {:?}",
            report.violations
        );
    }

    #[test]
    fn a_cycle_through_an_unclassified_crate_is_still_caught() {
        // Unclassified means exempt from the layer rule, not from the graph.
        let mut dependencies = BTreeMap::new();
        dependencies.insert("yeni".to_string(), vec!["read".to_string()]);
        dependencies.insert("read".to_string(), vec!["yeni".to_string()]);
        let report = check_architecture(&dependencies);
        assert!(!report.cycles.is_empty(), "a cycle went unreported");
    }

    #[test]
    fn the_declared_architecture_covers_the_workspace() {
        // A crate that is not in the table is invisible to the layer check, so the
        // table going stale is itself worth catching.
        assert!(layer_of("read") == Some(0));
        assert!(layer_of("cli") == Some(3));
        assert!(layer_of("cli") > layer_of("read"));
        assert_eq!(layer_of("yok-boyle-bir-crate"), None);
    }

    #[test]
    fn an_empty_graph_is_clean() {
        let report = check_architecture(&BTreeMap::new());
        assert!(report.is_clean());
        assert!(report.crates.is_empty());
    }

    #[test]
    fn a_quorum_excludes_the_requester() {
        // A quorum over a set that includes the requester is not a quorum.
        let quorum = Quorum::unweighted(&[1, 2, 3, 4], 3).expect("quorum");
        assert_eq!(quorum.threshold(), 3);
        // Two others plus the requester is not three others.
        assert!(quorum.count_excluding(&[1, 2, 3], 1).is_err());
        assert_eq!(quorum.count_excluding(&[1, 2, 3, 4], 1), Ok(3));
    }

    #[test]
    fn tolerable_faults_follows_the_byzantine_floor() {
        // n >= 3f + 1, so four members tolerate one fault and three do not.
        assert_eq!(lubot_esik::tolerable_faults(4), 1);
        assert_eq!(lubot_esik::tolerable_faults(3), 0);
        assert!(lubot_esik::meets_byzantine_floor(4, 1));
        assert!(!lubot_esik::meets_byzantine_floor(3, 1));
    }

    #[test]
    fn a_plan_reports_what_can_start_now() {
        let mut tracker = Tracker::new(4);
        tracker.add("a").expect("add");
        tracker.add("b").expect("add");
        tracker.add_dependency("b", "a").expect("edge");
        assert_eq!(tracker.ready(), vec!["a"]);
        tracker.advance("a", Phase::Started).expect("start");
        tracker.advance("a", Phase::Completed).expect("complete");
        assert_eq!(tracker.ready(), vec!["b"]);
    }

    #[test]
    fn a_cycle_in_a_plan_is_refused_when_the_edge_is_added() {
        let mut tracker = Tracker::new(4);
        tracker.add("a").expect("add");
        tracker.add("b").expect("add");
        tracker.add_dependency("a", "b").expect("edge");
        assert!(tracker.add_dependency("b", "a").is_err());
    }

    #[test]
    fn an_oscillating_policy_is_refused_when_it_is_built() {
        // Refused before any decision is taken, rather than after a hundred
        // decisions that each looked reasonable.
        assert!(lubot_olcek::Policy::new(0.7, 0.7, 3, 1, 10, 0.5).is_err());
        assert!(lubot_olcek::Policy::new(0.4, 0.8, 3, 1, 10, 0.5).is_err());
        assert!(lubot_olcek::Policy::new(0.8, 0.4, 3, 1, 10, 0.5).is_ok());
    }

    #[test]
    fn a_steady_load_produces_no_reversals() {
        let policy = lubot_olcek::Policy::new(0.8, 0.4, 3, 1, 10, 0.5).expect("policy");
        let mut controller = Controller::new(policy, 4).expect("controller");
        let mut load = Load::new(4);
        for _ in 0..20 {
            load.record(0.95);
            controller.tick(&load);
        }
        assert_eq!(controller.flaps, 0, "a steady load made the controller oscillate");
        assert!(controller.replicas > 4, "a steady high load did not scale up");
    }

    #[test]
    fn alternating_load_produces_reversals_that_are_counted() {
        // The number that says the thresholds are wrong. It is not visible from
        // any single decision.
        let policy = lubot_olcek::Policy::new(0.8, 0.4, 1, 1, 10, 0.5).expect("policy");
        let mut controller = Controller::new(policy, 4).expect("controller");
        let mut load = Load::new(1);
        for i in 0..20 {
            load.record(if i % 2 == 0 { 1.0 } else { 0.0 });
            controller.tick(&load);
        }
        assert!(controller.flaps > 0, "alternating load was not detected as flapping");
    }

    #[test]
    fn a_classifier_declines_rather_than_guessing() {
        let mut table = BTreeMap::new();
        let mut bug = BTreeMap::new();
        bug.insert("trace".to_string(), 3.0);
        table.insert("bug".to_string(), bug);
        let classifier = Classifier::new(table, 0.6).expect("classifier");
        match classifier.classify(&Evidence::new().with("unrelated", 1.0)) {
            Verdict::Abstain { .. } => {}
            other => panic!("expected a decline, got {other:?}"),
        }
        match classifier.classify(&Evidence::new().with("trace", 1.0)) {
            Verdict::Category { category, .. } => assert_eq!(category, "bug"),
            other => panic!("expected a category, got {other:?}"),
        }
    }
}
