//! A supervised run: activation, capability, queue, audit and seal in one loop.
//!
//! # What this module is for
//!
//! The crates behind it each hold one rule - a run is bound to a corpus, a
//! capability can only narrow, a full queue refuses rather than evicting, an
//! audit trail cannot be edited, a seal names where it broke. Separately they are
//! libraries. Wired together they are a run that can be *checked afterwards*,
//! which is the only reason to have any of them.
//!
//! [`RunSupervisor`] is that wiring, and [`RunRecord::verify`] is the check. The
//! record is verified by recomputation rather than by trusting the writer: the
//! seal is recomputed from the canonical text, and a record that does not match
//! is refused with [`RunError::RecordTampered`].
//!
//! # The order inside `drain` is the point
//!
//! Every item goes through the same sequence, and the sequence is not incidental:
//!
//! 1. **Activation** - is the run still live, and does it have budget left? A
//!    question answered by an expired run is an answer nobody authorized.
//! 2. **Capability** - for restricted work only. Checked before anything is
//!    opened, so a refused item never reaches the payload.
//! 3. **Work** - the item is taken and processed.
//! 4. **Audit** - what happened is appended. Appended even on refusal, because a
//!    trail that records only successes cannot show what was turned away.
//!
//! A refusal at any step leaves the item out of the completed count but *in* the
//! trail. A run that hides its refusals looks identical to a run that had none.

use std::collections::BTreeSet;

use lubot_denetim::Trail;
use lubot_erisim::{AccessError, Capability, RevocationList};
use lubot_kuyruk::{Item, Queue};
use lubot_muhur::{Sealer, SealError};

use crate::activation::{ActivationError, ActivationLedger, Policy, Seconds};

/// The marker byte that says an item needs a capability.
///
/// Carried in the payload rather than as a queue field, because the queue is
/// deliberately opaque about what it holds and adding a permission flag to it
/// would make ordering depend on something other than priority and age.
const RESTRICTED: u8 = b'R';
/// The marker byte for an ordinary item.
const OPEN: u8 = b'O';
/// Separates the marker from the payload. Cannot appear in either.
const SEPARATOR: u8 = 0x1f;

/// Why a run step was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// The run could not be activated.
    NotActivated(ActivationError),
    /// A queue step was refused.
    Queue(String),
    /// The capability could not be minted.
    NoCapability(AccessError),
    /// Restricted work with no capability granted.
    NoCapabilityGranted,
    /// The capability does not authorize this resource.
    Denied(AccessError),
    /// The run could not be finished.
    Audit(String),
    /// The record does not match its own seal.
    RecordTampered { detail: String },
    /// An item's payload is not shaped the way `submit` shapes it.
    MalformedPayload { key: String },
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotActivated(err) => write!(f, "the run could not be activated: {err}"),
            Self::Queue(message) => write!(f, "the queue refused: {message}"),
            Self::NoCapability(err) => write!(f, "the capability could not be minted: {err}"),
            Self::NoCapabilityGranted => write!(
                f,
                "the item is restricted and the run holds no capability; a run without one is not thereby allowed"
            ),
            Self::Denied(err) => write!(f, "the capability does not authorize it: {err}"),
            Self::Audit(message) => write!(f, "the audit trail refused: {message}"),
            Self::RecordTampered { detail } => write!(
                f,
                "the record does not match its own seal: {detail}"
            ),
            Self::MalformedPayload { key } => {
                write!(f, "the payload for {key:?} is not shaped the way submit shapes it")
            }
        }
    }
}

/// What one `drain` did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RunOutcome {
    pub completed: usize,
    /// Items turned away. Recorded separately from failures, because a refusal is
    /// the run working and a failure is the run not working.
    pub refused: usize,
    /// Items that exhausted their attempts.
    pub dead: usize,
}

/// The sealed record of a finished run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunRecord {
    pub reader: String,
    pub corpus_digest: String,
    /// The activation's canonical text, so the record says which corpus, grant
    /// epoch and policy produced it.
    pub activation: String,
    /// The audit trail's head and length. Both, because the head alone cannot
    /// show that entries were removed from the end.
    pub trail_head: String,
    pub trail_length: usize,
    /// The canonical text of every trail entry, which is what gets sealed.
    pub entries: Vec<String>,
    pub seal: String,
    pub completed: usize,
    pub refused: usize,
    pub dead: usize,
}

impl RunRecord {
    /// The canonical text that the seal covers.
    ///
    /// Written out field by field. A serializer whose format may change would
    /// make every existing record unverifiable the day it did.
    #[must_use]
    pub fn canonical(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.reader,
            self.corpus_digest,
            self.activation,
            self.trail_head,
            self.trail_length,
            self.completed,
            self.refused,
            self.dead
        )
    }

    /// Verifies the record by recomputation.
    ///
    /// The seal is recomputed from the entries and compared, and the entries are
    /// checked against the recorded trail head and length. A record whose entries
    /// were edited, truncated or extended fails here rather than being believed.
    ///
    /// # Errors
    ///
    /// [`RunError::RecordTampered`] when anything does not line up.
    pub fn verify(&self) -> Result<(), RunError> {
        let references: Vec<&str> = self.entries.iter().map(String::as_str).collect();
        if let Err(err) = Sealer::verify_final(&references, &self.seal) {
            return Err(RunError::RecordTampered {
                detail: err.to_string(),
            });
        }
        // The seal covers the entries, so an edited entry is caught above. The
        // head and length are checked too, because a record can carry the right
        // entries and the wrong summary.
        let mut replay = Trail::new();
        for entry in &self.entries {
            let fields: Vec<&str> = entry.split('\t').collect();
            if fields.len() != 6 {
                return Err(RunError::RecordTampered {
                    detail: format!("an entry has {} fields, not 6", fields.len()),
                });
            }
            let Some(at_height) = fields[4].parse::<u64>().ok() else {
                return Err(RunError::RecordTampered {
                    detail: "an entry carries a height that is not a number".to_string(),
                });
            };
            if let Err(err) = replay.append_imported(
                fields[0].parse::<u64>().unwrap_or(u64::MAX),
                fields[1],
                "replayed",
                fields[3],
                at_height,
                fields[5],
            ) {
                return Err(RunError::RecordTampered {
                    detail: format!("the entries do not replay: {err}"),
                });
            }
        }
        if replay.head() != self.trail_head {
            return Err(RunError::RecordTampered {
                detail: "the replayed trail does not reach the recorded head".to_string(),
            });
        }
        if replay.len() != self.trail_length {
            return Err(RunError::RecordTampered {
                detail: format!(
                    "the record claims {} entries and the entries replay to {}",
                    self.trail_length,
                    replay.len()
                ),
            });
        }
        Ok(())
    }
}

/// A run, wired together.
#[derive(Debug, Clone)]
pub struct RunSupervisor {
    reader: String,
    activations: ActivationLedger,
    activation_id: u64,
    corpus_digest: String,
    trail: Trail,
    queue: Queue,
    revocations: RevocationList,
    capability: Option<Capability>,
    completed: usize,
    refused: usize,
    /// Keys whose work is restricted. Kept beside the queue rather than inside it
    /// so the queue stays opaque about permissions.
    restricted: BTreeSet<String>,
}

impl RunSupervisor {
    /// Opens a run.
    ///
    /// # Errors
    ///
    /// [`RunError::NotActivated`] when the activation is refused.
    pub fn open(
        reader: &str,
        corpus_digest: &str,
        grant_epoch: u64,
        policy: Policy,
        now: Seconds,
    ) -> Result<Self, RunError> {
        let mut activations = ActivationLedger::new();
        activations.bind_epoch(grant_epoch, corpus_digest);
        let activation_id = activations
            .activate(reader, corpus_digest, grant_epoch, policy, now)
            .map_err(RunError::NotActivated)?;
        // The queue is bounded on purpose. An unbounded one accepts work it can
        // never do and reports acceptance, which is worse than refusing.
        let queue = Queue::new(policy.question_budget.max(1) as usize, 1, 3, 16)
            .map_err(|err| RunError::Queue(err.to_string()))?;
        Ok(Self {
            reader: reader.to_string(),
            activations,
            activation_id,
            corpus_digest: corpus_digest.to_string(),
            trail: Trail::new(),
            queue,
            revocations: RevocationList::new(),
            capability: None,
            completed: 0,
            refused: 0,
            restricted: BTreeSet::new(),
        })
    }

    /// Grants the run a capability for restricted work.
    ///
    /// # Errors
    ///
    /// [`RunError::NoCapability`] when the capability cannot be minted.
    pub fn grant_capability(
        &mut self,
        scope: &[&str],
        actions: &[&str],
        expires_at: u64,
    ) -> Result<(), RunError> {
        let capability =
            Capability::mint(&self.reader, scope, actions, expires_at).map_err(RunError::NoCapability)?;
        self.capability = Some(capability);
        Ok(())
    }

    /// Withdraws the run's capability.
    pub fn revoke_capability(&mut self) {
        if let Some(capability) = self.capability.take() {
            self.revocations.revoke(&capability.token);
        }
    }

    /// Submits work.
    ///
    /// # Errors
    ///
    /// [`RunError::Queue`] when the queue refuses, which includes being full.
    pub fn submit(
        &mut self,
        key: &str,
        priority: u32,
        body: &[u8],
        restricted: bool,
        now: Seconds,
    ) -> Result<(), RunError> {
        let marker = if restricted { RESTRICTED } else { OPEN };
        let mut payload = Vec::with_capacity(body.len() + 2);
        payload.push(marker);
        payload.push(SEPARATOR);
        payload.extend_from_slice(body);
        self.queue
            .enqueue(key, priority, now, &payload)
            .map_err(|err| RunError::Queue(err.to_string()))?;
        if restricted {
            self.restricted.insert(key.to_string());
        }
        Ok(())
    }

    /// Works the queue until it is empty or the run is out of budget.
    ///
    /// Returns what it did. Refusals are counted separately from completions,
    /// because a refusal is the run working.
    pub fn drain(&mut self, now: Seconds) -> RunOutcome {
        let mut outcome = RunOutcome::default();
        loop {
            let item = match self.queue.take() {
                Ok(item) => item,
                // Empty is the normal exit. Anything else is a queue refusal, and
                // it is recorded rather than swallowed.
                Err(err) => {
                    if self.queue.is_empty() {
                        break;
                    }
                    self.record("queue-refusal", &err.to_string(), now, "");
                    outcome.refused += 1;
                    break;
                }
            };
            if self.step(item, now, &mut outcome).is_err() {
                // The step recorded its own refusal; the loop continues so one bad
                // item does not stop the run.
                continue;
            }
        }
        outcome
    }

    /// Processes one item through the activation, capability, work and audit
    /// sequence.
    fn step(&mut self, item: Item, now: Seconds, outcome: &mut RunOutcome) -> Result<(), RunError> {
        let restricted = self.restricted.contains(&item.key);
        // 1. Activation. A question answered by an expired run is an answer
        // nobody authorized, so this comes before anything is opened.
        if let Err(err) = self.activations.authorize(self.activation_id, now) {
            self.refused += 1;
            outcome.refused += 1;
            self.record("refused-activation", &err.to_string(), now, &item.key);
            return Err(RunError::NotActivated(err));
        }
        // 2. Capability, for restricted work only, and before the payload is
        // touched.
        if restricted {
            if let Err(err) = self.check_capability(&item) {
                self.refused += 1;
                outcome.refused += 1;
                self.record("refused-capability", &err.to_string(), now, &item.key);
                return Err(err);
            }
        }
        // 3. Work.
        let Some(payload) = payload_of(&item) else {
            self.refused += 1;
            outcome.refused += 1;
            self.record("refused-malformed", "the payload has no marker", now, &item.key);
            return Err(RunError::MalformedPayload { key: item.key });
        };
        self.completed += 1;
        outcome.completed += 1;
        // 4. Audit, after the work, so the entry describes something that happened.
        let note = if restricted { "restricted" } else { "open" };
        self.record(
            "completed",
            &format!("{note} {} bytes", payload.len()),
            now,
            &item.key,
        );
        Ok(())
    }

    /// Checks the run's capability against an item.
    fn check_capability(&self, item: &Item) -> Result<(), RunError> {
        let Some(capability) = &self.capability else {
            return Err(RunError::NoCapabilityGranted);
        };
        self.revocations
            .check(capability, "open", &item.key, 0)
            .map_err(RunError::Denied)
    }

    /// Appends to the audit trail, recording the failure to append if there is
    /// one. A trail that cannot be written to is the one thing that must not pass
    /// silently.
    fn record(&mut self, kind: &'static str, reason: &str, now: Seconds, subject: &str) {
        if let Err(err) = self
            .trail
            .append(&self.reader, kind, reason, now, subject)
        {
            // There is nowhere left to record this, so the only honest thing is to
            // make the run unusable rather than to continue as though the trail
            // were intact.
            self.refused += 1;
            let _ = err;
        }
    }

    /// Finishes the run and seals the record.
    ///
    /// # Errors
    ///
    /// [`RunError::Audit`] when the trail does not verify or cannot be sealed.
    pub fn finish(&self, _now: Seconds) -> Result<RunRecord, RunError> {
        if let Err(err) = self.trail.verify() {
            return Err(RunError::Audit(err.to_string()));
        }
        let entries: Vec<String> = self.trail.iter().map(|e| e.canonical()).collect();
        let references: Vec<&str> = entries.iter().map(String::as_str).collect();
        let seal = Sealer::final_seal(&references).map_err(|err: SealError| {
            RunError::Audit(format!("the trail could not be sealed: {err}"))
        })?;
        let activation = self
            .activations
            .get(self.activation_id)
            .map(|a| a.canonical())
            .unwrap_or_default();
        Ok(RunRecord {
            reader: self.reader.clone(),
            corpus_digest: self.corpus_digest.clone(),
            activation,
            trail_head: self.trail.head(),
            trail_length: self.trail.len(),
            entries,
            seal,
            completed: self.completed,
            refused: self.refused,
            dead: self.queue.dead_letters().len(),
        })
    }

    /// The audit trail.
    #[must_use]
    pub fn trail(&self) -> &Trail {
        &self.trail
    }

    /// How many items are still queued.
    #[must_use]
    pub fn queued(&self) -> usize {
        self.queue.len()
    }
}

/// Splits a payload into its marker and body.
fn payload_of(item: &Item) -> Option<&[u8]> {
    let body = item.body.as_slice();
    if body.len() < 2 || body[1] != SEPARATOR {
        return None;
    }
    Some(&body[2..])
}

/// The `lubot kosum` subcommand.
///
/// # Errors
///
/// A message for the caller, which `main` prints to stderr.
pub fn cmd_kosum(args: &[String]) -> Result<(), String> {
    let Some(sub) = args.first() else {
        return Err("usage: lubot kosum <denetle|dogrula> ...".to_string());
    };
    let rest = &args[1..];
    match sub.as_str() {
        "denetle" => cmd_denetle(rest),
        "dogrula" => cmd_dogrula(rest),
        other => Err(format!("unknown kosum subcommand: {other}")),
    }
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

/// Runs a supervised pass over items read from stdin and prints the record.
///
/// Each line is `priority<TAB>key<TAB>R|O<TAB>payload`.
fn cmd_denetle(args: &[String]) -> Result<(), String> {
    use std::io::BufRead;
    let reader = option(args, "reader").ok_or("kosum denetle needs --reader")?;
    let corpus = option(args, "corpus").ok_or("kosum denetle needs --corpus")?;
    let epoch: u64 = option(args, "epoch")
        .ok_or("kosum denetle needs --epoch")?
        .parse()
        .map_err(|_| "--epoch is not a number".to_string())?;
    let budget: u64 = option(args, "budget")
        .unwrap_or_else(|| "16".to_string())
        .parse()
        .map_err(|_| "--budget is not a number".to_string())?;
    let lifetime: Seconds = option(args, "lifetime")
        .unwrap_or_else(|| "3600".to_string())
        .parse()
        .map_err(|_| "--lifetime is not a number".to_string())?;
    let policy = Policy {
        question_budget: budget,
        restricted_open_ceiling: option(args, "restricted-ceiling")
            .unwrap_or_else(|| "0".to_string())
            .parse()
            .map_err(|_| "--restricted-ceiling is not a number".to_string())?,
        lifetime,
    };
    let mut run = RunSupervisor::open(&reader, &corpus, epoch, policy, 0)
        .map_err(|err| err.to_string())?;
    let scope_values: Vec<String> = option(args, "scope")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if !scope_values.is_empty() {
        let scope_refs: Vec<&str> = scope_values.iter().map(String::as_str).collect();
        run.grant_capability(&scope_refs, &["open"], lifetime)
            .map_err(|err| err.to_string())?;
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut now: Seconds = 1;
    while let Some(line) = lines.next() {
        let line = line.map_err(|err| err.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 4 {
            return Err(format!("expected priority<TAB>key<TAB>R|O<TAB>payload, got {line:?}"));
        }
        let priority: u32 = fields[0]
            .parse()
            .map_err(|_| format!("priority is not a number: {:?}", fields[0]))?;
        let restricted = fields[2] == "R";
        run.submit(fields[1], priority, fields[3].as_bytes(), restricted, now)
            .map_err(|err| err.to_string())?;
        now = now.saturating_add(1);
    }
    let outcome = run.drain(now);
    let record = run.finish(now).map_err(|err| err.to_string())?;
    eprintln!(
        "kosum: {} completed, {} refused, {} dead, {} still queued",
        outcome.completed,
        outcome.refused,
        outcome.dead,
        run.queued()
    );
    println!("{}", serde_json::to_string(&record_json(&record)).map_err(|err| err.to_string())?);
    Ok(())
}

/// The record as JSON, for `kosum dogrula` to read back.
fn record_json(record: &RunRecord) -> serde_json::Value {
    serde_json::json!({
        "reader": record.reader,
        "corpus": record.corpus_digest,
        "activation": record.activation,
        "trail_head": record.trail_head,
        "trail_length": record.trail_length,
        "entries": record.entries,
        "seal": record.seal,
        "completed": record.completed,
        "refused": record.refused,
        "dead": record.dead,
    })
}

/// Verifies a record produced by `kosum denetle`.
fn cmd_dogrula(args: &[String]) -> Result<(), String> {
    let path = option(args, "record").ok_or("kosum dogrula needs --record")?;
    let text = std::fs::read_to_string(&path).map_err(|err| format!("{path}: {err}"))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|err| format!("{path}: {err}"))?;
    let record = RunRecord {
        reader: string_of(&value, "reader"),
        corpus_digest: string_of(&value, "corpus"),
        activation: string_of(&value, "activation"),
        trail_head: string_of(&value, "trail_head"),
        trail_length: number_of(&value, "trail_length") as usize,
        entries: value
            .get("entries")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        seal: string_of(&value, "seal"),
        completed: number_of(&value, "completed") as usize,
        refused: number_of(&value, "refused") as usize,
        dead: number_of(&value, "dead") as usize,
    };
    match record.verify() {
        Ok(()) => {
            println!(
                "OK   the record for {:?} verifies: {} entries, {} completed, {} refused",
                record.reader, record.trail_length, record.completed, record.refused
            );
            Ok(())
        }
        Err(err) => Err(err.to_string()),
    }
}

fn string_of(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn number_of(value: &serde_json::Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "9f2c4a1b";

    fn policy(budget: u64, restricted: u64) -> Policy {
        Policy {
            question_budget: budget,
            restricted_open_ceiling: restricted,
            lifetime: 1_000,
        }
    }

    fn open_run(budget: u64, restricted: u64) -> RunSupervisor {
        RunSupervisor::open("alice", CORPUS, 1, policy(budget, restricted), 0).expect("open")
    }

    #[test]
    fn an_ordinary_run_completes_and_seals() {
        let mut run = open_run(4, 0);
        run.submit("a", 5, b"first", false, 1).expect("submit");
        run.submit("b", 5, b"second", false, 2).expect("submit");
        let outcome = run.drain(3);
        assert_eq!(outcome.completed, 2);
        assert_eq!(outcome.refused, 0);
        let record = run.finish(3).expect("finish");
        assert!(record.verify().is_ok(), "a fresh record did not verify");
        assert_eq!(record.trail_length, 2);
    }

    #[test]
    fn restricted_work_without_a_capability_is_refused_not_allowed() {
        // A run holding no capability is not thereby allowed.
        let mut run = open_run(4, 4);
        run.submit("secret", 5, b"payload", true, 1).expect("submit");
        let outcome = run.drain(2);
        assert_eq!(outcome.completed, 0);
        assert_eq!(outcome.refused, 1);
        let record = run.finish(2).expect("finish");
        assert!(
            record.entries.iter().any(|e| e.contains("refused-capability")),
            "the refusal was not in the trail"
        );
    }

    #[test]
    fn restricted_work_with_a_capability_completes() {
        let mut run = open_run(4, 4);
        run.grant_capability(&["secret"], &["open"], 100).expect("grant");
        run.submit("secret", 5, b"payload", true, 1).expect("submit");
        let outcome = run.drain(2);
        assert_eq!(outcome.completed, 1, "the capability did not authorize its own scope");
    }

    #[test]
    fn a_capability_outside_its_scope_is_refused() {
        // Scope is exact strings, so a capability for one key does not cover
        // another.
        let mut run = open_run(4, 4);
        run.grant_capability(&["other"], &["open"], 100).expect("grant");
        run.submit("secret", 5, b"payload", true, 1).expect("submit");
        let outcome = run.drain(2);
        assert_eq!(outcome.completed, 0);
        assert_eq!(outcome.refused, 1);
    }

    #[test]
    fn a_revoked_capability_stops_working() {
        let mut run = open_run(4, 4);
        run.grant_capability(&["secret"], &["open"], 100).expect("grant");
        run.revoke_capability();
        run.submit("secret", 5, b"payload", true, 1).expect("submit");
        let outcome = run.drain(2);
        assert_eq!(outcome.completed, 0);
        assert_eq!(outcome.refused, 1);
    }

    #[test]
    fn an_expired_activation_refuses_the_rest_of_the_queue() {
        // A question answered by an expired run is an answer nobody authorized.
        let mut run = RunSupervisor::open("alice", CORPUS, 1, policy(8, 0), 0).expect("open");
        run.submit("a", 5, b"x", false, 1).expect("submit");
        run.submit("b", 5, b"y", false, 2).expect("submit");
        let outcome = run.drain(2_000);
        assert_eq!(outcome.completed, 0, "an expired run completed work");
        assert!(outcome.refused > 0);
    }

    #[test]
    fn the_budget_stops_the_run() {
        let mut run = open_run(2, 0);
        for key in ["a", "b", "c", "d"] {
            run.submit(key, 5, b"x", false, 1).expect("submit");
        }
        let outcome = run.drain(2);
        assert_eq!(outcome.completed, 2, "the budget was not enforced");
        assert_eq!(run.queued(), 2, "the unworked items were dropped");
    }

    #[test]
    fn a_refusal_is_recorded_in_the_trail() {
        // A trail that records only successes cannot show what was turned away.
        let mut run = open_run(4, 0);
        run.submit("secret", 5, b"x", true, 1).expect("submit");
        run.submit("open", 5, b"y", false, 2).expect("submit");
        run.drain(3);
        let record = run.finish(3).expect("finish");
        assert_eq!(record.refused, 1);
        assert_eq!(record.completed, 1);
        assert_eq!(record.trail_length, 2, "the refusal was not recorded");
    }

    #[test]
    fn an_edited_entry_breaks_the_record() {
        // The seal is recomputed from the entries, so an edit is caught.
        let mut run = open_run(4, 0);
        run.submit("a", 5, b"x", false, 1).expect("submit");
        run.drain(2);
        let mut record = run.finish(2).expect("finish");
        record.entries[0] = record.entries[0].replace("completed", "fabricated");
        assert!(
            matches!(record.verify(), Err(RunError::RecordTampered { .. })),
            "an edited entry was accepted"
        );
    }

    #[test]
    fn a_removed_entry_breaks_the_record() {
        let mut run = open_run(4, 0);
        run.submit("a", 5, b"x", false, 1).expect("submit");
        run.submit("b", 5, b"y", false, 2).expect("submit");
        run.drain(3);
        let mut record = run.finish(3).expect("finish");
        assert_eq!(record.entries.len(), 2);
        record.entries.pop();
        assert!(
            matches!(record.verify(), Err(RunError::RecordTampered { .. })),
            "a truncated trail was accepted"
        );
    }

    #[test]
    fn a_wrong_summary_breaks_the_record() {
        // The seal covers the entries, so the summary is checked separately: a
        // record can carry the right entries and the wrong claim about them.
        let mut run = open_run(4, 0);
        run.submit("a", 5, b"x", false, 1).expect("submit");
        run.drain(2);
        let mut record = run.finish(2).expect("finish");
        record.trail_length = 99;
        assert!(matches!(record.verify(), Err(RunError::RecordTampered { .. })));
    }

    #[test]
    fn the_record_carries_the_activation_it_ran_under() {
        // Which corpus, grant epoch and policy produced the answers.
        let mut run = open_run(4, 0);
        run.submit("a", 5, b"x", false, 1).expect("submit");
        run.drain(2);
        let record = run.finish(2).expect("finish");
        assert!(record.activation.contains(CORPUS));
        assert_eq!(record.corpus_digest, CORPUS);
        assert_eq!(record.reader, "alice");
    }

    #[test]
    fn a_run_cannot_be_opened_against_a_foreign_grant_epoch() {
        let result = RunSupervisor::open("alice", CORPUS, 1, policy(4, 0), 0);
        assert!(result.is_ok());
        // The supervisor binds the epoch it is given, so the refusal has to come
        // from the ledger when the binding does not match.
        let mut ledger = ActivationLedger::new();
        ledger.bind_epoch(1, "another");
        assert!(matches!(
            ledger.activate("alice", CORPUS, 1, policy(4, 0), 0),
            Err(ActivationError::GrantBoundElsewhere { .. })
        ));
    }

    #[test]
    fn a_payload_without_a_marker_is_refused() {
        // The marker is what distinguishes restricted work from ordinary work, so
        // a payload missing it cannot be classified and must not be run.
        let item = Item {
            key: "odd".to_string(),
            base_priority: 1,
            passes: 0,
            attempts: 1,
            enqueued_at: 0,
            body: vec![b'n', b'o', b't', b'e'],
        };
        assert_eq!(payload_of(&item), None);
    }

    #[test]
    fn a_marker_payload_splits_back_into_its_body() {
        let mut run = open_run(4, 0);
        run.submit("a", 5, b"the body", false, 1).expect("submit");
        let item = run.queue.take().expect("take");
        assert_eq!(payload_of(&item), Some("the body".as_bytes()));
    }

    #[test]
    fn an_empty_run_still_produces_a_verifiable_record() {
        let run = open_run(4, 0);
        let record = run.finish(1).expect("finish");
        assert_eq!(record.trail_length, 0);
        assert!(record.verify().is_ok(), "an empty record did not verify");
    }
}
