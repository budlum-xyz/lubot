//! # queue - uninterrupted work, bounded and resumable
//!
//! The autonomy power, in the shape this repository already enforces: work
//! continues over a queue until the queue is empty or a budget is spent;
//! every finished job is persisted before the next one starts, so an
//! interruption resumes where it stopped instead of redoing; and after every
//! job the configured check must pass, or the run halts with the remaining
//! jobs still pending. Nothing here loops forever: a run has a per-job
//! attempt cap, and a job that keeps crashing becomes `stalled` - it is never
//! retried silently.
//!
//! One rule deserves its name: an individual job answering `Out of scope` or
//! `Refused` is a **verdict**, not a failure. The queue moves on. Only a
//! structural break (the pipeline erroring, or the check failing) halts the
//! run, and halting is always loud.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A job crashes this many times before it is parked as stalled.
pub const MAX_ATTEMPTS: usize = 3;

/// What a job is.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Job {
    pub job_id: String,
    pub created_at: u64,
    pub question: String,
    #[serde(default)]
    pub corpus: Vec<String>,
    pub reader: String,
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub state: JobState,
    /// How many times the run has tried this job.
    #[serde(default)]
    pub attempts: u64,
    /// The verdict label of the last finished attempt (Grounded, Computed,
    /// OutOfScope, Refused, NotFound, or the crash message).
    #[serde(default)]
    pub verdict: Option<String>,
    #[serde(default)]
    pub citations: usize,
    /// Append-only trace of this job's history, oldest first.
    #[serde(default)]
    pub journal: Vec<String>,
}

/// The one place a job's life is decided.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    #[default]
    Pending,
    Done,
    Failed,
    Stalled,
    /// Cancelled by the operator before it ran; a cancelled job never runs.
    Cancelled,
}

impl Job {
    /// A job waiting for its first run.
    #[must_use]
    pub fn pending(
        job_id: &str,
        created_at: u64,
        question: &str,
        corpus: Vec<String>,
        reader: &str,
        effort: Option<String>,
    ) -> Self {
        Self {
            job_id: job_id.to_string(),
            created_at,
            question: question.to_string(),
            corpus,
            reader: reader.to_string(),
            effort,
            state: JobState::Pending,
            attempts: 0,
            verdict: None,
            citations: 0,
            journal: Vec::new(),
        }
    }

    /// Record a finished attempt (a verdict, crash message, or the halt
    /// reason) and advance the state machine.
    pub fn finish(&mut self, verdict: &str, citations: usize, reason: &str) {
        self.attempts += 1;
        self.verdict = Some(verdict.to_string());
        self.citations = citations;
        let entry = format!("#{} {}: {}", self.attempts, verdict, reason);
        self.journal.push(entry);
        if verdict == "crash" {
            self.state = if self.attempts >= MAX_ATTEMPTS as u64 {
                JobState::Stalled
            } else {
                JobState::Failed
            };
        } else {
            self.state = JobState::Done;
        }
    }

    /// Whether this job still needs work. A `Done` job is never re-run, a
    /// `Stalled` one is parked, and a `Failed` one is retried on its next run.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        matches!(self.state, JobState::Pending | JobState::Failed)
    }
}

/// Cancel a job that has not finished. Only `Pending` and `Failed` jobs can
/// be cancelled; a job that already reached a terminal state is refused, and
/// a job id that does not exist is refused, so the operator's command never
/// silently misses its target.
pub fn cancel(jobs: &mut [Job], job_id: &str) -> Result<String, String> {
    let Some(job) = jobs.iter_mut().find(|j| j.job_id == job_id) else {
        return Err(format!("queue: job `{job_id}` not found"));
    };
    if !matches!(job.state, JobState::Pending | JobState::Failed) {
        return Err(format!(
            "queue: job `{job_id}` is {:?} and can no longer be cancelled",
            job.state
        ));
    }
    job.state = JobState::Cancelled;
    job.journal.push(format!(
        "cancelled by operator at {}",
        crate::now_seconds()?
    ));
    Ok(job_id.to_string())
}

/// The operator's view of a queue: one line per job, oldest first, with the
/// state word, the effort ceiling and the attempt count.
pub fn list_md(jobs: &[Job]) -> String {
    let mut md = String::from("# Kuyruk\n\n");
    if jobs.is_empty() {
        md.push_str("_empty_\n");
        return md;
    }
    for job in jobs {
        let state = match job.state {
            JobState::Pending => "pending",
            JobState::Done => "done",
            JobState::Failed => "failed",
            JobState::Stalled => "stalled",
            JobState::Cancelled => "cancelled",
        };
        let effort = job.effort.as_deref().unwrap_or("-");
        md.push_str(&format!(
            "- `{}` {} effort={} attempts={} {:?}\n    {}\n",
            job.job_id,
            state,
            effort,
            job.attempts,
            job.verdict.as_deref().unwrap_or(""),
            job.question
        ));
    }
    md
}

/// Load a queue file; missing file means an empty queue.
///
/// # Errors
/// Parse failures are named and refuse the whole queue - a queue that cannot
/// be read must never be silently restarted empty.
pub fn load_queue(path: &Path) -> Result<Vec<Job>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut jobs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let job: Job = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), index + 1))?;
        jobs.push(job);
    }
    Ok(jobs)
}

/// Persist the queue atomically (tmp + rename), so an interruption cannot
/// leave a half-written job record.
///
/// # Errors
/// File creation, serialization or rename failures.
pub fn save_queue(path: &Path, jobs: &[Job]) -> Result<(), String> {
    let tmp = path.with_extension("jsonl.tmp");
    {
        use std::io::Write;
        let mut file =
            std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        for job in jobs {
            let line = serde_json::to_string(job).map_err(|e| e.to_string())?;
            writeln!(file, "{line}").map_err(|e| e.to_string())?;
        }
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// What processing one job produces: the verdict label and its citation
/// count, or the crash reason.
pub type JobResult = Result<(String, usize), String>;

/// The per-job worker and the post-job check, named so the queue's contract
/// is readable at the call site.
pub type CheckFn<'a> = dyn FnMut() -> Result<(), String> + 'a;

/// The outcome of one run over the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunReport {
    pub done: usize,
    pub crashed: usize,
    pub stalled: usize,
    /// True when a check failure stopped the run before the queue was empty.
    pub halted: bool,
    pub remaining: usize,
}

/// Drive the queue. For each pending job, `process` is called (it returns the
/// verdict label and the citation count, or an error for a crash); then the
/// mutable `check` must pass - the job is not allowed to be followed by an
/// unverified state. When `check` fails, the run halts and the remaining jobs
/// are left pending.
///
/// The `budget` is the number of jobs this run may process; a run always
/// stops at the budget even when the queue still has work, so an
/// uninterrupted worker is also a resumable one.
pub fn run_queue(
    jobs: &mut [Job],
    budget: usize,
    process: &mut dyn FnMut(&Job) -> JobResult,
    check: &mut CheckFn<'_>,
) -> RunReport {
    let mut report = RunReport {
        done: 0,
        crashed: 0,
        stalled: 0,
        halted: false,
        remaining: 0,
    };
    let mut touched = 0usize;
    for job in jobs.iter_mut() {
        if !job.is_pending() {
            continue;
        }
        if touched >= budget {
            break;
        }
        touched += 1;
        match process(job) {
            Ok((verdict, citations)) => {
                job.finish(&verdict, citations, "ok");
                report.done += 1;
            }
            Err(why) => {
                let was_stalled = job.attempts + 1 >= MAX_ATTEMPTS as u64;
                job.finish("crash", 0, &why);
                if was_stalled {
                    report.stalled += 1;
                } else {
                    report.crashed += 1;
                }
            }
        }
        if let Err(why) = check() {
            job.journal
                .push(format!("halt: check failed after this job: {why}"));
            report.halted = true;
            break;
        }
    }
    report.remaining = jobs.iter().filter(|j| j.is_pending()).count();
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str) -> Job {
        Job::pending(id, 1, "q", vec!["c.jsonl".to_string()], "r", None)
    }

    #[test]
    fn a_job_lives_through_the_state_machine() {
        let mut j = job("a");
        assert!(j.is_pending());
        j.finish("Grounded", 3, "ok");
        assert_eq!(j.state, JobState::Done);
        assert_eq!(j.attempts, 1);
        assert_eq!(j.verdict.as_deref(), Some("Grounded"));
        assert!(!j.is_pending());
        assert_eq!(j.journal.len(), 1);
    }

    #[test]
    fn a_crashed_job_is_retried_then_stalled() {
        let mut j = job("a");
        j.finish("crash", 0, "io");
        assert_eq!(j.state, JobState::Failed);
        assert!(j.is_pending());
        j.finish("crash", 0, "io");
        assert_eq!(j.state, JobState::Failed);
        j.finish("crash", 0, "io");
        assert_eq!(j.state, JobState::Stalled);
        assert!(!j.is_pending());
    }

    #[test]
    fn a_refusal_is_a_verdict_not_a_crash() {
        let mut j = job("a");
        j.finish("OutOfScope", 0, "ok");
        assert_eq!(j.state, JobState::Done);
        assert_eq!(j.attempts, 1);
    }

    #[test]
    fn the_budget_stops_a_run_and_the_rest_stays_pending() {
        let mut jobs = vec![job("a"), job("b"), job("c")];
        let mut check_ok = || Ok(());
        let report = run_queue(
            &mut jobs,
            2,
            &mut |_| Ok(("Grounded".to_string(), 1)),
            &mut check_ok,
        );
        assert_eq!(report.done, 2);
        assert_eq!(report.remaining, 1);
        assert!(!report.halted);
        assert_eq!(jobs[0].state, JobState::Done);
        assert_eq!(jobs[2].state, JobState::Pending);
    }

    #[test]
    fn done_jobs_are_never_redone_on_resume() {
        let mut jobs = vec![job("a"), job("b")];
        let mut check_ok = || Ok(());
        let first = run_queue(
            &mut jobs,
            1,
            &mut |_| Ok(("Grounded".to_string(), 1)),
            &mut check_ok,
        );
        assert_eq!(first.done, 1);
        assert_eq!(jobs[0].attempts, 1);
        let second = run_queue(
            &mut jobs,
            10,
            &mut |_| Ok(("Grounded".to_string(), 1)),
            &mut check_ok,
        );
        assert_eq!(second.done, 1);
        assert_eq!(jobs[0].attempts, 1, "resume redid a done job");
        assert_eq!(jobs[1].attempts, 1);
    }

    #[test]
    fn a_failing_check_halts_loudly_and_leaves_the_rest_pending() {
        let mut jobs = vec![job("a"), job("b")];
        let mut check_fails = || Err("gates red".to_string());
        let report = run_queue(
            &mut jobs,
            10,
            &mut |_| Ok(("Grounded".to_string(), 1)),
            &mut check_fails,
        );
        assert!(report.halted);
        assert_eq!(report.done, 1);
        assert_eq!(jobs[1].state, JobState::Pending);
        assert!(jobs[0].journal.iter().any(|e| e.contains("halt")));
    }

    #[test]
    fn queue_round_trips_through_the_file() {
        let dir = std::env::temp_dir().join(format!("lubot-queue-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q.jsonl");
        let jobs = vec![job("a"), job("b")];
        save_queue(&path, &jobs).unwrap();
        let loaded = load_queue(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].job_id, "a");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_operator_can_cancel_only_an_unfinished_job() {
        let mut jobs = vec![job("a"), job("b"), job("c")];
        jobs[1].finish("Grounded", 2, "ok");
        assert_eq!(cancel(&mut jobs, "a").unwrap(), "a");
        assert!(matches!(jobs[0].state, JobState::Cancelled));
        assert!(
            !jobs[0].is_pending(),
            "a cancelled job must never be re-run"
        );
        let err = cancel(&mut jobs, "b").unwrap_err();
        assert!(err.contains("can no longer be cancelled"), "{err}");
        let err = cancel(&mut jobs, "nope").unwrap_err();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn list_md_shows_states_effort_and_questions() {
        let mut jobs = vec![job("a"), job("b")];
        jobs[0].effort = Some("1.0x".to_string());
        jobs[1].finish("OutOfScope", 0, "ok");
        let md = list_md(&jobs);
        assert!(md.contains("`a`"), "{md}");
        assert!(md.contains("pending"), "{md}");
        assert!(md.contains("done"), "{md}");
        assert!(md.contains("1.0x"), "{md}");
        assert!(md.contains("q"), "{md}");
    }
}
