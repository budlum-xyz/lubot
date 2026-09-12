//! Task tracking where progress cannot quietly go backwards.
//!
//! # The two rules that make this more than a list
//!
//! **1. Progress is monotonic.** A task that was completed and is now pending has
//! not been updated, it has lost a fact, and nobody can tell whether that was a
//! deliberate re-open or a bug overwriting state. [`Tracker::advance`] only moves
//! forward along [`Phase`]; moving backwards is refused with
//! [`TrackError::Regression`] and the reason it was attempted is recorded. A
//! deliberate re-open goes through [`Tracker::reopen`], which is spelled
//! differently because it means something different and is auditable as such.
//!
//! **2. A task cannot start before its dependencies complete.** Enforced at
//! advance time, not at planning time, because a dependency can be added after
//! both tasks exist. And the check is not merely a convenience: a task marked
//! complete while a dependency is still pending means one of the two records is
//! wrong, and a tracker that allows it will report a finished plan that is not
//! finished.
//!
//! # Cycles
//!
//! A dependency cycle means nothing in it can ever complete, and that is not
//! visible until the plan stalls. [`Tracker::add_dependency`] refuses an edge
//! that would close a cycle, checking reachability first. The alternative - detect
//! the cycle when the plan stops moving - turns a validation error into a hung
//! worker with no explanation.
//!
//! # In-flight bound
//!
//! The number of tasks that may be started but not finished is capped. An
//! uncapped in-flight count is how a worker ends up holding a thousand half-done
//! tasks and finishing none, which looks like progress from the outside and is
//! not.

use std::collections::{BTreeMap, BTreeSet};

/// Where a task stands. Ordered, and the order is the only direction progress
/// moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// Known, not started.
    Pending,
    /// Started.
    Started,
    /// Finished successfully.
    Completed,
    /// Finished unsuccessfully. Terminal, and distinct from pending: a failed
    /// task was attempted, and treating it as not-yet-attempted would have it
    /// retried by anything that scans for pending work.
    Failed,
}

impl Phase {
    /// Whether this phase is terminal.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    /// A stable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// Why the tracker refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackError {
    /// The task id is already in use.
    DuplicateTask { id: String },
    /// No task with this id.
    UnknownTask { id: String },
    /// The requested move is backwards.
    Regression { id: String, from: Phase, to: Phase },
    /// A dependency has not completed, so this task cannot start or complete.
    DependencyIncomplete { id: String, dependency: String, phase: Phase },
    /// The dependency edge would close a cycle.
    Cycle { from: String, to: String },
    /// A task cannot depend on itself.
    SelfDependency { id: String },
    /// The in-flight bound is reached.
    TooManyInFlight { in_flight: usize, bound: usize },
    /// A dependency names a task that does not exist.
    UnknownDependency { id: String, dependency: String },
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateTask { id } => write!(f, "the task {id:?} already exists"),
            Self::UnknownTask { id } => write!(f, "there is no task {id:?}"),
            Self::Regression { id, from, to } => write!(
                f,
                "task {id:?} is {from} and cannot move back to {to}; a deliberate re-open is a separate, recorded action"
            ),
            Self::DependencyIncomplete {
                id,
                dependency,
                phase,
            } => write!(
                f,
                "task {id:?} cannot advance while its dependency {dependency:?} is still {phase}"
            ),
            Self::Cycle { from, to } => {
                write!(f, "making {to:?} depend on {from:?} would close a cycle, and nothing in it could ever complete")
            }
            Self::SelfDependency { id } => write!(f, "task {id:?} cannot depend on itself"),
            Self::TooManyInFlight { in_flight, bound } => write!(
                f,
                "{in_flight} tasks are in flight and the bound is {bound}; an uncapped count is how a worker finishes nothing"
            ),
            Self::UnknownDependency { id, dependency } => {
                write!(f, "task {id:?} depends on {dependency:?}, which does not exist")
            }
        }
    }
}

/// One task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    pub id: String,
    pub phase: Phase,
    /// What this task waits for.
    pub dependencies: BTreeSet<String>,
    /// Recorded re-opens. Kept because a task that has been re-opened six times
    /// is a different situation from one that has not, and the phase alone does
    /// not show it.
    pub reopens: u64,
    /// Why it was last re-opened, if it was.
    pub last_reopen_reason: String,
}

/// The tracker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tracker {
    tasks: BTreeMap<String, Task>,
    in_flight_bound: usize,
    /// Regressions that were refused. Counted because a caller trying to move
    /// tasks backwards is worth noticing even when every attempt is refused.
    pub refused_regressions: u64,
}

impl Tracker {
    /// A tracker with an in-flight bound.
    #[must_use]
    pub fn new(in_flight_bound: usize) -> Self {
        Self {
            tasks: BTreeMap::new(),
            in_flight_bound,
            refused_regressions: 0,
        }
    }

    /// Adds a task.
    ///
    /// # Errors
    ///
    /// [`TrackError::DuplicateTask`].
    pub fn add(&mut self, id: &str) -> Result<(), TrackError> {
        if self.tasks.contains_key(id) {
            return Err(TrackError::DuplicateTask { id: id.to_string() });
        }
        self.tasks.insert(
            id.to_string(),
            Task {
                id: id.to_string(),
                phase: Phase::Pending,
                dependencies: BTreeSet::new(),
                reopens: 0,
                last_reopen_reason: String::new(),
            },
        );
        Ok(())
    }

    /// Adds a dependency edge, refusing one that would close a cycle.
    ///
    /// # Errors
    ///
    /// [`TrackError::SelfDependency`], [`TrackError::UnknownTask`],
    /// [`TrackError::UnknownDependency`], or [`TrackError::Cycle`].
    pub fn add_dependency(&mut self, id: &str, depends_on: &str) -> Result<(), TrackError> {
        if id == depends_on {
            return Err(TrackError::SelfDependency { id: id.to_string() });
        }
        if !self.tasks.contains_key(id) {
            return Err(TrackError::UnknownTask { id: id.to_string() });
        }
        if !self.tasks.contains_key(depends_on) {
            return Err(TrackError::UnknownDependency {
                id: id.to_string(),
                dependency: depends_on.to_string(),
            });
        }
        // Would `id` become reachable from `depends_on`? If so, the edge closes a
        // cycle.
        if self.reaches(depends_on, id) {
            return Err(TrackError::Cycle {
                from: id.to_string(),
                to: depends_on.to_string(),
            });
        }
        if let Some(task) = self.tasks.get_mut(id) {
            task.dependencies.insert(depends_on.to_string());
        }
        Ok(())
    }

    /// Whether `target` is reachable from `start` by following dependencies.
    fn reaches(&self, start: &str, target: &str) -> bool {
        let mut stack = vec![start.to_string()];
        let mut seen = BTreeSet::new();
        while let Some(current) = stack.pop() {
            if !seen.insert(current.clone()) {
                continue;
            }
            let Some(task) = self.tasks.get(&current) else {
                continue;
            };
            for dependency in &task.dependencies {
                if dependency == target {
                    return true;
                }
                stack.push(dependency.clone());
            }
        }
        false
    }

    /// Moves a task forward.
    ///
    /// Refuses a backwards move, and refuses to start or complete a task whose
    /// dependencies are not complete.
    ///
    /// # Errors
    ///
    /// Any [`TrackError`] that applies.
    pub fn advance(&mut self, id: &str, to: Phase) -> Result<(), TrackError> {
        let Some(task) = self.tasks.get(id) else {
            return Err(TrackError::UnknownTask { id: id.to_string() });
        };
        let from = task.phase;
        if to < from {
            self.refused_regressions = self.refused_regressions.saturating_add(1);
            return Err(TrackError::Regression {
                id: id.to_string(),
                from,
                to,
            });
        }
        if to == from {
            return Ok(());
        }
        if to == Phase::Started && self.in_flight() >= self.in_flight_bound {
            return Err(TrackError::TooManyInFlight {
                in_flight: self.in_flight(),
                bound: self.in_flight_bound,
            });
        }
        // Dependencies must be complete before this task starts or completes.
        // Checked here rather than at planning time, because a dependency can be
        // added after both tasks exist.
        if to != Phase::Pending {
            for dependency in task.dependencies.clone() {
                let phase = self
                    .tasks
                    .get(&dependency)
                    .map_or(Phase::Pending, |t| t.phase);
                if phase != Phase::Completed {
                    return Err(TrackError::DependencyIncomplete {
                        id: id.to_string(),
                        dependency,
                        phase,
                    });
                }
            }
        }
        if let Some(task) = self.tasks.get_mut(id) {
            task.phase = to;
        }
        Ok(())
    }

    /// Re-opens a completed or failed task, with a reason.
    ///
    /// Separate from [`Self::advance`] on purpose: a re-open is not progress and
    /// must not look like it in the record.
    ///
    /// # Errors
    ///
    /// [`TrackError::UnknownTask`], or [`TrackError::Regression`] if the task was
    /// still pending or started, in which case there is nothing to re-open.
    pub fn reopen(&mut self, id: &str, reason: &str) -> Result<(), TrackError> {
        let Some(task) = self.tasks.get(id) else {
            return Err(TrackError::UnknownTask { id: id.to_string() });
        };
        if !task.phase.is_terminal() {
            return Err(TrackError::Regression {
                id: id.to_string(),
                from: task.phase,
                to: Phase::Pending,
            });
        }
        if let Some(task) = self.tasks.get_mut(id) {
            task.phase = Phase::Pending;
            task.reopens = task.reopens.saturating_add(1);
            task.last_reopen_reason = reason.to_string();
        }
        Ok(())
    }

    /// Reads a task.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Task> {
        self.tasks.get(id)
    }

    /// How many tasks are started but not finished.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.tasks
            .values()
            .filter(|t| t.phase == Phase::Started)
            .count()
    }

    /// How many tasks are in each phase. Reported per phase rather than as one
    /// number, because "twelve tasks" does not say how many are done.
    #[must_use]
    pub fn phase_counts(&self) -> [(Phase, usize); 4] {
        let mut counts = [
            (Phase::Pending, 0usize),
            (Phase::Started, 0),
            (Phase::Completed, 0),
            (Phase::Failed, 0),
        ];
        for task in self.tasks.values() {
            if let Some(slot) = counts.iter_mut().find(|(p, _)| *p == task.phase) {
                slot.1 = slot.1.saturating_add(1);
            }
        }
        counts
    }

    /// The tasks that may be started now: pending, with every dependency
    /// complete.
    ///
    /// The answer to "what should the worker do next", computed rather than
    /// queued, because a queue of ready tasks goes stale the moment any task
    /// finishes.
    #[must_use]
    pub fn ready(&self) -> Vec<&str> {
        self.tasks
            .values()
            .filter(|t| {
                t.phase == Phase::Pending
                    && t.dependencies.iter().all(|d| {
                        self.tasks.get(d).is_some_and(|dep| dep.phase == Phase::Completed)
                    })
            })
            .map(|t| t.id.as_str())
            .collect()
    }

    /// How many tasks exist.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the tracker holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> Tracker {
        Tracker::new(2)
    }

    #[test]
    fn a_task_starts_pending() {
        let mut t = tracker();
        t.add("a").expect("add");
        assert_eq!(t.get("a").map(|x| x.phase), Some(Phase::Pending));
    }

    #[test]
    fn progress_only_moves_forward() {
        // A task that was completed and is now pending has lost a fact, and
        // nobody can tell whether that was deliberate or a bug.
        let mut t = tracker();
        t.add("a").expect("add");
        t.advance("a", Phase::Started).expect("start");
        t.advance("a", Phase::Completed).expect("complete");
        assert_eq!(
            t.advance("a", Phase::Pending),
            Err(TrackError::Regression {
                id: "a".to_string(),
                from: Phase::Completed,
                to: Phase::Pending,
            })
        );
        assert_eq!(t.refused_regressions, 1, "a refused regression was not counted");
        assert_eq!(t.get("a").map(|x| x.phase), Some(Phase::Completed));
    }

    #[test]
    fn a_deliberate_reopen_is_recorded_separately() {
        // A re-open is not progress and must not look like it in the record.
        let mut t = tracker();
        t.add("a").expect("add");
        t.advance("a", Phase::Started).expect("start");
        t.advance("a", Phase::Completed).expect("complete");
        t.reopen("a", "the output was wrong").expect("reopen");
        let task = t.get("a").expect("task");
        assert_eq!(task.phase, Phase::Pending);
        assert_eq!(task.reopens, 1);
        assert_eq!(task.last_reopen_reason, "the output was wrong");
    }

    #[test]
    fn reopening_a_task_that_was_never_finished_is_refused() {
        let mut t = tracker();
        t.add("a").expect("add");
        assert!(matches!(
            t.reopen("a", "reason"),
            Err(TrackError::Regression { .. })
        ));
    }

    #[test]
    fn a_task_cannot_start_before_its_dependencies_complete() {
        // A task marked complete while a dependency is pending means one of the
        // two records is wrong.
        let mut t = tracker();
        t.add("a").expect("add");
        t.add("b").expect("add");
        t.add_dependency("b", "a").expect("edge");
        assert_eq!(
            t.advance("b", Phase::Started),
            Err(TrackError::DependencyIncomplete {
                id: "b".to_string(),
                dependency: "a".to_string(),
                phase: Phase::Pending,
            })
        );
        t.advance("a", Phase::Started).expect("start a");
        t.advance("a", Phase::Completed).expect("complete a");
        t.advance("b", Phase::Started).expect("now b can start");
    }

    #[test]
    fn a_failed_dependency_also_blocks() {
        // "Not completed" is the rule, not "not pending". A failed dependency
        // will never complete on its own.
        let mut t = tracker();
        t.add("a").expect("add");
        t.add("b").expect("add");
        t.add_dependency("b", "a").expect("edge");
        t.advance("a", Phase::Started).expect("start");
        t.advance("a", Phase::Failed).expect("fail");
        assert!(matches!(
            t.advance("b", Phase::Started),
            Err(TrackError::DependencyIncomplete { phase: Phase::Failed, .. })
        ));
    }

    #[test]
    fn a_dependency_cycle_is_refused_when_the_edge_is_added() {
        // Nothing in a cycle can ever complete, and that is not visible until the
        // plan stalls. Refusing the edge turns a hung worker into a validation
        // error.
        let mut t = tracker();
        for id in ["a", "b", "c"] {
            t.add(id).expect("add");
        }
        t.add_dependency("b", "a").expect("b waits for a");
        t.add_dependency("c", "b").expect("c waits for b");
        assert_eq!(
            t.add_dependency("a", "c"),
            Err(TrackError::Cycle {
                from: "a".to_string(),
                to: "c".to_string()
            })
        );
    }

    #[test]
    fn a_task_cannot_depend_on_itself() {
        let mut t = tracker();
        t.add("a").expect("add");
        assert_eq!(
            t.add_dependency("a", "a"),
            Err(TrackError::SelfDependency {
                id: "a".to_string()
            })
        );
    }

    #[test]
    fn a_dependency_on_a_task_that_does_not_exist_is_refused() {
        // Otherwise the dependency is unsatisfiable forever and the reason is
        // invisible.
        let mut t = tracker();
        t.add("a").expect("add");
        assert_eq!(
            t.add_dependency("a", "ghost"),
            Err(TrackError::UnknownDependency {
                id: "a".to_string(),
                dependency: "ghost".to_string(),
            })
        );
    }

    #[test]
    fn the_in_flight_bound_is_enforced() {
        // An uncapped in-flight count is how a worker holds a thousand half-done
        // tasks and finishes none, which looks like progress from outside.
        let mut t = Tracker::new(2);
        for id in ["a", "b", "c"] {
            t.add(id).expect("add");
        }
        t.advance("a", Phase::Started).expect("start a");
        t.advance("b", Phase::Started).expect("start b");
        assert!(matches!(
            t.advance("c", Phase::Started),
            Err(TrackError::TooManyInFlight { in_flight: 2, bound: 2 })
        ));
        assert_eq!(t.in_flight(), 2);
        t.advance("a", Phase::Completed).expect("finish a");
        t.advance("c", Phase::Started).expect("now c can start");
    }

    #[test]
    fn ready_reports_what_can_start_now() {
        // Computed rather than queued, because a queue of ready tasks goes stale
        // the moment any task finishes.
        let mut t = tracker();
        t.add("a").expect("add");
        t.add("b").expect("add");
        t.add_dependency("b", "a").expect("edge");
        assert_eq!(t.ready(), vec!["a"]);
        t.advance("a", Phase::Started).expect("start");
        assert!(t.ready().is_empty());
        t.advance("a", Phase::Completed).expect("complete");
        assert_eq!(t.ready(), vec!["b"]);
    }

    #[test]
    fn a_duplicate_task_id_is_refused() {
        let mut t = tracker();
        t.add("a").expect("add");
        assert_eq!(
            t.add("a"),
            Err(TrackError::DuplicateTask {
                id: "a".to_string()
            })
        );
    }

    #[test]
    fn a_failed_task_is_terminal_and_not_retried_as_pending() {
        // Treating a failed task as not-yet-attempted would have it retried by
        // anything that scans for pending work.
        let mut t = tracker();
        t.add("a").expect("add");
        t.advance("a", Phase::Started).expect("start");
        t.advance("a", Phase::Failed).expect("fail");
        assert!(Phase::Failed.is_terminal());
        assert!(t.ready().is_empty(), "a failed task looked like pending work");
        assert_eq!(t.get("a").map(|x| x.phase), Some(Phase::Failed));
    }

    #[test]
    fn phase_counts_report_each_phase_separately() {
        // "Twelve tasks" does not say how many are done.
        let mut t = tracker();
        t.add("a").expect("add");
        t.add("b").expect("add");
        t.advance("a", Phase::Started).expect("start");
        t.advance("a", Phase::Completed).expect("complete");
        let counts: BTreeMap<Phase, usize> = t.phase_counts().into_iter().collect();
        assert_eq!(counts.get(&Phase::Completed), Some(&1));
        assert_eq!(counts.get(&Phase::Pending), Some(&1));
        assert_eq!(counts.get(&Phase::Started), Some(&0));
    }

    #[test]
    fn advancing_to_the_current_phase_is_a_no_op_not_a_regression() {
        let mut t = tracker();
        t.add("a").expect("add");
        t.advance("a", Phase::Started).expect("start");
        assert!(t.advance("a", Phase::Started).is_ok());
        assert_eq!(t.refused_regressions, 0);
    }
}
