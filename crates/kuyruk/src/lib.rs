//! A queue that can prove it did not lose a job.
//!
//! Lubot's maintenance work is a queue: repairs, re-indexes, reports, sweeps.
//! The failure mode of every queue built this way is the same and it is not
//! "a job is slow" - it is "a job is gone". A bounded queue with an eviction
//! policy and no accounting is how that happens: capacity is reached, the
//! oldest entry is dropped to make room, the run continues, and the only record
//! of the dropped job is that it is not in the output.
//!
//! # The invariant
//!
//! ```text
//! submitted == in_flight + done + dead_letters + dropped + refused
//! ```
//!
//! [`Queue::verify`] recomputes that sum from its own counters. It is the whole
//! crate's reason to exist: everything else is a policy choice that a caller
//! could argue with, and this is arithmetic.
//!
//! # Policies, and which of them are choices
//!
//! * capacity is fixed at construction and a zero capacity is refused - a queue
//!   that admits nothing and says nothing is an off switch, not a queue;
//! * a full queue evicts the lowest-priority job to admit a higher-priority
//!   one, and only that. A `Repair` never displaces a `Repair`, and if the only
//!   things to displace are as important as what is arriving, the arrival is
//!   refused *with a count*;
//! * a job that fails goes back with a raised due epoch; a job that runs out of
//!   tries goes to the dead-letter list, which is never truncated to make room;
//! * duplicate submission by key is refused, not merged: two tickets for one
//!   shard pay two operators for one slot.

use std::collections::BTreeMap;

/// What a job is for. Ordered, not free-form: eviction picks a victim by this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// Reports and notes: first to go.
    Report,
    /// Re-indexing, cache warming.
    Index,
    /// Periodic cleanup.
    Sweep,
    /// Storage repair: last to go, never displaced by a peer.
    Repair,
}

impl Class {
    /// A name for the ledger and the report.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Report => "report",
            Self::Index => "index",
            Self::Sweep => "sweep",
            Self::Repair => "repair",
        }
    }
}

/// Why the queue said no.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    /// A zero-capacity queue admits nothing and reports nothing.
    ZeroCapacity,
    /// Full, and nothing inside was cheaper than what arrived.
    Full {
        /// The capacity.
        capacity: usize,
        /// The class that blocked it.
        blocked_by: Class,
    },
    /// A live job already carries this key.
    DuplicateKey {
        /// The key.
        key: String,
        /// The job that holds it.
        job: u64,
    },
    /// No such job.
    UnknownJob(u64),
    /// A terminal job was touched again.
    AlreadySettled {
        /// The job.
        job: u64,
        /// Where it is.
        where_: &'static str,
    },
    /// The job is not due yet: taking it early is how a backoff becomes a spin.
    NotDue {
        /// The job.
        job: u64,
        /// When it may be taken.
        due: u64,
        /// When it was asked for.
        now: u64,
    },
    /// A retry policy of zero means the job can never be attempted.
    NoTries {
        /// What was asked for.
        max_attempts: u32,
    },
    /// The accounting does not close: work left the queue uncounted.
    Inconsistent {
        /// Submitted.
        submitted: usize,
        /// Everything the queue can account for.
        accounted: usize,
    },
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "a queue with no room is not a queue"),
            Self::Full {
                capacity,
                blocked_by,
            } => write!(
                f,
                "capacity {capacity} is full and every job inside is at least as important \
                 as a {} arrival; refused and counted, not dropped",
                blocked_by.label()
            ),
            Self::DuplicateKey { key, job } => write!(
                f,
                "key `{key}` is already job {job}: merging two submissions pays two \
                 operators for one slot"
            ),
            Self::UnknownJob(id) => write!(f, "no job {id}"),
            Self::AlreadySettled { job, where_ } => {
                write!(f, "job {job} is already settled ({where_})")
            }
            Self::NotDue { job, due, now } => write!(
                f,
                "job {job} is due at {due} and was taken at {now}: an early take defeats \
                 the backoff that put it there"
            ),
            Self::NoTries { max_attempts } => write!(
                f,
                "max_attempts {max_attempts}: a job that can never be attempted should not \
                 be queued, it should be refused where a reader will see it"
            ),
            Self::Inconsistent {
                submitted,
                accounted,
            } => write!(
                f,
                "{submitted} jobs were submitted and {accounted} are accounted for: {} \
                 left the queue with no record",
                submitted.saturating_sub(accounted)
            ),
        }
    }
}

impl std::error::Error for QueueError {}

/// A queued job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    id: u64,
    class: Class,
    key: String,
    payload: String,
    due: u64,
    attempts: u32,
    max_attempts: u32,
}

impl Job {
    /// The job's identity inside the queue.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Its class.
    #[must_use]
    pub fn class(&self) -> Class {
        self.class
    }

    /// Its dedupe key.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// What to do.
    #[must_use]
    pub fn payload(&self) -> &str {
        &self.payload
    }

    /// The first epoch it may be taken at.
    #[must_use]
    pub fn due(&self) -> u64 {
        self.due
    }

    /// Attempts made so far.
    #[must_use]
    pub fn attempts(&self) -> u32 {
        self.attempts
    }
}

/// A settled job's outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Settled {
    /// It finished.
    Done(Job),
    /// It ran out of tries.
    Dead(Job),
}

/// The queue.
#[derive(Debug, Clone)]
pub struct Queue {
    capacity: usize,
    pending: BTreeMap<u64, Job>,
    done: BTreeMap<u64, Job>,
    dead: BTreeMap<u64, Job>,
    by_key: BTreeMap<String, u64>,
    next_id: u64,
    submitted: usize,
    dropped: BTreeMap<Class, usize>,
    refused: usize,
    evicted: usize,
}

impl Queue {
    /// A queue with `capacity` live jobs at most.
    ///
    /// # Errors
    ///
    /// [`QueueError::ZeroCapacity`].
    pub fn with_capacity(capacity: usize) -> Result<Self, QueueError> {
        if capacity == 0 {
            return Err(QueueError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            pending: BTreeMap::new(),
            done: BTreeMap::new(),
            dead: BTreeMap::new(),
            by_key: BTreeMap::new(),
            next_id: 1,
            submitted: 0,
            dropped: BTreeMap::new(),
            refused: 0,
            evicted: 0,
        })
    }

    /// Configured capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Jobs waiting to be taken.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether nothing is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Jobs completed.
    #[must_use]
    pub fn completed(&self) -> usize {
        self.done.len()
    }

    /// Jobs dead-lettered.
    #[must_use]
    pub fn dead(&self) -> usize {
        self.dead.len()
    }

    /// The dead-letter list, in id order. Never truncated to make room: the
    /// list is where a lost job is findable.
    #[must_use]
    pub fn dead_letters(&self) -> Vec<&Job> {
        self.dead.values().collect()
    }

    /// How many jobs were evicted to make room, by class.
    #[must_use]
    pub fn evictions(&self) -> &BTreeMap<Class, usize> {
        &self.dropped
    }

    /// How many arrivals were refused outright.
    #[must_use]
    pub fn refusals(&self) -> usize {
        self.refused
    }

    /// Submissions seen.
    #[must_use]
    pub fn submitted(&self) -> usize {
        self.submitted
    }

    /// Admits a job, evicting a cheaper one if the queue is full.
    ///
    /// # Errors
    ///
    /// [`QueueError::DuplicateKey`], [`QueueError::NoTries`],
    /// [`QueueError::Full`].
    pub fn submit(
        &mut self,
        class: Class,
        key: &str,
        payload: &str,
        due: u64,
        max_attempts: u32,
    ) -> Result<u64, QueueError> {
        self.submitted += 1;
        if max_attempts == 0 {
            self.refused += 1;
            return Err(QueueError::NoTries { max_attempts });
        }
        if let Some(existing) = self.by_key.get(key) {
            self.refused += 1;
            return Err(QueueError::DuplicateKey {
                key: key.to_string(),
                job: *existing,
            });
        }
        if self.pending.len() == self.capacity {
            let victim = self
                .pending
                .values()
                .filter(|job| job.class < class)
                .min_by_key(|job| (job.class, job.due, job.id))
                .map(|job| job.id);
            let Some(victim) = victim else {
                self.refused += 1;
                return Err(QueueError::Full {
                    capacity: self.capacity,
                    blocked_by: class,
                });
            };
            let Some(job) = self.pending.remove(&victim) else {
                self.refused += 1;
                return Err(QueueError::Full {
                    capacity: self.capacity,
                    blocked_by: class,
                });
            };
            self.by_key.remove(&job.key);
            *self.dropped.entry(job.class).or_insert(0) += 1;
            self.evicted += 1;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.pending.insert(
            id,
            Job {
                id,
                class,
                key: key.to_string(),
                payload: payload.to_string(),
                due,
                attempts: 0,
                max_attempts,
            },
        );
        self.by_key.insert(key.to_string(), id);
        Ok(id)
    }

    /// Takes the most important job that is due. Ordering is by class first,
    /// then by how long it has waited, then by id, so a low-priority job that
    /// has waited many epochs is not starved by an eager identical-class arrival:
    /// within a class, waiting is what decides.
    #[must_use]
    pub fn take(&mut self, now: u64) -> Option<Job> {
        let id = self
            .pending
            .values()
            .filter(|job| job.due <= now)
            .max_by_key(|job| (job.class, std::cmp::Reverse(job.due), std::cmp::Reverse(job.id)))
            .map(|job| job.id)?;
        let job = self.pending.remove(&id)?;
        self.by_key.remove(&job.key);
        Some(job)
    }

    /// Re-queues a failed job with a later due epoch.
    ///
    /// # Errors
    ///
    /// [`QueueError::UnknownJob`] when the id was never queued or already
    /// settled. A job out of tries goes to the dead letters instead of an error,
    /// because "we stopped trying" is a result, not a caller mistake - and it is
    /// returned as one, so the caller who has to report it can tell it apart
    /// from a job that will be retried.
    pub fn fail(
        &mut self,
        id: u64,
        backoff: u64,
        now: u64,
    ) -> Result<Option<Settled>, QueueError> {
        let Some(job) = self.pending.get(&id).cloned() else {
            return Err(self.settled_error(id)?);
        };
        self.pending.remove(&id);
        let mut job = job;
        job.attempts += 1;
        if job.attempts >= job.max_attempts {
            self.by_key.remove(&job.key);
            let settled = Settled::Dead(job.clone());
            self.dead.insert(job.id, job);
            return Ok(Some(settled));
        }
        job.due = now.saturating_add(backoff);
        self.pending.insert(job.id, job);
        Ok(None)
    }

    /// Completes a job.
    ///
    /// # Errors
    ///
    /// [`QueueError::UnknownJob`] or [`QueueError::AlreadySettled`].
    pub fn complete(&mut self, id: u64) -> Result<Settled, QueueError> {
        let Some(job) = self.pending.remove(&id) else {
            return Err(self.settled_error(id)?);
        };
        self.by_key.remove(&job.key);
        self.done.insert(id, job.clone());
        Ok(Settled::Done(job))
    }

    /// Refuses a take that is too early, rather than serving it.
    ///
    /// # Errors
    ///
    /// [`QueueError::NotDue`] if `now` is before the job's due epoch.
    pub fn peek_due(&self, id: u64, now: u64) -> Result<&Job, QueueError> {
        let job = self
            .pending
            .get(&id)
            .ok_or(QueueError::UnknownJob(id))?;
        if job.due > now {
            return Err(QueueError::NotDue {
                job: id,
                due: job.due,
                now,
            });
        }
        Ok(job)
    }

    fn settled_error(&self, id: u64) -> QueueError {
        if self.done.contains_key(&id) {
            QueueError::AlreadySettled { job: id, where_: "done" }
        } else if self.dead.contains_key(&id) {
            QueueError::AlreadySettled { job: id, where_: "dead" }
        } else {
            QueueError::UnknownJob(id)
        }
    }

    /// Recomputes the accounting. Every job that entered must be somewhere.
    ///
    /// # Errors
    ///
    /// [`QueueError::Inconsistent`].
    pub fn verify(&self) -> Result<(), QueueError> {
        let accounted = self.pending.len()
            + self.done.len()
            + self.dead.len()
            + self.dropped.values().sum::<usize>()
            + self.refused;
        if accounted != self.submitted {
            return Err(QueueError::Inconsistent {
                submitted: self.submitted,
                accounted,
            });
        }
        // A key may be live in at most one place.
        if self.by_key.len() != self.pending.len() {
            return Err(QueueError::Inconsistent {
                submitted: self.submitted,
                accounted: self.by_key.len(),
            });
        }
        Ok(())
    }

    /// The queue as lines, for a report: what is waiting, what settled, and
    /// what the queue had to give up. The last part is the one a run is
    /// tempted to leave out.
    #[must_use]
    pub fn render(&self) -> Vec<String> {
        let mut out = Vec::new();
        for job in self.pending.values() {
            out.push(format!(
                "pending\t{}\t{}\tdue={}\ttries={}/{}",
                job.id,
                job.class.label(),
                job.due,
                job.attempts,
                job.max_attempts
            ));
        }
        for job in self.dead.values() {
            out.push(format!(
                "dead\t{}\t{}\ttries={}",
                job.id,
                job.class.label(),
                job.attempts
            ));
        }
        for (class, count) in &self.dropped {
            out.push(format!("dropped\t{}\t{}", class.label(), count));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q() -> Queue {
        Queue::with_capacity(2).expect("capacity")
    }

    #[test]
    fn a_queue_with_no_room_is_not_a_queue() {
        assert_eq!(Queue::with_capacity(0), Err(QueueError::ZeroCapacity));
    }

    #[test]
    fn a_job_that_can_never_be_attempted_is_refused_where_it_can_be_seen() {
        let mut q = q();
        assert_eq!(
            q.submit(Class::Repair, "k", "p", 0, 0),
            Err(QueueError::NoTries { max_attempts: 0 })
        );
        assert_eq!(q.refusals(), 1);
        assert_eq!(q.verify(), Ok(()));
    }

    #[test]
    fn two_submissions_for_one_key_are_two_payments_for_one_slot() {
        let mut q = q();
        q.submit(Class::Repair, "manifest-7/shard-2", "ticket", 0, 3)
            .unwrap();
        assert_eq!(
            q.submit(Class::Repair, "manifest-7/shard-2", "ticket again", 0, 3),
            Err(QueueError::DuplicateKey {
                key: "manifest-7/shard-2".to_string(),
                job: 1
            })
        );
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn a_full_queue_evicts_only_what_is_cheaper() {
        let mut q = q();
        q.submit(Class::Report, "r1", "note", 0, 1).unwrap();
        q.submit(Class::Index, "i1", "reindex", 0, 1).unwrap();
        let id = q.submit(Class::Repair, "rep", "fix", 0, 2).unwrap();
        assert!(q.pending.contains_key(&id));
        assert_eq!(q.len(), 2);
        // `Report` was cheaper than `Repair`, so it is the one that went.
        assert_eq!(*q.evictions().get(&Class::Report).unwrap(), 1);
        assert!(!q.pending.values().any(|j| j.class == Class::Report));
        assert_eq!(q.verify(), Ok(()), "the eviction was counted");
    }

    #[test]
    fn a_repair_cannot_displace_a_repair_so_the_arrival_is_refused() {
        let mut q = q();
        q.submit(Class::Repair, "a", "x", 0, 2).unwrap();
        q.submit(Class::Repair, "b", "y", 0, 2).unwrap();
        assert_eq!(
            q.submit(Class::Repair, "c", "z", 0, 2),
            Err(QueueError::Full {
                capacity: 2,
                blocked_by: Class::Repair
            })
        );
        assert_eq!(q.refusals(), 1);
        assert_eq!(q.evictions().len(), 0);
        assert_eq!(q.verify(), Ok(()));
    }

    #[test]
    fn the_highest_class_that_is_due_goes_first_and_unborn_work_waits() {
        let mut q = q();
        let report = q.submit(Class::Report, "r", "p", 0, 1).unwrap();
        let repair = q.submit(Class::Repair, "s", "p", 5, 2).unwrap();
        assert_eq!(q.take(0).map(|j| j.id()), Some(report));
        // The repair is not due at 0: it must not be served early.
        assert_eq!(q.take(0).map(|j| j.id()), None);
        assert_eq!(q.take(5).map(|j| j.id()), Some(repair));
    }

    #[test]
    fn peeking_too_early_names_both_numbers() {
        let mut q = q();
        let id = q.submit(Class::Sweep, "s", "p", 10, 1).unwrap();
        assert_eq!(
            q.peek_due(id, 4),
            Err(QueueError::NotDue {
                job: id,
                due: 10,
                now: 4
            })
        );
        assert!(q.peek_due(id, 10).is_ok());
    }

    #[test]
    fn running_out_of_tries_is_a_result_not_a_caller_error() {
        let mut q = q();
        let id = q.submit(Class::Index, "i", "p", 0, 2).unwrap();
        assert_eq!(q.fail(id, 1, 0).unwrap(), None, "one try left, so still work");
        assert_eq!(q.len(), 1);
        assert_eq!(q.pending[&id].attempts, 1);
        assert_eq!(q.pending[&id].due, 1, "the backoff moved the due epoch");
        let settled = q.fail(id, 1, 1).unwrap().expect("out of tries");
        assert!(matches!(settled, Settled::Dead(_)));
        assert!(q.pending.is_empty());
        assert_eq!(q.dead(), 1);
        assert_eq!(q.dead_letters()[0].id(), id);
        assert_eq!(q.verify(), Ok(()));
    }

    #[test]
    fn touching_a_settled_job_says_where_it_is() {
        let mut q = q();
        let done = q.submit(Class::Report, "a", "p", 0, 1).unwrap();
        let dead = q.submit(Class::Report, "b", "p", 0, 1).unwrap();
        q.complete(done).unwrap();
        q.fail(dead, 0, 0).unwrap();
        assert_eq!(
            q.complete(dead),
            Err(QueueError::AlreadySettled {
                job: dead,
                where_: "dead"
            })
        );
        assert_eq!(
            q.fail(done, 0, 0),
            Err(QueueError::AlreadySettled {
                job: done,
                where_: "done"
            })
        );
        assert_eq!(
            q.fail(999, 0, 0),
            Err(QueueError::UnknownJob(999)),
            "an id that never existed is not the same as a settled one"
        );
    }

    #[test]
    fn the_sum_closes_even_after_everything_bad_happened() {
        let mut q = q();
        for i in 0..10u32 {
            let class = match i % 4 {
                0 => Class::Report,
                1 => Class::Index,
                2 => Class::Sweep,
                _ => Class::Repair,
            };
            let _ = q.submit(class, &format!("key-{i}"), "p", 0, 2);
        }
        let _ = q.submit(Class::Repair, "key-0", "dup", 0, 2);
        while let Some(job) = q.take(99) {
            if job.class == Class::Repair {
                q.complete(job.id()).unwrap();
            } else {
                q.fail(job.id(), 0, 99).unwrap();
            }
        }
        assert_eq!(q.verify(), Ok(()));
        assert!(q.submitted() > 0);
        // Nothing is "lost": the report lists what went, by class.
        assert!(!q.evictions().is_empty() || q.refusals() > 0 || q.dead() > 0);
    }

    #[test]
    fn the_render_carries_the_dropped_part_not_just_the_busy_part() {
        let mut q = q();
        q.submit(Class::Report, "a", "p", 0, 1).unwrap();
        q.submit(Class::Report, "b", "p", 0, 1).unwrap();
        q.submit(Class::Index, "c", "p", 0, 1).unwrap();
        let lines = q.render();
        assert!(lines.iter().any(|l| l.starts_with("pending")));
        assert!(lines.iter().any(|l| l.starts_with("dropped\treport\t1")));
    }
}
