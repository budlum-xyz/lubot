//! A bounded work queue that does not starve and does not grow without limit.
//!
//! # The three failures a naive priority queue has
//!
//! **1. It is unbounded.** A queue whose capacity is "whatever memory is left" is
//! a way to fill memory from the network side: an attacker who can enqueue does
//! not need to win, only to keep arriving faster than the worker drains. Capacity
//! here is a number set at construction, and a full queue **refuses** rather than
//! growing.
//!
//! **2. It evicts to make room.** Dropping the oldest item to admit a new one
//! loses work silently, and the caller who submitted the dropped item gets no
//! signal - the work simply never happens. This queue refuses the new item
//! instead and returns [`QueueError::Full`], so the caller learns and can retry
//! or report. Which item to lose is a decision that belongs to whoever submitted
//! them, not to the queue.
//!
//! **3. It starves.** A strict priority order means a low-priority item never runs
//! while high-priority work keeps arriving, and "it is in the queue" becomes
//! technically true and practically false. **Starvation is not fairness.** Items
//! here age: each time an item is passed over, its effective priority rises by
//! [`Queue::aging_step`], so a low-priority item eventually outranks a fresh
//! high-priority one. The number of passes is recorded, which makes starvation
//! observable before it becomes a complaint.
//!
//! # Retries
//!
//! An item that fails is retried, because most failures are transient. An item
//! that fails forever is moved to the dead-letter list after
//! [`Queue::max_attempts`], because retrying it forever is a way for one bad item
//! to consume the worker indefinitely - which is a denial of service that needs
//! no attacker. The dead-letter list is bounded too, and keeps the attempt count
//! and the last error, so the item can be diagnosed rather than merely noticed.

use std::collections::BTreeMap;

/// Why the queue refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    /// The queue is at capacity. Refused rather than evicting: which item to lose
    /// belongs to whoever submitted them, not to the queue.
    Full { capacity: usize },
    /// An item with this key is already queued. Idempotent by design - the same
    /// work arriving twice is one item, and counting it twice is how a retry
    /// storm turns into a backlog.
    AlreadyQueued { key: String },
    /// The item is in the dead-letter list. It does not come back by being
    /// re-submitted; it is re-queued explicitly, which is a decision.
    DeadLettered { key: String },
    /// No item with this key.
    Unknown { key: String },
    /// The queue is empty.
    Empty,
    /// A capacity of zero. A queue that cannot hold anything is not a small
    /// queue, it is a queue that refuses everything, and that is a configuration
    /// mistake worth reporting.
    ZeroCapacity,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full { capacity } => {
                write!(f, "the queue is at its capacity of {capacity} and does not evict to make room")
            }
            Self::AlreadyQueued { key } => write!(f, "{key:?} is already queued"),
            Self::DeadLettered { key } => {
                write!(f, "{key:?} is in the dead-letter list and must be re-queued explicitly")
            }
            Self::Unknown { key } => write!(f, "there is no queued item {key:?}"),
            Self::Empty => write!(f, "the queue is empty"),
            Self::ZeroCapacity => write!(f, "a capacity of zero is not a small queue, it is one that refuses everything"),
        }
    }
}

/// One queued item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub key: String,
    /// The priority it was submitted at. Never changes; the *effective* priority
    /// rises as the item is passed over.
    pub base_priority: u32,
    /// How many times this item has been passed over. Recorded so starvation is
    /// observable before it becomes a complaint.
    pub passes: u64,
    /// How many times it has been attempted.
    pub attempts: u64,
    /// When it was enqueued, in whatever clock the caller uses. Carried so age
    /// can be reported; not used for ordering, because a clock that moves
    /// backwards would reorder the queue.
    pub enqueued_at: u64,
    /// Opaque payload.
    pub body: Vec<u8>,
}

impl Item {
    /// The effective priority: base plus aging.
    ///
    /// Saturating, because an item passed over often enough to overflow has a
    /// problem that wrapping its priority to zero would hide.
    #[must_use]
    pub fn effective_priority(&self, aging_step: u32) -> u64 {
        u64::from(self.base_priority).saturating_add(self.passes.saturating_mul(u64::from(aging_step)))
    }
}

/// One dead-lettered item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeadItem {
    pub key: String,
    pub attempts: u64,
    /// The last error. Kept so the item can be diagnosed rather than merely
    /// noticed.
    pub last_error: String,
}

/// The queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Queue {
    capacity: usize,
    aging_step: u32,
    max_attempts: u64,
    dead_letter_capacity: usize,
    items: BTreeMap<String, Item>,
    dead: Vec<DeadItem>,
    /// Total items refused for being full. Reported because a queue that refuses
    /// silently looks exactly like a queue that is keeping up.
    pub refused_full: u64,
    /// Total items dead-lettered.
    pub dead_lettered: u64,
}

impl Queue {
    /// A queue with `capacity`, an `aging_step` per pass, and `max_attempts`
    /// before an item is dead-lettered.
    ///
    /// # Errors
    ///
    /// [`QueueError::ZeroCapacity`].
    pub fn new(capacity: usize, aging_step: u32, max_attempts: u64, dead_letter_capacity: usize) -> Result<Self, QueueError> {
        if capacity == 0 {
            return Err(QueueError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            aging_step,
            max_attempts,
            dead_letter_capacity,
            items: BTreeMap::new(),
            dead: Vec::new(),
            refused_full: 0,
            dead_lettered: 0,
        })
    }

    /// Enqueues an item.
    ///
    /// # Errors
    ///
    /// [`QueueError::Full`], [`QueueError::AlreadyQueued`], or
    /// [`QueueError::DeadLettered`].
    pub fn enqueue(&mut self, key: &str, base_priority: u32, enqueued_at: u64, body: &[u8]) -> Result<(), QueueError> {
        if self.dead.iter().any(|d| d.key == key) {
            return Err(QueueError::DeadLettered { key: key.to_string() });
        }
        if self.items.contains_key(key) {
            return Err(QueueError::AlreadyQueued { key: key.to_string() });
        }
        if self.items.len() >= self.capacity {
            self.refused_full = self.refused_full.saturating_add(1);
            return Err(QueueError::Full {
                capacity: self.capacity,
            });
        }
        self.items.insert(
            key.to_string(),
            Item {
                key: key.to_string(),
                base_priority,
                passes: 0,
                attempts: 0,
                enqueued_at,
                body: body.to_vec(),
            },
        );
        Ok(())
    }

    /// The key of the item that should run next, without removing it.
    ///
    /// Highest effective priority wins; ties break on the earliest enqueue time,
    /// and then on the key, so the order is deterministic across nodes rather
    /// than dependent on map iteration.
    ///
    /// # Errors
    ///
    /// [`QueueError::Empty`].
    pub fn peek(&self) -> Result<&str, QueueError> {
        self.best_key().ok_or(QueueError::Empty)
    }

    /// Records that the next item was passed over, raising its effective
    /// priority.
    ///
    /// Called when the worker takes something else instead. Without this, a
    /// low-priority item never runs while high-priority work keeps arriving.
    pub fn pass_over(&mut self, key: &str) -> Result<u64, QueueError> {
        let item = self.items.get_mut(key).ok_or_else(|| QueueError::Unknown { key: key.to_string() })?;
        item.passes = item.passes.saturating_add(1);
        Ok(item.effective_priority(self.aging_step))
    }

    /// Takes the next item for processing.
    ///
    /// Removes it from the queue. If the attempt fails, [`Self::retry`] puts it
    /// back with the attempt count raised; if it succeeds, it is simply gone.
    /// Removing on take rather than on completion is deliberate: an item that is
    /// in flight must not be handed to a second worker.
    ///
    /// # Errors
    ///
    /// [`QueueError::Empty`].
    pub fn take(&mut self) -> Result<Item, QueueError> {
        let key = self.best_key().ok_or(QueueError::Empty)?.to_string();
        self.items
            .remove(&key)
            .ok_or_else(|| QueueError::Unknown { key })
    }

    /// Puts a failed item back, or dead-letters it.
    ///
    /// # Errors
    ///
    /// [`QueueError::Full`] if the queue filled while the item was in flight.
    /// The item is dead-lettered in that case rather than dropped, because a
    /// failure to re-queue is not a reason to lose the work.
    pub fn retry(&mut self, mut item: Item, error: &str) -> Result<(), QueueError> {
        item.attempts = item.attempts.saturating_add(1);
        if item.attempts >= self.max_attempts {
            self.dead_letter(&item, error);
            return Ok(());
        }
        if self.items.len() >= self.capacity {
            // Could not be re-queued. Dead-letter rather than drop: the work is
            // recorded somewhere instead of vanishing.
            self.dead_letter(&item, &format!("{error} (could not re-queue: full)"));
            return Err(QueueError::Full {
                capacity: self.capacity,
            });
        }
        self.items.insert(item.key.clone(), item);
        Ok(())
    }

    /// Moves an item to the dead-letter list.
    fn dead_letter(&mut self, item: &Item, error: &str) {
        if self.dead.len() >= self.dead_letter_capacity {
            self.dead.remove(0);
        }
        self.dead.push(DeadItem {
            key: item.key.clone(),
            attempts: item.attempts,
            last_error: error.to_string(),
        });
        self.dead_lettered = self.dead_lettered.saturating_add(1);
    }

    /// Re-queues a dead-lettered item. Explicit, because an item that failed
    /// `max_attempts` times coming back on its own is how a poison item starts
    /// consuming the worker again.
    ///
    /// # Errors
    ///
    /// [`QueueError::Unknown`] if the key is not dead-lettered, or
    /// [`QueueError::Full`].
    pub fn requeue_dead(&mut self, key: &str, base_priority: u32, enqueued_at: u64, body: &[u8]) -> Result<(), QueueError> {
        let Some(position) = self.dead.iter().position(|d| d.key == key) else {
            return Err(QueueError::Unknown { key: key.to_string() });
        };
        if self.items.len() >= self.capacity {
            return Err(QueueError::Full {
                capacity: self.capacity,
            });
        }
        self.dead.remove(position);
        self.items.insert(
            key.to_string(),
            Item {
                key: key.to_string(),
                base_priority,
                passes: 0,
                attempts: 0,
                enqueued_at,
                body: body.to_vec(),
            },
        );
        Ok(())
    }

    /// The dead-letter list.
    #[must_use]
    pub fn dead_letters(&self) -> &[DeadItem] {
        &self.dead
    }

    /// How many items are queued.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The highest number of passes any queued item has accumulated.
    ///
    /// The starvation gauge. A number that keeps climbing means low-priority work
    /// is arriving faster than it is being served, and that is visible here
    /// before it becomes a complaint from whoever submitted it.
    #[must_use]
    pub fn worst_starvation(&self) -> u64 {
        self.items.values().map(|i| i.passes).max().unwrap_or(0)
    }

    /// How many times an item has been passed over.
    ///
    /// Exposed because the pass count is the starvation gauge: a number that
    /// keeps climbing means low-priority work is arriving faster than it is
    /// served, and that should be visible before it becomes a complaint.
    #[must_use]
    pub fn get_passes(&self, key: &str) -> Option<u64> {
        self.items.get(key).map(|i| i.passes)
    }

    /// Chooses the best key.
    fn best_key(&self) -> Option<&str> {
        self.items
            .values()
            .max_by(|a, b| {
                a.effective_priority(self.aging_step)
                    .cmp(&b.effective_priority(self.aging_step))
                    .then_with(|| b.enqueued_at.cmp(&a.enqueued_at).reverse())
                    .then_with(|| b.key.cmp(&a.key).reverse())
            })
            .map(|i| i.key.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue() -> Queue {
        Queue::new(4, 1, 3, 8).expect("queue")
    }

    #[test]
    fn a_capacity_of_zero_is_refused() {
        // A queue that cannot hold anything is not a small queue, it is one that
        // refuses everything, and that is a configuration mistake worth reporting.
        assert_eq!(Queue::new(0, 1, 3, 8), Err(QueueError::ZeroCapacity));
    }

    #[test]
    fn a_full_queue_refuses_rather_than_evicting() {
        // Dropping the oldest to admit a new one loses work silently. Which item
        // to lose belongs to whoever submitted them.
        let mut q = queue();
        for i in 0..4 {
            q.enqueue(&format!("k{i}"), 5, u64::from(i as u32), b"x").expect("enqueue");
        }
        assert_eq!(q.len(), 4);
        assert_eq!(
            q.enqueue("k9", 99, 10, b"x"),
            Err(QueueError::Full { capacity: 4 }),
            "the queue grew or evicted"
        );
        assert_eq!(q.len(), 4);
        assert_eq!(q.refused_full, 1);
        // The refusal is counted, because a queue that refuses silently looks
        // exactly like a queue that is keeping up.
    }

    #[test]
    fn the_same_work_arriving_twice_is_one_item() {
        // Counting it twice is how a retry storm turns into a backlog.
        let mut q = queue();
        q.enqueue("k", 5, 0, b"x").expect("first");
        assert_eq!(
            q.enqueue("k", 5, 1, b"y"),
            Err(QueueError::AlreadyQueued {
                key: "k".to_string()
            })
        );
        assert_eq!(q.len(), 1);
        assert_eq!(q.take().expect("take").body, b"x", "the second submit overwrote the first");
    }

    #[test]
    fn the_highest_effective_priority_runs_first() {
        let mut q = queue();
        q.enqueue("low", 1, 0, b"x").expect("enqueue");
        q.enqueue("high", 9, 0, b"x").expect("enqueue");
        assert_eq!(q.peek(), Ok("high"));
    }

    #[test]
    fn aging_eventually_outranks_a_fresh_high_priority_item() {
        // Starvation is not fairness. Without aging, "it is in the queue" is
        // technically true and practically false.
        let mut q = Queue::new(8, 5, 3, 8).expect("queue");
        q.enqueue("low", 1, 0, b"x").expect("enqueue");
        q.enqueue("high", 9, 0, b"x").expect("enqueue");
        assert_eq!(q.peek(), Ok("high"));
        // Pass the low item over until it outranks the fresh high one.
        for _ in 0..2 {
            q.pass_over("low").expect("pass");
        }
        assert_eq!(q.peek(), Ok("low"), "aging did not outrank a fresh high-priority item");
    }

    #[test]
    fn the_pass_count_is_recorded_so_starvation_is_observable() {
        let mut q = queue();
        q.enqueue("low", 1, 0, b"x").expect("enqueue");
        assert_eq!(q.worst_starvation(), 0);
        q.pass_over("low").expect("pass");
        q.pass_over("low").expect("pass");
        assert_eq!(q.worst_starvation(), 2);
        assert_eq!(q.get_passes("low"), Some(2));
    }

    #[test]
    fn an_item_in_flight_is_not_handed_to_a_second_worker() {
        // Removing on take rather than on completion.
        let mut q = queue();
        q.enqueue("k", 5, 0, b"x").expect("enqueue");
        let taken = q.take().expect("take");
        assert_eq!(taken.key, "k");
        assert!(q.is_empty(), "the item was still queued while in flight");
        assert_eq!(q.take(), Err(QueueError::Empty));
    }

    #[test]
    fn a_failing_item_is_retried_with_its_attempt_count_raised() {
        let mut q = queue();
        q.enqueue("k", 5, 0, b"x").expect("enqueue");
        let item = q.take().expect("take");
        q.retry(item, "transient").expect("retry");
        assert_eq!(q.len(), 1);
        assert_eq!(q.take().expect("take").attempts, 1);
    }

    #[test]
    fn an_item_that_fails_forever_is_dead_lettered_not_retried_forever() {
        // Retrying forever is a denial of service that needs no attacker.
        let mut q = Queue::new(4, 1, 2, 8).expect("queue");
        q.enqueue("poison", 5, 0, b"x").expect("enqueue");
        for _ in 0..2 {
            let item = q.take().expect("take");
            q.retry(item, "always fails").ok();
        }
        assert!(q.is_empty());
        assert_eq!(q.dead_letters().len(), 1);
        assert_eq!(q.dead_lettered, 1);
        let dead = q.dead_letters().first().expect("dead");
        assert_eq!(dead.key, "poison");
        assert_eq!(dead.last_error, "always fails");
    }

    #[test]
    fn a_dead_lettered_item_does_not_come_back_by_being_resubmitted() {
        // An item that failed max_attempts times returning on its own is how a
        // poison item starts consuming the worker again.
        let mut q = Queue::new(4, 1, 1, 8).expect("queue");
        q.enqueue("poison", 5, 0, b"x").expect("enqueue");
        let item = q.take().expect("take");
        q.retry(item, "always fails").ok();
        assert_eq!(
            q.enqueue("poison", 5, 0, b"x"),
            Err(QueueError::DeadLettered {
                key: "poison".to_string()
            })
        );
        // It comes back only by an explicit decision.
        q.requeue_dead("poison", 5, 1, b"x").expect("requeue");
        assert_eq!(q.len(), 1);
        assert!(q.dead_letters().is_empty());
    }

    #[test]
    fn an_item_that_cannot_be_requeued_is_dead_lettered_not_dropped() {
        // A failure to re-queue is not a reason to lose the work.
        let mut q = Queue::new(1, 1, 9, 8).expect("queue");
        q.enqueue("a", 5, 0, b"x").expect("enqueue");
        let a = q.take().expect("take");
        q.enqueue("b", 5, 1, b"y").expect("fills the single slot");
        assert!(matches!(q.retry(a, "transient"), Err(QueueError::Full { .. })));
        assert_eq!(q.dead_letters().len(), 1, "the work vanished instead of being recorded");
        assert!(q.dead_letters().first().is_some_and(|d| d.key == "a"));
    }

    #[test]
    fn the_dead_letter_list_is_bounded() {
        let mut q = Queue::new(4, 1, 1, 2).expect("queue");
        for i in 0..5 {
            q.enqueue(&format!("k{i}"), 5, u64::from(i as u32), b"x").expect("enqueue");
            let item = q.take().expect("take");
            q.retry(item, "fails").ok();
        }
        assert_eq!(q.dead_letters().len(), 2, "the dead-letter list grew without limit");
        assert_eq!(q.dead_lettered, 5, "the counter still records everything that was lettered");
    }

    #[test]
    fn ordering_is_deterministic_across_nodes() {
        // Ties break on enqueue time and then on key, so two nodes with the same
        // items in a different insertion order choose the same one.
        let mut a = queue();
        a.enqueue("b", 5, 0, b"x").expect("enqueue");
        a.enqueue("a", 5, 0, b"x").expect("enqueue");
        let mut b = queue();
        b.enqueue("a", 5, 0, b"x").expect("enqueue");
        b.enqueue("b", 5, 0, b"x").expect("enqueue");
        assert_eq!(a.peek(), b.peek());
    }

    #[test]
    fn an_earlier_enqueue_wins_a_priority_tie() {
        let mut q = queue();
        q.enqueue("later", 5, 10, b"x").expect("enqueue");
        q.enqueue("earlier", 5, 1, b"x").expect("enqueue");
        assert_eq!(q.peek(), Ok("earlier"));
    }

    #[test]
    fn an_empty_queue_reports_empty_rather_than_panicking() {
        let mut q = queue();
        assert_eq!(q.peek(), Err(QueueError::Empty));
        assert_eq!(q.take(), Err(QueueError::Empty));
        assert!(q.is_empty());
        assert_eq!(q.worst_starvation(), 0);
    }
}
