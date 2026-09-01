// A reader that panics stops answering; the panic path is closed here too.
#![forbid(unsafe_code)]
//! # lubot-grant - who is allowed to read what
//!
//! Public content is open. Everything else opens through a **view grant**: the
//! owner names a grantee and a content key id, and the grant carries an expiry.
//! A direct message is that same grant, issued by sending.
//!
//! Three things this module refuses to pretend:
//!
//! 1. **No key material lives here.** A grant is a permission record. Opening
//!    the bytes is the storage layer's job, and it asks this module first.
//! 2. **Revocation stops new opens, not old reads.** Bytes already read cannot
//!    be recalled. [`Decision::Revoked`] is therefore a distinct answer from
//!    [`Decision::NoGrant`]: one says "you had it and it is over", the other
//!    says "you never had it". Collapsing them would be a lie about the past.
//! 3. **A refusal is logged like an allowance.** A refusal that leaves no trace
//!    cannot be audited, and an audit log that only records successes measures
//!    nothing.

use std::collections::HashMap;

/// Wall-clock seconds. The caller supplies it, so a test can pin time and the
/// runtime cannot drift into "now is whatever the machine says".
pub type Seconds = u64;

/// What a content item is exposed as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Anyone may read it. No grant is consulted.
    Public,
    /// Readable only with a live grant for its key id.
    Restricted,
}

/// A single permission: `grantee` may open `key_id` until `expires_at`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewGrant {
    pub key_id: String,
    pub grantee: String,
    pub expires_at: Seconds,
}

/// The answer to "may this reader open this item?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Public content; no grant was needed.
    Public,
    /// A live grant was found.
    Granted,
    /// A grant existed and was withdrawn.
    Revoked,
    /// A grant existed and its expiry has passed.
    Expired,
    /// No grant was ever issued to this reader for this item.
    NoGrant,
}

impl Decision {
    /// True only for the two decisions that open bytes.
    #[must_use]
    pub fn opens(&self) -> bool {
        matches!(self, Decision::Public | Decision::Granted)
    }

    /// The stable word written to the audit log.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Decision::Public => "public",
            Decision::Granted => "granted",
            Decision::Revoked => "revoked",
            Decision::Expired => "expired",
            Decision::NoGrant => "no-grant",
        }
    }
}

/// One line of the audit trail. Refusals are recorded with the same shape as
/// allowances, so a reader of the log cannot tell them apart by absence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEntry {
    pub at: Seconds,
    pub reader: String,
    pub key_id: String,
    pub decision: Decision,
}

/// The permission table plus its audit trail.
#[derive(Debug, Default)]
pub struct GrantBook {
    live: HashMap<(String, String), Seconds>,
    revoked: Vec<(String, String)>,
    log: Vec<AuditEntry>,
}

impl GrantBook {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Issue a grant. Re-issuing after a revocation is allowed and clears the
    /// revocation: the owner is permitted to change their mind.
    pub fn issue(&mut self, grant: ViewGrant) {
        let key = (grant.key_id.clone(), grant.grantee.clone());
        self.revoked.retain(|r| r != &key);
        self.live.insert(key, grant.expires_at);
    }

    /// Withdraw a grant. Returns false when there was nothing to withdraw, so
    /// a caller cannot report a revocation that never happened.
    pub fn revoke(&mut self, key_id: &str, grantee: &str) -> bool {
        let key = (key_id.to_string(), grantee.to_string());
        let existed = self.live.remove(&key).is_some();
        if existed {
            self.revoked.push(key);
        }
        existed
    }

    /// Decide, and record the decision. This is the only entry point: there is
    /// no silent variant, because a silent check is an unlogged read.
    pub fn decide(
        &mut self,
        reader: &str,
        key_id: &str,
        visibility: Visibility,
        now: Seconds,
    ) -> Decision {
        let decision = self.judge(reader, key_id, visibility, now);
        self.log.push(AuditEntry {
            at: now,
            reader: reader.to_string(),
            key_id: key_id.to_string(),
            decision: decision.clone(),
        });
        decision
    }

    fn judge(&self, reader: &str, key_id: &str, visibility: Visibility, now: Seconds) -> Decision {
        if visibility == Visibility::Public {
            return Decision::Public;
        }
        let key = (key_id.to_string(), reader.to_string());
        if let Some(expiry) = self.live.get(&key) {
            if *expiry > now {
                return Decision::Granted;
            }
            return Decision::Expired;
        }
        if self.revoked.contains(&key) {
            return Decision::Revoked;
        }
        Decision::NoGrant
    }

    /// The audit trail, oldest first.
    #[must_use]
    pub fn audit(&self) -> &[AuditEntry] {
        &self.log
    }

    /// How many refusals the log holds. A deployment reporting zero refusals
    /// over a live corpus is reporting that its checks never ran.
    #[must_use]
    pub fn refusals(&self) -> usize {
        self.log.iter().filter(|e| !e.decision.opens()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grant(key: &str, who: &str, until: Seconds) -> ViewGrant {
        ViewGrant {
            key_id: key.to_string(),
            grantee: who.to_string(),
            expires_at: until,
        }
    }

    #[test]
    fn public_content_needs_no_grant() {
        let mut book = GrantBook::new();
        assert_eq!(
            book.decide("anyone", "post-1", Visibility::Public, 10),
            Decision::Public
        );
    }

    #[test]
    fn restricted_content_without_a_grant_is_refused() {
        let mut book = GrantBook::new();
        assert_eq!(
            book.decide("reader", "dm-1", Visibility::Restricted, 10),
            Decision::NoGrant
        );
    }

    #[test]
    fn a_live_grant_opens_and_an_expired_one_does_not() {
        let mut book = GrantBook::new();
        book.issue(grant("dm-1", "reader", 100));
        assert_eq!(
            book.decide("reader", "dm-1", Visibility::Restricted, 99),
            Decision::Granted
        );
        assert_eq!(
            book.decide("reader", "dm-1", Visibility::Restricted, 100),
            Decision::Expired
        );
    }

    #[test]
    fn revocation_is_not_the_same_answer_as_never_granted() {
        let mut book = GrantBook::new();
        book.issue(grant("dm-1", "reader", 100));
        assert!(book.revoke("dm-1", "reader"));
        assert_eq!(
            book.decide("reader", "dm-1", Visibility::Restricted, 10),
            Decision::Revoked
        );
        assert_eq!(
            book.decide("stranger", "dm-1", Visibility::Restricted, 10),
            Decision::NoGrant
        );
    }

    #[test]
    fn revoking_something_that_was_never_granted_reports_false() {
        let mut book = GrantBook::new();
        assert!(!book.revoke("dm-1", "reader"));
    }

    #[test]
    fn a_reissued_grant_overrides_the_revocation() {
        let mut book = GrantBook::new();
        book.issue(grant("dm-1", "reader", 100));
        book.revoke("dm-1", "reader");
        book.issue(grant("dm-1", "reader", 200));
        assert_eq!(
            book.decide("reader", "dm-1", Visibility::Restricted, 150),
            Decision::Granted
        );
    }

    #[test]
    fn a_grant_is_for_one_reader_only() {
        let mut book = GrantBook::new();
        book.issue(grant("dm-1", "reader", 100));
        assert_eq!(
            book.decide("someone-else", "dm-1", Visibility::Restricted, 10),
            Decision::NoGrant
        );
    }

    #[test]
    fn every_decision_is_logged_including_the_refusals() {
        let mut book = GrantBook::new();
        book.decide("a", "p", Visibility::Public, 1);
        book.decide("b", "d", Visibility::Restricted, 2);
        book.issue(grant("d", "c", 50));
        book.decide("c", "d", Visibility::Restricted, 3);
        assert_eq!(book.audit().len(), 3);
        assert_eq!(book.refusals(), 1);
        assert_eq!(book.audit()[1].decision.label(), "no-grant");
    }

    #[test]
    fn only_public_and_granted_open_bytes() {
        assert!(Decision::Public.opens());
        assert!(Decision::Granted.opens());
        assert!(!Decision::Revoked.opens());
        assert!(!Decision::Expired.opens());
        assert!(!Decision::NoGrant.opens());
    }

    #[test]
    fn no_key_material_is_stored_in_the_permission_record() {
        // The record carries an identifier and an expiry, and nothing that
        // could open bytes on its own. A field holding a key would make this
        // module the place a leak comes from.
        let g = grant("dm-1", "reader", 100);
        let printed = format!("{g:?}");
        assert!(printed.contains("dm-1"));
        assert!(!printed.to_lowercase().contains("secret"));
        assert!(!printed.to_lowercase().contains("key_material"));
    }
}
