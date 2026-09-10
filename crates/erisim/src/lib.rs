//! A grant that can be widened by its holder is not a grant, it is a role.
//!
//! Lubot asks to do things on other people's behalf: read a shard, mint a
//! record, move a handle. Every system that lets an agent act carries some
//! version of this ledger, and every version of it eventually produces the same
//! incident - a holder delegates to a helper, and the helper ends up able to do
//! *more* than the holder, because nobody re-checked the three numbers that
//! matter when a grant is copied: capability, scope, and expiry.
//!
//! # The rules this keeps
//!
//! * a grant is bounded by `expires_epoch` and by a use count; `0` for either is
//!   refused at issue time rather than admitted as a "no limit" sentinel,
//!   because a sentinel is what a bug reaches for;
//! * **scope is a path with a separator rule.** `src/storage` covers
//!   `src/storage/deal.rs` and does *not* cover `src/storagesecrets`. The naive
//!   prefix test - `starts_with` - is the bug this whole rule exists to pin, and
//!   the test that pins it is worth more than the rest of the file;
//! * a delegated grant may only be narrower than its parent, in all three
//!   dimensions, and the narrowing is checked against the *parent's recorded*
//!   bounds, not against whatever the child asks for;
//! * revocation is not deletion. A revoked grant keeps its record and its audit
//!   trail; a use after revocation fails with the grant id, not a generic denial.
//!
//! # No clock
//!
//! Every method that needs "now" takes it as `at`. A ledger that reads a clock
//! cannot be tested for expiry boundaries, and an expiry boundary tested at one
//! instant is not tested.
//!
//! # What is not here
//!
//! No keys, no signatures. This ledger decides what a token *permits*; it never
//! claims a token is genuine. Feeding it verified claims is the caller's job, and
//! the type says so by taking a `holder: &str` instead of pretending to know one.

use std::collections::BTreeMap;

/// What a grant permits, as an opaque label agreed between issuer and check.
///
/// Deliberately a string and not an enum: the vocabulary belongs to the module
/// being accessed, and a central enum of capabilities is a coupling point that
/// every module ends up editing - which is how capability lists drift into
/// granting more than their name says.
pub type Capability = String;

/// One issued grant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grant {
    id: u64,
    holder: String,
    capability: Capability,
    scope: String,
    issued_at: u64,
    expires_at: u64,
    uses_left: Option<u32>,
    uses_used: u32,
    revoked_at: Option<u64>,
    parent: Option<u64>,
}

impl Grant {
    /// The grant's identity.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Who holds it.
    #[must_use]
    pub fn holder(&self) -> &str {
        &self.holder
    }

    /// What it permits.
    #[must_use]
    pub fn capability(&self) -> &str {
        &self.capability
    }

    /// Where it permits it.
    #[must_use]
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// The epoch it was issued at.
    #[must_use]
    pub fn issued_at(&self) -> u64 {
        self.issued_at
    }

    /// The first epoch at which it is no longer valid.
    #[must_use]
    pub fn expires_at(&self) -> u64 {
        self.expires_at
    }

    /// Remaining uses, or `None` for unlimited-by-count.
    #[must_use]
    pub fn uses_left(&self) -> Option<u32> {
        self.uses_left
    }

    /// Uses consumed.
    #[must_use]
    pub fn uses_used(&self) -> u32 {
        self.uses_used
    }

    /// The epoch it was revoked at, if it was.
    #[must_use]
    pub fn revoked_at(&self) -> Option<u64> {
        self.revoked_at
    }

    /// The grant this one was narrowed from.
    #[must_use]
    pub fn parent(&self) -> Option<u64> {
        self.parent
    }

    /// Whether `target` is inside this grant's scope.
    ///
    /// A scope covers itself, and covers what hangs below the separator:
    /// `src/storage` covers `src/storage/deal.rs`, and neither covers
    /// `src/storage-deal` or `src/storagesecrets`. A bare prefix test would.
    #[must_use]
    pub fn covers(&self, target: &str) -> bool {
        if self.scope == target {
            return true;
        }
        let Some(rest) = target.strip_prefix(self.scope.as_str()) else {
            return false;
        };
        rest.starts_with('/')
    }
}

/// Why a request was refused, or why a ledger is inconsistent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantError {
    /// A zero-length life is a way to spell "no grant" without saying it.
    ZeroTtl {
        /// The requested ttl.
        ttl: u64,
    },
    /// A grant with no uses cannot be used, and is therefore a record of a
    /// refusal rather than a permission.
    ZeroUses,
    /// An empty scope covers everything by a prefix test, which is the opposite
    /// of what a reader will assume.
    EmptyScope,
    /// An empty capability matches nothing, and a check against nothing passes
    /// when the writer is tired.
    EmptyCapability,
    /// An empty holder is unattributable, and an unattributable action is not
    /// auditable.
    EmptyHolder,
    /// A grant that starts after or at its own expiry.
    Backwards {
        /// Issue epoch.
        issued_at: u64,
        /// Expiry epoch.
        expires_at: u64,
    },
    /// No such grant.
    UnknownGrant(u64),
    /// The grant was revoked at `at`, and the attempt is not older than that.
    Revoked {
        /// The grant.
        grant: u64,
        /// When it was revoked.
        at: u64,
    },
    /// `at` is at or after the expiry.
    Expired {
        /// The grant.
        grant: u64,
        /// When it died.
        expires_at: u64,
        /// When it was asked.
        at: u64,
    },
    /// The capability asked for is not the capability granted.
    CapabilityMismatch {
        /// The grant.
        grant: u64,
        /// What was asked.
        want: Capability,
        /// What was granted.
        got: Capability,
    },
    /// The target is outside the scope.
    ScopeEscape {
        /// The grant.
        grant: u64,
        /// What was asked for.
        want: String,
        /// What the grant covers.
        got: String,
    },
    /// The count is spent.
    UseLimitReached {
        /// The grant.
        grant: u64,
        /// What it allowed.
        limit: u32,
    },
    /// A delegation tried to widen. Reported as one of three named cases, not
    /// as "invalid", because the fix differs.
    DelegateWidens {
        /// The parent grant.
        parent: u64,
        /// Which dimension.
        dimension: &'static str,
    },
    /// The parent is not usable, so nothing can be narrowed from it.
    ParentUnusable(u64),
    /// A use was recorded against a grant that no longer permits it, or the
    /// ledger's counters disagree with its records.
    Inconsistent {
        /// What the counters said.
        detail: String,
    },
}

impl std::fmt::Display for GrantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTtl { ttl } => write!(
                f,
                "ttl {ttl}: a grant that lives no epochs is a refusal written as a permission"
            ),
            Self::ZeroUses => write!(
                f,
                "a grant with no uses cannot be used; keep it out of the ledger"
            ),
            Self::EmptyScope => write!(
                f,
                "an empty scope matches every path under a prefix test; say what you mean \
                 and grant nothing instead"
            ),
            Self::EmptyCapability => write!(
                f,
                "an empty capability is how a typo becomes a wildcard"
            ),
            Self::EmptyHolder => write!(f, "an unattributable grant is not auditable"),
            Self::Backwards {
                issued_at,
                expires_at,
            } => write!(
                f,
                "issued at {issued_at} expiring at {expires_at}: a grant born dead is a bug, \
                 not a policy"
            ),
            Self::UnknownGrant(id) => write!(f, "no grant {id}"),
            Self::Revoked { grant, at } => {
                write!(f, "grant {grant} was revoked at epoch {at}")
            }
            Self::Expired {
                grant,
                expires_at,
                at,
            } => write!(
                f,
                "grant {grant} expired at {expires_at} and was used at {at}: the boundary is \
                 `at >= expires_at`, not `>`, or an off-by-one buys an epoch"
            ),
            Self::CapabilityMismatch { grant, want, got } => write!(
                f,
                "grant {grant} permits `{got}` and was asked for `{want}`: no translation \
                 between capability strings, ever"
            ),
            Self::ScopeEscape { grant, want, got } => write!(
                f,
                "grant {grant} covers `{got}` and was used on `{want}`"
            ),
            Self::UseLimitReached { grant, limit } => write!(
                f,
                "grant {grant} allowed {limit} uses and has spent them"
            ),
            Self::DelegateWidens { parent, dimension } => write!(
                f,
                "a grant delegated from {parent} tried to widen its {dimension}: a delegate \
                 that can grow is a role, not a grant"
            ),
            Self::ParentUnusable(id) => write!(
                f,
                "grant {id} is expired, revoked or spent, so nothing may be narrowed from it"
            ),
            Self::Inconsistent { detail } => {
                write!(f, "the ledger does not match its own records: {detail}")
            }
        }
    }
}

impl std::error::Error for GrantError {}

/// One audit line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    seq: u64,
    at: u64,
    kind: &'static str,
    grant: u64,
    note: String,
}

impl Event {
    /// Its position in the trail.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The epoch it records.
    #[must_use]
    pub fn at(&self) -> u64 {
        self.at
    }

    /// `issue`, `use`, `revoke` or `refuse`.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        self.kind
    }

    /// The grant it is about.
    #[must_use]
    pub fn grant(&self) -> u64 {
        self.grant
    }

    /// The detail.
    #[must_use]
    pub fn note(&self) -> &str {
        &self.note
    }
}

/// The ledger.
#[derive(Debug, Clone)]
pub struct Ledger {
    grants: BTreeMap<u64, Grant>,
    events: Vec<Event>,
    next_id: u64,
    uses_granted: u32,
    uses_consumed: u32,
}

impl Default for Ledger {
    fn default() -> Self {
        Self {
            grants: BTreeMap::new(),
            events: Vec::new(),
            next_id: 1,
            uses_granted: 0,
            uses_consumed: 0,
        }
    }
}

impl Ledger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live records, revoked ones included: revocation is not
    /// deletion.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grants.len()
    }

    /// Whether nothing has ever been issued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grants.is_empty()
    }

    /// The audit trail.
    #[must_use]
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// A record.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Grant> {
        self.grants.get(&id)
    }

    /// Issues a grant.
    ///
    /// `ttl` is counted from `at`; `uses` of `None` means unlimited by count -
    /// still bounded by `at + ttl`, because a grant bounded by nothing is a role.
    ///
    /// # Errors
    ///
    /// The construction-shaped [`GrantError`] variants.
    pub fn issue(
        &mut self,
        at: u64,
        holder: &str,
        capability: &str,
        scope: &str,
        ttl: u64,
        uses: Option<u32>,
    ) -> Result<u64, GrantError> {
        if holder.trim().is_empty() {
            return Err(GrantError::EmptyHolder);
        }
        if capability.trim().is_empty() {
            return Err(GrantError::EmptyCapability);
        }
        if scope.trim().is_empty() {
            return Err(GrantError::EmptyScope);
        }
        if ttl == 0 {
            return Err(GrantError::ZeroTtl { ttl });
        }
        if uses == Some(0) {
            return Err(GrantError::ZeroUses);
        }
        let expires_at = at.saturating_add(ttl);
        if expires_at <= at {
            return Err(GrantError::Backwards {
                issued_at: at,
                expires_at,
            });
        }
        let id = self.next_id;
        self.next_id += 1;
        self.grants.insert(
            id,
            Grant {
                id,
                holder: holder.to_string(),
                capability: capability.to_string(),
                scope: scope.to_string(),
                issued_at: at,
                expires_at,
                uses_left: uses,
                uses_used: 0,
                revoked_at: None,
                parent: None,
            },
        );
        if let Some(n) = uses {
            self.uses_granted = self.uses_granted.saturating_add(n);
        }
        self.push_event(at, "issue", id, format!("{holder}:{capability} on {scope}"));
        Ok(id)
    }

    /// Narrows a grant into a new one for another holder.
    ///
    /// # Errors
    ///
    /// [`GrantError::UnknownGrant`], [`GrantError::ParentUnusable`], and
    /// [`GrantError::DelegateWidens`] naming which dimension widened.
    pub fn delegate(
        &mut self,
        at: u64,
        parent: u64,
        holder: &str,
        capability: &str,
        scope: &str,
        ttl: u64,
        uses: Option<u32>,
    ) -> Result<u64, GrantError> {
        let Some(p) = self.grants.get(&parent).cloned() else {
            return Err(GrantError::UnknownGrant(parent));
        };
        if p.revoked_at.is_some() || p.expires_at <= at || p.uses_left == Some(0) {
            return Err(GrantError::ParentUnusable(parent));
        }
        if capability != p.capability {
            return Err(GrantError::DelegateWidens {
                parent,
                dimension: "capability",
            });
        }
        if !p.covers(scope) {
            return Err(GrantError::DelegateWidens {
                parent,
                dimension: "scope",
            });
        }
        if ttl > p.expires_at.saturating_sub(at) {
            return Err(GrantError::DelegateWidens {
                parent,
                dimension: "expiry",
            });
        }
        if let Some(limit) = p.uses_left {
            let left = limit.saturating_sub(p.uses_used);
            if uses.unwrap_or(u32::MAX) > left {
                return Err(GrantError::DelegateWidens {
                    parent,
                    dimension: "use count",
                });
            }
        }
        // An unlimited parent may narrow to a counted child, and that is not
        // widening; nothing is checked for it above, deliberately.
        let id = self.next_id;
        self.next_id += 1;
        self.grants.insert(
            id,
            Grant {
                id,
                holder: holder.to_string(),
                capability: capability.to_string(),
                scope: scope.to_string(),
                issued_at: at,
                expires_at: at.saturating_add(ttl),
                uses_left: uses,
                uses_used: 0,
                revoked_at: None,
                parent: Some(parent),
            },
        );
        if let Some(n) = uses {
            self.uses_granted = self.uses_granted.saturating_add(n);
        }
        self.push_event(
            at,
            "delegate",
            id,
            format!("from {parent} to {holder} on {scope}"),
        );
        Ok(id)
    }

    /// Records a use of the grant, if it is permitted.
    ///
    /// # Errors
    ///
    /// Every refusal here is a distinct variant, because a caller deciding what
    /// to tell its user cannot decide from "denied".
    pub fn consume(
        &mut self,
        id: u64,
        at: u64,
        want_cap: &str,
        want_target: &str,
    ) -> Result<(), GrantError> {
        let Some(g) = self.grants.get(&id).cloned() else {
            return Err(GrantError::UnknownGrant(id));
        };
        if let Some(revoked_at) = g.revoked_at {
            self.push_event(at, "refuse", id, format!("revoked at {revoked_at}"));
            return Err(GrantError::Revoked {
                grant: id,
                at: revoked_at,
            });
        }
        if at >= g.expires_at {
            self.push_event(at, "refuse", id, format!("expired at {}", g.expires_at));
            return Err(GrantError::Expired {
                grant: id,
                expires_at: g.expires_at,
                at,
            });
        }
        if want_cap != g.capability {
            self.push_event(at, "refuse", id, format!("asked {want_cap}"));
            return Err(GrantError::CapabilityMismatch {
                grant: id,
                want: want_cap.to_string(),
                got: g.capability.clone(),
            });
        }
        if !g.covers(want_target) {
            self.push_event(at, "refuse", id, format!("asked {want_target}"));
            return Err(GrantError::ScopeEscape {
                grant: id,
                want: want_target.to_string(),
                got: g.scope.clone(),
            });
        }
        if let Some(limit) = g.uses_left {
            if g.uses_used >= limit {
                self.push_event(at, "refuse", id, format!("limit {limit} spent"));
                return Err(GrantError::UseLimitReached {
                    grant: id,
                    limit,
                });
            }
        }
        let Some(record) = self.grants.get_mut(&id) else {
            return Err(GrantError::UnknownGrant(id));
        };
        record.uses_used += 1;
        self.uses_consumed += 1;
        self.push_event(at, "use", id, format!("{want_cap} on {want_target}"));
        Ok(())
    }

    /// Revokes a grant. Returns whether anything changed, so a double revoke is
    /// visible instead of being silently idempotent.
    pub fn revoke(&mut self, id: u64, at: u64) -> bool {
        let Some(record) = self.grants.get_mut(&id) else {
            return false;
        };
        if record.revoked_at.is_some() {
            self.push_event(at, "refuse", id, "already revoked".to_string());
            return false;
        }
        record.revoked_at = Some(at);
        if record.parent.is_some() {
            // A child's existence must not outlive its parent's meaning: revoke
            // the whole subtree, or a revoked root keeps working through the
            // narrower copy someone made before the revocation.
            let children: Vec<u64> = self
                .grants
                .values()
                .filter(|g| g.parent == Some(id))
                .map(|g| g.id)
                .collect();
            for child in children {
                if let Some(c) = self.grants.get_mut(&child) {
                    if c.revoked_at.is_none() {
                        c.revoked_at = Some(at);
                        self.push_event(at, "revoke", child, format!("with parent {id}"));
                    }
                }
            }
        }
        self.push_event(at, "revoke", id, String::new());
        true
    }

    /// Recomputes the counters against the records.
    ///
    /// The trail is not trusted: an event log whose events do not match the
    /// records it describes is a record of someone's intentions, not of what
    /// happened.
    ///
    /// # Errors
    ///
    /// The first [`GrantError::Inconsistent`] found.
    pub fn verify(&self) -> Result<(), GrantError> {
        let mut granted = 0u32;
        let mut consumed = 0u32;
        for g in self.grants.values() {
            if let Some(limit) = g.uses_left {
                granted += limit;
            }
            if g.uses_used > g.uses_left.unwrap_or(u32::MAX) {
                return Err(GrantError::Inconsistent {
                    detail: format!(
                        "grant {} spent {} of {:?}",
                        g.id, g.uses_used, g.uses_left
                    ),
                });
            }
            if g.issued_at >= g.expires_at {
                return Err(GrantError::Inconsistent {
                    detail: format!("grant {} is born past its expiry", g.id),
                });
            }
            if let Some(parent) = g.parent {
                let Some(p) = self.grants.get(&parent) else {
                    return Err(GrantError::Inconsistent {
                        detail: format!(
                        "grant {} names a parent {} that is not here",
                        g.id, parent
                    ),
                    });
                };
                if g.capability != p.capability
                    || g.expires_at > p.expires_at
                    || !p.covers(&g.scope)
                {
                    return Err(GrantError::Inconsistent {
                        detail: format!("grant {} is wider than its parent {parent}", g.id),
                    });
                }
                if p.revoked_at.is_some() && g.revoked_at.is_none() {
                    return Err(GrantError::Inconsistent {
                        detail: format!(
                            "grant {} is live while its revoked parent {parent} is not",
                            g.id
                        ),
                    });
                }
            }
        }
        consumed += self
            .grants
            .values()
            .map(|g| g.uses_used)
            .sum::<u32>();
        if consumed != self.uses_consumed {
            return Err(GrantError::Inconsistent {
                detail: format!(
                    "records show {consumed} uses and the counter says {}",
                    self.uses_consumed
                ),
            });
        }
        if self.uses_granted < consumed {
            return Err(GrantError::Inconsistent {
                detail: format!(
                    "{} uses consumed against {} granted",
                    consumed, self.uses_granted
                ),
            });
        }
        let use_events = self.events.iter().filter(|e| e.kind == "use").count() as u32;
        if use_events != consumed {
            return Err(GrantError::Inconsistent {
                detail: format!(
                    "the trail has {use_events} use events and the records show {consumed}"
                ),
            });
        }
        if self.events.windows(2).any(|w| w[0].seq >= w[1].seq) {
            return Err(GrantError::Inconsistent {
                detail: "the trail is not strictly ordered".to_string(),
            });
        }
        Ok(())
    }

    fn push_event(&mut self, at: u64, kind: &'static str, grant: u64, note: String) {
        let seq = self.events.len() as u64 + 1;
        self.events.push(Event {
            seq,
            at,
            kind,
            grant,
            note,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> (Ledger, u64) {
        let mut l = Ledger::new();
        let id = l
            .issue(10, "agent", "read", "src/storage", 100, Some(2))
            .expect("issued");
        (l, id)
    }

    #[test]
    fn an_empty_scope_is_a_wildcard_and_is_refused() {
        let mut l = Ledger::new();
        assert_eq!(
            l.issue(0, "agent", "read", "", 10, None),
            Err(GrantError::EmptyScope)
        );
    }

    #[test]
    fn zero_limits_are_refused_at_issue_not_at_use() {
        let mut l = Ledger::new();
        assert_eq!(
            l.issue(0, "agent", "read", "src", 0, None),
            Err(GrantError::ZeroTtl { ttl: 0 })
        );
        assert_eq!(
            l.issue(0, "agent", "read", "src", 5, Some(0)),
            Err(GrantError::ZeroUses)
        );
        assert_eq!(
            l.issue(0, "", "read", "src", 5, None),
            Err(GrantError::EmptyHolder)
        );
        assert_eq!(
            l.issue(0, "agent", "", "src", 5, None),
            Err(GrantError::EmptyCapability)
        );
    }

    #[test]
    fn the_separator_rule_is_the_whole_point() {
        let (l, id) = ledger();
        let g = l.get(id).expect("grant");
        assert!(g.covers("src/storage"));
        assert!(g.covers("src/storage/deal.rs"));
        assert!(
            !g.covers("src/storagesecrets"),
            "a bare starts_with would say yes here, and that is the bug"
        );
        assert!(!g.covers("src/storage-deal"));
        assert!(!g.covers("src"));
    }

    #[test]
    fn expiry_is_inclusive_of_the_boundary_epoch() {
        let (mut l, id) = ledger();
        assert_eq!(l.consume(id, 109, "read", "src/storage/deal.rs").unwrap(), ());
        assert_eq!(
            l.consume(id, 110, "read", "src/storage/deal.rs"),
            Err(GrantError::Expired {
                grant: id,
                expires_at: 110,
                at: 110
            })
        );
    }

    #[test]
    fn the_use_count_is_enforced_and_the_limit_is_named() {
        let (mut l, id) = ledger();
        l.consume(id, 11, "read", "src/storage").unwrap();
        l.consume(id, 12, "read", "src/storage").unwrap();
        assert_eq!(
            l.consume(id, 13, "read", "src/storage"),
            Err(GrantError::UseLimitReached { grant: id, limit: 2 })
        );
    }

    #[test]
    fn no_translation_between_capability_strings() {
        let (mut l, id) = ledger();
        assert_eq!(
            l.consume(id, 11, "READ", "src/storage"),
            Err(GrantError::CapabilityMismatch {
                grant: id,
                want: "READ".to_string(),
                got: "read".to_string()
            })
        );
    }

    #[test]
    fn a_delegate_may_only_narrow() {
        let (mut l, id) = ledger();
        // narrower scope, shorter life, fewer uses: allowed
        let child = l
            .delegate(20, id, "helper", "read", "src/storage/deal.rs", 5, Some(1))
            .expect("a narrowing");
        assert_eq!(l.get(child).expect("child").parent(), Some(id));
        for (scope, dimension) in [("src", "scope"), ("src/storage/deal.rs/x", "scope")] {
            let err = l.delegate(20, id, "helper", "read", scope, 5, Some(1));
            if let Err(GrantError::DelegateWidens { dimension: got, .. }) = err {
                assert_eq!(got, dimension);
            } else {
                panic!("`{scope}` must not be delegable from `src/storage`");
            }
        }
        assert_eq!(
            l.delegate(20, id, "helper", "write", "src/storage", 5, Some(1)),
            Err(GrantError::DelegateWidens {
                parent: id,
                dimension: "capability"
            })
        );
        assert_eq!(
            l.delegate(20, id, "helper", "read", "src/storage", 1000, Some(1)),
            Err(GrantError::DelegateWidens {
                parent: id,
                dimension: "expiry"
            })
        );
        assert_eq!(
            l.delegate(20, id, "helper", "read", "src/storage", 5, Some(9)),
            Err(GrantError::DelegateWidens {
                parent: id,
                dimension: "use count"
            })
        );
    }

    #[test]
    fn revoking_a_parent_revokes_the_subtree() {
        let (mut l, id) = ledger();
        let child = l
            .delegate(20, id, "helper", "read", "src/storage/deal.rs", 5, Some(1))
            .unwrap();
        assert!(l.revoke(id, 30));
        assert!(!l.revoke(id, 31), "a double revoke reports itself");
        assert_eq!(
            l.consume(child, 32, "read", "src/storage/deal.rs"),
            Err(GrantError::Revoked {
                grant: child,
                at: 30
            }),
            "a narrower copy must not outlive what it was made from"
        );
    }

    #[test]
    fn a_spent_or_expired_parent_cannot_delegate() {
        let (mut l, id) = ledger();
        l.consume(id, 11, "read", "src/storage").unwrap();
        l.consume(id, 12, "read", "src/storage").unwrap();
        // The parent is spent but still live; the child limit check catches it.
        assert_eq!(
            l.delegate(13, id, "helper", "read", "src/storage", 1, Some(1)),
            Err(GrantError::DelegateWidens {
                parent: id,
                dimension: "use count"
            })
        );
        let err = l.delegate(200, id, "helper", "read", "src/storage", 1, None);
        assert_eq!(err, Err(GrantError::ParentUnusable(id)));
    }

    #[test]
    fn the_trail_must_match_the_records() {
        let (l, id) = ledger();
        assert_eq!(l.verify(), Ok(()));
        let mut broken = l.clone();
        broken.grants.get_mut(&id).expect("grant").uses_used = 5;
        assert!(matches!(
            broken.verify(),
            Err(GrantError::Inconsistent { .. })
        ));
    }

    #[test]
    fn a_widened_child_record_is_caught_after_the_fact() {
        let (l, id) = ledger();
        let child = l
            .delegate(20, id, "helper", "read", "src/storage/deal.rs", 5, Some(1))
            .unwrap();
        let mut broken = l.clone();
        broken.grants.get_mut(&child).expect("child").scope = "src".to_string();
        assert!(matches!(
            broken.verify(),
            Err(GrantError::Inconsistent { .. })
        ));
    }

    #[test]
    fn a_use_records_an_event_and_a_refusal_records_one_too() {
        let (mut l, id) = ledger();
        l.consume(id, 11, "read", "src/storage").unwrap();
        let _ = l.consume(id, 12, "read", "outside/path");
        let kinds: Vec<&str> = l.events().iter().map(Event::kind).collect();
        assert_eq!(kinds, vec!["issue", "use", "refuse"]);
        assert_eq!(l.events().len(), 3);
        assert_eq!(l.verify(), Ok(()));
    }

    #[test]
    fn unknown_grants_are_named_not_ignored() {
        let (mut l, _id) = ledger();
        assert_eq!(
            l.consume(9999, 11, "read", "src/storage"),
            Err(GrantError::UnknownGrant(9999))
        );
        assert!(!l.revoke(9999, 11));
        assert_eq!(
            l.delegate(11, 9999, "h", "read", "src", 1, None),
            Err(GrantError::UnknownGrant(9999))
        );
    }
}
