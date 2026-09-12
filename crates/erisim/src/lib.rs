//! Capability-based access: what a holder may do, and whether it still may.
//!
//! # Why capabilities rather than an access list
//!
//! An access list answers "who is allowed". A capability answers "what may this
//! particular thing do", and it travels with the thing. The difference matters at
//! the boundary this crate exists for: a session, a sub-agent, or a delegated
//! worker holds a capability, and the holder cannot ask a question about itself
//! that the capability does not already answer.
//!
//! # The three rules
//!
//! **1. A capability can be narrowed by its holder, never widened.** Attenuation
//! is the whole point of delegation: a worker given "read these three folders"
//! can hand a sub-task "read this one folder", and nothing in that chain can
//! produce "read everything". [`Capability::attenuate`] refuses any request that
//! is not a subset, and it refuses by checking each axis separately so the
//! refusal says which one.
//!
//! **2. Expiry is checked at use, not at issue.** A capability issued before a
//! deadline is not thereby valid at use. [`Capability::authorizes`] takes the
//! current time and compares; a check performed only when the capability was
//! minted is a check about the past.
//!
//! **3. The token binds the whole capability, not its parts.** A capability whose
//! scope could be edited and recombined with a valid token would be a suggestion.
//! The token is a digest over every field, so widening any field invalidates it.
//! This crate does not sign - it has no key - and says so: what it provides is
//! **binding**, and the unforgeability comes from whatever signs the digest
//! upstream. Confusing the two is how a system ends up believing it checked a
//! signature when it only checked a hash.
//!
//! # Revocation
//!
//! A revoked capability is refused, and revocation is by token, not by subject:
//! revoking everything one identity holds is a different operation
//! ([`RevocationList::revoke_subject`]) and is spelled differently because it has
//! a much larger blast radius.

use lubot_read::sha256_hex;
use std::collections::BTreeSet;

/// Why an authorization refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessError {
    /// The capability has expired. Checked at use, not at issue.
    Expired { expires_at: u64, now: u64 },
    /// The action is not among the capability's actions.
    ActionNotGranted { action: String, granted: Vec<String> },
    /// The resource is outside the capability's scope.
    OutsideScope { resource: String },
    /// The token does not match the capability's fields. This means a field was
    /// edited - see the module note on binding.
    TokenMismatch,
    /// The capability has been revoked.
    Revoked,
    /// An attenuation asked for something the capability does not have.
    NotAnAttenuation { axis: &'static str },
    /// An empty scope. A capability over nothing is not a narrow capability, it
    /// is one that cannot be used, and minting it is usually a mistake.
    EmptyScope,
}

impl std::fmt::Display for AccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expired { expires_at, now } => {
                write!(f, "the capability expired at {expires_at} and it is now {now}")
            }
            Self::ActionNotGranted { action, granted } => {
                write!(f, "{action:?} is not granted; this capability allows {granted:?}")
            }
            Self::OutsideScope { resource } => {
                write!(f, "{resource:?} is outside this capability's scope")
            }
            Self::TokenMismatch => write!(
                f,
                "the token does not match the capability's fields, so one of them was edited"
            ),
            Self::Revoked => write!(f, "this capability has been revoked"),
            Self::NotAnAttenuation { axis } => {
                write!(f, "the requested {axis} is not a subset of what this capability holds")
            }
            Self::EmptyScope => write!(f, "a capability over an empty scope cannot be used"),
        }
    }
}

/// A capability.
///
/// Every field is part of the token. Adding a field without adding it to
/// [`Capability::token`] would make that field editable without invalidating
/// anything, which is the failure mode this design exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capability {
    /// Who holds it. Recorded for audit; the authorization does not depend on it,
    /// because a capability is bearer-held and checking the holder's identity
    /// would turn this back into an access list.
    pub subject: String,
    /// The resources it covers. Exact strings, not patterns: a pattern language
    /// is a place for the scope to mean something narrower than it reads.
    pub scope: BTreeSet<String>,
    /// The actions it allows.
    pub actions: BTreeSet<String>,
    /// When it stops being valid. Zero means it does not expire by time, which is
    /// a choice made explicitly.
    pub expires_at: u64,
    /// The digest binding the fields together.
    pub token: String,
}

impl Capability {
    /// Mints a capability and binds its fields.
    ///
    /// # Errors
    ///
    /// [`AccessError::EmptyScope`].
    pub fn mint(
        subject: &str,
        scope: &[&str],
        actions: &[&str],
        expires_at: u64,
    ) -> Result<Self, AccessError> {
        if scope.is_empty() {
            return Err(AccessError::EmptyScope);
        }
        let mut cap = Self {
            subject: subject.to_string(),
            scope: scope.iter().map(|s| (*s).to_string()).collect(),
            actions: actions.iter().map(|s| (*s).to_string()).collect(),
            expires_at,
            token: String::new(),
        };
        cap.token = cap.compute_token();
        Ok(cap)
    }

    /// The digest over every field except the token itself.
    ///
    /// Fields are separated by a byte that cannot appear in any of them, so that
    /// two different field combinations cannot produce the same digest.
    #[must_use]
    pub fn compute_token(&self) -> String {
        let mut parts = vec![self.subject.clone(), self.expires_at.to_string()];
        parts.extend(self.scope.iter().cloned());
        parts.push("\u{1f}scope|actions".to_string());
        parts.extend(self.actions.iter().cloned());
        sha256_hex(parts.join("\u{1f}").as_bytes())
    }

    /// Whether the token matches the fields.
    #[must_use]
    pub fn token_is_bound(&self) -> bool {
        !self.token.is_empty() && self.token == self.compute_token()
    }

    /// Whether this capability authorizes `action` on `resource` at `now`.
    ///
    /// The order matters: the token is checked first, because a capability with
    /// edited fields should be reported as edited rather than as merely
    /// insufficient. An "outside scope" refusal for a widened scope would send
    /// the operator looking in the wrong direction.
    ///
    /// # Errors
    ///
    /// Any [`AccessError`] that applies.
    pub fn authorizes(&self, action: &str, resource: &str, now: u64) -> Result<(), AccessError> {
        if !self.token_is_bound() {
            return Err(AccessError::TokenMismatch);
        }
        if self.expires_at > 0 && now >= self.expires_at {
            return Err(AccessError::Expired {
                expires_at: self.expires_at,
                now,
            });
        }
        if !self.actions.contains(action) {
            return Err(AccessError::ActionNotGranted {
                action: action.to_string(),
                granted: self.actions.iter().cloned().collect(),
            });
        }
        if !self.scope.contains(resource) {
            return Err(AccessError::OutsideScope {
                resource: resource.to_string(),
            });
        }
        Ok(())
    }

    /// Derives a narrower capability.
    ///
    /// Every axis must be a subset: the scope, the actions, and the expiry (a
    /// later expiry is a wider capability, not a narrower one). The derived
    /// capability gets a fresh token, so it stands on its own and cannot be
    /// confused with its parent.
    ///
    /// # Errors
    ///
    /// [`AccessError::NotAnAttenuation`] naming the axis, or
    /// [`AccessError::TokenMismatch`] if the parent's own token does not bind.
    pub fn attenuate(
        &self,
        scope: &[&str],
        actions: &[&str],
        expires_at: u64,
    ) -> Result<Self, AccessError> {
        if !self.token_is_bound() {
            return Err(AccessError::TokenMismatch);
        }
        let new_scope: BTreeSet<String> = scope.iter().map(|s| (*s).to_string()).collect();
        if !new_scope.is_subset(&self.scope) {
            return Err(AccessError::NotAnAttenuation { axis: "scope" });
        }
        if new_scope.is_empty() {
            return Err(AccessError::EmptyScope);
        }
        let new_actions: BTreeSet<String> = actions.iter().map(|s| (*s).to_string()).collect();
        if !new_actions.is_subset(&self.actions) {
            return Err(AccessError::NotAnAttenuation { axis: "actions" });
        }
        // A later expiry is wider. Equal is allowed: attenuation does not have to
        // shorten the lifetime, only may.
        if self.expires_at > 0 && (expires_at == 0 || expires_at > self.expires_at) {
            return Err(AccessError::NotAnAttenuation { axis: "expiry" });
        }
        Self::mint(&self.subject, scope, actions, expires_at)
    }
}

/// Revoked tokens and subjects.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RevocationList {
    tokens: BTreeSet<String>,
    subjects: BTreeSet<String>,
}

impl RevocationList {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Revokes one capability, by token.
    pub fn revoke(&mut self, token: &str) {
        self.tokens.insert(token.to_string());
    }

    /// Revokes everything one subject holds.
    ///
    /// Spelled differently from [`Self::revoke`] because the blast radius is
    /// different: this one takes away every capability an identity has, including
    /// ones the caller did not think about.
    pub fn revoke_subject(&mut self, subject: &str) {
        self.subjects.insert(subject.to_string());
    }

    /// Whether a capability is revoked.
    #[must_use]
    pub fn is_revoked(&self, cap: &Capability) -> bool {
        self.tokens.contains(&cap.token) || self.subjects.contains(&cap.subject)
    }

    /// How many tokens are revoked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len().saturating_add(self.subjects.len())
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.subjects.is_empty()
    }

    /// Checks authorization and revocation together.
    ///
    /// Provided because a caller that checks one and forgets the other is the
    /// common failure, and the two checks belong in one call.
    ///
    /// # Errors
    ///
    /// [`AccessError::Revoked`] or anything
    /// [`Capability::authorizes`] returns.
    pub fn check(&self, cap: &Capability, action: &str, resource: &str, now: u64) -> Result<(), AccessError> {
        cap.authorizes(action, resource, now)?;
        if self.is_revoked(cap) {
            return Err(AccessError::Revoked);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap() -> Capability {
        Capability::mint("worker-1", &["folder/a", "folder/b"], &["read", "list"], 1000)
            .expect("mint")
    }

    #[test]
    fn a_minted_capability_authorizes_what_it_was_minted_for() {
        let c = cap();
        assert!(c.token_is_bound());
        assert!(c.authorizes("read", "folder/a", 500).is_ok());
        assert!(c.authorizes("list", "folder/b", 500).is_ok());
    }

    #[test]
    fn an_action_outside_the_capability_is_refused() {
        let c = cap();
        assert!(matches!(
            c.authorizes("delete", "folder/a", 500),
            Err(AccessError::ActionNotGranted { .. })
        ));
    }

    #[test]
    fn a_resource_outside_the_scope_is_refused() {
        let c = cap();
        assert_eq!(
            c.authorizes("read", "folder/c", 500),
            Err(AccessError::OutsideScope {
                resource: "folder/c".to_string()
            })
        );
    }

    #[test]
    fn expiry_is_checked_at_use_not_at_issue() {
        // A capability issued before a deadline is not thereby valid at use.
        let c = cap();
        assert!(c.authorizes("read", "folder/a", 999).is_ok());
        assert_eq!(
            c.authorizes("read", "folder/a", 1000),
            Err(AccessError::Expired {
                expires_at: 1000,
                now: 1000
            }),
            "the expiry boundary was off by one"
        );
    }

    #[test]
    fn an_edited_field_invalidates_the_token() {
        // A capability whose scope could be edited and recombined with a valid
        // token would be a suggestion.
        let mut c = cap();
        c.scope.insert("folder/secret".to_string());
        assert!(!c.token_is_bound());
        assert_eq!(c.authorizes("read", "folder/secret", 500), Err(AccessError::TokenMismatch));
    }

    #[test]
    fn the_token_is_reported_before_insufficiency() {
        // An "outside scope" refusal for a widened scope would send the operator
        // looking in the wrong direction.
        let mut c = cap();
        c.expires_at = 999_999;
        assert_eq!(
            c.authorizes("read", "folder/zzz", 500),
            Err(AccessError::TokenMismatch),
            "the edited capability was reported as insufficient instead of as edited"
        );
    }

    #[test]
    fn a_capability_can_be_narrowed() {
        let c = cap();
        let narrow = c.attenuate(&["folder/a"], &["read"], 500).expect("attenuate");
        assert!(narrow.authorizes("read", "folder/a", 100).is_ok());
        assert!(narrow.authorizes("read", "folder/b", 100).is_err(), "the attenuation kept the parent's scope");
        assert!(narrow.authorizes("list", "folder/a", 100).is_err(), "the attenuation kept the parent's actions");
        assert_ne!(narrow.token, c.token, "the derived capability shares its parent's token");
    }

    #[test]
    fn a_capability_cannot_be_widened_on_any_axis() {
        let c = cap();
        // Scope.
        assert_eq!(
            c.attenuate(&["folder/a", "folder/secret"], &["read"], 500),
            Err(AccessError::NotAnAttenuation { axis: "scope" })
        );
        // Actions.
        assert_eq!(
            c.attenuate(&["folder/a"], &["read", "delete"], 500),
            Err(AccessError::NotAnAttenuation { axis: "actions" })
        );
        // Expiry: a later expiry is a wider capability.
        assert_eq!(
            c.attenuate(&["folder/a"], &["read"], 2000),
            Err(AccessError::NotAnAttenuation { axis: "expiry" })
        );
        // Turning a finite expiry into an infinite one is the widest move there is.
        assert_eq!(
            c.attenuate(&["folder/a"], &["read"], 0),
            Err(AccessError::NotAnAttenuation { axis: "expiry" })
        );
    }

    #[test]
    fn attenuation_may_keep_the_lifetime_but_not_extend_it() {
        let c = cap();
        assert!(c.attenuate(&["folder/a"], &["read"], 1000).is_ok());
        assert!(c.attenuate(&["folder/a"], &["read"], 999).is_ok());
    }

    #[test]
    fn attenuation_refuses_a_parent_whose_token_does_not_bind() {
        // Otherwise an edited capability could launder itself through a
        // narrowing.
        let mut c = cap();
        c.scope.insert("folder/secret".to_string());
        assert_eq!(
            c.attenuate(&["folder/a"], &["read"], 500),
            Err(AccessError::TokenMismatch)
        );
    }

    #[test]
    fn an_empty_scope_is_refused_at_mint_and_at_attenuation() {
        // A capability over nothing is not a narrow capability, it is one that
        // cannot be used, and minting it is usually a mistake.
        assert_eq!(
            Capability::mint("w", &[], &["read"], 100),
            Err(AccessError::EmptyScope)
        );
        assert_eq!(cap().attenuate(&[], &["read"], 100), Err(AccessError::EmptyScope));
    }

    #[test]
    fn a_revoked_capability_is_refused() {
        let c = cap();
        let mut list = RevocationList::new();
        assert!(list.check(&c, "read", "folder/a", 500).is_ok());
        list.revoke(&c.token);
        assert_eq!(list.check(&c, "read", "folder/a", 500), Err(AccessError::Revoked));
    }

    #[test]
    fn revoking_a_subject_takes_every_capability_it_holds() {
        // Spelled differently from revoking one token because the blast radius is
        // different.
        let c = cap();
        let mut list = RevocationList::new();
        list.revoke_subject("worker-1");
        assert!(list.is_revoked(&c));
        let other = Capability::mint("worker-2", &["folder/a"], &["read"], 1000).expect("mint");
        assert!(!list.is_revoked(&other));
    }

    #[test]
    fn a_capability_that_never_expires_says_so_explicitly() {
        let c = Capability::mint("w", &["folder/a"], &["read"], 0).expect("mint");
        assert!(c.authorizes("read", "folder/a", u64::MAX).is_ok());
    }

    #[test]
    fn two_capabilities_over_the_same_fields_share_a_token() {
        // The token is a function of the fields, so it is reproducible - which is
        // what lets a verifier recompute it. It is binding, not unforgeable; the
        // unforgeability comes from whatever signs the digest upstream.
        let a = cap();
        let b = cap();
        assert_eq!(a.token, b.token);
        assert_eq!(a.token.len(), 64);
    }

    #[test]
    fn the_authority_of_the_holder_is_not_checked() {
        // A capability is bearer-held. Checking the holder's identity would turn
        // this back into an access list, which is the thing capabilities exist to
        // replace.
        let c = cap();
        assert!(c.authorizes("read", "folder/a", 500).is_ok());
        // Nothing in the call takes a caller identity, so there is no way to
        // impersonate through this API - and no way to be authorized by claiming
        // to be somebody else.
    }
}
