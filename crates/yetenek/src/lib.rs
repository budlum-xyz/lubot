//! A capability registry where "declared" and "works" are different states.
//!
//! # The rule that makes this more than a name list
//!
//! A capability that is registered but has never been exercised is a claim, not a
//! capability. A registry that hands out claims will route work to something that
//! fails on first use, and the failure surfaces in the caller's transaction
//! rather than at admission. [`Registry::register`] therefore requires the
//! declaration to carry a [`SelfTest`], and a capability only reaches
//! [`State::Ready`] after that test has passed.
//!
//! The self-test is a closure that runs against the capability's own
//! implementation, not against a description of it. A test that only checks the
//! declaration would pass for a capability whose implementation is broken, which
//! is the failure this exists to prevent.
//!
//! # Versions are exact
//!
//! [`Registry::acquire`] takes a version and returns that version or a refusal.
//! It never substitutes an older one: a caller written against version 2 does not
//! want version 1's behaviour with version 2's name. The refusal says which
//! versions *are* available, so the caller can decide rather than discover.
//!
//! # Degraded is a state, not an error
//!
//! A capability that works at reduced capacity is not unavailable, and treating
//! it as unavailable turns a partial outage into a total one. [`State::Degraded`]
//! is reported to the caller along with what is missing, and
//! [`Registry::acquire`] returns it rather than refusing. Whether degraded is
//! acceptable is the caller's decision; hiding it is not.
//!
//! # Revocation
//!
//! A revoked capability is refused at acquire time with
//! [`CapabilityError::Revoked`] and its reason carried through. It is not
//! removed from the registry, because a caller who saw it advertised an hour ago
//! needs to be told it was withdrawn rather than that it never existed.

use std::collections::BTreeMap;

/// The result of running a capability's self-test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelfTest {
    /// The test passed.
    Passed,
    /// The test failed, with what it observed.
    Failed { observed: String },
    /// The test could not run. Distinct from failure: a capability whose test
    /// errored is unverified, and reporting it as failed would assert something
    /// about its behaviour that was never observed.
    CouldNotRun { reason: String },
}

impl SelfTest {
    /// Whether the capability is verified by this result.
    #[must_use]
    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }
}

/// Where a capability stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Declared, self-test not yet run or not passed.
    Declared,
    /// Self-test passed; usable.
    Ready,
    /// Usable at reduced capacity. What is missing is in
    /// [`Capability::degradation`].
    Degraded,
    /// Withdrawn. Kept in the registry so a caller who saw it advertised is told
    /// it was withdrawn rather than that it never existed.
    Revoked,
}

impl State {
    /// A stable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Ready => "ready",
            Self::Degraded => "degraded",
            Self::Revoked => "revoked",
        }
    }
}

/// A declared capability.
#[derive(Debug, Clone)]
pub struct Declaration {
    /// The capability's name.
    pub name: String,
    /// The version this declaration implements.
    pub version: u32,
    /// The self-test. A closure rather than a stored result, because the result
    /// goes stale the moment the implementation changes.
    pub self_test: fn() -> SelfTest,
}

/// Why the registry refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// This exact name and version is already registered.
    AlreadyRegistered { name: String, version: u32 },
    /// No capability with this name.
    Unknown { name: String },
    /// The name exists but not at this version. `available` says which do.
    WrongVersion {
        name: String,
        requested: u32,
        available: Vec<u32>,
    },
    /// The capability was withdrawn.
    Revoked { name: String, reason: String },
    /// The self-test has not passed, so the capability is a claim rather than a
    /// capability.
    Unverified {
        name: String,
        version: u32,
        state: State,
    },
}

impl std::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRegistered { name, version } => {
                write!(f, "{name} version {version} is already registered")
            }
            Self::Unknown { name } => write!(f, "there is no capability named {name:?}"),
            Self::WrongVersion {
                name,
                requested,
                available,
            } => write!(
                f,
                "{name} has no version {requested}; the available versions are {available:?}"
            ),
            Self::Revoked { name, reason } => {
                write!(f, "{name} was withdrawn: {reason}")
            }
            Self::Unverified {
                name,
                version,
                state,
            } => write!(
                f,
                "{name} version {version} is {} and its self-test has not passed",
                state.label()
            ),
        }
    }
}

/// A registered capability.
#[derive(Debug, Clone)]
pub struct Capability {
    pub declaration: Declaration,
    pub state: State,
    /// What the last self-test observed.
    pub last_test: SelfTest,
    /// Present when [`State::Degraded`]: what is missing.
    pub degradation: String,
    /// Why it was revoked, when it was.
    pub revocation_reason: String,
}

/// What a caller gets back.
///
/// Owned rather than borrowed: `acquire` also updates the registry's refusal
/// counters, and a handle that borrowed the registry would tie the caller to
/// that borrow for as long as it held the handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handle {
    pub name: String,
    pub version: u32,
    pub state: State,
    /// Empty unless degraded.
    pub degradation: String,
}

/// The registry.
#[derive(Debug, Clone)]
pub struct Registry {
    // Keyed by (name, version) so two versions of one capability coexist, which
    // is what makes an exact-version request meaningful.
    entries: BTreeMap<(String, u32), Capability>,
    /// Acquisitions that were refused. Counted by kind so a caller asking for a
    /// version that does not exist is visible as a pattern rather than as noise.
    pub refused_wrong_version: u64,
    pub refused_unverified: u64,
    pub refused_revoked: u64,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            refused_wrong_version: 0,
            refused_unverified: 0,
            refused_revoked: 0,
        }
    }

    /// Registers a declaration in the [`State::Declared`] state.
    ///
    /// Registration does not make a capability usable. That is deliberate: a
    /// registry that handed out unexercised capabilities would route work to
    /// something that fails on first use, inside the caller's transaction.
    ///
    /// # Errors
    ///
    /// [`CapabilityError::AlreadyRegistered`].
    pub fn register(&mut self, declaration: Declaration) -> Result<(), CapabilityError> {
        let key = (declaration.name.clone(), declaration.version);
        if self.entries.contains_key(&key) {
            return Err(CapabilityError::AlreadyRegistered {
                name: key.0,
                version: key.1,
            });
        }
        self.entries.insert(
            key,
            Capability {
                declaration,
                state: State::Declared,
                last_test: SelfTest::CouldNotRun {
                    reason: "the self-test has not been run yet".to_string(),
                },
                degradation: String::new(),
                revocation_reason: String::new(),
            },
        );
        Ok(())
    }

    /// Runs the self-test and records what it observed.
    ///
    /// Passing moves the capability to [`State::Ready`]. Failing leaves it
    /// [`State::Declared`]: an unverified capability is not a broken one, and
    /// calling it broken would assert something about its behaviour that was
    /// never observed.
    ///
    /// # Errors
    ///
    /// [`CapabilityError::Unknown`] or [`CapabilityError::WrongVersion`].
    pub fn verify(&mut self, name: &str, version: u32) -> Result<SelfTest, CapabilityError> {
        let Some(capability) = self.entries.get_mut(&(name.to_string(), version)) else {
            return Err(self.version_error(name, version));
        };
        let result = (capability.declaration.self_test)();
        capability.last_test = result.clone();
        if result.is_passed() {
            // A revoked capability does not come back because its test passed.
            if capability.state != State::Revoked {
                capability.state = State::Ready;
            }
        }
        Ok(result)
    }

    /// Marks a capability degraded, with what is missing.
    ///
    /// Degraded is reported to callers rather than hidden: a capability that
    /// works at reduced capacity is not unavailable, and treating it as
    /// unavailable turns a partial outage into a total one.
    ///
    /// # Errors
    ///
    /// [`CapabilityError::Unknown`] or [`CapabilityError::WrongVersion`].
    pub fn mark_degraded(
        &mut self,
        name: &str,
        version: u32,
        missing: &str,
    ) -> Result<(), CapabilityError> {
        let Some(capability) = self.entries.get_mut(&(name.to_string(), version)) else {
            return Err(self.version_error(name, version));
        };
        if capability.state == State::Revoked {
            return Ok(());
        }
        capability.state = State::Degraded;
        capability.degradation = missing.to_string();
        Ok(())
    }

    /// Withdraws a capability.
    ///
    /// The entry stays. A caller who saw it advertised needs to be told it was
    /// withdrawn, not that it never existed.
    ///
    /// # Errors
    ///
    /// [`CapabilityError::Unknown`] or [`CapabilityError::WrongVersion`].
    pub fn revoke(
        &mut self,
        name: &str,
        version: u32,
        reason: &str,
    ) -> Result<(), CapabilityError> {
        let Some(capability) = self.entries.get_mut(&(name.to_string(), version)) else {
            return Err(self.version_error(name, version));
        };
        capability.state = State::Revoked;
        capability.revocation_reason = reason.to_string();
        Ok(())
    }

    /// Acquires a capability at an exact version.
    ///
    /// Never substitutes another version: a caller written against version 2 does
    /// not want version 1's behaviour under version 2's name.
    ///
    /// # Errors
    ///
    /// Any [`CapabilityError`] that applies.
    pub fn acquire(&mut self, name: &str, version: u32) -> Result<Handle, CapabilityError> {
        let key = (name.to_string(), version);
        let Some(found) = self.entries.get(&key).cloned() else {
            let error = self.version_error(name, version);
            if matches!(error, CapabilityError::WrongVersion { .. }) {
                self.refused_wrong_version = self.refused_wrong_version.saturating_add(1);
            }
            return Err(error);
        };
        match found.state {
            State::Revoked => {
                self.refused_revoked = self.refused_revoked.saturating_add(1);
                return Err(CapabilityError::Revoked {
                    name: name.to_string(),
                    reason: found.revocation_reason,
                });
            }
            State::Ready | State::Degraded => {}
            State::Declared => {
                self.refused_unverified = self.refused_unverified.saturating_add(1);
                return Err(CapabilityError::Unverified {
                    name: name.to_string(),
                    version,
                    state: State::Declared,
                });
            }
        }
        Ok(Handle {
            name: found.declaration.name,
            version: found.declaration.version,
            state: found.state,
            degradation: found.degradation,
        })
    }

    /// Reads a capability without acquiring it.
    #[must_use]
    pub fn get(&self, name: &str, version: u32) -> Option<&Capability> {
        self.entries.get(&(name.to_string(), version))
    }

    /// The versions registered under a name, in order.
    #[must_use]
    pub fn versions(&self, name: &str) -> Vec<u32> {
        self.entries
            .keys()
            .filter(|(n, _)| n == name)
            .map(|(_, v)| *v)
            .collect()
    }

    /// How many capabilities are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Builds the right error for a missing name or version.
    fn version_error(&self, name: &str, version: u32) -> CapabilityError {
        let available = self.versions(name);
        if available.is_empty() {
            return CapabilityError::Unknown {
                name: name.to_string(),
            };
        }
        CapabilityError::WrongVersion {
            name: name.to_string(),
            requested: version,
            available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passing() -> SelfTest {
        SelfTest::Passed
    }

    fn failing() -> SelfTest {
        SelfTest::Failed {
            observed: "the implementation returned nothing".to_string(),
        }
    }

    fn cannot_run() -> SelfTest {
        SelfTest::CouldNotRun {
            reason: "no implementation is wired up".to_string(),
        }
    }

    fn declare(name: &str, version: u32, test: fn() -> SelfTest) -> Declaration {
        Declaration {
            name: name.to_string(),
            version,
            self_test: test,
        }
    }

    fn registry_with_passing() -> Registry {
        let mut r = Registry::new();
        r.register(declare("summarise", 1, passing))
            .expect("register");
        r
    }

    #[test]
    fn a_registered_capability_is_not_yet_usable() {
        // A capability that is registered but never exercised is a claim. A
        // registry that hands out claims routes work into a caller's transaction
        // and fails there.
        let mut r = registry_with_passing();
        assert_eq!(
            r.get("summarise", 1).map(|c| c.state),
            Some(State::Declared)
        );
        assert_eq!(
            r.acquire("summarise", 1).map(|_| ()),
            Err(CapabilityError::Unverified {
                name: "summarise".to_string(),
                version: 1,
                state: State::Declared,
            })
        );
        assert_eq!(r.refused_unverified, 1);
    }

    #[test]
    fn a_capability_becomes_usable_after_its_self_test_passes() {
        let mut r = registry_with_passing();
        assert_eq!(r.verify("summarise", 1), Ok(SelfTest::Passed));
        assert_eq!(r.get("summarise", 1).map(|c| c.state), Some(State::Ready));
        let handle = r.acquire("summarise", 1).expect("acquire");
        assert_eq!(handle.state, State::Ready);
        assert!(handle.degradation.is_empty());
    }

    #[test]
    fn a_failed_self_test_leaves_the_capability_declared_not_broken() {
        // An unverified capability is not a broken one. Calling it broken would
        // assert something about its behaviour that was never observed.
        let mut r = Registry::new();
        r.register(declare("summarise", 1, failing))
            .expect("register");
        let result = r.verify("summarise", 1).expect("verify");
        assert!(matches!(result, SelfTest::Failed { .. }));
        assert_eq!(
            r.get("summarise", 1).map(|c| c.state),
            Some(State::Declared)
        );
        assert!(r.acquire("summarise", 1).is_err());
    }

    #[test]
    fn a_test_that_could_not_run_is_not_a_failure() {
        let mut r = Registry::new();
        r.register(declare("summarise", 1, cannot_run))
            .expect("register");
        assert!(matches!(
            r.verify("summarise", 1),
            Ok(SelfTest::CouldNotRun { .. })
        ));
        assert_eq!(
            r.get("summarise", 1).map(|c| c.state),
            Some(State::Declared)
        );
    }

    #[test]
    fn a_wrong_version_is_refused_and_never_substituted() {
        // A caller written against version 2 does not want version 1's behaviour
        // under version 2's name.
        let mut r = registry_with_passing();
        r.register(declare("summarise", 2, passing))
            .expect("register");
        r.verify("summarise", 1).expect("verify 1");
        r.verify("summarise", 2).expect("verify 2");
        assert_eq!(
            r.acquire("summarise", 3).map(|_| ()),
            Err(CapabilityError::WrongVersion {
                name: "summarise".to_string(),
                requested: 3,
                available: vec![1, 2],
            })
        );
        assert_eq!(r.refused_wrong_version, 1);
    }

    #[test]
    fn two_versions_of_one_capability_coexist() {
        let mut r = registry_with_passing();
        r.register(declare("summarise", 2, passing))
            .expect("register");
        r.verify("summarise", 1).expect("verify");
        // Version 2 is still declared, so only version 1 is acquirable. This is
        // what makes the exact-version rule meaningful.
        assert!(r.acquire("summarise", 1).is_ok());
        assert!(r.acquire("summarise", 2).is_err());
        assert_eq!(r.versions("summarise"), vec![1, 2]);
    }

    #[test]
    fn an_unknown_name_says_so_rather_than_listing_versions() {
        let mut r = registry_with_passing();
        assert_eq!(
            r.acquire("translate", 1).map(|_| ()),
            Err(CapabilityError::Unknown {
                name: "translate".to_string()
            })
        );
        assert_eq!(
            r.refused_wrong_version, 0,
            "an unknown name is not a version error"
        );
    }

    #[test]
    fn a_duplicate_registration_is_refused() {
        let mut r = registry_with_passing();
        assert_eq!(
            r.register(declare("summarise", 1, passing)),
            Err(CapabilityError::AlreadyRegistered {
                name: "summarise".to_string(),
                version: 1,
            })
        );
    }

    #[test]
    fn degradation_is_reported_to_the_caller_not_hidden() {
        // A capability at reduced capacity is not unavailable. Treating it as
        // unavailable turns a partial outage into a total one.
        let mut r = registry_with_passing();
        r.verify("summarise", 1).expect("verify");
        r.mark_degraded("summarise", 1, "the model endpoint is rate limited")
            .expect("degrade");
        let handle = r.acquire("summarise", 1).expect("acquire");
        assert_eq!(handle.state, State::Degraded);
        assert_eq!(handle.degradation, "the model endpoint is rate limited");
    }

    #[test]
    fn a_revoked_capability_is_refused_with_its_reason() {
        // The entry stays: a caller who saw it advertised needs to be told it was
        // withdrawn, not that it never existed.
        let mut r = registry_with_passing();
        r.verify("summarise", 1).expect("verify");
        r.revoke("summarise", 1, "the backing service was decommissioned")
            .expect("revoke");
        assert_eq!(
            r.acquire("summarise", 1).map(|_| ()),
            Err(CapabilityError::Revoked {
                name: "summarise".to_string(),
                reason: "the backing service was decommissioned".to_string(),
            })
        );
        assert_eq!(r.refused_revoked, 1);
        assert!(r.get("summarise", 1).is_some());
    }

    #[test]
    fn a_passing_test_does_not_unrevoke_a_capability() {
        // Revocation is a decision about the capability's standing, not a
        // statement about whether its code runs.
        let mut r = registry_with_passing();
        r.verify("summarise", 1).expect("verify");
        r.revoke("summarise", 1, "withdrawn").expect("revoke");
        r.verify("summarise", 1).expect("verify again");
        assert_eq!(r.get("summarise", 1).map(|c| c.state), Some(State::Revoked));
        assert!(r.acquire("summarise", 1).is_err());
    }

    #[test]
    fn verifying_an_unregistered_version_reports_the_available_ones() {
        let mut r = registry_with_passing();
        assert_eq!(
            r.verify("summarise", 9).map(|_| ()),
            Err(CapabilityError::WrongVersion {
                name: "summarise".to_string(),
                requested: 9,
                available: vec![1],
            })
        );
    }

    #[test]
    fn the_last_test_result_is_kept() {
        let mut r = registry_with_passing();
        assert!(matches!(
            r.get("summarise", 1).map(|c| c.last_test.clone()),
            Some(SelfTest::CouldNotRun { .. })
        ));
        r.verify("summarise", 1).expect("verify");
        assert_eq!(
            r.get("summarise", 1).map(|c| c.last_test.clone()),
            Some(SelfTest::Passed)
        );
    }

    #[test]
    fn an_empty_registry_has_no_capabilities() {
        let r = Registry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.versions("anything").is_empty());
    }
}
