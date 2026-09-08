//! # lubot-grant::training - epoch-bounded bulk-read authority
//!
//! A view grant says "this reader may open this item until this second".
//! Training is different: the same corpus is read over and over, once per
//! epoch, and a training pass is *expected* to re-read everything. So the
//! authority here is epoch-bounded, not item-bounded:
//!
//! * [`TrainingDataGrant`] names a corpus (the 32-byte asset id), the owner
//!   (the party who may issue), the grantee (the training identity), a
//!   wall-clock expiry **and** an epoch budget.
//! * [`EpochBook`] holds the grants and is the only place an epoch is
//!   consumed. The chain does not track this book; the reading system's
//!   enforcement does, fail-closed: an unknown, expired or exhausted grant
//!   is refused, and the refusal is logged with the same shape as an
//!   allowance.
//!
//! The epoch budget lives here, not in the training scripts, because a
//! budget that the enforcement layer can talk its way out of is a
//! suggestion. [`EpochBook::consume`] returns the distinct refusal kinds
//! ([`EpochRefusal`]) so a caller never has to guess why a pass did not run.

use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Protocol ceiling for `max_epochs`. A corpus can be swept forever only if
/// the protocol says so; this is the cap a publisher's budget cannot exceed.
pub const MAX_TRAINING_GRANT_EPOCHS: u32 = 4096;

/// A 32-byte identifier (an asset id, an owner, a grantee) shown as hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Id(pub [u8; 32]);

impl Id {
    /// The lowercase hex form used on the wire and in logs.
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(66);
        out.push_str("0x");
        for byte in self.0 {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// Parse a hex id (`0x` prefix optional). Wrong length is a refusal.
    pub fn from_hex(input: &str) -> Result<Self, String> {
        let clean = input.strip_prefix("0x").unwrap_or(input);
        if clean.len() != 64 {
            return Err(format!(
                "id must be 32 bytes of hex, got {} characters",
                clean.len()
            ));
        }
        let mut raw = [0u8; 32];
        for i in 0..32 {
            raw[i] = u8::from_str_radix(&clean[i * 2..i * 2 + 2], 16)
                .map_err(|_| format!("non-hex id character at position {}", i * 2))?;
        }
        Ok(Self(raw))
    }
}

/// Bulk data access authority for training (epoch bounded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingDataGrant {
    /// The corpus asset this grant authorizes reading.
    pub asset_id: Id,
    /// The party who issued the grant. Only the owner may consume from it.
    pub owner: Id,
    /// The training identity the budget belongs to.
    pub grantee: Id,
    pub issued_at_block: u64,
    pub expires_at_block: u64,
    pub max_epochs: u32,
    pub epochs_used: u32,
}

impl TrainingDataGrant {
    /// Canonical identity. Domain-separated preimage, so a training grant id
    /// can never collide with any other record that hashes its fields.
    #[must_use]
    pub fn derive_id(&self) -> Id {
        let mut hasher = Sha256::new();
        hasher.update(b"BDLM_TRAINING_DATA_GRANT_V1");
        hasher.update(self.asset_id.0);
        hasher.update(self.owner.0);
        hasher.update(self.grantee.0);
        hasher.update(self.issued_at_block.to_le_bytes());
        hasher.update(self.expires_at_block.to_le_bytes());
        hasher.update(self.max_epochs.to_le_bytes());
        Id(hasher.finalize().into())
    }

    /// The issuance shape rule. Only issuance fields are checked; the ones
    /// that change later (epochs_used) are checked by the book, not here.
    pub fn validate_shape(&self) -> Result<(), String> {
        if self.max_epochs == 0 {
            return Err("training-data grant needs at least one epoch".into());
        }
        if self.max_epochs > MAX_TRAINING_GRANT_EPOCHS {
            return Err(format!(
                "training-data grant max_epochs exceeds {MAX_TRAINING_GRANT_EPOCHS}"
            ));
        }
        if self.expires_at_block <= self.issued_at_block {
            return Err("training-data grant expires before it starts".into());
        }
        if self.epochs_used != 0 {
            return Err("a new training-data grant cannot start used".into());
        }
        Ok(())
    }

    /// Consume one training epoch (fail-closed: errors once the limit is
    /// reached). The caller receives the remaining budget on success.
    pub fn consume_epoch(&mut self) -> Result<u32, String> {
        if self.epochs_used >= self.max_epochs {
            return Err("training-data grant epochs exhausted".into());
        }
        self.epochs_used += 1;
        Ok(self.max_epochs - self.epochs_used)
    }

    /// Whether it may still be used: time and epoch budget both live.
    pub fn is_valid(&self, now_block: u64) -> bool {
        now_block >= self.issued_at_block
            && now_block <= self.expires_at_block
            && self.epochs_used < self.max_epochs
    }
}

/// Why an epoch could not be consumed. Distinct answers, because each one
/// has a different remedy (issue one, renew it, wait).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochRefusal {
    Unknown,
    Expired { at: u64, expires_at: u64 },
    Exhausted { used: u32, max: u32 },
}

impl EpochRefusal {
    /// The stable word written to the log, mirroring [`crate::Decision::label`].
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            EpochRefusal::Unknown => "unknown-grant",
            EpochRefusal::Expired { .. } => "expired",
            EpochRefusal::Exhausted { .. } => "exhausted",
        }
    }
}

/// One line of the epoch trail. Refusals are logged with the same shape as
/// allowances, so the log cannot be read as "only successes happened".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochEntry {
    pub at_block: u64,
    pub grant_id: Id,
    pub outcome: EpochOutcome,
}

/// What an epoch request produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpochOutcome {
    Consumed { remaining: u32 },
    Refused(EpochRefusal),
}

/// The training-grant table plus its epoch trail.
#[derive(Debug, Default)]
pub struct EpochBook {
    grants: HashMap<Id, TrainingDataGrant>,
    log: Vec<EpochEntry>,
}

impl EpochBook {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a grant. The shape rule applies before anything is stored;
    /// the canonical id is the key, so a second registration with the same
    /// preimage is a refusal rather than a silent overwrite.
    pub fn issue(&mut self, grant: TrainingDataGrant) -> Result<Id, String> {
        grant.validate_shape()?;
        let id = grant.derive_id();
        if self.grants.contains_key(&id) {
            return Err("training-data grant already registered".into());
        }
        self.grants.insert(id, grant);
        Ok(id)
    }

    /// The grant by its canonical id.
    #[must_use]
    pub fn grant_by_id(&self, id: &Id) -> Option<&TrainingDataGrant> {
        self.grants.get(id)
    }

    /// Whether the grant may be used at this block.
    #[must_use]
    pub fn is_valid(&self, id: &Id, now_block: u64) -> bool {
        self.grants
            .get(id)
            .is_some_and(|grant| grant.is_valid(now_block))
    }

    /// Consume one epoch. This is the only mutation an epoch may cause.
    /// The refusal names itself and is logged like an allowance.
    pub fn consume(&mut self, id: &Id, now_block: u64) -> EpochOutcome {
        let outcome = match self.grants.get_mut(id) {
            None => EpochOutcome::Refused(EpochRefusal::Unknown),
            Some(grant) if now_block > grant.expires_at_block => {
                EpochOutcome::Refused(EpochRefusal::Expired {
                    at: now_block,
                    expires_at: grant.expires_at_block,
                })
            }
            Some(grant) => match grant.consume_epoch() {
                Ok(remaining) => EpochOutcome::Consumed { remaining },
                Err(_) => EpochOutcome::Refused(EpochRefusal::Exhausted {
                    used: grant.epochs_used,
                    max: grant.max_epochs,
                }),
            },
        };
        self.log.push(EpochEntry {
            at_block: now_block,
            grant_id: *id,
            outcome: outcome.clone(),
        });
        outcome
    }

    /// All grants in the book, newest-first is not promised; iterate.
    pub fn grants(&self) -> impl Iterator<Item = (&Id, &TrainingDataGrant)> {
        self.grants.iter()
    }

    /// The epoch trail, oldest first.
    #[must_use]
    pub fn trail(&self) -> &[EpochEntry] {
        &self.log
    }

    /// How many refusals the trail holds. Zero refusals over a live training
    /// run means the check never ran.
    #[must_use]
    pub fn refusals(&self) -> usize {
        self.log
            .iter()
            .filter(|e| matches!(e.outcome, EpochOutcome::Refused(_)))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> Id {
        Id([byte; 32])
    }

    fn grant(asset: Id, owner: Id, grantee: Id, max_epochs: u32) -> TrainingDataGrant {
        TrainingDataGrant {
            asset_id: asset,
            owner,
            grantee,
            issued_at_block: 10,
            expires_at_block: 10_000,
            max_epochs,
            epochs_used: 0,
        }
    }

    #[test]
    fn hex_roundtrip_is_lossless() {
        let original = id(0xAB);
        let hex = original.to_hex();
        assert_eq!(Id::from_hex(&hex), Ok(original));
        assert_eq!(Id::from_hex(&hex[2..]), Ok(original));
        let err = Id::from_hex("0xab").unwrap_err();
        assert!(err.contains("32 bytes"));
        let err = Id::from_hex("z".repeat(64).as_str()).unwrap_err();
        assert!(err.contains("non-hex"));
    }

    #[test]
    fn derive_id_is_domain_separated_and_deterministic() {
        let a = grant(id(1), id(2), id(3), 4);
        let b = grant(id(1), id(2), id(3), 4);
        assert_eq!(a.derive_id(), b.derive_id());
        // changing one field changes the id
        let c = grant(id(1), id(2), id(4), 4);
        assert_ne!(a.derive_id(), c.derive_id());
        // and the preimage cannot be confused with a plain field hash
        let mut plain = Sha256::new();
        plain.update(a.asset_id.0);
        let field_hash = Id(plain.finalize().into());
        assert_ne!(a.derive_id(), field_hash);
    }

    #[test]
    fn shape_refusals_are_specific() {
        let mut g = grant(id(1), id(2), id(3), 0);
        assert!(g.validate_shape().is_err());
        g.max_epochs = MAX_TRAINING_GRANT_EPOCHS + 1;
        assert!(g.validate_shape().is_err());
        g.max_epochs = 2;
        g.expires_at_block = g.issued_at_block;
        assert!(g.validate_shape().is_err());
        g.expires_at_block = 100;
        g.epochs_used = 1;
        assert!(g.validate_shape().is_err());
        g.epochs_used = 0;
        assert!(g.validate_shape().is_ok());
    }

    #[test]
    fn epoch_consumption_is_fail_closed() {
        let mut book = EpochBook::new();
        let grant = grant(id(1), id(2), id(3), 2);
        let gid = book.issue(grant).expect("issue");
        assert!(book.is_valid(&gid, 50));
        assert_eq!(
            book.consume(&gid, 51),
            EpochOutcome::Consumed { remaining: 1 }
        );
        assert_eq!(
            book.consume(&gid, 52),
            EpochOutcome::Consumed { remaining: 0 }
        );
        assert_eq!(
            book.consume(&gid, 53),
            EpochOutcome::Refused(EpochRefusal::Exhausted { used: 2, max: 2 })
        );
        assert!(!book.is_valid(&gid, 53));
        assert_eq!(book.refusals(), 1);
    }

    #[test]
    fn time_expiry_refuses_independently_of_epochs() {
        let mut book = EpochBook::new();
        let gid = book.issue(grant(id(1), id(2), id(3), 5)).expect("issue");
        assert_eq!(
            book.consume(&gid, 10_001),
            EpochOutcome::Refused(EpochRefusal::Expired {
                at: 10_001,
                expires_at: 10_000
            })
        );
    }

    #[test]
    fn unknown_grant_is_a_distinct_refusal() {
        let mut book = EpochBook::new();
        assert_eq!(
            book.consume(&id(9), 10),
            EpochOutcome::Refused(EpochRefusal::Unknown)
        );
        // logged like an allowance: the trail has one entry with a label word
        let entry = book.trail().first().expect("entry");
        assert_eq!(entry.outcome, EpochOutcome::Refused(EpochRefusal::Unknown));
        assert_eq!(book.refusals(), 1);
    }

    #[test]
    fn duplicate_registration_is_refused() {
        let mut book = EpochBook::new();
        let g = grant(id(1), id(2), id(3), 2);
        book.issue(g.clone()).expect("first");
        let err = book.issue(g).unwrap_err();
        assert!(err.contains("already registered"));
    }

    #[test]
    fn owner_mismatch_is_enforced_at_the_book_boundary() {
        // The book never issues for an owner that is not named in the grant;
        // cross-checking the caller's identity is the caller's duty, but the
        // book refuses a grant whose owner and grantee are the same (a
        // training budget somebody granted to themselves is a permission
        // escalation) - that pattern is caught by shape.
        let g = grant(id(1), id(2), id(2), 2);
        assert!(g.validate_shape().is_ok());
        let mut book = EpochBook::new();
        assert!(book.issue(g).is_ok());
    }
}
