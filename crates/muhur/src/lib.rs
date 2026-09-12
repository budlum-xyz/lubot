//! The seal: an ordered hash chain over a batch's entries.
//!
//! # What a seal is, and what it is not
//!
//! A seal is not a signature. It carries no key and proves nothing about who
//! produced the batch. What it proves is narrower and, for the boundary this
//! crate exists for, sufficient: **these entries, in this order, are the entries
//! that were present when the batch was closed.**
//!
//! Signing belongs to the cold device. What the seal guarantees is that the
//! bytes the cold device is asked to sign are the bytes the hot device meant. A
//! media edited in transit, truncated by a flaky write, or reordered by a tool
//! that sorts lines changes the seal, and the cold reader refuses on the first
//! entry that disagrees.
//!
//! # Why a chain and not a single hash of the concatenation
//!
//! A single hash over the concatenated entries would also detect an edit. The
//! chain buys one thing that is worth having: **it can say where.**
//!
//! [`Sealer::verify`] walks the entries in order, recomputing the running seal,
//! and stops at the first one whose contribution does not match. The refusal
//! names that entry by index. An operator holding a batch of four hundred payout
//! lines and the message "the file is wrong" has a long afternoon; the same
//! operator holding "entry 217 disagrees" has a fix.
//!
//! The chain is also order-sensitive by construction, which the concatenation
//! would be too - but the chain makes the ordering *load-bearing in a way that
//! is visible*: swapping two entries changes every seal from the earlier one
//! onward, so a swapped pair cannot be localised to one index and will be
//! reported at the first of the two. That is the honest limitation, and it is
//! stated rather than hidden.
//!
//! # Determinism
//!
//! Every node must compute the same seal for the same entries, or the seal
//! proves nothing across nodes. The hash is SHA-256 over an explicit encoding,
//! and the encoding is written out here rather than delegated to a serializer:
//! a serializer whose output format is allowed to change would make every
//! existing seal unverifiable the day it did.

use lubot_read::sha256_hex;

/// The domain tag. Included in the first link so that a seal over one kind of
/// batch cannot be presented as a seal over another.
pub const SEAL_TAG: &[u8] = b"LUBOT-MUHU-V1";

/// One link's hex digest: 64 characters.
pub const LINK_HEX_LEN: usize = 64;

/// Why a seal did not verify.
///
/// Every variant carries enough to act on. A refusal that says only "mismatch"
/// is a refusal that sends the operator to read four hundred lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealError {
    /// The entry count differs. Reported before any hashing, because a
    /// truncated file is a different failure from a corrupted one and the fix
    /// is different.
    EntryCountMismatch { expected: usize, got: usize },
    /// The entry at `index` does not contribute what the running seal expects.
    /// `expected` is the seal up to and including that entry as the sealed batch
    /// recorded it; `got` is what these entries produce.
    EntryMismatch {
        index: usize,
        expected: String,
        got: String,
    },
    /// The final seal does not match the one recorded. Reached only when every
    /// entry matched, which means the recorded seal itself was altered.
    SealMismatch { expected: String, got: String },
    /// An entry was empty. An empty entry contributes nothing distinguishable
    /// from an absent one under a naive concatenation; refusing it removes the
    /// ambiguity rather than reasoning about it.
    EmptyEntry { index: usize },
}

impl std::fmt::Display for SealError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntryCountMismatch { expected, got } => {
                write!(f, "the batch carries {got} entries, the seal covers {expected}")
            }
            Self::EntryMismatch {
                index,
                expected,
                got,
            } => write!(
                f,
                "entry {index} disagrees: the sealed batch reaches {expected}, these entries reach {got}"
            ),
            Self::SealMismatch { expected, got } => {
                write!(f, "every entry matched but the recorded seal was altered: {expected} != {got}")
            }
            Self::EmptyEntry { index } => {
                write!(f, "entry {index} is empty and contributes nothing distinguishable")
            }
        }
    }
}

/// Builds and verifies seals.
///
/// Stateless: a [`Sealer`] holds nothing between calls, so two nodes sealing the
/// same entries cannot diverge through accumulated state.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sealer;

impl Sealer {
    /// The running seal after `entry`, given the seal so far.
    ///
    /// The previous link is hashed in first, then the entry, then a length
    /// prefix on the entry. The length prefix is what makes the encoding
    /// unambiguous: without it, entries `"ab"` and `"a","b"` would chain
    /// identically, and a batch could be split or merged without changing its
    /// seal.
    #[must_use]
    pub fn link(previous_hex: &str, entry: &str) -> String {
        let mut buf = String::with_capacity(previous_hex.len() + entry.len() + 16);
        buf.push_str(previous_hex);
        buf.push('\n');
        buf.push_str(&entry.len().to_string());
        buf.push(':');
        buf.push_str(entry);
        sha256_hex(buf.as_bytes())
    }

    /// The first link, before any entry.
    #[must_use]
    pub fn genesis() -> String {
        sha256_hex(SEAL_TAG)
    }

    /// Seals `entries` in order and returns the running link after each one.
    ///
    /// The vector is returned rather than only the final seal because the
    /// per-entry links are what make a mismatch localisable, and a caller that
    /// writes a batch wants to store them.
    ///
    /// # Errors
    ///
    /// [`SealError::EmptyEntry`] naming the first empty entry.
    pub fn seal(entries: &[&str]) -> Result<Vec<String>, SealError> {
        let mut links = Vec::with_capacity(entries.len());
        let mut running = Self::genesis();
        for (index, entry) in entries.iter().enumerate() {
            if entry.is_empty() {
                return Err(SealError::EmptyEntry { index });
            }
            running = Self::link(&running, entry);
            links.push(running.clone());
        }
        Ok(links)
    }

    /// The final seal for `entries`.
    ///
    /// # Errors
    ///
    /// [`SealError::EmptyEntry`] naming the first empty entry.
    pub fn final_seal(entries: &[&str]) -> Result<String, SealError> {
        Ok(Self::seal(entries)?.pop().unwrap_or_else(Self::genesis))
    }

    /// Verifies `entries` against a recorded set of per-entry links.
    ///
    /// Walks in order and stops at the first disagreement, which is the whole
    /// point: the caller learns which entry broke, not merely that something
    /// did.
    ///
    /// # Errors
    ///
    /// [`SealError::EntryCountMismatch`] first, then the first
    /// [`SealError::EntryMismatch`].
    pub fn verify(entries: &[&str], recorded: &[String]) -> Result<(), SealError> {
        if entries.len() != recorded.len() {
            return Err(SealError::EntryCountMismatch {
                expected: recorded.len(),
                got: entries.len(),
            });
        }
        let mut running = Self::genesis();
        for (index, entry) in entries.iter().enumerate() {
            if entry.is_empty() {
                return Err(SealError::EmptyEntry { index });
            }
            running = Self::link(&running, entry);
            let Some(expected) = recorded.get(index) else {
                // Unreachable: the lengths were just checked. Written as a
                // refusal rather than an index, because an index that is merely
                // probably in range is an abort waiting for an edit.
                return Err(SealError::EntryCountMismatch {
                    expected: recorded.len(),
                    got: entries.len(),
                });
            };
            if expected != &running {
                return Err(SealError::EntryMismatch {
                    index,
                    expected: expected.clone(),
                    got: running,
                });
            }
        }
        Ok(())
    }

    /// Verifies against a recorded final seal only.
    ///
    /// Cheaper and less useful: it cannot say where. Provided because a caller
    /// that stored only the final seal still needs to check it, and silently
    /// falling back to comparing nothing would be worse than the weaker check.
    ///
    /// # Errors
    ///
    /// [`SealError::SealMismatch`] or [`SealError::EmptyEntry`].
    pub fn verify_final(entries: &[&str], recorded_seal: &str) -> Result<(), SealError> {
        let computed = Self::final_seal(entries)?;
        if computed != recorded_seal {
            return Err(SealError::SealMismatch {
                expected: recorded_seal.to_string(),
                got: computed,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_entries_seal_to_the_same_value_on_every_call() {
        // Determinism is the property that makes a seal mean anything across
        // nodes. Two calls must agree, and the value must not depend on state.
        let a = Sealer::final_seal(&["one", "two", "three"]);
        let b = Sealer::final_seal(&["one", "two", "three"]);
        assert_eq!(a, b);
        assert_eq!(a.as_deref().map(str::len), Some(LINK_HEX_LEN));
    }

    #[test]
    fn reordering_changes_the_seal() {
        let a = Sealer::final_seal(&["one", "two"]);
        let b = Sealer::final_seal(&["two", "one"]);
        assert_ne!(a, b, "order is not load-bearing");
    }

    #[test]
    fn splitting_an_entry_does_not_preserve_the_seal() {
        // The length prefix exists for exactly this: without it, "ab" and
        // "a","b" would chain identically and a batch could be split or merged
        // without changing its seal.
        let joined = Sealer::final_seal(&["ab"]);
        let split = Sealer::final_seal(&["a", "b"]);
        assert_ne!(joined, split, "the encoding is ambiguous");
    }

    #[test]
    fn a_mismatch_names_the_entry_that_broke() {
        // The whole point of the chain: an operator holding four hundred lines
        // and "the file is wrong" has a long afternoon.
        let recorded = Sealer::seal(&["one", "two", "three", "four"]).expect("seal");
        let err = Sealer::verify(&["one", "TWO", "three", "four"], &recorded).unwrap_err();
        assert_eq!(
            err,
            SealError::EntryMismatch {
                index: 1,
                expected: recorded.get(1).cloned().unwrap_or_default(),
                got: Sealer::link(recorded.first().map_or("", String::as_str), "TWO"),
            }
        );
        assert!(
            matches!(err, SealError::EntryMismatch { index: 1, .. }),
            "the first disagreement must be reported, not a later one"
        );
    }

    #[test]
    fn a_truncated_batch_is_reported_as_a_count_not_a_corruption() {
        // A truncated file is a different failure from a corrupted one and the
        // fix is different, so it is reported before any hashing.
        let recorded = Sealer::seal(&["one", "two", "three"]).expect("seal");
        let err = Sealer::verify(&["one", "two"], &recorded).unwrap_err();
        assert_eq!(
            err,
            SealError::EntryCountMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn an_empty_entry_is_refused_rather_than_hashed() {
        // An empty entry contributes nothing distinguishable from an absent one
        // under a naive concatenation. Refusing it removes the ambiguity instead
        // of reasoning about it.
        let err = Sealer::seal(&["one", "", "three"]).unwrap_err();
        assert_eq!(err, SealError::EmptyEntry { index: 1 });
    }

    #[test]
    fn an_empty_batch_seals_to_the_genesis_link() {
        assert_eq!(Sealer::final_seal(&[]), Ok(Sealer::genesis()));
        assert!(Sealer::verify(&[], &[]).is_ok());
    }

    #[test]
    fn an_altered_recorded_seal_is_distinguishable_from_an_altered_entry() {
        // verify_final cannot say where; verify can. Both must refuse, and the
        // two refusals must not look alike, because they mean different things.
        let entries = ["one", "two"];
        let recorded = Sealer::seal(&entries).expect("seal");
        let mut tampered = recorded.clone();
        if let Some(last) = tampered.last_mut() {
            *last = Sealer::genesis();
        }
        assert!(matches!(
            Sealer::verify(&entries, &tampered),
            Err(SealError::EntryMismatch { index: 1, .. })
        ));
        assert!(matches!(
            Sealer::verify_final(&entries, &Sealer::genesis()),
            Err(SealError::SealMismatch { .. })
        ));
    }

    #[test]
    fn the_domain_tag_separates_one_batch_kind_from_another() {
        // A seal over one kind of batch must not be presentable as a seal over
        // another, so the tag is inside the first link.
        assert_eq!(Sealer::genesis(), sha256_hex(SEAL_TAG));
        assert_ne!(Sealer::genesis(), sha256_hex(b"SOMETHING-ELSE"));
    }

    #[test]
    fn per_entry_links_are_a_prefix_chain() {
        // Each link extends the previous one. A caller storing the links can
        // verify a prefix of a batch without the rest, which is what makes a
        // partially written batch checkable rather than merely suspect.
        let links = Sealer::seal(&["one", "two", "three"]).expect("seal");
        assert_eq!(links.len(), 3);
        let two = Sealer::seal(&["one", "two"]).expect("seal");
        assert_eq!(&links[..2], &two[..], "the chain is not a prefix chain");
    }
}
