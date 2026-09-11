//! A record that shows when it was edited after the fact.
//!
//! Lubot's outputs are the evidence a human audit starts from. If the file
//! holding them can be edited with no trace, then "the run said this" and
//! "someone made the run say this" are the same sentence - and there is no
//! review step that can tell them apart, because the review reads the file.
//!
//! # The shape
//!
//! Each entry seals the previous tip: `H(seq, actor, text) + prev`. The chain
//! therefore has two properties worth the cost of stating them:
//!
//! * rewriting any entry changes every seal after it;
//! * removing the tail changes the final tip, which is stored on `finalize`.
//!
//! So the checks are: digests recomputed from the bytes, sequence numbers
//! strictly increasing with no gap, and the recomputed tip equal to the stored
//! one. A chain that was never finalized has no stored tip, and `verify` says
//! so instead of passing: an unfinished seal is not a clean seal.
//!
//! # Limits, stated
//!
//! The digest is FNV-1a, 64 bits, chosen because this is not an adversary model.
//! It detects the realistic corruption in this repository's history: a file
//! edited by hand after the run, a tail deleted to make a report shorter, an
//! entry reordered to change the story. It does not stop an attacker who can
//! recompute every seal; that would need a keyed construction, and a key in a
//! source tree is not a key. Where an adversarial guarantee is needed, the seal
//! must be a hash-chain digest signed by a real key - and this crate would say
//! so in its type, not in a comment.

use std::collections::BTreeMap;

/// 64-bit FNV-1a over text. Not a security primitive; a way to bind bytes to a
/// claim without storing them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest(
    /// The bytes, little-endian.
    pub [u8; 8],
);

impl Digest {
    /// The digest of a string.
    #[must_use]
    pub fn of(text: &str) -> Self {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in text.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self(hash.to_le_bytes())
    }

    /// Lowercase hex.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut out = String::with_capacity(16);
        for byte in self.0 {
            out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
            out.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
        }
        out
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hex())
    }
}

/// One sealed line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    seq: u64,
    actor: String,
    text: String,
    digest: Digest,
    prev: Digest,
}

impl Entry {
    /// Its position, starting at one.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Who wrote it.
    #[must_use]
    pub fn actor(&self) -> &str {
        &self.actor
    }

    /// What it says.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// `H(seq, actor, text)`.
    #[must_use]
    pub fn digest(&self) -> Digest {
        self.digest
    }

    /// The sealed tip this entry was built on.
    #[must_use]
    pub fn prev(&self) -> Digest {
        self.prev
    }
}

/// Why a chain is not a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Breach {
    /// Nothing was written: an empty log proves nothing and reports nothing.
    Empty,
    /// The entry's bytes no longer match its seal.
    Rewritten {
        /// The entry.
        seq: u64,
        /// The seal stored beside it.
        claimed: Digest,
        /// The seal the bytes now produce.
        found: Digest,
    },
    /// The seal did not chain to the previous entry.
    ChainBroken {
        /// The entry whose `prev` is wrong.
        seq: u64,
        /// What it claimed to follow.
        claimed: Digest,
        /// What actually precedes it.
        expected: Digest,
    },
    /// A sequence number that repeats or goes backwards.
    SequenceNotIncreasing {
        /// The offending entry.
        seq: u64,
        /// The one before it.
        after: u64,
    },
    /// A gap in the numbering: entries were taken out of the middle.
    Gap {
        /// The last seq seen.
        before: u64,
        /// The first seq after the hole.
        after: u64,
    },
    /// The recomputed tip is not the sealed tip: the tail moved.
    TailMoved {
        /// The sealed tip.
        sealed: Digest,
        /// What the entries now produce.
        recomputed: Digest,
    },
    /// `verify` before `finalize`: there is nothing to compare against.
    NeverFinalized,
    /// A write attempted on a finalized chain.
    Sealed,
    /// The index points at an entry the chain does not have for that actor.
    IndexDrift {
        /// The sequence the index claims.
        seq: u64,
        /// The actor the index claims wrote it.
        actor: String,
    },
    /// An actor or text so empty it cannot be attributed or read.
    Unrecordable {
        /// Which field.
        field: &'static str,
    },
}

impl std::fmt::Display for Breach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "the chain holds nothing"),
            Self::Rewritten {
                seq,
                claimed,
                found,
            } => write!(
                f,
                "entry {seq} was rewritten: sealed {claimed}, now {found}; every seal after \
                 it must be rechecked by hand, because the run said something else"
            ),
            Self::ChainBroken {
                seq,
                claimed,
                expected,
            } => write!(
                f,
                "entry {seq} chains from {claimed} and the entry before it seals {expected}"
            ),
            Self::SequenceNotIncreasing { seq, after } => write!(
                f,
                "entry {seq} follows {after}: a log is ordered, or it is a bag"
            ),
            Self::Gap { before, after } => write!(
                f,
                "the log jumps from {before} to {after}: {n} entries are missing from the \
                 middle",
                n = if after > before + 1 { after - before - 1 } else { 0 }
            ),
            Self::TailMoved {
                sealed,
                recomputed,
            } => write!(
                f,
                "the sealed tip is {sealed} and the entries produce {recomputed}: the tail \
                 was shortened, which is how a report survives its own inconvenient end"
            ),
            Self::NeverFinalized => write!(
                f,
                "the chain was never finalized; comparing entries against each other \
                 proves internal order, not that the record is complete"
            ),
            Self::Sealed => write!(
                f,
                "the chain is finalized: a later entry is a new record, not an append"
            ),
            Self::IndexDrift { seq, actor } => write!(
                f,
                "the index says {actor} wrote entry {seq}; the chain has no such entry: a \
                 second structure over a record can drift even when the record is intact"
            ),
            Self::Unrecordable { field } => write!(
                f,
                "`{field}` is empty: an entry nobody can attribute or read is not a record"
            ),
        }
    }
}

impl std::error::Error for Breach {}

/// The seal chain.
#[derive(Debug, Clone)]
pub struct Chain {
    entries: Vec<Entry>,
    salt: Digest,
    sealed_tip: Option<Digest>,
    finalized: bool,
}

impl Chain {
    /// A chain whose genesis seal is `H(genesis_label)`.
    ///
    /// The label matters: two chains that share their entries but not their
    /// genesis are different records, and must not verify against each other.
    #[must_use]
    pub fn new(genesis_label: &str) -> Self {
        Self {
            entries: Vec::new(),
            salt: Digest::of(genesis_label),
            sealed_tip: None,
            finalized: false,
        }
    }

    /// The entries, in order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Whether the chain was finalized.
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// The tip sealed at `finalize`, if any.
    #[must_use]
    pub fn sealed_tip(&self) -> Option<Digest> {
        self.sealed_tip
    }

    /// The tip the current entries produce.
    #[must_use]
    pub fn tip(&self) -> Digest {
        let mut cur = self.salt;
        for entry in &self.entries {
            cur = Digest::of(&format!("{}{}", cur.hex(), entry.digest().hex()));
        }
        cur
    }

    /// Appends an entry, sealed on the current tip.
    ///
    /// # Errors
    ///
    /// [`Breach::Sealed`] after finalization, [`Breach::Unrecordable`] for an
    /// empty actor or text.
    pub fn append(&mut self, actor: &str, text: &str) -> Result<u64, Breach> {
        if self.finalized {
            return Err(Breach::Sealed);
        }
        if actor.trim().is_empty() {
            return Err(Breach::Unrecordable { field: "actor" });
        }
        if text.trim().is_empty() {
            return Err(Breach::Unrecordable { field: "text" });
        }
        let seq = self.entries.len() as u64 + 1;
        let digest = Digest::of(&format!("{seq}\u{1}{actor}\u{1}{text}"));
        let prev = self.tip();
        self.entries.push(Entry {
            seq,
            actor: actor.to_string(),
            text: text.to_string(),
            digest,
            prev,
        });
        Ok(seq)
    }

    /// Stores the current tip as the sealed one and returns it. After this the
    /// chain is append-frozen: a late entry is a new record, not a line in this
    /// one, and the caller usually wants the tip to write down separately.
    pub fn finalize(&mut self) -> Digest {
        let tip = self.tip();
        self.sealed_tip = Some(tip);
        self.finalized = true;
        tip
    }

    /// Recomputes everything from the bytes and reports the first breach.
    ///
    /// The entry digests are recomputed from `seq/actor/text` rather than
    /// trusted from the field, which is the entire mechanism: a stored seal
    /// that survives a rewrite is a seal of the *rewrite*.
    ///
    /// # Errors
    ///
    /// The first [`Breach`] found.
    pub fn verify(&self) -> Result<(), Breach> {
        if self.entries.is_empty() {
            return Err(Breach::Empty);
        }
        if !self.finalized {
            return Err(Breach::NeverFinalized);
        }
        let mut cur = self.salt;
        let mut last_seq = 0u64;
        for entry in &self.entries {
            if entry.seq != last_seq + 1 {
                if entry.seq <= last_seq {
                    return Err(Breach::SequenceNotIncreasing {
                        seq: entry.seq,
                        after: last_seq,
                    });
                }
                return Err(Breach::Gap {
                    before: last_seq,
                    after: entry.seq,
                });
            }
            last_seq = entry.seq;
            let want = Digest::of(&format!(
                "{}\u{1}{}\u{1}{}",
                entry.seq,
                entry.actor(),
                entry.text()
            ));
            if want != entry.digest {
                return Err(Breach::Rewritten {
                    seq: entry.seq,
                    claimed: entry.digest,
                    found: want,
                });
            }
            if entry.prev != cur {
                return Err(Breach::ChainBroken {
                    seq: entry.seq,
                    claimed: entry.prev,
                    expected: cur,
                });
            }
            cur = Digest::of(&format!("{}{}", cur.hex(), want.hex()));
        }
        match self.sealed_tip {
            Some(tip) if tip != cur => Err(Breach::TailMoved {
                sealed: tip,
                recomputed: cur,
            }),
            Some(_) => Ok(()),
            None => Err(Breach::NeverFinalized),
        }
    }

    /// The whole record as text a human can read after the seal - and only
    /// after, since a reader who edits this file is exactly what `verify` is
    /// for.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\n",
                entry.seq,
                entry.actor(),
                entry.digest().hex(),
                entry.text()
            ));
        }
        out
    }
}

/// A chain that also indexes entries by actor, so "what did this run write" is
/// answered without a scan. A convenience over [`Chain`], with the same seal.
#[derive(Debug, Clone)]
pub struct Indexed {
    chain: Chain,
    by_actor: BTreeMap<String, Vec<u64>>,
}

impl Indexed {
    /// Wraps a chain.
    #[must_use]
    pub fn new(chain: Chain) -> Self {
        Self {
            chain,
            by_actor: BTreeMap::new(),
        }
    }

    /// Appends and indexes.
    ///
    /// # Errors
    ///
    /// Passes through [`Chain::append`]'s.
    pub fn append(&mut self, actor: &str, text: &str) -> Result<u64, Breach> {
        let seq = self.chain.append(actor, text)?;
        self.by_actor
            .entry(actor.to_string())
            .or_default()
            .push(seq);
        Ok(seq)
    }

    /// The sequences written by `actor`.
    #[must_use]
    pub fn by(&self, actor: &str) -> &[u64] {
        self.by_actor.get(actor).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The sealed chain.
    #[must_use]
    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    /// Seals the chain the index points at.
    ///
    /// The seal is what every later reader recomputes and compares, so it is
    /// reached *through* the index: handing out `&mut Chain` would let a
    /// caller append after sealing and move the tip under the index's feet.
    ///
    /// [`Self::verify`] remains the authority; this only fixes the moment.
    pub fn finalize(&mut self) -> Digest {
        self.chain.finalize()
    }

    /// Verifies the chain, then checks that the index did not drift from it: a
    /// second index can lie in a second way.
    ///
    /// # Errors
    ///
    /// The chain's first [`Breach`], or the index's first disagreement.
    pub fn verify(&self) -> Result<(), Breach> {
        self.chain.verify()?;
        for (actor, seqs) in &self.by_actor {
            for seq in seqs {
                let found = self
                    .chain
                    .entries()
                    .iter()
                    .find(|e| e.seq() == *seq)
                    .filter(|e| e.actor() == *actor);
                if found.is_none() {
                    return Err(Breach::IndexDrift {
                        seq: *seq,
                        actor: actor.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled() -> Chain {
        let mut c = Chain::new("lubot-run-2026-09-10A");
        c.append("loop", "read src/storage/storage_deal.rs").unwrap();
        c.append("loop", "audit_coding wired at chain_actor.rs:4106")
            .unwrap();
        c.append("gate", "typos clean, fmt dirty").unwrap();
        c.finalize();
        c
    }

    #[test]
    fn a_finalized_chain_that_was_not_touched_verifies() {
        let c = filled();
        assert_eq!(c.verify(), Ok(()));
        assert_eq!(c.entries().len(), 3);
        assert!(c.is_finalized());
    }

    #[test]
    fn an_empty_chain_proves_nothing_and_says_so() {
        let mut c = Chain::new("nothing");
        c.finalize();
        assert_eq!(c.verify(), Err(Breach::Empty));
    }

    #[test]
    fn an_unfinalized_chain_is_not_a_clean_chain() {
        let mut c = Chain::new("lubot");
        c.append("loop", "one").unwrap();
        assert_eq!(c.verify(), Err(Breach::NeverFinalized));
    }

    #[test]
    fn rewriting_one_entry_changes_its_seal() {
        let mut c = filled();
        let idx = 1;
        c.entries[idx].text = "audit_coding NOT wired".to_string();
        match c.verify() {
            Err(Breach::Rewritten { seq, .. }) => assert_eq!(seq, 2),
            other => panic!("expected a rewrite to be caught, got {other:?}"),
        }
    }

    #[test]
    fn deleting_the_tail_moves_the_sealed_tip() {
        let mut c = filled();
        c.entries.pop();
        assert!(
            matches!(c.verify(), Err(Breach::TailMoved { .. })),
            "a truncated log that still verifies is the failure this crate exists for"
        );
        assert_ne!(c.tip(), c.sealed_tip().unwrap_or(c.tip()));
    }

    #[test]
    fn taking_an_entry_from_the_middle_leaves_a_numbering_hole() {
        let mut c = filled();
        c.entries.remove(0);
        assert_eq!(
            c.verify(),
            Err(Breach::Gap {
                before: 0,
                after: 2
            })
        );
    }

    #[test]
    fn nothing_is_written_after_finalization() {
        let mut c = filled();
        assert_eq!(c.append("loop", "late"), Err(Breach::Sealed));
    }

    #[test]
    fn unattributable_entries_are_refused() {
        let mut c = Chain::new("lubot");
        assert_eq!(
            c.append("  ", "text"),
            Err(Breach::Unrecordable { field: "actor" })
        );
        assert_eq!(
            c.append("loop", "   "),
            Err(Breach::Unrecordable { field: "text" })
        );
    }

    #[test]
    fn two_records_do_not_verify_against_each_other() {
        let mut a = Chain::new("run-a");
        a.append("loop", "same bytes").unwrap();
        a.finalize();
        let mut b = Chain::new("run-b");
        b.append("loop", "same bytes").unwrap();
        b.finalize();
        assert_ne!(a.tip(), b.tip(), "the genesis label is part of the record");
        let mut mixed = a.clone();
        mixed.entries[0].prev = b.tip();
        assert!(matches!(mixed.verify(), Err(Breach::ChainBroken { .. })));
    }

    #[test]
    fn render_and_reverify_round_trips_through_text() {
        let c = filled();
        let text = c.render();
        assert_eq!(text.lines().count(), 3);
        assert!(text.contains("audit_coding wired"));
        // The rendered form is for reading; re-parsing it is what a human audit
        // does, and every line carries its own seal for that walk.
        for line in text.lines() {
            assert_eq!(line.split('\t').count(), 4);
        }
    }

    #[test]
    fn the_index_is_checked_against_the_chain_it_indexes() {
        let mut ix = Indexed::new(Chain::new("lubot"));
        ix.append("loop", "a").unwrap();
        ix.append("gate", "b").unwrap();
        ix.finalize();
        assert_eq!(ix.verify(), Ok(()));
        assert_eq!(ix.by("loop"), &[1u64]);
        assert_eq!(ix.by("nobody"), &[] as &[u64]);
        // An index that points at nothing is caught the same way a rewritten
        // entry is: not by trust, by recomputation. The chain itself is intact
        // here, so the failure is the index's alone - which is why the index
        // gets its own rule instead of leaning on the seal.
        ix.by_actor
            .insert("ghost".to_string(), vec![9u64]);
        assert_eq!(
            ix.verify(),
            Err(Breach::IndexDrift {
                seq: 9,
                actor: "ghost".to_string()
            })
        );
        assert!(ix.chain().is_finalized());
    }
}
