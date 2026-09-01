#![forbid(unsafe_code)]
//! # lubot-read - where content comes from, and how it is proven
//!
//! Three channels, named once and closed here:
//!
//! | channel | what it is | how it is proven |
//! |---|---|---|
//! | [`SourceKind::Stored`] | content held by the network's storage | digest of the bytes |
//! | [`SourceKind::Granted`] | content opened by a view grant | digest of the bytes |
//! | [`SourceKind::Local`] | this repository's own files | digest of the bytes |
//!
//! **There is no fourth channel.** [`source_kind`] refuses anything else, and
//! that refusal is what makes the sentence "the loop is closed" a measurement
//! rather than a slogan.
//!
//! Every record carries the digest of its own bytes. [`verify_sha256`] compares
//! it, and a mismatch is a refusal - never a warning that a caller can ignore.

use sha2::{Digest, Sha256};

/// The three channels content may enter through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Stored,
    Granted,
    Local,
}

impl SourceKind {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            SourceKind::Stored => "stored",
            SourceKind::Granted => "granted",
            SourceKind::Local => "local",
        }
    }
}

/// Parse a channel name. Anything outside the three is refused: this is the
/// "no fourth channel" rule, in code.
///
/// # Errors
/// Returns the offending name when it is not one of the three channels.
pub fn source_kind(name: &str) -> Result<SourceKind, String> {
    match name {
        "stored" => Ok(SourceKind::Stored),
        "granted" => Ok(SourceKind::Granted),
        "local" => Ok(SourceKind::Local),
        other => Err(format!(
            "unknown source `{other}`: content enters through stored, granted or local, \
             and a fourth channel would be an unaudited one"
        )),
    }
}

/// Lowercase hex of the SHA-256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Fail-closed provenance check.
///
/// # Errors
/// Returns a description when the bytes do not hash to `expected`.
pub fn verify_sha256(bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual = sha256_hex(bytes);
    if actual == expected {
        return Ok(());
    }
    Err(format!(
        "digest mismatch: the record claims {expected}, the bytes hash to {actual}"
    ))
}

/// One readable item: its bytes, where they came from, and who may see them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// Stable identifier, also the key id a grant names.
    pub id: String,
    /// Human path or reference, used in citations.
    pub origin: String,
    pub kind: SourceKind,
    pub restricted: bool,
    pub body: String,
    pub digest: String,
}

impl Item {
    /// Build an item and compute its digest from the body it carries.
    #[must_use]
    pub fn new(id: &str, origin: &str, kind: SourceKind, restricted: bool, body: &str) -> Self {
        Self {
            id: id.to_string(),
            origin: origin.to_string(),
            kind,
            restricted,
            body: body.to_string(),
            digest: sha256_hex(body.as_bytes()),
        }
    }

    /// Re-check the body against the digest the record carries.
    ///
    /// # Errors
    /// Returns a description when the two disagree.
    pub fn verify(&self) -> Result<(), String> {
        verify_sha256(self.body.as_bytes(), &self.digest)
    }
}

/// A read surface over a fixed set of items.
///
/// The fixture backing is deliberate for the first version: the answer path is
/// provable against content that does not move under it. The same trait a live
/// node implements is [`Corpus::get`] - one method, so swapping the backing
/// cannot quietly widen what the reader can reach.
pub trait Corpus {
    /// Fetch an item by id.
    fn get(&self, id: &str) -> Option<&Item>;
    /// Every id the corpus holds, in insertion order.
    fn ids(&self) -> Vec<String>;
}

/// A corpus held in memory.
#[derive(Debug, Default)]
pub struct FixtureCorpus {
    items: Vec<Item>,
}

impl FixtureCorpus {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an item, refusing one whose digest does not match its body.
    ///
    /// # Errors
    /// Returns the verification failure.
    pub fn insert(&mut self, item: Item) -> Result<(), String> {
        item.verify()?;
        self.items.push(item);
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Corpus for FixtureCorpus {
    fn get(&self, id: &str) -> Option<&Item> {
        self.items.iter().find(|i| i.id == id)
    }

    fn ids(&self) -> Vec<String> {
        self.items.iter().map(|i| i.id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_channels_parse() {
        assert_eq!(source_kind("stored"), Ok(SourceKind::Stored));
        assert_eq!(source_kind("granted"), Ok(SourceKind::Granted));
        assert_eq!(source_kind("local"), Ok(SourceKind::Local));
    }

    #[test]
    fn reject_unknown_source() {
        let err = source_kind("scraped").unwrap_err();
        assert!(err.contains("unknown source"));
        assert!(err.contains("fourth channel"));
    }

    #[test]
    fn a_digest_matches_a_known_vector() {
        // SHA-256 of the empty input, the one vector that needs no trust.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn verification_fails_closed_on_a_changed_byte() {
        let good = sha256_hex(b"budlum");
        assert!(verify_sha256(b"budlum", &good).is_ok());
        let err = verify_sha256(b"budluM", &good).unwrap_err();
        assert!(err.contains("digest mismatch"));
    }

    #[test]
    fn an_item_carries_the_digest_of_its_own_body() {
        let item = Item::new("a", "docs/a.md", SourceKind::Local, false, "hello");
        assert!(item.verify().is_ok());
    }

    #[test]
    fn a_tampered_item_is_refused_at_the_door() {
        let mut item = Item::new("a", "docs/a.md", SourceKind::Local, false, "hello");
        item.body = "hello, but different".to_string();
        let mut corpus = FixtureCorpus::new();
        assert!(corpus.insert(item).is_err());
        assert!(corpus.is_empty());
    }

    #[test]
    fn the_corpus_returns_what_it_was_given_and_nothing_else() {
        let mut corpus = FixtureCorpus::new();
        corpus
            .insert(Item::new(
                "a",
                "docs/a.md",
                SourceKind::Local,
                false,
                "hello",
            ))
            .unwrap();
        assert_eq!(corpus.len(), 1);
        assert!(corpus.get("a").is_some());
        assert!(corpus.get("b").is_none());
        assert_eq!(corpus.ids(), vec!["a".to_string()]);
    }
}
