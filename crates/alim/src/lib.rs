#![forbid(unsafe_code)]
//! # lubot-alim - the intake line: manifests in, provenance out
//!
//! Working directive section 6 asks for four things: intake, the training step
//! that intake feeds, record-level provenance, and the hardware it runs on.
//! This crate is the first three, on the storage side, and nothing else.
//! It is deliberately small: every function here answers one question, and
//! every question has a refusal that names the rule it enforced.
//!
//! Three doors, in the order the directive names them:
//!
//! 1. [`manifest`] - a manifest is admitted whole or refused by name, and the
//!    source classes it may carry are a closed set (K2). The directive's K2 is
//!    the governing text here: the corpus surface is the Lubot tree, plus
//!    records the `doc` command admitted with record-based provenance (K3).
//!    A manifest from any other class is refused - never downgraded to a
//!    member of the set, because a silent downgrade is a widening nobody
//!    approved.
//! 2. [`provenance`] - one row per admitted record, carrying which manifest,
//!    which loader, when it was verified and which training step it entered.
//!    The ledger is append-only and refuses to be read in part: a malformed
//!    line refuses the file rather than being skipped.
//! 3. [`agirlik`] - the weight side of the same idea: a content address, an
//!    erasure-coding plan and a holder rule, all computed rather than guessed.
//!    The holder rule is the reason it exists: a placement that would lose the
//!    object when one holder disappears is refused, not reported.
//!
//! What this crate does not do is read the corpus, train, or reach the chain.
//! It takes bytes that already arrived, checks them against the rules that
//! admit bytes, and writes down where they came from.

pub mod agirlik;
pub mod manifest;
pub mod provenance;

use sha2::{Digest, Sha256};

/// The `sha256` digest of `bytes`, lowercase hex.
///
/// One function, used by every content address in this crate, so a digest
/// computed on any machine can be compared with one computed here.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// True when `value` has the shape of a lowercase `sha256` hex digest.
///
/// Shape only: whether the digest matches any bytes is
/// [`manifest::admit`]'s job, and it is a different question.
#[must_use]
pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// True when `path` is a relative path with no parent steps.
///
/// A manifest path is data, not instruction: `../../etc/passwd` names a file
/// outside the tree the intake was pointed at, so it is refused before any
/// bytes are resolved for it.
#[must_use]
pub fn is_safe_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

/// Why the intake refused.
///
/// A refusal is an answer, not an error log: each variant names the rule it
/// enforced, so the operator reads which rule spoke rather than that something
/// went wrong. `Display` prints [`Refusal::message`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Bytes that are not a manifest of this schema.
    Malformed(String),
    /// A manifest from a class outside the closed K2 set.
    UnknownSourceClass(String),
    /// A schema number this build does not read.
    UnsupportedSchema(u32),
    /// A `manifest_id` that is not a `sha256` hex digest.
    BadManifestId(String),
    /// A manifest with no loader name.
    MissingLoader,
    /// A manifest that claims no records.
    EmptyManifest,
    /// A record path that is absolute, empty, or walks up the tree.
    UnsafePath(String),
    /// A kind outside the four the corpus admits.
    UnknownKind { path: String, kind: String },
    /// A licence outside the closed set.
    UnknownLicence { path: String, licence: String },
    /// A record with no attribution.
    MissingAttribution(String),
    /// A record without the content id / asset id pair.
    MissingProvenancePair(String),
    /// A digest that is not a `sha256` hex digest.
    BadDigest { path: String, digest: String },
    /// No bytes were resolved for the record's path.
    MissingBytes(String),
    /// Bytes that do not hash to the digest the manifest claims.
    DigestMismatch {
        path: String,
        expected: String,
        measured: String,
    },
    /// A weight artefact with no bytes.
    EmptyArtefact,
    /// Erasure-coding parameters outside the addressable range.
    BadErasureParams { data: u32, parity: u32 },
    /// A shard index outside the data shards.
    ShardOutOfRange { index: u32, data: u32 },
    /// Fewer than two holders: one holder is not a placement.
    TooFewHolders(usize),
    /// A holder that would keep more shards than the code tolerates losing.
    HolderOverload {
        holder: String,
        shards: u32,
        tolerance: u32,
    },
    /// An I/O failure, named.
    Io(String),
    /// A ledger line that is not a row.
    MalformedLedger { line: u64, reason: String },
    /// A ledger whose steps go backwards.
    StepWentBackwards { line: u64, step: u64, previous: u64 },
}

impl Refusal {
    /// The rule this refusal enforced, as the constitution names it.
    #[must_use]
    pub fn rule(&self) -> &'static str {
        match self {
            Self::UnknownSourceClass(_) => "K2",
            Self::UnknownLicence { .. } | Self::MissingAttribution(_) => "K3",
            Self::MissingProvenancePair(_)
            | Self::BadManifestId(_)
            | Self::BadDigest { .. }
            | Self::DigestMismatch { .. } => "O",
            _ => "-",
        }
    }

    /// One line for the operator: the rule, then what it refused.
    #[must_use]
    pub fn message(&self) -> String {
        let body = match self {
            Self::Malformed(reason) => format!("malformed manifest: {reason}"),
            Self::UnknownSourceClass(name) => format!(
                "source class `{name}` is not admitted: the corpus surface is the \
                 budlum-xyz trees, `lubot`, `budlum` and `workspace`, plus \
                 `doc`-admitted records; a manifest from any other class is \
                 refused, not downgraded"
            ),
            Self::UnsupportedSchema(schema) => format!("unsupported manifest schema: {schema}"),
            Self::BadManifestId(id) => format!("`manifest_id` is not a digest: {id}"),
            Self::MissingLoader => "the manifest names no loader".to_string(),
            Self::EmptyManifest => "the manifest claims no records".to_string(),
            Self::UnsafePath(path) => {
                format!("path `{path}` is absolute, empty, or walks up the tree")
            }
            Self::UnknownKind { path, kind } => {
                format!("`{path}`: kind `{kind}` is outside the closed four")
            }
            Self::UnknownLicence { path, licence } => {
                format!("`{path}`: licence `{licence}` is outside the closed set")
            }
            Self::MissingAttribution(path) => format!("`{path}`: no attribution"),
            Self::MissingProvenancePair(path) => {
                format!("`{path}`: content id / asset id pair is incomplete")
            }
            Self::BadDigest { path, digest } => {
                format!("`{path}`: digest `{digest}` is not a digest")
            }
            Self::MissingBytes(path) => format!("`{path}`: no bytes were resolved"),
            Self::DigestMismatch {
                path,
                expected,
                measured,
            } => format!("`{path}`: digest mismatch, expected {expected}, measured {measured}"),
            Self::EmptyArtefact => "the artefact carries no bytes".to_string(),
            Self::BadErasureParams { data, parity } => format!(
                "erasure parameters data={data} parity={parity} are outside the \
                 addressable range (data>=1, parity>=1, data+parity<=255)"
            ),
            Self::ShardOutOfRange { index, data } => {
                format!("shard {index} is outside the {data} data shards")
            }
            Self::TooFewHolders(count) => {
                format!("{count} holder(s): a placement needs at least 2")
            }
            Self::HolderOverload {
                holder,
                shards,
                tolerance,
            } => format!(
                "holder `{holder}` would keep {shards} shards while the code tolerates \
                 losing {tolerance}: one holder's loss would lose the object"
            ),
            Self::Io(reason) => format!("i/o: {reason}"),
            Self::MalformedLedger { line, reason } => {
                format!("ledger line {line} is not a row: {reason}")
            }
            Self::StepWentBackwards {
                line,
                step,
                previous,
            } => format!("ledger line {line}: step {step} is behind the previous {previous}"),
        };
        format!("[{}] {body}", self.rule())
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message())
    }
}

impl std::error::Error for Refusal {}

#[cfg(test)]
mod tests {
    use super::{is_safe_relative, is_sha256_hex, sha256_hex, Refusal};

    #[test]
    fn digest_is_lowercase_hex_of_known_length() {
        let digest = sha256_hex(b"lubot");
        assert_eq!(digest.len(), 64);
        assert!(is_sha256_hex(&digest));
        // The empty-string digest is a published constant, so this test does
        // not merely check the shape: it checks the function against a value
        // that came from outside it.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn digest_separates_two_different_inputs() {
        assert_ne!(sha256_hex(b"lubot"), sha256_hex(b"lubot "));
    }

    #[test]
    fn digest_shape_is_checked_before_it_is_trusted() {
        assert!(is_sha256_hex(&sha256_hex(b"x")));
        assert!(!is_sha256_hex(""));
        assert!(!is_sha256_hex(&"A".repeat(64)));
        assert!(!is_sha256_hex(&"g".repeat(64)));
    }

    #[test]
    fn relative_paths_only() {
        assert!(is_safe_relative("a/b.md"));
        assert!(!is_safe_relative(""));
        assert!(!is_safe_relative("/etc/passwd"));
        assert!(!is_safe_relative("../up.md"));
        assert!(!is_safe_relative("a/../b.md"));
        assert!(!is_safe_relative("a//b.md"));
        assert!(!is_safe_relative("c:\\x"));
    }

    #[test]
    fn every_refusal_names_its_rule() {
        assert_eq!(Refusal::UnknownSourceClass("x".to_string()).rule(), "K2");
        assert_eq!(
            Refusal::UnknownLicence {
                path: "a".to_string(),
                licence: "GPL-3.0".to_string()
            }
            .rule(),
            "K3"
        );
        assert_eq!(Refusal::EmptyArtefact.rule(), "-");
        assert!(Refusal::EmptyArtefact
            .message()
            .starts_with("[-] the artefact"));
    }
}
