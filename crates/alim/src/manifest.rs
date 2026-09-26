//! Manifest admission: a manifest is admitted whole, or refused by name.
//!
//! The rules are the corpus rules this repository already enforces on the way
//! in, applied one step earlier - at the door where bytes arrive. Nothing here
//! reads a corpus, trains, or decides policy: it checks a claim against rules
//! that are already written down, and refuses rather than repairs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{is_safe_relative, is_sha256_hex, sha256_hex, Refusal};

/// The manifest schema this build reads.
pub const SCHEMA: u32 = 1;

/// The four record kinds the corpus admits. A fifth kind would need a ceiling
/// and an admission rule of its own, so it is refused here instead.
pub const KINDS: [&str; 4] = ["api", "behaviour", "doc", "markdown"];

/// The licences the corpus may carry: permissive with attribution, or ours.
pub const LICENCES: [&str; 3] = ["MIT", "Apache-2.0", "PolyForm-Shield-1.0.0"];

/// The corpus surface, as a closed set.
///
/// K2 names the budlum-xyz surface: this repository, the budlum core tree and
/// the workspace root documents - the three source names
/// `training/build_corpus.py` already carries in its sources manifest. K3 adds
/// the records the `doc` command admitted with record-based provenance. A
/// manifest from any other class is refused by name: widening the set is a
/// constitution change, and no code path here can perform one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceClass {
    /// This repository (K2).
    Lubot,
    /// The budlum core tree (K2).
    Budlum,
    /// The workspace root documents (K2).
    Workspace,
    /// Admitted through `lubot doc` with record-based provenance (K3).
    Doc,
}

impl SourceClass {
    /// Parse the wire form. An unknown name is refused, never mapped to the
    /// nearest member: "nearest" is exactly the downgrade this set exists to
    /// prevent.
    ///
    /// # Errors
    /// [`Refusal::UnknownSourceClass`] for any name outside the closed set.
    pub fn parse(name: &str) -> Result<Self, Refusal> {
        match name {
            "lubot" => Ok(Self::Lubot),
            "budlum" => Ok(Self::Budlum),
            "workspace" => Ok(Self::Workspace),
            "doc" => Ok(Self::Doc),
            other => Err(Refusal::UnknownSourceClass(other.to_string())),
        }
    }

    /// The wire form of this class: the source name the corpus builder uses.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Lubot => "lubot",
            Self::Budlum => "budlum",
            Self::Workspace => "workspace",
            Self::Doc => "doc",
        }
    }
}

/// One record a manifest claims, with the provenance pair the corpus schema
/// demands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// The `sha256` of the record's own text; recomputed on the way in.
    pub digest: String,
    /// The text's own content address.
    pub content_id: String,
    /// The asset the text belongs to.
    pub asset_id: String,
    /// One of [`KINDS`].
    pub kind: String,
    /// One of [`LICENCES`].
    pub licence: String,
    /// Who the text is attributed to.
    pub attribution: String,
    /// Where the bytes sit under the data root the caller points at.
    pub path: String,
}

/// A manifest as it arrives from storage.
///
/// Unknown fields are refused rather than ignored: a field this build cannot
/// account for is a field nobody checks, and a manifest that carries one is
/// describing something other than what this code reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// [`SCHEMA`], or the manifest is not one this build reads.
    pub schema: u32,
    /// The content address of the manifest itself.
    pub manifest_id: String,
    /// The wire form of [`SourceClass`].
    pub source_class: String,
    /// Who produced the manifest: an operator, a validator, a tool.
    pub loader: String,
    /// When the manifest was produced, in wall-clock seconds.
    pub created_at: u64,
    /// The records the manifest claims, each one checked before admission.
    pub records: Vec<Entry>,
}

/// What admission produced: the manifest's identity, the records it carried
/// and one digest over the admitted set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Admitted {
    /// The manifest's own content address.
    pub manifest_id: String,
    /// The admitted source class.
    pub source_class: SourceClass,
    /// The loader that produced the manifest.
    pub loader: String,
    /// The manifest's own timestamp.
    pub created_at: u64,
    /// The admitted records, in manifest order.
    pub records: Vec<Entry>,
    /// A digest over [`canonical`] of the admitted records, so two runs over
    /// the same manifest produce the same admission and a changed set cannot
    /// be mistaken for it.
    pub admission_digest: String,
}

/// Parse a manifest from the bytes that arrived.
///
/// # Errors
/// [`Refusal::Malformed`] when the bytes are not a manifest of this schema -
/// including a manifest carrying a field this build does not know.
pub fn parse(bytes: &[u8]) -> Result<Manifest, Refusal> {
    serde_json::from_slice(bytes).map_err(|error| Refusal::Malformed(error.to_string()))
}

/// Admit a manifest, or refuse it at the first rule it breaks.
///
/// `texts` resolves a record's `path` to the bytes that were stored for it;
/// the caller decides where those bytes live, this function decides whether
/// they are the bytes the manifest claims.
///
/// # Errors
/// The first refusal: an unsupported schema, a `manifest_id` that is not a
/// digest, no loader, no records, a source class outside the closed set, or
/// the first record that fails its own check - path shape, kind, licence,
/// attribution, provenance pair, digest shape, missing bytes, digest mismatch.
pub fn admit(manifest: &Manifest, texts: &BTreeMap<String, Vec<u8>>) -> Result<Admitted, Refusal> {
    if manifest.schema != SCHEMA {
        return Err(Refusal::UnsupportedSchema(manifest.schema));
    }
    if !is_sha256_hex(&manifest.manifest_id) {
        return Err(Refusal::BadManifestId(manifest.manifest_id.clone()));
    }
    if manifest.loader.trim().is_empty() {
        return Err(Refusal::MissingLoader);
    }
    let source_class = SourceClass::parse(&manifest.source_class)?;
    if manifest.records.is_empty() {
        return Err(Refusal::EmptyManifest);
    }
    let mut records = Vec::with_capacity(manifest.records.len());
    for entry in &manifest.records {
        check(entry, texts)?;
        records.push(entry.clone());
    }
    let admission_digest = sha256_hex(canonical(&records).as_bytes());
    Ok(Admitted {
        manifest_id: manifest.manifest_id.clone(),
        source_class,
        loader: manifest.loader.clone(),
        created_at: manifest.created_at,
        records,
        admission_digest,
    })
}

/// One record's checks, in the order the refusal should name them.
fn check(entry: &Entry, texts: &BTreeMap<String, Vec<u8>>) -> Result<(), Refusal> {
    if !is_safe_relative(&entry.path) {
        return Err(Refusal::UnsafePath(entry.path.clone()));
    }
    if !KINDS.contains(&entry.kind.as_str()) {
        return Err(Refusal::UnknownKind {
            path: entry.path.clone(),
            kind: entry.kind.clone(),
        });
    }
    if !LICENCES.contains(&entry.licence.as_str()) {
        return Err(Refusal::UnknownLicence {
            path: entry.path.clone(),
            licence: entry.licence.clone(),
        });
    }
    if entry.attribution.trim().is_empty() {
        return Err(Refusal::MissingAttribution(entry.path.clone()));
    }
    if entry.content_id.trim().is_empty() || entry.asset_id.trim().is_empty() {
        return Err(Refusal::MissingProvenancePair(entry.path.clone()));
    }
    if !is_sha256_hex(&entry.digest) {
        return Err(Refusal::BadDigest {
            path: entry.path.clone(),
            digest: entry.digest.clone(),
        });
    }
    let Some(bytes) = texts.get(&entry.path) else {
        return Err(Refusal::MissingBytes(entry.path.clone()));
    };
    let measured = sha256_hex(bytes);
    if measured != entry.digest {
        return Err(Refusal::DigestMismatch {
            path: entry.path.clone(),
            expected: entry.digest.clone(),
            measured,
        });
    }
    Ok(())
}

/// The canonical text the admission digest is taken over: one line per record,
/// fields separated by the unit separator, in manifest order.
///
/// Deliberately not JSON: a change in a serializer's formatting must never be
/// able to move an admission digest.
#[must_use]
pub fn canonical(records: &[Entry]) -> String {
    let mut out = String::new();
    for entry in records {
        for field in [
            entry.content_id.as_str(),
            entry.asset_id.as_str(),
            entry.kind.as_str(),
            entry.licence.as_str(),
            entry.digest.as_str(),
            entry.path.as_str(),
        ] {
            out.push_str(field);
            out.push('\u{1f}');
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{admit, canonical, parse, Entry, Manifest, SourceClass, KINDS, LICENCES, SCHEMA};
    use crate::sha256_hex;

    const TEXT: &str = "Alım hattı bir kaydı bütün kabul eder ya da hiç kabul etmez.\n";

    fn entry(path: &str, text: &str) -> Entry {
        Entry {
            digest: sha256_hex(text.as_bytes()),
            content_id: sha256_hex(text.as_bytes()),
            asset_id: sha256_hex(b"asset"),
            kind: "markdown".to_string(),
            licence: "MIT".to_string(),
            attribution: "lubot".to_string(),
            path: path.to_string(),
        }
    }

    fn manifest(records: Vec<Entry>) -> Manifest {
        Manifest {
            schema: SCHEMA,
            manifest_id: sha256_hex(b"manifest"),
            source_class: "lubot".to_string(),
            loader: "lubot-test".to_string(),
            created_at: 1_700_000_000,
            records,
        }
    }

    fn texts(paths: &[(&str, &str)]) -> BTreeMap<String, Vec<u8>> {
        paths
            .iter()
            .map(|(path, text)| ((*path).to_string(), text.as_bytes().to_vec()))
            .collect()
    }

    #[test]
    fn an_admissible_manifest_is_admitted_as_one() {
        let record = entry("a.md", TEXT);
        let bytes = serde_json::to_vec(&manifest(vec![record.clone()])).expect("serialize");
        let parsed = parse(&bytes).expect("parse");
        let admitted = admit(&parsed, &texts(&[("a.md", TEXT)])).expect("admit");
        assert_eq!(admitted.source_class, SourceClass::Lubot);
        assert_eq!(admitted.records, vec![record]);
        assert_eq!(admitted.admission_digest.len(), 64);
    }

    #[test]
    fn a_class_outside_the_surface_is_refused_by_name() {
        let mut subject = manifest(vec![entry("a.md", TEXT)]);
        subject.source_class = "bud_upload".to_string();
        let refusal = admit(&subject, &texts(&[("a.md", TEXT)])).expect_err("must refuse");
        assert_eq!(refusal.rule(), "K2");
        assert!(refusal.message().contains("bud_upload"));
        assert!(refusal.message().contains("not downgraded"));
    }

    #[test]
    fn an_unknown_field_is_not_ignored() {
        let mut value = serde_json::to_value(manifest(vec![entry("a.md", TEXT)])).expect("json");
        value["extra"] = serde_json::json!("something this build does not read");
        let bytes = serde_json::to_vec(&value).expect("serialize");
        let refusal = parse(&bytes).expect_err("must refuse");
        assert!(refusal.message().contains("malformed manifest"));
    }

    #[test]
    fn a_digest_that_does_not_match_the_bytes_is_refused() {
        let subject = manifest(vec![entry("a.md", TEXT)]);
        let refusal =
            admit(&subject, &texts(&[("a.md", "tampered bytes")])).expect_err("must refuse");
        assert!(refusal.message().contains("digest mismatch"));
        assert!(refusal.message().contains("expected"));
    }

    #[test]
    fn a_licence_outside_the_closed_set_is_refused() {
        let mut record = entry("a.md", TEXT);
        record.licence = "GPL-3.0".to_string();
        let refusal =
            admit(&manifest(vec![record]), &texts(&[("a.md", TEXT)])).expect_err("must refuse");
        assert_eq!(refusal.rule(), "K3");
        assert!(refusal.message().contains("GPL-3.0"));
    }

    #[test]
    fn a_path_that_walks_up_the_tree_is_refused_before_bytes_are_resolved() {
        let record = entry("../a.md", TEXT);
        let refusal =
            admit(&manifest(vec![record]), &texts(&[("../a.md", TEXT)])).expect_err("must refuse");
        assert!(refusal.message().contains("walks up the tree"));
    }

    #[test]
    fn missing_bytes_are_a_refusal_not_a_skip() {
        let refusal =
            admit(&manifest(vec![entry("a.md", TEXT)]), &BTreeMap::new()).expect_err("must refuse");
        assert!(refusal.message().contains("no bytes were resolved"));
    }

    #[test]
    fn the_manifest_level_rules_each_refuse() {
        let mut subject = manifest(vec![entry("a.md", TEXT)]);
        subject.schema = 2;
        assert!(admit(&subject, &texts(&[("a.md", TEXT)]))
            .expect_err("schema")
            .message()
            .contains("unsupported manifest schema"));

        let mut subject = manifest(vec![entry("a.md", TEXT)]);
        subject.manifest_id = "not-a-digest".to_string();
        assert!(admit(&subject, &texts(&[("a.md", TEXT)]))
            .expect_err("id")
            .message()
            .contains("is not a digest"));

        let mut subject = manifest(vec![entry("a.md", TEXT)]);
        subject.loader = "  ".to_string();
        assert!(admit(&subject, &texts(&[("a.md", TEXT)]))
            .expect_err("loader")
            .message()
            .contains("names no loader"));

        let subject = manifest(Vec::new());
        assert!(admit(&subject, &BTreeMap::new())
            .expect_err("empty")
            .message()
            .contains("claims no records"));
    }

    #[test]
    fn a_record_level_rule_each_refuse() {
        let mut record = entry("a.md", TEXT);
        record.kind = "image".to_string();
        assert!(admit(&manifest(vec![record]), &texts(&[("a.md", TEXT)]))
            .expect_err("kind")
            .message()
            .contains("closed four"));

        let mut record = entry("a.md", TEXT);
        record.attribution = String::new();
        assert!(admit(&manifest(vec![record]), &texts(&[("a.md", TEXT)]))
            .expect_err("attribution")
            .message()
            .contains("no attribution"));

        let mut record = entry("a.md", TEXT);
        record.asset_id = String::new();
        assert!(admit(&manifest(vec![record]), &texts(&[("a.md", TEXT)]))
            .expect_err("pair")
            .message()
            .contains("incomplete"));

        let mut record = entry("a.md", TEXT);
        record.digest = "short".to_string();
        assert!(admit(&manifest(vec![record]), &texts(&[("a.md", TEXT)]))
            .expect_err("digest")
            .message()
            .contains("is not a digest"));
    }

    #[test]
    fn the_admission_digest_moves_when_the_set_moves() {
        let first = entry("a.md", TEXT);
        let second = entry("b.md", "ikinci kayıt\n");
        let forward = canonical(&[first.clone(), second.clone()]);
        let backward = canonical(&[second, first.clone()]);
        assert_ne!(forward, backward);
        // The same logical set written twice is the same text, byte for byte.
        assert_eq!(
            forward,
            canonical(&[first, entry("b.md", "ikinci kayıt\n")])
        );
        assert_eq!(sha256_hex(forward.as_bytes()).len(), 64);
    }

    #[test]
    fn the_closed_sets_are_the_corpus_sets() {
        assert_eq!(KINDS.len(), 4);
        assert!(LICENCES.contains(&"PolyForm-Shield-1.0.0"));
        assert!(!KINDS.contains(&"image"));
        assert!(!LICENCES.contains(&"GPL-3.0"));
    }
}
