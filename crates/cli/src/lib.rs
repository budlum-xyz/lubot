#![forbid(unsafe_code)]
//! # lubot-cli - the runnable reader
//!
//! The library behind the `lubot` binary. It is deliberately thin over the
//! five crates: load a corpus (with the same fail-closed rules the corpus was
//! built with), answer through [`lubot_answer::Reader::ask`], render through the single
//! [`Answer::render_markdown`] exit, and keep a grant book plus an output
//! audit on disk.
//!
//! Loading rules, all fail-closed:
//!
//! 1. **Digest re-verification.** Every record carries a SHA-256 of its own
//!    text; a mismatch refuses the file, it does not skip a record.
//! 2. **Provenance pair.** Every record must carry a content id and an asset
//!    id (Aşama 6). A record without them is unanchored and is refused.
//! 3. **Licence.** Every record must name a licence (K3). A record that does
//!    not is refused.
//! 4. **Closed kind set.** `api`, `behaviour`, `doc`, `markdown` are the four
//!    kinds; anything else is refused, so a generating variant cannot slip in
//!    as a corpus kind.
//! 5. **Modality ceiling.** Text is capped at [`lubot_read::perception::MAX_TEXT_BYTES`].
//!
//! The audit file is a JSONL log of every `ask`: reader, question, answer
//! kind, citations, decision and refusals - refusals are recorded in the same
//! shape as allowances, so a log without refusals is a measurement.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod activation;
pub mod graph;
pub mod kosum;
pub mod odeme;
pub mod olcum;
pub mod queue;
pub mod ratchet;
pub mod sikistir;
pub mod soru;

use lubot_answer::Answer;
use lubot_grant::{GrantBook, Seconds, ViewGrant};

/// The licences Lubot may carry: permissive (with attribution) or our own.
/// The set is closed; a corpus record or document carrying anything outside
/// it is refused before its content is read.
pub const ALLOWED_LICENCES: [&str; 3] = ["MIT", "Apache-2.0", "PolyForm-Shield-1.0.0"];
use lubot_read::{perception, Corpus, Item, SourceKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Passages per answer; the reader indexes each record in these line chunks.
pub const LINES_PER_PASSAGE: usize = 16;
/// Best passages an answer may carry.
pub const PASSAGES_PER_ANSWER: usize = 5;
/// The four record kinds the corpus may hold. Anything else is refused.
pub const KINDS: [&str; 4] = ["api", "behaviour", "doc", "markdown"];

/// One corpus record, as the builder wrote it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub kind: String,
    pub text: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub source: String,
    pub digest: String,
    #[serde(default)]
    pub licence: String,
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub content_id: String,
    #[serde(default)]
    pub asset_id: String,
    #[serde(default)]
    pub restricted: bool,
    /// Aşama 6 labels, present only on code records.
    #[serde(default)]
    pub file_tests: u64,
    #[serde(default)]
    pub file_has_readme: bool,
}

impl Record {
    /// The fail-closed acceptance rules, in order.
    #[must_use]
    pub fn refusal(&self) -> Option<String> {
        if !KINDS.contains(&self.kind.as_str()) {
            return Some(format!("unknown kind `{}`", self.kind));
        }
        if self.text.is_empty() {
            return Some("empty text".to_string());
        }
        if let Err(why) = perception::check_units(1, self.text.len() as u64) {
            return Some(format!("text over ceiling: {}", why.label()));
        }
        if let Err(why) = lubot_read::verify_sha256(self.text.as_bytes(), &self.digest) {
            return Some(why);
        }
        if self.content_id.is_empty() || self.asset_id.is_empty() {
            return Some("missing provenance pair (content_id/asset_id)".to_string());
        }
        if self.licence.is_empty() {
            return Some("missing licence".to_string());
        }
        None
    }
}

/// Corpus-level metadata, carried alongside the items for summaries.
#[derive(Debug, Clone)]
pub struct ItemMeta {
    pub id: String,
    pub kind: String,
    pub licence: String,
    pub file_tests: u64,
    pub file_has_readme: bool,
}

/// A corpus held in memory, backed by verified records.
///
/// [`Corpus::get`] is hash-backed, because the reader asks for every id in
/// turn; a linear scan per id would make loading a 100k-record corpus a
/// quadratic operation.
#[derive(Debug, Default)]
pub struct LoadedCorpus {
    items: Vec<Item>,
    meta: Vec<ItemMeta>,
    by_id: HashMap<String, usize>,
}

impl LoadedCorpus {
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[must_use]
    pub fn meta(&self) -> &[ItemMeta] {
        &self.meta
    }
}

impl Corpus for LoadedCorpus {
    fn get(&self, id: &str) -> Option<&Item> {
        self.by_id.get(id).map(|&i| &self.items[i])
    }

    fn ids(&self) -> Vec<String> {
        self.items.iter().map(|i| i.id.clone()).collect()
    }
}

/// Open one `.jsonl.gz` (or plain `.jsonl`) corpus file, verifying every
/// record. A single refused record fails the whole load.
///
/// # Errors
/// The first concerns, with the count of refused records.
pub fn load_corpus(paths: &[PathBuf]) -> Result<LoadedCorpus, String> {
    let mut all = LoadedCorpus::default();
    let mut refused: Vec<String> = Vec::new();
    let mut count = 0usize;
    for path in paths {
        let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
            let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(file)))
        } else {
            let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Box::new(BufReader::new(file))
        };
        for (n, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            count += 1;
            let record: Record = match serde_json::from_str(trimmed) {
                Ok(r) => r,
                Err(e) => {
                    refused.push(format!("{}:{}: {e}", path.display(), n + 1));
                    continue;
                }
            };
            if let Some(why) = record.refusal() {
                refused.push(format!("{}:{}: {why}", path.display(), n + 1));
                continue;
            }
            let origin = if record.path.is_empty() {
                record.source.clone()
            } else {
                record.path.clone()
            };
            let item = Item::new(
                &record.content_id,
                &origin,
                SourceKind::Local,
                record.restricted,
                &record.text,
            );
            all.meta.push(ItemMeta {
                id: record.content_id.clone(),
                kind: record.kind.clone(),
                licence: record.licence.clone(),
                file_tests: record.file_tests,
                file_has_readme: record.file_has_readme,
            });
            all.by_id.insert(record.content_id, all.items.len());
            all.items.push(item);
        }
    }
    if refused.is_empty() {
        return Ok(all);
    }
    let shown: Vec<&str> = refused.iter().take(3).map(String::as_str).collect();
    Err(format!(
        "{}/{} records refused: {}",
        refused.len(),
        count,
        shown.join(" | ")
    ))
}

/// The immutable part of an audit line; the log appends one per `ask`.
#[derive(Debug, Clone, Serialize)]
pub struct AuditLine {
    pub at: Seconds,
    pub reader: String,
    pub question: String,
    pub answer: String,
    pub citations: Vec<String>,
    pub decision: Option<String>,
    pub refusals: usize,
    pub effort: Option<String>,
    pub budget: usize,
}

/// Append any JSONL line to a log file (creates it on first use).
///
/// # Errors
/// Filesystem failures.
pub fn append_json_line(path: &Path, value: &Value) -> Result<(), String> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("log {}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value).map_err(|e| format!("log: {e}"))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|e| format!("log: {e}"))
}

/// Append one audit line as JSONL.
///
/// # Errors
/// Filesystem failures.
pub fn append_audit(path: &Path, line: &AuditLine) -> Result<(), String> {
    let value = serde_json::to_value(line).map_err(|e| format!("audit: {e}"))?;
    append_json_line(path, &value)
}

/// The persisted grant book: view grants plus revoked pairs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BookFile {
    pub grants: Vec<StoredGrant>,
    pub revoked: Vec<(String, String)>,
}

/// A stored grant, without key material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredGrant {
    pub key_id: String,
    pub grantee: String,
    pub expires_at: Seconds,
}

impl BookFile {
    /// Load the book; a missing file is an empty book, never an error,
    /// because a book that has never been issued is a valid state.
    ///
    /// # Errors
    /// Parse failures.
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path).map_err(|e| format!("book: {e}"))?;
        serde_json::from_str(&raw).map_err(|e| format!("book {}: {e}", path.display()))
    }

    /// Save atomically (write to a temp path, then rename).
    ///
    /// # Errors
    /// Filesystem failures.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        let mut content = serde_json::to_vec_pretty(self).map_err(|e| format!("book: {e}"))?;
        content.push(b'\n');
        let mut file = File::create(&tmp).map_err(|e| format!("book: {e}"))?;
        file.write_all(&content).map_err(|e| format!("book: {e}"))?;
        file.sync_all().map_err(|e| format!("book: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("book: {e}"))
    }

    /// Materialise the in-memory grant book.
    #[must_use]
    pub fn to_book(&self) -> GrantBook {
        let mut book = GrantBook::new();
        for stored in &self.grants {
            book.issue(ViewGrant {
                key_id: stored.key_id.clone(),
                grantee: stored.grantee.clone(),
                expires_at: stored.expires_at,
            });
        }
        for (key_id, grantee) in &self.revoked {
            book.issue(ViewGrant {
                key_id: key_id.clone(),
                grantee: grantee.clone(),
                expires_at: 0,
            });
            book.revoke(key_id, grantee);
        }
        book
    }
}

/// The four modality ceilings and the effort range, as one document. The
/// numbers come from the perception module - the ceiling is the constant,
/// not a prose claim about it.
#[must_use]
pub fn ceilings_doc() -> String {
    use lubot_read::perception::{
        MAX_AUDIO_MS, MAX_IMAGE_PIXELS, MAX_TEXT_BYTES, MAX_VIDEO_FRAMES,
    };
    format!(
        "# Ceilings\n\n| modality | ceiling |\n|---|---|\n| text | {MAX_TEXT_BYTES} bytes |\n| image | {MAX_IMAGE_PIXELS} pixels |\n| audio | {MAX_AUDIO_MS} ms |\n| video | {MAX_VIDEO_FRAMES} frames |\n\nEffort range: 0.5x-10.0x\n"
    )
}

/// One batch run: many questions against one loaded corpus, so the index is
/// built once. The report is a single Markdown document; every question gets
/// its own verdict line and the summary counts them. Deterministic order.
/// One bad question is a failure of the whole run - fail-closed, like every
/// other surface here.
///
/// # Errors
/// `Err` for any question that does not produce a valid document, and for
/// question files the caller could not read.
// The arguments are the pipeline's own inputs in the order the report fixes,
// like `ask`; grouping them would hide a step.
#[allow(clippy::too_many_arguments)]
pub fn run_batch(
    corpus: &LoadedCorpus,
    reader: &str,
    questions: &[String],
    grants: &mut GrantBook,
    now: Seconds,
    effort: Option<&str>,
    audit_path: Option<&Path>,
    outputs_path: Option<&Path>,
) -> Result<String, String> {
    if questions.is_empty() {
        return Err("batch run: no questions".to_string());
    }
    let mut doc = String::from("# Batch report\n\n");
    let mut answerable = 0usize;
    for question in questions {
        let (answer, budget) = answer_of(corpus, reader, question, grants, now, effort)?;
        let label = answer_label(&answer);
        let citations = answer.citations().len();
        if matches!(answer, Answer::Grounded { .. } | Answer::Computed { .. }) {
            answerable += 1;
        }
        let markdown = answer
            .render_markdown()
            .map_err(|e| format!("schema: {e}"))?;
        write_trace(
            &answer,
            &markdown,
            budget,
            reader,
            question,
            grants,
            now,
            effort,
            audit_path,
            outputs_path,
        )?;
        doc.push_str(&format!(
            "- `{}` -> {}{}\n",
            question.trim().replace('`', "'"),
            label,
            if citations > 0 {
                format!(", {citations} citation(s)")
            } else {
                String::new()
            }
        ));
    }
    let refused = questions.len() - answerable;
    doc.push_str(&format!(
        "\n{answerable} answerable, {refused} refused/not-found\n"
    ));
    crate::validate_output(doc.as_bytes(), "batch report")?;
    Ok(doc)
}

/// Turn extracted rich-document text into corpus records: chunked,
/// hashed, provenance-tagged. Every record must pass the exact refusal
/// rules the reader enforces, or the whole document is refused.
///
/// # Errors
/// Empty extraction, chunking failure, or any record that fails [`Record::refusal`].
pub fn doc_records(
    text: &str,
    origin: &str,
    source: &str,
    licence: &str,
    attribution: &str,
    asset_id: &str,
    kind: &str,
) -> Result<Vec<Record>, String> {
    if !KINDS.contains(&kind) {
        return Err(format!("doc: unknown kind `{kind}`"));
    }
    let chunks = lubot_doc::chunk_text(text, 4000);
    if chunks.is_empty() {
        return Err("doc: nothing to record".to_string());
    }
    let mut records = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        let digest = lubot_read::sha256_hex(chunk.as_bytes());
        let record = Record {
            kind: kind.to_string(),
            text: chunk.clone(),
            path: format!("{origin}#{index}"),
            source: source.to_string(),
            digest,
            licence: licence.to_string(),
            attribution: attribution.to_string(),
            content_id: format!("{index}"),
            asset_id: asset_id.to_string(),
            restricted: false,
            file_tests: 0,
            file_has_readme: false,
        };
        if let Some(why) = record.refusal() {
            return Err(format!("doc record {}: {why}", index + 1));
        }
        records.push(record);
    }
    Ok(records)
}

/// Write records as one `.jsonl.gz` corpus file.
///
/// # Errors
/// File creation, serialization or gzip failures.
pub fn write_records_gz(path: &Path, records: &[Record]) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    for record in records {
        let line = serde_json::to_string(record).map_err(|e| e.to_string())?;
        enc.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        enc.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    enc.finish().map(|_| ()).map_err(|e| e.to_string())
}

/// The verdict of one ask, with the same audit and closed-loop writes as
/// [`ask`]: the answer label and its citation count. The queue uses this to
/// journal what a job produced.
///
/// # Errors
/// The same pipeline errors `ask` carries.
#[allow(clippy::too_many_arguments)]
pub fn ask_verdict(
    corpus: &LoadedCorpus,
    reader: &str,
    question: &str,
    grants: &mut GrantBook,
    now: Seconds,
    effort: Option<&str>,
    audit_path: Option<&Path>,
    outputs_path: Option<&Path>,
) -> Result<(String, usize), String> {
    let (answer, budget) = answer_of(corpus, reader, question, grants, now, effort)?;
    let markdown = answer
        .render_markdown()
        .map_err(|e| format!("schema: {e}"))?;
    write_trace(
        &answer,
        &markdown,
        budget,
        reader,
        question,
        grants,
        now,
        effort,
        audit_path,
        outputs_path,
    )?;
    Ok((answer_label(&answer).to_string(), answer.citations().len()))
}

/// Validate a document against the answer schema, with a named failure.
///
/// # Errors
/// The schema rejection, named.
pub fn validate_output(bytes: &[u8], name: &str) -> Result<(), String> {
    lubot_read::output_schema::validate_markdown_output(bytes)
        .map_err(|e| format!("{name} rejected by schema: {e}"))
}

/// Wall-clock seconds, fail-closed.
///
/// # Errors
/// Only when the system clock is before the Unix epoch.
pub fn now_seconds() -> Result<Seconds, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|e| format!("clock before unix epoch: {e}"))
}

/// Ask one question against a loaded corpus and render it.
///
/// The order is the report's: tool first, then permission, then search; and
/// the only exit is [`Answer::render_markdown`]. Audit is appended when an
/// audit path is given.
///
/// # Errors
/// Markdown schema failures, audit write failures.
/// The eight parameters are the pipeline's own inputs (corpus, reader,
/// question, grant book, clock, audit, closed loop, effort tier); splitting
/// them into a struct would hide the order the report fixes.
#[allow(clippy::too_many_arguments)]
/// The read step, shared by [`ask`] and [`ask_verdict`]: effort budget,
/// reader view, answer. The order here is the report's - tool first, then
/// permission, then search.
fn answer_of(
    corpus: &LoadedCorpus,
    reader: &str,
    question: &str,
    grants: &mut GrantBook,
    now: Seconds,
    effort: Option<&str>,
) -> Result<(Answer, usize), String> {
    let budget = match effort {
        Some(tag) => lubot_tools::operator::answer_budget(tag)?,
        None => PASSAGES_PER_ANSWER,
    };
    let reader_view = lubot_answer::Reader::new(corpus, LINES_PER_PASSAGE, budget);
    Ok((reader_view.ask(reader, question, grants, now), budget))
}

/// The audit and closed-loop writes shared by [`ask`] and [`ask_verdict`]:
/// one trace per ask, and (Aşama 9) only answers that carry content are
/// finalized - a refusal or a not-found has nothing to register.
// The arguments are the pipeline's own inputs in the order the report fixes;
// grouping them would hide the order, exactly as in `ask`.
#[allow(clippy::too_many_arguments)]
fn write_trace(
    answer: &Answer,
    markdown: &str,
    budget: usize,
    reader: &str,
    question: &str,
    grants: &mut GrantBook,
    now: Seconds,
    effort: Option<&str>,
    audit_path: Option<&Path>,
    outputs_path: Option<&Path>,
) -> Result<(), String> {
    if let Some(path) = audit_path {
        let line = AuditLine {
            at: now,
            reader: reader.to_string(),
            question: question.to_string(),
            answer: answer_label(answer).to_string(),
            citations: answer.citations(),
            decision: decision_word(answer).map(str::to_string),
            refusals: grants.refusals(),
            effort: effort.map(str::to_string),
            budget,
        };
        append_audit(path, &line)?;
    }
    if let Some(path) = outputs_path {
        if matches!(answer, Answer::Grounded { .. } | Answer::Computed { .. }) {
            let record = lubot_answer::output_registry::finalize_output(
                markdown,
                answer_label(answer),
                now,
            )?;
            let line = serde_json::json!({
                "content_id": record.content_id,
                "digest": record.digest,
                "asset_id": record.asset_id,
                "tag": record.tag,
                "kind": record.kind_label,
                "at": record.at,
            });
            append_json_line(path, &line)?;
        }
    }
    Ok(())
}

/// Ask one question against a loaded corpus and render it.
///
/// The order is the report's: tool first, then permission, then search; and
/// the only exit is [`Answer::render_markdown`]. Audit is appended when an
/// audit path is given.
///
/// # Errors
/// Markdown schema failures, audit write failures.
/// The eight parameters are the pipeline's own inputs (corpus, reader,
/// question, grant book, clock, audit, closed loop, effort tier); splitting
/// them into a struct would hide the order the report fixes.
#[allow(clippy::too_many_arguments)]
pub fn ask(
    corpus: &LoadedCorpus,
    reader: &str,
    question: &str,
    grants: &mut GrantBook,
    now: Seconds,
    audit_path: Option<&Path>,
    outputs_path: Option<&Path>,
    effort: Option<&str>,
) -> Result<String, String> {
    let (answer, budget) = answer_of(corpus, reader, question, grants, now, effort)?;
    let markdown = answer
        .render_markdown()
        .map_err(|e| format!("schema: {e}"))?;
    write_trace(
        &answer,
        &markdown,
        budget,
        reader,
        question,
        grants,
        now,
        effort,
        audit_path,
        outputs_path,
    )?;
    Ok(markdown)
}

/// The kind word written to the audit.
#[must_use]
pub fn answer_label(answer: &Answer) -> &'static str {
    match answer {
        Answer::Computed { .. } => "computed",
        Answer::ToolRefused { .. } => "tool-refused",
        Answer::Grounded { .. } => "grounded",
        Answer::NotFound => "not-found",
        Answer::Refused { .. } => "refused",
        Answer::OutOfScope { .. } => "out-of-scope",
    }
}

/// The permission word when the answer is a refusal.
#[must_use]
pub fn decision_word(answer: &Answer) -> Option<&str> {
    match answer {
        Answer::Refused { decision } => Some(decision),
        _ => None,
    }
}

/// Summary of a loaded corpus, for the `corpus` command.
#[must_use]
pub fn corpus_summary(corpus: &LoadedCorpus) -> Value {
    let mut by_kind: HashMap<&str, usize> = HashMap::new();
    let mut by_licence: HashMap<&str, usize> = HashMap::new();
    let mut labelled = 0usize;
    for meta in corpus.meta() {
        *by_kind.entry(meta.kind.as_str()).or_default() += 1;
        *by_licence.entry(meta.licence.as_str()).or_default() += 1;
        if meta.file_tests > 0 || meta.file_has_readme {
            labelled += 1;
        }
    }
    serde_json::json!({
        "items": corpus.len(),
        "by_kind": by_kind,
        "by_licence": by_licence,
        "records_with_labels": labelled,
    })
}

/// Content search directly against a loaded corpus: the top passages for a
/// question, with their citations, origins and licences. The index is the
/// same BM25 surface the reader uses, so `ara`'s answer and `ask`'s answer
/// agree on what the corpus holds.
///
/// # Errors
/// An empty question is refused before any search.
pub fn corpus_search(
    corpus: &LoadedCorpus,
    question: &str,
    limit: usize,
) -> Result<Vec<lubot_index::Passage>, String> {
    if question.trim().is_empty() {
        return Err("ara: question is empty".to_string());
    }
    let mut index = lubot_index::Index::new();
    for id in corpus.ids() {
        if let Some(item) = corpus.get(&id) {
            index.add(item, LINES_PER_PASSAGE);
        }
    }
    let allowed = corpus.ids();
    let mut hits = index.search(question, &allowed, limit.clamp(1, 5));
    hits.sort_by_key(|h| h.citation());
    Ok(hits)
}

/// The index's measured facts over a corpus: records, bytes, kind and
/// licence breakdowns, and the number of origins. Deterministic ordering,
/// so the same corpus always yields the same stats.
pub fn index_stats(corpus: &LoadedCorpus) -> Value {
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_licence: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_origin: BTreeMap<String, usize> = BTreeMap::new();
    let mut bytes = 0usize;
    for id in corpus.ids() {
        if let Some(item) = corpus.get(&id) {
            bytes += item.body.len();
            *by_origin.entry(item.origin.clone()).or_insert(0) += 1;
        }
    }
    for meta in corpus.meta() {
        *by_kind.entry(meta.kind.as_str()).or_insert(0) += 1;
        *by_licence.entry(meta.licence.as_str()).or_insert(0) += 1;
    }
    serde_json::json!({
        "items": corpus.len(),
        "bytes": bytes,
        "by_kind": by_kind,
        "by_licence": by_licence,
        "origins": by_origin.len(),
    })
}

/// The reading plan (müfredat) for a corpus: a deterministic order - text
/// documents first, then every other kind, each group by origin - with the
/// verification binding (digest) of every record. `--out` writes the
/// per-record syllabus; the markdown prints the plan summary and the
/// ordered origin list, so the plan is readable and measurable.
pub fn curriculum_md(corpus: &LoadedCorpus, out: Option<&Path>) -> Result<String, String> {
    struct Entry {
        origin: String,
        kind: String,
        licence: String,
        digest: String,
        bytes: usize,
    }
    let mut entries: Vec<Entry> = Vec::with_capacity(corpus.len());
    for meta in corpus.meta() {
        let item = corpus
            .get(&meta.id)
            .ok_or_else(|| format!("mufredat: metadata id `{}` has no item", meta.id))?;
        entries.push(Entry {
            origin: item.origin.clone(),
            kind: meta.kind.clone(),
            licence: meta.licence.clone(),
            digest: lubot_read::sha256_hex(item.body.as_bytes()),
            bytes: item.body.len(),
        });
    }
    entries.sort_by(|a, b| {
        let rank = |k: &str| if k == "markdown" { 0 } else { 1 };
        rank(&a.kind)
            .cmp(&rank(&b.kind))
            .then_with(|| a.origin.cmp(&b.origin))
            .then_with(|| a.digest.cmp(&b.digest))
    });
    if let Some(path) = out {
        use std::io::Write;
        let mut file =
            std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for entry in &entries {
            let line = serde_json::json!({
                "origin": entry.origin,
                "kind": entry.kind,
                "licence": entry.licence,
                "digest": entry.digest,
                "bytes": entry.bytes,
            });
            writeln!(file, "{line}").map_err(|e| e.to_string())?;
        }
    }
    let mut by_origin: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for entry in &entries {
        let slot = by_origin.entry(&entry.origin).or_insert((0, 0));
        slot.0 += 1;
        slot.1 += entry.bytes;
    }
    let mut md = String::from("# Müfredat\n\n");
    md.push_str(&format!(
        "{} records, {} bytes, {} origins. Reading order: markdown first, then the rest; each group by origin.\n\n",
        entries.len(),
        entries.iter().map(|e| e.bytes).sum::<usize>(),
        by_origin.len()
    ));
    for (origin, (count, bytes)) in &by_origin {
        md.push_str(&format!("- `{origin}`: {count} record(s), {bytes} bytes\n"));
    }
    if out.is_some() {
        md.push_str(&format!(
            "\nFull syllabus written to {} ({} line(s)).\n",
            out.map(|p| p.display().to_string()).unwrap_or_default(),
            entries.len()
        ));
    }
    Ok(md)
}

/// Effort comparison (karsılaştır): the same question, same reader, same
/// corpus, at several effort ceilings, each against its own copy of the
/// grant book - so one effort's refusals never pollute another's. The
/// report shows the budget each ceiling admits and what each run spent.
pub fn compare_efforts(
    corpus: &LoadedCorpus,
    reader: &str,
    question: &str,
    grants: &GrantBook,
    now: Seconds,
    efforts: &[String],
) -> Result<String, String> {
    if efforts.is_empty() {
        return Err("karsilastir: at least one --effort tag is required".to_string());
    }
    let mut md = String::from("# Effort karsilastirma\n\n");
    md.push_str(&format!("Soru: `{question}` | reader: `{reader}`\n\n"));
    let mut seen: Vec<&str> = Vec::new();
    for tag in efforts {
        if seen.contains(&tag.as_str()) {
            return Err(format!("karsilastir: duplicate effort `{tag}`"));
        }
        seen.push(tag);
        let mut fresh = grants.clone();
        let (answer, budget) = answer_of(corpus, reader, question, &mut fresh, now, Some(tag))?;
        let markdown = answer
            .render_markdown()
            .map_err(|e| format!("schema: {e}"))?;
        md.push_str(&format!(
            "- `{tag}`: budget {} passage(s) -> **{}**, {} citation(s), {} bytes\n",
            budget,
            answer_label(&answer),
            answer.citations().len(),
            markdown.len()
        ));
    }
    validate_output(md.as_bytes(), "karsilastir")?;
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_grant::{Decision, Visibility};

    fn record(id: &str, body: &str) -> Record {
        Record {
            kind: "markdown".to_string(),
            text: body.to_string(),
            path: format!("docs/{id}.md"),
            source: "test".to_string(),
            digest: lubot_read::sha256_hex(body.as_bytes()),
            licence: "MIT".to_string(),
            attribution: "test".to_string(),
            content_id: id.to_string(),
            asset_id: "a".repeat(64),
            restricted: false,
            file_tests: 0,
            file_has_readme: false,
        }
    }

    fn write_gz(path: &Path, records: &[Record]) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut enc = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        for record in records {
            let line = serde_json::to_string(record).map_err(|e| e.to_string())?;
            enc.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
            enc.write_all(b"\n").map_err(|e| e.to_string())?;
        }
        enc.finish().map_err(|e| e.to_string())?;
        Ok(())
    }

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lubot-cli-test-{}-{}", std::process::id(), name))
    }

    #[test]
    fn load_accepts_verified_records_and_computes_digest() {
        let path = tmp("ok.jsonl.gz");
        let r = record("1", "A view grant names a grantee and a key id.");
        write_gz(&path, &[r]).unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        assert_eq!(corpus.len(), 1);
        drop(corpus);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_refuses_a_digest_mismatch() {
        let path = tmp("bad-digest.jsonl.gz");
        let mut r = record("2", "the body says one thing");
        r.digest = "0".repeat(64);
        write_gz(&path, &[r]).unwrap();
        let err = load_corpus(std::slice::from_ref(&path)).unwrap_err();
        assert!(err.contains("digest mismatch"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_refuses_a_record_without_provenance() {
        let path = tmp("no-provenance.jsonl.gz");
        let mut r = record("3", "no anchor");
        r.asset_id.clear();
        write_gz(&path, &[r]).unwrap();
        let err = load_corpus(std::slice::from_ref(&path)).unwrap_err();
        assert!(err.contains("provenance"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_refuses_an_unknown_kind() {
        let path = tmp("bad-kind.jsonl.gz");
        let mut r = record("4", "not a real kind");
        r.kind = "invented".to_string();
        write_gz(&path, &[r]).unwrap();
        let err = load_corpus(std::slice::from_ref(&path)).unwrap_err();
        assert!(err.contains("unknown kind"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_refuses_text_over_the_ceiling() {
        let path = tmp("over.jsonl.gz");
        let big = "x".repeat(perception::MAX_TEXT_BYTES as usize + 1);
        let mut r = record("5", &big);
        r.digest = lubot_read::sha256_hex(r.text.as_bytes());
        write_gz(&path, &[r]).unwrap();
        let err = load_corpus(std::slice::from_ref(&path)).unwrap_err();
        assert!(err.contains("ceiling"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_grounded_question_returns_rendered_markdown() {
        let path = tmp("grounded.jsonl.gz");
        let r = record("6", "Revocation stops new opens, not old reads.");
        write_gz(&path, &[r]).unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let mut grants = GrantBook::new();
        let md = ask(
            &corpus,
            "reader",
            "what does revocation stop?",
            &mut grants,
            1,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(md.starts_with("# Answer"), "{md}");
        assert!(md.contains("Revocation stops new opens"), "{md}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn arithmetic_never_touches_the_corpus() {
        let path = tmp("arith.jsonl.gz");
        let r = record("7", "unrelated");
        write_gz(&path, &[r]).unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let mut grants = GrantBook::new();
        let md = ask(
            &corpus,
            "reader",
            "12 * 12 = ?",
            &mut grants,
            1,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(md.starts_with("## calculator"), "{md}");
        assert!(md.contains("144"), "{md}");
        assert!(grants.audit().is_empty(), "the tool path opens nothing");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_restricted_item_is_refused_until_a_grant_opens_it() {
        let path = tmp("restricted.jsonl.gz");
        let mut r = record("8", "a private settlement note");
        r.restricted = true;
        write_gz(&path, &[r]).unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let mut grants = GrantBook::new();
        let md = ask(
            &corpus,
            "reader",
            "settlement note",
            &mut grants,
            1,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(md.starts_with("# Refused"), "{md}");
        assert_eq!(grants.refusals(), 1);
        grants.issue(ViewGrant {
            key_id: "8".to_string(),
            grantee: "reader".to_string(),
            expires_at: 100,
        });
        let md = ask(
            &corpus,
            "reader",
            "settlement note",
            &mut grants,
            10,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(md.starts_with("# Answer"), "{md}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_book_file_round_trips_grants_and_revocations() {
        let path = tmp("book.json");
        let book = BookFile {
            grants: vec![StoredGrant {
                key_id: "dm-1".to_string(),
                grantee: "reader".to_string(),
                expires_at: 200,
            }],
            revoked: vec![("dm-2".to_string(), "reader".to_string())],
        };
        book.save(&path).unwrap();
        let loaded = BookFile::load(&path).unwrap();
        let mut grants = loaded.to_book();
        assert_eq!(
            grants.decide("reader", "dm-1", Visibility::Restricted, 150),
            Decision::Granted
        );
        assert_eq!(
            grants.decide("reader", "dm-2", Visibility::Restricted, 150),
            Decision::Revoked
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_outputs_file_carries_the_finalized_handoff() {
        let path = tmp("corpus.jsonl.gz");
        let r = record("9", "The finalized output is a Markdown document.");
        write_gz(&path, &[r]).unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let out = tmp("outputs.jsonl");
        let mut grants = GrantBook::new();
        let _ = ask(
            &corpus,
            "reader",
            "what is the finalized output?",
            &mut grants,
            1,
            None,
            Some(out.as_path()),
            None,
        )
        .unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        let value: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(value["tag"], "ai-inference");
        assert_eq!(value["content_id"], value["digest"]);
        assert_eq!(value["kind"], "grounded");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn audit_lines_are_appended_and_valid_json() {
        let path = tmp("audit.jsonl");
        let line = AuditLine {
            at: 7,
            reader: "reader".to_string(),
            question: "q".to_string(),
            answer: "not-found".to_string(),
            citations: vec![],
            decision: None,
            refusals: 0,
            effort: None,
            budget: 5,
        };
        append_audit(&path, &line).unwrap();
        append_audit(&path, &line).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2);
        for l in text.lines() {
            let parsed: Value = serde_json::from_str(l).unwrap();
            assert_eq!(parsed["answer"], "not-found");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn search_finds_the_right_passage_and_refuses_an_empty_question() {
        let path = tmp("search.jsonl.gz");
        write_gz(
            &path,
            &[
                record(
                    "s1",
                    "The private settlement note mentions a schedule of 3 payments.",
                ),
                record("b1", "A blob is a binary large object stored in a bucket."),
            ],
        )
        .unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let hits = corpus_search(&corpus, "settlement schedule", 3).unwrap();
        assert!(!hits.is_empty(), "the settlement passage must be found");
        assert!(hits.iter().any(|h| h.text.contains("settlement")));
        let err = corpus_search(&corpus, "   ", 3).unwrap_err();
        assert!(err.contains("empty"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn index_stats_are_deterministic() {
        let path = tmp("stats.jsonl.gz");
        write_gz(
            &path,
            &[
                record("s1", "A grant names a grantee."),
                record("s2", "The epoch ledger is fail-closed."),
            ],
        )
        .unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let a = index_stats(&corpus);
        let b = index_stats(&corpus);
        assert_eq!(a, b);
        assert_eq!(a["items"], 2);
        assert_eq!(a["by_kind"]["markdown"], 2);
        assert_eq!(a["by_licence"]["MIT"], 2);
        assert!(a["origins"].as_u64().unwrap() >= 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn curriculum_lists_markdown_first_with_digests() {
        let path = tmp("mufredat.jsonl.gz");
        let mut code = record("c1", "pub fn repay(amount: u64) -> u64 { amount + 1 }");
        code.kind = "doc".to_string();
        code.path = "src/lib.rs".to_string();
        write_gz(
            &path,
            &[
                code,
                record("m1", "# Design\n\nThe engine batches by ledger day."),
            ],
        )
        .unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let syllabus = tmp("mufredat-out.jsonl");
        let md = curriculum_md(&corpus, Some(&syllabus)).unwrap();
        assert!(md.contains("markdown first"), "{md}");
        assert!(md.contains("2 records"), "{md}");
        let out = std::fs::read_to_string(&syllabus).unwrap();
        assert_eq!(out.lines().count(), 2);
        let first: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
        assert_eq!(first["digest"].as_str().unwrap().len(), 64);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&syllabus);
    }

    #[test]
    fn compare_efforts_reports_budgets_even_when_the_answer_is_empty() {
        use lubot_grant::GrantBook;
        let path = tmp("karsilastir.jsonl.gz");
        write_gz(
            &path,
            &[record(
                "m1",
                "# Design\n\nThe engine batches by ledger day.",
            )],
        )
        .unwrap();
        let corpus = load_corpus(std::slice::from_ref(&path)).unwrap();
        let mut grants = GrantBook::default();
        for i in 0..3 {
            grants.issue(lubot_grant::ViewGrant {
                key_id: format!("key-{i}"),
                grantee: "reader".to_string(),
                expires_at: 9_999_999_999,
            });
        }
        let md = compare_efforts(
            &corpus,
            "reader",
            "what is the text ceiling?",
            &grants,
            1_768_000_000,
            &["0.5x".to_string(), "5.0x".to_string()],
        )
        .unwrap();
        assert!(md.contains("budget 2 passage(s)"), "{md}");
        assert!(md.contains("budget 10 passage(s)"), "{md}");
        let err = compare_efforts(
            &corpus,
            "reader",
            "q?",
            &grants,
            1_768_000_000,
            &["1.0x".to_string(), "1.0x".to_string()],
        )
        .unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
        let _ = std::fs::remove_file(&path);
    }
}
