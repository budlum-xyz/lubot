//! The provenance ledger: one row per admitted record, append-only.
//!
//! Directive step 15 asks for four fields per training input - which manifest,
//! which loader, when it was verified, which training step it entered - and
//! K3 asks that provenance be recorded per record rather than per run. A row
//! here is exactly that, and nothing else: no byte counts that were not
//! measured, no summary that stands in for the rows.
//!
//! Two rules make the ledger usable as evidence rather than as a log:
//!
//! - **Whole or refused.** A line that is not a row refuses the file. A ledger
//!   read in part would let a gap pass as continuity.
//! - **Steps never go backwards.** A row's `admitted_step` may equal or exceed
//!   the row before it. An intake that claims to feed an earlier step than an
//!   intake that already happened is describing a history that never ran.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::manifest::{Admitted, KINDS, LICENCES};
use crate::Refusal;

/// One admitted record's provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// The manifest the record arrived in.
    pub manifest_id: String,
    /// Who produced that manifest.
    pub loader: String,
    /// When the bytes were verified, in wall-clock seconds.
    pub verified_at: u64,
    /// The training step this intake feeds.
    pub admitted_step: u64,
    /// The record's content address.
    pub content_id: String,
    /// Where the record's bytes sit.
    pub path: String,
    /// The record's kind.
    pub kind: String,
    /// The record's licence.
    pub licence: String,
}

/// One row per admitted record, in manifest order.
#[must_use]
pub fn rows_for(admitted: &Admitted, verified_at: u64, admitted_step: u64) -> Vec<Row> {
    admitted
        .records
        .iter()
        .map(|entry| Row {
            manifest_id: admitted.manifest_id.clone(),
            loader: admitted.loader.clone(),
            verified_at,
            admitted_step,
            content_id: entry.content_id.clone(),
            path: entry.path.clone(),
            kind: entry.kind.clone(),
            licence: entry.licence.clone(),
        })
        .collect()
}

/// Append a validated batch to a validated ledger, one JSON object per line.
///
/// Single-writer contract: the caller must serialize writers to this path.
/// This function does not provide cross-process locking or a multi-file
/// transaction. Serialization is not filesystem atomicity: an I/O failure
/// may leave torn JSON, which the next read refuses rather than skips. A cut
/// exactly between complete rows cannot be detected without an outside anchor.
///
/// # Errors
/// Invalid rows or backwards steps refuse before opening the write handle.
/// [`Refusal::Io`] names read, write or sync failures. An empty batch creates
/// no file, but still checks an existing ledger before reporting success.
pub fn append(path: &Path, rows: &[Row]) -> Result<(), Refusal> {
    let previous = match std::fs::File::open(path) {
        Ok(file) => read_rows(BufReader::new(file))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => LedgerRead {
            rows: Vec::new(),
            physical_lines: 0,
            newline: true,
        },
        Err(error) => return Err(Refusal::Io(format!("{}: {error}", path.display()))),
    };
    let mut last_step = previous.rows.last().map(|row| row.admitted_step);
    let mut block = String::new();
    if !rows.is_empty() && !previous.newline {
        // A complete final JSON object without LF is valid input. Complete
        // its framing without rewriting its bytes before adding another row.
        block.push('\n');
    }
    for (index, row) in rows.iter().enumerate() {
        let line = previous.physical_lines.saturating_add(line_number(index));
        check_row(row, line)?;
        check_previous(row, last_step, line)?;
        let encoded = serde_json::to_string(row)
            .map_err(|error| Refusal::Io(format!("serialize row: {error}")))?;
        block.push_str(&encoded);
        block.push('\n');
        last_step = Some(row.admitted_step);
    }
    if block.is_empty() {
        return Ok(());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| Refusal::Io(format!("{}: {error}", path.display())))?;
    file.write_all(block.as_bytes())
        .and_then(|()| file.sync_data())
        .map_err(|error| Refusal::Io(format!("{}: {error}", path.display())))
}

struct LedgerRead {
    rows: Vec<Row>,
    physical_lines: u64,
    newline: bool,
}

fn check_row(row: &Row, line: u64) -> Result<(), Refusal> {
    let reason = if !crate::is_sha256_hex(&row.manifest_id) {
        Some("manifest_id is not a sha256 digest")
    } else if row.loader.trim().is_empty() {
        Some("loader is empty")
    } else if row.content_id.trim().is_empty() {
        // The manifest contract treats content_id as an opaque, nonempty id;
        // do not silently turn that into a new digest-only schema here.
        Some("content_id is empty")
    } else if !crate::is_safe_relative(&row.path) {
        Some("path is not safe and relative")
    } else if !KINDS.contains(&row.kind.as_str()) {
        Some("kind is outside the admitted set")
    } else if !LICENCES.contains(&row.licence.as_str()) {
        Some("licence is outside the admitted set")
    } else {
        None
    };
    match reason {
        Some(reason) => Err(Refusal::MalformedLedger {
            line,
            reason: reason.to_string(),
        }),
        None => Ok(()),
    }
}

fn check_previous(row: &Row, previous: Option<u64>, line: u64) -> Result<(), Refusal> {
    if let Some(seen) = previous {
        if row.admitted_step < seen {
            return Err(Refusal::StepWentBackwards {
                line,
                step: row.admitted_step,
                previous: seen,
            });
        }
    }
    Ok(())
}

fn read_rows(mut reader: impl BufRead) -> Result<LedgerRead, Refusal> {
    let mut result = LedgerRead {
        rows: Vec::new(),
        physical_lines: 0,
        newline: true,
    };
    let mut line = String::new();
    loop {
        line.clear();
        let count = reader
            .read_line(&mut line)
            .map_err(|error| Refusal::Io(error.to_string()))?;
        if count == 0 {
            break;
        }
        result.physical_lines = result.physical_lines.saturating_add(1);
        result.newline = line.ends_with('\n');
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(&line).map_err(|error| Refusal::MalformedLedger {
            line: result.physical_lines,
            reason: error.to_string(),
        })?;
        check_row(&row, result.physical_lines)?;
        check_previous(
            &row,
            result.rows.last().map(|r| r.admitted_step),
            result.physical_lines,
        )?;
        result.rows.push(row);
    }
    Ok(result)
}

/// Read the ledger back, whole, preserving physical line numbers in refusals.
///
/// # Errors
/// [`Refusal::Io`] when the file cannot be read,
/// [`Refusal::MalformedLedger`] when JSON or row metadata is invalid, and
/// [`Refusal::StepWentBackwards`] when steps are not non-decreasing.
pub fn read(path: &Path) -> Result<Vec<Row>, Refusal> {
    let file = std::fs::File::open(path)
        .map_err(|error| Refusal::Io(format!("{}: {error}", path.display())))?;
    Ok(read_rows(BufReader::new(file))?.rows)
}

/// The step a new intake may claim against a ledger that already exists.
///
/// # Errors
/// [`Refusal::StepWentBackwards`] when `step` is behind the last row's step.
pub fn check_step(rows: &[Row], step: u64) -> Result<(), Refusal> {
    match rows.last() {
        Some(last) if step < last.admitted_step => Err(Refusal::StepWentBackwards {
            line: u64::try_from(rows.len()).unwrap_or(u64::MAX),
            step,
            previous: last.admitted_step,
        }),
        _ => Ok(()),
    }
}

/// Steps must never go backwards.
///
/// # Errors
/// [`Refusal::StepWentBackwards`] at the first row that is behind the one
/// before it.
pub fn check_monotonic(rows: &[Row]) -> Result<(), Refusal> {
    let mut previous: Option<u64> = None;
    for (index, row) in rows.iter().enumerate() {
        check_previous(row, previous, line_number(index))?;
        previous = Some(row.admitted_step);
    }
    Ok(())
}

/// Line numbers are one-based for the operator; the counter is zero-based.
fn line_number(index: usize) -> u64 {
    u64::try_from(index).unwrap_or(u64::MAX).saturating_add(1)
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::PathBuf;

    use super::{append, check_monotonic, check_step, read, rows_for, Row};
    use crate::manifest::{Admitted, Entry, SourceClass};
    use crate::sha256_hex;

    fn admitted() -> Admitted {
        let make = |path: &str| Entry {
            digest: sha256_hex(path.as_bytes()),
            content_id: sha256_hex(path.as_bytes()),
            asset_id: sha256_hex(b"asset"),
            kind: "markdown".to_string(),
            licence: "MIT".to_string(),
            attribution: "lubot".to_string(),
            path: path.to_string(),
        };
        Admitted {
            manifest_id: sha256_hex(b"manifest"),
            source_class: SourceClass::Lubot,
            loader: "lubot-test".to_string(),
            created_at: 1_700_000_000,
            records: vec![make("a.md"), make("b.md")],
            admission_digest: sha256_hex(b"admission"),
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("lubot-alim-{name}-{}", std::process::id()));
        path
    }

    #[test]
    fn one_row_per_record_carries_the_four_fields() {
        let rows = rows_for(&admitted(), 1_700_000_100, 7);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].admitted_step, 7);
        assert_eq!(rows[0].verified_at, 1_700_000_100);
        assert_eq!(rows[0].loader, "lubot-test");
        assert_eq!(rows[1].path, "b.md");
        assert_eq!(rows[0].manifest_id, rows[1].manifest_id);
    }

    #[test]
    fn appended_rows_read_back_whole() {
        let path = scratch("roundtrip.jsonl");
        let _ = std::fs::remove_file(&path);
        let rows = rows_for(&admitted(), 1_700_000_100, 1);
        append(&path, &rows).expect("append");
        append(&path, &rows_for(&admitted(), 1_700_000_200, 2)).expect("append");
        let read_back = read(&path).expect("read");
        assert_eq!(read_back.len(), 4);
        assert_eq!(read_back[3].admitted_step, 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_malformed_line_refuses_the_whole_ledger() {
        let path = scratch("malformed.jsonl");
        let rows = rows_for(&admitted(), 1, 1);
        append(&path, &rows).expect("append");
        let mut file = OpenOptions::new().append(true).open(&path).expect("open");
        file.write_all(b"not a row\n").expect("write");
        let refusal = read(&path).expect_err("must refuse");
        assert!(refusal.message().contains("not a row"));
        assert!(refusal.message().contains("line 3"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn steps_that_go_backwards_are_refused() {
        let path = scratch("backwards.jsonl");
        let rows = rows_for(&admitted(), 1, 5);
        append(&path, &rows).expect("append");
        assert!(check_step(&rows, 5).is_ok());
        assert!(check_step(&rows, 6).is_ok());
        let refusal = check_step(&rows, 4).expect_err("must refuse");
        assert!(refusal.message().contains("behind the previous 5"));

        let mut first = Row {
            manifest_id: "m".to_string(),
            loader: "l".to_string(),
            verified_at: 1,
            admitted_step: 9,
            content_id: "c".to_string(),
            path: "a.md".to_string(),
            kind: "markdown".to_string(),
            licence: "MIT".to_string(),
        };
        let mut second = first.clone();
        second.admitted_step = 8;
        assert!(check_monotonic(&[first.clone()]).is_ok());
        assert!(check_monotonic(&[first.clone(), second]).is_err());
        first.admitted_step = 8;
        assert!(check_monotonic(&[first]).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unknown_field_in_a_row_is_refused() {
        let path = scratch("unknown.jsonl");
        std::fs::write(
            &path,
            "{\"manifest_id\":\"m\",\"loader\":\"l\",\"verified_at\":1,\"admitted_step\":1,\
             \"content_id\":\"c\",\"path\":\"a.md\",\"kind\":\"markdown\",\"licence\":\"MIT\",\
             \"guessed\":true}\n",
        )
        .expect("write");
        let refusal = read(&path).expect_err("must refuse");
        assert!(refusal.message().contains("not a row"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_ledger_is_an_io_refusal_not_an_empty_one() {
        let refusal = read(&scratch("absent.jsonl")).expect_err("must refuse");
        assert!(refusal.message().starts_with("[-] i/o:"));
    }

    #[test]
    fn append_rejects_backwards_batch_without_creating_file() {
        let path = scratch("batch-order.jsonl");
        let _ = std::fs::remove_file(&path);
        let mut rows = rows_for(&admitted(), 1, 4);
        rows[1].admitted_step = 3;
        assert!(matches!(
            append(&path, &rows),
            Err(crate::Refusal::StepWentBackwards { line: 2, .. })
        ));
        assert!(!path.exists());
    }

    #[test]
    fn append_rejects_backwards_boundary_without_changing_bytes() {
        let path = scratch("boundary-order.jsonl");
        let _ = std::fs::remove_file(&path);
        append(&path, &rows_for(&admitted(), 1, 5)).expect("initial");
        let before = std::fs::read(&path).expect("before");
        assert!(matches!(
            append(&path, &rows_for(&admitted(), 2, 4)),
            Err(crate::Refusal::StepWentBackwards { line: 3, .. })
        ));
        assert_eq!(std::fs::read(&path).expect("after"), before);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn append_refuses_corrupt_existing_ledger_without_changing_bytes() {
        let path = scratch("corrupt-existing.jsonl");
        for bytes in [b"not-json\n".as_slice(), b"{\"manifest_id\":"] {
            std::fs::write(&path, bytes).expect("fixture");
            assert!(append(&path, &rows_for(&admitted(), 1, 1)).is_err());
            assert_eq!(std::fs::read(&path).expect("after"), bytes);
        }
        std::fs::remove_file(path).expect("cleanup");
    }

    fn invalid_rows() -> Vec<Row> {
        let row = rows_for(&admitted(), 1, 1).remove(0);
        let mut variants = vec![row; 6];
        variants[0].manifest_id = "not-a-digest".to_string();
        variants[1].loader = " \t".to_string();
        variants[2].content_id = " ".to_string();
        variants[3].path = "../outside".to_string();
        variants[4].kind = "unapproved".to_string();
        variants[5].licence = "unapproved".to_string();
        variants
    }

    #[test]
    fn read_refuses_each_invalid_metadata_field() {
        let path = scratch("metadata-read.jsonl");
        for row in invalid_rows() {
            std::fs::write(&path, serde_json::to_vec(&row).expect("serialize")).expect("fixture");
            assert!(
                matches!(
                    read(&path),
                    Err(crate::Refusal::MalformedLedger { line: 1, .. })
                ),
                "{row:?}"
            );
        }
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn append_refuses_invalid_later_row_without_creating_file() {
        let path = scratch("metadata-batch.jsonl");
        let _ = std::fs::remove_file(&path);
        for row in invalid_rows() {
            let valid = rows_for(&admitted(), 1, 1).remove(0);
            assert!(matches!(
                append(&path, &[valid, row]),
                Err(crate::Refusal::MalformedLedger { line: 2, .. })
            ));
            assert!(!path.exists());
        }
    }

    #[test]
    fn read_preserves_physical_line_number_across_blank_lines() {
        let path = scratch("physical-line.jsonl");
        let first = serde_json::to_string(&rows_for(&admitted(), 1, 3)[0]).expect("serialize");
        let second = serde_json::to_string(&rows_for(&admitted(), 1, 2)[0]).expect("serialize");
        std::fs::write(&path, format!("\n{first}\n \t\n{second}\n")).expect("fixture");
        assert!(matches!(
            read(&path),
            Err(crate::Refusal::StepWentBackwards { line: 4, .. })
        ));
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn append_completes_missing_newline_without_rewriting_prefix() {
        let path = scratch("missing-newline.jsonl");
        let row = rows_for(&admitted(), 1, 1).remove(0);
        let before = serde_json::to_vec(&row).expect("serialize");
        std::fs::write(&path, &before).expect("fixture");
        append(&path, &rows_for(&admitted(), 2, 2)).expect("append");
        assert!(std::fs::read(&path).expect("bytes").starts_with(&before));
        assert_eq!(read(&path).expect("read").len(), 3);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn empty_batch_does_not_create_a_ledger() {
        let path = scratch("empty-batch.jsonl");
        let _ = std::fs::remove_file(&path);
        append(&path, &[]).expect("empty");
        assert!(!path.exists());
    }

    #[test]
    fn opaque_content_id_and_equal_steps_stay_compatible() {
        let path = scratch("opaque-id.jsonl");
        let _ = std::fs::remove_file(&path);
        let mut rows = rows_for(&admitted(), 0, 0);
        rows[0].content_id = "opaque-admitted-id".to_string();
        append(&path, &rows).expect("append");
        append(&path, &rows).expect("equal steps");
        assert_eq!(read(&path).expect("read"), [rows.clone(), rows].concat());
        std::fs::remove_file(path).expect("cleanup");
    }
}
