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

use crate::manifest::Admitted;
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

/// Append rows, one JSON object per line.
///
/// # Errors
/// [`Refusal::Io`] when the ledger cannot be opened or written. A partly
/// written line is impossible: each row is serialized before the write and
/// written with its newline in one call.
pub fn append(path: &Path, rows: &[Row]) -> Result<(), Refusal> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| Refusal::Io(format!("{}: {error}", path.display())))?;
    let mut block = String::new();
    for row in rows {
        let line = serde_json::to_string(row)
            .map_err(|error| Refusal::Io(format!("serialize row: {error}")))?;
        block.push_str(&line);
        block.push('\n');
    }
    file.write_all(block.as_bytes())
        .map_err(|error| Refusal::Io(format!("{}: {error}", path.display())))
}

/// Read the ledger back, whole.
///
/// # Errors
/// [`Refusal::Io`] when the file cannot be read,
/// [`Refusal::MalformedLedger`] when a line is not a row, and
/// [`Refusal::StepWentBackwards`] when the steps are not non-decreasing.
pub fn read(path: &Path) -> Result<Vec<Row>, Refusal> {
    let file = std::fs::File::open(path)
        .map_err(|error| Refusal::Io(format!("{}: {error}", path.display())))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| Refusal::Io(format!("{}: {error}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(&line).map_err(|error| Refusal::MalformedLedger {
            line: line_number(index),
            reason: error.to_string(),
        })?;
        rows.push(row);
    }
    check_monotonic(&rows)?;
    Ok(rows)
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
        if let Some(seen) = previous {
            if row.admitted_step < seen {
                return Err(Refusal::StepWentBackwards {
                    line: line_number(index),
                    step: row.admitted_step,
                    previous: seen,
                });
            }
        }
        previous = Some(row.admitted_step);
    }
    Ok(())
}

/// Line numbers are one-based for the operator; the counter is zero-based.
fn line_number(index: usize) -> u64 {
    u64::try_from(index).unwrap_or(u64::MAX) + 1
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
}
