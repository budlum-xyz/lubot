//! # lubot-read::output_schema - the shape a reply must keep
//!
//! Every reply this system produces is Markdown, and a reply that fails the
//! schema check is **rejected and regenerated**, never downgraded to a format
//! it was not. This module is the schema; the assembly applies it before a
//! reply leaves, so a malformed reply never reaches a citation or a record.
//!
//! The checks are deliberately the ones a machine can agree on:
//!
//! 1. strict UTF-8 (no replacement of undecodable bytes);
//! 2. non-empty after trimming;
//! 3. heading hierarchy does not skip a level while descending;
//! 4. code fences are balanced (`CommonMark` toggling);
//! 5. tables are well formed: separator row after the header, consistent
//!    column count.
//!
//! Nothing here judges prose. A single paragraph is valid Markdown; the rule
//! is about the shape the contract promises, not about style.

use std::fmt;

/// Why an output was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputSchemaError {
    NotUtf8,
    Empty,
    HeadingSkip { line: usize, from: usize, to: usize },
    UnbalancedFence { line: usize },
    TableMismatch { line: usize },
    TooLarge { bytes: usize },
}

impl fmt::Display for OutputSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotUtf8 => write!(f, "output is not valid UTF-8"),
            Self::Empty => write!(f, "output is empty"),
            Self::HeadingSkip { line, from, to } => write!(
                f,
                "heading on line {line} skips a level: {} after {}",
                "#".repeat(*to),
                "#".repeat(*from)
            ),
            Self::UnbalancedFence { line } => write!(f, "unbalanced code fence around line {line}"),
            Self::TableMismatch { line } => write!(f, "malformed table around line {line}"),
            Self::TooLarge { bytes } => write!(f, "output too large: {bytes} bytes"),
        }
    }
}

impl std::error::Error for OutputSchemaError {}

/// The analyzer's own ceiling, independent of any wrapper's byte limit.
pub const MAX_OUTPUT_BYTES: usize = 64 * 1024;

/// Length of a line of `|`-separated cells, or 0 when the line is not a row.
fn row_columns(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    if !(trimmed.starts_with('|') && trimmed.ends_with('|')) {
        return None;
    }
    Some(trimmed.matches('|').count() - 1)
}

/// Is this a separator row (`|---|---|`, `|:--|`, `|---:|` style)?
fn is_separator_row(line: &str) -> bool {
    let Some(cols) = row_columns(line) else {
        return false;
    };
    if cols == 0 {
        return false;
    }
    let cells = line
        .trim()
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect::<Vec<_>>();
    cells.iter().all(|c| {
        !c.is_empty() && c.contains('-') && c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')
    })
}

/// The fence to wrap `text` in so it stays one code block.
///
/// A passage being quoted may itself contain fence lines - reading a document
/// about code means reading fence lines - and `CommonMark` closes a fence only
/// with a run at least as long as the opener. Choosing the wrapper from the content
/// keeps the quotation faithful: nothing inside is rewritten, the fence simply
/// grows around it.
#[must_use]
pub fn fence_for(text: &str) -> String {
    let longest = text
        .lines()
        .map(|line| line.trim_start().chars().take_while(|c| *c == '`').count())
        .max()
        .unwrap_or(0);
    "`".repeat(longest.max(2) + 1)
}

/// Validate a reply against the Markdown schema.
///
/// # Errors
///
/// [`OutputSchemaError`] naming the first violation. The caller refuses the
/// reply; it is never converted into a near-miss format.
pub fn validate_markdown_output(bytes: &[u8]) -> Result<(), OutputSchemaError> {
    if bytes.len() > MAX_OUTPUT_BYTES {
        return Err(OutputSchemaError::TooLarge { bytes: bytes.len() });
    }
    let text = std::str::from_utf8(bytes).map_err(|_| OutputSchemaError::NotUtf8)?;
    if text.trim().is_empty() {
        return Err(OutputSchemaError::Empty);
    }

    let mut level: usize = 0;
    let mut seen_heading = false;
    // CommonMark: a fence opened with N backticks is closed only by a line of
    // backticks at least that long with nothing else on it. Modelling that
    // faithfully is what lets a quotation contain a shorter fence - a passage
    // that talks about code may well contain ``` - without the document
    // silently unbalancing. The alternative, stripping the passage, would
    // change what was read; the schema refuses rather than rewrites.
    let mut in_fence = false;
    let mut fence_open_line = 0usize;
    let mut fence_len = 0usize;
    let mut table_buffer: Vec<(usize, String)> = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw.trim();

        // Fence toggling before anything else: inside a fence nothing is a
        // heading or a table row. A closing fence has to be at least as long as
        // the opening one and carry nothing else; a shorter run of backticks is
        // content.
        if trimmed.starts_with("```") {
            let run = trimmed.chars().take_while(|c| *c == '`').count();
            let only_backticks = trimmed[run..].trim().is_empty();
            if !in_fence {
                in_fence = true;
                fence_open_line = line_no;
                fence_len = run;
            } else if run >= fence_len && only_backticks {
                in_fence = false;
            }
            if !table_buffer.is_empty() {
                check_table(&table_buffer)?;
                table_buffer.clear();
            }
            continue;
        }
        if in_fence {
            continue;
        }

        // Heading: a run of '#'s followed by a space, at the start of the line.
        let lead_hashes = trimmed.chars().take_while(|c| *c == '#').count();
        if lead_hashes > 0 && trimmed[lead_hashes..].starts_with(' ') {
            let new_level = lead_hashes;
            // The first heading may be any level; only a *descent* is bounded.
            if seen_heading && new_level > level + 1 {
                return Err(OutputSchemaError::HeadingSkip {
                    line: line_no,
                    from: level,
                    to: new_level,
                });
            }
            seen_heading = true;
            level = new_level;
            if !table_buffer.is_empty() {
                check_table(&table_buffer)?;
                table_buffer.clear();
            }
            continue;
        }

        // Table rows accumulate; a blank or non-row line closes the block.
        let is_row = row_columns(trimmed).is_some();
        if is_row {
            table_buffer.push((line_no, trimmed.to_string()));
            continue;
        }
        if !table_buffer.is_empty() {
            check_table(&table_buffer)?;
            table_buffer.clear();
        }
    }

    if in_fence {
        return Err(OutputSchemaError::UnbalancedFence {
            line: fence_open_line,
        });
    }
    if !table_buffer.is_empty() {
        check_table(&table_buffer)?;
    }
    Ok(())
}

/// Validate one table block: header, separator, consistent columns.
fn check_table(rows: &[(usize, String)]) -> Result<(), OutputSchemaError> {
    let (first_line, header) = &rows[0];
    let header_cols = row_columns(header).unwrap_or(0);
    if rows.len() == 1 {
        // A single `|` line is not a table; nothing to enforce.
        return Ok(());
    }
    let (_, second) = &rows[1];
    if !is_separator_row(second) || row_columns(second) != Some(header_cols) {
        return Err(OutputSchemaError::TableMismatch { line: *first_line });
    }
    for (line, row) in rows.iter().skip(2) {
        if row_columns(row) != Some(header_cols) {
            return Err(OutputSchemaError::TableMismatch { line: *line });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_paragraph_is_valid() {
        assert!(validate_markdown_output(b"just words\n\nmore words").is_ok());
    }

    #[test]
    fn a_document_with_headings_is_valid() {
        let doc = "# Title\n\n## Section\n\n### Sub\n\ntext\n";
        assert!(validate_markdown_output(doc.as_bytes()).is_ok());
    }

    #[test]
    fn skipping_a_level_while_descending_is_refused() {
        let doc = "# Title\n\n### Sub\n";
        let err = validate_markdown_output(doc.as_bytes()).unwrap_err();
        assert_eq!(
            err,
            OutputSchemaError::HeadingSkip {
                line: 3,
                from: 1,
                to: 3
            }
        );
    }

    #[test]
    fn ascending_to_any_level_is_allowed() {
        // Ascend from 3 to 1 (any shallower level), then descend by exactly
        // one: both are fine.
        let doc = "### Deep\n\n# Back\n\n## Shallower descent\n";
        assert!(validate_markdown_output(doc.as_bytes()).is_ok());
    }

    #[test]
    fn unbalanced_fence_is_refused() {
        let doc = "text\n```rust\nlet x = 1;\n";
        let err = validate_markdown_output(doc.as_bytes()).unwrap_err();
        assert_eq!(err, OutputSchemaError::UnbalancedFence { line: 2 });
    }

    #[test]
    fn balanced_fence_is_accepted_even_with_hash_rows_inside() {
        let doc = "## Title\n\n```\n# not a heading\n```\n";
        assert!(validate_markdown_output(doc.as_bytes()).is_ok());
    }

    #[test]
    fn a_passage_containing_its_own_fence_is_quoted_in_a_longer_one() {
        let passage = "here is how:\n```rust\nlet x = 1;\n```\n";
        let fence = fence_for(passage);
        let doc = format!("## Passage\n\n{fence}\n{passage}{fence}\n");
        assert!(validate_markdown_output(doc.as_bytes()).is_ok(), "{doc}");
        // A wrapper that is not longer than the passage's own fence leaks: the
        // inner fence closes the outer one and the document ends unbalanced.
        let naif = format!("## Passage\n\n```\n{passage}```\n");
        assert!(matches!(
            validate_markdown_output(naif.as_bytes()),
            Err(OutputSchemaError::UnbalancedFence { .. })
        ));
    }

    #[test]
    fn a_table_with_a_separator_and_consistent_columns_is_valid() {
        let doc = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n";
        assert!(validate_markdown_output(doc.as_bytes()).is_ok());
    }

    #[test]
    fn a_table_without_a_separator_row_is_refused() {
        let doc = "| a | b |\n| 1 | 2 |\n";
        let err = validate_markdown_output(doc.as_bytes()).unwrap_err();
        assert!(matches!(err, OutputSchemaError::TableMismatch { .. }));
    }

    #[test]
    fn a_table_with_a_mismatched_column_is_refused() {
        let doc = "| a | b |\n|---|---|\n| 1 | 2 | 3 |\n";
        let err = validate_markdown_output(doc.as_bytes()).unwrap_err();
        assert!(matches!(err, OutputSchemaError::TableMismatch { .. }));
    }

    #[test]
    fn empty_output_is_refused() {
        assert_eq!(
            validate_markdown_output(b"   \n  "),
            Err(OutputSchemaError::Empty)
        );
    }

    #[test]
    fn invalid_utf8_is_refused() {
        assert_eq!(
            validate_markdown_output(&[0xff, 0xfe, 0x00]),
            Err(OutputSchemaError::NotUtf8)
        );
    }

    #[test]
    fn oversized_output_is_refused() {
        let big = vec![b'a'; MAX_OUTPUT_BYTES + 1];
        assert!(matches!(
            validate_markdown_output(&big),
            Err(OutputSchemaError::TooLarge { .. })
        ));
    }

    #[test]
    fn a_single_pipe_line_is_not_treated_as_a_table() {
        assert!(validate_markdown_output(b"| not a table at all").is_ok());
    }
}
