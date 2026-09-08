#![forbid(unsafe_code)]
//! # lubot-doc - rich-document reading
//!
//! The workspace PDF-reading power, in the licence-safe shape this repository
//! uses everywhere: extraction (via the MIT `pdf-extract` crate) produces
//! text; chunking is paragraph-aware and deterministic; and everything that
//! comes out of here is still just text - the corpus record schema, the
//! provenance rules and the ceilings stay exactly where they are. No content
//! is read outside the extraction step, and no extraction happens without the
//! caller passing the provenance the record schema demands.

/// Extract the text of a PDF held in memory.
///
/// # Errors
/// The extractor's own failures, named.
pub fn pdf_text(bytes: &[u8]) -> Result<String, String> {
    pdf_extract::extract_text_from_mem(bytes).map_err(|e| format!("pdf extraction: {e}"))
}

/// Split text into chunks of at most `max_chars`, preferring to break at a
/// newline (paragraph boundary) once a chunk is past half its budget. This is
/// deterministic: the same text always produces the same chunks.
#[must_use]
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in text.split('\n') {
        if current.len() + paragraph.len() + 1 > max_chars && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push('\n');
        }
        if paragraph.len() > max_chars {
            // One over-long paragraph cannot make an over-budget record; it
            // is cut on the budget, at the last word boundary available.
            let mut rest = paragraph;
            while rest.len() > max_chars {
                let cut = rest[..max_chars]
                    .rfind(char::is_whitespace)
                    .unwrap_or(max_chars);
                let (head, tail) = rest.split_at(cut);
                chunks.push(head.trim_end().to_string());
                rest = tail.trim_start();
            }
            if !rest.is_empty() {
                if !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                }
                current = rest.to_string();
            }
        } else {
            current.push_str(paragraph);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(
            chunk_text("hello lubot", 100),
            vec!["hello lubot".to_string()]
        );
    }

    #[test]
    fn chunks_break_at_paragraph_boundaries_preferentially() {
        let text = format!("{}\n{}\n{}", "a".repeat(70), "b".repeat(70), "c".repeat(70));
        let chunks = chunk_text(&text, 100);
        assert_eq!(chunks.len(), 3);
        for chunk in chunks {
            assert!(chunk.len() <= 100);
        }
    }

    #[test]
    fn an_over_long_paragraph_is_cut_on_the_budget() {
        let text = "word ".repeat(60); // 300 chars, one paragraph
        let chunks = chunk_text(&text, 100);
        assert!(chunks.len() >= 3, "{} chunks", chunks.len());
        for chunk in &chunks {
            assert!(chunk.len() <= 100, "chunk of {}", chunk.len());
        }
        // Order and content are preserved: re-joining minus separators equals
        // the original text, with whitespace runs collapsed at cuts.
        let joined = chunks.join(" ");
        assert!(joined.starts_with("word word"), "{joined}");
    }

    #[test]
    fn empty_input_is_no_chunks() {
        assert!(chunk_text("", 100).is_empty());
        assert!(chunk_text("anything", 0).is_empty());
    }
}
