#![forbid(unsafe_code)]
//! # lubot-index - finding the passage, and masking what must not be kept
//!
//! Retrieval here is deliberately plain: split an item into passages that keep
//! their line numbers, score them against the question's terms, return the best
//! few. A citation is only worth something if it points at lines that exist, so
//! a passage carries the range it came from and nothing is stitched together
//! from two places.
//!
//! [`mask_secrets`] runs **before** anything is written into the index. A mask
//! applied on the way out would still leave the credential in the store, so the
//! order is the whole claim.

use lubot_read::Item;

/// A slice of an item, with the lines it occupies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passage {
    pub item_id: String,
    pub origin: String,
    pub first_line: usize,
    pub last_line: usize,
    pub text: String,
}

impl Passage {
    /// The citation string an answer carries: origin plus line range.
    #[must_use]
    pub fn citation(&self) -> String {
        if self.first_line == self.last_line {
            return format!("{}:{}", self.origin, self.first_line);
        }
        format!("{}:{}-{}", self.origin, self.first_line, self.last_line)
    }
}

/// Replace anything that looks like a credential with a fixed marker.
///
/// The rule is coarse on purpose: a long opaque token, or a `key=value` pair
/// whose name suggests a secret. Missing a real secret is expensive; masking an
/// innocent string costs a passage.
#[must_use]
pub fn mask_secrets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (i, line) in text.lines().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let lower = line.to_lowercase();
        let names = [
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "private_key",
        ];
        if let Some(eq) = line.find(['=', ':']) {
            let name = lower[..eq].trim().to_string();
            if names.iter().any(|n| name.ends_with(n) || name == *n) {
                out.push_str(&line[..=eq]);
                out.push_str(" [masked]");
                continue;
            }
        }
        let mut masked_line = String::with_capacity(line.len());
        for word in line.split_inclusive(char::is_whitespace) {
            let trimmed = word.trim();
            if trimmed.len() >= 32
                && trimmed
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                masked_line.push_str("[masked]");
                if word.len() > trimmed.len() {
                    masked_line.push(' ');
                }
            } else {
                masked_line.push_str(word);
            }
        }
        out.push_str(&masked_line);
    }
    if text.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// The searchable form of the corpus.
#[derive(Debug, Default)]
pub struct Index {
    passages: Vec<Passage>,
}

impl Index {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Split an item into passages of at most `lines_per_passage` lines and add
    /// them. The body is masked on this write path, before it is stored.
    pub fn add(&mut self, item: &Item, lines_per_passage: usize) {
        let step = lines_per_passage.max(1);
        let masked = mask_secrets(&item.body);
        let lines: Vec<&str> = masked.lines().collect();
        let mut start = 0;
        while start < lines.len() {
            let end = (start + step).min(lines.len());
            let text = lines[start..end].join("\n");
            if !text.trim().is_empty() {
                self.passages.push(Passage {
                    item_id: item.id.clone(),
                    origin: item.origin.clone(),
                    first_line: start + 1,
                    last_line: end,
                    text,
                });
            }
            start = end;
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.passages.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    /// The best `limit` passages for `question`, among items in `allowed`.
    ///
    /// Scoring is term overlap, weighted by how rare the term is across the
    /// index. Nothing here is a language model; a passage that shares no term
    /// with the question is never returned, so "no relevant passage" stays a
    /// possible outcome.
    #[must_use]
    pub fn search(&self, question: &str, allowed: &[String], limit: usize) -> Vec<Passage> {
        let terms = terms_of(question);
        if terms.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f64, &Passage)> = Vec::new();
        for passage in &self.passages {
            if !allowed.contains(&passage.item_id) {
                continue;
            }
            let body = passage.text.to_lowercase();
            let mut score = 0.0;
            for term in &terms {
                if body.contains(term.as_str()) {
                    score += 1.0 / (1.0 + self.document_frequency(term) as f64);
                }
            }
            if score > 0.0 {
                scored.push((score, passage));
            }
        }
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.first_line.cmp(&b.1.first_line))
        });
        scored
            .into_iter()
            .take(limit)
            .map(|(_, p)| p.clone())
            .collect()
    }

    fn document_frequency(&self, term: &str) -> usize {
        self.passages
            .iter()
            .filter(|p| p.text.to_lowercase().contains(term))
            .count()
    }
}

/// Lowercase words of at least three characters.
fn terms_of(question: &str) -> Vec<String> {
    question
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_read::SourceKind;

    fn item(id: &str, body: &str) -> Item {
        Item::new(id, &format!("docs/{id}.md"), SourceKind::Local, false, body)
    }

    #[test]
    fn a_passage_keeps_the_lines_it_came_from() {
        let mut index = Index::new();
        index.add(&item("a", "one\ntwo\nthree\nfour"), 2);
        assert_eq!(index.len(), 2);
        let hits = index.search("three", &["a".to_string()], 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].first_line, 3);
        assert_eq!(hits[0].last_line, 4);
        assert_eq!(hits[0].citation(), "docs/a.md:3-4");
    }

    #[test]
    fn a_single_line_citation_has_no_range() {
        let mut index = Index::new();
        index.add(&item("a", "alpha"), 4);
        let hits = index.search("alpha", &["a".to_string()], 5);
        assert_eq!(hits[0].citation(), "docs/a.md:1");
    }

    #[test]
    fn a_question_sharing_no_term_returns_nothing() {
        let mut index = Index::new();
        index.add(&item("a", "grants and refusals"), 4);
        assert!(index
            .search("photosynthesis", &["a".to_string()], 5)
            .is_empty());
    }

    #[test]
    fn content_the_reader_may_not_open_is_never_searched() {
        let mut index = Index::new();
        index.add(&item("public", "the grant book records refusals"), 4);
        index.add(&item("private", "the grant book records refusals"), 4);
        let hits = index.search("grant", &["public".to_string()], 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].item_id, "public");
    }

    #[test]
    fn a_rare_term_outranks_a_common_one() {
        let mut index = Index::new();
        index.add(&item("a", "grant grant grant"), 1);
        index.add(&item("b", "grant revocation"), 1);
        let hits = index.search("grant revocation", &["a".to_string(), "b".to_string()], 1);
        assert_eq!(hits[0].item_id, "b");
    }

    #[test]
    fn masking_happens_before_storage_not_after() {
        let mut index = Index::new();
        index.add(&item("a", "password = hunter2hunter2hunter2"), 4);
        let stored = index.search("password", &["a".to_string()], 5);
        assert_eq!(stored.len(), 1);
        assert!(stored[0].text.contains("[masked]"));
        assert!(!stored[0].text.contains("hunter2hunter2hunter2"));
    }

    #[test]
    fn redact_model_strings() {
        assert_eq!(mask_secrets("api_key: abc"), "api_key: [masked]");
        assert_eq!(mask_secrets("SECRET=x"), "SECRET= [masked]");
        let long = "a".repeat(40);
        assert_eq!(mask_secrets(&long), "[masked]");
        assert_eq!(mask_secrets("a short line"), "a short line");
        assert_eq!(mask_secrets("ratio: 3"), "ratio: 3");
    }

    #[test]
    fn masking_preserves_the_shape_of_a_document() {
        let text = "line one\npassword = averylongsecretvaluethatkeepsgoing\nline three";
        let masked = mask_secrets(text);
        assert_eq!(masked.lines().count(), 3);
        assert!(masked.starts_with("line one"));
        assert!(masked.ends_with("line three"));
    }

    #[test]
    fn an_empty_question_retrieves_nothing() {
        let mut index = Index::new();
        index.add(&item("a", "content"), 4);
        assert!(index.search("", &["a".to_string()], 5).is_empty());
        assert!(index.search("a I", &["a".to_string()], 5).is_empty());
    }
}
