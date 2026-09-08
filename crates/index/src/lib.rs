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

/// A passage must cover this share of the question's distinct terms.
pub const MIN_TERM_COVERAGE: f64 = 0.5;
/// BM25 parameters, fixed so a change in scoring is a measured change.
pub const BM25_K1: f64 = 1.2;
pub const BM25_B: f64 = 0.75;

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
        let names = [
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "private_key",
        ];
        if let Some(eq) = line.find(['=', ':']) {
            // The separator index is a char boundary in `line`; slicing the
            // lowercased copy with it would be a guess, because lowercase can
            // change byte offsets (e.g. `İ` becomes `i` + a combining dot).
            // So the candidate name is sliced from the original and lowercased
            // only for the comparison.
            let name = line[..eq].trim().to_lowercase();
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
    /// Scoring is BM25 over normalised terms: a question is matched against a
    /// passage only through a closed set of operations, and nothing here is a
    /// language model. Two guards keep "no relevant passage" a real outcome:
    ///
    /// 1. **Term coverage** - at least half of the question's distinct terms
    ///    must be found in a passage (after stopword removal), otherwise the
    ///    hit is coincidence, not evidence.
    /// 2. **Fuzzy tolerance of one edit** - a misspelled query term of five
    ///    or more characters may match a corpus word one edit away, so a
    ///    typo does not silently become "nothing found".
    ///
    /// Normalisation runs on both sides (NFD, combining marks stripped,
    /// lowercased) so Turkish `İ` on one side and `i` on the other are the
    /// same term. Document frequencies are computed once, before the scan;
    /// computing them per passage would make a full-corpus query quadratic.
    #[must_use]
    pub fn search(&self, question: &str, allowed: &[String], limit: usize) -> Vec<Passage> {
        use std::collections::HashSet;
        let terms = terms_of(question);
        if terms.is_empty() {
            return Vec::new();
        }
        let allowed_set: HashSet<&str> = allowed.iter().map(String::as_str).collect();
        let n = self.passages.len() as f64;
        if n == 0.0 {
            return Vec::new();
        }
        // One normalised read of each passage per query; the lowercase forms
        // double as the frequency source.
        let mut docs: Vec<(&Passage, String)> = Vec::with_capacity(self.passages.len());
        let mut total_terms = 0usize;
        for passage in &self.passages {
            if allowed_set.contains(passage.item_id.as_str()) {
                let norm = normalize(&passage.text);
                total_terms += norm.split_whitespace().count();
                docs.push((passage, norm));
            }
        }
        let avgdl = total_terms as f64 / n;
        let df: Vec<usize> = terms
            .iter()
            .map(|term| {
                docs.iter()
                    .filter(|(_, norm)| norm_contains(norm, term))
                    .count()
            })
            .collect();
        // A document with no term at all is uninteresting; the cost of the
        // fuzzy pass is paid only for the docs that already share a term.
        let idf: Vec<f64> = df
            .iter()
            .map(|count| (1.0 + (n - *count as f64 + 0.5) / (*count as f64 + 0.5)).ln())
            .collect();
        let mut scored: Vec<(f64, &Passage)> = Vec::new();
        for (passage, norm) in &docs {
            let words: Vec<&str> = norm.split_whitespace().collect();
            let mut score = 0.0;
            let mut covered = 0usize;
            for (i, term) in terms.iter().enumerate() {
                if df[i] == 0 || !norm_contains(norm, term) {
                    continue;
                }
                covered += 1;
                let mut tf = words.iter().filter(|w| *w == term).count() as f64;
                if tf == 0.0 {
                    tf = 1.0; // fuzzy or compound match counts once
                }
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (words.len() as f64 / avgdl));
                score += idf[i] * (tf * (BM25_K1 + 1.0)) / denom.max(1e-9);
            }
            let coverage = covered as f64 / terms.len() as f64;
            if score > 0.0 && coverage >= MIN_TERM_COVERAGE {
                scored.push((score, *passage));
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
}

/// Neutral word list, closed on purpose: `what is a view grant` is really
/// two terms, not six. Everything else stays a term.
const STOPWORDS: [&str; 24] = [
    "the", "are", "was", "were", "and", "for", "with", "this", "that", "what", "which", "how",
    "why", "does", "can", "you", "about", "there", "nedir", "nasil", "nasıl", "gibi", "icin",
    "için",
];

/// NFD, combining marks stripped, lowercased: `İSTANBUL` and `istanbul` are
/// the same term on both sides of the match.
#[must_use]
pub fn normalize(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let mut out = String::with_capacity(text.len());
    for c in text.nfd() {
        if unicode_normalization::char::is_combining_mark(c) {
            continue;
        }
        out.extend(c.to_lowercase());
    }
    out
}

/// Normalised, stopword-free words of at least three characters.
fn terms_of(question: &str) -> Vec<String> {
    normalize(question)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
        .map(str::to_string)
        .collect()
}

/// Does the normalised passage contain the term, exactly or one edit away?
/// The fuzzy branch runs only over unique words of similar length, so the
/// cost is bounded by the number of distinct words in the passage.
fn norm_contains(norm: &str, term: &str) -> bool {
    if norm.contains(term) {
        return true;
    }
    if term.chars().count() < 5 {
        return false;
    }
    let min_len = term.chars().count().saturating_sub(1);
    let max_len = term.chars().count() + 1;
    norm.split_whitespace().any(|word| {
        word.chars().count() >= min_len
            && word.chars().count() <= max_len
            && word != term
            && within_one(term, word)
    })
}

/// Levenshtein distance of at most one, with a length-gate and early exit.
fn within_one(a: &str, b: &str) -> bool {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > 1 {
        return false;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            let best = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            cur.push(best);
        }
        prev = cur;
    }
    *prev.last().unwrap_or(&0) <= 1
}

/// Deterministic context compaction: keep the highest-ranked passages that
/// fit a character budget and return how many were dropped. The input order
/// is the ranking order (the caller's search output), so no rescoring
/// happens here - the choice is purely "how much of the ranked evidence
/// survives the window".
#[must_use]
pub fn compact(passages: Vec<Passage>, budget: usize) -> (Vec<Passage>, usize) {
    let mut kept = Vec::new();
    let mut used = 0usize;
    let mut dropped = 0usize;
    for passage in passages {
        let cost = passage.text.len() + 2; // one bullet line, one newline
        if used + cost <= budget {
            used += cost;
            kept.push(passage);
        } else {
            dropped += 1;
        }
    }
    (kept, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_read::SourceKind;

    fn item(id: &str, body: &str) -> Item {
        Item::new(id, &format!("docs/{id}.md"), SourceKind::Local, false, body)
    }

    #[test]
    fn compaction_keeps_the_top_passages_that_fit() {
        let mut index = Index::new();
        index.add(&item("a", "first passage with some length"), 16);
        index.add(&item("b", "second passage with a bit more"), 16);
        let allowed: Vec<String> = vec!["a".to_string(), "b".to_string()];
        let passages = index.search("first second passage length", &allowed, 10);
        assert_eq!(passages.len(), 2);
        let budget = passages[0].text.len() + 2;
        let (kept, dropped) = compact(passages, budget);
        assert_eq!(kept.len(), 1);
        assert_eq!(dropped, 1);
        assert_eq!(kept[0].item_id, "a");
    }

    #[test]
    fn an_over_budget_window_drops_everything_but_reports_it() {
        let passages = vec![
            Passage {
                item_id: "a".into(),
                origin: "o".into(),
                first_line: 1,
                last_line: 1,
                text: "12345".into(),
            },
            Passage {
                item_id: "b".into(),
                origin: "o".into(),
                first_line: 1,
                last_line: 1,
                text: "12345".into(),
            },
        ];
        let (kept, dropped) = compact(passages, 3);
        assert!(kept.is_empty());
        assert_eq!(dropped, 2);
    }

    #[test]
    fn an_uncapped_window_passes_everything_through() {
        let passages = vec![
            Passage {
                item_id: "a".into(),
                origin: "o".into(),
                first_line: 1,
                last_line: 1,
                text: "12345".into(),
            },
            Passage {
                item_id: "b".into(),
                origin: "o".into(),
                first_line: 1,
                last_line: 1,
                text: "12345".into(),
            },
        ];
        let (kept, dropped) = compact(passages, 10_000);
        assert_eq!(kept.len(), 2);
        assert_eq!(dropped, 0);
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
    fn masking_survives_lowercase_byte_offsets() {
        // Turkish uppercase `İ` lowercases into two chars, so a byte index
        // measured on the original line is not a boundary of the lowercased
        // copy. The masker must never slice the lowercased line.
        let line = "İÇERİK_API_KEY=abc123";
        let masked = mask_secrets(line);
        assert!(masked.contains("[masked]"), "{masked}");
        let plain = mask_secrets("İÇERİK=değer\ndeğer");
        assert_eq!(plain, "İÇERİK=değer\ndeğer");
    }

    #[test]
    fn turkish_casing_is_normalized_on_both_sides() {
        let mut index = Index::new();
        index.add(&item("a", "İSTANBUL veri merkezi ölçümleri"), 4);
        assert_eq!(index.search("istanbul", &["a".to_string()], 5).len(), 1);
    }

    #[test]
    fn a_one_edit_typo_still_matches() {
        let mut index = Index::new();
        index.add(&item("a", "the modality ceiling is four"), 4);
        assert_eq!(index.search("modaliti", &["a".to_string()], 5).len(), 1);
    }

    #[test]
    fn a_query_sharing_only_one_term_is_not_evidence_enough() {
        let mut index = Index::new();
        index.add(&item("a", "grants and refusals are logged"), 4);
        // "grants" matches, but the other two terms find nothing: coverage is
        // below the floor, so the hit is coincidence, not an answer.
        assert!(index
            .search("grants photosynthesis nucleus", &["a".to_string()], 5)
            .is_empty());
    }

    #[test]
    fn stopwords_do_not_dilute_coverage() {
        let mut index = Index::new();
        index.add(&item("a", "view grant names a grantee and a key id"), 4);
        let hits = index.search("what is a view grant?", &["a".to_string()], 5);
        assert_eq!(hits.len(), 1);
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

    #[test]
    fn turkish_uppercase_forms_match_the_same_term() {
        let mut index = Index::new();
        let mut it = item("t1", "İstanbul modları: görsel, metin, ses.");
        index.add(&it, 4);
        it = item("t2", "ısparta bir şehirdir.");
        index.add(&it, 4);
        let allowed = vec!["t1".to_string(), "t2".to_string()];
        // İSTANBUL (uppercase dotted I) must hit the İstanbul passage.
        let hits = index.search("İSTANBUL modaliteler", &allowed, 3);
        assert!(
            !hits.is_empty(),
            "Turkish İ must fold to the same term as i"
        );
        assert_eq!(hits[0].item_id, "t1");
        // Dotless ı is a distinct letter (never folded onto i by case rules),
        // and the one-edit tolerance is what bridges it - measured, not accidental.
        let hits = index.search("isparta", &allowed, 3);
        assert!(
            hits.iter().any(|h| h.item_id == "t2"),
            "the ı/i gap is one edit"
        );
    }
}
