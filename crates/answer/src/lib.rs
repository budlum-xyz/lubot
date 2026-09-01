#![forbid(unsafe_code)]
//! # lubot-answer - the shape of a reply
//!
//! This is where the parts meet: a question arrives, the tool router gets first
//! refusal, the grant book decides what may be opened, the index finds the
//! passages, and the reply is assembled with the citations attached.
//!
//! Three rules the assembly enforces, because prose cannot:
//!
//! 1. **Every claim carries a citation.** [`Answer::Grounded`] cannot be built
//!    without at least one passage.
//! 2. **Nothing found is a valid answer.** [`Answer::NotFound`] exists so the
//!    system has somewhere to go other than inventing one.
//! 3. **A refusal names itself.** [`Answer::Refused`] carries the decision word
//!    from the grant book, so "revoked" is never reported as "not found".

use lubot_grant::{Decision, GrantBook, Seconds, Visibility};
use lubot_index::{Index, Passage};
use lubot_read::Corpus;
use lubot_tools::{route, Route};

/// What the reader gets back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    /// Computed by a tool. The model does not restate arithmetic.
    Computed { tool: &'static str, value: String },
    /// A tool recognised the question and could not answer it.
    ToolRefused { tool: &'static str, reason: String },
    /// Passages that answer the question, each with where it came from.
    Grounded { passages: Vec<Passage> },
    /// The corpus was searched and holds nothing relevant.
    NotFound,
    /// Everything relevant was behind a permission the reader does not have.
    Refused { decision: String },
}

impl Answer {
    /// The citations behind this answer, in order. Empty for everything that is
    /// not grounded in a passage - including the computed answers, whose
    /// evidence is the computation itself.
    #[must_use]
    pub fn citations(&self) -> Vec<String> {
        match self {
            Answer::Grounded { passages } => passages.iter().map(Passage::citation).collect(),
            _ => Vec::new(),
        }
    }
}

/// The reading loop over one corpus.
pub struct Reader<'a, C: Corpus> {
    corpus: &'a C,
    index: Index,
    passages_per_answer: usize,
}

impl<'a, C: Corpus> Reader<'a, C> {
    /// Build a reader and index every item the corpus holds.
    #[must_use]
    pub fn new(corpus: &'a C, lines_per_passage: usize, passages_per_answer: usize) -> Self {
        let mut index = Index::new();
        for id in corpus.ids() {
            if let Some(item) = corpus.get(&id) {
                index.add(item, lines_per_passage);
            }
        }
        Self {
            corpus,
            index,
            passages_per_answer,
        }
    }

    /// Answer one question for one reader at one moment.
    ///
    /// The order is not an implementation detail: the tool is consulted before
    /// any content is opened, so a question that needs no data never triggers a
    /// permission check, and permission is settled before the index is
    /// searched, so a refused item is never scored.
    pub fn ask(
        &self,
        reader: &str,
        question: &str,
        grants: &mut GrantBook,
        now: Seconds,
    ) -> Answer {
        match route(question) {
            Route::Tool { name, result } => {
                return Answer::Computed {
                    tool: name,
                    value: result,
                }
            }
            Route::ToolFailed { name, reason } => {
                return Answer::ToolRefused { tool: name, reason }
            }
            Route::Model => {}
        }

        let mut allowed: Vec<String> = Vec::new();
        let mut refusal: Option<Decision> = None;
        for id in self.corpus.ids() {
            let Some(item) = self.corpus.get(&id) else {
                continue;
            };
            let visibility = if item.restricted {
                Visibility::Restricted
            } else {
                Visibility::Public
            };
            let decision = grants.decide(reader, &item.id, visibility, now);
            if decision.opens() {
                allowed.push(item.id.clone());
            } else if refusal.is_none() {
                refusal = Some(decision);
            }
        }

        let passages = self
            .index
            .search(question, &allowed, self.passages_per_answer);
        if !passages.is_empty() {
            return Answer::Grounded { passages };
        }
        match refusal {
            Some(decision) if allowed.is_empty() => Answer::Refused {
                decision: decision.label().to_string(),
            },
            _ => Answer::NotFound,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_grant::ViewGrant;
    use lubot_read::{FixtureCorpus, Item, SourceKind};

    fn corpus() -> FixtureCorpus {
        let mut c = FixtureCorpus::new();
        c.insert(Item::new(
            "public-doc",
            "docs/grants.md",
            SourceKind::Local,
            false,
            "A view grant names a grantee and a key id.\nRevocation stops new opens.",
        ))
        .unwrap();
        c.insert(Item::new(
            "dm-1",
            "dm/1",
            SourceKind::Granted,
            true,
            "The private note mentions a settlement schedule.",
        ))
        .unwrap();
        c
    }

    #[test]
    fn arithmetic_never_reaches_the_reading_path() {
        let c = corpus();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        assert_eq!(
            reader.ask("someone", "74830 * 1291 = ?", &mut grants, 1),
            Answer::Computed {
                tool: "calculator",
                value: "96605530".to_string()
            }
        );
        // No content was opened, so no permission was consulted.
        assert!(grants.audit().is_empty());
    }

    #[test]
    fn a_public_question_is_answered_with_a_citation() {
        let c = corpus();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        let answer = reader.ask("someone", "what does revocation do?", &mut grants, 1);
        match &answer {
            Answer::Grounded { passages } => {
                assert_eq!(passages[0].item_id, "public-doc");
                assert_eq!(answer.citations(), vec!["docs/grants.md:1-2".to_string()]);
            }
            other => panic!("expected a grounded answer, got {other:?}"),
        }
    }

    #[test]
    fn private_content_is_invisible_without_a_grant() {
        let c = corpus();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        let answer = reader.ask(
            "someone",
            "what is the settlement schedule?",
            &mut grants,
            1,
        );
        assert_eq!(answer, Answer::NotFound);
        assert_eq!(grants.refusals(), 1);
    }

    #[test]
    fn the_same_question_is_answered_once_the_grant_exists() {
        let c = corpus();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        grants.issue(ViewGrant {
            key_id: "dm-1".to_string(),
            grantee: "someone".to_string(),
            expires_at: 100,
        });
        let answer = reader.ask(
            "someone",
            "what is the settlement schedule?",
            &mut grants,
            1,
        );
        match answer {
            Answer::Grounded { passages } => assert_eq!(passages[0].item_id, "dm-1"),
            other => panic!("expected a grounded answer, got {other:?}"),
        }
    }

    #[test]
    fn an_expired_grant_closes_the_content_again() {
        let c = corpus();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        grants.issue(ViewGrant {
            key_id: "dm-1".to_string(),
            grantee: "someone".to_string(),
            expires_at: 100,
        });
        let answer = reader.ask(
            "someone",
            "what is the settlement schedule?",
            &mut grants,
            200,
        );
        assert_eq!(answer, Answer::NotFound);
        assert!(grants
            .audit()
            .iter()
            .any(|e| e.decision == Decision::Expired));
    }

    #[test]
    fn a_question_the_corpus_does_not_cover_is_not_invented() {
        let c = corpus();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        assert_eq!(
            reader.ask("someone", "who won the match last night?", &mut grants, 1),
            Answer::NotFound
        );
    }

    #[test]
    fn an_impossible_computation_is_reported_not_answered() {
        let c = corpus();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        match reader.ask("someone", "what is 1 / 0", &mut grants, 1) {
            Answer::ToolRefused { reason, .. } => assert!(reason.contains("division by zero")),
            other => panic!("expected a tool refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_reader_with_nothing_open_gets_the_refusal_word() {
        let mut c = FixtureCorpus::new();
        c.insert(Item::new(
            "dm-1",
            "dm/1",
            SourceKind::Granted,
            true,
            "the settlement schedule is monthly",
        ))
        .unwrap();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        grants.issue(ViewGrant {
            key_id: "dm-1".to_string(),
            grantee: "someone".to_string(),
            expires_at: 100,
        });
        grants.revoke("dm-1", "someone");
        assert_eq!(
            reader.ask("someone", "settlement schedule", &mut grants, 1),
            Answer::Refused {
                decision: "revoked".to_string()
            }
        );
    }

    #[test]
    fn only_grounded_answers_carry_citations() {
        assert!(Answer::NotFound.citations().is_empty());
        assert!(Answer::Computed {
            tool: "calculator",
            value: "4".to_string()
        }
        .citations()
        .is_empty());
    }
}
