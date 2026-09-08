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
use lubot_tools::{route, scope, Route};

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
    /// The question is outside the reader's scope (generation, secret hunts).
    /// The refusal names the reason, like every other refusal here.
    OutOfScope { reason: &'static str },
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

    /// The Markdown document for this answer.
    ///
    /// This is the single exit of the answer surface: any producer of a reply
    /// goes through here, and the produced text is schema-validated before it
    /// returns. A reply that fails the schema is an error here - it is never
    /// downgraded to a format it was not, and it never leaves unvalidated.
    pub fn render_markdown(&self) -> Result<String, lubot_read::output_schema::OutputSchemaError> {
        let text = match self {
            Answer::Computed { tool, value } => {
                format!("## {tool}\n\n```\n{value}\n```\n")
            }
            Answer::ToolRefused { tool, reason } => {
                format!("## {tool}\n\nThe tool refused: {reason}\n")
            }
            Answer::Grounded { passages } => {
                let mut doc = String::from("# Answer\n\n");
                for passage in passages {
                    doc.push_str(&format!("- {}\n\n", passage.text));
                }
                doc
            }
            Answer::NotFound => String::from("# No answer\n\nNothing relevant was found.\n"),
            Answer::Refused { decision } => {
                format!("# Refused\n\nThe decision word is: {decision}\n")
            }
            Answer::OutOfScope { reason } => {
                format!("# Out of scope\n\n{reason}\n")
            }
        };
        lubot_read::output_schema::validate_markdown_output(text.as_bytes())?;
        Ok(text)
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
        // Scope before anything opens: a generation request or a secret hunt
        // is refused here, and no grant is consulted for a refusal that the
        // question itself justifies.
        if let Some(kind) = scope::scope_refusal(question) {
            return Answer::OutOfScope {
                reason: kind.reason(),
            };
        }
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

pub mod output_registry;

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

#[cfg(test)]
mod render_tests {
    use super::*;
    use lubot_index::Passage;

    #[test]
    fn every_answer_variant_renders_valid_markdown() {
        let computed = Answer::Computed {
            tool: "calculator",
            value: "1/3".to_string(),
        };
        assert!(computed.render_markdown().is_ok());

        let refused = Answer::ToolRefused {
            tool: "calculator",
            reason: "overflow".to_string(),
        };
        assert!(refused.render_markdown().is_ok());

        let grounded = Answer::Grounded {
            passages: vec![Passage {
                item_id: "i".to_string(),
                origin: "doc.md".to_string(),
                first_line: 1,
                last_line: 1,
                text: "a measured claim".to_string(),
            }],
        };
        let text = grounded.render_markdown().expect("grounded renders");
        assert!(!text.contains("doc.md")); // citation is in Answer::citations, not in the doc body
        assert!(grounded
            .citations()
            .first()
            .is_some_and(|c| c == "doc.md:1"));

        let not_found = Answer::NotFound;
        assert!(not_found.render_markdown().is_ok());

        let decision = Answer::Refused {
            decision: "no-grant".to_string(),
        };
        assert!(decision.render_markdown().is_ok());
    }

    #[test]
    fn a_generation_request_is_refused_before_anything_opens() {
        use lubot_read::{FixtureCorpus, Item, SourceKind};
        let mut c = FixtureCorpus::new();
        c.insert(Item::new(
            "public-doc",
            "docs/grants.md",
            SourceKind::Local,
            false,
            "A view grant names a grantee and a key id.",
        ))
        .unwrap();
        let reader = Reader::new(&c, 2, 3);
        let mut grants = GrantBook::new();
        let answer = reader.ask("someone", "write a haiku about data", &mut grants, 1);
        assert!(matches!(answer, Answer::OutOfScope { .. }));
        assert!(grants.audit().is_empty(), "a scope refusal opens nothing");
        let md = answer
            .render_markdown()
            .expect("out-of-scope renders valid");
        assert!(md.starts_with("# Out of scope"), "{md}");
    }

    #[test]
    fn an_unbalanced_fence_in_a_passage_is_refused_not_fixed() {
        let grounded = Answer::Grounded {
            passages: vec![Passage {
                item_id: "i".to_string(),
                origin: "code.rs".to_string(),
                first_line: 1,
                last_line: 3,
                text: "look:\n```rust\nlet x = 1;".to_string(),
            }],
        };
        let err = grounded.render_markdown().unwrap_err();
        assert!(matches!(
            err,
            lubot_read::output_schema::OutputSchemaError::UnbalancedFence { .. }
        ));
    }
}
