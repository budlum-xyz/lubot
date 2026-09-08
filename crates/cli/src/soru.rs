//! # soru - decision batteries, in the ask_user shape
//!
//! The ask-user tool is an agent power; this module is its Lubot shape. A
//! battery is a JSON document of questions, each with 2-4 options and a
//! `allow_custom_response` switch, validated by the same rules the tool
//! enforces (a question with one option is not a question; a duplicated id
//! is a corrupted file). Answers are appended to a JSONL record with the
//! answer time, so a decision battery is audit-able like everything else in
//! this repository.
//!
//! The battery validates as a whole: one broken question refuses the entire
//! document, never a silent skip.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// One option of one question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SoruSecenek {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub description: String,
}

/// One question, in the ask_user wire shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Soru {
    pub id: String,
    pub question: String,
    pub options: Vec<SoruSecenek>,
    #[serde(default = "default_true")]
    pub allow_custom_response: bool,
}

fn default_true() -> bool {
    true
}

/// A whole battery. `note` is free text; `sorular` is the set, closed to
/// what the validator admits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Batarya {
    #[serde(default)]
    pub note: String,
    pub sorular: Vec<Soru>,
}

impl Batarya {
    /// Parse and validate a battery document.
    ///
    /// # Errors
    /// Parse failures, and the first rule violation: unique non-empty ids,
    /// 2-4 options per question, unique non-empty option ids, non-empty
    /// question text. One violation refuses the whole document.
    pub fn load(text: &str) -> Result<Self, String> {
        let battery: Batarya =
            serde_json::from_str(text).map_err(|e| format!("soru bataryasi: {e}"))?;
        battery.validate()?;
        Ok(battery)
    }

    /// The validation rules, one line each, in order.
    ///
    /// # Errors
    /// The first violation found, with the offending id.
    pub fn validate(&self) -> Result<(), String> {
        if self.sorular.is_empty() {
            return Err("soru bataryasi: bos batarya".to_string());
        }
        let mut seen: Vec<&str> = Vec::new();
        for soru in &self.sorular {
            if soru.id.trim().is_empty() {
                return Err("soru bataryasi: bos soru id".to_string());
            }
            if seen.contains(&soru.id.as_str()) {
                return Err(format!("soru bataryasi: yinelenen id `{}`", soru.id));
            }
            seen.push(&soru.id);
            if soru.question.trim().is_empty() {
                return Err(format!("soru `{}`: bos soru metni", soru.id));
            }
            if !(2..=4).contains(&soru.options.len()) {
                return Err(format!(
                    "soru `{}`: {} secenek (2-4 olmali)",
                    soru.id,
                    soru.options.len()
                ));
            }
            let mut option_ids: Vec<&str> = Vec::new();
            for option in &soru.options {
                if option.id.trim().is_empty() || option.label.trim().is_empty() {
                    return Err(format!("soru `{}`: bos secenek", soru.id));
                }
                if option_ids.contains(&option.id.as_str()) {
                    return Err(format!(
                        "soru `{}`: yinelenen secenek `{}`",
                        soru.id, option.id
                    ));
                }
                option_ids.push(&option.id);
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.sorular.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sorular.is_empty()
    }

    #[must_use]
    pub fn by_id(&self, id: &str) -> Option<&Soru> {
        self.sorular.iter().find(|s| s.id == id)
    }

    /// The battery as a Markdown document: one table row per question.
    #[must_use]
    pub fn list_md(&self) -> String {
        let mut doc =
            String::from("# Soru bataryasi\n\n| id | soru | secenek | ozel |\n|---|---|---|---|\n");
        for soru in &self.sorular {
            doc.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                soru.id,
                soru.question.replace('|', "/"),
                soru.options.len(),
                if soru.allow_custom_response {
                    "evet"
                } else {
                    "hayir"
                }
            ));
        }
        doc.push_str(&format!("\n{} soru\n", self.sorular.len()));
        doc
    }

    /// One question, as the ask_user tool expects it (JSON on one line).
    #[must_use]
    pub fn question_json(&self, id: &str) -> Option<String> {
        let soru = self.by_id(id)?;
        Some(
            serde_json::to_string(soru)
                .unwrap_or_else(|_| "{\"id\":\"serialize-error\"}".to_string()),
        )
    }
}

/// Record one answer: id, choice, note, answer time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Cevap {
    pub id: String,
    pub secim: String,
    #[serde(default)]
    pub not: String,
    pub at: u64,
}

/// Append an answer to the record file (one JSON line per answer).
///
/// # Errors
/// File creation or serialization failures.
pub fn record_answer(
    path: &Path,
    id: &str,
    battery: &Batarya,
    secim: &str,
    not: &str,
    at: u64,
) -> Result<(), String> {
    let soru = battery
        .by_id(id)
        .ok_or_else(|| format!("soru `{id}` bataryada yok"))?;
    let known = soru.options.iter().any(|o| o.id == secim);
    if !known && !soru.allow_custom_response {
        return Err(format!(
            "soru `{id}`: `{secim}` seceneklerde yok ve ozel cevap kapali"
        ));
    }
    let cevap = Cevap {
        id: id.to_string(),
        secim: secim.to_string(),
        not: not.to_string(),
        at,
    };
    let line = serde_json::to_string(&cevap).map_err(|e| e.to_string())?;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))
}

/// Read the answer record: distinct answered ids and the total.
#[must_use]
pub fn durum(path: &Path, total: usize) -> (usize, usize) {
    if !path.exists() {
        return (0, total);
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return (0, total);
    };
    let mut answered: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Ok(cevap) = serde_json::from_str::<Cevap>(line) {
            if !answered.contains(&cevap.id) {
                answered.push(cevap.id);
            }
        }
    }
    (answered.len(), total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "note": "t",
        "sorular": [
            {"id": "a", "question": "A?", "options": [
                {"id": "a1", "label": "bir"}, {"id": "a2", "label": "iki"}]},
            {"id": "b", "question": "B?", "allow_custom_response": false,
             "options": [{"id": "b1", "label": "bir"}, {"id": "b2", "label": "iki"}]}
        ]
    }"#;

    #[test]
    fn a_battery_loads_and_lists() {
        let battery = Batarya::load(FIXTURE).expect("fixture geçerli");
        assert_eq!(battery.len(), 2);
        assert!(battery.by_id("b").is_some());
        let md = battery.list_md();
        assert!(md.contains("| a | A? | 2 | evet |"));
    }

    #[test]
    fn a_duplicate_id_refuses_the_whole_battery() {
        let bad = FIXTURE.replace("\"id\": \"b\"", "\"id\": \"a\"");
        let err = Batarya::load(&bad).expect_err("yinelenen id reddedilir");
        assert!(err.contains("yinelenen"), "{err}");
    }

    #[test]
    fn one_option_is_not_a_question() {
        let bad = FIXTURE.replace(
            "{\"id\": \"b1\", \"label\": \"bir\"}, {\"id\": \"b2\", \"label\": \"iki\"}",
            "{\"id\": \"b1\", \"label\": \"bir\"}",
        );
        let err = Batarya::load(&bad).expect_err("tek secenek reddedilir");
        assert!(err.contains("2-4"), "{err}");
    }

    #[test]
    fn an_unknown_choice_is_refused_when_custom_is_off() {
        let battery = Batarya::load(FIXTURE).unwrap();
        let dir = std::env::temp_dir().join(format!("lubot-soru-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cevaplar.jsonl");
        let err = record_answer(&path, "b", &battery, "c", "", 1).expect_err("ozel cevap kapali");
        assert!(err.contains("kapali"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn answers_round_trip_and_durum_counts_distinct_ids() {
        let battery = Batarya::load(FIXTURE).unwrap();
        let dir = std::env::temp_dir().join(format!("lubot-soru2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cevaplar.jsonl");
        record_answer(&path, "a", &battery, "a1", "ilk", 10).unwrap();
        record_answer(&path, "a", &battery, "a2", "ikinci", 11).unwrap();
        record_answer(&path, "b", &battery, "b1", "", 12).unwrap();
        let (answered, total) = durum(&path, battery.len());
        assert_eq!(answered, 2);
        assert_eq!(total, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
