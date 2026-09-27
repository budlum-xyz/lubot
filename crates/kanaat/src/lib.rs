#![forbid(unsafe_code)]
//! # kanaat - evidence in, verdict out
//!
//! `kanaat` decides. It takes a question, a closed list of options and a set of
//! evidence items, and returns exactly one of three shapes: a choice, an
//! escalation, or a refusal. Every verdict carries its reasoning, because a
//! decision without its numbers cannot be reviewed and cannot be argued with.
//!
//! # The three shapes, and why there are only three
//!
//! | shape | when | what the caller does |
//! |---|---|---|
//! | [`Hukum::Secim`] | one candidate clears every floor and leads by a margin | acts on it |
//! | [`Hukum::Yukselt`] | the evidence is real but does not settle the question | asks a human, or gathers more |
//! | [`Hukum::Red`] | the question or the evidence cannot support any verdict | stops |
//!
//! There is no fourth "best guess" shape. A guess and a decision are different
//! acts, and a type that lets them share a return value is a type that makes the
//! difference invisible at the call site.
//!
//! # The floors, in order
//!
//! Escalation and refusal are decided by *separate* checks, and the order is
//! fixed and stated in the doctrine page: a missing prerequisite first, then a
//! weak candidate, then the doctrine's confidence threshold. The order matters
//! because the reported reason is the first one that fired, and two runs on the
//! same case must name the same reason.
//!
//! | floor | what it protects against |
//! |---|---|
//! | evidence floor | deciding with nothing to stand on |
//! | overlap floor | a candidate that shares no vocabulary with the evidence |
//! | coverage floor | a candidate that dodges most of the question |
//! | support floor | a single weak mention |
//! | margin floor | two candidates that are genuinely tied |
//!
//! # What this crate is not
//!
//! It does not read files, load a corpus, or print. It takes text in, returns
//! verdicts; the CLI owns I/O and Markdown. It has no model and no weights: the
//! arithmetic here is exact and repeatable, and a verdict is reproducible from
//! the case alone - which is the property that makes a battery meaningful.

mod defter;
mod metin;
mod puanlama;

use serde::{Deserialize, Serialize};

use lubot_tomurcuk::{tek_bas, Guven, Karar, Politika, Puan, PuanKarari, Sonuc, YukseltmeNedeni};

pub use defter::{Defter, DefterHatasi, Kayit};
pub use puanlama::SecenekPuan;

/// The battery format this build understands.
///
/// A battery written for another version is refused rather than run: an
/// expectation set that was written against different thresholds would report
/// failures that are not defects, and the noise would hide the real ones.
pub const BATARYA_SURUMU: u32 = 1;

/// One piece of evidence.
///
/// Serializable as well as deserializable: a case can be written out again, so
/// a verdict can be replayed from the case it was made on rather than from a
/// description of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kanit {
    /// Stable identity, quoted by the reasoning.
    pub kimlik: String,
    /// The evidence text itself.
    pub metin: String,
    /// Optional weight; absent means `1.0`. A zero-weight item contributes
    /// nothing, which is how an item is kept in a case for the record without
    /// letting it vote.
    #[serde(default)]
    pub agirlik: Option<f64>,
}

/// One question with a closed set of options and the evidence for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dava {
    /// The question being decided.
    pub soru: String,
    /// The options. The order is part of the case: a verdict names an index.
    pub secenekler: Vec<String>,
    /// The evidence.
    pub kanitlar: Vec<Kanit>,
}

/// The thresholds a verdict is measured against.
#[derive(Debug, Clone, PartialEq)]
pub struct Ayarlar {
    /// How many evidence items are needed before any decision is allowed.
    pub en_az_kanit: usize,
    /// The share of the question a candidate must address to be decided on.
    pub en_az_kapsam: f64,
    /// The lead the best candidate must hold over the runner-up.
    pub marj_esigi: f64,
    /// The lead at which the margin stops adding confidence.
    pub marj_tam: f64,
    /// The support at which support stops adding confidence.
    pub destek_tam: usize,
    /// Multiplier applied to a candidate that contradicts the evidence's
    /// polarity. It reduces the score, it does not veto: a candidate may
    /// disagree with one item of a set that mostly supports it.
    pub olumsuzluk_cezasi: f64,
    /// Bonus for agreeing with the evidence on a number.
    pub sayi_bonusu: f64,
    /// Penalty for quoting a different number in the same unit.
    pub sayi_celiskisi: f64,
    /// The decision head's own policy: confidence threshold, k-of-n.
    pub politika: Politika,
}

impl Default for Ayarlar {
    /// The declared defaults. Every one is a policy choice, and the doctrine
    /// page prints them, so a reader can disagree with a specific number rather
    /// than with "the engine".
    fn default() -> Self {
        Self {
            en_az_kanit: 1,
            en_az_kapsam: 0.34,
            marj_esigi: 0.15,
            marj_tam: 0.60,
            destek_tam: 3,
            olumsuzluk_cezasi: 0.5,
            sayi_bonusu: 0.10,
            sayi_celiskisi: 0.25,
            politika: Politika::varsayilan(),
        }
    }
}

/// Why a decision was escalated instead of taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Yukseltme {
    /// Fewer evidence items than the floor.
    DestekYetersiz {
        /// The support found.
        destek: usize,
        /// The support required.
        gereken: usize,
    },
    /// The best candidate covers less of the question than the floor.
    KapsamDusuk {
        /// The coverage found.
        kapsam: f64,
        /// The coverage required.
        gereken: f64,
    },
    /// The lead over the runner-up is below the floor.
    MarjYetersiz {
        /// The margin found.
        marj: f64,
        /// The margin required.
        gereken: f64,
    },
    /// The arithmetic was satisfied but the decision head's own confidence
    /// threshold was not: the head declines rather than deciding alone. The
    /// reason is the head's own, carried through rather than renamed.
    GuvenEsigi(YukseltmeNedeni),
}

/// Why no verdict was possible at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedSebebi {
    /// The question was empty.
    SoruBos,
    /// No options were supplied.
    SecenekYok,
    /// No evidence was supplied.
    KanitYok,
    /// No candidate shares any token with the evidence.
    EslesmeYok,
    /// The decision head refused.
    GuvenRed,
}

/// The numbers a choice was made on.
#[derive(Debug, Clone, PartialEq)]
pub struct Gerekce {
    /// Lead of the winner over the runner-up, saturated at one.
    pub marj: f64,
    /// Coverage of the winner.
    pub kapsam: f64,
    /// Support of the winner.
    pub destek: usize,
    /// The final score of the winner.
    pub deger: f64,
    /// The evidence items that matched, by identity.
    pub dayanaklar: Vec<String>,
}

/// A verdict that names an option.
#[derive(Debug, Clone, PartialEq)]
pub struct Secim {
    /// Index into the case's options.
    pub indeks: usize,
    /// The winner's score.
    pub puan: Puan,
    /// The head's confidence in the choice.
    pub guven: Guven,
    /// Every candidate's measured position, in case order.
    pub puanlar: Vec<SecenekPuan>,
    /// The reasoning.
    pub gerekce: Gerekce,
}

/// The result of one case.
#[derive(Debug, Clone, PartialEq)]
pub enum Hukum {
    /// One option was chosen.
    Secim(Secim),
    /// The evidence did not settle it.
    Yukselt(Yukseltme),
    /// The case cannot be decided.
    Red(RedSebebi),
}

impl Hukum {
    /// The stable label a report and a battery both quote.
    #[must_use]
    pub fn etiket(&self) -> String {
        hukum_metni(self)
    }
}

/// The label of a verdict: `secim:1`, `yukselt`, `ret`.
///
/// The form is deliberately machine-readable *and* stable: a battery compares
/// these strings, so a change here changes every expectation at once and cannot
/// be made quietly.
#[must_use]
pub fn hukum_metni(hukum: &Hukum) -> String {
    match hukum {
        Hukum::Secim(secim) => format!("secim:{}", secim.indeks),
        Hukum::Yukselt(_) => "yukselt".to_string(),
        Hukum::Red(_) => "ret".to_string(),
    }
}

/// Decides one case.
///
/// The order of the checks is part of the contract; see the module docs.
#[must_use]
pub fn karar_ver(dava: &Dava, ayar: &Ayarlar) -> Hukum {
    if dava.soru.trim().is_empty() {
        return Hukum::Red(RedSebebi::SoruBos);
    }
    if dava.secenekler.is_empty() {
        return Hukum::Red(RedSebebi::SecenekYok);
    }
    if dava.kanitlar.is_empty() {
        return Hukum::Red(RedSebebi::KanitYok);
    }
    let dokum = puanlama::Dokum::kur(&dava.kanitlar);
    let mut puanlar: Vec<SecenekPuan> = dava
        .secenekler
        .iter()
        .enumerate()
        .map(|(sira, aday)| {
            let mut puan = puanlama::puanla(aday, &dava.soru, &dava.kanitlar, &dokum);
            puan.indeks = sira;
            // The two corrections are applied after scoring so that the report
            // shows the corrected number: a reader should not have to redo the
            // penalty to understand the verdict.
            if puan.olumsuzluk_celiskisi {
                puan.puan *= ayar.olumsuzluk_cezasi;
            }
            if puan.sayi_celiskisi {
                puan.puan *= ayar.sayi_celiskisi;
            }
            puan
        })
        .collect();

    let hepsi_bos = puanlar.iter().all(|p| p.eslesen_jetonlar.is_empty());
    if hepsi_bos {
        return Hukum::Red(RedSebebi::EslesmeYok);
    }

    // Rank by score, then by index: equal scores must not be ordered by a hash
    // order or a clock, and `total_cmp` is the only comparison that is total
    // over floats including NaN.
    puanlar.sort_by(|a, b| {
        b.puan
            .total_cmp(&a.puan)
            .then_with(|| a.indeks.cmp(&b.indeks))
    });
    let en_iyi = puanlar[0].clone();
    let ikinci = puanlar.get(1);

    if en_iyi.destek < ayar.en_az_kanit {
        return Hukum::Yukselt(Yukseltme::DestekYetersiz {
            destek: en_iyi.destek,
            gereken: ayar.en_az_kanit,
        });
    }
    if en_iyi.kapsam < ayar.en_az_kapsam {
        return Hukum::Yukselt(Yukseltme::KapsamDusuk {
            kapsam: en_iyi.kapsam,
            gereken: ayar.en_az_kapsam,
        });
    }
    let marj = match ikinci {
        None => 1.0,
        Some(ikinci) => {
            if en_iyi.puan <= 0.0 {
                0.0
            } else {
                ((en_iyi.puan - ikinci.puan) / en_iyi.puan).max(0.0)
            }
        }
    };
    if marj < ayar.marj_esigi {
        return Hukum::Yukselt(Yukseltme::MarjYetersiz {
            marj,
            gereken: ayar.marj_esigi,
        });
    }

    // Confidence: the margin, the coverage and the support, each saturating.
    // Three weak signals that each mean something on their own, averaged - not
    // multiplied, because multiplying three fractions understates a verdict
    // that is strong on two axes.
    let marj_payi = (marj / ayar.marj_tam).clamp(0.0, 1.0);
    let destek_payi = (en_iyi.destek as f64 / ayar.destek_tam as f64).clamp(0.0, 1.0);
    let guven_degeri = (marj_payi + en_iyi.kapsam.clamp(0.0, 1.0) + destek_payi) / 3.0;
    let guven = Guven(Puan::yeni(guven_degeri).unwrap_or(Puan::SIFIR));

    // The head decides, or declines. `tek_bas` is the doctrine's own door: the
    // confidence that the arithmetic produced is handed to it rather than
    // interpreted here, so the threshold lives in exactly one place.
    let karar: Karar = Karar::Puan(PuanKarari {
        deger: Puan::yeni(en_iyi.puan.clamp(0.0, 1.0)).unwrap_or(Puan::SIFIR),
        guven,
    });
    match tek_bas(karar, &ayar.politika) {
        Ok(Sonuc::Kesin(_)) => {}
        Ok(Sonuc::Yukselt(neden)) => return Hukum::Yukselt(Yukseltme::GuvenEsigi(neden)),
        Ok(Sonuc::Red) | Err(_) => return Hukum::Red(RedSebebi::GuvenRed),
    }

    let dayanaklar: Vec<String> = dava
        .kanitlar
        .iter()
        .filter(|kanit| {
            let jetonlar = metin::jetonlar(&kanit.metin);
            en_iyi
                .eslesen_jetonlar
                .iter()
                .any(|jeton| jetonlar.contains(jeton))
        })
        .map(|kanit| kanit.kimlik.clone())
        .collect();

    Hukum::Secim(Secim {
        indeks: en_iyi.indeks,
        puan: Puan::yeni(en_iyi.puan.clamp(0.0, 1.0)).unwrap_or(Puan::SIFIR),
        guven,
        gerekce: Gerekce {
            marj,
            kapsam: en_iyi.kapsam,
            destek: en_iyi.destek,
            deger: en_iyi.puan,
            dayanaklar,
        },
        puanlar,
    })
}

/// One case in a battery: a case, and what the engine is expected to say.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Vaka {
    /// The case's name, quoted by the report.
    pub ad: String,
    /// The case itself.
    pub dava: Dava,
    /// The expected verdict.
    pub beklenen: Beklenen,
}

/// The expected shape of a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Beklenen {
    /// A choice of a named index.
    Secim {
        /// The expected index.
        indeks: usize,
    },
    /// An escalation.
    Yukselt,
    /// A refusal.
    Ret,
}

/// A battery of cases.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Batarya {
    /// The format version.
    pub surum: u32,
    /// The cases.
    pub vakalar: Vec<Vaka>,
}

/// Why a battery could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BataryaHatasi {
    /// The text is not the JSON this build expects.
    Json(String),
    /// The version is not [`BATARYA_SURUMU`].
    Surum(u32),
    /// The battery carries no cases, so running it would prove nothing.
    Bos,
}

/// Loads a battery, refusing a version this build does not know.
///
/// # Errors
/// [`BataryaHatasi`] for malformed JSON, an unknown version, or an empty list.
pub fn batarya_oku(metin: &str) -> Result<Batarya, BataryaHatasi> {
    let batarya: Batarya =
        serde_json::from_str(metin).map_err(|hata| BataryaHatasi::Json(hata.to_string()))?;
    if batarya.surum != BATARYA_SURUMU {
        return Err(BataryaHatasi::Surum(batarya.surum));
    }
    if batarya.vakalar.is_empty() {
        return Err(BataryaHatasi::Bos);
    }
    Ok(batarya)
}

/// The label of an expectation, in the same form as [`hukum_metni`].
#[must_use]
pub fn beklenen_metni(beklenen: Beklenen) -> String {
    match beklenen {
        Beklenen::Secim { indeks } => format!("secim:{indeks}"),
        Beklenen::Yukselt => "yukselt".to_string(),
        Beklenen::Ret => "ret".to_string(),
    }
}

/// One case's outcome.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct VakaSonucu {
    /// The case's name.
    pub ad: String,
    /// The expectation, as a label.
    pub beklenen: String,
    /// What the engine said, as a label.
    pub gelen: String,
    /// Whether they agree.
    pub dogru: bool,
}

/// A battery's outcome.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BataryaRaporu {
    /// The battery version that was run.
    pub surum: u32,
    /// How many cases ran.
    pub toplam: usize,
    /// How many agreed with their expectation.
    pub dogru: usize,
    /// How many did not.
    pub yanlis: usize,
    /// The cases, in battery order.
    pub vakalar: Vec<VakaSonucu>,
}

impl BataryaRaporu {
    /// The share of cases that agreed.
    #[must_use]
    pub fn oran(&self) -> f64 {
        if self.toplam == 0 {
            0.0
        } else {
            self.dogru as f64 / self.toplam as f64
        }
    }
}

/// Runs a battery against a set of thresholds.
#[must_use]
pub fn batarya_kos(batarya: &Batarya, ayar: &Ayarlar) -> BataryaRaporu {
    let vakalar: Vec<VakaSonucu> = batarya
        .vakalar
        .iter()
        .map(|vaka| {
            let gelen = hukum_metni(&karar_ver(&vaka.dava, ayar));
            let beklenen = beklenen_metni(vaka.beklenen);
            VakaSonucu {
                ad: vaka.ad.clone(),
                dogru: gelen == beklenen,
                beklenen,
                gelen,
            }
        })
        .collect();
    let dogru = vakalar.iter().filter(|v| v.dogru).count();
    BataryaRaporu {
        surum: batarya.surum,
        toplam: vakalar.len(),
        dogru,
        yanlis: vakalar.len() - dogru,
        vakalar,
    }
}

/// The battery compiled into the binary.
///
/// Compiled in rather than read from disk: a report that depends on the working
/// directory is a report whose numbers change when the command is run from
/// somewhere else, and the difference would look like an engine change.
pub const GOMULU_BATARYA: &str = include_str!("../../../training/eval/kanaat-bataryasi.json");

#[cfg(test)]
mod tests {
    use super::*;

    fn dava(soru: &str, secenekler: &[&str], kanitlar: &[(&str, &str)]) -> Dava {
        Dava {
            soru: soru.to_string(),
            secenekler: secenekler.iter().map(|s| (*s).to_string()).collect(),
            kanitlar: kanitlar
                .iter()
                .map(|(kimlik, metin)| Kanit {
                    kimlik: (*kimlik).to_string(),
                    metin: (*metin).to_string(),
                    agirlik: None,
                })
                .collect(),
        }
    }

    #[test]
    fn a_clear_winner_is_chosen_with_its_reasoning() {
        let dava = dava(
            "yedek acildi mi",
            &["yedek acildi", "yedek kapandi"],
            &[
                ("k1", "yedek acildi ve dogrulandi"),
                ("k2", "yedek acildi kaydi tutuldu"),
            ],
        );
        match karar_ver(&dava, &Ayarlar::default()) {
            Hukum::Secim(secim) => {
                assert_eq!(secim.indeks, 0);
                assert!(secim.gerekce.marj > 0.15);
                assert!(secim.gerekce.kapsam >= 0.34);
                assert!(secim.gerekce.dayanaklar.contains(&"k1".to_string()));
                assert_eq!(secim.puanlar.len(), 2);
            }
            digeri => panic!("secim bekleniyordu: {digeri:?}"),
        }
    }

    #[test]
    fn no_evidence_is_refused_not_guessed() {
        let dava = dava("soru", &["a", "b"], &[]);
        assert_eq!(
            karar_ver(&dava, &Ayarlar::default()),
            Hukum::Red(RedSebebi::KanitYok)
        );
    }

    #[test]
    fn an_empty_question_is_refused() {
        let dava = dava("   ", &["a"], &[("k1", "a")]);
        assert_eq!(
            karar_ver(&dava, &Ayarlar::default()),
            Hukum::Red(RedSebebi::SoruBos)
        );
    }

    #[test]
    fn no_candidates_is_refused() {
        let dava = dava("soru", &[], &[("k1", "a")]);
        assert_eq!(
            karar_ver(&dava, &Ayarlar::default()),
            Hukum::Red(RedSebebi::SecenekYok)
        );
    }

    #[test]
    fn evidence_that_shares_nothing_is_refused() {
        let dava = dava(
            "soru",
            &["tamamen ilgisiz cumle"],
            &[("k1", "baska bir sey")],
        );
        assert_eq!(
            karar_ver(&dava, &Ayarlar::default()),
            Hukum::Red(RedSebebi::EslesmeYok)
        );
    }

    #[test]
    fn a_single_weak_mention_is_escalated_for_coverage() {
        let dava = dava(
            "kayit acildi mi ve denetim tamamlandi mi ve rapor onaylandi mi ve yedek dogrulandi mi",
            &["kayit acildi"],
            &[("k1", "kayit acildi")],
        );
        match karar_ver(&dava, &Ayarlar::default()) {
            Hukum::Yukselt(Yukseltme::KapsamDusuk { kapsam, gereken }) => {
                assert!(kapsam < gereken);
            }
            digeri => panic!("kapsam yukseltmesi bekleniyordu, gelen {digeri:?}"),
        }
    }

    #[test]
    fn two_equal_candidates_escalate_on_the_margin() {
        let dava = dava(
            "kayit acildi mi",
            &["kayit acildi", "kayit acildi"],
            &[("k1", "kayit acildi")],
        );
        match karar_ver(&dava, &Ayarlar::default()) {
            Hukum::Yukselt(Yukseltme::MarjYetersiz { marj, gereken }) => {
                assert!(marj.abs() < 1e-12);
                assert!(gereken > 0.0);
            }
            digeri => panic!("marj yukseltmesi bekleniyordu, gelen {digeri:?}"),
        }
    }

    #[test]
    fn option_order_is_part_of_the_answer() {
        let a = dava(
            "kayit acildi mi",
            &["kayit acildi", "kayit acilmadi"],
            &[("k1", "kayit acildi")],
        );
        let b = dava(
            "kayit acildi mi",
            &["kayit acilmadi", "kayit acildi"],
            &[("k1", "kayit acildi")],
        );
        let ha = hukum_metni(&karar_ver(&a, &Ayarlar::default()));
        let hb = hukum_metni(&karar_ver(&b, &Ayarlar::default()));
        assert_eq!(ha, "secim:0");
        assert_eq!(hb, "secim:1");
    }

    #[test]
    fn a_doctrine_threshold_above_the_confidence_escalates() {
        let dava = dava(
            "yedek acildi mi",
            &["yedek acildi", "yedek kapandi"],
            &[("k1", "yedek acildi")],
        );
        let mut ayar = Ayarlar::default();
        ayar.politika.guven_esigi = Guven(Puan::BIR);
        match karar_ver(&dava, &ayar) {
            Hukum::Yukselt(Yukseltme::GuvenEsigi(_)) => {}
            digeri => panic!("doktrin yukseltmesi bekleniyordu, gelen {digeri:?}"),
        }
    }

    #[test]
    fn a_disagreeing_polarity_is_penalised_not_vetoed() {
        let dava = dava(
            "yedek acildi mi",
            &["yedek acildi", "yedek acilmadi"],
            &[("k1", "yedek acilmadi ve kayit kapandi")],
        );
        match karar_ver(&dava, &Ayarlar::default()) {
            Hukum::Secim(secim) => {
                assert_eq!(secim.indeks, 1, "kanitin dedigi secilmeli");
                assert!(secim.puanlar.iter().any(|p| p.olumsuzluk_celiskisi));
            }
            Hukum::Yukselt(_) => {}
            Hukum::Red(neden) => panic!("ret beklenmiyordu: {neden:?}"),
        }
    }

    #[test]
    fn malformed_json_is_refused_with_the_parser_message() {
        let hata = batarya_oku("{").unwrap_err();
        assert!(matches!(hata, BataryaHatasi::Json(_)), "{hata:?}");
    }

    #[test]
    fn a_battery_of_the_wrong_version_is_refused() {
        let hata = batarya_oku(r#"{"surum": 99, "vakalar": []}"#).unwrap_err();
        assert_eq!(hata, BataryaHatasi::Surum(99));
    }

    #[test]
    fn an_empty_battery_proves_nothing_and_is_refused() {
        let hata = batarya_oku(r#"{"surum": 1, "vakalar": []}"#).unwrap_err();
        assert_eq!(hata, BataryaHatasi::Bos);
    }

    #[test]
    fn the_compiled_in_battery_runs_and_every_case_agrees() {
        let batarya = batarya_oku(GOMULU_BATARYA).expect("gomulu batarya okunmali");
        let rapor = batarya_kos(&batarya, &Ayarlar::default());
        assert_eq!(rapor.toplam, batarya.vakalar.len());
        assert_eq!(
            rapor.yanlis,
            0,
            "tutmayan vakalar: {:?}",
            rapor
                .vakalar
                .iter()
                .filter(|v| !v.dogru)
                .collect::<Vec<_>>()
        );
        assert!((rapor.oran() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_same_case_decides_the_same_way_twice() {
        let dava = dava(
            "kayit acildi mi",
            &["kayit acildi", "kayit kapandi"],
            &[("k1", "kayit acildi ve surdu"), ("k2", "kayit acildi")],
        );
        let a = karar_ver(&dava, &Ayarlar::default());
        let b = karar_ver(&dava, &Ayarlar::default());
        assert_eq!(a, b);
    }
}
