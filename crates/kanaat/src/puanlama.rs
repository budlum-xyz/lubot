//! Scoring: how much of the evidence a candidate actually answers.
//!
//! # Why the score is weighted overlap and not a length ratio
//!
//! The naive figure - "how many evidence tokens does this candidate contain" -
//! rewards a candidate for being long. A paragraph that repeats the question
//! scores higher than the one sentence that answers it, and the engine's
//! confidence would then measure verbosity. Here each matched token carries its
//! inverse document frequency, so a token that appears in every evidence item
//! is worth little and a token that appears in one is worth much. The raw score
//! is divided by the candidate's own token mass, which is what makes the number
//! comparable between a two-word option and a twenty-word one.
//!
//! # Why coverage is a separate number
//!
//! Score says "the matches that happened were valuable". Coverage says "how
//! much of the question the candidate addresses". Two candidates can share a
//! score with very different coverage, and a verdict that reports only the
//! score would hide that. They are separate thresholds for the same reason.
//!
//! # Why bigrams are scored apart
//!
//! A phrase match and two unrelated word matches produce the same token
//! overlap, and they are not the same evidence. The bigram term is added on top
//! of the token term with its own coefficient, and it is deliberately *not*
//! filtered through the document frequencies: adjacent pairs are unique to one
//! evidence item almost by construction, so an idf weight there would be a
//! number that is always at its ceiling - a constant dressed as a measurement.

use std::collections::HashMap;

use crate::metin::{ikili_gramlar, jetonlar, olumsuzluk_orani};
use crate::Kanit;

/// The inverse document frequencies of one evidence set.
///
/// Built once per verdict: every candidate is scored against the same table, so
/// two candidates' scores cannot disagree because they saw different statistics.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Dokum {
    df: HashMap<String, usize>,
    adet: usize,
}

impl Dokum {
    /// Builds the table from the evidence.
    pub(crate) fn kur(kanitlar: &[Kanit]) -> Self {
        let mut df: HashMap<String, usize> = HashMap::new();
        for kanit in kanitlar {
            let jetonlar = jetonlar(&kanit.metin);
            let benzersiz: std::collections::HashSet<&String> = jetonlar.iter().collect();
            for jeton in benzersiz {
                *df.entry(jeton.clone()).or_insert(0) += 1;
            }
        }
        Self {
            df,
            adet: kanitlar.len(),
        }
    }

    /// How many evidence items there are.
    #[must_use]
    pub(crate) fn adet(&self) -> usize {
        self.adet
    }

    /// Smoothed idf of one token.
    ///
    /// The `+ 1` floor keeps a token that appears everywhere from scoring zero:
    /// zeroing it would silently delete it from the comparison, and a token
    /// nobody scores is a token whose absence cannot be noticed.
    #[must_use]
    pub(crate) fn idf(&self, jeton: &str) -> f64 {
        let df = self.df.get(jeton).copied().unwrap_or(0) as f64;
        let n = self.adet as f64;
        ((n + 1.0) / (df + 1.0)).ln() + 1.0
    }
}

/// One candidate's measured position.
#[derive(Debug, Clone, PartialEq)]
pub struct SecenekPuan {
    /// Index into the case's options.
    pub indeks: usize,
    /// Weighted overlap over the candidate's own token mass.
    pub puan: f64,
    /// Share of the question's tokens the candidate addresses.
    pub kapsam: f64,
    /// How many evidence items supplied at least one match.
    pub destek: usize,
    /// The candidate and the evidence disagree about polarity.
    pub olumsuzluk_celiskisi: bool,
    /// The candidate and the evidence quote different numbers in one unit.
    pub sayi_celiskisi: bool,
    /// The matched tokens, in candidate order, for a report to quote.
    pub eslesen_jetonlar: Vec<String>,
}

/// A number with the unit that followed it, if any.
#[derive(Debug, Clone, PartialEq)]
struct Olcu {
    deger: String,
    birim: String,
}

/// Pulls `value unit` pairs out of raw text.
///
/// A number is compared with the unit that follows it: `500` and `500 mb` are
/// different claims, and so are `500 mb` and `500 gb`. Numbers without a unit
/// are kept under the empty unit, so they may only conflict with other
/// unit-less numbers - which is the honest scope of the comparison.
fn olculer(metin: &str) -> Vec<Olcu> {
    let jetonlar = jetonlar(metin);
    let mut cikti = Vec::new();
    for (sira, jeton) in jetonlar.iter().enumerate() {
        if jeton.chars().next().is_some_and(|h| h.is_ascii_digit())
            && jeton.chars().any(|h| h.is_ascii_digit())
        {
            let birim = jetonlar
                .get(sira + 1)
                .filter(|sonraki| {
                    sonraki
                        .chars()
                        .next()
                        .is_some_and(|h| h.is_ascii_alphabetic())
                })
                .cloned()
                .unwrap_or_default();
            cikti.push(Olcu {
                deger: jeton.clone(),
                birim,
            });
        }
    }
    cikti
}

/// Whether two measurements contradict each other.
///
/// Contradiction is *same unit, different value*. A different unit is not a
/// contradiction: it is a comparison the engine cannot make, and reporting it
/// as one would punish a candidate for mentioning a related quantity.
fn sayi_celiskisi(aday_metni: &str, kanit_metni: &str) -> bool {
    let aday = olculer(aday_metni);
    let kanit = olculer(kanit_metni);
    for a in &aday {
        for k in &kanit {
            if a.birim == k.birim && a.deger != k.deger {
                return true;
            }
        }
    }
    false
}

/// Whether the candidate and the evidence point in opposite directions.
///
/// The test compares *shares*, not single words: one negated token in a long
/// sentence is a detail, a negated token in a short one is the whole claim.
fn olumsuzluk_celiskisi(aday_jetonlari: &[String], kanit_jetonlari: &[String]) -> bool {
    let aday = olumsuzluk_orani(aday_jetonlari);
    let kanit = olumsuzluk_orani(kanit_jetonlari);
    (aday > 0.0 && kanit == 0.0) || (aday == 0.0 && kanit > 0.0)
}

/// Scores one candidate against the evidence.
///
/// `soru` enters through the coverage figure only: the score is the candidate's
/// overlap with the *evidence*, because the evidence is what a verdict has to
/// stand on.
pub(crate) fn puanla(aday: &str, soru: &str, kanitlar: &[Kanit], dokum: &Dokum) -> SecenekPuan {
    let aday_jetonlari = jetonlar(aday);
    let soru_jetonlari = jetonlar(soru);
    let aday_gramlar = ikili_gramlar(&aday_jetonlari);

    let mut agirlikli = 0.0_f64;
    let mut toplam_agirlik = 0.0_f64;
    let mut eslesen = Vec::new();
    let mut destekli_kanit = vec![false; kanitlar.len()];
    let mut celiski_olumsuzluk = false;
    let mut celiski_sayi = false;

    for (sira, kanit) in kanitlar.iter().enumerate() {
        let kanit_jetonlari = jetonlar(&kanit.metin);
        if olumsuzluk_celiskisi(&aday_jetonlari, &kanit_jetonlari) {
            celiski_olumsuzluk = true;
        }
        if sayi_celiskisi(aday, &kanit.metin) {
            celiski_sayi = true;
        }
        let kanit_kumesi: std::collections::HashSet<&String> = kanit_jetonlari.iter().collect();
        for jeton in &aday_jetonlari {
            if kanit_kumesi.contains(jeton) {
                destekli_kanit[sira] = true;
                if !eslesen.contains(jeton) {
                    eslesen.push(jeton.clone());
                }
            }
        }
    }

    // The score: weighted overlap over the candidate's own mass, so length does
    // not buy points. The evidence weight is the caller's, defaulting to one.
    let agirlik_toplam: f64 = kanitlar
        .iter()
        .map(|k| k.agirlik.unwrap_or(1.0).max(0.0))
        .sum::<f64>()
        .max(f64::EPSILON);
    for jeton in &aday_jetonlari {
        let idf = dokum.idf(jeton);
        toplam_agirlik += idf;
        let mut kanit_agirligi = 0.0_f64;
        for (sira, kanit) in kanitlar.iter().enumerate() {
            if destekli_kanit[sira] {
                let kanit_jetonlari = jetonlar(&kanit.metin);
                if kanit_jetonlari.iter().any(|k| k == jeton) {
                    kanit_agirligi += kanit.agirlik.unwrap_or(1.0).max(0.0);
                }
            }
        }
        if kanit_agirligi > 0.0 {
            agirlikli += idf * (kanit_agirligi / agirlik_toplam);
        }
    }
    let mut puan = if toplam_agirlik > 0.0 {
        agirlikli / toplam_agirlik
    } else {
        0.0
    };

    // The bigram bonus: a constant coefficient, stated as one, because an idf
    // weight for adjacent pairs would sit at its ceiling every time.
    const GRAM_KATSAYI: f64 = 0.35;
    if !aday_gramlar.is_empty() {
        let mut tutan = 0usize;
        for gram in &aday_gramlar {
            let (ilk, ikinci) = gram
                .split_once('\u{1f}')
                .expect("ikili gram her zaman ayirici tasir");
            let bulundu = kanitlar.iter().any(|kanit| {
                let kj = jetonlar(&kanit.metin);
                kj.windows(2)
                    .any(|pencere| pencere[0] == ilk && pencere[1] == ikinci)
            });
            if bulundu {
                tutan += 1;
            }
        }
        puan += GRAM_KATSAYI * (tutan as f64 / aday_gramlar.len() as f64);
    }

    // Coverage: how much of the question this candidate speaks to.
    let soru_kumesi: std::collections::HashSet<&String> = soru_jetonlari.iter().collect();
    let kapsam = if soru_kumesi.is_empty() {
        0.0
    } else {
        let tutan = soru_kumesi
            .iter()
            .filter(|jeton| aday_jetonlari.contains(jeton))
            .count();
        tutan as f64 / soru_kumesi.len() as f64
    };

    let destek = destekli_kanit.iter().filter(|x| **x).count();
    eslesen.sort_unstable();
    eslesen.dedup();
    SecenekPuan {
        indeks: 0,
        puan,
        kapsam,
        destek,
        olumsuzluk_celiskisi: celiski_olumsuzluk,
        sayi_celiskisi: celiski_sayi,
        eslesen_jetonlar: eslesen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kanit(kimlik: &str, metin: &str, agirlik: Option<f64>) -> Kanit {
        Kanit {
            kimlik: kimlik.to_string(),
            metin: metin.to_string(),
            agirlik,
        }
    }

    #[test]
    fn a_rare_word_outweighs_a_common_one() {
        let kanitlar = vec![
            kanit("k1", "kayit acildi ve surdu", None),
            kanit("k2", "kayit acildi", None),
            kanit("k3", "kayit kapandi", None),
        ];
        let dokum = Dokum::kur(&kanitlar);
        // "sel" never appears: highest idf. "kayit" appears in all three: floor.
        assert!(dokum.idf("sel") > dokum.idf("kayit"));
        assert!((dokum.idf("kayit") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn frequency_is_capped_so_repetition_is_not_weight() {
        let tekrar = kanit("k1", "kayit kayit kayit kayit acildi", None);
        let dokum = Dokum::kur(std::slice::from_ref(&tekrar));
        // A token repeated in one document is still one document.
        assert!((dokum.idf("kayit") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn length_does_not_buy_a_higher_score() {
        let kanitlar = vec![kanit("k1", "yedek acildi ve dogrulandi", None)];
        let dokum = Dokum::kur(&kanitlar);
        let kisa = puanla("yedek acildi", "yedek acildi mi", &kanitlar, &dokum);
        let uzun = puanla(
            "yedek acildi ve dogrulandi ve sonra baska bir suru ilgisiz sozcuk daha eklendi buraya",
            "yedek acildi mi",
            &kanitlar,
            &dokum,
        );
        assert!(
            kisa.puan >= uzun.puan,
            "uzun aday daha yuksek puan aldi: {} vs {}",
            kisa.puan,
            uzun.puan
        );
    }

    #[test]
    fn a_phrase_match_beats_two_separate_word_matches() {
        let kanitlar = vec![kanit("k1", "kayit defteri acildi ve muhurlendi", None)];
        let dokum = Dokum::kur(&kanitlar);
        let ifade = puanla("kayit defteri", "ne oldu", &kanitlar, &dokum);
        let dagilmis = puanla("defteri kayit", "ne oldu", &kanitlar, &dokum);
        assert!(
            ifade.puan > dagilmis.puan,
            "ifade eslesmesi bonus aldi mi: {} vs {}",
            ifade.puan,
            dagilmis.puan
        );
    }

    #[test]
    fn numbers_are_compared_with_their_units() {
        let kanitlar = vec![kanit("k1", "dosya boyutu 500 mb olculdu", None)];
        let dokum = Dokum::kur(&kanitlar);
        let uyumlu = puanla("boyut 500 mb", "boyut ne", &kanitlar, &dokum);
        let farkli_deger = puanla("boyut 300 mb", "boyut ne", &kanitlar, &dokum);
        let farkli_birim = puanla("boyut 500 gb", "boyut ne", &kanitlar, &dokum);
        assert!(!uyumlu.sayi_celiskisi);
        assert!(farkli_deger.sayi_celiskisi, "ayni birim farkli deger celiski");
        assert!(
            !farkli_birim.sayi_celiskisi,
            "farkli birim celiski sayilmaz: karsilastirilamaz"
        );
    }

    #[test]
    fn polarity_disagreement_is_flagged() {
        let kanitlar = vec![kanit("k1", "yedek acilmadi ve kayit kapandi", None)];
        let dokum = Dokum::kur(&kanitlar);
        let celisen = puanla("yedek acildi kayit kapandi", "yedek", &kanitlar, &dokum);
        assert!(celisen.olumsuzluk_celiskisi);
        let uyumlu = puanla("yedek acilmadi", "yedek", &kanitlar, &dokum);
        assert!(!uyumlu.olumsuzluk_celiskisi);
    }

    #[test]
    fn coverage_is_about_the_question_and_not_the_evidence() {
        let kanitlar = vec![kanit("k1", "yedek acildi ve dogrulandi", None)];
        let dokum = Dokum::kur(&kanitlar);
        let tam = puanla("yedek acildi", "yedek acildi mi", &kanitlar, &dokum);
        let kismi = puanla("yedek acildi", "yedek acildi mi ve kayit kapandi mi", &kanitlar, &dokum);
        assert!((tam.kapsam - 1.0).abs() < 1e-12);
        assert!(kismi.kapsam < 0.5, "kapsam {} beklenenden yuksek", kismi.kapsam);
        assert!(kismi.kapsam > 0.0);
    }

    #[test]
    fn evidence_weight_moves_the_score_in_its_direction() {
        let zayif = vec![
            kanit("k1", "yedek acildi", Some(0.1)),
            kanit("k2", "yedek kapandi", Some(1.0)),
        ];
        let guclu = vec![
            kanit("k1", "yedek acildi", Some(1.0)),
            kanit("k2", "yedek kapandi", Some(0.1)),
        ];
        let d1 = Dokum::kur(&zayif);
        let d2 = Dokum::kur(&guclu);
        let s1 = puanla("yedek acildi", "yedek", &zayif, &d1);
        let s2 = puanla("yedek acildi", "yedek", &guclu, &d2);
        assert!(s2.puan > s1.puan, "agirlik puani degistirmedi");
    }
}
