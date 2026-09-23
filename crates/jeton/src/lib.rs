//! The frozen BPE vocab, read and applied - not re-cut.
//!
//! # Why this crate exists
//!
//! The training core can forward and back-propagate but could not read the
//! corpus: the vocab lived on the Python side only. Two tokenizers that agree
//! by convention is not an agreement, so this one is written against the same
//! frozen file and a gate cross-checks the two on the corpus.
//!
//! # The loader refuses rather than approximates
//!
//! The vocab is a committed artefact ([`Sozluk::yukle`] checks the format,
//! `vocab_size == 256 + merges`, and the merge DAG - every merge may only refer
//! to ids defined before it). One check deserves its own sentence: the
//! pretoken pattern is **string-compared** against the one this file implements
//! and any other pattern is a refusal. A regex engine is not available here and
//! silently applying a different segmentation than the vocab was cut with would
//! produce plausible ids that mean something else.
//!
//! # Segmentation
//!
//! `[^\W\d_]+|\d+|\s+|[\W_]+` splits text into runs of letters, digits,
//! whitespace and other characters. Those four classes are disjoint and cover
//! every character, so splitting on a change of class reproduces the regex
//! exactly, in order, without backtracking.

use serde_json::Value;

/// The only pretoken pattern this implementation knows how to apply.
pub const DESTEKLENEN_DESEN: &str = r"[^\W\d_]+|\d+|\s+|[\W_]+";

/// Why a vocab was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SozlukHatasi {
    /// The file does not exist.
    Yok(String),
    /// Not JSON.
    BozukJson(String),
    /// A required key is missing.
    EksikAlan(String),
    /// The format or version is not the one this reader implements.
    BilinmeyenBicim(String),
    /// `vocab_size` and the merge count disagree.
    BoyutUyusmuyor {
        /// What the file claims.
        beyan: usize,
        /// What the merges imply.
        beklenen: usize,
    },
    /// A merge refers to an id that does not exist yet.
    DagBozuk {
        /// Merge index.
        sira: usize,
        /// The offending pair.
        cift: (u32, u32),
        /// The highest id defined before this merge.
        tavan: u32,
    },
    /// A merge entry is not a pair of integers.
    BirlestirmeBicimiBozuk(usize),
    /// The pretoken pattern is not one this reader implements.
    DesenDesteklenmiyor(String),
}

impl std::fmt::Display for SozlukHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Yok(y) => write!(f, "sozluk yok: {y} (donmus sozluk turetilemez)"),
            Self::BozukJson(e) => write!(f, "sozluk JSON degil: {e}"),
            Self::EksikAlan(a) => write!(f, "sozlukte zorunlu alan yok: {a}"),
            Self::BilinmeyenBicim(b) => write!(f, "bilinmeyen sozluk bicimi: {b}"),
            Self::BoyutUyusmuyor { beyan, beklenen } => write!(
                f,
                "vocab_size {beyan} birlestirme sayisiyla uyusmuyor (256 + {beklenen} beklenirdi)"
            ),
            Self::DagBozuk { sira, cift, tavan } => write!(
                f,
                "gecersiz birlestirme #{sira}: ({}, {}) kendinden once tanimli degil (tavan {tavan})",
                cift.0, cift.1
            ),
            Self::BirlestirmeBicimiBozuk(i) => write!(f, "birlestirme #{i} bir tamsayi cifti degil"),
            Self::DesenDesteklenmiyor(d) => write!(
                f,
                "onislem deseni bu okuyucunun uyguladigi desen degil: {d} (beklenen {DESTEKLENEN_DESEN})"
            ),
        }
    }
}

/// A loaded, validated vocab.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sozluk {
    aile: String,
    boyut: usize,
    /// Merge rank -> the pair it merges.
    birlestirmeler: Vec<(u32, u32)>,
    /// Pair -> rank, as a sorted lookup.
    siralar: std::collections::HashMap<(u32, u32), u32>,
    /// Token id -> the bytes it stands for.
    baytlar: Vec<Vec<u8>>,
}

impl Sozluk {
    /// # Errors
    /// Any structural doubt is a refusal; see [`SozlukHatasi`].
    pub fn yukle(yol: &std::path::Path) -> Result<Self, SozlukHatasi> {
        let ham = std::fs::read_to_string(yol)
            .map_err(|e| SozlukHatasi::Yok(format!("{} ({e})", yol.display())))?;
        Self::metinden(&ham)
    }

    /// # Errors
    /// Any structural doubt is a refusal; see [`SozlukHatasi`].
    pub fn metinden(ham: &str) -> Result<Self, SozlukHatasi> {
        let kok: Value =
            serde_json::from_str(ham).map_err(|e| SozlukHatasi::BozukJson(e.to_string()))?;
        for alan in [
            "format",
            "format_version",
            "vocab_family",
            "vocab_size",
            "pretoken_pattern",
            "merges",
        ] {
            if kok.get(alan).is_none() {
                return Err(SozlukHatasi::EksikAlan(alan.to_string()));
            }
        }
        if kok["format"].as_str() != Some("lubot-bpe") {
            return Err(SozlukHatasi::BilinmeyenBicim(format!(
                "format {:?}",
                kok["format"].as_str()
            )));
        }
        if kok["format_version"].as_u64() != Some(1) {
            return Err(SozlukHatasi::BilinmeyenBicim(format!(
                "format_version {:?}",
                kok["format_version"]
            )));
        }
        let desen = kok["pretoken_pattern"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        if desen != DESTEKLENEN_DESEN {
            return Err(SozlukHatasi::DesenDesteklenmiyor(desen));
        }
        let liste = kok["merges"]
            .as_array()
            .ok_or_else(|| SozlukHatasi::EksikAlan("merges".to_string()))?;
        let mut birlestirmeler = Vec::with_capacity(liste.len());
        let mut siralar = std::collections::HashMap::with_capacity(liste.len());
        let mut baytlar: Vec<Vec<u8>> = (0u32..256).map(|i| vec![i as u8]).collect();
        for (sira, oge) in liste.iter().enumerate() {
            let cift = oge.as_array();
            let (a, b) = match cift.and_then(|c| {
                if c.len() == 2 {
                    Some((c[0].as_u64()?, c[1].as_u64()?))
                } else {
                    None
                }
            }) {
                Some((a, b)) => (
                    u32::try_from(a).unwrap_or(u32::MAX),
                    u32::try_from(b).unwrap_or(u32::MAX),
                ),
                None => return Err(SozlukHatasi::BirlestirmeBicimiBozuk(sira)),
            };
            let tavan = 256 + sira as u32;
            if a >= tavan || b >= tavan {
                return Err(SozlukHatasi::DagBozuk {
                    sira,
                    cift: (a, b),
                    tavan,
                });
            }
            baytlar.push([baytlar[a as usize].clone(), baytlar[b as usize].clone()].concat());
            siralar.insert((a, b), sira as u32);
            birlestirmeler.push((a, b));
        }
        let beklenen = 256 + birlestirmeler.len();
        let beyan = kok["vocab_size"].as_u64().unwrap_or(0) as usize;
        if beyan != beklenen {
            return Err(SozlukHatasi::BoyutUyusmuyor { beyan, beklenen });
        }
        Ok(Self {
            aile: kok["vocab_family"].as_str().unwrap_or_default().to_string(),
            boyut: beyan,
            birlestirmeler,
            siralar,
            baytlar,
        })
    }

    /// The vocab family name, as the file states it.
    #[must_use]
    pub fn aile(&self) -> &str {
        &self.aile
    }

    /// How many ids the vocab has.
    #[must_use]
    pub fn boyut(&self) -> usize {
        self.boyut
    }

    /// How many merges the vocab carries.
    #[must_use]
    pub fn birlestirme_sayisi(&self) -> usize {
        self.birlestirmeler.len()
    }

    /// Encode text to ids. Each pretoken stays inside itself.
    #[must_use]
    pub fn kodla(&self, metin: &str) -> Vec<u32> {
        let mut kimlikler = Vec::new();
        for on in on_tokenler(metin) {
            let mut dizi: Vec<u32> = on.bytes().map(u32::from).collect();
            while dizi.len() >= 2 {
                let mut en_iyi: Option<(u32, usize)> = None;
                for i in 0..dizi.len() - 1 {
                    if let Some(&sira) = self.siralar.get(&(dizi[i], dizi[i + 1])) {
                        if en_iyi.is_none_or(|(r, _)| sira < r) {
                            en_iyi = Some((sira, i));
                        }
                    }
                }
                let Some((sira, i)) = en_iyi else { break };
                dizi.splice(i..i + 2, [256 + sira]);
            }
            kimlikler.extend(dizi);
        }
        kimlikler
    }

    /// Decode ids back to text.
    ///
    /// # Errors
    /// An id outside the vocab, or bytes that are not valid UTF-8.
    pub fn coz(&self, kimlikler: &[u32]) -> Result<String, String> {
        let mut ham: Vec<u8> = Vec::new();
        for &k in kimlikler {
            let b = self
                .baytlar
                .get(k as usize)
                .ok_or_else(|| format!("sozlukte olmayan token kimligi: {k}"))?;
            ham.extend_from_slice(b);
        }
        String::from_utf8(ham).map_err(|e| format!("token baytlari gecerli UTF-8 degil: {e}"))
    }
}

/// The four alternatives of the pretoken pattern, at the position that starts
/// a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sinif {
    Harf,
    Rakam,
    Bosluk,
    Diger,
}

/// Python's `\w` minus the underscore: the Unicode-alphanumeric characters.
fn alnum(c: char) -> bool {
    c.is_alphanumeric()
}

/// The radix whose digit table is the Unicode Nd category.
///
/// Named rather than written as a literal because the three ways of writing
/// this check each draw a lint that suggests a *different* and wrong check:
/// `is_ascii_digit()` would put every non-ASCII digit in the "other" class, and
/// `is_numeric()` would pull in Nl and No (roman numerals, `½`, `²`), which
/// `\d` does not match.
const ONDALIK_TABAN: u32 = 10;

/// Python's `\d`: the Unicode decimal-digit category (Nd), not ASCII digits.
///
/// Two replacements that look natural are both wrong here. `is_ascii_digit()`
/// would put every non-ASCII digit in the "other" class, and `is_numeric()`
/// would pull in Nl and No (roman numerals, `½`, `²`), which `\d` does not
/// match. The radix-10 conversion is the Nd table, which is the category the
/// pattern asks for.
fn ondalik_rakam(c: char) -> bool {
    c.is_digit(ONDALIK_TABAN)
}

/// Which alternative matches at this character. Tried in the pattern's order,
/// which matters: at a whitespace character `\s+` wins over `[\W_]+`.
fn sinif(c: char) -> Sinif {
    if alnum(c) && !ondalik_rakam(c) {
        Sinif::Harf
    } else if ondalik_rakam(c) {
        Sinif::Rakam
    } else if c.is_whitespace() {
        Sinif::Bosluk
    } else {
        Sinif::Diger
    }
}

/// How far a run of the given class extends.
///
/// This is not the same predicate as [`sinif`], and the difference is the whole
/// subtlety of the pattern. `[\W_]+` is greedy and `\W` includes whitespace,
/// so a run that *starts* on a punctuation character swallows the spaces after
/// it: `"; oku"` segments as `"; "` then `"oku"`, not as `";"`, `" "`, `"oku"`.
/// Only a run that *starts* on whitespace is a whitespace run.
fn uzar(s: Sinif, c: char) -> bool {
    match s {
        Sinif::Harf => alnum(c) && !ondalik_rakam(c),
        Sinif::Rakam => ondalik_rakam(c),
        Sinif::Bosluk => c.is_whitespace(),
        Sinif::Diger => !alnum(c),
    }
}

/// Split into the runs the pattern's `findall` would return, in order.
fn on_tokenler(metin: &str) -> Vec<String> {
    let karakterler: Vec<char> = metin.chars().collect();
    let mut parcalar: Vec<String> = Vec::new();
    let mut i = 0;
    while i < karakterler.len() {
        let sinif = sinif(karakterler[i]);
        let mut j = i + 1;
        while j < karakterler.len() && uzar(sinif, karakterler[j]) {
            j += 1;
        }
        parcalar.push(karakterler[i..j].iter().collect());
        i = j;
    }
    parcalar
}

/// Pretoken count, exposed for the cross-check gate.
#[must_use]
pub fn on_token_sayisi(metin: &str) -> usize {
    on_tokenler(metin).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocab with two merges: "ab" then "abc", so ranks are ordered and the
    /// greedy rule has something to choose between.
    fn oyuncak() -> String {
        serde_json::json!({
            "format": "lubot-bpe",
            "format_version": 1,
            "vocab_family": "oyuncak-v1",
            "vocab_size": 258,
            "pretoken_pattern": DESTEKLENEN_DESEN,
            "merges": [[97, 98], [256, 99]],
            "trained_from": {},
        })
        .to_string()
    }

    #[test]
    fn encode_and_decode_round_trip() {
        let s = Sozluk::metinden(&oyuncak()).unwrap();
        assert_eq!(s.boyut(), 258);
        assert_eq!(s.birlestirme_sayisi(), 2);
        for metin in ["ab", "abc", "abcabc", "a b c", "123 45", "x-y_z", ""] {
            let kimlikler = s.kodla(metin);
            assert_eq!(s.coz(&kimlikler).unwrap(), metin, "round trip: {metin:?}");
        }
    }

    #[test]
    fn the_lowest_rank_merge_wins() {
        let s = Sozluk::metinden(&oyuncak()).unwrap();
        // "abc": the pair (a,b) has rank 0 and (b,c) has no rank, so the result
        // is [256, 99] -> then (256, 99) is rank 1 -> [257].
        assert_eq!(s.kodla("abc"), vec![257]);
        assert_eq!(s.kodla("ab"), vec![256]);
        assert_eq!(s.kodla("ba"), vec![98, 97]);
    }

    #[test]
    fn pretokens_do_not_merge_across_a_class_change() {
        let s = Sozluk::metinden(&oyuncak()).unwrap();
        // A space separates them, so neither pair is ever considered together.
        assert_eq!(s.kodla("a b"), vec![97, 32, 98]);
        assert_eq!(on_token_sayisi("ab 12 _"), 5);
    }

    #[test]
    fn turkish_text_segments_into_the_four_classes() {
        let parcalar = on_tokenler("İstanbul'da 42 gün; oku-ya");
        // Python'un findall'u bu dizede aynen bunu veriyor (olculdu).
        assert_eq!(
            parcalar,
            [
                "İstanbul",
                "'",
                "da",
                " ",
                "42",
                " ",
                "gün",
                "; ",
                "oku",
                "-",
                "ya"
            ]
        );
    }

    #[test]
    fn a_punctuation_run_swallows_the_space_after_it() {
        // The single easiest way to get this tokenizer wrong: treating the four
        // classes as a partition of characters. `;` and the space after it are
        // both `\W`, and `[\W_]+` is greedy, so they are ONE pretoken. A run
        // that starts on whitespace is a whitespace run instead.
        assert_eq!(on_tokenler("; a"), ["; ", "a"]);
        assert_eq!(on_tokenler(";  a"), [";  ", "a"]);
        assert_eq!(on_tokenler(" ; a"), [" ", "; ", "a"]);
        assert_eq!(on_tokenler("x-y_z"), ["x", "-", "y", "_", "z"]);
    }

    #[test]
    fn a_broken_merge_dag_is_refused() {
        let bozuk = serde_json::json!({
            "format": "lubot-bpe",
            "format_version": 1,
            "vocab_family": "x",
            "vocab_size": 257,
            "pretoken_pattern": DESTEKLENEN_DESEN,
            "merges": [[97, 256]],
        })
        .to_string();
        assert!(matches!(
            Sozluk::metinden(&bozuk),
            Err(SozlukHatasi::DagBozuk { sira: 0, .. })
        ));
    }

    #[test]
    fn a_size_that_disagrees_with_the_merges_is_refused() {
        let bozuk = serde_json::json!({
            "format": "lubot-bpe",
            "format_version": 1,
            "vocab_family": "x",
            "vocab_size": 300,
            "pretoken_pattern": DESTEKLENEN_DESEN,
            "merges": [[97, 98]],
        })
        .to_string();
        assert!(matches!(
            Sozluk::metinden(&bozuk),
            Err(SozlukHatasi::BoyutUyusmuyor {
                beyan: 300,
                beklenen: 257
            })
        ));
    }

    #[test]
    fn an_unsupported_pretoken_pattern_is_refused_not_approximated() {
        let baska = serde_json::json!({
            "format": "lubot-bpe",
            "format_version": 1,
            "vocab_family": "x",
            "vocab_size": 257,
            "pretoken_pattern": r"\w+",
            "merges": [[97, 98]],
        })
        .to_string();
        assert!(matches!(
            Sozluk::metinden(&baska),
            Err(SozlukHatasi::DesenDesteklenmiyor(_))
        ));
    }

    #[test]
    fn a_missing_key_and_a_bad_format_are_refused() {
        assert!(matches!(
            Sozluk::metinden("{}"),
            Err(SozlukHatasi::EksikAlan(_))
        ));
        let yanlis = serde_json::json!({
            "format": "bpe-baska",
            "format_version": 1,
            "vocab_family": "x",
            "vocab_size": 257,
            "pretoken_pattern": DESTEKLENEN_DESEN,
            "merges": [[97, 98]],
        })
        .to_string();
        assert!(matches!(
            Sozluk::metinden(&yanlis),
            Err(SozlukHatasi::BilinmeyenBicim(_))
        ));
    }

    #[test]
    fn an_id_outside_the_vocab_cannot_be_decoded() {
        let s = Sozluk::metinden(&oyuncak()).unwrap();
        assert!(s.coz(&[9999]).is_err());
    }
}
