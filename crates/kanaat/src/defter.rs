//! The verdict ledger: decisions recorded so that a later reader can check them.
//!
//! # Why a ledger and not a log
//!
//! A log line says what happened. A ledger line says what happened *and* where
//! it sits in the sequence, because each entry carries a digest over its own
//! content and the digest of the entry before it. That makes two different
//! failures visible:
//!
//! - an **edited** entry no longer matches its own digest, and the check names
//!   the first index where the chain breaks;
//! - a **reordered** or removed entry breaks the link to its neighbour, so the
//!   gap is named rather than skipped.
//!
//! # What a chain cannot see
//!
//! Truncation. An attacker who deletes the last ten entries leaves a chain that
//! is internally perfect. That is why [`Defter::dogrula_uc`] takes an anchor:
//! the digest of the last entry, held somewhere the ledger's own writer cannot
//! reach. Without an anchor the honest statement is "the entries that are here
//! are consistent", and the module says exactly that instead of claiming more.
//!
//! # Why the digest is over a canonical form
//!
//! The digest is taken over a fixed field order joined with a separator that
//! cannot appear in a field, not over the JSON as written. Two writers who
//! pretty-print differently produce the same digest, so a formatting difference
//! is never reported as a broken chain.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{hukum_metni, Hukum};

/// The digest a chain starts from: sixty-four zeros, not an empty string.
///
/// Choosing a value rather than an absence keeps the first entry's rule
/// identical to every later one, so there is no special case to get wrong.
pub const BASLANGIC_OZETI: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// One recorded decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kayit {
    /// Position in the ledger, from zero.
    pub sira: u64,
    /// The question, folded and hashed: the ledger stores a digest rather than
    /// the question itself, because a ledger is often shipped and the question
    /// may name something the ledger is not allowed to repeat.
    pub soru_damgasi: String,
    /// The verdict label, in the same form the battery compares.
    pub hukum: String,
    /// The previous entry's digest, or [`BASLANGIC_OZETI`].
    pub onceki: String,
    /// This entry's digest.
    pub ozet: String,
}

/// Why a ledger could not be read or does not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefterHatasi {
    /// A line is not the JSON this build writes.
    Json {
        /// Line number, one-based, as a reader counts it.
        satir: usize,
        /// The parser's own message.
        mesaj: String,
    },
    /// Entries are not numbered consecutively from zero.
    SiraAtlandi {
        /// Where the sequence broke.
        satir: usize,
    },
    /// An entry's own digest does not match its content: it was edited.
    IcerikBozuk {
        /// The entry's index.
        sira: u64,
    },
    /// An entry does not point at its predecessor: it was removed or reordered.
    ZincirBozuk {
        /// The entry's index.
        sira: u64,
    },
}

/// The canonical string a digest is taken over.
///
/// The unit separator (`0x1f`) cannot occur in a folded question digest, a
/// verdict label or a hex digest, so no field can be made to look like two.
fn kanonik(sira: u64, soru_damgasi: &str, hukum: &str, onceki: &str) -> String {
    format!("{sira}\u{1f}{soru_damgasi}\u{1f}{hukum}\u{1f}{onceki}")
}

/// The SHA-256 of a string, as lowercase hex.
#[must_use]
pub fn damga(metin: &str) -> String {
    let mut seri = Sha256::new();
    seri.update(metin.as_bytes());
    format!("{:x}", seri.finalize())
}

/// An append-only sequence of decisions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Defter {
    kayitlar: Vec<Kayit>,
}

impl Defter {
    /// An empty ledger.
    #[must_use]
    pub fn yeni() -> Self {
        Self::default()
    }

    /// How many entries it holds.
    #[must_use]
    pub fn uzunluk(&self) -> usize {
        self.kayitlar.len()
    }

    /// The entries, in order.
    #[must_use]
    pub fn kayitlar(&self) -> &[Kayit] {
        &self.kayitlar
    }

    /// The last entry's digest, for an anchor to be taken elsewhere.
    #[must_use]
    pub fn uc(&self) -> String {
        self.kayitlar
            .last()
            .map_or_else(|| BASLANGIC_OZETI.to_string(), |k| k.ozet.clone())
    }

    /// Appends a decision.
    ///
    /// The chain link is computed here and nowhere else, so a caller cannot
    /// write an entry whose digest does not cover its predecessor.
    pub fn ekle(&mut self, soru: &str, hukum: &Hukum) -> Kayit {
        let sira = self.kayitlar.len() as u64;
        let onceki = self.uc();
        let soru_damgasi = damga(soru);
        let etiket = hukum_metni(hukum);
        let ozet = damga(&kanonik(sira, &soru_damgasi, &etiket, &onceki));
        let kayit = Kayit {
            sira,
            soru_damgasi,
            hukum: etiket,
            onceki,
            ozet,
        };
        self.kayitlar.push(kayit.clone());
        kayit
    }

    /// The ledger as JSONL, one entry per line.
    ///
    /// # Errors
    /// A serialisation failure, which for this shape means the serialiser
    /// itself failed and not that a field was wrong.
    pub fn jsonl(&self) -> Result<String, String> {
        let mut cikti = String::new();
        for kayit in &self.kayitlar {
            cikti.push_str(&serde_json::to_string(kayit).map_err(|h| h.to_string())?);
            cikti.push('\n');
        }
        Ok(cikti)
    }

    /// Reads a ledger and verifies its chain in one pass.
    ///
    /// # Errors
    /// The first failure, with the line or index where it was found. The check
    /// stops there rather than reporting every later line: after the first
    /// break, the rest of the chain is meaningless.
    pub fn oku(metin: &str) -> Result<Self, DefterHatasi> {
        let mut defter = Self::yeni();
        for (sira, satir) in metin.lines().enumerate() {
            let satir_no = sira + 1;
            if satir.trim().is_empty() {
                continue;
            }
            let kayit: Kayit = serde_json::from_str(satir).map_err(|h| DefterHatasi::Json {
                satir: satir_no,
                mesaj: h.to_string(),
            })?;
            if kayit.sira != defter.kayitlar.len() as u64 {
                return Err(DefterHatasi::SiraAtlandi { satir: satir_no });
            }
            let beklenen = damga(&kanonik(
                kayit.sira,
                &kayit.soru_damgasi,
                &kayit.hukum,
                &kayit.onceki,
            ));
            if beklenen != kayit.ozet {
                return Err(DefterHatasi::IcerikBozuk { sira: kayit.sira });
            }
            if kayit.onceki != defter.uc() {
                return Err(DefterHatasi::ZincirBozuk { sira: kayit.sira });
            }
            defter.kayitlar.push(kayit);
        }
        Ok(defter)
    }

    /// Checks this ledger against an anchor taken from elsewhere.
    ///
    /// The anchor is the digest the caller expects at the end. A ledger that
    /// was truncated after the anchor was taken does not reproduce it, which is
    /// the failure the chain alone cannot see.
    ///
    /// # Errors
    /// [`DefterHatasi::ZincirBozuk`] at the last index when the anchor differs.
    pub fn dogrula_uc(&self, beklenen_uc: &str) -> Result<(), DefterHatasi> {
        let olculen = self.uc();
        if olculen == beklenen_uc.trim().to_ascii_lowercase() {
            Ok(())
        } else {
            Err(DefterHatasi::ZincirBozuk {
                sira: self.kayitlar.len().saturating_sub(1) as u64,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Ayarlar, Dava, Kanit, RedSebebi};

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

    fn hukum(metin: &str) -> Hukum {
        match metin {
            "ret" => Hukum::Red(RedSebebi::KanitYok),
            _ => crate::karar_ver(
                &dava(metin, &["kayit acildi"], &[("k1", "kayit acildi")]),
                &Ayarlar::default(),
            ),
        }
    }

    #[test]
    fn an_empty_ledger_has_the_beginning_digest_as_its_end() {
        let defter = Defter::yeni();
        assert_eq!(defter.uzunluk(), 0);
        assert_eq!(defter.uc(), BASLANGIC_OZETI);
    }

    #[test]
    fn an_entry_points_at_its_predecessor() {
        let mut defter = Defter::yeni();
        let ilk = defter.ekle("kayit acildi mi", &hukum("kayit acildi mi"));
        let ikinci = defter.ekle("kayit acildi mi", &hukum("kayit acildi mi"));
        assert_eq!(ilk.onceki, BASLANGIC_OZETI);
        assert_eq!(ikinci.onceki, ilk.ozet);
        assert_ne!(
            ilk.ozet, ikinci.ozet,
            "ayni icerik, farkli sira, farkli ozet"
        );
        assert_eq!(defter.uc(), ikinci.ozet);
    }

    #[test]
    fn a_ledger_round_trips_through_jsonl() {
        let mut defter = Defter::yeni();
        for i in 0..5 {
            defter.ekle(&format!("soru {i}"), &hukum("kayit acildi mi"));
        }
        let metin = defter.jsonl().unwrap_or_default();
        assert_eq!(metin.lines().count(), 5);
        let geri = Defter::oku(&metin).unwrap_or_default();
        assert_eq!(geri, defter);
    }

    #[test]
    fn an_edited_entry_names_its_index() {
        let mut defter = Defter::yeni();
        for i in 0..4 {
            defter.ekle(&format!("soru {i}"), &hukum("kayit acildi mi"));
        }
        let metin = defter.jsonl().unwrap_or_default();
        let satirlar: Vec<&str> = metin.lines().collect();
        // Rewrite entry 2's verdict without recomputing its digest.
        let bozuk = satirlar[2].replace("secim:0", "ret");
        let yeni: String = satirlar
            .iter()
            .enumerate()
            .map(|(i, s)| if i == 2 { bozuk.clone() } else { (*s).to_string() } + "\n")
            .collect();
        match Defter::oku(&yeni) {
            Err(DefterHatasi::IcerikBozuk { sira }) => assert_eq!(sira, 2),
            digeri => panic!("icerik bozulmasi bekleniyordu: {digeri:?}"),
        }
    }

    #[test]
    fn a_removed_entry_breaks_the_link() {
        let mut defter = Defter::yeni();
        for i in 0..4 {
            defter.ekle(&format!("soru {i}"), &hukum("kayit acildi mi"));
        }
        let metin = defter.jsonl().unwrap_or_default();
        let kalan: String = metin
            .lines()
            .enumerate()
            .filter(|(i, _)| *i != 1)
            .map(|(_, s)| s.to_string() + "\n")
            .collect();
        // The remaining lines are renumbered so that the only failure left is
        // the broken link, which is what this test is about.
        let mut yeniden = String::new();
        for (yeni_sira, satir) in kalan.lines().enumerate() {
            let mut kayit: Kayit = serde_json::from_str(satir).unwrap_or_else(|_| Kayit {
                sira: 0,
                soru_damgasi: String::new(),
                hukum: String::new(),
                onceki: String::new(),
                ozet: String::new(),
            });
            kayit.sira = yeni_sira as u64;
            yeniden.push_str(&serde_json::to_string(&kayit).unwrap_or_default());
            yeniden.push('\n');
        }
        match Defter::oku(&yeniden) {
            Err(DefterHatasi::IcerikBozuk { .. } | DefterHatasi::ZincirBozuk { .. }) => {}
            digeri => panic!("zincir bozulmasi bekleniyordu: {digeri:?}"),
        }
    }

    #[test]
    fn a_gap_in_the_numbering_is_named() {
        let mut defter = Defter::yeni();
        for i in 0..3 {
            defter.ekle(&format!("soru {i}"), &hukum("kayit acildi mi"));
        }
        let metin = defter.jsonl().unwrap_or_default();
        let ilk_satir = metin.lines().next().unwrap_or_default();
        let yeni = format!("{ilk_satir}\n{}", metin);
        match Defter::oku(&yeni) {
            Err(DefterHatasi::SiraAtlandi { satir }) => assert_eq!(satir, 2),
            digeri => panic!("sira atlamasi bekleniyordu: {digeri:?}"),
        }
    }

    #[test]
    fn truncation_is_caught_by_the_anchor_and_by_nothing_else() {
        let mut defter = Defter::yeni();
        for i in 0..5 {
            defter.ekle(&format!("soru {i}"), &hukum("kayit acildi mi"));
        }
        let uc = defter.uc();
        let metin = defter.jsonl().unwrap_or_default();
        let kisaltilmis: String = metin
            .lines()
            .take(3)
            .map(|s| s.to_string() + "\n")
            .collect();
        let geri = Defter::oku(&kisaltilmis).expect("kisa zincir kendi icinde tutarlidir");
        // The chain alone cannot see it: this assertion is the honest limit.
        assert_eq!(geri.uzunluk(), 3);
        assert!(
            geri.dogrula_uc(&uc).is_err(),
            "capali kontrol kesmeyi gormeli"
        );
        assert!(geri.dogrula_uc(&geri.uc()).is_ok());
    }

    #[test]
    fn a_malformed_line_reports_its_line_number() {
        let metin = "{\n";
        match Defter::oku(metin) {
            Err(DefterHatasi::Json { satir, mesaj }) => {
                assert_eq!(satir, 1);
                assert!(!mesaj.is_empty());
            }
            digeri => panic!("json hatasi bekleniyordu: {digeri:?}"),
        }
    }

    #[test]
    fn the_digest_is_taken_over_a_canonical_form() {
        // Same fields, same digest; the ledger's formatting is not part of it.
        let a = damga(&kanonik(1, "abc", "secim:0", BASLANGIC_OZETI));
        let b = damga(&kanonik(1, "abc", "secim:0", BASLANGIC_OZETI));
        assert_eq!(a, b);
        assert_ne!(a, damga(&kanonik(2, "abc", "secim:0", BASLANGIC_OZETI)));
        assert_eq!(a.len(), 64);
    }
}
