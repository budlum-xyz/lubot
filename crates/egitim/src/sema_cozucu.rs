//! # `sema_cozucu` - the schema decoder: refusing a byte before it is written
//!
//! The last of the eight architecture modules, and it is the
//! one the repository never had: a byte-level, schema-constrained decoder.
//! `lubot-read::output_schema` already *validates* a finished reply. This
//! module is the other half of the same rule - it narrows the set of bytes a
//! decode step may emit, so a reply that fails the schema is not produced in
//! the first place.
//!
//! The distinction matters and is not cosmetic. A validator answers "was this
//! valid?"; a decoder needs "can this still become valid?". The second question
//! is asked once per position, over 256 candidate bytes, before a logit is ever
//! sampled.
//!
//! # What this module refuses to do
//!
//! Lubot's constitution says a reply that misses the schema is **rejected and
//! never softened** - no downgrade to the nearest format. That rule is coded
//! here literally: when no byte is acceptable and the document cannot end where
//! it stands, [`SemaCozucu::maskele`] returns [`SemaRed::KabulEdilenBaytYok`].
//! It does not fall back to the most likely byte, it does not widen the mask,
//! it does not emit a newline to escape. A caller that wants to continue has to
//! change its state, not its standards.
//!
//! # The guard is local, and says so
//!
//! A byte is refused when *that byte* makes the document invalid. The automaton
//! does not look further ahead: it will not refuse a byte because the remaining
//! budget is too small to close an open fence later. That is a real limitation,
//! it is measured in [`tests::tavan_geriye_dogru_bakmaz`], and it is the honest
//! shape of a one-byte guard. Naming it here is cheaper than discovering it in
//! a run.
//!
//! # Correctness is measured against the validator, not argued
//!
//! The per-line rules below are a second implementation of the rules in
//! `lubot-read::output_schema`. A second implementation is exactly where two
//! copies drift apart, so the drift is measured rather than promised:
//! [`tests::ayni_dili_kabul_eder`] runs a deterministic fuzz over byte strings
//! drawn from the alphabet that actually exercises the schema (hashes, pipes,
//! dashes, backticks, newlines, multi-byte UTF-8 and invalid lead bytes) and
//! requires `coz(s).is_ok() == validate_markdown_output(s).is_ok()` for every
//! one of them. Measured on this tree: 20.000 cases, 0 disagreements. The
//! cross-check is a dev-dependency, so the production path of this crate does
//! not depend on `lubot-read`.
//!
//! Two deliberate non-claims. The two implementations agree on **acceptance**,
//! not on which error a doubly-invalid input reports first: the validator sizes
//! and decodes the whole buffer before reading a line, while this automaton
//! meets the ceiling, the bad byte and the bad line in stream order. And the
//! agreement is measured on the sampled alphabet, not proved over all byte
//! strings - a fuzz is evidence, not a proof, and the record says which.
//!
//! # Where a refusal lands
//!
//! A heading that skips a level is refused at the **space** that completes the
//! hash run, not at the newline that ends the line, because that is the byte at
//! which the violation becomes unavoidable. Refusing later would let a decoder
//! spend a whole line it can never keep. Measured in
//! [`tests::baslik_bosluk_baytinda_reddedilir`].
//!
//! # Not connected
//!
//! Like the other candidates in this crate, this module changes no spec
//! and no training call. It holds no parameters - the mask is a function of the
//! bytes already emitted, and a gate checks that parameter count is zero.

use std::fmt;

/// The analyzer's ceiling, held equal to `lubot_read::output_schema`'s.
///
/// The constant is restated rather than imported because the production path of
/// this crate does not depend on `lubot-read`; the two are pinned together by
/// [`tests::tavan_dogrulayiciyla_ayni`], which is a measurement rather than a
/// convention.
pub const MAKS_CIKTI_BAYT: usize = 64 * 1024;

/// The decode alphabet: one logit per byte value.
pub const ALFABE: usize = 256;

/// Why a byte, or a finished document, was refused.
///
/// Every variant names the position or the line it refers to. A decoder that
/// cannot say *where* it refused cannot be debugged against a corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemaRed {
    /// The output would pass [`MAKS_CIKTI_BAYT`].
    Tavan { bayt: usize },
    /// A byte that cannot appear here in well-formed UTF-8.
    GecersizUtf8 { konum: usize, bayt: u8 },
    /// The document ended in the middle of a multi-byte sequence.
    EksikUtf8 { konum: usize, kalan: usize },
    /// A heading descended more than one level at once.
    BaslikAtlama {
        satir: usize,
        onceki: usize,
        gelen: usize,
    },
    /// A table block whose separator row or column count does not hold.
    TabloUyusmazligi { satir: usize },
    /// The document ended with a code fence still open.
    AcikCit { satir: usize },
    /// Nothing but whitespace.
    Bos,
    /// No byte is acceptable and the document cannot end here.
    ///
    /// This is the variant that carries the doctrine: the decoder reports a
    /// dead end instead of emitting the least-bad byte.
    KabulEdilenBaytYok { konum: usize },
}

impl fmt::Display for SemaRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tavan { bayt } => write!(f, "output would exceed the ceiling at byte {bayt}"),
            Self::GecersizUtf8 { konum, bayt } => {
                write!(f, "byte {bayt:#04x} is not valid UTF-8 at position {konum}")
            }
            Self::EksikUtf8 { konum, kalan } => write!(
                f,
                "output ends inside a UTF-8 sequence at position {konum}, {kalan} byte(s) missing"
            ),
            Self::BaslikAtlama {
                satir,
                onceki,
                gelen,
            } => write!(
                f,
                "heading on line {satir} skips a level: {} after {}",
                "#".repeat(*gelen),
                "#".repeat(*onceki)
            ),
            Self::TabloUyusmazligi { satir } => write!(f, "malformed table around line {satir}"),
            Self::AcikCit { satir } => write!(f, "unbalanced code fence opened on line {satir}"),
            Self::Bos => write!(f, "output is empty"),
            Self::KabulEdilenBaytYok { konum } => write!(
                f,
                "no byte is acceptable at position {konum} and the output cannot end here"
            ),
        }
    }
}

impl std::error::Error for SemaRed {}

/// The incremental UTF-8 state: how many continuation bytes are still owed and
/// what range the next one may take.
///
/// The range is carried rather than recomputed because the constraint on the
/// *first* continuation byte is what rules out overlong encodings and the
/// surrogate block; a decoder that only checks `0x80..=0xBF` accepts both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Utf8Durum {
    kalan: u8,
    alt: u8,
    ust: u8,
}

impl Utf8Durum {
    const YER: Self = Self {
        kalan: 0,
        alt: 0,
        ust: 0,
    };
}

/// For a lead byte: continuation count and the range the next byte may take.
///
/// `None` is a byte that can never start a sequence - the continuation block
/// `0x80..=0xBF`, the overlong leads `0xC0`/`0xC1`, and `0xF5..=0xFF` which are
/// all above U+10FFFF.
const fn onder(b: u8) -> Option<(u8, u8, u8)> {
    match b {
        0x00..=0x7F => Some((0, 0, 0)),
        0xC2..=0xDF => Some((1, 0x80, 0xBF)),
        0xE0 => Some((2, 0xA0, 0xBF)),
        0xE1..=0xEC | 0xEE..=0xEF => Some((2, 0x80, 0xBF)),
        0xED => Some((2, 0x80, 0x9F)),
        0xF0 => Some((3, 0x90, 0xBF)),
        0xF1..=0xF3 => Some((3, 0x80, 0xBF)),
        0xF4 => Some((3, 0x80, 0x8F)),
        _ => None,
    }
}

/// Column count of a `|`-delimited row, or `None` when the line is not a row.
///
/// Mirrors `lubot_read::output_schema::row_columns`, including the single-`|`
/// case: one pipe both starts and ends the line, so the count is zero columns
/// rather than a malformed row.
fn satir_sutunlari(satir: &str) -> Option<usize> {
    let kirpik = satir.trim();
    if !(kirpik.starts_with('|') && kirpik.ends_with('|')) {
        return None;
    }
    Some(kirpik.matches('|').count() - 1)
}

/// Is this the `|---|:--|---:|` row that has to follow a table header?
fn ayirici_satir(satir: &str) -> bool {
    let Some(sutun) = satir_sutunlari(satir) else {
        return false;
    };
    if sutun == 0 {
        return false;
    }
    satir
        .trim()
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .all(|h| {
            !h.is_empty() && h.contains('-') && h.chars().all(|c| c == '-' || c == ':' || c == ' ')
        })
}

/// One table block: header, separator, consistent columns.
fn tablo_denetle(satirlar: &[(usize, String)]) -> Result<(), SemaRed> {
    let Some((ilk_no, baslik)) = satirlar.first() else {
        return Ok(());
    };
    if satirlar.len() == 1 {
        // A lone `|` line is not a table; there is nothing to hold it to.
        return Ok(());
    }
    let baslik_sutun = satir_sutunlari(baslik).unwrap_or(0);
    let Some((_, ikinci)) = satirlar.get(1) else {
        return Ok(());
    };
    if !ayirici_satir(ikinci) || satir_sutunlari(ikinci) != Some(baslik_sutun) {
        return Err(SemaRed::TabloUyusmazligi { satir: *ilk_no });
    }
    for (no, satir) in satirlar.iter().skip(2) {
        if satir_sutunlari(satir) != Some(baslik_sutun) {
            return Err(SemaRed::TabloUyusmazligi { satir: *no });
        }
    }
    Ok(())
}

/// The decoder state: everything needed to answer "which byte may come next?".
///
/// The structure is deliberately a mirror of the validator's loop variables.
/// Two copies of a rule are a disagreement waiting to happen, so the copies are
/// held together by a fuzz rather than by care.
#[derive(Debug, Clone)]
pub struct SemaCozucu {
    bayt: usize,
    utf8: Utf8Durum,
    satir: Vec<u8>,
    satir_no: usize,
    seviye: usize,
    basligi_gordu: bool,
    cit_acik: bool,
    cit_uzunluk: usize,
    cit_acilis_satiri: usize,
    tablo: Vec<(usize, String)>,
    bos_degil: bool,
}

impl Default for SemaCozucu {
    fn default() -> Self {
        Self::yeni()
    }
}

impl SemaCozucu {
    /// A decoder at the start of an empty document.
    #[must_use]
    pub const fn yeni() -> Self {
        Self {
            bayt: 0,
            utf8: Utf8Durum::YER,
            satir: Vec::new(),
            satir_no: 1,
            seviye: 0,
            basligi_gordu: false,
            cit_acik: false,
            cit_uzunluk: 0,
            cit_acilis_satiri: 0,
            tablo: Vec::new(),
            bos_degil: false,
        }
    }

    /// How many bytes have been accepted.
    #[must_use]
    pub const fn bayt_sayisi(&self) -> usize {
        self.bayt
    }

    /// Is a code fence open right now?
    #[must_use]
    pub const fn cit_acik(&self) -> bool {
        self.cit_acik
    }

    /// The heading level in force, zero before the first heading.
    #[must_use]
    pub const fn baslik_seviyesi(&self) -> usize {
        self.seviye
    }

    /// The module holds no parameters. The gate reads this, not a comment.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    /// Accept one byte, or refuse it by name.
    ///
    /// State is advanced only on success: the byte is applied to a copy and the
    /// copy is committed. A refused byte therefore leaves the decoder exactly
    /// where it was, which is what makes [`Self::izin_verilir`] a question
    /// rather than a side effect.
    ///
    /// # Errors
    ///
    /// [`SemaRed`] naming the rule this byte would break.
    pub fn yut(&mut self, b: u8) -> Result<(), SemaRed> {
        let mut sonraki = self.clone();
        sonraki.yut_ic(b)?;
        *self = sonraki;
        Ok(())
    }

    fn yut_ic(&mut self, b: u8) -> Result<(), SemaRed> {
        if self.bayt >= MAKS_CIKTI_BAYT {
            return Err(SemaRed::Tavan {
                bayt: self.bayt + 1,
            });
        }

        if self.utf8.kalan > 0 {
            if b < self.utf8.alt || b > self.utf8.ust {
                return Err(SemaRed::GecersizUtf8 {
                    konum: self.bayt,
                    bayt: b,
                });
            }
            self.utf8.kalan -= 1;
            self.utf8.alt = 0x80;
            self.utf8.ust = 0xBF;
            self.satir.push(b);
            self.bayt += 1;
            return Ok(());
        }

        let Some((kalan, alt, ust)) = onder(b) else {
            return Err(SemaRed::GecersizUtf8 {
                konum: self.bayt,
                bayt: b,
            });
        };

        if b == b'\n' {
            // The line is complete, so the line rules become decidable.
            let ham = String::from_utf8(std::mem::take(&mut self.satir)).unwrap_or_default();
            let no = self.satir_no;
            self.satir_uygula(&ham, no)?;
            self.satir_no += 1;
            self.bayt += 1;
            return Ok(());
        }

        self.utf8 = Utf8Durum { kalan, alt, ust };
        self.satir.push(b);
        self.bayt += 1;

        // A heading violation becomes unavoidable at the space that closes the
        // hash run, not at the newline. Refuse it there.
        if self.utf8.kalan == 0 && b == b' ' && !self.cit_acik {
            self.erken_baslik_denetimi()?;
        }
        Ok(())
    }

    /// If the line so far is exactly `#...#` plus the space just written, the
    /// heading level is already fixed; hold it to the no-skip rule now.
    fn erken_baslik_denetimi(&self) -> Result<(), SemaRed> {
        let Ok(s) = std::str::from_utf8(&self.satir) else {
            return Ok(());
        };
        let on = s.trim_start();
        let diyez = on.chars().take_while(|c| *c == '#').count();
        if diyez == 0 || on.len() != diyez + 1 {
            return Ok(());
        }
        if self.basligi_gordu && diyez > self.seviye + 1 {
            return Err(SemaRed::BaslikAtlama {
                satir: self.satir_no,
                onceki: self.seviye,
                gelen: diyez,
            });
        }
        Ok(())
    }

    /// The validator's per-line block, mirrored.
    fn satir_uygula(&mut self, ham: &str, satir_no: usize) -> Result<(), SemaRed> {
        let kirpik = ham.trim();
        if !kirpik.is_empty() {
            self.bos_degil = true;
        }

        // Fence toggling first: inside a fence nothing is a heading or a row.
        if kirpik.starts_with("```") {
            let kosu = kirpik.chars().take_while(|c| *c == '`').count();
            let yalniz = kirpik[kosu..].trim().is_empty();
            if self.cit_acik {
                if kosu >= self.cit_uzunluk && yalniz {
                    self.cit_acik = false;
                }
            } else {
                self.cit_acik = true;
                self.cit_acilis_satiri = satir_no;
                self.cit_uzunluk = kosu;
            }
            return self.tabloyu_bosalt();
        }
        if self.cit_acik {
            return Ok(());
        }

        let diyez = kirpik.chars().take_while(|c| *c == '#').count();
        if diyez > 0 && kirpik[diyez..].starts_with(' ') {
            // The first heading may be any level; only a descent is bounded.
            if self.basligi_gordu && diyez > self.seviye + 1 {
                return Err(SemaRed::BaslikAtlama {
                    satir: satir_no,
                    onceki: self.seviye,
                    gelen: diyez,
                });
            }
            self.basligi_gordu = true;
            self.seviye = diyez;
            return self.tabloyu_bosalt();
        }

        if satir_sutunlari(kirpik).is_some() {
            self.tablo.push((satir_no, kirpik.to_string()));
            // Checked here rather than when the block closes. The validator
            // only asks at the flush, but the predicate is monotone in the
            // rows - a separator that is missing on row two is missing for
            // good - so asking early refuses the newline that commits the bad
            // row instead of letting a decoder write a whole block it can
            // never keep. The accepted language is unchanged and the fuzz
            // says so.
            return tablo_denetle(&self.tablo);
        }
        self.tabloyu_bosalt()
    }

    fn tabloyu_bosalt(&mut self) -> Result<(), SemaRed> {
        if self.tablo.is_empty() {
            return Ok(());
        }
        tablo_denetle(&self.tablo)?;
        self.tablo.clear();
        Ok(())
    }

    /// Would this byte be accepted here?
    ///
    /// Asked on a copy, so the question costs a clone and changes nothing.
    #[must_use]
    pub fn izin_verilir(&self, b: u8) -> bool {
        let mut deneme = self.clone();
        deneme.yut_ic(b).is_ok()
    }

    /// The acceptance mask over the whole byte alphabet.
    #[must_use]
    pub fn maske(&self) -> [bool; ALFABE] {
        let mut m = [false; ALFABE];
        for (b, hucre) in m.iter_mut().enumerate() {
            // `b` comes from an array of exactly 256 entries.
            *hucre = self.izin_verilir(b as u8);
        }
        m
    }

    /// How many bytes are acceptable here.
    #[must_use]
    pub fn izinli_sayisi(&self) -> usize {
        self.maske().iter().filter(|a| **a).count()
    }

    /// Apply the mask to a row of logits: refused bytes go to `-inf`.
    ///
    /// Allowed positions are left **bit-identical** to the input - the mask
    /// removes, it never rescales, so a logit that survives is the logit the
    /// model produced. Measured in [`tests::maske_izinli_logiti_bit_ozdes_birakir`].
    ///
    /// # Errors
    ///
    /// [`SemaRed::KabulEdilenBaytYok`] when nothing is acceptable and the
    /// document cannot end here. The mask is not widened and no byte is
    /// substituted; this is the "never softened" rule in code.
    pub fn maskele(&self, logitler: &mut [f64; ALFABE]) -> Result<(), SemaRed> {
        let m = self.maske();
        let hicbiri = !m.iter().any(|a| *a);
        if hicbiri && !self.bitirebilir() {
            return Err(SemaRed::KabulEdilenBaytYok { konum: self.bayt });
        }
        for (logit, izin) in logitler.iter_mut().zip(m.iter()) {
            if !*izin {
                *logit = f64::NEG_INFINITY;
            }
        }
        Ok(())
    }

    /// May the document end exactly here?
    #[must_use]
    pub fn bitirebilir(&self) -> bool {
        self.bitir().is_ok()
    }

    /// Close the document: the pending line, the open fence, the last table.
    ///
    /// # Errors
    ///
    /// [`SemaRed`] naming what the finished document breaks.
    pub fn bitir(&self) -> Result<(), SemaRed> {
        if self.utf8.kalan > 0 {
            return Err(SemaRed::EksikUtf8 {
                konum: self.bayt,
                kalan: self.utf8.kalan as usize,
            });
        }
        let mut son = self.clone();
        // `str::lines` yields no trailing empty line for a text that ends with
        // a newline, so a pending line is only a line when it has bytes.
        if !son.satir.is_empty() {
            let ham = String::from_utf8(std::mem::take(&mut son.satir)).unwrap_or_default();
            let no = son.satir_no;
            son.satir_uygula(&ham, no)?;
        }
        if son.cit_acik {
            return Err(SemaRed::AcikCit {
                satir: son.cit_acilis_satiri,
            });
        }
        son.tabloyu_bosalt()?;
        if !son.bos_degil {
            return Err(SemaRed::Bos);
        }
        Ok(())
    }
}

/// Feed a whole buffer through the decoder and close it.
///
/// The convenience form used by the cross-check: it answers the same question
/// the validator answers, by the decoder's route.
///
/// # Errors
///
/// [`SemaRed`] naming the first rule the buffer breaks in stream order.
pub fn coz(baytlar: &[u8]) -> Result<(), SemaRed> {
    let mut d = SemaCozucu::yeni();
    for b in baytlar {
        d.yut(*b)?;
    }
    d.bitir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_read::output_schema::{validate_markdown_output, MAX_OUTPUT_BYTES};

    fn besle(s: &str) -> SemaCozucu {
        let mut d = SemaCozucu::yeni();
        for b in s.as_bytes() {
            assert!(d.yut(*b).is_ok(), "byte {b:#04x} refused in {s:?}");
        }
        d
    }

    #[test]
    fn tavan_dogrulayiciyla_ayni() {
        assert_eq!(MAKS_CIKTI_BAYT, MAX_OUTPUT_BYTES);
    }

    #[test]
    fn modul_parametre_tutmaz() {
        assert_eq!(SemaCozucu::parametre_sayisi(), 0);
    }

    #[test]
    fn duz_paragraf_gecer() {
        assert!(coz(b"just words\n\nmore words").is_ok());
    }

    #[test]
    fn bos_belge_reddedilir() {
        assert_eq!(coz(b""), Err(SemaRed::Bos));
        assert_eq!(coz(b"   \n\t\n"), Err(SemaRed::Bos));
    }

    #[test]
    fn inen_basliklar_gecer() {
        assert!(coz(b"# Title\n\n## Section\n\n### Sub\n\ntext\n").is_ok());
    }

    #[test]
    fn baslik_bosluk_baytinda_reddedilir() {
        // "# Title\n" then "###" is still fine - a hash run is not yet a
        // heading. The violation lands on the space, byte 11.
        let mut d = besle("# Title\n###");
        assert_eq!(d.bayt_sayisi(), 11);
        assert!(!d.izin_verilir(b' '));
        assert_eq!(
            d.yut(b' '),
            Err(SemaRed::BaslikAtlama {
                satir: 2,
                onceki: 1,
                gelen: 3
            })
        );
        // Refused, and the decoder did not move.
        assert_eq!(d.bayt_sayisi(), 11);
        // The legal descent is still open at the same position.
        assert!(d.izin_verilir(b'#'));
        assert!(besle("# Title\n## Sub").bitirebilir());
    }

    #[test]
    fn ilk_baslik_herhangi_bir_seviye_olabilir() {
        assert!(coz(b"#### Deep first\n").is_ok());
        assert!(besle("").izin_verilir(b'#'));
    }

    #[test]
    fn cit_kapanana_kadar_bitirilemez() {
        let acik = besle("```rust\nfn main() {}\n");
        assert!(!acik.bitirebilir());
        assert!(acik.cit_acik());
        assert_eq!(acik.bitir(), Err(SemaRed::AcikCit { satir: 1 }));

        let kapali = besle("```rust\nfn main() {}\n```\n");
        assert!(kapali.bitirebilir());
        assert!(!kapali.cit_acik());
    }

    #[test]
    fn cit_icinde_baslik_kurali_askida() {
        // Inside a fence a `### ` run is content, so the no-skip rule may not
        // fire there. This is the rule the early check has to respect.
        assert!(coz(b"# T\n```\n##### not a heading\n```\n").is_ok());
        let d = besle("# T\n```\n####");
        assert!(d.cit_acik());
        assert!(d.izin_verilir(b' '));
    }

    #[test]
    fn tablo_ayirici_satiri_ister() {
        assert!(coz(b"| a | b |\n|---|---|\n| 1 | 2 |\n").is_ok());
        assert_eq!(
            coz(b"| a | b |\n| 1 | 2 |\n"),
            Err(SemaRed::TabloUyusmazligi { satir: 1 })
        );
        assert_eq!(
            coz(b"| a | b |\n|---|---|\n| 1 |\n"),
            Err(SemaRed::TabloUyusmazligi { satir: 3 })
        );
    }

    #[test]
    fn tek_boru_satiri_tablo_degildir() {
        assert!(coz(b"|\n").is_ok());
        assert!(coz(b"| lone |\n").is_ok());
    }

    #[test]
    fn tablo_satiri_kapanista_denetlenir() {
        // The block closes on the newline of the offending row, so that is
        // where the refusal lands.
        let mut d = besle("| a | b |\n| 1 | 2 |");
        assert!(!d.izin_verilir(b'\n'));
        assert_eq!(d.yut(b'\n'), Err(SemaRed::TabloUyusmazligi { satir: 1 }));
    }

    #[test]
    fn gecersiz_utf8_onder_baytlari_hic_izinli_degil() {
        let d = SemaCozucu::yeni();
        for b in 0x80u8..=0xC1 {
            assert!(!d.izin_verilir(b), "{b:#04x} should never start a sequence");
        }
        for b in 0xF5u8..=0xFF {
            assert!(!d.izin_verilir(b), "{b:#04x} is above U+10FFFF");
        }
    }

    #[test]
    fn asiri_uzun_ve_vekil_diziler_reddedilir() {
        // C0 80 (overlong NUL) is refused at the lead byte.
        assert!(matches!(
            coz(&[0xC0, 0x80]),
            Err(SemaRed::GecersizUtf8 { .. })
        ));
        // E0 80 80 (overlong) is refused at the first continuation byte.
        let d = besle("");
        let mut e0 = d.clone();
        assert!(e0.yut(0xE0).is_ok());
        assert!(!e0.izin_verilir(0x80));
        assert!(e0.izin_verilir(0xA0));
        // ED A0 80 is a surrogate; the range stops at 0x9F.
        let mut ed = d;
        assert!(ed.yut(0xED).is_ok());
        assert!(!ed.izin_verilir(0xA0));
        assert!(ed.izin_verilir(0x9F));
    }

    #[test]
    fn yarim_dizi_ile_bitirilemez() {
        let mut d = SemaCozucu::yeni();
        assert!(d.yut(b'a').is_ok());
        assert!(d.yut(0xC3).is_ok());
        assert!(!d.bitirebilir());
        assert_eq!(d.bitir(), Err(SemaRed::EksikUtf8 { konum: 2, kalan: 1 }));
        assert!(d.yut(0xA7).is_ok());
        assert!(d.bitirebilir());
    }

    #[test]
    fn maske_izinli_logiti_bit_ozdes_birakir() {
        let d = besle("# T\nmetin");
        let taban: [f64; ALFABE] = std::array::from_fn(|i| (i as f64) * 0.5 - 13.25);
        let mut bir = taban;
        assert!(d.maskele(&mut bir).is_ok());
        let m = d.maske();
        for (i, ((yeni, eski), izin)) in bir.iter().zip(taban.iter()).zip(m.iter()).enumerate() {
            if *izin {
                assert_eq!(
                    yeni.to_bits(),
                    eski.to_bits(),
                    "allowed byte {i} was rewritten"
                );
            } else {
                assert!(yeni.is_infinite() && yeni.is_sign_negative());
            }
        }
        // Idempotent, bit for bit.
        let mut iki = bir;
        assert!(d.maskele(&mut iki).is_ok());
        for (a, b) in iki.iter().zip(bir.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn maske_hicbir_zaman_tum_alfabeyi_acmaz() {
        // The "a fresh mask is a no-op" shape the other candidates have does
        // not exist here and the reason is structural, not an omission: UTF-8
        // alone forbids 0x80..=0xC1 at every ground state, so the mask is
        // always a real restriction. Stating it as a test keeps a later reader
        // from "fixing" it.
        let d = SemaCozucu::yeni();
        let izinli = d.izinli_sayisi();
        assert!(izinli < ALFABE, "mask opened the whole alphabet");
        assert_eq!(izinli, 128 + (0xF4 - 0xC2 + 1));
    }

    #[test]
    fn cikmaz_yumusatilmaz() {
        // A table whose second row is not a separator: the block can never be
        // made valid by anything written after it, and because the row check
        // is eager the newline that would commit the bad row is the byte that
        // gets refused.
        let mut d = besle("| a | b |\n| 1 | 2 |");
        assert!(!d.izin_verilir(b'\n'));
        // The document also cannot end here.
        assert!(!d.bitirebilir());
        // A caller that insists gets a named refusal, not a byte.
        assert!(matches!(
            d.yut(b'\n'),
            Err(SemaRed::TabloUyusmazligi { .. })
        ));
    }

    #[test]
    fn tavan_geriye_dogru_bakmaz() {
        // The declared limitation, measured. A single byte below the ceiling
        // with a fence still open: the guard allows the byte, because the byte
        // itself breaks nothing, and the document only fails at `bitir`.
        let mut d = SemaCozucu::yeni();
        for b in b"```\n" {
            assert!(d.yut(*b).is_ok());
        }
        d.bayt = MAKS_CIKTI_BAYT - 1;
        assert!(d.izin_verilir(b'x'), "the local guard allows this byte");
        assert!(!d.bitirebilir(), "yet the document can never be finished");
        assert!(d.yut(b'x').is_ok());
        assert_eq!(
            d.yut(b'y'),
            Err(SemaRed::Tavan {
                bayt: MAKS_CIKTI_BAYT + 1
            })
        );
    }

    #[test]
    fn tavan_dolunca_maskele_reddeder() {
        let mut d = besle("```\n");
        d.bayt = MAKS_CIKTI_BAYT;
        assert_eq!(d.izinli_sayisi(), 0);
        let mut logitler = [0.0f64; ALFABE];
        assert_eq!(
            d.maskele(&mut logitler),
            Err(SemaRed::KabulEdilenBaytYok {
                konum: MAKS_CIKTI_BAYT
            })
        );
        // Nothing was softened: the logits were not touched on the refusal path.
        assert!(logitler.iter().all(|l| l.to_bits() == 0.0f64.to_bits()));
    }

    #[test]
    fn tavanda_bitirilebilen_belge_maskeyi_gecer() {
        // Same ceiling, but the document is closeable: no byte is allowed and
        // that is not an error, it is the end.
        let mut d = besle("metin\n");
        d.bayt = MAKS_CIKTI_BAYT;
        assert_eq!(d.izinli_sayisi(), 0);
        assert!(d.bitirebilir());
        let mut logitler = [1.0f64; ALFABE];
        assert!(d.maskele(&mut logitler).is_ok());
        assert!(logitler.iter().all(|l| l.is_infinite()));
    }

    /// A deterministic LCG - a fuzz that reruns identically on every machine.
    struct Rastgele(u64);

    impl Rastgele {
        fn sonraki(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }

        fn altinda(&mut self, n: usize) -> usize {
            (self.sonraki() % (n as u64)) as usize
        }
    }

    #[test]
    fn ayni_dili_kabul_eder() {
        // The alphabet is chosen to hit the schema, not to be uniform over
        // bytes: random bytes are almost all invalid UTF-8 and would measure
        // the UTF-8 decoder over and over while never building a table.
        const PARCALAR: [&[u8]; 16] = [
            b"#",
            b" ",
            b"|",
            b"-",
            b":",
            b"\n",
            b"`",
            b"a",
            b"\r",
            b"\t",
            b"|---|",
            b"| a |",
            b"## ",
            b"```",
            "ç".as_bytes(),
            &[0xC3],
        ];
        let mut r = Rastgele(0x5E3A_C02D_1F77_0001);
        let mut vaka = 0usize;
        let mut kabul = 0usize;
        for _ in 0..20_000 {
            let n = 1 + r.altinda(14);
            let mut s: Vec<u8> = Vec::new();
            for _ in 0..n {
                s.extend_from_slice(PARCALAR[r.altinda(PARCALAR.len())]);
            }
            let bizim = coz(&s).is_ok();
            let onun = validate_markdown_output(&s).is_ok();
            assert_eq!(
                bizim,
                onun,
                "disagreement on {:?}: decoder={bizim} validator={onun}",
                String::from_utf8_lossy(&s)
            );
            vaka += 1;
            if bizim {
                kabul += 1;
            }
        }
        assert_eq!(vaka, 20_000);
        // A fuzz that never accepts anything measures nothing.
        assert!(kabul > 1_000, "only {kabul} of {vaka} cases were accepted");
    }

    #[test]
    fn bayt_bayt_kabul_tam_kabulle_ayni() {
        // Feeding a valid document one byte at a time must never refuse a byte:
        // if the whole is acceptable, no prefix step may be a dead end.
        let belgeler = [
            "# T\n\n## S\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\ncode\n```\n",
            "düz metin, çok baytlı karakterlerle: ğüşiöç\n",
            "|\n\n| a |\n|---|\n| b |\n",
        ];
        for belge in belgeler {
            assert!(
                validate_markdown_output(belge.as_bytes()).is_ok(),
                "fixture is not valid: {belge:?}"
            );
            let mut d = SemaCozucu::yeni();
            for (i, b) in belge.as_bytes().iter().enumerate() {
                assert!(
                    d.izin_verilir(*b),
                    "byte {i} ({b:#04x}) refused in a valid document"
                );
                assert!(d.yut(*b).is_ok());
            }
            assert!(d.bitirebilir());
        }
    }
}
