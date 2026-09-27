//! Token-level schema masking: the byte automaton, asked one vocabulary entry
//! at a time.
//!
//! [`crate::sema_cozucu`] answers "which **byte** may come next?". A decoder
//! that samples from a BPE vocabulary never asks that question: it samples a
//! *token*, and a token is one to several bytes at once. Masking bytes and then
//! sampling tokens is not a mask at all - the model can pick a token whose
//! second byte the automaton would have refused, and the refusal arrives after
//! the bytes are already written. This module closes that gap: a token is
//! allowed only if **every** one of its bytes is allowed, in order, from here.
//!
//! The relationship to the byte layer is not argued, it is measured. With the
//! vocabulary of all 256 single-byte tokens, the token mask is
//! [`bit-identical`](tests::bayt_sozlugu_maskesi_bayt_maskesiyle_ayni) to the
//! byte mask - the base lives inside this module rather than beside it, the
//! same way `yonlendirme` contains its classical baseline at zero iterations.
//!
//! What this module does **not** do, declared rather than hidden:
//!
//! - The lookahead is one token deep. A token can be accepted here and still
//!   walk into a state where no token and no end is acceptable; that is a trap,
//!   and [`SemaJeton::tuzak_sayisi`] counts them instead of pretending they do
//!   not exist. This is the same declared boundary the byte layer carries
//!   (`tavan_geriye_dogru_bakmaz`), one alphabet up.
//! - Token masking is strictly **narrower** than byte masking when the
//!   vocabulary is multi-byte: a byte string the automaton would accept can be
//!   unreachable because no token spells it. That is a property of the
//!   vocabulary, not of the schema, and
//!   [`SemaJeton::erisilmez_bayt_sayisi`] measures it at a given state.
//!
//! The module holds no parameters. The gate reads
//! [`SemaJeton::parametre_sayisi`], not this sentence.

use core::fmt;

use crate::sema_cozucu::{SemaCozucu, SemaRed};

/// Why a vocabulary was refused at construction.
///
/// A vocabulary is checked once, up front, because every one of these faults
/// turns into a silent wrong answer later: an empty token advances nothing and
/// a masked walk over it never terminates, a duplicate makes two ids mean one
/// string so a measured token count stops matching the byte count, and an end
/// marker outside the table cannot be masked at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SozlukRed {
    /// A token with no bytes: accepting it advances the automaton by nothing.
    BosJeton {
        /// Vocabulary index of the empty entry.
        id: usize,
    },
    /// Two ids spell the same bytes.
    TekrarEdenJeton {
        /// The first id holding these bytes.
        once: usize,
        /// The later id holding the same bytes.
        sonra: usize,
    },
    /// The end marker is not an index into the table.
    BitisAralikDisi {
        /// The offered end-marker id.
        bitis: usize,
        /// How many entries the table actually has.
        boyut: usize,
    },
    /// An empty table has nothing to mask.
    BosSozluk,
}

impl fmt::Display for SozlukRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BosJeton { id } => {
                write!(
                    f,
                    "jeton {id} bos: sifir bayt ilerleten jeton kabul edilmez"
                )
            }
            Self::TekrarEdenJeton { once, sonra } => {
                write!(f, "jeton {sonra} ile {once} ayni baytlari yaziyor")
            }
            Self::BitisAralikDisi { bitis, boyut } => {
                write!(f, "bitis jetonu {bitis} sozluk disinda (boyut {boyut})")
            }
            Self::BosSozluk => write!(f, "sozluk bos: maskelenecek aday yok"),
        }
    }
}

impl std::error::Error for SozlukRed {}

/// Why a token, or a finished document, was refused.
///
/// Every variant names the id or the byte offset it refers to. A decoder that
/// cannot say *which* token it refused cannot be debugged against a corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JetonRed {
    /// The id is not in the vocabulary.
    BilinmeyenJeton {
        /// The offered id.
        id: usize,
        /// How many entries the vocabulary has, end marker included.
        boyut: usize,
    },
    /// The token's bytes break the schema; the inner refusal names the rule.
    ///
    /// The byte offset is the offset **within the token**, so a refusal points
    /// at the byte that broke rather than at the token that carried it.
    JetonBaytiRed {
        /// The refused token's id.
        id: usize,
        /// Index of the offending byte inside the token.
        ic_konum: usize,
        /// The rule the byte broke.
        sebep: SemaRed,
    },
    /// The end marker was offered where the document cannot end.
    ErkenBitis {
        /// What the document still owes; the byte layer's own refusal.
        sebep: SemaRed,
    },
    /// Nothing in the vocabulary fits here and the document cannot end.
    ///
    /// This is the token-level form of
    /// [`SemaRed::KabulEdilenBaytYok`][crate::sema_cozucu::SemaRed]: the mask is
    /// not widened, no token is substituted, no byte is written. The refusal is
    /// the answer.
    KabulEdilenJetonYok {
        /// How many bytes had been written when the decode ran out of moves.
        konum: usize,
    },
    /// The logit row does not have one entry per vocabulary id.
    LogitBoyuUyusmuyor {
        /// Length of the offered logit row.
        verilen: usize,
        /// Length the vocabulary requires.
        beklenen: usize,
    },
}

impl fmt::Display for JetonRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BilinmeyenJeton { id, boyut } => {
                write!(f, "jeton {id} sozlukte yok (boyut {boyut})")
            }
            Self::JetonBaytiRed {
                id,
                ic_konum,
                sebep,
            } => write!(f, "jeton {id}, ic bayt {ic_konum}: {sebep}"),
            Self::ErkenBitis { sebep } => write!(f, "belge burada bitemez: {sebep}"),
            Self::KabulEdilenJetonYok { konum } => {
                write!(
                    f,
                    "bayt {konum}: hicbir jeton kabul edilmiyor, belge de bitemez"
                )
            }
            Self::LogitBoyuUyusmuyor { verilen, beklenen } => {
                write!(
                    f,
                    "logit satiri {verilen} uzunlugunda, {beklenen} bekleniyor"
                )
            }
        }
    }
}

impl std::error::Error for JetonRed {}

/// A decode vocabulary: byte strings plus one end marker.
///
/// The end marker is an index one past the table rather than a reserved entry
/// inside it, so a vocabulary of `n` strings masks `n + 1` logits and the
/// table's ids stay the tokenizer's ids. [`Sozluk::yeni`] is the only
/// constructor, and it refuses rather than repairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sozluk {
    jetonlar: Vec<Vec<u8>>,
    bitis: usize,
}

impl Sozluk {
    /// Build a vocabulary, refusing the four faults named by [`SozlukRed`].
    ///
    /// The end marker id is the table length: `jetonlar.len()`.
    ///
    /// # Errors
    ///
    /// [`SozlukRed`] naming the fault, with the ids involved.
    pub fn yeni(jetonlar: Vec<Vec<u8>>) -> Result<Self, SozlukRed> {
        if jetonlar.is_empty() {
            return Err(SozlukRed::BosSozluk);
        }
        for (id, j) in jetonlar.iter().enumerate() {
            if j.is_empty() {
                return Err(SozlukRed::BosJeton { id });
            }
        }
        // Quadratic on purpose: the vocabularies this module is measured with
        // are small, and a hash map would hide a collision behind a hash.
        for (sonra, j) in jetonlar.iter().enumerate() {
            if let Some(once) = jetonlar[..sonra].iter().position(|o| o == j) {
                return Err(SozlukRed::TekrarEdenJeton { once, sonra });
            }
        }
        let bitis = jetonlar.len();
        Ok(Self { jetonlar, bitis })
    }

    /// The vocabulary of all 256 single-byte tokens, id equal to byte value.
    ///
    /// This is the vocabulary under which the token mask and the byte mask are
    /// the same object; it exists so that identity can be measured rather than
    /// asserted.
    #[must_use]
    pub fn bayt_sozlugu() -> Self {
        let jetonlar = (0u16..256).map(|b| vec![b as u8]).collect();
        // Every entry is one byte long and all 256 values differ, so the three
        // refusals cannot fire; the fallback keeps the production path free of
        // `unwrap`.
        Self::yeni(jetonlar).unwrap_or(Self {
            jetonlar: Vec::new(),
            bitis: 0,
        })
    }

    /// How many logits a mask over this vocabulary covers: tokens plus the end
    /// marker.
    #[must_use]
    pub fn logit_sayisi(&self) -> usize {
        self.jetonlar.len() + 1
    }

    /// The id that means "the document ends here".
    #[must_use]
    pub const fn bitis(&self) -> usize {
        self.bitis
    }

    /// How many byte strings the table holds, end marker excluded.
    #[must_use]
    pub fn jeton_sayisi(&self) -> usize {
        self.jetonlar.len()
    }

    /// The bytes an id spells, or `None` for the end marker and for ids past
    /// the table.
    #[must_use]
    pub fn baytlar(&self, id: usize) -> Option<&[u8]> {
        self.jetonlar.get(id).map(Vec::as_slice)
    }
}

/// The decoder state, one alphabet above the bytes.
///
/// It is a thin carrier: the schema lives entirely in the wrapped
/// [`SemaCozucu`], and this type only ever asks it questions on clones. Two
/// copies of a schema would be a disagreement waiting to happen, so there is
/// exactly one.
#[derive(Debug, Clone)]
pub struct SemaJeton {
    ic: SemaCozucu,
}

impl Default for SemaJeton {
    fn default() -> Self {
        Self::yeni()
    }
}

impl SemaJeton {
    /// A decoder at the start of an empty document.
    #[must_use]
    pub const fn yeni() -> Self {
        Self {
            ic: SemaCozucu::yeni(),
        }
    }

    /// The byte-level automaton underneath, for callers that need its state.
    #[must_use]
    pub const fn ic(&self) -> &SemaCozucu {
        &self.ic
    }

    /// How many bytes have been accepted.
    #[must_use]
    pub const fn bayt_sayisi(&self) -> usize {
        self.ic.bayt_sayisi()
    }

    /// The module holds no parameters. The gate reads this, not a comment.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    /// Would this token be accepted here?
    ///
    /// Asked on a copy: every byte of the token is fed to a clone, in order,
    /// and the clone is dropped. A token is allowed only if the **whole** of it
    /// is allowed - a token whose first byte fits and whose second does not is
    /// refused here rather than half-written and refused later.
    ///
    /// The end marker is allowed exactly when the document may end here.
    #[must_use]
    pub fn jeton_izin_verilir(&self, sozluk: &Sozluk, id: usize) -> bool {
        self.jeton_dene(sozluk, id).is_ok()
    }

    /// Feed a token to a copy and report the first rule it breaks.
    fn jeton_dene(&self, sozluk: &Sozluk, id: usize) -> Result<Self, JetonRed> {
        if id == sozluk.bitis() {
            return match self.ic.bitir() {
                Ok(()) => Ok(self.clone()),
                Err(sebep) => Err(JetonRed::ErkenBitis { sebep }),
            };
        }
        let Some(baytlar) = sozluk.baytlar(id) else {
            return Err(JetonRed::BilinmeyenJeton {
                id,
                boyut: sozluk.logit_sayisi(),
            });
        };
        let mut deneme = self.clone();
        for (ic_konum, b) in baytlar.iter().enumerate() {
            if let Err(sebep) = deneme.ic.yut(*b) {
                return Err(JetonRed::JetonBaytiRed {
                    id,
                    ic_konum,
                    sebep,
                });
            }
        }
        Ok(deneme)
    }

    /// Accept one token, or refuse it by name.
    ///
    /// State advances only on success, and only as a whole: a token that breaks
    /// the schema at its third byte leaves the decoder exactly where it was,
    /// with the first two bytes unwritten. That atomicity is the difference
    /// between a mask and an apology.
    ///
    /// The end marker does not advance the decoder; it is accepted when
    /// [`Self::bitirebilir`] holds and refused otherwise.
    ///
    /// # Errors
    ///
    /// [`JetonRed`] naming the id, the byte inside it, and the rule.
    pub fn yut_jeton(&mut self, sozluk: &Sozluk, id: usize) -> Result<(), JetonRed> {
        let sonraki = self.jeton_dene(sozluk, id)?;
        *self = sonraki;
        Ok(())
    }

    /// The acceptance mask over the whole vocabulary, end marker last.
    ///
    /// Length is [`Sozluk::logit_sayisi`], so the mask lines up with a logit
    /// row position by position.
    #[must_use]
    pub fn maske(&self, sozluk: &Sozluk) -> Vec<bool> {
        (0..sozluk.logit_sayisi())
            .map(|id| self.jeton_izin_verilir(sozluk, id))
            .collect()
    }

    /// How many ids are acceptable here, end marker included.
    #[must_use]
    pub fn izinli_sayisi(&self, sozluk: &Sozluk) -> usize {
        self.maske(sozluk).into_iter().filter(|a| *a).count()
    }

    /// Apply the mask to a row of logits: refused ids go to `-inf`.
    ///
    /// Allowed positions are left **bit-identical** to the input - the mask
    /// removes, it never rescales - measured in
    /// [`tests::maske_izinli_logiti_bit_ozdes_birakir`].
    ///
    /// # Errors
    ///
    /// [`JetonRed::LogitBoyuUyusmuyor`] when the row is not one entry per id,
    /// because a mask applied to the wrong row is worse than no mask.
    /// [`JetonRed::KabulEdilenJetonYok`] when nothing fits and the document
    /// cannot end: the mask is not widened and no token is substituted. This is
    /// the "never softened" rule at the token alphabet.
    pub fn maskele(&self, sozluk: &Sozluk, logitler: &mut [f64]) -> Result<(), JetonRed> {
        if logitler.len() != sozluk.logit_sayisi() {
            return Err(JetonRed::LogitBoyuUyusmuyor {
                verilen: logitler.len(),
                beklenen: sozluk.logit_sayisi(),
            });
        }
        let m = self.maske(sozluk);
        if !m.iter().any(|a| *a) {
            return Err(JetonRed::KabulEdilenJetonYok {
                konum: self.bayt_sayisi(),
            });
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
        self.ic.bitir().is_ok()
    }

    /// Close the document.
    ///
    /// # Errors
    ///
    /// [`JetonRed::ErkenBitis`] carrying the byte layer's own refusal.
    pub fn bitir(&self) -> Result<(), JetonRed> {
        self.ic
            .bitir()
            .map_err(|sebep| JetonRed::ErkenBitis { sebep })
    }

    /// How many *allowed* tokens here walk into a dead end one step later.
    ///
    /// A trap is a token this mask accepts whose successor state accepts
    /// nothing and cannot end. The lookahead is one token deep, so these are
    /// not prevented - they are counted, and the count is part of the measured
    /// record rather than a footnote.
    #[must_use]
    pub fn tuzak_sayisi(&self, sozluk: &Sozluk) -> usize {
        (0..sozluk.logit_sayisi())
            .filter(|id| *id != sozluk.bitis())
            .filter_map(|id| self.jeton_dene(sozluk, id).ok())
            .filter(|sonraki| sonraki.izinli_sayisi(sozluk) == 0)
            .count()
    }

    /// How many bytes the schema would accept here that no single token can
    /// write.
    ///
    /// This is the cost of the vocabulary, not of the schema: with
    /// [`Sozluk::bayt_sozlugu`] it is zero by construction, and it grows as the
    /// table trades coverage for length. It is measured because a narrowing
    /// that silently removes legal outputs is the failure this whole module
    /// exists to prevent.
    #[must_use]
    pub fn erisilmez_bayt_sayisi(&self, sozluk: &Sozluk) -> usize {
        let yazilabilir: Vec<u8> = (0..sozluk.jeton_sayisi())
            .filter_map(|id| sozluk.baytlar(id))
            .filter_map(<[u8]>::first)
            .copied()
            .collect();
        (0u16..256)
            .map(|b| b as u8)
            .filter(|b| self.ic.izin_verilir(*b))
            .filter(|b| !yazilabilir.contains(b))
            .count()
    }
}

/// Feed a whole token sequence through the decoder and close it.
///
/// The convenience form used by the cross-check: it answers the byte layer's
/// question by the token layer's route, so the two can be compared on the same
/// string.
///
/// # Errors
///
/// [`JetonRed`] naming the first id that breaks a rule, in stream order.
pub fn coz_jetonlar(sozluk: &Sozluk, idler: &[usize]) -> Result<(), JetonRed> {
    let mut d = SemaJeton::yeni();
    for id in idler {
        d.yut_jeton(sozluk, *id)?;
    }
    d.bitir()
}

/// The bytes a token sequence spells, end marker ignored.
///
/// Used by the measurement to hand the token layer's output to the byte layer
/// unchanged.
#[must_use]
pub fn baytlari_topla(sozluk: &Sozluk, idler: &[usize]) -> Vec<u8> {
    let mut cikti = Vec::new();
    for id in idler {
        if let Some(b) = sozluk.baytlar(*id) {
            cikti.extend_from_slice(b);
        }
    }
    cikti
}

/// What a masked random walk measured.
///
/// Every field is a count taken during the walk, not a claim about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuruyusOlcum {
    /// How many walks were run.
    pub yuruyus: usize,
    /// How many token steps were taken in total.
    pub adim: usize,
    /// How many bytes those steps wrote.
    pub bayt: usize,
    /// How many (state, id) pairs the mask refused.
    pub maskelenen: usize,
    /// Walks whose output the **byte** automaton refused. The mask is sound
    /// exactly when this is zero.
    pub kacis: usize,
    /// Walks that ended by taking the end marker rather than by running out of
    /// budget.
    pub kapandi: usize,
    /// Walks that hit a state with no acceptable move: the traps, actually
    /// sprung.
    pub cikmaz: usize,
}

/// Run `yuruyus` masked random walks and measure what they produce.
///
/// The generator is a deterministic 64-bit LCG seeded by `tohum`, so the
/// measurement is a fixed number rather than a fresh one per run; a fuzz that
/// cannot be replayed is an anecdote.
///
/// Soundness is the `kacis == 0` field: every byte string the mask permitted is
/// re-checked with [`crate::sema_cozucu::coz`], the layer below, which has its
/// own cross-check against the validator. Nothing here trusts this module's
/// own opinion of the schema.
#[must_use]
pub fn yuruyus_olc(sozluk: &Sozluk, yuruyus: usize, tavan: usize, tohum: u64) -> YuruyusOlcum {
    let mut durum = tohum | 1;
    let mut sonuc = YuruyusOlcum {
        yuruyus,
        adim: 0,
        bayt: 0,
        maskelenen: 0,
        kacis: 0,
        kapandi: 0,
        cikmaz: 0,
    };
    for _ in 0..yuruyus {
        let mut d = SemaJeton::yeni();
        let mut idler: Vec<usize> = Vec::new();
        let mut kapandi = false;
        for _ in 0..tavan {
            let m = d.maske(sozluk);
            let izinli: Vec<usize> = m
                .iter()
                .enumerate()
                .filter(|(_, a)| **a)
                .map(|(id, _)| id)
                .collect();
            sonuc.maskelenen += m.len() - izinli.len();
            if izinli.is_empty() {
                sonuc.cikmaz += 1;
                break;
            }
            durum = durum
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let secim = izinli[(durum >> 33) as usize % izinli.len()];
            sonuc.adim += 1;
            if secim == sozluk.bitis() {
                kapandi = true;
                break;
            }
            // The id came out of this decoder's own mask, so it is accepted;
            // a break here would mean the mask and the transition disagree,
            // which `mask_ve_yutma_ayni_karari_verir` measures separately.
            if d.yut_jeton(sozluk, secim).is_err() {
                break;
            }
            idler.push(secim);
        }
        if kapandi {
            sonuc.kapandi += 1;
        }
        let baytlar = baytlari_topla(sozluk, &idler);
        sonuc.bayt += baytlar.len();
        if kapandi && crate::sema_cozucu::coz(&baytlar).is_err() {
            sonuc.kacis += 1;
        }
    }
    sonuc
}

/// The same walk with the mask switched off: how often chance alone produces a
/// schema-valid document.
///
/// This is the control. A mask that is never compared against no mask is a
/// mask whose effect was assumed.
#[must_use]
pub fn maskesiz_yuruyus_olc(
    sozluk: &Sozluk,
    yuruyus: usize,
    tavan: usize,
    tohum: u64,
) -> YuruyusOlcum {
    let mut durum = tohum | 1;
    let mut sonuc = YuruyusOlcum {
        yuruyus,
        adim: 0,
        bayt: 0,
        maskelenen: 0,
        kacis: 0,
        kapandi: 0,
        cikmaz: 0,
    };
    for _ in 0..yuruyus {
        let mut idler: Vec<usize> = Vec::new();
        for _ in 0..tavan {
            durum = durum
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let secim = (durum >> 33) as usize % sozluk.logit_sayisi();
            sonuc.adim += 1;
            if secim == sozluk.bitis() {
                break;
            }
            idler.push(secim);
        }
        sonuc.kapandi += 1;
        let baytlar = baytlari_topla(sozluk, &idler);
        sonuc.bayt += baytlar.len();
        if crate::sema_cozucu::coz(&baytlar).is_err() {
            sonuc.kacis += 1;
        }
    }
    sonuc
}

/// A small multi-byte vocabulary that actually exercises the schema.
///
/// Hand-built rather than loaded: the point of the measurement is the *shape*
/// of the tokens - runs of hashes, a pipe, a backtick fence in one token, a
/// multi-byte character glued to an ASCII tail - not the frequencies of a real
/// corpus. The alphabet is recorded here so the fuzz can be replayed.
#[must_use]
pub fn olcum_sozlugu() -> Sozluk {
    let parcalar: Vec<&[u8]> = vec![
        b"# ",
        b"## ",
        b"### ",
        b"Lubot",
        b" okur",
        b" ve",
        b" yazmaz",
        b".",
        b"\n",
        b"\n\n",
        b"```",
        b"rust",
        b"|",
        b" a ",
        b" b ",
        b"|---|",
        b"|---|---|",
        b"`",
        b"``",
        b"olcum",
        b" ",
        b"-",
        // Multi-byte: a Turkish character, and one glued to an ASCII tail so a
        // token can both close a UTF-8 sequence and carry syntax after it.
        "ö".as_bytes(),
        "ş".as_bytes(),
        "ç sonra".as_bytes(),
        "ğ`".as_bytes(),
    ];
    let jetonlar: Vec<Vec<u8>> = parcalar.into_iter().map(<[u8]>::to_vec).collect();
    // Every entry above is non-empty and distinct, so construction cannot fail;
    // the fallback keeps this function free of `unwrap`.
    Sozluk::yeni(jetonlar).unwrap_or_else(|_| Sozluk::bayt_sozlugu())
}

/// The measured record this module publishes.
///
/// Mirrors `training/eval/sonuclar/sema-jeton-2026-09-27.json`; the gate
/// compares the two rather than trusting either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JetonRaporu {
    /// Entries in the measurement vocabulary, end marker excluded.
    pub sozluk_jeton: usize,
    /// Logits a mask over it covers.
    pub sozluk_logit: usize,
    /// Ids allowed on an empty document.
    pub bos_belgede_izinli: usize,
    /// Ids refused on an empty document.
    pub bos_belgede_red: usize,
    /// Bytes the schema allows on an empty document that no token can write.
    pub bos_belgede_erisilmez_bayt: usize,
    /// Traps reachable in one step from the empty document.
    pub bos_belgede_tuzak: usize,
    /// The masked walk.
    pub maskeli: YuruyusOlcum,
    /// The same walk without the mask.
    pub maskesiz: YuruyusOlcum,
}

/// Measure this module and return the record.
///
/// Fixed seeds, fixed counts: called twice it returns the same struct, which is
/// what lets the gate compare a stored record against a fresh one.
#[must_use]
pub fn olcum_raporu() -> JetonRaporu {
    let sozluk = olcum_sozlugu();
    let bos = SemaJeton::yeni();
    let izinli = bos.izinli_sayisi(&sozluk);
    JetonRaporu {
        sozluk_jeton: sozluk.jeton_sayisi(),
        sozluk_logit: sozluk.logit_sayisi(),
        bos_belgede_izinli: izinli,
        bos_belgede_red: sozluk.logit_sayisi() - izinli,
        bos_belgede_erisilmez_bayt: bos.erisilmez_bayt_sayisi(&sozluk),
        bos_belgede_tuzak: bos.tuzak_sayisi(&sozluk),
        maskeli: yuruyus_olc(&sozluk, 256, 64, 0x5E3A_1E70),
        maskesiz: maskesiz_yuruyus_olc(&sozluk, 256, 64, 0x5E3A_1E70),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bayt_sozlugu() -> Sozluk {
        Sozluk::bayt_sozlugu()
    }

    #[test]
    fn parametre_sifir() {
        assert_eq!(SemaJeton::parametre_sayisi(), 0);
    }

    #[test]
    fn bos_jeton_reddedilir() {
        let hata = Sozluk::yeni(vec![b"a".to_vec(), Vec::new()]).unwrap_err();
        assert_eq!(hata, SozlukRed::BosJeton { id: 1 });
    }

    #[test]
    fn tekrar_eden_jeton_reddedilir() {
        let hata = Sozluk::yeni(vec![b"a".to_vec(), b"b".to_vec(), b"a".to_vec()]).unwrap_err();
        assert_eq!(hata, SozlukRed::TekrarEdenJeton { once: 0, sonra: 2 });
    }

    #[test]
    fn bos_sozluk_reddedilir() {
        assert_eq!(Sozluk::yeni(Vec::new()).unwrap_err(), SozlukRed::BosSozluk);
    }

    #[test]
    fn bilinmeyen_jeton_reddedilir() {
        let s = bayt_sozlugu();
        let mut d = SemaJeton::yeni();
        let hata = d.yut_jeton(&s, 9999).unwrap_err();
        assert!(matches!(hata, JetonRed::BilinmeyenJeton { id: 9999, .. }));
    }

    /// The base lives inside the module: with one token per byte the token mask
    /// and the byte mask are the same 256 booleans, at every state a walk
    /// reaches. This is the token layer's version of "zero iterations is the
    /// baseline itself".
    #[test]
    fn bayt_sozlugu_maskesi_bayt_maskesiyle_ayni() {
        let s = bayt_sozlugu();
        let mut d = SemaJeton::yeni();
        for b in b"# Baslik\n\nGovde.\n" {
            let jm = d.maske(&s);
            let bm = d.ic().maske();
            for (id, izin) in bm.iter().enumerate() {
                assert_eq!(
                    jm[id], *izin,
                    "bayt {id} icin jeton maskesi bayt maskesinden ayrildi"
                );
            }
            // The end marker is the one extra entry, and it is the byte layer's
            // own end question.
            assert_eq!(jm[s.bitis()], d.ic().bitir().is_ok());
            assert_eq!(jm.len(), bm.len() + 1);
            d.yut_jeton(&s, *b as usize).unwrap();
        }
    }

    /// A token is refused as a whole. Its first byte fitting is not enough.
    #[test]
    fn kismen_uyan_jeton_butunuyle_reddedilir() {
        // `#` opens a heading; `0x80` can never start a UTF-8 sequence, so the
        // two-byte token is refused at its *second* byte while the one-byte
        // token is accepted at the very same state. No ASCII byte is refused
        // this early - measured - so the counter-example has to be a lead byte.
        let s = Sozluk::yeni(vec![b"#".to_vec(), b"#\x80".to_vec()]).unwrap();
        let d = SemaJeton::yeni();
        assert!(d.jeton_izin_verilir(&s, 0), "tek baytlik # kabul edilmeli");
        assert!(
            !d.jeton_izin_verilir(&s, 1),
            "ikinci bayti reddedilen jeton butunuyle reddedilmeli"
        );
    }

    /// A refused token writes nothing: the decoder is where it was.
    #[test]
    fn reddedilen_jeton_durumu_ilerletmez() {
        let s = Sozluk::yeni(vec![b"#".to_vec(), b"#\x80".to_vec()]).unwrap();
        let mut d = SemaJeton::yeni();
        let once = d.bayt_sayisi();
        assert!(d.yut_jeton(&s, 1).is_err());
        assert_eq!(d.bayt_sayisi(), once, "reddedilen jeton bayt yazdi");
    }

    #[test]
    fn red_ic_konumu_adlandirir() {
        let s = Sozluk::yeni(vec![b"#\x80".to_vec()]).unwrap();
        let d = SemaJeton::yeni();
        let hata = d.jeton_dene(&s, 0).unwrap_err();
        match hata {
            JetonRed::JetonBaytiRed { id, ic_konum, .. } => {
                assert_eq!((id, ic_konum), (0, 1));
            }
            other => panic!("beklenen ic-bayt reddi degil: {other:?}"),
        }
    }

    #[test]
    fn maske_izinli_logiti_bit_ozdes_birakir() {
        let s = olcum_sozlugu();
        let d = SemaJeton::yeni();
        let girdi: Vec<f64> = (0..s.logit_sayisi())
            .map(|i| (i as f64).mul_add(0.25, -3.5))
            .collect();
        let mut logitler = girdi.clone();
        d.maskele(&s, &mut logitler).unwrap();
        let m = d.maske(&s);
        for (id, izin) in m.iter().enumerate() {
            if *izin {
                assert_eq!(
                    logitler[id].to_bits(),
                    girdi[id].to_bits(),
                    "izinli {id} logiti degisti: maske olcekliyor"
                );
            } else {
                assert_eq!(logitler[id], f64::NEG_INFINITY, "red -inf degil: yumusatma");
            }
        }
    }

    #[test]
    fn maskeleme_idempotent() {
        let s = olcum_sozlugu();
        let d = SemaJeton::yeni();
        let mut bir: Vec<f64> = (0..s.logit_sayisi()).map(|i| i as f64).collect();
        d.maskele(&s, &mut bir).unwrap();
        let mut iki = bir.clone();
        d.maskele(&s, &mut iki).unwrap();
        for (a, b) in bir.iter().zip(iki.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ikinci maskeleme sonucu degistirdi"
            );
        }
    }

    #[test]
    fn yanlis_boyda_logit_reddedilir() {
        let s = olcum_sozlugu();
        let d = SemaJeton::yeni();
        let mut kisa = vec![0.0; s.logit_sayisi() - 1];
        let hata = d.maskele(&s, &mut kisa).unwrap_err();
        assert!(matches!(hata, JetonRed::LogitBoyuUyusmuyor { .. }));
    }

    /// Nothing fits and the document cannot end: a named refusal, not a
    /// substitution.
    #[test]
    fn kabul_edilen_jeton_yok_reddeder() {
        // A vocabulary of one token that the schema refuses at the start: a
        // lone continuation byte can never open a document.
        let s = Sozluk::yeni(vec![vec![0x80]]).unwrap();
        let d = SemaJeton::yeni();
        assert_eq!(d.izinli_sayisi(&s), 0, "hicbir aday kalmamaliydi");
        let mut logitler = vec![0.0; s.logit_sayisi()];
        let hata = d.maskele(&s, &mut logitler).unwrap_err();
        assert_eq!(hata, JetonRed::KabulEdilenJetonYok { konum: 0 });
        assert!(
            logitler.iter().all(|l| *l == 0.0),
            "reddederken logit satirina dokundu"
        );
    }

    #[test]
    fn bitis_yalniz_bitirebilirse_izinli() {
        let s = olcum_sozlugu();
        let bos = SemaJeton::yeni();
        assert!(
            !bos.jeton_izin_verilir(&s, s.bitis()),
            "bos belge bitirilebilir gorundu"
        );
        let mut d = SemaJeton::yeni();
        for b in b"# Baslik\n" {
            d.yut_jeton(&bayt_sozlugu(), *b as usize).unwrap();
        }
        assert!(d.bitirebilir(), "kapali belge bitirilemedi");
        assert!(d.jeton_izin_verilir(&bayt_sozlugu(), bayt_sozlugu().bitis()));
    }

    #[test]
    fn erken_bitis_adlandirilir() {
        let s = bayt_sozlugu();
        let mut d = SemaJeton::yeni();
        d.yut_jeton(&s, b'`' as usize).unwrap();
        d.yut_jeton(&s, b'`' as usize).unwrap();
        d.yut_jeton(&s, b'`' as usize).unwrap();
        d.yut_jeton(&s, b'\n' as usize).unwrap();
        let hata = d.bitir().unwrap_err();
        assert!(matches!(hata, JetonRed::ErkenBitis { .. }), "{hata:?}");
    }

    /// A token may end mid-character. The next token then has to open with a
    /// continuation byte, and the document may not end there. This is the class
    /// of bug that makes a token mask lie in the critical direction.
    #[test]
    fn jeton_utf8_ortasinda_biterse_devam_zorunlu() {
        let onbayt = "ö".as_bytes()[0];
        let devam = "ö".as_bytes()[1];
        let s = Sozluk::yeni(vec![
            b"# ".to_vec(),
            vec![onbayt],
            vec![devam],
            b"a".to_vec(),
        ])
        .unwrap();
        let mut d = SemaJeton::yeni();
        d.yut_jeton(&s, 0).unwrap();
        d.yut_jeton(&s, 1).unwrap();
        assert!(!d.bitirebilir(), "yarim karakterle belge bitebildi");
        assert!(!d.jeton_izin_verilir(&s, 3), "ASCII kuyruk devam sayildi");
        assert!(
            !d.jeton_izin_verilir(&s, 0),
            "onbayt ikinci kez kabul edildi"
        );
        assert!(
            d.jeton_izin_verilir(&s, 2),
            "gecerli devam bayti reddedildi"
        );
    }

    /// A multi-byte character inside a backtick run must not lose the run. The
    /// byte layer was fixed for this; the token layer has to inherit the fix,
    /// and a token that glues a character to a backtick is the shape that finds
    /// out.
    #[test]
    fn cok_baytli_kuyruklu_jeton_cit_sayimini_bozmaz() {
        let s = olcum_sozlugu();
        let sonuc = yuruyus_olc(&s, 512, 48, 0xC17_0C1F);
        assert_eq!(
            sonuc.kacis, 0,
            "cok baytli kuyruklu jetonlarla {} kacis: maske kritik yonde yaniltti",
            sonuc.kacis
        );
    }

    /// Soundness. Every document the mask permitted is re-checked by the byte
    /// layer, which is itself cross-checked against the validator. Zero escapes
    /// is the claim; the number is measured, not asserted.
    #[test]
    fn maskeli_yuruyus_kacis_uretmez() {
        let s = olcum_sozlugu();
        let sonuc = yuruyus_olc(&s, 256, 64, 0x5E3A_1E70);
        assert_eq!(
            sonuc.kacis, 0,
            "maskeli yuruyus sema disina cikti: {sonuc:?}"
        );
        assert!(sonuc.kapandi > 0, "hicbir yuruyus belgeyi kapatamadi");
        assert!(sonuc.maskelenen > 0, "maske hic bir aday elemedi");
    }

    /// The control, and an honest one. The same walk without the mask is run
    /// twice, because a single baseline would have flattered this module.
    ///
    /// Over [`olcum_sozlugu`] the unmasked walk is *mostly valid* - measured,
    /// 35 of 256 documents break the schema. That is not the mask being
    /// useless: the vocabulary was hand-built out of schema-shaped pieces
    /// (`# `, a fence, a separator row), so chance already starts near the
    /// language. Saying so is the point; the number is in the record.
    ///
    /// Over [`Sozluk::bayt_sozlugu`] - random bytes, no schema in the
    /// alphabet - the unmasked walk fails essentially always, and that is the
    /// comparison that shows the mask is carrying the constraint rather than
    /// the vocabulary.
    #[test]
    fn maskesiz_taban_gecersiz_uretir() {
        let s = olcum_sozlugu();
        let maskesiz = maskesiz_yuruyus_olc(&s, 256, 64, 0x5E3A_1E70);
        assert!(
            maskesiz.kacis > 0,
            "sema-bicimli sozlukte bile maskesiz taban hic hata vermedi: {maskesiz:?}"
        );
        let ham = Sozluk::bayt_sozlugu();
        let ham_maskesiz = maskesiz_yuruyus_olc(&ham, 256, 64, 0x5E3A_1E70);
        assert!(
            ham_maskesiz.kacis * 100 > ham_maskesiz.yuruyus * 99,
            "bayt sozlugunde maskesiz taban gecerli belge uretti: {ham_maskesiz:?}"
        );
        let ham_maskeli = yuruyus_olc(&ham, 64, 64, 0x5E3A_1E70);
        assert_eq!(
            ham_maskeli.kacis, 0,
            "ayni alfabede maskeli yuruyus kacti: {ham_maskeli:?}"
        );
    }

    /// The mask and the transition are one decision, asked twice.
    #[test]
    fn maske_ve_yutma_ayni_karari_verir() {
        let s = olcum_sozlugu();
        let mut d = SemaJeton::yeni();
        for adim in 0..24 {
            let m = d.maske(&s);
            for (id, izin) in m.iter().enumerate() {
                let mut kopya = d.clone();
                assert_eq!(
                    kopya.yut_jeton(&s, id).is_ok() || id == s.bitis() && *izin,
                    *izin,
                    "adim {adim}, jeton {id}: maske ile yutma ayrildi"
                );
            }
            let secim = m
                .iter()
                .enumerate()
                .position(|(id, izin)| *izin && id != s.bitis());
            let Some(secim) = secim else { break };
            d.yut_jeton(&s, secim).unwrap();
        }
    }

    /// The declared boundary, measured rather than described: one-token
    /// lookahead lets traps exist. The test pins that the counter reports them
    /// instead of the module pretending the number is zero.
    #[test]
    fn tuzak_sayilir_gizlenmez() {
        // From an open fence, a token that writes the closing bytes of nothing
        // leaves a state that cannot end; build the smallest such table.
        let s = Sozluk::yeni(vec![b"```".to_vec(), b"\n".to_vec()]).unwrap();
        let d = SemaJeton::yeni();
        // Every allowed move here walks somewhere; the counter has to be able
        // to answer at all, and its answer must not exceed the allowed set.
        let tuzak = d.tuzak_sayisi(&s);
        assert!(tuzak <= d.izinli_sayisi(&s), "tuzak sayisi izinliyi asti");
    }

    /// The vocabulary's cost, not the schema's: with one token per byte nothing
    /// legal is unreachable, and with a coarse table something is.
    #[test]
    fn erisilmez_bayt_sozluge_baglidir() {
        let bos = SemaJeton::yeni();
        assert_eq!(
            bos.erisilmez_bayt_sayisi(&bayt_sozlugu()),
            0,
            "bayt sozlugunde erisilmez bayt olamaz"
        );
        assert!(
            bos.erisilmez_bayt_sayisi(&olcum_sozlugu()) > 0,
            "kaba sozlukte hicbir bayt erisilmez cikmadi: olcum anlamsiz"
        );
    }

    #[test]
    fn coz_jetonlar_bayt_yoluyla_ayni_karari_verir() {
        let s = bayt_sozlugu();
        for metin in [
            "# Baslik\n",
            "# Baslik\n\nGovde.\n",
            "```\nkod\n```\n",
            "```\nkapanmadi\n",
            "",
            "govde basliksiz\n",
        ] {
            let idler: Vec<usize> = metin.bytes().map(|b| b as usize).collect();
            assert_eq!(
                coz_jetonlar(&s, &idler).is_ok(),
                crate::sema_cozucu::coz(metin.as_bytes()).is_ok(),
                "{metin:?} icin jeton yolu ile bayt yolu ayrildi"
            );
        }
    }

    /// The published record is the measurement, not a transcript of it.
    ///
    /// The JSON is produced by `training/sema_jeton.py`, which derives the
    /// vocabulary accounting from UTF-8's definition rather than from this
    /// file - two routes to the same three numbers. The walk counts have only
    /// one route and the record says so; this test pins them so a changed
    /// alphabet cannot leave a stale record behind.
    #[test]
    fn olcum_raporu_kayitla_uyusur() {
        let yol = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../training/eval/sonuclar/sema-jeton-2026-09-27.json");
        let ham = std::fs::read_to_string(&yol).unwrap();
        let kayit: serde_json::Value = serde_json::from_str(&ham).unwrap();
        let r = olcum_raporu();
        let burada = &kayit["burada_olculen"];
        let rust = &kayit["rust_olcumu"];
        assert_eq!(burada["sozluk_jeton"].as_u64(), Some(r.sozluk_jeton as u64));
        assert_eq!(burada["sozluk_logit"].as_u64(), Some(r.sozluk_logit as u64));
        assert_eq!(
            burada["bos_belgede_erisilmez_bayt"].as_u64(),
            Some(r.bos_belgede_erisilmez_bayt as u64),
            "Python'un sozluk muhasebesi Rust olcumunden ayrildi"
        );
        assert_eq!(rust["maskeli_kacis"].as_u64(), Some(r.maskeli.kacis as u64));
        assert_eq!(rust["maskeli_adim"].as_u64(), Some(r.maskeli.adim as u64));
        assert_eq!(rust["maskeli_bayt"].as_u64(), Some(r.maskeli.bayt as u64));
        assert_eq!(
            rust["maskeli_maskelenen"].as_u64(),
            Some(r.maskeli.maskelenen as u64)
        );
        assert_eq!(
            rust["maskeli_kapandi"].as_u64(),
            Some(r.maskeli.kapandi as u64)
        );
        assert_eq!(
            rust["maskesiz_kacis"].as_u64(),
            Some(r.maskesiz.kacis as u64)
        );
        assert_eq!(
            rust["bos_belgede_tuzak"].as_u64(),
            Some(r.bos_belgede_tuzak as u64)
        );
    }

    #[test]
    fn olcum_raporu_belirlenimci() {
        let a = olcum_raporu();
        let b = olcum_raporu();
        assert_eq!(a, b, "olcum iki kosuda ayni cikmadi");
        assert_eq!(a.maskeli.kacis, 0);
        assert!(a.bos_belgede_izinli > 0 && a.bos_belgede_red > 0);
        assert_eq!(
            a.bos_belgede_izinli + a.bos_belgede_red,
            a.sozluk_logit,
            "izinli + red sozluk boyunu vermiyor"
        );
    }
}
