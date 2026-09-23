//! The decision head: a small System 1 that decides and never writes.
//!
//! # Why a separate head
//!
//! Most of what the runtime does is not free text. It is a closed choice: which
//! tool to route to, whether this passage is relevant, whether the request must
//! be refused, whether a citation actually supports the claim. Those are
//! decisions, and a decision does not need an autoregressive model. [`Karar`]
//! is the whole output surface of this crate: three closed shapes and nothing
//! else. There is no way to ask it for a sentence, because there is no type
//! here that could hold one - the `decision-head-has-no-generation-surface`
//! gate refuses the source if a `String`, a `format!` or a `char` appears
//! outside the tests.
//!
//! The name follows the house motif: seed, block, bud. The bud is the small
//! thing that opens first.
//!
//! # The order is fixed
//!
//! Every decision point runs one order and never skips a tier
//! ([`Oncelik::SIRALI`]): deterministic code first - a calculator, a digest
//! check, mechanisms that already exist and cannot be wrong about arithmetic -
//! then this head, and only then the generative model. [`kademe_atlandi`]
//! refuses a route that jumped a tier, because a route that skips the
//! deterministic tier is a route that asked a model to do arithmetic.
//!
//! # Confidence is not enough; calibration is
//!
//! A decision taken with confidence below [`GUVEN_ESIGI`] is not taken: it is
//! escalated ([`Sonuc::Yukselt`]) so a wrong guess cannot become a quiet error.
//! And the confidence is only usable if recorded outcomes say so:
//! [`GuvenDefteri`] keeps the buckets and reports the worst gap between what
//! the head claimed and what it actually got right. An empty ledger is not a
//! clean ledger - with nothing recorded, the head is not allowed to decide
//! alone ([`GuvenDefteri::tek_bas_guvenli`]), which is what makes moving a
//! decision from the rule base to the head a measured step rather than an
//! opinion.
//!
//! # k-of-n is not the chain's threshold
//!
//! [`KONSENSUS_K`] of [`KONSENSUS_N`] heads must agree before a decision stands
//! on its own. That is a *model-internal* consensus over independently
//! initialised heads. It is deliberately **not** the chain's operator admission
//! threshold (K5): different mechanism, different surface, no shared constant,
//! no shared RPC. The separation is enforced structurally rather than by a
//! comment - this crate depends on neither chain crate, so it cannot read the
//! other's number even by accident. `model_consensus_is_not_the_chain_threshold`
//! checks that the manifest still says so.

use lubot_anlama::{BucketReport, Calibration, Outcome};

/// Bottom of the effort band the chain states (`0.5x-10.0x`).
pub const EFOR_TABAN: f64 = 0.5;
/// Top of the effort band the chain states (`0.5x-10.0x`).
pub const EFOR_TAVAN: f64 = 10.0;
/// How many of the heads must agree for a decision to stand alone.
///
/// Not the chain's operator admission threshold; see the module docs.
pub const KONSENSUS_K: usize = 2;
/// How many heads vote.
pub const KONSENSUS_N: usize = 3;
/// Below this confidence the head does not decide; it escalates.
pub const GUVEN_ESIGI: f64 = 0.6;

/// A closed decision point.
///
/// The list is closed on purpose: a decision point that is not named here does
/// not exist, and adding one is a change to this file rather than a runtime
/// choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Secenek {
    /// Which tool the question goes to, before any model is asked.
    AracYonlendirici,
    /// Whether reading this needs permission.
    IzinKarari,
    /// Whether a passage is relevant to the question.
    IndeksAramasi,
    /// Whether a citation actually supports the claim.
    AlintiDestegi,
    /// Whether the question is outside what Lubot reads at all.
    KapsamReddi,
    /// Which language the question is in.
    DilTespiti,
}

impl Secenek {
    /// Every decision point, in routing order.
    pub const HEPSI: [Self; 6] = [
        Self::AracYonlendirici,
        Self::IzinKarari,
        Self::IndeksAramasi,
        Self::AlintiDestegi,
        Self::KapsamReddi,
        Self::DilTespiti,
    ];

    /// The fixed label of this point. A label, not text: it comes from this
    /// match and nowhere else.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::AracYonlendirici => "arac-yonlendirici",
            Self::IzinKarari => "izin-karari",
            Self::IndeksAramasi => "indeks-aramasi",
            Self::AlintiDestegi => "alinti-destegi",
            Self::KapsamReddi => "kapsam-reddi",
            Self::DilTespiti => "dil-tespiti",
        }
    }
}

/// A number in `0..=1`, and nothing else.
///
/// Constructing one is fallible on purpose: a probability outside the interval,
/// or a `NaN`, is a bug in the caller and is refused here rather than being
/// clamped into something that looks like a measurement.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Puan(f64);

impl Puan {
    /// The bottom of the interval.
    pub const SIFIR: Self = Self(0.0);
    /// The top of the interval.
    pub const BIR: Self = Self(1.0);

    /// A number in `0..=1`, or `None` if it is not one.
    #[must_use]
    pub fn yeni(deger: f64) -> Option<Self> {
        // A `NaN` is not in the range either, so the interval check refuses it
        // without a separate branch.
        if (0.0..=1.0).contains(&deger) {
            Some(Self(deger))
        } else {
            None
        }
    }

    /// The value, which is in `0..=1` because the constructor said so.
    #[must_use]
    pub fn deger(self) -> f64 {
        self.0
    }

    /// `1 - self`, the confidence of the opposite.
    #[must_use]
    pub fn tumleyen(self) -> Self {
        Self(1.0 - self.0)
    }

    /// The weaker of two, for "as confident as the weakest head".
    #[must_use]
    pub fn en_az(self, b: Self) -> Self {
        if self.0 <= b.0 {
            self
        } else {
            b
        }
    }
}

/// Calibrated confidence attached to a decision.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Guven(pub Puan);

/// A pick from a closed list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SecenekKarari {
    /// What was picked.
    pub secim: Secenek,
    /// How confident the head is, calibrated.
    pub guven: Guven,
}

/// A graded score, for relevance ordering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PuanKarari {
    /// The score, in `0..=1`.
    pub deger: Puan,
    /// How confident the head is, calibrated.
    pub guven: Guven,
}

/// A yes or no, with the probability of the yes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvetHayirKarari {
    /// The decision.
    pub evet: bool,
    /// Probability of `evet`; the no-probability is its complement.
    pub olasilik: Puan,
    /// How confident the head is, calibrated.
    pub guven: Guven,
}

/// The whole output surface of this crate: three closed shapes.
///
/// There is deliberately no fourth variant and no variant that carries text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Karar {
    /// A pick from a closed list.
    Secenek(SecenekKarari),
    /// A graded score.
    Puan(PuanKarari),
    /// A yes or no with a probability.
    EvetHayir(EvetHayirKarari),
}

/// Which of the three closed shapes a [`Karar`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KararTipi {
    /// A pick from a closed list.
    Secenek,
    /// A graded score.
    Puan,
    /// A yes or no.
    EvetHayir,
}

impl Karar {
    /// The shape of this decision.
    #[must_use]
    pub fn tipi(self) -> KararTipi {
        match self {
            Self::Secenek(_) => KararTipi::Secenek,
            Self::Puan(_) => KararTipi::Puan,
            Self::EvetHayir(_) => KararTipi::EvetHayir,
        }
    }

    /// The calibrated confidence carried by any shape.
    #[must_use]
    pub fn guven(self) -> Guven {
        match self {
            Self::Secenek(k) => k.guven,
            Self::Puan(k) => k.guven,
            Self::EvetHayir(k) => k.guven,
        }
    }
}

/// Where a decision is taken, in the one allowed order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Oncelik {
    /// Existing deterministic code: calculator, digest check.
    Belirlenimci,
    /// This head: one typed pass, no generation.
    KararBasi,
    /// The generative model, only when free text is really needed.
    Uretken,
}

impl Oncelik {
    /// The fixed order. Deterministic code, then the head, then generation.
    pub const SIRALI: [Self; 3] = [Self::Belirlenimci, Self::KararBasi, Self::Uretken];

    /// The fixed label of this tier.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::Belirlenimci => "belirlenimci-kod",
            Self::KararBasi => "karar-basi",
            Self::Uretken => "uretken-model",
        }
    }
}

/// Whether a route skipped a tier.
///
/// A route that reaches a higher tier without passing through the lower ones is
/// a route that asked a model to do what a calculator already does.
#[must_use]
pub fn kademe_atlandi(gidilen: &[Oncelik]) -> bool {
    let mut beklenen = 0usize;
    for tier in gidilen {
        let found = Oncelik::SIRALI.iter().position(|t| t == tier);
        match found {
            Some(idx) if idx > beklenen => return true,
            Some(_) => beklenen = beklenen.saturating_add(1),
            None => return true,
        }
    }
    false
}

/// How much answering an effort request buys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EforButcesi {
    /// How many heads vote.
    pub baslik_sayisi: usize,
    /// Whether agreement of [`KONSENSUS_K`] of them is required.
    pub konsensus_zorunlu: bool,
    /// Whether the generative model may be reached at all.
    pub uretken_izinli: bool,
}

/// Why an effort request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EforHatasi {
    /// Outside the band the chain states (`0.5x-10.0x`).
    AralikDisi,
}

/// What an effort request buys: the low-effort end is answered by the head
/// alone, the high-effort end may reach the generative model.
///
/// # Errors
/// [`EforHatasi::AralikDisi`] when the request is outside `0.5x-10.0x`.
pub fn efor_butcesi(efor: f64) -> Result<EforButcesi, EforHatasi> {
    if !(EFOR_TABAN..=EFOR_TAVAN).contains(&efor) {
        return Err(EforHatasi::AralikDisi);
    }
    if efor < 1.0 {
        // Cheapest end: one head, no consensus, no generation.
        return Ok(EforButcesi {
            baslik_sayisi: 1,
            konsensus_zorunlu: false,
            uretken_izinli: false,
        });
    }
    if efor <= 2.0 {
        // Middle: three heads must agree, still no generation.
        return Ok(EforButcesi {
            baslik_sayisi: KONSENSUS_N,
            konsensus_zorunlu: true,
            uretken_izinli: false,
        });
    }
    Ok(EforButcesi {
        baslik_sayisi: KONSENSUS_N,
        konsensus_zorunlu: true,
        uretken_izinli: true,
    })
}

/// The policy a decision is taken under.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Politika {
    /// Below this, escalate instead of deciding.
    pub guven_esigi: Guven,
    /// How many votes are needed.
    pub konsensus_k: usize,
    /// How many heads vote.
    pub konsensus_n: usize,
}

/// Why a policy was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolitikaHatasi {
    /// `k` is zero, or greater than `n`, or `n` is zero.
    KonsensusGecersiz,
    /// The confidence threshold is not a probability.
    EsikAralikDisi,
}

impl Politika {
    /// The declared defaults: [`GUVEN_ESIGI`], [`KONSENSUS_K`] of
    /// [`KONSENSUS_N`].
    #[must_use]
    pub fn varsayilan() -> Self {
        Self {
            guven_esigi: Guven(Puan(GUVEN_ESIGI)),
            konsensus_k: KONSENSUS_K,
            konsensus_n: KONSENSUS_N,
        }
    }

    /// # Errors
    /// [`PolitikaHatasi::KonsensusGecersiz`] when `k` cannot be reached out of
    /// `n`; [`PolitikaHatasi::EsikAralikDisi`] when the threshold is not in
    /// `0..=1`.
    pub fn dogrula(self) -> Result<(), PolitikaHatasi> {
        if self.konsensus_k == 0 || self.konsensus_n == 0 || self.konsensus_k > self.konsensus_n {
            return Err(PolitikaHatasi::KonsensusGecersiz);
        }
        let esik = self.guven_esigi.0.deger();
        if !(0.0..=1.0).contains(&esik) {
            return Err(PolitikaHatasi::EsikAralikDisi);
        }
        Ok(())
    }
}

/// Why a decision was escalated instead of acted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YukseltmeNedeni {
    /// Confidence below the threshold: a wrong guess must not be quiet.
    GuvenEsikAltinda,
    /// The heads did not reach `k` of `n`.
    KonsensusYok,
    /// The heads did not all return the same closed shape.
    TipKarisik,
    /// Fewer votes arrived than the policy asks for.
    OySayisiEksik,
}

/// What a decision point ends in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sonuc {
    /// Act on this.
    Kesin(Karar),
    /// Do not act; hand it to the next tier, with the reason.
    Yukselt(YukseltmeNedeni),
    /// Refuse. A refusal is an answer, not a failure to answer.
    Red,
}

/// A vote count and what it produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Konsensus {
    /// The outcome.
    pub sonuc: Sonuc,
    /// Votes for yes.
    pub evet_oyu: usize,
    /// Votes for no.
    pub hayir_oyu: usize,
    /// How many votes the policy asked for.
    pub beklenen_n: usize,
}

/// One head deciding on its own.
///
/// # Errors
/// The policy's own validation errors.
pub fn tek_bas(karar: Karar, politika: &Politika) -> Result<Sonuc, PolitikaHatasi> {
    politika.dogrula()?;
    if karar.guven().0 < politika.guven_esigi.0 {
        return Ok(Sonuc::Yukselt(YukseltmeNedeni::GuvenEsikAltinda));
    }
    Ok(Sonuc::Kesin(karar))
}

/// Several heads deciding together: `k` of `n` must agree, and disagreement is
/// reported rather than resolved by picking the first vote.
///
/// # Errors
/// The policy's own validation errors.
pub fn konsensus(oylar: &[Karar], politika: &Politika) -> Result<Konsensus, PolitikaHatasi> {
    politika.dogrula()?;
    let beklenen_n = politika.konsensus_n;
    if oylar.len() < beklenen_n {
        return Ok(Konsensus {
            sonuc: Sonuc::Yukselt(YukseltmeNedeni::OySayisiEksik),
            evet_oyu: 0,
            hayir_oyu: 0,
            beklenen_n,
        });
    }
    let ilk = oylar[0].tipi();
    if oylar.iter().any(|o| o.tipi() != ilk) {
        return Ok(Konsensus {
            sonuc: Sonuc::Yukselt(YukseltmeNedeni::TipKarisik),
            evet_oyu: 0,
            hayir_oyu: 0,
            beklenen_n,
        });
    }
    let mut evet = 0usize;
    let mut hayir = 0usize;
    let mut en_zayif = Puan::BIR;
    for oy in oylar {
        en_zayif = en_zayif.en_az(oy.guven().0);
        match oy {
            Karar::EvetHayir(k) => {
                if k.evet {
                    evet = evet.saturating_add(1);
                } else {
                    hayir = hayir.saturating_add(1);
                }
            }
            // A non-boolean shape has no yes or no to count; it is decided by
            // the weakest head's confidence alone, which the threshold check
            // below handles.
            Karar::Secenek(_) | Karar::Puan(_) => {}
        }
    }
    // Whichever side reached `k` decides; a side that did not is a deadlock and
    // is reported rather than resolved by taking the first vote.
    let anlasan = if evet >= politika.konsensus_k {
        Some(evet)
    } else if hayir >= politika.konsensus_k {
        Some(hayir)
    } else {
        None
    };
    let sonuc = match anlasan {
        None => Sonuc::Yukselt(YukseltmeNedeni::KonsensusYok),
        // Agreement is not enough on its own: the weakest head still has to
        // clear the threshold, or the consensus would launder a weak vote.
        Some(_) if en_zayif < politika.guven_esigi.0 => {
            Sonuc::Yukselt(YukseltmeNedeni::GuvenEsikAltinda)
        }
        Some(_) => Sonuc::Kesin(oylar[0]),
    };
    Ok(Konsensus {
        sonuc,
        evet_oyu: evet,
        hayir_oyu: hayir,
        beklenen_n,
    })
}

/// Recorded outcomes for the head's confidence.
///
/// A thin record over [`Calibration`]: the head does not keep a second
/// calibration machine, because two counters for one question give two answers.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GuvenDefteri {
    ic: Calibration,
}

impl GuvenDefteri {
    /// An empty record. Empty is not clean: see [`Self::tek_bas_guvenli`].
    #[must_use]
    pub fn yeni() -> Self {
        Self {
            ic: Calibration::new(),
        }
    }

    /// Record what the head claimed and whether it was right.
    pub fn kaydet(&mut self, guven: Guven, dogru_mu: bool) {
        self.ic.record(Outcome {
            confidence: guven.0.deger(),
            correct: dogru_mu,
        });
    }

    /// How many outcomes are recorded.
    #[must_use]
    pub fn toplam(&self) -> u64 {
        self.ic.total()
    }

    /// The bucket where claimed and observed confidence disagree most.
    #[must_use]
    pub fn en_kotu_bosluk(&self) -> Option<BucketReport> {
        self.ic.worst_gap()
    }

    /// Whether the head may decide alone.
    ///
    /// `false` when nothing is recorded: an unmeasured head has not earned the
    /// right to decide without the next tier, and saying otherwise would turn
    /// "nobody checked" into "checked and fine".
    #[must_use]
    pub fn tek_bas_guvenli(&self, tolerans: f64) -> bool {
        if self.ic.total() == 0 {
            return false;
        }
        match self.ic.worst_gap() {
            Some(gap) => gap.gap.abs() <= tolerans,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evet(olasilik: f64) -> Karar {
        Karar::EvetHayir(EvetHayirKarari {
            evet: true,
            olasilik: Puan::yeni(olasilik).unwrap(),
            guven: Guven(Puan::yeni(olasilik).unwrap()),
        })
    }

    fn hayir(guven: f64) -> Karar {
        Karar::EvetHayir(EvetHayirKarari {
            evet: false,
            olasilik: Puan::yeni(1.0 - guven).unwrap(),
            guven: Guven(Puan::yeni(guven).unwrap()),
        })
    }

    #[test]
    fn out_of_interval_numbers_are_refused() {
        assert!(Puan::yeni(-0.001).is_none());
        assert!(Puan::yeni(1.001).is_none());
        assert!(Puan::yeni(f64::NAN).is_none());
        assert_eq!(Puan::yeni(0.0), Some(Puan::SIFIR));
        assert_eq!(Puan::yeni(1.0), Some(Puan::BIR));
    }

    #[test]
    fn low_confidence_escalates_instead_of_deciding() {
        let politika = Politika::varsayilan();
        let sonuc = tek_bas(
            Karar::EvetHayir(EvetHayirKarari {
                evet: true,
                olasilik: Puan::yeni(0.55).unwrap(),
                guven: Guven(Puan::yeni(0.55).unwrap()),
            }),
            &politika,
        )
        .unwrap();
        assert_eq!(sonuc, Sonuc::Yukselt(YukseltmeNedeni::GuvenEsikAltinda));
    }

    #[test]
    fn an_unreachable_consensus_policy_is_refused() {
        for politika in [
            Politika {
                guven_esigi: Guven(Puan::yeni(0.6).unwrap()),
                konsensus_k: 4,
                konsensus_n: 3,
            },
            Politika {
                guven_esigi: Guven(Puan::yeni(0.6).unwrap()),
                konsensus_k: 0,
                konsensus_n: 3,
            },
            Politika {
                guven_esigi: Guven(Puan::yeni(0.6).unwrap()),
                konsensus_k: 2,
                konsensus_n: 0,
            },
        ] {
            assert!(politika.dogrula().is_err());
        }
    }

    #[test]
    fn two_of_three_agrees_and_one_of_three_does_not() {
        let politika = Politika::varsayilan();
        let agrees = konsensus(&[evet(0.9), evet(0.8), hayir(0.7)], &politika).unwrap();
        assert_eq!(agrees.evet_oyu, 2);
        assert_eq!(agrees.hayir_oyu, 1);
        assert!(matches!(agrees.sonuc, Sonuc::Kesin(_)));

        let split = konsensus(&[evet(0.9), hayir(0.8), hayir(0.7)], &politika).unwrap();
        assert_eq!(split.evet_oyu, 1);
        assert_eq!(split.hayir_oyu, 2);
        assert!(matches!(split.sonuc, Sonuc::Kesin(_)), "two noes agree");

        let none = konsensus(
            &[evet(0.9), evet(0.8), evet(0.7)],
            &Politika {
                konsensus_k: 3,
                ..Politika::varsayilan()
            },
        )
        .unwrap();
        assert!(matches!(none.sonuc, Sonuc::Kesin(_)));

        let deadlock = konsensus(
            &[evet(0.9), hayir(0.8), evet(0.2)],
            &Politika {
                konsensus_k: 3,
                ..Politika::varsayilan()
            },
        )
        .unwrap();
        assert_eq!(
            deadlock.sonuc,
            Sonuc::Yukselt(YukseltmeNedeni::KonsensusYok)
        );
    }

    #[test]
    fn mixed_shapes_and_missing_votes_are_refused_not_averaged() {
        let politika = Politika::varsayilan();
        let mixed = konsensus(
            &[
                evet(0.9),
                Karar::Puan(PuanKarari {
                    deger: Puan::yeni(0.9).unwrap(),
                    guven: Guven(Puan::yeni(0.9).unwrap()),
                }),
                evet(0.8),
            ],
            &politika,
        )
        .unwrap();
        assert_eq!(mixed.sonuc, Sonuc::Yukselt(YukseltmeNedeni::TipKarisik));

        let few = konsensus(&[evet(0.9), evet(0.8)], &politika).unwrap();
        assert_eq!(few.sonuc, Sonuc::Yukselt(YukseltmeNedeni::OySayisiEksik));
        assert_eq!(few.beklenen_n, KONSENSUS_N);
    }

    #[test]
    fn a_skipped_tier_is_refused() {
        assert!(kademe_atlandi(&[Oncelik::Belirlenimci, Oncelik::Uretken]));
        assert!(kademe_atlandi(&[Oncelik::Uretken]));
        assert!(!kademe_atlandi(&Oncelik::SIRALI));
        assert!(!kademe_atlandi(&[
            Oncelik::Belirlenimci,
            Oncelik::KararBasi
        ]));
    }

    #[test]
    fn effort_outside_the_stated_band_is_refused() {
        assert_eq!(efor_butcesi(0.49), Err(EforHatasi::AralikDisi));
        assert_eq!(efor_butcesi(10.01), Err(EforHatasi::AralikDisi));
        assert_eq!(efor_butcesi(f64::NAN), Err(EforHatasi::AralikDisi));
        let cheap = efor_butcesi(EFOR_TABAN).unwrap();
        assert_eq!(cheap.baslik_sayisi, 1);
        assert!(!cheap.uretken_izinli);
        let mid = efor_butcesi(1.5).unwrap();
        assert!(mid.konsensus_zorunlu);
        assert!(!mid.uretken_izinli);
        let full = efor_butcesi(EFOR_TAVAN).unwrap();
        assert!(full.uretken_izinli);
    }

    #[test]
    fn an_unmeasured_ledger_does_not_permit_deciding_alone() {
        let defter = GuvenDefteri::yeni();
        assert_eq!(defter.toplam(), 0);
        assert!(defter.en_kotu_bosluk().is_none());
        assert!(!defter.tek_bas_guvenli(0.5));

        let mut olculmus = GuvenDefteri::yeni();
        for _ in 0..40 {
            olculmus.kaydet(Guven(Puan::yeni(0.85).unwrap()), true);
        }
        assert_eq!(olculmus.toplam(), 40);
        assert!(olculmus.en_kotu_bosluk().is_some());
        assert!(olculmus.tek_bas_guvenli(0.5));
        assert!(!olculmus.tek_bas_guvenli(0.0));
    }

    #[test]
    fn model_consensus_is_not_the_chain_threshold() {
        // K5 keeps the chain's operator admission rule chain-side. One is a
        // model-internal vote count, the other an admission rule between
        // operators; they must never share a constant or an RPC surface. The
        // check is structural: with no dependency on either chain crate, this
        // head cannot read the other's number at all.
        let manifest = include_str!("../Cargo.toml");
        for zincir in ["lubot-grant", "lubot-tools"] {
            assert!(
                !manifest.contains(zincir),
                "the head must not depend on {zincir}; that is how the two consensus mechanisms would merge"
            );
        }
        assert_eq!(KONSENSUS_N, KONSENSUS_K + 1);
    }

    #[test]
    fn the_output_surface_is_three_closed_shapes() {
        let kararlar = [
            Karar::Secenek(SecenekKarari {
                secim: Secenek::AracYonlendirici,
                guven: Guven(Puan::yeni(0.9).unwrap()),
            }),
            Karar::Puan(PuanKarari {
                deger: Puan::yeni(0.4).unwrap(),
                guven: Guven(Puan::yeni(0.9).unwrap()),
            }),
            evet(0.9),
        ];
        let tipler: Vec<KararTipi> = kararlar.iter().map(|k| k.tipi()).collect();
        assert_eq!(
            tipler,
            vec![KararTipi::Secenek, KararTipi::Puan, KararTipi::EvetHayir]
        );
        assert_eq!(Secenek::HEPSI.len(), 6);
        assert_eq!(Secenek::DilTespiti.ad(), "dil-tespiti");
        assert_eq!(Puan::yeni(0.25).unwrap().tumleyen().deger(), 0.75);
    }
}
