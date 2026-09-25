//! The ladder, and the ceiling it is measured against.
//!
//! # The gap this closes
//!
//! `crates/egitim` states the rule the operator side runs on: an operator
//! answers with the machine it actually owns. The rule has never been
//! enforceable, for one concrete reason recorded in the workspace audit of
//! 2026-08-21: *an operator has nowhere to advertise a ceiling*. The bond
//! carries an amount and nothing else, so "can this machine serve this depth?"
//! has no term in it that refers to the machine. Effort tiers are therefore
//! priced but not gated.
//!
//! A ladder supplies the missing term. If a container's tensors are tagged with
//! the rung they belong to, and rung `0` is the part every depth needs, then a
//! machine's ceiling is a **depth**: the largest `k` whose rungs `0..=k` fit in
//! the bytes the machine will give up. That is measurable on the machine, it is
//! a single small integer, and it is exactly the shape a declaration needs.
//!
//! # What "fits" means, precisely
//!
//! Three quantities, kept apart because conflating them is how a ceiling
//! becomes a guess:
//!
//! - **Resident bytes.** Rungs the reader keeps in memory. This is what the
//!   ceiling is about.
//! - **Streamed bytes.** Rungs left in the mapped file and read group by group.
//!   Correct but slower, and how much slower is a property of the storage, not
//!   of this crate - so this crate reports the byte count and refuses to
//!   convert it into a time.
//! - **Scratch.** One quantisation group of floats, from
//!   `lubot_nicem::Nicemlenmis::en_buyuk_calisma_alani`. Constant in the model
//!   size, which is the whole reason a large model fits at all.
//!
//! # What is deliberately not decided here
//!
//! Whether a shallower depth is *good enough* is not a memory question and this
//! module does not answer it. Truncating a network at rung `k` produces a
//! different model, and whether that model still passes the exam set is the
//! exam set's business (`training/sinav.py`). [`Merdiven::sec`] returns what
//! fits; nothing here says what is acceptable, and a caller that treats "fits"
//! as "works" has made a claim this crate did not make.

use std::collections::BTreeMap;
use std::fmt;

use crate::bicim::{Kapsayici, Kayit};

/// Why a depth could not be chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerdivenHatasi {
    /// Even rung zero does not fit. The machine cannot serve this container at
    /// any depth, which is a refusal and not a depth of zero: a reader that
    /// reports depth zero looks like a working shallow model.
    TabanSigmiyor { taban_bayt: u64, tavan_bayt: u64 },
    /// The container has no rung zero. Every depth needs the shared part, so a
    /// container without one is malformed rather than merely deep.
    TabanYok,
    /// A rung is missing from the middle of the ladder. Rungs must be
    /// contiguous or "depth k" does not name a network.
    KademeBosluk { eksik: u8 },
}

impl fmt::Display for MerdivenHatasi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TabanSigmiyor {
                taban_bayt,
                tavan_bayt,
            } => write!(
                f,
                "taban kademe {taban_bayt} bayt, tavan {tavan_bayt} bayt: hicbir derinlik sigmaz"
            ),
            Self::TabanYok => write!(f, "kapsayicida 0. kademe yok"),
            Self::KademeBosluk { eksik } => {
                write!(f, "merdivende {eksik}. kademe eksik")
            }
        }
    }
}

impl std::error::Error for MerdivenHatasi {}

/// What a machine will give up, and where that number came from.
///
/// Two fields carry a label rather than a number because the repository's rule
/// is that an unmeasured figure is written as unmeasured. [`Tavan::olculdu`]
/// says whether the byte budget was read off this machine or supplied by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tavan {
    /// Bytes the reader may hold resident.
    pub yerlesik_bayt: u64,
    /// Whether that number was measured on the machine or declared by an
    /// operator. A declared ceiling is still a ceiling; it is just not
    /// evidence about the machine.
    pub olculdu: bool,
    /// Free text saying where the number came from, carried into the report so
    /// a ceiling can never appear without its provenance.
    pub kaynak: String,
}

impl Tavan {
    /// A ceiling measured from a machine's usable memory.
    ///
    /// `pay` is the fraction of usable memory the reader may take. It is a
    /// parameter and not a constant because the right fraction depends on what
    /// else the operator runs; the caller has to choose it, and the choice ends
    /// up in [`Tavan::kaynak`].
    ///
    /// # Panics
    ///
    /// Never. `pay` is clamped to `0.0..=1.0`, because a caller that passes
    /// `1.5` has made an error that should produce a small ceiling rather than
    /// a huge one.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn olcumden(kullanilabilir_bayt: u64, pay: f64, kaynak: &str) -> Self {
        let pay = pay.clamp(0.0, 1.0);
        let bayt = (kullanilabilir_bayt as f64 * pay) as u64;
        Self {
            yerlesik_bayt: bayt,
            olculdu: true,
            kaynak: format!("{kaynak}; pay {pay:.2}"),
        }
    }

    /// A ceiling an operator states without measuring one.
    #[must_use]
    pub fn beyandan(yerlesik_bayt: u64, kaynak: &str) -> Self {
        Self {
            yerlesik_bayt,
            olculdu: false,
            kaynak: format!("beyan: {kaynak}"),
        }
    }
}

/// Bytes on one rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kademe {
    /// Rung index; `0` is the shared part.
    pub no: u8,
    /// Payload bytes the tensors on this rung claim.
    pub bayt: u64,
    /// Number of tensors.
    pub tensor: usize,
    /// Weights on this rung.
    pub agirlik: u64,
}

/// The ladder of a container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Merdiven {
    kademeler: Vec<Kademe>,
}

/// A chosen depth and what it costs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Secim {
    /// The deepest rung kept resident. Rungs `0..=derinlik` are resident.
    pub derinlik: u8,
    /// Resident bytes.
    pub yerlesik_bayt: u64,
    /// Bytes left in the file, read group by group when needed.
    pub akan_bayt: u64,
    /// Rungs that did not fit.
    pub akan_kademe: Vec<u8>,
    /// Weights held resident.
    pub yerlesik_agirlik: u64,
    /// Weights in the container as a whole.
    pub toplam_agirlik: u64,
}

impl Secim {
    /// Fraction of the model held resident, by bytes. Reported because it is
    /// the number an operator can compare across machines; it is not a quality
    /// figure and nothing here says it is.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn yerlesiklik(&self) -> f64 {
        let toplam = self.yerlesik_bayt + self.akan_bayt;
        if toplam == 0 {
            return 1.0;
        }
        self.yerlesik_bayt as f64 / toplam as f64
    }

    /// Whether the whole container is resident.
    #[must_use]
    pub fn tam(&self) -> bool {
        self.akan_bayt == 0
    }
}

impl Merdiven {
    /// Build the ladder from a container's directory.
    ///
    /// # Errors
    ///
    /// [`MerdivenHatasi::TabanYok`] if nothing sits on rung zero, and
    /// [`MerdivenHatasi::KademeBosluk`] if the rungs are not contiguous.
    pub fn kur(k: &Kapsayici<'_>) -> Result<Self, MerdivenHatasi> {
        Self::kayitlardan(k.kayitlar())
    }

    /// Build the ladder from records directly, so the ladder can be reasoned
    /// about without a file in hand.
    ///
    /// # Errors
    ///
    /// As [`Self::kur`].
    pub fn kayitlardan(kayitlar: &[Kayit]) -> Result<Self, MerdivenHatasi> {
        let mut toplam: BTreeMap<u8, (u64, usize, u64)> = BTreeMap::new();
        for kayit in kayitlar {
            let giris = toplam.entry(kayit.kademe).or_insert((0, 0, 0));
            giris.0 += kayit.bayt();
            giris.1 += 1;
            giris.2 += kayit.agirlik();
        }
        if !toplam.contains_key(&0) {
            return Err(MerdivenHatasi::TabanYok);
        }
        let en_ust = toplam.keys().copied().max().unwrap_or(0);
        for no in 0..=en_ust {
            if !toplam.contains_key(&no) {
                return Err(MerdivenHatasi::KademeBosluk { eksik: no });
            }
        }
        let kademeler = toplam
            .into_iter()
            .map(|(no, (bayt, tensor, agirlik))| Kademe {
                no,
                bayt,
                tensor,
                agirlik,
            })
            .collect();
        Ok(Self { kademeler })
    }

    /// The rungs, in order.
    #[must_use]
    pub fn kademeler(&self) -> &[Kademe] {
        &self.kademeler
    }

    /// Deepest rung present.
    #[must_use]
    pub fn en_derin(&self) -> u8 {
        self.kademeler.last().map_or(0, |k| k.no)
    }

    /// Bytes needed to hold rungs `0..=derinlik`.
    #[must_use]
    pub fn bayt(&self, derinlik: u8) -> u64 {
        self.kademeler
            .iter()
            .filter(|k| k.no <= derinlik)
            .map(|k| k.bayt)
            .sum()
    }

    /// Choose the deepest rung that fits under a ceiling.
    ///
    /// # Errors
    ///
    /// [`MerdivenHatasi::TabanSigmiyor`] when rung zero alone exceeds the
    /// ceiling. There is no depth to report in that case, and reporting zero
    /// would look like a shallow model rather than a refusal.
    pub fn sec(&self, tavan: &Tavan) -> Result<Secim, MerdivenHatasi> {
        let ilk_bayt = self.kademeler.first().map_or(0, |k| k.bayt);
        if ilk_bayt > tavan.yerlesik_bayt {
            return Err(MerdivenHatasi::TabanSigmiyor {
                taban_bayt: ilk_bayt,
                tavan_bayt: tavan.yerlesik_bayt,
            });
        }
        let mut yerlesik = 0u64;
        let mut yerlesik_agirlik = 0u64;
        let mut derinlik = 0u8;
        let mut akan = 0u64;
        let mut akan_kademe = Vec::new();
        let mut doldu = false;
        for k in &self.kademeler {
            if !doldu && yerlesik + k.bayt <= tavan.yerlesik_bayt {
                yerlesik += k.bayt;
                yerlesik_agirlik += k.agirlik;
                derinlik = k.no;
            } else {
                // Once a rung does not fit, the rungs above it are streamed
                // too. Skipping one rung and taking the next would produce a
                // network with a hole in it, which is not a shallower model but
                // a broken one.
                doldu = true;
                akan += k.bayt;
                akan_kademe.push(k.no);
            }
        }
        Ok(Secim {
            derinlik,
            yerlesik_bayt: yerlesik,
            akan_bayt: akan,
            akan_kademe,
            yerlesik_agirlik,
            toplam_agirlik: self.kademeler.iter().map(|k| k.agirlik).sum(),
        })
    }

    /// A one-line declaration an operator can publish, and the only shape this
    /// crate offers for one.
    ///
    /// Carries the provenance of the ceiling with it, so a declaration derived
    /// from an unmeasured ceiling cannot be read as a measurement.
    #[must_use]
    pub fn beyan(&self, tavan: &Tavan, secim: &Secim) -> String {
        format!(
            "derinlik {} / {} | yerlesik {} bayt | akan {} bayt | yerlesiklik {:.3} | tavan {} ({})",
            secim.derinlik,
            self.en_derin(),
            secim.yerlesik_bayt,
            secim.akan_bayt,
            secim.yerlesiklik(),
            if tavan.olculdu { "olculdu" } else { "OLCULMEDI" },
            tavan.kaynak
        )
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use lubot_nicem::grup::Genislik;

    fn kayit(ad: &str, kademe: u8, bayt: u64) -> Kayit {
        // A synthetic record: the ladder only reads the rung and the byte
        // counts, so the shape fields are set to something self-consistent.
        Kayit {
            ad: ad.to_string(),
            genislik: Genislik::Bit(2),
            kademe,
            grup: 128,
            satir: 1,
            son_eksen: 128,
            olcek_ofset: 0,
            olcek_bayt: 0,
            yuk_ofset: 0,
            yuk_bayt: bayt,
        }
    }

    fn merdiven(boylar: &[(u8, u64)]) -> Merdiven {
        let kayitlar: Vec<Kayit> = boylar
            .iter()
            .enumerate()
            .map(|(i, (k, b))| kayit(&format!("t{i}"), *k, *b))
            .collect();
        Merdiven::kayitlardan(&kayitlar).expect("valid ladder")
    }

    #[test]
    fn the_deepest_rung_that_fits_is_chosen_and_the_rest_is_streamed() {
        let m = merdiven(&[(0, 1000), (1, 500), (2, 500), (3, 500)]);
        let t = Tavan::beyandan(2000, "test");
        let s = m.sec(&t).expect("base fits");
        assert_eq!(s.derinlik, 2);
        assert_eq!(s.yerlesik_bayt, 2000);
        assert_eq!(s.akan_bayt, 500);
        assert_eq!(s.akan_kademe, vec![3]);
        assert!(!s.tam());
        assert!((s.yerlesiklik() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn a_ceiling_that_holds_everything_reports_a_full_residency() {
        let m = merdiven(&[(0, 100), (1, 100)]);
        let s = m.sec(&Tavan::beyandan(10_000, "test")).expect("fits");
        assert_eq!(s.derinlik, 1);
        assert_eq!(s.akan_bayt, 0);
        assert!(s.tam());
        assert!((s.yerlesiklik() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_machine_that_cannot_hold_the_base_is_refused_not_given_depth_zero() {
        // Depth zero would look like a working shallow model. It is not one.
        let m = merdiven(&[(0, 5000), (1, 100)]);
        let e = m.sec(&Tavan::beyandan(1000, "test")).expect_err("refuses");
        assert_eq!(
            e,
            MerdivenHatasi::TabanSigmiyor {
                taban_bayt: 5000,
                tavan_bayt: 1000
            }
        );
        assert!(e.to_string().contains("hicbir derinlik"), "{e}");
    }

    #[test]
    fn a_rung_that_does_not_fit_stops_the_climb_rather_than_being_skipped() {
        // Rung 1 is large, rung 2 is small. A greedy filler that skipped rung 1
        // and took rung 2 would report depth 2 for a network missing a layer.
        let m = merdiven(&[(0, 100), (1, 9000), (2, 100)]);
        let s = m.sec(&Tavan::beyandan(500, "test")).expect("base fits");
        assert_eq!(s.derinlik, 0);
        assert_eq!(s.akan_kademe, vec![1, 2]);
        assert_eq!(s.akan_bayt, 9100);
    }

    #[test]
    fn a_ladder_with_a_hole_in_it_is_refused() {
        let kayitlar = vec![kayit("a", 0, 10), kayit("b", 1, 10), kayit("c", 3, 10)];
        assert_eq!(
            Merdiven::kayitlardan(&kayitlar).err(),
            Some(MerdivenHatasi::KademeBosluk { eksik: 2 })
        );
    }

    #[test]
    fn a_ladder_with_no_base_is_refused() {
        let kayitlar = vec![kayit("a", 1, 10)];
        assert_eq!(
            Merdiven::kayitlardan(&kayitlar).err(),
            Some(MerdivenHatasi::TabanYok)
        );
    }

    #[test]
    fn a_measured_ceiling_and_a_declared_one_are_not_reported_the_same_way() {
        let m = merdiven(&[(0, 100), (1, 100)]);
        let olculen = Tavan::olcumden(1_000_000, 0.5, "free(1) on this box");
        assert!(olculen.olculdu);
        assert_eq!(olculen.yerlesik_bayt, 500_000);
        let beyan = Tavan::beyandan(500_000, "operator");
        assert!(!beyan.olculdu);
        assert_eq!(olculen.yerlesik_bayt, beyan.yerlesik_bayt);

        let s = m.sec(&olculen).expect("fits");
        assert!(m.beyan(&olculen, &s).contains("olculdu"));
        assert!(m.beyan(&beyan, &s).contains("OLCULMEDI"));
    }

    #[test]
    fn a_nonsense_share_is_clamped_rather_than_amplified() {
        assert_eq!(Tavan::olcumden(1000, 1.5, "x").yerlesik_bayt, 1000);
        assert_eq!(Tavan::olcumden(1000, -1.0, "x").yerlesik_bayt, 0);
        assert!(Tavan::olcumden(1000, 0.25, "x").kaynak.contains("0.25"));
    }

    #[test]
    fn the_byte_total_for_a_depth_is_the_sum_of_its_rungs() {
        let m = merdiven(&[(0, 100), (1, 200), (2, 400)]);
        assert_eq!(m.bayt(0), 100);
        assert_eq!(m.bayt(1), 300);
        assert_eq!(m.bayt(2), 700);
        assert_eq!(m.bayt(9), 700);
        assert_eq!(m.en_derin(), 2);
        assert_eq!(m.kademeler().len(), 3);
        assert_eq!(m.kademeler()[1].tensor, 1);
    }

    #[test]
    fn a_larger_model_on_the_same_machine_gives_a_shallower_depth_not_a_failure() {
        // The property the whole crate exists for, stated as a test: growing
        // the model does not stop the machine serving, it lowers what the
        // machine may declare.
        let kucuk = merdiven(&[(0, 100), (1, 100), (2, 100)]);
        let buyuk = merdiven(&[(0, 100), (1, 1000), (2, 1000)]);
        let t = Tavan::beyandan(1200, "same machine");
        assert_eq!(kucuk.sec(&t).expect("fits").derinlik, 2);
        assert_eq!(buyuk.sec(&t).expect("fits").derinlik, 1);
    }
}
