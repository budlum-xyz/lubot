//! # merdiven - reading the stack at several depths instead of only the top
//!
//! A stack of `n` layers produces `n` hidden states per token and throws away
//! all but the last. That is a choice, not a law, and it costs two things:
//!
//! - **A decision that needed four layers pays for twenty.** If a level part
//!   way up already separates the classes, the rest of the stack is spent to
//!   produce a number that was already there.
//! - **The pooling head sees one opinion.** [`crate::sonda`] pools over
//!   *levels* as well as tokens; with one level the level axis is a formality
//!   and the gains it learns have nothing to choose between.
//!
//! This module names the levels. A ladder is a rising list of layer counts:
//! `[2, 4, 6]` means "read the state after two layers, after four, after
//! six". [`Merdiven::hucreler`] runs the backbone once and returns all of
//! them in the layout the pooling head expects: `jeton * kademe * d_model`.
//!
//! ## What is measured rather than argued
//!
//! - **One forward pass, not three.** The cells come from a single call to
//!   [`crate::Omurga::ileri_kademeli`]; running the stack once per level would
//!   be the same numbers at three times the cost, and
//!   `son_seviye_ileri_ile_bit_ozdes` pins the top level against
//!   [`crate::Omurga::ileri`] bit for bit.
//! - **Every level passes the same final norm.** Comparing a raw state with a
//!   normalised one would make the deeper level look larger for a reason that
//!   has nothing to do with what it knows.
//! - **The embedding is not a level.** Exit zero has read nothing - no layer
//!   has looked at another token yet - so it is refused rather than offered
//!   as a cheap level.
//!
//! ## What is not here
//!
//! No early-exit *policy*. [`Merdiven::butceye_gore`] truncates a ladder to a
//! depth budget, but which level is good enough for a given input is a
//! decision that needs a trained confidence head and a measured error rate,
//! and neither exists in this crate. Choosing a level by hand and calling it
//! adaptive computation would be a claim with no number under it.

use crate::{Omurga, OmurgaHatasi};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MerdivenHatasi {
    /// A ladder with no rungs.
    Bos,
    /// More levels than layers: two levels would have to share a layer.
    CokFazlaSeviye { seviye: usize, n_katman: usize },
    /// The stack has no layers.
    KatmanYok,
    /// Exits must rise strictly.
    SiraBozuk { onceki: usize, gelen: usize },
    /// An exit at zero (nothing has been read) or past the last layer.
    AralikDisi { kademe: usize, n_katman: usize },
    /// The budget leaves no rung standing.
    ButceBos { oran: f64 },
}

impl core::fmt::Display for MerdivenHatasi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bos => write!(f, "the ladder has no rungs"),
            Self::CokFazlaSeviye { seviye, n_katman } => {
                write!(f, "{seviye} levels asked of {n_katman} layers")
            }
            Self::KatmanYok => write!(f, "the stack has no layers"),
            Self::SiraBozuk { onceki, gelen } => {
                write!(f, "exits must rise: {gelen} follows {onceki}")
            }
            Self::AralikDisi { kademe, n_katman } => {
                write!(f, "exit {kademe} is outside 1..={n_katman}")
            }
            Self::ButceBos { oran } => write!(f, "a budget of {oran} leaves no rung"),
        }
    }
}

impl std::error::Error for MerdivenHatasi {}

/// A rising list of exit depths over a stack of known height.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Merdiven {
    n_katman: usize,
    kademeler: Vec<usize>,
}

impl Merdiven {
    /// A ladder from an explicit list.
    ///
    /// # Errors
    ///
    /// [`MerdivenHatasi::Bos`], [`MerdivenHatasi::KatmanYok`],
    /// [`MerdivenHatasi::SiraBozuk`], [`MerdivenHatasi::AralikDisi`].
    pub fn yeni(n_katman: usize, kademeler: &[usize]) -> Result<Self, MerdivenHatasi> {
        if n_katman == 0 {
            return Err(MerdivenHatasi::KatmanYok);
        }
        if kademeler.is_empty() {
            return Err(MerdivenHatasi::Bos);
        }
        let mut onceki = 0usize;
        for kademe in kademeler {
            if *kademe == 0 || *kademe > n_katman {
                return Err(MerdivenHatasi::AralikDisi {
                    kademe: *kademe,
                    n_katman,
                });
            }
            if *kademe <= onceki {
                return Err(MerdivenHatasi::SiraBozuk {
                    onceki,
                    gelen: *kademe,
                });
            }
            onceki = *kademe;
        }
        Ok(Self {
            n_katman,
            kademeler: kademeler.to_vec(),
        })
    }

    /// `seviye` rungs spread over the stack, the last one at the top.
    ///
    /// The spacing is `round(i * n / seviye)` counted from the top down, so
    /// the top rung is always the last layer and the gaps differ by at most
    /// one layer. Evenly spaced is not obviously the right schedule - it is
    /// the one that assumes nothing, and the schedule that assumes something
    /// needs a measurement this crate cannot make.
    ///
    /// # Errors
    ///
    /// [`MerdivenHatasi::KatmanYok`], [`MerdivenHatasi::Bos`],
    /// [`MerdivenHatasi::CokFazlaSeviye`].
    pub fn esit_aralikli(n_katman: usize, seviye: usize) -> Result<Self, MerdivenHatasi> {
        if n_katman == 0 {
            return Err(MerdivenHatasi::KatmanYok);
        }
        if seviye == 0 {
            return Err(MerdivenHatasi::Bos);
        }
        if seviye > n_katman {
            return Err(MerdivenHatasi::CokFazlaSeviye { seviye, n_katman });
        }
        let mut kademeler: Vec<usize> = (1..=seviye)
            .map(|i| (i * n_katman).div_ceil(seviye))
            .collect();
        kademeler.dedup();
        // Dedup can only shorten a rising list, and the list is rising by
        // construction, so the ladder stays valid.
        Self::yeni(n_katman, &kademeler)
    }

    /// The ladder truncated to a fraction of the stack's depth.
    ///
    /// A serving path that may spend only two thirds of the stack keeps the
    /// rungs at or below that depth. The top rung of the truncated ladder is
    /// then *not* the top of the stack, and that is the point.
    ///
    /// # Errors
    ///
    /// [`MerdivenHatasi::ButceBos`] when no rung survives, plus whatever
    /// [`Merdiven::yeni`] refuses.
    pub fn butceye_gore(&self, oran: f64) -> Result<Self, MerdivenHatasi> {
        let tavan = if oran <= 0.0 {
            0
        } else if oran >= 1.0 {
            self.n_katman
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let t = (oran * (self.n_katman as f64)).floor() as usize;
            t
        };
        let kalan: Vec<usize> = self
            .kademeler
            .iter()
            .copied()
            .filter(|k| *k <= tavan)
            .collect();
        if kalan.is_empty() {
            return Err(MerdivenHatasi::ButceBos { oran });
        }
        Self::yeni(self.n_katman, &kalan)
    }

    #[must_use]
    pub fn n_katman(&self) -> usize {
        self.n_katman
    }

    #[must_use]
    pub fn seviye(&self) -> usize {
        self.kademeler.len()
    }

    #[must_use]
    pub fn kademeler(&self) -> &[usize] {
        &self.kademeler
    }

    /// Does the ladder read the top of the stack?
    ///
    /// A ladder that does not is a truncated one, and a caller comparing its
    /// numbers with a full run is comparing two models.
    #[must_use]
    pub fn tepeyi_okur(&self) -> bool {
        self.kademeler.last() == Some(&self.n_katman)
    }

    /// Each rung as a fraction of the stack's depth.
    #[must_use]
    pub fn oranlar(&self) -> Vec<f64> {
        self.kademeler
            .iter()
            .map(|k| (*k as f64) / (self.n_katman as f64))
            .collect()
    }

    /// The layers between one rung and the next, first rung included.
    ///
    /// The gap is the compute a level costs over the one below it, so a
    /// schedule is cheap or expensive in exactly these numbers.
    #[must_use]
    pub fn araliklar(&self) -> Vec<usize> {
        let mut onceki = 0usize;
        let mut cikis = Vec::with_capacity(self.kademeler.len());
        for kademe in &self.kademeler {
            cikis.push(kademe - onceki);
            onceki = *kademe;
        }
        cikis
    }

    /// Run the backbone once and return every rung's state, token-major.
    ///
    /// # Errors
    ///
    /// Whatever [`Omurga::ileri_kademeli`] refuses, and
    /// [`OmurgaHatasi::KademeAralikDisi`] when the ladder was built for a
    /// stack of another height.
    pub fn hucreler(&self, omurga: &Omurga, jetonlar: &[u32]) -> Result<Vec<f32>, OmurgaHatasi> {
        if omurga.yapilandirma().n_katman != self.n_katman {
            return Err(OmurgaHatasi::KademeAralikDisi {
                kademe: self.n_katman,
                n_katman: omurga.yapilandirma().n_katman,
            });
        }
        omurga.ileri_kademeli(jetonlar, &self.kademeler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sonda::{Bas, Sonda, SondaSekli};
    use crate::{Tohum, Yapilandirma};

    fn omurga() -> Omurga {
        let yap = Yapilandirma::kucuk_aday();
        match Omurga::yeni(yap, 3) {
            Ok(o) => o,
            Err(e) => panic!("omurga: {e}"),
        }
    }

    fn merdiven(n: usize, seviye: usize) -> Merdiven {
        match Merdiven::esit_aralikli(n, seviye) {
            Ok(m) => m,
            Err(e) => panic!("merdiven: {e}"),
        }
    }

    #[test]
    fn katmansiz_yigin_reddedilir() {
        assert_eq!(Merdiven::yeni(0, &[1]), Err(MerdivenHatasi::KatmanYok));
        assert_eq!(
            Merdiven::esit_aralikli(0, 1),
            Err(MerdivenHatasi::KatmanYok)
        );
    }

    #[test]
    fn bos_merdiven_reddedilir() {
        assert_eq!(Merdiven::yeni(4, &[]), Err(MerdivenHatasi::Bos));
        assert_eq!(Merdiven::esit_aralikli(4, 0), Err(MerdivenHatasi::Bos));
    }

    #[test]
    fn sifirinci_kademe_reddedilir() {
        // The embedding has read nothing: no layer has looked at another
        // token yet, so it is not a cheap level.
        assert_eq!(
            Merdiven::yeni(4, &[0, 2]),
            Err(MerdivenHatasi::AralikDisi {
                kademe: 0,
                n_katman: 4
            })
        );
    }

    #[test]
    fn yiginin_ustunde_kademe_reddedilir() {
        assert_eq!(
            Merdiven::yeni(4, &[2, 5]),
            Err(MerdivenHatasi::AralikDisi {
                kademe: 5,
                n_katman: 4
            })
        );
    }

    #[test]
    fn tekrarlanan_kademe_reddedilir() {
        assert_eq!(
            Merdiven::yeni(6, &[2, 2]),
            Err(MerdivenHatasi::SiraBozuk {
                onceki: 2,
                gelen: 2
            })
        );
    }

    #[test]
    fn azalan_kademe_reddedilir() {
        assert_eq!(
            Merdiven::yeni(6, &[4, 3]),
            Err(MerdivenHatasi::SiraBozuk {
                onceki: 4,
                gelen: 3
            })
        );
    }

    #[test]
    fn katmandan_cok_seviye_reddedilir() {
        assert_eq!(
            Merdiven::esit_aralikli(3, 4),
            Err(MerdivenHatasi::CokFazlaSeviye {
                seviye: 4,
                n_katman: 3
            })
        );
    }

    #[test]
    fn esit_aralikli_bilinen_degerler() {
        assert_eq!(merdiven(6, 3).kademeler(), &[2, 4, 6]);
        assert_eq!(merdiven(6, 1).kademeler(), &[6]);
        assert_eq!(merdiven(6, 6).kademeler(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(merdiven(20, 4).kademeler(), &[5, 10, 15, 20]);
    }

    #[test]
    fn esit_aralikli_bolunmeyen_derinlik() {
        // 7 layers, 3 levels: the gaps differ by at most one layer and the
        // top rung is still the top of the stack.
        let m = merdiven(7, 3);
        assert_eq!(m.kademeler(), &[3, 5, 7]);
        let araliklar = m.araliklar();
        let enb = araliklar.iter().copied().max().unwrap_or(0);
        let enk = araliklar.iter().copied().min().unwrap_or(0);
        assert!(enb - enk <= 1, "{araliklar:?}");
    }

    #[test]
    fn esit_aralikli_hep_tepeyi_okur() {
        for n in 1..=24usize {
            for s in 1..=n.min(6) {
                let m = merdiven(n, s);
                assert!(m.tepeyi_okur(), "n={n} s={s}: {:?}", m.kademeler());
            }
        }
    }

    #[test]
    fn kademeler_hep_artar() {
        for n in 1..=24usize {
            for s in 1..=n.min(6) {
                let k = merdiven(n, s).kademeler().to_vec();
                assert!(k.windows(2).all(|w| w[0] < w[1]), "n={n} s={s}: {k:?}");
            }
        }
    }

    #[test]
    fn araliklar_derinlige_toplanir() {
        let m = merdiven(20, 4);
        assert_eq!(m.araliklar().iter().sum::<usize>(), 20);
    }

    #[test]
    fn oranlar_sifirla_bir_arasinda() {
        let m = merdiven(20, 4);
        let o = m.oranlar();
        assert_eq!(o.len(), 4);
        assert!(o.iter().all(|v| *v > 0.0 && *v <= 1.0));
        assert!((o[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn butce_ustteki_basamaklari_keser() {
        let m = merdiven(20, 4); // 5, 10, 15, 20
        let kesik = match m.butceye_gore(0.6) {
            Ok(k) => k,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(kesik.kademeler(), &[5, 10]);
        assert!(!kesik.tepeyi_okur());
        assert_eq!(kesik.n_katman(), 20);
    }

    #[test]
    fn butce_bir_iken_merdiven_degismez() {
        let m = merdiven(20, 4);
        let ayni = match m.butceye_gore(1.0) {
            Ok(k) => k,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(ayni, m);
    }

    #[test]
    fn butce_sifir_reddedilir() {
        let m = merdiven(20, 4);
        assert_eq!(
            m.butceye_gore(0.0),
            Err(MerdivenHatasi::ButceBos { oran: 0.0 })
        );
    }

    #[test]
    fn hucre_uzunlugu_sekle_uyar() {
        let o = omurga();
        let m = merdiven(o.yapilandirma().n_katman, 2);
        let jetonlar = [1u32, 2, 3, 4];
        let h = match m.hucreler(&o, &jetonlar) {
            Ok(h) => h,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(
            h.len(),
            jetonlar.len() * m.seviye() * o.yapilandirma().d_model
        );
    }

    #[test]
    fn son_seviye_ileri_ile_bit_ozdes() {
        // One forward pass, not one per level: the top rung must be exactly
        // what the ordinary forward pass produces, bit for bit.
        let o = omurga();
        let n = o.yapilandirma().n_katman;
        let d = o.yapilandirma().d_model;
        let m = merdiven(n, 3);
        let jetonlar = [5u32, 1, 9];
        let hucreler = match m.hucreler(&o, &jetonlar) {
            Ok(h) => h,
            Err(e) => panic!("{e}"),
        };
        let duz = match o.ileri(&jetonlar) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let seviye = m.seviye();
        for t in 0..jetonlar.len() {
            let son = &hucreler[(t * seviye + seviye - 1) * d..(t * seviye + seviye) * d];
            let beklenen = &duz[t * d..(t + 1) * d];
            for (a, b) in son.iter().zip(beklenen.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "jeton {t}");
            }
        }
    }

    #[test]
    fn tek_seviyeli_merdiven_ileriyle_ayni() {
        let o = omurga();
        let m = merdiven(o.yapilandirma().n_katman, 1);
        let jetonlar = [2u32, 7];
        let (a, b) = match (m.hucreler(&o, &jetonlar), o.ileri(&jetonlar)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => panic!("kosu reddetti"),
        };
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }

    #[test]
    fn alt_seviye_ust_seviyeden_farkli() {
        // If two rungs agreed the stack would be doing nothing between them,
        // and the level axis would be decoration.
        let o = omurga();
        let d = o.yapilandirma().d_model;
        let m = match Merdiven::yeni(o.yapilandirma().n_katman, &[1, o.yapilandirma().n_katman]) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        let h = match m.hucreler(&o, &[3u32, 4]) {
            Ok(h) => h,
            Err(e) => panic!("{e}"),
        };
        let alt = &h[0..d];
        let ust = &h[d..2 * d];
        assert!(alt.iter().zip(ust.iter()).any(|(a, b)| a != b));
    }

    #[test]
    fn baska_boydaki_yigin_reddedilir() {
        let o = omurga();
        let m = merdiven(o.yapilandirma().n_katman + 1, 1);
        assert!(matches!(
            m.hucreler(&o, &[1u32]),
            Err(OmurgaHatasi::KademeAralikDisi { .. })
        ));
    }

    #[test]
    fn bos_dizi_reddedilir() {
        let o = omurga();
        let m = merdiven(o.yapilandirma().n_katman, 2);
        assert!(matches!(m.hucreler(&o, &[]), Err(OmurgaHatasi::BosDizi)));
    }

    #[test]
    fn hucreler_belirlenimci() {
        let o = omurga();
        let m = merdiven(o.yapilandirma().n_katman, 2);
        let jetonlar = [1u32, 1, 2, 3];
        let (a, b) = match (m.hucreler(&o, &jetonlar), m.hucreler(&o, &jetonlar)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => panic!("kosu reddetti"),
        };
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }

    #[test]
    fn hucreler_havuzlamaya_dogrudan_girer() {
        // The layout is the pooling head's layout; if it were not, this test
        // would be the place it showed up rather than a silent reshape.
        let o = omurga();
        let d = o.yapilandirma().d_model;
        let m = merdiven(o.yapilandirma().n_katman, 2);
        let jetonlar = [1u32, 2, 3];
        let hucreler = match m.hucreler(&o, &jetonlar) {
            Ok(h) => h,
            Err(e) => panic!("{e}"),
        };
        let sekil = match SondaSekli::yeni(m.seviye(), 2, 2, d) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        };
        let mut tohum = Tohum::yeni(11);
        let bas = match Sonda::yeni(sekil, Bas::Guven, &mut tohum) {
            Ok(b) => b,
            Err(e) => panic!("{e}"),
        };
        match bas.ileri(&hucreler, jetonlar.len(), None, None) {
            Ok(v) => assert_eq!(v.len(), 1),
            Err(e) => panic!("havuz: {e}"),
        }
    }

    #[test]
    fn hata_metinleri_ayirt_edilir() {
        let metinler = [
            MerdivenHatasi::Bos.to_string(),
            MerdivenHatasi::KatmanYok.to_string(),
            MerdivenHatasi::CokFazlaSeviye {
                seviye: 2,
                n_katman: 1,
            }
            .to_string(),
            MerdivenHatasi::SiraBozuk {
                onceki: 2,
                gelen: 1,
            }
            .to_string(),
            MerdivenHatasi::AralikDisi {
                kademe: 9,
                n_katman: 4,
            }
            .to_string(),
            MerdivenHatasi::ButceBos { oran: 0.1 }.to_string(),
        ];
        for (i, a) in metinler.iter().enumerate() {
            for b in metinler.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }
}
