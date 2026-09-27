//! # dizi - packing several records into one window without letting them mix
//!
//! Records are short and windows are long, so a window holds several records
//! or it holds mostly padding. Both choices go wrong quietly:
//!
//! - **Padding that attends.** A padded position carries a real embedding
//!   (usually the pad token's) and, unless something stops it, every real
//!   token attends to it. The loss still falls, the numbers still look like
//!   numbers, and the model has learned to lean on a token that means
//!   nothing.
//! - **Records that attend to each other.** Packing two documents into one
//!   window and leaving attention alone lets a citation in the second one
//!   answer a question from the first. Measured from the outside this looks
//!   like reading comprehension.
//!
//! This module builds the packing and the mask that goes with it, and the
//! mask is the point: [`Paket::gorulebilir`] answers "may query `q` read key
//! `k`" with the record boundary and the window radius folded in, and the
//! tests below measure that the answer is no in exactly the cases where it
//! must be no.
//!
//! ## Measured, not asserted
//!
//! - `dolgu_hicbir_seyi_gormez` and `dolguyu_kimse_gormez` - padding is
//!   isolated in both directions. One direction is not enough: a padded
//!   position that cannot read but can be read is still a channel.
//! - `kayitlar_birbirini_gormez` - no pair of positions from two records is
//!   ever visible, at any radius, including the global radius.
//! - `her_jeton_kendini_gorur` - the diagonal survives every mask, because a
//!   row with no visible key has no softmax at all.
//! - `maske_simetrik` - this backbone is bidirectional; a mask that is not
//!   symmetric has smuggled in a causal rule that nothing declared.
//!
//! ## What is not here
//!
//! No tokenizer and no record source: [`Paket`] is built from lengths that
//! someone else measured. And no training-time loss masking - the pad
//! positions are marked, but what a trainer does with them is not this
//! crate's business, because this crate has no backward pass.

/// Where one record sits inside the packed window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yerlesim {
    /// Index of the first position.
    pub bas: usize,
    /// How many positions the record occupies.
    pub uzunluk: usize,
}

impl Yerlesim {
    #[must_use]
    pub fn son(&self) -> usize {
        self.bas + self.uzunluk
    }

    #[must_use]
    pub fn icerir(&self, konum: usize) -> bool {
        konum >= self.bas && konum < self.son()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiziHatasi {
    /// A window of length zero holds nothing.
    SifirPencere,
    /// A record of length zero is not a record.
    SifirKayit { sira: usize },
    /// The records do not fit.
    PencereYetmez { gereken: usize, pencere: usize },
    /// A position outside the window.
    KonumDisarida { konum: usize, pencere: usize },
}

impl core::fmt::Display for DiziHatasi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SifirPencere => write!(f, "the window has length zero"),
            Self::SifirKayit { sira } => write!(f, "record {sira} has length zero"),
            Self::PencereYetmez { gereken, pencere } => {
                write!(f, "{gereken} positions asked of a {pencere} window")
            }
            Self::KonumDisarida { konum, pencere } => {
                write!(f, "position {konum} is outside a {pencere} window")
            }
        }
    }
}

impl std::error::Error for DiziHatasi {}

/// How far a position may look inside its own record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Erim {
    /// The whole record, however long it is.
    Kayit,
    /// A symmetric window of the given radius, clipped by the record.
    Yaricap(usize),
}

/// Several records laid end to end in one window, plus the padding tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paket {
    pencere: usize,
    yerlesimler: Vec<Yerlesim>,
    /// Record index per position; `None` is padding.
    sahip: Vec<Option<usize>>,
}

impl Paket {
    /// Lay the records out in order and pad the rest.
    ///
    /// # Errors
    ///
    /// [`DiziHatasi::SifirPencere`], [`DiziHatasi::SifirKayit`],
    /// [`DiziHatasi::PencereYetmez`].
    pub fn paketle(pencere: usize, uzunluklar: &[usize]) -> Result<Self, DiziHatasi> {
        if pencere == 0 {
            return Err(DiziHatasi::SifirPencere);
        }
        let mut gereken = 0usize;
        for (sira, u) in uzunluklar.iter().enumerate() {
            if *u == 0 {
                return Err(DiziHatasi::SifirKayit { sira });
            }
            gereken += *u;
        }
        if gereken > pencere {
            return Err(DiziHatasi::PencereYetmez { gereken, pencere });
        }
        let mut yerlesimler = Vec::with_capacity(uzunluklar.len());
        let mut sahip = vec![None; pencere];
        let mut bas = 0usize;
        for (sira, u) in uzunluklar.iter().enumerate() {
            for yuva in sahip.iter_mut().skip(bas).take(*u) {
                *yuva = Some(sira);
            }
            yerlesimler.push(Yerlesim { bas, uzunluk: *u });
            bas += u;
        }
        Ok(Self {
            pencere,
            yerlesimler,
            sahip,
        })
    }

    #[must_use]
    pub fn pencere(&self) -> usize {
        self.pencere
    }

    #[must_use]
    pub fn kayit_sayisi(&self) -> usize {
        self.yerlesimler.len()
    }

    #[must_use]
    pub fn yerlesimler(&self) -> &[Yerlesim] {
        &self.yerlesimler
    }

    /// How many positions carry a record.
    #[must_use]
    pub fn dolu(&self) -> usize {
        self.sahip.iter().filter(|s| s.is_some()).count()
    }

    /// How many positions are padding.
    #[must_use]
    pub fn dolgu(&self) -> usize {
        self.pencere - self.dolu()
    }

    /// The fraction of the window that carries a record.
    ///
    /// The number a packing strategy lives or dies by, and the reason it is
    /// reported rather than assumed: a window that is two thirds padding
    /// spends two thirds of its attention on nothing.
    #[must_use]
    pub fn doluluk(&self) -> f64 {
        (self.dolu() as f64) / (self.pencere as f64)
    }

    /// Which record owns a position, or `None` for padding.
    #[must_use]
    pub fn sahip(&self, konum: usize) -> Option<usize> {
        self.sahip.get(konum).copied().flatten()
    }

    /// The token mask a pooling head needs: true where a record sits.
    #[must_use]
    pub fn tut_maskesi(&self) -> Vec<bool> {
        self.sahip.iter().map(Option::is_some).collect()
    }

    /// May the query at `q` read the key at `k`?
    ///
    /// Padding is invisible in both directions, records never see each other,
    /// and a radius is clipped by the record it sits in.
    #[must_use]
    pub fn gorulebilir(&self, q: usize, k: usize, erim: Erim) -> bool {
        let (Some(a), Some(b)) = (self.sahip(q), self.sahip(k)) else {
            return false;
        };
        if a != b {
            return false;
        }
        match erim {
            Erim::Kayit => true,
            Erim::Yaricap(r) => q.abs_diff(k) <= r,
        }
    }

    /// How many keys the query at `konum` may read.
    ///
    /// # Errors
    ///
    /// [`DiziHatasi::KonumDisarida`].
    pub fn gorulen_sayisi(&self, konum: usize, erim: Erim) -> Result<usize, DiziHatasi> {
        if konum >= self.pencere {
            return Err(DiziHatasi::KonumDisarida {
                konum,
                pencere: self.pencere,
            });
        }
        Ok((0..self.pencere)
            .filter(|k| self.gorulebilir(konum, *k, erim))
            .count())
    }

    /// Lay token ids out in the window, padding the rest with `dolgu_jeton`.
    ///
    /// # Errors
    ///
    /// [`DiziHatasi::PencereYetmez`] when the ids do not match the layout.
    pub fn yerlestir(&self, kayitlar: &[&[u32]], dolgu_jeton: u32) -> Result<Vec<u32>, DiziHatasi> {
        let gereken: usize = kayitlar.iter().map(|k| k.len()).sum();
        if kayitlar.len() != self.yerlesimler.len() || gereken > self.pencere {
            return Err(DiziHatasi::PencereYetmez {
                gereken,
                pencere: self.pencere,
            });
        }
        let mut cikis = vec![dolgu_jeton; self.pencere];
        for (kayit, yer) in kayitlar.iter().zip(self.yerlesimler.iter()) {
            if kayit.len() != yer.uzunluk {
                return Err(DiziHatasi::PencereYetmez {
                    gereken: kayit.len(),
                    pencere: yer.uzunluk,
                });
            }
            cikis[yer.bas..yer.son()].copy_from_slice(kayit);
        }
        Ok(cikis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paket() -> Paket {
        match Paket::paketle(10, &[3, 4]) {
            Ok(p) => p,
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn sifir_pencere_reddedilir() {
        assert_eq!(Paket::paketle(0, &[1]), Err(DiziHatasi::SifirPencere));
    }

    #[test]
    fn sifir_uzunluklu_kayit_reddedilir() {
        assert_eq!(
            Paket::paketle(8, &[3, 0]),
            Err(DiziHatasi::SifirKayit { sira: 1 })
        );
    }

    #[test]
    fn sigmayan_kayitlar_reddedilir() {
        assert_eq!(
            Paket::paketle(5, &[3, 4]),
            Err(DiziHatasi::PencereYetmez {
                gereken: 7,
                pencere: 5
            })
        );
    }

    #[test]
    fn bos_paket_gecerli() {
        let p = match Paket::paketle(4, &[]) {
            Ok(p) => p,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(p.kayit_sayisi(), 0);
        assert_eq!(p.dolu(), 0);
        assert_eq!(p.dolgu(), 4);
    }

    #[test]
    fn yerlesimler_ard_arda() {
        let p = paket();
        assert_eq!(p.yerlesimler()[0], Yerlesim { bas: 0, uzunluk: 3 });
        assert_eq!(p.yerlesimler()[1], Yerlesim { bas: 3, uzunluk: 4 });
        assert_eq!(p.yerlesimler()[1].son(), 7);
    }

    #[test]
    fn sahiplik_konum_konum_dogru() {
        let p = paket();
        assert_eq!(p.sahip(0), Some(0));
        assert_eq!(p.sahip(2), Some(0));
        assert_eq!(p.sahip(3), Some(1));
        assert_eq!(p.sahip(6), Some(1));
        assert_eq!(p.sahip(7), None);
        assert_eq!(p.sahip(99), None);
    }

    #[test]
    fn doluluk_olculur() {
        let p = paket();
        assert_eq!(p.dolu(), 7);
        assert_eq!(p.dolgu(), 3);
        assert!((p.doluluk() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn tut_maskesi_dolguyu_disarida_birakir() {
        let p = paket();
        let m = p.tut_maskesi();
        assert_eq!(m.len(), 10);
        assert_eq!(m.iter().filter(|v| **v).count(), 7);
        assert!(!m[7] && !m[8] && !m[9]);
    }

    #[test]
    fn her_jeton_kendini_gorur() {
        // A row with no visible key has no softmax; the diagonal is what
        // keeps every row defined.
        let p = paket();
        for konum in 0..p.pencere() {
            if p.sahip(konum).is_some() {
                assert!(p.gorulebilir(konum, konum, Erim::Kayit));
                assert!(p.gorulebilir(konum, konum, Erim::Yaricap(0)));
            }
        }
    }

    #[test]
    fn kayitlar_birbirini_gormez() {
        let p = paket();
        for erim in [Erim::Kayit, Erim::Yaricap(0), Erim::Yaricap(9)] {
            for q in 0..p.pencere() {
                for k in 0..p.pencere() {
                    if let (Some(a), Some(b)) = (p.sahip(q), p.sahip(k)) {
                        if a != b {
                            assert!(!p.gorulebilir(q, k, erim), "{q}->{k} {erim:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dolgu_hicbir_seyi_gormez() {
        let p = paket();
        for k in 0..p.pencere() {
            assert!(!p.gorulebilir(8, k, Erim::Kayit));
        }
    }

    #[test]
    fn dolguyu_kimse_gormez() {
        // The other direction: a padded position that cannot read but can be
        // read is still a channel into the model.
        let p = paket();
        for q in 0..p.pencere() {
            assert!(!p.gorulebilir(q, 8, Erim::Kayit));
        }
    }

    #[test]
    fn maske_simetrik() {
        // Bidirectional by declaration: an asymmetric mask would be a causal
        // rule nobody wrote down.
        let p = paket();
        for erim in [Erim::Kayit, Erim::Yaricap(1), Erim::Yaricap(3)] {
            for q in 0..p.pencere() {
                for k in 0..p.pencere() {
                    assert_eq!(
                        p.gorulebilir(q, k, erim),
                        p.gorulebilir(k, q, erim),
                        "{q}/{k} {erim:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn yaricap_kayitla_kirpilir() {
        let p = paket();
        // Position 3 starts the second record; radius 2 would reach position
        // 1, but the record boundary stops it.
        assert!(!p.gorulebilir(3, 1, Erim::Yaricap(2)));
        assert!(p.gorulebilir(3, 4, Erim::Yaricap(2)));
        assert!(p.gorulebilir(3, 5, Erim::Yaricap(2)));
        assert!(!p.gorulebilir(3, 6, Erim::Yaricap(2)));
    }

    #[test]
    fn gorulen_sayisi_yaricapla_buyur() {
        let p = paket();
        let dar = match p.gorulen_sayisi(4, Erim::Yaricap(1)) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let genis = match p.gorulen_sayisi(4, Erim::Yaricap(3)) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let tam = match p.gorulen_sayisi(4, Erim::Kayit) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(dar, 3);
        assert_eq!(genis, 4);
        assert_eq!(tam, 4);
        assert!(dar <= genis && genis <= tam);
    }

    #[test]
    fn gorulen_sayisi_kayit_disinda_sifir() {
        let p = paket();
        match p.gorulen_sayisi(9, Erim::Kayit) {
            Ok(v) => assert_eq!(v, 0),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn pencere_disi_konum_reddedilir() {
        let p = paket();
        assert_eq!(
            p.gorulen_sayisi(10, Erim::Kayit),
            Err(DiziHatasi::KonumDisarida {
                konum: 10,
                pencere: 10
            })
        );
    }

    #[test]
    fn genis_yaricap_kaydin_tamamini_gorur() {
        let p = paket();
        for konum in 3..7 {
            let genis = match p.gorulen_sayisi(konum, Erim::Yaricap(usize::MAX / 2)) {
                Ok(v) => v,
                Err(e) => panic!("{e}"),
            };
            let tam = match p.gorulen_sayisi(konum, Erim::Kayit) {
                Ok(v) => v,
                Err(e) => panic!("{e}"),
            };
            assert_eq!(genis, tam);
        }
    }

    #[test]
    fn yerlestirme_dolguyu_doldurur() {
        let p = paket();
        let a = [1u32, 2, 3];
        let b = [4u32, 5, 6, 7];
        let kayitlar: Vec<&[u32]> = vec![&a, &b];
        let dizi = match p.yerlestir(&kayitlar, 0) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(dizi, vec![1, 2, 3, 4, 5, 6, 7, 0, 0, 0]);
    }

    #[test]
    fn yerlestirme_uzunluk_uyusmazsa_reddeder() {
        let p = paket();
        let a = [1u32, 2];
        let b = [4u32, 5, 6, 7];
        let kayitlar: Vec<&[u32]> = vec![&a, &b];
        assert!(p.yerlestir(&kayitlar, 0).is_err());
    }

    #[test]
    fn yerlestirme_kayit_sayisi_denetlenir() {
        let p = paket();
        let a = [1u32, 2, 3];
        let kayitlar: Vec<&[u32]> = vec![&a];
        assert!(p.yerlestir(&kayitlar, 0).is_err());
    }

    #[test]
    fn tek_kayit_paketi_bolunmemis_diziyle_ayni_maskeyi_verir() {
        // One record filling the window: the packing must not invent a
        // boundary that was not there.
        let tek = match Paket::paketle(5, &[5]) {
            Ok(p) => p,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(tek.dolgu(), 0);
        for q in 0..5 {
            for k in 0..5 {
                assert!(tek.gorulebilir(q, k, Erim::Kayit));
            }
        }
    }

    #[test]
    fn yerlesim_icerir_sinirlari_dogru() {
        let y = Yerlesim { bas: 2, uzunluk: 3 };
        assert!(!y.icerir(1));
        assert!(y.icerir(2));
        assert!(y.icerir(4));
        assert!(!y.icerir(5));
    }

    #[test]
    fn hata_metinleri_ayirt_edilir() {
        let metinler = [
            DiziHatasi::SifirPencere.to_string(),
            DiziHatasi::SifirKayit { sira: 0 }.to_string(),
            DiziHatasi::PencereYetmez {
                gereken: 2,
                pencere: 1,
            }
            .to_string(),
            DiziHatasi::KonumDisarida {
                konum: 2,
                pencere: 1,
            }
            .to_string(),
        ];
        for (i, a) in metinler.iter().enumerate() {
            for b in metinler.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }
}
