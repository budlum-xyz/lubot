//! Rotary positions, with the pairing named rather than assumed.
//!
//! # Why the pairing is a field and not a comment
//!
//! Two different maps are both called "rotary position embedding". One rotates
//! the adjacent pair `(x[2j], x[2j+1])`. The other splits the head in half and
//! rotates `(x[j], x[j + d/2])`. Both are orthogonal, both preserve the norm,
//! and both give the relative-position property that makes the method work, so
//! a model built with one and served with the other does not crash, does not
//! warn, and does not produce a number anybody can look at and call wrong. It
//! just answers slightly worse, forever.
//!
//! That is the failure this module is shaped against. [`Eslesme`] is a value the
//! configuration carries, [`Rope::eslesme`] reports it, and
//! [`Rope::olculen_eslesme`] *measures* it back out of the implementation by
//! probing a basis vector. A test asserts the two agree. A declaration that
//! cannot be checked against the code is a comment, and comments do not rotate
//! anything.
//!
//! # What the tests actually prove
//!
//! - the rotation is orthogonal (the norm of a head is unchanged);
//! - position zero is the identity;
//! - the relative property holds: the inner product of a rotated query at `m`
//!   and a rotated key at `n` depends only on `m - n`, which is the entire
//!   reason to use rotary positions instead of learned ones;
//! - the two pairings genuinely differ, so the label carries information.

/// Which coordinates are rotated together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eslesme {
    /// `(x[2j], x[2j+1])` - the neighbouring pair.
    KomsuCift,
    /// `(x[j], x[j + d/2])` - the half split.
    YariyaBolme,
}

impl Eslesme {
    /// The stable name written into a shape signature.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::KomsuCift => "komsu-cift",
            Self::YariyaBolme => "yariya-bolme",
        }
    }
}

/// Why a rotation was refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KonumHatasi {
    /// A head of odd width has a coordinate with no partner.
    TekBoyut { d_head: usize },
    /// A head of width zero is not a head.
    SifirBoyut,
    /// The base of the frequency ladder must be greater than one, or every pair
    /// turns at the same rate and the positions stop being distinguishable.
    GecersizTaban { taban: f64 },
    /// The slice handed in is not one whole head.
    YanlisUzunluk { beklenen: usize, gelen: usize },
}

impl std::fmt::Display for KonumHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TekBoyut { d_head } => write!(
                f,
                "head width {d_head} is odd, so one coordinate has no partner to rotate with"
            ),
            Self::SifirBoyut => write!(f, "a head of width zero is not a head"),
            Self::GecersizTaban { taban } => write!(
                f,
                "the frequency base {taban} is not greater than one, so every pair would turn at the same rate"
            ),
            Self::YanlisUzunluk { beklenen, gelen } => write!(
                f,
                "expected one head of {beklenen} values, got {gelen}"
            ),
        }
    }
}

/// The rotary map for one head width.
#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    d_head: usize,
    taban: f64,
    eslesme: Eslesme,
}

impl Rope {
    /// Builds the map.
    ///
    /// # Errors
    ///
    /// [`KonumHatasi::SifirBoyut`], [`KonumHatasi::TekBoyut`] or
    /// [`KonumHatasi::GecersizTaban`].
    pub fn yeni(d_head: usize, taban: f64, eslesme: Eslesme) -> Result<Self, KonumHatasi> {
        if d_head == 0 {
            return Err(KonumHatasi::SifirBoyut);
        }
        if !d_head.is_multiple_of(2) {
            return Err(KonumHatasi::TekBoyut { d_head });
        }
        if taban <= 1.0 || !taban.is_finite() {
            return Err(KonumHatasi::GecersizTaban { taban });
        }
        Ok(Self {
            d_head,
            taban,
            eslesme,
        })
    }

    #[must_use]
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    #[must_use]
    pub fn taban(&self) -> f64 {
        self.taban
    }

    /// The pairing this map was built with.
    #[must_use]
    pub fn eslesme(&self) -> Eslesme {
        self.eslesme
    }

    /// How many pairs one head carries.
    #[must_use]
    pub fn cift_sayisi(&self) -> usize {
        self.d_head / 2
    }

    /// The two coordinates pair `j` occupies.
    #[must_use]
    pub fn cift_indeksleri(&self, j: usize) -> (usize, usize) {
        match self.eslesme {
            Eslesme::KomsuCift => (2 * j, 2 * j + 1),
            Eslesme::YariyaBolme => (j, j + self.cift_sayisi()),
        }
    }

    /// Cosine and sine for pair `j` at `pos`.
    #[must_use]
    pub fn aci(&self, j: usize, pos: usize) -> (f32, f32) {
        let ust = -2.0 * (j as f64) / (self.d_head as f64);
        let theta = self.taban.powf(ust);
        let aci = (pos as f64) * theta;
        (aci.cos() as f32, aci.sin() as f32)
    }

    /// Rotates one head in place.
    ///
    /// # Errors
    ///
    /// [`KonumHatasi::YanlisUzunluk`] when the slice is not one whole head.
    pub fn uygula(&self, kafa: &mut [f32], pos: usize) -> Result<(), KonumHatasi> {
        if kafa.len() != self.d_head {
            return Err(KonumHatasi::YanlisUzunluk {
                beklenen: self.d_head,
                gelen: kafa.len(),
            });
        }
        for j in 0..self.cift_sayisi() {
            let (a, b) = self.cift_indeksleri(j);
            let (cos, sin) = self.aci(j, pos);
            let x0 = kafa[a];
            let x1 = kafa[b];
            kafa[a] = x0 * cos - x1 * sin;
            kafa[b] = x0 * sin + x1 * cos;
        }
        Ok(())
    }

    /// Reads the pairing back out of the implementation instead of trusting the
    /// field.
    ///
    /// A basis vector is placed at coordinate zero and rotated by one position.
    /// Whichever coordinate the mass leaks into names the pairing: index `1`
    /// for the neighbouring pair, index `d/2` for the half split. This is the
    /// check that makes [`Rope::eslesme`] a measurement rather than a promise.
    #[must_use]
    pub fn olculen_eslesme(&self) -> Eslesme {
        let mut sonda = vec![0.0f32; self.d_head];
        sonda[0] = 1.0;
        // The only failure mode of `uygula` is a length mismatch, and the probe
        // is allocated at exactly the right length one line above.
        if self.uygula(&mut sonda, 1).is_err() {
            return self.eslesme;
        }
        let yari = self.cift_sayisi();
        let komsu = sonda.get(1).copied().unwrap_or(0.0).abs();
        let bolme = sonda.get(yari).copied().unwrap_or(0.0).abs();
        if bolme > komsu {
            Eslesme::YariyaBolme
        } else {
            Eslesme::KomsuCift
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nokta(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum()
    }

    fn norm(a: &[f32]) -> f64 {
        nokta(a, a).sqrt()
    }

    fn ornek(d: usize) -> Vec<f32> {
        // A fixed, uneven vector: a constant one would hide a pairing mistake,
        // because every coordinate would be interchangeable.
        (0..d)
            .map(|i| ((i as f32) * 0.37).sin() + 0.11 * (i as f32))
            .collect()
    }

    #[test]
    fn tek_boyut_reddedilir() {
        assert_eq!(
            Rope::yeni(7, 10_000.0, Eslesme::KomsuCift),
            Err(KonumHatasi::TekBoyut { d_head: 7 })
        );
    }

    #[test]
    fn sifir_boyut_reddedilir() {
        assert_eq!(
            Rope::yeni(0, 10_000.0, Eslesme::KomsuCift),
            Err(KonumHatasi::SifirBoyut)
        );
    }

    #[test]
    fn gecersiz_taban_reddedilir() {
        assert_eq!(
            Rope::yeni(8, 1.0, Eslesme::KomsuCift),
            Err(KonumHatasi::GecersizTaban { taban: 1.0 })
        );
    }

    #[test]
    fn yanlis_uzunluk_reddedilir() {
        let rope = Rope::yeni(8, 10_000.0, Eslesme::KomsuCift).unwrap();
        let mut kisa = vec![0.0; 6];
        assert_eq!(
            rope.uygula(&mut kisa, 3),
            Err(KonumHatasi::YanlisUzunluk {
                beklenen: 8,
                gelen: 6
            })
        );
    }

    #[test]
    fn sifir_konum_birim_doniisumdur() {
        for eslesme in [Eslesme::KomsuCift, Eslesme::YariyaBolme] {
            let rope = Rope::yeni(16, 10_000.0, eslesme).unwrap();
            let x = ornek(16);
            let mut y = x.clone();
            rope.uygula(&mut y, 0).unwrap();
            for (a, b) in x.iter().zip(y.iter()) {
                assert!((a - b).abs() < 1e-6, "position zero moved a coordinate");
            }
        }
    }

    #[test]
    fn dondurme_normu_korur() {
        for eslesme in [Eslesme::KomsuCift, Eslesme::YariyaBolme] {
            let rope = Rope::yeni(32, 10_000.0, eslesme).unwrap();
            let x = ornek(32);
            let once = norm(&x);
            for pos in [1usize, 7, 64, 4095] {
                let mut y = x.clone();
                rope.uygula(&mut y, pos).unwrap();
                let sonra = norm(&y);
                assert!(
                    (once - sonra).abs() < 1e-4,
                    "the map is not orthogonal at {pos}: {once} -> {sonra}"
                );
            }
        }
    }

    #[test]
    fn baginti_yalniz_konum_farkina_baglidir() {
        // The whole point of rotary positions: <R_m q, R_n k> is a function of
        // m - n alone. If this fails the module is not doing rotary positions,
        // whatever the doc comment says.
        for eslesme in [Eslesme::KomsuCift, Eslesme::YariyaBolme] {
            let rope = Rope::yeni(32, 10_000.0, eslesme).unwrap();
            let q = ornek(32);
            let k: Vec<f32> = ornek(32).iter().rev().copied().collect();
            let mut referans: Option<f64> = None;
            for kaydirma in [0usize, 3, 11, 40] {
                let mut qr = q.clone();
                let mut kr = k.clone();
                rope.uygula(&mut qr, 5 + kaydirma).unwrap();
                rope.uygula(&mut kr, 2 + kaydirma).unwrap();
                let deger = nokta(&qr, &kr);
                match referans {
                    None => referans = Some(deger),
                    Some(ilk) => assert!(
                        (ilk - deger).abs() < 1e-3,
                        "the inner product moved with absolute position: {ilk} vs {deger}"
                    ),
                }
            }
        }
    }

    #[test]
    fn iki_eslesme_ayni_sey_degildir() {
        let komsu = Rope::yeni(16, 10_000.0, Eslesme::KomsuCift).unwrap();
        let bolme = Rope::yeni(16, 10_000.0, Eslesme::YariyaBolme).unwrap();
        let x = ornek(16);
        let mut a = x.clone();
        let mut b = x.clone();
        komsu.uygula(&mut a, 9).unwrap();
        bolme.uygula(&mut b, 9).unwrap();
        let fark: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max);
        assert!(
            fark > 1e-3,
            "the two pairings produced the same output, so the label carries no information"
        );
    }

    #[test]
    fn beyan_edilen_eslesme_olculenle_ayni() {
        // The declaration is checked against the code, not against a comment.
        for eslesme in [Eslesme::KomsuCift, Eslesme::YariyaBolme] {
            let rope = Rope::yeni(16, 10_000.0, eslesme).unwrap();
            assert_eq!(
                rope.eslesme(),
                rope.olculen_eslesme(),
                "the declared pairing and the implemented pairing disagree"
            );
        }
    }

    #[test]
    fn cift_indeksleri_ortusmeden_kapsar() {
        for eslesme in [Eslesme::KomsuCift, Eslesme::YariyaBolme] {
            let rope = Rope::yeni(16, 10_000.0, eslesme).unwrap();
            let mut gorulen = vec![0u8; 16];
            for j in 0..rope.cift_sayisi() {
                let (a, b) = rope.cift_indeksleri(j);
                gorulen[a] += 1;
                gorulen[b] += 1;
            }
            assert!(
                gorulen.iter().all(|n| *n == 1),
                "the pairing does not cover every coordinate exactly once: {gorulen:?}"
            );
        }
    }

    #[test]
    fn aci_ilk_cift_icin_en_hizli_doner() {
        let rope = Rope::yeni(32, 10_000.0, Eslesme::YariyaBolme).unwrap();
        let (c0, s0) = rope.aci(0, 1);
        let (c_son, s_son) = rope.aci(rope.cift_sayisi() - 1, 1);
        let hizli = s0.atan2(c0).abs();
        let yavas = s_son.atan2(c_son).abs();
        assert!(
            hizli > yavas,
            "the frequency ladder does not descend: {hizli} vs {yavas}"
        );
    }

    #[test]
    fn deterministik() {
        let rope = Rope::yeni(16, 10_000.0, Eslesme::YariyaBolme).unwrap();
        let x = ornek(16);
        let mut a = x.clone();
        let mut b = x;
        rope.uygula(&mut a, 17).unwrap();
        rope.uygula(&mut b, 17).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ad_kararli() {
        assert_eq!(Eslesme::KomsuCift.ad(), "komsu-cift");
        assert_eq!(Eslesme::YariyaBolme.ad(), "yariya-bolme");
    }

    #[test]
    fn hata_metni_sayiyi_tasir() {
        let metin = KonumHatasi::TekBoyut { d_head: 7 }.to_string();
        assert!(metin.contains('7'), "the message hides which width failed");
        assert!(!KonumHatasi::SifirBoyut.to_string().is_empty());
        assert!(KonumHatasi::GecersizTaban { taban: 0.5 }
            .to_string()
            .contains("0.5"));
        assert!(KonumHatasi::YanlisUzunluk {
            beklenen: 8,
            gelen: 6
        }
        .to_string()
        .contains('6'));
    }
}
