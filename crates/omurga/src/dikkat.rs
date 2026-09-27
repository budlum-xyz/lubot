//! Grouped-query attention, with the softmax written so it cannot silently
//! produce a NaN.
//!
//! # What "grouped-query" buys
//!
//! Every query head keeps its own projection; the key and value heads are
//! fewer, and several query heads read the same pair. The parameters saved are
//! real but secondary - the thing that actually matters on a small machine is
//! the size of the state a served model has to keep resident, and that state is
//! keys and values. Halving the number of key/value heads halves it.
//!
//! The grouping is not free and this module does not pretend otherwise: query
//! heads sharing a key head cannot specialise their retrieval independently.
//! Whether that costs anything on this corpus is a measurement nobody here has
//! taken, so nothing in this file claims it does not.
//!
//! # The masked row
//!
//! A masked position is scored `-inf`, and `exp(-inf) = 0`, which is the
//! behaviour that makes masking exact rather than approximate. But a row where
//! *every* position is masked has no maximum to subtract and sums to zero, and
//! the natural implementation divides zero by zero and writes NaN into the
//! residual stream, where it spreads to every later token and every later layer
//! before anything notices. [`yumusak_azami`] refuses that row
//! ([`DikkatHatasi::TumuMaskeli`]) instead of returning a plausible vector.
//! [`crate::pencere::Plan`] guarantees a token always sees itself, so the
//! refusal should be unreachable from the backbone - which is exactly why it is
//! tested directly.

/// Why attention was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DikkatHatasi {
    /// Zero heads, or a head of width zero.
    SifirKafa,
    /// The query heads do not divide evenly among the key/value heads, so some
    /// group would be a different size than the others.
    BolunmezGrup { n_sorgu: usize, n_kv: usize },
    /// There are more key/value heads than query heads, which is not a grouping.
    FazlaKvKafa { n_sorgu: usize, n_kv: usize },
    /// A buffer is not a whole number of positions.
    YanlisUzunluk { beklenen_kati: usize, gelen: usize },
    /// Query, key and value do not agree on how many positions there are.
    UyumsuzDizi { sorgu: usize, anahtar: usize },
    /// Every position in a row was masked, so the row has no distribution.
    TumuMaskeli { sorgu: usize },
    /// The head index is past the end.
    KafaYok { kafa: usize, n_kafa: usize },
}

impl std::fmt::Display for DikkatHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SifirKafa => write!(f, "zero heads, or a head of width zero"),
            Self::BolunmezGrup { n_sorgu, n_kv } => write!(
                f,
                "{n_sorgu} query heads do not divide evenly among {n_kv} key/value heads"
            ),
            Self::FazlaKvKafa { n_sorgu, n_kv } => write!(
                f,
                "{n_kv} key/value heads for {n_sorgu} query heads is not a grouping"
            ),
            Self::YanlisUzunluk {
                beklenen_kati,
                gelen,
            } => write!(
                f,
                "{gelen} values is not a whole number of positions of {beklenen_kati}"
            ),
            Self::UyumsuzDizi { sorgu, anahtar } => write!(
                f,
                "the query has {sorgu} positions and the key has {anahtar}"
            ),
            Self::TumuMaskeli { sorgu } => write!(
                f,
                "every position was masked for query {sorgu}, so there is no distribution to take"
            ),
            Self::KafaYok { kafa, n_kafa } => {
                write!(f, "head {kafa} is past the end of {n_kafa} heads")
            }
        }
    }
}

/// Turns scores into a distribution in place, subtracting the maximum first.
///
/// # Errors
///
/// [`DikkatHatasi::TumuMaskeli`] when no position is finite.
pub fn yumusak_azami(skorlar: &mut [f32], sorgu: usize) -> Result<(), DikkatHatasi> {
    let enbuyuk = skorlar
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !enbuyuk.is_finite() {
        return Err(DikkatHatasi::TumuMaskeli { sorgu });
    }
    let mut toplam = 0.0f64;
    for v in skorlar.iter_mut() {
        let e = if v.is_finite() {
            f64::from(*v - enbuyuk).exp()
        } else {
            0.0
        };
        *v = e as f32;
        toplam += e;
    }
    if toplam <= 0.0 {
        return Err(DikkatHatasi::TumuMaskeli { sorgu });
    }
    for v in skorlar.iter_mut() {
        *v = (f64::from(*v) / toplam) as f32;
    }
    Ok(())
}

/// The head layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gqa {
    n_sorgu: usize,
    n_kv: usize,
    d_head: usize,
}

impl Gqa {
    /// # Errors
    ///
    /// [`DikkatHatasi::SifirKafa`], [`DikkatHatasi::FazlaKvKafa`] or
    /// [`DikkatHatasi::BolunmezGrup`].
    pub fn yeni(n_sorgu: usize, n_kv: usize, d_head: usize) -> Result<Self, DikkatHatasi> {
        if n_sorgu == 0 || n_kv == 0 || d_head == 0 {
            return Err(DikkatHatasi::SifirKafa);
        }
        if n_kv > n_sorgu {
            return Err(DikkatHatasi::FazlaKvKafa { n_sorgu, n_kv });
        }
        if !n_sorgu.is_multiple_of(n_kv) {
            return Err(DikkatHatasi::BolunmezGrup { n_sorgu, n_kv });
        }
        Ok(Self {
            n_sorgu,
            n_kv,
            d_head,
        })
    }

    #[must_use]
    pub fn n_sorgu(&self) -> usize {
        self.n_sorgu
    }

    #[must_use]
    pub fn n_kv(&self) -> usize {
        self.n_kv
    }

    #[must_use]
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    /// How many query heads share one key/value head.
    #[must_use]
    pub fn grup_boyutu(&self) -> usize {
        self.n_sorgu / self.n_kv
    }

    /// Which key/value head this query head reads.
    ///
    /// # Errors
    ///
    /// [`DikkatHatasi::KafaYok`].
    pub fn kv_indeksi(&self, kafa: usize) -> Result<usize, DikkatHatasi> {
        if kafa >= self.n_sorgu {
            return Err(DikkatHatasi::KafaYok {
                kafa,
                n_kafa: self.n_sorgu,
            });
        }
        Ok(kafa / self.grup_boyutu())
    }

    /// `1/sqrt(d_head)`.
    ///
    /// The other scale this repository uses is `1/d_k`
    /// (`training/model_spec.json`, the `lubot-a1` family). The two are not
    /// interchangeable and the difference is a marked architectural decision,
    /// not something a new crate settles by picking one: this backbone is a
    /// separate family and declares `1/sqrt(d_head)` here so the divergence is
    /// visible in one place instead of buried in a forward pass.
    #[must_use]
    pub fn olcek(&self) -> f32 {
        1.0 / (self.d_head as f32).sqrt()
    }

    fn dizi_uzunlugu(&self, q: &[f32], k: &[f32], v: &[f32]) -> Result<usize, DikkatHatasi> {
        let q_satir = self.n_sorgu * self.d_head;
        let kv_satir = self.n_kv * self.d_head;
        if !q.len().is_multiple_of(q_satir) {
            return Err(DikkatHatasi::YanlisUzunluk {
                beklenen_kati: q_satir,
                gelen: q.len(),
            });
        }
        if !k.len().is_multiple_of(kv_satir) || !v.len().is_multiple_of(kv_satir) {
            return Err(DikkatHatasi::YanlisUzunluk {
                beklenen_kati: kv_satir,
                gelen: if !k.len().is_multiple_of(kv_satir) {
                    k.len()
                } else {
                    v.len()
                },
            });
        }
        let dizi = q.len() / q_satir;
        let k_dizi = k.len() / kv_satir;
        let v_dizi = v.len() / kv_satir;
        if k_dizi != dizi || v_dizi != dizi {
            return Err(DikkatHatasi::UyumsuzDizi {
                sorgu: dizi,
                anahtar: if k_dizi != dizi { k_dizi } else { v_dizi },
            });
        }
        Ok(dizi)
    }

    /// Attention over a whole sequence.
    ///
    /// `q` is `dizi` rows of `n_sorgu * d_head`; `k` and `v` are `dizi` rows of
    /// `n_kv * d_head`. `gorulebilir(sorgu, anahtar)` decides the mask. The
    /// result is `dizi` rows of `n_sorgu * d_head`, heads concatenated in order.
    ///
    /// # Errors
    ///
    /// [`DikkatHatasi::YanlisUzunluk`], [`DikkatHatasi::UyumsuzDizi`] or
    /// [`DikkatHatasi::TumuMaskeli`].
    pub fn ileri(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gorulebilir: &dyn Fn(usize, usize) -> bool,
    ) -> Result<Vec<f32>, DikkatHatasi> {
        let dizi = self.dizi_uzunlugu(q, k, v)?;
        let q_satir = self.n_sorgu * self.d_head;
        let kv_satir = self.n_kv * self.d_head;
        let mut cikti = vec![0.0f32; dizi * q_satir];
        let mut skorlar = vec![0.0f32; dizi];
        for kafa in 0..self.n_sorgu {
            let kv = self.kv_indeksi(kafa)?;
            let q_ofset = kafa * self.d_head;
            let kv_ofset = kv * self.d_head;
            for sorgu in 0..dizi {
                let qv = &q[sorgu * q_satir + q_ofset..sorgu * q_satir + q_ofset + self.d_head];
                for (anahtar, yuva) in skorlar.iter_mut().enumerate() {
                    if !gorulebilir(sorgu, anahtar) {
                        *yuva = f32::NEG_INFINITY;
                        continue;
                    }
                    let kvv = &k[anahtar * kv_satir + kv_ofset
                        ..anahtar * kv_satir + kv_ofset + self.d_head];
                    let mut nokta = 0.0f64;
                    for (a, b) in qv.iter().zip(kvv.iter()) {
                        nokta += f64::from(*a) * f64::from(*b);
                    }
                    *yuva = (nokta * f64::from(self.olcek())) as f32;
                }
                yumusak_azami(&mut skorlar, sorgu)?;
                let hedef =
                    &mut cikti[sorgu * q_satir + q_ofset..sorgu * q_satir + q_ofset + self.d_head];
                for (anahtar, agirlik) in skorlar.iter().enumerate() {
                    if *agirlik == 0.0 {
                        continue;
                    }
                    let vv = &v[anahtar * kv_satir + kv_ofset
                        ..anahtar * kv_satir + kv_ofset + self.d_head];
                    for (yuva, deger) in hedef.iter_mut().zip(vv.iter()) {
                        *yuva += agirlik * deger;
                    }
                }
            }
        }
        Ok(cikti)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hepsi(_: usize, _: usize) -> bool {
        true
    }

    #[test]
    fn yumusak_azami_bire_toplar() {
        let mut s = vec![1.0, 2.0, 3.0, -1.0];
        yumusak_azami(&mut s, 0).unwrap();
        let toplam: f32 = s.iter().sum();
        assert!((toplam - 1.0).abs() < 1e-6, "sums to {toplam}");
    }

    #[test]
    fn yumusak_azami_buyuk_sayida_tasmaz() {
        let mut s = vec![1.0e30, 1.0e30 + 1.0, 0.0];
        yumusak_azami(&mut s, 0).unwrap();
        assert!(s.iter().all(|v| v.is_finite()), "overflowed: {s:?}");
        let toplam: f32 = s.iter().sum();
        assert!((toplam - 1.0).abs() < 1e-5);
    }

    #[test]
    fn yumusak_azami_maskeyi_tam_sifirlar() {
        let mut s = vec![1.0, f32::NEG_INFINITY, 1.0];
        yumusak_azami(&mut s, 0).unwrap();
        assert_eq!(s[1], 0.0, "a masked position kept weight");
        assert!((s[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn yumusak_azami_tumu_maskeli_reddeder() {
        let mut s = vec![f32::NEG_INFINITY; 3];
        assert_eq!(
            yumusak_azami(&mut s, 7),
            Err(DikkatHatasi::TumuMaskeli { sorgu: 7 })
        );
    }

    #[test]
    fn yumusak_azami_bos_dizi_reddeder() {
        let mut s: Vec<f32> = Vec::new();
        assert_eq!(
            yumusak_azami(&mut s, 0),
            Err(DikkatHatasi::TumuMaskeli { sorgu: 0 })
        );
    }

    #[test]
    fn sifir_kafa_reddedilir() {
        assert_eq!(Gqa::yeni(0, 1, 4), Err(DikkatHatasi::SifirKafa));
        assert_eq!(Gqa::yeni(4, 0, 4), Err(DikkatHatasi::SifirKafa));
        assert_eq!(Gqa::yeni(4, 2, 0), Err(DikkatHatasi::SifirKafa));
    }

    #[test]
    fn bolunmez_grup_reddedilir() {
        assert_eq!(
            Gqa::yeni(6, 4, 8),
            Err(DikkatHatasi::BolunmezGrup {
                n_sorgu: 6,
                n_kv: 4
            })
        );
    }

    #[test]
    fn fazla_kv_kafa_reddedilir() {
        assert_eq!(
            Gqa::yeni(2, 4, 8),
            Err(DikkatHatasi::FazlaKvKafa {
                n_sorgu: 2,
                n_kv: 4
            })
        );
    }

    #[test]
    fn grup_esleme_dogru() {
        let gqa = Gqa::yeni(8, 2, 4).unwrap();
        assert_eq!(gqa.grup_boyutu(), 4);
        assert_eq!(gqa.kv_indeksi(0), Ok(0));
        assert_eq!(gqa.kv_indeksi(3), Ok(0));
        assert_eq!(gqa.kv_indeksi(4), Ok(1));
        assert_eq!(gqa.kv_indeksi(7), Ok(1));
        assert_eq!(
            gqa.kv_indeksi(8),
            Err(DikkatHatasi::KafaYok { kafa: 8, n_kafa: 8 })
        );
    }

    #[test]
    fn alanlar_geri_okunur() {
        let gqa = Gqa::yeni(8, 2, 4).unwrap();
        assert_eq!(gqa.n_sorgu(), 8);
        assert_eq!(gqa.n_kv(), 2);
        assert_eq!(gqa.d_head(), 4);
        assert!((gqa.olcek() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tek_konum_degeri_aynen_dondurur() {
        let gqa = Gqa::yeni(2, 1, 3).unwrap();
        let q = vec![0.5f32; 2 * 3];
        let k = vec![0.5f32; 3];
        let v = vec![1.0, 2.0, 3.0];
        let y = gqa.ileri(&q, &k, &v, &hepsi).unwrap();
        // One position: the distribution is a point mass, so the output is v.
        assert_eq!(y.len(), 2 * 3);
        for kafa in 0..2 {
            for (i, beklenen) in v.iter().enumerate() {
                assert!((y[kafa * 3 + i] - beklenen).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn kv_paylasan_kafalar_ayni_degeri_okur() {
        // Two query heads with identical queries and one shared kv head must
        // produce identical outputs; if the grouping indexes wrongly they will
        // not.
        let gqa = Gqa::yeni(2, 1, 2).unwrap();
        let dizi = 3;
        // Both heads get the *same* query, so any difference in the output can
        // only come from the grouping picking the wrong key/value head.
        let mut q: Vec<f32> = Vec::new();
        for s in 0..dizi {
            let kafa = [(s as f32) * 0.3, (s as f32) * 0.3 + 0.1];
            q.extend_from_slice(&kafa);
            q.extend_from_slice(&kafa);
        }
        let k: Vec<f32> = (0..dizi * 2).map(|i| (i as f32) * 0.2).collect();
        let v: Vec<f32> = (0..dizi * 2).map(|i| (i as f32) * 0.3).collect();
        let y = gqa.ileri(&q, &k, &v, &hepsi).unwrap();
        for s in 0..dizi {
            for i in 0..2 {
                let a = y[s * 4 + i];
                let b = y[s * 4 + 2 + i];
                assert!(
                    (a - b).abs() < 1e-6,
                    "shared kv heads diverged at ({s}, {i}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn maske_pencere_disini_kesiyor() {
        // Position 0 may only see itself; its output must equal v[0] exactly.
        let gqa = Gqa::yeni(1, 1, 2).unwrap();
        let dizi = 4;
        let q = vec![1.0f32; dizi * 2];
        let k = vec![1.0f32; dizi * 2];
        let v: Vec<f32> = (0..dizi * 2).map(|i| (i as f32) + 1.0).collect();
        let sadece_kendisi = |s: usize, a: usize| s == a;
        let y = gqa.ileri(&q, &k, &v, &sadece_kendisi).unwrap();
        for s in 0..dizi {
            assert!((y[s * 2] - v[s * 2]).abs() < 1e-5);
            assert!((y[s * 2 + 1] - v[s * 2 + 1]).abs() < 1e-5);
        }
    }

    #[test]
    fn ciktinin_her_satiri_degerlerin_disbukey_birlesimi() {
        // Attention is an average: no output coordinate may leave the range of
        // the values it averaged. A sign error in the softmax breaks this.
        let gqa = Gqa::yeni(2, 2, 2).unwrap();
        let dizi = 5;
        let q: Vec<f32> = (0..dizi * 4).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
        let k: Vec<f32> = (0..dizi * 4).map(|i| ((i * 5 % 11) as f32) - 5.0).collect();
        let v: Vec<f32> = (0..dizi * 4).map(|i| ((i * 3 % 17) as f32) - 8.0).collect();
        let y = gqa.ileri(&q, &k, &v, &hepsi).unwrap();
        let en_kucuk = v.iter().copied().fold(f32::INFINITY, f32::min);
        let en_buyuk = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for deger in &y {
            assert!(
                *deger >= en_kucuk - 1e-4 && *deger <= en_buyuk + 1e-4,
                "{deger} is outside [{en_kucuk}, {en_buyuk}]"
            );
        }
    }

    #[test]
    fn tam_gruplama_coklu_kafayla_ayni() {
        // With n_kv == n_sorgu the grouping is the identity, which is the
        // property that lets a grouped model be compared against a plain one.
        let gqa = Gqa::yeni(3, 3, 2).unwrap();
        assert_eq!(gqa.grup_boyutu(), 1);
        for kafa in 0..3 {
            assert_eq!(gqa.kv_indeksi(kafa), Ok(kafa));
        }
    }

    #[test]
    fn yanlis_uzunluk_reddedilir() {
        let gqa = Gqa::yeni(2, 1, 2).unwrap();
        let sonuc = gqa.ileri(&[0.0; 5], &[0.0; 2], &[0.0; 2], &hepsi);
        assert_eq!(
            sonuc,
            Err(DikkatHatasi::YanlisUzunluk {
                beklenen_kati: 4,
                gelen: 5
            })
        );
    }

    #[test]
    fn uyumsuz_dizi_reddedilir() {
        let gqa = Gqa::yeni(2, 1, 2).unwrap();
        let sonuc = gqa.ileri(&[0.0; 8], &[0.0; 2], &[0.0; 2], &hepsi);
        assert_eq!(
            sonuc,
            Err(DikkatHatasi::UyumsuzDizi {
                sorgu: 2,
                anahtar: 1
            })
        );
    }

    #[test]
    fn deterministik() {
        let gqa = Gqa::yeni(2, 1, 2).unwrap();
        let q: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25).collect();
        let k: Vec<f32> = (0..4).map(|i| (i as f32) * 0.5).collect();
        let v: Vec<f32> = (0..4).map(|i| (i as f32) * 0.75).collect();
        assert_eq!(
            gqa.ileri(&q, &k, &v, &hepsi).unwrap(),
            gqa.ileri(&q, &k, &v, &hepsi).unwrap()
        );
    }

    #[test]
    fn hata_metinleri_sayilari_tasir() {
        assert!(!DikkatHatasi::SifirKafa.to_string().is_empty());
        assert!(DikkatHatasi::BolunmezGrup {
            n_sorgu: 6,
            n_kv: 4
        }
        .to_string()
        .contains('6'));
        assert!(DikkatHatasi::FazlaKvKafa {
            n_sorgu: 2,
            n_kv: 4
        }
        .to_string()
        .contains('4'));
        assert!(DikkatHatasi::YanlisUzunluk {
            beklenen_kati: 4,
            gelen: 5
        }
        .to_string()
        .contains('5'));
        assert!(DikkatHatasi::UyumsuzDizi {
            sorgu: 2,
            anahtar: 1
        }
        .to_string()
        .contains('2'));
        assert!(DikkatHatasi::TumuMaskeli { sorgu: 7 }
            .to_string()
            .contains('7'));
        assert!(DikkatHatasi::KafaYok { kafa: 8, n_kafa: 8 }
            .to_string()
            .contains('8'));
    }
}
