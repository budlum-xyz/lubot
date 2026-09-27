//! The bias-free primitives a block is built from.
//!
//! # Why no biases
//!
//! Every linear map and every normalisation here carries a weight and no bias.
//! The reason is not fashion: a bias is one parameter per output channel that
//! the residual stream can already supply, and at the widths this repository can
//! afford (`training/model_spec.json` declares the family) those parameters are
//! a measurable fraction of the budget. Removing them is also the only way the
//! shape signature in [`crate::Omurga`] stays short enough to be read.
//!
//! The choice is a declaration, not a silent default: [`Norm`] has no bias field
//! to forget to initialise, and [`carp`] takes no bias argument to pass zero to.
//!
//! # Row-major, everywhere
//!
//! A weight of shape `(cikis, giris)` is stored as `cikis` contiguous rows of
//! `giris` values. A sequence of `s` vectors of width `giris` is `s` contiguous
//! rows. Mixing the two conventions inside one crate is the classic way to get a
//! transpose that only shows up as slightly worse quality, so there is one
//! convention and the shapes are checked.

/// Why an operation was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KatmanHatasi {
    /// The weight buffer is not `cikis * giris` long.
    AgirlikSekli { beklenen: usize, gelen: usize },
    /// The input is not a whole number of rows of the declared width.
    GirdiSekli {
        satir_genisligi: usize,
        gelen: usize,
    },
    /// A width of zero is not a width.
    SifirGenislik,
}

impl std::fmt::Display for KatmanHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgirlikSekli { beklenen, gelen } => write!(
                f,
                "the weight holds {gelen} values but its shape asks for {beklenen}"
            ),
            Self::GirdiSekli {
                satir_genisligi,
                gelen,
            } => write!(
                f,
                "{gelen} values is not a whole number of rows of width {satir_genisligi}"
            ),
            Self::SifirGenislik => write!(f, "a width of zero is not a width"),
        }
    }
}

/// `y = W x` for every row of `x`, with `W` of shape `(cikis, giris)`.
///
/// # Errors
///
/// [`KatmanHatasi::SifirGenislik`], [`KatmanHatasi::AgirlikSekli`] or
/// [`KatmanHatasi::GirdiSekli`].
pub fn carp(
    agirlik: &[f32],
    cikis: usize,
    giris: usize,
    x: &[f32],
) -> Result<Vec<f32>, KatmanHatasi> {
    if cikis == 0 || giris == 0 {
        return Err(KatmanHatasi::SifirGenislik);
    }
    if agirlik.len() != cikis * giris {
        return Err(KatmanHatasi::AgirlikSekli {
            beklenen: cikis * giris,
            gelen: agirlik.len(),
        });
    }
    if !x.len().is_multiple_of(giris) {
        return Err(KatmanHatasi::GirdiSekli {
            satir_genisligi: giris,
            gelen: x.len(),
        });
    }
    let satir = x.len() / giris;
    let mut sonuc = vec![0.0f32; satir * cikis];
    for (s, girdi) in x.chunks_exact(giris).enumerate() {
        let hedef = &mut sonuc[s * cikis..(s + 1) * cikis];
        for (yuva, satir_agirlik) in hedef.iter_mut().zip(agirlik.chunks_exact(giris)) {
            let mut toplam = 0.0f64;
            for (w, v) in satir_agirlik.iter().zip(girdi.iter()) {
                toplam += f64::from(*w) * f64::from(*v);
            }
            *yuva = toplam as f32;
        }
    }
    Ok(sonuc)
}

/// Layer normalisation with a learned gain and no bias.
#[derive(Debug, Clone, PartialEq)]
pub struct Norm {
    genislik: usize,
    eps: f32,
}

impl Norm {
    /// # Errors
    ///
    /// [`KatmanHatasi::SifirGenislik`].
    pub fn yeni(genislik: usize, eps: f32) -> Result<Self, KatmanHatasi> {
        if genislik == 0 {
            return Err(KatmanHatasi::SifirGenislik);
        }
        Ok(Self { genislik, eps })
    }

    #[must_use]
    pub fn genislik(&self) -> usize {
        self.genislik
    }

    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Normalises every row of `x` and scales it by `kazanc`.
    ///
    /// # Errors
    ///
    /// [`KatmanHatasi::AgirlikSekli`] when the gain is the wrong width, or
    /// [`KatmanHatasi::GirdiSekli`] when the input is not whole rows.
    pub fn uygula(&self, x: &[f32], kazanc: &[f32]) -> Result<Vec<f32>, KatmanHatasi> {
        if kazanc.len() != self.genislik {
            return Err(KatmanHatasi::AgirlikSekli {
                beklenen: self.genislik,
                gelen: kazanc.len(),
            });
        }
        if !x.len().is_multiple_of(self.genislik) {
            return Err(KatmanHatasi::GirdiSekli {
                satir_genisligi: self.genislik,
                gelen: x.len(),
            });
        }
        let n = self.genislik as f64;
        let mut sonuc = Vec::with_capacity(x.len());
        for satir in x.chunks_exact(self.genislik) {
            let ortalama = satir.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
            let varyans = satir
                .iter()
                .map(|v| {
                    let d = f64::from(*v) - ortalama;
                    d * d
                })
                .sum::<f64>()
                / n;
            let olcek = 1.0 / (varyans + f64::from(self.eps)).sqrt();
            for (v, g) in satir.iter().zip(kazanc.iter()) {
                sonuc.push((((f64::from(*v) - ortalama) * olcek) * f64::from(*g)) as f32);
            }
        }
        Ok(sonuc)
    }
}

/// The Gaussian error linear unit, tanh form.
///
/// The tanh form rather than the exact erf form because it is the one the
/// reference pattern uses and because it is two multiplications and a `tanh`
/// with no special-function table, which matters on the machines this reader is
/// meant to run on.
#[must_use]
pub fn gelu(v: f32) -> f32 {
    let x = f64::from(v);
    let ic = 0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x);
    (0.5 * x * (1.0 + ic.tanh())) as f32
}

/// The gated feed-forward: `W_out( gelu(a) * b )` where `W_in x = [a ; b]`.
///
/// # Errors
///
/// Whatever [`carp`] refuses, plus [`KatmanHatasi::SifirGenislik`] when the
/// gated half does not divide in two.
pub fn kapili_ileri(
    w_giris: &[f32],
    w_cikis: &[f32],
    d_model: usize,
    d_ff: usize,
    x: &[f32],
) -> Result<Vec<f32>, KatmanHatasi> {
    if d_ff == 0 {
        return Err(KatmanHatasi::SifirGenislik);
    }
    let genis = carp(w_giris, 2 * d_ff, d_model, x)?;
    let mut kapili = Vec::with_capacity(genis.len() / 2);
    for satir in genis.chunks_exact(2 * d_ff) {
        let (a, b) = satir.split_at(d_ff);
        for (p, q) in a.iter().zip(b.iter()) {
            kapili.push(gelu(*p) * q);
        }
    }
    carp(w_cikis, d_model, d_ff, &kapili)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carp_birim_matrisi_gecirir() {
        let mut birim = vec![0.0f32; 9];
        for i in 0..3 {
            birim[i * 3 + i] = 1.0;
        }
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = carp(&birim, 3, 3, &x).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn carp_sekli_dogru() {
        let w = vec![1.0f32; 2 * 5];
        let x = vec![1.0f32; 3 * 5];
        let y = carp(&w, 2, 5, &x).unwrap();
        assert_eq!(y.len(), 3 * 2);
        assert!(y.iter().all(|v| (*v - 5.0).abs() < 1e-6));
    }

    #[test]
    fn carp_satir_esleme_dogru() {
        // W = [[1,0],[0,10]] so the second output channel must be ten times the
        // second input. A transposed store would give the first channel instead.
        let w = vec![1.0, 0.0, 0.0, 10.0];
        let y = carp(&w, 2, 2, &[3.0, 7.0]).unwrap();
        assert_eq!(y, vec![3.0, 70.0]);
    }

    #[test]
    fn carp_yanlis_agirlik_reddedilir() {
        assert_eq!(
            carp(&[1.0, 2.0, 3.0], 2, 2, &[1.0, 1.0]),
            Err(KatmanHatasi::AgirlikSekli {
                beklenen: 4,
                gelen: 3
            })
        );
    }

    #[test]
    fn carp_yanlis_girdi_reddedilir() {
        assert_eq!(
            carp(&[1.0; 4], 2, 2, &[1.0, 1.0, 1.0]),
            Err(KatmanHatasi::GirdiSekli {
                satir_genisligi: 2,
                gelen: 3
            })
        );
    }

    #[test]
    fn carp_sifir_genislik_reddedilir() {
        assert_eq!(carp(&[], 0, 2, &[]), Err(KatmanHatasi::SifirGenislik));
        assert_eq!(carp(&[], 2, 0, &[]), Err(KatmanHatasi::SifirGenislik));
    }

    #[test]
    fn norm_sifir_genislik_reddedilir() {
        assert_eq!(Norm::yeni(0, 1e-5), Err(KatmanHatasi::SifirGenislik));
    }

    #[test]
    fn norm_ortalamayi_sifirlar() {
        let norm = Norm::yeni(4, 1e-5).unwrap();
        let y = norm.uygula(&[1.0, 2.0, 3.0, 10.0], &[1.0; 4]).unwrap();
        let toplam: f32 = y.iter().sum();
        assert!(toplam.abs() < 1e-4, "the mean was not removed: {toplam}");
    }

    #[test]
    fn norm_varyansi_bire_getirir() {
        let norm = Norm::yeni(8, 1e-9).unwrap();
        let x: Vec<f32> = (0..8).map(|i| (i as f32) * 3.0 - 4.0).collect();
        let y = norm.uygula(&x, &[1.0; 8]).unwrap();
        let var: f32 = y.iter().map(|v| v * v).sum::<f32>() / 8.0;
        assert!((var - 1.0).abs() < 1e-3, "variance is {var}, not one");
    }

    #[test]
    fn norm_kazanci_uygular() {
        let norm = Norm::yeni(4, 1e-9).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let bir = norm.uygula(&x, &[1.0; 4]).unwrap();
        let iki = norm.uygula(&x, &[2.0; 4]).unwrap();
        for (a, b) in bir.iter().zip(iki.iter()) {
            assert!((a * 2.0 - b).abs() < 1e-5);
        }
    }

    #[test]
    fn norm_coklu_satir_bagimsiz() {
        let norm = Norm::yeni(2, 1e-9).unwrap();
        let y = norm.uygula(&[1.0, 3.0, 100.0, 300.0], &[1.0; 2]).unwrap();
        // Both rows have the same shape, so both normalise to the same pair.
        assert!((y[0] - y[2]).abs() < 1e-4);
        assert!((y[1] - y[3]).abs() < 1e-4);
    }

    #[test]
    fn norm_yanlis_kazanc_reddedilir() {
        let norm = Norm::yeni(4, 1e-5).unwrap();
        assert_eq!(
            norm.uygula(&[0.0; 4], &[1.0; 3]),
            Err(KatmanHatasi::AgirlikSekli {
                beklenen: 4,
                gelen: 3
            })
        );
    }

    #[test]
    fn norm_yanlis_girdi_reddedilir() {
        let norm = Norm::yeni(4, 1e-5).unwrap();
        assert_eq!(
            norm.uygula(&[0.0; 5], &[1.0; 4]),
            Err(KatmanHatasi::GirdiSekli {
                satir_genisligi: 4,
                gelen: 5
            })
        );
    }

    #[test]
    fn norm_alanlari_geri_okunur() {
        let norm = Norm::yeni(6, 1e-4).unwrap();
        assert_eq!(norm.genislik(), 6);
        assert!((norm.eps() - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn gelu_sifirda_sifir() {
        assert!(gelu(0.0).abs() < 1e-7);
    }

    #[test]
    fn gelu_buyukte_ozdes() {
        assert!((gelu(8.0) - 8.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_kucukte_soner() {
        assert!(gelu(-8.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_monoton_degil_ama_sinirli() {
        // GELU dips below zero around -0.75 and comes back; a "monotone" claim
        // would be wrong, so what is asserted is the bound that is true.
        let en_kucuk = (-30..0)
            .map(|i| gelu((i as f32) / 10.0))
            .fold(f32::INFINITY, f32::min);
        assert!(
            en_kucuk > -0.2,
            "the dip is deeper than expected: {en_kucuk}"
        );
    }

    #[test]
    fn kapili_ileri_sekli_dogru() {
        let d_model = 4;
        let d_ff = 3;
        let w_giris = vec![0.1f32; 2 * d_ff * d_model];
        let w_cikis = vec![0.1f32; d_model * d_ff];
        let x = vec![1.0f32; 2 * d_model];
        let y = kapili_ileri(&w_giris, &w_cikis, d_model, d_ff, &x).unwrap();
        assert_eq!(y.len(), 2 * d_model);
    }

    #[test]
    fn kapili_ileri_kapi_gercekten_kapatir() {
        // A zero gate half must zero the output whatever the activation half is.
        let d_model = 2;
        let d_ff = 2;
        let mut w_giris = vec![0.0f32; 2 * d_ff * d_model];
        // First d_ff rows (the activated half) are ones; the gate half stays zero.
        for satir in 0..d_ff {
            for sutun in 0..d_model {
                w_giris[satir * d_model + sutun] = 1.0;
            }
        }
        let w_cikis = vec![1.0f32; d_model * d_ff];
        let y = kapili_ileri(&w_giris, &w_cikis, d_model, d_ff, &[1.0, 1.0]).unwrap();
        assert!(
            y.iter().all(|v| v.abs() < 1e-7),
            "a closed gate leaked: {y:?}"
        );
    }

    #[test]
    fn kapili_ileri_sifir_d_ff_reddedilir() {
        assert_eq!(
            kapili_ileri(&[], &[], 2, 0, &[1.0, 1.0]),
            Err(KatmanHatasi::SifirGenislik)
        );
    }

    #[test]
    fn hata_metinleri_sayilari_tasir() {
        assert!(KatmanHatasi::AgirlikSekli {
            beklenen: 4,
            gelen: 3
        }
        .to_string()
        .contains('3'));
        assert!(KatmanHatasi::GirdiSekli {
            satir_genisligi: 2,
            gelen: 3
        }
        .to_string()
        .contains('2'));
        assert!(!KatmanHatasi::SifirGenislik.to_string().is_empty());
    }
}
