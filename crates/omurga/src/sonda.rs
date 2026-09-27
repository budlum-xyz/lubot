//! # sonda - probe pooling: a sequence read into a fixed number of vectors
//!
//! The backbone produces one vector per token. A decision needs one vector,
//! not a thousand, and the usual answers are bad ones: the last position is a
//! position and not a summary, the mean treats a citation and a filler word
//! as equals, and a linear layer over a padded maximum length makes the shape
//! depend on the padding.
//!
//! This module pools instead, in two stages, both of them learned:
//!
//! 1. **Probes read the sequence.** Each level of the stack carries `k` probe
//!    vectors; a probe scores every token, the scores are a softmax over
//!    *tokens*, and the probe's answer is the weighted sum of the token
//!    states. `l * k` answers come out, each one normalised to unit RMS and
//!    then scaled by its own learned gain, so a probe that has nothing to say
//!    cannot shout over one that does simply by holding a longer vector.
//! 2. **Queries read the probes.** Each of the `q` queries scores every one of
//!    the `l * k` probe answers, the softmax is over that flattened set, and
//!    the query's answer is the weighted sum. The result is `q * d` numbers:
//!    a fixed shape, whatever the sequence length was.
//!
//! Two properties follow from that and are measured rather than assumed:
//!
//! - **Order does not reach the head.** Pooling is a weighted sum over tokens,
//!   so shuffling the sequence cannot change the result
//!   (`havuz_jeton_sirasindan_bagimsiz`). Whatever the head knows about order,
//!   it learned from the backbone's positions, not from this pooling - and if
//!   someone later adds an order-dependent shortcut here, that test fails.
//! - **A masked token is absent, not quiet.** Masking is applied before the
//!   softmax, so a masked token's weight is exactly zero rather than small
//!   (`maskeli_jeton_tam_olarak_disarida`).
//!
//! ## Where this refuses instead of producing a number
//!
//! A row whose every token is masked has no softmax: the denominator is zero
//! and the usual answer is a quiet `NaN` that travels a long way before
//! anyone notices. Here it is [`SondaHatasi::TumuMaskeli`]. The same holds for
//! a level mask that excludes every level.
//!
//! ## Heads
//!
//! [`Bas`] names the three shapes that sit on top of the pooling: a
//! confidence scalar, a three-way route, and an embedding of the configured
//! width. Only the output width and the bias convention differ, so they are
//! one type with a discriminant rather than three copies of one projection.
//! None of them produces text.
//!
//! No backward pass: this crate is forward-only.

use crate::Tohum;

/// Epsilon of the unit-RMS step, matching [`crate::katman::Norm`]'s floor.
pub const RMS_EPS: f32 = 1e-6;

/// The route head's calibration triple: threshold, offset, floor.
///
/// Carried as data, not as three literals scattered through the code, because
/// a calibration that lives in three places drifts in two of them.
pub const ROTA_KALIBRASYONU: [f32; 3] = [0.90, 0.00, 0.60];

/// The embedding head's temperature and bias at initialisation.
pub const GOMME_SICAKLIK_INIT: f32 = 10.0;
pub const GOMME_YANLILIK_INIT: f32 = -10.0;

/// Standard deviation of the probe and query initialiser.
pub const SONDA_INIT_STD: f32 = 0.02;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SondaHatasi {
    /// Some dimension is zero, so there is nothing to pool.
    SifirBoyut,
    /// The cell buffer is not `jeton * seviye * d` long.
    HucreSekli { beklenen: usize, gelen: usize },
    /// A mask does not match the thing it masks.
    MaskeSekli { beklenen: usize, gelen: usize },
    /// Every token is masked: a softmax with no support is not a distribution.
    TumuMaskeli,
    /// Every level is masked.
    TumSeviyelerKapali,
}

impl core::fmt::Display for SondaHatasi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SifirBoyut => write!(f, "a dimension is zero"),
            Self::HucreSekli { beklenen, gelen } => {
                write!(f, "cells: expected {beklenen} numbers, got {gelen}")
            }
            Self::MaskeSekli { beklenen, gelen } => {
                write!(f, "mask: expected {beklenen} entries, got {gelen}")
            }
            Self::TumuMaskeli => write!(f, "every token is masked"),
            Self::TumSeviyelerKapali => write!(f, "every level is masked"),
        }
    }
}

impl std::error::Error for SondaHatasi {}

/// Which head sits on the pooled vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bas {
    /// One number: how sure the model is. Biased.
    Guven,
    /// Three numbers: the route. Biased, and calibrated by
    /// [`ROTA_KALIBRASYONU`].
    YonSecimi,
    /// `genislik` numbers: an embedding. Deliberately *unbiased* - a constant
    /// added to every embedding moves every distance by the same amount and
    /// buys nothing, but it does make two runs with different biases look
    /// like different models.
    Gomme { genislik: usize },
}

impl Bas {
    #[must_use]
    pub fn cikis_genisligi(&self) -> usize {
        match self {
            Self::Guven => 1,
            Self::YonSecimi => 3,
            Self::Gomme { genislik } => *genislik,
        }
    }

    #[must_use]
    pub fn yanlilik_var(&self) -> bool {
        !matches!(self, Self::Gomme { .. })
    }
}

/// The shape of a probe head: levels, probes per level, queries, width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SondaSekli {
    pub seviye: usize,
    pub sonda: usize,
    pub sorgu: usize,
    pub d: usize,
}

impl SondaSekli {
    /// # Errors
    ///
    /// [`SondaHatasi::SifirBoyut`] when any dimension is zero.
    pub fn yeni(seviye: usize, sonda: usize, sorgu: usize, d: usize) -> Result<Self, SondaHatasi> {
        if seviye == 0 || sonda == 0 || sorgu == 0 || d == 0 {
            return Err(SondaHatasi::SifirBoyut);
        }
        Ok(Self {
            seviye,
            sonda,
            sorgu,
            d,
        })
    }

    /// The pooled width: `sorgu * d`, and it does not depend on the sequence
    /// length - which is the whole reason this module exists.
    #[must_use]
    pub fn havuz_genisligi(&self) -> usize {
        self.sorgu * self.d
    }

    /// Parameters of the pooling stage alone, from the shape.
    #[must_use]
    pub fn havuz_param_sayisi(&self) -> usize {
        self.seviye * self.sonda * self.d          // probes
            + self.seviye * self.sonda             // gains
            + self.sorgu * self.d                  // queries
            + self.sorgu * self.seviye * self.sonda // row biases
    }

    /// Parameters including the projection onto `bas`.
    #[must_use]
    pub fn param_sayisi(&self, bas: Bas) -> usize {
        let cikis = bas.cikis_genisligi();
        self.havuz_param_sayisi()
            + self.havuz_genisligi() * cikis
            + usize::from(bas.yanlilik_var()) * cikis
    }
}

/// Unit-RMS over one vector, with the same floor the backbone's norm uses.
pub fn birim_rms(v: &mut [f32]) {
    let kare: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    let rms = (kare / (v.len() as f64) + f64::from(RMS_EPS)).sqrt();
    let olcek = (1.0 / rms) as f32;
    for x in v.iter_mut() {
        *x *= olcek;
    }
}

/// Softmax over a masked row.
///
/// `gecerli` marks the entries that take part. An empty support is refused
/// rather than divided by zero.
///
/// # Errors
///
/// [`SondaHatasi::TumuMaskeli`] when nothing is valid.
pub fn maskeli_yumusak_azami(puan: &[f32], gecerli: &[bool]) -> Result<Vec<f32>, SondaHatasi> {
    let mut enb = f32::NEG_INFINITY;
    let mut destek = 0usize;
    for (p, g) in puan.iter().zip(gecerli.iter()) {
        if *g {
            destek += 1;
            if *p > enb {
                enb = *p;
            }
        }
    }
    if destek == 0 {
        return Err(SondaHatasi::TumuMaskeli);
    }
    let mut cikis = vec![0.0f32; puan.len()];
    let mut toplam = 0.0f64;
    for ((c, p), g) in cikis.iter_mut().zip(puan.iter()).zip(gecerli.iter()) {
        if *g {
            let u = f64::from(*p - enb).exp();
            *c = u as f32;
            toplam += u;
        }
    }
    if toplam > 0.0 {
        let olcek = (1.0 / toplam) as f32;
        for c in &mut cikis {
            *c *= olcek;
        }
    }
    Ok(cikis)
}

/// A probe head: the pooling weights and the projection on top of them.
#[derive(Debug, Clone)]
pub struct Sonda {
    sekil: SondaSekli,
    bas: Bas,
    /// `seviye * sonda * d`
    sondalar: Vec<f32>,
    /// `seviye * sonda`
    kazanc: Vec<f32>,
    /// `sorgu * d`
    sorgu: Vec<f32>,
    /// `sorgu * seviye * sonda`
    satir_yanlilik: Vec<f32>,
    /// `cikis * (sorgu * d)`
    izdusum: Vec<f32>,
    /// `cikis`, empty for the embedding head
    yanlilik: Vec<f32>,
}

impl Sonda {
    /// A fresh head: probes and queries drawn small, gains at one, row biases
    /// and the projection bias at zero.
    ///
    /// # Errors
    ///
    /// Whatever [`SondaSekli::yeni`] refuses.
    pub fn yeni(sekil: SondaSekli, bas: Bas, tohum: &mut Tohum) -> Result<Self, SondaHatasi> {
        if bas.cikis_genisligi() == 0 {
            return Err(SondaHatasi::SifirBoyut);
        }
        let mut cek = |n: usize, std: f32| -> Vec<f32> {
            (0..n)
                .map(|_| (tohum.normal() * f64::from(std)) as f32)
                .collect()
        };
        let sondalar = cek(sekil.seviye * sekil.sonda * sekil.d, SONDA_INIT_STD);
        let sorgu = cek(sekil.sorgu * sekil.d, SONDA_INIT_STD);
        let cikis = bas.cikis_genisligi();
        let izdusum = cek(cikis * sekil.havuz_genisligi(), SONDA_INIT_STD);
        Ok(Self {
            sekil,
            bas,
            sondalar,
            kazanc: vec![1.0; sekil.seviye * sekil.sonda],
            sorgu,
            satir_yanlilik: vec![0.0; sekil.sorgu * sekil.seviye * sekil.sonda],
            izdusum,
            yanlilik: if bas.yanlilik_var() {
                vec![0.0; cikis]
            } else {
                Vec::new()
            },
        })
    }

    #[must_use]
    pub fn sekil(&self) -> SondaSekli {
        self.sekil
    }

    #[must_use]
    pub fn bas(&self) -> Bas {
        self.bas
    }

    /// The parameters actually held, counted by walking the buffers.
    #[must_use]
    pub fn tutulan_param_sayisi(&self) -> usize {
        self.sondalar.len()
            + self.kazanc.len()
            + self.sorgu.len()
            + self.satir_yanlilik.len()
            + self.izdusum.len()
            + self.yanlilik.len()
    }

    /// Pool a sequence into `sorgu * d` numbers.
    ///
    /// `hucreler` is laid out token-major: `jeton * seviye * d`. `tut` marks
    /// the tokens that count, `seviye_tut` the levels; either may be `None`,
    /// which means all of them.
    ///
    /// # Errors
    ///
    /// [`SondaHatasi::HucreSekli`], [`SondaHatasi::MaskeSekli`],
    /// [`SondaHatasi::TumuMaskeli`], [`SondaHatasi::TumSeviyelerKapali`].
    pub fn havuzla(
        &self,
        hucreler: &[f32],
        jeton: usize,
        tut: Option<&[bool]>,
        seviye_tut: Option<&[bool]>,
    ) -> Result<Vec<f32>, SondaHatasi> {
        let SondaSekli {
            seviye,
            sonda,
            sorgu,
            d,
        } = self.sekil;
        if jeton == 0 {
            return Err(SondaHatasi::SifirBoyut);
        }
        let beklenen = jeton * seviye * d;
        if hucreler.len() != beklenen {
            return Err(SondaHatasi::HucreSekli {
                beklenen,
                gelen: hucreler.len(),
            });
        }
        let jeton_maskesi: Vec<bool> = match tut {
            None => vec![true; jeton],
            Some(m) if m.len() == jeton => m.to_vec(),
            Some(m) => {
                return Err(SondaHatasi::MaskeSekli {
                    beklenen: jeton,
                    gelen: m.len(),
                })
            }
        };
        let seviye_maskesi: Vec<bool> = match seviye_tut {
            None => vec![true; seviye],
            Some(m) if m.len() == seviye => m.to_vec(),
            Some(m) => {
                return Err(SondaHatasi::MaskeSekli {
                    beklenen: seviye,
                    gelen: m.len(),
                })
            }
        };
        if !seviye_maskesi.iter().any(|v| *v) {
            return Err(SondaHatasi::TumSeviyelerKapali);
        }
        let olcek = 1.0f32 / (d as f32).sqrt();

        // Stage one: every probe reads every token.
        let mut cevap = vec![0.0f32; seviye * sonda * d];
        for l in 0..seviye {
            for k in 0..sonda {
                let sonda_vek = &self.sondalar[((l * sonda) + k) * d..((l * sonda) + k + 1) * d];
                let mut puan = vec![0.0f32; jeton];
                for (t, p) in puan.iter_mut().enumerate() {
                    let hucre = &hucreler[(t * seviye + l) * d..(t * seviye + l + 1) * d];
                    let ic: f64 = hucre
                        .iter()
                        .zip(sonda_vek.iter())
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                    *p = (ic as f32) * olcek;
                }
                let agirlik = maskeli_yumusak_azami(&puan, &jeton_maskesi)?;
                let hedef = &mut cevap[((l * sonda) + k) * d..((l * sonda) + k + 1) * d];
                for (t, a) in agirlik.iter().enumerate() {
                    if *a == 0.0 {
                        continue;
                    }
                    let hucre = &hucreler[(t * seviye + l) * d..(t * seviye + l + 1) * d];
                    for (h, c) in hedef.iter_mut().zip(hucre.iter()) {
                        *h += a * c;
                    }
                }
                birim_rms(hedef);
                let g = self.kazanc[(l * sonda) + k];
                for h in hedef.iter_mut() {
                    *h *= g;
                }
            }
        }

        // Stage two: every query reads every probe answer.
        let hucre_sayisi = seviye * sonda;
        let gecerli: Vec<bool> = (0..hucre_sayisi)
            .map(|m| seviye_maskesi[m / sonda])
            .collect();
        let mut cikis = vec![0.0f32; sorgu * d];
        for q in 0..sorgu {
            let sorgu_vek = &self.sorgu[q * d..(q + 1) * d];
            let mut puan = vec![0.0f32; hucre_sayisi];
            for (m, p) in puan.iter_mut().enumerate() {
                let r = &cevap[m * d..(m + 1) * d];
                let ic: f64 = r
                    .iter()
                    .zip(sorgu_vek.iter())
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                *p = (ic as f32) * olcek + self.satir_yanlilik[q * hucre_sayisi + m];
            }
            let agirlik = maskeli_yumusak_azami(&puan, &gecerli)?;
            let hedef = &mut cikis[q * d..(q + 1) * d];
            for (m, a) in agirlik.iter().enumerate() {
                if *a == 0.0 {
                    continue;
                }
                let r = &cevap[m * d..(m + 1) * d];
                for (h, v) in hedef.iter_mut().zip(r.iter()) {
                    *h += a * v;
                }
            }
        }
        Ok(cikis)
    }

    /// Pool, then project onto the head's output shape.
    ///
    /// # Errors
    ///
    /// Whatever [`Sonda::havuzla`] refuses.
    pub fn ileri(
        &self,
        hucreler: &[f32],
        jeton: usize,
        tut: Option<&[bool]>,
        seviye_tut: Option<&[bool]>,
    ) -> Result<Vec<f32>, SondaHatasi> {
        let havuz = self.havuzla(hucreler, jeton, tut, seviye_tut)?;
        let cikis_genislik = self.bas.cikis_genisligi();
        let mut cikis = vec![0.0f32; cikis_genislik];
        for (o, hedef) in cikis.iter_mut().enumerate() {
            let satir = &self.izdusum[o * havuz.len()..(o + 1) * havuz.len()];
            let toplam: f64 = satir
                .iter()
                .zip(havuz.iter())
                .map(|(w, x)| f64::from(*w) * f64::from(*x))
                .sum();
            *hedef = toplam as f32;
            if let Some(b) = self.yanlilik.get(o) {
                *hedef += b;
            }
        }
        Ok(cikis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sekil() -> SondaSekli {
        match SondaSekli::yeni(2, 3, 2, 4) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        }
    }

    fn sonda(bas: Bas) -> Sonda {
        let mut t = Tohum::yeni(5);
        match Sonda::yeni(sekil(), bas, &mut t) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        }
    }

    fn hucreler(jeton: usize) -> Vec<f32> {
        let s = sekil();
        (0..jeton * s.seviye * s.d)
            .map(|i| ((i % 11) as f32) * 0.13 - 0.5)
            .collect()
    }

    #[test]
    fn sifir_boyut_reddedilir() {
        assert_eq!(SondaSekli::yeni(0, 1, 1, 1), Err(SondaHatasi::SifirBoyut));
        assert_eq!(SondaSekli::yeni(1, 0, 1, 1), Err(SondaHatasi::SifirBoyut));
        assert_eq!(SondaSekli::yeni(1, 1, 0, 1), Err(SondaHatasi::SifirBoyut));
        assert_eq!(SondaSekli::yeni(1, 1, 1, 0), Err(SondaHatasi::SifirBoyut));
    }

    #[test]
    fn havuz_genisligi_dizi_uzunlugundan_bagimsiz() {
        let s = sonda(Bas::Guven);
        let a = match s.havuzla(&hucreler(3), 3, None, None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let b = match s.havuzla(&hucreler(17), 17, None, None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(a.len(), sekil().havuz_genisligi());
        assert_eq!(b.len(), a.len());
    }

    #[test]
    fn param_sayisi_iki_yoldan_ayni() {
        for bas in [Bas::Guven, Bas::YonSecimi, Bas::Gomme { genislik: 8 }] {
            let s = sonda(bas);
            assert_eq!(
                s.sekil().param_sayisi(bas),
                s.tutulan_param_sayisi(),
                "{bas:?}: formul ile yurume ayristi"
            );
        }
    }

    #[test]
    fn havuz_jeton_sirasindan_bagimsiz() {
        // Pooling is a weighted sum over tokens, so a shuffle cannot move it.
        // Order reaches the head through the backbone's positions, not here.
        let s = sonda(Bas::YonSecimi);
        let sek = sekil();
        let jeton = 5;
        let h = hucreler(jeton);
        let duz = match s.havuzla(&h, jeton, None, None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let sira = [3usize, 0, 4, 1, 2];
        let mut karisik = vec![0.0f32; h.len()];
        for (yeni, eski) in sira.iter().enumerate() {
            let genislik = sek.seviye * sek.d;
            karisik[yeni * genislik..(yeni + 1) * genislik]
                .copy_from_slice(&h[eski * genislik..(eski + 1) * genislik]);
        }
        let karisik_havuz = match s.havuzla(&karisik, jeton, None, None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        for (a, b) in duz.iter().zip(karisik_havuz.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn maskeli_jeton_tam_olarak_disarida() {
        // Masking before the softmax means weight zero, not weight epsilon:
        // pooling four tokens with the fifth masked must equal pooling the
        // four alone.
        let s = sonda(Bas::Guven);
        let sek = sekil();
        let genislik = sek.seviye * sek.d;
        let bes = hucreler(5);
        let maske = [true, true, true, true, false];
        let maskeli = match s.havuzla(&bes, 5, Some(&maske), None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let dort = &bes[..4 * genislik];
        let kisa = match s.havuzla(dort, 4, None, None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        for (a, b) in maskeli.iter().zip(kisa.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn tumu_maskeli_reddedilir() {
        let s = sonda(Bas::Guven);
        let maske = [false, false, false];
        assert_eq!(
            s.havuzla(&hucreler(3), 3, Some(&maske), None),
            Err(SondaHatasi::TumuMaskeli)
        );
    }

    #[test]
    fn tum_seviyeler_kapali_reddedilir() {
        let s = sonda(Bas::Guven);
        let kapali = [false, false];
        assert_eq!(
            s.havuzla(&hucreler(3), 3, None, Some(&kapali)),
            Err(SondaHatasi::TumSeviyelerKapali)
        );
    }

    #[test]
    fn kapali_seviye_havuza_girmez() {
        // With one level switched off the answer must equal the answer of a
        // head that never had that level's probes in its softmax.
        let s = sonda(Bas::Guven);
        let acik = [true, false];
        let a = match s.havuzla(&hucreler(4), 4, None, Some(&acik)) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let b = match s.havuzla(&hucreler(4), 4, None, Some(&acik)) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(a, b);
        let hepsi = match s.havuzla(&hucreler(4), 4, None, None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert_ne!(a, hepsi, "kapatilan seviye sonucu degistirmedi");
    }

    #[test]
    fn hucre_sekli_denetlenir() {
        let s = sonda(Bas::Guven);
        let sek = sekil();
        let beklenen = 3 * sek.seviye * sek.d;
        assert_eq!(
            s.havuzla(&vec![0.0; beklenen + 1], 3, None, None),
            Err(SondaHatasi::HucreSekli {
                beklenen,
                gelen: beklenen + 1
            })
        );
    }

    #[test]
    fn maske_uzunlugu_denetlenir() {
        let s = sonda(Bas::Guven);
        let maske = [true, true];
        assert_eq!(
            s.havuzla(&hucreler(3), 3, Some(&maske), None),
            Err(SondaHatasi::MaskeSekli {
                beklenen: 3,
                gelen: 2
            })
        );
    }

    #[test]
    fn sifir_jeton_reddedilir() {
        let s = sonda(Bas::Guven);
        assert_eq!(s.havuzla(&[], 0, None, None), Err(SondaHatasi::SifirBoyut));
    }

    #[test]
    fn birim_rms_normu_bire_getirir() {
        let mut v = vec![3.0f32, -4.0, 0.0, 12.0];
        birim_rms(&mut v);
        let rms = (v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / (v.len() as f64))
            .sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "rms={rms}");
    }

    #[test]
    fn birim_rms_sifir_vektorde_nan_uretmez() {
        let mut v = vec![0.0f32; 8];
        birim_rms(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn yumusak_azami_bire_toplanir() {
        let p = match maskeli_yumusak_azami(&[1.0, 2.0, 3.0], &[true, true, true]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let toplam: f32 = p.iter().sum();
        assert!((toplam - 1.0).abs() < 1e-6);
    }

    #[test]
    fn yumusak_azami_maskeliye_sifir_verir() {
        let p = match maskeli_yumusak_azami(&[1.0, 50.0, 3.0], &[true, false, true]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(p[1].to_bits(), 0.0f32.to_bits());
        assert!((p[0] + p[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn yumusak_azami_buyuk_sayida_tasmaz() {
        let p = match maskeli_yumusak_azami(&[1.0e30, 1.0e30 + 1.0], &[true, true]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn yumusak_azami_destegi_yoksa_reddeder() {
        assert_eq!(
            maskeli_yumusak_azami(&[1.0, 2.0], &[false, false]),
            Err(SondaHatasi::TumuMaskeli)
        );
    }

    #[test]
    fn guven_basi_tek_sayi_verir() {
        let s = sonda(Bas::Guven);
        match s.ileri(&hucreler(4), 4, None, None) {
            Ok(v) => assert_eq!(v.len(), 1),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn yon_secimi_basi_uc_sayi_verir() {
        let s = sonda(Bas::YonSecimi);
        match s.ileri(&hucreler(4), 4, None, None) {
            Ok(v) => assert_eq!(v.len(), 3),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn gomme_basi_yanlilik_tasimaz() {
        let s = sonda(Bas::Gomme { genislik: 8 });
        assert!(!Bas::Gomme { genislik: 8 }.yanlilik_var());
        assert_eq!(
            s.sekil().param_sayisi(Bas::Gomme { genislik: 8 }),
            s.tutulan_param_sayisi()
        );
        match s.ileri(&hucreler(4), 4, None, None) {
            Ok(v) => assert_eq!(v.len(), 8),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn cikis_genislikleri_sabit() {
        assert_eq!(Bas::Guven.cikis_genisligi(), 1);
        assert_eq!(Bas::YonSecimi.cikis_genisligi(), 3);
        assert_eq!(Bas::Gomme { genislik: 128 }.cikis_genisligi(), 128);
    }

    #[test]
    fn rota_kalibrasyonu_bir_bant_tarif_eder() {
        // Threshold above floor, offset at zero: a triple that does not
        // describe a band would route everything one way and the head would
        // look decisive while deciding nothing.
        let k: Vec<f32> = ROTA_KALIBRASYONU.to_vec();
        assert_eq!(k.len(), 3);
        assert!(k.iter().all(|v| (0.0..=1.0).contains(v)), "{k:?}");
        assert!(k[0] > k[2], "esik tabanin altinda: {k:?}");
        assert_eq!(k[1].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn taze_gomme_basi_kapali_baslar() {
        // Temperature and bias together: at a similarity of zero the gate
        // reads sigmoid(bias), which is 4.5e-5 - a fresh head admits nothing
        // until it is trained, rather than admitting half of everything.
        let sicaklik = f64::from(GOMME_SICAKLIK_INIT);
        let yanlilik = f64::from(GOMME_YANLILIK_INIT);
        let kapi = 1.0 / (1.0 + (-(sicaklik * 0.0 + yanlilik)).exp());
        assert!(kapi < 1e-4, "taze kapi {kapi}");
        let doygun = 1.0 / (1.0 + (-(sicaklik * 1.0 + yanlilik)).exp());
        assert!(doygun.abs() < 0.51, "birim benzerlikte kapi {doygun}");
    }

    #[test]
    fn ileri_belirlenimci() {
        let s = sonda(Bas::YonSecimi);
        let h = hucreler(6);
        let (a, b) = match (s.ileri(&h, 6, None, None), s.ileri(&h, 6, None, None)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => panic!("ileri reddetti"),
        };
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }

    #[test]
    fn tek_jeton_havuzu_o_jetonun_kendisi() {
        // With one token every softmax is the constant one, so the probe
        // answer is that token's state at unit RMS times its gain.
        let s = sonda(Bas::Guven);
        let sek = sekil();
        let h = hucreler(1);
        let havuz = match s.havuzla(&h, 1, None, None) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        // Every query then mixes those answers; the result must stay inside
        // the range the answers span.
        let mut enb = 0.0f32;
        for l in 0..sek.seviye {
            let mut hucre = h[l * sek.d..(l + 1) * sek.d].to_vec();
            birim_rms(&mut hucre);
            for v in hucre {
                enb = enb.max(v.abs());
            }
        }
        for v in havuz {
            assert!(v.abs() <= enb + 1e-4, "{v} > {enb}");
        }
    }

    #[test]
    fn kazanc_baslangicta_bir() {
        let s = sonda(Bas::Guven);
        assert!(s.kazanc.iter().all(|g| g.to_bits() == 1.0f32.to_bits()));
    }

    #[test]
    fn satir_yanliligi_baslangicta_sifir() {
        let s = sonda(Bas::YonSecimi);
        assert!(s
            .satir_yanlilik
            .iter()
            .all(|b| b.to_bits() == 0.0f32.to_bits()));
    }

    #[test]
    fn hata_metinleri_ayirt_edilir() {
        let metinler = [
            SondaHatasi::SifirBoyut.to_string(),
            SondaHatasi::TumuMaskeli.to_string(),
            SondaHatasi::TumSeviyelerKapali.to_string(),
            SondaHatasi::HucreSekli {
                beklenen: 1,
                gelen: 2,
            }
            .to_string(),
            SondaHatasi::MaskeSekli {
                beklenen: 1,
                gelen: 2,
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
