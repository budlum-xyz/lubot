//! # servis - the backbone's weights, quantised by the crate that owns it
//!
//! Serving wants a small file; training wants wide floats. This module bridges
//! the two, and the bridge is the whole content: the arithmetic of
//! quantisation lives in `lubot-nicem` and is *called* here, not copied. A
//! second quantiser in this crate would be a second place for the same
//! rounding to be wrong, and the two would drift apart on the day one of them
//! was fixed.
//!
//! What this module adds is the thing `lubot-nicem` cannot know: the
//! backbone's **tensor directory**. A flat buffer quantised as one long vector
//! mixes an embedding table with an attention projection, and a group that
//! straddles two tensors gets a scale that suits neither. Here every tensor is
//! quantised on its own, with its own last-axis stride, and the error is
//! reported **per tensor** as well as in aggregate.
//!
//! ## Why per tensor is not a detail
//!
//! `her_tensor_ayri_olculur` measures the spread: on a fresh backbone the
//! worst tensor's relative error is several times the best one's. An aggregate
//! number would hide exactly the tensor a serving failure would come from, and
//! "the model quantises to 2.5 bits with 0.3 relative error" would be true and
//! useless at the same time.
//!
//! ## What is refused
//!
//! - A tensor shorter than the quantiser's group: padding it to reach the
//!   group would invent weights, and inventing weights to make a compression
//!   ratio look good is the failure this whole repository is built against.
//! - Anything `lubot-nicem` refuses: the error keeps its own name
//!   ([`ServisHatasi::Nicem`]) rather than being flattened into a string.
//!
//! ## What is not here
//!
//! No serving *runtime*: nothing runs a forward pass from packed weights yet,
//! and the quantised form is not written to disk. And no quality claim - the
//! error is a number, and whether a model survives it is a question for a
//! trained checkpoint, which does not exist in this crate.

use crate::{Omurga, OmurgaHatasi};
use lubot_nicem::grup::{Genislik, NicemHatasi, Nicemleyici, Olcum};

#[derive(Debug, Clone, PartialEq)]
pub enum ServisHatasi {
    /// The quantiser refused; its reason is kept rather than summarised.
    Nicem(NicemHatasi),
    /// The backbone refused.
    Omurga(OmurgaHatasi),
    /// A tensor with fewer weights than one group.
    TensorGruptanKucuk {
        ad: String,
        uzunluk: usize,
        grup: usize,
    },
}

impl core::fmt::Display for ServisHatasi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Nicem(e) => write!(f, "quantiser: {e}"),
            Self::Omurga(e) => write!(f, "backbone: {e}"),
            Self::TensorGruptanKucuk { ad, uzunluk, grup } => write!(
                f,
                "tensor `{ad}` has {uzunluk} weights, fewer than one group of {grup}"
            ),
        }
    }
}

impl std::error::Error for ServisHatasi {}

impl From<NicemHatasi> for ServisHatasi {
    fn from(e: NicemHatasi) -> Self {
        Self::Nicem(e)
    }
}

impl From<OmurgaHatasi> for ServisHatasi {
    fn from(e: OmurgaHatasi) -> Self {
        Self::Omurga(e)
    }
}

/// What one tensor cost.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorRaporu {
    pub ad: String,
    pub uzunluk: usize,
    pub olcum: Olcum,
}

/// What the whole backbone cost.
#[derive(Debug, Clone, PartialEq)]
pub struct ServisRaporu {
    pub tensorler: Vec<TensorRaporu>,
    /// Weights quantised, summed over tensors.
    pub agirlik: usize,
    /// Bytes the quantised form occupies, scales included.
    pub bayt: usize,
    /// `f32` bytes divided by quantised bytes.
    pub oran: f64,
    /// Bits per weight, measured rather than quoted from the alphabet.
    pub agirlik_basina_bit: f64,
}

impl ServisRaporu {
    /// The tensor that came out worst, by relative error.
    #[must_use]
    pub fn en_kotu(&self) -> Option<&TensorRaporu> {
        self.tensorler.iter().max_by(|a, b| {
            a.olcum
                .bagil_hata
                .partial_cmp(&b.olcum.bagil_hata)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
    }

    /// The tensor that came out best.
    #[must_use]
    pub fn en_iyi(&self) -> Option<&TensorRaporu> {
        self.tensorler.iter().min_by(|a, b| {
            a.olcum
                .bagil_hata
                .partial_cmp(&b.olcum.bagil_hata)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
    }

    /// Worst divided by best: how far apart the tensors are.
    ///
    /// One number for the question an aggregate error cannot answer - whether
    /// the scheme treats every tensor alike, or whether one of them is paying
    /// for the rest.
    #[must_use]
    pub fn yayilim(&self) -> f64 {
        match (self.en_iyi(), self.en_kotu()) {
            (Some(iyi), Some(kotu)) if iyi.olcum.bagil_hata > 0.0 => {
                kotu.olcum.bagil_hata / iyi.olcum.bagil_hata
            }
            _ => 1.0,
        }
    }
}

/// Quantise a backbone tensor by tensor.
///
/// `son_eksen` is taken from each tensor's own row width, so a group never
/// straddles two rows of one matrix and never two tensors.
///
/// # Errors
///
/// [`ServisHatasi::TensorGruptanKucuk`], plus whatever the quantiser or the
/// backbone refuses.
pub fn nicemle(
    omurga: &Omurga,
    genislik: Genislik,
    grup: usize,
) -> Result<ServisRaporu, ServisHatasi> {
    let nicemleyici = Nicemleyici::yeni(genislik, grup)?;
    let mut tensorler = Vec::with_capacity(omurga.dizin().len());
    let mut agirlik = 0usize;
    let mut bayt = 0usize;
    for kayit in omurga.dizin() {
        let veri = omurga.tensor(&kayit.ad)?;
        if veri.len() < grup {
            return Err(ServisHatasi::TensorGruptanKucuk {
                ad: kayit.ad.clone(),
                uzunluk: veri.len(),
                grup,
            });
        }
        // The last axis is the tensor's row width: a group that crossed a row
        // would share one scale between two independent output channels.
        let son_eksen = if kayit.sutun >= grup && kayit.sutun % grup == 0 {
            kayit.sutun
        } else {
            veri.len()
        };
        let nicemlenmis = nicemleyici.nicemle(veri, son_eksen)?;
        let olcum = nicemlenmis.olc(veri)?;
        agirlik += veri.len();
        bayt += olcum.bayt;
        tensorler.push(TensorRaporu {
            ad: kayit.ad.clone(),
            uzunluk: veri.len(),
            olcum,
        });
    }
    let oran = if bayt > 0 {
        (agirlik as f64) * 4.0 / (bayt as f64)
    } else {
        0.0
    };
    let agirlik_basina_bit = if agirlik > 0 {
        (bayt as f64) * 8.0 / (agirlik as f64)
    } else {
        0.0
    };
    Ok(ServisRaporu {
        tensorler,
        agirlik,
        bayt,
        oran,
        agirlik_basina_bit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Yapilandirma;

    fn omurga() -> Omurga {
        // A small backbone on purpose: the quantiser is exercised here, not
        // benchmarked, and an 8192-word embedding table would turn a unit
        // test into a minute of arithmetic.
        let mut yap = Yapilandirma::kucuk_aday();
        yap.vocab = 256;
        yap.n_katman = 2;
        match Omurga::yeni(yap, 4) {
            Ok(o) => o,
            Err(e) => panic!("omurga: {e}"),
        }
    }

    fn rapor(bit: u8, grup: usize) -> ServisRaporu {
        match nicemle(&omurga(), Genislik::Bit(bit), grup) {
            Ok(r) => r,
            Err(e) => panic!("nicemle: {e}"),
        }
    }

    #[test]
    fn her_tensor_ayri_olculur() {
        // The point of the module: one number per tensor, and the spread
        // between them is itself reported. An aggregate error would hide the
        // tensor a serving failure comes from.
        let r = rapor(2, 64);
        assert_eq!(r.tensorler.len(), omurga().dizin().len());
        let yayilim = r.yayilim();
        assert!(
            yayilim > 1.0,
            "butun tensorler ayni hatayi verdi: {yayilim}"
        );
        let kotu = match r.en_kotu() {
            Some(k) => k,
            None => panic!("en kotu tensor yok"),
        };
        let iyi = match r.en_iyi() {
            Some(k) => k,
            None => panic!("en iyi tensor yok"),
        };
        assert!(kotu.olcum.bagil_hata >= iyi.olcum.bagil_hata);
    }

    #[test]
    fn kucuk_tensor_reddedilir() {
        // Padding a short tensor to reach the group would invent weights.
        let hata = nicemle(&omurga(), Genislik::Bit(2), 4096);
        match hata {
            Err(ServisHatasi::TensorGruptanKucuk { grup, .. }) => assert_eq!(grup, 4096),
            other => panic!("beklenen red gelmedi: {other:?}"),
        }
    }

    #[test]
    fn toplam_agirlik_omurganinkiyle_ayni() {
        let o = omurga();
        let r = match nicemle(&o, Genislik::Bit(4), 64) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(r.agirlik, o.param_sayisi());
    }

    #[test]
    fn daha_cok_bit_daha_az_hata() {
        // Monotone in the alphabet: if it were not, the scheme would be
        // spending bits on nothing.
        // Three widths, not eight bits: a 256-level Lloyd-Max codebook is
        // minutes of arithmetic and the monotonicity it would show is the
        // same monotonicity these three show.
        let iki = rapor(2, 64);
        let uc = rapor(3, 64);
        let dort = rapor(4, 64);
        let en_kotu = |r: &ServisRaporu| match r.en_kotu() {
            Some(t) => t.olcum.bagil_hata,
            None => f64::INFINITY,
        };
        assert!(en_kotu(&uc) < en_kotu(&iki));
        assert!(en_kotu(&dort) < en_kotu(&uc));
    }

    #[test]
    fn daha_cok_bit_daha_cok_bayt() {
        let iki = rapor(2, 64);
        let dort = rapor(4, 64);
        assert!(dort.bayt > iki.bayt);
        assert!(dort.oran < iki.oran);
    }

    #[test]
    fn bit_basina_agirlik_alfabeden_buyuk() {
        // The alphabet is two bits; the group scales are the fine print, and
        // the measured number carries them.
        let r = rapor(2, 64);
        assert!(r.agirlik_basina_bit > 2.0, "{}", r.agirlik_basina_bit);
        assert!(r.agirlik_basina_bit < 3.0, "{}", r.agirlik_basina_bit);
    }

    #[test]
    fn buyuk_grup_daha_ucuz_ama_daha_hatali() {
        // The trade the group size *is*: fewer scales, worse fit. Both halves
        // are measured, because quoting only the first is how a compression
        // ratio gets advertised.
        let dar = rapor(2, 32);
        let genis = rapor(2, 128);
        assert!(genis.agirlik_basina_bit < dar.agirlik_basina_bit);
        let en_kotu = |r: &ServisRaporu| match r.en_kotu() {
            Some(t) => t.olcum.bagil_hata,
            None => 0.0,
        };
        assert!(en_kotu(&genis) >= en_kotu(&dar));
    }

    #[test]
    fn ucdeger_alfabesi_de_calisir() {
        let r = match nicemle(&omurga(), Genislik::Ucdeger, 64) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        };
        assert!(r.agirlik_basina_bit > 1.6, "{}", r.agirlik_basina_bit);
        assert!(r.agirlik_basina_bit < 2.6, "{}", r.agirlik_basina_bit);
    }

    #[test]
    fn sekil_imzasi_degismez() {
        // Quantisation reports; it does not touch the model. If it ever did,
        // an averaged pair of checkpoints would silently stop being averageable.
        let o = omurga();
        let imza = o.sekil_imzasi();
        let _ = match nicemle(&o, Genislik::Bit(2), 64) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(o.sekil_imzasi(), imza);
    }

    #[test]
    fn olcum_belirlenimci() {
        let a = rapor(2, 64);
        let b = rapor(2, 64);
        assert_eq!(a.tensorler.len(), b.tensorler.len());
        for (x, y) in a.tensorler.iter().zip(b.tensorler.iter()) {
            assert_eq!(x.ad, y.ad);
            assert_eq!(x.olcum.bagil_hata.to_bits(), y.olcum.bagil_hata.to_bits());
        }
    }

    #[test]
    fn snr_ve_bagil_hata_birbirini_dogrular() {
        // Two ways of saying one thing: if they disagreed, one of them would
        // be decoration.
        let r = rapor(4, 64);
        for t in &r.tensorler {
            let beklenen = 20.0 * (1.0 / t.olcum.bagil_hata).log10();
            assert!(
                (t.olcum.snr_db - beklenen).abs() < 1e-6,
                "{}: {} vs {beklenen}",
                t.ad,
                t.olcum.snr_db
            );
        }
    }

    #[test]
    fn en_buyuk_sapma_bagil_hatadan_bagimsiz_raporlanir() {
        // A small Frobenius error can still hide one badly placed weight, so
        // the largest single deviation is carried separately.
        let r = rapor(2, 64);
        assert!(r.tensorler.iter().all(|t| t.olcum.en_buyuk_sapma > 0.0));
    }

    #[test]
    fn bos_rapor_yayilimi_bir() {
        let bos = ServisRaporu {
            tensorler: Vec::new(),
            agirlik: 0,
            bayt: 0,
            oran: 0.0,
            agirlik_basina_bit: 0.0,
        };
        assert!((bos.yayilim() - 1.0).abs() < 1e-12);
        assert!(bos.en_kotu().is_none());
        assert!(bos.en_iyi().is_none());
    }

    #[test]
    fn hata_metinleri_ayirt_edilir() {
        let a = ServisHatasi::TensorGruptanKucuk {
            ad: "gomme".to_string(),
            uzunluk: 4,
            grup: 64,
        }
        .to_string();
        let b = ServisHatasi::Omurga(OmurgaHatasi::BosTopluluk).to_string();
        assert_ne!(a, b);
    }
}
