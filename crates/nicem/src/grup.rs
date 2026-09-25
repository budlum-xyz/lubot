//! The quantiser itself: rotate a group, keep its norm, spend two bits on the
//! direction, and measure what that cost.
//!
//! # The pipeline, in one place
//!
//! For each group of `g` weights along the reduction axis:
//!
//! 1. **Rotate.** [`crate::hadamard::wht`] spreads outliers across the group.
//!    Orthonormal, so the norm is unchanged and the step is reversible.
//! 2. **Split scale from direction.** The group becomes one number - its
//!    Euclidean norm, stored as `f16` - and a unit vector. This is the choice
//!    that makes two bits viable: absmax scaling spends its range on the single
//!    largest coordinate, norm scaling spends it on the group's energy, and
//!    after a rotation those are nearly the same thing anyway.
//! 3. **Quantise the direction.** Each coordinate of the unit vector goes to
//!    the nearest level of the codebook from [`crate::kodkitabi`], which was
//!    solved for exactly this distribution.
//! 4. **Pack.** [`crate::paket`] writes the indices with no slack.
//!
//! Reading runs the same four steps backwards, and because the rotation is its
//! own inverse there is no second table and no second code path.
//!
//! # What is measured, and what is therefore not claimed
//!
//! [`Nicemlenmis::olc`] compares the reconstruction against the original and
//! reports relative Frobenius error, signal-to-noise ratio, bits per weight and
//! the compression ratio. Those four numbers are computed from the tensor in
//! front of it. Nothing in this crate says a quantised model is "as good as"
//! anything: that is an evaluation question, it belongs to the exam set, and a
//! compression ratio is not evidence about it.
//!
//! The one prediction that *is* available is the codebook's own distortion
//! ([`crate::kodkitabi::Kodkitabi::bozulma`]), computed for a Gaussian source.
//! If a real tensor comes out far worse than that, the rotation did not
//! Gaussianise it, and [`Nicemlenmis::olc`] is where that shows up.
//!
//! # Why the whole tensor is never reconstructed on the device
//!
//! [`Nicemlenmis::carp`] multiplies by a vector one group at a time, so the
//! largest live buffer is `g` floats regardless of how many parameters the
//! tensor has. [`Nicemlenmis::en_buyuk_calisma_alani`] returns that number and
//! a test pins it. This is the entire mechanism behind "a large model on a
//! small device": the parameters live packed in mapped bytes, and only one
//! group of them is ever unpacked.

use crate::hadamard::{wht, HadamardHatasi};
use crate::kodkitabi::{self, Kodkitabi, KodkitabiHatasi};
use crate::paket::{self, PaketHatasi};
use crate::yarim;
use std::fmt;

/// Groups smaller than this cannot amortise their own `f16` scale: at `g = 8`
/// the scale alone costs two bits per weight.
pub const EN_KUCUK_GRUP: usize = 16;

/// Alphabet width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genislik {
    /// Three levels, packed base-3 at 1.6 bits per weight.
    Ucdeger,
    /// `2^bit` levels, `bit` in `1..=8`.
    Bit(u8),
}

impl Genislik {
    /// Number of levels in the alphabet.
    #[must_use]
    pub fn seviye_sayisi(self) -> usize {
        match self {
            Self::Ucdeger => 3,
            Self::Bit(b) => 1usize << b,
        }
    }

    /// Bits per weight including the group scale.
    #[must_use]
    pub fn agirlik_basina_bit(self, grup: usize) -> f64 {
        match self {
            Self::Ucdeger => paket::agirlik_basina_bit_ucdeger(grup),
            Self::Bit(b) => paket::agirlik_basina_bit(b, grup),
        }
    }

    /// A short, stable name for the container's tensor directory.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::Ucdeger => "ucdeger",
            Self::Bit(1) => "b1",
            Self::Bit(2) => "b2",
            Self::Bit(3) => "b3",
            Self::Bit(4) => "b4",
            Self::Bit(5) => "b5",
            Self::Bit(6) => "b6",
            Self::Bit(7) => "b7",
            Self::Bit(_) => "b8",
        }
    }
}

/// Why quantising or reading was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum NicemHatasi {
    /// The group size is not a power of two, so there is no fast rotation.
    GrupIkininKuvvetiDegil { grup: usize },
    /// The group size is below [`EN_KUCUK_GRUP`].
    GrupCokKucuk { grup: usize },
    /// The tensor length is not a multiple of its declared last axis.
    EksenUyumsuz { uzunluk: usize, eksen: usize },
    /// The tensor is empty.
    Bos,
    /// A non-finite weight. A NaN cannot be quantised to a level and would
    /// poison its whole group's norm; the refusal names the position so the
    /// upstream defect is findable.
    SonluDegil { konum: usize },
    /// The codebook solve failed.
    Kodkitabi(KodkitabiHatasi),
    /// The packing refused.
    Paket(PaketHatasi),
    /// The rotation refused.
    Hadamard(HadamardHatasi),
    /// Stored payload does not match the declared shape.
    BoyutUyumsuz { beklenen: usize, var: usize },
}

impl fmt::Display for NicemHatasi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GrupIkininKuvvetiDegil { grup } => {
                write!(f, "grup {grup} ikinin kuvveti degil")
            }
            Self::GrupCokKucuk { grup } => write!(
                f,
                "grup {grup} cok kucuk: olcek maliyeti asgari {EN_KUCUK_GRUP} ile amorti olur"
            ),
            Self::EksenUyumsuz { uzunluk, eksen } => {
                write!(f, "uzunluk {uzunluk} son eksen {eksen} ile bolunmuyor")
            }
            Self::Bos => write!(f, "bos tensor nicemlenmez"),
            Self::SonluDegil { konum } => write!(f, "konum {konum} sonlu degil"),
            Self::Kodkitabi(e) => write!(f, "kodkitabi: {e}"),
            Self::Paket(e) => write!(f, "paket: {e}"),
            Self::Hadamard(e) => write!(f, "donusum: {e}"),
            Self::BoyutUyumsuz { beklenen, var } => {
                write!(f, "yuk boyutu uyumsuz: {beklenen} beklendi, {var} var")
            }
        }
    }
}

impl std::error::Error for NicemHatasi {}

impl From<KodkitabiHatasi> for NicemHatasi {
    fn from(e: KodkitabiHatasi) -> Self {
        Self::Kodkitabi(e)
    }
}
impl From<PaketHatasi> for NicemHatasi {
    fn from(e: PaketHatasi) -> Self {
        Self::Paket(e)
    }
}
impl From<HadamardHatasi> for NicemHatasi {
    fn from(e: HadamardHatasi) -> Self {
        Self::Hadamard(e)
    }
}

/// A configured quantiser. Holds the solved codebook so a tensor-by-tensor pass
/// over a whole model solves it once.
#[derive(Debug, Clone)]
pub struct Nicemleyici {
    genislik: Genislik,
    grup: usize,
    kitap: Kodkitabi,
    seviyeler: Vec<f32>,
}

impl Nicemleyici {
    /// Build a quantiser and solve its codebook.
    ///
    /// # Errors
    ///
    /// [`NicemHatasi::GrupIkininKuvvetiDegil`] or [`NicemHatasi::GrupCokKucuk`]
    /// for an unusable group size, and [`NicemHatasi::Kodkitabi`] if the solve
    /// refuses the width.
    pub fn yeni(genislik: Genislik, grup: usize) -> Result<Self, NicemHatasi> {
        if !grup.is_power_of_two() {
            return Err(NicemHatasi::GrupIkininKuvvetiDegil { grup });
        }
        if grup < EN_KUCUK_GRUP {
            return Err(NicemHatasi::GrupCokKucuk { grup });
        }
        let kitap = match genislik {
            Genislik::Ucdeger => kodkitabi::coz_ucdeger()?,
            Genislik::Bit(b) => kodkitabi::coz(b)?,
        };
        let seviyeler = kitap.birim_norm_icin(grup);
        Ok(Self {
            genislik,
            grup,
            kitap,
            seviyeler,
        })
    }

    /// The solved codebook, for reporting.
    #[must_use]
    pub fn kitap(&self) -> &Kodkitabi {
        &self.kitap
    }

    /// Group size.
    #[must_use]
    pub fn grup(&self) -> usize {
        self.grup
    }

    /// Alphabet width.
    #[must_use]
    pub fn genislik(&self) -> Genislik {
        self.genislik
    }

    /// Quantise a row-major tensor whose reduction axis is the last one.
    ///
    /// The last axis is padded with zeros up to a multiple of the group size.
    /// Padding with zeros rather than with edge values is deliberate: a zero
    /// contributes nothing to the group norm, so the padding cannot move the
    /// scale of the real weights beside it.
    ///
    /// # Errors
    ///
    /// [`NicemHatasi::Bos`] for an empty tensor, [`NicemHatasi::EksenUyumsuz`]
    /// if the length is not a multiple of `son_eksen`, and
    /// [`NicemHatasi::SonluDegil`] naming the first non-finite weight.
    pub fn nicemle(&self, agirlik: &[f32], son_eksen: usize) -> Result<Nicemlenmis, NicemHatasi> {
        if agirlik.is_empty() || son_eksen == 0 {
            return Err(NicemHatasi::Bos);
        }
        if agirlik.len() % son_eksen != 0 {
            return Err(NicemHatasi::EksenUyumsuz {
                uzunluk: agirlik.len(),
                eksen: son_eksen,
            });
        }
        for (i, v) in agirlik.iter().enumerate() {
            if !v.is_finite() {
                return Err(NicemHatasi::SonluDegil { konum: i });
            }
        }

        let satir = agirlik.len() / son_eksen;
        let grup_basina = son_eksen.div_ceil(self.grup);
        let toplam_grup = satir * grup_basina;

        let mut olcekler = Vec::with_capacity(toplam_grup);
        let mut indeksler = Vec::with_capacity(toplam_grup * self.grup);
        let mut tampon = vec![0.0f32; self.grup];

        for r in 0..satir {
            let temel = r * son_eksen;
            for g in 0..grup_basina {
                let bas = g * self.grup;
                let son = ((g + 1) * self.grup).min(son_eksen);
                tampon.fill(0.0);
                tampon[..son - bas].copy_from_slice(&agirlik[temel + bas..temel + son]);
                wht(&mut tampon)?;
                let norm = (tampon
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>())
                .sqrt();
                // The scale goes through f16 *before* the direction is
                // quantised, so the encoder optimises against the number the
                // reader will actually see rather than against an f32 it will
                // never hold.
                #[allow(clippy::cast_possible_truncation)]
                let norm_yarim = yarim::f32_to_yarim(norm as f32);
                let norm_geri = yarim::yarim_to_f32(norm_yarim);
                olcekler.push(norm_yarim);
                if norm_geri <= 0.0 {
                    // An all-zero group: every index is the level nearest zero,
                    // and the scale is zero, so the reconstruction is exact.
                    let sifir = Kodkitabi::en_yakin(&self.seviyeler, 0.0);
                    #[allow(clippy::cast_possible_truncation)]
                    indeksler.extend(std::iter::repeat_n(sifir as u8, self.grup));
                    continue;
                }
                for v in &tampon {
                    let birim = v / norm_geri;
                    #[allow(clippy::cast_possible_truncation)]
                    indeksler.push(Kodkitabi::en_yakin(&self.seviyeler, birim) as u8);
                }
            }
        }

        let yuk = match self.genislik {
            Genislik::Ucdeger => paket::paketle_ucdeger(&indeksler)?,
            Genislik::Bit(b) => paket::paketle(&indeksler, b)?,
        };

        Ok(Nicemlenmis {
            genislik: self.genislik,
            grup: self.grup,
            son_eksen,
            uzunluk: agirlik.len(),
            olcekler,
            yuk,
            seviyeler: self.seviyeler.clone(),
        })
    }
}

/// A quantised tensor: the packed indices, one `f16` scale per group, and
/// enough shape to put them back.
#[derive(Debug, Clone, PartialEq)]
pub struct Nicemlenmis {
    genislik: Genislik,
    grup: usize,
    son_eksen: usize,
    uzunluk: usize,
    olcekler: Vec<u16>,
    yuk: Vec<u8>,
    seviyeler: Vec<f32>,
}

/// What a quantisation actually cost, measured against the tensor it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Olcum {
    /// `||w - w'||_F / ||w||_F`.
    pub bagil_hata: f64,
    /// `20 log10(1 / relative error)`, in decibels.
    pub snr_db: f64,
    /// Largest single-weight absolute deviation. Reported beside the aggregate
    /// because a small Frobenius error can still hide one badly placed weight.
    pub en_buyuk_sapma: f64,
    /// Bits per weight actually spent.
    pub agirlik_basina_bit: f64,
    /// Bytes the quantised form occupies, scales included.
    pub bayt: usize,
    /// `f32 bytes / quantised bytes`.
    pub oran: f64,
    /// The codebook's own predicted distortion, as an SNR, for comparison.
    pub kitap_snr_db: f64,
}

impl Nicemlenmis {
    /// Alphabet width.
    #[must_use]
    pub fn genislik(&self) -> Genislik {
        self.genislik
    }

    /// Group size.
    #[must_use]
    pub fn grup(&self) -> usize {
        self.grup
    }

    /// Length of the reduction axis.
    #[must_use]
    pub fn son_eksen(&self) -> usize {
        self.son_eksen
    }

    /// Number of weights.
    #[must_use]
    pub fn uzunluk(&self) -> usize {
        self.uzunluk
    }

    /// Number of rows.
    #[must_use]
    pub fn satir(&self) -> usize {
        self.uzunluk / self.son_eksen
    }

    /// Groups per row.
    #[must_use]
    pub fn grup_basina(&self) -> usize {
        self.son_eksen.div_ceil(self.grup)
    }

    /// The packed index bytes, for the container to write straight out.
    #[must_use]
    pub fn yuk(&self) -> &[u8] {
        &self.yuk
    }

    /// The `f16` group scales.
    #[must_use]
    pub fn olcekler(&self) -> &[u16] {
        &self.olcekler
    }

    /// Bytes on disk: packed indices plus two bytes per group scale.
    #[must_use]
    pub fn bayt(&self) -> usize {
        self.yuk.len() + self.olcekler.len() * 2
    }

    /// The largest live scratch buffer any read path needs, in floats.
    ///
    /// This is one group. It does not depend on the size of the tensor, which
    /// is the property the whole format exists for, so it is exposed as a
    /// function and pinned by a test rather than asserted in prose.
    #[must_use]
    pub fn en_buyuk_calisma_alani(&self) -> usize {
        self.grup
    }

    /// Reconstruct the full `f32` tensor.
    ///
    /// Useful for measurement and for a host with memory to spare. A device
    /// that does not have the memory should use [`Self::carp`], which never
    /// materialises more than one group.
    ///
    /// # Errors
    ///
    /// [`NicemHatasi::Paket`] if the payload is shorter than the shape needs,
    /// and [`NicemHatasi::BoyutUyumsuz`] if the scale count disagrees with the
    /// shape.
    pub fn coz(&self) -> Result<Vec<f32>, NicemHatasi> {
        let mut cikti = vec![0.0f32; self.uzunluk];
        self.her_grup(|satir, bas, son, degerler| {
            let temel = satir * self.son_eksen;
            cikti[temel + bas..temel + son].copy_from_slice(&degerler[..son - bas]);
        })?;
        Ok(cikti)
    }

    /// Multiply by a vector on the reduction axis, one group at a time.
    ///
    /// `y[r] = sum_c w[r][c] * x[c]`. The tensor is never reconstructed: each
    /// group is unpacked into a buffer of `grup` floats, consumed, and
    /// overwritten. Peak extra memory is [`Self::en_buyuk_calisma_alani`]
    /// floats however large the tensor is.
    ///
    /// # Errors
    ///
    /// [`NicemHatasi::EksenUyumsuz`] if `x` is not as long as the reduction
    /// axis, plus the refusals of [`Self::coz`].
    pub fn carp(&self, x: &[f32]) -> Result<Vec<f32>, NicemHatasi> {
        if x.len() != self.son_eksen {
            return Err(NicemHatasi::EksenUyumsuz {
                uzunluk: x.len(),
                eksen: self.son_eksen,
            });
        }
        let mut y = vec![0.0f32; self.satir()];
        self.her_grup(|satir, bas, son, degerler| {
            let mut toplam = 0.0f32;
            for (i, v) in degerler[..son - bas].iter().enumerate() {
                toplam += v * x[bas + i];
            }
            y[satir] += toplam;
        })?;
        Ok(y)
    }

    /// Walk every group, handing the caller the dequantised values.
    ///
    /// The single place the read path exists, so [`Self::coz`] and
    /// [`Self::carp`] cannot drift apart: a bug in one would be a bug in both,
    /// which is preferable to a bug in one of them.
    fn her_grup<F: FnMut(usize, usize, usize, &[f32])>(&self, mut f: F) -> Result<(), NicemHatasi> {
        let grup_basina = self.grup_basina();
        let beklenen_olcek = self.satir() * grup_basina;
        if self.olcekler.len() != beklenen_olcek {
            return Err(NicemHatasi::BoyutUyumsuz {
                beklenen: beklenen_olcek,
                var: self.olcekler.len(),
            });
        }
        let toplam_indeks = beklenen_olcek * self.grup;
        let indeksler = match self.genislik {
            Genislik::Ucdeger => paket::coz_ucdeger(&self.yuk, toplam_indeks)?,
            Genislik::Bit(b) => paket::coz(&self.yuk, b, toplam_indeks)?,
        };

        let mut tampon = vec![0.0f32; self.grup];
        for satir in 0..self.satir() {
            for g in 0..grup_basina {
                let grup_no = satir * grup_basina + g;
                let norm = yarim::yarim_to_f32(self.olcekler[grup_no]);
                let pencere = &indeksler[grup_no * self.grup..(grup_no + 1) * self.grup];
                for (t, idx) in tampon.iter_mut().zip(pencere.iter()) {
                    *t = self.seviyeler[*idx as usize] * norm;
                }
                wht(&mut tampon)?;
                let bas = g * self.grup;
                let son = ((g + 1) * self.grup).min(self.son_eksen);
                f(satir, bas, son, &tampon);
            }
        }
        Ok(())
    }

    /// Measure the round trip against the tensor it came from.
    ///
    /// # Errors
    ///
    /// [`NicemHatasi::BoyutUyumsuz`] if `asil` is not the tensor this was built
    /// from, plus the refusals of [`Self::coz`].
    pub fn olc(&self, asil: &[f32]) -> Result<Olcum, NicemHatasi> {
        if asil.len() != self.uzunluk {
            return Err(NicemHatasi::BoyutUyumsuz {
                beklenen: self.uzunluk,
                var: asil.len(),
            });
        }
        let geri = self.coz()?;
        let mut hata = 0.0f64;
        let mut guc = 0.0f64;
        let mut en_buyuk = 0.0f64;
        for (a, b) in asil.iter().zip(geri.iter()) {
            let d = f64::from(*a) - f64::from(*b);
            hata += d * d;
            guc += f64::from(*a) * f64::from(*a);
            en_buyuk = en_buyuk.max(d.abs());
        }
        let bagil = if guc > 0.0 { (hata / guc).sqrt() } else { 0.0 };
        let bayt = self.bayt();
        #[allow(clippy::cast_precision_loss)]
        let oran = (self.uzunluk as f64 * 4.0) / (bayt as f64);
        let kitap_snr = self.kitap_snr();
        Ok(Olcum {
            bagil_hata: bagil,
            snr_db: if bagil > 0.0 {
                20.0 * (1.0 / bagil).log10()
            } else {
                f64::INFINITY
            },
            en_buyuk_sapma: en_buyuk,
            agirlik_basina_bit: self.genislik.agirlik_basina_bit(self.grup),
            bayt,
            oran,
            kitap_snr_db: kitap_snr,
        })
    }

    /// The codebook's predicted SNR, re-solved for reporting. Cheap relative to
    /// a tensor pass and it keeps [`Olcum`] self-describing.
    fn kitap_snr(&self) -> f64 {
        let kitap = match self.genislik {
            Genislik::Ucdeger => kodkitabi::coz_ucdeger(),
            Genislik::Bit(b) => kodkitabi::coz(b),
        };
        kitap.map_or(f64::NAN, |k| k.snr_db())
    }
}

#[cfg(test)]
// Exact float comparison is the *subject* of several of these tests: a round
// trip that is only approximately equal has lost something, and the cast and
// single-character-name lints are noise inside small numeric fixtures.
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;

    /// A deterministic pseudo-normal tensor: twelve uniforms summed and
    /// centred, which is the Irwin-Hall construction. Fixed recurrence, so the
    /// numbers below are reproducible on any machine.
    fn normalimsi(n: usize, olcek: f32) -> Vec<f32> {
        let mut durum = 0x1234_5678u32;
        let mut cikti = Vec::with_capacity(n);
        for _ in 0..n {
            let mut toplam = 0.0f64;
            for _ in 0..12 {
                durum = durum.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                toplam += f64::from(durum >> 8) / f64::from(1u32 << 24);
            }
            #[allow(clippy::cast_possible_truncation)]
            cikti.push(((toplam - 6.0) as f32) * olcek);
        }
        cikti
    }

    #[test]
    fn two_bits_and_a_group_of_128_cost_exactly_two_point_one_two_five() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let w = normalimsi(128 * 32, 0.05);
        let q = n.nicemle(&w, 128).expect("quantises");
        let o = q.olc(&w).expect("measures");
        assert!((o.agirlik_basina_bit - 2.125).abs() < 1e-12);
        // 4096 weights: 1024 bytes of indices plus 32 scales of 2 bytes.
        assert_eq!(o.bayt, 1024 + 64);
        assert!((o.oran - 16384.0 / 1088.0).abs() < 1e-9, "{}", o.oran);
    }

    #[test]
    fn the_measured_error_tracks_the_codebook_prediction() {
        // The codebook's distortion is derived for a Gaussian. After the
        // rotation a real group is close enough to Gaussian that the measured
        // SNR should land within a few dB of the prediction. If it does not,
        // the rotation is not doing its job - which is exactly the finding this
        // comparison exists to surface.
        for bit in [2u8, 3, 4] {
            let n = Nicemleyici::yeni(Genislik::Bit(bit), 128).expect("valid");
            let w = normalimsi(128 * 64, 0.02);
            let q = n.nicemle(&w, 128).expect("quantises");
            let o = q.olc(&w).expect("measures");
            let fark = (o.snr_db - o.kitap_snr_db).abs();
            assert!(
                fark < 3.0,
                "bit {bit}: measured {:.2} dB vs predicted {:.2} dB",
                o.snr_db,
                o.kitap_snr_db
            );
        }
    }

    #[test]
    fn more_bits_are_monotonically_better() {
        let w = normalimsi(128 * 40, 0.1);
        let mut onceki = f64::INFINITY;
        for bit in 1..=6u8 {
            let n = Nicemleyici::yeni(Genislik::Bit(bit), 128).expect("valid");
            let q = n.nicemle(&w, 128).expect("quantises");
            let o = q.olc(&w).expect("measures");
            assert!(
                o.bagil_hata < onceki,
                "bit {bit} was not better than the width below it"
            );
            onceki = o.bagil_hata;
        }
    }

    #[test]
    fn the_rotation_is_load_bearing_and_this_measures_how_much() {
        // The realistic adversarial case is a heavy tail: a Gaussian bulk with
        // a few weights an order of magnitude larger. Without the rotation an
        // absmax scale is set by those few and every ordinary weight rounds
        // into a level sized for them. This computes the rotation-free baseline
        // inline, so "rotation matters" carries a number instead of a claim.
        let mut w = normalimsi(128 * 8, 0.01);
        for r in 0..8 {
            w[r * 128 + 17] = 0.15;
            w[r * 128 + 96] = -0.14;
        }
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let q = n.nicemle(&w, 128).expect("quantises");
        let donduruldu = q.olc(&w).expect("measures").bagil_hata;

        let dondurulmedi = absmax_iki_bit(&w, 128);
        assert!(
            donduruldu < dondurulmedi,
            "rotated {donduruldu:.4} was not better than unrotated {dondurulmedi:.4}"
        );
        // Measured on this fixture: roughly 0.35 against 0.55.
        assert!(donduruldu < 0.40, "{donduruldu}");
        assert!(dondurulmedi > 0.50, "{dondurulmedi}");
    }

    /// Rotation-free absmax baseline at two bits, for the comparison above and
    /// the one below. Four levels on an integer grid, so zero is available -
    /// the strongest two-bit baseline rather than a straw one.
    fn absmax_iki_bit(w: &[f32], grup: usize) -> f64 {
        let mut hata = 0.0f64;
        let mut guc = 0.0f64;
        for blok in w.chunks(grup) {
            let absmax = blok.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let adim = if absmax > 0.0 { absmax / 2.0 } else { 1.0 };
            for v in blok {
                let geri = (v / adim).round().clamp(-2.0, 1.0) * adim;
                let d = f64::from(*v) - f64::from(geri);
                hata += d * d;
                guc += f64::from(*v) * f64::from(*v);
            }
        }
        (hata / guc).sqrt()
    }

    #[test]
    fn a_pure_spike_is_the_rotations_worst_case_and_this_records_it() {
        // Honest counter-case, found by measurement and kept rather than
        // hidden. A group that is one large weight and nothing else rotates
        // into a vector whose coordinates all have the *same* magnitude. That
        // is the least Gaussian thing a unit vector can be, and a Gaussian
        // codebook has no level at 1/sqrt(g): the two-bit book offers 0.4528
        // and 1.5104 sigma and the truth sits between them, so every coordinate
        // is placed about half a level away.
        //
        // The consequence is stated rather than worked around: the rotation is
        // a bet that trained weight groups are not spikes. `Olcum` reports the
        // per-tensor error so that bet is checked per tensor, and a tensor that
        // comes out near this number is a tensor that should not be quantised
        // at two bits.
        let mut w = vec![0.0f32; 128 * 4];
        for r in 0..4 {
            w[r * 128 + 5] = 100.0;
        }
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let q = n.nicemle(&w, 128).expect("quantises");
        let donduruldu = q.olc(&w).expect("measures").bagil_hata;
        assert!(
            donduruldu > 0.4,
            "the worst case got better; re-read the comment: {donduruldu}"
        );
        // Four bits has a level near 1/sqrt(g) and recovers most of it, which
        // is the mitigation an operator has.
        let n4 = Nicemleyici::yeni(Genislik::Bit(4), 128).expect("valid");
        let q4 = n4.nicemle(&w, 128).expect("quantises");
        let dort = q4.olc(&w).expect("measures").bagil_hata;
        assert!(dort < donduruldu / 2.0, "four bits did not help: {dort}");
    }

    #[test]
    fn multiplying_never_materialises_more_than_one_group() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let w = normalimsi(128 * 50, 0.03);
        let q = n.nicemle(&w, 128).expect("quantises");
        assert_eq!(q.en_buyuk_calisma_alani(), 128);
        assert_eq!(q.uzunluk(), 6400);
        // The working set does not grow with the tensor: same group size on a
        // tensor eight times larger.
        let buyuk = normalimsi(128 * 400, 0.03);
        let qb = n.nicemle(&buyuk, 128).expect("quantises");
        assert_eq!(qb.en_buyuk_calisma_alani(), q.en_buyuk_calisma_alani());
        assert_eq!(qb.uzunluk(), 51_200);
    }

    #[test]
    fn the_streaming_product_agrees_with_the_reconstructed_one() {
        let n = Nicemleyici::yeni(Genislik::Bit(3), 64).expect("valid");
        let w = normalimsi(64 * 20, 0.2);
        let q = n.nicemle(&w, 64).expect("quantises");
        let x: Vec<f32> = (0..64).map(|i| ((i % 5) as f32) - 2.0).collect();
        let akan = q.carp(&x).expect("multiplies");
        let tam = q.coz().expect("reconstructs");
        for (r, y) in akan.iter().enumerate() {
            let beklenen: f32 = (0..64).map(|c| tam[r * 64 + c] * x[c]).sum();
            assert!((y - beklenen).abs() < 1e-3, "row {r}: {y} vs {beklenen}");
        }
    }

    #[test]
    fn a_short_reduction_axis_is_padded_with_zeros_that_do_not_move_the_scale() {
        // 100 columns with a group of 64: the second group is 36 real weights
        // and 28 zeros. Padding with the edge value would inflate that group's
        // norm and shrink every real weight in it.
        let n = Nicemleyici::yeni(Genislik::Bit(4), 64).expect("valid");
        let w = normalimsi(100 * 3, 0.1);
        let q = n.nicemle(&w, 100).expect("quantises");
        assert_eq!(q.grup_basina(), 2);
        assert_eq!(q.olcekler().len(), 6);
        let geri = q.coz().expect("reconstructs");
        assert_eq!(geri.len(), 300);
        let o = q.olc(&w).expect("measures");
        assert!(o.bagil_hata < 0.15, "{}", o.bagil_hata);
    }

    #[test]
    fn an_all_zero_group_reconstructs_exactly_and_does_not_divide_by_zero() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 64).expect("valid");
        let w = vec![0.0f32; 64 * 4];
        let q = n.nicemle(&w, 64).expect("quantises");
        let geri = q.coz().expect("reconstructs");
        assert!(geri.iter().all(|v| v.abs() < 1e-12), "{:?}", &geri[..4]);
        let o = q.olc(&w).expect("measures");
        assert_eq!(o.bagil_hata, 0.0);
    }

    #[test]
    fn ternary_sits_between_one_and_two_bits_in_both_size_and_error() {
        let w = normalimsi(128 * 40, 0.05);
        let bir = Nicemleyici::yeni(Genislik::Bit(1), 128).expect("valid");
        let uc = Nicemleyici::yeni(Genislik::Ucdeger, 128).expect("valid");
        let iki = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let ob = bir.nicemle(&w, 128).expect("q").olc(&w).expect("m");
        let ou = uc.nicemle(&w, 128).expect("q").olc(&w).expect("m");
        let oi = iki.nicemle(&w, 128).expect("q").olc(&w).expect("m");
        assert!(
            ob.bayt < ou.bayt && ou.bayt < oi.bayt,
            "{ob:?} {ou:?} {oi:?}"
        );
        assert!(ob.bagil_hata > ou.bagil_hata && ou.bagil_hata > oi.bagil_hata);
        assert!((ou.agirlik_basina_bit - 1.725).abs() < 1e-12);
    }

    #[test]
    fn a_non_finite_weight_is_refused_with_its_position() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 64).expect("valid");
        let mut w = vec![0.1f32; 128];
        w[73] = f32::NAN;
        assert_eq!(
            n.nicemle(&w, 64).err(),
            Some(NicemHatasi::SonluDegil { konum: 73 })
        );
        w[73] = f32::INFINITY;
        assert_eq!(
            n.nicemle(&w, 64).err(),
            Some(NicemHatasi::SonluDegil { konum: 73 })
        );
    }

    #[test]
    fn unusable_group_sizes_are_refused_at_construction() {
        assert_eq!(
            Nicemleyici::yeni(Genislik::Bit(2), 100).err(),
            Some(NicemHatasi::GrupIkininKuvvetiDegil { grup: 100 })
        );
        assert_eq!(
            Nicemleyici::yeni(Genislik::Bit(2), 8).err(),
            Some(NicemHatasi::GrupCokKucuk { grup: 8 })
        );
    }

    #[test]
    fn a_tensor_whose_length_disagrees_with_its_axis_is_refused() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 64).expect("valid");
        assert_eq!(
            n.nicemle(&[0.0; 100], 64).err(),
            Some(NicemHatasi::EksenUyumsuz {
                uzunluk: 100,
                eksen: 64
            })
        );
        assert_eq!(n.nicemle(&[], 64).err(), Some(NicemHatasi::Bos));
    }

    #[test]
    fn the_product_refuses_a_vector_of_the_wrong_length() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 64).expect("valid");
        let q = n.nicemle(&normalimsi(64 * 2, 0.1), 64).expect("quantises");
        assert!(matches!(
            q.carp(&[0.0; 10]),
            Err(NicemHatasi::EksenUyumsuz { .. })
        ));
    }

    #[test]
    fn quantising_the_same_tensor_twice_gives_the_same_bytes() {
        // Two operators must produce identical files or the container's digest
        // is not a digest of anything.
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let w = normalimsi(128 * 16, 0.07);
        let a = n.nicemle(&w, 128).expect("quantises");
        let b = n.nicemle(&w, 128).expect("quantises");
        assert_eq!(a.yuk(), b.yuk());
        assert_eq!(a.olcekler(), b.olcekler());
        assert_eq!(a, b);
    }

    #[test]
    fn every_width_name_is_distinct_because_the_directory_stores_it() {
        let mut adlar = vec![Genislik::Ucdeger.ad()];
        for b in 1..=8u8 {
            adlar.push(Genislik::Bit(b).ad());
        }
        let mut sirali = adlar.clone();
        sirali.sort_unstable();
        sirali.dedup();
        assert_eq!(sirali.len(), adlar.len(), "{adlar:?}");
    }
}
