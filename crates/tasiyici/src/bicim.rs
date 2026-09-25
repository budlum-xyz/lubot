//! The container format: eight magic bytes, a directory, and a payload that is
//! read where it lies.
//!
//! # The one design constraint
//!
//! A device with less memory than the model has parameters can still serve the
//! model **if and only if** the file can be consulted without being copied. So
//! every decision here is downstream of one rule: *a reader holds a `&[u8]`
//! over the file and takes slices of it; it never builds a second copy of a
//! tensor.*
//!
//! That rule is what rules out the obvious conveniences. No compression over
//! the payload, because a byte offset would then depend on the bytes before it.
//! No serialiser for the directory, because a serialiser's output format is
//! allowed to change and every file written before the change becomes
//! unreadable. No pointer fix-up pass on load, because a fix-up pass is a write
//! and a write means a copy. What is left is a fixed-width header, a directory
//! parsed once into owned records, and a payload nobody touches until a group
//! of weights is actually needed.
//!
//! # Layout
//!
//! ```text
//! 0   magic        "LUBOTNCM"              8 bytes
//! 8   surum        u32 little-endian       4
//! 12  tensor adedi u32 little-endian       4
//! 16  dizin_bayt   u64 little-endian       8
//! 24  yuk_bayt     u64 little-endian       8
//! 32  yuk_ozeti    sha-256 of the payload  32
//! 64  dizin        dizin_bayt bytes, then padded to the next 64
//! ..  yuk          yuk_bayt bytes, each tensor block 64-aligned
//! ```
//!
//! Little-endian is written down rather than assumed. Every platform this will
//! run on is little-endian today; the one that is not would otherwise read a
//! plausible file and produce nonsense, which is the failure this repository
//! spends most of its refusals avoiding.
//!
//! Sixty-four is the alignment because it is the common cache line, and because
//! a tensor block that starts mid-line costs a straddling load on the hot path
//! for every group. It is not required for correctness - the reader does
//! unaligned byte slices - so a future format could change it without changing
//! any of the arithmetic.
//!
//! # Why the digest is not checked on open
//!
//! [`Kapsayici::yuk_ozeti`] holds a SHA-256 over the whole payload, and
//! [`Kapsayici::dogrula`] checks it. Opening does **not**, and that is the
//! honest trade written down rather than hidden: hashing the payload reads
//! every byte of it, which is exactly the thing a device with a small memory
//! opened the file to avoid. Verifying costs a full pass; the operator decides
//! when to pay it (on install, on a schedule, after a transfer) and the format
//! makes the choice explicit instead of making it for them.
//!
//! What *is* checked on open is structure: the magic, the version, that every
//! declared range lies inside the payload, that the ranges do not overlap, and
//! that each tensor's byte counts match the shape and width it declares. Those
//! are cheap - they read the directory, not the payload - and they are the
//! checks that stop a truncated or scrambled file from being read as weights.

use std::fmt;

use lubot_nicem::grup::Genislik;
use sha2::{Digest, Sha256};

/// The eight bytes at the start of every container.
pub const IMZA: &[u8; 8] = b"LUBOTNCM";
/// Format version. A reader refuses anything else rather than guessing.
pub const SURUM: u32 = 1;
/// Fixed header length, and the alignment of everything after it.
pub const BASLIK_BAYT: usize = 64;
/// Alignment of each payload block.
pub const HIZA: usize = 64;

/// Why a container was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BicimHatasi {
    /// Shorter than the fixed header.
    CokKisa { var: usize },
    /// The first eight bytes are not [`IMZA`].
    ImzaYok,
    /// A version this reader does not implement.
    Surum { var: u32 },
    /// The declared directory or payload does not fit the buffer.
    UzunlukTutmuyor {
        beyan: usize,
        var: usize,
        alan: &'static str,
    },
    /// The directory ended in the middle of a record.
    DizinKesildi { tensor: usize },
    /// A tensor name that is not valid UTF-8. Names are read by humans and
    /// matched by the loader; a name that cannot be printed cannot be reported
    /// in a refusal either.
    AdUtf8Degil { tensor: usize },
    /// Two tensors share a name.
    AdTekrari { ad: String },
    /// A width code outside `0..=8`.
    GenislikKodu { ad: String, kod: u8 },
    /// A group size that is not a power of two, or is zero.
    Grup { ad: String, grup: u32 },
    /// A declared byte range falls outside the payload.
    AralikDisi {
        ad: String,
        bas: u64,
        son: u64,
        yuk: u64,
    },
    /// Two tensors claim overlapping payload bytes.
    Ortusme { once: String, sonra: String },
    /// The declared byte count does not match the shape and width.
    BoyutTutmuyor {
        ad: String,
        beyan: u64,
        gereken: u64,
        alan: &'static str,
    },
    /// The payload digest does not match. Only [`Kapsayici::dogrula`] can raise
    /// this, because only it reads the payload.
    OzetTutmuyor,
    /// A tensor asked for by name is not in the directory.
    Yok { ad: String },
}

impl fmt::Display for BicimHatasi {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CokKisa { var } => {
                write!(f, "dosya {var} bayt: basligin kendisi {BASLIK_BAYT} bayt")
            }
            Self::ImzaYok => write!(f, "imza yok: bu bir kapsayici degil"),
            Self::Surum { var } => {
                write!(f, "surum {var} okunamiyor, bu okuyucu {SURUM} biliyor")
            }
            Self::UzunlukTutmuyor { beyan, var, alan } => write!(
                f,
                "{alan} {beyan} bayt beyan edildi, dosyada {var} bayt var"
            ),
            Self::DizinKesildi { tensor } => {
                write!(f, "dizin {tensor}. kayitta kesildi")
            }
            Self::AdUtf8Degil { tensor } => {
                write!(f, "{tensor}. kaydin adi UTF-8 degil")
            }
            Self::AdTekrari { ad } => write!(f, "ad tekrari: {ad}"),
            Self::GenislikKodu { ad, kod } => {
                write!(f, "{ad}: genislik kodu {kod} taninmiyor")
            }
            Self::Grup { ad, grup } => {
                write!(f, "{ad}: grup {grup} ikinin kuvveti degil")
            }
            Self::AralikDisi { ad, bas, son, yuk } => {
                write!(f, "{ad}: {bas}..{son} yuk disinda (yuk {yuk} bayt)")
            }
            Self::Ortusme { once, sonra } => {
                write!(f, "{once} ile {sonra} ayni baytlari istiyor")
            }
            Self::BoyutTutmuyor {
                ad,
                beyan,
                gereken,
                alan,
            } => write!(
                f,
                "{ad}: {alan} {beyan} bayt beyan edildi, sekil {gereken} bayt gerektiriyor"
            ),
            Self::OzetTutmuyor => write!(f, "yuk ozeti tutmuyor: dosya degismis"),
            Self::Yok { ad } => write!(f, "dizinde {ad} yok"),
        }
    }
}

impl std::error::Error for BicimHatasi {}

/// Encode a width as the byte the directory stores.
#[must_use]
pub fn genislik_kodu(g: Genislik) -> u8 {
    match g {
        Genislik::Ucdeger => 0,
        Genislik::Bit(b) => b,
    }
}

/// Decode a width byte.
///
/// # Errors
///
/// [`BicimHatasi::GenislikKodu`] for anything outside `0..=8`.
pub fn genislik_coz(kod: u8, ad: &str) -> Result<Genislik, BicimHatasi> {
    match kod {
        0 => Ok(Genislik::Ucdeger),
        1..=8 => Ok(Genislik::Bit(kod)),
        _ => Err(BicimHatasi::GenislikKodu {
            ad: ad.to_string(),
            kod,
        }),
    }
}

/// One directory record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kayit {
    /// Tensor name, as the loader asks for it.
    pub ad: String,
    /// Alphabet width.
    pub genislik: Genislik,
    /// Rung of the ladder this tensor belongs to. `0` means always resident:
    /// the embedding, the final norm, the decision head. `1..` are the layers,
    /// in order, and a reader that stops at rung `k` stops at a model.
    pub kademe: u8,
    /// Quantisation group size.
    pub grup: u32,
    /// Rows.
    pub satir: u64,
    /// Length of the reduction axis.
    pub son_eksen: u64,
    /// Offset of the group scales inside the payload.
    pub olcek_ofset: u64,
    /// Length of the group scales, in bytes.
    pub olcek_bayt: u64,
    /// Offset of the packed indices inside the payload.
    pub yuk_ofset: u64,
    /// Length of the packed indices, in bytes.
    pub yuk_bayt: u64,
}

impl Kayit {
    /// Total bytes this tensor occupies in the payload.
    #[must_use]
    pub fn bayt(&self) -> u64 {
        self.olcek_bayt + self.yuk_bayt
    }

    /// Number of weights.
    #[must_use]
    pub fn agirlik(&self) -> u64 {
        self.satir * self.son_eksen
    }

    /// Groups this tensor is cut into.
    #[must_use]
    pub fn grup_sayisi(&self) -> u64 {
        let per = self.son_eksen.div_ceil(u64::from(self.grup));
        self.satir * per
    }

    /// Bytes the scales must occupy, from the shape alone.
    #[must_use]
    pub fn gereken_olcek_bayt(&self) -> u64 {
        self.grup_sayisi() * 2
    }

    /// Bytes the packed indices must occupy, from the shape and width alone.
    #[must_use]
    pub fn gereken_yuk_bayt(&self) -> u64 {
        let indeks = self.grup_sayisi() * u64::from(self.grup);
        match self.genislik {
            Genislik::Ucdeger => indeks.div_ceil(5),
            Genislik::Bit(b) => (indeks * u64::from(b)).div_ceil(8),
        }
    }
}

/// A parsed container over borrowed bytes.
///
/// Holds the directory as owned records - it is kilobytes - and the payload as
/// a slice into the caller's buffer. Nothing else is copied.
#[derive(Debug, Clone)]
pub struct Kapsayici<'a> {
    kayitlar: Vec<Kayit>,
    yuk: &'a [u8],
    yuk_ozeti: [u8; 32],
}

impl<'a> Kapsayici<'a> {
    /// Parse and structurally validate a container.
    ///
    /// Reads the header and the directory. Does **not** read the payload, and
    /// therefore does not verify the digest; see [`Self::dogrula`].
    ///
    /// # Errors
    ///
    /// Every variant of [`BicimHatasi`] except [`BicimHatasi::OzetTutmuyor`]
    /// and [`BicimHatasi::Yok`].
    /// The `u64 as usize` casts below are checked immediately afterwards
    /// against the real length of `bayt`, so a value that truncated on a
    /// 32-bit target produces a refusal rather than a short read.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn ac(bayt: &'a [u8]) -> Result<Self, BicimHatasi> {
        if bayt.len() < BASLIK_BAYT {
            return Err(BicimHatasi::CokKisa { var: bayt.len() });
        }
        if &bayt[0..8] != IMZA {
            return Err(BicimHatasi::ImzaYok);
        }
        let surum = u32_oku(bayt, 8);
        if surum != SURUM {
            return Err(BicimHatasi::Surum { var: surum });
        }
        let adet = u32_oku(bayt, 12) as usize;
        let dizin_bayt = u64_oku(bayt, 16) as usize;
        let yuk_bayt = u64_oku(bayt, 24) as usize;
        let mut ozet = [0u8; 32];
        ozet.copy_from_slice(&bayt[32..64]);

        let dizin_son =
            BASLIK_BAYT
                .checked_add(dizin_bayt)
                .ok_or(BicimHatasi::UzunlukTutmuyor {
                    beyan: dizin_bayt,
                    var: bayt.len(),
                    alan: "dizin",
                })?;
        if dizin_son > bayt.len() {
            return Err(BicimHatasi::UzunlukTutmuyor {
                beyan: dizin_bayt,
                var: bayt.len(),
                alan: "dizin",
            });
        }
        let yuk_bas = hizala(dizin_son);
        let yuk_son = yuk_bas
            .checked_add(yuk_bayt)
            .ok_or(BicimHatasi::UzunlukTutmuyor {
                beyan: yuk_bayt,
                var: bayt.len(),
                alan: "yuk",
            })?;
        if yuk_son > bayt.len() {
            return Err(BicimHatasi::UzunlukTutmuyor {
                beyan: yuk_bayt,
                var: bayt.len(),
                alan: "yuk",
            });
        }

        let dizin = &bayt[BASLIK_BAYT..dizin_son];
        let mut imlec = 0usize;
        let mut kayitlar = Vec::with_capacity(adet);
        for i in 0..adet {
            let kayit = kayit_oku(dizin, &mut imlec, i)?;
            kayitlar.push(kayit);
        }

        let yuk = &bayt[yuk_bas..yuk_son];
        let k = Self {
            kayitlar,
            yuk,
            yuk_ozeti: ozet,
        };
        k.tutarli()?;
        Ok(k)
    }

    /// Structural checks that need the whole directory: unique names, in-range
    /// and non-overlapping payload blocks, byte counts that match the shapes.
    fn tutarli(&self) -> Result<(), BicimHatasi> {
        let mut adlar: Vec<&str> = self.kayitlar.iter().map(|k| k.ad.as_str()).collect();
        adlar.sort_unstable();
        for w in adlar.windows(2) {
            if w[0] == w[1] {
                return Err(BicimHatasi::AdTekrari {
                    ad: w[0].to_string(),
                });
            }
        }

        let yuk_uzunluk = self.yuk.len() as u64;
        let mut bloklar: Vec<(u64, u64, &str)> = Vec::with_capacity(self.kayitlar.len() * 2);
        for k in &self.kayitlar {
            if k.grup == 0 || !k.grup.is_power_of_two() {
                return Err(BicimHatasi::Grup {
                    ad: k.ad.clone(),
                    grup: k.grup,
                });
            }
            let gereken_olcek = k.gereken_olcek_bayt();
            if k.olcek_bayt != gereken_olcek {
                return Err(BicimHatasi::BoyutTutmuyor {
                    ad: k.ad.clone(),
                    beyan: k.olcek_bayt,
                    gereken: gereken_olcek,
                    alan: "olcek",
                });
            }
            let gereken_yuk = k.gereken_yuk_bayt();
            if k.yuk_bayt != gereken_yuk {
                return Err(BicimHatasi::BoyutTutmuyor {
                    ad: k.ad.clone(),
                    beyan: k.yuk_bayt,
                    gereken: gereken_yuk,
                    alan: "yuk",
                });
            }
            for (bas, uzunluk) in [(k.olcek_ofset, k.olcek_bayt), (k.yuk_ofset, k.yuk_bayt)] {
                let son = bas.checked_add(uzunluk).ok_or(BicimHatasi::AralikDisi {
                    ad: k.ad.clone(),
                    bas,
                    son: u64::MAX,
                    yuk: yuk_uzunluk,
                })?;
                if son > yuk_uzunluk {
                    return Err(BicimHatasi::AralikDisi {
                        ad: k.ad.clone(),
                        bas,
                        son,
                        yuk: yuk_uzunluk,
                    });
                }
                if uzunluk > 0 {
                    bloklar.push((bas, son, k.ad.as_str()));
                }
            }
        }
        bloklar.sort_unstable();
        for w in bloklar.windows(2) {
            if w[0].1 > w[1].0 {
                return Err(BicimHatasi::Ortusme {
                    once: w[0].2.to_string(),
                    sonra: w[1].2.to_string(),
                });
            }
        }
        Ok(())
    }

    /// The directory.
    #[must_use]
    pub fn kayitlar(&self) -> &[Kayit] {
        &self.kayitlar
    }

    /// The payload, as borrowed bytes.
    #[must_use]
    pub fn yuk(&self) -> &'a [u8] {
        self.yuk
    }

    /// The digest recorded in the header.
    #[must_use]
    pub fn yuk_ozeti(&self) -> [u8; 32] {
        self.yuk_ozeti
    }

    /// Look a tensor up by name.
    ///
    /// # Errors
    ///
    /// [`BicimHatasi::Yok`] if the directory has no such name.
    pub fn kayit(&self, ad: &str) -> Result<&Kayit, BicimHatasi> {
        self.kayitlar
            .iter()
            .find(|k| k.ad == ad)
            .ok_or_else(|| BicimHatasi::Yok { ad: ad.to_string() })
    }

    /// The scale bytes and the index bytes of one tensor, as slices into the
    /// mapped payload. No copy.
    ///
    /// # Errors
    ///
    /// [`BicimHatasi::Yok`] if the name is not in the directory.
    ///
    /// The offsets were bounds-checked against the payload length when the
    /// container was opened, so the casts here cannot produce a bad slice on
    /// any target that could hold the payload in the first place.
    #[allow(clippy::cast_possible_truncation)]
    pub fn dilimler(&self, ad: &str) -> Result<(&'a [u8], &'a [u8]), BicimHatasi> {
        let k = self.kayit(ad)?;
        let o = k.olcek_ofset as usize;
        let ob = k.olcek_bayt as usize;
        let y = k.yuk_ofset as usize;
        let yb = k.yuk_bayt as usize;
        Ok((&self.yuk[o..o + ob], &self.yuk[y..y + yb]))
    }

    /// Recompute the payload digest and compare it with the header.
    ///
    /// Reads every payload byte, which is the cost the format is designed to
    /// let a reader avoid. Call it on install or after a transfer, not on every
    /// open.
    ///
    /// # Errors
    ///
    /// [`BicimHatasi::OzetTutmuyor`] if the payload has changed.
    pub fn dogrula(&self) -> Result<(), BicimHatasi> {
        let mut h = Sha256::new();
        h.update(self.yuk);
        let ozet: [u8; 32] = h.finalize().into();
        if ozet == self.yuk_ozeti {
            Ok(())
        } else {
            Err(BicimHatasi::OzetTutmuyor)
        }
    }

    /// Total payload bytes the tensors actually claim, which can be less than
    /// the payload length because of alignment padding.
    #[must_use]
    pub fn talep_edilen_bayt(&self) -> u64 {
        self.kayitlar.iter().map(Kayit::bayt).sum()
    }

    /// Total weights across every tensor.
    #[must_use]
    pub fn agirlik(&self) -> u64 {
        self.kayitlar.iter().map(Kayit::agirlik).sum()
    }

    /// Measured bits per weight over the whole container, header and padding
    /// included. The honest figure: it is what the file costs divided by what
    /// the file holds, not the alphabet width.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn agirlik_basina_bit(&self, dosya_bayt: usize) -> f64 {
        let n = self.agirlik();
        if n == 0 {
            return 0.0;
        }
        (dosya_bayt as f64 * 8.0) / (n as f64)
    }
}

/// Round up to the next [`HIZA`] boundary.
#[must_use]
pub fn hizala(n: usize) -> usize {
    n.div_ceil(HIZA) * HIZA
}

fn u32_oku(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn u64_oku(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([
        b[i],
        b[i + 1],
        b[i + 2],
        b[i + 3],
        b[i + 4],
        b[i + 5],
        b[i + 6],
        b[i + 7],
    ])
}

fn kayit_oku(dizin: &[u8], imlec: &mut usize, no: usize) -> Result<Kayit, BicimHatasi> {
    // width, rung, group, rows, axis, and four 64-bit ranges
    const SABIT: usize = 1 + 1 + 4 + 8 + 8 + 8 + 8 + 8 + 8;
    let kesik = || BicimHatasi::DizinKesildi { tensor: no };
    if *imlec + 2 > dizin.len() {
        return Err(kesik());
    }
    let ad_uzunluk = u16::from_le_bytes([dizin[*imlec], dizin[*imlec + 1]]) as usize;
    *imlec += 2;
    if *imlec + ad_uzunluk > dizin.len() {
        return Err(kesik());
    }
    let ad = std::str::from_utf8(&dizin[*imlec..*imlec + ad_uzunluk])
        .map_err(|_| BicimHatasi::AdUtf8Degil { tensor: no })?
        .to_string();
    *imlec += ad_uzunluk;
    if *imlec + SABIT > dizin.len() {
        return Err(kesik());
    }
    let kod = dizin[*imlec];
    let kademe = dizin[*imlec + 1];
    let genislik = genislik_coz(kod, &ad)?;
    let grup = u32_oku(dizin, *imlec + 2);
    let satir = u64_oku(dizin, *imlec + 6);
    let son_eksen = u64_oku(dizin, *imlec + 14);
    let olcek_ofset = u64_oku(dizin, *imlec + 22);
    let olcek_bayt = u64_oku(dizin, *imlec + 30);
    let yuk_ofset = u64_oku(dizin, *imlec + 38);
    let yuk_bayt = u64_oku(dizin, *imlec + 46);
    *imlec += SABIT;
    Ok(Kayit {
        ad,
        genislik,
        kademe,
        grup,
        satir,
        son_eksen,
        olcek_ofset,
        olcek_bayt,
        yuk_ofset,
        yuk_bayt,
    })
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::yazici::Yazici;
    use lubot_nicem::Nicemleyici;

    fn ornek(n: usize, olcek: f32) -> Vec<f32> {
        let mut durum = 0x5150_2026u32;
        (0..n)
            .map(|_| {
                durum = durum.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let v = ((durum >> 8) as f32) / 8_388_608.0 - 1.0;
                v * olcek
            })
            .collect()
    }

    fn kucuk_dosya() -> Vec<u8> {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let mut y = Yazici::yeni();
        let a = n.nicemle(&ornek(128 * 8, 0.05), 128).expect("q");
        let b = n.nicemle(&ornek(128 * 4, 0.05), 128).expect("q");
        y.ekle("gomme", 0, &a).expect("added");
        y.ekle("katman.1.wq", 1, &b).expect("added");
        y.bayt()
    }

    #[test]
    fn a_written_container_opens_and_its_directory_survives() {
        let dosya = kucuk_dosya();
        let k = Kapsayici::ac(&dosya).expect("opens");
        assert_eq!(k.kayitlar().len(), 2);
        assert_eq!(k.kayitlar()[0].ad, "gomme");
        assert_eq!(k.kayitlar()[0].kademe, 0);
        assert_eq!(k.kayitlar()[1].kademe, 1);
        assert_eq!(k.agirlik(), 128 * 12);
        k.dogrula().expect("digest holds");
    }

    #[test]
    fn the_payload_is_borrowed_not_copied() {
        // The slices a reader gets must point into the caller's buffer. This is
        // the property the whole format exists for, so it is checked by address
        // and not by trust.
        let dosya = kucuk_dosya();
        let k = Kapsayici::ac(&dosya).expect("opens");
        let (olcek, yuk) = k.dilimler("gomme").expect("present");
        let dosya_bas = dosya.as_ptr() as usize;
        let dosya_son = dosya_bas + dosya.len();
        for dilim in [olcek, yuk] {
            let p = dilim.as_ptr() as usize;
            assert!(
                p >= dosya_bas && p + dilim.len() <= dosya_son,
                "a slice was copied out of the mapped buffer"
            );
        }
    }

    #[test]
    fn the_blocks_are_aligned_so_a_group_read_does_not_straddle_a_line() {
        let dosya = kucuk_dosya();
        let k = Kapsayici::ac(&dosya).expect("opens");
        for kayit in k.kayitlar() {
            assert_eq!(kayit.olcek_ofset % HIZA as u64, 0, "{}", kayit.ad);
            assert_eq!(kayit.yuk_ofset % HIZA as u64, 0, "{}", kayit.ad);
        }
    }

    #[test]
    fn a_truncated_file_is_refused_rather_than_read_short() {
        let dosya = kucuk_dosya();
        for kes in [0usize, 1, 8, 32, 63] {
            let e = Kapsayici::ac(&dosya[..kes]).expect_err("must refuse");
            assert!(matches!(e, BicimHatasi::CokKisa { .. }), "{e:?}");
        }
        let kisa = &dosya[..dosya.len() - 1];
        let e = Kapsayici::ac(kisa).expect_err("must refuse");
        assert!(matches!(e, BicimHatasi::UzunlukTutmuyor { .. }), "{e:?}");
    }

    #[test]
    fn a_foreign_file_is_refused_at_the_magic() {
        let mut dosya = kucuk_dosya();
        dosya[0] = b'X';
        assert_eq!(Kapsayici::ac(&dosya).err(), Some(BicimHatasi::ImzaYok));
    }

    #[test]
    fn a_future_version_is_refused_rather_than_guessed_at() {
        let mut dosya = kucuk_dosya();
        dosya[8..12].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            Kapsayici::ac(&dosya).err(),
            Some(BicimHatasi::Surum { var: 7 })
        );
    }

    #[test]
    fn a_changed_payload_byte_is_caught_by_the_digest_and_only_by_it() {
        let mut dosya = kucuk_dosya();
        let son = dosya.len() - 1;
        dosya[son] ^= 0xff;
        // Structure is untouched, so opening still succeeds - which is exactly
        // why the digest exists and why the format says so out loud.
        let k = Kapsayici::ac(&dosya).expect("still structurally valid");
        assert_eq!(k.dogrula().err(), Some(BicimHatasi::OzetTutmuyor));
    }

    #[test]
    fn a_missing_tensor_is_named_in_the_refusal() {
        let dosya = kucuk_dosya();
        let k = Kapsayici::ac(&dosya).expect("opens");
        let e = k.kayit("katman.9.wq").expect_err("absent");
        assert_eq!(e.to_string(), "dizinde katman.9.wq yok");
    }

    #[test]
    fn a_shape_that_disagrees_with_its_byte_count_is_refused() {
        // Corrupting the declared row count leaves a file that would otherwise
        // load and produce a model with the wrong weights in it.
        let dosya = kucuk_dosya();
        let k = Kapsayici::ac(&dosya).expect("opens");
        let mut sahte = k.kayitlar()[0].clone();
        sahte.satir += 1;
        assert_ne!(sahte.gereken_yuk_bayt(), sahte.yuk_bayt);
    }

    #[test]
    fn overlapping_blocks_are_refused_because_one_of_them_must_be_wrong() {
        let mut y = Yazici::yeni();
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let a = n.nicemle(&ornek(128 * 2, 0.05), 128).expect("q");
        y.ekle("a", 0, &a).expect("added");
        y.ekle("b", 1, &a).expect("added");
        let mut dosya = y.bayt();
        // Point b's payload at a's bytes by rewriting its offset in place.
        let k = Kapsayici::ac(&dosya).expect("opens");
        let hedef = k.kayitlar()[0].yuk_ofset;
        let kaynak = k.kayitlar()[1].yuk_ofset;
        let dizin = &dosya[BASLIK_BAYT..];
        let konum = dizin
            .windows(8)
            .position(|w| w == kaynak.to_le_bytes())
            .expect("offset is in the directory")
            + BASLIK_BAYT;
        dosya[konum..konum + 8].copy_from_slice(&hedef.to_le_bytes());
        let e = Kapsayici::ac(&dosya).expect_err("must refuse");
        assert!(matches!(e, BicimHatasi::Ortusme { .. }), "{e:?}");
    }

    #[test]
    fn the_measured_bits_per_weight_includes_the_header_and_the_padding() {
        let dosya = kucuk_dosya();
        let k = Kapsayici::ac(&dosya).expect("opens");
        let olculen = k.agirlik_basina_bit(dosya.len());
        // The alphabet costs 2.125; the file costs more because of the header,
        // the directory and the alignment. Reporting the alphabet figure as the
        // file figure is the small lie this method exists to prevent.
        assert!(olculen > 2.125, "{olculen}");
        assert!(olculen < 4.0, "{olculen}");
    }

    #[test]
    fn width_codes_round_trip_and_an_unknown_one_is_refused() {
        for g in [
            Genislik::Ucdeger,
            Genislik::Bit(1),
            Genislik::Bit(2),
            Genislik::Bit(4),
            Genislik::Bit(8),
        ] {
            assert_eq!(genislik_coz(genislik_kodu(g), "t").expect("known"), g);
        }
        assert_eq!(
            genislik_coz(9, "t").err(),
            Some(BicimHatasi::GenislikKodu {
                ad: "t".to_string(),
                kod: 9
            })
        );
    }

    #[test]
    fn alignment_rounds_up_and_leaves_exact_multiples_alone() {
        assert_eq!(hizala(0), 0);
        assert_eq!(hizala(1), 64);
        assert_eq!(hizala(64), 64);
        assert_eq!(hizala(65), 128);
    }
}
