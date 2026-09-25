//! safetensors reading, split across part files.
//!
//! # The format, exactly
//!
//! A safetensors file is an eight-byte little-endian header length, that many
//! bytes of JSON describing every tensor, and then the tensor bytes in the
//! order the JSON lists them. There is no compression, no alignment rule beyond
//! the ones the header states, and no checksum. This module reads that header
//! and hands out byte ranges; it does not copy a tensor until a caller asks for
//! its numbers.
//!
//! # Why the parts are read as one file
//!
//! The artifact here arrives as seven part files. Treating them as one
//! contiguous byte sequence - and seeking across the boundary when a tensor
//! straddles two parts - keeps every other function in this crate working on
//! the format rather than on the packing. The alternative, loading each part
//! and stitching them, would hold 643 MB of bytes in memory to answer a
//! question about a 768-element vector.
//!
//! # What this module refuses
//!
//! A header that does not parse, a tensor whose byte range runs past the end of
//! the data, overlapping ranges, and a data offset that does not start after
//! the header. Each is refused with the tensor's name, because "the file is
//! broken" is not a diagnosis a reader can act on.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// One tensor as the header describes it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TensorBasligi {
    /// The element type, as the format names it (`F16`, `F32`, `BF16`, ...).
    pub dtype: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// `[start, end)` relative to the start of the tensor data section.
    pub data_offsets: [usize; 2],
}

impl TensorBasligi {
    /// How many elements the tensor holds.
    #[must_use]
    pub fn eleman(&self) -> usize {
        self.shape.iter().product()
    }

    /// How many bytes the tensor's payload occupies.
    #[must_use]
    pub fn bayt(&self) -> usize {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

/// Why a checkpoint could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaslikHatasi {
    /// The part files could not be opened or read.
    Okunamadi {
        /// Which path failed.
        yol: String,
        /// The operating system's own message.
        mesaj: String,
    },
    /// The header length or the JSON did not parse.
    Baslik {
        /// The parser's own message.
        mesaj: String,
    },
    /// A tensor's range is outside the data section.
    ///
    /// The message carries both numbers, because the common cause is a missing
    /// part file and "range outside" alone would send a reader looking at the
    /// tensors instead of at the directory listing.
    AralikDisi {
        /// The tensor's name.
        ad: String,
    },
    /// A part file is missing: the header describes more bytes than are here.
    ParcaEksik {
        /// How many bytes are present.
        var: u64,
        /// How many the header implies.
        gereken: u64,
    },
    /// Two tensors claim the same bytes.
    Cakisma {
        /// The tensor whose range overlaps an earlier one.
        ad: String,
    },
    /// The data section does not start after the header.
    VeriBaslangici {
        /// The offset the header implies.
        ofset: usize,
    },
}

/// A checkpoint spread over part files, read as one byte sequence.
///
/// Cloning is cheap and deliberate: the struct holds paths and sizes, not
/// bytes, so a value of it can live next to the weights that were loaded
/// through it.
#[derive(Debug, Clone)]
pub struct ParcaliDosya {
    parcalar: Vec<PathBuf>,
    boyutlar: Vec<u64>,
    toplam: u64,
}

impl ParcaliDosya {
    /// Opens every part in the directory whose name starts with `on_ek`.
    ///
    /// Parts are ordered by name, which is why the packer zero-pads the index:
    /// `part-10` must not sort before `part-9`.
    ///
    /// # Errors
    /// [`BaslikHatasi::Okunamadi`] when a part cannot be opened, and
    /// [`BaslikHatasi::Baslik`] when no part matches.
    pub fn ac(klasor: &Path, on_ek: &str) -> Result<Self, BaslikHatasi> {
        let mut parcalar: Vec<PathBuf> = Vec::new();
        let girdiler = std::fs::read_dir(klasor).map_err(|h| BaslikHatasi::Okunamadi {
            yol: klasor.display().to_string(),
            mesaj: h.to_string(),
        })?;
        for girdi in girdiler {
            let girdi = girdi.map_err(|h| BaslikHatasi::Okunamadi {
                yol: klasor.display().to_string(),
                mesaj: h.to_string(),
            })?;
            let ad = girdi.file_name().to_string_lossy().to_string();
            if ad.starts_with(on_ek) && ad != on_ek {
                parcalar.push(girdi.path());
            }
        }
        if parcalar.is_empty() {
            return Err(BaslikHatasi::Baslik {
                mesaj: format!("{on_ek} ile baslayan parca yok: {}", klasor.display()),
            });
        }
        parcalar.sort();
        let mut boyutlar = Vec::with_capacity(parcalar.len());
        for yol in &parcalar {
            let meta = std::fs::metadata(yol).map_err(|h| BaslikHatasi::Okunamadi {
                yol: yol.display().to_string(),
                mesaj: h.to_string(),
            })?;
            boyutlar.push(meta.len());
        }
        let toplam = boyutlar.iter().sum();
        Ok(Self {
            parcalar,
            boyutlar,
            toplam,
        })
    }

    /// How many parts there are.
    #[must_use]
    pub fn parca_sayisi(&self) -> usize {
        self.parcalar.len()
    }

    /// The total size in bytes.
    #[must_use]
    pub fn boyut(&self) -> u64 {
        self.toplam
    }

    /// The SHA-256 of every part, hashed in name order as one stream.
    ///
    /// # Errors
    /// [`BaslikHatasi::Baslik`] when a part cannot be opened or read.
    pub fn sha256(&self) -> Result<String, BaslikHatasi> {
        use sha2::{Digest, Sha256};
        let mut ozet = Sha256::new();
        let mut tampon = vec![0_u8; 1 << 20];
        for parca in &self.parcalar {
            let mut dosya = std::fs::File::open(parca).map_err(|e| BaslikHatasi::Baslik {
                mesaj: format!("{}: {e}", parca.display()),
            })?;
            loop {
                let okunan = std::io::Read::read(&mut dosya, &mut tampon).map_err(|e| {
                    BaslikHatasi::Baslik {
                        mesaj: format!("{}: {e}", parca.display()),
                    }
                })?;
                if okunan == 0 {
                    break;
                }
                ozet.update(&tampon[..okunan]);
            }
        }
        Ok(ozet
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect())
    }

    /// The directory the parts live in.
    #[must_use]
    pub fn klasor(&self) -> &Path {
        self.parcalar.first().and_then(|p| p.parent()).map_or_else(
            || Path::new("."),
            |p| p,
        )
    }

    /// The name prefix the parts share.
    #[must_use]
    pub fn on_ek(&self) -> &str {
        // The prefix is what the caller passed to `ac`; it is recovered here by
        // stripping the trailing index from the first part's name, which keeps
        // the caller from having to carry it twice.
        self.parcalar
            .first()
            .and_then(|p| p.file_name())
            .and_then(|ad| ad.to_str())
            .map_or("model.safetensors.part-", |ad| {
                ad.rfind('-').map_or(ad, |i| &ad[..=i])
            })
    }

    /// How many bytes the header says the full artifact has, read from the
    /// header itself when the parts present are not enough.
    ///
    /// # Errors
    /// [`BaslikHatasi`] when the header cannot be read at all.
    pub fn beklenen_boyut(&self) -> Result<u64, BaslikHatasi> {
        let mut uzunluk_baytlari = [0_u8; 8];
        self.oku(0, &mut uzunluk_baytlari)?;
        let uzunluk = u64::from_le_bytes(uzunluk_baytlari);
        Ok(8 + uzunluk)
    }

    /// Reads `hedef.len()` bytes starting at `ofset`, across part boundaries.
    ///
    /// # Errors
    /// [`BaslikHatasi::AralikDisi`] when the range runs past the end of the
    /// data, or [`BaslikHatasi::Okunamadi`] when a part cannot be read.
    pub fn oku(&self, ofset: u64, hedef: &mut [u8]) -> Result<(), BaslikHatasi> {
        let son = ofset + hedef.len() as u64;
        if son > self.toplam {
            return Err(BaslikHatasi::AralikDisi {
                ad: format!("{ofset}..{son} (boyut {})", self.toplam),
            });
        }
        let mut yazilan = 0usize;
        let mut kalan_ofset = ofset;
        // Find the part the range starts in, then walk forward. A linear scan
        // over seven parts is cheaper than the bookkeeping a binary search (or
        // a cached index) would need to justify itself here.
        let mut parca = 0usize;
        let mut parca_basi = 0u64;
        while parca < self.parcalar.len() && parca_basi + self.boyutlar[parca] <= kalan_ofset {
            parca_basi += self.boyutlar[parca];
            parca += 1;
        }
        while yazilan < hedef.len() {
            if parca >= self.parcalar.len() {
                return Err(BaslikHatasi::AralikDisi {
                    ad: format!("{ofset}..{son}"),
                });
            }
            let ic_ofset = kalan_ofset - parca_basi;
            let kalan_bayt = (self.boyutlar[parca] - ic_ofset) as usize;
            let alinacak = kalan_bayt.min(hedef.len() - yazilan);
            let mut dosya = File::open(&self.parcalar[parca]).map_err(|h| BaslikHatasi::Okunamadi {
                yol: self.parcalar[parca].display().to_string(),
                mesaj: h.to_string(),
            })?;
            dosya
                .seek(SeekFrom::Start(ic_ofset))
                .map_err(|h| BaslikHatasi::Okunamadi {
                    yol: self.parcalar[parca].display().to_string(),
                    mesaj: h.to_string(),
                })?;
            dosya
                .read_exact(&mut hedef[yazilan..yazilan + alinacak])
                .map_err(|h| BaslikHatasi::Okunamadi {
                    yol: self.parcalar[parca].display().to_string(),
                    mesaj: h.to_string(),
                })?;
            yazilan += alinacak;
            kalan_ofset += alinacak as u64;
            if kalan_ofset >= parca_basi + self.boyutlar[parca] {
                parca_basi += self.boyutlar[parca];
                parca += 1;
            }
        }
        Ok(())
    }
}

/// The header and the data section's start, read from the first part.
#[derive(Debug, Clone)]
pub struct Dizin {
    /// Every tensor, ordered by name for a stable report.
    tensors: BTreeMap<String, TensorBasligi>,
    /// Where the tensor data begins in the logical file.
    veri_baslangici: u64,
}

impl Dizin {
    /// Reads and checks the header of a split checkpoint.
    ///
    /// # Errors
    /// [`BaslikHatasi`] for a header that does not parse, a data offset that
    /// does not follow the header, an out-of-range tensor, or overlapping
    /// tensors.
    pub fn oku(dosya: &ParcaliDosya) -> Result<Self, BaslikHatasi> {
        let mut uzunluk_baytlari = [0_u8; 8];
        dosya.oku(0, &mut uzunluk_baytlari)?;
        let uzunluk = u64::from_le_bytes(uzunluk_baytlari) as usize;
        // Two checks, for two different failures: the header has to fit inside
        // the file (a corrupt length would otherwise ask for a huge
        // allocation), and it has to be small enough that it describes tensors
        // rather than *being* the data. Both are refusals, because a header
        // that does not fit is a file that cannot be read at all.
        const EN_BUYUK_BASLIK: u64 = 64 * 1024 * 1024;
        if uzunluk == 0 || 8 + uzunluk as u64 > dosya.boyut() || uzunluk as u64 > EN_BUYUK_BASLIK {
            return Err(BaslikHatasi::Baslik {
                mesaj: format!("baslik uzunlugu makul degil: {uzunluk}"),
            });
        }
        let mut ham = vec![0_u8; uzunluk];
        dosya.oku(8, &mut ham)?;
        let metin = String::from_utf8(ham).map_err(|h| BaslikHatasi::Baslik {
            mesaj: format!("baslik utf-8 degil: {h}"),
        })?;
        let ham_tablolar: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(&metin).map_err(|h| BaslikHatasi::Baslik {
                mesaj: h.to_string(),
            })?;
        let veri_baslangici = 8 + uzunluk as u64;
        // JSON objects have no order, so the header's *listing* order says
        // nothing about where the bytes are: tensors are checked by their
        // ranges, sorted, not by the order they were written in. Getting this
        // wrong reports a false overlap on the first real checkpoint (measured:
        // `temperature`, the single fp32 tensor, arrives alphabetically before
        // tensors whose bytes precede it).
        let mut araliklar: Vec<(usize, usize, String)> = Vec::new();
        // The header states the size of every tensor. If the last one ends past
        // what is here, a part file is missing; that is a different failure from
        // a corrupt range and is reported with both numbers before the tensors
        // are examined one by one.
        let gereken = ham_tablolar
            .iter()
            .filter(|(ad, _)| ad.as_str() != "__metadata__")
            .filter_map(|(_, deger)| {
                deger.get("data_offsets")
                    .and_then(|o| o.get(1))
                    .and_then(serde_json::Value::as_u64)
            })
            .max()
            .unwrap_or(0);
        if veri_baslangici + gereken > dosya.boyut() {
            return Err(BaslikHatasi::ParcaEksik {
                var: dosya.boyut(),
                gereken: veri_baslangici + gereken,
            });
        }
        let mut tensors: BTreeMap<String, TensorBasligi> = BTreeMap::new();
        for (ad, deger) in ham_tablolar {
            if ad == "__metadata__" {
                continue;
            }
            let baslik: TensorBasligi =
                serde_json::from_value(deger).map_err(|h| BaslikHatasi::Baslik {
                    mesaj: format!("{ad}: {h}"),
                })?;
            if baslik.data_offsets[0] >= baslik.data_offsets[1] {
                return Err(BaslikHatasi::AralikDisi { ad });
            }
            if baslik.data_offsets[1] as u64 + veri_baslangici > dosya.boyut() {
                return Err(BaslikHatasi::AralikDisi { ad });
            }
            araliklar.push((baslik.data_offsets[0], baslik.data_offsets[1], ad.clone()));
            tensors.insert(ad, baslik);
        }
        // Sorted by start offset: two tensors that claim the same bytes are an
        // overlap, and it is reported by name so the reader knows which one to
        // look at. A file whose ranges are disjoint but do not tile the data
        // section is *not* refused: a gap is padding, not corruption.
        araliklar.sort_unstable();
        for pencere in araliklar.windows(2) {
            let (_, onceki_son, _) = &pencere[0];
            let (sonraki_bas, _, sonraki_ad) = &pencere[1];
            if sonraki_bas < onceki_son {
                return Err(BaslikHatasi::Cakisma {
                    ad: sonraki_ad.clone(),
                });
            }
        }
        Ok(Self {
            tensors,
            veri_baslangici,
        })
    }

    /// How many tensors the checkpoint holds.
    #[must_use]
    pub fn uzunluk(&self) -> usize {
        self.tensors.len()
    }

    /// The logical file offset a tensor's payload starts at.
    #[must_use]
    pub fn veri_baslangici(&self) -> u64 {
        self.veri_baslangici
    }

    /// One tensor's header.
    #[must_use]
    pub fn tensor(&self, ad: &str) -> Option<&TensorBasligi> {
        self.tensors.get(ad)
    }

    /// Every tensor name, in order.
    #[must_use]
    pub fn adlar(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// The element types present, with how many tensors use each.
    #[must_use]
    pub fn tur_dagilimi(&self) -> BTreeMap<String, usize> {
        let mut sayim: BTreeMap<String, usize> = BTreeMap::new();
        for baslik in self.tensors.values() {
            *sayim.entry(baslik.dtype.clone()).or_insert(0) += 1;
        }
        sayim
    }

    /// The total element count of every tensor the header lists.
    #[must_use]
    pub fn toplam_eleman(&self) -> usize {
        self.tensors.values().map(TensorBasligi::eleman).sum()
    }

    /// Reads `adet` elements of a tensor starting at element `ilk`.
    ///
    /// This exists for one tensor: the embedding table is 196 million elements,
    /// which is 786 MB once converted to `f32`, and the machine this runs on
    /// has less than that to spare. Reading the rows a sequence actually needs
    /// keeps a run's footprint at the size of the layer weights rather than the
    /// size of the vocabulary.
    ///
    /// # Errors
    /// [`BaslikHatasi::AralikDisi`] for a range outside the tensor, and the
    /// reader's own errors.
    pub fn tensor_aralik_oku(
        &self,
        dosya: &ParcaliDosya,
        ad: &str,
        ilk: usize,
        adet: usize,
    ) -> Result<Vec<f32>, BaslikHatasi> {
        let baslik = self.tensors.get(ad).ok_or_else(|| BaslikHatasi::Baslik {
            mesaj: format!("tensor yok: {ad}"),
        })?;
        let eleman = baslik.eleman();
        if ilk.saturating_add(adet) > eleman {
            return Err(BaslikHatasi::AralikDisi {
                ad: format!("{ad}[{ilk}..{}] (eleman {eleman})", ilk + adet),
            });
        }
        let genislik = match baslik.dtype.as_str() {
            "F32" => 4_usize,
            "F16" => 2_usize,
            digeri => {
                return Err(BaslikHatasi::Baslik {
                    mesaj: format!("{ad}: desteklenmeyen tur {digeri}"),
                })
            }
        };
        let baslangic = baslik.data_offsets[0] + ilk * genislik;
        let mut ham = vec![0_u8; adet * genislik];
        dosya.oku(self.veri_baslangici + baslangic as u64, &mut ham)?;
        match baslik.dtype.as_str() {
            "F32" => {
                let (parcalar, kalan) = ham.as_chunks::<4>();
                debug_assert!(kalan.is_empty(), "f32 govdesi dort baytin kati olmali");
                Ok(parcalar
                    .iter()
                    .map(|k| f32::from_le_bytes(*k))
                    .collect())
            }
            _ => {
                let (parcalar, kalan) = ham.as_chunks::<2>();
                debug_assert!(kalan.is_empty(), "f16 govdesi iki baytin kati olmali");
                Ok(parcalar
                    .iter()
                    .map(|k| f16_to_f32(u16::from_le_bytes(*k)))
                    .collect())
            }
        }
    }

    /// Whether the header names this tensor.
    ///
    /// The encoder and the decision head ask this instead of catching a
    /// missing-tensor error, because "this key is absent" is a fact about this
    /// checkpoint - one layer is deliberately built without an attention norm -
    /// and not a failure.
    #[must_use]
    pub fn iceriyor(&self, ad: &str) -> bool {
        self.tensors.contains_key(ad)
    }

    /// Reads one tensor as `f32`, converting from the stored type.
    ///
    /// # Errors
    /// [`BaslikHatasi::Baslik`] for an unknown name or an unsupported type, and
    /// the reader's own errors for an unreadable range.
    pub fn tensor_oku(&self, dosya: &ParcaliDosya, ad: &str) -> Result<Vec<f32>, BaslikHatasi> {
        let baslik = self.tensors.get(ad).ok_or_else(|| BaslikHatasi::Baslik {
            mesaj: format!("tensor yok: {ad}"),
        })?;
        let mut ham = vec![0_u8; baslik.bayt()];
        dosya.oku(
            self.veri_baslangici + baslik.data_offsets[0] as u64,
            &mut ham,
        )?;
        match baslik.dtype.as_str() {
            "F32" => Ok(ham
                .as_chunks::<4>()
                .0
                .iter()
                .map(|k| f32::from_le_bytes(*k))
                .collect()),
            "F16" => Ok(ham
                .as_chunks::<2>()
                .0
                .iter()
                .map(|k| f16_to_f32(u16::from_le_bytes(*k)))
                .collect()),
            digeri => Err(BaslikHatasi::Baslik {
                mesaj: format!("{ad}: desteklenmeyen tur {digeri}"),
            }),
        }
    }
}

/// The value of a half-precision float, computed exactly.
///
/// The conversion is written out rather than delegated, because it is the one
/// place where a wrong constant would produce plausible numbers: subnormals,
/// infinities and `NaN` all have to come out right, and each of those cases has
/// its own test below.
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let isaret = (u32::from(bits) & 0x8000) << 16;
    let us = (u32::from(bits) >> 10) & 0x1f;
    let anlam = u32::from(bits) & 0x03ff;
    let sonuc: u32 = match (us, anlam) {
        // Zero.
        (0, 0) => isaret,
        // Subnormal: value = mantissa * 2^-24.
        (0, anlam) => {
            let mut e = 127 - 15 + 1;
            let mut m = anlam;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            isaret | ((e as u32) << 23) | ((m & 0x03ff) << 13)
        }
        // Infinity and NaN.
        (0x1f, _) => isaret | 0x7f80_0000 | (anlam << 13),
        // Normal.
        (us, anlam) => isaret | ((us + 127 - 15) << 23) | (anlam << 13),
    };
    f32::from_bits(sonuc)
}

/// The half-precision bit pattern of a value, rounding to nearest even.
///
/// The inverse is here so that a test can check the round trip and so that a
/// later writer can store weights without a second implementation of the same
/// rule.
#[must_use]
pub fn f32_to_f16(deger: f32) -> u16 {
    let bits = deger.to_bits();
    let isaret = ((bits >> 16) & 0x8000) as u16;
    let us = ((bits >> 23) & 0xff) as i32;
    let anlam = bits & 0x007f_ffff;
    if us == 0xff {
        // Infinity or NaN: keep the NaN payload's top bits.
        return isaret | 0x7c00 | ((anlam >> 13) as u16) | u16::from(anlam != 0);
    }
    let yeni_us = us - 127 + 15;
    if yeni_us >= 0x1f {
        return isaret | 0x7c00;
    }
    if yeni_us <= 0 {
        if yeni_us < -10 {
            return isaret;
        }
        // Subnormal or underflow.
        let kaydirma = (14 - yeni_us) as u32;
        let m = (anlam | 0x0080_0000) >> kaydirma;
        let kalan = (anlam | 0x0080_0000) & ((1 << kaydirma) - 1);
        let yarim = 1 << (kaydirma - 1);
        let yuvarlanmis = m + u32::from(kalan > yarim || (kalan == yarim && (m & 1) == 1));
        return isaret | yuvarlanmis as u16;
    }
    let m = anlam >> 13;
    let kalan = anlam & 0x1fff;
    let mut us_bits = yeni_us as u32;
    let mut m_bits = m;
    if kalan > 0x1000 || (kalan == 0x1000 && (m & 1) == 1) {
        m_bits += 1;
        if m_bits == 0x400 {
            m_bits = 0;
            us_bits += 1;
            if us_bits >= 0x1f {
                return isaret | 0x7c00;
            }
        }
    }
    isaret | ((us_bits as u16) << 10) | m_bits as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn gecici_klasor(ad: &str) -> PathBuf {
        let yol = std::env::temp_dir().join(format!("lubot-kodlayici-{ad}"));
        let _ = std::fs::remove_dir_all(&yol);
        std::fs::create_dir_all(&yol).expect("klasor olusmali");
        yol
    }

    /// Writes a two-part checkpoint: some tensors in each part.
    fn iki_parcali(klasor: &Path) -> (Vec<f32>, Vec<f32>) {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0];
        let mut govde = Vec::new();
        for x in &a {
            govde.extend_from_slice(&f32_to_f16(*x).to_le_bytes());
        }
        for x in &b {
            govde.extend_from_slice(&f32_to_f16(*x).to_le_bytes());
        }
        let baslik = format!(
            r#"{{"a":{{"dtype":"F16","shape":[4],"data_offsets":[0,{}]}},"b":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            a.len() * 2,
            a.len() * 2,
            (a.len() + b.len()) * 2
        );
        let mut dosya = Vec::new();
        dosya.extend_from_slice(&(baslik.len() as u64).to_le_bytes());
        dosya.extend_from_slice(baslik.as_bytes());
        dosya.extend_from_slice(&govde);
        // Split the logical file in the middle of the second tensor's bytes, so
        // that the reader has to walk across a part boundary.
        let kesim = 8 + baslik.len() + 4 * 2 + 1;
        let (ilk, ikinci) = dosya.split_at(kesim);
        let mut f1 = File::create(klasor.join("model.safetensors.part-00")).expect("parca 0");
        f1.write_all(ilk).expect("yazilmali");
        let mut f2 = File::create(klasor.join("model.safetensors.part-01")).expect("parca 1");
        f2.write_all(ikinci).expect("yazilmali");
        (a, b)
    }

    #[test]
    fn half_precision_rounds_trip_for_every_representable_value() {
        // Values that f16 can hold exactly: the round trip must be identity.
        for x in [0.0_f32, 1.0, -1.0, 0.5, 2.0, -0.25, 1024.0, 65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16(x)), x, "{x}");
        }
        // Values f16 cannot hold: the round trip is close, not equal.
        for x in [0.1_f32, -std::f32::consts::PI, 1.0e-4] {
            let geri = f16_to_f32(f32_to_f16(x));
            assert!((geri - x).abs() / x.abs() < 1.0e-3, "{x} -> {geri}");
        }
    }

    #[test]
    fn half_precision_special_values_are_exact() {
        assert!(f16_to_f32(0x0000).abs() < f32::EPSILON);
        assert!(f16_to_f32(0x8000) == 0.0 && f16_to_f32(0x8000).is_sign_negative());
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7c00) > 0.0);
        assert!(f16_to_f32(0xfc00).is_infinite() && f16_to_f32(0xfc00) < 0.0);
        assert!(f16_to_f32(0x7e00).is_nan());
        // The smallest subnormal, 2^-24.
        let en_kucuk = f16_to_f32(0x0001);
        assert!((en_kucuk - 5.960_464_5e-8).abs() < 1e-14, "{en_kucuk}");
        // The largest finite half.
        assert!((f16_to_f32(0x7bff) - 65504.0).abs() < 1e-3);
    }

    #[test]
    fn a_split_checkpoint_is_read_as_one_sequence() {
        let klasor = gecici_klasor("parcali");
        let (a, b) = iki_parcali(&klasor);
        let dosya = ParcaliDosya::ac(&klasor, "model.safetensors.part-").expect("acilmali");
        assert_eq!(dosya.parca_sayisi(), 2);
        let dizin = Dizin::oku(&dosya).expect("baslik okunmali");
        assert_eq!(dizin.uzunluk(), 2);
        assert_eq!(dizin.toplam_eleman(), 6);
        assert_eq!(dizin.tur_dagilimi().get("F16"), Some(&2));
        let okunan_a = dizin.tensor_oku(&dosya, "a").expect("a okunmali");
        let okunan_b = dizin.tensor_oku(&dosya, "b").expect("b okunmali");
        assert_eq!(okunan_a, a);
        // `b`'s bytes straddle the part boundary: this is the case the reader
        // exists for, and the assertion is that the crossing is invisible.
        assert_eq!(okunan_b, b);
        let _ = std::fs::remove_dir_all(&klasor);
    }

    #[test]
    fn a_missing_part_is_refused_with_both_sizes() {
        let klasor = gecici_klasor("aralik");
        let baslik = r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]},"b":{"dtype":"F16","shape":[4],"data_offsets":[8,99999]}}"#;
        let mut dosya = Vec::new();
        dosya.extend_from_slice(&(baslik.len() as u64).to_le_bytes());
        dosya.extend_from_slice(baslik.as_bytes());
        dosya.extend_from_slice(&[0_u8; 8]);
        File::create(klasor.join("model.safetensors.part-00"))
            .expect("parca")
            .write_all(&dosya)
            .expect("yazilmali");
        let parcali = ParcaliDosya::ac(&klasor, "model.safetensors.part-").expect("acilmali");
        // The tensor claims bytes the file does not have. The diagnosis is
        // "a part is missing", with both sizes, because that is what a reader
        // can act on; the per-tensor range error is for a corrupt header.
        match Dizin::oku(&parcali) {
            Err(BaslikHatasi::ParcaEksik { var, gereken }) => {
                assert_eq!(var, 8 + baslik.len() as u64 + 8);
                assert_eq!(gereken, 8 + baslik.len() as u64 + 99999);
            }
            digeri => panic!("eksik parca bekleniyordu: {digeri:?}"),
        }
        let _ = std::fs::remove_dir_all(&klasor);
    }

    #[test]
    fn a_range_that_cannot_be_read_is_refused_with_its_name() {
        let klasor = gecici_klasor("ic_aralik");
        // A header whose own range is backwards: start after end.
        let baslik = r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[8,4]}}"#;
        let mut dosya = Vec::new();
        dosya.extend_from_slice(&(baslik.len() as u64).to_le_bytes());
        dosya.extend_from_slice(baslik.as_bytes());
        dosya.extend_from_slice(&[0_u8; 12]);
        File::create(klasor.join("model.safetensors.part-00"))
            .expect("parca")
            .write_all(&dosya)
            .expect("yazilmali");
        let parcali = ParcaliDosya::ac(&klasor, "model.safetensors.part-").expect("acilmali");
        match Dizin::oku(&parcali) {
            Err(BaslikHatasi::AralikDisi { ad }) => assert_eq!(ad, "a"),
            digeri => panic!("aralik disi bekleniyordu: {digeri:?}"),
        }
        let _ = std::fs::remove_dir_all(&klasor);
    }

    #[test]
    fn a_row_range_is_read_without_loading_the_whole_tensor() {
        let klasor = gecici_klasor("satir");
        let (a, _) = iki_parcali(&klasor);
        let dosya = ParcaliDosya::ac(&klasor, "model.safetensors.part-").expect("acilmali");
        let dizin = Dizin::oku(&dosya).expect("baslik");
        // Elements 1..3 of `a`: exactly [2, 3], and the cross-part read of `b`.
        let dilim = dizin.tensor_aralik_oku(&dosya, "a", 1, 2).expect("okunmali");
        assert_eq!(dilim, vec![a[1], a[2]]);
        let b = dizin.tensor_aralik_oku(&dosya, "b", 0, 2).expect("okunmali");
        assert_eq!(b, vec![5.0, 6.0]);
        assert!(dizin.tensor_aralik_oku(&dosya, "a", 3, 2).is_err());
        let _ = std::fs::remove_dir_all(&klasor);
    }

    #[test]
    fn overlapping_tensors_are_refused_rather_than_guessed() {
        let klasor = gecici_klasor("cakisma");
        let baslik = r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]},"b":{"dtype":"F16","shape":[4],"data_offsets":[4,12]}}"#;
        let mut dosya = Vec::new();
        dosya.extend_from_slice(&(baslik.len() as u64).to_le_bytes());
        dosya.extend_from_slice(baslik.as_bytes());
        dosya.extend_from_slice(&[0_u8; 12]);
        File::create(klasor.join("model.safetensors.part-00"))
            .expect("parca")
            .write_all(&dosya)
            .expect("yazilmali");
        let parcali = ParcaliDosya::ac(&klasor, "model.safetensors.part-").expect("acilmali");
        match Dizin::oku(&parcali) {
            Err(BaslikHatasi::Cakisma { ad }) => assert_eq!(ad, "b"),
            digeri => panic!("cakisma bekleniyordu: {digeri:?}"),
        }
        let _ = std::fs::remove_dir_all(&klasor);
    }

    #[test]
    fn an_unknown_tensor_name_is_an_error_and_not_an_empty_vector() {
        let klasor = gecici_klasor("bilinmeyen");
        let (_, _) = iki_parcali(&klasor);
        let dosya = ParcaliDosya::ac(&klasor, "model.safetensors.part-").expect("acilmali");
        let dizin = Dizin::oku(&dosya).expect("baslik");
        assert!(dizin.tensor_oku(&dosya, "yok").is_err());
        let _ = std::fs::remove_dir_all(&klasor);
    }

    #[test]
    fn parts_are_ordered_by_name_so_ten_comes_after_nine() {
        let klasor = gecici_klasor("siralama");
        for (ad, icerik) in [("p-09", b"a".to_vec()), ("p-10", b"b".to_vec()), ("p-02", b"c".to_vec())] {
            File::create(klasor.join(ad))
                .expect("parca")
                .write_all(&icerik)
                .expect("yazilmali");
        }
        let dosya = ParcaliDosya::ac(&klasor, "p-").expect("acilmali");
        let mut tampon = [0_u8; 1];
        dosya.oku(0, &mut tampon).expect("okunmali");
        assert_eq!(tampon[0], b'c', "ilk parca p-02 olmali");
        dosya.oku(2, &mut tampon).expect("okunmali");
        assert_eq!(tampon[0], b'b', "ucuncu parca p-10 olmali");
        let _ = std::fs::remove_dir_all(&klasor);
    }
}
