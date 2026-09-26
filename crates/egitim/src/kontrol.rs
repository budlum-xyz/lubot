//! The checkpoint format: one file that is the same file twice.
//!
//! # Why the format is this fussy
//!
//! A checkpoint is not a cache of weights, it is a *claim about a run*: these
//! weights, at this step, from this corpus, with this vocabulary family, under
//! this seed. Everything downstream - a resumed run, an exam, a comparison
//! against an earlier round - is only meaningful if that claim travels with the
//! numbers. So the file carries the run's identity in a header, and it carries
//! a SHA-256 over every byte before it, because a checkpoint with one flipped
//! byte loads perfectly well and produces a slightly different model.
//!
//! # What is written, in order
//!
//! ```text
//! LUBOTCKPT | version:u8 | precision:u8 | flags:u16le | header_len:u32le
//! header JSON | named blocks | sha256(everything before)
//! ```
//!
//! The header is JSON because it has to be readable by a human deciding
//! whether a checkpoint is the one they want; the blocks are binary because a
//! million weights in JSON is a megabyte of digits. The precision byte is not a
//! hint: it says whether the block values are `f64` or `f32`, and a reader that
//! guessed would turn a storage decision into a different model.
//!
//! # Order of reading
//!
//! The fixed-width fields are read *before* the header, so a file with a
//! damaged precision byte is refused as a precision problem rather than as a
//! JSON problem. That ordering is visible in the error values
//! ([`KontrolHatasi::Hassasiyet`], [`KontrolHatasi::Bayrak`]) rather than being
//! an implementation detail, because "the file is damaged here" and "the header
//! is damaged somewhere" are different things to whoever has to fix it.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::kosu::KosuRaporu;
use crate::{Adamw, Parametreler, Spec};

/// Magic bytes: a checkpoint that does not start with these is not one.
pub const SIHIR: &[u8; 9] = b"LUBOTCKPT";
/// Format version. A reader refuses a version it does not know.
pub const SURUM: u8 = 1;
/// Trailing digest length, in bytes. How much of the file is the check on
/// the rest of it: part of the format, not part of its public surface.
pub(crate) const OZET_UZUNLUK: usize = 32;

/// How the block values were stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hassasiyet {
    /// Every weight as an `f64`.
    F64,
    /// Every weight rounded to `f32` and stored in four bytes.
    F32,
}

impl Hassasiyet {
    /// The byte the format writes.
    #[must_use]
    pub fn kod(self) -> u8 {
        match self {
            Self::F64 => 0,
            Self::F32 => 1,
        }
    }

    /// How many bytes one weight takes.
    #[must_use]
    pub fn bayt(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
        }
    }

    /// The label reports use.
    #[must_use]
    pub fn etiket(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
        }
    }

    fn koddan(kod: u8) -> Option<Self> {
        match kod {
            0 => Some(Self::F64),
            1 => Some(Self::F32),
            _ => None,
        }
    }
}

/// The optimizer's state, so a run can continue where it stopped.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerDurumu {
    /// Steps taken.
    pub adim: u64,
    /// Learning rate at that step.
    pub ogrenme_orani: f64,
    /// Weight decay.
    pub agirlik_sonumu: f64,
    /// First moments.
    pub m: Vec<f64>,
    /// Second moments.
    pub v: Vec<f64>,
}

/// Why a checkpoint was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum KontrolHatasi {
    /// The file does not start with [`SIHIR`].
    Sihir,
    /// The version byte names a format this reader does not know.
    Surum(u8),
    /// The precision byte is neither `f64` nor `f32`.
    Hassasiyet(u8),
    /// The flag word carries a bit whose meaning is not defined here.
    Bayrak(u16),
    /// The header is not the JSON this format writes.
    Baslik(String),
    /// A block's name or length does not match the format's order.
    Blok(String),
    /// The blocks do not rebuild the spec's shape.
    Sekil,
    /// The trailing digest does not match the bytes before it.
    Ozet { beklenen: String, bulunan: String },
    /// The file ends before the digest does.
    Kisa { gerekli: usize, var: usize },
    /// The optimizer's moments do not cover the parameters.
    Optimizer { oge: usize, m: usize, v: usize },
    /// The file could not be read or written.
    Io(String),
}

impl std::fmt::Display for KontrolHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sihir => write!(f, "dosya LUBOTCKPT ile baslamiyor"),
            Self::Surum(v) => write!(f, "bilinmeyen surum {v} (bu okuyucu {SURUM})"),
            Self::Hassasiyet(b) => write!(f, "bilinmeyen hassasiyet bayti {b} (0=f64, 1=f32)"),
            Self::Bayrak(b) => write!(f, "tanimsiz bayrak biti: 0x{b:04x}"),
            Self::Baslik(s) => write!(f, "baslik okunamadi: {s}"),
            Self::Blok(s) => write!(f, "blok okunamadi: {s}"),
            Self::Sekil => write!(f, "bloklar spec'in sekline uymuyor"),
            Self::Ozet { beklenen, bulunan } => {
                write!(f, "ozet tutmuyor: beklenen {beklenen}, dosyada {bulunan}")
            }
            Self::Kisa { gerekli, var } => {
                write!(f, "dosya kisa: {gerekli} bayt gerekli, {var} var")
            }
            Self::Optimizer { oge, m, v } => write!(
                f,
                "optimiser durumu parametreleri kaplamiyor: {oge} parametre, {m} ve {v} moment"
            ),
            Self::Io(s) => write!(f, "{s}"),
        }
    }
}

/// Everything a later reader needs to know about the run behind the weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Kontrol {
    /// The architecture.
    pub spec: Spec,
    /// The weights.
    pub parametreler: Parametreler,
    /// Step the run reached.
    pub adim: u64,
    /// Epoch the run completed.
    pub epoch: u32,
    /// The run's seed.
    pub tohum: u64,
    /// Vocabulary family the run tokenised with.
    pub sozluk_aile: String,
    /// Digest of the `content_id`s the run read.
    pub korpus_ozeti: String,
    /// Last training loss.
    pub egitim_kaybi: f64,
    /// Last validation loss, if one was measured.
    pub dogrulama_kaybi: Option<f64>,
    /// Best validation loss reached.
    pub en_iyi_dogrulama: Option<f64>,
    /// How far into the next epoch the run got. A checkpoint that says only
    /// "epoch 3" cannot continue epoch 3: the windows it already read would be
    /// read again.
    pub devam_konum: usize,
    /// How the values were stored.
    pub hassasiyet: Hassasiyet,
    /// The optimizer's state, if the run carried it.
    pub optimizer: Option<OptimizerDurumu>,
}

impl Kontrol {
    /// Build a checkpoint from the state a finished run left behind.
    ///
    /// The step, epoch, seed and measurements come from the run's own report,
    /// not from the caller's memory of them: a checkpoint that disagrees with
    /// the run that produced it can be neither resumed nor compared, and the
    /// disagreement would show up only as a loss curve that moved.
    ///
    /// The optimizer's moments are carried along, because a checkpoint without
    /// them can start a fresh run but cannot continue the one it came from -
    /// and the difference between those two is a measurable one.
    #[must_use]
    pub fn kosudan(
        rapor: &KosuRaporu,
        parametreler: &Parametreler,
        optimizer: &Adamw,
        sozluk_aile: &str,
        korpus_ozeti: &str,
        hassasiyet: Hassasiyet,
    ) -> Self {
        let (adim, m, v) = optimizer.durum();
        let son_dogrulama = rapor.dogrulama_egrisi.last().map(|d| d.kayip);
        Self {
            spec: rapor.ayar.spec,
            parametreler: parametreler.clone(),
            adim: rapor.adim,
            epoch: rapor.epoch,
            tohum: rapor.ayar.tohum,
            sozluk_aile: sozluk_aile.to_string(),
            korpus_ozeti: korpus_ozeti.to_string(),
            egitim_kaybi: rapor.son_kaybi,
            dogrulama_kaybi: son_dogrulama,
            en_iyi_dogrulama: rapor.en_iyi_dogrulama,
            devam_konum: rapor.devam_konum,
            hassasiyet,
            optimizer: Some(OptimizerDurumu {
                adim,
                ogrenme_orani: optimizer.ogrenme_orani,
                agirlik_sonumu: optimizer.agirlik_sonumu,
                m: m.to_vec(),
                v: v.to_vec(),
            }),
        }
    }

    /// Write the checkpoint, returning the SHA-256 of the bytes written.
    ///
    /// The digest is computed over the same bytes that go to disk, in the same
    /// order, so the returned value is the file's identity and not a second
    /// computation that could drift from it.
    ///
    /// # Errors
    /// [`KontrolHatasi::Io`] when the file cannot be written,
    /// [`KontrolHatasi::Sekil`] when the tensors are not the spec's,
    /// [`KontrolHatasi::Optimizer`] when the moment vectors do not match the
    /// parameter count.
    pub fn yaz(&self, yol: &Path) -> Result<String, KontrolHatasi> {
        let mut govde = self.baytlar()?;
        let ozet = ozetle(&govde);
        govde.extend_from_slice(&ozet);
        std::fs::write(yol, &govde).map_err(|e| KontrolHatasi::Io(e.to_string()))?;
        Ok(hex(&ozet))
    }

    /// Load a checkpoint and verify it against its own digest.
    ///
    /// # Errors
    /// [`KontrolHatasi::Io`] when the file cannot be read; otherwise whatever
    /// [`Kontrol::baytlardan`] refuses.
    pub fn yukle(yol: &Path) -> Result<Self, KontrolHatasi> {
        let ham = std::fs::read(yol).map_err(|e| KontrolHatasi::Io(e.to_string()))?;
        Self::baytlardan(&ham)
    }

    /// The bytes before the digest: what [`Kontrol::yaz`] writes, minus the
    /// trailing SHA-256.
    ///
    /// # Errors
    /// See [`Kontrol::yaz`].
    pub fn baytlar(&self) -> Result<Vec<u8>, KontrolHatasi> {
        if !self.parametreler.sekil_dogru(self.spec) {
            return Err(KontrolHatasi::Sekil);
        }
        let oge = self.parametreler.toplam_ogeler();
        if let Some(opt) = &self.optimizer {
            if opt.m.len() != oge || opt.v.len() != oge {
                return Err(KontrolHatasi::Optimizer {
                    oge,
                    m: opt.m.len(),
                    v: opt.v.len(),
                });
            }
        }
        // The fixed-width head first: a reader can refuse a damaged precision
        // byte without parsing anything.
        let mut govde: Vec<u8> = Vec::new();
        govde.extend_from_slice(SIHIR);
        govde.push(SURUM);
        govde.push(self.hassasiyet.kod());
        govde.extend_from_slice(&0u16.to_le_bytes());
        let baslik = self.baslik_json();
        let baslik_baytlari = baslik.as_bytes();
        govde.extend_from_slice(&(baslik_baytlari.len() as u32).to_le_bytes());
        govde.extend_from_slice(baslik_baytlari);
        for (ad, blok) in self.parametreler.bloklar_adli() {
            govde.extend_from_slice(&(ad.len() as u16).to_le_bytes());
            govde.extend_from_slice(ad.as_bytes());
            govde.extend_from_slice(&(blok.len() as u64).to_le_bytes());
            for deger in blok {
                match self.hassasiyet {
                    Hassasiyet::F64 => govde.extend_from_slice(&deger.to_le_bytes()),
                    Hassasiyet::F32 => govde.extend_from_slice(&(*deger as f32).to_le_bytes()),
                }
            }
        }
        // The moments follow the weights, in the same encoding and with the
        // same naming rule: a checkpoint that carried only their count could
        // be loaded but not continued, and "could not continue" is exactly the
        // difference a resumed run is supposed to eliminate.
        if let Some(opt) = &self.optimizer {
            for (ad, blok) in [("optimizer.m", &opt.m), ("optimizer.v", &opt.v)] {
                govde.extend_from_slice(&(ad.len() as u16).to_le_bytes());
                govde.extend_from_slice(ad.as_bytes());
                govde.extend_from_slice(&(blok.len() as u64).to_le_bytes());
                for deger in blok {
                    match self.hassasiyet {
                        Hassasiyet::F64 => govde.extend_from_slice(&deger.to_le_bytes()),
                        Hassasiyet::F32 => govde.extend_from_slice(&(*deger as f32).to_le_bytes()),
                    }
                }
            }
        }
        Ok(govde)
    }

    fn baslik_json(&self) -> String {
        let spec = self.spec;
        let kayit = serde_json::json!({
            "surum": SURUM,
            "spec": {
                "vocab": spec.vocab,
                "d_model": spec.d_model,
                "n_layers": spec.n_layers,
                "n_heads": spec.n_heads,
                "n_kv_heads": spec.n_kv_heads,
                "d_ff": spec.d_ff,
                "max_seq_len": spec.max_seq_len,
            },
            "adim": self.adim,
            "epoch": self.epoch,
            "tohum": self.tohum,
            "sozluk_aile": self.sozluk_aile,
            "korpus_ozeti": self.korpus_ozeti,
            "egitim_kaybi": self.egitim_kaybi,
            "dogrulama_kaybi": self.dogrulama_kaybi,
            "en_iyi_dogrulama": self.en_iyi_dogrulama,
            "devam_konum": self.devam_konum,
            "hassasiyet": self.hassasiyet.etiket(),
            "optimizer": self.optimizer.as_ref().map(|o| serde_json::json!({
                "adim": o.adim,
                "ogrenme_orani": o.ogrenme_orani,
                "agirlik_sonumu": o.agirlik_sonumu,
                "oge": o.m.len(),
            })),
        });
        kayit.to_string()
    }

    /// Parse and verify a checkpoint from bytes.
    ///
    /// # Errors
    /// Every refusal in [`KontrolHatasi`]: the fixed-width fields first, then
    /// the header, then the blocks, then the digest.
    pub fn baytlardan(ham: &[u8]) -> Result<Self, KontrolHatasi> {
        if ham.len() < SIHIR.len() + 1 {
            return Err(KontrolHatasi::Kisa {
                gerekli: SIHIR.len() + 1,
                var: ham.len(),
            });
        }
        if &ham[..SIHIR.len()] != SIHIR {
            return Err(KontrolHatasi::Sihir);
        }
        let mut konum = SIHIR.len();
        let surum = *ham.get(konum).ok_or(KontrolHatasi::Kisa {
            gerekli: konum + 1,
            var: ham.len(),
        })?;
        konum += 1;
        if surum != SURUM {
            return Err(KontrolHatasi::Surum(surum));
        }
        let hassasiyet_kodu = *ham.get(konum).ok_or(KontrolHatasi::Kisa {
            gerekli: konum + 1,
            var: ham.len(),
        })?;
        konum += 1;
        let hassasiyet = Hassasiyet::koddan(hassasiyet_kodu)
            .ok_or(KontrolHatasi::Hassasiyet(hassasiyet_kodu))?;
        let bayrak = oku_u16(ham, &mut konum)?;
        if bayrak != 0 {
            return Err(KontrolHatasi::Bayrak(bayrak));
        }
        let baslik_uzunluk = oku_u32(ham, &mut konum)? as usize;
        if konum + baslik_uzunluk > ham.len() {
            return Err(KontrolHatasi::Kisa {
                gerekli: konum + baslik_uzunluk,
                var: ham.len(),
            });
        }
        let baslik_metni = std::str::from_utf8(&ham[konum..konum + baslik_uzunluk])
            .map_err(|e| KontrolHatasi::Baslik(e.to_string()))?;
        konum += baslik_uzunluk;
        let baslik: serde_json::Value =
            serde_json::from_str(baslik_metni).map_err(|e| KontrolHatasi::Baslik(e.to_string()))?;

        let spec = spec_oku(&baslik)?;
        let blok_bayt = hassasiyet.bayt();
        let mut parametreler = Parametreler::sifir(spec);
        let mut okunan: Vec<(&'static str, usize)> = Vec::new();
        for ad in Parametreler::blok_adlari() {
            let ad_uzunluk = oku_u16(ham, &mut konum)? as usize;
            if konum + ad_uzunluk > ham.len() {
                return Err(KontrolHatasi::Kisa {
                    gerekli: konum + ad_uzunluk,
                    var: ham.len(),
                });
            }
            let yazilan_ad = std::str::from_utf8(&ham[konum..konum + ad_uzunluk])
                .map_err(|e| KontrolHatasi::Blok(e.to_string()))?;
            konum += ad_uzunluk;
            if yazilan_ad != ad {
                return Err(KontrolHatasi::Blok(format!(
                    "beklenen `{ad}`, dosyada `{yazilan_ad}`"
                )));
            }
            let oge = oku_u64(ham, &mut konum)? as usize;
            let gerekli = konum + oge * blok_bayt;
            if gerekli > ham.len() {
                return Err(KontrolHatasi::Kisa {
                    gerekli,
                    var: ham.len(),
                });
            }
            let mut degerler: Vec<f64> = Vec::with_capacity(oge);
            for _ in 0..oge {
                let parca = &ham[konum..konum + blok_bayt];
                degerler.push(match hassasiyet {
                    Hassasiyet::F64 => f64::from_le_bytes(parca.try_into().map_err(|_| {
                        KontrolHatasi::Blok(format!("`{ad}` f64 olarak okunamadi"))
                    })?),
                    Hassasiyet::F32 => {
                        f64::from(f32::from_le_bytes(parca.try_into().map_err(|_| {
                            KontrolHatasi::Blok(format!("`{ad}` f32 olarak okunamadi"))
                        })?))
                    }
                });
                konum += blok_bayt;
            }
            okunan.push((ad, oge));
            if !parametreler.blok_ata(ad, degerler) {
                return Err(KontrolHatasi::Sekil);
            }
        }
        if !parametreler.sekil_dogru(spec) {
            return Err(KontrolHatasi::Sekil);
        }
        let oge = parametreler.toplam_ogeler();
        let optimizer = optimizer_oku(&baslik, oge)?;
        let optimizer = match optimizer {
            None => None,
            Some(mut o) => {
                o.m = blok_oku(ham, &mut konum, "optimizer.m", oge, hassasiyet)?;
                o.v = blok_oku(ham, &mut konum, "optimizer.v", oge, hassasiyet)?;
                Some(o)
            }
        };

        if ham.len() < konum + OZET_UZUNLUK {
            return Err(KontrolHatasi::Kisa {
                gerekli: konum + OZET_UZUNLUK,
                var: ham.len(),
            });
        }
        let beklenen = ozetle(&ham[..konum]);
        let bulunan = &ham[konum..konum + OZET_UZUNLUK];
        if bulunan != beklenen.as_slice() {
            return Err(KontrolHatasi::Ozet {
                beklenen: hex(&beklenen),
                bulunan: hex(bulunan),
            });
        }

        Ok(Self {
            spec,
            parametreler,
            adim: baslik
                .get("adim")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| KontrolHatasi::Baslik("`adim` yok".to_string()))?,
            epoch: baslik
                .get("epoch")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| KontrolHatasi::Baslik("`epoch` yok".to_string()))?
                as u32,
            tohum: baslik
                .get("tohum")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| KontrolHatasi::Baslik("`tohum` yok".to_string()))?,
            sozluk_aile: baslik
                .get("sozluk_aile")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| KontrolHatasi::Baslik("`sozluk_aile` yok".to_string()))?
                .to_string(),
            korpus_ozeti: baslik
                .get("korpus_ozeti")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| KontrolHatasi::Baslik("`korpus_ozeti` yok".to_string()))?
                .to_string(),
            egitim_kaybi: baslik
                .get("egitim_kaybi")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| KontrolHatasi::Baslik("`egitim_kaybi` yok".to_string()))?,
            dogrulama_kaybi: baslik
                .get("dogrulama_kaybi")
                .and_then(serde_json::Value::as_f64),
            en_iyi_dogrulama: baslik
                .get("en_iyi_dogrulama")
                .and_then(serde_json::Value::as_f64),
            devam_konum: baslik
                .get("devam_konum")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize,
            hassasiyet,
            optimizer,
        })
    }

    /// Whether the file stored full precision.
    #[must_use]
    pub fn hassas(&self) -> bool {
        self.hassasiyet == Hassasiyet::F64
    }

    /// The same checkpoint at `f32` precision.
    ///
    /// The weights are rounded and the label says so; the returned value is what
    /// would be written, not an alternative view of the same numbers.
    #[must_use]
    pub fn f32_kopya(&self) -> Self {
        let mut kopya = self.clone();
        kopya.hassasiyet = Hassasiyet::F32;
        kopya.parametreler.yuvarla_f32();
        kopya
    }

    /// The optimizer rebuilt from the file.
    ///
    /// `None` when the checkpoint was written without one - which is a
    /// different thing from "resume from the start", and the caller is expected
    /// to say which it means.
    #[must_use]
    pub fn optimizer_yeniden(&self) -> Option<Adamw> {
        let o = self.optimizer.as_ref()?;
        Adamw::durumdan(
            o.m.clone(),
            o.v.clone(),
            o.adim,
            o.ogrenme_orani,
            o.agirlik_sonumu,
        )
        .ok()
    }
}

/// Read one named block of `oge` values at the file's precision.
fn blok_oku(
    ham: &[u8],
    konum: &mut usize,
    beklenen_ad: &str,
    oge: usize,
    hassasiyet: Hassasiyet,
) -> Result<Vec<f64>, KontrolHatasi> {
    let ad_uzunluk = oku_u16(ham, konum)? as usize;
    if *konum + ad_uzunluk > ham.len() {
        return Err(KontrolHatasi::Kisa {
            gerekli: *konum + ad_uzunluk,
            var: ham.len(),
        });
    }
    let ad = std::str::from_utf8(&ham[*konum..*konum + ad_uzunluk])
        .map_err(|e| KontrolHatasi::Blok(e.to_string()))?;
    *konum += ad_uzunluk;
    if ad != beklenen_ad {
        return Err(KontrolHatasi::Blok(format!(
            "beklenen `{beklenen_ad}`, dosyada `{ad}`"
        )));
    }
    let yazilan = oku_u64(ham, konum)? as usize;
    if yazilan != oge {
        return Err(KontrolHatasi::Blok(format!(
            "`{beklenen_ad}` {yazilan} deger tasiyor, {oge} bekleniyordu"
        )));
    }
    let blok_bayt = hassasiyet.bayt();
    let gerekli = *konum + oge * blok_bayt;
    if gerekli > ham.len() {
        return Err(KontrolHatasi::Kisa {
            gerekli,
            var: ham.len(),
        });
    }
    let mut degerler: Vec<f64> = Vec::with_capacity(oge);
    for _ in 0..oge {
        let parca = &ham[*konum..*konum + blok_bayt];
        degerler.push(match hassasiyet {
            Hassasiyet::F64 => f64::from_le_bytes(parca.try_into().map_err(|_| {
                KontrolHatasi::Blok(format!("`{beklenen_ad}` f64 olarak okunamadi"))
            })?),
            Hassasiyet::F32 => f64::from(f32::from_le_bytes(parca.try_into().map_err(|_| {
                KontrolHatasi::Blok(format!("`{beklenen_ad}` f32 olarak okunamadi"))
            })?)),
        });
        *konum += blok_bayt;
    }
    Ok(degerler)
}

fn spec_oku(baslik: &serde_json::Value) -> Result<Spec, KontrolHatasi> {
    let alan = |ad: &str| -> Result<usize, KontrolHatasi> {
        baslik
            .get("spec")
            .and_then(|s| s.get(ad))
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| KontrolHatasi::Baslik(format!("spec.{ad} yok")))
    };
    let n_heads = alan("n_heads")?;
    // Eski kontrol noktalarinda KV paylasimi yoktu: alan yoksa tam dikkat
    // demektir ve dosya oldugu gibi okunmaya devam eder. Yeni alan yalnizca
    // paylasimli bir spec yazildiginda anlami vardir.
    let n_kv_heads = baslik
        .get("spec")
        .and_then(|s| s.get("n_kv_heads"))
        .and_then(serde_json::Value::as_u64)
        .map_or(n_heads, |v| v as usize);
    let spec = Spec {
        vocab: alan("vocab")?,
        d_model: alan("d_model")?,
        n_layers: alan("n_layers")?,
        n_heads,
        n_kv_heads,
        d_ff: alan("d_ff")?,
        max_seq_len: alan("max_seq_len")?,
    };
    spec.dogrula().map_err(|_| KontrolHatasi::Sekil)?;
    Ok(spec)
}

fn optimizer_oku(
    baslik: &serde_json::Value,
    oge: usize,
) -> Result<Option<OptimizerDurumu>, KontrolHatasi> {
    let Some(o) = baslik.get("optimizer") else {
        return Ok(None);
    };
    if o.is_null() {
        return Ok(None);
    }
    let oge_beyan = o
        .get("oge")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| KontrolHatasi::Baslik("optimizer.oge yok".to_string()))?
        as usize;
    if oge_beyan != oge {
        return Err(KontrolHatasi::Optimizer {
            oge,
            m: oge_beyan,
            v: oge_beyan,
        });
    }
    Ok(Some(OptimizerDurumu {
        adim: o
            .get("adim")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| KontrolHatasi::Baslik("optimizer.adim yok".to_string()))?,
        ogrenme_orani: o
            .get("ogrenme_orani")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| KontrolHatasi::Baslik("optimizer.ogrenme_orani yok".to_string()))?,
        agirlik_sonumu: o
            .get("agirlik_sonumu")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| KontrolHatasi::Baslik("optimizer.agirlik_sonumu yok".to_string()))?,
        m: Vec::new(),
        v: Vec::new(),
    }))
}

fn oku_u16(ham: &[u8], konum: &mut usize) -> Result<u16, KontrolHatasi> {
    let parca = ham.get(*konum..*konum + 2).ok_or(KontrolHatasi::Kisa {
        gerekli: *konum + 2,
        var: ham.len(),
    })?;
    *konum += 2;
    Ok(u16::from_le_bytes(parca.try_into().map_err(|_| {
        KontrolHatasi::Kisa {
            gerekli: *konum,
            var: ham.len(),
        }
    })?))
}

fn oku_u32(ham: &[u8], konum: &mut usize) -> Result<u32, KontrolHatasi> {
    let parca = ham.get(*konum..*konum + 4).ok_or(KontrolHatasi::Kisa {
        gerekli: *konum + 4,
        var: ham.len(),
    })?;
    *konum += 4;
    Ok(u32::from_le_bytes(parca.try_into().map_err(|_| {
        KontrolHatasi::Kisa {
            gerekli: *konum,
            var: ham.len(),
        }
    })?))
}

fn oku_u64(ham: &[u8], konum: &mut usize) -> Result<u64, KontrolHatasi> {
    let parca = ham.get(*konum..*konum + 8).ok_or(KontrolHatasi::Kisa {
        gerekli: *konum + 8,
        var: ham.len(),
    })?;
    *konum += 8;
    Ok(u64::from_le_bytes(parca.try_into().map_err(|_| {
        KontrolHatasi::Kisa {
            gerekli: *konum,
            var: ham.len(),
        }
    })?))
}

fn ozetle(baytlar: &[u8]) -> [u8; OZET_UZUNLUK] {
    let mut ozet = Sha256::new();
    ozet.update(baytlar);
    let cikti = ozet.finalize();
    let mut dizi = [0u8; OZET_UZUNLUK];
    dizi.copy_from_slice(&cikti);
    dizi
}

/// Lowercase hex, the form the header and the reports use.
#[must_use]
pub fn hex(baytlar: &[u8]) -> String {
    let mut s = String::with_capacity(baytlar.len() * 2);
    for b in baytlar {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::veri::Kayit;
    use crate::INIT_STD_EMBEDDING;

    fn spec() -> Spec {
        Spec {
            vocab: 16,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            d_ff: 16,
            max_seq_len: 8,
        }
    }

    /// The bytes a file holds: the body plus the digest the writer appends.
    fn tam(k: &Kontrol) -> Vec<u8> {
        let mut ham = k.baytlar().expect("bytes");
        ham.extend_from_slice(&ozetle(&ham));
        ham
    }

    fn kontrol() -> Kontrol {
        let spec = spec();
        let p = Parametreler::mup_init(spec, 9, INIT_STD_EMBEDDING);
        let opt = Adamw::yeni(p.toplam_ogeler(), 0.01, 0.1).expect("optimizer");
        let (adim, m, v) = opt.durum();
        Kontrol {
            spec,
            parametreler: p,
            adim: 7,
            epoch: 2,
            tohum: 9,
            sozluk_aile: "lubot-bpe-v2".to_string(),
            korpus_ozeti: "a".repeat(64),
            egitim_kaybi: 3.5,
            dogrulama_kaybi: Some(3.6),
            en_iyi_dogrulama: Some(3.4),
            devam_konum: 5,
            hassasiyet: Hassasiyet::F64,
            optimizer: Some(OptimizerDurumu {
                adim,
                ogrenme_orani: 0.01,
                agirlik_sonumu: 0.1,
                m: m.to_vec(),
                v: v.to_vec(),
            }),
        }
    }

    #[test]
    fn a_checkpoint_round_trips_through_its_bytes() {
        let k = kontrol();
        let ham = tam(&k);
        let geri = Kontrol::baytlardan(&ham).expect("load");
        assert_eq!(geri.spec, k.spec);
        assert_eq!(geri.parametreler, k.parametreler);
        assert_eq!(geri.adim, k.adim);
        assert_eq!(geri.epoch, k.epoch);
        assert_eq!(geri.tohum, k.tohum);
        assert_eq!(geri.sozluk_aile, k.sozluk_aile);
        assert_eq!(geri.korpus_ozeti, k.korpus_ozeti);
        assert_eq!(geri.egitim_kaybi, k.egitim_kaybi);
        assert_eq!(geri.en_iyi_dogrulama, k.en_iyi_dogrulama);
        assert_eq!(geri.devam_konum, k.devam_konum);
        assert_eq!(geri.hassasiyet, Hassasiyet::F64);
        let o = geri.optimizer.expect("optimizer");
        assert_eq!(o.m, k.optimizer.as_ref().expect("o").m);
        assert_eq!(o.v, k.optimizer.as_ref().expect("o").v);
    }

    #[test]
    fn a_written_file_carries_its_own_digest() {
        let dizin = std::env::temp_dir().join("lubot-kontrol-testi");
        std::fs::create_dir_all(&dizin).expect("temp dir");
        let yol = dizin.join("a.ckpt");
        let k = kontrol();
        let ozet = k.yaz(&yol).expect("write");
        let ham = std::fs::read(&yol).expect("read");
        assert_eq!(&ham[..4], b"LUBO");
        assert_eq!(ham.len(), k.baytlar().expect("bytes").len() + OZET_UZUNLUK);
        assert_eq!(ozet, hex(&ozetle(&ham[..ham.len() - OZET_UZUNLUK])));
        assert_eq!(Kontrol::yukle(&yol).expect("load").adim, k.adim);
        std::fs::remove_file(&yol).ok();
    }

    #[test]
    fn a_flipped_byte_is_refused_by_the_digest() {
        let k = kontrol();
        let mut ham = tam(&k);
        let orta = ham.len() / 2;
        ham[orta] ^= 0x01;
        match Kontrol::baytlardan(&ham) {
            Err(KontrolHatasi::Ozet { .. }) => {}
            other => panic!("bozuk bayt kabul edildi: {other:?}"),
        }
    }

    #[test]
    fn a_damaged_fixed_width_field_is_named_before_the_header() {
        let k = kontrol();
        let ham = tam(&k);
        let mut hassasiyet_bozuk = ham.clone();
        hassasiyet_bozuk[SIHIR.len() + 1] = 7;
        assert_eq!(
            Kontrol::baytlardan(&hassasiyet_bozuk),
            Err(KontrolHatasi::Hassasiyet(7))
        );
        let mut bayrak_bozuk = ham.clone();
        bayrak_bozuk[SIHIR.len() + 2] = 1;
        assert_eq!(
            Kontrol::baytlardan(&bayrak_bozuk),
            Err(KontrolHatasi::Bayrak(1))
        );
    }

    #[test]
    fn an_unknown_version_or_magic_is_refused() {
        let k = kontrol();
        let ham = tam(&k);
        let mut surum = ham.clone();
        surum[SIHIR.len()] = 9;
        assert_eq!(Kontrol::baytlardan(&surum), Err(KontrolHatasi::Surum(9)));
        let mut sihir = ham.clone();
        sihir[0] = b'X';
        assert_eq!(Kontrol::baytlardan(&sihir), Err(KontrolHatasi::Sihir));
        assert_eq!(
            Kontrol::baytlardan(&ham[..3]),
            Err(KontrolHatasi::Kisa {
                gerekli: SIHIR.len() + 1,
                var: 3
            })
        );
    }

    #[test]
    fn a_truncated_file_is_refused_as_short() {
        let k = kontrol();
        let ham = k.baytlar().expect("bytes");
        for kes in [ham.len() - 1, ham.len() / 2, 30] {
            match Kontrol::baytlardan(&ham[..kes]) {
                Err(KontrolHatasi::Kisa { .. }) => {}
                other => panic!("kesik dosya ({kes}) kabul edildi: {other:?}"),
            }
        }
    }

    #[test]
    fn the_f32_copy_says_what_it_is_and_rounds() {
        let k = kontrol();
        let dusuk = k.f32_kopya();
        assert_eq!(dusuk.hassasiyet, Hassasiyet::F32);
        assert_eq!(dusuk.hassasiyet.bayt(), 4);
        assert!(!dusuk.hassas());
        assert!(k.hassas());
        let ham = tam(&dusuk);
        let geri = Kontrol::baytlardan(&ham).expect("load");
        assert_eq!(geri.hassasiyet, Hassasiyet::F32);
        assert_eq!(geri.parametreler, dusuk.parametreler);
        assert!(
            ham.len() < tam(&k).len(),
            "f32 dosyasi f64 dosyasindan kucuk olmali"
        );
    }

    #[test]
    fn the_optimizer_rebuilds_where_the_run_stopped() {
        let k = kontrol();
        let opt = k.optimizer_yeniden().expect("optimizer");
        let (adim, m, v) = opt.durum();
        assert_eq!(adim, 0);
        assert_eq!(m.len(), k.parametreler.toplam_ogeler());
        assert_eq!(v.len(), m.len());
        let mut kopya = k.clone();
        kopya.optimizer = None;
        assert!(kopya.optimizer_yeniden().is_none());
    }

    #[test]
    fn a_shape_that_does_not_match_the_spec_is_refused_on_write() {
        let mut k = kontrol();
        k.parametreler.embedding.pop();
        assert_eq!(k.baytlar(), Err(KontrolHatasi::Sekil));
        let mut k = kontrol();
        if let Some(o) = k.optimizer.as_mut() {
            o.m.pop();
        }
        assert!(matches!(k.baytlar(), Err(KontrolHatasi::Optimizer { .. })));
    }

    #[test]
    fn a_run_hands_its_own_report_to_the_checkpoint() {
        // Kosudan: raporun adimi, epoch'u, tohumu ve kayiplari kontrol
        // noktasina gecer; cagiranin hatirladigi degil, kosunun soyledigi.
        let spec = spec();
        let mut p = Parametreler::mup_init(spec, 4, INIT_STD_EMBEDDING);
        let mut opt = Adamw::yeni(p.toplam_ogeler(), 0.02, 0.1).expect("optimizer");
        let kayitlar: Vec<Kayit> = (0..4)
            .map(|i| Kayit {
                kimlik: format!("k-{i}"),
                jetonlar: (0..40).map(|j| ((i + j) % 15) as u32).collect(),
            })
            .collect();
        let bolum = crate::veri::bolumle(kayitlar, 0.25).expect("split");
        let (egitim, dogrulama) = crate::veri::pencereler(&bolum, 8).expect("windows");
        let ayar = crate::kosu::KosuAyari {
            spec,
            pencere_uzunlugu: 8,
            tohum: 4,
            ogrenme_orani: 0.02,
            agirlik_sonumu: 0.1,
            isinma_adimi: 1,
            toplam_adim: 4,
            planlanan_adim: 4,
            baslangic_adim: 0,
            baslangic_en_iyi: None,
            baslangic_epoch: 0,
            baslangic_konum: 0,
            yigin: 1,
            hassasiyet: Hassasiyet::F64,
            iplik: 1,
            kirpma: 1.0,
            dogrulama_her: 2,
            epoch_tavani: 2,
        };
        let rapor =
            crate::kosu::egitim_kosu(&ayar, &egitim, &dogrulama, &mut p, &mut opt, |_, _| {})
                .expect("run");
        let k = Kontrol::kosudan(
            &rapor,
            &p,
            &opt,
            "test-aile",
            &"b".repeat(64),
            Hassasiyet::F64,
        );
        assert_eq!(k.adim, rapor.adim);
        assert_eq!(k.epoch, rapor.epoch);
        assert_eq!(k.tohum, 4);
        assert_eq!(k.egitim_kaybi, rapor.son_kaybi);
        assert_eq!(k.en_iyi_dogrulama, rapor.en_iyi_dogrulama);
        assert_eq!(k.sozluk_aile, "test-aile");
        let ham = tam(&k);
        let geri = Kontrol::baytlardan(&ham).expect("load");
        assert_eq!(geri.parametreler, p);
    }
}
