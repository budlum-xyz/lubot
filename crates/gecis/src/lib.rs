#![forbid(unsafe_code)]
//! # lubot-gecis - the transition line: a source tree in, a plan out
//!
//! Bringing an outside repository into this workspace by hand costs a day per
//! component, and almost none of that day is the hard part. The hard part is
//! realising the idea in Rust, measured; the rest is inventory (what files,
//! what symbols, what imports), plumbing (the workspace member line, the cli
//! dependency line, the CRATES row, the gate registration) and bookkeeping
//! (what is realised, what is still waiting). This crate automates exactly
//! that rest, and refuses to automate the hard part:
//!
//! 1. [`say`] - a census. Walks a source tree deterministically, records file
//!    kinds, line counts, public symbols and imports, and folds them into one
//!    tree digest. The census is reproducible: the same bytes in the same
//!    paths give the same digest, so a plan can name the source it was built
//!    from.
//! 2. [`plan`] - a realization plan. One module per source file that carries
//!    symbols, with the symbols as the contract, imports split into in-tree
//!    and external, and the plumbing lines written in this repository's own
//!    style - the gate checks them byte against the real lines, so a drift in
//!    either direction is caught.
//! 3. [`durum`] - coverage of the realized crate, at module level by default
//!    and at symbol level when a name mapping is given. A module that exists
//!    but covers none of its contract is a fact this reports rather than
//!    hides.
//!
//! What this crate does not do is generate bodies. A generated stub would
//! compile, pass the lints and carry no idea; the realization is written by
//! hand under the contract the plan states, and the coverage number says how
//! much of that contract is still open. The census also never records the
//! source tree's root name: paths are stored relative to it, because which
//! repository a component came from is provenance, not architecture.
//!
//! Limits, stated rather than smoothed over: a directory-level package import
//! matches by directory name, a comma-separated import records only its first
//! name, and symbol-level coverage needs an explicit mapping because source
//! and realized names are in different languages.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use lubot_alim::manifest::LICENCES;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Directory names a census refuses to enter. They are build output, version
/// control or interpreter state: none of them is source, and walking them
/// would make the census depend on how often the source was built.
pub const ATLAMA: [&str; 5] = [".git", "target", "node_modules", "__pycache__", ".venv"];

/// The limits a census enforces, so a source tree cannot turn the walk into
/// an unbounded one. A refusal names the limit it hit, with the count seen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sinirlar {
    /// En çok kaç dosya kaydedilir.
    pub en_cok_dosya: usize,
    /// En çok kaç sembol toplanır.
    pub en_cok_sembol: usize,
    /// Bir dosyanın izin verilen en büyük bayt uzunluğu.
    pub en_cok_bayt: u64,
}

/// Varsayılan sınır lar: 4096 dosya, 8192 sembol, 16 MiB.
pub const SINIRLAR: Sinirlar = Sinirlar {
    en_cok_dosya: 4096,
    en_cok_sembol: 8192,
    en_cok_bayt: 16 * 1024 * 1024,
};

/// Bir reddin adı, reddi veren kuralla birlikte. Her reddin kendini adlandırması
/// gerekir; adı olmayan red, operatöre hiçbir şey söylemez.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GecisHatasi {
    /// Kaynak yok ya da dizin değil.
    KaynakYok(String),
    /// Sembolik bağ izlenmez: sayım bir ağacın özeti, bir bağın hedefi değil.
    BagIzlenmez(String),
    /// Dosya sayısı sınırı aşıldı.
    DosyaSiniri { sinir: usize, gorulen: usize },
    /// Tek bir dosya bayt sınırını aştı.
    BaytSiniri { yol: String, sinir: u64, boyut: u64 },
    /// Sembol sayısı sınırı aşıldı.
    SembolSiniri { sinir: usize, gorulen: usize },
    /// Ağaçta hiç dosya kalmadı: plan boş olurdu.
    BosAgac(String),
    /// Lisans kapalı kümede değil.
    LisansKapali(String),
    /// Hedef ad küçük harf, rakam ve tire dışında bir şey taşıyor ya da harfle başlamıyor.
    AdGecersiz(String),
    /// İki kaynak dosyası aynı modül adına indirgendi.
    AdCakismasi {
        ad: String,
        birinci: String,
        ikinci: String,
    },
    /// Dosya okunamadı.
    Okunamadi { yol: String, neden: String },
}

impl fmt::Display for GecisHatasi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GecisHatasi::KaynakYok(yol) => write!(f, "kaynak yok ya da dizin degil: {yol}"),
            GecisHatasi::BagIzlenmez(yol) => write!(f, "sembolik bag izlenmez: {yol}"),
            GecisHatasi::DosyaSiniri { sinir, gorulen } => {
                write!(
                    f,
                    "dosya sayisi siniri asildi: sinir {sinir}, gorulen {gorulen}"
                )
            }
            GecisHatasi::BaytSiniri { yol, sinir, boyut } => {
                write!(
                    f,
                    "dosya bayt siniri asildi: {yol} {boyut} bayt (sinir {sinir})"
                )
            }
            GecisHatasi::SembolSiniri { sinir, gorulen } => {
                write!(
                    f,
                    "sembol sayisi siniri asildi: sinir {sinir}, gorulen {gorulen}"
                )
            }
            GecisHatasi::BosAgac(yol) => write!(f, "kaynak agac bos: {yol}"),
            GecisHatasi::LisansKapali(lisans) => write!(f, "lisans kapali kumede degil: {lisans}"),
            GecisHatasi::AdGecersiz(ad) => {
                write!(
                    f,
                    "ad gecersiz (kucuk harf, rakam, tire; harf ya da rakamla baslar): {ad}"
                )
            }
            GecisHatasi::AdCakismasi {
                ad,
                birinci,
                ikinci,
            } => {
                write!(f, "modul adi cakisti: {ad} ({birinci} ve {ikinci})")
            }
            GecisHatasi::Okunamadi { yol, neden } => write!(f, "okunamadi: {yol}: {neden}"),
        }
    }
}

impl std::error::Error for GecisHatasi {}

/// Bir dosyanın cinsinden ne olduğu. Cins, sayımın ne tarayacağını belirler:
/// kod taranır, veri sayılır, ikili yalnızca özetlenir.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DosyaTuru {
    /// Rust kaynak dosyası.
    Rust,
    /// Python kaynak dosyası.
    Python,
    /// Markdown belgesi.
    Markdown,
    /// JSON verisi.
    Json,
    /// TOML verisi.
    Toml,
    /// YAML verisi.
    Yaml,
    /// Uzantısı bilinmeyen ama ikili olmayan dosya.
    Metin,
    /// İlk 256 baytında NUL taşıyan dosya.
    Ikili,
}

impl DosyaTuru {
    /// Cinsin kayıttaki adı.
    pub fn as_str(self) -> &'static str {
        match self {
            DosyaTuru::Rust => "rust",
            DosyaTuru::Python => "python",
            DosyaTuru::Markdown => "markdown",
            DosyaTuru::Json => "json",
            DosyaTuru::Toml => "toml",
            DosyaTuru::Yaml => "yaml",
            DosyaTuru::Metin => "metin",
            DosyaTuru::Ikili => "ikili",
        }
    }
}

/// Bir sembolün cinsi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SembolTuru {
    /// `fn` ya da `def`.
    Fonksiyon,
    /// `struct` ya da `type`.
    Yapi,
    /// `enum`.
    Enum,
    /// `const` ya da Python'da büyük harfli sabit.
    Sabit,
    /// `trait`.
    Trait,
    /// Python `class`.
    Sinif,
    /// Markdown başlığı.
    Baslik,
}

/// Kaynakta yakalanan bir sembol: adı, cinsi ve satırı.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sembol {
    /// Sembolün adı.
    pub ad: String,
    /// Cinsi.
    pub tur: SembolTuru,
    /// Bir tabanlı satır numarası.
    pub satir: u32,
}

/// Sayımda yer alan bir dosya.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DosyaKaydi {
    /// Kaynak köküne göre göreli yol, eğik çizgilerle.
    pub yol: String,
    /// Cinsi.
    pub tur: DosyaTuru,
    /// Satır sayısı; ikili dosyada 0.
    pub satir: u32,
    /// İçeriğin sha256 özeti, küçük harfli onaltılık.
    pub ozet: String,
    /// Yakalanan semboller.
    pub semboller: Vec<Sembol>,
    /// İçe aktarma adlarının ilk bölümleri, görüldükleri sırayla.
    pub alimlar: Vec<String>,
}

/// Bir kaynak ağacının sayımı.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sayim {
    /// Kök dizinin adı. Bilgi olarak taşınır; ağaç özetine girmez.
    pub kok_adi: String,
    /// Sayımın dayandığı lisans; alım hattının kapalı kümesiyle eşlenir.
    pub lisans: String,
    /// Dosyalar, yola göre sıralı.
    pub dosyalar: Vec<DosyaKaydi>,
    /// Gezilen dizinlerin göreli yolları, sıralı.
    pub dizinler: Vec<String>,
    /// Ağaç özeti: dosya yolu ve özetinden üretilen tek sha256.
    pub agac_ozeti: String,
    /// Metin dosyalarının toplam satırı.
    pub toplam_satir: u32,
}

/// Planlanan bir modül: bir kaynak dosyanın sözleşmesi.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModulPlani {
    /// Normalleştirilmiş modül adı (tire ayraçlı).
    pub ad: String,
    /// Kaynak dosyanın göreli yolu.
    pub kaynak_yol: String,
    /// Sözleşme: gerçeklenmesi beklenen sembol adları.
    pub semboller: Vec<String>,
    /// Ağaç içinde eşleşen içe aktarmalar.
    pub ic_alimlar: Vec<String>,
    /// Dışarıya giden içe aktarmalar.
    pub dis_alimlar: Vec<String>,
}

/// Planın ürettiği tesisat satırları. Ağacın kendi biçiminde yazılırlar; kapı
/// bunları gerçek satırlarla bayt bayt karşılaştırır.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tesisat {
    /// Kök `Cargo.toml` üye satırı.
    pub uye_satiri: String,
    /// `crates/cli/Cargo.toml` bağımlılık satırı.
    pub cli_bagimliligi: String,
    /// `docs/CRATES.md` satır şablonu; ölçüm doldurur.
    pub crates_satiri: String,
    /// `gates/check.py` kayıt satırı şablonu.
    pub kapi_kaydi: String,
}

/// Bir sayımdan kurulan gerçekleme planı.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// Hedef crate adı.
    pub hedef_crate: String,
    /// Sayımın lisansı.
    pub lisans: String,
    /// Planın dayandığı ağaç özeti.
    pub kaynak_ozeti: String,
    /// Modül planları, ada göre sıralı.
    pub moduller: Vec<ModulPlani>,
    /// Veri dosyaları (json, toml, yaml, markdown), yola göre sıralı.
    pub veri_dosyalari: Vec<String>,
    /// Sözleşmenin toplam sembol sayısı: herkese açık her sembol için en az
    /// bir test yazılması hedeflenir. Hedef, iddia değil.
    pub test_amaci: u32,
    /// Tesisat satırları.
    pub tesisat: Tesisat,
}

/// Gerçekleşen crate'te bir modülün durumu.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModulDurumu {
    /// Modül adı.
    pub ad: String,
    /// Karşılık gelen `src/<ad>.rs` var mı.
    pub var: bool,
    /// Gerçekleşen dosyadaki `pub` öğe sayısı.
    pub sembol: u32,
    /// Gerçekleşen dosyadaki test işlevlerinin sayısı.
    pub test: u32,
    /// Eşleme verildiyse, eşlenen sembollerin kapsamı.
    pub simge_kapsami: Option<f32>,
}

/// Bir planın gerçekleşme kapsamı.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Durum {
    /// Planın hedef crate adı.
    pub hedef_crate: String,
    /// Planın ağaç özeti.
    pub plan_ozeti: String,
    /// Modül durumları, planın sırasıyla.
    pub moduller: Vec<ModulDurumu>,
    /// Modül düzeyi kapsam: var olan modüllerin oranı.
    pub modul_kapsami: f32,
    /// Eşleme verildiyse sembol düzeyi kapsam; verilmediyse `None`.
    pub simge_kapsami: Option<f32>,
    /// Henüz gerçekleşmemiş modüllerin adları.
    pub bekleyen: Vec<String>,
}

/// `bytes` içindeki `--ad` değerini döndürür; yoksa `None`.
/// Bir kaynak ağacını varsayılan sınırlarla sayar.
///
/// # Hatalar
///
/// Kaynak yoksa [`GecisHatasi::KaynakYok`], lisans kapalı kümede değilse
/// [`GecisHatasi::LisansKapali`], sembolik bağ görülse
/// [`GecisHatasi::BagIzlenmez`], sınırlar aşılırsa adı geçen sınır hatası.
pub fn say(kok: &Path, lisans: &str) -> Result<Sayim, GecisHatasi> {
    say_sinirli(kok, lisans, &SINIRLAR)
}

/// Bir kaynak ağacını verilen sınırlarla sayar.
///
/// # Hatalar
///
/// [`say`] ile aynı; ek olarak verilen sınırlar aşıldığında o sınırın adını
/// taşıyan hata döner.
pub fn say_sinirli(kok: &Path, lisans: &str, sinir: &Sinirlar) -> Result<Sayim, GecisHatasi> {
    if !LICENCES.contains(&lisans) {
        return Err(GecisHatasi::LisansKapali(lisans.to_string()));
    }
    let meta = fs::symlink_metadata(kok).map_err(|e| meta_hatasi(kok, &e))?;
    if meta.file_type().is_symlink() {
        return Err(GecisHatasi::BagIzlenmez(kok.display().to_string()));
    }
    if !meta.is_dir() {
        return Err(GecisHatasi::KaynakYok(kok.display().to_string()));
    }
    let mut dosyalar = Vec::new();
    let mut dizinler = Vec::new();
    let mut sembol_sayisi = 0usize;
    gez(
        kok,
        kok,
        sinir,
        &mut dosyalar,
        &mut dizinler,
        &mut sembol_sayisi,
    )?;
    if dosyalar.is_empty() {
        return Err(GecisHatasi::BosAgac(kok.display().to_string()));
    }
    dosyalar.sort_by(|a, b| a.yol.cmp(&b.yol));
    dizinler.sort();
    let agac_ozeti = agac_ozeti(&dosyalar);
    let toplam_satir = dosyalar.iter().map(|d| d.satir).sum();
    Ok(Sayim {
        kok_adi: kok
            .file_name()
            .map(|a| a.to_string_lossy().into_owned())
            .unwrap_or_default(),
        lisans: lisans.to_string(),
        dosyalar,
        dizinler,
        agac_ozeti,
        toplam_satir,
    })
}

/// Metadata okunamadiysa ayir: olmayan bir yol `KaynakYok`'tur, diger her
/// hata `Okunamadi` ve nedenini tasir.
fn meta_hatasi(yol: &Path, e: &std::io::Error) -> GecisHatasi {
    if e.kind() == std::io::ErrorKind::NotFound {
        GecisHatasi::KaynakYok(yol.display().to_string())
    } else {
        GecisHatasi::Okunamadi {
            yol: yol.display().to_string(),
            neden: e.to_string(),
        }
    }
}

fn gez(
    kok: &Path,
    dizin: &Path,
    sinir: &Sinirlar,
    dosyalar: &mut Vec<DosyaKaydi>,
    dizinler: &mut Vec<String>,
    sembol_sayisi: &mut usize,
) -> Result<(), GecisHatasi> {
    let mut adlar: Vec<PathBuf> = fs::read_dir(dizin)
        .map_err(|e| GecisHatasi::Okunamadi {
            yol: dizin.display().to_string(),
            neden: e.to_string(),
        })?
        .filter_map(|girdi| girdi.ok().map(|g| g.path()))
        .collect();
    adlar.sort();
    for yol in adlar {
        let meta = fs::symlink_metadata(&yol).map_err(|e| GecisHatasi::Okunamadi {
            yol: yol.display().to_string(),
            neden: e.to_string(),
        })?;
        if meta.file_type().is_symlink() {
            return Err(GecisHatasi::BagIzlenmez(yol.display().to_string()));
        }
        if meta.is_dir() {
            let ad = yol
                .file_name()
                .map(|a| a.to_string_lossy().into_owned())
                .unwrap_or_default();
            if ATLAMA.contains(&ad.as_str()) {
                continue;
            }
            dizinler.push(goreli_yol(kok, &yol));
            gez(kok, &yol, sinir, dosyalar, dizinler, sembol_sayisi)?;
        } else {
            if dosyalar.len() >= sinir.en_cok_dosya {
                return Err(GecisHatasi::DosyaSiniri {
                    sinir: sinir.en_cok_dosya,
                    gorulen: dosyalar.len() + 1,
                });
            }
            let kayit = dosya_kaydi(kok, &yol, sinir)?;
            *sembol_sayisi += kayit.semboller.len();
            if *sembol_sayisi > sinir.en_cok_sembol {
                return Err(GecisHatasi::SembolSiniri {
                    sinir: sinir.en_cok_sembol,
                    gorulen: *sembol_sayisi,
                });
            }
            dosyalar.push(kayit);
        }
    }
    Ok(())
}

fn dosya_kaydi(kok: &Path, yol: &Path, sinir: &Sinirlar) -> Result<DosyaKaydi, GecisHatasi> {
    let bayt = fs::read(yol).map_err(|e| GecisHatasi::Okunamadi {
        yol: yol.display().to_string(),
        neden: e.to_string(),
    })?;
    if bayt.len() as u64 > sinir.en_cok_bayt {
        return Err(GecisHatasi::BaytSiniri {
            yol: goreli_yol(kok, yol),
            sinir: sinir.en_cok_bayt,
            boyut: bayt.len() as u64,
        });
    }
    let tur = tur_belirle(yol, &bayt);
    let satir = if tur == DosyaTuru::Ikili {
        0
    } else {
        satir_say(&bayt)
    };
    let (semboller, alimlar) = match tur {
        DosyaTuru::Rust | DosyaTuru::Python | DosyaTuru::Markdown => tara(&bayt, tur),
        _ => (Vec::new(), Vec::new()),
    };
    Ok(DosyaKaydi {
        yol: goreli_yol(kok, yol),
        tur,
        satir,
        ozet: sha256_hex(&bayt),
        semboller,
        alimlar,
    })
}

/// Bir dosyanın cinsini uzantısından, bilinmeyen uzantıda ilk 256 baytın
/// NUL taşıyıp taşımadığından belirler.
fn tur_belirle(yol: &Path, bayt: &[u8]) -> DosyaTuru {
    match yol
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("rs") => return DosyaTuru::Rust,
        Some("py") => return DosyaTuru::Python,
        Some("md") => return DosyaTuru::Markdown,
        Some("json") => return DosyaTuru::Json,
        Some("toml") => return DosyaTuru::Toml,
        Some("yaml" | "yml") => return DosyaTuru::Yaml,
        Some("txt") => return DosyaTuru::Metin,
        _ => {}
    }
    if bayt.iter().take(256).any(|b| *b == 0) {
        DosyaTuru::Ikili
    } else {
        DosyaTuru::Metin
    }
}

fn satir_say(bayt: &[u8]) -> u32 {
    let yeni_satir = bayt.iter().filter(|b| **b == b'\n').count() as u32;
    match bayt.last() {
        Some(b) if *b != b'\n' => yeni_satir + 1,
        _ => yeni_satir,
    }
}

/// Metin dosyasının sembollerini ve içe aktarmalarını tarar. Yalnızca satır
/// başlangıçlarına bakar; bir ifadenin ortasındaki ad kayda girmez.
fn tara(bayt: &[u8], tur: DosyaTuru) -> (Vec<Sembol>, Vec<String>) {
    let metin = String::from_utf8_lossy(bayt);
    let mut semboller = Vec::new();
    let mut alimlar = Vec::new();
    for (no, satir) in metin.lines().enumerate() {
        let girinti = satir.len() - satir.trim_start().len();
        let k = satir.trim_start();
        let satir_no = no as u32 + 1;
        match tur {
            DosyaTuru::Rust => {
                if let Some((ad, tur)) = rust_sembolu(k) {
                    semboller.push(Sembol {
                        ad,
                        tur,
                        satir: satir_no,
                    });
                }
                if let Some(rest) = k.strip_prefix("use ") {
                    let ilk = ilk_kelime(rest);
                    if !ilk.is_empty() {
                        alimlar.push(ilk);
                    }
                }
            }
            DosyaTuru::Python => {
                if girinti == 0 {
                    if let Some(ad) = k.strip_prefix("def ") {
                        semboller.push(Sembol {
                            ad: ilk_kelime(ad),
                            tur: SembolTuru::Fonksiyon,
                            satir: satir_no,
                        });
                    } else if let Some(ad) = k.strip_prefix("async def ") {
                        semboller.push(Sembol {
                            ad: ilk_kelime(ad),
                            tur: SembolTuru::Fonksiyon,
                            satir: satir_no,
                        });
                    } else if let Some(ad) = k.strip_prefix("class ") {
                        semboller.push(Sembol {
                            ad: ilk_kelime(ad),
                            tur: SembolTuru::Sinif,
                            satir: satir_no,
                        });
                    } else if let Some(ad) = python_sabiti(k) {
                        semboller.push(Sembol {
                            ad,
                            tur: SembolTuru::Sabit,
                            satir: satir_no,
                        });
                    }
                }
                if let Some(rest) = k.strip_prefix("import ") {
                    let ilk = ilk_kelime(rest);
                    if !ilk.is_empty() {
                        alimlar.push(ilk);
                    }
                } else if let Some(rest) = k.strip_prefix("from ") {
                    let ilk = ilk_kelime(rest);
                    if !ilk.is_empty() {
                        alimlar.push(ilk);
                    }
                }
            }
            DosyaTuru::Markdown => {
                if k.starts_with('#') {
                    let ad = k.trim_matches('#').trim();
                    if !ad.is_empty() {
                        semboller.push(Sembol {
                            ad: ad.to_string(),
                            tur: SembolTuru::Baslik,
                            satir: satir_no,
                        });
                    }
                }
            }
            _ => {}
        }
    }
    (semboller, alimlar)
}

/// Rust'ta sayımın gördüğü herkese açık öğe önekleri ve cinsleri.
const ONEKLER: [(&str, SembolTuru); 6] = [
    ("pub fn ", SembolTuru::Fonksiyon),
    ("pub struct ", SembolTuru::Yapi),
    ("pub enum ", SembolTuru::Enum),
    ("pub const ", SembolTuru::Sabit),
    ("pub trait ", SembolTuru::Trait),
    ("pub type ", SembolTuru::Yapi),
];

fn rust_sembolu(k: &str) -> Option<(String, SembolTuru)> {
    let (govde, tur) = ONEKLER
        .iter()
        .find_map(|(onek, tur)| k.strip_prefix(onek).map(|govde| (govde, *tur)))?;
    let ad = ilk_kelime(govde);
    if ad.is_empty() {
        None
    } else {
        Some((ad, tur))
    }
}

fn python_sabiti(k: &str) -> Option<String> {
    let esittir = k.find('=')?;
    let ad = k[..esittir].trim();
    if ad.is_empty() || ad.len() > 64 {
        return None;
    }
    let gecerli = ad
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    let ilk_harf = ad
        .chars()
        .next()
        .map(|c| c.is_ascii_uppercase() || c == '_');
    if gecerli && ilk_harf == Some(true) {
        Some(ad.to_string())
    } else {
        None
    }
}

fn ilk_kelime(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Windows yol ayracı. Kaynakta iki-eğik-cizgi kaçışı (`\\`) yazıldığında
/// sınırlayıcı sayımını yapan durum makinesi karakter sabitinin kapanış
/// tırnağını kaçış sanıp yutuyor; bu yüzden ayracı `\u{5c}` ile yazarız -
/// aynı karakter, sayımı bozmayan yazım.
const WIN_AYRAC: char = '\u{5c}';

fn goreli_yol(kok: &Path, yol: &Path) -> String {
    yol.strip_prefix(kok)
        .unwrap_or(yol)
        .display()
        .to_string()
        .replace(WIN_AYRAC, "/")
}

fn sha256_hex(bayt: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bayt);
    let ozet = h.finalize();
    let mut s = String::with_capacity(ozet.len() * 2);
    for b in ozet {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn agac_ozeti(dosyalar: &[DosyaKaydi]) -> String {
    let mut h = Sha256::new();
    for d in dosyalar {
        h.update(d.yol.as_bytes());
        h.update([0]);
        h.update(d.ozet.as_bytes());
        h.update(b"\n");
    }
    let ozet = h.finalize();
    let mut s = String::with_capacity(ozet.len() * 2);
    for b in ozet {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Türkçe harfleri ASCII'ye indirger, kalanları tire yapar ve tireleri kısar.
fn ascii_kebab(s: &str) -> String {
    let duz: String = s
        .chars()
        .map(|c| match c {
            'ç' | 'Ç' => 'c',
            'ğ' | 'Ğ' => 'g',
            'ı' | 'I' | 'İ' => 'i',
            'ö' | 'Ö' => 'o',
            'ş' | 'Ş' => 's',
            'ü' | 'Ü' => 'u',
            diger => diger,
        })
        .flat_map(|c| c.to_lowercase())
        .collect();
    let mut cikti = String::with_capacity(duz.len());
    let mut onceki_tire = false;
    for c in duz.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            cikti.push(c);
            onceki_tire = false;
        } else if !onceki_tire && !cikti.is_empty() {
            cikti.push('-');
            onceki_tire = true;
        }
    }
    while cikti.ends_with('-') {
        cikti.pop();
    }
    cikti
}

fn ad_denetle(ad: &str) -> Result<(), GecisHatasi> {
    let gecerli = !ad.is_empty()
        && ad.len() <= 32
        && ad
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && ad.chars().next().map(|c| c.is_ascii_lowercase()) == Some(true)
        && !ad.ends_with('-')
        && !ad.contains("--");
    if gecerli {
        Ok(())
    } else {
        Err(GecisHatasi::AdGecersiz(ad.to_string()))
    }
}

fn modul_adi(yol: &str) -> Result<String, GecisHatasi> {
    let govde = match yol.rfind('.') {
        Some(nokta) if nokta > yol.rfind('/').map(|b| b + 1).unwrap_or(0) => &yol[..nokta],
        _ => yol,
    };
    let ad: String = govde
        .split('/')
        .map(ascii_kebab)
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if ad.is_empty() {
        return Err(GecisHatasi::AdGecersiz(yol.to_string()));
    }
    Ok(ad)
}

fn ic_mi(alim: &str, sayim: &Sayim) -> bool {
    let ilk = alim.split('.').next().unwrap_or(alim);
    let kok = ascii_kebab(ilk);
    if kok.is_empty() {
        return false;
    }
    let saplama = |d: &DosyaKaydi| -> Option<String> {
        let yol = &d.yol;
        let taban = yol.rsplit('/').next().unwrap_or(yol);
        let govde = match taban.rfind('.') {
            Some(n) => &taban[..n],
            None => taban,
        };
        Some(ascii_kebab(govde))
    };
    sayim
        .dosyalar
        .iter()
        .filter(|d| matches!(d.tur, DosyaTuru::Rust | DosyaTuru::Python))
        .any(|d| saplama(d).as_deref() == Some(kok.as_str()))
        || sayim
            .dizinler
            .iter()
            .any(|d| ascii_kebab(d.rsplit('/').next().unwrap_or(d)) == kok)
}

/// Bir sayımdan gerçekleme planı kurar.
///
/// # Hatalar
///
/// Hedef ad geçersizse [`GecisHatasi::AdGecersiz`], iki dosya aynı modül adına
/// inerse [`GecisHatasi::AdCakismasi`].
pub fn plan(sayim: &Sayim, hedef: &str) -> Result<Plan, GecisHatasi> {
    ad_denetle(hedef)?;
    let mut moduller: Vec<ModulPlani> = Vec::new();
    for d in &sayim.dosyalar {
        if !matches!(d.tur, DosyaTuru::Rust | DosyaTuru::Python) || d.semboller.is_empty() {
            continue;
        }
        let ad = modul_adi(&d.yol)?;
        let (ic, dis): (Vec<String>, Vec<String>) =
            d.alimlar.iter().cloned().partition(|a| ic_mi(a, sayim));
        moduller.push(ModulPlani {
            ad,
            kaynak_yol: d.yol.clone(),
            semboller: d.semboller.iter().map(|s| s.ad.clone()).collect(),
            ic_alimlar: ic,
            dis_alimlar: dis,
        });
    }
    moduller.sort_by(|a, b| a.ad.cmp(&b.ad));
    for i in 0..moduller.len() {
        for j in i + 1..moduller.len() {
            if moduller[i].ad == moduller[j].ad {
                return Err(GecisHatasi::AdCakismasi {
                    ad: moduller[i].ad.clone(),
                    birinci: moduller[i].kaynak_yol.clone(),
                    ikinci: moduller[j].kaynak_yol.clone(),
                });
            }
        }
    }
    let veri_dosyalari: Vec<String> = sayim
        .dosyalar
        .iter()
        .filter(|d| {
            matches!(
                d.tur,
                DosyaTuru::Json | DosyaTuru::Toml | DosyaTuru::Yaml | DosyaTuru::Markdown
            )
        })
        .map(|d| d.yol.clone())
        .collect();
    let test_amaci = moduller.iter().map(|m| m.semboller.len() as u32).sum();
    Ok(Plan {
        hedef_crate: hedef.to_string(),
        lisans: sayim.lisans.clone(),
        kaynak_ozeti: sayim.agac_ozeti.clone(),
        moduller,
        veri_dosyalari,
        test_amaci,
        tesisat: Tesisat {
            uye_satiri: format!("    \"crates/{hedef}\","),
            cli_bagimliligi: format!("lubot-{hedef} = {{ path = \"../{hedef}\", version = \"0.1.0\" }}"),
            crates_satiri: format!(
                "| `{hedef}` | olculecek | olculecek | gecis hatti: sozlesme plan.json'da, kapsam durum.json'da |"
            ),
            kapi_kaydi: format!(
                "    \"{hedef}-hatti-kapisi\": (gate_{hedef}_hatti_kapisi, selftest_{hedef}_hatti_kapisi),"
            ),
        },
    })
}

/// Durum sayacının iğnesi: kapanış parantezini bilinçli atar, böylece
/// kaynağın kendisi test özniteliğinin tam yazımını taşımaz ve test sayan
/// kapı bu iğneyi bir test sanıp saymaz.
const IGNESI_TEST: &str = "#[test";

fn icerir_kelime(metin: &str, ad: &str) -> bool {
    metin
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|k| k == ad)
}

/// Bir planın gerçekleşen karşılığını ölçer.
///
/// # Hatalar
///
/// Gerçekleşen dizin yoksa [`GecisHatasi::KaynakYok`], okunamıyorsa
/// [`GecisHatasi::Okunamadi`].
pub fn durum(
    pln: &Plan,
    gerceklesen: &Path,
    esleme: Option<&BTreeMap<String, String>>,
) -> Result<Durum, GecisHatasi> {
    let meta = fs::symlink_metadata(gerceklesen).map_err(|e| meta_hatasi(gerceklesen, &e))?;
    if !meta.is_dir() {
        return Err(GecisHatasi::KaynakYok(gerceklesen.display().to_string()));
    }
    let mut moduller = Vec::new();
    let mut bekleyen = Vec::new();
    let mut var_sayisi = 0usize;
    let mut eslenen_toplam = 0usize;
    let mut eslenen_kapsanan = 0usize;
    for m in &pln.moduller {
        let dosya = gerceklesen
            .join("src")
            .join(format!("{}.rs", m.ad.replace('-', "_")));
        if !dosya.is_file() {
            moduller.push(ModulDurumu {
                ad: m.ad.clone(),
                var: false,
                sembol: 0,
                test: 0,
                simge_kapsami: None,
            });
            bekleyen.push(m.ad.clone());
            continue;
        }
        var_sayisi += 1;
        let metin = fs::read_to_string(&dosya).map_err(|e| GecisHatasi::Okunamadi {
            yol: dosya.display().to_string(),
            neden: e.to_string(),
        })?;
        let sembol = metin
            .lines()
            .filter(|l| l.trim_start().starts_with("pub "))
            .count() as u32;
        let test = metin.matches(IGNESI_TEST).count() as u32;
        let simge_kapsami = esleme.map(|e| {
            let mut toplam = 0usize;
            let mut kapsanan = 0usize;
            for s in &m.semboller {
                if let Some(rust_adi) = e.get(&format!("{}:{}", m.ad, s)) {
                    toplam += 1;
                    if icerir_kelime(&metin, rust_adi) {
                        kapsanan += 1;
                    }
                }
            }
            eslenen_toplam += toplam;
            eslenen_kapsanan += kapsanan;
            if toplam == 0 {
                0.0
            } else {
                kapsanan as f32 / toplam as f32
            }
        });
        moduller.push(ModulDurumu {
            ad: m.ad.clone(),
            var: true,
            sembol,
            test,
            simge_kapsami,
        });
    }
    let modul_kapsami = if pln.moduller.is_empty() {
        0.0
    } else {
        var_sayisi as f32 / pln.moduller.len() as f32
    };
    let simge_kapsami = esleme.map(|_| {
        if eslenen_toplam == 0 {
            0.0
        } else {
            eslenen_kapsanan as f32 / eslenen_toplam as f32
        }
    });
    Ok(Durum {
        hedef_crate: pln.hedef_crate.clone(),
        plan_ozeti: pln.kaynak_ozeti.clone(),
        moduller,
        modul_kapsami,
        simge_kapsami,
        bekleyen,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn sablon_koku() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("ornek-kaynak")
    }

    fn sablon_sayim() -> Sayim {
        say(&sablon_koku(), "MIT").expect("sablon sayildi")
    }

    #[test]
    fn sayim_dosya_cinslerini_gorur() {
        let s = sablon_sayim();
        let tur = |yol: &str| {
            s.dosyalar
                .iter()
                .find(|d| d.yol == yol)
                .unwrap_or_else(|| panic!("kayit yok: {yol}"))
                .tur
        };
        assert_eq!(tur("README.md"), DosyaTuru::Markdown);
        assert_eq!(tur("ayar.toml"), DosyaTuru::Toml);
        assert_eq!(tur("katman/rope.py"), DosyaTuru::Python);
        assert_eq!(tur("ndim.py"), DosyaTuru::Python);
        assert_eq!(tur("veri.json"), DosyaTuru::Json);
        assert_eq!(s.dosyalar.len(), 5);
    }

    #[test]
    fn sayim_sembolleri_yakalar() {
        let s = sablon_sayim();
        let ndim = s
            .dosyalar
            .iter()
            .find(|d| d.yol == "ndim.py")
            .unwrap_or_else(|| panic!("ndim kaydi yok"));
        let adlar: Vec<&str> = ndim.semboller.iter().map(|x| x.ad.as_str()).collect();
        assert!(adlar.contains(&"SOFTMAX_EPS"));
        assert!(adlar.contains(&"softmax"));
        assert!(adlar.contains(&"Katman"));
        let rope = s
            .dosyalar
            .iter()
            .find(|d| d.yol == "katman/rope.py")
            .unwrap_or_else(|| panic!("rope kaydi yok"));
        assert!(rope.semboller.iter().any(|x| x.ad == "rope_acisi"));
        let oku = s
            .dosyalar
            .iter()
            .find(|d| d.yol == "README.md")
            .unwrap_or_else(|| panic!("readme kaydi yok"));
        assert!(oku.semboller.iter().any(|x| x.tur == SembolTuru::Baslik));
    }

    #[test]
    fn sayim_girintili_tanimlari_atlar() {
        // ndim.py icindeki `def ileri` girintili oldugu icin modul sembolu
        // degil, sinif uyesi olarak kalmali: sayim satir basina bakar.
        let s = sablon_sayim();
        let ndim = s
            .dosyalar
            .iter()
            .find(|d| d.yol == "ndim.py")
            .unwrap_or_else(|| panic!("ndim kaydi yok"));
        assert!(!ndim.semboller.iter().any(|x| x.ad == "ileri"));
    }

    #[test]
    fn sayim_ozeti_yinelenebilir() {
        let a = sablon_sayim();
        let b = sablon_sayim();
        assert_eq!(a.agac_ozeti, b.agac_ozeti);
        assert_eq!(a.dosyalar, b.dosyalar);
    }

    #[test]
    fn sayim_dosyalari_yola_gore_siralar() {
        let s = sablon_sayim();
        let yollar: Vec<&str> = s.dosyalar.iter().map(|d| d.yol.as_str()).collect();
        let mut beklenen = yollar.clone();
        beklenen.sort();
        assert_eq!(yollar, beklenen);
        assert!(s.dizinler.iter().any(|d| d == "katman"));
    }

    #[test]
    fn sayim_lisans_kapaliysa_reddeder() {
        let e = say(&sablon_koku(), "GPL-3.0").expect_err("kapali lisans gecti");
        assert_eq!(e, GecisHatasi::LisansKapali("GPL-3.0".to_string()));
        assert!(e.to_string().contains("GPL-3.0"));
    }

    #[test]
    fn sayim_kaynak_yoksa_reddeder() {
        let e = say(Path::new("/tmp/gecis-böyle-bir-yol-yok"), "MIT")
            .expect_err("olmayan kaynak gecti");
        assert!(matches!(e, GecisHatasi::KaynakYok(_)));
    }

    #[test]
    fn sayim_bos_agaci_reddeder() {
        let kok = std::env::temp_dir().join("gecis-bos-agac-testi");
        let _ = fs::remove_dir_all(&kok);
        fs::create_dir_all(&kok).expect("gecici dizin kuruldu");
        let e = say(&kok, "MIT").expect_err("bos agac gecti");
        let _ = fs::remove_dir_all(&kok);
        assert!(matches!(e, GecisHatasi::BosAgac(_)));
    }

    #[cfg(unix)]
    #[test]
    fn sayim_bag_izlemez() {
        use std::os::unix::fs::symlink;
        let kok = std::env::temp_dir().join("gecis-bag-testi");
        let hedef = std::env::temp_dir().join("gecis-bag-hedefi");
        let _ = fs::remove_dir_all(&kok);
        let _ = fs::remove_dir_all(&hedef);
        fs::create_dir_all(&hedef).expect("hedef kuruldu");
        fs::create_dir_all(&kok).expect("kok kuruldu");
        fs::write(hedef.join("a.py"), "def f():\n    pass\n").expect("yazildi");
        symlink(&hedef, kok.join("bag")).expect("bag kuruldu");
        let e = say(&kok, "MIT").expect_err("bag izlendi");
        let _ = fs::remove_dir_all(&kok);
        let _ = fs::remove_dir_all(&hedef);
        assert!(matches!(e, GecisHatasi::BagIzlenmez(_)));
    }

    #[test]
    fn tur_bilinmeyen_uzantida_nul_ezer() {
        let p = Path::new("dosya.ver");
        assert_eq!(tur_belirle(p, b"merhaba"), DosyaTuru::Metin);
        assert_eq!(tur_belirle(p, b"me\x00rhaba"), DosyaTuru::Ikili);
        assert_eq!(tur_belirle(Path::new("x.rs"), b""), DosyaTuru::Rust);
    }

    #[test]
    fn satir_sayimi_tanimlidir() {
        assert_eq!(satir_say(b""), 0);
        assert_eq!(satir_say(b"a\n"), 1);
        assert_eq!(satir_say(b"a\nb"), 2);
        assert_eq!(satir_say(b"\n"), 1);
    }

    #[test]
    fn sinirlar_adini_tasiyarak_red_eder() {
        let kok = sablon_koku();
        let e = say_sinirli(
            &kok,
            "MIT",
            &Sinirlar {
                en_cok_dosya: 2,
                en_cok_sembol: 8192,
                en_cok_bayt: 1 << 20,
            },
        )
        .expect_err("dosya siniri gecmedi");
        assert!(matches!(e, GecisHatasi::DosyaSiniri { sinir: 2, .. }));
        let e = say_sinirli(
            &kok,
            "MIT",
            &Sinirlar {
                en_cok_dosya: 4096,
                en_cok_sembol: 3,
                en_cok_bayt: 1 << 20,
            },
        )
        .expect_err("sembol siniri gecmedi");
        assert!(matches!(e, GecisHatasi::SembolSiniri { sinir: 3, .. }));
        let e = say_sinirli(
            &kok,
            "MIT",
            &Sinirlar {
                en_cok_dosya: 4096,
                en_cok_sembol: 8192,
                en_cok_bayt: 8,
            },
        )
        .expect_err("bayt siniri gecmedi");
        assert!(matches!(e, GecisHatasi::BaytSiniri { sinir: 8, .. }));
    }

    #[test]
    fn plan_modulleri_ve_alimlari_boler() {
        let p = plan(&sablon_sayim(), "ornek").expect("plan kuruldu");
        let adlar: Vec<&str> = p.moduller.iter().map(|m| m.ad.as_str()).collect();
        assert_eq!(adlar, vec!["katman-rope", "ndim"]);
        let ndim = p
            .moduller
            .iter()
            .find(|m| m.ad == "ndim")
            .unwrap_or_else(|| panic!("ndim plani yok"));
        assert_eq!(ndim.ic_alimlar, Vec::<String>::new());
        assert_eq!(ndim.dis_alimlar, vec!["json".to_string()]);
        let rope = p
            .moduller
            .iter()
            .find(|m| m.ad == "katman-rope")
            .unwrap_or_else(|| panic!("rope plani yok"));
        assert_eq!(rope.ic_alimlar, vec!["ndim".to_string()]);
        assert_eq!(
            p.veri_dosyalari,
            vec![
                "README.md".to_string(),
                "ayar.toml".to_string(),
                "veri.json".to_string()
            ]
        );
    }

    #[test]
    fn plan_tesisat_satirlari_agacin_bicimindedir() {
        let p = plan(&sablon_sayim(), "gecis").expect("plan kuruldu");
        assert_eq!(p.tesisat.uye_satiri, "    \"crates/gecis\",");
        assert_eq!(
            p.tesisat.cli_bagimliligi,
            "lubot-gecis = { path = \"../gecis\", version = \"0.1.0\" }"
        );
    }

    #[test]
    fn plan_test_amaci_sozlesmeyi_sayar() {
        let p = plan(&sablon_sayim(), "ornek").expect("plan kuruldu");
        let sozlesme: usize = p.moduller.iter().map(|m| m.semboller.len()).sum();
        assert_eq!(p.test_amaci as usize, sozlesme);
        assert!(p.test_amaci >= 5);
    }

    #[test]
    fn plan_gecersiz_adi_reddeder() {
        for ad in [
            "ABC",
            "1x",
            "a_b",
            "a--b",
            "a-",
            "",
            "cok-uzun-bir-crate-adi-ki-aslinda-cok-daha-uzun",
        ] {
            assert!(plan(&sablon_sayim(), ad).is_err(), "ad gecti: {ad}");
        }
    }

    #[test]
    fn plan_cakisan_adlari_reddeder() {
        let kok = std::env::temp_dir().join("gecis-cakisma-testi");
        let _ = fs::remove_dir_all(&kok);
        fs::create_dir_all(kok.join("a")).expect("dizin kuruldu");
        fs::write(kok.join("a").join("b.py"), "def f():\n    pass\n").expect("yazildi");
        fs::write(kok.join("a b.py"), "def g():\n    pass\n").expect("yazildi");
        let s = say(&kok, "MIT").expect("sayildi");
        let e = plan(&s, "ornek").expect_err("cakisma gecti");
        let _ = fs::remove_dir_all(&kok);
        assert!(matches!(e, GecisHatasi::AdCakismasi { .. }));
        assert!(e.to_string().contains("a-b"));
    }

    #[test]
    fn plan_json_gidis_gelisi_kaybeder_mi() {
        let p = plan(&sablon_sayim(), "ornek").expect("plan kuruldu");
        let metin = serde_json::to_string(&p).expect("yazildi");
        let geri: Plan = serde_json::from_str(&metin).expect("okundu");
        assert_eq!(p, geri);
    }

    #[test]
    fn durum_yariya_kadar_gerceklesmeyi_gorur() {
        let p = plan(&sablon_sayim(), "ornek").expect("plan kuruldu");
        let kok = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("ornek-gerceklesen");
        let d = durum(&p, &kok, None).expect("durum olculdu");
        assert!((d.modul_kapsami - 0.5).abs() < 1e-6);
        assert_eq!(d.bekleyen, vec!["katman-rope".to_string()]);
        let ndim = d
            .moduller
            .iter()
            .find(|m| m.ad == "ndim")
            .unwrap_or_else(|| panic!("ndim durumu yok"));
        assert!(ndim.var);
        assert!(ndim.test >= 1);
        assert!(d.simge_kapsami.is_none());
    }

    #[test]
    fn durum_eslemeyle_simge_duzeyine_iner() {
        let p = plan(&sablon_sayim(), "ornek").expect("plan kuruldu");
        let kok = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("ornek-gerceklesen");
        let mut esleme = BTreeMap::new();
        esleme.insert("ndim:softmax".to_string(), "yumusat".to_string());
        esleme.insert("ndim:Katman".to_string(), "YokBoyleBirSey".to_string());
        let d = durum(&p, &kok, Some(&esleme)).expect("durum olculdu");
        let kapsam = d.simge_kapsami.expect("esleme verildi");
        assert!((kapsam - 0.5).abs() < 1e-6, "kapsam {kapsam}");
    }

    #[test]
    fn durum_kaynak_yoksa_reddeder() {
        let p = plan(&sablon_sayim(), "ornek").expect("plan kuruldu");
        let e = durum(&p, Path::new("/tmp/gecis-gerceklesmemis"), None)
            .expect_err("olmayan gerceklesen gecti");
        assert!(matches!(e, GecisHatasi::KaynakYok(_)));
    }

    #[test]
    fn ascii_kebab_turkce_harfleri_indirger() {
        assert_eq!(ascii_kebab("Ölçü"), "olcu");
        assert_eq!(ascii_kebab("Katman-Rope"), "katman-rope");
        assert_eq!(ascii_kebab("İşlem -- ağacı"), "islem-agaci");
        assert_eq!(ascii_kebab("---"), "");
    }

    #[test]
    fn modul_adi_yolu_bilesen_bilesen_cevirir() {
        assert_eq!(modul_adi("katman/rope.py").expect("ad"), "katman-rope");
        assert_eq!(modul_adi("ndim.py").expect("ad"), "ndim");
        assert_eq!(modul_adi("Ölçü/İşlem.py").expect("ad"), "olcu-islem");
        assert!(modul_adi("----.py").is_err());
    }

    #[test]
    fn kelime_aramasi_sinirlarla_calisir() {
        assert!(icerir_kelime("pub fn yumusat(x: &[f32])", "yumusat"));
        assert!(!icerir_kelime("pub fn yumusat(x: &[f32])", "yumus"));
        assert!(!icerir_kelime("yumusatma_fn", "yumusat"));
    }

    #[test]
    fn ic_alim_dizin_adiyla_eslesir() {
        let s = sablon_sayim();
        assert!(ic_mi("ndim", &s));
        assert!(ic_mi("katman.rope", &s));
        assert!(!ic_mi("json", &s));
        assert!(!ic_mi("numpy", &s));
    }

    #[test]
    fn rust_sembolleri_oneklerle_yakalanir() {
        assert_eq!(
            rust_sembolu("pub fn foo<T>(x: T)"),
            Some(("foo".to_string(), SembolTuru::Fonksiyon))
        );
        assert_eq!(
            rust_sembolu("pub struct Kayit {"),
            Some(("Kayit".to_string(), SembolTuru::Yapi))
        );
        assert_eq!(rust_sembolu("fn gizli()"), None);
        assert_eq!(rust_sembolu("pub "), None);
    }

    #[test]
    fn hata_mesajlari_adi_adi_soyler() {
        assert!(GecisHatasi::KaynakYok("/a/b".into())
            .to_string()
            .contains("/a/b"));
        assert!(GecisHatasi::BagIzlenmez("/a/b".into())
            .to_string()
            .contains("/a/b"));
        assert!(GecisHatasi::BosAgac("/a/b".into())
            .to_string()
            .contains("/a/b"));
        assert!(GecisHatasi::LisansKapali("X".into())
            .to_string()
            .contains("X"));
        assert!(GecisHatasi::AdGecersiz("Y".into())
            .to_string()
            .contains("Y"));
        assert!(GecisHatasi::Okunamadi {
            yol: "/c".into(),
            neden: "neden".into()
        }
        .to_string()
        .contains("/c"));
    }
}
