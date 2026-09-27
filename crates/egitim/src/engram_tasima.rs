//! Engram tablosunun tasinmasi: tablo, kontrol noktasinin **yaninda** giden,
//! kendi ozetini tasiyan ikinci bir dosyadir.
//!
//! # Neden ayri bir dosya
//!
//! Direktif 7.1, engram tablosunu model durumunun parcasi sayar ve kontrol
//! noktasi mekanizmasiyla saklanmasini ister. Kontrol noktasi bicimi (`kontrol`,
//! surum 1) bloklarini `Parametreler::blok_adlari()` sirasiyla okur; engram ise
//! hicbir aileye bagli degildir (M1/M2/M3 acik, operator karari). Bagli olmayan
//! bir tabloyu surum 1'in blok sirasina sokmak, her mevcut kontrol noktasini
//! "eski bicim" yapar ve aile karari verilmeden bicimi degistirir. Bu yuzden
//! tablo ayni kurallarla (sihir, surum, hassasiyet, JSON baslik, adli blok,
//! kuyrukta SHA-256) ama **kendi** dosyasinda tasinir ve basligi, ait oldugu
//! kontrol noktasinin ozetini tasir: hangi agirliklarla birlikte egitildigi
//! dosyanin icinde yazar, yanina konan bir ada bagli degildir.
//!
//! # Yazilan, sirasiyla
//!
//! ```text
//! LUBOTENGR | surum:u8 | hassasiyet:u8 | bayrak:u16le | baslik_uzunluk:u32le
//! baslik JSON | "engram.tablo" blok | sha256(oncesindeki her bayt)
//! ```
//!
//! Baslik `n`, `tablo`, `d_kv`, `adim`, `hassasiyet` ve `kontrol_ozeti`
//! (sahip kontrol noktasinin hex ozeti; yoksa `null`) tasir. Okuyucu once
//! sabit genislikli alanlari, sonra basligi, sonra blogu, en son ozeti
//! dogrular: bozuk bir hassasiyet bayti JSON hatasi olarak degil hassasiyet
//! hatasi olarak geri doner.

use std::path::Path;

use crate::engram::{EngramSekilHatasi, EngramSpec};
use crate::kontrol::{hex, Hassasiyet, Kontrol, OZET_UZUNLUK};

/// Sihir baytlari: bununla baslamayan dosya engram tasiyicisi degildir.
pub const ENGRAM_SIHIR: &[u8; 9] = b"LUBOTENGR";
/// Bicim surumu. Okuyucu bilmedigi surumu reddeder.
pub const ENGRAM_SURUM: u8 = 1;
/// Tek blogun adi.
pub const ENGRAM_BLOK_ADI: &str = "engram.tablo";

/// Neden bir engram dosyasi reddedildi.
#[derive(Debug, Clone, PartialEq)]
pub enum EngramTasimaHatasi {
    /// Dosya [`ENGRAM_SIHIR`] ile baslamiyor.
    Sihir,
    /// Surum bayti bu okuyucunun bilmedigi bir bicimi adlandiriyor.
    Surum(u8),
    /// Hassasiyet bayti ne `f64` ne `f32`.
    Hassasiyet(u8),
    /// Bayrak sozcugunde burada tanimsiz bir bit var.
    Bayrak(u16),
    /// Baslik bu bicimin yazdigi JSON degil.
    Baslik(String),
    /// Basliktaki sekil gecersiz.
    Sekil(EngramSekilHatasi),
    /// Blok adi ya da uzunlugu bicimin sirasina uymuyor.
    Blok(String),
    /// Tablo uzunlugu seklin parametre sayisina esit degil.
    Uzunluk { beklenen: usize, var: usize },
    /// Kuyruktaki ozet oncesindeki baytlarla uyusmuyor.
    Ozet { beklenen: String, bulunan: String },
    /// Dosya ozetten once bitiyor.
    Kisa { gerekli: usize, var: usize },
    /// Dosya okunamadi ya da yazilamadi.
    Io(String),
}

impl std::fmt::Display for EngramTasimaHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sihir => write!(f, "engram tasiyicisi degil (sihir baytlari yok)"),
            Self::Surum(s) => write!(f, "bilinmeyen engram bicim surumu {s}"),
            Self::Hassasiyet(k) => write!(f, "bilinmeyen hassasiyet kodu {k}"),
            Self::Bayrak(b) => write!(f, "tanimsiz bayrak {b:#06x}"),
            Self::Baslik(m) => write!(f, "baslik okunamadi: {m}"),
            Self::Sekil(s) => write!(f, "engram sekli gecersiz: {s:?}"),
            Self::Blok(m) => write!(f, "blok: {m}"),
            Self::Uzunluk { beklenen, var } => {
                write!(f, "tablo uzunlugu {var}, sekil {beklenen} istiyor")
            }
            Self::Ozet { beklenen, bulunan } => {
                write!(f, "ozet uyusmuyor: dosya {beklenen}, hesaplanan {bulunan}")
            }
            Self::Kisa { gerekli, var } => write!(f, "dosya kisa: {var} bayt, {gerekli} gerekli"),
            Self::Io(m) => write!(f, "g/c: {m}"),
        }
    }
}

/// Tasinan engram tablosu ve kime ait oldugu.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramTasiyici {
    /// Tablonun sekli.
    pub spec: EngramSpec,
    /// Tablo degerleri: anahtarlar `[0, tablo*d_kv)`, degerler sonra.
    pub tablo: Vec<f64>,
    /// Tablonun ait oldugu kosunun adimi.
    pub adim: u64,
    /// Sahip kontrol noktasinin hex ozeti; tablo tek basina egitildiyse yok.
    pub kontrol_ozeti: Option<String>,
    /// Degerlerin nasil saklandigi.
    pub hassasiyet: Hassasiyet,
}

impl EngramTasiyici {
    /// Tabloyu sekliyle dogrulayarak kurar.
    ///
    /// # Errors
    /// [`EngramTasimaHatasi::Uzunluk`] tablo uzunlugu `spec.parametre_sayisi()`
    /// degilse.
    pub fn yeni(
        spec: EngramSpec,
        tablo: Vec<f64>,
        adim: u64,
        kontrol_ozeti: Option<String>,
        hassasiyet: Hassasiyet,
    ) -> Result<Self, EngramTasimaHatasi> {
        let beklenen = spec.parametre_sayisi();
        if tablo.len() != beklenen {
            return Err(EngramTasimaHatasi::Uzunluk {
                beklenen,
                var: tablo.len(),
            });
        }
        Ok(Self {
            spec,
            tablo,
            adim,
            kontrol_ozeti,
            hassasiyet,
        })
    }

    /// Bir kontrol noktasina baglar: o dosyanin ozetini basliga yazar.
    ///
    /// Ozet, kontrol noktasinin `baytlar()` cikisi uzerinden hesaplanir; yani
    /// `Kontrol::yaz`'in dondurdugu degerle aynidir. Adim da oradan alinir:
    /// tablo ile agirliklarin farkli adimlari iddia etmesi mumkun degildir.
    ///
    /// # Errors
    /// Kontrol noktasinin baytlari uretilemezse ([`crate::kontrol::KontrolHatasi`]
    /// metni ile [`EngramTasimaHatasi::Blok`]).
    pub fn bagla(&mut self, kontrol: &Kontrol) -> Result<String, EngramTasimaHatasi> {
        let govde = kontrol
            .baytlar()
            .map_err(|e| EngramTasimaHatasi::Blok(e.to_string()))?;
        let ozet = hex(&ozetle(&govde));
        self.kontrol_ozeti = Some(ozet.clone());
        self.adim = kontrol.adim;
        Ok(ozet)
    }

    /// Bu tablo verilen kontrol noktasina mi ait?
    ///
    /// `None` bagsiz bir tablo icin doner: bagsizlik bir cevap degil, cevabin
    /// yoklugudur ve `false` ile karistirilmaz.
    ///
    /// # Errors
    /// Kontrol noktasinin baytlari uretilemezse.
    pub fn ait_mi(&self, kontrol: &Kontrol) -> Result<Option<bool>, EngramTasimaHatasi> {
        let Some(beklenen) = &self.kontrol_ozeti else {
            return Ok(None);
        };
        let govde = kontrol
            .baytlar()
            .map_err(|e| EngramTasimaHatasi::Blok(e.to_string()))?;
        Ok(Some(
            hex(&ozetle(&govde)) == *beklenen && kontrol.adim == self.adim,
        ))
    }

    /// Dosyayi yazar, yazilan baytlarin SHA-256'sini dondurur.
    ///
    /// # Errors
    /// [`EngramTasimaHatasi::Io`] yazilamazsa; [`EngramTasimaHatasi::Uzunluk`]
    /// tablo sekle uymuyorsa.
    pub fn yaz(&self, yol: &Path) -> Result<String, EngramTasimaHatasi> {
        let mut govde = self.baytlar()?;
        let ozet = ozetle(&govde);
        govde.extend_from_slice(&ozet);
        std::fs::write(yol, &govde).map_err(|e| EngramTasimaHatasi::Io(e.to_string()))?;
        Ok(hex(&ozet))
    }

    /// Dosyayi yukler ve kendi ozetine karsi dogrular.
    ///
    /// # Errors
    /// [`EngramTasimaHatasi::Io`] okunamazsa, sonra [`Self::baytlardan`]'in her reddi.
    pub fn yukle(yol: &Path) -> Result<Self, EngramTasimaHatasi> {
        let ham = std::fs::read(yol).map_err(|e| EngramTasimaHatasi::Io(e.to_string()))?;
        Self::baytlardan(&ham)
    }

    /// Ozetsiz govde: sabit alanlar, baslik, tek blok.
    ///
    /// # Errors
    /// [`EngramTasimaHatasi::Uzunluk`] tablo sekle uymuyorsa.
    pub fn baytlar(&self) -> Result<Vec<u8>, EngramTasimaHatasi> {
        let beklenen = self.spec.parametre_sayisi();
        if self.tablo.len() != beklenen {
            return Err(EngramTasimaHatasi::Uzunluk {
                beklenen,
                var: self.tablo.len(),
            });
        }
        let mut govde: Vec<u8> = Vec::new();
        govde.extend_from_slice(ENGRAM_SIHIR);
        govde.push(ENGRAM_SURUM);
        govde.push(self.hassasiyet.kod());
        govde.extend_from_slice(&0u16.to_le_bytes());
        let baslik = self.baslik_json();
        let baslik_baytlari = baslik.as_bytes();
        let baslik_uzunluk = u32::try_from(baslik_baytlari.len())
            .map_err(|_| EngramTasimaHatasi::Baslik("baslik u32'ye sigmiyor".to_string()))?;
        govde.extend_from_slice(&baslik_uzunluk.to_le_bytes());
        govde.extend_from_slice(baslik_baytlari);
        let ad_uzunluk = u16::try_from(ENGRAM_BLOK_ADI.len())
            .map_err(|_| EngramTasimaHatasi::Blok("blok adi u16'ya sigmiyor".to_string()))?;
        govde.extend_from_slice(&ad_uzunluk.to_le_bytes());
        govde.extend_from_slice(ENGRAM_BLOK_ADI.as_bytes());
        govde.extend_from_slice(&(self.tablo.len() as u64).to_le_bytes());
        for deger in &self.tablo {
            match self.hassasiyet {
                Hassasiyet::F64 => govde.extend_from_slice(&deger.to_le_bytes()),
                #[allow(clippy::cast_possible_truncation)]
                Hassasiyet::F32 => govde.extend_from_slice(&(*deger as f32).to_le_bytes()),
            }
        }
        Ok(govde)
    }

    fn baslik_json(&self) -> String {
        serde_json::json!({
            "surum": ENGRAM_SURUM,
            "n": self.spec.n,
            "tablo": self.spec.tablo,
            "d_kv": self.spec.d_kv,
            "adim": self.adim,
            "hassasiyet": self.hassasiyet.etiket(),
            "kontrol_ozeti": self.kontrol_ozeti,
        })
        .to_string()
    }

    /// Baytlardan cozer ve dogrular: sabit alanlar, baslik, blok, ozet.
    ///
    /// # Errors
    /// [`EngramTasimaHatasi`]'nin her reddi, bu sirayla.
    pub fn baytlardan(ham: &[u8]) -> Result<Self, EngramTasimaHatasi> {
        let sabit = ENGRAM_SIHIR.len() + 2 + 2 + 4;
        if ham.len() < sabit {
            return Err(EngramTasimaHatasi::Kisa {
                gerekli: sabit,
                var: ham.len(),
            });
        }
        if &ham[..ENGRAM_SIHIR.len()] != ENGRAM_SIHIR {
            return Err(EngramTasimaHatasi::Sihir);
        }
        let mut konum = ENGRAM_SIHIR.len();
        let surum = ham[konum];
        konum += 1;
        if surum != ENGRAM_SURUM {
            return Err(EngramTasimaHatasi::Surum(surum));
        }
        let kod = ham[konum];
        konum += 1;
        let hassasiyet = hassasiyet_koddan(kod).ok_or(EngramTasimaHatasi::Hassasiyet(kod))?;
        let bayrak = u16::from_le_bytes([ham[konum], ham[konum + 1]]);
        konum += 2;
        if bayrak != 0 {
            return Err(EngramTasimaHatasi::Bayrak(bayrak));
        }
        let baslik_uzunluk =
            u32::from_le_bytes([ham[konum], ham[konum + 1], ham[konum + 2], ham[konum + 3]])
                as usize;
        konum += 4;
        let baslik_sonu = konum
            .checked_add(baslik_uzunluk)
            .ok_or(EngramTasimaHatasi::Baslik("uzunluk tasti".to_string()))?;
        if baslik_sonu > ham.len() {
            return Err(EngramTasimaHatasi::Kisa {
                gerekli: baslik_sonu,
                var: ham.len(),
            });
        }
        let baslik_metni = std::str::from_utf8(&ham[konum..baslik_sonu])
            .map_err(|e| EngramTasimaHatasi::Baslik(e.to_string()))?;
        konum = baslik_sonu;
        let baslik: serde_json::Value = serde_json::from_str(baslik_metni)
            .map_err(|e| EngramTasimaHatasi::Baslik(e.to_string()))?;
        let alan = |ad: &str| -> Result<usize, EngramTasimaHatasi> {
            let sayi = baslik
                .get(ad)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    EngramTasimaHatasi::Baslik(format!("`{ad}` yok ya da sayi degil"))
                })?;
            usize::try_from(sayi)
                .map_err(|_| EngramTasimaHatasi::Baslik(format!("`{ad}` sigmiyor")))
        };
        let spec = EngramSpec::yeni(alan("n")?, alan("tablo")?, alan("d_kv")?)
            .map_err(EngramTasimaHatasi::Sekil)?;
        let adim = baslik
            .get("adim")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| EngramTasimaHatasi::Baslik("`adim` yok".to_string()))?;
        let etiket = baslik
            .get("hassasiyet")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| EngramTasimaHatasi::Baslik("`hassasiyet` yok".to_string()))?;
        if etiket != hassasiyet.etiket() {
            return Err(EngramTasimaHatasi::Baslik(format!(
                "baslik `{etiket}` diyor, bayt `{}`",
                hassasiyet.etiket()
            )));
        }
        let kontrol_ozeti = match baslik.get("kontrol_ozeti") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(_) => {
                return Err(EngramTasimaHatasi::Baslik(
                    "`kontrol_ozeti` metin degil".to_string(),
                ))
            }
        };

        // Tek blok, adiyla.
        if konum + 2 > ham.len() {
            return Err(EngramTasimaHatasi::Kisa {
                gerekli: konum + 2,
                var: ham.len(),
            });
        }
        let ad_uzunluk = usize::from(u16::from_le_bytes([ham[konum], ham[konum + 1]]));
        konum += 2;
        if konum + ad_uzunluk + 8 > ham.len() {
            return Err(EngramTasimaHatasi::Kisa {
                gerekli: konum + ad_uzunluk + 8,
                var: ham.len(),
            });
        }
        let ad = std::str::from_utf8(&ham[konum..konum + ad_uzunluk])
            .map_err(|e| EngramTasimaHatasi::Blok(e.to_string()))?;
        if ad != ENGRAM_BLOK_ADI {
            return Err(EngramTasimaHatasi::Blok(format!(
                "beklenen `{ENGRAM_BLOK_ADI}`, dosyada `{ad}`"
            )));
        }
        konum += ad_uzunluk;
        let mut sekiz = [0u8; 8];
        sekiz.copy_from_slice(&ham[konum..konum + 8]);
        konum += 8;
        let oge = usize::try_from(u64::from_le_bytes(sekiz))
            .map_err(|_| EngramTasimaHatasi::Blok("oge sayisi sigmiyor".to_string()))?;
        let beklenen = spec.parametre_sayisi();
        if oge != beklenen {
            return Err(EngramTasimaHatasi::Uzunluk { beklenen, var: oge });
        }
        let bayt = hassasiyet.bayt();
        let blok_sonu = konum
            .checked_add(
                oge.checked_mul(bayt)
                    .ok_or(EngramTasimaHatasi::Blok("blok uzunlugu tasti".to_string()))?,
            )
            .ok_or(EngramTasimaHatasi::Blok("blok sonu tasti".to_string()))?;
        if blok_sonu + OZET_UZUNLUK > ham.len() {
            return Err(EngramTasimaHatasi::Kisa {
                gerekli: blok_sonu + OZET_UZUNLUK,
                var: ham.len(),
            });
        }
        let mut tablo: Vec<f64> = Vec::with_capacity(oge);
        for parca in ham[konum..blok_sonu].chunks_exact(bayt) {
            tablo.push(match hassasiyet {
                Hassasiyet::F64 => {
                    let mut d = [0u8; 8];
                    d.copy_from_slice(parca);
                    f64::from_le_bytes(d)
                }
                Hassasiyet::F32 => {
                    let mut d = [0u8; 4];
                    d.copy_from_slice(parca);
                    f64::from(f32::from_le_bytes(d))
                }
            });
        }
        konum = blok_sonu;
        if konum + OZET_UZUNLUK != ham.len() {
            return Err(EngramTasimaHatasi::Blok(format!(
                "ozetten sonra {} fazla bayt",
                ham.len() - konum - OZET_UZUNLUK
            )));
        }
        let beklenen_ozet = hex(&ham[konum..]);
        let bulunan_ozet = hex(&ozetle(&ham[..konum]));
        if beklenen_ozet != bulunan_ozet {
            return Err(EngramTasimaHatasi::Ozet {
                beklenen: beklenen_ozet,
                bulunan: bulunan_ozet,
            });
        }
        Ok(Self {
            spec,
            tablo,
            adim,
            kontrol_ozeti,
            hassasiyet,
        })
    }
}

fn hassasiyet_koddan(kod: u8) -> Option<Hassasiyet> {
    [Hassasiyet::F64, Hassasiyet::F32]
        .into_iter()
        .find(|h| h.kod() == kod)
}

fn ozetle(baytlar: &[u8]) -> [u8; OZET_UZUNLUK] {
    use sha2::{Digest, Sha256};
    let mut ozet = Sha256::new();
    ozet.update(baytlar);
    let cikti = ozet.finalize();
    let mut dizi = [0u8; OZET_UZUNLUK];
    dizi.copy_from_slice(&cikti);
    dizi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::belirgin_doldur;

    fn spec() -> EngramSpec {
        EngramSpec::yeni(3, 16, 4).expect("sekil")
    }

    fn tasiyici(hassasiyet: Hassasiyet) -> EngramTasiyici {
        let s = spec();
        EngramTasiyici::yeni(s, belirgin_doldur(s, 7), 12, None, hassasiyet).expect("kurulmali")
    }

    #[test]
    fn f64_yuvarlak_yol_bit_ozdes() {
        let t = tasiyici(Hassasiyet::F64);
        let mut govde = t.baytlar().expect("govde");
        govde.extend_from_slice(&ozetle(&govde));
        let geri = EngramTasiyici::baytlardan(&govde).expect("okunmali");
        assert_eq!(geri.spec, t.spec);
        assert_eq!(geri.adim, 12);
        assert!(geri
            .tablo
            .iter()
            .zip(t.tablo.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn f32_yuvarlak_yol_yalniz_yuvarlama_kadar_sapar() {
        let t = tasiyici(Hassasiyet::F32);
        let mut govde = t.baytlar().expect("govde");
        govde.extend_from_slice(&ozetle(&govde));
        let geri = EngramTasiyici::baytlardan(&govde).expect("okunmali");
        for (a, b) in geri.tablo.iter().zip(t.tablo.iter()) {
            #[allow(clippy::cast_possible_truncation)]
            let yuvarlanan = f64::from(*b as f32);
            assert_eq!(a.to_bits(), yuvarlanan.to_bits());
        }
        // Dosya boyu hassasiyetten turetilir: 4 bayt/deger.
        assert_eq!(
            govde.len(),
            9 + 1
                + 1
                + 2
                + 4
                + t.baslik_json().len()
                + 2
                + ENGRAM_BLOK_ADI.len()
                + 8
                + t.tablo.len() * 4
                + OZET_UZUNLUK
        );
    }

    #[test]
    fn uzunluk_sekle_bagli() {
        let s = spec();
        let hata = EngramTasiyici::yeni(s, vec![0.0; 5], 0, None, Hassasiyet::F64).unwrap_err();
        assert_eq!(
            hata,
            EngramTasimaHatasi::Uzunluk {
                beklenen: s.parametre_sayisi(),
                var: 5
            }
        );
    }

    #[test]
    fn tek_bayt_degisince_ozet_reddeder() {
        let t = tasiyici(Hassasiyet::F64);
        let mut govde = t.baytlar().expect("govde");
        govde.extend_from_slice(&ozetle(&govde));
        let orta = govde.len() / 2;
        govde[orta] ^= 0x01;
        assert!(matches!(
            EngramTasiyici::baytlardan(&govde),
            Err(EngramTasimaHatasi::Ozet { .. })
        ));
    }

    #[test]
    fn sabit_alanlar_basliktan_once_reddedilir() {
        let t = tasiyici(Hassasiyet::F64);
        let mut govde = t.baytlar().expect("govde");
        govde.extend_from_slice(&ozetle(&govde));
        let mut sihir = govde.clone();
        sihir[0] = b'X';
        assert_eq!(
            EngramTasiyici::baytlardan(&sihir),
            Err(EngramTasimaHatasi::Sihir)
        );
        let mut surum = govde.clone();
        surum[9] = 9;
        assert_eq!(
            EngramTasiyici::baytlardan(&surum),
            Err(EngramTasimaHatasi::Surum(9))
        );
        let mut hassas = govde.clone();
        hassas[10] = 7;
        assert_eq!(
            EngramTasiyici::baytlardan(&hassas),
            Err(EngramTasimaHatasi::Hassasiyet(7))
        );
        let mut bayrak = govde.clone();
        bayrak[11] = 1;
        assert_eq!(
            EngramTasiyici::baytlardan(&bayrak),
            Err(EngramTasimaHatasi::Bayrak(1))
        );
        // Kisa dosya: ozet olmadan.
        govde.truncate(govde.len() - 1);
        assert!(matches!(
            EngramTasiyici::baytlardan(&govde),
            Err(EngramTasimaHatasi::Kisa { .. })
        ));
    }

    #[test]
    fn basliktaki_sekil_engram_kurallariyla_dogrulanir() {
        let t = tasiyici(Hassasiyet::F64);
        let govde = t.baytlar().expect("govde");
        let metin = String::from_utf8_lossy(&govde).into_owned();
        // n = 9 ust siniri asar; baslik uzunlugu degismesin diye 3 -> 9.
        let bozuk_metin = metin.replacen("\"n\":3", "\"n\":9", 1);
        assert_ne!(metin, bozuk_metin);
        let mut bozuk = bozuk_metin.into_bytes();
        bozuk.extend_from_slice(&ozetle(&bozuk));
        assert_eq!(
            EngramTasiyici::baytlardan(&bozuk),
            Err(EngramTasimaHatasi::Sekil(
                EngramSekilHatasi::NUstSinirAsildi
            ))
        );
    }

    #[test]
    fn bagsiz_tablo_cevap_vermez_bagli_tablo_sahibini_tanir() {
        use crate::kontrol::OptimizerDurumu;
        use crate::{Adamw, Parametreler, Spec, INIT_STD_EMBEDDING};

        let s = Spec {
            vocab: 16,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            qkv_dokunus: 0,
            qk_norm: false,
            d_ff: 16,
            max_seq_len: 8,
        };
        let p = Parametreler::mup_init(s, 9, INIT_STD_EMBEDDING);
        let opt = Adamw::yeni(p.toplam_ogeler(), 0.01, 0.1).expect("optimizer");
        let (adim, m, v) = opt.durum();
        let kontrol = Kontrol {
            spec: s,
            parametreler: p,
            adim: 7,
            epoch: 2,
            tohum: 9,
            sozluk_aile: "aile".to_string(),
            korpus_ozeti: "a".repeat(64),
            egitim_kaybi: 3.5,
            dogrulama_kaybi: None,
            en_iyi_dogrulama: None,
            devam_konum: 0,
            hassasiyet: Hassasiyet::F64,
            optimizer: Some(OptimizerDurumu {
                adim,
                ogrenme_orani: 0.01,
                agirlik_sonumu: 0.1,
                m: m.to_vec(),
                v: v.to_vec(),
            }),
        };

        let mut t = tasiyici(Hassasiyet::F64);
        assert_eq!(t.ait_mi(&kontrol).expect("hesap"), None);
        let ozet = t.bagla(&kontrol).expect("baglanmali");
        assert_eq!(ozet.len(), 64);
        assert_eq!(t.adim, kontrol.adim);
        assert_eq!(t.ait_mi(&kontrol).expect("hesap"), Some(true));

        // Baglilik dosyadan gecer.
        let mut govde = t.baytlar().expect("govde");
        govde.extend_from_slice(&ozetle(&govde));
        let geri = EngramTasiyici::baytlardan(&govde).expect("okunmali");
        assert_eq!(geri.ait_mi(&kontrol).expect("hesap"), Some(true));

        // Baska bir kontrol noktasi: ayni sekil, farkli agirlik.
        let mut baska = kontrol.clone();
        baska.parametreler.bloklar_mut()[0][0] += 1.0;
        assert_eq!(geri.ait_mi(&baska).expect("hesap"), Some(false));
    }

    #[test]
    fn dosyaya_yaz_ve_yukle() {
        let dizin = std::env::temp_dir().join(format!(
            "lubot-engram-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dizin).expect("dizin");
        let yol = dizin.join("engram.bin");
        let t = tasiyici(Hassasiyet::F32);
        let ozet = t.yaz(&yol).expect("yazilmali");
        let geri = EngramTasiyici::yukle(&yol).expect("yuklenmeli");
        assert_eq!(geri.spec, t.spec);
        assert_eq!(geri.hassasiyet, Hassasiyet::F32);
        let ham = std::fs::read(&yol).expect("oku");
        assert_eq!(hex(&ham[ham.len() - OZET_UZUNLUK..]), ozet);
        std::fs::remove_dir_all(&dizin).expect("temizlik");
    }

    /// Olcum kaydi: `training/engram_tasima.py` bu satiri kosar ve okur.
    #[test]
    fn olcum_raporu_engram_tasima() {
        let t64 = tasiyici(Hassasiyet::F64);
        let t32 = tasiyici(Hassasiyet::F32);
        let parametre = t64.spec.parametre_sayisi();
        let mut g64 = t64.baytlar().expect("govde");
        g64.extend_from_slice(&ozetle(&g64));
        let mut g32 = t32.baytlar().expect("govde");
        g32.extend_from_slice(&ozetle(&g32));
        let geri64 = EngramTasiyici::baytlardan(&g64).expect("okunmali");
        let geri32 = EngramTasiyici::baytlardan(&g32).expect("okunmali");
        let bit_ozdes = usize::from(
            geri64
                .tablo
                .iter()
                .zip(t64.tablo.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        );
        let f32_sapma = geri32
            .tablo
            .iter()
            .zip(t32.tablo.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let ust_sinir = t32
            .tablo
            .iter()
            .fold(0.0f64, |m, b| m.max(b.abs() * f64::from(f32::EPSILON)));
        let f32_sapma_sinir_icinde = usize::from(f32_sapma <= ust_sinir);
        // Her bayt cevrilince ozet reddediyor mu (ozetin kendisi haric).
        let govde_uzunluk = g64.len() - OZET_UZUNLUK;
        let mut yakalanan = 0usize;
        for i in 0..govde_uzunluk {
            let mut bozuk = g64.clone();
            bozuk[i] ^= 0x80;
            if EngramTasiyici::baytlardan(&bozuk).is_err() {
                yakalanan += 1;
            }
        }
        let bayt_64 = g64.len();
        let bayt_32 = g32.len();
        println!(
            "engram-tasima | n={} tablo={} d_kv={} parametre={parametre} bayt_64={bayt_64} bayt_32={bayt_32} bit_ozdes={bit_ozdes} f32_sapma={f32_sapma:.6e} f32_sapma_sinir_icinde={f32_sapma_sinir_icinde} cevrilen_bayt={govde_uzunluk} yakalanan={yakalanan}",
            t64.spec.n, t64.spec.tablo, t64.spec.d_kv
        );
    }
}
