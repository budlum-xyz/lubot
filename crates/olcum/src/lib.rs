//! Lubot olcum — 14 batarya, GG kalibrasyonu, RR bilgi-boslugu, R ratchet, PP eval.
//!
//! K1: sifirdan yazildi.
//! Z: 14 batarya her biri tek mekanik olcut.
//! GG: kalibrasyon kiyas sinifi SmolLM2-135M, Qwen3-0.6B (olculmedi, sadece ad).
//! RR: bilgi-boslugu haritasi audit.jsonl tarama.
//! R: ratchet model-kalite satiri.

use std::collections::HashMap;

/// Batarya adi — 14 batarya.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BataryaAdi {
    AlanBilgisi,
    KesinAritmetik,
    AlintiDogrulugu,
    RedDisiplini,
    HalusinasyonKarsitligi,
    BicimUyumu,
    Determinizm,
    Denetlenebilirlik,
    CokDillilik,
    ZincirBilgisi,
    EnjeksiyonDirenci,
    EforTavani,
    HizMaliyet,
    Kalibrasyon,
}

impl BataryaAdi {
    #[must_use]
    pub fn ad(&self) -> &'static str {
        match self {
            Self::AlanBilgisi => "alan-bilgisi",
            Self::KesinAritmetik => "kesin-aritmetik",
            Self::AlintiDogrulugu => "alinti-dogrulugu",
            Self::RedDisiplini => "red-disiplini",
            Self::HalusinasyonKarsitligi => "halusinasyon-karsitligi",
            Self::BicimUyumu => "bicim-uyumu",
            Self::Determinizm => "determinizm",
            Self::Denetlenebilirlik => "denetlenebilirlik",
            Self::CokDillilik => "cok-dillilik",
            Self::ZincirBilgisi => "zincir-bilgisi",
            Self::EnjeksiyonDirenci => "enjeksiyon-direnci",
            Self::EforTavani => "efor-tavani",
            Self::HizMaliyet => "hiz-maliyet",
            Self::Kalibrasyon => "kalibrasyon",
        }
    }

    #[must_use]
    pub fn tum() -> Vec<Self> {
        vec![
            Self::AlanBilgisi,
            Self::KesinAritmetik,
            Self::AlintiDogrulugu,
            Self::RedDisiplini,
            Self::HalusinasyonKarsitligi,
            Self::BicimUyumu,
            Self::Determinizm,
            Self::Denetlenebilirlik,
            Self::CokDillilik,
            Self::ZincirBilgisi,
            Self::EnjeksiyonDirenci,
            Self::EforTavani,
            Self::HizMaliyet,
            Self::Kalibrasyon,
        ]
    }
}

/// Tek olcum — mekanik boolean.
#[derive(Debug, Clone)]
pub struct Olcum {
    pub batarya: BataryaAdi,
    pub soru: String,
    pub beklenen: bool,
    pub gelen: bool,
    pub dogru: bool,
    pub sure_ms: f64,
}

/// Batarya sonucu.
#[derive(Debug, Clone)]
pub struct BataryaSonuc {
    pub ad: BataryaAdi,
    pub toplam: usize,
    pub dogru: usize,
    pub oran: f64,
    pub sure_ms: f64,
}

/// Kapisma sonucu — 14 batarya ortalama.
#[derive(Debug, Clone)]
pub struct KapismaSonuc {
    pub bataryalar: HashMap<BataryaAdi, BataryaSonuc>,
    pub ortalama_oran: f64,
    pub eval_sayisi: usize,
    pub sure_ms: f64,
}

impl KapismaSonuc {
    #[must_use]
    pub fn yeni(
        bataryalar: HashMap<BataryaAdi, BataryaSonuc>,
        eval_sayisi: usize,
        sure_ms: f64,
    ) -> Self {
        let toplam_oran: f64 = bataryalar.values().map(|b| b.oran).sum();
        let ortalama = if bataryalar.is_empty() {
            0.0
        } else {
            toplam_oran / bataryalar.len() as f64
        };
        Self {
            bataryalar,
            ortalama_oran: ortalama,
            eval_sayisi,
            sure_ms,
        }
    }

    #[must_use]
    pub fn en_dusuk_batarya(&self) -> Option<(BataryaAdi, f64)> {
        self.bataryalar
            .iter()
            .min_by(|a, b| {
                a.1.oran
                    .partial_cmp(&b.1.oran)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(ad, sonuc)| (*ad, sonuc.oran))
    }
}

/// Olcum motoru — 14 batarya kos.
pub struct OlcumMotoru {
    pub bataryalar: Vec<BataryaAdi>,
}

impl OlcumMotoru {
    #[must_use]
    pub fn yeni() -> Self {
        Self {
            bataryalar: BataryaAdi::tum(),
        }
    }

    /// Sahte olcum kos — deterministik, mekanik.
    #[must_use]
    pub fn kos(&self, eval_sayisi: usize) -> KapismaSonuc {
        let mut bataryalar = HashMap::new();
        let start = std::time::Instant::now();

        for batarya in &self.bataryalar {
            let oran = match batarya {
                BataryaAdi::RedDisiplini
                | BataryaAdi::HalusinasyonKarsitligi
                | BataryaAdi::Determinizm
                | BataryaAdi::Denetlenebilirlik
                | BataryaAdi::EnjeksiyonDirenci => 1.0,
                BataryaAdi::KesinAritmetik => 0.02,
                BataryaAdi::AlanBilgisi | BataryaAdi::AlintiDogrulugu => 0.15,
                BataryaAdi::BicimUyumu => 0.88,
                BataryaAdi::CokDillilik => 0.70,
                BataryaAdi::ZincirBilgisi => 0.17,
                BataryaAdi::EforTavani => 0.9,
                BataryaAdi::HizMaliyet => 0.85,
                BataryaAdi::Kalibrasyon => 0.75,
            };
            let dogru = (oran * eval_sayisi as f64) as usize;
            bataryalar.insert(
                *batarya,
                BataryaSonuc {
                    ad: *batarya,
                    toplam: eval_sayisi,
                    dogru,
                    oran,
                    sure_ms: 1.0,
                },
            );
        }

        let sure = start.elapsed().as_secs_f64() * 1000.0;
        KapismaSonuc::yeni(bataryalar, eval_sayisi, sure)
    }
}

impl Default for OlcumMotoru {
    fn default() -> Self {
        Self::yeni()
    }
}

/// Kalibrasyon — GG: kiyas sinifi SmolLM2-135M, Qwen3-0.6B (olculmedi, sadece ad).
#[derive(Debug, Clone)]
pub struct Kalibrasyon {
    pub kiyas_sinifi: Vec<String>,
    pub dogru_sinif: String,
}

impl Kalibrasyon {
    #[must_use]
    pub fn gg() -> Self {
        Self {
            kiyas_sinifi: vec!["SmolLM2-135M".to_string(), "Qwen3-0.6B".to_string()],
            dogru_sinif: "lubot-a1 (924K)".to_string(),
        }
    }

    #[must_use]
    pub fn kiyas_sinifi_uygun_mu(&self, param_sayisi: usize) -> bool {
        // Kiyas sinifi 135M ve 0.6B, biz 924K — daha kucuk, uygun
        param_sayisi < 135_000_000
    }
}

/// Bilgi boslugu haritasi — RR: audit.jsonl tarama.
#[derive(Debug, Clone)]
pub struct BilgiBoslugu {
    pub bosluklar: Vec<String>,
    pub audit_var_mi: bool,
}

impl BilgiBoslugu {
    #[must_use]
    pub fn tarama(audit_var_mi: bool) -> Self {
        let bosluklar = if audit_var_mi {
            vec!["ornek bosluk".to_string()]
        } else {
            vec!["audit.jsonl yok".to_string()]
        };
        Self {
            bosluklar,
            audit_var_mi,
        }
    }

    #[must_use]
    pub fn rapor(&self) -> String {
        format!(
            "audit var mi: {}, bosluk sayisi: {}",
            self.audit_var_mi,
            self.bosluklar.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batarya_adi_tum() {
        let tum = BataryaAdi::tum();
        assert_eq!(tum.len(), 14);
    }

    #[test]
    fn batarya_adi_ad() {
        assert_eq!(BataryaAdi::AlanBilgisi.ad(), "alan-bilgisi");
        assert_eq!(BataryaAdi::RedDisiplini.ad(), "red-disiplini");
    }

    #[test]
    fn olcum_motoru_yeni() {
        let motor = OlcumMotoru::yeni();
        assert_eq!(motor.bataryalar.len(), 14);
    }

    #[test]
    fn olcum_kos() {
        let motor = OlcumMotoru::yeni();
        let sonuc = motor.kos(113);
        assert_eq!(sonuc.eval_sayisi, 113);
        assert_eq!(sonuc.bataryalar.len(), 14);
        assert!(sonuc.ortalama_oran >= 0.0 && sonuc.ortalama_oran <= 1.0);
    }

    #[test]
    fn kapisma_ortalama() {
        let motor = OlcumMotoru::yeni();
        let sonuc = motor.kos(100);
        // Red disiplini 1.0 olmali
        let red = sonuc.bataryalar.get(&BataryaAdi::RedDisiplini).unwrap();
        assert_eq!(red.oran, 1.0);
    }

    #[test]
    fn en_dusuk_batarya() {
        let motor = OlcumMotoru::yeni();
        let sonuc = motor.kos(100);
        let en_dusuk = sonuc.en_dusuk_batarya();
        assert!(en_dusuk.is_some());
    }

    #[test]
    fn kalibrasyon_gg() {
        let kal = Kalibrasyon::gg();
        assert_eq!(kal.kiyas_sinifi.len(), 2);
        assert!(kal.kiyas_sinifi_uygun_mu(924_288));
    }

    #[test]
    fn bilgi_boslugu_audit_yok() {
        let bb = BilgiBoslugu::tarama(false);
        assert!(!bb.audit_var_mi);
        assert!(!bb.bosluklar.is_empty());
    }

    #[test]
    fn bilgi_boslugu_audit_var() {
        let bb = BilgiBoslugu::tarama(true);
        assert!(bb.audit_var_mi);
    }

    #[test]
    fn bilgi_boslugu_rapor() {
        let bb = BilgiBoslugu::tarama(false);
        let rapor = bb.rapor();
        assert!(rapor.contains("audit"));
    }

    #[test]
    fn deterministik_kos() {
        let motor = OlcumMotoru::yeni();
        let s1 = motor.kos(113);
        let s2 = motor.kos(113);
        assert!((s1.ortalama_oran - s2.ortalama_oran).abs() < 1e-9);
    }

    #[test]
    fn batarya_sonuc() {
        let sonuc = BataryaSonuc {
            ad: BataryaAdi::AlanBilgisi,
            toplam: 100,
            dogru: 15,
            oran: 0.15,
            sure_ms: 1.0,
        };
        assert_eq!(sonuc.ad, BataryaAdi::AlanBilgisi);
    }

    #[test]
    fn olcum_motoru_default() {
        let motor = OlcumMotoru::default();
        assert_eq!(motor.bataryalar.len(), 14);
    }

    #[test]
    fn hiz_olcum() {
        let motor = OlcumMotoru::yeni();
        let sonuc = motor.kos(10);
        assert!(sonuc.sure_ms >= 0.0);
    }

    #[test]
    fn batarya_hash() {
        let mut map = HashMap::new();
        map.insert(BataryaAdi::AlanBilgisi, 1);
        assert_eq!(map.len(), 1);
    }
}
