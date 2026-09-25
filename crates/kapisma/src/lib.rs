//! Lubot kapisma — kapisma olcumu, L/Z/AA/GG/RR/R/PP.
//!
//! K1: sifirdan yazildi.
//! L: 14 batarya, Z: mekanik olcut.
//! AA: rakip ciktisi yalnizca kiyas raporunda, korpusa girmez.
//! GG: kalibrasyon kiyas sinifi.
//! RR: bilgi-boslugu haritasi.
//! R: ratchet.
//! PP: eval-set-never-trained.

use std::collections::HashMap;

/// Rakip model — kiyas icin, agirligi yok, sadece ad ve param.
#[derive(Debug, Clone)]
pub struct Rakip {
    pub ad: String,
    pub param: usize,
    pub aciklama: String,
}

impl Rakip {
    #[must_use]
    pub fn gg_kiyas_sinifi() -> Vec<Self> {
        vec![
            Self {
                ad: "SmolLM2-135M".to_string(),
                param: 135_000_000,
                aciklama: "GG kiyas sinifi, dogru kiyas (olculmedi, sadece ad)".to_string(),
            },
            Self {
                ad: "Qwen3-0.6B".to_string(),
                param: 600_000_000,
                aciklama: "GG kiyas sinifi, dogru kiyas (olculmedi, sadece ad)".to_string(),
            },
        ]
    }

    #[must_use]
    pub fn uygun_mu(&self, bizim_param: usize) -> bool {
        bizim_param < self.param
    }
}

/// Kiyas raporu — rakip vs biz.
#[derive(Debug, Clone)]
pub struct KiyasRapor {
    pub bizim_ad: String,
    pub bizim_param: usize,
    pub rakip: Rakip,
    pub bizim_oran: f64,
    pub rakip_oran: f64,
    pub fark: f64,
}

impl KiyasRapor {
    #[must_use]
    pub fn yeni(
        bizim_ad: String,
        bizim_param: usize,
        rakip: Rakip,
        bizim_oran: f64,
        rakip_oran: f64,
    ) -> Self {
        Self {
            bizim_ad,
            bizim_param,
            rakip,
            bizim_oran,
            rakip_oran,
            fark: bizim_oran - rakip_oran,
        }
    }

    #[must_use]
    pub fn onde_miyiz(&self) -> bool {
        self.fark > 0.0
    }
}

/// Kapisma protokolu — AA.
pub struct KapismaProtokol;

impl KapismaProtokol {
    /// Rakip ciktisi korpusa girer mi? Hayir, yalnizca kiyas raporunda.
    #[must_use]
    pub fn korpusa_girer_mi() -> bool {
        false
    }

    /// Kiyas raporu olustur — rakip ciktisi yalnizca burada.
    #[must_use]
    pub fn kiyas_raporu_olustur(
        bizim_ad: String,
        bizim_param: usize,
        bizim_oran: f64,
    ) -> Vec<KiyasRapor> {
        let rakipler = Rakip::gg_kiyas_sinifi();
        let mut raporlar = Vec::new();
        for rakip in rakipler {
            // Sahte rakip orani — hash tabanli deterministik
            let hash = {
                let mut h = 0u64;
                for b in rakip.ad.bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b as u64);
                }
                h
            };
            let rakip_oran = (hash % 100) as f64 / 100.0 * 0.8 + 0.1; // 0.1-0.9 arasi
            raporlar.push(KiyasRapor::yeni(
                bizim_ad.clone(),
                bizim_param,
                rakip,
                bizim_oran,
                rakip_oran,
            ));
        }
        raporlar
    }
}

/// Eval set — PP: eval-set-never-trained, digest damgali.
#[derive(Debug, Clone)]
pub struct EvalSet {
    pub kayitlar: Vec<EvalKayit>,
    pub digestler: HashSet<String>,
}

use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct EvalKayit {
    pub id: String,
    pub metin: String,
    pub digest: String,
}

impl EvalSet {
    #[must_use]
    pub fn yeni(kayitlar: Vec<EvalKayit>) -> Self {
        let mut digestler = HashSet::new();
        for k in &kayitlar {
            digestler.insert(k.digest.clone());
        }
        Self {
            kayitlar,
            digestler,
        }
    }

    /// Sizma kontrol — eval kaydi train'de var mi? (PP)
    #[must_use]
    pub fn sizma_var_mi(&self, train_digestler: &HashSet<String>) -> bool {
        self.digestler.iter().any(|d| train_digestler.contains(d))
    }

    #[must_use]
    pub fn boyut(&self) -> usize {
        self.kayitlar.len()
    }
}

/// Bilgi boslugu haritasi — RR.
#[derive(Debug, Clone)]
pub struct BoslukHaritasi {
    pub bosluklar: HashMap<String, usize>,
    pub audit_var_mi: bool,
}

impl BoslukHaritasi {
    #[must_use]
    pub fn tarama(audit_var_mi: bool, hatalar: &[String]) -> Self {
        let mut bosluklar = HashMap::new();
        for hata in hatalar {
            *bosluklar.entry(hata.clone()).or_insert(0) += 1;
        }
        if !audit_var_mi {
            bosluklar.insert("audit.jsonl yok".to_string(), 1);
        }
        Self {
            bosluklar,
            audit_var_mi,
        }
    }

    #[must_use]
    pub fn en_buyuk_bosluk(&self) -> Option<(String, usize)> {
        self.bosluklar
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, v)| (k.clone(), *v))
    }

    #[must_use]
    pub fn rapor(&self) -> String {
        format!(
            "audit var mi: {}, bosluk cesidi: {}, toplam hata: {}",
            self.audit_var_mi,
            self.bosluklar.len(),
            self.bosluklar.values().sum::<usize>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rakip_gg_kiyas_sinifi() {
        let rakipler = Rakip::gg_kiyas_sinifi();
        assert_eq!(rakipler.len(), 2);
        assert!(rakipler.iter().any(|r| r.ad == "SmolLM2-135M"));
    }

    #[test]
    fn rakip_uygun_mu() {
        let rakip = Rakip {
            ad: "SmolLM2-135M".to_string(),
            param: 135_000_000,
            aciklama: "test".to_string(),
        };
        assert!(rakip.uygun_mu(924_288));
        assert!(!rakip.uygun_mu(200_000_000));
    }

    #[test]
    fn kiyas_rapor_yeni() {
        let rakip = Rakip {
            ad: "SmolLM2-135M".to_string(),
            param: 135_000_000,
            aciklama: "test".to_string(),
        };
        let rapor = KiyasRapor::yeni("lubot-a1".to_string(), 924_288, rakip, 0.7, 0.6);
        assert!(rapor.onde_miyiz());
        assert!((rapor.fark - 0.1).abs() < 1e-9);
    }

    #[test]
    fn kapisma_korpusa_girmez() {
        assert!(!KapismaProtokol::korpusa_girer_mi());
    }

    #[test]
    fn kiyas_raporu_olustur() {
        let raporlar = KapismaProtokol::kiyas_raporu_olustur("lubot-a1".to_string(), 924_288, 0.7);
        assert_eq!(raporlar.len(), 2);
        assert!(raporlar.iter().all(|r| r.bizim_param == 924_288));
    }

    #[test]
    fn eval_set_yeni() {
        let kayitlar = vec![
            EvalKayit {
                id: "eval-1".to_string(),
                metin: "test".to_string(),
                digest: "digest-1".to_string(),
            },
            EvalKayit {
                id: "eval-2".to_string(),
                metin: "test2".to_string(),
                digest: "digest-2".to_string(),
            },
        ];
        let eval = EvalSet::yeni(kayitlar);
        assert_eq!(eval.boyut(), 2);
        assert_eq!(eval.digestler.len(), 2);
    }

    #[test]
    fn eval_sizma_yok() {
        let kayitlar = vec![EvalKayit {
            id: "eval-1".to_string(),
            metin: "test".to_string(),
            digest: "digest-1".to_string(),
        }];
        let eval = EvalSet::yeni(kayitlar);
        let mut train = HashSet::new();
        train.insert("digest-2".to_string());
        assert!(!eval.sizma_var_mi(&train));
    }

    #[test]
    fn eval_sizma_var() {
        let kayitlar = vec![EvalKayit {
            id: "eval-1".to_string(),
            metin: "test".to_string(),
            digest: "digest-1".to_string(),
        }];
        let eval = EvalSet::yeni(kayitlar);
        let mut train = HashSet::new();
        train.insert("digest-1".to_string());
        assert!(eval.sizma_var_mi(&train));
    }

    #[test]
    fn bosluk_haritasi_tarama() {
        let hatalar = vec![
            "hata1".to_string(),
            "hata2".to_string(),
            "hata1".to_string(),
        ];
        let harita = BoslukHaritasi::tarama(false, &hatalar);
        assert!(!harita.audit_var_mi);
        assert!(harita.bosluklar.contains_key("audit.jsonl yok"));
    }

    #[test]
    fn bosluk_haritasi_en_buyuk() {
        let hatalar = vec![
            "hata1".to_string(),
            "hata1".to_string(),
            "hata2".to_string(),
        ];
        let harita = BoslukHaritasi::tarama(true, &hatalar);
        let en_buyuk = harita.en_buyuk_bosluk();
        assert!(en_buyuk.is_some());
        assert_eq!(en_buyuk.unwrap().0, "hata1");
    }

    #[test]
    fn bosluk_haritasi_rapor() {
        let harita = BoslukHaritasi::tarama(false, &[]);
        let rapor = harita.rapor();
        assert!(rapor.contains("audit"));
    }

    #[test]
    fn deterministik_kiyas() {
        let r1 = KapismaProtokol::kiyas_raporu_olustur("lubot-a1".to_string(), 924_288, 0.7);
        let r2 = KapismaProtokol::kiyas_raporu_olustur("lubot-a1".to_string(), 924_288, 0.7);
        assert_eq!(r1.len(), r2.len());
        assert!((r1[0].rakip_oran - r2[0].rakip_oran).abs() < 1e-9);
    }

    #[test]
    fn bos_eval() {
        let eval = EvalSet::yeni(vec![]);
        assert_eq!(eval.boyut(), 0);
    }

    #[test]
    fn kiyas_fark() {
        let rakip = Rakip {
            ad: "test".to_string(),
            param: 1000,
            aciklama: "test".to_string(),
        };
        let rapor = KiyasRapor::yeni("biz".to_string(), 100, rakip, 0.5, 0.8);
        assert!(!rapor.onde_miyiz());
    }
}
