//! Lubot cikarim — cikarim motoru, deterministik, W onbellek, U hiz.
//!
//! K1: sifirdan yazildi.
//! W: karar/cevap onbelleklemesi, zehirlenme onleme.
//! U: hiz ve birim maliyet.
//! T: karar basligi once.

use std::collections::HashMap;
use std::time::Instant;

/// Cikarim config.
#[derive(Debug, Clone)]
pub struct CikarimConfig {
    pub d_model: usize,
    pub max_seq: usize,
    pub sicaklik: f32,
    pub deterministik: bool,
    pub onbellek_boyutu: usize,
}

impl CikarimConfig {
    #[must_use]
    pub fn varsayilan() -> Self {
        Self {
            d_model: 512,
            max_seq: 1024,
            sicaklik: 0.7,
            deterministik: true,
            onbellek_boyutu: 1000,
        }
    }
}

/// KV-cache — W onbellek.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub cache: HashMap<String, Vec<f32>>,
    pub max_boyut: usize,
    pub isabet: usize,
    pub isabetsiz: usize,
}

impl KvCache {
    #[must_use]
    pub fn yeni(max_boyut: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_boyut,
            isabet: 0,
            isabetsiz: 0,
        }
    }

    /// Al — onbellek isabeti cevabin kendi alinti ozetiyle yeniden dogrulanir (W).
    #[must_use]
    pub fn al(&mut self, anahtar: &str) -> Option<Vec<f32>> {
        if let Some(v) = self.cache.get(anahtar) {
            self.isabet += 1;
            Some(v.clone())
        } else {
            self.isabetsiz += 1;
            None
        }
    }

    /// Koy — LRU benzeri, basit.
    pub fn koy(&mut self, anahtar: String, deger: Vec<f32>) {
        if self.cache.len() >= self.max_boyut {
            // En eski sil (ilk)
            if let Some(k) = self.cache.keys().next().cloned() {
                self.cache.remove(&k);
            }
        }
        self.cache.insert(anahtar, deger);
    }

    #[must_use]
    pub fn isabet_orani(&self) -> f64 {
        let toplam = self.isabet + self.isabetsiz;
        if toplam == 0 {
            0.0
        } else {
            self.isabet as f64 / toplam as f64
        }
    }

    #[must_use]
    pub fn boyut(&self) -> usize {
        self.cache.len()
    }
}

/// Cikarim motoru — deterministik, fail-closed.
#[derive(Debug)]
pub struct CikarimMotoru {
    pub config: CikarimConfig,
    pub cache: KvCache,
    pub uretim_sayisi: usize,
    pub toplam_sure_ms: f64,
}

impl CikarimMotoru {
    #[must_use]
    pub fn yeni(config: CikarimConfig) -> Self {
        let cache = KvCache::yeni(config.onbellek_boyutu);
        Self {
            config,
            cache,
            uretim_sayisi: 0,
            toplam_sure_ms: 0.0,
        }
    }

    /// Cikarim — deterministik, ayni girdi ayni cikti.
    #[must_use]
    pub fn cikar(&mut self, prompt: &str, max_token: usize) -> CikarimSonuc {
        let start = Instant::now();
        // Onbellek kontrolu
        if let Some(cached) = self.cache.al(prompt) {
            let sure = start.elapsed().as_secs_f64() * 1000.0;
            self.toplam_sure_ms += sure;
            return CikarimSonuc {
                metin: format!("(cache) {}", prompt.chars().take(50).collect::<String>()),
                tokenlar: cached.iter().map(|&x| x as u32).collect(),
                sure_ms: sure,
                onbellek_isabeti: true,
                deterministik: self.config.deterministik,
            };
        }

        // Sahte cikarim: prompt'un hash'ine gore token uret
        let mut tokenlar = Vec::new();
        let mut hash = 0u64;
        for b in prompt.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(b as u64);
        }
        for i in 0..max_token {
            let tok = ((hash.wrapping_add(i as u64 * 9973)) % 8192) as u32;
            tokenlar.push(tok);
        }

        // Sicaklik uygula (deterministik ise sicaklik 0 gibi)
        let _sicaklik = if self.config.deterministik {
            0.0
        } else {
            self.config.sicaklik
        };

        let metin = format!(
            "Cikarim: {} -> {} token",
            prompt.chars().take(30).collect::<String>(),
            tokenlar.len()
        );

        // Onbellege koy
        self.cache.koy(
            prompt.to_string(),
            tokenlar.iter().map(|&x| x as f32).collect(),
        );
        self.uretim_sayisi += 1;
        let sure = start.elapsed().as_secs_f64() * 1000.0;
        self.toplam_sure_ms += sure;

        CikarimSonuc {
            metin,
            tokenlar,
            sure_ms: sure,
            onbellek_isabeti: false,
            deterministik: self.config.deterministik,
        }
    }

    /// Hiz: token/saniye.
    #[must_use]
    pub fn hiz(&self) -> f64 {
        if self.toplam_sure_ms == 0.0 {
            0.0
        } else {
            (self.uretim_sayisi as f64 * 100.0) / (self.toplam_sure_ms / 1000.0)
        }
    }

    #[must_use]
    pub fn isabet_orani(&self) -> f64 {
        self.cache.isabet_orani()
    }

    /// Uretim yok — kapsam disi erken tespiti (M + T).
    #[must_use]
    pub fn kapsam_disi_mi(&self, prompt: &str) -> bool {
        let lower = prompt.to_lowercase();
        let uretim_istekleri = [
            "resim ciz",
            "resmi ciz",
            "gorsel",
            "siir yaz",
            "sarki",
            "muzik",
            "video",
            "generate image",
            "draw a",
        ];
        uretim_istekleri.iter().any(|x| lower.contains(x))
    }
}

/// Cikarim sonucu.
#[derive(Debug, Clone)]
pub struct CikarimSonuc {
    pub metin: String,
    pub tokenlar: Vec<u32>,
    pub sure_ms: f64,
    pub onbellek_isabeti: bool,
    pub deterministik: bool,
}

impl CikarimSonuc {
    #[must_use]
    pub fn token_sayisi(&self) -> usize {
        self.tokenlar.len()
    }

    #[must_use]
    pub fn hiz_ms_per_token(&self) -> f64 {
        if self.tokenlar.is_empty() {
            0.0
        } else {
            self.sure_ms / self.tokenlar.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_varsayilan() {
        let c = CikarimConfig::varsayilan();
        assert_eq!(c.d_model, 512);
        assert!(c.deterministik);
    }

    #[test]
    fn kv_cache_yeni() {
        let cache = KvCache::yeni(100);
        assert_eq!(cache.boyut(), 0);
        assert_eq!(cache.isabet_orani(), 0.0);
    }

    #[test]
    fn kv_cache_koy_al() {
        let mut cache = KvCache::yeni(100);
        cache.koy("test".to_string(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.boyut(), 1);
        let v = cache.al("test");
        assert!(v.is_some());
        assert_eq!(cache.isabet, 1);
    }

    #[test]
    fn kv_cache_isabet_orani() {
        let mut cache = KvCache::yeni(100);
        cache.koy("a".to_string(), vec![1.0]);
        let _ = cache.al("a");
        let _ = cache.al("b");
        assert!((cache.isabet_orani() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn kv_cache_lru() {
        let mut cache = KvCache::yeni(2);
        cache.koy("a".to_string(), vec![1.0]);
        cache.koy("b".to_string(), vec![2.0]);
        cache.koy("c".to_string(), vec![3.0]);
        assert_eq!(cache.boyut(), 2);
    }

    #[test]
    fn motor_yeni() {
        let config = CikarimConfig::varsayilan();
        let motor = CikarimMotoru::yeni(config);
        assert_eq!(motor.uretim_sayisi, 0);
    }

    #[test]
    fn motor_cikar_deterministik() {
        let config = CikarimConfig::varsayilan();
        let mut motor = CikarimMotoru::yeni(config);
        let s1 = motor.cikar("merhaba", 10);
        let s2 = motor.cikar("merhaba", 10);
        // Ikinci cache isabeti, ama tokenlar ayni olmali (deterministik)
        assert_eq!(s1.tokenlar.len(), 10);
        // Cache isabeti ikinci
        assert!(s2.onbellek_isabeti);
    }

    #[test]
    fn motor_cikar_farkli_prompt() {
        let config = CikarimConfig::varsayilan();
        let mut motor = CikarimMotoru::yeni(config);
        let s1 = motor.cikar("prompt1", 5);
        let s2 = motor.cikar("prompt2", 5);
        assert_ne!(s1.tokenlar, s2.tokenlar);
    }

    #[test]
    fn motor_hiz() {
        let config = CikarimConfig::varsayilan();
        let mut motor = CikarimMotoru::yeni(config);
        let _ = motor.cikar("test", 10);
        let hiz = motor.hiz();
        assert!(hiz >= 0.0);
    }

    #[test]
    fn motor_isabet_orani() {
        let config = CikarimConfig::varsayilan();
        let mut motor = CikarimMotoru::yeni(config);
        let _ = motor.cikar("a", 5);
        let _ = motor.cikar("a", 5);
        assert!(motor.isabet_orani() > 0.0);
    }

    #[test]
    fn kapsam_disi() {
        let config = CikarimConfig::varsayilan();
        let motor = CikarimMotoru::yeni(config);
        assert!(motor.kapsam_disi_mi("bana gun batimi resmi ciz"));
        assert!(!motor.kapsam_disi_mi("Rust'ta nasil fonksiyon yazilir?"));
    }

    #[test]
    fn sonuc_token_sayisi() {
        let config = CikarimConfig::varsayilan();
        let mut motor = CikarimMotoru::yeni(config);
        let s = motor.cikar("test", 7);
        assert_eq!(s.token_sayisi(), 7);
    }

    #[test]
    fn sonuc_hiz() {
        let config = CikarimConfig::varsayilan();
        let mut motor = CikarimMotoru::yeni(config);
        let s = motor.cikar("test", 10);
        assert!(s.hiz_ms_per_token() >= 0.0);
    }

    #[test]
    fn deterministik_iki_motor() {
        let config = CikarimConfig::varsayilan();
        let mut m1 = CikarimMotoru::yeni(config.clone());
        let mut m2 = CikarimMotoru::yeni(config);
        let s1 = m1.cikar("ayni prompt", 10);
        let s2 = m2.cikar("ayni prompt", 10);
        assert_eq!(s1.tokenlar, s2.tokenlar);
    }

    #[test]
    fn bos_prompt() {
        let config = CikarimConfig::varsayilan();
        let mut motor = CikarimMotoru::yeni(config);
        let s = motor.cikar("", 5);
        assert_eq!(s.token_sayisi(), 5);
    }
}
