//! Lubot derin — derin-dar mimari, agirlik baglama, model spec.
//!
//! K1: sifirdan yazildi.
//! F: sifirdan egitim iskeleti, derin-dar (12 katman, 512 gizli).
//! II: agirlik baglama — embedding = readout^T, logit_scale 1/d_model.
//! K6: tavan altinda (97M).

use std::collections::HashMap;

/// Model config — derin-dar.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub ad: String,
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub max_seq: usize,
    pub dropout: f32,
    pub agirlik_baglama: bool,
}

impl Config {
    #[must_use]
    pub fn lubot_a1() -> Self {
        // training/model_spec.json ile uyumlu: 64,8,2,256 -> 924K param
        Self {
            ad: "lubot-a1-derin-dar".to_string(),
            d_model: 64,
            n_layer: 8,
            n_head: 2,
            d_ff: 256,
            vocab: 8192,
            max_seq: 256,
            dropout: 0.1,
            agirlik_baglama: true,
        }
    }

    #[must_use]
    pub fn kucuk() -> Self {
        Self {
            ad: "lubot-kucuk".to_string(),
            d_model: 32,
            n_layer: 4,
            n_head: 2,
            d_ff: 128,
            vocab: 8192,
            max_seq: 128,
            dropout: 0.1,
            agirlik_baglama: true,
        }
    }

    #[must_use]
    pub fn buyuk() -> Self {
        // Buyuk versiyon: 512,12,8,2048 -> 37M param (K6 tavan 97M altinda)
        Self {
            ad: "lubot-b1-derin-dar".to_string(),
            d_model: 512,
            n_layer: 12,
            n_head: 8,
            d_ff: 2048,
            vocab: 8192,
            max_seq: 1024,
            dropout: 0.1,
            agirlik_baglama: true,
        }
    }

    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        let embedding = self.vocab * self.d_model;
        let qkv = self.n_layer * 3 * self.d_model * self.d_model;
        let out = self.n_layer * self.d_model * self.d_model;
        let mlp1 = self.n_layer * self.d_model * self.d_ff;
        let mlp2 = self.n_layer * self.d_ff * self.d_model;
        let ln = self.n_layer * 2 * self.d_model + self.d_model;
        let total = embedding + qkv + out + mlp1 + mlp2 + ln;
        if self.agirlik_baglama {
            total
        } else {
            total + embedding
        }
    }

    #[must_use]
    pub fn basina_param(&self) -> HashMap<String, usize> {
        let mut m = HashMap::new();
        m.insert("embedding".to_string(), self.vocab * self.d_model);
        m.insert(
            "dikkat_qkv".to_string(),
            self.n_layer * 3 * self.d_model * self.d_model,
        );
        m.insert(
            "dikkat_out".to_string(),
            self.n_layer * self.d_model * self.d_model,
        );
        m.insert("mlp1".to_string(), self.n_layer * self.d_model * self.d_ff);
        m.insert("mlp2".to_string(), self.n_layer * self.d_ff * self.d_model);
        m.insert(
            "layernorm".to_string(),
            self.n_layer * 2 * self.d_model + self.d_model,
        );
        m
    }

    #[must_use]
    pub fn derin_dar_mi(&self) -> bool {
        self.n_layer * 64 > self.d_model
    }

    #[must_use]
    pub fn kafa_boyutu(&self) -> usize {
        self.d_model / self.n_head
    }

    #[must_use]
    pub fn tavan_altinda_mi(&self, tavan: usize) -> bool {
        self.param_sayisi() < tavan
    }
}

/// Agirlik baglama — embedding ve readout ayni agirlik.
#[derive(Debug, Clone)]
pub struct AgirlikBaglama {
    pub bagli_mi: bool,
    pub logit_scale: f64,
}

impl AgirlikBaglama {
    #[must_use]
    pub fn yeni(bagli_mi: bool, d_model: usize) -> Self {
        Self {
            bagli_mi,
            logit_scale: 1.0 / d_model as f64,
        }
    }

    #[must_use]
    pub fn aciklama(&self) -> String {
        if self.bagli_mi {
            format!(
                "agirlik baglama acik, logit_scale 1/d_model = {}",
                self.logit_scale
            )
        } else {
            "agirlik baglama kapali".to_string()
        }
    }
}

/// Model spec — JSON benzeri, training/model_spec.py'den.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub config: Config,
    pub mu_lr: HashMap<String, f64>,
    pub mu_init: HashMap<String, f64>,
    pub agirlik_baglama: AgirlikBaglama,
    pub notlar: String,
}

impl ModelSpec {
    #[must_use]
    pub fn lubot_a1() -> Self {
        let config = Config::lubot_a1();
        let mut mu_lr = HashMap::new();
        mu_lr.insert("embedding".to_string(), 0.001);
        mu_lr.insert("hidden".to_string(), 0.0005);
        mu_lr.insert("readout".to_string(), 1.56e-05);
        mu_lr.insert("layernorm".to_string(), 0.001);

        let mut mu_init = HashMap::new();
        mu_init.insert("embedding".to_string(), 1.0);
        mu_init.insert("hidden".to_string(), 0.044);
        mu_init.insert("readout".to_string(), 0.0019);
        mu_init.insert("layernorm".to_string(), 1.0);

        Self {
            agirlik_baglama: AgirlikBaglama::yeni(true, config.d_model),
            config,
            mu_lr,
            mu_init,
            notlar: "Tensor Programs V + Lingle 2024 yontem ilhami, olculmedi".to_string(),
        }
    }

    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        self.config.param_sayisi()
    }

    #[must_use]
    pub fn json_benzeri(&self) -> String {
        format!(
            "{{\"ad\":\"{}\",\"d_model\":{},\"n_layer\":{},\"n_head\":{},\"d_ff\":{},\"vocab\":{},\"param\":{},\"baglama\":{}}}",
            self.config.ad,
            self.config.d_model,
            self.config.n_layer,
            self.config.n_head,
            self.config.d_ff,
            self.config.vocab,
            self.param_sayisi(),
            self.config.agirlik_baglama
        )
    }
}

/// Derinlik genislik orani — U bolumu hiz/maliyet.
#[derive(Debug, Clone)]
pub struct Oran {
    pub derinlik: usize,
    pub genislik: usize,
    pub oran: f64,
}

impl Oran {
    #[must_use]
    pub fn yeni(derinlik: usize, genislik: usize) -> Self {
        Self {
            derinlik,
            genislik,
            oran: derinlik as f64 / genislik as f64,
        }
    }

    #[must_use]
    pub fn hiz_tahmini_ms(&self, seq_len: usize) -> f64 {
        self.derinlik as f64 * seq_len as f64 * self.genislik as f64 / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_lubot_a1() {
        let c = Config::lubot_a1();
        assert_eq!(c.d_model, 64);
        assert_eq!(c.n_layer, 8);
        assert!(c.agirlik_baglama);
    }

    #[test]
    fn config_kucuk() {
        let c = Config::kucuk();
        assert_eq!(c.d_model, 32);
        assert!(c.param_sayisi() < Config::lubot_a1().param_sayisi());
    }

    #[test]
    fn param_sayisi_yaklasik() {
        let c = Config::lubot_a1();
        let p = c.param_sayisi();
        // model_spec.json: 924288
        assert!(p > 800_000 && p < 1_100_000, "param {}", p);
    }

    #[test]
    fn config_buyuk() {
        let c = Config::buyuk();
        assert_eq!(c.d_model, 512);
        assert_eq!(c.n_layer, 12);
        assert!(c.tavan_altinda_mi(97_000_000));
    }

    #[test]
    fn basina_param() {
        let c = Config::lubot_a1();
        let m = c.basina_param();
        assert!(m.contains_key("embedding"));
        assert!(m.contains_key("dikkat_qkv"));
    }

    #[test]
    fn derin_dar() {
        let c = Config::lubot_a1();
        assert!(c.derin_dar_mi());
    }

    #[test]
    fn kafa_boyutu() {
        let c = Config::lubot_a1();
        assert_eq!(c.kafa_boyutu(), 32);
    }

    #[test]
    fn tavan_altinda() {
        let c = Config::lubot_a1();
        assert!(c.tavan_altinda_mi(97_000_000));
    }

    #[test]
    fn agirlik_baglama() {
        let ab = AgirlikBaglama::yeni(true, 512);
        assert!(ab.bagli_mi);
        assert!((ab.logit_scale - 1.0 / 512.0).abs() < 1e-9);
    }

    #[test]
    fn agirlik_baglama_aciklama() {
        let ab = AgirlikBaglama::yeni(true, 512);
        let a = ab.aciklama();
        assert!(a.contains("baglama acik"));
    }

    #[test]
    fn model_spec_lubot_a1() {
        let spec = ModelSpec::lubot_a1();
        assert_eq!(spec.config.ad, "lubot-a1-derin-dar");
        assert!(spec.param_sayisi() > 400_000);
    }

    #[test]
    fn model_spec_json() {
        let spec = ModelSpec::lubot_a1();
        let j = spec.json_benzeri();
        assert!(j.contains("lubot-a1-derin-dar"));
    }

    #[test]
    fn oran_hiz() {
        let o = Oran::yeni(12, 512);
        let hiz = o.hiz_tahmini_ms(1024);
        assert!(hiz > 0.0);
    }

    #[test]
    fn config_klon() {
        let c1 = Config::lubot_a1();
        let c2 = c1.clone();
        assert_eq!(c1, c2);
    }

    #[test]
    fn deterministik_param() {
        let c1 = Config::lubot_a1();
        let c2 = Config::lubot_a1();
        assert_eq!(c1.param_sayisi(), c2.param_sayisi());
    }

    #[test]
    fn baglama_param_farki() {
        let mut c1 = Config::lubot_a1();
        c1.agirlik_baglama = true;
        let mut c2 = Config::lubot_a1();
        c2.agirlik_baglama = false;
        assert!(c1.param_sayisi() < c2.param_sayisi());
    }
}
