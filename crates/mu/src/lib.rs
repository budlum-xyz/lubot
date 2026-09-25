//! Lubot mu — μP olcekleme, Tensor Programs V + Lingle 2024 yontem ilhami.
//!
//! K1: sifirdan yazildi, yalnizca metodoloji notu (olculmedi).
//! II: derin-dar mimari icin olcek tablosu.
//! HH: guclu weight_decay ayri.
//! F: sifirdan egitim iskeleti.

use std::collections::HashMap;

/// Model olcegi — d_model, n_layer, n_head, d_ff, vocab.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Olcek {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub vocab: usize,
}

impl Olcek {
    #[must_use]
    pub fn yeni(d_model: usize, n_layer: usize, n_head: usize, d_ff: usize, vocab: usize) -> Self {
        Self {
            d_model,
            n_layer,
            n_head,
            d_ff,
            vocab,
        }
    }

    /// Parametre sayisi — derin-dar icin yaklasik.
    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        let embedding = self.vocab * self.d_model;
        let dikkat_qkv = self.n_layer * 3 * self.d_model * self.d_model;
        let dikkat_out = self.n_layer * self.d_model * self.d_model;
        let mlp_1 = self.n_layer * self.d_model * self.d_ff;
        let mlp_2 = self.n_layer * self.d_ff * self.d_model;
        let layernorm = self.n_layer * 2 * self.d_model + self.d_model;
        embedding + dikkat_qkv + dikkat_out + mlp_1 + mlp_2 + layernorm
    }

    /// Derin-dar mi? (n_layer > d_model / 64)
    #[must_use]
    pub fn derin_dar_mi(&self) -> bool {
        self.n_layer * 64 > self.d_model
    }
}

/// μP tablosu — init std ve LR per grup.
#[derive(Debug, Clone)]
pub struct MuP {
    pub olcek: Olcek,
    pub taban_d_model: usize,
    pub taban_lr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Grup {
    Embedding,
    Hidden,
    Readout,
    Layernorm,
    Bias,
}

impl MuP {
    #[must_use]
    pub fn yeni(olcek: Olcek, taban_d_model: usize, taban_lr: f64) -> Self {
        Self {
            olcek,
            taban_d_model,
            taban_lr,
        }
    }

    /// Init std — μP kuralina gore.
    /// Embedding: 1.0, Hidden: 1/sqrt(d_model), Readout: 1/d_model
    #[must_use]
    pub fn init_std(&self, grup: Grup) -> f64 {
        match grup {
            Grup::Embedding => 1.0,
            Grup::Hidden => 1.0 / (self.olcek.d_model as f64).sqrt(),
            Grup::Readout => 1.0 / (self.olcek.d_model as f64),
            Grup::Layernorm => 1.0,
            Grup::Bias => 0.0,
        }
    }

    /// LR olcegi — μP.
    /// Embedding: sabit, Hidden: 1/d_model, Readout: 1/d_model
    /// Lingle 2024: readout LR daha kucuk olmali, yoksa loss patlar.
    #[must_use]
    pub fn lr_olcek(&self, grup: Grup) -> f64 {
        let oran = self.taban_d_model as f64 / self.olcek.d_model as f64;
        match grup {
            Grup::Embedding => self.taban_lr,
            Grup::Hidden => self.taban_lr * oran,
            Grup::Readout => self.taban_lr * oran * oran, // 1/d_model^2 gibi
            Grup::Layernorm => self.taban_lr,
            Grup::Bias => self.taban_lr,
        }
    }

    /// LR tablosu — tum gruplar icin.
    #[must_use]
    pub fn lr_tablosu(&self) -> HashMap<Grup, f64> {
        let mut tablo = HashMap::new();
        tablo.insert(Grup::Embedding, self.lr_olcek(Grup::Embedding));
        tablo.insert(Grup::Hidden, self.lr_olcek(Grup::Hidden));
        tablo.insert(Grup::Readout, self.lr_olcek(Grup::Readout));
        tablo.insert(Grup::Layernorm, self.lr_olcek(Grup::Layernorm));
        tablo.insert(Grup::Bias, self.lr_olcek(Grup::Bias));
        tablo
    }

    /// Init tablosu.
    #[must_use]
    pub fn init_tablosu(&self) -> HashMap<Grup, f64> {
        let mut tablo = HashMap::new();
        tablo.insert(Grup::Embedding, self.init_std(Grup::Embedding));
        tablo.insert(Grup::Hidden, self.init_std(Grup::Hidden));
        tablo.insert(Grup::Readout, self.init_std(Grup::Readout));
        tablo.insert(Grup::Layernorm, self.init_std(Grup::Layernorm));
        tablo.insert(Grup::Bias, self.init_std(Grup::Bias));
        tablo
    }

    /// Logit scale — 1/d_model (isaretli karar, training/model_spec.py).
    #[must_use]
    pub fn logit_scale(&self) -> f64 {
        1.0 / self.olcek.d_model as f64
    }

    /// Weight decay ayri — HH: embedding icin 0.0, hidden icin 0.1, readout icin 0.01
    #[must_use]
    pub fn weight_decay(&self, grup: Grup) -> f64 {
        match grup {
            Grup::Embedding => 0.0,
            Grup::Hidden => 0.1,
            Grup::Readout => 0.01,
            Grup::Layernorm => 0.0,
            Grup::Bias => 0.0,
        }
    }

    /// Olcek kontrol — K6: tavan altinda mi?
    #[must_use]
    pub fn tavan_altinda_mi(&self, tavan: usize) -> bool {
        self.olcek.param_sayisi() < tavan
    }
}

/// MuP spec dosyasi — training/model_spec.py'den okunan JSON benzeri.
#[derive(Debug, Clone)]
pub struct MuSpec {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub lr_tablosu: HashMap<String, f64>,
    pub init_tablosu: HashMap<String, f64>,
    pub logit_scale: f64,
}

impl MuSpec {
    #[must_use]
    pub fn ornek() -> Self {
        // lubot-a1-derin-dar: 64,8,2,256,8192 -> 924K
        let olcek = Olcek::yeni(64, 8, 2, 256, 8192);
        let mup = MuP::yeni(olcek.clone(), 256, 0.001);
        let mut lr_tablosu = HashMap::new();
        for (grup, lr) in mup.lr_tablosu() {
            let ad = match grup {
                Grup::Embedding => "embedding",
                Grup::Hidden => "hidden",
                Grup::Readout => "readout",
                Grup::Layernorm => "layernorm",
                Grup::Bias => "bias",
            };
            lr_tablosu.insert(ad.to_string(), lr);
        }
        let mut init_tablosu = HashMap::new();
        for (grup, std) in mup.init_tablosu() {
            let ad = match grup {
                Grup::Embedding => "embedding",
                Grup::Hidden => "hidden",
                Grup::Readout => "readout",
                Grup::Layernorm => "layernorm",
                Grup::Bias => "bias",
            };
            init_tablosu.insert(ad.to_string(), std);
        }
        Self {
            d_model: olcek.d_model,
            n_layer: olcek.n_layer,
            n_head: olcek.n_head,
            d_ff: olcek.d_ff,
            vocab: olcek.vocab,
            lr_tablosu,
            init_tablosu,
            logit_scale: mup.logit_scale(),
        }
    }

    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        Olcek::yeni(
            self.d_model,
            self.n_layer,
            self.n_head,
            self.d_ff,
            self.vocab,
        )
        .param_sayisi()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ornek_olcek() -> Olcek {
        Olcek::yeni(64, 8, 2, 256, 8192)
    }

    #[test]
    fn olcek_param_sayisi() {
        let o = ornek_olcek();
        let p = o.param_sayisi();
        // 924K civari olmali (training/model_spec.py'den) — tying ile 924K, tying'siz 1.4M, ama Olcek param_sayisi embedding+... hesaplar
        // Olcek::param_sayisi embedding + qkv + out + mlp1+mlp2+ln = 924K civari
        assert!(p > 800_000 && p < 2_000_000, "param {}", p);
    }

    #[test]
    fn derin_dar_kontrol() {
        let o = Olcek::yeni(512, 12, 8, 2048, 8192);
        assert!(o.derin_dar_mi());
        let sig = Olcek::yeni(1024, 2, 8, 4096, 8192);
        assert!(!sig.derin_dar_mi());
    }

    #[test]
    fn mup_init_std() {
        let o = ornek_olcek();
        let m = MuP::yeni(o, 256, 0.001);
        assert_eq!(m.init_std(Grup::Embedding), 1.0);
        assert!(m.init_std(Grup::Hidden) < 1.0);
        assert!(m.init_std(Grup::Readout) < m.init_std(Grup::Hidden));
    }

    #[test]
    fn mup_lr_olcek() {
        // Buyuk model icin test: d_model >= taban => emb >= hid >= read
        let o = Olcek::yeni(512, 12, 8, 2048, 8192);
        let m = MuP::yeni(o, 256, 0.001);
        let emb = m.lr_olcek(Grup::Embedding);
        let hid = m.lr_olcek(Grup::Hidden);
        let read = m.lr_olcek(Grup::Readout);
        assert!(emb >= hid);
        assert!(hid >= read);
    }

    #[test]
    fn mup_lr_tablosu() {
        let o = ornek_olcek();
        let m = MuP::yeni(o, 256, 0.001);
        let tablo = m.lr_tablosu();
        assert_eq!(tablo.len(), 5);
    }

    #[test]
    fn mup_init_tablosu() {
        let o = ornek_olcek();
        let m = MuP::yeni(o, 256, 0.001);
        let tablo = m.init_tablosu();
        assert_eq!(tablo.len(), 5);
    }

    #[test]
    fn logit_scale() {
        let o = ornek_olcek();
        let m = MuP::yeni(o.clone(), 256, 0.001);
        let ls = m.logit_scale();
        assert!((ls - 1.0 / o.d_model as f64).abs() < 1e-9);
    }

    #[test]
    fn weight_decay() {
        let o = ornek_olcek();
        let m = MuP::yeni(o, 256, 0.001);
        assert_eq!(m.weight_decay(Grup::Embedding), 0.0);
        assert_eq!(m.weight_decay(Grup::Hidden), 0.1);
    }

    #[test]
    fn tavan_altinda() {
        let o = ornek_olcek();
        let m = MuP::yeni(o, 256, 0.001);
        assert!(m.tavan_altinda_mi(97_000_000));
        assert!(!m.tavan_altinda_mi(100_000));
    }

    #[test]
    fn mu_spec_ornek() {
        let spec = MuSpec::ornek();
        assert_eq!(spec.d_model, 64);
        assert_eq!(spec.n_layer, 8);
        assert!(spec.param_sayisi() > 800_000);
    }

    #[test]
    fn mu_spec_lr_tablosu() {
        let spec = MuSpec::ornek();
        assert!(spec.lr_tablosu.contains_key("embedding"));
        assert!(spec.lr_tablosu.contains_key("hidden"));
        assert!(spec.lr_tablosu.contains_key("readout"));
    }

    #[test]
    fn mu_spec_init_tablosu() {
        let spec = MuSpec::ornek();
        assert!(spec.init_tablosu.contains_key("embedding"));
    }

    #[test]
    fn olcek_klon() {
        let o1 = ornek_olcek();
        let o2 = o1.clone();
        assert_eq!(o1, o2);
    }

    #[test]
    fn grup_hash() {
        let mut map = HashMap::new();
        map.insert(Grup::Embedding, 1.0);
        map.insert(Grup::Hidden, 2.0);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn deterministik_param() {
        let o1 = ornek_olcek();
        let o2 = ornek_olcek();
        assert_eq!(o1.param_sayisi(), o2.param_sayisi());
    }
}
