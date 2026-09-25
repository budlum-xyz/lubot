//! Lubot egitim — egitim cekirdegi Rust, F/II/HH/J/MM/BB/R.
//!
//! K1: sifirdan yazildi, torch/transformers yok.
//! F: sifirdan on-egitim iskeleti, muP tablosu model_spec.py'den.
//! II: agirlik baglama logit_scale 1/d_model.
//! HH: guclu weight_decay 0.1.
//! J: derin-dar + agirlik baglama.
//! MM: muhendislik iskeleti vs veri ayrimi — stdlib + minimal.
//! BB: karar basligi once.
//! R: ratchet.

use std::collections::HashMap;

/// Model agirliklari — basit HashMap tabanli (gercek tensor yok, iskelet).
#[derive(Debug, Clone)]
pub struct Agirliklar {
    pub params: HashMap<String, Vec<f32>>,
    pub d_model: usize,
    pub vocab: usize,
}

impl Agirliklar {
    #[must_use]
    pub fn yeni(d_model: usize, vocab: usize) -> Self {
        let mut params = HashMap::new();
        // Embedding: vocab x d_model
        params.insert("embedding".to_string(), vec![0.0; vocab * d_model]);
        // Her katman icin qkv, out, mlp1, mlp2, ln1, ln2
        for layer in 0..12 {
            params.insert(
                format!("layer_{layer}_qkv"),
                vec![0.0; 3 * d_model * d_model],
            );
            params.insert(format!("layer_{layer}_out"), vec![0.0; d_model * d_model]);
            params.insert(format!("layer_{layer}_mlp1"), vec![0.0; d_model * 2048]);
            params.insert(format!("layer_{layer}_mlp2"), vec![0.0; 2048 * d_model]);
            params.insert(format!("layer_{layer}_ln1"), vec![0.0; d_model]);
            params.insert(format!("layer_{layer}_ln2"), vec![0.0; d_model]);
        }
        params.insert("final_ln".to_string(), vec![0.0; d_model]);
        // Readout: d_model x vocab, baglama aciksa embedding ile ayni ref (iskelet)
        params.insert("readout".to_string(), vec![0.0; d_model * vocab]);

        Self {
            params,
            d_model,
            vocab,
        }
    }

    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        self.params.values().map(|v| v.len()).sum()
    }

    /// Hash tabanli init — hizli, deterministik (training/run_pretrain.py'deki gibi).
    pub fn init_hash(&mut self, seed: u64) {
        for (name, vals) in self.params.iter_mut() {
            // Basit hash: seed + name hash + index
            let name_hash = {
                let mut h = 0u64;
                for b in name.bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b as u64);
                }
                h
            };
            for (i, v) in vals.iter_mut().enumerate() {
                let h = seed
                    .wrapping_add(name_hash)
                    .wrapping_add(i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                // -1..1 arasi
                let f = ((h % 10000) as f32 / 5000.0) - 1.0;
                // muP init std: embedding 1.0, hidden 1/sqrt(d), readout 1/d
                let std = if name.contains("embedding") {
                    1.0
                } else if name.contains("readout") {
                    1.0 / self.d_model as f32
                } else if name.contains("ln") {
                    1.0
                } else {
                    1.0 / (self.d_model as f32).sqrt()
                };
                *v = f * std * 0.02;
            }
        }
    }

    /// Sparse embedding update — sadece batch'teki tokenlar guncellenir (U hiz).
    pub fn sparse_embedding_update(&mut self, batch_tokens: &[u32], grad: f32, lr: f32) {
        if let Some(emb) = self.params.get_mut("embedding") {
            for &tok in batch_tokens {
                let tok = tok as usize % self.vocab;
                let start = tok * self.d_model;
                let end = start + self.d_model;
                if end <= emb.len() {
                    for val in &mut emb[start..end] {
                        *val -= grad * lr;
                    }
                }
            }
        }
    }
}

/// AdamW optimizer — basit.
#[derive(Debug, Clone)]
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub m: HashMap<String, Vec<f32>>,
    pub v: HashMap<String, Vec<f32>>,
    pub step: usize,
}

impl AdamW {
    #[must_use]
    pub fn yeni(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
        }
    }

    /// Adim — tum parametreler icin.
    pub fn adim(&mut self, agirliklar: &mut Agirliklar, grads: &HashMap<String, Vec<f32>>) {
        self.step += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        for (name, param) in agirliklar.params.iter_mut() {
            let grad = match grads.get(name) {
                Some(g) => g,
                None => continue,
            };
            let m = self
                .m
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; param.len()]);
            let v = self
                .v
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; param.len()]);

            for i in 0..param.len() {
                let g = grad[i];
                m[i] = (self.beta1 as f32) * m[i] + (1.0 - self.beta1 as f32) * g;
                v[i] = (self.beta2 as f32) * v[i] + (1.0 - self.beta2 as f32) * g * g;

                let m_hat = m[i] / bias_correction1 as f32;
                let v_hat = v[i] / bias_correction2 as f32;

                // weight decay ayri — HH
                let wd = if name.contains("embedding") {
                    0.0
                } else if name.contains("readout") {
                    0.01
                } else if name.contains("ln") || name.contains("bias") {
                    0.0
                } else {
                    self.weight_decay as f32
                };

                param[i] -=
                    self.lr as f32 * (m_hat / (v_hat.sqrt() + self.eps as f32) + wd * param[i]);
            }
        }
    }
}

/// Ileri gecis iskeleti — gercek matmul yok, ama sekil ve olcum var.
#[derive(Debug, Clone)]
pub struct Ileri {
    pub d_model: usize,
    pub n_layer: usize,
    pub seq_len: usize,
}

impl Ileri {
    #[must_use]
    pub fn yeni(d_model: usize, n_layer: usize, seq_len: usize) -> Self {
        Self {
            d_model,
            n_layer,
            seq_len,
        }
    }

    /// Ileri gecis — loss dondur (sahte, ama deterministik).
    #[must_use]
    pub fn ileri(&self, tokens: &[u32], agirliklar: &Agirliklar) -> f32 {
        // Basit: tokenlarin hash'i + agirliklarin ortalamasi
        let mut loss = 0.0f32;
        for &tok in tokens {
            loss += (tok as f32 % 10.0) * 0.001;
        }
        // Agirliklarin ortalamasi
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for vals in agirliklar.params.values() {
            for &v in vals.iter().take(100) {
                sum += v.abs();
                count += 1;
            }
        }
        if count > 0 {
            loss += sum / count as f32 * 0.01;
        }
        loss += self.n_layer as f32 * 0.0001;
        loss
    }

    /// Hiz olcumu: ornek basina ms.
    #[must_use]
    pub fn hiz_olcum(&self, agirliklar: &Agirliklar, batch: usize, tekrar: usize) -> f64 {
        let tokens: Vec<u32> = (0..self.seq_len as u32).collect();
        let start = std::time::Instant::now();
        for _ in 0..tekrar {
            let _ = self.ileri(&tokens, agirliklar);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if tekrar * batch == 0 {
            0.0
        } else {
            elapsed / (tekrar * batch) as f64
        }
    }
}

/// Egitim dongusu — epoch, batch, loss.
#[derive(Debug, Clone)]
pub struct EgitimDongusu {
    pub config: EgitimConfig,
    pub agirliklar: Agirliklar,
    pub optimizer: AdamW,
    pub ileri: Ileri,
}

#[derive(Debug, Clone)]
pub struct EgitimConfig {
    pub d_model: usize,
    pub vocab: usize,
    pub n_layer: usize,
    pub seq_len: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub max_steps: usize,
}

impl EgitimConfig {
    #[must_use]
    pub fn lubot_a1() -> Self {
        // model_spec.json ile uyumlu: 64,8192,8
        Self {
            d_model: 64,
            vocab: 8192,
            n_layer: 8,
            seq_len: 128,
            batch_size: 8,
            epochs: 1,
            lr: 0.001,
            weight_decay: 0.1,
            max_steps: 100,
        }
    }

    #[must_use]
    pub fn kucuk() -> Self {
        Self {
            d_model: 32,
            vocab: 1024,
            n_layer: 2,
            seq_len: 32,
            batch_size: 2,
            epochs: 1,
            lr: 0.001,
            weight_decay: 0.1,
            max_steps: 5,
        }
    }
}

impl EgitimDongusu {
    #[must_use]
    pub fn yeni(config: EgitimConfig) -> Self {
        let mut agirliklar = Agirliklar::yeni(config.d_model, config.vocab);
        agirliklar.init_hash(42);
        let optimizer = AdamW::yeni(config.lr, config.weight_decay);
        let ileri = Ileri::yeni(config.d_model, config.n_layer, config.seq_len);
        Self {
            config,
            agirliklar,
            optimizer,
            ileri,
        }
    }

    /// Bir epoch calistir — sahte data ile.
    pub fn epoch_calistir(&mut self) -> Vec<f32> {
        let mut losses = Vec::new();
        let steps = self.config.max_steps.min(100); // hizli icin 100
        for _ in 0..steps {
            let tokens: Vec<u32> = (0..self.config.seq_len as u32)
                .map(|i| (i * 7 + 3) % self.config.vocab as u32)
                .collect();
            let loss = self.ileri.ileri(&tokens, &self.agirliklar);
            losses.push(loss);

            // Sahte grad
            let mut grads = HashMap::new();
            for (name, vals) in &self.agirliklar.params {
                let g = vals.iter().map(|_| 0.001).collect();
                grads.insert(name.clone(), g);
            }
            self.optimizer.adim(&mut self.agirliklar, &grads);

            // Sparse embedding update (U hiz)
            self.agirliklar
                .sparse_embedding_update(&tokens, 0.001, self.config.lr as f32);
        }
        losses
    }

    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        self.agirliklar.param_sayisi()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agirliklar_yeni() {
        let a = Agirliklar::yeni(64, 1024);
        assert!(a.param_sayisi() > 50_000);
    }

    #[test]
    fn agirliklar_init_hash() {
        let mut a = Agirliklar::yeni(32, 256);
        a.init_hash(42);
        let sum: f32 = a.params.values().flat_map(|v| v.iter()).take(100).sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn agirliklar_deterministik() {
        let mut a1 = Agirliklar::yeni(32, 256);
        a1.init_hash(42);
        let mut a2 = Agirliklar::yeni(32, 256);
        a2.init_hash(42);
        assert_eq!(
            a1.params["embedding"][0].to_bits(),
            a2.params["embedding"][0].to_bits()
        );
    }

    #[test]
    fn sparse_update() {
        let mut a = Agirliklar::yeni(64, 1024);
        a.init_hash(42);
        let before = a.params["embedding"][0];
        a.sparse_embedding_update(&[0], 0.1, 0.001);
        let after = a.params["embedding"][0];
        assert_ne!(before.to_bits(), after.to_bits());
    }

    #[test]
    fn adamw_yeni() {
        let opt = AdamW::yeni(0.001, 0.1);
        assert_eq!(opt.lr, 0.001);
        assert_eq!(opt.step, 0);
    }

    #[test]
    fn adamw_adim() {
        let mut a = Agirliklar::yeni(32, 256);
        a.init_hash(42);
        let mut opt = AdamW::yeni(0.001, 0.1);
        let mut grads = HashMap::new();
        grads.insert("embedding".to_string(), vec![0.001; 32 * 256]);
        opt.adim(&mut a, &grads);
        assert_eq!(opt.step, 1);
    }

    #[test]
    fn ileri_yeni() {
        let il = Ileri::yeni(64, 8, 32);
        assert_eq!(il.d_model, 64);
    }

    #[test]
    fn ileri_loss() {
        let il = Ileri::yeni(32, 2, 16);
        let a = Agirliklar::yeni(32, 256);
        let tokens = vec![1, 2, 3, 4, 5];
        let loss = il.ileri(&tokens, &a);
        assert!(loss >= 0.0);
    }

    #[test]
    fn ileri_deterministik() {
        let il = Ileri::yeni(32, 2, 16);
        let a = Agirliklar::yeni(32, 256);
        let tokens = vec![1, 2, 3];
        let l1 = il.ileri(&tokens, &a);
        let l2 = il.ileri(&tokens, &a);
        assert_eq!(l1.to_bits(), l2.to_bits());
    }

    #[test]
    fn hiz_olcum() {
        let il = Ileri::yeni(32, 2, 16);
        let a = Agirliklar::yeni(32, 256);
        let hiz = il.hiz_olcum(&a, 2, 2);
        assert!(hiz >= 0.0);
    }

    #[test]
    fn egitim_config() {
        let c = EgitimConfig::kucuk();
        assert_eq!(c.d_model, 32);
        assert_eq!(c.batch_size, 2);
    }

    #[test]
    fn egitim_dongusu_yeni() {
        let c = EgitimConfig::kucuk();
        let d = EgitimDongusu::yeni(c);
        assert!(d.param_sayisi() > 10_000);
    }

    #[test]
    fn egitim_epoch() {
        let mut c = EgitimConfig::kucuk();
        c.max_steps = 2;
        let mut d = EgitimDongusu::yeni(c);
        let losses = d.epoch_calistir();
        assert_eq!(losses.len(), 2);
        assert!(losses.iter().all(|&l| l >= 0.0));
    }

    #[test]
    fn egitim_param_sayisi() {
        let c = EgitimConfig::kucuk();
        let d = EgitimDongusu::yeni(c);
        let p = d.param_sayisi();
        assert!(p > 10_000);
    }

    #[test]
    fn tavan_altinda() {
        let c = EgitimConfig::lubot_a1();
        let d = EgitimDongusu::yeni(c);
        assert!(d.param_sayisi() < 97_000_000);
    }

    #[test]
    fn agirliklar_klon() {
        let a1 = Agirliklar::yeni(256, 1024);
        let a2 = a1.clone();
        assert_eq!(a1.param_sayisi(), a2.param_sayisi());
    }
}
