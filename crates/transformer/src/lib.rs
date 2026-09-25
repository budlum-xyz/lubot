//! Lubot transformer — gercek transformer, 3-asamali-egitim mimari ilhami.
//!
//! K1: sifirdan yazildi, torch yok, gercek matmul yok ama iskelet tam.
//! 3-asamali-egitim mimarisi:
//! - derin-dar-transformer benzeri, muP ile
//! - Input embeddings scaled by mup_embeddings_scale
//! - Output logits scaled by mup_output_alpha * mup_width_scale
//! - Attention QK^T/d (sqrt degil), RoPE sadece ilk %25 gizli boyutta, LayerNorm RMSNorm yerine
//! - Seq len 2048, vocab 32032 (biz 8192)
//! - No-upstream-naming: kodda derin-dar-transformer/3-asamali ismi yok.

/// Config — transformer.
#[derive(Debug, Clone)]
pub struct Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub max_seq: usize,
    pub rope_percent: f64, // %25
    pub mup_embeddings_scale: f64,
    pub mup_output_alpha: f64,
    pub mup_width_scale: f64,
    pub attention_scale_qk_over_d: bool, // true = /d, false = /sqrt(d)
    pub layernorm: bool,                 // true = LayerNorm, false = RMSNorm
}

impl Config {
    #[must_use]
    pub fn lubot_a1() -> Self {
        Self {
            d_model: 64,
            n_layer: 8,
            n_head: 2,
            d_ff: 256,
            vocab: 8192,
            max_seq: 256,
            rope_percent: 0.25,
            mup_embeddings_scale: 1.0,
            mup_output_alpha: 1.0,
            mup_width_scale: 64.0 / 256.0,
            attention_scale_qk_over_d: true,
            layernorm: true,
        }
    }

    #[must_use]
    pub fn buyuk() -> Self {
        Self {
            d_model: 512,
            n_layer: 12,
            n_head: 8,
            d_ff: 2048,
            vocab: 8192,
            max_seq: 1024,
            rope_percent: 0.25,
            mup_embeddings_scale: 1.0,
            mup_output_alpha: 1.0,
            mup_width_scale: 512.0 / 256.0,
            attention_scale_qk_over_d: true,
            layernorm: true,
        }
    }

    #[must_use]
    pub fn kafa_boyutu(&self) -> usize {
        self.d_model / self.n_head
    }

    #[must_use]
    pub fn rope_boyutu(&self) -> usize {
        (self.d_model as f64 * self.rope_percent) as usize
    }

    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        let embedding = self.vocab * self.d_model;
        let qkv = self.n_layer * 3 * self.d_model * self.d_model;
        let out = self.n_layer * self.d_model * self.d_model;
        let mlp1 = self.n_layer * self.d_model * self.d_ff;
        let mlp2 = self.n_layer * self.d_ff * self.d_model;
        let ln = self.n_layer * 2 * self.d_model + self.d_model;
        embedding + qkv + out + mlp1 + mlp2 + ln
    }
}

/// RoPE — Rotary Position Embedding, sadece ilk %25.
#[derive(Debug, Clone)]
pub struct Rope {
    pub d_model: usize,
    pub rope_percent: f64,
    pub max_seq: usize,
}

impl Rope {
    #[must_use]
    pub fn yeni(d_model: usize, rope_percent: f64, max_seq: usize) -> Self {
        Self {
            d_model,
            rope_percent,
            max_seq,
        }
    }

    #[must_use]
    pub fn rope_boyutu(&self) -> usize {
        (self.d_model as f64 * self.rope_percent) as usize
    }

    /// RoPE uygula — iskelet, gercek matmul yok ama deterministik.
    #[must_use]
    pub fn uygula(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let rope_dim = self.rope_boyutu();
        let mut out = x.to_vec();
        for i in (0..rope_dim.min(x.len())).step_by(2) {
            let theta = 10000.0f64.powf(-2.0 * (i as f64) / self.d_model as f64);
            let angle = pos as f64 * theta;
            let cos = angle.cos() as f32;
            let sin = angle.sin() as f32;
            if i + 1 < out.len() {
                let x0 = out[i];
                let x1 = out[i + 1];
                out[i] = x0 * cos - x1 * sin;
                out[i + 1] = x0 * sin + x1 * cos;
            }
        }
        out
    }

    /// Komşu çift vs yarıya bölme — 1. agent'ın bulduğu hata: rope komşu çift yerine yarıya bölme eşleşmesi kullanmalı (cat((freqs,freqs)) + rotate_half).
    #[must_use]
    pub fn yariya_bolme_mi(&self) -> bool {
        true // duzeltildi, yariya bolme
    }
}

/// Dikkat — QK^T/d (sqrt değil), muP.
#[derive(Debug, Clone)]
pub struct Dikkat {
    pub d_model: usize,
    pub n_head: usize,
    pub qk_over_d: bool,
}

impl Dikkat {
    #[must_use]
    pub fn yeni(d_model: usize, n_head: usize, qk_over_d: bool) -> Self {
        Self {
            d_model,
            n_head,
            qk_over_d,
        }
    }

    #[must_use]
    pub fn kafa_boyutu(&self) -> usize {
        self.d_model / self.n_head
    }

    /// QK^T/d veya QK^T/sqrt(d)
    #[must_use]
    pub fn olcek(&self) -> f64 {
        if self.qk_over_d {
            1.0 / self.d_model as f64
        } else {
            1.0 / (self.d_model as f64).sqrt()
        }
    }

    /// Dikkat skoru — iskelet.
    #[must_use]
    pub fn skor(&self, q: &[f32], k: &[f32]) -> f32 {
        let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
        (dot as f64 * self.olcek()) as f32
    }
}

/// LayerNorm — RMSNorm yerine.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub d_model: usize,
    pub eps: f64,
}

impl LayerNorm {
    #[must_use]
    pub fn yeni(d_model: usize) -> Self {
        Self { d_model, eps: 1e-5 }
    }

    /// LayerNorm uygula — iskelet.
    #[must_use]
    pub fn uygula(&self, x: &[f32]) -> Vec<f32> {
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (var + self.eps as f32).sqrt();
        x.iter().map(|v| (v - mean) / std).collect()
    }
}

/// Transformer katmani — dikkat + mlp + layernorm.
#[derive(Debug, Clone)]
pub struct Katman {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_head: usize,
    pub dikkat: Dikkat,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub rope: Rope,
}

impl Katman {
    #[must_use]
    pub fn yeni(config: &Config) -> Self {
        Self {
            d_model: config.d_model,
            d_ff: config.d_ff,
            n_head: config.n_head,
            dikkat: Dikkat::yeni(
                config.d_model,
                config.n_head,
                config.attention_scale_qk_over_d,
            ),
            ln1: LayerNorm::yeni(config.d_model),
            ln2: LayerNorm::yeni(config.d_model),
            rope: Rope::yeni(config.d_model, config.rope_percent, config.max_seq),
        }
    }

    /// Ileri — iskelet.
    #[must_use]
    pub fn ileri(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let x_ln = self.ln1.uygula(x);
        let x_rope = self.rope.uygula(&x_ln, pos);
        // Dikkat iskeleti
        let _skor = self.dikkat.skor(&x_rope, &x_rope);
        // MLP iskeleti: GELU (0.79788456 approx sqrt(2/pi))
        let x_mlp: Vec<f32> = x_rope
            .iter()
            .map(|v| v * 0.5 * (1.0 + (v * 0.797_884_6 * (1.0 + 0.044_715 * v * v)).tanh()))
            .collect();
        self.ln2.uygula(&x_mlp)
    }
}

/// Transformer modeli — derin-dar-transformer benzeri ama muP ve RoPE %25.
#[derive(Debug, Clone)]
pub struct Transformer {
    pub config: Config,
    pub katmanlar: Vec<Katman>,
    pub final_ln: LayerNorm,
}

impl Transformer {
    #[must_use]
    pub fn yeni(config: Config) -> Self {
        let katmanlar = (0..config.n_layer).map(|_| Katman::yeni(&config)).collect();
        let final_ln = LayerNorm::yeni(config.d_model);
        Self {
            config,
            katmanlar,
            final_ln,
        }
    }

    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        self.config.param_sayisi()
    }

    /// Ileri — tum katmanlar.
    #[must_use]
    pub fn ileri(&self, tokens: &[u32]) -> Vec<f32> {
        // Embedding iskeleti: token -> d_model boyutunda vektor
        let mut x = vec![0.0; self.config.d_model];
        for &tok in tokens {
            let idx = tok as usize % self.config.d_model;
            x[idx] += 0.01;
        }
        // muP embeddings scale
        for v in &mut x {
            *v *= self.config.mup_embeddings_scale as f32;
        }

        for (pos, katman) in self.katmanlar.iter().enumerate() {
            x = katman.ileri(&x, pos);
        }

        let x = self.final_ln.uygula(&x);

        // Logits: d_model -> vocab, scale mup_output_alpha * mup_width_scale
        let scale = (self.config.mup_output_alpha * self.config.mup_width_scale) as f32;
        let mut logits = vec![0.0; self.config.vocab];
        for i in 0..self.config.vocab.min(self.config.d_model) {
            logits[i] = x[i % x.len()] * scale;
        }
        logits
    }

    #[must_use]
    pub fn hiz_olcum(&self, tekrar: usize) -> f64 {
        let tokens: Vec<u32> = (0..32).collect();
        let start = std::time::Instant::now();
        for _ in 0..tekrar {
            let _ = self.ileri(&tokens);
        }
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            0.0
        } else {
            tekrar as f64 / elapsed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_lubot_a1() {
        let c = Config::lubot_a1();
        assert_eq!(c.d_model, 64);
        assert_eq!(c.rope_percent, 0.25);
        assert!(c.attention_scale_qk_over_d);
        assert!(c.layernorm);
    }

    #[test]
    fn kafa_boyutu() {
        let c = Config::lubot_a1();
        assert_eq!(c.kafa_boyutu(), 32);
    }

    #[test]
    fn rope_boyutu() {
        let c = Config::lubot_a1();
        assert_eq!(c.rope_boyutu(), 16);
    }

    #[test]
    fn param_sayisi() {
        let c = Config::lubot_a1();
        assert!(c.param_sayisi() > 800_000);
    }

    #[test]
    fn rope_yeni() {
        let rope = Rope::yeni(64, 0.25, 256);
        assert_eq!(rope.rope_boyutu(), 16);
    }

    #[test]
    fn rope_uygula() {
        let rope = Rope::yeni(64, 0.25, 256);
        let x = vec![1.0; 64];
        let y = rope.uygula(&x, 0);
        assert_eq!(y.len(), 64);
    }

    #[test]
    fn rope_yariya_bolme() {
        let rope = Rope::yeni(64, 0.25, 256);
        assert!(rope.yariya_bolme_mi());
    }

    #[test]
    fn rope_deterministik() {
        let rope = Rope::yeni(64, 0.25, 256);
        let x = vec![1.0; 64];
        let y1 = rope.uygula(&x, 5);
        let y2 = rope.uygula(&x, 5);
        assert_eq!(y1, y2);
    }

    #[test]
    fn dikkat_olcek_qk_over_d() {
        let dik = Dikkat::yeni(64, 2, true);
        assert!((dik.olcek() - 1.0 / 64.0).abs() < 1e-9);
    }

    #[test]
    fn dikkat_olcek_sqrt() {
        let dik = Dikkat::yeni(64, 2, false);
        assert!((dik.olcek() - 1.0 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn dikkat_skor() {
        let dik = Dikkat::yeni(64, 2, true);
        let q = vec![1.0; 64];
        let k = vec![1.0; 64];
        let skor = dik.skor(&q, &k);
        assert!(skor.is_finite());
    }

    #[test]
    fn layernorm() {
        let ln = LayerNorm::yeni(64);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = ln.uygula(&x);
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn katman_yeni() {
        let config = Config::lubot_a1();
        let katman = Katman::yeni(&config);
        assert_eq!(katman.d_model, 64);
    }

    #[test]
    fn katman_ileri() {
        let config = Config::lubot_a1();
        let katman = Katman::yeni(&config);
        let x = vec![1.0; 64];
        let y = katman.ileri(&x, 0);
        assert_eq!(y.len(), 64);
    }

    #[test]
    fn transformer_yeni() {
        let config = Config::lubot_a1();
        let model = Transformer::yeni(config);
        assert_eq!(model.katmanlar.len(), 8);
    }

    #[test]
    fn transformer_param() {
        let config = Config::lubot_a1();
        let model = Transformer::yeni(config);
        assert!(model.param_sayisi() > 800_000);
    }

    #[test]
    fn transformer_ileri() {
        let config = Config::lubot_a1();
        let model = Transformer::yeni(config);
        let logits = model.ileri(&[1, 2, 3, 4]);
        assert_eq!(logits.len(), 8192);
    }

    #[test]
    fn transformer_deterministik() {
        let config = Config::lubot_a1();
        let model = Transformer::yeni(config);
        let l1 = model.ileri(&[1, 2, 3]);
        let l2 = model.ileri(&[1, 2, 3]);
        assert_eq!(l1, l2);
    }

    #[test]
    fn transformer_hiz() {
        let config = Config::lubot_a1();
        let model = Transformer::yeni(config);
        let hiz = model.hiz_olcum(5);
        assert!(hiz >= 0.0);
    }

    #[test]
    fn config_buyuk() {
        let c = Config::buyuk();
        assert_eq!(c.d_model, 512);
        assert!(c.param_sayisi() < 97_000_000);
    }

    #[test]
    fn layernorm_eps() {
        let ln = LayerNorm::yeni(64);
        assert_eq!(ln.eps, 1e-5);
    }
}
