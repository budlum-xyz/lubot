//! Lubot paralel — paralellik boyutlari, CrystalCoder altyapi ilhami.
//!
//! K1: sifirdan yazildi.
//! CrystalCoder: 224 GPU, batch 2240 (224*10), CG-1 4 exaFLOPS 54M core 64-node, mixed-precision BF16/FP32.
//! Biz: paralellik boyutlari, batch hesaplama, mixed-precision iskeleti.

/// Paralellik config.
#[derive(Debug, Clone)]
pub struct ParalelConfig {
    pub gpu_sayisi: usize,
    pub batch_per_gpu: usize,
    pub toplam_batch: usize,
    pub d_model: usize,
    pub seq_len: usize,
    pub bf16: bool,
    pub fp32_weights: bool,
}

impl ParalelConfig {
    #[must_use]
    pub fn yeni(gpu_sayisi: usize, batch_per_gpu: usize, d_model: usize, seq_len: usize) -> Self {
        Self {
            gpu_sayisi,
            batch_per_gpu,
            toplam_batch: gpu_sayisi * batch_per_gpu,
            d_model,
            seq_len,
            bf16: true,
            fp32_weights: true,
        }
    }

    #[must_use]
    pub fn crystal_benzeri() -> Self {
        // CrystalCoder: 224 GPU, batch 2240 (224*10)
        Self::yeni(224, 10, 4096, 2048)
    }

    #[must_use]
    pub fn lubot() -> Self {
        // Lubot: kucuk olcek, 1 GPU, batch 8
        Self::yeni(1, 8, 64, 256)
    }

    #[must_use]
    pub fn toplam_batch(&self) -> usize {
        self.toplam_batch
    }

    #[must_use]
    pub fn token_per_batch(&self) -> usize {
        self.toplam_batch * self.seq_len
    }

    #[must_use]
    pub fn mixed_precision_aciklama(&self) -> String {
        if self.bf16 && self.fp32_weights {
            "BF16 activ/grad, FP32 weights (CrystalCoder benzeri)".to_string()
        } else {
            "FP32".to_string()
        }
    }
}

/// CG-1 benzeri supercomputer iskeleti — 4 exaFLOPS 54M core 64-node (isimsiz, sadece olcek).
#[derive(Debug, Clone)]
pub struct SuperComputer {
    pub exaflops: f64,
    pub core_sayisi: usize,
    pub node_sayisi: usize,
    pub aciklama: String,
}

impl SuperComputer {
    #[must_use]
    pub fn cg1_benzeri() -> Self {
        Self {
            exaflops: 4.0,
            core_sayisi: 54_000_000,
            node_sayisi: 64,
            aciklama: "4 exaFLOPS, 54M core, 64-node (CG-1 benzeri olcek, isim yok)".to_string(),
        }
    }

    #[must_use]
    pub fn lubot_sandbox() -> Self {
        Self {
            exaflops: 0.001,
            core_sayisi: 4,
            node_sayisi: 1,
            aciklama: "sandbox CPU (K6 olculdu)".to_string(),
        }
    }

    #[must_use]
    pub fn hiz_orani(&self, other: &Self) -> f64 {
        self.exaflops / other.exaflops
    }
}

/// Batch hesaplama.
#[derive(Debug, Clone)]
pub struct BatchHesap {
    pub config: ParalelConfig,
}

impl BatchHesap {
    #[must_use]
    pub fn yeni(config: ParalelConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn adim_sayisi(&self, toplam_token: usize) -> usize {
        let token_per_batch = self.config.token_per_batch();
        toplam_token.checked_div(token_per_batch).unwrap_or(0)
    }

    #[must_use]
    pub fn sure_tahmini_saat(&self, toplam_token: usize, token_per_second_per_gpu: f64) -> f64 {
        let total_tps = token_per_second_per_gpu * self.config.gpu_sayisi as f64;
        if total_tps == 0.0 {
            0.0
        } else {
            (toplam_token as f64 / total_tps) / 3600.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paralel_yeni() {
        let c = ParalelConfig::yeni(2, 4, 64, 128);
        assert_eq!(c.toplam_batch(), 8);
    }

    #[test]
    fn crystal_benzeri() {
        let c = ParalelConfig::crystal_benzeri();
        assert_eq!(c.gpu_sayisi, 224);
        assert_eq!(c.toplam_batch(), 2240);
    }

    #[test]
    fn lubot() {
        let c = ParalelConfig::lubot();
        assert_eq!(c.gpu_sayisi, 1);
        assert_eq!(c.batch_per_gpu, 8);
    }

    #[test]
    fn token_per_batch() {
        let c = ParalelConfig::yeni(2, 4, 64, 128);
        assert_eq!(c.token_per_batch(), 8 * 128);
    }

    #[test]
    fn mixed_precision() {
        let c = ParalelConfig::lubot();
        let aciklama = c.mixed_precision_aciklama();
        assert!(aciklama.contains("BF16"));
    }

    #[test]
    fn supercomputer_cg1() {
        let sc = SuperComputer::cg1_benzeri();
        assert_eq!(sc.exaflops, 4.0);
        assert_eq!(sc.core_sayisi, 54_000_000);
    }

    #[test]
    fn supercomputer_sandbox() {
        let sc = SuperComputer::lubot_sandbox();
        assert!(sc.exaflops < 1.0);
    }

    #[test]
    fn hiz_orani() {
        let cg1 = SuperComputer::cg1_benzeri();
        let sandbox = SuperComputer::lubot_sandbox();
        let oran = cg1.hiz_orani(&sandbox);
        assert!(oran > 1000.0);
    }

    #[test]
    fn batch_hesap_adim() {
        let config = ParalelConfig::yeni(1, 8, 64, 256);
        let hesap = BatchHesap::yeni(config);
        let adim = hesap.adim_sayisi(47_000);
        assert!(adim > 0);
    }

    #[test]
    fn batch_sure_tahmini() {
        let config = ParalelConfig::yeni(1, 8, 64, 256);
        let hesap = BatchHesap::yeni(config);
        let sure = hesap.sure_tahmini_saat(47_000, 1000.0);
        assert!(sure >= 0.0);
    }

    #[test]
    fn deterministik() {
        let c1 = ParalelConfig::lubot();
        let c2 = ParalelConfig::lubot();
        assert_eq!(c1.toplam_batch(), c2.toplam_batch());
    }
}
