//! Lubot checkpoint — 143 checkpoint, data bucket, seffaflik, 3-asamali-egitim ilhami.
//!
//! K1: sifirdan yazildi.
//! 3-asamali-egitim: 143 checkpoint (3-asamali) 360 checkpoint (Amber), her checkpoint icin data bucket, training log, metrics, code, preprocessing tamamen acik, lit-derin-dar-transformer.
//! Biz: 143 checkpoint, data bucket per checkpoint, loss azalir, grad_norm, provenance.

use std::collections::HashMap;

/// Checkpoint — her checkpoint icin.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub id: usize,
    pub adim: usize,
    pub epoch: usize,
    pub loss: f64,
    pub grad_norm: f64,
    pub lr: f64,
    pub token_sayisi: usize,
    pub veri_kovasi: String,
    pub agirlik_yolu: String,
    pub optimizer_yolu: String,
    pub fim_orani: f64,
}

impl Checkpoint {
    #[must_use]
    pub fn yeni(id: usize, adim: usize, loss: f64, token_sayisi: usize) -> Self {
        Self {
            id,
            adim,
            epoch: adim / 1000,
            loss,
            grad_norm: 1.0 / (1.0 + adim as f64 * 0.001),
            lr: 0.001 * (0.99f64).powf(adim as f64 / 1000.0),
            token_sayisi,
            veri_kovasi: format!("bucket-{:04}", id),
            agirlik_yolu: format!("checkpoints/{:04}/model.bin", id),
            optimizer_yolu: format!("checkpoints/{:04}/optimizer.bin", id),
            fim_orani: if id >= 120 { 0.3 } else { 0.0 },
        }
    }

    #[must_use]
    pub fn loss_azalir_mi(&self, onceki: &Self) -> bool {
        self.loss <= onceki.loss + 0.1 // kucuk dalgalanma tolere
    }
}

/// Checkpoint yoneticisi — 143 checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointYonetici {
    pub checkpoints: Vec<Checkpoint>,
    pub toplam_token: usize,
}

impl CheckpointYonetici {
    #[must_use]
    pub fn yeni(toplam_checkpoint: usize, toplam_token: usize) -> Self {
        let mut checkpoints = Vec::new();
        let token_per_checkpoint = toplam_token / toplam_checkpoint.max(1);
        for i in 0..toplam_checkpoint {
            let adim = i * 100;
            let loss = 10.0 - (i as f64 * 0.05).min(8.0) + (i as f64 * 0.001).sin() * 0.1;
            let token = token_per_checkpoint * (i + 1);
            checkpoints.push(Checkpoint::yeni(i, adim, loss, token));
        }
        Self {
            checkpoints,
            toplam_token,
        }
    }

    #[must_use]
    pub fn lubot() -> Self {
        Self::yeni(143, 47_000)
    }

    #[must_use]
    pub fn checkpoint_sayisi(&self) -> usize {
        self.checkpoints.len()
    }

    #[must_use]
    pub fn son_checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    #[must_use]
    pub fn bucket_haritasi(&self) -> HashMap<String, usize> {
        let mut harita = HashMap::new();
        for cp in &self.checkpoints {
            *harita.entry(cp.veri_kovasi.clone()).or_insert(0) += 1;
        }
        harita
    }

    #[must_use]
    pub fn seffaflik_raporu(&self) -> String {
        format!(
            "Checkpoint Yonetici: {} checkpoint, toplam {} token, son loss {:.3}, seffaflik: her checkpoint icin data bucket + agirlik + optimizer + metrics acik",
            self.checkpoint_sayisi(),
            self.toplam_token,
            self.son_checkpoint().map_or(0.0, |c| c.loss)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_yeni() {
        let cp = Checkpoint::yeni(0, 0, 10.0, 1000);
        assert_eq!(cp.id, 0);
        assert_eq!(cp.loss, 10.0);
    }

    #[test]
    fn loss_azalir() {
        let cp1 = Checkpoint::yeni(0, 0, 10.0, 1000);
        let cp2 = Checkpoint::yeni(1, 100, 9.0, 2000);
        assert!(cp2.loss_azalir_mi(&cp1));
    }

    #[test]
    fn yonetici_yeni() {
        let yonetici = CheckpointYonetici::yeni(10, 10000);
        assert_eq!(yonetici.checkpoint_sayisi(), 10);
    }

    #[test]
    fn lubot_143() {
        let yonetici = CheckpointYonetici::lubot();
        assert_eq!(yonetici.checkpoint_sayisi(), 143);
        assert_eq!(yonetici.toplam_token, 47_000);
    }

    #[test]
    fn son_checkpoint() {
        let yonetici = CheckpointYonetici::lubot();
        assert!(yonetici.son_checkpoint().is_some());
    }

    #[test]
    fn bucket_haritasi() {
        let yonetici = CheckpointYonetici::lubot();
        let harita = yonetici.bucket_haritasi();
        assert_eq!(harita.len(), 143);
    }

    #[test]
    fn seffaflik_raporu() {
        let yonetici = CheckpointYonetici::lubot();
        let rapor = yonetici.seffaflik_raporu();
        assert!(rapor.contains("143"));
        assert!(rapor.contains("seffaflik"));
    }

    #[test]
    fn fim_orani() {
        let yonetici = CheckpointYonetici::lubot();
        let son = yonetici.son_checkpoint().unwrap();
        assert_eq!(son.fim_orani, 0.3);
        let ilk = &yonetici.checkpoints[0];
        assert_eq!(ilk.fim_orani, 0.0);
    }

    #[test]
    fn deterministik() {
        let y1 = CheckpointYonetici::lubot();
        let y2 = CheckpointYonetici::lubot();
        assert_eq!(y1.checkpoint_sayisi(), y2.checkpoint_sayisi());
    }

    #[test]
    fn grad_norm_azalir() {
        let yonetici = CheckpointYonetici::lubot();
        let ilk = &yonetici.checkpoints[0];
        let son = yonetici.son_checkpoint().unwrap();
        assert!(son.grad_norm < ilk.grad_norm);
    }

    #[test]
    fn lr_azalir() {
        let yonetici = CheckpointYonetici::lubot();
        let ilk = &yonetici.checkpoints[0];
        let son = yonetici.son_checkpoint().unwrap();
        assert!(son.lr < ilk.lr);
    }

    #[test]
    fn agirlik_yolu() {
        let cp = Checkpoint::yeni(5, 500, 5.0, 5000);
        assert!(cp.agirlik_yolu.contains("0005"));
    }
}
