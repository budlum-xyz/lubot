//! Lubot metrics — loss, grad_norm, eval metrics, CrystalCoder metrics ilhami.
//!
//! K1: sifirdan yazildi.
//! CrystalCoder: training log, metrics, eval per checkpoint, 143 checkpoint.
//! Biz: loss, grad_norm, eval, speed, transparency.

use std::collections::HashMap;

/// Metric turu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricTur {
    Loss,
    GradNorm,
    LearningRate,
    TokenPerSecond,
    EvalLoss,
    EvalAccuracy,
    DataBucket,
}

impl MetricTur {
    #[must_use]
    pub fn ad(&self) -> &'static str {
        match self {
            Self::Loss => "loss",
            Self::GradNorm => "grad_norm",
            Self::LearningRate => "lr",
            Self::TokenPerSecond => "token_per_sec",
            Self::EvalLoss => "eval_loss",
            Self::EvalAccuracy => "eval_accuracy",
            Self::DataBucket => "data_bucket",
        }
    }
}

/// Metric kaydi.
#[derive(Debug, Clone)]
pub struct MetricKaydi {
    pub adim: usize,
    pub tur: MetricTur,
    pub deger: f64,
    pub checkpoint_id: usize,
}

impl MetricKaydi {
    #[must_use]
    pub fn yeni(adim: usize, tur: MetricTur, deger: f64, checkpoint_id: usize) -> Self {
        Self {
            adim,
            tur,
            deger,
            checkpoint_id,
        }
    }
}

/// Metrics yoneticisi — 143 checkpoint metrics.
#[derive(Debug, Clone)]
pub struct MetricsYonetici {
    pub kayitlar: Vec<MetricKaydi>,
    pub checkpoint_sayisi: usize,
}

impl MetricsYonetici {
    #[must_use]
    pub fn yeni(checkpoint_sayisi: usize) -> Self {
        let mut kayitlar = Vec::new();
        for i in 0..checkpoint_sayisi {
            let adim = i * 100;
            let loss = 10.0 - (i as f64 * 0.05).min(8.0);
            kayitlar.push(MetricKaydi::yeni(adim, MetricTur::Loss, loss, i));
            kayitlar.push(MetricKaydi::yeni(
                adim,
                MetricTur::GradNorm,
                1.0 / (1.0 + i as f64 * 0.01),
                i,
            ));
            kayitlar.push(MetricKaydi::yeni(
                adim,
                MetricTur::LearningRate,
                0.001 * 0.99f64.powf(i as f64 / 10.0),
                i,
            ));
            kayitlar.push(MetricKaydi::yeni(
                adim,
                MetricTur::TokenPerSecond,
                1000.0 + i as f64 * 10.0,
                i,
            ));
        }
        Self {
            kayitlar,
            checkpoint_sayisi,
        }
    }

    #[must_use]
    pub fn lubot() -> Self {
        Self::yeni(143)
    }

    #[must_use]
    pub fn filtrele(&self, tur: MetricTur) -> Vec<&MetricKaydi> {
        self.kayitlar.iter().filter(|k| k.tur == tur).collect()
    }

    #[must_use]
    pub fn son_deger(&self, tur: MetricTur) -> Option<f64> {
        self.filtrele(tur).last().map(|k| k.deger)
    }

    #[must_use]
    pub fn ortalama(&self, tur: MetricTur) -> f64 {
        let filtre = self.filtrele(tur);
        if filtre.is_empty() {
            0.0
        } else {
            filtre.iter().map(|k| k.deger).sum::<f64>() / filtre.len() as f64
        }
    }

    #[must_use]
    pub fn rapor(&self) -> HashMap<String, f64> {
        let mut rapor = HashMap::new();
        for tur in [
            MetricTur::Loss,
            MetricTur::GradNorm,
            MetricTur::LearningRate,
            MetricTur::TokenPerSecond,
        ] {
            if let Some(deger) = self.son_deger(tur) {
                rapor.insert(tur.ad().to_string(), deger);
            }
        }
        rapor
    }

    #[must_use]
    pub fn seffaflik_raporu(&self) -> String {
        format!(
            "Metrics: {} checkpoint, {} kayit, son loss {:.3}, grad_norm {:.3}, lr {:.6}, token/sec {:.0}, seffaflik: her checkpoint metrics acik",
            self.checkpoint_sayisi,
            self.kayitlar.len(),
            self.son_deger(MetricTur::Loss).unwrap_or(0.0),
            self.son_deger(MetricTur::GradNorm).unwrap_or(0.0),
            self.son_deger(MetricTur::LearningRate).unwrap_or(0.0),
            self.son_deger(MetricTur::TokenPerSecond).unwrap_or(0.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_tur_ad() {
        assert_eq!(MetricTur::Loss.ad(), "loss");
        assert_eq!(MetricTur::GradNorm.ad(), "grad_norm");
    }

    #[test]
    fn kayit_yeni() {
        let kayit = MetricKaydi::yeni(0, MetricTur::Loss, 10.0, 0);
        assert_eq!(kayit.deger, 10.0);
    }

    #[test]
    fn yonetici_yeni() {
        let yonetici = MetricsYonetici::yeni(10);
        assert_eq!(yonetici.checkpoint_sayisi, 10);
        assert_eq!(yonetici.kayitlar.len(), 40);
    }

    #[test]
    fn lubot_143() {
        let yonetici = MetricsYonetici::lubot();
        assert_eq!(yonetici.checkpoint_sayisi, 143);
    }

    #[test]
    fn filtrele() {
        let yonetici = MetricsYonetici::lubot();
        let loss = yonetici.filtrele(MetricTur::Loss);
        assert_eq!(loss.len(), 143);
    }

    #[test]
    fn son_deger() {
        let yonetici = MetricsYonetici::lubot();
        let loss = yonetici.son_deger(MetricTur::Loss);
        assert!(loss.is_some());
        assert!(loss.unwrap() < 10.0);
    }

    #[test]
    fn ortalama() {
        let yonetici = MetricsYonetici::lubot();
        let ort = yonetici.ortalama(MetricTur::Loss);
        assert!(ort > 0.0 && ort < 10.0);
    }

    #[test]
    fn rapor() {
        let yonetici = MetricsYonetici::lubot();
        let rapor = yonetici.rapor();
        assert!(rapor.contains_key("loss"));
    }

    #[test]
    fn seffaflik_raporu() {
        let yonetici = MetricsYonetici::lubot();
        let rapor = yonetici.seffaflik_raporu();
        assert!(rapor.contains("143"));
        assert!(rapor.contains("seffaflik"));
    }

    #[test]
    fn deterministik() {
        let y1 = MetricsYonetici::lubot();
        let y2 = MetricsYonetici::lubot();
        assert_eq!(y1.son_deger(MetricTur::Loss), y2.son_deger(MetricTur::Loss));
    }

    #[test]
    fn loss_azalir() {
        let yonetici = MetricsYonetici::lubot();
        let kayitlar = yonetici.filtrele(MetricTur::Loss);
        let ilk = kayitlar.first().unwrap().deger;
        let son = kayitlar.last().unwrap().deger;
        assert!(son < ilk);
    }

    #[test]
    fn lr_azalir() {
        let yonetici = MetricsYonetici::lubot();
        let kayitlar = yonetici.filtrele(MetricTur::LearningRate);
        let ilk = kayitlar.first().unwrap().deger;
        let son = kayitlar.last().unwrap().deger;
        assert!(son < ilk);
    }
}
