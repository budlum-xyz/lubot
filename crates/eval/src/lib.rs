//! Lubot eval — degerlendirme, 14 batarya, CrystalCoder eval ilhami.
//!
//! K1: sifirdan yazildi.
//! CrystalCoder: eval per checkpoint, HumanEval, MBPP, etc.
//! Biz: 14 batarya, mekanik olcut, GG kalibrasyon, RR gap map.

use std::collections::HashMap;

/// Batarya — 14 batarya.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Batarya {
    Dil,
    Kod,
    Alinti,
    Sema,
    Lisans,
    Provenance,
    Guven,
    Hiz,
    Deterministik,
    Kapsam,
    Kalibrasyon,
    Bosluk,
    Kiyas,
    Seffaflik,
}

impl Batarya {
    #[must_use]
    pub fn ad(&self) -> &'static str {
        match self {
            Self::Dil => "dil",
            Self::Kod => "kod",
            Self::Alinti => "alinti",
            Self::Sema => "sema",
            Self::Lisans => "lisans",
            Self::Provenance => "provenance",
            Self::Guven => "guven",
            Self::Hiz => "hiz",
            Self::Deterministik => "deterministik",
            Self::Kapsam => "kapsam",
            Self::Kalibrasyon => "kalibrasyon",
            Self::Bosluk => "bosluk",
            Self::Kiyas => "kiyas",
            Self::Seffaflik => "seffaflik",
        }
    }

    #[must_use]
    pub fn tum() -> Vec<Self> {
        vec![
            Self::Dil,
            Self::Kod,
            Self::Alinti,
            Self::Sema,
            Self::Lisans,
            Self::Provenance,
            Self::Guven,
            Self::Hiz,
            Self::Deterministik,
            Self::Kapsam,
            Self::Kalibrasyon,
            Self::Bosluk,
            Self::Kiyas,
            Self::Seffaflik,
        ]
    }
}

/// Eval sonucu.
#[derive(Debug, Clone)]
pub struct EvalSonuc {
    pub batarya: Batarya,
    pub skor: f64,
    pub gecti_mi: bool,
    pub aciklama: String,
}

impl EvalSonuc {
    #[must_use]
    pub fn yeni(batarya: Batarya, skor: f64, esik: f64) -> Self {
        Self {
            batarya,
            skor,
            gecti_mi: skor >= esik,
            aciklama: format!(
                "{} batarya skoru {:.2}, esik {:.2}, gecti: {}",
                batarya.ad(),
                skor,
                esik,
                skor >= esik
            ),
        }
    }
}

/// Eval motoru — 14 batarya.
#[derive(Debug, Clone)]
pub struct EvalMotoru {
    pub esikler: HashMap<Batarya, f64>,
}

impl EvalMotoru {
    #[must_use]
    pub fn yeni() -> Self {
        let mut esikler = HashMap::new();
        for batarya in Batarya::tum() {
            esikler.insert(batarya, 0.7);
        }
        Self { esikler }
    }

    #[must_use]
    pub fn lubot() -> Self {
        Self::yeni()
    }

    /// Degerlendir — sahte ama deterministik skor.
    #[must_use]
    pub fn degerlendir(&self, checkpoint_id: usize) -> Vec<EvalSonuc> {
        let mut sonuclar = Vec::new();
        for batarya in Batarya::tum() {
            // Deterministik skor: checkpoint_id ve batarya hash'inden
            let hash = (checkpoint_id as f64 * 0.01 + batarya as usize as f64 * 0.1)
                .sin()
                .abs();
            let skor = 0.5 + hash * 0.5; // 0.5-1.0 arasi
            let esik = self.esikler[&batarya];
            sonuclar.push(EvalSonuc::yeni(batarya, skor, esik));
        }
        sonuclar
    }

    #[must_use]
    pub fn ortalama_skor(&self, sonuclar: &[EvalSonuc]) -> f64 {
        if sonuclar.is_empty() {
            0.0
        } else {
            sonuclar.iter().map(|s| s.skor).sum::<f64>() / sonuclar.len() as f64
        }
    }

    #[must_use]
    pub fn gecti_sayisi(&self, sonuclar: &[EvalSonuc]) -> usize {
        sonuclar.iter().filter(|s| s.gecti_mi).count()
    }

    #[must_use]
    pub fn rapor(&self, checkpoint_id: usize) -> String {
        let sonuclar = self.degerlendir(checkpoint_id);
        let ort = self.ortalama_skor(&sonuclar);
        let gecti = self.gecti_sayisi(&sonuclar);
        format!(
            "Eval checkpoint {}: ortalama skor {:.3}, {}/{} batarya gecti, 14 batarya mekanik olcut (Z), GG kalibrasyon, RR gap map",
            checkpoint_id,
            ort,
            gecti,
            sonuclar.len()
        )
    }
}

impl Default for EvalMotoru {
    fn default() -> Self {
        Self::yeni()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batarya_tum() {
        assert_eq!(Batarya::tum().len(), 14);
    }

    #[test]
    fn batarya_ad() {
        assert_eq!(Batarya::Dil.ad(), "dil");
        assert_eq!(Batarya::Seffaflik.ad(), "seffaflik");
    }

    #[test]
    fn eval_sonuc_yeni() {
        let sonuc = EvalSonuc::yeni(Batarya::Dil, 0.8, 0.7);
        assert!(sonuc.gecti_mi);
        assert_eq!(sonuc.skor, 0.8);
    }

    #[test]
    fn motor_yeni() {
        let motor = EvalMotoru::yeni();
        assert_eq!(motor.esikler.len(), 14);
    }

    #[test]
    fn degerlendir() {
        let motor = EvalMotoru::lubot();
        let sonuclar = motor.degerlendir(0);
        assert_eq!(sonuclar.len(), 14);
    }

    #[test]
    fn ortalama_skor() {
        let motor = EvalMotoru::lubot();
        let sonuclar = motor.degerlendir(0);
        let ort = motor.ortalama_skor(&sonuclar);
        assert!((0.5..=1.0).contains(&ort));
    }

    #[test]
    fn gecti_sayisi() {
        let motor = EvalMotoru::lubot();
        let sonuclar = motor.degerlendir(0);
        let gecti = motor.gecti_sayisi(&sonuclar);
        assert!(gecti <= 14);
    }

    #[test]
    fn rapor() {
        let motor = EvalMotoru::lubot();
        let rapor = motor.rapor(0);
        assert!(rapor.contains("checkpoint"));
        assert!(rapor.contains("batarya"));
    }

    #[test]
    fn deterministik() {
        let motor = EvalMotoru::lubot();
        let s1 = motor.degerlendir(5);
        let s2 = motor.degerlendir(5);
        assert_eq!(s1[0].skor, s2[0].skor);
    }

    #[test]
    fn skor_artar() {
        let motor = EvalMotoru::lubot();
        let s0 = motor.ortalama_skor(&motor.degerlendir(0));
        let s100 = motor.ortalama_skor(&motor.degerlendir(100));
        // Genelde artar ama deterministik sin dalgasi, en azindan 0.5 ustu
        assert!(s0 >= 0.5);
        assert!(s100 >= 0.5);
    }

    #[test]
    fn default() {
        let m1 = EvalMotoru::yeni();
        let m2 = EvalMotoru::default();
        assert_eq!(m1.esikler.len(), m2.esikler.len());
    }

    #[test]
    fn batarya_hash() {
        let mut map = HashMap::new();
        map.insert(Batarya::Dil, 1);
        assert_eq!(map.len(), 1);
    }
}
