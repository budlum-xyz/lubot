//! Lubot kendinden — kendinden-damitma, G/H/CC.
//!
//! K1: sifirdan yazildi.
//! G: self-instruct dongusu, checkpoint loss'a gore aday uret.
//! H: mekanik juri (alinti, sema, lisans, provenance).
//! CC: hata madenciligi.

use std::collections::HashMap;

/// Aday — model tarafindan uretilen sentetik kayit.
#[derive(Debug, Clone)]
pub struct Aday {
    pub id: String,
    pub metin: String,
    pub kaynak: String,
    pub tur: String,
    pub guven: f64,
}

/// Juri sonucu — mekanik.
#[derive(Debug, Clone)]
pub struct JuriSonuc {
    pub aday_id: String,
    pub gecti: bool,
    pub sebepler: Vec<String>,
    pub puan: f64,
}

/// Mekanik juri — alinti, sema, lisans, provenance.
pub struct MekanikJuri;

impl MekanikJuri {
    #[must_use]
    pub fn degerlendir(aday: &Aday) -> JuriSonuc {
        let mut sebepler = Vec::new();
        let mut gecti = true;
        let mut puan: f64 = 1.0;

        // Alinti kontrol — metin bos mu?
        if aday.metin.trim().is_empty() {
            gecti = false;
            sebepler.push("bos metin".to_string());
            puan -= 0.5;
        }

        // Sema kontrol — id ve tur var mi?
        if aday.id.is_empty() || aday.tur.is_empty() {
            gecti = false;
            sebepler.push("sema hatasi".to_string());
            puan -= 0.3;
        }

        // Lisans kontrol — kaynak PolyForm Shield mi?
        if !aday.kaynak.contains("PolyForm") && !aday.kaynak.contains("lubot") {
            gecti = false;
            sebepler.push("lisans hatasi".to_string());
            puan -= 0.4;
        }

        // Provenance — id asset_id gibi mi?
        if !aday.id.contains('-') {
            gecti = false;
            sebepler.push("provenance hatasi".to_string());
            puan -= 0.2;
        }

        // Guven esigi
        if aday.guven < 0.5 {
            gecti = false;
            sebepler.push("dusuk guven".to_string());
            puan -= 0.2;
        }

        JuriSonuc {
            aday_id: aday.id.clone(),
            gecti,
            sebepler,
            puan: puan.max(0.0),
        }
    }

    #[must_use]
    pub fn toplu_degerlendir(adaylar: &[Aday]) -> Vec<JuriSonuc> {
        adaylar.iter().map(Self::degerlendir).collect()
    }
}

/// Kendinden-damitma dongusu.
pub struct DamitmaDongusu {
    pub checkpoint_loss: f64,
    pub esik: f64,
}

impl DamitmaDongusu {
    #[must_use]
    pub fn yeni(checkpoint_loss: f64, esik: f64) -> Self {
        Self {
            checkpoint_loss,
            esik,
        }
    }

    /// Aday uret — checkpoint loss'a gore.
    /// Loss dusukse daha fazla aday, yuksekse daha az.
    #[must_use]
    pub fn aday_uret(&self, sayi: usize) -> Vec<Aday> {
        let mut adaylar = Vec::new();
        let gercek_sayi = if self.checkpoint_loss < 0.01 {
            sayi
        } else if self.checkpoint_loss < 0.1 {
            sayi / 2
        } else {
            sayi / 4
        };

        for i in 0..gercek_sayi {
            let guven = 1.0 - self.checkpoint_loss - (i as f64 * 0.001);
            adaylar.push(Aday {
                id: format!("aday-{}-{}", self.checkpoint_loss, i),
                metin: format!("Sentetik metin {} — loss {}", i, self.checkpoint_loss),
                kaynak: "lubot (kendi eser; PolyForm-Shield-1.0.0)".to_string(),
                tur: "sentetik".to_string(),
                guven: guven.clamp(0.0, 1.0),
            });
        }
        adaylar
    }

    /// Filtrele — juri ile.
    #[must_use]
    pub fn filtrele(&self, adaylar: Vec<Aday>) -> (Vec<Aday>, Vec<Aday>, f64) {
        let mut gecen = Vec::new();
        let mut kalan = Vec::new();

        for aday in adaylar {
            let sonuc = MekanikJuri::degerlendir(&aday);
            if sonuc.gecti && sonuc.puan >= self.esik {
                gecen.push(aday);
            } else {
                kalan.push(aday);
            }
        }

        let oran = if gecen.len() + kalan.len() == 0 {
            0.0
        } else {
            gecen.len() as f64 / (gecen.len() + kalan.len()) as f64
        };

        (gecen, kalan, oran)
    }

    /// Tam dongu — uret, degerlendir, filtrele.
    #[must_use]
    pub fn dongu(&self, sayi: usize) -> DamitmaSonuc {
        let adaylar = self.aday_uret(sayi);
        let toplam = adaylar.len();
        let (gecen, kalan, oran) = self.filtrele(adaylar);
        DamitmaSonuc {
            toplam,
            gecen: gecen.len(),
            kalan: kalan.len(),
            oran,
            checkpoint_loss: self.checkpoint_loss,
        }
    }
}

/// Damitma sonucu.
#[derive(Debug, Clone)]
pub struct DamitmaSonuc {
    pub toplam: usize,
    pub gecen: usize,
    pub kalan: usize,
    pub oran: f64,
    pub checkpoint_loss: f64,
}

impl DamitmaSonuc {
    #[must_use]
    pub fn basarili_mi(&self) -> bool {
        self.oran >= 0.8
    }
}

/// Hata madenciligi — kalan adaylardan hata turlerini cikar.
pub struct HataMadenciligi;

impl HataMadenciligi {
    #[must_use]
    pub fn madenle(kalan: &[Aday]) -> HashMap<String, usize> {
        let mut hatalar = HashMap::new();
        for aday in kalan {
            let sonuc = MekanikJuri::degerlendir(aday);
            for sebep in sonuc.sebepler {
                *hatalar.entry(sebep).or_insert(0) += 1;
            }
        }
        hatalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ornek_aday(id: &str, guven: f64) -> Aday {
        Aday {
            id: id.to_string(),
            metin: "ornek metin".to_string(),
            kaynak: "lubot (kendi eser; PolyForm-Shield-1.0.0)".to_string(),
            tur: "sentetik".to_string(),
            guven,
        }
    }

    #[test]
    fn aday_yeni() {
        let a = ornek_aday("aday-1", 0.9);
        assert_eq!(a.guven, 0.9);
    }

    #[test]
    fn juri_bos_metin() {
        let mut a = ornek_aday("aday-1", 0.9);
        a.metin = "".to_string();
        let sonuc = MekanikJuri::degerlendir(&a);
        assert!(!sonuc.gecti);
        assert!(sonuc.sebepler.contains(&"bos metin".to_string()));
    }

    #[test]
    fn juri_sema_hatasi() {
        let mut a = ornek_aday("", 0.9);
        a.id = "".to_string();
        let sonuc = MekanikJuri::degerlendir(&a);
        assert!(!sonuc.gecti);
    }

    #[test]
    fn juri_lisans_hatasi() {
        let mut a = ornek_aday("aday-1", 0.9);
        a.kaynak = "dis kaynak".to_string();
        let sonuc = MekanikJuri::degerlendir(&a);
        assert!(!sonuc.gecti);
    }

    #[test]
    fn juri_dusuk_guven() {
        let a = ornek_aday("aday-1", 0.3);
        let sonuc = MekanikJuri::degerlendir(&a);
        assert!(!sonuc.gecti);
    }

    #[test]
    fn juri_gecti() {
        let a = ornek_aday("aday-1", 0.9);
        let sonuc = MekanikJuri::degerlendir(&a);
        assert!(sonuc.gecti);
        assert!(sonuc.puan > 0.8);
    }

    #[test]
    fn toplu_degerlendir() {
        let adaylar = vec![ornek_aday("a-1", 0.9), ornek_aday("a-2", 0.3)];
        let sonuclar = MekanikJuri::toplu_degerlendir(&adaylar);
        assert_eq!(sonuclar.len(), 2);
    }

    #[test]
    fn damitma_yeni() {
        let d = DamitmaDongusu::yeni(0.002, 0.5);
        assert_eq!(d.checkpoint_loss, 0.002);
    }

    #[test]
    fn aday_uret_loss_dusuk() {
        let d = DamitmaDongusu::yeni(0.001, 0.5);
        let adaylar = d.aday_uret(100);
        assert_eq!(adaylar.len(), 100);
    }

    #[test]
    fn aday_uret_loss_yuksek() {
        let d = DamitmaDongusu::yeni(0.5, 0.5);
        let adaylar = d.aday_uret(100);
        assert!(adaylar.len() < 100);
    }

    #[test]
    fn filtrele() {
        let d = DamitmaDongusu::yeni(0.002, 0.5);
        let adaylar = vec![ornek_aday("a-1", 0.9), ornek_aday("a-2", 0.3)];
        let (gecen, kalan, oran) = d.filtrele(adaylar);
        assert_eq!(gecen.len(), 1);
        assert_eq!(kalan.len(), 1);
        assert!((oran - 0.5).abs() < 1e-9);
    }

    #[test]
    fn dongu() {
        let d = DamitmaDongusu::yeni(0.002, 0.5);
        let sonuc = d.dongu(20);
        assert!(sonuc.toplam > 0);
        assert!(sonuc.oran >= 0.0 && sonuc.oran <= 1.0);
    }

    #[test]
    fn basarili_mi() {
        let sonuc = DamitmaSonuc {
            toplam: 100,
            gecen: 90,
            kalan: 10,
            oran: 0.9,
            checkpoint_loss: 0.002,
        };
        assert!(sonuc.basarili_mi());
    }

    #[test]
    fn hata_madenciligi() {
        let kalan = vec![ornek_aday("a-1", 0.3), ornek_aday("a-2", 0.2)];
        let hatalar = HataMadenciligi::madenle(&kalan);
        assert!(!hatalar.is_empty());
    }

    #[test]
    fn deterministik_aday() {
        let d = DamitmaDongusu::yeni(0.002, 0.5);
        let a1 = d.aday_uret(10);
        let a2 = d.aday_uret(10);
        assert_eq!(a1.len(), a2.len());
        assert_eq!(a1[0].id, a2[0].id);
    }
}
