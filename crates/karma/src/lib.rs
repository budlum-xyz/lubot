//! Lubot karma — veri hazirlama, preprocessing, data bucket, CrystalCoder veri karisimi ilhami.
//!
//! K1: sifirdan yazildi.
//! CrystalCoder veri: SlimPajama 690B + StarCoder 291B, 3 asama 345B/927B/100B+10B, FIM 0.3 SPM 0.5.
//! Biz: gercek 893 + sentetik 152 + derleyici 5 + mufredat 88 = 1138, 3 asama, FIM 0.3, data bucket.

use std::collections::HashMap;

/// Veri kaynagi — kendi agacimiz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VeriKaynagi {
    Gercek,
    Sentetik,
    Derleyici,
    Mufredat,
}

impl VeriKaynagi {
    #[must_use]
    pub fn ad(&self) -> &'static str {
        match self {
            Self::Gercek => "gercek",
            Self::Sentetik => "sentetik",
            Self::Derleyici => "derleyici",
            Self::Mufredat => "mufredat",
        }
    }
}

/// Data bucket — her checkpoint icin.
#[derive(Debug, Clone)]
pub struct DataBucket {
    pub id: usize,
    pub kaynak: VeriKaynagi,
    pub token_sayisi: usize,
    pub dosya_yolu: String,
    pub fim_orani: f64,
}

impl DataBucket {
    #[must_use]
    pub fn yeni(id: usize, kaynak: VeriKaynagi, token_sayisi: usize, fim_orani: f64) -> Self {
        Self {
            id,
            kaynak,
            token_sayisi,
            dosya_yolu: format!("data/bucket-{:04}-{}.jsonl", id, kaynak.ad()),
            fim_orani,
        }
    }
}

/// Preprocessing — BPE + FIM + ozel token.
#[derive(Debug, Clone)]
pub struct Preprocessing {
    pub vocab_boyutu: usize,
    pub fim_orani: f64,
    pub spm_orani: f64,
    pub ozel_token_sayisi: usize,
}

impl Preprocessing {
    #[must_use]
    pub fn yeni(
        vocab_boyutu: usize,
        fim_orani: f64,
        spm_orani: f64,
        ozel_token_sayisi: usize,
    ) -> Self {
        Self {
            vocab_boyutu,
            fim_orani,
            spm_orani,
            ozel_token_sayisi,
        }
    }

    #[must_use]
    pub fn lubot() -> Self {
        Self::yeni(8214, 0.3, 0.5, 22)
    }

    /// FIM uygula — %0.3.
    #[must_use]
    pub fn fim_uygula(&self, metin: &str) -> String {
        if metin.len() < 20 {
            return metin.to_string();
        }
        let orta = metin.len() / 2;
        format!(
            "<fim_prefix>{} <fim_middle>{} <fim_suffix>{}",
            &metin[..orta / 2],
            &metin[orta / 2..orta + orta / 2],
            &metin[orta + orta / 2..]
        )
    }

    /// Token sayisi tahmini — 1 token ~4 byte.
    #[must_use]
    pub fn token_tahmini(&self, metin: &str) -> usize {
        metin.len() / 4
    }
}

/// 3 asamali veri karisimi — CrystalCoder benzeri ama Lubot verisiyle.
#[derive(Debug, Clone)]
pub struct UcAsamaliKarisim {
    pub asama1: AsamaKarisim,
    pub asama2: AsamaKarisim,
    pub asama3: AsamaKarisim,
    pub toplam_token: usize,
}

#[derive(Debug, Clone)]
pub struct AsamaKarisim {
    pub asama: usize,
    pub token_sayisi: usize,
    pub dagilim: HashMap<VeriKaynagi, usize>,
    pub aciklama: String,
}

impl UcAsamaliKarisim {
    #[must_use]
    pub fn lubot() -> Self {
        // Stage1: 345B SlimPajama ilk yari benzeri — biz gercek %50 15K
        let mut d1 = HashMap::new();
        d1.insert(VeriKaynagi::Gercek, 15_000);
        let asama1 = AsamaKarisim {
            asama: 1,
            token_sayisi: 15_000,
            dagilim: d1,
            aciklama: "asama-1: ilk yari gercek veri — dil temeli (Crystal Stage1 345B SlimPajama ilk yari benzeri)".to_string(),
        };

        // Stage2: 927B = 345B SlimPajama diger yari + 2*291B StarCoder benzeri — biz gercek %50 + sentetik 2 epoch + derleyici 26K
        let mut d2 = HashMap::new();
        d2.insert(VeriKaynagi::Gercek, 13_000);
        d2.insert(VeriKaynagi::Sentetik, 13_000);
        d2.insert(VeriKaynagi::Derleyici, 100);
        let asama2 = AsamaKarisim {
            asama: 2,
            token_sayisi: 26_000,
            dagilim: d2,
            aciklama: "asama-2: diger yari gercek + 2 epoch sentetik + derleyici — kod+dil dengesi (Crystal Stage2 927B benzeri)".to_string(),
        };

        // Stage3: 100B Python/web + 10B SlimPajama + FIM 0.3 SPM 0.5 benzeri — biz mufredat %90 + gercek %10 6K FIM 0.3
        let mut d3 = HashMap::new();
        d3.insert(VeriKaynagi::Mufredat, 5_400);
        d3.insert(VeriKaynagi::Gercek, 600);
        let asama3 = AsamaKarisim {
            asama: 3,
            token_sayisi: 6_000,
            dagilim: d3,
            aciklama: "asama-3: mufredat Python/web + gercek %10 — uzmanlasma FIM 0.3 (Crystal Stage3 100B+10B benzeri)".to_string(),
        };

        Self {
            toplam_token: 15_000 + 26_000 + 6_000,
            asama1,
            asama2,
            asama3,
        }
    }

    #[must_use]
    pub fn toplam_token(&self) -> usize {
        self.toplam_token
    }

    #[must_use]
    pub fn bucket_olustur(&self) -> Vec<DataBucket> {
        let mut buckets = Vec::new();
        let mut id = 0;
        for (kaynak, token) in &self.asama1.dagilim {
            buckets.push(DataBucket::yeni(id, *kaynak, *token, 0.0));
            id += 1;
        }
        for (kaynak, token) in &self.asama2.dagilim {
            buckets.push(DataBucket::yeni(id, *kaynak, *token, 0.0));
            id += 1;
        }
        for (kaynak, token) in &self.asama3.dagilim {
            buckets.push(DataBucket::yeni(id, *kaynak, *token, 0.3));
            id += 1;
        }
        buckets
    }

    #[must_use]
    pub fn seffaflik_raporu(&self) -> String {
        format!(
            "3 asamali karisim: toplam {} token, asama1 {} token ({}), asama2 {} token ({}), asama3 {} token ({}), FIM 0.3 SPM 0.5",
            self.toplam_token,
            self.asama1.token_sayisi,
            self.asama1.aciklama,
            self.asama2.token_sayisi,
            self.asama2.aciklama,
            self.asama3.token_sayisi,
            self.asama3.aciklama
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn veri_kaynagi_ad() {
        assert_eq!(VeriKaynagi::Gercek.ad(), "gercek");
        assert_eq!(VeriKaynagi::Sentetik.ad(), "sentetik");
    }

    #[test]
    fn data_bucket_yeni() {
        let bucket = DataBucket::yeni(0, VeriKaynagi::Gercek, 1000, 0.0);
        assert_eq!(bucket.id, 0);
        assert_eq!(bucket.token_sayisi, 1000);
    }

    #[test]
    fn preprocessing_yeni() {
        let prep = Preprocessing::lubot();
        assert_eq!(prep.vocab_boyutu, 8214);
        assert_eq!(prep.fim_orani, 0.3);
    }

    #[test]
    fn fim_uygula() {
        let prep = Preprocessing::lubot();
        let metin = "fn main() { println!(\"merhaba dunya nasilsin\"); }";
        let sonuc = prep.fim_uygula(metin);
        assert!(sonuc.contains("<fim_prefix>"));
    }

    #[test]
    fn token_tahmini() {
        let prep = Preprocessing::lubot();
        let token = prep.token_tahmini("hello world test");
        assert!(token > 0);
    }

    #[test]
    fn uc_asamali_lubot() {
        let karisim = UcAsamaliKarisim::lubot();
        assert_eq!(karisim.toplam_token(), 47_000);
        assert_eq!(karisim.asama1.asama, 1);
    }

    #[test]
    fn bucket_olustur() {
        let karisim = UcAsamaliKarisim::lubot();
        let buckets = karisim.bucket_olustur();
        assert!(!buckets.is_empty());
        assert!(buckets.iter().any(|b| b.fim_orani == 0.3));
    }

    #[test]
    fn seffaflik_raporu() {
        let karisim = UcAsamaliKarisim::lubot();
        let rapor = karisim.seffaflik_raporu();
        assert!(rapor.contains("3 asamali"));
        assert!(rapor.contains("FIM"));
    }

    #[test]
    fn deterministik() {
        let k1 = UcAsamaliKarisim::lubot();
        let k2 = UcAsamaliKarisim::lubot();
        assert_eq!(k1.toplam_token(), k2.toplam_token());
    }

    #[test]
    fn veri_kaynagi_hash() {
        let mut map = HashMap::new();
        map.insert(VeriKaynagi::Gercek, 1);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn asama_aciklama() {
        let karisim = UcAsamaliKarisim::lubot();
        assert!(karisim.asama1.aciklama.contains("dil temeli"));
    }
}
