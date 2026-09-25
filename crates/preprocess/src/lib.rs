//! Lubot preprocess — BPE + FIM + ozel token preprocessing, CrystalCoder preprocessing ilhami.
//!
//! K1: sifirdan yazildi.
//! CrystalCoder: preprocessing code tamamen acik, FIM 4 token, 14 ozel kod metadata, 4 instruction, 32032 vocab, FIM rate 0.3 SPM 0.5, StarCoder yontemi.
//! Biz: 8214 vocab, FIM 4, kod 14, instruction 4, FIM 0.3, kendi verisi.

use std::collections::HashMap;

/// Preprocess config.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub vocab_boyutu: usize,
    pub fim_orani: f64,
    pub spm_orani: f64,
    pub ozel_token_sayisi: usize,
    pub max_seq: usize,
}

impl PreprocessConfig {
    #[must_use]
    pub fn lubot() -> Self {
        Self {
            vocab_boyutu: 8214,
            fim_orani: 0.3,
            spm_orani: 0.5,
            ozel_token_sayisi: 22,
            max_seq: 256,
        }
    }

    #[must_use]
    pub fn crystal_benzeri() -> Self {
        Self {
            vocab_boyutu: 32032,
            fim_orani: 0.3,
            spm_orani: 0.5,
            ozel_token_sayisi: 22,
            max_seq: 2048,
        }
    }
}

/// FIM — Fill-in-the-Middle.
#[derive(Debug, Clone)]
pub struct Fim {
    pub oran: f64,
    pub spm_orani: f64,
}

impl Fim {
    #[must_use]
    pub fn yeni(oran: f64, spm_orani: f64) -> Self {
        Self { oran, spm_orani }
    }

    #[must_use]
    pub fn lubot() -> Self {
        Self::yeni(0.3, 0.5)
    }

    /// FIM uygula — prefix/middle/suffix.
    #[must_use]
    pub fn uygula(&self, metin: &str) -> String {
        if metin.len() < 20 {
            return metin.to_string();
        }
        let orta = metin.len() / 2;
        format!(
            "<fim_prefix>{}<fim_suffix>{}<fim_middle>{}<fim_pad>",
            &metin[..orta / 2],
            &metin[orta + orta / 2..],
            &metin[orta / 2..orta + orta / 2]
        )
    }

    #[must_use]
    pub fn spm_mi(&self, rastgele: f64) -> bool {
        rastgele < self.spm_orani
    }
}

/// Ozel token.
#[derive(Debug, Clone)]
pub struct OzelToken {
    pub metin: String,
    pub id: u32,
    pub tur: OzelTur,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OzelTur {
    Fim,
    KodMetadata,
    Instruction,
}

/// Preprocessor — tum preprocessing.
#[derive(Debug, Clone)]
pub struct Preprocessor {
    pub config: PreprocessConfig,
    pub fim: Fim,
    pub ozel_map: HashMap<String, u32>,
}

impl Preprocessor {
    #[must_use]
    pub fn yeni(config: PreprocessConfig) -> Self {
        let mut ozel_map = HashMap::new();
        let mut next_id = (config.vocab_boyutu - config.ozel_token_sayisi) as u32;

        for token in ["<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<fim_pad>"] {
            ozel_map.insert(token.to_string(), next_id);
            next_id += 1;
        }

        for token in [
            "<dosya_adi>",
            "<jupyter_baslangic>",
            "<repo_adi>",
            "<dosya_turu>",
            "<lisans>",
            "<asset_id>",
            "<content_id>",
            "<provenance>",
            "<kanit>",
            "<alinti>",
            "<grant>",
            "<operator>",
            "<epoch>",
            "<checkpoint>",
        ] {
            ozel_map.insert(token.to_string(), next_id);
            next_id += 1;
        }

        for token in ["<sys_baslangic>", "<im_baslangic>", "<im_bitis>", "<cevap>"] {
            ozel_map.insert(token.to_string(), next_id);
            next_id += 1;
        }

        let fim = Fim::yeni(config.fim_orani, config.spm_orani);

        Self {
            config,
            fim,
            ozel_map,
        }
    }

    #[must_use]
    pub fn lubot() -> Self {
        Self::yeni(PreprocessConfig::lubot())
    }

    #[must_use]
    pub fn encode(&self, metin: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut remaining = metin.to_string();

        for (ozel_metin, &id) in &self.ozel_map {
            if remaining.contains(ozel_metin) {
                remaining = remaining.replace(ozel_metin, "");
                ids.push(id);
            }
        }

        for b in remaining.bytes() {
            ids.push(b as u32);
        }

        ids.truncate(self.config.max_seq);
        ids
    }

    #[must_use]
    pub fn preprocess(&self, metin: &str, fim_uygula: bool) -> Vec<u32> {
        let metin = if fim_uygula {
            self.fim.uygula(metin)
        } else {
            metin.to_string()
        };
        self.encode(&metin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_lubot() {
        let c = PreprocessConfig::lubot();
        assert_eq!(c.vocab_boyutu, 8214);
        assert_eq!(c.fim_orani, 0.3);
    }

    #[test]
    fn fim_yeni() {
        let fim = Fim::lubot();
        assert_eq!(fim.oran, 0.3);
    }

    #[test]
    fn fim_uygula() {
        let fim = Fim::lubot();
        let metin = "fn main() { println!(\"merhaba dunya nasilsin\"); }";
        let sonuc = fim.uygula(metin);
        assert!(sonuc.contains("<fim_prefix>"));
        assert!(sonuc.contains("<fim_middle>"));
    }

    #[test]
    fn spm_mi() {
        let fim = Fim::lubot();
        assert!(fim.spm_mi(0.3));
        assert!(!fim.spm_mi(0.6));
    }

    #[test]
    fn preprocessor_yeni() {
        let prep = Preprocessor::lubot();
        assert_eq!(prep.ozel_map.len(), 22);
    }

    #[test]
    fn encode() {
        let prep = Preprocessor::lubot();
        let ids = prep.encode("abc");
        assert_eq!(ids, vec![97, 98, 99]);
    }

    #[test]
    fn encode_ozel() {
        let prep = Preprocessor::lubot();
        let ids = prep.encode("<fim_prefix> hello");
        assert!(ids.contains(&prep.ozel_map["<fim_prefix>"]));
    }

    #[test]
    fn preprocess_fim() {
        let prep = Preprocessor::lubot();
        let ids = prep.preprocess("fn main() { test uzun metin burasi }", true);
        assert!(!ids.is_empty());
    }

    #[test]
    fn max_seq() {
        let prep = Preprocessor::lubot();
        let uzun = "a".repeat(1000);
        let ids = prep.encode(&uzun);
        assert!(ids.len() <= 256);
    }

    #[test]
    fn deterministik() {
        let prep = Preprocessor::lubot();
        let ids1 = prep.encode("test");
        let ids2 = prep.encode("test");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn crystal_config() {
        let c = PreprocessConfig::crystal_benzeri();
        assert_eq!(c.vocab_boyutu, 32032);
        assert_eq!(c.max_seq, 2048);
    }

    #[test]
    fn ozel_tur() {
        let token = OzelToken {
            metin: "<fim_prefix>".to_string(),
            id: 8192,
            tur: OzelTur::Fim,
        };
        assert_eq!(token.tur, OzelTur::Fim);
    }
}
