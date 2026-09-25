//! Lubot bpe-gelismis — gelismis BPE, FIM ve ozel tokenlar, 3-asamali-egitim ilhami.
//!
//! K1: sifirdan yazildi.
//! 3-asamali-egitim tokenizer: 4 FIM token (<|fim_prefix|>), 14 ozel kod metadata (<|filename|>, <|jupyter_start|>, <|reponame|> kod-korpus yontemi), 4 instruction (<|sys_start|>, <|im_start|>), vocab 32032.
//! Biz: kendi vocab 8192 + FIM 4 + ozel kod 14 + instruction 4 = 8214, Lubot adlariyla.

use std::collections::HashMap;

/// Ozel token turu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OzelTur {
    Fim,
    KodMetadata,
    Instruction,
    Genel,
}

/// Ozel token.
#[derive(Debug, Clone)]
pub struct OzelToken {
    pub metin: String,
    pub id: u32,
    pub tur: OzelTur,
}

/// Gelismis BPE sozluk.
#[derive(Debug, Clone)]
pub struct BpeGelismis {
    pub vocab_boyutu: usize,
    pub base_vocab: usize,
    pub fim_tokenlar: Vec<OzelToken>,
    pub kod_tokenlar: Vec<OzelToken>,
    pub instruction_tokenlar: Vec<OzelToken>,
    pub ozel_map: HashMap<String, u32>,
    pub merges: HashMap<(u32, u32), u32>,
}

impl BpeGelismis {
    #[must_use]
    pub fn yeni() -> Self {
        let base_vocab = 8192;
        let mut ozel_map = HashMap::new();
        let mut fim_tokenlar = Vec::new();
        let mut kod_tokenlar = Vec::new();
        let mut instruction_tokenlar = Vec::new();

        let mut next_id = base_vocab as u32;

        // FIM 4 token
        for (i, token) in ["<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<fim_pad>"]
            .iter()
            .enumerate()
        {
            let id = next_id + i as u32;
            ozel_map.insert(token.to_string(), id);
            fim_tokenlar.push(OzelToken {
                metin: token.to_string(),
                id,
                tur: OzelTur::Fim,
            });
        }
        next_id += 4;

        // Kod metadata 14 token — kod-korpus yontemi ilham, Lubot adlariyla
        for (i, token) in [
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
        ]
        .iter()
        .enumerate()
        {
            let id = next_id + i as u32;
            ozel_map.insert(token.to_string(), id);
            kod_tokenlar.push(OzelToken {
                metin: token.to_string(),
                id,
                tur: OzelTur::KodMetadata,
            });
        }
        next_id += 14;

        // Instruction 4 token
        for (i, token) in ["<sys_baslangic>", "<im_baslangic>", "<im_bitis>", "<cevap>"]
            .iter()
            .enumerate()
        {
            let id = next_id + i as u32;
            ozel_map.insert(token.to_string(), id);
            instruction_tokenlar.push(OzelToken {
                metin: token.to_string(),
                id,
                tur: OzelTur::Instruction,
            });
        }
        next_id += 4;

        let vocab_boyutu = next_id as usize;

        Self {
            vocab_boyutu,
            base_vocab,
            fim_tokenlar,
            kod_tokenlar,
            instruction_tokenlar,
            ozel_map,
            merges: HashMap::new(),
        }
    }

    #[must_use]
    pub fn vocab_boyutu(&self) -> usize {
        self.vocab_boyutu
    }

    #[must_use]
    pub fn ozel_sayisi(&self) -> usize {
        self.fim_tokenlar.len() + self.kod_tokenlar.len() + self.instruction_tokenlar.len()
    }

    #[must_use]
    pub fn fim_sayisi(&self) -> usize {
        self.fim_tokenlar.len()
    }

    #[must_use]
    pub fn kod_sayisi(&self) -> usize {
        self.kod_tokenlar.len()
    }

    #[must_use]
    pub fn instruction_sayisi(&self) -> usize {
        self.instruction_tokenlar.len()
    }

    #[must_use]
    pub fn ozel_id(&self, metin: &str) -> Option<u32> {
        self.ozel_map.get(metin).copied()
    }

    /// Encode — ozel tokenlari koru, gerisini byte'lara bol.
    #[must_use]
    pub fn encode(&self, metin: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut remaining = metin.to_string();

        // Ozel tokenlari ara ve koru
        for (ozel_metin, &id) in &self.ozel_map {
            if remaining.contains(ozel_metin) {
                // Basit: ozel token varsa onu id olarak ekle, metinden cikar
                remaining = remaining.replace(ozel_metin, "");
                ids.push(id);
            }
        }

        // Kalan byte'lar
        for b in remaining.bytes() {
            ids.push(b as u32);
        }

        ids
    }

    /// Decode — id'lerden metne.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut metin = String::new();
        let ters_map: HashMap<u32, String> =
            self.ozel_map.iter().map(|(k, &v)| (v, k.clone())).collect();

        for &id in ids {
            if let Some(ozel) = ters_map.get(&id) {
                metin.push_str(ozel);
            } else if id < 256 {
                metin.push(id as u8 as char);
            } else {
                metin.push('?');
            }
        }
        metin
    }

    /// FIM uygula — %0.3 oran.
    #[must_use]
    pub fn fim_uygula(&self, metin: &str, oran: f64) -> String {
        if oran <= 0.0 || metin.len() < 20 {
            return metin.to_string();
        }
        let orta = metin.len() / 2;
        let prefix = &metin[..orta / 2];
        let middle = &metin[orta / 2..orta + orta / 2];
        let suffix = &metin[orta + orta / 2..];
        format!(
            "<fim_prefix>{} <fim_middle>{} <fim_suffix>{} <fim_pad>",
            prefix, middle, suffix
        )
    }

    #[must_use]
    pub fn tum_ozel_tokenlar(&self) -> Vec<OzelToken> {
        let mut tum = Vec::new();
        tum.extend(self.fim_tokenlar.clone());
        tum.extend(self.kod_tokenlar.clone());
        tum.extend(self.instruction_tokenlar.clone());
        tum
    }
}

impl Default for BpeGelismis {
    fn default() -> Self {
        Self::yeni()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yeni_vocab_boyutu() {
        let bpe = BpeGelismis::yeni();
        assert_eq!(bpe.base_vocab, 8192);
        assert_eq!(bpe.vocab_boyutu(), 8192 + 4 + 14 + 4);
    }

    #[test]
    fn ozel_sayisi() {
        let bpe = BpeGelismis::yeni();
        assert_eq!(bpe.ozel_sayisi(), 22);
        assert_eq!(bpe.fim_sayisi(), 4);
        assert_eq!(bpe.kod_sayisi(), 14);
        assert_eq!(bpe.instruction_sayisi(), 4);
    }

    #[test]
    fn ozel_id() {
        let bpe = BpeGelismis::yeni();
        assert!(bpe.ozel_id("<fim_prefix>").is_some());
        assert!(bpe.ozel_id("<dosya_adi>").is_some());
        assert!(bpe.ozel_id("<sys_baslangic>").is_some());
        assert!(bpe.ozel_id("olmayan").is_none());
    }

    #[test]
    fn encode_ozel() {
        let bpe = BpeGelismis::yeni();
        let ids = bpe.encode("<fim_prefix> hello");
        assert!(ids.contains(&bpe.ozel_id("<fim_prefix>").unwrap()));
    }

    #[test]
    fn encode_byte() {
        let bpe = BpeGelismis::yeni();
        let ids = bpe.encode("abc");
        assert_eq!(ids, vec![97, 98, 99]);
    }

    #[test]
    fn decode_ozel() {
        let bpe = BpeGelismis::yeni();
        let id = bpe.ozel_id("<fim_prefix>").unwrap();
        let metin = bpe.decode(&[id]);
        assert_eq!(metin, "<fim_prefix>");
    }

    #[test]
    fn decode_byte() {
        let bpe = BpeGelismis::yeni();
        let metin = bpe.decode(&[97, 98, 99]);
        assert_eq!(metin, "abc");
    }

    #[test]
    fn fim_uygula() {
        let bpe = BpeGelismis::yeni();
        let metin = "fn main() { println!(\"merhaba dunya nasilsin\"); }";
        let sonuc = bpe.fim_uygula(metin, 0.3);
        assert!(sonuc.contains("<fim_prefix>"));
        assert!(sonuc.contains("<fim_middle>"));
    }

    #[test]
    fn fim_oran_sifir() {
        let bpe = BpeGelismis::yeni();
        let metin = "test metin";
        let sonuc = bpe.fim_uygula(metin, 0.0);
        assert_eq!(sonuc, metin);
    }

    #[test]
    fn fim_kisa() {
        let bpe = BpeGelismis::yeni();
        let sonuc = bpe.fim_uygula("kisa", 0.3);
        assert_eq!(sonuc, "kisa");
    }

    #[test]
    fn tum_ozel() {
        let bpe = BpeGelismis::yeni();
        let tum = bpe.tum_ozel_tokenlar();
        assert_eq!(tum.len(), 22);
    }

    #[test]
    fn deterministik_encode() {
        let bpe = BpeGelismis::yeni();
        let ids1 = bpe.encode("test");
        let ids2 = bpe.encode("test");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn bos_metin() {
        let bpe = BpeGelismis::yeni();
        let ids = bpe.encode("");
        assert!(ids.is_empty());
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

    #[test]
    fn default() {
        let bpe1 = BpeGelismis::yeni();
        let bpe2 = BpeGelismis::default();
        assert_eq!(bpe1.vocab_boyutu(), bpe2.vocab_boyutu());
    }

    #[test]
    fn vocab_artisi() {
        let bpe = BpeGelismis::yeni();
        assert!(bpe.vocab_boyutu() > bpe.base_vocab);
    }

    #[test]
    fn kod_metadata_14() {
        let bpe = BpeGelismis::yeni();
        assert_eq!(bpe.kod_tokenlar.len(), 14);
        // kod-korpus yontemi ilham ama Lubot adlariyla — no-upstream-naming
        assert!(bpe.kod_tokenlar.iter().any(|t| t.metin == "<dosya_adi>"));
    }
}
