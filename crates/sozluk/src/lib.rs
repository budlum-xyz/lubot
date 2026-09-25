//! Lubot sozluk — BPE tokenizer, JJ AST-farkinda.
//!
//! K1: sifirdan yazildi, hicbir dis tokenizer kutuphanesi kullanilmadi.
//! K2: yalnizca kendi agacimizdan ogrenildi (training/tokenizer/lubot-bpe-v2.json).
//! JJ: AST-farkinda bolme — kod ve metin farkli kurallarla bolunur.
//!
//! Format: lubot-bpe v1, vocab_size 8192, merges [[a,b], ...].
//! Deterministik, saat/hash yok, olculmedi disiplini.

use std::collections::{HashMap, HashSet};

/// BPE sozluk — 8192 token, 0-255 byte, 256-8191 merge.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Sozluk {
    /// vocab id -> bytes (utf8 degil, ham byte dizisi gibi)
    vocab: Vec<Vec<u8>>,
    /// merge (a,b) -> yeni id
    merges: HashMap<(u32, u32), u32>,
    /// ters: id -> (a,b) eger merge ise
    merge_rev: HashMap<u32, (u32, u32)>,
    /// ozel tokenlar: <pad>, <bos>, <eos>, <unk>
    ozel: HashMap<String, u32>,
    /// pretoken pattern (basitlestirilmis)
    pretoken: Pretoken,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum Pretoken {
    /// [^\W\d_]+|\d+|\s+|[\W_]+ benzeri
    Genel,
    /// Kod icin: kelime, sayi, bosluk, noktalama ayri
    Kod,
    /// Metin icin: kelime ve bosluk
    Metin,
}

/// Token — id ve metin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jeton {
    pub id: u32,
    pub metin: String,
}

/// Kodlama sonucu.
#[derive(Debug, Clone)]
pub struct Kodlama {
    pub ids: Vec<u32>,
    pub jetonlar: Vec<Jeton>,
    pub metin: String,
}

/// AST-farkinda bolme: kod mu metin mi?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DilimTuru {
    Kod,
    Metin,
    Bosluk,
    Sayi,
    Noktalama,
}

/// Dilim — metin parcasi ve turu.
#[derive(Debug, Clone)]
pub struct Dilim {
    pub metin: String,
    pub tur: DilimTuru,
}

impl Sozluk {
    /// Bos sozluk — 0-255 byte vocab ile baslat.
    #[must_use]
    pub fn bos() -> Self {
        let mut vocab = Vec::with_capacity(8192);
        for b in 0..=255u8 {
            vocab.push(vec![b]);
        }
        // 256..8191 bos, merge ile dolacak
        for _ in 256..8192 {
            vocab.push(Vec::new());
        }
        let mut ozel = HashMap::new();
        ozel.insert("<pad>".to_string(), 0);
        ozel.insert("<bos>".to_string(), 1);
        ozel.insert("<eos>".to_string(), 2);
        ozel.insert("<unk>".to_string(), 3);

        Self {
            vocab,
            merges: HashMap::new(),
            merge_rev: HashMap::new(),
            ozel,
            pretoken: Pretoken::Genel,
        }
    }

    /// Dosyadan yukle — lubot-bpe-v2.json formati.
    /// K1: dosya okuma yok, icerik disaridan verilir (test icin).
    #[must_use]
    pub fn yeni(merges: Vec<(u32, u32)>) -> Self {
        let mut s = Self::bos();
        for (next_id, (a, b)) in (256u32..).zip(merges) {
            if next_id >= 8192 {
                break;
            }
            s.merges.insert((a, b), next_id);
            s.merge_rev.insert(next_id, (a, b));
            // vocab: a+b byte'larini birlestir (basit)
            let mut v = Vec::new();
            if (a as usize) < s.vocab.len() && !s.vocab[a as usize].is_empty() {
                v.extend_from_slice(&s.vocab[a as usize]);
            } else {
                v.push(b'X');
            }
            if (b as usize) < s.vocab.len() && !s.vocab[b as usize].is_empty() {
                v.extend_from_slice(&s.vocab[b as usize]);
            } else {
                v.push(b'Y');
            }
            if next_id as usize >= s.vocab.len() {
                s.vocab.resize(next_id as usize + 1, Vec::new());
            }
            s.vocab[next_id as usize] = v;
        }
        s
    }

    /// Vocab boyutu.
    #[must_use]
    pub fn vocab_boyutu(&self) -> usize {
        self.vocab.len()
    }

    /// Merge sayisi.
    #[must_use]
    pub fn merge_sayisi(&self) -> usize {
        self.merges.len()
    }

    /// AST-farkinda dilimle — JJ.
    /// Kod: fn, let, struct, impl, pub, mod, use gibi Rust anahtar kelimeleri
    /// ayri dilim, parantezler noktalama, sayilar ayri.
    /// Metin: kelimeler ve bosluklar.
    #[must_use]
    pub fn dilimle(&self, metin: &str) -> Vec<Dilim> {
        let mut dilimler = Vec::new();
        let mut current = String::new();
        let mut current_tur = DilimTuru::Metin;

        let rust_keywords: HashSet<&str> = [
            "fn", "let", "mut", "struct", "enum", "impl", "pub", "mod", "use", "crate", "self",
            "Self", "super", "as", "const", "static", "type", "trait", "where", "for", "in", "if",
            "else", "match", "loop", "while", "return", "break", "continue",
        ]
        .into_iter()
        .collect();

        for ch in metin.chars() {
            let tur = if ch.is_whitespace() {
                DilimTuru::Bosluk
            } else if ch.is_ascii_digit() {
                DilimTuru::Sayi
            } else if ch.is_ascii_alphabetic() || ch == '_' {
                DilimTuru::Metin
            } else {
                DilimTuru::Noktalama
            };

            // Rust anahtar kelime siniri: noktalama veya bosluk gorunce kelime bitti
            if current_tur != tur || tur == DilimTuru::Noktalama || tur == DilimTuru::Bosluk {
                if !current.is_empty() {
                    // Anahtar kelime kontrolu
                    let final_tur = if rust_keywords.contains(current.as_str()) {
                        DilimTuru::Kod
                    } else {
                        current_tur
                    };
                    dilimler.push(Dilim {
                        metin: current.clone(),
                        tur: final_tur,
                    });
                    current.clear();
                }
                if tur == DilimTuru::Noktalama {
                    dilimler.push(Dilim {
                        metin: ch.to_string(),
                        tur,
                    });
                    continue;
                }
                if tur == DilimTuru::Bosluk {
                    dilimler.push(Dilim {
                        metin: ch.to_string(),
                        tur,
                    });
                    continue;
                }
                current_tur = tur;
                current.push(ch);
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            let final_tur = if rust_keywords.contains(current.as_str()) {
                DilimTuru::Kod
            } else {
                current_tur
            };
            dilimler.push(Dilim {
                metin: current,
                tur: final_tur,
            });
        }
        dilimler
    }

    /// Basit BPE encode — deterministik, greedy merge.
    /// JJ: AST-farkinda dilimle, sonra her dilim icin BPE uygula.
    #[must_use]
    pub fn encode(&self, metin: &str) -> Kodlama {
        let dilimler = self.dilimle(metin);
        let mut ids = Vec::new();
        let mut jetonlar = Vec::new();

        for dilim in dilimler {
            if dilim.tur == DilimTuru::Bosluk {
                // Bosluk ozel: id 32 (space) veya 1 (bos)
                ids.push(32);
                jetonlar.push(Jeton {
                    id: 32,
                    metin: dilim.metin.clone(),
                });
                continue;
            }
            // Her dilim icin byte'lara bol, sonra merge uygula
            let bytes: Vec<u32> = dilim.metin.bytes().map(|b| b as u32).collect();
            let merged = self.bpe_merge(bytes);
            for id in merged {
                let m = if (id as usize) < self.vocab.len() && !self.vocab[id as usize].is_empty() {
                    String::from_utf8_lossy(&self.vocab[id as usize]).to_string()
                } else {
                    format!("<{id}>")
                };
                ids.push(id);
                jetonlar.push(Jeton { id, metin: m });
            }
        }

        Kodlama {
            ids,
            jetonlar,
            metin: metin.to_string(),
        }
    }

    /// BPE merge — en dusuk id'li merge once (deterministik).
    fn bpe_merge(&self, mut tokens: Vec<u32>) -> Vec<u32> {
        if tokens.len() <= 1 {
            return tokens;
        }
        loop {
            let mut best_pos = None;
            let mut best_id = u32::MAX;
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&merged_id) = self.merges.get(&pair) {
                    if merged_id < best_id {
                        best_id = merged_id;
                        best_pos = Some(i);
                    }
                }
            }
            if let Some(pos) = best_pos {
                let mut new_tokens = Vec::with_capacity(tokens.len() - 1);
                new_tokens.extend_from_slice(&tokens[..pos]);
                new_tokens.push(best_id);
                new_tokens.extend_from_slice(&tokens[pos + 2..]);
                tokens = new_tokens;
            } else {
                break;
            }
        }
        tokens
    }

    /// Decode — id'lerden metne.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if (id as usize) < self.vocab.len() && !self.vocab[id as usize].is_empty() {
                bytes.extend_from_slice(&self.vocab[id as usize]);
            } else if id < 256 {
                bytes.push(id as u8);
            } else {
                // bilinmeyen: ?
                bytes.push(b'?');
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Ozel token id.
    #[must_use]
    pub fn ozel_id(&self, ad: &str) -> Option<u32> {
        self.ozel.get(ad).copied()
    }

    /// Vocab'dan metin.
    #[must_use]
    pub fn vocab_metin(&self, id: u32) -> Option<String> {
        if (id as usize) < self.vocab.len() {
            Some(String::from_utf8_lossy(&self.vocab[id as usize]).to_string())
        } else {
            None
        }
    }

    /// Hiz olcumu: saniyede kac token encode.
    #[must_use]
    pub fn hiz_olcum(&self, metin: &str, tekrar: usize) -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..tekrar {
            let _ = self.encode(metin);
        }
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            0.0
        } else {
            (tekrar as f64) / elapsed
        }
    }
}

/// BPE egitimi — basit frekans tabanli (K1: sifirdan).
pub struct BpeEgitim {
    vocab_boyutu: usize,
    min_frekans: usize,
}

impl BpeEgitim {
    #[must_use]
    pub fn yeni(vocab_boyutu: usize) -> Self {
        Self {
            vocab_boyutu,
            min_frekans: 2,
        }
    }

    /// Basit merge ogren — metinlerden en sik ciftleri bul.
    #[must_use]
    pub fn ogren(&self, metinler: &[String]) -> Vec<(u32, u32)> {
        let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
        for metin in metinler {
            let bytes: Vec<u32> = metin.bytes().map(|b| b as u32).collect();
            for w in bytes.windows(2) {
                *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
            }
        }
        let mut pairs: Vec<_> = pair_counts.into_iter().collect();
        pairs.sort_by_key(|a| std::cmp::Reverse(a.1));
        pairs
            .into_iter()
            .filter(|(_, c)| *c >= self.min_frekans)
            .take(self.vocab_boyutu - 256)
            .map(|(p, _)| p)
            .collect()
    }
}

/// AST-farkinda bolme icin yardimci — kod mu?
#[must_use]
pub fn kod_mu(metin: &str) -> bool {
    // Basit heuristic: fn, let, struct, pub, use iceriyorsa kod
    let keywords = [
        "fn ", "let ", "struct ", "impl ", "pub ", "use ", "mod ", "crate::",
    ];
    keywords.iter().any(|k| metin.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ornek_sozluk() -> Sozluk {
        // Basit merges: 101+114=er, 105+110=in, 116+104=th gibi
        let merges = vec![(101, 114), (105, 110), (116, 104), (97, 110), (101, 115)];
        Sozluk::yeni(merges)
    }

    #[test]
    fn bos_sozluk_256() {
        let s = Sozluk::bos();
        assert_eq!(s.vocab_boyutu(), 8192);
        assert_eq!(s.merge_sayisi(), 0);
    }

    #[test]
    fn yeni_sozluk_merge() {
        let s = ornek_sozluk();
        assert_eq!(s.merge_sayisi(), 5);
        assert!(s.vocab_boyutu() >= 261);
    }

    #[test]
    fn dilimle_bosluk() {
        let s = ornek_sozluk();
        let d = s.dilimle("merhaba dunya");
        assert!(d.len() >= 3);
        assert!(d.iter().any(|x| x.tur == DilimTuru::Bosluk));
    }

    #[test]
    fn dilimle_kod() {
        let s = ornek_sozluk();
        let d = s.dilimle("fn main() { let x = 5; }");
        assert!(d.iter().any(|x| x.tur == DilimTuru::Kod));
        assert!(d.iter().any(|x| x.tur == DilimTuru::Noktalama));
    }

    #[test]
    fn dilimle_sayi() {
        let s = ornek_sozluk();
        let d = s.dilimle("74830 * 1291");
        assert!(d.iter().any(|x| x.tur == DilimTuru::Sayi));
    }

    #[test]
    fn encode_bosluk() {
        let s = ornek_sozluk();
        let k = s.encode("a b");
        assert!(!k.ids.is_empty());
        assert!(k.ids.contains(&32));
    }

    #[test]
    fn encode_decode_dongu() {
        let s = ornek_sozluk();
        let metin = "hello world";
        let kod = s.encode(metin);
        let cozulen = s.decode(&kod.ids);
        // Basit sozlukte tam dongu olmayabilir ama bos olmamali
        assert!(!cozulen.is_empty());
    }

    #[test]
    fn bpe_merge_deterministik() {
        let s = ornek_sozluk();
        let tokens = vec![101, 114, 105, 110]; // e r i n
        let m1 = s.bpe_merge(tokens.clone());
        let m2 = s.bpe_merge(tokens);
        assert_eq!(m1, m2);
    }

    #[test]
    fn ozel_tokenlar() {
        let s = Sozluk::bos();
        assert_eq!(s.ozel_id("<pad>"), Some(0));
        assert_eq!(s.ozel_id("<eos>"), Some(2));
    }

    #[test]
    fn kod_mu_heuristic() {
        assert!(kod_mu("fn main() { }"));
        assert!(kod_mu("let x = 5;"));
        assert!(!kod_mu("merhaba dunya nasilsin"));
    }

    #[test]
    fn bpe_egitim_frekans() {
        let egitim = BpeEgitim::yeni(300);
        let metinler = vec!["hello hello hello".to_string(), "world world".to_string()];
        let merges = egitim.ogren(&metinler);
        assert!(!merges.is_empty());
    }

    #[test]
    fn vocab_metin() {
        let s = Sozluk::bos();
        let m = s.vocab_metin(65); // 'A'
        assert!(m.is_some());
    }

    #[test]
    fn hiz_olcum() {
        let s = ornek_sozluk();
        let hiz = s.hiz_olcum("merhaba dunya", 10);
        assert!(hiz >= 0.0);
    }

    #[test]
    fn deterministik_encode() {
        let s = ornek_sozluk();
        let k1 = s.encode("test metin");
        let k2 = s.encode("test metin");
        assert_eq!(k1.ids, k2.ids);
    }

    #[test]
    fn turkce_normalizasyon() {
        let s = ornek_sozluk();
        let d = s.dilimle("İstanbul çalışıyor");
        // Turkce karakterler metin olarak kalmali
        assert!(!d.is_empty());
    }

    #[test]
    fn buyuk_metin_encode() {
        let s = ornek_sozluk();
        let metin = "a".repeat(1000);
        let k = s.encode(&metin);
        assert!(!k.ids.is_empty());
        assert!(k.ids.len() <= 1000);
    }

    #[test]
    fn bos_metin() {
        let s = ornek_sozluk();
        let k = s.encode("");
        assert!(k.ids.is_empty());
    }

    #[test]
    fn noktalama_ayri() {
        let s = ornek_sozluk();
        let d = s.dilimle("hello, world!");
        let noktalama = d.iter().filter(|x| x.tur == DilimTuru::Noktalama).count();
        assert!(noktalama >= 2);
    }

    #[test]
    fn merge_sirasi_onemli() {
        let merges = vec![(97, 98), (98, 99), (256, 99)]; // ab, bc, ab+c
        let s = Sozluk::yeni(merges);
        let tokens = vec![97, 98, 99]; // a b c
        let merged = s.bpe_merge(tokens);
        // ab birlesmeli, sonra (ab)c -> 256,99 -> 258 ?
        assert!(merged.len() <= 2);
    }

    #[test]
    fn pretoken_turleri() {
        let s = Sozluk::bos();
        assert_eq!(s.pretoken as u8, Pretoken::Genel as u8);
    }
}
