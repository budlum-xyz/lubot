//! Defter — karar defteri, sha256 zinciri (kanaat benzeri, ama karar odakli).
//!
//! Her karar bir onceki kararın hash'ini tasir, zincir bozulursa ret.
//! Provenance: asset_id + content_id, licence, attribution.

use sha2::{Digest, Sha256};

/// Bir karar kaydi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kayit {
    pub sira: usize,
    pub soru: String,
    pub hukum: String,
    pub guven: String,
    pub onceki_hash: String,
    pub hash: String,
    pub asset_id: String,
    pub content_id: String,
}

impl Kayit {
    #[must_use]
    pub fn hash_hesapla(soru: &str, hukum: &str, guven: &str, onceki_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(soru.as_bytes());
        hasher.update(b"|");
        hasher.update(hukum.as_bytes());
        hasher.update(b"|");
        hasher.update(guven.as_bytes());
        hasher.update(b"|");
        hasher.update(onceki_hash.as_bytes());
        hex::encode(hasher.finalize())
    }
}

/// Karar defteri.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Defter {
    pub kayitlar: Vec<Kayit>,
}

impl Defter {
    #[must_use]
    pub fn yeni() -> Self {
        Self {
            kayitlar: Vec::new(),
        }
    }

    #[must_use]
    pub fn bos_mu(&self) -> bool {
        self.kayitlar.is_empty()
    }

    #[must_use]
    pub fn son_hash(&self) -> String {
        self.kayitlar
            .last()
            .map(|k| k.hash.clone())
            .unwrap_or_else(|| "0".repeat(64))
    }

    /// Yeni karar ekle, hash zincirini koru.
    pub fn ekle(&mut self, soru: &str, hukum: &str, guven: f64) -> &Kayit {
        let onceki = self.son_hash();
        let guven_str = format!("{guven:.4}");
        let hash = Kayit::hash_hesapla(soru, hukum, &guven_str, &onceki);
        let asset_id = {
            let mut hasher = Sha256::new();
            hasher.update(b"BDLM_LUBOT_CORPUS_ASSET_V1|karar-defteri");
            hex::encode(hasher.finalize())
        };
        let content_id = {
            let mut hasher = Sha256::new();
            hasher.update(soru.as_bytes());
            hasher.update(hukum.as_bytes());
            hex::encode(hasher.finalize())
        };

        let kayit = Kayit {
            sira: self.kayitlar.len(),
            soru: soru.to_string(),
            hukum: hukum.to_string(),
            guven: guven_str,
            onceki_hash: onceki,
            hash,
            asset_id,
            content_id,
        };
        self.kayitlar.push(kayit);
        let idx = self.kayitlar.len() - 1;
        &self.kayitlar[idx]
    }

    /// Zinciri dogrula: her kayit onceki hash'i dogru tasimali.
    pub fn dogrula(&self) -> Result<(), String> {
        let mut beklenen_onceki = "0".repeat(64);
        for kayit in &self.kayitlar {
            if kayit.onceki_hash != beklenen_onceki {
                return Err(format!(
                    "kayit {} onceki hash uyusmazligi: beklenen {}, gelen {}",
                    kayit.sira, beklenen_onceki, kayit.onceki_hash
                ));
            }
            let hesaplanan =
                Kayit::hash_hesapla(&kayit.soru, &kayit.hukum, &kayit.guven, &kayit.onceki_hash);
            if hesaplanan != kayit.hash {
                return Err(format!(
                    "kayit {} hash uyusmazligi: beklenen {}, gelen {}",
                    kayit.sira, hesaplanan, kayit.hash
                ));
            }
            beklenen_onceki = kayit.hash.clone();
        }
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.kayitlar.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kayitlar.is_empty()
    }
}

impl Default for Defter {
    fn default() -> Self {
        Self::yeni()
    }
}

// Hex encode icin minimal, sha2 disinda bagimlilik yok — stdlib ile yap
mod hex {
    #[must_use]
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        let bytes = bytes.as_ref();
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defter_bos_hash() {
        let d = Defter::yeni();
        assert_eq!(d.son_hash(), "0".repeat(64));
    }

    #[test]
    fn defter_ekle_ve_dogrula() {
        let mut d = Defter::yeni();
        d.ekle("soru1", "secim:0", 0.9);
        d.ekle("soru2", "ret", 0.1);
        assert_eq!(d.len(), 2);
        assert!(d.dogrula().is_ok());
    }

    #[test]
    fn defter_zincir_bozulursa_ret() {
        let mut d = Defter::yeni();
        d.ekle("soru1", "secim:0", 0.9);
        d.ekle("soru2", "ret", 0.1);
        // Zinciri boz
        d.kayitlar[1].onceki_hash = "bozuk".to_string();
        assert!(d.dogrula().is_err());
    }

    #[test]
    fn defter_hash_deterministik() {
        let h1 = Kayit::hash_hesapla("soru", "secim:0", "0.9000", &"0".repeat(64));
        let h2 = Kayit::hash_hesapla("soru", "secim:0", "0.9000", &"0".repeat(64));
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }
}
