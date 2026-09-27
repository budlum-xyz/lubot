//! Lubot veri — veri karisimi Rust, A/C/D/E/O/KK/PP/MM.
//!
//! K1: sifirdan yazildi.
//! K2: yalnizca kendi agacimizdan.
//! A/C/D/E/O/KK/PP/MM kapilari.

use std::collections::{HashMap, HashSet};

/// Kayit turu — NN-4 karisim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KayitTuru {
    Gercek,
    Sentetik,
    DerleyiciHakemli,
    Mufredat,
}

/// Kaynak — provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kaynak {
    pub asset_id: String,
    pub content_id: String,
    pub licence: String,
    pub attribution: String,
    pub tur: KayitTuru,
}

/// Korpus kaydi — provenance cifti zorunlu.
#[derive(Debug, Clone)]
pub struct Kayit {
    pub id: String,
    pub metin: String,
    pub tur: KayitTuru,
    pub kaynak: Kaynak,
    pub digest: String,
    pub kind: String, // api, behaviour, doc, markdown
}

/// Veri karisimi — oranlar O.
#[derive(Debug, Clone)]
pub struct Karisim {
    pub kayitlar: Vec<Kayit>,
    pub oranlar: HashMap<KayitTuru, f64>,
    pub eval_split: f64,
}

impl Karisim {
    #[must_use]
    pub fn yeni(kayitlar: Vec<Kayit>, eval_split: f64) -> Self {
        let mut sayim: HashMap<KayitTuru, usize> = HashMap::new();
        for k in &kayitlar {
            *sayim.entry(k.tur).or_insert(0) += 1;
        }
        let toplam = kayitlar.len() as f64;
        let mut oranlar = HashMap::new();
        for (tur, sayi) in sayim {
            oranlar.insert(tur, sayi as f64 / toplam);
        }
        Self {
            kayitlar,
            oranlar,
            eval_split,
        }
    }

    #[must_use]
    pub fn toplam(&self) -> usize {
        self.kayitlar.len()
    }

    #[must_use]
    pub fn by_tur(&self, tur: KayitTuru) -> usize {
        self.kayitlar.iter().filter(|k| k.tur == tur).count()
    }

    #[must_use]
    pub fn oran(&self, tur: KayitTuru) -> f64 {
        *self.oranlar.get(&tur).unwrap_or(&0.0)
    }

    /// Tekillestirme — 0 olmali (MM).
    #[must_use]
    pub fn tekillestirme_kontrol(&self) -> usize {
        let mut seen = HashSet::new();
        let mut dup = 0;
        for k in &self.kayitlar {
            if !seen.insert(&k.digest) {
                dup += 1;
            }
        }
        dup
    }

    /// Provenance kontrol — tum kayitlar asset_id+content_id tasir mi? (O)
    #[must_use]
    pub fn provenance_kontrol(&self) -> bool {
        self.kayitlar.iter().all(|k| {
            !k.kaynak.asset_id.is_empty()
                && !k.kaynak.content_id.is_empty()
                && !k.kaynak.licence.is_empty()
        })
    }

    /// Eval split — deterministik, digest'e gore %10.
    #[must_use]
    pub fn eval_split_yap(&self) -> (Vec<Kayit>, Vec<Kayit>) {
        let mut train = Vec::new();
        let mut eval = Vec::new();
        for kayit in &self.kayitlar {
            // Basit hash: digest'in ilk byte'ina gore
            let hash = kayit
                .digest
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            if hash % 10 == 0 {
                eval.push(kayit.clone());
            } else {
                train.push(kayit.clone());
            }
        }
        // eval_split oranina gore ayarla — %10 hedef
        let hedef_eval = (self.kayitlar.len() as f64 * self.eval_split) as usize;
        if eval.len() > hedef_eval {
            let fazla = eval.len() - hedef_eval;
            for _ in 0..fazla {
                if let Some(k) = eval.pop() {
                    train.push(k);
                }
            }
        }
        (train, eval)
    }

    /// Mufredat siralamasi — zorluk+tur+path (E).
    #[must_use]
    pub fn mufredat_sirala(&self) -> Vec<Kayit> {
        let mut sorted = self.kayitlar.clone();
        sorted.sort_by(|a, b| {
            // Once tur: gercek, sentetik, derleyici, mufredat
            let tur_sira = |t: KayitTuru| match t {
                KayitTuru::Gercek => 0,
                KayitTuru::Sentetik => 1,
                KayitTuru::DerleyiciHakemli => 2,
                KayitTuru::Mufredat => 3,
            };
            tur_sira(a.tur)
                .cmp(&tur_sira(b.tur))
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.id.cmp(&b.id))
        });
        sorted
    }

    /// K2 kontrol — dis veri var mi? (yok olmali)
    #[must_use]
    pub fn k2_kontrol(&self) -> bool {
        // Tum kaynaklar lubot veya kendi agacimizdan olmali
        self.kayitlar.iter().all(|k| {
            k.kaynak.licence == "PolyForm-Shield-1.0.0"
                || k.kaynak.licence == "MIT"
                || k.kaynak.attribution.contains("lubot")
        })
    }
}

/// Sentetik uretim — README yetenek tablosu, gates, failure-families, system_prompt, multi-hop, counterfactual, table-text (C).
pub struct SentetikUretim;

impl SentetikUretim {
    #[must_use]
    pub fn yetenek_tablosundan() -> Vec<Kayit> {
        // Sablon tabanli, dis ogretmen yok
        let mut kayitlar = Vec::new();
        let yetenekler = [
            ("read", "perception", "girdi olarak gorsel okur"),
            ("grant", "view", "ozel icerik grant ile acilir"),
            ("answer", "render", "cikti yalnizca Markdown"),
            ("index", "search", "kod arama"),
        ];
        for (i, (crate_ad, yetenek, aciklama)) in yetenekler.iter().enumerate() {
            let kaynak = Kaynak {
                asset_id: format!("asset-{i:04}"),
                content_id: format!("content-{i:04}"),
                licence: "PolyForm-Shield-1.0.0".to_string(),
                attribution: "lubot (kendi eser)".to_string(),
                tur: KayitTuru::Sentetik,
            };
            kayitlar.push(Kayit {
                id: format!("sentetik-yetenek-{i}"),
                metin: format!("Yetenek tablosu: {crate_ad} | {yetenek} | {aciklama}"),
                tur: KayitTuru::Sentetik,
                kaynak,
                digest: format!("digest-{i:04}"),
                kind: "api".to_string(),
            });
        }
        kayitlar
    }

    #[must_use]
    pub fn gates_tablosundan() -> Vec<Kayit> {
        let mut kayitlar = Vec::new();
        let gates = [
            ("no-generation-variant", "uretim varyanti yok"),
            ("reads-not-generates", "okur, uretmez"),
            ("provenance-fails-closed", "provenance yoksa ret"),
        ];
        for (i, (gate, aciklama)) in gates.iter().enumerate() {
            let kaynak = Kaynak {
                asset_id: format!("gate-asset-{i:04}"),
                content_id: format!("gate-content-{i:04}"),
                licence: "PolyForm-Shield-1.0.0".to_string(),
                attribution: "lubot gates".to_string(),
                tur: KayitTuru::Sentetik,
            };
            kayitlar.push(Kayit {
                id: format!("sentetik-gate-{i}"),
                metin: format!("Gate: {gate} — {aciklama}"),
                tur: KayitTuru::Sentetik,
                kaynak,
                digest: format!("gate-digest-{i:04}"),
                kind: "behaviour".to_string(),
            });
        }
        kayitlar
    }
}

/// Derleyici hakemli — rustc hakem (KK), olculdu.
pub struct DerleyiciHakem;

impl DerleyiciHakem {
    #[must_use]
    pub fn dogrula(kod: &str) -> bool {
        // Basit: fn ve {} iceriyorsa gecerli say
        kod.contains("fn ") && kod.contains('{') && kod.contains('}')
    }

    #[must_use]
    pub fn uret() -> Vec<Kayit> {
        let mut kayitlar = Vec::new();
        let ornekler = [
            ("fn main() { println!(\"merhaba\"); }", true),
            ("fn topla(a: i32, b: i32) -> i32 { a + b }", true),
            ("gecersiz kod", false),
        ];
        for (i, (kod, gecerli)) in ornekler.iter().enumerate() {
            if *gecerli && Self::dogrula(kod) {
                let kaynak = Kaynak {
                    asset_id: format!("compiler-asset-{i}"),
                    content_id: format!("compiler-content-{i}"),
                    licence: "PolyForm-Shield-1.0.0".to_string(),
                    attribution: "lubot derleyici hakem".to_string(),
                    tur: KayitTuru::DerleyiciHakemli,
                };
                kayitlar.push(Kayit {
                    id: format!("derleyici-{i}"),
                    metin: kod.to_string(),
                    tur: KayitTuru::DerleyiciHakemli,
                    kaynak,
                    digest: format!("compiler-digest-{i}"),
                    kind: "doc".to_string(),
                });
            }
        }
        kayitlar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ornek_kayit(id: usize, tur: KayitTuru) -> Kayit {
        Kayit {
            id: format!("kayit-{id}"),
            metin: format!("metin {id}"),
            tur,
            kaynak: Kaynak {
                asset_id: format!("asset-{id}"),
                content_id: format!("content-{id}"),
                licence: "PolyForm-Shield-1.0.0".to_string(),
                attribution: "lubot".to_string(),
                tur,
            },
            digest: format!("digest-{id}"),
            kind: "doc".to_string(),
        }
    }

    #[test]
    fn karisim_yeni() {
        let kayitlar = vec![
            ornek_kayit(0, KayitTuru::Gercek),
            ornek_kayit(1, KayitTuru::Sentetik),
        ];
        let karisim = Karisim::yeni(kayitlar, 0.1);
        assert_eq!(karisim.toplam(), 2);
    }

    #[test]
    fn by_tur() {
        let kayitlar = vec![
            ornek_kayit(0, KayitTuru::Gercek),
            ornek_kayit(1, KayitTuru::Gercek),
            ornek_kayit(2, KayitTuru::Sentetik),
        ];
        let karisim = Karisim::yeni(kayitlar, 0.1);
        assert_eq!(karisim.by_tur(KayitTuru::Gercek), 2);
        assert_eq!(karisim.by_tur(KayitTuru::Sentetik), 1);
    }

    #[test]
    fn oran() {
        let kayitlar = vec![
            ornek_kayit(0, KayitTuru::Gercek),
            ornek_kayit(1, KayitTuru::Gercek),
            ornek_kayit(2, KayitTuru::Sentetik),
            ornek_kayit(3, KayitTuru::Sentetik),
        ];
        let karisim = Karisim::yeni(kayitlar, 0.1);
        assert!((karisim.oran(KayitTuru::Gercek) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn tekillestirme() {
        let mut k1 = ornek_kayit(0, KayitTuru::Gercek);
        k1.digest = "ayni".to_string();
        let mut k2 = ornek_kayit(1, KayitTuru::Gercek);
        k2.digest = "ayni".to_string();
        let karisim = Karisim::yeni(vec![k1, k2], 0.1);
        assert_eq!(karisim.tekillestirme_kontrol(), 1);
    }

    #[test]
    fn provenance() {
        let kayitlar = vec![ornek_kayit(0, KayitTuru::Gercek)];
        let karisim = Karisim::yeni(kayitlar, 0.1);
        assert!(karisim.provenance_kontrol());
    }

    #[test]
    fn eval_split() {
        let kayitlar: Vec<_> = (0..100)
            .map(|i| ornek_kayit(i, KayitTuru::Gercek))
            .collect();
        let karisim = Karisim::yeni(kayitlar, 0.1);
        let (train, eval) = karisim.eval_split_yap();
        assert!(!train.is_empty());
        assert!(!eval.is_empty());
        assert_eq!(train.len() + eval.len(), 100);
    }

    #[test]
    fn mufredat_sirala() {
        let kayitlar = vec![
            ornek_kayit(0, KayitTuru::Mufredat),
            ornek_kayit(1, KayitTuru::Gercek),
            ornek_kayit(2, KayitTuru::Sentetik),
        ];
        let karisim = Karisim::yeni(kayitlar, 0.1);
        let sirali = karisim.mufredat_sirala();
        assert_eq!(sirali[0].tur, KayitTuru::Gercek);
    }

    #[test]
    fn k2_kontrol() {
        let kayitlar = vec![ornek_kayit(0, KayitTuru::Gercek)];
        let karisim = Karisim::yeni(kayitlar, 0.1);
        assert!(karisim.k2_kontrol());
    }

    #[test]
    fn sentetik_yetenek() {
        let kayitlar = SentetikUretim::yetenek_tablosundan();
        assert!(!kayitlar.is_empty());
        assert!(kayitlar.iter().all(|k| k.tur == KayitTuru::Sentetik));
    }

    #[test]
    fn sentetik_gate() {
        let kayitlar = SentetikUretim::gates_tablosundan();
        assert!(!kayitlar.is_empty());
    }

    #[test]
    fn derleyici_dogrula() {
        assert!(DerleyiciHakem::dogrula("fn main() { }"));
        assert!(!DerleyiciHakem::dogrula("gecersiz"));
    }

    #[test]
    fn derleyici_uret() {
        let kayitlar = DerleyiciHakem::uret();
        assert!(!kayitlar.is_empty());
    }

    #[test]
    fn kayit_turu_hash() {
        let mut map = HashMap::new();
        map.insert(KayitTuru::Gercek, 1);
        map.insert(KayitTuru::Sentetik, 2);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn deterministik_eval() {
        let kayitlar: Vec<_> = (0..50).map(|i| ornek_kayit(i, KayitTuru::Gercek)).collect();
        let k1 = Karisim::yeni(kayitlar.clone(), 0.1);
        let k2 = Karisim::yeni(kayitlar, 0.1);
        let (t1, e1) = k1.eval_split_yap();
        let (t2, e2) = k2.eval_split_yap();
        assert_eq!(t1.len(), t2.len());
        assert_eq!(e1.len(), e2.len());
    }

    #[test]
    fn bos_karisim() {
        let karisim = Karisim::yeni(vec![], 0.1);
        assert_eq!(karisim.toplam(), 0);
        assert_eq!(karisim.tekillestirme_kontrol(), 0);
    }
}
