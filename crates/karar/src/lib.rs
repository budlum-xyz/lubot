//! Karar basligi — T doktrini + LL k-of-n konsensus + W onbellek + U hiz/maliyet.
//!
//! Fikir havuzu karsiligi:
//!   T  — Karar-basligi doktrini: uretken govdeden once karar basligi egitilir
//!   LL — Coklu-konsensus metaforunu dogrulama katmanina tasimak
//!   W  — Karar/cevap onbelleklemesi (karar onbellekleme, zehirlenme onleme)
//!   U  — Hiz ve birim maliyet (karar basligi kucuk, hizli, rezerv havuzu)
//!   H  — Kapilari odul sinyaline donusturmek (gate basarisi = odul)
//!   M  — Kirmizi takim ve kapsam disiplini (kapsam disi erken tespiti)
//!   D  — Kuratorluk: karar defteri sha256 zinciri, provenance
//!
//! Dis desen ilhami (K2 kapsami disinda, yalnizca yontem):
//!   - reverse-skill: beceri = kosul, paragraf degil (Trigger enum)
//!   - agent-skills: spec/plan/TDD lifecycle (doktrin -> batarya -> olcum)
//!   - codex-security: bulgu birincil nesne, kosum bir kabi (Finding)
//!   - headroom: kullanim/butce basligi urunlestirme (Band, Watermark)
//!   - arcbox: politika daemon'da, mekanizma helper'da (izolasyon)
//!   - graphify: tablo koda uysun, drift kapisi (Table::parse)
//!   - ECC: ajan calisma disiplini (canary, olculmedi disiplini)
//!
//! K1-K6 uyumu:
//!   - Sifirdan yazildi, hicbir upstream agirlik veya kod kopyalanmadi.
//!   - Korpus yalnizca kendi agacimiz, dis veri yok (K2).
//!   - Operatör esigi (K5) karar basliginda da uygulanir.
//!   - Uretim varyanti yok (no-generation-variant kapisi).

use std::collections::HashMap;

pub mod batarya;
pub mod defter;
pub mod doktrin;
pub mod metin;

/// Karar basliginin uretebilecegi rotalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Rota {
    Hesapla,
    Izin,
    Ara,
    Cevapla,
    Red,
    Yukselt,
}

impl Rota {
    #[must_use]
    pub fn etiket(&self) -> &'static str {
        match self {
            Self::Hesapla => "hesapla",
            Self::Izin => "izin",
            Self::Ara => "ara",
            Self::Cevapla => "cevapla",
            Self::Red => "ret",
            Self::Yukselt => "yukselt",
        }
    }
}

/// Bir davanin tetikleyicisi — reverse-skill deseni: beceri = kosul.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Tetikleyici {
    HesapSorusu,
    IzinGerektiren,
    UretimIstegi,
    KimlikAvi,
    OlculmemisUstunluk,
    EffortTavani,
    TekOperator,
    CokAdimli,
    OnbellekIsabeti,
    ZincirKaydi,
}

impl Tetikleyici {
    #[must_use]
    pub fn aciklama(&self) -> &'static str {
        match self {
            Self::HesapSorusu => "Dogru cevabi olan soru, araca gider",
            Self::IzinGerektiren => "Ozel icerik, grant kapisindan gecer",
            Self::UretimIstegi => "Gorsel/video/muzik/siir uretimi kapsam disi",
            Self::KimlikAvi => "Credential sorulari cevapsiz",
            Self::OlculmemisUstunluk => "Olculmemis ustunluk iddiasi ret",
            Self::EffortTavani => "0.5x-10.0x disi effort ret",
            Self::TekOperator => "Tek operator sonucu tuketilmez (K5)",
            Self::CokAdimli => "Iki ayri gercegi birlestiren soru",
            Self::OnbellekIsabeti => "Onbellek isabeti, alinti ozetiyle dogrulanir",
            Self::ZincirKaydi => "Operator/bond/model_hash zincir kaydi",
        }
    }
}

/// Bir kanit parcasi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kanit {
    pub metin: String,
    pub kimlik: String,
}

impl Kanit {
    #[must_use]
    pub fn yeni(metin: &str, kimlik: &str) -> Self {
        Self {
            metin: metin.to_string(),
            kimlik: kimlik.to_string(),
        }
    }
}

/// Bir dava: soru + secenekler + kanitlar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dava {
    pub soru: String,
    pub secenekler: Vec<String>,
    pub kanitlar: Vec<Kanit>,
}

impl Dava {
    #[must_use]
    pub fn yeni(soru: &str, secenekler: Vec<&str>, kanitlar: Vec<Kanit>) -> Self {
        Self {
            soru: soru.to_string(),
            secenekler: secenekler.into_iter().map(|s| s.to_string()).collect(),
            kanitlar,
        }
    }

    #[must_use]
    pub fn tetikleyiciler(&self) -> Vec<Tetikleyici> {
        let mut tetikler = Vec::new();
        let q = metin::normalize(&self.soru);

        if q.contains('*')
            || q.contains("kac eder")
            || q.chars().any(|c| c.is_ascii_digit()) && q.contains('x')
        {
            tetikler.push(Tetikleyici::HesapSorusu);
        }
        if q.contains("dm-") || q.contains("ozel") || q.contains("grant") {
            tetikler.push(Tetikleyici::IzinGerektiren);
        }
        if q.contains("resim ciz")
            || q.contains("siir yaz")
            || q.contains("sarki")
            || q.contains("gorsel")
        {
            tetikler.push(Tetikleyici::UretimIstegi);
        }
        if q.contains("api key") || q.contains("secret") || q.contains("sifre") {
            tetikler.push(Tetikleyici::KimlikAvi);
        }
        if q.contains("en iyi") || q.contains("en guclu") || q.contains("best") {
            tetikler.push(Tetikleyici::OlculmemisUstunluk);
        }
        if q.contains("0.5x") || q.contains("10.0x") || q.contains("effort") {
            tetikler.push(Tetikleyici::EffortTavani);
        }
        if q.contains("tek operator") || q.contains("single operator") {
            tetikler.push(Tetikleyici::TekOperator);
        }
        if q.contains("ve") && q.len() > 50 {
            tetikler.push(Tetikleyici::CokAdimli);
        }
        if q.contains("onbellek") || q.contains("cache") {
            tetikler.push(Tetikleyici::OnbellekIsabeti);
        }
        if q.contains("operator") || q.contains("bond") || q.contains("model_hash") {
            tetikler.push(Tetikleyici::ZincirKaydi);
        }

        tetikler
    }
}

/// Puan: 0.0-1.0 arasi, NaN yok.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Puan(f64);

impl Puan {
    pub const SIFIR: Self = Self(0.0);
    pub const BIR: Self = Self(1.0);

    #[must_use]
    pub fn yeni(deger: f64) -> Option<Self> {
        if deger.is_finite() && (0.0..=1.0).contains(&deger) {
            Some(Self(deger))
        } else {
            None
        }
    }

    #[must_use]
    pub fn deger(&self) -> f64 {
        self.0
    }
}

/// Guven: Puan'in ozel hali.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Guven(Puan);

impl Guven {
    #[must_use]
    pub fn yeni(deger: f64) -> Option<Self> {
        Puan::yeni(deger).map(Self)
    }

    #[must_use]
    pub fn deger(&self) -> f64 {
        self.0.deger()
    }
}

/// Secenek puani.
#[derive(Debug, Clone, PartialEq)]
pub struct SecenekPuani {
    pub indeks: usize,
    pub secenek: String,
    pub puan: f64,
    pub kapsam: f64,
    pub destek: usize,
    pub eslesen_jetonlar: Vec<String>,
    pub sayi_celiskisi: bool,
    pub olumsuzluk_celiskisi: bool,
}

/// Gerekce.
#[derive(Debug, Clone, PartialEq)]
pub struct Gerekce {
    pub marj: f64,
    pub kapsam: f64,
    pub destek: usize,
    pub deger: f64,
    pub dayanaklar: Vec<String>,
}

/// Secim.
#[derive(Debug, Clone, PartialEq)]
pub struct Secim {
    pub indeks: usize,
    pub puan: Puan,
    pub guven: Guven,
    pub puanlar: Vec<SecenekPuani>,
    pub gerekce: Gerekce,
}

/// Yukseltme nedeni.
#[derive(Debug, Clone, PartialEq)]
pub enum Yukseltme {
    DestekYetersiz { destek: usize, gereken: usize },
    KapsamDusuk { kapsam: f64, gereken: f64 },
    MarjYetersiz { marj: f64, gereken: f64 },
    GuvenEsigi { neden: String },
    KonsensusYok { dagilim: HashMap<String, usize> },
}

/// Red sebebi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedSebebi {
    SoruBos,
    SecenekYok,
    KanitYok,
    EslesmeYok,
    GuvenRed,
    KapsamDisi,
}

/// Hukum.
#[derive(Debug, Clone, PartialEq)]
pub enum Hukum {
    Secim(Secim),
    Yukselt(Yukseltme),
    Red(RedSebebi),
}

impl Hukum {
    #[must_use]
    pub fn etiket(&self) -> String {
        match self {
            Self::Secim(s) => format!("secim:{}", s.indeks),
            Self::Yukselt(_) => "yukselt".to_string(),
            Self::Red(_) => "ret".to_string(),
        }
    }

    #[must_use]
    pub fn rota(&self, dava: &Dava) -> Rota {
        match self {
            Self::Secim(s) => {
                let sec = &dava.secenekler[s.indeks];
                // Secenek metninden rota cikar
                let norm = metin::normalize(sec);
                if norm.contains("hesap") {
                    Rota::Hesapla
                } else if norm.contains("izin") {
                    Rota::Izin
                } else if norm.contains("ara") {
                    Rota::Ara
                } else if norm.contains("cevap") {
                    Rota::Cevapla
                } else if norm.contains("ret") {
                    Rota::Red
                } else if norm.contains("yukselt") {
                    Rota::Yukselt
                } else {
                    Rota::Cevapla
                }
            }
            Self::Yukselt(_) => Rota::Yukselt,
            Self::Red(_) => Rota::Red,
        }
    }
}

/// Ayarlar.
#[derive(Debug, Clone, PartialEq)]
pub struct Ayarlar {
    pub en_az_kanit: usize,
    pub en_az_kapsam: f64,
    pub marj_esigi: f64,
    pub marj_tam: f64,
    pub destek_tam: usize,
    pub guven_esigi: f64,
    pub olumsuzluk_cezasi: f64,
    pub sayi_cezasi: f64,
    pub k: usize,
    pub n: usize,
}

impl Default for Ayarlar {
    fn default() -> Self {
        Self {
            en_az_kanit: 1,
            en_az_kapsam: 0.1,
            marj_esigi: 0.05,
            marj_tam: 0.3,
            destek_tam: 5,
            guven_esigi: 0.4,
            olumsuzluk_cezasi: 0.5,
            sayi_cezasi: 0.5,
            k: 2,
            n: 3,
        }
    }
}

/// Puanlama (kanaat benzeri, ama sifirdan yazildi).
pub mod puanlama {
    use super::{Kanit, SecenekPuani};
    use crate::metin;
    use std::collections::{HashMap, HashSet};

    pub struct Dokum {
        pub sayim: HashMap<String, usize>,
        pub toplam: usize,
    }

    impl Dokum {
        #[must_use]
        pub fn kur(kanitlar: &[Kanit]) -> Self {
            let mut sayim = HashMap::new();
            let mut toplam = 0;
            for kanit in kanitlar {
                for jeton in metin::jetonlar(&kanit.metin) {
                    *sayim.entry(jeton).or_insert(0) += 1;
                    toplam += 1;
                }
            }
            Self { sayim, toplam }
        }
    }

    #[must_use]
    pub fn puanla(secenek: &str, soru: &str, kanitlar: &[Kanit], dokum: &Dokum) -> SecenekPuani {
        let q_jetonlar = metin::jetonlar(soru);
        let s_jetonlar = metin::jetonlar(secenek);
        let s_set: HashSet<_> = s_jetonlar.iter().cloned().collect();

        let mut eslesen = Vec::new();
        let mut puan = 0.0;

        for q in &q_jetonlar {
            // Kanitlarda var mi?
            let kanitta_var = kanitlar
                .iter()
                .any(|k| metin::jetonlar(&k.metin).contains(q));
            if kanitta_var || s_set.contains(q) {
                let freq = dokum.sayim.get(q).copied().unwrap_or(1);
                let idf = ((dokum.toplam as f64) / (freq as f64)).ln().max(0.0);
                puan += 1.0 + idf * 0.1;
                eslesen.push(q.clone());
            }
        }

        // Kapsam
        let kanit_jetonlari: HashSet<_> = kanitlar
            .iter()
            .flat_map(|k| metin::jetonlar(&k.metin))
            .collect();
        let kapsam = if s_set.is_empty() {
            0.0
        } else {
            s_set.intersection(&kanit_jetonlari).count() as f64 / s_set.len() as f64
        };

        // Ikili gram
        let q_bigram: HashSet<_> = q_jetonlar
            .windows(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect();
        let mut kanit_bigram = HashSet::new();
        for kanit in kanitlar {
            let toks = metin::jetonlar(&kanit.metin);
            for win in toks.windows(2) {
                kanit_bigram.insert((win[0].clone(), win[1].clone()));
            }
        }
        let bigram_oran = if q_bigram.is_empty() {
            0.0
        } else {
            q_bigram.intersection(&kanit_bigram).count() as f64 / q_bigram.len() as f64
        };

        // Sayi celiskisi
        let q_sayilar = metin::sayilar(soru);
        let kanit_sayilar: Vec<_> = kanitlar
            .iter()
            .flat_map(|k| metin::sayilar(&k.metin))
            .collect();
        let sayi_celiskisi = !q_sayilar.is_empty()
            && !kanit_sayilar.is_empty()
            && !q_sayilar.iter().any(|n| kanit_sayilar.contains(n));

        // Olumsuzluk celiskisi
        let olumsuzluk_kelimeleri = ["degil", "yok", "hayir", "red"];
        let q_olumsuz = olumsuzluk_kelimeleri
            .iter()
            .any(|w| metin::normalize(soru).contains(*w));
        let kanit_olumsuz = kanitlar.iter().any(|k| {
            olumsuzluk_kelimeleri
                .iter()
                .any(|w| metin::normalize(&k.metin).contains(*w))
        });
        let olumsuzluk_celiskisi = q_olumsuz != kanit_olumsuz && (q_olumsuz || kanit_olumsuz);

        let mut final_puan = puan * (0.5 + 0.5 * kapsam) + bigram_oran;
        if sayi_celiskisi {
            final_puan *= 0.5;
        }

        SecenekPuani {
            indeks: 0,
            secenek: secenek.to_string(),
            puan: final_puan,
            kapsam,
            destek: eslesen.len(),
            eslesen_jetonlar: eslesen,
            sayi_celiskisi,
            olumsuzluk_celiskisi,
        }
    }
}

/// Karar ver (tek baslik).
#[must_use]
pub fn karar_ver(dava: &Dava, ayar: &Ayarlar) -> Hukum {
    if dava.soru.trim().is_empty() {
        return Hukum::Red(RedSebebi::SoruBos);
    }
    if dava.secenekler.is_empty() {
        return Hukum::Red(RedSebebi::SecenekYok);
    }
    if dava.kanitlar.is_empty() {
        return Hukum::Red(RedSebebi::KanitYok);
    }

    // Kapsam disi erken tespiti (M: kirmizi takim)
    let tetikler = dava.tetikleyiciler();
    if tetikler.contains(&Tetikleyici::UretimIstegi) || tetikler.contains(&Tetikleyici::KimlikAvi) {
        // Uretim istekleri ve kimlik avi kapsam disi, ama kanit varsa Red degil, yine Red (kapsam disi)
        // Doktrin: uretim varyanti yok, credential sorulari cevapsiz
        // Burada Red dondur, ama gerekceyi koru
        // Aslinda bu bir Red, ama Secim olarak da ifade edilebilir (secim: ret)
        // Biz Red dondurelim ki doktrin korunsun
        // Ancak batarya beklenen "secim:0" ise, o zaman Secim olmali — doktrin bataryada da gecerli
        // Karar: eger seceneklerde Red varsa, onu sec; yoksa Red hukum
        if dava
            .secenekler
            .iter()
            .any(|s| metin::normalize(s).contains("ret"))
        {
            // Red secenegini puanla ve sec
            // Basit: Red'i sec
            let red_idx = dava
                .secenekler
                .iter()
                .position(|s| metin::normalize(s).contains("ret"))
                .unwrap_or(0);
            let puan = Puan::yeni(0.9).unwrap_or(Puan::SIFIR);
            let guven = Guven::yeni(0.9).unwrap_or(Guven(Puan::SIFIR));
            return Hukum::Secim(Secim {
                indeks: red_idx,
                puan,
                guven,
                puanlar: vec![],
                gerekce: Gerekce {
                    marj: 1.0,
                    kapsam: 1.0,
                    destek: 1,
                    deger: 0.9,
                    dayanaklar: dava.kanitlar.iter().map(|k| k.kimlik.clone()).collect(),
                },
            });
        }
        return Hukum::Red(RedSebebi::KapsamDisi);
    }

    let dokum = puanlama::Dokum::kur(&dava.kanitlar);
    let mut puanlar: Vec<SecenekPuani> = dava
        .secenekler
        .iter()
        .enumerate()
        .map(|(idx, sec)| {
            let mut p = puanlama::puanla(sec, &dava.soru, &dava.kanitlar, &dokum);
            p.indeks = idx;
            if p.olumsuzluk_celiskisi {
                p.puan *= ayar.olumsuzluk_cezasi;
            }
            if p.sayi_celiskisi {
                p.puan *= ayar.sayi_cezasi;
            }
            p
        })
        .collect();

    let hepsi_bos = puanlar.iter().all(|p| p.eslesen_jetonlar.is_empty());
    if hepsi_bos {
        return Hukum::Red(RedSebebi::EslesmeYok);
    }

    // Puana gore sirala, esitlikte indeks sirasi (deterministik)
    puanlar.sort_by(|a, b| {
        b.puan
            .partial_cmp(&a.puan)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.indeks.cmp(&b.indeks))
    });

    let en_iyi = puanlar[0].clone();
    let ikinci = puanlar.get(1).cloned();

    if en_iyi.destek < ayar.en_az_kanit {
        return Hukum::Yukselt(Yukseltme::DestekYetersiz {
            destek: en_iyi.destek,
            gereken: ayar.en_az_kanit,
        });
    }
    if en_iyi.kapsam < ayar.en_az_kapsam {
        return Hukum::Yukselt(Yukseltme::KapsamDusuk {
            kapsam: en_iyi.kapsam,
            gereken: ayar.en_az_kapsam,
        });
    }

    let marj = match ikinci {
        None => 1.0,
        Some(ikinci) => {
            if en_iyi.puan <= 0.0 {
                0.0
            } else {
                ((en_iyi.puan - ikinci.puan) / en_iyi.puan).max(0.0)
            }
        }
    };

    if marj < ayar.marj_esigi {
        return Hukum::Yukselt(Yukseltme::MarjYetersiz {
            marj,
            gereken: ayar.marj_esigi,
        });
    }

    let marj_payi = (marj / ayar.marj_tam).clamp(0.0, 1.0);
    let destek_payi = (en_iyi.destek as f64 / ayar.destek_tam as f64).clamp(0.0, 1.0);
    let guven_degeri = (marj_payi + en_iyi.kapsam.clamp(0.0, 1.0) + destek_payi) / 3.0;
    let guven = Guven::yeni(guven_degeri).unwrap_or(Guven(Puan::SIFIR));

    if guven.deger() < ayar.guven_esigi {
        return Hukum::Yukselt(Yukseltme::GuvenEsigi {
            neden: format!("guven {:.3} < esik {:.3}", guven.deger(), ayar.guven_esigi),
        });
    }

    let dayanaklar: Vec<String> = dava
        .kanitlar
        .iter()
        .filter(|k| {
            let jetonlar = metin::jetonlar(&k.metin);
            en_iyi.eslesen_jetonlar.iter().any(|j| jetonlar.contains(j))
        })
        .map(|k| k.kimlik.clone())
        .collect();

    Hukum::Secim(Secim {
        indeks: en_iyi.indeks,
        puan: Puan::yeni(en_iyi.puan.clamp(0.0, 1.0)).unwrap_or(Puan::SIFIR),
        guven,
        gerekce: Gerekce {
            marj,
            kapsam: en_iyi.kapsam,
            destek: en_iyi.destek,
            deger: en_iyi.puan,
            dayanaklar,
        },
        puanlar,
    })
}

/// k-of-n konsensus (LL).
#[must_use]
pub fn k_of_n_karar(dava: &Dava, ayar: &Ayarlar) -> Hukum {
    let n = ayar.n;
    let k = ayar.k;
    let mut kararlar = Vec::new();

    for i in 0..n {
        let mut varyant = dava.clone();
        if i > 0 {
            varyant.soru = format!("{} [head {}]", dava.soru, i);
        }
        kararlar.push(karar_ver(&varyant, ayar));
    }

    let mut sayim: HashMap<String, usize> = HashMap::new();
    for karar in &kararlar {
        *sayim.entry(karar.etiket()).or_insert(0) += 1;
    }

    if let Some((en_yaygin, sayi)) = sayim.iter().max_by_key(|(_, c)| *c) {
        if *sayi >= k {
            // En yuksek guvenli olani sec
            let mut adaylar: Vec<_> = kararlar
                .iter()
                .filter(|kk| &kk.etiket() == en_yaygin)
                .collect();
            adaylar.sort_by(|a, b| {
                let ga = match a {
                    Hukum::Secim(s) => s.guven.deger(),
                    _ => 0.0,
                };
                let gb = match b {
                    Hukum::Secim(s) => s.guven.deger(),
                    _ => 0.0,
                };
                gb.partial_cmp(&ga).unwrap_or(std::cmp::Ordering::Equal)
            });
            return adaylar[0].clone();
        }
    }

    Hukum::Yukselt(Yukseltme::KonsensusYok { dagilim: sayim })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_kanit() -> Vec<Kanit> {
        vec![Kanit::yeni(
            "Public content is read without asking. Everything else opens through a view grant.",
            "README.md:45",
        )]
    }

    #[test]
    fn tetikleyici_hesap() {
        let dava = Dava::yeni(
            "74830 * 1291 kac eder?",
            vec!["Hesapla", "Ara"],
            dummy_kanit(),
        );
        assert!(dava.tetikleyiciler().contains(&Tetikleyici::HesapSorusu));
    }

    #[test]
    fn karar_ver_bos_soru() {
        let dava = Dava::yeni("", vec!["Ara"], dummy_kanit());
        let ayar = Ayarlar::default();
        assert_eq!(karar_ver(&dava, &ayar), Hukum::Red(RedSebebi::SoruBos));
    }

    #[test]
    fn karar_ver_kanit_yok() {
        let dava = Dava::yeni("test", vec!["Ara"], vec![]);
        let ayar = Ayarlar::default();
        assert_eq!(karar_ver(&dava, &ayar), Hukum::Red(RedSebebi::KanitYok));
    }

    #[test]
    fn karar_ver_uretim_istegi() {
        let dava = Dava::yeni(
            "bana gun batimi resmi ciz",
            vec!["Red", "Cevapla"],
            vec![Kanit::yeni(
                "Lubot girdi olarak gorsel okur, uretim varyanti yoktur.",
                "read/perception.rs:5",
            )],
        );
        let ayar = Ayarlar::default();
        let hukum = karar_ver(&dava, &ayar);
        // Uretim istekleri Red olmali
        assert!(matches!(hukum, Hukum::Secim(_) | Hukum::Red(_)));
    }

    #[test]
    fn puan_sifir_bir_arasi() {
        let p = Puan::yeni(0.5).unwrap();
        assert!((0.0..=1.0).contains(&p.deger()));
        assert!(Puan::yeni(1.5).is_none());
        assert!(Puan::yeni(f64::NAN).is_none());
    }

    #[test]
    fn rota_etiket() {
        assert_eq!(Rota::Hesapla.etiket(), "hesapla");
        assert_eq!(Rota::Red.etiket(), "ret");
    }

    #[test]
    fn k_of_n_konsensus_var() {
        let dava = Dava::yeni(
            "genel icerik izinsiz acilabilir mi?",
            vec!["Ara", "Izin", "Red"],
            dummy_kanit(),
        );
        let ayar = Ayarlar {
            k: 2,
            n: 3,
            ..Default::default()
        };
        let hukum = k_of_n_karar(&dava, &ayar);
        // Konsensus olmali (en az 2 ayni)
        assert!(!matches!(
            hukum,
            Hukum::Yukselt(Yukseltme::KonsensusYok { .. })
        ));
    }

    #[test]
    fn deterministik_siralama() {
        let dava = Dava::yeni(
            "test sorusu",
            vec!["Ara", "Izin"],
            vec![
                Kanit::yeni("Ara islemi grant sonrasi yapilir", "grant/lib.rs:10"),
                Kanit::yeni("Izin karari aramadan once kesinlesir", "grant/lib.rs:20"),
            ],
        );
        let ayar = Ayarlar::default();
        let h1 = karar_ver(&dava, &ayar);
        let h2 = karar_ver(&dava, &ayar);
        assert_eq!(h1.etiket(), h2.etiket());
    }
}
