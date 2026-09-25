//! Örnekleyici çekirdeği: logitleri bir sonraki jetona çeviren matematik.
//!
//! Bu modül *karar* vermez, metin yazmaz, ağ çalıştırmaz. Tek işi bir logit
//! vektörünü, verilen ayarla bir olasılık dağılımına indirmek ve o dağılımdan
//! deterministik bir akıştan bir indeks çekmek. Ağın ileri geçişi
//! [`crate::Cikarim`]'de kalır; ikisini birleştiren döngü ayrı bir yüzeydir ve
//! bu modülün dışındadır.
//!
//! # Sıfır sıcaklık bir dağılım değil, bir karardır
//!
//! `sicaklik = 0` örnekleme değildir: en büyük logitli jeton alınır. Bu ayrım
//! kodda da açıkça yazılıdır, çünkü "0'a böl" ile "0 sıcaklık" aynı şey değil
//! ve sessizce NaN üretmek yerine söylenmesi gerekir.
//!
//! # Eşitlik eşitliktir
//!
//! Aynı logitli iki jeton arasında "hangisi daha iyi" diye bir şey yoktur.
//! Eşitlik küçük indeks lehine kırılır; böylece sıralama girdilerin fonksiyonu
//! olur, sıralama algoritmasının keyfinin değil (kardeş kural:
//! [`crate::Cikarim::pasaj_sirala`]).

/// Örnekleme ayarı.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ayarlar {
    /// Sıcaklık: logitler bu değere bölünür. `0` → açgözlü (argmax).
    pub sicaklik: f64,
    /// En iyi kaç jeton tutulur. `0` → süzgeç kapalı.
    pub top_k: usize,
    /// Nucleus eşiği: toplam olasılığı bu değere ulaşan en küçük küme tutulur.
    /// `1.0` → süzgeç kapalı.
    pub top_p: f64,
}

impl Default for Ayarlar {
    fn default() -> Self {
        Self {
            sicaklik: 1.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// Örnekleyicinin ret sebepleri.
#[derive(Debug, Clone, PartialEq)]
pub enum OrnekHatasi {
    /// Ayar aralık dışı: negatif/NaN sıcaklık, `(0,1]` dışı top-p.
    GecersizAyar(String),
    /// Logit vektörü boş: çekilecek jeton yok.
    BosLogit,
    /// Tüm logitler sonlu değil (NaN/±sonsuz) ve dağılım kurulamıyor.
    CozulemezLogit,
}

impl std::fmt::Display for OrnekHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GecersizAyar(ne) => write!(f, "gecersiz ornekleme ayari: {ne}"),
            Self::BosLogit => write!(f, "logit vektoru bos: orneklenecek jeton yok"),
            Self::CozulemezLogit => write!(f, "logitler sonlu degil: dagilim kurulamiyor"),
        }
    }
}

impl std::error::Error for OrnekHatasi {}

impl Ayarlar {
    /// Ayarın kurulabilir olduğunu söyler.
    ///
    /// # Errors
    /// [`OrnekHatasi::GecersizAyar`] — sıcaklık negatif/NaN/sonsuz, `top_p`
    /// `(0,1]` dışında.
    pub fn dogrula(&self) -> Result<(), OrnekHatasi> {
        if !self.sicaklik.is_finite() || self.sicaklik < 0.0 {
            return Err(OrnekHatasi::GecersizAyar(format!(
                "sicaklik {} sonlu ve negatif olmayan olmali",
                self.sicaklik
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(OrnekHatasi::GecersizAyar(format!(
                "top_p {} (0,1] araliginda olmali",
                self.top_p
            )));
        }
        Ok(())
    }
}

/// Determinist akış: splitmix64. Aynı tohum aynı diziyi verir.
///
/// `rand` bağımlılığı yoktur (K1: from-scratch): dağıtımın kendisi kadar
/// küçük bir akış yeter, ve ölçülebilir olması gerekir.
#[derive(Debug, Clone)]
pub(crate) struct Rastgele {
    durum: u64,
}

impl Rastgele {
    /// Tohumdan akış kurar.
    #[must_use]
    pub const fn tohumdan(tohum: u64) -> Self {
        Self { durum: tohum }
    }

    /// Sonraki 64 bit.
    pub fn sonraki_u64(&mut self) -> u64 {
        self.durum = self.durum.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.durum;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `[0,1)` aralığında bir sayı (53 bit duyarlık).
    pub fn birim(&mut self) -> f64 {
        (self.sonraki_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Logitleri dağılıma indirir: sıcaklık → top-k → nucleus, sonra normalize.
///
/// Dönen vektör olasılığa göre azalan sıralıdır; eşitlikte küçük indeks önce
/// gelir. Dağılım yalnız tutulan jetonlar üzerinde toplamı 1 yapar.
///
/// # Errors
/// [`OrnekHatasi::GecersizAyar`], [`OrnekHatasi::BosLogit`],
/// [`OrnekHatasi::CozulemezLogit`].
/// [`dagilim`], artı: yasak jeton listesi.
///
/// Yasak, geçerli bir logit üzerine konmuş bir sınırdır (ör. tekrarlanan
/// n-gram'ın devamı), bu yüzden aday listesinden *çıkarılır*; logit `-inf`
/// yapılmaz. İkisi ayrı sözleşmedir: sonlu olmayan logit "dağılım kurulamaz"
/// demektir ve orada durmak doğrudur, yasak ise sınırlı ama kurulabilir bir
/// dağılımdır.
///
/// # Errors
/// [`OrnekHatasi::BosLogit`], [`OrnekHatasi::CozulemezLogit`] (sonlu olmayan
/// logit) ve yasağın bütün adayları kapatması durumunda yine
/// [`OrnekHatasi::CozulemezLogit`].
pub(crate) fn dagilim_maskele(
    logitler: &[f64],
    ayar: &Ayarlar,
    yasak: &[u32],
) -> Result<Vec<(u32, f64)>, OrnekHatasi> {
    ayar.dogrula()?;
    if logitler.is_empty() {
        return Err(OrnekHatasi::BosLogit);
    }
    if logitler.iter().any(|l| !l.is_finite()) {
        // NaN ya da ±sonsuz: softmax kurulamaz. Sessizce sifira cevirmek
        // yerine ret - sayi uydurmak yerine "bu logitlerle dagilim olmaz".
        return Err(OrnekHatasi::CozulemezLogit);
    }
    let serbest = |sira: usize| -> bool { !yasak.contains(&(sira as u32)) };
    if !(0..logitler.len()).any(serbest) {
        return Err(OrnekHatasi::CozulemezLogit);
    }
    if ayar.sicaklik == 0.0 {
        // Açgözlü: tek jeton, olasılık 1. Eşitlikte küçük indeks.
        let mut en_iyi: Option<usize> = None;
        for (sira, logit) in logitler.iter().enumerate() {
            if !serbest(sira) {
                continue;
            }
            match en_iyi {
                Some(mevcut) if logitler[mevcut] >= *logit => {}
                _ => en_iyi = Some(sira),
            }
        }
        let sira = en_iyi.ok_or(OrnekHatasi::CozulemezLogit)?;
        return Ok(vec![(sira as u32, 1.0)]);
    }
    let olcek = 1.0 / ayar.sicaklik;
    let en_buyuk = logitler
        .iter()
        .enumerate()
        .filter(|(sira, _)| serbest(*sira))
        .map(|(_, logit)| *logit)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut agirlik: Vec<(u32, f64)> = logitler
        .iter()
        .enumerate()
        .filter(|(sira, _)| serbest(*sira))
        .map(|(sira, logit)| (sira as u32, ((logit - en_buyuk) * olcek).exp()))
        .collect();
    // Eşitlikte küçük indeks önce: sıralama girdilerin fonksiyonu olur.
    agirlik.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    if ayar.top_k > 0 && ayar.top_k < agirlik.len() {
        agirlik.truncate(ayar.top_k);
    }
    let toplam: f64 = agirlik.iter().map(|(_, w)| *w).sum();
    if !toplam.is_finite() || toplam <= 0.0 {
        return Err(OrnekHatasi::CozulemezLogit);
    }
    for (_, w) in &mut agirlik {
        *w /= toplam;
    }
    if ayar.top_p < 1.0 {
        let mut kumulatif = 0.0;
        let mut kes = agirlik.len();
        for (sira, (_, p)) in agirlik.iter().enumerate() {
            kumulatif += p;
            if kumulatif >= ayar.top_p {
                kes = sira + 1;
                break;
            }
        }
        agirlik.truncate(kes.max(1));
        let yeni: f64 = agirlik.iter().map(|(_, p)| *p).sum();
        for (_, p) in &mut agirlik {
            *p /= yeni;
        }
    }
    Ok(agirlik)
}

/// Dağılımdan bir jeton çeker.
///
/// # Errors
/// [`OrnekHatasi::BosLogit`] — dağılım boşsa (kurulmuş bir dağılım boş olamaz;
/// çağıran ham bir vektör geçirdiyse ret edilir).
pub(crate) fn ornekle(dagilim: &[(u32, f64)], rastgele: &mut Rastgele) -> Result<u32, OrnekHatasi> {
    if dagilim.is_empty() {
        return Err(OrnekHatasi::BosLogit);
    }
    let cekilis = rastgele.birim();
    let mut kumulatif = 0.0;
    for (jeton, olasilik) in dagilim {
        kumulatif += *olasilik;
        if cekilis < kumulatif {
            return Ok(*jeton);
        }
    }
    // Yuvarlama artığı: son jeton kuyruğu toplar.
    Ok(dagilim[dagilim.len() - 1].0)
}

#[cfg(test)]
mod testler {
    use super::*;

    /// Iki adimin kisa yolu; test yazimini kısaltir, uretim yuzeyine girmez
    /// (kullanilmayan herkese acik kisa yol, kapinin listesine takilirdi).
    /// Test yazimini kisaltir: maskesiz dagilim.
    fn dagilim(logitler: &[f64], ayar: &Ayarlar) -> Result<Vec<(u32, f64)>, OrnekHatasi> {
        dagilim_maskele(logitler, ayar, &[])
    }

    fn sec(logitler: &[f64], ayar: &Ayarlar, rastgele: &mut Rastgele) -> Result<u32, OrnekHatasi> {
        ornekle(&dagilim_maskele(logitler, ayar, &[])?, rastgele)
    }

    fn esit(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn sicaklik_sifir_argmax_verir() {
        let logitler = [0.5, 2.0, 1.0];
        let ayar = Ayarlar {
            sicaklik: 0.0,
            ..Ayarlar::default()
        };
        let d = dagilim(&logitler, &ayar).expect("dagilim");
        assert_eq!(d, vec![(1, 1.0)]);
        // Eşitlikte küçük indeks kazanır - "hangisi daha iyi" diye bir şey yok.
        let esit_logitler = [1.0, 1.0, 1.0];
        let d2 = dagilim(&esit_logitler, &ayar).expect("dagilim");
        assert_eq!(d2, vec![(0, 1.0)]);
    }

    #[test]
    fn sicaklik_orani_olcekler() {
        // Iki logit (0, 1): p1/p0 = exp((l1-l0)/T) = exp(1/T). Sicaklik boluyor mu?
        let logitler = [0.0, 1.0];
        for sicaklik in [0.5f64, 1.0, 2.0] {
            let beklenen = (1.0f64 / sicaklik).exp();
            let d = dagilim(
                &logitler,
                &Ayarlar {
                    sicaklik,
                    ..Ayarlar::default()
                },
            )
            .expect("dagilim");
            let p0 = d.iter().find(|(j, _)| *j == 0).expect("jeton 0").1;
            let p1 = d.iter().find(|(j, _)| *j == 1).expect("jeton 1").1;
            esit(p1 / p0, beklenen);
        }
    }

    #[test]
    fn top_k_kume_boyutunu_kirpar() {
        let logitler = [3.0, 2.0, 1.0, 0.0, -1.0];
        let d1 = dagilim(
            &logitler,
            &Ayarlar {
                top_k: 1,
                ..Ayarlar::default()
            },
        )
        .expect("dagilim");
        assert_eq!(d1.len(), 1);
        assert_eq!(d1[0].0, 0);
        let d3 = dagilim(
            &logitler,
            &Ayarlar {
                top_k: 3,
                ..Ayarlar::default()
            },
        )
        .expect("dagilim");
        assert_eq!(d3.len(), 3);
        assert_eq!(
            d3.iter().map(|(j, _)| *j).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        // top_k = 0 süzgeci kapatır.
        let acik = dagilim(&logitler, &Ayarlar::default()).expect("dagilim");
        assert_eq!(acik.len(), 5);
    }

    #[test]
    fn nucleus_en_kucuk_kumeyi_tutar() {
        // Buyuk bosluk: ilk jetonun olasiligi ~0,73, ikincisi ~0,27.
        let logitler = [1.0f64, 0.0];
        let d = dagilim(
            &logitler,
            &Ayarlar {
                top_p: 0.5,
                ..Ayarlar::default()
            },
        )
        .expect("dagilim");
        assert_eq!(d.len(), 1, "0,5 esiginde kume tek jetona inmeliydi");
        let d2 = dagilim(
            &logitler,
            &Ayarlar {
                top_p: 0.99,
                ..Ayarlar::default()
            },
        )
        .expect("dagilim");
        assert_eq!(d2.len(), 2);
        esit(d2.iter().map(|(_, p)| *p).sum::<f64>(), 1.0);
        // top_p = 1.0 süzgeci kapatır.
        let d3 = dagilim(&logitler, &Ayarlar::default()).expect("dagilim");
        assert_eq!(d3.len(), 2);
    }

    #[test]
    fn dagilim_toplami_birdir_ve_azalan_siralidir() {
        let logitler = [0.3, -1.0, 2.5, 0.7, -0.2];
        let d = dagilim(
            &logitler,
            &Ayarlar {
                sicaklik: 0.8,
                top_p: 0.9,
                ..Ayarlar::default()
            },
        )
        .expect("dagilim");
        esit(d.iter().map(|(_, p)| *p).sum::<f64>(), 1.0);
        for cift in d.windows(2) {
            assert!(cift[0].1 >= cift[1].1 - 1e-15, "siralama bozuk: {d:?}");
        }
    }

    #[test]
    fn ayni_tohum_ayni_diziyi_verir() {
        let logitler = [0.4, 0.4, 0.4, 0.4];
        let ayar = Ayarlar {
            sicaklik: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let mut a = Rastgele::tohumdan(42);
        let mut b = Rastgele::tohumdan(42);
        let dizi_a: Vec<u32> = (0..64)
            .map(|_| sec(&logitler, &ayar, &mut a).expect("sec"))
            .collect();
        let dizi_b: Vec<u32> = (0..64)
            .map(|_| sec(&logitler, &ayar, &mut b).expect("sec"))
            .collect();
        assert_eq!(dizi_a, dizi_b);
        // Tohum degisince dizi degismeli (duz dagilimda esitlik neredeyse imkansiz).
        let mut c = Rastgele::tohumdan(43);
        let dizi_c: Vec<u32> = (0..64)
            .map(|_| sec(&logitler, &ayar, &mut c).expect("sec"))
            .collect();
        assert_ne!(dizi_a, dizi_c);
    }

    #[test]
    fn ornekleme_dagilimi_gezer_ama_sinirin_disina_cikmaz() {
        let logitler = [0.0, 0.0, 0.0, 0.0];
        let ayar = Ayarlar {
            top_k: 2,
            ..Ayarlar::default()
        };
        let mut r = Rastgele::tohumdan(7);
        let mut gorulen = std::collections::BTreeSet::new();
        for _ in 0..200 {
            gorulen.insert(sec(&logitler, &ayar, &mut r).expect("sec"));
        }
        assert_eq!(
            gorulen.iter().copied().collect::<Vec<_>>(),
            vec![0, 1],
            "top_k sinirinin disina cikildi ya da hic gezilmedi"
        );
    }

    #[test]
    fn gecersiz_ayar_ve_bos_girdi_reddedilir() {
        assert!(matches!(
            dagilim(
                &[1.0],
                &Ayarlar {
                    sicaklik: -1.0,
                    ..Ayarlar::default()
                }
            ),
            Err(OrnekHatasi::GecersizAyar(_))
        ));
        assert!(matches!(
            dagilim(
                &[1.0],
                &Ayarlar {
                    sicaklik: f64::NAN,
                    ..Ayarlar::default()
                }
            ),
            Err(OrnekHatasi::GecersizAyar(_))
        ));
        assert!(matches!(
            dagilim(
                &[1.0],
                &Ayarlar {
                    top_p: 0.0,
                    ..Ayarlar::default()
                }
            ),
            Err(OrnekHatasi::GecersizAyar(_))
        ));
        assert!(matches!(
            dagilim(
                &[1.0],
                &Ayarlar {
                    top_p: 1.5,
                    ..Ayarlar::default()
                }
            ),
            Err(OrnekHatasi::GecersizAyar(_))
        ));
        assert!(matches!(
            dagilim(&[], &Ayarlar::default()),
            Err(OrnekHatasi::BosLogit)
        ));
        assert!(matches!(
            dagilim(&[f64::NAN, 1.0], &Ayarlar::default()),
            Err(OrnekHatasi::CozulemezLogit)
        ));
        assert!(matches!(
            ornekle(&[], &mut Rastgele::tohumdan(1)),
            Err(OrnekHatasi::BosLogit)
        ));
    }

    #[test]
    fn akis_birim_araliginda_ve_tek_duzeyde_kalir() {
        let mut r = Rastgele::tohumdan(20260924);
        let mut en_kucuk = f64::INFINITY;
        let mut en_buyuk = f64::NEG_INFINITY;
        for _ in 0..10_000 {
            let x = r.birim();
            assert!((0.0..1.0).contains(&x), "birim() aralik disi: {x}");
            en_kucuk = en_kucuk.min(x);
            en_buyuk = en_buyuk.max(x);
        }
        // Duz bir akista hem alt hem ust yari gorulmeli: sabit sayi ureten bir
        // akis "deterministik" olurdu ama ornekleyici olamazdi.
        assert!(en_kucuk < 0.25 && en_buyuk > 0.75, "{en_kucuk}..{en_buyuk}");
    }
}
