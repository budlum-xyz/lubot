//! 6.7 — kapisma tabanli aday secimi.
//!
//! Tek cumle: **esit skor esit raporlanir.**
//!
//! Iki aday ayni sayiyi aldiginda aralarinda bir sira **uydurmak**, olcumun
//! tasimadigi bir bilgi uretmektir. Cazip, cunku "bir kazanan lazim"; ve
//! zararsiz gorunur, cunku fark kucuktur. Zararli olmasinin sebebi su: bir
//! kez uydurulan sira, sonraki turda tabana donusur ve gurultu kalicilasir.
//!
//! Bu yuzden buradaki siralama uc kurala baglidir:
//!
//! 1. **Skor esitse esit.** [`Sonuc::sira`] ayni skoru alan adaylara ayni
//!    sirayi verir; sonraki sira **atlanir** (1, 2, 2, 4), cunku uc adayin
//!    ikisi berabere ise ucuncusu ucuncu degildir.
//! 2. **Beraberlik cagiran sirasiyla bozulur**, gizlice degil. Bir liste
//!    gerekiyorsa [`Kapisma::siralama`] girdi sirasini korur ve bunu beyan
//!    eder; skorun icine kucuk bir sayi eklemek gibi bir "sessiz bozucu" yok.
//! 3. **Kazanan ancak tek ise vardir.** [`Kapisma::kazanan`] berabere durumda
//!    `None` doner. "Ilkini al" demek, 1. kurali arka kapidan delmek olurdu.
//!
//! Skorlar `f64` ve karsilastirma **bit duzeyinde degil**, beyanli bir
//! toleransla yapiliyor: ayni hesaptan gelen iki skor son bitte ayrilabilir ve
//! bu ayrimi "fark" saymak, tam olarak uydurulmus siranin baska bir bicimi
//! olurdu. Tolerans sabiti [`ESITLIK_TOLERANSI`] ile beyanli.
//!
//! NaN skoru **reddedilir**. Siralanabilir olmayan bir sayi, siralamaya
//! sokuldugunda hangi karsilastirmanin once yapildigina gore farkli sonuc
//! verir - yani belirlenimciligi sessizce bitirir.

use core::fmt;

/// Iki skorun esit sayildigi mutlak fark.
pub const ESITLIK_TOLERANSI: f64 = 1e-12;

/// Aday ya da kapisma kurulamadi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecimRed {
    /// Adin bos olmasi.
    AdsizAday,
    /// Skor sonlu degil (NaN ya da sonsuz).
    SiralanamazSkor { ad: String },
    /// Ayni ad iki kez.
    AdTekrar { ad: String },
    /// Hic aday yok: bos bir kapismanin kazanani olmaz.
    AdaySiz,
}

impl fmt::Display for SecimRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdsizAday => write!(f, "adsiz aday kapismaya giremez"),
            Self::SiralanamazSkor { ad } => {
                write!(f, "{ad} icin skor siralanamaz (NaN ya da sonsuz)")
            }
            Self::AdTekrar { ad } => write!(f, "{ad} zaten kapismada"),
            Self::AdaySiz => write!(f, "aday yok: bos kapismanin kazanani olmaz"),
        }
    }
}

/// Bir aday ve skoru.
#[derive(Debug, Clone, PartialEq)]
pub struct Aday {
    ad: String,
    skor: f64,
}

impl Aday {
    /// # Errors
    /// Ad bossa ya da skor sonlu degilse reddeder.
    pub fn yeni(ad: &str, skor: f64) -> Result<Self, SecimRed> {
        if ad.trim().is_empty() {
            return Err(SecimRed::AdsizAday);
        }
        if !skor.is_finite() {
            return Err(SecimRed::SiralanamazSkor {
                ad: ad.trim().to_string(),
            });
        }
        Ok(Self {
            ad: ad.trim().to_string(),
            skor,
        })
    }

    #[must_use]
    pub fn ad(&self) -> &str {
        &self.ad
    }

    #[must_use]
    pub const fn skor(&self) -> f64 {
        self.skor
    }
}

/// Siralamadaki bir satir.
#[derive(Debug, Clone, PartialEq)]
pub struct Sonuc {
    ad: String,
    skor: f64,
    sira: usize,
    berabere: bool,
    girdi_sirasi: usize,
}

impl Sonuc {
    #[must_use]
    pub fn ad(&self) -> &str {
        &self.ad
    }

    #[must_use]
    pub const fn skor(&self) -> f64 {
        self.skor
    }

    /// 1'den baslar. Beraberlikte **ayni** sira verilir ve sonraki atlanir.
    #[must_use]
    pub const fn sira(&self) -> usize {
        self.sira
    }

    /// Bu satir en az bir baskasiyla ayni skoru mu paylasiyor?
    #[must_use]
    pub const fn berabere(&self) -> bool {
        self.berabere
    }

    /// Adayin kapismaya verilis sirasi. Beraberligin **beyanli** bozucusu;
    /// gizli bir bozucu yok.
    #[must_use]
    pub const fn girdi_sirasi(&self) -> usize {
        self.girdi_sirasi
    }
}

/// Kapisma.
#[derive(Debug, Clone, Default)]
pub struct Kapisma {
    adaylar: Vec<Aday>,
}

impl Kapisma {
    #[must_use]
    pub const fn yeni() -> Self {
        Self {
            adaylar: Vec::new(),
        }
    }

    /// Bu modul parametre tutmaz.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    #[must_use]
    pub fn sayi(&self) -> usize {
        self.adaylar.len()
    }

    /// # Errors
    /// Ad tekrarliysa reddeder.
    pub fn ekle(&mut self, aday: Aday) -> Result<(), SecimRed> {
        if self.adaylar.iter().any(|a| a.ad == aday.ad) {
            return Err(SecimRed::AdTekrar { ad: aday.ad });
        }
        self.adaylar.push(aday);
        Ok(())
    }

    /// Iki skor esit sayilir mi?
    #[must_use]
    pub fn esit(a: f64, b: f64) -> bool {
        (a - b).abs() <= ESITLIK_TOLERANSI
    }

    /// Skora gore azalan siralama; beraberlik girdi sirasiyla bozulur ve
    /// **isaretlenir**.
    ///
    /// # Errors
    /// Aday yoksa reddeder.
    pub fn siralama(&self) -> Result<Vec<Sonuc>, SecimRed> {
        if self.adaylar.is_empty() {
            return Err(SecimRed::AdaySiz);
        }
        let mut idx: Vec<usize> = (0..self.adaylar.len()).collect();
        idx.sort_by(|&x, &y| {
            let (ax, ay) = (&self.adaylar[x], &self.adaylar[y]);
            if Self::esit(ax.skor, ay.skor) {
                // Beraberlik cagiran sirasiyla bozulur, gizlice degil.
                return x.cmp(&y);
            }
            // NaN yapici tarafindan disarida birakildi; `partial_cmp` burada
            // her zaman `Some` veriyor ve verdigi yerde belirlenimci.
            ay.skor
                .partial_cmp(&ax.skor)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut sonuclar: Vec<Sonuc> = Vec::with_capacity(idx.len());
        let mut sira = 0;
        let mut onceki_skor: Option<f64> = None;
        for (yer, &i) in idx.iter().enumerate() {
            let a = &self.adaylar[i];
            let ayni = onceki_skor.is_some_and(|s| Self::esit(s, a.skor));
            if !ayni {
                sira = yer + 1;
            }
            onceki_skor = Some(a.skor);
            sonuclar.push(Sonuc {
                ad: a.ad.clone(),
                skor: a.skor,
                sira,
                berabere: false,
                girdi_sirasi: i,
            });
        }
        // Beraberlik isareti: ayni sirayi paylasan her satir isaretlenir.
        for k in 0..sonuclar.len() {
            let paylasan = sonuclar
                .iter()
                .filter(|s| s.sira == sonuclar[k].sira)
                .count();
            if paylasan > 1 {
                if let Some(s) = sonuclar.get_mut(k) {
                    s.berabere = true;
                }
            }
        }
        Ok(sonuclar)
    }

    /// Tek kazanan. Berabere durumda `None`.
    ///
    /// "Ilkini al" demek, esit skorun esit raporlanmasi kuralini arka kapidan
    /// delmek olurdu; bu yuzden burada bir liste degil bir **Option** var.
    ///
    /// # Errors
    /// Aday yoksa reddeder.
    pub fn kazanan(&self) -> Result<Option<&Aday>, SecimRed> {
        let sira = self.siralama()?;
        let Some(bas) = sira.first() else {
            return Err(SecimRed::AdaySiz);
        };
        if bas.berabere() {
            return Ok(None);
        }
        Ok(self.adaylar.iter().find(|a| a.ad == bas.ad))
    }

    /// Ilk sirayi paylasanlar. Kazanan yoksa **kimin** paylastigi yine de
    /// raporlanir: "karar veremedim" ile "hic aday yoktu" ayni sey degil.
    ///
    /// # Errors
    /// Aday yoksa reddeder.
    pub fn zirve(&self) -> Result<Vec<String>, SecimRed> {
        let sira = self.siralama()?;
        let Some(bas) = sira.first() else {
            return Err(SecimRed::AdaySiz);
        };
        Ok(sira
            .iter()
            .filter(|s| s.sira == bas.sira)
            .map(|s| s.ad.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kur(skorlar: &[(&str, f64)]) -> Kapisma {
        let mut k = Kapisma::yeni();
        for (ad, s) in skorlar {
            k.ekle(Aday::yeni(ad, *s).unwrap()).unwrap();
        }
        k
    }

    #[test]
    fn adsiz_aday_kurulamaz() {
        assert_eq!(Aday::yeni("  ", 1.0), Err(SecimRed::AdsizAday));
    }

    #[test]
    fn nan_skor_reddedilir() {
        assert!(matches!(
            Aday::yeni("a", f64::NAN),
            Err(SecimRed::SiralanamazSkor { .. })
        ));
    }

    #[test]
    fn sonsuz_skor_reddedilir() {
        assert!(matches!(
            Aday::yeni("a", f64::INFINITY),
            Err(SecimRed::SiralanamazSkor { .. })
        ));
    }

    #[test]
    fn ad_tekrari_reddedilir() {
        let mut k = kur(&[("a", 1.0)]);
        assert!(matches!(
            k.ekle(Aday::yeni("a", 2.0).unwrap()),
            Err(SecimRed::AdTekrar { .. })
        ));
    }

    #[test]
    fn bos_kapismanin_kazanani_yok() {
        let k = Kapisma::yeni();
        assert_eq!(k.siralama().err(), Some(SecimRed::AdaySiz));
        assert_eq!(k.kazanan().err(), Some(SecimRed::AdaySiz));
    }

    #[test]
    fn azalan_siralama() {
        let k = kur(&[("dusuk", 0.1), ("yuksek", 0.9), ("orta", 0.5)]);
        let s = k.siralama().unwrap();
        assert_eq!(s[0].ad(), "yuksek");
        assert_eq!(s[1].ad(), "orta");
        assert_eq!(s[2].ad(), "dusuk");
        assert_eq!(s[0].sira(), 1);
        assert_eq!(s[2].sira(), 3);
    }

    /// Bu modulun tek cumlesi.
    #[test]
    fn esit_skor_esit_sira_alir() {
        let k = kur(&[("a", 0.5), ("b", 0.5), ("c", 0.1)]);
        let s = k.siralama().unwrap();
        assert_eq!(s[0].sira(), 1);
        assert_eq!(s[1].sira(), 1, "esit skora farkli sira verildi");
        assert!(s[0].berabere() && s[1].berabere());
    }

    /// Uc adayin ikisi berabere ise ucuncusu ucuncudur, ikinci degil.
    #[test]
    fn beraberlikten_sonra_sira_atlanir() {
        let k = kur(&[("a", 0.5), ("b", 0.5), ("c", 0.1)]);
        let s = k.siralama().unwrap();
        assert_eq!(s[2].sira(), 3, "beraberlikten sonra sira atlanmadi");
        assert!(!s[2].berabere());
    }

    #[test]
    fn berabere_zirvede_kazanan_yok() {
        let k = kur(&[("a", 0.5), ("b", 0.5)]);
        assert!(
            k.kazanan().unwrap().is_none(),
            "beraberlikte kazanan uyduruldu"
        );
    }

    #[test]
    fn berabere_olsa_da_zirve_raporlanir() {
        let k = kur(&[("a", 0.5), ("b", 0.5), ("c", 0.1)]);
        let z = k.zirve().unwrap();
        assert_eq!(z, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn tek_kazanan_dondurulur() {
        let k = kur(&[("a", 0.9), ("b", 0.5)]);
        assert_eq!(k.kazanan().unwrap().map(Aday::ad), Some("a"));
    }

    #[test]
    fn tolerans_icindeki_fark_esit_sayilir() {
        let k = kur(&[("a", 0.5), ("b", 0.5 + ESITLIK_TOLERANSI / 2.0)]);
        assert!(
            k.kazanan().unwrap().is_none(),
            "son bit farki kazanan uydurdu"
        );
    }

    #[test]
    fn tolerans_disindaki_fark_ayirir() {
        let k = kur(&[("a", 0.5), ("b", 0.5 + ESITLIK_TOLERANSI * 100.0)]);
        assert_eq!(k.kazanan().unwrap().map(Aday::ad), Some("b"));
    }

    /// Beraberligin bozucusu **beyanli**: girdi sirasi.
    #[test]
    fn beraberlik_girdi_sirasiyla_bozulur() {
        let k = kur(&[("once", 0.5), ("sonra", 0.5)]);
        let s = k.siralama().unwrap();
        assert_eq!(s[0].ad(), "once");
        assert_eq!(s[0].girdi_sirasi(), 0);
        assert_eq!(s[1].girdi_sirasi(), 1);
    }

    #[test]
    fn siralama_belirlenimci() {
        let k = kur(&[("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.9)]);
        assert_eq!(k.siralama().unwrap(), k.siralama().unwrap());
    }

    #[test]
    fn hepsi_berabere_ise_hepsi_zirvede() {
        let k = kur(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        assert_eq!(k.zirve().unwrap().len(), 3);
        assert!(k.kazanan().unwrap().is_none());
        let s = k.siralama().unwrap();
        assert!(s.iter().all(|x| x.sira() == 1 && x.berabere()));
    }

    #[test]
    fn negatif_skorlar_siralanir() {
        let k = kur(&[("a", -1.0), ("b", -0.5), ("c", -2.0)]);
        let s = k.siralama().unwrap();
        assert_eq!(s[0].ad(), "b");
        assert_eq!(s[2].ad(), "c");
    }

    #[test]
    fn tek_adayin_kazanani_kendisi() {
        let k = kur(&[("tek", 0.0)]);
        assert_eq!(k.kazanan().unwrap().map(Aday::ad), Some("tek"));
        assert_eq!(k.zirve().unwrap(), vec!["tek".to_string()]);
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Kapisma::parametre_sayisi(), 0);
        assert_eq!(kur(&[("a", 1.0)]).sayi(), 1);
    }
}
