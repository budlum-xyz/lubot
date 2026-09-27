//! 6.4 — kapi + ratchet regresyon reddi.
//!
//! Otonom bir dongude en tehlikeli an, modelin **kotulestigi** an degil;
//! kotulestigi halde yayimlandigi andir. Bu modul o ani imkansiz kilmaya
//! calisir ve bunu tek bir siralamayla yapar:
//!
//! **Once ratchet okunur, sonra kayip.**
//!
//! Sirasi ters olsaydi - once kayba bakip "iyilesmis, gecir" deseydik - kayip
//! bir skalerdir ve her zaman bir yonde hareket eder; ratchet ise bir
//! **sozlesmedir** ve yonu vardir. Kaybi iyilestirip test sayisini dusuren bir
//! tur, kayba bakan bir kapidan gecer. Buradan gecmez.
//!
//! Ratchet'in yonu anahtar basina beyan edilir ve iki yon vardir:
//!
//! - **Yukari** (`tests`, `gates`, `corpus`, `tokens`): dusmek regresyondur.
//! - **Asagi** (`pedantic`): yukselmek regresyondur.
//!
//! Ucuncu bir yon - "serbest" - bilincli olarak **yok**. Serbest bir anahtar,
//! ratchet'te yer kaplayan ama hicbir seyi engellemeyen bir satirdir; boyle
//! bir satir varsa okuyan kisi onu da korunuyor sanir. Yonu olmayan sey
//! ratchet'e girmez.
//!
//! Bilinmeyen anahtar da reddedilir (`YonsuzAnahtar`). Cazip alternatif -
//! "tanimadigim anahtari yoksay" - ratchet'e yeni bir anahtar eklemeyi
//! sessizce etkisiz hale getirirdi: yazarsin, korur sanirsin, korumaz.
//!
//! Ve son olarak: bir anahtarin **kaybolmasi** da regresyondur
//! (`AnahtarKayboldu`). Ratchet'ten satir silmek, o satirin korudugu seyi
//! serbest birakmanin en sessiz yoludur.

use core::fmt;

/// Bir ratchet anahtarinin izin verdigi yon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Yon {
    /// Yalniz artabilir; dusmek regresyon.
    Yukari,
    /// Yalniz azalabilir; yukselmek regresyon.
    Asagi,
}

impl Yon {
    #[must_use]
    pub const fn ad(self) -> &'static str {
        match self {
            Self::Yukari => "yukari",
            Self::Asagi => "asagi",
        }
    }
}

/// Bu depoda tanimli ratchet anahtarlari ve yonleri. Liste **kapali**:
/// burada olmayan bir anahtar ratchet'e giremez.
pub const ANAHTARLAR: [(&str, Yon); 7] = [
    ("tests", Yon::Yukari),
    ("gates", Yon::Yukari),
    ("corpus", Yon::Yukari),
    ("tokens", Yon::Yukari),
    ("bootstrap", Yon::Yukari),
    ("exam", Yon::Yukari),
    ("pedantic", Yon::Asagi),
];

/// Bir anahtarin yonu, biliniyorsa.
#[must_use]
pub fn yon(anahtar: &str) -> Option<Yon> {
    let mut i = 0;
    while i < ANAHTARLAR.len() {
        let (ad, y) = ANAHTARLAR[i];
        if konst_esit(ad, anahtar) {
            return Some(y);
        }
        i += 1;
    }
    None
}

fn konst_esit(a: &str, b: &str) -> bool {
    a.as_bytes() == b.as_bytes()
}

/// Bir turun neden reddedildigi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegresyonRed {
    /// Yukari yonlu anahtar dustu.
    Dustu {
        anahtar: String,
        taban: u64,
        olculen: u64,
    },
    /// Asagi yonlu anahtar yukseldi.
    Yukseldi {
        anahtar: String,
        taban: u64,
        olculen: u64,
    },
    /// Ratchet'te olan bir anahtar olcumde yok: silmek de regresyondur.
    AnahtarKayboldu { anahtar: String },
    /// Yonu beyan edilmemis anahtar. Yoksaymak, ratchet'i sessizce delerdi.
    YonsuzAnahtar { anahtar: String },
    /// Kapi kosusu yesil degil. Kayip ne olursa olsun burada durulur.
    KapiKirmizi { kapi: String, sebep: String },
    /// Hicbir kapi kosmamis: "kapi yok" ile "kapi gecti" ayni sey degil.
    KapiKosmadi,
}

impl fmt::Display for RegresyonRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dustu {
                anahtar,
                taban,
                olculen,
            } => write!(f, "{anahtar} dustu: taban {taban}, olculen {olculen}"),
            Self::Yukseldi {
                anahtar,
                taban,
                olculen,
            } => write!(f, "{anahtar} yukseldi: taban {taban}, olculen {olculen}"),
            Self::AnahtarKayboldu { anahtar } => {
                write!(f, "{anahtar} olcumde yok: ratchet'ten satir silinmis")
            }
            Self::YonsuzAnahtar { anahtar } => {
                write!(f, "{anahtar} icin yon beyan edilmemis: ratchet'e giremez")
            }
            Self::KapiKirmizi { kapi, sebep } => write!(f, "kapi {kapi} kirmizi: {sebep}"),
            Self::KapiKosmadi => write!(f, "hicbir kapi kosmadi: yesil sayilmaz"),
        }
    }
}

/// Anahtar/deger ciftleri. `std::collections::HashMap` yerine siralı vektor:
/// iterasyon sirasinin belirlenimci olmasi, red listesinin her kosuda ayni
/// sirada cikmasi demek, ve bir red listesinin sirasi degisirse "ayni hata
/// mi?" sorusu olculemez olur.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sayac {
    girdiler: Vec<(String, u64)>,
}

impl Sayac {
    #[must_use]
    pub const fn bos() -> Self {
        Self {
            girdiler: Vec::new(),
        }
    }

    /// Ekler ya da gunceller; anahtarlar alfabetik sirada tutulur.
    pub fn koy(&mut self, anahtar: &str, deger: u64) {
        match self
            .girdiler
            .binary_search_by(|(a, _)| a.as_str().cmp(anahtar))
        {
            Ok(i) => {
                if let Some(g) = self.girdiler.get_mut(i) {
                    g.1 = deger;
                }
            }
            Err(i) => self.girdiler.insert(i, (anahtar.to_string(), deger)),
        }
    }

    #[must_use]
    pub fn al(&self, anahtar: &str) -> Option<u64> {
        self.girdiler
            .binary_search_by(|(a, _)| a.as_str().cmp(anahtar))
            .ok()
            .and_then(|i| self.girdiler.get(i).map(|g| g.1))
    }

    #[must_use]
    pub fn anahtarlar(&self) -> Vec<&str> {
        self.girdiler.iter().map(|(a, _)| a.as_str()).collect()
    }

    #[must_use]
    pub fn sayi(&self) -> usize {
        self.girdiler.len()
    }
}

/// Bir kapi kosusunun sonucu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KapiSonucu {
    ad: String,
    gecti: bool,
    sebep: String,
}

impl KapiSonucu {
    #[must_use]
    pub fn gecti(ad: &str) -> Self {
        Self {
            ad: ad.to_string(),
            gecti: true,
            sebep: String::new(),
        }
    }

    #[must_use]
    pub fn kirmizi(ad: &str, sebep: &str) -> Self {
        Self {
            ad: ad.to_string(),
            gecti: false,
            sebep: sebep.to_string(),
        }
    }

    #[must_use]
    pub fn ad(&self) -> &str {
        &self.ad
    }

    #[must_use]
    pub const fn yesil(&self) -> bool {
        self.gecti
    }
}

/// Bir turun yayima hak kazanip kazanmadigi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yargi {
    redler: Vec<RegresyonRed>,
    kapi_sayisi: usize,
    anahtar_sayisi: usize,
}

impl Yargi {
    /// Tur gecti mi? **Tek** kosul: hic red yok.
    #[must_use]
    pub fn gecti(&self) -> bool {
        self.redler.is_empty()
    }

    #[must_use]
    pub fn redler(&self) -> &[RegresyonRed] {
        &self.redler
    }

    #[must_use]
    pub const fn kapi_sayisi(&self) -> usize {
        self.kapi_sayisi
    }

    #[must_use]
    pub const fn anahtar_sayisi(&self) -> usize {
        self.anahtar_sayisi
    }
}

/// Fren: once kapilar, sonra ratchet, **sonra** kayip.
#[derive(Debug, Clone)]
pub struct Fren {
    taban: Sayac,
}

impl Fren {
    #[must_use]
    pub const fn yeni(taban: Sayac) -> Self {
        Self { taban }
    }

    #[must_use]
    pub const fn taban(&self) -> &Sayac {
        &self.taban
    }

    /// Bu modul parametre tutmaz.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    /// Turu yargilar. Kayip **bilerek** parametre degil: bu fonksiyonun kaybi
    /// gormemesi, kaybin bir redi ezememesini yapisal olarak garanti eder.
    #[must_use]
    pub fn yargila(&self, olculen: &Sayac, kapilar: &[KapiSonucu]) -> Yargi {
        let mut redler = Vec::new();

        if kapilar.is_empty() {
            redler.push(RegresyonRed::KapiKosmadi);
        }
        for k in kapilar {
            if !k.yesil() {
                redler.push(RegresyonRed::KapiKirmizi {
                    kapi: k.ad.clone(),
                    sebep: k.sebep.clone(),
                });
            }
        }

        for anahtar in self.taban.anahtarlar() {
            let Some(taban_deger) = self.taban.al(anahtar) else {
                continue;
            };
            let Some(y) = yon(anahtar) else {
                redler.push(RegresyonRed::YonsuzAnahtar {
                    anahtar: anahtar.to_string(),
                });
                continue;
            };
            let Some(olculen_deger) = olculen.al(anahtar) else {
                redler.push(RegresyonRed::AnahtarKayboldu {
                    anahtar: anahtar.to_string(),
                });
                continue;
            };
            match y {
                Yon::Yukari if olculen_deger < taban_deger => {
                    redler.push(RegresyonRed::Dustu {
                        anahtar: anahtar.to_string(),
                        taban: taban_deger,
                        olculen: olculen_deger,
                    });
                }
                Yon::Asagi if olculen_deger > taban_deger => {
                    redler.push(RegresyonRed::Yukseldi {
                        anahtar: anahtar.to_string(),
                        taban: taban_deger,
                        olculen: olculen_deger,
                    });
                }
                _ => {}
            }
        }

        // Olcumde olup tabanda olmayan anahtar red degil: yeni bir olcut
        // eklemek serbest, kaldirmak degil. Ama sayilir.
        Yargi {
            redler,
            kapi_sayisi: kapilar.len(),
            anahtar_sayisi: self.taban.sayi(),
        }
    }

    /// Yeni tabani dondurur: **yalniz** yargi gectiyse. Gecmemis bir turun
    /// tabani ilerletmesi, ratchet'in tanimini ortadan kaldirirdi.
    #[must_use]
    pub fn taban_ilerlet(&self, olculen: &Sayac, kapilar: &[KapiSonucu]) -> Option<Sayac> {
        if !self.yargila(olculen, kapilar).gecti() {
            return None;
        }
        let mut yeni = self.taban.clone();
        for anahtar in olculen.anahtarlar() {
            if let (Some(d), Some(_)) = (olculen.al(anahtar), yon(anahtar)) {
                yeni.koy(anahtar, d);
            }
        }
        Some(yeni)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn taban() -> Sayac {
        let mut s = Sayac::bos();
        s.koy("tests", 1600);
        s.koy("gates", 109);
        s.koy("pedantic", 0);
        s
    }

    fn yesil_kapilar() -> Vec<KapiSonucu> {
        vec![KapiSonucu::gecti("a"), KapiSonucu::gecti("b")]
    }

    #[test]
    fn yonler_kapali_listede() {
        assert_eq!(yon("tests"), Some(Yon::Yukari));
        assert_eq!(yon("pedantic"), Some(Yon::Asagi));
        assert_eq!(yon("uydurma"), None);
    }

    #[test]
    fn ayni_sayilar_gecer() {
        let f = Fren::yeni(taban());
        let y = f.yargila(&taban(), &yesil_kapilar());
        assert!(y.gecti(), "{:?}", y.redler());
    }

    #[test]
    fn yukari_anahtar_dususu_reddedilir() {
        let f = Fren::yeni(taban());
        let mut o = taban();
        o.koy("tests", 1599);
        let y = f.yargila(&o, &yesil_kapilar());
        assert!(!y.gecti());
        assert_eq!(
            y.redler()[0],
            RegresyonRed::Dustu {
                anahtar: "tests".to_string(),
                taban: 1600,
                olculen: 1599
            }
        );
    }

    #[test]
    fn asagi_anahtar_yukselisi_reddedilir() {
        let f = Fren::yeni(taban());
        let mut o = taban();
        o.koy("pedantic", 1);
        let y = f.yargila(&o, &yesil_kapilar());
        assert!(!y.gecti());
        assert!(matches!(y.redler()[0], RegresyonRed::Yukseldi { .. }));
    }

    #[test]
    fn asagi_anahtar_dususu_kabul() {
        let mut t = taban();
        t.koy("pedantic", 13);
        let f = Fren::yeni(t);
        let mut o = taban();
        o.koy("pedantic", 0);
        assert!(f.yargila(&o, &yesil_kapilar()).gecti());
    }

    /// Ratchet'ten satir silmek, korudugu seyi serbest birakmanin en sessiz
    /// yolu; o yuzden silinmesi de regresyon.
    #[test]
    fn kaybolan_anahtar_regresyondur() {
        let f = Fren::yeni(taban());
        let mut o = Sayac::bos();
        o.koy("tests", 9999);
        o.koy("pedantic", 0);
        let y = f.yargila(&o, &yesil_kapilar());
        assert!(!y.gecti());
        assert!(y.redler().iter().any(|r| matches!(
            r,
            RegresyonRed::AnahtarKayboldu { anahtar } if anahtar == "gates"
        )));
    }

    #[test]
    fn yonsuz_anahtar_yoksayilmaz() {
        let mut t = taban();
        t.koy("uydurma", 5);
        let f = Fren::yeni(t);
        let mut o = taban();
        o.koy("uydurma", 5);
        let y = f.yargila(&o, &yesil_kapilar());
        assert!(!y.gecti(), "yonu olmayan anahtar sessizce gecti");
        assert!(y
            .redler()
            .iter()
            .any(|r| matches!(r, RegresyonRed::YonsuzAnahtar { .. })));
    }

    #[test]
    fn kirmizi_kapi_her_seyi_durdurur() {
        let f = Fren::yeni(taban());
        let mut o = taban();
        o.koy("tests", 99_999); // her sey "iyilesmis" olsa bile
        let kapilar = vec![KapiSonucu::gecti("a"), KapiSonucu::kirmizi("b", "kanarya")];
        let y = f.yargila(&o, &kapilar);
        assert!(!y.gecti(), "kirmizi kapiyi iyilesme ezdi");
    }

    /// "Kapi yok" ile "kapi gecti" ayni sey degil.
    #[test]
    fn hic_kapi_kosmazsa_yesil_sayilmaz() {
        let f = Fren::yeni(taban());
        let y = f.yargila(&taban(), &[]);
        assert!(!y.gecti());
        assert_eq!(y.redler()[0], RegresyonRed::KapiKosmadi);
    }

    #[test]
    fn yeni_anahtar_eklemek_serbest() {
        let f = Fren::yeni(taban());
        let mut o = taban();
        o.koy("corpus", 7075);
        assert!(f.yargila(&o, &yesil_kapilar()).gecti());
    }

    #[test]
    fn taban_yalniz_gecen_turla_ilerler() {
        let f = Fren::yeni(taban());
        let mut kotu = taban();
        kotu.koy("tests", 1);
        assert!(
            f.taban_ilerlet(&kotu, &yesil_kapilar()).is_none(),
            "dusen tur tabani ilerletti"
        );
        let mut iyi = taban();
        iyi.koy("tests", 1700);
        let yeni = f.taban_ilerlet(&iyi, &yesil_kapilar()).unwrap();
        assert_eq!(yeni.al("tests"), Some(1700));
    }

    #[test]
    fn taban_ilerletirken_yonsuz_anahtar_tasinmaz() {
        let f = Fren::yeni(taban());
        let mut iyi = taban();
        iyi.koy("gecici_olcum", 42);
        let yeni = f.taban_ilerlet(&iyi, &yesil_kapilar()).unwrap();
        assert_eq!(
            yeni.al("gecici_olcum"),
            None,
            "yonu olmayan anahtar tabana sizdi"
        );
    }

    #[test]
    fn red_listesi_belirlenimci_sirada() {
        let mut t = taban();
        t.koy("corpus", 100);
        t.koy("tokens", 100);
        let f = Fren::yeni(t);
        let mut o = Sayac::bos();
        o.koy("tests", 1);
        o.koy("gates", 1);
        o.koy("corpus", 1);
        o.koy("tokens", 1);
        o.koy("pedantic", 0);
        let bir = f.yargila(&o, &yesil_kapilar());
        let iki = f.yargila(&o, &yesil_kapilar());
        assert_eq!(bir.redler(), iki.redler(), "red listesi sirasi oynak");
        let adlar: Vec<&str> = bir
            .redler()
            .iter()
            .filter_map(|r| match r {
                RegresyonRed::Dustu { anahtar, .. } => Some(anahtar.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(adlar, vec!["corpus", "gates", "tests", "tokens"]);
    }

    #[test]
    fn sayac_koy_gunceller_cogaltmaz() {
        let mut s = Sayac::bos();
        s.koy("a", 1);
        s.koy("a", 2);
        assert_eq!(s.sayi(), 1);
        assert_eq!(s.al("a"), Some(2));
    }

    #[test]
    fn sayac_alfabetik_tutulur() {
        let mut s = Sayac::bos();
        s.koy("z", 1);
        s.koy("a", 1);
        s.koy("m", 1);
        assert_eq!(s.anahtarlar(), vec!["a", "m", "z"]);
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Fren::parametre_sayisi(), 0);
    }

    #[test]
    fn butun_anahtarlarin_yonu_var() {
        // Kapali listenin kendisi de denetlenir: yonu olmayan bir satir
        // eklenirse bu test dusr.
        for (ad, _) in ANAHTARLAR {
            assert!(yon(ad).is_some(), "{ad} icin yon cozulemedi");
        }
        assert_eq!(ANAHTARLAR.len(), 7);
    }
}
