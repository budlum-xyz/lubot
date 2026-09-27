//! 6.5 — kontrol noktasi yayini ve soy zinciri.
//!
//! Tek cumle: **soyu dogrulanamayan kontrol noktasi yayimlanmaz.**
//!
//! Bir kontrol noktasinin soyu, "hangi veriden hangi turda ciktigi"nin tek
//! kaydidir. Otonom bir dongude bu kaydin ozel bir agirligi var: modeli kimse
//! elle yayimlamadigi icin, yanlis bir halka aylarca fark edilmeden zincirin
//! icinde kalir ve her yeni tur onun uzerine insa eder.
//!
//! Zincirin tasidigi dort kural:
//!
//! 1. **Kok tektir.** Atasi olmayan ikinci bir halka, zinciri sessizce ormana
//!    cevirir; `IkinciKok` ile reddedilir.
//! 2. **Ata once gelir.** Bilinmeyen bir ataya isaret eden halka
//!    `AtaBulunamadi` ile reddedilir - "sonra baglariz" diye bekleyen bir
//!    halka, hicbir zaman baglanmayan halkadir.
//! 3. **Adim geri gitmez.** Ayni ata uzerine daha kucuk adimli bir halka
//!    yazmak, zaman cizgisini catallamanin sessiz yoludur.
//! 4. **Ozet degismez.** Yayimlanmis bir halkanin ozeti guncellenemez; bir
//!    halka yeniden yazilacaksa **yeni bir halka** olur. Zincirin degeri tam
//!    olarak burada: gecmis duzeltilmez, uzerine yazilir.
//!
//! Ozet burada **dogrulanir, hesaplanmaz**: karma islevi `crates/karma`'nin
//! isi ve ikinci bir uygulama iki kopyanin ayrismasi demekti. Bu modul ozetin
//! **seklini** denetler (64 onaltilik basamak, kucuk harf) ve zincirdeki
//! **tekilligini** garanti eder.

use core::fmt;

/// Ozet uzunlugu (sha256, onaltilik).
pub const OZET_BASAMAK: usize = 64;

/// Zincire yazilamadi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YayinRed {
    /// Ozet 64 kucuk harfli onaltilik basamak degil.
    OzetBicimsiz { uzunluk: usize },
    /// Ozet buyuk harf iceriyor: ayni ozet iki yazimda iki farkli dize olurdu.
    OzetBuyukHarf,
    /// Bu ozet zincirde zaten var.
    OzetTekrar { ozet: String },
    /// Atasi olmayan ikinci halka.
    IkinciKok { mevcut_kok: String },
    /// Ata zincirde yok.
    AtaBulunamadi { ata: String },
    /// Adim atadan kucuk ya da esit.
    AdimGeriGitti { ata_adim: u64, gelen: u64 },
    /// Yayimlanmis halkanin ozeti degistirilmek istendi.
    GecmisDuzeltilemez { ozet: String },
    /// Yargi gecmemis: regresyon freni kirmiziyken yayin yok.
    YargiGecmedi,
    /// Kontrol noktasi bos: sifir baytlik agirlik yayimlanmaz.
    BosKontrolNoktasi,
}

impl fmt::Display for YayinRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OzetBicimsiz { uzunluk } => {
                write!(f, "ozet {uzunluk} basamak, {OZET_BASAMAK} bekleniyor")
            }
            Self::OzetBuyukHarf => write!(f, "ozet buyuk harf iceriyor"),
            Self::OzetTekrar { ozet } => write!(f, "ozet {ozet} zincirde zaten var"),
            Self::IkinciKok { mevcut_kok } => {
                write!(f, "zincirin koku zaten var: {mevcut_kok}")
            }
            Self::AtaBulunamadi { ata } => write!(f, "ata {ata} zincirde yok"),
            Self::AdimGeriGitti { ata_adim, gelen } => {
                write!(f, "adim {gelen}, ata adimi {ata_adim}: geri gidilemez")
            }
            Self::GecmisDuzeltilemez { ozet } => {
                write!(f, "{ozet} yayimlanmis: gecmis duzeltilmez, uzerine yazilir")
            }
            Self::YargiGecmedi => write!(f, "regresyon yargisi gecmedi: yayin yok"),
            Self::BosKontrolNoktasi => write!(f, "sifir baytlik kontrol noktasi yayimlanmaz"),
        }
    }
}

/// Zincirdeki bir halka.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Halka {
    ozet: String,
    ata: Option<String>,
    adim: u64,
    bayt: u64,
    veri_ozeti: String,
}

impl Halka {
    #[must_use]
    pub fn ozet(&self) -> &str {
        &self.ozet
    }

    #[must_use]
    pub fn ata(&self) -> Option<&str> {
        self.ata.as_deref()
    }

    #[must_use]
    pub const fn adim(&self) -> u64 {
        self.adim
    }

    #[must_use]
    pub const fn bayt(&self) -> u64 {
        self.bayt
    }

    /// Bu halkanin uzerinde egitildigi veri partisinin ozeti. Agirlik ozetinden
    /// **ayri** tutulur: ayni veriden iki farkli agirlik cikabilir (tohum), ve
    /// ikisini tek alanda toplamak o ayrimi yok ederdi.
    #[must_use]
    pub fn veri_ozeti(&self) -> &str {
        &self.veri_ozeti
    }
}

/// Ozetin seklini denetler.
///
/// # Errors
/// Uzunluk yanlissa, buyuk harf ya da onaltilik olmayan karakter varsa.
pub fn ozet_dogrula(ozet: &str) -> Result<(), YayinRed> {
    if ozet.len() != OZET_BASAMAK {
        return Err(YayinRed::OzetBicimsiz {
            uzunluk: ozet.len(),
        });
    }
    for c in ozet.chars() {
        if c.is_ascii_uppercase() {
            return Err(YayinRed::OzetBuyukHarf);
        }
        if !c.is_ascii_hexdigit() {
            return Err(YayinRed::OzetBicimsiz {
                uzunluk: ozet.len(),
            });
        }
    }
    Ok(())
}

/// Soy zinciri.
#[derive(Debug, Clone, Default)]
pub struct Yayinci {
    halkalar: Vec<Halka>,
    reddedilen: u64,
}

impl Yayinci {
    #[must_use]
    pub const fn yeni() -> Self {
        Self {
            halkalar: Vec::new(),
            reddedilen: 0,
        }
    }

    #[must_use]
    pub fn uzunluk(&self) -> usize {
        self.halkalar.len()
    }

    #[must_use]
    pub fn bos(&self) -> bool {
        self.halkalar.is_empty()
    }

    /// Reddedilen yayin denemelerinin sayisi. **Gizlenmez.**
    #[must_use]
    pub const fn reddedilen(&self) -> u64 {
        self.reddedilen
    }

    /// Bu modul parametre tutmaz.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    #[must_use]
    pub fn bul(&self, ozet: &str) -> Option<&Halka> {
        self.halkalar.iter().find(|h| h.ozet == ozet)
    }

    #[must_use]
    pub fn kok(&self) -> Option<&Halka> {
        self.halkalar.iter().find(|h| h.ata.is_none())
    }

    /// En son yayimlanan halka.
    #[must_use]
    pub fn ucu(&self) -> Option<&Halka> {
        self.halkalar.last()
    }

    /// Yeni bir halka yayimlar.
    ///
    /// `yargi_gecti` bu fonksiyonun **ilk** kontrolu: 6.4'un freni kirmiziysa
    /// hicbir sekil kontrolune bile gecilmez, cunku gecilirse hata mesaji
    /// "ozetin bicimi bozuk" olur ve asil sebep - regresyon - kaybolur.
    ///
    /// # Errors
    /// Yukaridaki dort kuraldan biri cignenirse.
    pub fn yayimla(
        &mut self,
        ozet: &str,
        ata: Option<&str>,
        adim: u64,
        bayt: u64,
        veri_ozeti: &str,
        yargi_gecti: bool,
    ) -> Result<&Halka, YayinRed> {
        let sonuc = self.yayimla_ic(ozet, ata, adim, bayt, veri_ozeti, yargi_gecti);
        if sonuc.is_err() {
            self.reddedilen = self.reddedilen.saturating_add(1);
        }
        sonuc?;
        self.halkalar.last().ok_or(YayinRed::BosKontrolNoktasi)
    }

    fn yayimla_ic(
        &mut self,
        ozet: &str,
        ata: Option<&str>,
        adim: u64,
        bayt: u64,
        veri_ozeti: &str,
        yargi_gecti: bool,
    ) -> Result<(), YayinRed> {
        if !yargi_gecti {
            return Err(YayinRed::YargiGecmedi);
        }
        if bayt == 0 {
            return Err(YayinRed::BosKontrolNoktasi);
        }
        ozet_dogrula(ozet)?;
        ozet_dogrula(veri_ozeti)?;
        if self.bul(ozet).is_some() {
            return Err(YayinRed::OzetTekrar {
                ozet: ozet.to_string(),
            });
        }
        match ata {
            None => {
                if let Some(k) = self.kok() {
                    return Err(YayinRed::IkinciKok {
                        mevcut_kok: k.ozet.clone(),
                    });
                }
            }
            Some(a) => {
                let Some(ata_halka) = self.bul(a) else {
                    return Err(YayinRed::AtaBulunamadi { ata: a.to_string() });
                };
                if adim <= ata_halka.adim {
                    return Err(YayinRed::AdimGeriGitti {
                        ata_adim: ata_halka.adim,
                        gelen: adim,
                    });
                }
            }
        }
        self.halkalar.push(Halka {
            ozet: ozet.to_string(),
            ata: ata.map(ToString::to_string),
            adim,
            bayt,
            veri_ozeti: veri_ozeti.to_string(),
        });
        Ok(())
    }

    /// Bir halkadan koke giden yol. Zincir tutarliysa her zaman kokte biter;
    /// bitmiyorsa `None` doner ve bu **bulunmasi gereken** bir durumdur.
    #[must_use]
    pub fn soy(&self, ozet: &str) -> Option<Vec<&Halka>> {
        let mut yol = Vec::new();
        let mut su_an = self.bul(ozet)?;
        loop {
            yol.push(su_an);
            match su_an.ata() {
                None => return Some(yol),
                Some(a) => {
                    // Dongu koruması: zincir uzunlugunu asan bir yuruyus,
                    // dogrulanabilir bir soy degildir.
                    if yol.len() > self.halkalar.len() {
                        return None;
                    }
                    su_an = self.bul(a)?;
                }
            }
        }
    }

    /// Zincirin butunlugu: tek kok, her ata mevcut, adimlar monoton.
    #[must_use]
    pub fn butun(&self) -> bool {
        if self.halkalar.is_empty() {
            return true;
        }
        let kokler = self.halkalar.iter().filter(|h| h.ata.is_none()).count();
        if kokler != 1 {
            return false;
        }
        self.halkalar
            .iter()
            .all(|h| self.soy(&h.ozet).is_some_and(|y| !y.is_empty()))
    }
}

/// Testler ve olcumler icin belirlenimci sahte ozet uretir. Gercek bir karma
/// **degildir** ve adi bunu soyluyor: `crates/karma` ile karistirilmasin diye
/// `sahte_` oneki tasiyor.
#[must_use]
pub fn sahte_ozet(tohum: u64) -> String {
    let mut s = String::with_capacity(OZET_BASAMAK);
    // `tohum | 1` **degil**: o esleme 2 ile 3'u ayni duruma gonderiyordu ve
    // iki farkli tohum ayni ozeti uretiyordu. Tekli kaydirma+1 birebir.
    let mut durum = tohum.wrapping_mul(2).wrapping_add(1);
    while s.len() < OZET_BASAMAK {
        durum = durum
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s.push_str(&format!("{durum:016x}"));
    }
    s.truncate(OZET_BASAMAK);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zincir_kur() -> (Yayinci, String) {
        let mut y = Yayinci::yeni();
        let kok = sahte_ozet(1);
        y.yayimla(&kok, None, 0, 1024, &sahte_ozet(100), true)
            .unwrap();
        (y, kok)
    }

    #[test]
    fn sahte_ozet_bicimi_gecerli() {
        for t in 0..32 {
            let o = sahte_ozet(t);
            assert_eq!(o.len(), OZET_BASAMAK);
            assert!(ozet_dogrula(&o).is_ok(), "{o}");
        }
    }

    #[test]
    fn kisa_ozet_reddedilir() {
        assert!(matches!(
            ozet_dogrula("abc"),
            Err(YayinRed::OzetBicimsiz { .. })
        ));
    }

    #[test]
    fn buyuk_harfli_ozet_reddedilir() {
        let o = "A".repeat(OZET_BASAMAK);
        assert_eq!(ozet_dogrula(&o), Err(YayinRed::OzetBuyukHarf));
    }

    #[test]
    fn onaltilik_olmayan_ozet_reddedilir() {
        let o = "z".repeat(OZET_BASAMAK);
        assert!(matches!(
            ozet_dogrula(&o),
            Err(YayinRed::OzetBicimsiz { .. })
        ));
    }

    /// Bu modulun tek cumlesi, ilk kontrol olarak.
    #[test]
    fn yargi_gecmeden_yayin_yok() {
        let mut y = Yayinci::yeni();
        let r = y.yayimla(&sahte_ozet(1), None, 0, 1024, &sahte_ozet(2), false);
        assert_eq!(r.err(), Some(YayinRed::YargiGecmedi));
        assert!(y.bos());
        assert_eq!(y.reddedilen(), 1);
    }

    /// Yargi kontrolu **once** gelir: bozuk ozetli ve yargisiz bir deneme
    /// "yargi gecmedi" demeli, "ozet bozuk" degil - yoksa asil sebep kaybolur.
    #[test]
    fn yargi_redi_bicim_redini_onceler() {
        let mut y = Yayinci::yeni();
        let r = y.yayimla("bozuk", None, 0, 1024, "de bozuk", false);
        assert_eq!(r.err(), Some(YayinRed::YargiGecmedi));
    }

    #[test]
    fn bos_kontrol_noktasi_yayimlanmaz() {
        let mut y = Yayinci::yeni();
        let r = y.yayimla(&sahte_ozet(1), None, 0, 0, &sahte_ozet(2), true);
        assert_eq!(r.err(), Some(YayinRed::BosKontrolNoktasi));
    }

    #[test]
    fn kok_yayimlanir() {
        let (y, kok) = zincir_kur();
        assert_eq!(y.uzunluk(), 1);
        assert_eq!(y.kok().map(Halka::ozet), Some(kok.as_str()));
    }

    #[test]
    fn ikinci_kok_reddedilir() {
        let (mut y, kok) = zincir_kur();
        let r = y.yayimla(&sahte_ozet(2), None, 1, 1024, &sahte_ozet(3), true);
        assert_eq!(r.err(), Some(YayinRed::IkinciKok { mevcut_kok: kok }));
    }

    #[test]
    fn bilinmeyen_ata_reddedilir() {
        let (mut y, _) = zincir_kur();
        let hayalet = sahte_ozet(999);
        let r = y.yayimla(
            &sahte_ozet(2),
            Some(&hayalet),
            1,
            1024,
            &sahte_ozet(3),
            true,
        );
        assert_eq!(r.err(), Some(YayinRed::AtaBulunamadi { ata: hayalet }));
    }

    #[test]
    fn adim_geri_gitmez() {
        let (mut y, kok) = zincir_kur();
        let r = y.yayimla(&sahte_ozet(2), Some(&kok), 0, 1024, &sahte_ozet(3), true);
        assert_eq!(
            r.err(),
            Some(YayinRed::AdimGeriGitti {
                ata_adim: 0,
                gelen: 0
            })
        );
    }

    #[test]
    fn ayni_ozet_iki_kez_yayimlanmaz() {
        let (mut y, kok) = zincir_kur();
        let r = y.yayimla(&kok, Some(&kok), 5, 1024, &sahte_ozet(3), true);
        assert_eq!(r.err(), Some(YayinRed::OzetTekrar { ozet: kok }));
    }

    #[test]
    fn soy_kokte_biter() {
        let (mut y, kok) = zincir_kur();
        let a = sahte_ozet(2);
        let b = sahte_ozet(3);
        y.yayimla(&a, Some(&kok), 10, 1024, &sahte_ozet(10), true)
            .unwrap();
        y.yayimla(&b, Some(&a), 20, 1024, &sahte_ozet(11), true)
            .unwrap();
        let yol = y.soy(&b).unwrap();
        assert_eq!(yol.len(), 3);
        assert_eq!(yol[0].ozet(), b);
        assert_eq!(yol[2].ozet(), kok);
        assert!(yol[2].ata().is_none());
    }

    #[test]
    fn catallanma_serbest_ama_adim_monoton() {
        // Ayni atadan iki cocuk: dallanma (6.9) icin gerekli, yasak degil.
        let (mut y, kok) = zincir_kur();
        y.yayimla(&sahte_ozet(2), Some(&kok), 10, 1, &sahte_ozet(10), true)
            .unwrap();
        y.yayimla(&sahte_ozet(3), Some(&kok), 10, 1, &sahte_ozet(11), true)
            .unwrap();
        assert_eq!(y.uzunluk(), 3);
        assert!(y.butun());
    }

    #[test]
    fn veri_ozeti_agirlik_ozetinden_ayri() {
        let (mut y, kok) = zincir_kur();
        let a = sahte_ozet(2);
        let veri = sahte_ozet(77);
        y.yayimla(&a, Some(&kok), 10, 1024, &veri, true).unwrap();
        let h = y.bul(&a).unwrap();
        assert_eq!(h.veri_ozeti(), veri);
        assert_ne!(h.veri_ozeti(), h.ozet());
    }

    #[test]
    fn bicimsiz_veri_ozeti_reddedilir() {
        let (mut y, kok) = zincir_kur();
        let r = y.yayimla(&sahte_ozet(2), Some(&kok), 10, 1024, "kisa", true);
        assert!(matches!(r.err(), Some(YayinRed::OzetBicimsiz { .. })));
    }

    #[test]
    fn red_zinciri_degistirmez() {
        let (mut y, kok) = zincir_kur();
        let once = y.uzunluk();
        let _ = y.yayimla(&sahte_ozet(2), Some("yok"), 10, 1024, &sahte_ozet(3), true);
        let _ = y.yayimla(&kok, Some(&kok), 10, 1024, &sahte_ozet(3), true);
        assert_eq!(y.uzunluk(), once, "red zincire yazdi");
        assert_eq!(y.reddedilen(), 2);
    }

    #[test]
    fn bos_zincir_butun_sayilir() {
        assert!(Yayinci::yeni().butun());
    }

    #[test]
    fn uzun_zincir_butun() {
        let (mut y, kok) = zincir_kur();
        let mut onceki = kok;
        for i in 2..50u64 {
            let o = sahte_ozet(i);
            y.yayimla(&o, Some(&onceki), i * 10, 512, &sahte_ozet(i + 1000), true)
                .unwrap();
            onceki = o;
        }
        assert_eq!(y.uzunluk(), 49);
        assert!(y.butun());
        assert_eq!(y.soy(&onceki).map(|v| v.len()), Some(49));
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Yayinci::parametre_sayisi(), 0);
    }
}
