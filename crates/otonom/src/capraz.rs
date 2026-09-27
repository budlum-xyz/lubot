//! 6.6 — validator capraz dogrulamasi.
//!
//! Tek cumle: **bir dogrulayici cogunluk degildir.**
//!
//! Otonom bir dongude "dogrulandi" kelimesi cok ucuz. Bir dogrulayici kosar,
//! yesil der, tur gecer. Sorun su ki tek dogrulayicinin yesili, dogrulayicinin
//! kendisi bozuldugunda da yesildir - ve bozuldugunu soyleyecek ikinci bir ses
//! yoktur.
//!
//! Bu yuzden burada yazilan kural sayisal: bir sonuc ancak **en az iki
//! bagimsiz** dogrulayici ayni yargiya vardiginda kabul edilir, ve
//! "bagimsiz"in tanimi beyanlidir - ayni `kaynak`tan gelen iki dogrulayici
//! **bir** sayilir. Ayni kod tabanini iki kez kosturmak capraz dogrulama
//! degil, tekrardir.
//!
//! Cekimserlik de sayilir ve **yesile eklenmez**. "Karar veremedim" ile
//! "gecti" arasindaki farki silmek, kurulun en sik rastlanan cokme bicimi:
//! yeterince cekimser varsa azinliktaki tek bir yesil cogunluk gibi gorunur.
//!
//! Beraberlik reddir, gecis degil. Esit sayida yesil ve kirmizi varsa kurul
//! **karar vermemistir**; bunu gecis saymak, kararsizligi onaya cevirmek olur.

use core::fmt;

/// Bir sonucun kabulu icin gereken en az bagimsiz onay.
pub const EN_AZ_BAGIMSIZ_ONAY: usize = 2;

/// Tek bir dogrulayicinin oyu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Oy {
    /// Gecti.
    Yesil,
    /// Gecmedi.
    Kirmizi,
    /// Karar veremedi. **Yesile eklenmez.**
    Cekimser,
}

impl Oy {
    #[must_use]
    pub const fn ad(self) -> &'static str {
        match self {
            Self::Yesil => "yesil",
            Self::Kirmizi => "kirmizi",
            Self::Cekimser => "cekimser",
        }
    }
}

/// Kurul kurulamadi ya da yargi verilemedi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KurulRed {
    /// Dogrulayicinin adi bos.
    AdsizDogrulayici,
    /// Kaynak beyan edilmemis: bagimsizlik olculemez.
    KaynaksizDogrulayici { ad: String },
    /// Ayni ad iki kez kayitli.
    AdTekrar { ad: String },
}

impl fmt::Display for KurulRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdsizDogrulayici => write!(f, "adsiz dogrulayici kurula giremez"),
            Self::KaynaksizDogrulayici { ad } => {
                write!(f, "{ad} icin kaynak beyan edilmemis: bagimsizlik olculemez")
            }
            Self::AdTekrar { ad } => write!(f, "{ad} zaten kurulda"),
        }
    }
}

/// Bir dogrulayici: adi ve **kaynagi**.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dogrulayici {
    ad: String,
    kaynak: String,
}

impl Dogrulayici {
    /// # Errors
    /// Ad ya da kaynak bossa reddeder.
    pub fn yeni(ad: &str, kaynak: &str) -> Result<Self, KurulRed> {
        if ad.trim().is_empty() {
            return Err(KurulRed::AdsizDogrulayici);
        }
        if kaynak.trim().is_empty() {
            return Err(KurulRed::KaynaksizDogrulayici { ad: ad.to_string() });
        }
        Ok(Self {
            ad: ad.trim().to_string(),
            kaynak: kaynak.trim().to_string(),
        })
    }

    #[must_use]
    pub fn ad(&self) -> &str {
        &self.ad
    }

    /// Bagimsizligin olculdugu alan. Ayni kaynaktan gelen iki dogrulayici bir
    /// sayilir.
    #[must_use]
    pub fn kaynak(&self) -> &str {
        &self.kaynak
    }
}

/// Kurulun yargisi, sayilariyla.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yargi {
    kabul: bool,
    yesil_oy: usize,
    kirmizi_oy: usize,
    cekimser_oy: usize,
    bagimsiz_yesil: usize,
    bagimsiz_kirmizi: usize,
    sebep: &'static str,
}

impl Yargi {
    #[must_use]
    pub const fn kabul(&self) -> bool {
        self.kabul
    }

    #[must_use]
    pub const fn yesil_oy(&self) -> usize {
        self.yesil_oy
    }

    #[must_use]
    pub const fn kirmizi_oy(&self) -> usize {
        self.kirmizi_oy
    }

    /// Cekimser oylar ayri sayilir; yesile eklenmez.
    #[must_use]
    pub const fn cekimser_oy(&self) -> usize {
        self.cekimser_oy
    }

    /// Farkli kaynaklardan gelen yesil sayisi. Kabul kosulu budur, `yesil_oy`
    /// degil.
    #[must_use]
    pub const fn bagimsiz_yesil(&self) -> usize {
        self.bagimsiz_yesil
    }

    #[must_use]
    pub const fn bagimsiz_kirmizi(&self) -> usize {
        self.bagimsiz_kirmizi
    }

    /// Kararin makine-okunur gerekcesi.
    #[must_use]
    pub const fn sebep(&self) -> &'static str {
        self.sebep
    }
}

/// Dogrulayici kurulu.
#[derive(Debug, Clone, Default)]
pub struct Kurul {
    uyeler: Vec<Dogrulayici>,
}

impl Kurul {
    #[must_use]
    pub const fn yeni() -> Self {
        Self { uyeler: Vec::new() }
    }

    /// Bu modul parametre tutmaz.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    #[must_use]
    pub fn uye_sayisi(&self) -> usize {
        self.uyeler.len()
    }

    /// Kuruldaki farkli kaynak sayisi: kurulun **gercek** genisligi.
    #[must_use]
    pub fn kaynak_sayisi(&self) -> usize {
        let mut kaynaklar: Vec<&str> = self.uyeler.iter().map(Dogrulayici::kaynak).collect();
        kaynaklar.sort_unstable();
        kaynaklar.dedup();
        kaynaklar.len()
    }

    /// # Errors
    /// Ad tekrarliysa reddeder.
    pub fn ekle(&mut self, d: Dogrulayici) -> Result<(), KurulRed> {
        if self.uyeler.iter().any(|u| u.ad == d.ad) {
            return Err(KurulRed::AdTekrar { ad: d.ad });
        }
        self.uyeler.push(d);
        Ok(())
    }

    /// Oylari yargilar.
    ///
    /// `oylar` uye adiyla eslesir; kurulda olmayan bir ad **sessizce
    /// yoksayilir degil**, hic sayilmaz ve bu `cekimser` olarak raporlanir -
    /// yani oyu gelmeyen uye kurulun genisligini dusurur, yesilini degil.
    #[must_use]
    pub fn yargila(&self, oylar: &[(&str, Oy)]) -> Yargi {
        let mut yesil = 0;
        let mut kirmizi = 0;
        let mut cekimser = 0;
        let mut yesil_kaynak: Vec<&str> = Vec::new();
        let mut kirmizi_kaynak: Vec<&str> = Vec::new();

        for uye in &self.uyeler {
            let oy = oylar
                .iter()
                .find(|(ad, _)| *ad == uye.ad)
                .map_or(Oy::Cekimser, |(_, o)| *o);
            match oy {
                Oy::Yesil => {
                    yesil += 1;
                    yesil_kaynak.push(uye.kaynak());
                }
                Oy::Kirmizi => {
                    kirmizi += 1;
                    kirmizi_kaynak.push(uye.kaynak());
                }
                Oy::Cekimser => cekimser += 1,
            }
        }
        yesil_kaynak.sort_unstable();
        yesil_kaynak.dedup();
        kirmizi_kaynak.sort_unstable();
        kirmizi_kaynak.dedup();

        let bagimsiz_yesil = yesil_kaynak.len();
        let bagimsiz_kirmizi = kirmizi_kaynak.len();

        let (kabul, sebep) = if kirmizi > 0 {
            (false, "kirmizi oy var")
        } else if bagimsiz_yesil < EN_AZ_BAGIMSIZ_ONAY {
            (false, "bagimsiz onay yetersiz")
        } else if bagimsiz_yesil == bagimsiz_kirmizi {
            (false, "beraberlik karar degildir")
        } else {
            (true, "en az iki bagimsiz onay")
        };

        Yargi {
            kabul,
            yesil_oy: yesil,
            kirmizi_oy: kirmizi,
            cekimser_oy: cekimser,
            bagimsiz_yesil,
            bagimsiz_kirmizi,
            sebep,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kurul3() -> Kurul {
        let mut k = Kurul::yeni();
        k.ekle(Dogrulayici::yeni("a", "kaynak-1").unwrap()).unwrap();
        k.ekle(Dogrulayici::yeni("b", "kaynak-2").unwrap()).unwrap();
        k.ekle(Dogrulayici::yeni("c", "kaynak-3").unwrap()).unwrap();
        k
    }

    #[test]
    fn adsiz_dogrulayici_kurulamaz() {
        assert_eq!(Dogrulayici::yeni(" ", "k"), Err(KurulRed::AdsizDogrulayici));
    }

    #[test]
    fn kaynaksiz_dogrulayici_kurulamaz() {
        assert!(matches!(
            Dogrulayici::yeni("a", ""),
            Err(KurulRed::KaynaksizDogrulayici { .. })
        ));
    }

    #[test]
    fn ad_tekrari_reddedilir() {
        let mut k = Kurul::yeni();
        k.ekle(Dogrulayici::yeni("a", "k1").unwrap()).unwrap();
        let r = k.ekle(Dogrulayici::yeni("a", "k2").unwrap());
        assert!(matches!(r, Err(KurulRed::AdTekrar { .. })));
    }

    /// Bu modulun tek cumlesi.
    #[test]
    fn tek_yesil_yetmez() {
        let k = kurul3();
        let y = k.yargila(&[("a", Oy::Yesil)]);
        assert!(!y.kabul());
        assert_eq!(y.sebep(), "bagimsiz onay yetersiz");
        assert_eq!(y.bagimsiz_yesil(), 1);
        assert_eq!(y.cekimser_oy(), 2);
    }

    #[test]
    fn iki_bagimsiz_yesil_yeter() {
        let k = kurul3();
        let y = k.yargila(&[("a", Oy::Yesil), ("b", Oy::Yesil)]);
        assert!(y.kabul(), "{}", y.sebep());
        assert_eq!(y.bagimsiz_yesil(), 2);
    }

    /// Ayni kod tabanini iki kez kosturmak capraz dogrulama degil, tekrardir.
    #[test]
    fn ayni_kaynaktan_iki_yesil_bir_sayilir() {
        let mut k = Kurul::yeni();
        k.ekle(Dogrulayici::yeni("a", "ayni").unwrap()).unwrap();
        k.ekle(Dogrulayici::yeni("b", "ayni").unwrap()).unwrap();
        let y = k.yargila(&[("a", Oy::Yesil), ("b", Oy::Yesil)]);
        assert!(!y.kabul(), "ayni kaynak iki sayildi");
        assert_eq!(y.yesil_oy(), 2);
        assert_eq!(y.bagimsiz_yesil(), 1, "bagimsizlik kaynaktan olculmedi");
    }

    #[test]
    fn tek_kirmizi_her_seyi_durdurur() {
        let k = kurul3();
        let y = k.yargila(&[("a", Oy::Yesil), ("b", Oy::Yesil), ("c", Oy::Kirmizi)]);
        assert!(!y.kabul());
        assert_eq!(y.sebep(), "kirmizi oy var");
    }

    /// Cekimseri yesile eklemek, kurulun en sik cokme bicimi.
    #[test]
    fn cekimser_yesile_eklenmez() {
        let k = kurul3();
        let y = k.yargila(&[("a", Oy::Yesil), ("b", Oy::Cekimser), ("c", Oy::Cekimser)]);
        assert!(!y.kabul());
        assert_eq!(y.yesil_oy(), 1);
        assert_eq!(y.cekimser_oy(), 2);
    }

    #[test]
    fn oyu_gelmeyen_uye_cekimser_sayilir() {
        let k = kurul3();
        let y = k.yargila(&[("a", Oy::Yesil)]);
        assert_eq!(y.cekimser_oy(), 2, "oyu gelmeyen uye sayilmadi");
    }

    #[test]
    fn kurulda_olmayan_ad_oy_kullanamaz() {
        let k = kurul3();
        let y = k.yargila(&[
            ("a", Oy::Yesil),
            ("hayalet", Oy::Yesil),
            ("hayalet2", Oy::Yesil),
        ]);
        assert!(!y.kabul(), "kurul disi oy sayildi");
        assert_eq!(y.yesil_oy(), 1);
    }

    #[test]
    fn hic_oy_yoksa_kabul_yok() {
        let k = kurul3();
        let y = k.yargila(&[]);
        assert!(!y.kabul());
        assert_eq!(y.cekimser_oy(), 3);
    }

    #[test]
    fn bos_kurul_kabul_edemez() {
        let k = Kurul::yeni();
        let y = k.yargila(&[]);
        assert!(!y.kabul());
        assert_eq!(y.bagimsiz_yesil(), 0);
    }

    #[test]
    fn kaynak_sayisi_kurulun_gercek_genisligi() {
        let mut k = Kurul::yeni();
        k.ekle(Dogrulayici::yeni("a", "ayni").unwrap()).unwrap();
        k.ekle(Dogrulayici::yeni("b", "ayni").unwrap()).unwrap();
        k.ekle(Dogrulayici::yeni("c", "baska").unwrap()).unwrap();
        assert_eq!(k.uye_sayisi(), 3);
        assert_eq!(k.kaynak_sayisi(), 2, "genislik uye sayisiyla karistirildi");
    }

    #[test]
    fn oy_adlari_kararli() {
        assert_eq!(Oy::Yesil.ad(), "yesil");
        assert_eq!(Oy::Kirmizi.ad(), "kirmizi");
        assert_eq!(Oy::Cekimser.ad(), "cekimser");
    }

    #[test]
    fn yargi_belirlenimci() {
        let k = kurul3();
        let oylar = [("c", Oy::Yesil), ("a", Oy::Yesil), ("b", Oy::Cekimser)];
        assert_eq!(k.yargila(&oylar), k.yargila(&oylar));
        // Oy sirasi karari degistirmemeli.
        let ters = [("b", Oy::Cekimser), ("a", Oy::Yesil), ("c", Oy::Yesil)];
        assert_eq!(k.yargila(&oylar), k.yargila(&ters));
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Kurul::parametre_sayisi(), 0);
    }

    #[test]
    fn esik_iki() {
        assert_eq!(EN_AZ_BAGIMSIZ_ONAY, 2);
    }
}
