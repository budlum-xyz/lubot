//! 6.1 — tetikleyici ve toplulastirma.
//!
//! Otonom dongunun ilk sorusu "ne zaman egitilir?" degil, **"ne zaman
//! egitilmez?"** Bir dongu ancak reddedebildigi olcude otonomdur: kendi
//! kendini tetikleyen ve hicbir kosulda durmayan bir sey dongu degil,
//! kacaktir.
//!
//! Bu yuzden burada yazilan sey bir zamanlayici degil, bir **red kumesi**:
//!
//! - **Bos parti egitilmez.** Sure esigi dolsa bile. Zaman gecmesi veri
//!   degildir; bos bir parti uzerinde kosan bir tur, sifir gradyanla bir
//!   kontrol noktasi yayimlar ve soy zincirine anlamsiz bir halka ekler.
//! - **Geriye giden saat reddedilir.** Kayitlarin damgasi monoton olmak
//!   zorunda; olmadigi anda `SaatGeriGitti` ile durulur. Sessizce siralamak,
//!   "hangi veri hangi turda vardi" sorusunu olculemez hale getirir.
//! - **Sogumadan yeniden tetiklenmez.** Esik dolar dolmaz ikinci bir tur
//!   acmak, ayni veriyi iki kez gormek demektir; `soguma_saniye` bunu
//!   yapisal olarak imkansiz kilar, disiplinle degil.
//! - **Tasma gizlenmez.** Sayac `u32` tavanina dayanirsa `SayacTasti` ile
//!   reddedilir; sarmalanan bir sayac, esigi hicbir zaman gecmeyen bir
//!   sayaca donusur ve dongu sessizce olur.
//!
//! Esigin iki yuzu var ve **hangisinin atesledigi kaydedilir**: sayi mi, sure
//! mi, yoksa ayni anda ikisi mi. Bu ayrim rapor suslemesi degil — sayiyla
//! tetiklenen bir dongu veri akisina, sureyle tetiklenen bir dongu takvime
//! bagimlidir, ve ikisinin karisimi ancak ayri ayri olculdugunde okunur.
//!
//! Saat disaridan verilir (`simdi` parametresi). Gercek bir saat okumak, bu
//! modulu belirlenimci olmaktan cikarirdi; testler ayni yurumeyi tekrar
//! kosabiliyorsa, uretimdeki bir tetikleme de yeniden uretilebilir demektir.

use core::fmt;

/// Bir partide toplanabilecek en fazla kayit. Tavan keyfi degil: `u32`
/// sayacinin sarmalanmasindan **once** duran bir sinir olmasi gerekiyordu, ve
/// bellekte tutulan damga vektorunun sinirsiz buyumemesi de ayni sayiyla
/// kapaniyor.
pub const MAKS_PARTI: u32 = 1_000_000;

/// Esigin kabul ettigi en uzun pencere (30 gun). Bundan uzun bir pencere,
/// "bekle" demenin kibar bir bicimi olurdu; dongunun durdugu bir kipe ihtiyac
/// varsa o kip `Esik::yok()` ile **adlandirilarak** kurulur.
pub const MAKS_PENCERE_SANIYE: u64 = 30 * 24 * 60 * 60;

/// Esik kurulamadi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EsikRed {
    /// Iki esik de sifir: boyle bir tetikleyici ya hic ates etmez ya da her
    /// kayitta eder; hangisi oldugu okunamaz, o yuzden kurulmaz.
    IkiEsikDeSifir,
    /// Sayi esigi parti tavanindan buyuk: asla dolmaz.
    SayiEsigiTavaninUstunde { istenen: u32, tavan: u32 },
    /// Pencere tavandan uzun.
    PencereCokUzun { istenen: u64, tavan: u64 },
    /// Soguma penceresi pencerenin kendisinden uzun: dongu kilitlenirdi.
    SogumaPencereyiAsiyor { soguma: u64, pencere: u64 },
}

impl fmt::Display for EsikRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IkiEsikDeSifir => {
                write!(f, "iki esik de sifir: tetikleyicinin davranisi okunamaz")
            }
            Self::SayiEsigiTavaninUstunde { istenen, tavan } => {
                write!(f, "sayi esigi {istenen}, parti tavani {tavan}: asla dolmaz")
            }
            Self::PencereCokUzun { istenen, tavan } => {
                write!(f, "pencere {istenen} sn, tavan {tavan} sn")
            }
            Self::SogumaPencereyiAsiyor { soguma, pencere } => {
                write!(
                    f,
                    "soguma {soguma} sn, pencere {pencere} sn: dongu kilitlenir"
                )
            }
        }
    }
}

/// Toplulastirma sirasinda kayit reddedildi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BiriktirmeRed {
    /// Damga bir oncekinden kucuk. Sessizce siralamak, "hangi veri hangi
    /// turda vardi" sorusunu olculemez yapardi.
    SaatGeriGitti { onceki: u64, gelen: u64 },
    /// Parti tavani doldu ve tur hala acilmadi.
    SayacTasti { tavan: u32 },
    /// Kaydin kimligi bos: kaynagi olmayan bir kayit provenance zincirine
    /// giremez (6.8 ile ayni kural, burada girişte uygulaniyor).
    KimliksizKayit,
    /// Ayni kimlik ayni partide iki kez: tekrar, esigi hak edilmeden doldurur.
    TekrarEdenKimlik { kimlik: String },
}

impl fmt::Display for BiriktirmeRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SaatGeriGitti { onceki, gelen } => {
                write!(f, "damga geri gitti: onceki {onceki}, gelen {gelen}")
            }
            Self::SayacTasti { tavan } => write!(f, "parti tavani {tavan} doldu"),
            Self::KimliksizKayit => write!(f, "kimliksiz kayit toplulastirmaya giremez"),
            Self::TekrarEdenKimlik { kimlik } => {
                write!(f, "kimlik {kimlik} bu partide zaten var")
            }
        }
    }
}

/// Esigin hangi yuzu atesledi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tetik {
    /// Kayit sayisi esige ulasti.
    Sayi,
    /// Pencere doldu ve parti bos degildi.
    Sure,
    /// Ikisi ayni degerlendirmede dolду.
    Ikisi,
}

impl Tetik {
    /// Kayit icin kisa ad. Rapor metni degil, **anahtar**: olcum kaydinda
    /// bu dize karsilastirilir.
    #[must_use]
    pub const fn ad(self) -> &'static str {
        match self {
            Self::Sayi => "sayi",
            Self::Sure => "sure",
            Self::Ikisi => "ikisi",
        }
    }
}

/// Tetikleme esigi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Esik {
    kayit_sayisi: u32,
    pencere_saniye: u64,
    soguma_saniye: u64,
}

impl Esik {
    /// Dogrulayan yapici. Gecersiz bir esik **kurulamaz**; sonradan
    /// denetlenen bir esik, denetlenmeyen bir esiktir.
    ///
    /// # Errors
    /// Iki esik de sifirsa, sayi esigi parti tavanini asiyorsa, pencere
    /// tavandan uzunsa ya da soguma pencereyi asiyorsa reddeder.
    pub const fn yeni(
        kayit_sayisi: u32,
        pencere_saniye: u64,
        soguma_saniye: u64,
    ) -> Result<Self, EsikRed> {
        if kayit_sayisi == 0 && pencere_saniye == 0 {
            return Err(EsikRed::IkiEsikDeSifir);
        }
        if kayit_sayisi > MAKS_PARTI {
            return Err(EsikRed::SayiEsigiTavaninUstunde {
                istenen: kayit_sayisi,
                tavan: MAKS_PARTI,
            });
        }
        if pencere_saniye > MAKS_PENCERE_SANIYE {
            return Err(EsikRed::PencereCokUzun {
                istenen: pencere_saniye,
                tavan: MAKS_PENCERE_SANIYE,
            });
        }
        if pencere_saniye > 0 && soguma_saniye > pencere_saniye {
            return Err(EsikRed::SogumaPencereyiAsiyor {
                soguma: soguma_saniye,
                pencere: pencere_saniye,
            });
        }
        Ok(Self {
            kayit_sayisi,
            pencere_saniye,
            soguma_saniye,
        })
    }

    /// Yalniz sayiyla tetiklenen esik: takvimden bagimsiz dongu.
    ///
    /// # Errors
    /// `kayit_sayisi` sifir ya da tavan ustuyse reddeder.
    pub const fn yalniz_sayi(kayit_sayisi: u32) -> Result<Self, EsikRed> {
        Self::yeni(kayit_sayisi, 0, 0)
    }

    /// Yalniz sureyle tetiklenen esik: veri akisindan bagimsiz dongu.
    /// Bos parti yine egitilmez.
    ///
    /// # Errors
    /// `pencere_saniye` sifir ya da tavan ustuyse reddeder.
    pub const fn yalniz_sure(pencere_saniye: u64) -> Result<Self, EsikRed> {
        Self::yeni(0, pencere_saniye, 0)
    }

    #[must_use]
    pub const fn kayit_sayisi(&self) -> u32 {
        self.kayit_sayisi
    }

    #[must_use]
    pub const fn pencere_saniye(&self) -> u64 {
        self.pencere_saniye
    }

    #[must_use]
    pub const fn soguma_saniye(&self) -> u64 {
        self.soguma_saniye
    }

    /// Bu esik sayi yuzunu kullaniyor mu?
    #[must_use]
    pub const fn sayi_acik(&self) -> bool {
        self.kayit_sayisi > 0
    }

    /// Bu esik sure yuzunu kullaniyor mu?
    #[must_use]
    pub const fn sure_acik(&self) -> bool {
        self.pencere_saniye > 0
    }
}

/// Biriktiriciye giren bir kayit. Icerigi degil **sekli** tasiniyor: bu modul
/// metni gormez, cunku gormesi gerekmiyor ve gormesi 6.2'nin isini bulanik
/// hale getirirdi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kayit {
    kimlik: String,
    damga: u64,
    bayt: u32,
}

impl Kayit {
    /// # Errors
    /// Kimlik bossa reddeder.
    pub fn yeni(kimlik: &str, damga: u64, bayt: u32) -> Result<Self, BiriktirmeRed> {
        if kimlik.trim().is_empty() {
            return Err(BiriktirmeRed::KimliksizKayit);
        }
        Ok(Self {
            kimlik: kimlik.to_string(),
            damga,
            bayt,
        })
    }

    #[must_use]
    pub fn kimlik(&self) -> &str {
        &self.kimlik
    }

    #[must_use]
    pub const fn damga(&self) -> u64 {
        self.damga
    }

    #[must_use]
    pub const fn bayt(&self) -> u32 {
        self.bayt
    }
}

/// Bir degerlendirmenin sonucu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Karar {
    tetik: Option<Tetik>,
    parti_kayit: u32,
    parti_bayt: u64,
    gecen_saniye: u64,
    soguma_kalan: u64,
}

impl Karar {
    /// Tur acildi mi?
    #[must_use]
    pub const fn acildi(&self) -> bool {
        self.tetik.is_some()
    }

    #[must_use]
    pub const fn tetik(&self) -> Option<Tetik> {
        self.tetik
    }

    #[must_use]
    pub const fn parti_kayit(&self) -> u32 {
        self.parti_kayit
    }

    #[must_use]
    pub const fn parti_bayt(&self) -> u64 {
        self.parti_bayt
    }

    #[must_use]
    pub const fn gecen_saniye(&self) -> u64 {
        self.gecen_saniye
    }

    /// Soguma bitene kadar kalan saniye. Sifirsa soguma engel degil.
    #[must_use]
    pub const fn soguma_kalan(&self) -> u64 {
        self.soguma_kalan
    }
}

/// Kayitlari toplayan ve esigi degerlendiren yapi.
#[derive(Debug, Clone)]
pub struct Biriktirici {
    esik: Esik,
    kimlikler: Vec<String>,
    parti_bayt: u64,
    ilk_damga: Option<u64>,
    /// Pencerenin **kayittan bagimsiz** baslangici. `ilk_damga` ile ayni sey
    /// degil ve ayni olmamasi bu modulun bir bulgusu: bos bir partide
    /// `ilk_damga` yoktur, yani pencerenin dolup dolmadigi ona bakilarak
    /// **olculemez** - ve olculemedigi surece "bos parti reddedildi" olayi
    /// hic sayilmaz. Sayilmayan bir red, olmayan bir reddir.
    pencere_basi: Option<u64>,
    son_damga: u64,
    son_tur_damgasi: Option<u64>,
    acilan_tur: u32,
    reddedilen_bos: u32,
    reddedilen_soguma: u32,
}

impl Biriktirici {
    #[must_use]
    pub const fn yeni(esik: Esik) -> Self {
        Self {
            esik,
            kimlikler: Vec::new(),
            parti_bayt: 0,
            ilk_damga: None,
            pencere_basi: None,
            son_damga: 0,
            son_tur_damgasi: None,
            acilan_tur: 0,
            reddedilen_bos: 0,
            reddedilen_soguma: 0,
        }
    }

    #[must_use]
    pub const fn esik(&self) -> &Esik {
        &self.esik
    }

    /// Partideki kayit sayisi.
    #[must_use]
    pub fn parti_kayit(&self) -> u32 {
        // Uzunluk `MAKS_PARTI` ile sinirli, yani daraltma kayipsiz.
        u32::try_from(self.kimlikler.len()).unwrap_or(u32::MAX)
    }

    #[must_use]
    pub const fn parti_bayt(&self) -> u64 {
        self.parti_bayt
    }

    /// Simdiye kadar acilan tur sayisi.
    #[must_use]
    pub const fn acilan_tur(&self) -> u32 {
        self.acilan_tur
    }

    /// Pencere doldugu halde bos oldugu icin reddedilen degerlendirme sayisi.
    /// **Gizlenmez**: bos parti reddi bu dongunun en sik calisan kuralidir ve
    /// sayilmadigi surece calistigi bilinemez.
    #[must_use]
    pub const fn reddedilen_bos(&self) -> u32 {
        self.reddedilen_bos
    }

    /// Esik dolu oldugu halde soguma yuzunden acilmayan tur sayisi.
    #[must_use]
    pub const fn reddedilen_soguma(&self) -> u32 {
        self.reddedilen_soguma
    }

    /// Bu modul parametre tutmaz: bir zamanlama kurali, ogrenilen bir sey
    /// degil.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    /// Kayit ekler.
    ///
    /// # Errors
    /// Damga geri giderse, tavan dolarsa ya da kimlik tekrarsa reddeder; red
    /// halinde **hicbir sey degismez** (atomik).
    pub fn ekle(&mut self, kayit: &Kayit) -> Result<(), BiriktirmeRed> {
        if let Some(ilk) = self.ilk_damga {
            let _ = ilk;
            if kayit.damga < self.son_damga {
                return Err(BiriktirmeRed::SaatGeriGitti {
                    onceki: self.son_damga,
                    gelen: kayit.damga,
                });
            }
        }
        if self.parti_kayit() >= MAKS_PARTI {
            return Err(BiriktirmeRed::SayacTasti { tavan: MAKS_PARTI });
        }
        if self.kimlikler.iter().any(|k| k == kayit.kimlik()) {
            return Err(BiriktirmeRed::TekrarEdenKimlik {
                kimlik: kayit.kimlik().to_string(),
            });
        }
        if self.ilk_damga.is_none() {
            self.ilk_damga = Some(kayit.damga);
        }
        self.son_damga = kayit.damga;
        self.parti_bayt = self.parti_bayt.saturating_add(u64::from(kayit.bayt));
        self.kimlikler.push(kayit.kimlik().to_string());
        Ok(())
    }

    /// Verilen an icin esigi degerlendirir. Tur acilirsa parti **bosaltilir**
    /// ve soguma baslar; acilmazsa hicbir sey degismez (sayaclar haric).
    pub fn degerlendir(&mut self, simdi: u64) -> Karar {
        let kayit = self.parti_kayit();
        if self.pencere_basi.is_none() {
            self.pencere_basi = Some(simdi);
        }
        // Dolu partide pencere **verinin geldigi** andan olculur; bos partide
        // veri yoktur, o yuzden pencerenin kendi baslangicindan olculur. Tek
        // bir alan kullanilsaydi bu iki soru birbirini yerdi.
        let olcum_basi = self.ilk_damga.or(self.pencere_basi);
        let gecen = olcum_basi.map_or(0, |b| simdi.saturating_sub(b));
        let soguma_kalan = self.soguma_kalan(simdi);

        let sayi_doldu = self.esik.sayi_acik() && kayit >= self.esik.kayit_sayisi;
        let sure_doldu = self.esik.sure_acik() && gecen >= self.esik.pencere_saniye;

        // Bos parti egitilmez. Sure esigi dolsa bile: zaman gecmesi veri
        // degildir.
        if kayit == 0 {
            if sure_doldu || sayi_doldu {
                self.reddedilen_bos = self.reddedilen_bos.saturating_add(1);
                // Pencere yeniden baslar: aksi halde bir kez dolan bos
                // pencere her degerlendirmede yeniden sayilir ve sayac
                // "kac kez reddedildi" degil "kac kez bakildi" olurdu.
                self.pencere_basi = Some(simdi);
            }
            return Karar {
                tetik: None,
                parti_kayit: 0,
                parti_bayt: 0,
                gecen_saniye: gecen,
                soguma_kalan,
            };
        }

        let tetik = match (sayi_doldu, sure_doldu) {
            (true, true) => Some(Tetik::Ikisi),
            (true, false) => Some(Tetik::Sayi),
            (false, true) => Some(Tetik::Sure),
            (false, false) => None,
        };

        let Some(tetik) = tetik else {
            return Karar {
                tetik: None,
                parti_kayit: kayit,
                parti_bayt: self.parti_bayt,
                gecen_saniye: gecen,
                soguma_kalan,
            };
        };

        if soguma_kalan > 0 {
            self.reddedilen_soguma = self.reddedilen_soguma.saturating_add(1);
            return Karar {
                tetik: None,
                parti_kayit: kayit,
                parti_bayt: self.parti_bayt,
                gecen_saniye: gecen,
                soguma_kalan,
            };
        }

        let karar = Karar {
            tetik: Some(tetik),
            parti_kayit: kayit,
            parti_bayt: self.parti_bayt,
            gecen_saniye: gecen,
            soguma_kalan: 0,
        };
        self.kimlikler.clear();
        self.parti_bayt = 0;
        self.ilk_damga = None;
        self.pencere_basi = Some(simdi);
        self.son_tur_damgasi = Some(simdi);
        self.acilan_tur = self.acilan_tur.saturating_add(1);
        karar
    }

    fn soguma_kalan(&self, simdi: u64) -> u64 {
        match self.son_tur_damgasi {
            None => 0,
            Some(son) => {
                let gecen = simdi.saturating_sub(son);
                self.esik.soguma_saniye.saturating_sub(gecen)
            }
        }
    }
}

/// Belirlenimci bir yurumenin olcumu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TetikYuruyusOlcumu {
    pub adim: u32,
    pub kayit: u32,
    pub acilan_tur: u32,
    pub sayiyla: u32,
    pub sureyle: u32,
    pub ikisiyle: u32,
    pub reddedilen_bos: u32,
    pub reddedilen_soguma: u32,
}

/// Belirlenimci bir akis uzerinde tetikleyiciyi olcer.
///
/// Akis LCG ile uretiliyor: her `adim` bir saniye ilerliyor ve tohumun
/// belirledigi bir olasilikla bir kayit dusuyor. Amac gercekci bir trafik
/// modeli degil - amac **ayni tohumun ayni sayilari vermesi**, yani bir
/// regresyonun fark edilebilmesi.
#[must_use]
pub fn tetik_yuruyusu_olc(
    esik: Esik,
    adim: u32,
    kayit_sansi: u32,
    tohum: u64,
) -> TetikYuruyusOlcumu {
    let mut b = Biriktirici::yeni(esik);
    let mut durum = tohum | 1;
    let mut olcum = TetikYuruyusOlcumu {
        adim,
        kayit: 0,
        acilan_tur: 0,
        sayiyla: 0,
        sureyle: 0,
        ikisiyle: 0,
        reddedilen_bos: 0,
        reddedilen_soguma: 0,
    };
    for t in 0..u64::from(adim) {
        durum = durum
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let zar = u32::try_from((durum >> 33) % 100).unwrap_or(0);
        if zar < kayit_sansi {
            let kimlik = format!("k{t}");
            if let Ok(k) = Kayit::yeni(&kimlik, t, 128) {
                if b.ekle(&k).is_ok() {
                    olcum.kayit = olcum.kayit.saturating_add(1);
                }
            }
        }
        let karar = b.degerlendir(t);
        match karar.tetik() {
            Some(Tetik::Sayi) => olcum.sayiyla = olcum.sayiyla.saturating_add(1),
            Some(Tetik::Sure) => olcum.sureyle = olcum.sureyle.saturating_add(1),
            Some(Tetik::Ikisi) => olcum.ikisiyle = olcum.ikisiyle.saturating_add(1),
            None => {}
        }
    }
    olcum.acilan_tur = b.acilan_tur();
    olcum.reddedilen_bos = b.reddedilen_bos();
    olcum.reddedilen_soguma = b.reddedilen_soguma();
    olcum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iki_esik_de_sifir_reddedilir() {
        assert_eq!(Esik::yeni(0, 0, 0), Err(EsikRed::IkiEsikDeSifir));
    }

    #[test]
    fn asla_dolmayacak_sayi_esigi_reddedilir() {
        let r = Esik::yeni(MAKS_PARTI + 1, 0, 0);
        assert!(matches!(r, Err(EsikRed::SayiEsigiTavaninUstunde { .. })));
    }

    #[test]
    fn cok_uzun_pencere_reddedilir() {
        let r = Esik::yeni(0, MAKS_PENCERE_SANIYE + 1, 0);
        assert!(matches!(r, Err(EsikRed::PencereCokUzun { .. })));
    }

    #[test]
    fn pencereyi_asan_soguma_reddedilir() {
        let r = Esik::yeni(0, 100, 101);
        assert!(matches!(r, Err(EsikRed::SogumaPencereyiAsiyor { .. })));
    }

    /// Bu modulun tek cumlesi: **zaman gecmesi veri degildir.**
    #[test]
    fn bos_parti_sure_dolsa_bile_egitilmez() {
        let esik = Esik::yalniz_sure(10).unwrap();
        let mut b = Biriktirici::yeni(esik);
        let karar = b.degerlendir(1_000_000);
        assert!(!karar.acildi(), "bos parti uzerinde tur acildi");
        assert_eq!(b.acilan_tur(), 0);
    }

    #[test]
    fn bos_red_sayilir_gizlenmez() {
        // Pencere 10 sn; her 100 sn'de bir bakiliyor. Ilk bakis pencereyi
        // baslatir (henuz dolmamistir), sonraki dortu dolu pencere uzerinde
        // bos parti bulur ve reddeder.
        let esik = Esik::yalniz_sure(10).unwrap();
        let mut b = Biriktirici::yeni(esik);
        for t in 0..5 {
            let _ = b.degerlendir(t * 100);
        }
        assert_eq!(b.reddedilen_bos(), 4, "bos red sayaci calismiyor");
        assert_eq!(b.acilan_tur(), 0, "bos parti uzerinde tur acildi");
    }

    /// Bos red sayaci "kac kez reddedildi"yi sayar, "kac kez bakildi"yi degil.
    #[test]
    fn bos_red_her_bakista_degil_her_pencerede_sayilir() {
        let esik = Esik::yalniz_sure(100).unwrap();
        let mut b = Biriktirici::yeni(esik);
        for t in 0..50 {
            let _ = b.degerlendir(t);
        }
        assert_eq!(b.reddedilen_bos(), 0, "pencere dolmadan red sayildi");
        let _ = b.degerlendir(100);
        assert_eq!(b.reddedilen_bos(), 1);
        let _ = b.degerlendir(101);
        assert_eq!(b.reddedilen_bos(), 1, "ayni pencere iki kez sayildi");
    }

    #[test]
    fn sayi_esigi_tam_esitlikte_ates_eder() {
        let esik = Esik::yalniz_sayi(3).unwrap();
        let mut b = Biriktirici::yeni(esik);
        for i in 0..2 {
            b.ekle(&Kayit::yeni(&format!("a{i}"), i, 10).unwrap())
                .unwrap();
            assert!(!b.degerlendir(i).acildi(), "esik dolmadan acildi");
        }
        b.ekle(&Kayit::yeni("a2", 2, 10).unwrap()).unwrap();
        let karar = b.degerlendir(2);
        assert_eq!(karar.tetik(), Some(Tetik::Sayi));
        assert_eq!(karar.parti_kayit(), 3);
        assert_eq!(karar.parti_bayt(), 30);
    }

    #[test]
    fn tur_acilinca_parti_bosalir() {
        let esik = Esik::yalniz_sayi(2).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 0, 10).unwrap()).unwrap();
        b.ekle(&Kayit::yeni("b", 1, 10).unwrap()).unwrap();
        assert!(b.degerlendir(1).acildi());
        assert_eq!(
            b.parti_kayit(),
            0,
            "parti bosalmadi: ayni veri iki tur gorulur"
        );
        assert_eq!(b.parti_bayt(), 0);
    }

    #[test]
    fn sure_esigi_dolu_partide_ates_eder() {
        let esik = Esik::yalniz_sure(10).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 100, 10).unwrap()).unwrap();
        assert!(!b.degerlendir(105).acildi(), "pencere dolmadan acildi");
        let karar = b.degerlendir(110);
        assert_eq!(karar.tetik(), Some(Tetik::Sure));
        assert_eq!(karar.gecen_saniye(), 10);
    }

    /// Hangi yuzun atesledigi kaydedilir: sayi akisa, sure takvime bagimlidir.
    #[test]
    fn iki_esik_ayni_anda_dolarsa_ikisi_raporlanir() {
        let esik = Esik::yeni(2, 10, 0).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 0, 1).unwrap()).unwrap();
        b.ekle(&Kayit::yeni("b", 1, 1).unwrap()).unwrap();
        let karar = b.degerlendir(10);
        assert_eq!(karar.tetik(), Some(Tetik::Ikisi));
        assert_eq!(Tetik::Ikisi.ad(), "ikisi");
    }

    #[test]
    fn saat_geri_giderse_reddedilir() {
        let esik = Esik::yalniz_sayi(10).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 100, 1).unwrap()).unwrap();
        let r = b.ekle(&Kayit::yeni("b", 99, 1).unwrap());
        assert_eq!(
            r,
            Err(BiriktirmeRed::SaatGeriGitti {
                onceki: 100,
                gelen: 99
            })
        );
    }

    #[test]
    fn reddedilen_kayit_partiyi_degistirmez() {
        let esik = Esik::yalniz_sayi(10).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 100, 42).unwrap()).unwrap();
        let once_kayit = b.parti_kayit();
        let once_bayt = b.parti_bayt();
        assert!(b.ekle(&Kayit::yeni("b", 99, 99).unwrap()).is_err());
        assert_eq!(b.parti_kayit(), once_kayit, "red kayit sayisini degistirdi");
        assert_eq!(b.parti_bayt(), once_bayt, "red bayt sayisini degistirdi");
    }

    #[test]
    fn ayni_damga_kabul_edilir_geri_gitme_degil() {
        let esik = Esik::yalniz_sayi(10).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 100, 1).unwrap()).unwrap();
        assert!(b.ekle(&Kayit::yeni("b", 100, 1).unwrap()).is_ok());
    }

    #[test]
    fn tekrar_eden_kimlik_esigi_hak_etmeden_doldurmaz() {
        let esik = Esik::yalniz_sayi(2).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("ayni", 0, 1).unwrap()).unwrap();
        let r = b.ekle(&Kayit::yeni("ayni", 1, 1).unwrap());
        assert!(matches!(r, Err(BiriktirmeRed::TekrarEdenKimlik { .. })));
        assert!(!b.degerlendir(1).acildi(), "tekrar esigi doldurdu");
    }

    #[test]
    fn kimliksiz_kayit_kurulamaz() {
        assert_eq!(Kayit::yeni("   ", 0, 1), Err(BiriktirmeRed::KimliksizKayit));
    }

    /// Soguma disiplinle degil **yapisal olarak** uygulanir.
    #[test]
    fn soguma_ikinci_turu_engeller() {
        let esik = Esik::yeni(1, 100, 50).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 0, 1).unwrap()).unwrap();
        assert!(b.degerlendir(0).acildi());
        b.ekle(&Kayit::yeni("b", 1, 1).unwrap()).unwrap();
        let karar = b.degerlendir(10);
        assert!(!karar.acildi(), "soguma icinde ikinci tur acildi");
        assert_eq!(karar.soguma_kalan(), 40);
        assert_eq!(b.reddedilen_soguma(), 1);
    }

    #[test]
    fn soguma_bitince_tur_acilir() {
        let esik = Esik::yeni(1, 100, 50).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 0, 1).unwrap()).unwrap();
        assert!(b.degerlendir(0).acildi());
        b.ekle(&Kayit::yeni("b", 1, 1).unwrap()).unwrap();
        assert!(!b.degerlendir(49).acildi());
        assert!(b.degerlendir(50).acildi(), "soguma bittigi halde acilmadi");
        assert_eq!(b.acilan_tur(), 2);
    }

    #[test]
    fn soguma_reddi_partiyi_bosaltmaz() {
        let esik = Esik::yeni(1, 100, 50).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 0, 7).unwrap()).unwrap();
        assert!(b.degerlendir(0).acildi());
        b.ekle(&Kayit::yeni("b", 1, 7).unwrap()).unwrap();
        let _ = b.degerlendir(10);
        assert_eq!(b.parti_kayit(), 1, "soguma reddi partiyi yedi");
        assert_eq!(b.parti_bayt(), 7);
    }

    #[test]
    fn gecen_sure_ilk_kayittan_olculur() {
        let esik = Esik::yalniz_sure(1000).unwrap();
        let mut b = Biriktirici::yeni(esik);
        b.ekle(&Kayit::yeni("a", 500, 1).unwrap()).unwrap();
        let karar = b.degerlendir(700);
        assert_eq!(karar.gecen_saniye(), 200);
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Biriktirici::parametre_sayisi(), 0);
    }

    #[test]
    fn yuruyus_belirlenimci() {
        let esik = Esik::yeni(8, 60, 5).unwrap();
        let a = tetik_yuruyusu_olc(esik, 500, 20, 0x7E11_CA01);
        let b = tetik_yuruyusu_olc(esik, 500, 20, 0x7E11_CA01);
        assert_eq!(a, b, "ayni tohum farkli sonuc verdi");
    }

    #[test]
    fn yuruyuste_her_tur_dolu_partiyle_acilir() {
        let esik = Esik::yeni(8, 60, 5).unwrap();
        let olcum = tetik_yuruyusu_olc(esik, 2000, 15, 0x51AA_7C01);
        assert_eq!(
            olcum.sayiyla + olcum.sureyle + olcum.ikisiyle,
            olcum.acilan_tur,
            "tetik dagilimi tur sayisiyla tutmuyor"
        );
        assert!(olcum.acilan_tur > 0, "hic tur acilmadi: olcum bos");
        assert!(
            olcum.kayit >= olcum.acilan_tur,
            "turlar kayitlardan cok: bos parti egitilmis olmali"
        );
    }

    #[test]
    fn yalniz_sure_esiginde_sayi_tetigi_gorulmez() {
        let esik = Esik::yalniz_sure(30).unwrap();
        let olcum = tetik_yuruyusu_olc(esik, 1000, 10, 0x5E_11_00_22);
        assert_eq!(olcum.sayiyla, 0, "kapali yuz ates etti");
        assert_eq!(olcum.ikisiyle, 0, "kapali yuz ates etti");
    }

    #[test]
    fn yalniz_sayi_esiginde_sure_tetigi_gorulmez() {
        let esik = Esik::yalniz_sayi(5).unwrap();
        let olcum = tetik_yuruyusu_olc(esik, 1000, 10, 0x5E_11_00_22);
        assert_eq!(olcum.sureyle, 0, "kapali yuz ates etti");
        assert_eq!(olcum.ikisiyle, 0, "kapali yuz ates etti");
    }
}
