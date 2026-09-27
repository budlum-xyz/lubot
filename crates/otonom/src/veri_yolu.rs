//! 6.2 — veri yolu ayrimi: omurga MLM mi, karar basligi mi?
//!
//! Bir kayit iki yerden birine gider ve **kendi sekli** karar verir, kaydin
//! ya da toplayanin dilegi degil. Ayrim su:
//!
//! - **Omurga (MLM)** dil ogrenir. Ona giden sey duz metindir: etiketi
//!   olmayan, yeterince uzun, maskelenecek jetonu bulunan bir govde.
//! - **Karar basligi** kapali bir kume uzerinde siniflandirir. Ona giden sey
//!   `(govde, etiket)` ciftidir ve etiket **beyan edilmis** kumede olmak
//!   zorundadir.
//!
//! Ucuncu bir yol yok, ve olmamasi bilincli: "ikisine de" demek, etiketli
//! veriyi hem denetimli hem denetimsiz saymak, yani ayni orneyi iki kez
//! kredilendirmek olurdu. Bir kayit tam olarak bir yola gider ya da
//! **adlandirilmis bir redle** hicbir yere gitmez.
//!
//! Reddin adlandirilmasi burada ozellikle onemli, cunku alternatifi cok
//! cazip: "etiketi tanimadim, omurgaya atayim." O davranis, karar basliginin
//! kapali kumesini sessizce delen tek seydir - etiket kumesine girmeyen her
//! sey dil verisi olarak birikir ve kimse fark etmez. Bu yuzden
//! `EtiketTaninmiyor` bir **red**, bir yonlendirme degil.
//!
//! Modul metni okur ama **anlamaz**: jeton sayimi bosluga gore yapilir, cunku
//! gercek sozlukle sayim `crates/jeton`'un isi ve iki kopya ayrisirdi. Burada
//! olculen sey "yeterince uzun mu" sorusudur, "kac jeton" degil; ikisi ayni
//! sey degil ve bu ayrim rapor edilir.

use core::fmt;

/// Omurgaya gitmek icin gereken en az bosluk-ayrilmis parca sayisi.
/// Tek kelimelik bir govdede maskelenecek baglam yoktur: MLM kaybi tanimli
/// olur ama ogretici degildir, ve "tanimli ama ogretici degil" tam olarak
/// sessizce birikip olcumu kirleten sinifitir.
pub const EN_AZ_PARCA: usize = 4;

/// Bir govdenin kabul edilen en buyuk boyu (64 KiB), `sema_cozucu`'nun cikti
/// tavaniyla ayni sayi. Ayni olmasi tesaduf degil: uretilemeyecek uzunlukta
/// bir govde uzerinde egitmek, olculmeyen bir rejimde egitmektir.
pub const MAKS_GOVDE_BAYT: usize = 64 * 1024;

/// Kayit hicbir yola gitmedi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YolRed {
    /// Govde bos ya da yalniz bosluk.
    BosGovde,
    /// Govde tavandan uzun.
    GovdeCokUzun { bayt: usize, tavan: usize },
    /// Parca sayisi omurga icin yetersiz ve etiket de yok.
    CokKisa { parca: usize, en_az: usize },
    /// Etiket verilmis ama beyan edilen kumede degil. **Omurgaya dusurulmez.**
    EtiketTaninmiyor { etiket: String },
    /// Etiket kumesi bos kurulmus: kapali kume bos olamaz.
    EtiketKumesiBos,
    /// Govde gecerli UTF-8 degil (bayt girisinde).
    GecersizUtf8 { konum: usize },
}

impl fmt::Display for YolRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BosGovde => write!(f, "bos govde hicbir yola gitmez"),
            Self::GovdeCokUzun { bayt, tavan } => {
                write!(f, "govde {bayt} bayt, tavan {tavan}")
            }
            Self::CokKisa { parca, en_az } => {
                write!(f, "{parca} parca, omurga icin en az {en_az} gerekiyor")
            }
            Self::EtiketTaninmiyor { etiket } => {
                write!(f, "etiket {etiket} kapali kumede yok: omurgaya dusurulmez")
            }
            Self::EtiketKumesiBos => write!(f, "kapali etiket kumesi bos kurulamaz"),
            Self::GecersizUtf8 { konum } => write!(f, "bayt {konum}: gecersiz UTF-8"),
        }
    }
}

/// Kaydin gidecegi yol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Yol {
    /// Omurga: maskeli dil modeli.
    Omurga,
    /// Karar basligi: kapali kume siniflandirmasi.
    KararBasligi,
}

impl Yol {
    #[must_use]
    pub const fn ad(self) -> &'static str {
        match self {
            Self::Omurga => "omurga",
            Self::KararBasligi => "karar-basligi",
        }
    }
}

/// Karar basliginin kapali etiket kumesi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EtiketKumesi {
    etiketler: Vec<String>,
}

impl EtiketKumesi {
    /// Dogrulayan yapici: bos kume ve tekrar reddedilir.
    ///
    /// # Errors
    /// Kume bossa `EtiketKumesiBos` doner. Tekrar eden etiketler **sessizce
    /// tekillestirilmez**; tekillestirme, kapali kumenin kac elemani oldugu
    /// sorusunu bulanik yapardi, o yuzden tekrar da bos kume gibi bir kurulum
    /// hatasi sayilir ve ilk tekrar reddin icinde adlandirilir.
    pub fn yeni(etiketler: &[&str]) -> Result<Self, YolRed> {
        if etiketler.is_empty() {
            return Err(YolRed::EtiketKumesiBos);
        }
        let mut toplanan: Vec<String> = Vec::with_capacity(etiketler.len());
        for e in etiketler {
            let kirpik = e.trim();
            if kirpik.is_empty() {
                return Err(YolRed::EtiketKumesiBos);
            }
            if toplanan.iter().any(|t| t == kirpik) {
                return Err(YolRed::EtiketTaninmiyor {
                    etiket: format!("{kirpik} (tekrar)"),
                });
            }
            toplanan.push(kirpik.to_string());
        }
        Ok(Self {
            etiketler: toplanan,
        })
    }

    #[must_use]
    pub fn sayi(&self) -> usize {
        self.etiketler.len()
    }

    #[must_use]
    pub fn icerir(&self, etiket: &str) -> bool {
        self.etiketler.iter().any(|e| e == etiket.trim())
    }

    #[must_use]
    pub fn etiketler(&self) -> &[String] {
        &self.etiketler
    }
}

/// Yol karari, gerekcesiyle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YolKarari {
    yol: Yol,
    parca: usize,
    bayt: usize,
    etiket: Option<String>,
}

impl YolKarari {
    #[must_use]
    pub const fn yol(&self) -> Yol {
        self.yol
    }

    #[must_use]
    pub const fn parca(&self) -> usize {
        self.parca
    }

    #[must_use]
    pub const fn bayt(&self) -> usize {
        self.bayt
    }

    #[must_use]
    pub fn etiket(&self) -> Option<&str> {
        self.etiket.as_deref()
    }
}

/// Yol ayirici.
#[derive(Debug, Clone)]
pub struct Ayirici {
    kume: EtiketKumesi,
    omurga: u64,
    karar: u64,
    red: u64,
}

impl Ayirici {
    #[must_use]
    pub const fn yeni(kume: EtiketKumesi) -> Self {
        Self {
            kume,
            omurga: 0,
            karar: 0,
            red: 0,
        }
    }

    #[must_use]
    pub const fn kume(&self) -> &EtiketKumesi {
        &self.kume
    }

    #[must_use]
    pub const fn omurga_sayisi(&self) -> u64 {
        self.omurga
    }

    #[must_use]
    pub const fn karar_sayisi(&self) -> u64 {
        self.karar
    }

    /// Hicbir yola gitmeyen kayitlarin sayisi. **Gizlenmez**: bu sayinin
    /// buyumesi, ya veri kaynaginin ya etiket kumesinin bozuldugunu soyler ve
    /// her ikisi de sessiz kalirsa aylarca fark edilmez.
    #[must_use]
    pub const fn red_sayisi(&self) -> u64 {
        self.red
    }

    /// Bu modul parametre tutmaz: yonlendirme bir kural, ogrenilen bir sey
    /// degil.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    /// Kaydi yola ayirir.
    ///
    /// # Errors
    /// Bos govde, tavan asimi, omurga icin fazla kisa govde ve taninmayan
    /// etiket reddedilir. Taninmayan etiket **omurgaya dusurulmez**.
    pub fn ayir(&mut self, govde: &str, etiket: Option<&str>) -> Result<YolKarari, YolRed> {
        let sonuc = self.karar_ver(govde, etiket);
        match &sonuc {
            Ok(k) => match k.yol {
                Yol::Omurga => self.omurga = self.omurga.saturating_add(1),
                Yol::KararBasligi => self.karar = self.karar.saturating_add(1),
            },
            Err(_) => self.red = self.red.saturating_add(1),
        }
        sonuc
    }

    /// Sayaclara dokunmadan ayni karari verir. Bir kaydin nereye gidecegini
    /// **sormak**, onu yollamakla ayni sey olmamali; olsaydi, bir onizleme
    /// aracı olcumu kirletirdi.
    ///
    /// # Errors
    /// [`Ayirici::ayir`] ile ayni redler.
    pub fn karar_ver(&self, govde: &str, etiket: Option<&str>) -> Result<YolKarari, YolRed> {
        let bayt = govde.len();
        if bayt > MAKS_GOVDE_BAYT {
            return Err(YolRed::GovdeCokUzun {
                bayt,
                tavan: MAKS_GOVDE_BAYT,
            });
        }
        let parcalar: Vec<&str> = govde.split_whitespace().collect();
        if parcalar.is_empty() {
            return Err(YolRed::BosGovde);
        }
        match etiket {
            Some(e) => {
                let kirpik = e.trim();
                if !self.kume.icerir(kirpik) {
                    return Err(YolRed::EtiketTaninmiyor {
                        etiket: kirpik.to_string(),
                    });
                }
                Ok(YolKarari {
                    yol: Yol::KararBasligi,
                    parca: parcalar.len(),
                    bayt,
                    etiket: Some(kirpik.to_string()),
                })
            }
            None => {
                if parcalar.len() < EN_AZ_PARCA {
                    return Err(YolRed::CokKisa {
                        parca: parcalar.len(),
                        en_az: EN_AZ_PARCA,
                    });
                }
                Ok(YolKarari {
                    yol: Yol::Omurga,
                    parca: parcalar.len(),
                    bayt,
                    etiket: None,
                })
            }
        }
    }

    /// Bayt dizisinden ayirir: UTF-8 dogrulamasi **burada** yapilir, cunku
    /// gecersiz bir govde omurgaya da karar basligina da gidemez ve
    /// "en yakin gecerli diziye dusurme" yasak.
    ///
    /// # Errors
    /// Gecersiz UTF-8 icin `GecersizUtf8`, sonrasinda [`Ayirici::ayir`]
    /// redleri.
    pub fn ayir_bayt(&mut self, govde: &[u8], etiket: Option<&str>) -> Result<YolKarari, YolRed> {
        match core::str::from_utf8(govde) {
            Ok(s) => self.ayir(s, etiket),
            Err(e) => {
                self.red = self.red.saturating_add(1);
                Err(YolRed::GecersizUtf8 {
                    konum: e.valid_up_to(),
                })
            }
        }
    }
}

/// Bir ayirma yurumesinin olcumu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YolYuruyusOlcumu {
    pub girdi: u64,
    pub omurga: u64,
    pub karar: u64,
    pub red: u64,
    pub etiketsiz_kisa_red: u64,
    pub taninmayan_etiket_red: u64,
}

impl YolYuruyusOlcumu {
    /// Her girdi tam olarak bir sonuca gitti mi? Bu esitlik bozulursa bir
    /// kayit ya kayboldu ya iki kez sayildi.
    #[must_use]
    pub const fn hesap_kapaniyor(&self) -> bool {
        self.omurga + self.karar + self.red == self.girdi
    }
}

/// Belirlenimci bir karisim uzerinde ayiriciyi olcer.
#[must_use]
pub fn yol_yuruyusu_olc(kume: &EtiketKumesi, adim: u32, tohum: u64) -> YolYuruyusOlcumu {
    let mut a = Ayirici::yeni(kume.clone());
    let mut durum = tohum | 1;
    let mut olcum = YolYuruyusOlcumu {
        girdi: 0,
        omurga: 0,
        karar: 0,
        red: 0,
        etiketsiz_kisa_red: 0,
        taninmayan_etiket_red: 0,
    };
    for i in 0..adim {
        durum = durum
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let zar = (durum >> 33) % 100;
        let kelime = usize::try_from((durum >> 17) % 8).unwrap_or(0) + 1;
        let govde = vec!["sozcuk"; kelime].join(" ");
        let etiket: Option<String> = if zar < 40 {
            let idx = usize::try_from((durum >> 7) % 8).unwrap_or(0);
            if idx < kume.sayi() {
                kume.etiketler().get(idx).cloned()
            } else {
                Some("bilinmeyen".to_string())
            }
        } else {
            None
        };
        let _ = i;
        olcum.girdi += 1;
        match a.ayir(&govde, etiket.as_deref()) {
            Ok(k) => match k.yol() {
                Yol::Omurga => olcum.omurga += 1,
                Yol::KararBasligi => olcum.karar += 1,
            },
            Err(YolRed::CokKisa { .. }) => {
                olcum.red += 1;
                olcum.etiketsiz_kisa_red += 1;
            }
            Err(YolRed::EtiketTaninmiyor { .. }) => {
                olcum.red += 1;
                olcum.taninmayan_etiket_red += 1;
            }
            Err(_) => olcum.red += 1,
        }
    }
    olcum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kume() -> EtiketKumesi {
        EtiketKumesi::yeni(&["evet", "hayir", "bilmiyorum"]).unwrap()
    }

    #[test]
    fn bos_etiket_kumesi_kurulamaz() {
        assert_eq!(EtiketKumesi::yeni(&[]), Err(YolRed::EtiketKumesiBos));
    }

    #[test]
    fn bos_etiket_adi_kurulamaz() {
        assert_eq!(
            EtiketKumesi::yeni(&["evet", "  "]),
            Err(YolRed::EtiketKumesiBos)
        );
    }

    #[test]
    fn tekrar_eden_etiket_sessizce_tekillestirilmez() {
        let r = EtiketKumesi::yeni(&["evet", "evet"]);
        assert!(matches!(r, Err(YolRed::EtiketTaninmiyor { .. })));
    }

    #[test]
    fn etiketsiz_uzun_govde_omurgaya_gider() {
        let mut a = Ayirici::yeni(kume());
        let k = a.ayir("bu govde yeterince uzun bir metin", None).unwrap();
        assert_eq!(k.yol(), Yol::Omurga);
        assert_eq!(k.parca(), 6);
        assert!(k.etiket().is_none());
    }

    #[test]
    fn etiketli_govde_karar_basligina_gider() {
        let mut a = Ayirici::yeni(kume());
        let k = a.ayir("kisa", Some("evet")).unwrap();
        assert_eq!(k.yol(), Yol::KararBasligi);
        assert_eq!(k.etiket(), Some("evet"));
    }

    /// Bu modulun tek cumlesi: **taninmayan etiket omurgaya dusurulmez.**
    #[test]
    fn taninmayan_etiket_omurgaya_dusurulmez() {
        let mut a = Ayirici::yeni(kume());
        let r = a.ayir("bu govde yeterince uzun bir metin", Some("belki"));
        assert_eq!(
            r,
            Err(YolRed::EtiketTaninmiyor {
                etiket: "belki".to_string()
            })
        );
        assert_eq!(a.omurga_sayisi(), 0, "red omurgaya dusmus");
        assert_eq!(a.red_sayisi(), 1);
    }

    #[test]
    fn kisa_etiketsiz_govde_reddedilir() {
        let mut a = Ayirici::yeni(kume());
        let r = a.ayir("cok kisa", None);
        assert_eq!(
            r,
            Err(YolRed::CokKisa {
                parca: 2,
                en_az: EN_AZ_PARCA
            })
        );
    }

    #[test]
    fn kisa_govde_etiketliyse_kabul_edilir() {
        // Karar basligi icin uzunluk olcutu yok: bir etiket zaten denetimdir.
        let mut a = Ayirici::yeni(kume());
        assert!(a.ayir("evet", Some("evet")).is_ok());
    }

    #[test]
    fn bos_govde_reddedilir() {
        let mut a = Ayirici::yeni(kume());
        assert_eq!(a.ayir("   \t\n ", None), Err(YolRed::BosGovde));
        assert_eq!(a.ayir("   ", Some("evet")), Err(YolRed::BosGovde));
    }

    #[test]
    fn tavan_asan_govde_reddedilir() {
        let mut a = Ayirici::yeni(kume());
        let uzun = "a ".repeat(MAKS_GOVDE_BAYT);
        let r = a.ayir(&uzun, None);
        assert!(matches!(r, Err(YolRed::GovdeCokUzun { .. })));
    }

    #[test]
    fn gecersiz_utf8_hicbir_yola_gitmez() {
        let mut a = Ayirici::yeni(kume());
        let r = a.ayir_bayt(b"iyi metin \xff burada", None);
        assert!(matches!(r, Err(YolRed::GecersizUtf8 { .. })));
        assert_eq!(a.omurga_sayisi(), 0);
        assert_eq!(a.red_sayisi(), 1);
    }

    #[test]
    fn karar_ver_sayaclari_kirletmez() {
        let a = Ayirici::yeni(kume());
        let _ = a.karar_ver("bu govde yeterince uzun bir metin", None);
        let _ = a.karar_ver("x", None);
        assert_eq!(a.omurga_sayisi(), 0, "onizleme olcumu kirletti");
        assert_eq!(a.red_sayisi(), 0, "onizleme olcumu kirletti");
    }

    #[test]
    fn karar_ver_ile_ayir_ayni_karari_verir() {
        let mut a = Ayirici::yeni(kume());
        let ornekler: [(&str, Option<&str>); 6] = [
            ("bu govde yeterince uzun bir metin", None),
            ("cok kisa", None),
            ("kisa", Some("evet")),
            ("kisa", Some("belki")),
            ("", None),
            ("  bir iki uc dort  ", None),
        ];
        for (g, e) in ornekler {
            let sessiz = a.karar_ver(g, e);
            let sayan = a.ayir(g, e);
            assert_eq!(sessiz, sayan, "onizleme ile gercek karar ayrildi: {g:?}");
        }
    }

    #[test]
    fn etiket_bosluklari_kirpilir() {
        let mut a = Ayirici::yeni(kume());
        let k = a.ayir("kisa", Some("  evet  ")).unwrap();
        assert_eq!(k.etiket(), Some("evet"));
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Ayirici::parametre_sayisi(), 0);
    }

    #[test]
    fn hesap_kapaniyor() {
        let olcum = yol_yuruyusu_olc(&kume(), 2000, 0x4E_D1_5A_01);
        assert!(
            olcum.hesap_kapaniyor(),
            "kayit kayboldu ya da iki kez sayildi: {olcum:?}"
        );
        assert_eq!(olcum.girdi, 2000);
    }

    #[test]
    fn yuruyus_belirlenimci() {
        let a = yol_yuruyusu_olc(&kume(), 500, 0x4E_D1_5A_01);
        let b = yol_yuruyusu_olc(&kume(), 500, 0x4E_D1_5A_01);
        assert_eq!(a, b);
    }

    #[test]
    fn iki_yol_da_gercekten_kullaniliyor() {
        // Kontrol: olcum tek yola yigilirsa ayrim olculmemis olur.
        let olcum = yol_yuruyusu_olc(&kume(), 2000, 0x4E_D1_5A_01);
        assert!(olcum.omurga > 0, "omurga yolu hic kullanilmadi");
        assert!(olcum.karar > 0, "karar yolu hic kullanilmadi");
        assert!(
            olcum.taninmayan_etiket_red > 0,
            "taninmayan etiket hic denenmedi: red yolu olculmedi"
        );
    }
}
