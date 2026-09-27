//! 6.9 — cok dilli dallanma ve model corbasi (model soup).
//!
//! Tek cumle: **ortak atasi olmayan modeller ortalanmaz.**
//!
//! Model corbasi - birden cok ince ayarli modelin agirliklarini ortalamak -
//! ancak modeller ayni baslangictan ayrilmissa anlamli. Ortak atasi olmayan
//! iki agi ortalamak, iki farkli dili harf harf ortalamaya benzer: sonuc bir
//! sey gibi gorunur, hicbir sey degildir. Kayip egrisi bunu soylemez, cunku
//! kayip her zaman bir sayi verir.
//!
//! Bu yuzden burada ortalama almadan **once** soy sorulur. [`Corba::karistir`]
//! ilk olarak ortak atayi arar; yoksa `OrtakAtaYok` ile durur ve hicbir
//! aritmetik yapilmaz.
//!
//! Uc kural daha:
//!
//! - **Agirliklar toplami bir.** Toplami bir olmayan bir karisim, ortalama
//!   degil olceklemedir; iki isi tek fonksiyona sikistirmak, olceklemeyi
//!   kazara yapmayi mumkun kilar. Tolerans beyanli.
//! - **Negatif agirlik yok.** "Bu dalin tersini al" demek, corba degil
//!   cikarma; farkli bir islem, farkli bir ad ister.
//! - **Sekil esitligi.** Farkli uzunluktaki agirlik vektorleri, kisa olani
//!   sifirla doldurularak birlestirilmez; `SekilUyusmuyor` ile reddedilir.
//!
//! Dil dallanmasi tarafinda kural daha basit ama ayni cinsten: bir dal bir
//! **dil etiketi** tasir ve iki dal ayni etiketi tasiyorsa bu bir hata degil,
//! ama `dil_dagilimi()` ile **gorunur** olmak zorunda. Tek dile yigilmis bir
//! corbayi "cok dilli" diye raporlamak, olcumun tasimadigi bir iddia olurdu.

use core::fmt;

/// Agirliklar toplaminin birden sapabilecegi miktar.
pub const TOPLAM_TOLERANSI: f64 = 1e-9;

/// Karistirma reddedildi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorbaRed {
    /// Hic dal yok.
    DalYok,
    /// Dallarin ortak atasi yok: ortalama anlamsiz.
    OrtakAtaYok,
    /// Agirlik sayisi dal sayisiyla uyusmuyor.
    AgirlikSayisiUyusmuyor { dal: usize, agirlik: usize },
    /// Agirliklar toplami bir degil.
    ToplamBirDegil,
    /// Negatif agirlik: bu islem cikarma, corba degil.
    NegatifAgirlik { dal: usize },
    /// Agirlik vektorlerinin uzunlugu farkli.
    SekilUyusmuyor { beklenen: usize, gelen: usize },
    /// Bir agirlik sonlu degil.
    SonluDegil,
    /// Dal adi bos.
    AdsizDal,
    /// Dil etiketi bos: dagilim olculemez.
    DilsizDal { ad: String },
}

impl fmt::Display for CorbaRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DalYok => write!(f, "dal yok: karistirilacak bir sey yok"),
            Self::OrtakAtaYok => {
                write!(f, "ortak ata yok: iliskisiz modeller ortalanmaz")
            }
            Self::AgirlikSayisiUyusmuyor { dal, agirlik } => {
                write!(f, "{dal} dal, {agirlik} agirlik")
            }
            Self::ToplamBirDegil => write!(f, "agirliklar toplami bir degil: bu olcekleme"),
            Self::NegatifAgirlik { dal } => {
                write!(f, "dal {dal} icin negatif agirlik: bu cikarma, corba degil")
            }
            Self::SekilUyusmuyor { beklenen, gelen } => {
                write!(
                    f,
                    "sekil {gelen}, {beklenen} bekleniyor: sifirla doldurulmaz"
                )
            }
            Self::SonluDegil => write!(f, "agirlik sonlu bir sayi degil"),
            Self::AdsizDal => write!(f, "adsiz dal"),
            Self::DilsizDal { ad } => write!(f, "{ad} icin dil etiketi yok"),
        }
    }
}

/// Bir dal: adi, atasi, dili ve agirliklari.
#[derive(Debug, Clone, PartialEq)]
pub struct Dal {
    ad: String,
    ata: String,
    dil: String,
    agirliklar: Vec<f64>,
}

impl Dal {
    /// # Errors
    /// Bos ad, bos dil, bos agirlik vektoru ya da sonlu olmayan agirlik.
    pub fn yeni(ad: &str, ata: &str, dil: &str, agirliklar: &[f64]) -> Result<Self, CorbaRed> {
        if ad.trim().is_empty() {
            return Err(CorbaRed::AdsizDal);
        }
        if dil.trim().is_empty() {
            return Err(CorbaRed::DilsizDal {
                ad: ad.trim().to_string(),
            });
        }
        if agirliklar.is_empty() {
            return Err(CorbaRed::SekilUyusmuyor {
                beklenen: 1,
                gelen: 0,
            });
        }
        if agirliklar.iter().any(|w| !w.is_finite()) {
            return Err(CorbaRed::SonluDegil);
        }
        Ok(Self {
            ad: ad.trim().to_string(),
            ata: ata.trim().to_string(),
            dil: dil.trim().to_string(),
            agirliklar: agirliklar.to_vec(),
        })
    }

    #[must_use]
    pub fn ad(&self) -> &str {
        &self.ad
    }

    #[must_use]
    pub fn ata(&self) -> &str {
        &self.ata
    }

    #[must_use]
    pub fn dil(&self) -> &str {
        &self.dil
    }

    #[must_use]
    pub fn agirliklar(&self) -> &[f64] {
        &self.agirliklar
    }

    #[must_use]
    pub fn boy(&self) -> usize {
        self.agirliklar.len()
    }
}

/// Karisimin sonucu, **gerekcesiyle**.
#[derive(Debug, Clone, PartialEq)]
pub struct Karisim {
    agirliklar: Vec<f64>,
    ata: String,
    dal_sayisi: usize,
    dil_sayisi: usize,
}

impl Karisim {
    #[must_use]
    pub fn agirliklar(&self) -> &[f64] {
        &self.agirliklar
    }

    /// Karisimin dayandigi ortak ata.
    #[must_use]
    pub fn ata(&self) -> &str {
        &self.ata
    }

    #[must_use]
    pub const fn dal_sayisi(&self) -> usize {
        self.dal_sayisi
    }

    /// Karisimda kac **farkli** dil var. Tek dile yigilmis bir corbayi
    /// "cok dilli" diye raporlamayi imkansiz kilar.
    #[must_use]
    pub const fn dil_sayisi(&self) -> usize {
        self.dil_sayisi
    }

    /// Gercekten cok dilli mi? Iddia degil, sayidan turetilmis.
    #[must_use]
    pub const fn cok_dilli(&self) -> bool {
        self.dil_sayisi > 1
    }
}

/// Dal kumesi ve karistirici.
#[derive(Debug, Clone, Default)]
pub struct Corba {
    dallar: Vec<Dal>,
}

impl Corba {
    #[must_use]
    pub const fn yeni() -> Self {
        Self { dallar: Vec::new() }
    }

    /// Bu modul parametre tutmaz: karistirma bir islem, ogrenilen bir sey
    /// degil. (Karistirilan agirliklar cagirana ait; bu tip onlari tutmaz,
    /// sadece bir sonuc uretir.)
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    #[must_use]
    pub fn sayi(&self) -> usize {
        self.dallar.len()
    }

    /// Dal ekler.
    ///
    /// # Errors
    /// Sekil ilk dalla uyusmuyorsa reddeder: kisa olan sifirla doldurulmaz.
    pub fn ekle(&mut self, dal: Dal) -> Result<(), CorbaRed> {
        if let Some(ilk) = self.dallar.first() {
            if ilk.boy() != dal.boy() {
                return Err(CorbaRed::SekilUyusmuyor {
                    beklenen: ilk.boy(),
                    gelen: dal.boy(),
                });
            }
        }
        self.dallar.push(dal);
        Ok(())
    }

    /// Butun dallarin ortak atasi, varsa.
    #[must_use]
    pub fn ortak_ata(&self) -> Option<&str> {
        let ilk = self.dallar.first()?;
        if self.dallar.iter().all(|d| d.ata == ilk.ata) {
            Some(&ilk.ata)
        } else {
            None
        }
    }

    /// Dil basina dal sayisi, dil adina gore siralanmis.
    #[must_use]
    pub fn dil_dagilimi(&self) -> Vec<(String, usize)> {
        let mut diller: Vec<&str> = self.dallar.iter().map(Dal::dil).collect();
        diller.sort_unstable();
        diller.dedup();
        diller
            .into_iter()
            .map(|d| {
                (
                    d.to_string(),
                    self.dallar.iter().filter(|x| x.dil == d).count(),
                )
            })
            .collect()
    }

    /// Esit agirlikli corba.
    ///
    /// # Errors
    /// [`Corba::karistir`] ile ayni redler.
    pub fn esit_karistir(&self) -> Result<Karisim, CorbaRed> {
        if self.dallar.is_empty() {
            return Err(CorbaRed::DalYok);
        }
        let n = self.dallar.len();
        // `n >= 1` ve `u32` donusumu kayipsiz: dal sayisi bellekte tutulan bir
        // vektorun uzunlugu.
        let pay = 1.0 / f64::from(u32::try_from(n).unwrap_or(u32::MAX));
        let agirliklar = vec![pay; n];
        self.karistir(&agirliklar)
    }

    /// Verilen agirliklarla karistirir.
    ///
    /// **Ilk** kontrol ortak atadir: yoksa hicbir aritmetik yapilmaz. Sirasi
    /// ters olsaydi, iliskisiz modellerin ortalamasi hesaplanir ve ancak
    /// sonra atilirdi - yani bir kere hesaplanmis olurdu, ve hesaplanan sey
    /// bir gun loglanirdi.
    ///
    /// # Errors
    /// Ortak ata yoksa, agirlik sayisi tutmuyorsa, toplam bir degilse,
    /// negatif ya da sonsuz agirlik varsa.
    pub fn karistir(&self, agirliklar: &[f64]) -> Result<Karisim, CorbaRed> {
        if self.dallar.is_empty() {
            return Err(CorbaRed::DalYok);
        }
        let Some(ata) = self.ortak_ata() else {
            return Err(CorbaRed::OrtakAtaYok);
        };
        let ata = ata.to_string();

        if agirliklar.len() != self.dallar.len() {
            return Err(CorbaRed::AgirlikSayisiUyusmuyor {
                dal: self.dallar.len(),
                agirlik: agirliklar.len(),
            });
        }
        for (i, w) in agirliklar.iter().enumerate() {
            if !w.is_finite() {
                return Err(CorbaRed::SonluDegil);
            }
            if *w < 0.0 {
                return Err(CorbaRed::NegatifAgirlik { dal: i });
            }
        }
        let toplam: f64 = agirliklar.iter().sum();
        if (toplam - 1.0).abs() > TOPLAM_TOLERANSI {
            return Err(CorbaRed::ToplamBirDegil);
        }

        let boy = self.dallar.first().map_or(0, Dal::boy);
        let mut cikti = vec![0.0f64; boy];
        for (dal, w) in self.dallar.iter().zip(agirliklar.iter()) {
            if dal.boy() != boy {
                return Err(CorbaRed::SekilUyusmuyor {
                    beklenen: boy,
                    gelen: dal.boy(),
                });
            }
            for (c, d) in cikti.iter_mut().zip(dal.agirliklar.iter()) {
                *c += w * d;
            }
        }

        let dil_sayisi = self.dil_dagilimi().len();
        Ok(Karisim {
            agirliklar: cikti,
            ata,
            dal_sayisi: self.dallar.len(),
            dil_sayisi,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corba_kur() -> Corba {
        let mut c = Corba::yeni();
        c.ekle(Dal::yeni("tr", "kok", "tr", &[1.0, 2.0, 3.0]).unwrap())
            .unwrap();
        c.ekle(Dal::yeni("en", "kok", "en", &[3.0, 2.0, 1.0]).unwrap())
            .unwrap();
        c
    }

    #[test]
    fn adsiz_dal_kurulamaz() {
        assert_eq!(Dal::yeni("", "k", "tr", &[1.0]), Err(CorbaRed::AdsizDal));
    }

    #[test]
    fn dilsiz_dal_kurulamaz() {
        assert!(matches!(
            Dal::yeni("a", "k", "  ", &[1.0]),
            Err(CorbaRed::DilsizDal { .. })
        ));
    }

    #[test]
    fn bos_agirlik_vektoru_reddedilir() {
        assert!(matches!(
            Dal::yeni("a", "k", "tr", &[]),
            Err(CorbaRed::SekilUyusmuyor { .. })
        ));
    }

    #[test]
    fn sonsuz_agirlikli_dal_reddedilir() {
        assert_eq!(
            Dal::yeni("a", "k", "tr", &[f64::NAN]),
            Err(CorbaRed::SonluDegil)
        );
    }

    /// Bu modulun tek cumlesi.
    #[test]
    fn ortak_atasi_olmayan_dallar_ortalanmaz() {
        let mut c = Corba::yeni();
        c.ekle(Dal::yeni("a", "kok-1", "tr", &[1.0, 1.0]).unwrap())
            .unwrap();
        c.ekle(Dal::yeni("b", "kok-2", "en", &[2.0, 2.0]).unwrap())
            .unwrap();
        assert_eq!(c.ortak_ata(), None);
        assert_eq!(c.esit_karistir(), Err(CorbaRed::OrtakAtaYok));
    }

    /// Ortak ata kontrolu **ilk** sirada: bozuk agirliklarla ve iliskisiz
    /// dallarla cagrildiginda "ortak ata yok" demeli, aritmetige girmemeli.
    #[test]
    fn ortak_ata_kontrolu_agirlik_kontrolunu_onceler() {
        let mut c = Corba::yeni();
        c.ekle(Dal::yeni("a", "kok-1", "tr", &[1.0]).unwrap())
            .unwrap();
        c.ekle(Dal::yeni("b", "kok-2", "en", &[2.0]).unwrap())
            .unwrap();
        assert_eq!(c.karistir(&[0.9, 0.9]), Err(CorbaRed::OrtakAtaYok));
    }

    #[test]
    fn esit_karisim_ortalamadir() {
        let c = corba_kur();
        let k = c.esit_karistir().unwrap();
        assert_eq!(k.agirliklar(), &[2.0, 2.0, 2.0]);
        assert_eq!(k.ata(), "kok");
        assert_eq!(k.dal_sayisi(), 2);
    }

    #[test]
    fn agirlikli_karisim() {
        let c = corba_kur();
        let k = c.karistir(&[0.25, 0.75]).unwrap();
        assert!((k.agirliklar()[0] - 2.5).abs() < 1e-12);
        assert!((k.agirliklar()[1] - 2.0).abs() < 1e-12);
        assert!((k.agirliklar()[2] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn toplami_bir_olmayan_agirlik_reddedilir() {
        let c = corba_kur();
        assert_eq!(c.karistir(&[0.5, 0.4]), Err(CorbaRed::ToplamBirDegil));
        assert_eq!(c.karistir(&[0.6, 0.6]), Err(CorbaRed::ToplamBirDegil));
    }

    #[test]
    fn negatif_agirlik_cikarmadir_corba_degil() {
        let c = corba_kur();
        assert_eq!(
            c.karistir(&[-0.5, 1.5]),
            Err(CorbaRed::NegatifAgirlik { dal: 0 })
        );
    }

    #[test]
    fn agirlik_sayisi_dal_sayisiyla_ayni_olmali() {
        let c = corba_kur();
        assert_eq!(
            c.karistir(&[1.0]),
            Err(CorbaRed::AgirlikSayisiUyusmuyor { dal: 2, agirlik: 1 })
        );
    }

    /// Kisa olan sifirla **doldurulmaz**.
    #[test]
    fn farkli_sekilli_dal_eklenemez() {
        let mut c = corba_kur();
        let r = c.ekle(Dal::yeni("kisa", "kok", "de", &[1.0]).unwrap());
        assert_eq!(
            r,
            Err(CorbaRed::SekilUyusmuyor {
                beklenen: 3,
                gelen: 1
            })
        );
        assert_eq!(c.sayi(), 2, "reddedilen dal yine de eklendi");
    }

    #[test]
    fn bos_corba_karistirilamaz() {
        assert_eq!(Corba::yeni().esit_karistir(), Err(CorbaRed::DalYok));
        assert_eq!(Corba::yeni().karistir(&[]), Err(CorbaRed::DalYok));
    }

    #[test]
    fn tek_dal_kendisidir() {
        let mut c = Corba::yeni();
        c.ekle(Dal::yeni("tek", "kok", "tr", &[1.0, 2.0]).unwrap())
            .unwrap();
        let k = c.esit_karistir().unwrap();
        assert_eq!(k.agirliklar(), &[1.0, 2.0]);
        assert!(!k.cok_dilli(), "tek dil cok dilli sayildi");
    }

    /// Tek dile yigilmis bir corbayi "cok dilli" diye raporlamak imkansiz.
    #[test]
    fn tek_dile_yigilmis_corba_cok_dilli_degildir() {
        let mut c = Corba::yeni();
        for i in 0..5 {
            c.ekle(Dal::yeni(&format!("d{i}"), "kok", "tr", &[1.0]).unwrap())
                .unwrap();
        }
        let k = c.esit_karistir().unwrap();
        assert_eq!(k.dal_sayisi(), 5);
        assert_eq!(k.dil_sayisi(), 1);
        assert!(!k.cok_dilli());
    }

    #[test]
    fn dil_dagilimi_gorunur() {
        let mut c = corba_kur();
        c.ekle(Dal::yeni("tr2", "kok", "tr", &[0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        assert_eq!(
            c.dil_dagilimi(),
            vec![("en".to_string(), 1), ("tr".to_string(), 2)]
        );
    }

    #[test]
    fn cok_dilli_karisim_isaretlenir() {
        let k = corba_kur().esit_karistir().unwrap();
        assert!(k.cok_dilli());
        assert_eq!(k.dil_sayisi(), 2);
    }

    #[test]
    fn karisim_belirlenimci() {
        let c = corba_kur();
        assert_eq!(c.karistir(&[0.3, 0.7]), c.karistir(&[0.3, 0.7]));
    }

    #[test]
    fn sifir_agirlikli_dal_katkisizdir() {
        let c = corba_kur();
        let k = c.karistir(&[0.0, 1.0]).unwrap();
        assert_eq!(k.agirliklar(), &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn tolerans_icinde_toplam_kabul() {
        let c = corba_kur();
        assert!(c.karistir(&[0.5, 0.5 + TOPLAM_TOLERANSI / 2.0]).is_ok());
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Corba::parametre_sayisi(), 0);
    }
}
