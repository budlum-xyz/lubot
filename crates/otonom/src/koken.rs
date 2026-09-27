//! 6.8 — katki provenance defteri (**odulsuz**).
//!
//! Tek cumle: **katki kaydedilir, odullendirilmez.**
//!
//! Direktifte bu maddenin yaninda parantez icinde "odulsuz" yaziyor ve bu
//! parantez modulun tamamini belirliyor. Bir katki defteri, uzerine bir odul
//! alani eklendigi anda bir muhasebe sistemine donusur; muhasebe sistemi de
//! oyunlastirilir. Bu depoda defterin isi tek: **bir kaydin nereden geldigini
//! sonradan sorulabilir kilmak.**
//!
//! Bu yuzden burada yapisal olarak imkansiz olan seyler var:
//!
//! - Bir katkiya sayisal bir **deger** iliştirilemez. `Katki`'da miktar,
//!   puan, agirlik, pay alani yok ve olmamasi kasitli.
//! - Katkicilar **siralanamaz**. `Defter` bir liderlik tablosu vermez; verdigi
//!   tek toplu sayi `katkici_sayisi`, ve o da tekil kimlik sayisidir.
//! - Bir kaynak **silinemez**. Kayit eklenir; gecmis duzeltilmez.
//!
//! Defterin gercek isi *soru cevaplamak*: "bu kayit nereden geldi?",
//! "bu lisans altinda kac kayit var?", "su kaynaktan gelen kayitlar hangi
//! turlara girdi?" Bunlar denetim sorulari, tesvik sorulari degil.
//!
//! Lisans burada **beyan**dir ve kapali bir kumedir: taninmayan bir lisans
//! reddedilir, "bilinmiyor"a dusurulmez. K2 (korpus yalniz budlum-xyz yuzeyi)
//! tam olarak bu redle uygulanir - yuzey disindan gelen bir kayit deftere
//! **giremez**, girip sonra filtrelenmez.

use core::fmt;

/// Kabul edilen lisans beyanlari. Kume **kapali**.
pub const LISANSLAR: [&str; 4] = ["kendi-agac", "cc0-budlum", "mit-budlum", "apache2-budlum"];

/// Kabul edilen kaynak yuzeyleri (K2). Kume **kapali**.
pub const YUZEYLER: [&str; 3] = ["budlum-xyz/lubot", "budlum-xyz/workspace", "zincir-kayit"];

/// Deftere yazilamadi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KokenRed {
    /// Kayit kimligi bos.
    KimliksizKayit,
    /// Lisans kapali kumede degil. **"bilinmiyor"a dusurulmez.**
    LisansTaninmiyor { lisans: String },
    /// Kaynak yuzeyi kapali kumede degil (K2 ihlali).
    YuzeyDisiKaynak { kaynak: String },
    /// Katkici kimligi bos.
    KimliksizKatkici,
    /// Ayni kayit kimligi iki kez.
    KayitTekrar { kimlik: String },
    /// Bilinmeyen kayit sorgulandi.
    KayitYok { kimlik: String },
}

impl fmt::Display for KokenRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KimliksizKayit => write!(f, "kimliksiz kayit deftere giremez"),
            Self::LisansTaninmiyor { lisans } => {
                write!(
                    f,
                    "lisans {lisans} kapali kumede yok: bilinmiyora dusurulmez"
                )
            }
            Self::YuzeyDisiKaynak { kaynak } => {
                write!(f, "kaynak {kaynak} yuzey disinda (K2): deftere giremez")
            }
            Self::KimliksizKatkici => write!(f, "kimliksiz katkici deftere giremez"),
            Self::KayitTekrar { kimlik } => write!(f, "kayit {kimlik} deftere zaten yazilmis"),
            Self::KayitYok { kimlik } => write!(f, "kayit {kimlik} defterde yok"),
        }
    }
}

/// Lisans kapali kumede mi?
#[must_use]
pub fn lisans_tanimli(lisans: &str) -> bool {
    LISANSLAR.contains(&lisans)
}

/// Kaynak yuzeyi kapali kumede mi?
#[must_use]
pub fn yuzey_tanimli(kaynak: &str) -> bool {
    YUZEYLER.contains(&kaynak)
}

/// Bir katki kaydi.
///
/// **Bilincli olarak yok olan alanlar:** miktar, puan, agirlik, pay, odul.
/// Bu tipin alan listesi modulun sozlesmesidir; bir deger alani eklemek
/// kapinin kirmizi yanmasi demek.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Katki {
    kimlik: String,
    katkici: String,
    kaynak: String,
    lisans: String,
    damga: u64,
}

impl Katki {
    /// # Errors
    /// Bos kimlik, bos katkici, taninmayan lisans, yuzey disi kaynak.
    pub fn yeni(
        kimlik: &str,
        katkici: &str,
        kaynak: &str,
        lisans: &str,
        damga: u64,
    ) -> Result<Self, KokenRed> {
        if kimlik.trim().is_empty() {
            return Err(KokenRed::KimliksizKayit);
        }
        if katkici.trim().is_empty() {
            return Err(KokenRed::KimliksizKatkici);
        }
        if !yuzey_tanimli(kaynak.trim()) {
            return Err(KokenRed::YuzeyDisiKaynak {
                kaynak: kaynak.trim().to_string(),
            });
        }
        if !lisans_tanimli(lisans.trim()) {
            return Err(KokenRed::LisansTaninmiyor {
                lisans: lisans.trim().to_string(),
            });
        }
        Ok(Self {
            kimlik: kimlik.trim().to_string(),
            katkici: katkici.trim().to_string(),
            kaynak: kaynak.trim().to_string(),
            lisans: lisans.trim().to_string(),
            damga,
        })
    }

    #[must_use]
    pub fn kimlik(&self) -> &str {
        &self.kimlik
    }

    #[must_use]
    pub fn katkici(&self) -> &str {
        &self.katkici
    }

    #[must_use]
    pub fn kaynak(&self) -> &str {
        &self.kaynak
    }

    #[must_use]
    pub fn lisans(&self) -> &str {
        &self.lisans
    }

    #[must_use]
    pub const fn damga(&self) -> u64 {
        self.damga
    }
}

/// Provenance defteri. Ekleme yapilir, silme yapilmaz.
#[derive(Debug, Clone, Default)]
pub struct Defter {
    kayitlar: Vec<Katki>,
    /// Hangi kaydin hangi turlara girdigi.
    turlar: Vec<(String, u64)>,
    reddedilen: u64,
}

impl Defter {
    #[must_use]
    pub const fn yeni() -> Self {
        Self {
            kayitlar: Vec::new(),
            turlar: Vec::new(),
            reddedilen: 0,
        }
    }

    /// Bu modul parametre tutmaz.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    #[must_use]
    pub fn sayi(&self) -> usize {
        self.kayitlar.len()
    }

    /// Reddedilen yazma denemeleri. **Gizlenmez**: bu sayinin buyumesi K2
    /// yuzeyinin disindan veri geldigini soyler.
    #[must_use]
    pub const fn reddedilen(&self) -> u64 {
        self.reddedilen
    }

    /// # Errors
    /// Kayit kimligi tekrarliysa reddeder.
    pub fn yaz(&mut self, katki: Katki) -> Result<(), KokenRed> {
        if self.kayitlar.iter().any(|k| k.kimlik == katki.kimlik) {
            self.reddedilen = self.reddedilen.saturating_add(1);
            return Err(KokenRed::KayitTekrar {
                kimlik: katki.kimlik,
            });
        }
        self.kayitlar.push(katki);
        Ok(())
    }

    /// Dogrulayip yazar; red sayaci burada da isler.
    ///
    /// # Errors
    /// [`Katki::yeni`] ve [`Defter::yaz`] redleri.
    pub fn yaz_ham(
        &mut self,
        kimlik: &str,
        katkici: &str,
        kaynak: &str,
        lisans: &str,
        damga: u64,
    ) -> Result<(), KokenRed> {
        match Katki::yeni(kimlik, katkici, kaynak, lisans, damga) {
            Ok(k) => self.yaz(k),
            Err(e) => {
                self.reddedilen = self.reddedilen.saturating_add(1);
                Err(e)
            }
        }
    }

    /// Bir kaydin kokenini sorar. Defterin **asil** isi.
    ///
    /// # Errors
    /// Kayit defterde yoksa.
    pub fn koken(&self, kimlik: &str) -> Result<&Katki, KokenRed> {
        self.kayitlar
            .iter()
            .find(|k| k.kimlik == kimlik)
            .ok_or_else(|| KokenRed::KayitYok {
                kimlik: kimlik.to_string(),
            })
    }

    /// Bir kaydi bir tura bagalar. Ayni kayit birden cok tura girebilir.
    ///
    /// # Errors
    /// Kayit defterde yoksa.
    pub fn tura_bagla(&mut self, kimlik: &str, tur: u64) -> Result<(), KokenRed> {
        if self.koken(kimlik).is_err() {
            return Err(KokenRed::KayitYok {
                kimlik: kimlik.to_string(),
            });
        }
        if !self.turlar.iter().any(|(k, t)| k == kimlik && *t == tur) {
            self.turlar.push((kimlik.to_string(), tur));
        }
        Ok(())
    }

    /// Bir kaydin girdigi turlar, artan sirada.
    #[must_use]
    pub fn turlari(&self, kimlik: &str) -> Vec<u64> {
        let mut t: Vec<u64> = self
            .turlar
            .iter()
            .filter(|(k, _)| k == kimlik)
            .map(|(_, t)| *t)
            .collect();
        t.sort_unstable();
        t
    }

    /// Bir turdaki kayitlarin kimlikleri, defter sirasinda.
    #[must_use]
    pub fn turdaki(&self, tur: u64) -> Vec<&str> {
        self.kayitlar
            .iter()
            .filter(|k| self.turlari(&k.kimlik).contains(&tur))
            .map(|k| k.kimlik.as_str())
            .collect()
    }

    /// Bir lisans altindaki kayit sayisi. Denetim sorusu.
    #[must_use]
    pub fn lisans_sayisi(&self, lisans: &str) -> usize {
        self.kayitlar.iter().filter(|k| k.lisans == lisans).count()
    }

    /// Tekil katkici sayisi. **Liderlik tablosu degil**: bir sayi, bir liste
    /// degil, ve katkiciya gore kirilim yok.
    #[must_use]
    pub fn katkici_sayisi(&self) -> usize {
        let mut adlar: Vec<&str> = self.kayitlar.iter().map(Katki::katkici).collect();
        adlar.sort_unstable();
        adlar.dedup();
        adlar.len()
    }

    /// Her kaydin lisansi kapali kumede mi? Defterin kendi butunlugu.
    #[must_use]
    pub fn butun(&self) -> bool {
        self.kayitlar
            .iter()
            .all(|k| lisans_tanimli(&k.lisans) && yuzey_tanimli(&k.kaynak))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defter() -> Defter {
        let mut d = Defter::yeni();
        d.yaz_ham("k1", "ayaz", "budlum-xyz/lubot", "kendi-agac", 10)
            .unwrap();
        d.yaz_ham("k2", "ayaz", "budlum-xyz/workspace", "kendi-agac", 20)
            .unwrap();
        d.yaz_ham("k3", "baskasi", "zincir-kayit", "cc0-budlum", 30)
            .unwrap();
        d
    }

    #[test]
    fn kapali_kumeler_beyanli() {
        assert_eq!(LISANSLAR.len(), 4);
        assert_eq!(YUZEYLER.len(), 3);
        assert!(lisans_tanimli("kendi-agac"));
        assert!(!lisans_tanimli("gpl"));
        assert!(yuzey_tanimli("budlum-xyz/lubot"));
        assert!(!yuzey_tanimli("github.com/baska"));
    }

    /// K2 burada uygulanir: yuzey disi kayit **girip sonra filtrelenmez**.
    #[test]
    fn yuzey_disi_kaynak_deftere_giremez() {
        let mut d = Defter::yeni();
        let r = d.yaz_ham("k", "a", "huggingface.co/model", "kendi-agac", 1);
        assert!(matches!(r, Err(KokenRed::YuzeyDisiKaynak { .. })));
        assert_eq!(d.sayi(), 0);
        assert_eq!(d.reddedilen(), 1);
    }

    /// Bu modulun ikinci kurali: taninmayan lisans **bilinmiyora
    /// dusurulmez**.
    #[test]
    fn taninmayan_lisans_bilinmiyora_dusurulmez() {
        let mut d = Defter::yeni();
        let r = d.yaz_ham("k", "a", "budlum-xyz/lubot", "gpl-3.0", 1);
        assert_eq!(
            r,
            Err(KokenRed::LisansTaninmiyor {
                lisans: "gpl-3.0".to_string()
            })
        );
        assert_eq!(d.sayi(), 0);
    }

    #[test]
    fn kimliksiz_kayit_reddedilir() {
        let mut d = Defter::yeni();
        assert_eq!(
            d.yaz_ham("  ", "a", "budlum-xyz/lubot", "kendi-agac", 1),
            Err(KokenRed::KimliksizKayit)
        );
    }

    #[test]
    fn kimliksiz_katkici_reddedilir() {
        let mut d = Defter::yeni();
        assert_eq!(
            d.yaz_ham("k", "", "budlum-xyz/lubot", "kendi-agac", 1),
            Err(KokenRed::KimliksizKatkici)
        );
    }

    #[test]
    fn kayit_tekrari_reddedilir() {
        let mut d = defter();
        let r = d.yaz_ham("k1", "ayaz", "budlum-xyz/lubot", "kendi-agac", 99);
        assert!(matches!(r, Err(KokenRed::KayitTekrar { .. })));
        assert_eq!(d.sayi(), 3);
    }

    #[test]
    fn koken_sorulabilir() {
        let d = defter();
        let k = d.koken("k3").unwrap();
        assert_eq!(k.katkici(), "baskasi");
        assert_eq!(k.kaynak(), "zincir-kayit");
        assert_eq!(k.lisans(), "cc0-budlum");
        assert_eq!(k.damga(), 30);
    }

    #[test]
    fn bilinmeyen_kayit_sorusu_reddedilir() {
        let d = defter();
        assert!(matches!(d.koken("yok"), Err(KokenRed::KayitYok { .. })));
    }

    #[test]
    fn kayit_birden_cok_tura_girebilir() {
        let mut d = defter();
        d.tura_bagla("k1", 1).unwrap();
        d.tura_bagla("k1", 3).unwrap();
        d.tura_bagla("k1", 2).unwrap();
        assert_eq!(d.turlari("k1"), vec![1, 2, 3]);
    }

    #[test]
    fn ayni_tura_iki_kez_baglama_cogaltmaz() {
        let mut d = defter();
        d.tura_bagla("k1", 5).unwrap();
        d.tura_bagla("k1", 5).unwrap();
        assert_eq!(d.turlari("k1"), vec![5]);
    }

    #[test]
    fn bilinmeyen_kayit_tura_baglanamaz() {
        let mut d = defter();
        assert!(matches!(
            d.tura_bagla("yok", 1),
            Err(KokenRed::KayitYok { .. })
        ));
    }

    #[test]
    fn turdaki_kayitlar_listelenir() {
        let mut d = defter();
        d.tura_bagla("k1", 7).unwrap();
        d.tura_bagla("k3", 7).unwrap();
        assert_eq!(d.turdaki(7), vec!["k1", "k3"]);
        assert!(d.turdaki(8).is_empty());
    }

    #[test]
    fn lisans_sayisi_denetim_sorusu() {
        let d = defter();
        assert_eq!(d.lisans_sayisi("kendi-agac"), 2);
        assert_eq!(d.lisans_sayisi("cc0-budlum"), 1);
        assert_eq!(d.lisans_sayisi("mit-budlum"), 0);
    }

    /// Defter bir liderlik tablosu vermez.
    #[test]
    fn katkici_sayisi_tekil_sayidir_liste_degil() {
        let d = defter();
        assert_eq!(d.katkici_sayisi(), 2);
    }

    #[test]
    fn defter_butun() {
        assert!(defter().butun());
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Defter::parametre_sayisi(), 0);
    }

    /// Sozlesme testi: `Katki`'nin alan listesi modulun kendisi. Bir deger
    /// alani eklenirse bu desen eslesmesi derlenmez.
    #[test]
    fn katkida_deger_alani_yok() {
        let k = Katki::yeni("k", "a", "budlum-xyz/lubot", "kendi-agac", 1).unwrap();
        let Katki {
            kimlik,
            katkici,
            kaynak,
            lisans,
            damga,
        } = k;
        assert_eq!(kimlik, "k");
        assert_eq!(katkici, "a");
        assert_eq!(kaynak, "budlum-xyz/lubot");
        assert_eq!(lisans, "kendi-agac");
        assert_eq!(damga, 1);
    }
}
