//! 6.3 — bf16 egit, CQ2 servis et.
//!
//! Bu modulun tek cumlesi: **egitim sayisi ile servis sayisi ayni sey
//! degildir.** Kulaga bariz geliyor ve tam da bu yuzden tehlikeli - bariz
//! oldugu icin kimse kontrol etmiyor.
//!
//! Somut hata sudur: model bf16 ile egitilir, degerlendirme bf16 agirliklarla
//! kosar, sonuc kaydedilir, sonra servis CQ2 ile yapilir. Kaydedilen sayi
//! **servis edilen modelin sayisi degildir** ama raporda oyle durur. Fark
//! kucukse kimse fark etmez; fark buyudugunde de raporun hangi tarafi yanlis
//! oldugu artik bilinemez, cunku ikisi hic ayri olculmemistir.
//!
//! Burada kurulan sey bu ayrimin **tipli** hali:
//!
//! - [`Numerik`] bir kipi adlandirir ve her kipin bit butcesi beyanlidir.
//! - [`Bolme`] bir turda hangi kipin egitimde, hangisinin serviste oldugunu
//!   tutar; ikisi **ayni olmak zorunda degil** ama **ayri olculmek zorunda**.
//! - [`Olcum`] bir sayiyi kipiyle birlikte tasir. Kipsiz bir sayi bu modulden
//!   gecemez: `Olcum::yeni` kip ister ve kip `Numerik`'tir, dize degil.
//!
//! Ve bir red: [`Bolme::rapor`] iki kip icin de olcum yoksa
//! `EksikOlcum` doner. Cazip alternatif - "servis olcumu yoksa egitim
//! olcumunu yaz" - tam olarak yukaridaki hatanin kendisi.
//!
//! Bit butceleri `crates/nicem`'in aritmetigiyle ayni yerden okunur olsun
//! diye burada **yeniden hesaplanmaz**, beyan edilir ve `bit_butcesi()`
//! testinde 2 + 16/128 = 2.125 esitligi ile capraz kontrol edilir.

use core::fmt;

/// Nicemleme grubu basina eleman sayisi (CQ2'nin olcek paylasimi).
pub const GRUP: usize = 128;

/// Nicemleme olceginin maliyeti, agirlik grubu basina bit (fp16).
pub const OLCEK_BIT: f64 = 16.0;

/// Uclu paketlemede agirlik basina bit: 5 uclu deger 8 bitte, yani 1.6.
/// Entropi tabani log2(3) = 1.58496 **degil**; paketleme orani ile entropi
/// tabani ayni sey olmadigi icin ikisi ayri yazilir ve burada paketleme
/// orani kullanilir - `bit-budget-is-arithmetic` kapisi da bunu kullaniyor.
pub const UCLUK_BIT: f64 = 1.6;

/// Sayisal kip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Numerik {
    /// Egitim kipi: agirlik basina 16 bit, olcek yok.
    Bf16,
    /// Servis kipi: agirlik basina 2 bit + grup basina fp16 olcek.
    Cq2,
    /// Uclu kip: agirlik basina 1.6 bit + grup basina fp16 olcek.
    ///
    /// 1.6, log2(3) = 1.58496'nin yukari yuvarlanmisi degil; deponun
    /// `bit-budget-is-arithmetic` kapisinda **beyan ettigi** paketleme
    /// oranidir (5 uclu deger 8 bitte). Burada log2(3) yazmak ikinci bir
    /// sayi uretirdi ve iki sayidan hangisinin "gercek" oldugu sorusu
    /// olculemez olurdu; entropi tabani ile paketleme orani ayni sey degil.
    Ucluk,
    /// Sikistirilmamis referans; yalniz capraz kontrolde kullanilir.
    Fp32,
}

impl Numerik {
    #[must_use]
    pub const fn ad(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Cq2 => "cq2",
            Self::Ucluk => "ucluk",
            Self::Fp32 => "fp32",
        }
    }

    /// Agirlik basina bit butcesi, olcek payi dahil.
    #[must_use]
    pub fn bit_butcesi(self) -> f64 {
        let grup = f64::from(u32::try_from(GRUP).unwrap_or(1));
        match self {
            Self::Bf16 => 16.0,
            Self::Fp32 => 32.0,
            Self::Cq2 => 2.0 + OLCEK_BIT / grup,
            // 1.6 = 8 bit / 5 deger; kapinin beyan ettigi paketleme orani.
            Self::Ucluk => UCLUK_BIT + OLCEK_BIT / grup,
        }
    }

    /// Bu kip egitimde kullanilabilir mi? CQ2 ile egitmek bu depoda
    /// yapilmiyor ve yapilmadigi **beyan** ediliyor: sessizce izin vermek,
    /// bir gun birinin denemesi ve raporun ayni gorunmesi demekti.
    #[must_use]
    pub const fn egitime_uygun(self) -> bool {
        matches!(self, Self::Bf16 | Self::Fp32)
    }

    /// Bu kip serviste kullanilabilir mi?
    #[must_use]
    pub const fn servise_uygun(self) -> bool {
        matches!(self, Self::Cq2 | Self::Ucluk | Self::Bf16)
    }
}

impl fmt::Display for Numerik {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.ad())
    }
}

/// Bolme kurulamadi ya da rapor verilemedi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KosumRed {
    /// Egitim kipi egitime uygun degil.
    EgitimeUygunsuz { kip: &'static str },
    /// Servis kipi servise uygun degil.
    ServiseUygunsuz { kip: &'static str },
    /// Bir kipin olcumu yok. **Otekinin olcumu yazilmaz.**
    EksikOlcum { kip: &'static str },
    /// Olcum baska bir kip icin verilmis.
    KipUyusmuyor {
        beklenen: &'static str,
        gelen: &'static str,
    },
    /// Olcum sonlu bir sayi degil.
    SonluDegil,
}

impl fmt::Display for KosumRed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EgitimeUygunsuz { kip } => write!(f, "{kip} ile egitilmiyor"),
            Self::ServiseUygunsuz { kip } => write!(f, "{kip} ile servis edilmiyor"),
            Self::EksikOlcum { kip } => {
                write!(f, "{kip} olcumu yok: otekinin sayisi onun yerine yazilmaz")
            }
            Self::KipUyusmuyor { beklenen, gelen } => {
                write!(f, "olcum {gelen} kipinde, {beklenen} bekleniyordu")
            }
            Self::SonluDegil => write!(f, "olcum sonlu bir sayi degil"),
        }
    }
}

/// Kipiyle birlikte tasinan bir sayi. Kipsiz sayi bu modulden gecemez.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Olcum {
    kip: Numerik,
    deger: f64,
}

impl Olcum {
    /// # Errors
    /// Deger sonlu degilse reddeder.
    pub fn yeni(kip: Numerik, deger: f64) -> Result<Self, KosumRed> {
        if !deger.is_finite() {
            return Err(KosumRed::SonluDegil);
        }
        Ok(Self { kip, deger })
    }

    #[must_use]
    pub const fn kip(&self) -> Numerik {
        self.kip
    }

    #[must_use]
    pub const fn deger(&self) -> f64 {
        self.deger
    }
}

/// Egitim/servis bolmesi.
#[derive(Debug, Clone, PartialEq)]
pub struct Bolme {
    egitim: Numerik,
    servis: Numerik,
    egitim_olcumu: Option<Olcum>,
    servis_olcumu: Option<Olcum>,
}

impl Bolme {
    /// # Errors
    /// Kipler rollerine uygun degilse reddeder.
    pub const fn yeni(egitim: Numerik, servis: Numerik) -> Result<Self, KosumRed> {
        if !egitim.egitime_uygun() {
            return Err(KosumRed::EgitimeUygunsuz { kip: egitim.ad() });
        }
        if !servis.servise_uygun() {
            return Err(KosumRed::ServiseUygunsuz { kip: servis.ad() });
        }
        Ok(Self {
            egitim,
            servis,
            egitim_olcumu: None,
            servis_olcumu: None,
        })
    }

    /// Bu deponun beyan ettigi varsayilan bolme: bf16 egit, CQ2 servis et.
    ///
    /// # Errors
    /// Pratikte donmez; `yeni`'nin sozlesmesini korumak icin `Result`.
    pub const fn varsayilan() -> Result<Self, KosumRed> {
        Self::yeni(Numerik::Bf16, Numerik::Cq2)
    }

    #[must_use]
    pub const fn egitim_kipi(&self) -> Numerik {
        self.egitim
    }

    #[must_use]
    pub const fn servis_kipi(&self) -> Numerik {
        self.servis
    }

    /// Egitim ve servis ayni kipte mi? Ayni olmasi hata degil - ama
    /// **bilinmesi** gerekiyor, cunku ayniysa aradaki fark olculemez.
    #[must_use]
    pub fn ayni_kip(&self) -> bool {
        self.egitim == self.servis
    }

    /// Bu modul parametre tutmaz.
    #[must_use]
    pub const fn parametre_sayisi() -> usize {
        0
    }

    /// Egitim tarafinin olcumunu yazar.
    ///
    /// # Errors
    /// Olcum baska bir kipe aitse reddeder.
    pub fn egitim_olcumu(&mut self, olcum: Olcum) -> Result<(), KosumRed> {
        if olcum.kip != self.egitim {
            return Err(KosumRed::KipUyusmuyor {
                beklenen: self.egitim.ad(),
                gelen: olcum.kip.ad(),
            });
        }
        self.egitim_olcumu = Some(olcum);
        Ok(())
    }

    /// Servis tarafinin olcumunu yazar.
    ///
    /// # Errors
    /// Olcum baska bir kipe aitse reddeder.
    pub fn servis_olcumu(&mut self, olcum: Olcum) -> Result<(), KosumRed> {
        if olcum.kip != self.servis {
            return Err(KosumRed::KipUyusmuyor {
                beklenen: self.servis.ad(),
                gelen: olcum.kip.ad(),
            });
        }
        self.servis_olcumu = Some(olcum);
        Ok(())
    }

    /// Iki tarafi da tasiyan rapor.
    ///
    /// # Errors
    /// Iki olcumden biri eksikse `EksikOlcum`. Otekinin sayisi onun yerine
    /// **yazilmaz**.
    pub fn rapor(&self) -> Result<Rapor, KosumRed> {
        let egitim = self.egitim_olcumu.ok_or(KosumRed::EksikOlcum {
            kip: self.egitim.ad(),
        })?;
        let servis = self.servis_olcumu.ok_or(KosumRed::EksikOlcum {
            kip: self.servis.ad(),
        })?;
        Ok(Rapor {
            egitim,
            servis,
            fark: servis.deger - egitim.deger,
            egitim_bit: self.egitim.bit_butcesi(),
            servis_bit: self.servis.bit_butcesi(),
        })
    }
}

/// Iki tarafi ayri ayri tasiyan rapor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rapor {
    pub egitim: Olcum,
    pub servis: Olcum,
    /// Servis eksi egitim. Isaret korunur: servis tarafi kotuyse negatif
    /// degil **pozitif** cikmasi gerektigi gibi bir donusum yapilmaz, cunku
    /// olcunun yonu olcunun kendisine ait.
    pub fark: f64,
    pub egitim_bit: f64,
    pub servis_bit: f64,
}

impl Rapor {
    /// Servis tarafinin egitim tarafina gore sikistirma orani.
    #[must_use]
    pub fn sikistirma(&self) -> f64 {
        if self.servis_bit <= 0.0 {
            return f64::INFINITY;
        }
        self.egitim_bit / self.servis_bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_butcesi_aritmetikle_tutuyor() {
        // crates/nicem'in kapisiyla ayni sayi: 2 + 16/128 = 2.125.
        assert!((Numerik::Cq2.bit_butcesi() - 2.125).abs() < 1e-12);
        assert!((Numerik::Ucluk.bit_butcesi() - 1.725).abs() < 1e-12);
        assert!((Numerik::Bf16.bit_butcesi() - 16.0).abs() < 1e-12);
    }

    #[test]
    fn cq2_ile_egitilmez() {
        let r = Bolme::yeni(Numerik::Cq2, Numerik::Cq2);
        assert_eq!(r, Err(KosumRed::EgitimeUygunsuz { kip: "cq2" }));
    }

    #[test]
    fn fp32_ile_servis_edilmez() {
        let r = Bolme::yeni(Numerik::Bf16, Numerik::Fp32);
        assert_eq!(r, Err(KosumRed::ServiseUygunsuz { kip: "fp32" }));
    }

    #[test]
    fn varsayilan_bolme_bf16_cq2() {
        let b = Bolme::varsayilan().unwrap();
        assert_eq!(b.egitim_kipi(), Numerik::Bf16);
        assert_eq!(b.servis_kipi(), Numerik::Cq2);
        assert!(!b.ayni_kip());
    }

    /// Bu modulun tek cumlesi.
    #[test]
    fn eksik_servis_olcumu_egitim_olcumuyle_doldurulmaz() {
        let mut b = Bolme::varsayilan().unwrap();
        b.egitim_olcumu(Olcum::yeni(Numerik::Bf16, 0.42).unwrap())
            .unwrap();
        let r = b.rapor();
        assert_eq!(r, Err(KosumRed::EksikOlcum { kip: "cq2" }));
    }

    #[test]
    fn eksik_egitim_olcumu_servis_olcumuyle_doldurulmaz() {
        let mut b = Bolme::varsayilan().unwrap();
        b.servis_olcumu(Olcum::yeni(Numerik::Cq2, 0.5).unwrap())
            .unwrap();
        assert_eq!(b.rapor(), Err(KosumRed::EksikOlcum { kip: "bf16" }));
    }

    #[test]
    fn yanlis_kipli_olcum_reddedilir() {
        let mut b = Bolme::varsayilan().unwrap();
        let r = b.egitim_olcumu(Olcum::yeni(Numerik::Cq2, 0.1).unwrap());
        assert_eq!(
            r,
            Err(KosumRed::KipUyusmuyor {
                beklenen: "bf16",
                gelen: "cq2"
            })
        );
    }

    #[test]
    fn sonsuz_olcum_reddedilir() {
        assert_eq!(
            Olcum::yeni(Numerik::Bf16, f64::INFINITY),
            Err(KosumRed::SonluDegil)
        );
        assert_eq!(
            Olcum::yeni(Numerik::Bf16, f64::NAN),
            Err(KosumRed::SonluDegil)
        );
    }

    #[test]
    fn rapor_iki_tarafi_da_tasir() {
        let mut b = Bolme::varsayilan().unwrap();
        b.egitim_olcumu(Olcum::yeni(Numerik::Bf16, 0.40).unwrap())
            .unwrap();
        b.servis_olcumu(Olcum::yeni(Numerik::Cq2, 0.44).unwrap())
            .unwrap();
        let r = b.rapor().unwrap();
        assert!((r.egitim.deger() - 0.40).abs() < 1e-12);
        assert!((r.servis.deger() - 0.44).abs() < 1e-12);
        assert!((r.fark - 0.04).abs() < 1e-12, "fark isareti kaybolmus");
    }

    #[test]
    fn farkin_isareti_korunur() {
        let mut b = Bolme::varsayilan().unwrap();
        b.egitim_olcumu(Olcum::yeni(Numerik::Bf16, 0.50).unwrap())
            .unwrap();
        b.servis_olcumu(Olcum::yeni(Numerik::Cq2, 0.30).unwrap())
            .unwrap();
        assert!(b.rapor().unwrap().fark < 0.0);
    }

    #[test]
    fn sikistirma_orani_olculur() {
        let mut b = Bolme::varsayilan().unwrap();
        b.egitim_olcumu(Olcum::yeni(Numerik::Bf16, 1.0).unwrap())
            .unwrap();
        b.servis_olcumu(Olcum::yeni(Numerik::Cq2, 1.0).unwrap())
            .unwrap();
        let s = b.rapor().unwrap().sikistirma();
        assert!((s - 16.0 / 2.125).abs() < 1e-12, "olculen {s}");
    }

    #[test]
    fn ayni_kip_bolmesi_kurulabilir_ama_isaretlenir() {
        let mut b = Bolme::yeni(Numerik::Bf16, Numerik::Bf16).unwrap();
        assert!(b.ayni_kip(), "ayni kip isaretlenmedi: fark olculemez");
        b.egitim_olcumu(Olcum::yeni(Numerik::Bf16, 1.0).unwrap())
            .unwrap();
        b.servis_olcumu(Olcum::yeni(Numerik::Bf16, 1.0).unwrap())
            .unwrap();
        assert!((b.rapor().unwrap().fark).abs() < 1e-12);
    }

    #[test]
    fn kip_adlari_kararli() {
        assert_eq!(Numerik::Bf16.to_string(), "bf16");
        assert_eq!(Numerik::Cq2.to_string(), "cq2");
        assert_eq!(Numerik::Ucluk.to_string(), "ucluk");
        assert_eq!(Numerik::Fp32.to_string(), "fp32");
    }

    #[test]
    fn rol_uygunlugu_kapali() {
        assert!(Numerik::Bf16.egitime_uygun());
        assert!(Numerik::Fp32.egitime_uygun());
        assert!(!Numerik::Cq2.egitime_uygun());
        assert!(!Numerik::Ucluk.egitime_uygun());
        assert!(Numerik::Cq2.servise_uygun());
        assert!(Numerik::Ucluk.servise_uygun());
        assert!(!Numerik::Fp32.servise_uygun());
    }

    #[test]
    fn parametre_tutmaz() {
        assert_eq!(Bolme::parametre_sayisi(), 0);
    }
}
