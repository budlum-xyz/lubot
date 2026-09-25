//! Eğitim ölçüm çatısı: koşunun sayılarını *iddia* değil *ölçüm* olarak tutar.
//!
//! Bu modül eğitim yapmaz; bir koşunun sayılarını toplar, dönüştürür ve
//! yazdırılabilir hâle getirir. Amacı tek: raporda geçen her sayının nereden
//! geldiğinin tek bir yerde durması. Kayıp eğrisinin ortalaması, en iyi adımı,
//! jeton/saniye hızı, karmaşıklık (perplexity) ve bayt başına bit — hepsi aynı
//! sözleşmeye uyar:
//!
//! * **Anlamsız girdi sessizce yutulmaz.** NaN/sonsuz/negatif bir kayıp
//!   [`OlcumHatasi::GecersizKayip`] ile geri döner; ortalamaya karışmaz.
//! * **Tanımsız dönüşüm `None`'dur, uydurma değil.** `kayip = 0` için
//!   perplexity `1`'dir (doğru), ama negatif bir kayıp için perplexity
//!   tanımsızdır: `None` döner, `1` dönmez.
//! * **Ölçüm birikimlidir ve sıfırlanabilir.** Bir koşu parça parça
//!   koşulabildiği için (resume) istatistik nesnesi taşınabilir olmalıdır;
//!   [`KayipIstatistigi`] taşınabilir bir değerdir.
//!
//! # Neden bayt başına bit
//!
//! Jeton başına kayıp, jetonlayıcıya bağlıdır: aynı model, aynı metin, farklı
//! sözlükle farklı `kayip` verir. Bayt başına bit jetonlayıcıdan bağımsızdır ve
//! bu yüzden iki farklı sözlükle eğitilmiş iki modeli karşılaştırmanın tek
//! dürüst yoludur. Ölçüm için metnin **bayt** sayısı gerekir; jeton sayısından
//! türetmek, sözlüğü ölçümün içine geri sokar.

/// Ölçüm katmanının ret sebepleri.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OlcumHatasi {
    /// Kayıp sonlu ve negatif olmayan bir sayı değil.
    GecersizKayip(String),
    /// Ölçüm için gereken sayı sıfır (ör. sıfır jetonla hız ölçülemez).
    SifirPayda(String),
    /// Yumuşatma katsayısı `(0,1]` dışında.
    GecersizKatsayi(String),
}

impl std::fmt::Display for OlcumHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GecersizKayip(ne) => write!(f, "gecersiz kayip: {ne}"),
            Self::SifirPayda(ne) => write!(f, "olcum icin payda sifir: {ne}"),
            Self::GecersizKatsayi(ne) => write!(f, "gecersiz katsayi: {ne}"),
        }
    }
}

impl std::error::Error for OlcumHatasi {}

/// Bir koşunun kayıp eğrisi: adım adım toplanan, taşınabilir istatistik.
///
/// Üstel hareketli ortalama (EMA) ayrı tutulur, çünkü eğrinin son hâlini
/// anlatmanın yolu ortalamadan farklıdır: ortalama tüm koşuyu, EMA son pencereleri
/// temsil eder. İkisinin arasındaki fark, kaybın hâlâ düşüp düşmediğinin
/// ölçüsüdür ve rapor bunu ayrı sayı olarak taşır.
#[derive(Debug, Clone, PartialEq)]
pub struct KayipIstatistigi {
    adim: u64,
    toplam: f64,
    en_kucuk: f64,
    en_buyuk: f64,
    ema: f64,
    ema_katsayisi: f64,
    en_kucuk_adim: u64,
}

impl Default for KayipIstatistigi {
    fn default() -> Self {
        Self::yeni(0.05)
    }
}

impl KayipIstatistigi {
    /// Katsayı `(0,1]` olmalı: `1.0` "yalnız son değer", küçük değer "uzun hafıza".
    #[must_use]
    pub fn yeni(ema_katsayisi: f64) -> Self {
        let katsayi = if ema_katsayisi.is_finite() && ema_katsayisi > 0.0 && ema_katsayisi <= 1.0 {
            ema_katsayisi
        } else {
            0.05
        };
        Self {
            adim: 0,
            toplam: 0.0,
            en_kucuk: f64::INFINITY,
            en_buyuk: f64::NEG_INFINITY,
            ema: f64::NAN,
            ema_katsayisi: katsayi,
            en_kucuk_adim: 0,
        }
    }

    /// Bir adımın kaybını ekler.
    ///
    /// # Errors
    /// [`OlcumHatasi::GecersizKayip`] — NaN, sonsuz ya da negatif kayıp.
    pub fn ekle(&mut self, kayip: f64) -> Result<(), OlcumHatasi> {
        if !kayip.is_finite() || kayip < 0.0 {
            return Err(OlcumHatasi::GecersizKayip(format!("{kayip}")));
        }
        self.adim += 1;
        self.toplam += kayip;
        if kayip < self.en_kucuk {
            self.en_kucuk = kayip;
            self.en_kucuk_adim = self.adim;
        }
        if kayip > self.en_buyuk {
            self.en_buyuk = kayip;
        }
        self.ema = if self.ema.is_nan() {
            kayip
        } else {
            self.ema_katsayisi
                .mul_add(kayip, (1.0 - self.ema_katsayisi) * self.ema)
        };
        Ok(())
    }

    /// Kaç adım ölçüldü.
    #[must_use]
    pub fn adim(&self) -> u64 {
        self.adim
    }

    /// Ölçüm var mı (tek adım bile yoksa `false`).
    #[must_use]
    pub fn dolu(&self) -> bool {
        self.adim > 0
    }

    /// Aritmetik ortalama; hiç adım yoksa `None`.
    #[must_use]
    pub fn ortalama(&self) -> Option<f64> {
        if self.adim == 0 {
            return None;
        }
        Some(self.toplam / self.adim as f64)
    }

    /// En küçük kayıp ve hangi adımda görüldüğü; hiç adım yoksa `None`.
    #[must_use]
    pub fn en_iyi(&self) -> Option<(u64, f64)> {
        if self.adim == 0 {
            return None;
        }
        Some((self.en_kucuk_adim, self.en_kucuk))
    }

    /// En büyük kayıp; hiç adım yoksa `None`.
    #[must_use]
    pub fn en_kotu(&self) -> Option<f64> {
        if self.adim == 0 {
            return None;
        }
        Some(self.en_buyuk)
    }

    /// Üstel hareketli ortalama; hiç adım yoksa `NaN`.
    #[must_use]
    pub fn ema(&self) -> f64 {
        self.ema
    }

    /// Eğrinin salınımı: en iyi ile en kötü arasındaki fark.
    #[must_use]
    pub fn salinim(&self) -> Option<f64> {
        Some(self.en_kotu()? - self.en_iyi()?.1)
    }

    /// İki uç arasındaki göreli düşüş, yüzde olarak; başlangıç sıfırsa `None`.
    ///
    /// "Kayıp %X düştü" cümlesinin ölçülmüş hâli budur: ilk adımın kaybına
    /// göre yüzde. Sıfıra bölmek yerine `None` döner.
    #[must_use]
    pub fn dusus_yuzdesi(&self, ilk: f64, son: f64) -> Option<f64> {
        if !ilk.is_finite() || ilk <= 0.0 || !son.is_finite() {
            return None;
        }
        Some((ilk - son) / ilk * 100.0)
    }

    /// Ölçümü sıfırlar; `ema_katsayisi` korunur.
    pub fn sifirla(&mut self) {
        let katsayi = self.ema_katsayisi;
        *self = Self::yeni(katsayi);
    }
}

/// Jeton ve süre sayacı: hız ve kalan süre tahmini.
///
/// Kalan süre tahmini **ölçülmüş** hıza dayanır; bir tahminin tahmin olduğunu
/// söylemenin yolu, onu üreten hızı yanında taşımaktır. Hiç jeton ölçülmemişse
/// tahmin de `None`'dur.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct JetonSayaci {
    jeton: u64,
    saniye: f64,
}

impl JetonSayaci {
    /// Boş sayaç.
    #[must_use]
    pub fn yeni() -> Self {
        Self::default()
    }

    /// Bir ölçüm turu ekler: şu kadar jeton, şu kadar saniyede.
    ///
    /// # Errors
    /// [`OlcumHatasi::SifirPayda`] — negatif/sonlu olmayan süre.
    pub fn ekle(&mut self, jeton: u64, saniye: f64) -> Result<(), OlcumHatasi> {
        if !saniye.is_finite() || saniye < 0.0 {
            return Err(OlcumHatasi::SifirPayda(format!("sure {saniye}")));
        }
        self.jeton += jeton;
        self.saniye += saniye;
        Ok(())
    }

    /// Ölçülen toplam jeton.
    #[must_use]
    pub fn jeton(&self) -> u64 {
        self.jeton
    }

    /// Ölçülen toplam süre (saniye).
    #[must_use]
    pub fn saniye(&self) -> f64 {
        self.saniye
    }

    /// Saniyedeki jeton; süre ölçülmemişse `None`.
    #[must_use]
    pub fn jeton_saniye(&self) -> Option<f64> {
        if self.saniye <= 0.0 {
            return None;
        }
        Some(self.jeton as f64 / self.saniye)
    }

    /// Jeton başına milisaniye; jeton yoksa `None`.
    #[must_use]
    pub fn milisaniye_jeton(&self) -> Option<f64> {
        if self.jeton == 0 {
            return None;
        }
        Some(self.saniye * 1000.0 / self.jeton as f64)
    }

    /// Kalan jeton için ölçülmüş hızdan süre tahmini (saniye); hız yoksa `None`.
    #[must_use]
    pub fn tahmini_sure(&self, kalan_jeton: u64) -> Option<f64> {
        let hiz = self.jeton_saniye()?;
        if hiz <= 0.0 {
            return None;
        }
        Some(kalan_jeton as f64 / hiz)
    }
}

/// Karmaşıklık (perplexity): `exp(kayip)`.
///
/// Negatif ya da sonlu olmayan bir kayıp için `None`. `f64::exp` taşarsa
/// (`kayip` büyükken) yine `None`: taşan bir sayıyı "ölçüm" diye yazmak,
/// rapordaki en yanıltıcı satır olurdu.
#[must_use]
pub fn perplexity(kayip: f64) -> Option<f64> {
    if !kayip.is_finite() || kayip < 0.0 {
        return None;
    }
    let p = kayip.exp();
    if p.is_finite() {
        Some(p)
    } else {
        None
    }
}

/// Bayt başına bit: `kayip_toplam / ln(2) / bayt`.
///
/// Sözlükten bağımsız karşılaştırma ölçüsü. Jeton ya da bayt sıfırsa `None`.
#[must_use]
pub fn bit_basina_bayt(kayip_toplam: f64, jeton: u64, bayt: u64) -> Option<f64> {
    if !kayip_toplam.is_finite() || kayip_toplam < 0.0 || jeton == 0 || bayt == 0 {
        return None;
    }
    Some(kayip_toplam / std::f64::consts::LN_2 / bayt as f64)
}

/// Bir koşunun rapora giren ölçüleri.
///
/// Bu yapı *hesap yapmaz*; yalnız ölçülmüş sayıları taşır ve yazdırır. Böylece
/// raporda görünen her sayının kaynağı tek bir alan olur ve iki yerde iki farklı
/// hesap yapılması mümkün olmaz.
#[derive(Debug, Clone, PartialEq)]
pub struct KosuOlculeri {
    /// Ölçülen adım sayısı.
    pub adim: u64,
    /// Kayıp istatistiği.
    pub kayip: KayipIstatistigi,
    /// Doğrulama kayıplarının istatistiği.
    pub dogrulama: KayipIstatistigi,
    /// Jeton/süre sayacı.
    pub sayac: JetonSayaci,
    /// Doğrulama için geçen süre (saniye); hiç doğrulama yoksa `None`.
    pub dogrulama_saniye: Option<f64>,
    /// Ölçülen kayıp için bayt sayısı; yoksa `None` (bayt başına bit ölçülemez).
    pub bayt: Option<u64>,
}

impl KosuOlculeri {
    /// Boş ölçüm.
    #[must_use]
    pub fn yeni() -> Self {
        Self {
            adim: 0,
            kayip: KayipIstatistigi::default(),
            dogrulama: KayipIstatistigi::default(),
            sayac: JetonSayaci::yeni(),
            dogrulama_saniye: None,
            bayt: None,
        }
    }

    /// Eğitim adımının kaybını ve o adımda harcanan jetonu/süreyi ekler.
    ///
    /// # Errors
    /// [`KayipIstatistigi::ekle`] ve [`JetonSayaci::ekle`] hataları.
    pub fn adim_ekle(&mut self, kayip: f64, jeton: u64, saniye: f64) -> Result<(), OlcumHatasi> {
        self.kayip.ekle(kayip)?;
        self.sayac.ekle(jeton, saniye)?;
        self.adim += 1;
        Ok(())
    }

    /// Doğrulama kaybını ve süresini ekler.
    ///
    /// # Errors
    /// [`KayipIstatistigi::ekle`], ve negatif/sonlu olmayan süre.
    pub fn dogrulama_ekle(&mut self, kayip: f64, saniye: f64) -> Result<(), OlcumHatasi> {
        self.dogrulama.ekle(kayip)?;
        if !saniye.is_finite() || saniye < 0.0 {
            return Err(OlcumHatasi::SifirPayda(format!(
                "dogrulama suresi {saniye}"
            )));
        }
        self.dogrulama_saniye = Some(self.dogrulama_saniye.unwrap_or(0.0) + saniye);
        Ok(())
    }

    /// Eğitim kaybının karmaşıklığı; ölçüm yoksa `None`.
    #[must_use]
    pub fn perplexity(&self) -> Option<f64> {
        perplexity(self.kayip.ema())
    }

    /// Doğrulama kaybının karmaşıklığı; ölçüm yoksa `None`.
    #[must_use]
    pub fn dogrulama_perplexity(&self) -> Option<f64> {
        perplexity(self.dogrulama.ema())
    }

    /// Bayt başına bit; bayt ya da jeton ölçülmemişse `None`.
    #[must_use]
    pub fn bit_basina_bayt(&self) -> Option<f64> {
        let bayt = self.bayt?;
        let kayip_toplam = self
            .kayip
            .ortalama()
            .map(|o| o * self.sayac.jeton() as f64)?;
        bit_basina_bayt(kayip_toplam, self.sayac.jeton(), bayt)
    }

    /// Markdown rapor satırları: tablo gövdesi (başlık çağıranın işi).
    #[must_use]
    pub fn markdown_satirlari(&self) -> Vec<String> {
        let sayi = |d: Option<f64>, basamak: usize| match d {
            Some(x) if x.is_finite() => format!("{x:.basamak$}"),
            _ => "olculmedi".to_string(),
        };
        let mut satirlar = vec![
            format!("| adim | {} |", self.adim),
            format!("| kayip (son EMA) | {} |", sayi(Some(self.kayip.ema()), 6)),
            format!("| kayip (ortalama) | {} |", sayi(self.kayip.ortalama(), 6)),
            format!(
                "| en iyi adim | {} |",
                match self.kayip.en_iyi() {
                    Some((adim, kayip)) => format!("{adim} ({kayip:.6})"),
                    None => "olculmedi".to_string(),
                }
            ),
            format!("| perplexity | {} |", sayi(self.perplexity(), 3)),
            format!(
                "| dogrulama kaybi | {} |",
                sayi(self.dogrulama.ortalama(), 6)
            ),
            format!(
                "| dogrulama perplexity | {} |",
                sayi(self.dogrulama_perplexity(), 3)
            ),
            format!("| jeton | {} |", self.sayac.jeton()),
            format!("| jeton/saniye | {}", " |"),
            format!("| bit/bayt | {} |", sayi(self.bit_basina_bayt(), 4)),
        ];
        // jeton/saniye satiri olcumsuzken "olculmedi" yazmali: yukaridaki
        // sablonu burada tamamlamak, iki yerde iki bicim uretmekten iyidir.
        if let Some(hedef) = satirlar
            .iter_mut()
            .find(|s| s.starts_with("| jeton/saniye"))
        {
            *hedef = format!("| jeton/saniye | {} |", sayi(self.sayac.jeton_saniye(), 2));
        }
        satirlar
    }
}

#[cfg(test)]
mod testler {
    use super::*;

    #[test]
    fn perplexity_ln10_tam_ondur() {
        let p = perplexity(std::f64::consts::LN_10).expect("ln 10");
        assert!((p - 10.0).abs() < 1e-9, "{p}");
        assert_eq!(perplexity(0.0), Some(1.0));
        assert_eq!(perplexity(-1.0), None);
        assert_eq!(perplexity(f64::NAN), None);
        assert_eq!(perplexity(1e9), None, "tasan perplexity olcum sayilmaz");
    }

    #[test]
    fn kayip_istatistigi_ortalama_en_iyi_ve_ema_tutar() {
        let mut i = KayipIstatistigi::yeni(1.0);
        assert!(!i.dolu());
        assert_eq!(i.ortalama(), None);
        assert_eq!(i.en_iyi(), None);
        i.ekle(4.0).expect("ilk");
        i.ekle(2.0).expect("ikinci");
        i.ekle(6.0).expect("ucuncu");
        assert_eq!(i.adim(), 3);
        assert!((i.ortalama().unwrap_or(f64::NAN) - 4.0).abs() < 1e-12);
        assert_eq!(i.en_iyi(), Some((2, 2.0)));
        assert_eq!(i.en_kotu(), Some(6.0));
        // katsayi 1.0: EMA her zaman son deger.
        assert!((i.ema() - 6.0).abs() < 1e-12);
        assert!((i.salinim().unwrap_or(f64::NAN) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn gecersiz_kayip_ortalamaya_karismaz() {
        let mut i = KayipIstatistigi::default();
        assert!(i.ekle(f64::NAN).is_err());
        assert!(i.ekle(-0.5).is_err());
        assert!(i.ekle(f64::INFINITY).is_err());
        assert_eq!(i.adim(), 0);
        i.ekle(1.0).expect("gecerli");
        assert_eq!(i.adim(), 1);
    }

    #[test]
    fn ema_katsayisi_gecersizse_varsayilana_duser_ve_sifirlama_korur() {
        let mut i = KayipIstatistigi::yeni(0.0);
        i.ekle(2.0).expect("kayit");
        i.ekle(4.0).expect("kayit");
        // Varsayilan katsayi 0.05: EMA 2 -> 2.1
        assert!((i.ema() - 2.1).abs() < 1e-12, "{}", i.ema());
        i.sifirla();
        assert_eq!(i.adim(), 0);
        i.ekle(5.0).expect("kayit");
        assert!((i.ema() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn dusus_yuzdesi_sifira_bolmez() {
        let i = KayipIstatistigi::default();
        assert_eq!(i.dusus_yuzdesi(0.0, 1.0), None);
        assert_eq!(i.dusus_yuzdesi(4.0, 1.0), Some(75.0));
        assert_eq!(i.dusus_yuzdesi(f64::NAN, 1.0), None);
    }

    #[test]
    fn sayac_hiz_ve_kalan_sure_tahmini_verir() {
        let mut s = JetonSayaci::yeni();
        assert_eq!(s.jeton_saniye(), None);
        s.ekle(1000, 2.0).expect("ilk tur");
        assert!((s.jeton_saniye().unwrap_or(f64::NAN) - 500.0).abs() < 1e-9);
        assert!((s.milisaniye_jeton().unwrap_or(f64::NAN) - 2.0).abs() < 1e-9);
        assert!((s.tahmini_sure(250).unwrap_or(f64::NAN) - 0.5).abs() < 1e-9);
        assert!(s.ekle(10, -1.0).is_err());
        assert_eq!(s.jeton(), 1000, "gecersiz tur sayaci buyutmedi");
    }

    #[test]
    fn bit_basina_bayt_sozlukten_bagimsiz_olculur() {
        // 1 nat toplam kayip, ln2 nat = 1 bit -> 1 bayt icin 1 bit/bayt.
        let b = bit_basina_bayt(std::f64::consts::LN_2, 1, 1).unwrap_or(f64::NAN);
        assert!((b - 1.0).abs() < 1e-12, "{b}");
        assert_eq!(bit_basina_bayt(1.0, 0, 10), None);
        assert_eq!(bit_basina_bayt(1.0, 10, 0), None);
        assert_eq!(bit_basina_bayt(-1.0, 10, 10), None);
    }

    #[test]
    fn kosu_olculeri_adim_ve_dogrulamayi_ayri_tutar() {
        let mut o = KosuOlculeri::yeni();
        o.adim_ekle(2.0, 256, 0.5).expect("adim 1");
        o.adim_ekle(1.0, 256, 0.5).expect("adim 2");
        o.dogrulama_ekle(1.5, 0.25).expect("dogrulama");
        o.bayt = Some(1024);
        assert_eq!(o.adim, 2);
        assert_eq!(o.sayac.jeton(), 512);
        assert!((o.sayac.jeton_saniye().unwrap_or(f64::NAN) - 512.0).abs() < 1e-9);
        assert!((o.dogrulama.ortalama().unwrap_or(f64::NAN) - 1.5).abs() < 1e-12);
        assert!(o.perplexity().is_some() && o.dogrulama_perplexity().is_some());
        assert!(o.bit_basina_bayt().is_some());
    }

    #[test]
    fn markdown_olculmeyeni_olculmedi_diye_yazar() {
        let o = KosuOlculeri::yeni();
        let satirlar = o.markdown_satirlari();
        let metin = satirlar.join("\n");
        assert!(metin.contains("| perplexity | olculmedi |"), "{metin}");
        assert!(metin.contains("| jeton/saniye | olculmedi |"), "{metin}");
        assert!(metin.contains("| adim | 0 |"), "{metin}");
        let mut dolu = KosuOlculeri::yeni();
        dolu.adim_ekle(1.0, 256, 0.5).expect("adim");
        let metin = dolu.markdown_satirlari().join("\n");
        assert!(!metin.contains("| jeton/saniye | olculmedi |"), "{metin}");
    }
}
