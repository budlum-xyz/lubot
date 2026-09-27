//! Kalibre edilmis guven: ham kayitlardan sicaklik ve bant olcumu.
//!
//! # Neden bu modul var
//!
//! [`crate::GuvenDefteri`] bir iddiayi kaydeder ve kovalarda ne kadar
//! yanildigini soyler. Soyledigi sey **olculen** bir sayidir ama **duzeltilmis**
//! bir sayi degildir: bir karar basi sistematik olarak asiri guvenliyse, defter
//! bunu gosterir, gidermez. Bu modul o bosluğu kapatir: ham `(puan, dogru)`
//! kayitlarindan tek bir **sicaklik** uydurur (egitim hatasini dusuren T),
//! duzeltilmis puanlar uzerinden **bantlari** olcer ve bir guveni uc basamaktan
//! birine koyar: yesil (tek bas karar verir), orta (k-of-n), kirmizi (yukselt).
//!
//! # Esik kodda degil, olcumde
//!
//! Tasarim notu (3.7) bant sinirlarinin **olcumle** oturmasini, kodda sabit
//! deger durmamasini ister. Burada oyle: kirmizi ve yesil sinirlar kayitlardan
//! turer, `hedef` (yesil bandin tutmasi gereken isabet) cagirandan gelir ve
//! varsayilani [`HEDEF_KESINLIK`], veri yetmezse modul **reddeder** - bos bir
//! defterin temiz sayilmamasi kuralinin aynisi: olculemeyen bir bant, guvenli
//! sayilan bir bant degildir.
//!
//! # Tek kalibrasyon makinesi
//!
//! Kova muhasebesi burada yeniden yazilmaz; [`lubot_anlama::Calibration`]
//! kullanilir. Ikinci bir sayac, ayni soruya iki cevap verir.
//!
//! # Olculen sey
//!
//! Modul kendi testleriyle olculur: bilinen bir sicaklikla uretilmis veride
//! uydurmanin onu bulmasi, egitim hatasinin dusmesi, ECE'nin kalibrasyon
//! bozukluguna gore artmasi, bantlarin veriden turedigi ve sinir davranisinin
//! (tam sinirda hangi basamak) test edilmesi. Hicbiri iddia degil, hepsi
//! kosan bir testtir.

use lubot_anlama::{Calibration, Outcome, CALIBRATION_BUCKETS};

use crate::{tek_bas, Guven, Karar, Politika, Sonuc, YukseltmeNedeni};

/// Yesil bandin tutmasi gereken isabet, cagiran baska bir sey istemezse.
pub const HEDEF_KESINLIK: f64 = 0.9;
/// Bir bandin olculebilmesi icin gereken en az kayit.
pub const ASGARI_DESTEK: u64 = 5;
/// Sicaklik uydurmasi icin gereken en az kayit.
pub const ASGARI_KAYIT: usize = 8;
/// Aranan sicaklik araligi: alt ucu.
pub const SICAKLIK_ALT: f64 = 0.05;
/// Aranan sicaklik araligi: ust ucu.
pub const SICAKLIK_UST: f64 = 20.0;
/// Altin oran aramasinin daralma katsayisi.
const ALTIN_ORAN: f64 = 0.618_033_988_749_894_9;
/// Yazı-tura esigi: bunun altinda kalan bir bolge isabetsiz sayilir.
pub const KIRMIZI_TAVAN: f64 = 0.5;
/// Altin oran aramasinin tur sayisi; araligi ~0.618^tur kadar daraltir.
const ARAMA_TURU: usize = 90;

/// Neden bir olcum reddedildi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KalibrasyonHatasi {
    /// Elde yeterli kayit yok.
    YetersizKayit,
    /// Puan araligin disinda ya da bir ucta (logit'i sayi degil).
    GecersizPuan,
    /// Sicaklik araligin disinda.
    GecersizSicaklik,
    /// Hedef isabet `0.5..1.0` araliginin disinda.
    GecersizHedef,
    /// Yesil bandi olusturacak bir esik bulunamadi.
    YesilBantYok,
    /// Kirmizi bandi olusturacak bir esik bulunamadi.
    KirmiziBantYok,
    /// Iki bant kesisiyor: ayni guven hem yesil hem kirmizi olamaz.
    BantlarKesisiyor,
}

impl KalibrasyonHatasi {
    /// Sabit etiket. Metin uretmez, bu eslesmeden gelir.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::YetersizKayit => "yetersiz-kayit",
            Self::GecersizPuan => "gecersiz-puan",
            Self::GecersizSicaklik => "gecersiz-sicaklik",
            Self::GecersizHedef => "gecersiz-hedef",
            Self::YesilBantYok => "yesil-bant-yok",
            Self::KirmiziBantYok => "kirmizi-bant-yok",
            Self::BantlarKesisiyor => "bantlar-kesisiyor",
        }
    }
}

/// Bir kayit: basligin soyledigi puan ve gercekte dogru olup olmadigi.
///
/// Iki uctaki puan kabul edilmez (`0.0`, `1.0`): logit'i sayi degildir ve
/// "sonsuz guven" diye bir olcum yoktur. Uctaki bir kayit sessizce
/// kirpilmaz - kayit reddedilir, cunku kirpilan bir olcum olcum degildir.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HamKayit {
    /// Basligin soyledigi puan.
    pub puan: f64,
    /// Sonuc gercekten dogru muydu.
    pub dogru: bool,
}

impl HamKayit {
    /// Bir kayit, ya da aralik disindaysa `None`.
    #[must_use]
    pub fn yeni(puan: f64, dogru: bool) -> Option<Self> {
        if puan > 0.0 && puan < 1.0 && puan.is_finite() {
            Some(Self { puan, dogru })
        } else {
            None
        }
    }
}

/// Uydurmanin sonucu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SicaklikFit {
    /// Uydurulan sicaklik. `1.0` duzeltme yapilmadigi anlamina gelir.
    pub sicaklik: f64,
    /// Ham puanlarin egitim hatasi (`T = 1`).
    pub nll_ham: f64,
    /// Duzeltilmis puanlarin egitim hatasi.
    pub nll_fit: f64,
    /// Kac kayit uzerinde uyduruldu.
    pub kayit: u64,
}

impl SicaklikFit {
    /// Egitim hatasi gercekten dustu mu.
    ///
    /// `true` donmesi bir iddia degil, iki olculen sayinin karsilastirmasi;
    /// esitlik (duzeltmenin bir sey kazandirmadigi hal) `false` doner.
    #[must_use]
    pub fn kazanc_var(self) -> bool {
        self.nll_fit < self.nll_ham
    }
}

/// Olculen bantlar.
///
/// Sinirlar kayitlardan turer ve kova kenarlarina oturur: `yesil_alt`, bu ve
/// uzerindeki bos olmayan her kovada isabetin `hedef`i tuttugu en dusuk kenar;
/// `kirmizi_ust`, bu ve altindaki bos olmayan her kovada isabetin
/// [`KIRMIZI_TAVAN`]i asmadigi en yuksek kovanin ust kenari (bu kenar orta
/// bandin ilk noktasidir, kirmiziya dahil degildir).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bantlar {
    /// Kirmizi bandin ust siniri (haric).
    pub kirmizi_ust: f64,
    /// Yesil bandin alt siniri (dahil).
    pub yesil_alt: f64,
    /// Yesil bandin tutmasi gereken isabet.
    pub hedef: f64,
    /// Bantlarin uzerinde olculdugu kayit sayisi.
    pub kayit: u64,
    /// Bantlarin uzerinde olculdugu sicaklik.
    pub sicaklik: f64,
    /// Destek esiginin altinda kaldigi icin taramayi kesmeyen bos olmayan kova
    /// sayisi. Bu kovalar "olculmedi" sayilir: tek kayitlik bir kova o bolge
    /// hakkinda bir sey soylemez, ama sifir da soylemez - sayilir ve yazilir.
    pub destek_alti: u64,
}

/// Uc basamak: kararin kendi basina mi, k-of-n mi, yukseltme mi oldugu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basamak {
    /// Yesil: karar basi tek basina karar verebilir (politika esigi de tutuyorsa).
    TekBas,
    /// Orta: karar k-of-n oylamadan gecmeden durmaz.
    Konsensus,
    /// Kirmizi: karar yukseltilir; bas bu bandda kendi basina karar vermez.
    Yukselt,
}

impl Basamak {
    /// Sabit etiket.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::TekBas => "tek-bas",
            Self::Konsensus => "konsensus",
            Self::Yukselt => "yukselt",
        }
    }
}

/// `p / (1 - p)`, log uzayinda. Uctaki puanlarda `None`.
#[must_use]
pub fn logit(puan: f64) -> Option<f64> {
    if puan <= 0.0 || puan >= 1.0 || !puan.is_finite() {
        return None;
    }
    Some((puan / (1.0 - puan)).ln())
}

/// `1 / (1 + e^-z)`.
#[must_use]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Bir ham puanin sicaklikla duzeltilmis hali.
#[must_use]
pub fn kalibre_puan(puan: f64, sicaklik: f64) -> Option<f64> {
    if !(SICAKLIK_ALT..=SICAKLIK_UST).contains(&sicaklik) || !sicaklik.is_finite() {
        return None;
    }
    let z = logit(puan)?;
    Some(sigmoid(z / sicaklik))
}

/// Kayitlari kanonik bir siraya dizer.
///
/// Toplama sirasi bir kayan nokta toplaminin son basamaklarini degistirir;
/// girdi sirasi degistiginde ayni sayiyi vermek icin kayitlar once
/// `(puan, dogru)` uzerinden tam siraya dizilir.
fn kanonik(kayitlar: &[HamKayit]) -> Vec<HamKayit> {
    let mut sirali = kayitlar.to_vec();
    sirali.sort_by(|a, b| a.puan.total_cmp(&b.puan).then(a.dogru.cmp(&b.dogru)));
    sirali
}

/// Kayitlarin ortalama negatif log-olabilirligi (`T = 1` icin ham puanlar).
fn nll(kayitlar: &[HamKayit], sicaklik: f64) -> Option<f64> {
    if kayitlar.len() < ASGARI_KAYIT {
        return None;
    }
    let mut toplam = 0.0;
    for kayit in kayitlar {
        let q = kalibre_puan(kayit.puan, sicaklik)?;
        let (p_dogru, p_yanlis) = if kayit.dogru {
            (q, 1.0 - q)
        } else {
            (1.0 - q, q)
        };
        // log(0) yok: kalibre_puan ucta bir deger uretmez cunku kayitlar ucta
        // degil, ama bir sigma cok kucuk olabilir; taban log'u sonlu tutar.
        toplam -= p_dogru.max(f64::MIN_POSITIVE).ln();
        let _ = p_yanlis;
    }
    Some(toplam / kayitlar.len() as f64)
}

/// Altin oran aramasi: tek tepeli bir fonksiyonun en kucuk yerini bulur.
fn altin_arama(f: &dyn Fn(f64) -> f64, alt: f64, ust: f64) -> f64 {
    let mut a = alt;
    let mut b = ust;
    let mut c = b - (b - a) * ALTIN_ORAN;
    let mut d = a + (b - a) * ALTIN_ORAN;
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..ARAMA_TURU {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * ALTIN_ORAN;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * ALTIN_ORAN;
            fd = f(d);
        }
    }
    a.midpoint(b)
}

/// Ham kayitlardan sicakligi uydurur.
///
/// Arama log-sicaklik uzayinda yapilir (sicaklik bir olcek carpanidir, toplamsal
/// degil) ve deterministiktir: ayni kayitlar ayni sayiyi verir.
///
/// # Errors
/// [`KalibrasyonHatasi::YetersizKayit`] - `ASGARI_KAYIT` altinda kayit;
/// [`KalibrasyonHatasi::GecersizSicaklik`] - her iki ucun da egitim hatasi sonlu
/// degilse (bu, kayitlarin bozuk oldugu anlamina gelir, modulun degil).
pub fn sicaklik_uyarla(kayitlar: &[HamKayit]) -> Result<SicaklikFit, KalibrasyonHatasi> {
    if kayitlar.len() < ASGARI_KAYIT {
        return Err(KalibrasyonHatasi::YetersizKayit);
    }
    let kanonik = kanonik(kayitlar);
    let nll_ham = nll(&kanonik, 1.0).ok_or(KalibrasyonHatasi::GecersizSicaklik)?;
    let a = SICAKLIK_ALT.ln();
    let b = SICAKLIK_UST.ln();
    let hedef = |x: f64| nll(&kanonik, x.exp()).unwrap_or(f64::INFINITY);
    let en_iyi_x = altin_arama(&hedef, a, b);
    let sicaklik = en_iyi_x.exp();
    let nll_fit = nll(&kanonik, sicaklik).ok_or(KalibrasyonHatasi::GecersizSicaklik)?;
    // Uc noktalardan daha iyi olmayan bir sonuc "uydurma" sayilmaz: aramanin
    // tuzagi (dar bir aralik) sessizce en iyi sanilmasin diye uc degerler de
    // denir ve en kucuk olan kazanir. Esitlikte ham sicaklik kalir.
    let adaylar = [(1.0, nll_ham), (sicaklik, nll_fit)];
    let (kazanan_s, kazanan_nll) = adaylar
        .iter()
        .copied()
        .fold((1.0, nll_ham), |en_iyi, aday| {
            if aday.1 < en_iyi.1 {
                aday
            } else {
                en_iyi
            }
        });
    Ok(SicaklikFit {
        sicaklik: kazanan_s,
        nll_ham,
        nll_fit: kazanan_nll,
        kayit: kayitlar.len() as u64,
    })
}

/// Beklenen kalibrasyon hatasi (ECE): kova ortalama |iddia - isabet|.
///
/// Kova muhasebesi [`lubot_anlama::Calibration`]'dan gelir; burada yalnizca
/// kovalarin agirlikli ortalamasi alinir. Bos kayitta `None` doner: olculmemis
/// bir bas, sifir hata ile ayni sey degildir.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ece(kayitlar: &[HamKayit], sicaklik: f64) -> Option<f64> {
    if kayitlar.is_empty() || !(SICAKLIK_ALT..=SICAKLIK_UST).contains(&sicaklik) {
        return None;
    }
    let mut kalibrasyon = Calibration::new();
    for kayit in kayitlar {
        let q = kalibre_puan(kayit.puan, sicaklik)?;
        kalibrasyon.record(Outcome {
            confidence: q,
            correct: kayit.dogru,
        });
    }
    let toplam = kalibrasyon.total();
    if toplam == 0 {
        return None;
    }
    let mut agirlikli = 0.0;
    for rapor in kalibrasyon.gaps() {
        if rapor.samples == 0 {
            continue;
        }
        agirlikli += rapor.gap.abs() * rapor.samples as f64;
    }
    Some(agirlikli / toplam as f64)
}

/// Bir kovanin genisligi.
///
/// Bant sinirlari kova kenarlaridir: ara deger **uydurulmaz**, olcumun kendi
/// cozunurlugu neyse bantlar o kadar incedir. Kova sayisi
/// [`lubot_anlama::CALIBRATION_BUCKETS`] ile aynidir - ikinci bir kova sayaci
/// ayni soruya iki cevap verir.
#[must_use]
pub fn kova_genisligi() -> f64 {
    1.0 / CALIBRATION_BUCKETS as f64
}

/// Kovalari duzeltilmis puanlarla doldurur.
fn kovalar(kayitlar: &[HamKayit], sicaklik: f64) -> Result<Calibration, KalibrasyonHatasi> {
    let mut kalibrasyon = Calibration::new();
    for kayit in kayitlar {
        let q = kalibre_puan(kayit.puan, sicaklik).ok_or(KalibrasyonHatasi::GecersizPuan)?;
        kalibrasyon.record(Outcome {
            confidence: q,
            correct: kayit.dogru,
        });
    }
    Ok(kalibrasyon)
}

/// Bantlari kayitlardan olcer.
///
/// Yesil bandin alt siniri, **bu ve uzerindeki bos olmayan her kovada** isabet
/// `hedef`i tutan en dusuk kova kenari; kirmizi bandin ust siniri, **bu ve
/// altindaki bos olmayan her kovada** isabet [`KIRMIZI_TAVAN`]i asmayan en
/// yuksek kovanin ust kenari. Iki bolge arasinda kalan her sey orta banttir.
/// Sinirlar kova kenarlaridir: olcumun cozunurlugu neyse bant o kadar ince.
///
/// # Errors
/// - [`KalibrasyonHatasi::YetersizKayit`] - `ASGARI_KAYIT` altinda kayit;
/// - [`KalibrasyonHatasi::GecersizPuan`] - puanlardan biri duzeltilemiyorsa;
/// - [`KalibrasyonHatasi::GecersizSicaklik`], [`KalibrasyonHatasi::GecersizHedef`];
/// - [`KalibrasyonHatasi::YesilBantYok`] / [`KalibrasyonHatasi::KirmiziBantYok`] -
///   hicbir kesim `ASGARI_DESTEK` kadar kayitla ve butun kovalariyla kosulu
///   saglamiyorsa. Bu bir eksiklik degil, fail-closed bir sonuctur: olculemeyen
///   bandi "yok" saymak, her guvene yesil demek olurdu;
/// - [`KalibrasyonHatasi::BantlarKesisiyor`] - kirmizi band yesil bandin ustune
///   tasiyorsa (iki bolge ayni anda hem guvenilir hem guvenilmez olamaz).
#[allow(clippy::cast_precision_loss)]
pub fn bantlari_olc(
    kayitlar: &[HamKayit],
    sicaklik: f64,
    hedef: f64,
) -> Result<Bantlar, KalibrasyonHatasi> {
    if kayitlar.len() < ASGARI_KAYIT {
        return Err(KalibrasyonHatasi::YetersizKayit);
    }
    if !(KIRMIZI_TAVAN < hedef && hedef < 1.0) {
        return Err(KalibrasyonHatasi::GecersizHedef);
    }
    if !(SICAKLIK_ALT..=SICAKLIK_UST).contains(&sicaklik) {
        return Err(KalibrasyonHatasi::GecersizSicaklik);
    }
    let kalibrasyon = kovalar(kayitlar, sicaklik)?;
    let raporlar = kalibrasyon.gaps();
    let genislik = kova_genisligi();

    // Yesil: tepeden asagi inerken butun olculmus kovalar hedefi tutuyor mu.
    // "Olculmus" = destek esigini gecmis; tek kayitlik bir kova bir bolge
    // hakkinda hukum vermez, ne lehte ne aleyhte. Boyle kovalar sayilir ve
    // `destek_alti` olarak raporlanir.
    let mut yesil: Option<f64> = None;
    let mut destek: u64 = 0;
    let mut destek_alti: u64 = 0;
    for rapor in raporlar.iter().rev() {
        if rapor.samples > 0 && rapor.samples < ASGARI_DESTEK {
            destek_alti += 1;
            destek += rapor.samples;
            continue;
        }
        if rapor.samples > 0 && rapor.observed < hedef {
            break;
        }
        destek += rapor.samples;
        if rapor.samples > 0 && destek >= ASGARI_DESTEK {
            yesil = Some(rapor.low);
        }
    }

    // Kirmizi: tabandan yukari cikarken butun olculmus kovalar tavani asmiyor mu.
    let mut kirmizi: Option<f64> = None;
    let mut destek: u64 = 0;
    for rapor in &raporlar {
        if rapor.samples > 0 && rapor.samples < ASGARI_DESTEK {
            destek_alti += 1;
            destek += rapor.samples;
            continue;
        }
        if rapor.samples > 0 && rapor.observed > KIRMIZI_TAVAN {
            break;
        }
        destek += rapor.samples;
        if rapor.samples > 0 && destek >= ASGARI_DESTEK {
            kirmizi = Some(rapor.low + genislik);
        }
    }

    let yesil_alt = yesil.ok_or(KalibrasyonHatasi::YesilBantYok)?;
    let kirmizi_ust = kirmizi.ok_or(KalibrasyonHatasi::KirmiziBantYok)?;
    if kirmizi_ust > yesil_alt {
        return Err(KalibrasyonHatasi::BantlarKesisiyor);
    }
    Ok(Bantlar {
        kirmizi_ust,
        yesil_alt,
        hedef,
        kayit: kayitlar.len() as u64,
        sicaklik,
        destek_alti,
    })
}

/// Bir guveni basamaga koyar.
///
/// Sinir davranisi tanimli: `guven >= yesil_alt` yesil bandin icindedir (yesil
/// sinir dahil), `guven < kirmizi_ust` kirmizi bandin icindedir (kirmizi ust
/// kenar haric; o kenar orta bandin ilk noktasidir). Arada kalan her sey orta
/// banddir, iki sinir da bu bandin disinda kalir.
#[must_use]
pub fn basamak(guven: f64, bantlar: &Bantlar) -> Basamak {
    if guven >= bantlar.yesil_alt {
        Basamak::TekBas
    } else if guven < bantlar.kirmizi_ust {
        Basamak::Yukselt
    } else {
        Basamak::Konsensus
    }
}

/// Basamagi mevcut karar zincirine baglar.
///
/// Yesil bandda karar kendi basina verilebilir - ama politika esigi hala
/// gecerlidir (`tek_bas`), cunku bant olcumu esigi **degistirmez**, ona bir taban
/// daha ekler. Orta bandda `tek_bas` cagrilmaz: karar konsensusa gider, orada
/// yoksa yukseltilir. Kirmizi bandda hic denenmez.
#[must_use]
pub fn zincir(guven: Guven, karar: Karar, bantlar: &Bantlar, politika: &Politika) -> Sonuc {
    match basamak(guven.0.deger(), bantlar) {
        Basamak::TekBas => match tek_bas(karar, politika) {
            Ok(sonuc) => sonuc,
            // Politika gecersizse (k/n ya da esik bozuksa) karar verilmez.
            Err(_) => Sonuc::Yukselt(YukseltmeNedeni::GuvenEsikAltinda),
        },
        // Elle karar verilmez; cagiran k-of-n oylamayi kosar. Oylama yoksa
        // karar yukselir - bu, bandin istedigi seydir.
        Basamak::Konsensus => Sonuc::Yukselt(YukseltmeNedeni::KonsensusYok),
        Basamak::Yukselt => Sonuc::Yukselt(YukseltmeNedeni::GuvenEsikAltinda),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EvetHayirKarari, Puan};

    /// Deterministik dolgu: ayni tohum her makinede ayni sayilari verir.
    fn tohumlu(n: usize, tohum: u64) -> Vec<f64> {
        let mut sayac = tohum.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                sayac = sayac
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (sayac >> 33) as f64 / (1u64 << 31) as f64
            })
            .collect()
    }

    /// Bilinen bir sicaklikla uretilmis kayitlar: etiket, `T` ile duzeltilmis
    /// puandan orneklenir; yani verinin gercek sicakligi `T`'dir.
    fn bilinen_sicaklik(kayit_sayisi: usize, t: f64, tohum: u64) -> Vec<HamKayit> {
        let mut kayitlar = Vec::new();
        for (i, u) in tohumlu(kayit_sayisi, tohum).into_iter().enumerate() {
            // Puanlar log uzayda duzgun dagilsin: 0.02 .. 0.98.
            let puan = 0.02 + 0.96 * (i as f64 + u) / kayit_sayisi as f64;
            let puan = puan.clamp(0.001, 0.999);
            let q = kalibre_puan(puan, t).expect("kalibre puan");
            let dogru = u < q;
            kayitlar.push(HamKayit::yeni(puan, dogru).expect("kayit"));
        }
        kayitlar
    }

    /// Asiri guvenli kayitlar: puanlar yuksek, isabet dusuk.
    fn asiri_guvenli(n: usize) -> Vec<HamKayit> {
        let mut kayitlar = Vec::new();
        let degerler = tohumlu(n, 11);
        for (i, u) in degerler.into_iter().enumerate() {
            let puan = 0.75 + 0.24 * u;
            let dogru = i % 3 == 0;
            kayitlar.push(HamKayit::yeni(puan, dogru).expect("kayit"));
        }
        kayitlar
    }

    #[test]
    fn az_kayit_uyarlamayi_reddeder() {
        let az: Vec<HamKayit> = (0..ASGARI_KAYIT - 1)
            .map(|i| HamKayit::yeni(0.5 + 0.01 * i as f64, true).expect("kayit"))
            .collect();
        assert_eq!(sicaklik_uyarla(&az), Err(KalibrasyonHatasi::YetersizKayit));
        assert_eq!(
            bantlari_olc(&az, 1.0, HEDEF_KESINLIK),
            Err(KalibrasyonHatasi::YetersizKayit)
        );
    }

    #[test]
    fn uctaki_puanlar_reddedilir() {
        assert!(HamKayit::yeni(0.0, true).is_none());
        assert!(HamKayit::yeni(1.0, true).is_none());
        assert!(HamKayit::yeni(f64::NAN, true).is_none());
        assert!(HamKayit::yeni(-0.1, true).is_none());
        assert!(HamKayit::yeni(0.5, true).is_some());
        assert_eq!(logit(0.0), None);
        assert_eq!(logit(1.0), None);
    }

    #[test]
    fn bilinen_sicakligi_bulur() {
        for (t, tohum) in [(2.5, 3u64), (0.5, 5), (1.0, 7)] {
            let kayitlar = bilinen_sicaklik(400, t, tohum);
            let fit = sicaklik_uyarla(&kayitlar).expect("fit");
            let bagil = (fit.sicaklik - t).abs() / t;
            assert!(
                bagil < 0.25,
                "T={t} icin uydurulan {:.4} (bagil sapma {bagil:.3})",
                fit.sicaklik
            );
        }
    }

    #[test]
    fn sicaklik_egitim_hatasini_dusurur() {
        let kayitlar = asiri_guvenli(200);
        let fit = sicaklik_uyarla(&kayitlar).expect("fit");
        assert!(
            fit.kazanc_var(),
            "ham {:.6} duzeltilmis {:.6}",
            fit.nll_ham,
            fit.nll_fit
        );
        assert!(fit.sicaklik > 1.0, "asiri guvenli veri T>1 istemeli");
    }

    #[test]
    fn ece_kalibrasyon_bozukluguna_gore_artar() {
        let iyi = bilinen_sicaklik(2000, 1.0, 13);
        let kotu = bilinen_sicaklik(2000, 3.0, 13);
        let ece_iyi = ece(&iyi, 1.0).expect("ece");
        let ece_kotu = ece(&kotu, 1.0).expect("ece");
        assert!(
            ece_kotu > ece_iyi,
            "kotu {ece_kotu:.4} iyi {ece_iyi:.4} olmamali"
        );
    }

    #[test]
    fn ece_bos_kayitta_yok() {
        assert_eq!(ece(&[], 1.0), None);
        // Kalibrasyon kovalari crate disindan gorunmez ama ayrisan bant sayisi
        // dogrudan olculebilir: iki ayri kova, ayri gap uretir.
        let mut kalibrasyon = Calibration::new();
        kalibrasyon.record(Outcome {
            confidence: 0.15,
            correct: true,
        });
        kalibrasyon.record(Outcome {
            confidence: 0.95,
            correct: false,
        });
        let dolu: Vec<_> = kalibrasyon
            .gaps()
            .into_iter()
            .filter(|r| r.samples > 0)
            .collect();
        assert_eq!(dolu.len(), 2);
    }

    #[test]
    fn duzeltme_yesil_bandi_uretemez_ama_hatayi_gercekten_duzeltir() {
        // Ham puanlar asiri guvenli (gercek sicaklik 3.0). Uydurma egitim
        // hatasini ve ECE'yi gercekten dusurur - ama puanlarin kendisi 0.9
        // isabet tasimiyor: en iyi kova 0.708'de kaliyor. Duzeltme bilgi
        // uretemez, yalnizca var olani ortaya cikarir. Bu yuzden hedef 0.9'da
        // yesil band YOKTUR ve bas kendi basina karar veremez (fail-closed).
        let kayitlar = bilinen_sicaklik(2000, 3.0, 17);
        let fit = sicaklik_uyarla(&kayitlar).expect("fit");
        assert!(fit.kazanc_var(), "duzeltme egitim hatasini dusurmeli");
        assert_eq!(
            bantlari_olc(&kayitlar, fit.sicaklik, HEDEF_KESINLIK),
            Err(KalibrasyonHatasi::YesilBantYok)
        );
        // Hedef olculen seviyeye indirilince yesil band ortaya cikar: sinir
        // olcumden gelir, sabitten degil.
        let alcak = bantlari_olc(&kayitlar, fit.sicaklik, 0.7).expect("0.7 hedefiyle bantlar");
        assert!(alcak.yesil_alt >= 0.6 && alcak.yesil_alt <= 0.8);
        assert!(alcak.kirmizi_ust <= alcak.yesil_alt);
    }

    #[test]
    fn hedef_dustukce_yesil_band_genisler() {
        let kayitlar = bilinen_sicaklik(2000, 1.0, 23);
        let fit = sicaklik_uyarla(&kayitlar).expect("fit");
        let sik = bantlari_olc(&kayitlar, fit.sicaklik, 0.95).expect("0.95");
        let gevsek = bantlari_olc(&kayitlar, fit.sicaklik, 0.8).expect("0.8");
        assert!(
            gevsek.yesil_alt <= sik.yesil_alt,
            "gevsek hedef {:.2} daha yuksek sinir verdi ({:.2})",
            gevsek.hedef,
            gevsek.yesil_alt
        );
    }

    #[test]
    fn ece_duzeltmeyle_duser() {
        let kayitlar = bilinen_sicaklik(2000, 3.0, 19);
        let ham = ece(&kayitlar, 1.0).expect("ece");
        let fit = sicaklik_uyarla(&kayitlar).expect("fit");
        let duzeltilmis = ece(&kayitlar, fit.sicaklik).expect("ece");
        assert!(
            duzeltilmis < ham,
            "duzeltilmis {duzeltilmis:.4} ham {ham:.4} olmamali"
        );
    }

    #[test]
    fn basamak_sinirlari_tam_sinirda_uygulanir() {
        let bantlar = Bantlar {
            kirmizi_ust: 0.5,
            yesil_alt: 0.8,
            hedef: HEDEF_KESINLIK,
            kayit: 100,
            sicaklik: 1.0,
            destek_alti: 0,
        };
        // Yesil sinir dahil, kirmizi ust kenar haric.
        assert_eq!(basamak(0.8, &bantlar), Basamak::TekBas);
        assert_eq!(basamak(0.79, &bantlar), Basamak::Konsensus);
        assert_eq!(basamak(0.5, &bantlar), Basamak::Konsensus);
        assert_eq!(basamak(0.499_9, &bantlar), Basamak::Yukselt);
        assert_eq!(basamak(0.0, &bantlar), Basamak::Yukselt);
        assert_eq!(basamak(1.0, &bantlar), Basamak::TekBas);
    }

    #[test]
    fn zincir_basamaga_gore_karar_verir() {
        let bantlar = Bantlar {
            kirmizi_ust: 0.5,
            yesil_alt: 0.8,
            hedef: HEDEF_KESINLIK,
            kayit: 100,
            sicaklik: 1.0,
            destek_alti: 0,
        };
        let politika = Politika::varsayilan();
        let karar = Karar::EvetHayir(EvetHayirKarari {
            evet: true,
            olasilik: Puan::yeni(0.9).expect("puan"),
            guven: Guven(Puan::yeni(0.9).expect("puan")),
        });
        let yesil = zincir(
            Guven(Puan::yeni(0.9).expect("puan")),
            karar,
            &bantlar,
            &politika,
        );
        assert!(matches!(yesil, Sonuc::Kesin(_)));
        let orta = zincir(
            Guven(Puan::yeni(0.6).expect("puan")),
            karar,
            &bantlar,
            &politika,
        );
        assert_eq!(orta, Sonuc::Yukselt(YukseltmeNedeni::KonsensusYok));
        let kirmizi = zincir(
            Guven(Puan::yeni(0.2).expect("puan")),
            karar,
            &bantlar,
            &politika,
        );
        assert_eq!(kirmizi, Sonuc::Yukselt(YukseltmeNedeni::GuvenEsikAltinda));
        // Yesil band, politika esigini gevsetmez: esik altindaki guven yine yukselir.
        let yesil_band_dusuk_guven = zincir(
            Guven(Puan::yeni(0.85).expect("puan")),
            Karar::EvetHayir(EvetHayirKarari {
                evet: true,
                olasilik: Puan::yeni(0.55).expect("puan"),
                guven: Guven(Puan::yeni(0.55).expect("puan")),
            }),
            &bantlar,
            &politika,
        );
        assert_eq!(
            yesil_band_dusuk_guven,
            Sonuc::Yukselt(YukseltmeNedeni::GuvenEsikAltinda)
        );
    }

    #[test]
    fn ayni_girdi_ayni_sayilari_verir() {
        let kayitlar = bilinen_sicaklik(2000, 1.0, 37);
        let a = sicaklik_uyarla(&kayitlar).expect("fit");
        let b = sicaklik_uyarla(&kayitlar).expect("fit");
        assert_eq!(a.sicaklik.to_bits(), b.sicaklik.to_bits());
        let ba = bantlari_olc(&kayitlar, a.sicaklik, HEDEF_KESINLIK).expect("bantlar");
        let bb = bantlari_olc(&kayitlar, b.sicaklik, HEDEF_KESINLIK).expect("bantlar");
        assert_eq!(ba.yesil_alt.to_bits(), bb.yesil_alt.to_bits());
        assert_eq!(ba.kirmizi_ust.to_bits(), bb.kirmizi_ust.to_bits());
    }

    #[test]
    fn kayit_sirasi_sonucu_degistirmez() {
        let kayitlar = bilinen_sicaklik(2000, 1.0, 41);
        let mut ters = kayitlar.clone();
        ters.reverse();
        let a = sicaklik_uyarla(&kayitlar).expect("fit");
        let b = sicaklik_uyarla(&ters).expect("fit");
        // Bit-esitlik: toplama sirasi kanonik hale getirildi, yani girdi
        // sirasi sonucun son basamagini bile degistirmez.
        assert_eq!(
            a.sicaklik.to_bits(),
            b.sicaklik.to_bits(),
            "sira fit'i degistirdi: {} vs {}",
            a.sicaklik,
            b.sicaklik
        );
        let ba = bantlari_olc(&kayitlar, a.sicaklik, HEDEF_KESINLIK).expect("bantlar");
        let bb = bantlari_olc(&ters, b.sicaklik, HEDEF_KESINLIK).expect("bantlar");
        assert!((ba.yesil_alt - bb.yesil_alt).abs() < 1e-12);
        assert!((ba.kirmizi_ust - bb.kirmizi_ust).abs() < 1e-12);
    }

    #[test]
    fn bantlar_olculen_kayitlari_kapsar() {
        let kayitlar = bilinen_sicaklik(2000, 1.0, 43);
        let fit = sicaklik_uyarla(&kayitlar).expect("fit");
        let bantlar = bantlari_olc(&kayitlar, fit.sicaklik, HEDEF_KESINLIK).expect("bantlar");
        let mut yesil = 0u64;
        let mut kirmizi = 0u64;
        for kayit in &kayitlar {
            let q = kalibre_puan(kayit.puan, fit.sicaklik).expect("puan");
            match basamak(q, &bantlar) {
                Basamak::TekBas => yesil += 1,
                Basamak::Yukselt => kirmizi += 1,
                Basamak::Konsensus => {}
            }
        }
        // Bantlar bos kume olmamali: her iki bolgede de kayit var ve toplamlari
        // butunu asmaz.
        assert!(yesil >= ASGARI_DESTEK, "yesil band bos");
        assert!(kirmizi >= ASGARI_DESTEK, "kirmizi band bos");
        assert!(yesil + kirmizi <= kayitlar.len() as u64);
    }
}
