//! Kademe egitimi: yiginin her derinligi dagitilabilir bir modeldir (aday).
//!
//! Servis tarafindaki derinlik merdiveni (`tasiyici`) her ara derinligi okunabilir
//! bir basamak sayar; kesit ailesi her derinlikte kosulabilen agirliklar verir.
//! Bu ikisinin egitimdeki karsiligi eksikti: egitim hedefi yalniz en derin
//! cikisa kayip yuklerse, ara basamaklar "tasinir ama hic egitilmez" durumunda
//! kalir. Bu modul o eksik parcanin **adayimidir**: ara kademelerin cikislari da
//! kayip tasir, kayip kademeler uzerine agirlikli bilesimle dagitilir.
//!
//! Bu bir egitim hedefi degisikligidir ve mimari karar olarak isaretlidir
//! (M4). Bu yuzden modul **bagli degildir**: `model_spec.json` degismez,
//! `lubot-a1` ailesi degismez, egitim dongusu bu modulu cagirmaz. Aday, kendi
//! olculen sozlesmesiyle durur; baglanma karari operator damgasi bekler.
//!
//! Sozlesme (hepsi testte olculur, iddia degil):
//!
//! * **Taban iceride**: son kademeye agirlik 1, digerlerine 0 verilirse
//!   kademeli kayip, taban (yalniz en derin cikis) kaybinin **ta kendisidir** -
//!   bit duzeyinde ayni sayi, `to_bits` ile olculur. Yani modul tabani icinde
//!   tasir; kademeler kapaliyken davranis degismez.
//! * **Dogrusal bilesim**: kademeli kayip, her kademeye ayrica olculen taban
//!   kayiplarinin bildirilen agirliklarla toplamidir.
//! * **Gradyan elle yazildi ve sonlu farkla denetlendi**: her kademe cikisinin
//!   her boyutu icin analitik gradyan, merkezi sonlu farkla karsilastirilir;
//!   en kotu goreli sapma esik altinda olmalidir.
//! * **Her kademeden gradyan akar**: pozitif agirlikli kademenin gradyan
//!   normu sifirdan buyuktur; agirligi sifir olan kademenin gradyani **tam**
//!   sifirdir (bit duzeyinde).
//! * **Kararli bicim**: kayip, log-sum-exp biciminde hesaplanir; buyuk logitler
//!   (orn. ±1000) kaybi sonsuz yapamaz, gradyan NaN uretemez.
//! * **Agirliklar bire normalize edilir**: spec kurulurken agirliklar toplami
//!   bire indirgenir; olcum bunu dogrular. Sema secimi (uniform, azalan,
//!   harmonik) kodda sabit degildir - olcum kararidir, `olculmedi` listesinde
//!   durur.
//!
//! Ogrenilebilir parametresi yoktur: kayip bilesimi bir hesaplamadir, agirlik
//! degildir. Butcesi bu yuzden parametre muhasebesinde sifirdir ve bunu
//! soylemek `parametre_sayisi` ile mumkundur.

/// Derinligin ust siniri (K6 butcesi icinde kalinir; daha derin yiginlar
/// olculmeden bu adayin kapsamina girmez).
pub const DERINLIK_UST_SINIRI: usize = 32;
/// Tek spec'in tasiyabilecegi kademelerin (ara + son) ust siniri.
pub const KADEME_UST_SINIRI: usize = 8;

/// Kademe egitiminin sekli: hangi derinlikler kayip tasir, hangi agirlikla.
#[derive(Debug, Clone, PartialEq)]
pub struct KademeSpec {
    /// Yiginin tam derinligi; son kademe daima budur.
    pub ust_derinlik: usize,
    /// Kayip tasiyan ara derinlikler; kesin artan sirada, `1..ust_derinlik`
    /// araliginda.
    pub ara_kademeler: Vec<usize>,
    /// Normalize edilmis kadem agirliklari; son ele son derinlige karsilik
    /// gelir. Toplamlari 1'dir (olculur).
    pub agirliklar: Vec<f64>,
}

/// Sekil hatasi: spec ya da kayip girdisi kurulmadan once reddedilir.
#[derive(Debug, Clone, PartialEq)]
pub enum KademeHatasi {
    /// Tam derinlik sifir olamaz.
    SifirDerinlik,
    /// Tam derinlik ust siniri asiyor.
    DerinlikUstSiniriAsildi(usize),
    /// Toplam kademe sayisi (ara + son) ust siniri asiyor.
    KademeUstSiniriAsildi(usize),
    /// Ara kademe `1..ust_derinlik` araliginda degil: `(kademe, ust)`.
    KademeAralikDisi(usize, usize),
    /// Ara kademeler kesin artan sirada degil: `(ilk, ikinci)`.
    KademelerAzalmaz(usize, usize),
    /// Agirlik sayisi kademe sayisiyla uyusmuyor: `(gelen, beklenen)`.
    AgirlikUzunluk(usize, usize),
    /// Agirlik sonlu degil (NaN ya da sonsuz).
    GecersizAgirlik(f64),
    /// Agirlik negatif.
    NegatifAgirlik(f64),
    /// Agirliklarin toplami sifir; hicbir kademe kayip tasimaz.
    SifirToplamAgirlik,
    /// Cikis sayisi kademe sayisiyla uyusmuyor: `(gelen, beklenen)`.
    CikisUzunluk(usize, usize),
    /// Bir kademede genislik hedefle uyusmuyor: `(sira, gelen, beklenen)`.
    CikisGenislik(usize, usize, usize),
    /// Kademede logit sonlu degil: `(sira, deger)`.
    GecersizCikis(usize, f64),
    /// Hedef bos.
    HedefBos,
    /// Hedef girdisi sonlu ve negatif olmayan degil.
    GecersizHedef(f64),
    /// Hedefin toplami sifir; dagilim tanimsiz.
    SifirToplamHedef,
}

impl KademeSpec {
    /// Sekli dogrular, agirliklari bire normalize eder. Hatali sekil ile
    /// modul kurulmaz.
    ///
    /// # Errors
    ///
    /// Derinlik, kademe araligi/sirasi, agirlik sayisi/degeri/toplami
    /// sozlesmeye aykiriysa adlandirilmis [`KademeHatasi`] doner; sessiz
    /// duzeltme yoktur.
    pub fn yeni(
        ust_derinlik: usize,
        ara_kademeler: &[usize],
        agirliklar: &[f64],
    ) -> Result<Self, KademeHatasi> {
        if ust_derinlik == 0 {
            return Err(KademeHatasi::SifirDerinlik);
        }
        if ust_derinlik > DERINLIK_UST_SINIRI {
            return Err(KademeHatasi::DerinlikUstSiniriAsildi(ust_derinlik));
        }
        if ara_kademeler.len() + 1 > KADEME_UST_SINIRI {
            return Err(KademeHatasi::KademeUstSiniriAsildi(ara_kademeler.len() + 1));
        }
        for (sira, kademe) in ara_kademeler.iter().enumerate() {
            if *kademe == 0 || *kademe >= ust_derinlik {
                return Err(KademeHatasi::KademeAralikDisi(*kademe, ust_derinlik));
            }
            if sira > 0 && *kademe <= ara_kademeler[sira - 1] {
                return Err(KademeHatasi::KademelerAzalmaz(
                    ara_kademeler[sira - 1],
                    *kademe,
                ));
            }
        }
        if agirliklar.len() != ara_kademeler.len() + 1 {
            return Err(KademeHatasi::AgirlikUzunluk(
                agirliklar.len(),
                ara_kademeler.len() + 1,
            ));
        }
        for agirlik in agirliklar {
            if !agirlik.is_finite() {
                return Err(KademeHatasi::GecersizAgirlik(*agirlik));
            }
            if *agirlik < 0.0 {
                return Err(KademeHatasi::NegatifAgirlik(*agirlik));
            }
        }
        let toplam: f64 = agirliklar.iter().sum();
        if toplam <= 0.0 {
            return Err(KademeHatasi::SifirToplamAgirlik);
        }
        let normalize = agirliklar.iter().map(|a| a / toplam).collect();
        Ok(Self {
            ust_derinlik,
            ara_kademeler: ara_kademeler.to_vec(),
            agirliklar: normalize,
        })
    }

    /// Kayip tasiyan kademelerin tam listesi: ara kademeler + tam derinlik.
    #[must_use]
    pub fn kademeler(&self) -> Vec<usize> {
        let mut tumu = self.ara_kademeler.clone();
        tumu.push(self.ust_derinlik);
        tumu
    }

    /// Ogrenilebilir parametre sayisi: kayip bilesimi agirlik tutmaz.
    #[must_use]
    pub fn parametre_sayisi(&self) -> usize {
        0
    }
}

/// Kademeli kayip sonucu: toplam, kademelerin kendi kayiplari ve her kademe
/// cikisina elle yazilmis gradyan.
#[derive(Debug, Clone, PartialEq)]
pub struct KademeKayip {
    /// Agirlikli bilesim: `toplam = Σ agirlik_kademe · kayip_kademe`.
    pub toplam: f64,
    /// Her kademeye ayrica olculen taban kaybi (agirliksiz).
    pub kademe_kayiplari: Vec<f64>,
    /// Her kademe cikisinin her boyutuna gradyan.
    pub gradyanlar: Vec<Vec<f64>>,
    /// Kullanilan (normalize edilmis) agirliklar; olcum bunlari dogrular.
    pub agirliklar: Vec<f64>,
    /// Kayip tasiyan kademelerin derinlikleri.
    pub kademeler: Vec<usize>,
}

/// Log-sum-exp: `en_buyuk + ln(Σ exp(x - en_buyuk))`. Degerler cagri oncesi
/// dogrulanmis sonlu logitlerdir; bu yuzden sonuc daima sonludur.
fn log_toplam_us(degerler: &[f64]) -> f64 {
    let en_buyuk = degerler.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !en_buyuk.is_finite() {
        return f64::NEG_INFINITY;
    }
    let toplam: f64 = degerler.iter().map(|d| (d - en_buyuk).exp()).sum();
    en_buyuk + toplam.ln()
}

/// Hedefi dogrular ve dagilima indirger (toplami 1). Hedef sayim olarak da
/// verilebilir: `[2, 0, 4]` ile `[1, 0, 2]` ayni dagilimdir.
fn hedefi_normalize(hedef: &[f64]) -> Result<Vec<f64>, KademeHatasi> {
    if hedef.is_empty() {
        return Err(KademeHatasi::HedefBos);
    }
    for deger in hedef {
        if !deger.is_finite() || *deger < 0.0 {
            return Err(KademeHatasi::GecersizHedef(*deger));
        }
    }
    let toplam: f64 = hedef.iter().sum();
    if toplam <= 0.0 {
        return Err(KademeHatasi::SifirToplamHedef);
    }
    Ok(hedef.iter().map(|d| d / toplam).collect())
}

/// Kararli capraz entropi: `lse(cikis) - Σ dagilim · cikis`.
///
/// Yumusak olasilik `exp(o - lse)` biciminde ara deger uretmez; buyuk logitler
/// altta yuvarlanip sifir olsa bile kayip sonlu kalir. Gradyan tarafinda
/// yumusak olasilik ayrica ve yalnizca fark icin hesaplanir.
fn kararli_capraz_entropi(cikis: &[f64], dagilim: &[f64]) -> f64 {
    let lse = log_toplam_us(cikis);
    let beklenen: f64 = dagilim.iter().zip(cikis).map(|(p, o)| p * o).sum();
    lse - beklenen
}

/// Tek cikisin yumusak olasiliklari: `exp(o - lse)`.
fn yumusak_olasilik(cikis: &[f64]) -> Vec<f64> {
    let lse = log_toplam_us(cikis);
    cikis.iter().map(|o| (o - lse).exp()).collect()
}

/// Taban kayip: kademeler kapaliyken egitimin tasidigi tek kayip - yalniz en
/// derin cikisin capraz entropisi. Kademeli kayip, son kademeye agirlik 1
/// verilince buna **bit duzeyinde** esittir (test olcer).
///
/// # Errors
///
/// Hedef bos/gecersiz ya da cikis genisligi hedefle uyusmuyorsa adlandirilmis
/// [`KademeHatasi`] doner.
pub fn taban_kayip(son_cikis: &[f64], hedef: &[f64]) -> Result<f64, KademeHatasi> {
    let dagilim = hedefi_normalize(hedef)?;
    if son_cikis.len() != hedef.len() {
        return Err(KademeHatasi::CikisGenislik(0, son_cikis.len(), hedef.len()));
    }
    for (sira, deger) in son_cikis.iter().enumerate() {
        if !deger.is_finite() {
            return Err(KademeHatasi::GecersizCikis(sira, *deger));
        }
    }
    Ok(kararli_capraz_entropi(son_cikis, &dagilim))
}

/// Kademeli kayip: ileri gecis + elle yazilmis geri gecis.
///
/// Ileri: her kademeye `L_k = lse(o_k) - Σ dagilim · o_k`, toplam
/// `Σ agirlik_k · L_k`. Geri: kademeye gradyan
/// `agirlik_k · (yumsak_olasilik - dagilim)` - carpim-zincirinin ta kendisi,
/// bilesim katsayisi sadece distan carpar.
///
/// # Errors
///
/// Hedef bos/gecersiz, cikis sayisi kademe sayisiyla uyusmuyor, genislik
/// uyusmuyor ya da bir logit sonlu degilse adlandirilmis [`KademeHatasi`]
/// doner.
pub fn kademeli_kayip(
    spec: &KademeSpec,
    cikislar: &[Vec<f64>],
    hedef: &[f64],
) -> Result<KademeKayip, KademeHatasi> {
    let dagilim = hedefi_normalize(hedef)?;
    let kademeler = spec.kademeler();
    if cikislar.len() != kademeler.len() {
        return Err(KademeHatasi::CikisUzunluk(cikislar.len(), kademeler.len()));
    }
    for (sira, cikis) in cikislar.iter().enumerate() {
        if cikis.len() != hedef.len() {
            return Err(KademeHatasi::CikisGenislik(sira, cikis.len(), hedef.len()));
        }
        for (boyut, deger) in cikis.iter().enumerate() {
            if !deger.is_finite() {
                return Err(KademeHatasi::GecersizCikis(boyut, *deger));
            }
        }
    }
    let mut kademe_kayiplari = Vec::with_capacity(kademeler.len());
    let mut gradyanlar = Vec::with_capacity(kademeler.len());
    let mut toplam = 0.0;
    for (sira, cikis) in cikislar.iter().enumerate() {
        let kayip = kararli_capraz_entropi(cikis, &dagilim);
        // Agirlik sifirsa 0 * kayip = 0 (kayip sonlu oldugundan NaN yok);
        // agirlik birdense kaybin kendisi. Toplama sirasi sabittir: ayni
        // girdi her makinede ayni bitleri verir.
        toplam += spec.agirliklar[sira] * kayip;
        kademe_kayiplari.push(kayip);
        let olasilik = yumusak_olasilik(cikis);
        let gradyan = olasilik
            .iter()
            .zip(&dagilim)
            .map(|(q, p)| spec.agirliklar[sira] * (q - p))
            .collect();
        gradyanlar.push(gradyan);
    }
    Ok(KademeKayip {
        toplam,
        kademe_kayiplari,
        gradyanlar,
        agirliklar: spec.agirliklar.clone(),
        kademeler,
    })
}

/// Ornek kademe cikislari: tohumsuz tam sayi karma ile uretilir (tekrarlanabilir,
/// makineden makineye degismez). Her kademeye bir cikis; degerler [-4, 4]
/// araliginda logitlerdir.
#[must_use]
pub fn ornek_cikislar(kademe_sayisi: usize, genislik: usize, egim: u64) -> Vec<Vec<f64>> {
    let mut cikislar = Vec::with_capacity(kademe_sayisi);
    for k in 0..kademe_sayisi {
        let mut cikis = Vec::with_capacity(genislik);
        for b in 0..genislik {
            let karisim = (k as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((b as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                .wrapping_add(egim);
            let x = ((karisim >> 11) & 0xFFFF) as f64 / 65535.0;
            cikis.push(x * 8.0 - 4.0);
        }
        cikislar.push(cikis);
    }
    cikislar
}

/// Ornek hedef: sayim olarak verilir (0..3); bazi elemanlar sifirdir - dagilim
/// teste sifir olasilikli siniflari da tasir.
#[must_use]
pub fn ornek_hedef(genislik: usize, egim: u64) -> Vec<f64> {
    let mut hedef = Vec::with_capacity(genislik);
    for b in 0..genislik {
        let karisim = 0xD1B5_4A32_D192_ED03u64
            .wrapping_add((b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(egim);
        hedef.push(((karisim >> 11) & 0x3) as f64);
    }
    hedef
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(ara: &[usize], agirliklar: &[f64]) -> KademeSpec {
        KademeSpec::yeni(6, ara, agirliklar)
            .unwrap_or_else(|hata| panic!("spec kurulmadi: {hata:?}"))
    }

    #[test]
    fn sekil_hatalari_reddedilir() {
        assert!(matches!(
            KademeSpec::yeni(0, &[2], &[0.5, 0.5]),
            Err(KademeHatasi::SifirDerinlik)
        ));
        assert!(matches!(
            KademeSpec::yeni(DERINLIK_UST_SINIRI + 1, &[2], &[0.5, 0.5]),
            Err(KademeHatasi::DerinlikUstSiniriAsildi(_))
        ));
        assert!(matches!(
            KademeSpec::yeni(10, &[1, 2, 3, 4, 5, 6, 7, 8, 9], &[0.1; 10]),
            Err(KademeHatasi::KademeUstSiniriAsildi(10))
        ));
        assert!(matches!(
            KademeSpec::yeni(6, &[6], &[0.5, 0.5]),
            Err(KademeHatasi::KademeAralikDisi(6, 6))
        ));
        assert!(matches!(
            KademeSpec::yeni(6, &[0], &[0.5, 0.5]),
            Err(KademeHatasi::KademeAralikDisi(0, 6))
        ));
        assert!(matches!(
            KademeSpec::yeni(6, &[4, 2], &[0.4, 0.3, 0.3]),
            Err(KademeHatasi::KademelerAzalmaz(4, 2))
        ));
        assert!(matches!(
            KademeSpec::yeni(6, &[2, 4], &[1.0]),
            Err(KademeHatasi::AgirlikUzunluk(1, 3))
        ));
        assert!(matches!(
            KademeSpec::yeni(6, &[2, 4], &[0.5, f64::NAN, 0.5]),
            Err(KademeHatasi::GecersizAgirlik(_))
        ));
        assert!(matches!(
            KademeSpec::yeni(6, &[2, 4], &[0.5, -0.1, 0.5]),
            Err(KademeHatasi::NegatifAgirlik(-0.1))
        ));
        assert!(matches!(
            KademeSpec::yeni(6, &[2, 4], &[0.0, 0.0, 0.0]),
            Err(KademeHatasi::SifirToplamAgirlik)
        ));
        let s = spec(&[2, 4], &[0.25, 0.25, 0.5]);
        let hedef = ornek_hedef(16, 11);
        assert!(matches!(
            kademeli_kayip(&s, &ornek_cikislar(2, 16, 3), &hedef),
            Err(KademeHatasi::CikisUzunluk(2, 3))
        ));
        assert!(matches!(
            kademeli_kayip(&s, &ornek_cikislar(3, 8, 3), &hedef),
            Err(KademeHatasi::CikisGenislik(0, 8, 16))
        ));
        let mut bozuk = ornek_cikislar(3, 16, 3);
        bozuk[1][4] = f64::INFINITY;
        assert!(matches!(
            kademeli_kayip(&s, &bozuk, &hedef),
            Err(KademeHatasi::GecersizCikis(4, _))
        ));
        assert!(matches!(
            kademeli_kayip(&s, &ornek_cikislar(3, 16, 3), &[]),
            Err(KademeHatasi::HedefBos)
        ));
        let negatif_hedef = vec![1.0, -0.5, 1.0];
        assert!(matches!(
            taban_kayip(&[0.0; 3], &negatif_hedef),
            Err(KademeHatasi::GecersizHedef(-0.5))
        ));
        assert!(matches!(
            taban_kayip(&[0.0; 3], &[0.0; 3]),
            Err(KademeHatasi::SifirToplamHedef)
        ));
    }

    #[test]
    fn agirliklar_bire_normalize_edilir() {
        // 2:1:1 orani normalize sonrasi da ayni orandir ve toplam 1'dir.
        let s = spec(&[2, 4], &[2.0, 1.0, 1.0]);
        let toplam: f64 = s.agirliklar.iter().sum();
        assert!(
            (toplam - 1.0).abs() < 1e-12,
            "agirlik toplami {toplam} degil 1"
        );
        assert!((s.agirliklar[0] - 0.5).abs() < 1e-12, "oran bozuldu");
        // Cikarilan kademe listede yok: spec'in kademe listesi sozdur.
        assert_eq!(s.kademeler(), vec![2, 4, 6], "kademeler ara + tam derinlik");
    }

    #[test]
    fn parametre_sayisi_sifirdir() {
        let s = spec(&[2, 4], &[1.0, 1.0, 2.0]);
        assert_eq!(s.parametre_sayisi(), 0, "kayip bicimi parametre tutmaz");
    }

    #[test]
    fn son_kademe_tabanin_kendisi() {
        // Son kademe agirligi 1, digerleri 0: kademeli kayip taban kaybin ta
        // kendisidir - bit duzeyinde (0.0 * sonlu = 0.0, 1.0 * L = L).
        let hedef = ornek_hedef(16, 11);
        let cikislar = ornek_cikislar(3, 16, 7);
        let taban_spec = spec(&[2, 4], &[0.0, 0.0, 1.0]);
        let kademeli = kademeli_kayip(&taban_spec, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("kademeli kosmadi: {hata:?}"));
        let taban = taban_kayip(&cikislar[2], &hedef)
            .unwrap_or_else(|hata| panic!("taban kosmadi: {hata:?}"));
        assert_eq!(
            kademeli.toplam.to_bits(),
            taban.to_bits(),
            "son kademe agirligi 1 iken kademeli kayip tabanla bit-ozdes olmali"
        );
    }

    #[test]
    fn kademeli_kayip_agirlikli_dogrusal_kombinasyon() {
        // Toplam, kademelerin bagimsiz olculen taban kayiplarinin bildirilen
        // agirliklarla toplamidir; bilesim katsayisi baska bir sey olamaz.
        let agirliklar = [0.25, 0.25, 0.5];
        let s = spec(&[2, 4], &agirliklar);
        let hedef = ornek_hedef(16, 11);
        let cikislar = ornek_cikislar(3, 16, 7);
        let sonuc = kademeli_kayip(&s, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("kademeli kosmadi: {hata:?}"));
        let mut beklenen = 0.0;
        for (sira, cikis) in cikislar.iter().enumerate() {
            let taban = taban_kayip(cikis, &hedef)
                .unwrap_or_else(|hata| panic!("kademe kaybi olculemedi: {hata:?}"));
            assert!(
                (taban - sonuc.kademe_kayiplari[sira]).abs() < 1e-12,
                "kademe kaybi bagimsiz olcumden sapti"
            );
            beklenen += s.agirliklar[sira] * taban;
        }
        assert!(
            (sonuc.toplam - beklenen).abs() < 1e-12,
            "kademeli toplam {beklenen} olmali, {} geldi",
            sonuc.toplam
        );
        assert_eq!(sonuc.kademeler, vec![2, 4, 6], "kademeler spec'ten geldi");
    }

    #[test]
    fn kademeli_kayip_kademelerin_arasindadir() {
        // Uniform agirliklarda toplam, kademe kayiplarinin konveks
        // bilesimidir: en kucuk ve en buyuk kademe kaybinin arasinda kalir.
        let s = spec(&[2, 4], &[1.0, 1.0, 1.0]);
        let hedef = ornek_hedef(16, 11);
        let cikislar = ornek_cikislar(3, 16, 7);
        let sonuc = kademeli_kayip(&s, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("kademeli kosmadi: {hata:?}"));
        let en_kucuk = sonuc
            .kademe_kayiplari
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let en_buyuk = sonuc
            .kademe_kayiplari
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            sonuc.toplam >= en_kucuk && sonuc.toplam <= en_buyuk,
            "toplam {} kademelerin arasinda degil ({en_kucuk}..{en_buyuk})",
            sonuc.toplam
        );
    }

    #[test]
    fn gradyan_sonlu_farkla_uyumlu() {
        // Elle yazilan geri gecis, her kademe cikisinin her boyutunda merkezi
        // sonlu farkla karsilastirilir. Gorel sapma cozulebilir gradyanlarda
        // esik altinda, cozulemeyenlerde mutlak taban uygulanir.
        let s = spec(&[2, 4], &[0.25, 0.25, 0.5]);
        let hedef = ornek_hedef(16, 11);
        let cikislar = ornek_cikislar(3, 16, 7);
        let sonuc = kademeli_kayip(&s, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("kademeli kosmadi: {hata:?}"));
        let adim = 1e-5;
        let mut en_kotu = 0.0f64;
        for sira in 0..cikislar.len() {
            for boyut in 0..cikislar[sira].len() {
                let mut arti = cikislar.clone();
                let mut eksi = cikislar.clone();
                arti[sira][boyut] += adim;
                eksi[sira][boyut] -= adim;
                let l_arti = kademeli_kayip(&s, &arti, &hedef)
                    .unwrap_or_else(|hata| panic!("ileri arti kosmadi: {hata:?}"))
                    .toplam;
                let l_eksi = kademeli_kayip(&s, &eksi, &hedef)
                    .unwrap_or_else(|hata| panic!("ileri eksi kosmadi: {hata:?}"))
                    .toplam;
                let sayisal = (l_arti - l_eksi) / (2.0 * adim);
                let analitik = sonuc.gradyanlar[sira][boyut];
                let fark = (sayisal - analitik).abs();
                // Gorel sipmis payda tabani: kayip ~2.3, adim 1e-5 iken sonlu
                // farkin yuvarlama gurultusu ~eps * kayip / adim = 2.5e-11;
                // bu gurultu ~2.5e-5 buyuklugundeki bir gradyanda 1e-6 GORECE
                // hata olarak okunur - gradyan yanlis degil, cozunurluk siniri.
                // Bu yuzden payda 1e-4 tabanina oturur (cekirdegin ayni dersi,
                // bu modulun olculmus gurultu olcegine uyarlanmis hali).
                let gorel = fark / analitik.abs().max(1e-4);
                en_kotu = en_kotu.max(gorel);
            }
        }
        assert!(
            en_kotu < 1e-6,
            "gradyan sonlu farkla uyusmadi: en kotu sapma {en_kotu:.3e}"
        );
    }

    #[test]
    fn her_kademeden_gradyan_akar() {
        let hedef = ornek_hedef(16, 11);
        let cikislar = ornek_cikislar(3, 16, 7);
        // Pozitif agirlikli her kademeden gradyan akar.
        let s = spec(&[2, 4], &[1.0, 1.0, 1.0]);
        let sonuc = kademeli_kayip(&s, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("kademeli kosmadi: {hata:?}"));
        for (sira, gradyan) in sonuc.gradyanlar.iter().enumerate() {
            let norm: f64 = gradyan.iter().copied().map(f64::abs).sum();
            assert!(
                norm > 0.0,
                "{sira}. kademenin gradyani akmali (norm {norm})"
            );
        }
        // Agirligi sifir olan kademeye gradyan tam sifirdir: 0.0 * sonlu =
        // 0.0 (isaretli sifir dahi olabilir; bit duzeyinde denetlenir).
        let s_sifir = spec(&[2, 4], &[1.0, 0.0, 0.0]);
        let sonuc_sifir = kademeli_kayip(&s_sifir, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("kademeli kosmadi: {hata:?}"));
        for deger in &sonuc_sifir.gradyanlar[1] {
            assert!(
                deger.to_bits() == 0.0f64.to_bits() || deger.to_bits() == (-0.0f64).to_bits(),
                "sifir agirlikli kademeye gradyan degil sifir geldi: {deger}"
            );
        }
    }

    #[test]
    fn buyuk_logitler_kararli_kalir() {
        // Log-sum-exp bicimi: ±1000 logitler kaybi sonsuz yapmaz, gradyan NaN
        // uretmez (yumusak olasilik altta yuvarlanip sifir olsa bile).
        let s = spec(&[3], &[0.5, 0.5]);
        let hedef = vec![1.0, 0.0, 1.0, 0.0];
        let cikislar = vec![
            vec![1000.0, -1000.0, 999.0, -999.0],
            vec![-1000.0, 1000.0, -999.0, 999.0],
        ];
        let sonuc = kademeli_kayip(&s, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("buyuk logit kabul edilmedi: {hata:?}"));
        assert!(
            sonuc.toplam.is_finite(),
            "kayip sonlu olmali, {} geldi",
            sonuc.toplam
        );
        for gradyan in &sonuc.gradyanlar {
            for deger in gradyan {
                assert!(deger.is_finite(), "gradyan sonlu olmali: {deger}");
            }
        }
    }

    #[test]
    fn hedef_sayim_olarak_da_kabul_edilir() {
        // [2, 0, 2] ile [1, 0, 1] ayni dagilimdir; kayip ayni olmali.
        let cikis = vec![0.3, -0.2, 0.9];
        let a = taban_kayip(&cikis, &[2.0, 0.0, 2.0])
            .unwrap_or_else(|hata| panic!("sayim hedefi kabul edilmedi: {hata:?}"));
        let b = taban_kayip(&cikis, &[1.0, 0.0, 1.0])
            .unwrap_or_else(|hata| panic!("dagilim hedefi kabul edilmedi: {hata:?}"));
        assert!(
            (a - b).abs() < 1e-15,
            "ayni dagilimin kaybi farkli: {a} vs {b}"
        );
    }

    #[test]
    fn olcum_raporu() {
        // Olcum kaydinin kaynagi bu satirdir; betik (training/kademe.py)
        // kosup okur, sayilari kendisi uretmez. Derinlik 6, ara kademeler
        // [2, 4], agirliklar [0.25, 0.25, 0.5].
        let s = spec(&[2, 4], &[0.25, 0.25, 0.5]);
        let hedef = ornek_hedef(16, 11);
        let cikislar = ornek_cikislar(3, 16, 7);
        let sonuc = kademeli_kayip(&s, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("kademeli kosmadi: {hata:?}"));
        let taban_spec = spec(&[2, 4], &[0.0, 0.0, 1.0]);
        let sadece_son = kademeli_kayip(&taban_spec, &cikislar, &hedef)
            .unwrap_or_else(|hata| panic!("taban spec kosmadi: {hata:?}"))
            .toplam;
        let taban = taban_kayip(&cikislar[2], &hedef)
            .unwrap_or_else(|hata| panic!("taban kosmadi: {hata:?}"));
        let taban_fark = sadece_son - taban;
        // Gradyan sapmasi: en kotu gorel/mutlak sapma (sonlu fark).
        let adim = 1e-5;
        let mut gradyan_sapma = 0.0f64;
        for sira in 0..cikislar.len() {
            for boyut in 0..cikislar[sira].len() {
                let mut arti = cikislar.clone();
                let mut eksi = cikislar.clone();
                arti[sira][boyut] += adim;
                eksi[sira][boyut] -= adim;
                let l_arti = kademeli_kayip(&s, &arti, &hedef)
                    .unwrap_or_else(|hata| panic!("ileri arti kosmadi: {hata:?}"))
                    .toplam;
                let l_eksi = kademeli_kayip(&s, &eksi, &hedef)
                    .unwrap_or_else(|hata| panic!("ileri eksi kosmadi: {hata:?}"))
                    .toplam;
                let sayisal = (l_arti - l_eksi) / (2.0 * adim);
                let analitik = sonuc.gradyanlar[sira][boyut];
                let fark = (sayisal - analitik).abs();
                // Gorel sipmis payda tabani - ustteki testle ayni gerekce:
                // cozunurluk sinirinin altindaki gradyani gurultu olarak okumak.
                let olcek = analitik.abs().max(1e-4);
                gradyan_sapma = gradyan_sapma.max(fark / olcek);
            }
        }
        let agirlik_toplam: f64 = s.agirliklar.iter().sum();
        println!(
            "kademe | derinlik={} kademe={} parametre={} agirlik_toplam={} taban_fark={} gradyan_sapma={} kayip={}",
            s.ust_derinlik,
            s.kademeler().len(),
            s.parametre_sayisi(),
            agirlik_toplam,
            taban_fark,
            gradyan_sapma,
            sonuc.toplam
        );
        assert!(
            (agirlik_toplam - 1.0).abs() < 1e-12,
            "agirliklar normalize degil: {agirlik_toplam}"
        );
        assert_eq!(
            taban_fark.to_bits(),
            0.0f64.to_bits(),
            "taban farki sifir olmali, {taban_fark} geldi"
        );
        assert!(
            gradyan_sapma < 1e-6,
            "gradyan sapmasi esik ustunde: {gradyan_sapma:.3e}"
        );
    }
}
