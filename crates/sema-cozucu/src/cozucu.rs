//! Şema-maskeli decode: dağılımdan yalnız şemanın kabul ettiği jetonları seçmek.
//!
//! Buradaki tek kural: bir jeton, otomatın şu anki durumundan geçmiyorsa
//! **seçilemez**. Seçilecek hiçbir jeton kalmadıysa decode reddeder
//! ([`CozumHatasi::Cikmaz`]); maske gevşetilmez, en yakın geçerli jetona
//! "düşürülmez", çıktı sonradan onarılmaz. Lubot'un "şemayı geçemeyen çıktı
//! reddedilir, asla yumuşatılmaz" kuralının decode tarafı budur.
//!
//! ## Seçim kuralı
//!
//! Maskelenmiş küme içinde en yüksek logit kazanır; eşitlikte küçük indeks
//! kazanır. İki eşit sayı arasına bir sıra uydurmak, olmayan bir farkı
//! uydurmaktır — bu, `lubot-cikarim`'in aday sıralamasındaki kuralın aynıdır.
//! `NaN` hiçbir zaman kazanamaz: bir dağılımın bozuk olduğu yerde decode'un
//! sessizce bir jeton seçmesi, bozukluğu gizlemenin en ucuz yoludur.
//!
//! ## Ne değildir
//!
//! Sıcaklık, top-k, çekirdek örnekleme ve tekrar cezası burada yok: onlar
//! dağıtımın işi (`lubot-cikarim::ornekleyici`) ve maskeden **önce**
//! uygulanırlar. Bu modül dağıtımın şekliyle ilgilenmez; hangi jetonların hiç
//! seçilemeyeceğiyle ilgilenir. Durma kuralı da burada değil: [`coz`] durmayı
//! çağıranın verdiği bir yüklemle yapar, çünkü "belge bitti" kararı şemanın
//! değil görevin kararıdır.

use crate::otomat::Yuruyus;
use crate::SemaHatasi;
use std::cmp::Ordering;

/// Bir decode adımının sonucu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adim {
    /// Seçilen jetonun sözlükteki indeksi.
    pub jeton: usize,
    /// Bu adımda maskelenen jeton sayısı.
    pub maskelenen: usize,
}

/// Decode'un neden durduğu.
///
/// Üçü şemanın reddi, biri çağıranın hatası: hiçbiri "yaklaşık geçerli bir
/// çıktı" üretmez.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CozumHatasi {
    /// Maskelenecek jeton kalmadı: hiçbir jeton şu anki durumdan geçmiyor.
    Cikmaz {
        /// Kaçıncı adımda.
        adim: usize,
        /// Hangi satırda.
        satir: usize,
    },
    /// Decode bitti ama belge şemayı geçmiyor.
    KabulEdilmez {
        /// Şemanın söylediği.
        sebep: SemaHatasi,
    },
    /// Dağılım, izinli hiçbir jeton için sonlu bir logit taşımıyordu.
    ///
    /// `NaN` bir jetonun puanı değil, bozuk bir dağılımın işaretidir; bozuk bir
    /// dağılımdan sessizce bir jeton seçmek bozukluğu gizler.
    DagilimBozuk {
        /// Maskeden geçen jeton sayısı.
        izinli: usize,
    },
    /// Logit vektörü sözlükle aynı uzunlukta değil.
    LogitUyusmaz {
        /// Sözlükteki jeton sayısı.
        beklenen: usize,
        /// Verilen logit sayısı.
        verilen: usize,
    },
    /// Boş sözlükle decode istendi.
    BosSozluk,
    /// `izinli` geçti ama `ilerle` reddetti: iki yol ayrıldı, bu bir iç hatadır.
    IcTutarsizlik {
        /// Otomatın reddi.
        hata: SemaHatasi,
    },
}

impl std::fmt::Display for CozumHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cikmaz { adim, satir } => write!(
                f,
                "step {adim}, line {satir}: no token in the vocabulary is allowed here"
            ),
            Self::KabulEdilmez { sebep } => write!(f, "the document is refused: {sebep}"),
            Self::DagilimBozuk { izinli } => write!(
                f,
                "the distribution held no finite logit for any of the {izinli} allowed tokens"
            ),
            Self::LogitUyusmaz { beklenen, verilen } => write!(
                f,
                "the vocabulary holds {beklenen} tokens but {verilen} logits were given"
            ),
            Self::BosSozluk => write!(f, "an empty vocabulary cannot decode anything"),
            Self::IcTutarsizlik { hata } => write!(
                f,
                "the mask allowed a token the automaton then refused: {hata}"
            ),
        }
    }
}

impl std::error::Error for CozumHatasi {}

/// Bir koşunun sayıları.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CozumOlcum {
    /// Kaç adım koştu.
    pub adim: usize,
    /// Toplam maskelenen jeton sayısı (adım başına maskelenenlerin toplamı).
    pub maskelenen: usize,
    /// Üretilen bayt sayısı.
    pub bayt: usize,
    /// Adım adım seçilen jeton ve o adımda maskelenen jeton sayısı.
    ///
    /// Bu iz olmadan "maske hiç çalışmadı" ile "maske çok çalıştı" ayırt
    /// edilemez; `maskelenen` toplamı tek başına hangi adımlarda maskenin
    /// işlediğini söylemez.
    pub adimlar: Vec<Adim>,
    /// Belge şemayı geçti mi.
    pub kabul: bool,
}

/// Tek bir maskeli adım.
///
/// # Errors
///
/// [`CozumHatasi::BosSozluk`] boş bir sözlükte, [`CozumHatasi::LogitUyusmaz`]
/// logit vektörü sözlükle aynı uzunlukta olmadığında,
/// [`CozumHatasi::Cikmaz`] hiçbir jeton şemadan geçmediğinde.
pub fn adim(y: &mut Yuruyus, logitler: &[f64], sozluk: &[&[u8]]) -> Result<Adim, CozumHatasi> {
    if sozluk.is_empty() {
        return Err(CozumHatasi::BosSozluk);
    }
    if logitler.len() != sozluk.len() {
        return Err(CozumHatasi::LogitUyusmaz {
            beklenen: sozluk.len(),
            verilen: logitler.len(),
        });
    }
    let mut secim: Option<(usize, f64)> = None;
    let mut maskelenen = 0usize;
    for (i, jeton) in sozluk.iter().enumerate() {
        if !y.izinli(jeton) {
            maskelenen += 1;
            continue;
        }
        let puan = logitler[i];
        if puan.is_nan() {
            // Bu atlamadan vazgeçilemez: `NaN` bir kere başa geçerse kendinden
            // sonraki hiçbir sonlu değer onu geçemez (karşılaştırma kurulamaz)
            // ve seçilen jetonun logiti `NaN` olur. Yani bu satır yalnız
            // okunabilirlik değil, seçimin kendisi.
            continue;
        }
        let daha_iyi = match secim {
            // Eşitlikte mevcut (küçük indeksli) seçim kalır; kıyas kurulamazsa
            // (yalnız `NaN` ile olur, yukarıda elendi) bu jeton geçemez.
            Some((_, en_iyi)) => puan.partial_cmp(&en_iyi) == Some(Ordering::Greater),
            None => true,
        };
        if !daha_iyi {
            continue;
        }
        secim = Some((i, puan));
    }
    let (secilen, _) = match secim {
        Some(secim) => secim,
        None if maskelenen == sozluk.len() => {
            return Err(CozumHatasi::Cikmaz {
                adim: 0,
                satir: y.satir(),
            })
        }
        None => {
            return Err(CozumHatasi::DagilimBozuk {
                izinli: sozluk.len() - maskelenen,
            })
        }
    };
    y.ilerle(sozluk[secilen])
        .map_err(|hata| CozumHatasi::IcTutarsizlik { hata })?;
    Ok(Adim {
        jeton: secilen,
        maskelenen,
    })
}

/// Maskeli decode döngüsü.
///
/// `logitler` her adımda şu anki ön eke bakarak bir dağıtım verir; `dur`
/// belgenin bittiğini söyler. Döngü adım tavanıyla da sınırlıdır: tavan dolar
/// ve belge hâlâ kabul edilmiyorsa sonuç bir hata olarak döner, yarım belge
/// "başarılı" sayılmaz.
///
/// # Errors
///
/// [`adim`]'ın bütün reddleri, ayrıca [`CozumHatasi::AdimBitti`] ve
/// [`CozumHatasi::KabulEdilmez`].
pub fn coz<F, D>(
    y: &mut Yuruyus,
    mut logitler: F,
    sozluk: &[&[u8]],
    en_fazla_adim: usize,
    mut dur: D,
) -> Result<CozumOlcum, CozumHatasi>
where
    F: FnMut(&Yuruyus) -> Vec<f64>,
    D: FnMut(&Yuruyus) -> bool,
{
    let mut olcum = CozumOlcum::default();
    if sozluk.is_empty() {
        return Err(CozumHatasi::BosSozluk);
    }
    while olcum.adim < en_fazla_adim {
        if dur(y) {
            break;
        }
        let dagilim = logitler(y);
        let a = adim(y, &dagilim, sozluk).map_err(|h| h.adimla(olcum.adim))?;
        olcum.adim += 1;
        olcum.maskelenen += a.maskelenen;
        olcum.adimlar.push(a);
    }
    olcum.bayt = y.bayt_sayisi();
    // Kabulün tek kaynağı `kapat`: aynı soruyu iki yerden sormak, iki ayrı
    // cevabın mümkün olduğu bir yer açar. Red nedeni de böylece kaybolmaz.
    match y.kapat() {
        Ok(()) => {
            olcum.kabul = true;
            Ok(olcum)
        }
        Err(sebep) => Err(CozumHatasi::KabulEdilmez { sebep }),
    }
}

impl CozumHatasi {
    /// Reddi, geldiği adımın numarasıyla taşı.
    ///
    /// `adim` kendi başına adım numarasını bilmez (tek bir adımın kaçıncı adım
    /// olduğu döngünün işi); çıkmaz raporunda adım numarası olmadan iki ayrı
    /// red birbirinden ayırt edilemez.
    #[must_use]
    fn adimla(self, adim: usize) -> Self {
        match self {
            Self::Cikmaz { satir, .. } => Self::Cikmaz { adim, satir },
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sema;
    use lubot_read::output_schema::validate_markdown_output;

    /// Sentetik sözlük: 256 tek bayt + çok baytlı jetonlar.
    ///
    /// Gerçek bir kontrol noktası yok, o yüzden logitler de sentetik. Bu bir
    /// kalite ölçümü değil: ölçülen şey maskenin ne yaptığı.
    fn sozluk() -> Vec<&'static [u8]> {
        let mut s: Vec<&'static [u8]> = Vec::with_capacity(263);
        for b in 0..=255u16 {
            #[allow(clippy::cast_possible_truncation)]
            let bayt = b as u8;
            s.push(&TEK_BAYT[bayt as usize]);
        }
        s.extend_from_slice(&[
            b"## ", b"\n\n", b"|---|", b"```", b"ilik", b"olcu", b"| ", b"---",
        ]);
        s
    }

    /// 0..=255 için tek baytlık dilimler.
    static TEK_BAYT: [[u8; 1]; 256] = {
        let mut tablo = [[0u8; 1]; 256];
        let mut i = 0;
        while i < 256 {
            #[allow(clippy::cast_possible_truncation)]
            {
                tablo[i] = [i as u8];
            }
            i += 1;
        }
        tablo
    };

    /// Deterministik logit kaynağı: aynı tohum aynı belgeyi verir.
    fn logit_kaynagi(tohum: u64) -> impl FnMut(&Yuruyus) -> Vec<f64> {
        let mut durum = tohum
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let boyut = sozluk().len();
        move |_y: &Yuruyus| {
            (0..boyut)
                .map(|_| {
                    durum = durum
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((durum >> 33) as f64 / (1u64 << 31) as f64) - 0.5
                })
                .collect()
        }
    }

    /// Belge bitti mi: kabul edilebilir bir ön ek ve satır sonunda.
    fn dur(y: &Yuruyus) -> bool {
        y.kabul() && y.belge().ends_with(b"\n")
    }

    #[test]
    fn maske_secimi_bozmaz() {
        // Bütün jetonların kabul edildiği bir durumda maskeli seçim düz
        // argmax'tır: maske bir tercih değil, bir kısıttır.
        let s = sozluk();
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        let logitler: Vec<f64> = (0..s.len()).map(|i| (i as f64) * 0.001).collect();
        let a = adim(&mut y, &logitler, &s).expect("ilk adımda her şey geçer");
        let duz = logitler
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("sonlu"))
            .map(|(i, _)| i)
            .expect("boş değil");
        assert_eq!(a.jeton, duz);
    }

    #[test]
    fn esitlikte_kucuk_indeks_kazanir_ve_nan_kazanamaz() {
        let s = sozluk();
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        let mut logitler = vec![0.5f64; s.len()];
        logitler[7] = f64::NAN;
        let a = adim(&mut y, &logitler, &s).expect("adım");
        assert_eq!(a.jeton, 0, "eşitlikte küçük indeks kazanır");
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        let tek_nan = {
            let mut l = vec![f64::NAN; s.len()];
            l[3] = 0.1;
            l
        };
        let a = adim(&mut y, &tek_nan, &s).expect("adım");
        assert_eq!(a.jeton, 3, "NaN hiçbir zaman kazanamaz");
    }

    #[test]
    fn bos_sozluk_ve_uyusmayan_logit_reddedilir() {
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        assert_eq!(adim(&mut y, &[], &[]), Err(CozumHatasi::BosSozluk));
        let s = sozluk();
        assert_eq!(
            adim(&mut y, &[0.0, 1.0], &s),
            Err(CozumHatasi::LogitUyusmaz {
                beklenen: s.len(),
                verilen: 2
            })
        );
    }

    #[test]
    fn olcum_raporu_sema_cozucu() {
        // Karta özel ölçüm satırı. Sayılar bu testin kendisinden basılır;
        // `training/sema_cozucu.py` satırı koşar ve okur, kendisi sayı üretmez.
        let s = sozluk();
        let mut belge_sayisi = 0usize;
        let mut toplam_adim = 0usize;
        let mut toplam_maskelenen = 0usize;
        let mut toplam_bayt = 0usize;
        let mut dogrulayici_kacis = 0usize;
        let mut naif_gecersiz = 0usize;
        let mut naif_belge = 0usize;

        for tohum in 1u64..=8 {
            let mut y = Yuruyus::yeni(Sema::varsayilan());
            let kaynak = logit_kaynagi(tohum);
            let o = coz(&mut y, kaynak, &s, 400, dur).expect("koşu kabul edilmedi");
            belge_sayisi += 1;
            toplam_adim += o.adim;
            toplam_maskelenen += o.maskelenen;
            toplam_bayt += o.bayt;
            assert!(o.kabul);
            if validate_markdown_output(y.belge()).is_err() {
                dogrulayici_kacis += 1;
            }

            // Aynı dağıtım, maske olmadan: maskenin iş yaptığı burada görünür.
            // Bu bir kalite karşılaştırması değil, maskenin varlık sebebidir.
            // Naif yolun bir durumu **yoktur**; dağıtım kaynağının imzası bir
            // yürüyüş istediği için burada boş bir yürüyüş duruyor (kaynak
            // argümanı yok sayıyor). Fark tam olarak bu: aynı logit dizisi,
            // otomat olmadan.
            let naif = Yuruyus::yeni(Sema::varsayilan());
            let mut naif_kaynak = logit_kaynagi(tohum);
            let mut naif_cikti: Vec<u8> = Vec::new();
            for _ in 0..o.adim {
                let dagilim = naif_kaynak(&naif);
                let (secilen, _) = dagilim
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("boş değil");
                // Naif yol otomatı hiç tutmaz: yalnız bayt biriktirir. Fark bu.
                naif_cikti.extend_from_slice(s[secilen]);
            }
            naif_belge += 1;
            if validate_markdown_output(&naif_cikti).is_err() {
                naif_gecersiz += 1;
            }
        }

        // Rastgele yürüyüşler: otomatın kabul ettiği her belge doğrulayıcıdan
        // geçmek zorunda. Kaçış sayısı sıfır değilse daraltma yalan söylüyor.
        let mut rastgele_kacis = 0usize;
        let mut rastgele_belge = 0usize;
        for tohum in 11u64..=210 {
            let mut kaynak = logit_kaynagi(tohum);
            let mut y = Yuruyus::yeni(Sema::varsayilan());
            for _ in 0..120 {
                if y.kabul() && y.bayt_sayisi() > 8 {
                    break;
                }
                let dagilim = kaynak(&y);
                if adim(&mut y, &dagilim, &s).is_err() {
                    break;
                }
            }
            if y.kabul() {
                rastgele_belge += 1;
                if validate_markdown_output(y.belge()).is_err() {
                    rastgele_kacis += 1;
                }
            }
        }

        // Red yolları ölçülür: bir red yolu hiç koşulmadıysa o kuralın çalıştığı
        // bilinmez. Üçü de burada gerçekten koşuyor.
        let mut cikmaz_red = 0usize;
        let mut y = Yuruyus::yeni(Sema { en_fazla_bayt: 3 });
        y.ilerle(b"abc").expect("tavana kadar");
        if adim(&mut y, &vec![0.0; s.len()], &s).is_err() {
            cikmaz_red += 1;
        }

        let mut adim_bitti_red = 0usize;
        let mut y = kos("|").expect("tek satır");
        y.ilerle(b"\n").expect("satır sonu");
        let kaynak = |_y: &Yuruyus| {
            // Sıfır sütunlu bir başlıktan sonra gelen her satır tabloyu bozar:
            // bu dağıtım bilerek oraya zorluyor.
            let mut l = vec![-1.0f64; s.len()];
            l[b'|' as usize] = 1.0;
            l[b'-' as usize] = 0.5;
            l
        };
        // Dur yüklemi bilerek `false`: bu ön ek hiçbir adımda kabul edilebilir
        // olmuyor, yani koşu ancak tavanla biter ve red orada görünür.
        if coz(&mut y, kaynak, &s, 6, |_| false).is_err() {
            adim_bitti_red += 1;
        }

        let mut utf8_red = 0usize;
        for dizi in [
            &b"\xc0\xaf"[..],
            &b"\xed\xa0\x80"[..],
            &b"\xff"[..],
            &b"\xf4\x90\x80\x80"[..],
        ] {
            let y = Yuruyus::yeni(Sema::varsayilan());
            if !y.izinli(dizi) {
                utf8_red += 1;
            }
        }

        // Olçüt kayıttan önce yazıldı ve burada da duruyor: kaçış varsa CI
        // kâğıt üzerinde değil, test düzeyinde kırmızı olur.
        assert_eq!(
            dogrulayici_kacis, 0,
            "otomat doğrulayıcıdan geniş çıktı: kaçış var"
        );
        assert_eq!(rastgele_kacis, 0, "rastgele yürüyüşte kaçış var");
        assert!(
            naif_gecersiz > 0,
            "maskesiz taban hiç geçersiz belge üretmedi: maske ölçülmüyor"
        );
        assert!(
            cikmaz_red >= 1 && adim_bitti_red >= 1 && utf8_red == 4,
            "bir red yolu hiç koşmadı: kuralın çalıştığı bilinmez"
        );

        println!(
            "sema-cozucu | sozluk={} belge={belge_sayisi} adim={toplam_adim} maskelenen={toplam_maskelenen} bayt={toplam_bayt} rastgele_belge={rastgele_belge} dogrulayici_kacis={dogrulayici_kacis} rastgele_kacis={rastgele_kacis} naif_belge={naif_belge} naif_gecersiz={naif_gecersiz} cikmaz_red={cikmaz_red} adim_bitti_red={adim_bitti_red} utf8_red={utf8_red}",
            s.len(),
        );
    }

    fn kos(metin: &str) -> Result<Yuruyus, SemaHatasi> {
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        for b in metin.as_bytes() {
            y.ilerle(&[*b])?;
        }
        Ok(y)
    }

    #[test]
    fn cikmaz_adim_numarasi_tasir() {
        let mut y = Yuruyus::yeni(Sema { en_fazla_bayt: 2 });
        y.ilerle(b"ab").expect("tavana kadar");
        let h = adim(&mut y, &vec![0.0; sozluk().len()], &sozluk()).expect_err("çıkmaz");
        assert_eq!(h.adimla(17), CozumHatasi::Cikmaz { adim: 17, satir: 1 });
        assert!(!h.to_string().is_empty());
        // Diğer reddler adım numarası taşımaz, oldukları gibi kalır.
        assert_eq!(CozumHatasi::BosSozluk.adimla(4), CozumHatasi::BosSozluk);
        assert!(!CozumHatasi::DagilimBozuk { izinli: 9 }
            .to_string()
            .is_empty());
        assert!(!CozumHatasi::LogitUyusmaz {
            beklenen: 3,
            verilen: 2
        }
        .to_string()
        .is_empty());
        assert!(!CozumHatasi::KabulEdilmez {
            sebep: SemaHatasi::Bos
        }
        .to_string()
        .is_empty());
        assert!(!CozumHatasi::IcTutarsizlik {
            hata: SemaHatasi::Bos
        }
        .to_string()
        .is_empty());
    }

    #[test]
    fn adim_tavani_acik_citle_biten_belgeyi_basari_saymaz() {
        // Açık bir çit, adım tavanı dolunca kendiliğinden kapanmaz: koşu red
        // döner ve nedeni çitin kendisidir. Yarım bir belgeyi "başarılı"
        // saymak, şemayı geçmeyen bir çıktıyı kullanıcıya vermenin en kolay
        // yoludur.
        let s = sozluk();
        let mut y = kos("```\n").expect("açık çit geçerli bir ön ek");
        let kaynak = |_y: &Yuruyus| {
            let mut l = vec![-1.0f64; s.len()];
            l[b'a' as usize] = 1.0;
            l
        };
        let h = coz(&mut y, kaynak, &s, 4, |_| false).expect_err("tavan");
        assert_eq!(
            h,
            CozumHatasi::KabulEdilmez {
                sebep: SemaHatasi::CitDengesiz { satir: 1 }
            }
        );
        // Tavan dolup belge **kabul edilebilir** kaldıysa bu bir red değildir:
        // adım sayısı çağıranın bütçesidir, şemanın kararı değil.
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        let o = coz(&mut y, logit_kaynagi(5), &s, 4, |_| false).expect("belge geçerli");
        assert!(o.kabul);
        assert_eq!(o.adim, 4);
    }

    #[test]
    fn bozuk_dagilim_sessizce_secmez() {
        let s = sozluk();
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        let h = adim(&mut y, &vec![f64::NAN; s.len()], &s).expect_err("bozuk dağılım");
        // Boş bir belgede maske 264 jetonun 186'sını bırakır: 0x80..=0xBF (64),
        // 0xC0..=0xC1 (2), 0xF5..=0xFF (11) ve `\r` (1) = 78 red. Sayı
        // yazılmış bir beklenti değil, UTF-8 kuralının kendisi.
        assert_eq!(h, CozumHatasi::DagilimBozuk { izinli: 186 });
    }
}
