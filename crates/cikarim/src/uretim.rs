//! Üretim döngüsü: bir bağlamdan jeton dizisi üretir.
//!
//! Bu modül bir *kullanıcı yüzeyi* değildir: metni çözmez, biçimlendirmez,
//! ekrana yazmaz. Jeton kimlikleri üretir ve neden durduğunu söyler. Bağlamı
//! metne çeviren, üretileni çözen ve çıktıyı şema doğrulamasından geçiren
//! katman çağırandır ([`lubot`](https://github.com/) CLI'sindeki `sohbet`).
//!
//! # Neden ayrı bir modül, [`crate::puanla`] varken
//!
//! Puanlama "bu metin bu bağlamdan sonra ne kadar beklenirdi" sorusunu yanıtlar
//! ve her çağrıda önbelleği baştan kurar. Üretim ise bağlamı **bir kez** okur,
//! sonra her adımda tek jeton ilerler: aynı önbellek, artımlı ilerleme. İkisini
//! aynı fonksiyona sıkıştırmak, puanlamanın "sızıntısız" sözleşmesini
//! (jetonu, onu içeren durumdan puanlama) üretimin ihtiyacı olan "durumu bir
//! sonraki jeton için kullan" ile karıştırırdı.
//!
//! # Sızıntı kuralı burada da geçerli
//!
//! Bir jeton, ancak kendisinden önceki durumdan örneklendiğinde üretimdir.
//! Döngü önce dağılımı kurar, jetonu çeker, **sonra** çekilen jetonu ileri
//! geçirir; ters sıra (jetonu önce ileri geçirip sonra ondan örneklemek)
//! modelin kendi yazdığını okuması olurdu ve üretilen metin bağlama değil,
//! dağılımın kendi kuyruğuna bağlanırdı.

use crate::ornekleyici::{self, Ayarlar, OrnekHatasi, Rastgele};
use crate::{Cikarim, CikarimHatasi};

/// Üretimin ayarı.
#[derive(Debug, Clone)]
pub struct UretimAyari {
    /// Örnekleme ayarı (sıcaklık, top-k, nucleus).
    pub ayar: Ayarlar,
    /// Örneklayicinin tohumu: aynı tohum, aynı metin.
    pub tohum: u64,
    /// Üretilecek en çok jeton.
    pub en_cok_jeton: usize,
    /// Bu jetonlardan biri çekilirse döngü durur (ör. `</s>`).
    pub dur_jetonlari: Vec<u32>,
    /// Bağlam penceresinin üstüne çıkmamak için: pencere dolarsa en eski
    /// jetonlar düşülür (kayan pencere).
    pub pencere_kaydir: bool,
}

impl Default for UretimAyari {
    fn default() -> Self {
        Self { ayar: Ayarlar::default(), tohum: 0, en_cok_jeton: 64,
               dur_jetonlari: Vec::new(), pencere_kaydir: true }
    }
}

/// Üretimin neden durduğu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durma {
    /// İstenen jeton sayısı doldu.
    Uzunluk,
    /// Durma jetonu çekildi.
    DurJetonu,
}

/// Bir üretim koşusunun ölçümü.
#[derive(Debug, Clone, PartialEq)]
pub struct UretimRaporu {
    /// Üretilen jeton sayısı.
    pub jeton: usize,
    /// Neden durdu.
    pub durma: Durma,
    /// Çekilen jetonların ortalama log-olasılığı (üretimin kendi güveni).
    pub ortalama_log_olasilik: f64,
    /// Kayan pencere kaç kez kaydırıldı.
    pub kaydirma: usize,
}

/// Üretim döngüsü: yüklenmiş bir kontrol noktası + ayar + akış.
pub struct Uretic {
    cikarim: Cikarim,
    ayar: UretimAyari,
    rastgele: Rastgele,
}

impl Uretic {
    /// Yüklenmiş çıkarımı ve ayarı alır; akışı tohumdan kurar.
    #[must_use]
    pub fn yeni(cikarim: Cikarim, ayar: UretimAyari) -> Self {
        let rastgele = Rastgele::tohumdan(ayar.tohum);
        Self { cikarim, ayar, rastgele }
    }

    /// Çıkarımın kendi görünümü.
    #[must_use]
    pub fn cikarim(&self) -> &Cikarim {
        &self.cikarim
    }

    /// Bağlamdan devam ederek jeton üretir.
    ///
    /// Bağlam boş olamaz: boş bir bağlamdan üretim, modelin sözlük dağılımından
    /// metin uydurması olurdu; bu proje "uydurma"yı bir cevap saymaz.
    ///
    /// # Errors
    /// [`CikarimHatasi::BosGirdi`], [`CikarimHatasi::KimlikAraligi`],
    /// [`CikarimHatasi::PencereAsimi`] ve [`OrnekHatasi`] zarfları.
    pub fn uret(&mut self, baglam: &[u32]) -> Result<(Vec<u32>, UretimRaporu), UretimHatasi> {
        if baglam.is_empty() {
            return Err(UretimHatasi::BosBaglam);
        }
        self.ayar.ayar.dogrula().map_err(UretimHatasi::Ornek)?;
        if self.ayar.en_cok_jeton == 0 {
            return Err(UretimHatasi::SifirUzunluk);
        }
        let pencere = self.cikarim.spec().max_seq_len;
        // Baglam pencereye sigmiyorsa kayan pencere acikken kirpilir: en eski
        // jetonlar duser, son jetonlar kalir (baglamin sonu soruyu tasir).
        let mut gecmis: Vec<u32> = if baglam.len() > pencere {
            if !self.ayar.pencere_kaydir {
                return Err(UretimHatasi::Cikarim(CikarimHatasi::PencereAsimi {
                    istenen: baglam.len(),
                    tavan: pencere,
                }));
            }
            baglam[baglam.len() - pencere..].to_vec()
        } else {
            baglam.to_vec()
        };
        let mut uretilen = Vec::with_capacity(self.ayar.en_cok_jeton);
        let mut toplam_log = 0.0f64;
        let mut kaydirma = 0usize;
        let mut durma = Durma::Uzunluk;
        while uretilen.len() < self.ayar.en_cok_jeton {
            // Onbellek her adimda bastan kurulur: `ileri_konum` gizli durumu
            // disariya verir, ama onbellek iceride yasar. Kayan pencere
            // gerektiginde zaten bastan kurmak zorundayiz; sadelik burada
            // olcumden daha degerli (pencere 256, adim basina maliyet dogrusal).
            let mut onbellek = crate::Onbellek::yeni(self.cikarim.spec());
            let mut son_gizli = None;
            for jeton in &gecmis {
                son_gizli = Some(self.cikarim.ileri_konum(*jeton, &mut onbellek).map_err(UretimHatasi::Cikarim)?);
            }
            let gizli = son_gizli.ok_or(UretimHatasi::BosBaglam)?;
            let logitler = self.cikarim.logitler(&gizli);
            let dagilim = ornekleyici::dagilim(&logitler, &self.ayar.ayar).map_err(UretimHatasi::Ornek)?;
            let jeton = ornekleyici::ornekle(&dagilim, &mut self.rastgele).map_err(UretimHatasi::Ornek)?;
            let olasilik = dagilim
                .iter()
                .find(|(j, _)| *j == jeton)
                .map_or(f64::NEG_INFINITY, |(_, p)| p.ln());
            toplam_log += olasilik;
            uretilen.push(jeton);
            gecmis.push(jeton);
            if self.ayar.dur_jetonlari.contains(&jeton) {
                durma = Durma::DurJetonu;
                break;
            }
            if gecmis.len() >= pencere {
                if !self.ayar.pencere_kaydir {
                    break;
                }
                let fazla = gecmis.len() - pencere + 1;
                gecmis.drain(0..fazla);
                kaydirma += fazla;
            }
        }
        let rapor = UretimRaporu {
            jeton: uretilen.len(),
            durma,
            ortalama_log_olasilik: if uretilen.is_empty() { f64::NAN } else { toplam_log / uretilen.len() as f64 },
            kaydirma,
        };
        Ok((uretilen, rapor))
    }
}

/// Üretimin ret sebepleri.
#[derive(Debug, Clone, PartialEq)]
pub enum UretimHatasi {
    /// Bağlam boş: buradan üretim, uydurmadır.
    BosBaglam,
    /// `en_cok_jeton = 0`: üretim istenmemiş.
    SifirUzunluk,
    /// Çıkarım katmanının reddi.
    Cikarim(CikarimHatasi),
    /// Örnekleyicinin reddi.
    Ornek(OrnekHatasi),
}

impl std::fmt::Display for UretimHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BosBaglam => write!(f, "baglam bos: bos baglamdan uretim uydurmadir"),
            Self::SifirUzunluk => write!(f, "en_cok_jeton sifir: uretim istenmemis"),
            Self::Cikarim(e) => write!(f, "cikarim: {e}"),
            Self::Ornek(e) => write!(f, "ornekleyici: {e}"),
        }
    }
}

impl std::error::Error for UretimHatasi {}

#[cfg(test)]
mod testler {
    use super::*;

    /// Küçük, hızlı bir kontrol noktası: eğitim çekirdeğinin kendi ürettiği
    /// ilk-parametreler. Ölçülen şey dil değil, döngünün sözleşmesi - ve
    /// sözleşme dil bilgisine bağlı değil.
    fn oyuncak() -> Cikarim {
        let spec = crate::Spec::lubot_a1();
        let p = lubot_egitim::Parametreler::mup_init(spec, 7, lubot_egitim::INIT_STD_EMBEDDING);
        let opt = lubot_egitim::Adamw::yeni(p.toplam_ogeler(), 0.01, 0.1).expect("optimizer");
        let (adim, m, v) = opt.durum();
        Cikarim::kontrollden(lubot_egitim::kontrol::Kontrol {
            spec,
            parametreler: p,
            adim: 1,
            epoch: 1,
            tohum: 7,
            sozluk_aile: "test-aile".to_string(),
            korpus_ozeti: "c".repeat(64),
            egitim_kaybi: 9.0,
            dogrulama_kaybi: Some(9.0),
            en_iyi_dogrulama: Some(9.0),
            devam_konum: 0,
            hassasiyet: lubot_egitim::kontrol::Hassasiyet::F64,
            optimizer: Some(lubot_egitim::kontrol::OptimizerDurumu {
                adim, ogrenme_orani: 0.01, agirlik_sonumu: 0.1, m: m.to_vec(), v: v.to_vec(),
            }),
        })
    }

    #[test]
    fn bos_baglam_reddedilir() {
        let mut u = Uretic::yeni(oyuncak(), UretimAyari::default());
        assert!(matches!(u.uret(&[]), Err(UretimHatasi::BosBaglam)));
    }

    #[test]
    fn ayni_tohum_ayni_jetonlari_verir_ve_uzunluk_tutar() {
        let baglam = [3u32, 17, 42, 8];
        let ayar = UretimAyari { tohum: 20260924, en_cok_jeton: 12, ..UretimAyari::default() };
        let (a, rapor_a) = Uretic::yeni(oyuncak(), ayar.clone()).uret(&baglam).expect("uret");
        let (b, rapor_b) = Uretic::yeni(oyuncak(), ayar).uret(&baglam).expect("uret");
        assert_eq!(a, b, "ayni tohum ayni diziyi vermeliydi");
        assert_eq!(rapor_a, rapor_b);
        assert_eq!(a.len(), 12);
        assert!(rapor_a.ortalama_log_olasilik.is_finite());
    }

    #[test]
    fn dur_jetonu_donguyu_keser() {
        let baglam = [1u32, 2, 3];
        // Butun jetonlar durma jetoni: ilk adimda kesilmeli.
        let hepsi: Vec<u32> = (0..8192).collect();
        let ayar = UretimAyari { tohum: 5, en_cok_jeton: 50, dur_jetonlari: hepsi,
                                 ..UretimAyari::default() };
        let (uretilen, rapor) = Uretic::yeni(oyuncak(), ayar).uret(&baglam).expect("uret");
        assert_eq!(uretilen.len(), 1);
        assert_eq!(rapor.durma, Durma::DurJetonu);
    }

    #[test]
    fn sicaklik_sifir_acgozlu_ve_tekrarlanabilir() {
        let baglam = [11u32, 12, 13];
        let ayar = UretimAyari { ayar: Ayarlar { sicaklik: 0.0, ..Ayarlar::default() },
                                 tohum: 1, en_cok_jeton: 6, ..UretimAyari::default() };
        let (a, _) = Uretic::yeni(oyuncak(), ayar.clone()).uret(&baglam).expect("uret");
        let (b, _) = Uretic::yeni(oyuncak(), ayar).uret(&baglam).expect("uret");
        assert_eq!(a, b);
    }

    #[test]
    fn pencere_asimi_ve_kayan_pencere() {
        let spec = crate::Spec::lubot_a1();
        let uzun: Vec<u32> = (0..spec.max_seq_len + 20).map(|i| (i % 100) as u32).collect();
        let mut u = Uretic::yeni(oyuncak(), UretimAyari {
            pencere_kaydir: false, tohum: 2, en_cok_jeton: 4, ..UretimAyari::default() });
        assert!(matches!(u.uret(&uzun), Err(UretimHatasi::Cikarim(_))), "tasmasina izin verildi");
        let mut u2 = Uretic::yeni(oyuncak(), UretimAyari {
            pencere_kaydir: true, tohum: 2, en_cok_jeton: 4, ..UretimAyari::default() });
        let (uretilen, _) = u2.uret(&uzun).expect("kayan pencere uretmeliydi");
        assert_eq!(uretilen.len(), 4);
    }
}
