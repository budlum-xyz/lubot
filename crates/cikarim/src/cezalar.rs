//! Örnekleme cezaları: dağılım kurulmadan **önce** logitlere uygulanan düzeltmeler.
//!
//! Bu modül metin yazmaz, jeton seçmez, akış tüketmez. Tek işi bir logit
//! vektörünü, bağlamdan okunan kanıtla düzeltmek ve *ne kadarını düzelttiğini*
//! söylemek. Ölçülen cevap [`CezaRaporu`]'dur: kaç jeton cezalandırıldı, kaç
//! gram yasaklandı, kaç durma jetonu ertelendi. Cezasız bir koşu da bu modülden
//! geçer ve raporu "hiçbir şeye dokunulmadı" der.
//!
//! # Neden ceza, neden burada
//!
//! Eğitilmiş küçük bir model, serbest üretimde aynı jetona ya da aynı kısa
//! diziye takılıp kalır; çıktı "[...] [] <u : a an zone /" gibi tekrarlardan
//! ibaret kalır. Bu bir örnekleme hatasıdır, modelin *bildiği* bir şeyin
//! yanlış okunması değil: dağılımın kuyruğu kendi kendini besler. Düzeltme
//! bu yüzden sıcaklık/top-k/top-p ile aynı yerde durur ve aynı sözleşmeye
//! uyar — her ayar doğrulanır, her etki raporlanır, hiçbir şey sessizce
//! değişmez.
//!
//! # Cezalar sırayla ve bir kez uygulanır
//!
//! Sıra sabittir: tekrar → varlık → sıklık → n-gram yasağı → durma erteleme.
//! Toplama yerine sıralamanın anlamı, aynı jetonun iki cezayı da yemesidir
//! (görülmüş *ve* yasaklanmış olabilir); yasak en sonda gelir ve kazanır,
//! çünkü yasak bir tercih değil bir sınırdır.
//!
//! # Yasak listesi boşaltılamaz
//!
//! Bütün adaylar yasaklanırsa örnekleme yapacak bir şey kalmaz. O durumda
//! dağılımdan bir şey uydurmak yerine [`CezaHatasi::BosAday`] döner: "hiçbir
//! jeton kalmadı" demek, rastgele bir jetonu "seçilmiş" gibi sunmaktan
//! dürüsttür.

/// Örnekleme cezalarının ayarı. Hepsi kapatılabilir; varsayılan **kapalıdır**.
///
/// Kapalı varsayılan bir karardır, tembellik değil: ceza ölçülen bir
/// müdahaledir ve ölçüm, neyin değiştiğini bilmeden yapılamaz. Açık bir ceza
/// çıktıyı değiştirir; bu yüzden her koşunun raporunda hangi cezaların açık
/// olduğu ve kaç jetona dokunduğu yazılıdır.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cezalar {
    /// Görülmüş jetonun logitini böler (pozitifse) ya da çarpar (negatifse).
    /// `1.0` kapalı demektir; `1.0`'ın altı ödüllendirme olurdu ve reddedilir.
    pub tekrar_cezasi: f64,
    /// Görülmüş her jetonun logitinden düşülen sabit. `0.0` kapalı.
    pub varlik_cezasi: f64,
    /// Görülme sayısıyla çarpılarak düşülen sabit. `0.0` kapalı.
    pub siklik_cezasi: f64,
    /// Yasaklanan n-gram uzunluğu. `0` ya da `1` kapalı; `2` "ikili dizi
    /// tekrarlanmasın" demektir.
    pub kac_gram: usize,
    /// Bu jeton sayısına kadar durma jetonları yasaklanır: çıktı tek jetonluk
    /// bir dizi olarak bitmesin diye.
    pub en_az_jeton: usize,
}

impl Default for Cezalar {
    fn default() -> Self {
        Self {
            tekrar_cezasi: 1.0,
            varlik_cezasi: 0.0,
            siklik_cezasi: 0.0,
            kac_gram: 0,
            en_az_jeton: 0,
        }
    }
}

impl Cezalar {
    /// Hiçbir ceza yok: varsayılanın adlandırılmış hâli.
    #[must_use]
    pub fn kapali() -> Self {
        Self::default()
    }

    /// Açık bir ceza var mı. Rapor "ceza yok" ile "ceza var ama dokunmadı"
    /// arasındaki farkı bu ayrımla tutar.
    #[must_use]
    pub fn etkin(&self) -> bool {
        self.tekrar_cezasi > 1.0
            || self.varlik_cezasi > 0.0
            || self.siklik_cezasi > 0.0
            || self.kac_gram >= 2
            || self.en_az_jeton > 0
    }

    /// Ayarın kurulabilir olduğunu söyler.
    ///
    /// # Errors
    /// [`CezaHatasi::GecersizCezalar`] — sonlu olmayan bir katsayı, `1.0`'ın
    /// altında bir tekrar cezası, negatif bir ceza ya da `en_az_jeton`u
    /// anlamsız kılan bir n-gram.
    pub fn dogrula(&self) -> Result<(), CezaHatasi> {
        for (ad, deger) in [
            ("tekrar cezasi", self.tekrar_cezasi),
            ("varlik cezasi", self.varlik_cezasi),
            ("siklik cezasi", self.siklik_cezasi),
        ] {
            if !deger.is_finite() {
                return Err(CezaHatasi::GecersizCezalar(format!(
                    "{ad} sonlu degil: {deger}"
                )));
            }
        }
        if self.tekrar_cezasi < 1.0 {
            return Err(CezaHatasi::GecersizCezalar(format!(
                "tekrar cezasi {} < 1.0: bu ceza degil odul olurdu",
                self.tekrar_cezasi
            )));
        }
        if self.varlik_cezasi < 0.0 || self.siklik_cezasi < 0.0 {
            return Err(CezaHatasi::GecersizCezalar(format!(
                "varlik/siklik cezasi negatif olamaz: {} / {}",
                self.varlik_cezasi, self.siklik_cezasi
            )));
        }
        if self.kac_gram == 1 {
            return Err(CezaHatasi::GecersizCezalar(
                "kac_gram 1: tek jetonluk dizi yasagi tanimsiz (0 kapali demektir, 2 ikili)"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Bir ceza turunun ölçümü.
///
/// Rapor, "ceza uygulandı" iddiasının kanıtıdır: kaç jetonun logiti değişti,
/// kaç jeton yasak listesine girdi, kaç durma jetonu ertelendi.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CezaRaporu {
    /// Logiti bu turda değişen **farklı** jeton sayısı.
    pub cezalanan_jeton: usize,
    /// Yasak listesine giren farklı jeton sayısı (n-gram ve durma erteleme).
    pub yasaklanan_jeton: usize,
    /// Bitmesin diye ertelenen durma jetonu sayısı.
    pub ertelenen_durma: usize,
    /// Bu turda etkin bir ceza var mıydı.
    pub etkin: bool,
}

impl CezaRaporu {
    /// Hiçbir şeye dokunulmamış tur.
    #[must_use]
    pub fn bos() -> Self {
        Self::default()
    }

    /// İki turu (ya da iki jetonu) tek raporda toplar.
    pub fn ekle(&mut self, digeri: &Self) {
        self.cezalanan_jeton += digeri.cezalanan_jeton;
        self.yasaklanan_jeton += digeri.yasaklanan_jeton;
        self.ertelenen_durma += digeri.ertelenen_durma;
        self.etkin = self.etkin || digeri.etkin;
    }

    /// Tek satırlık okunur özet.
    #[must_use]
    pub fn ozet(&self) -> String {
        if !self.etkin {
            return "ceza: kapali".to_string();
        }
        format!(
            "ceza: adim basina toplam {} jeton geriletilmis, {} yasak, {} erteleme",
            self.cezalanan_jeton, self.yasaklanan_jeton, self.ertelenen_durma
        )
    }
}

/// Cezaların ret sebepleri.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CezaHatasi {
    /// Ayar kurulamaz (bkz. [`Cezalar::dogrula`]).
    GecersizCezalar(String),
    /// Bütün adaylar yasaklandı: örneklenecek jeton kalmadı.
    BosAday,
}

impl std::fmt::Display for CezaHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GecersizCezalar(ne) => write!(f, "gecersiz ceza ayari: {ne}"),
            Self::BosAday => write!(
                f,
                "butun adaylar yasaklandi: cezalar dagilimi bosaltti, orneklenecek jeton yok"
            ),
        }
    }
}

impl std::error::Error for CezaHatasi {}

/// Bir ceza turunun çıktısı: düzeltilmiş logitler, yasak listesi, ölçüm.
///
/// Yasak, logiti `-inf` yaparak değil **ayrı bir liste** olarak taşınır. Sebep
/// örnekleyicinin sözleşmesi: sonlu olmayan bir logit "dağılım kurulamaz"
/// demektir ve orada durmak doğrudur. Yasak ise geçerli bir logit üzerine
/// konmuş bir sınırdır; ikisini aynı yere yazmak, "bu jeton yasak" ile "bu
/// logit bozuk" ayrımını silerdi.
#[derive(Debug, Clone, PartialEq)]
pub struct CezaSonucu {
    /// Ölçeklenmiş/geriletIlmiş logitler; hepsi sonlu kalır.
    pub logitler: Vec<f64>,
    /// Bu turda örneklenemeyecek jetonlar (n-gram yasağı + durma erteleme).
    pub yasak: Vec<u32>,
    /// Ne yapıldığının ölçümü.
    pub rapor: CezaRaporu,
}

/// Logitleri bağlamdan okunan kanıtla düzeltir ve ne yaptığını döndürür.
///
/// `gecmis` cezaların okunduğu bağlamdır (istem + üretilen); `uretilen` yalnız
/// uzunluk kuralı için gerekir ve ondan ayrı tutulur, çünkü "kaç jeton
/// üretildi" ile "bağlamda ne var" ayrı sorulardır.
///
/// # Errors
/// [`CezaHatasi::GecersizCezalar`] ve [`CezaHatasi::BosAday`].
pub fn uygula(
    cezalar: &Cezalar,
    logitler: &[f64],
    gecmis: &[u32],
    uretilen: &[u32],
    dur_jetonlari: &[u32],
) -> Result<CezaSonucu, CezaHatasi> {
    cezalar.dogrula()?;
    let mut rapor = CezaRaporu {
        etkin: cezalar.etkin(),
        ..CezaRaporu::default()
    };
    let mut duzeltilmis = logitler.to_vec();
    if !cezalar.etkin() {
        return Ok(CezaSonucu {
            logitler: duzeltilmis,
            yasak: Vec::new(),
            rapor,
        });
    }
    let mut yasak: Vec<u32> = Vec::new();

    // Görülme sayımı: bağlamdaki her jeton bir kez sayılır, sıra korunmaz
    // (sayım sıraya bağlı değil; bağlı olsaydı aynı bağlam iki farklı ceza
    // verirdi ve tohum tekrarlanabilirliği metin uzunluğuna takılırdı).
    let mut sayim: Vec<(u32, u32)> = Vec::new();
    for jeton in gecmis {
        match sayim.iter_mut().find(|(j, _)| j == jeton) {
            Some((_, n)) => *n += 1,
            None => sayim.push((*jeton, 1)),
        }
    }

    let mut dokunulan = vec![false; duzeltilmis.len()];
    for (jeton, adet) in &sayim {
        let Some(logit) = duzeltilmis.get_mut(*jeton as usize) else {
            continue;
        };
        let onceki = *logit;
        if cezalar.tekrar_cezasi > 1.0 {
            *logit = if *logit > 0.0 {
                *logit / cezalar.tekrar_cezasi
            } else {
                *logit * cezalar.tekrar_cezasi
            };
        }
        if cezalar.varlik_cezasi > 0.0 {
            *logit -= cezalar.varlik_cezasi;
        }
        if cezalar.siklik_cezasi > 0.0 {
            *logit -= cezalar.siklik_cezasi * f64::from(*adet);
        }
        if *logit != onceki {
            dokunulan[*jeton as usize] = true;
        }
    }

    // n-gram yasağı: son (n-1) jetonun ardından bağlamda daha önce gelen
    // devamlar yasaklanır. Böylece dizi tekrarı jeton tekrarından daha güçlü
    // kesilir - gözlenen bozulmanın asıl biçimi buydu.
    if cezalar.kac_gram >= 2 && gecmis.len() >= cezalar.kac_gram {
        let n = cezalar.kac_gram;
        let onek = &gecmis[gecmis.len() - (n - 1)..];
        let mut yasaklar: Vec<u32> = Vec::new();
        for pencere in gecmis.windows(n) {
            if &pencere[..n - 1] == onek {
                let aday = pencere[n - 1];
                if !yasaklar.contains(&aday) {
                    yasaklar.push(aday);
                }
            }
        }
        for jeton in yasaklar {
            if (jeton as usize) < duzeltilmis.len() && !yasak.contains(&jeton) {
                yasak.push(jeton);
                dokunulan[jeton as usize] = true;
            }
        }
    }

    // Durma erteleme: çıktı en az bu kadar jeton olsun. Durma jetonu bir
    // tercih değil, bir sınır; ertelemesi de yasakla yapılır.
    if cezalar.en_az_jeton > 0 && uretilen.len() < cezalar.en_az_jeton {
        for jeton in dur_jetonlari {
            if (*jeton as usize) < duzeltilmis.len() && !yasak.contains(jeton) {
                yasak.push(*jeton);
                dokunulan[*jeton as usize] = true;
                rapor.ertelenen_durma += 1;
            }
        }
    }

    // Hiçbir aday kalmadıysa dağılım kurulamaz; uydurmak yerine söylenir.
    let aday = duzeltilmis
        .iter()
        .enumerate()
        .any(|(sira, logit)| logit.is_finite() && !yasak.contains(&(sira as u32)));
    if !aday {
        return Err(CezaHatasi::BosAday);
    }

    rapor.cezalanan_jeton = dokunulan.iter().filter(|d| **d).count();
    rapor.yasaklanan_jeton = yasak.len();
    Ok(CezaSonucu {
        logitler: duzeltilmis,
        yasak,
        rapor,
    })
}

#[cfg(test)]
mod testler {
    use super::*;

    fn ayar_tekrar(c: f64) -> Cezalar {
        Cezalar {
            tekrar_cezasi: c,
            ..Cezalar::default()
        }
    }

    #[test]
    fn varsayilan_kapalidir_ve_hicbir_seyi_degistirmez() {
        let cezalar = Cezalar::default();
        assert!(!cezalar.etkin());
        let logitler = vec![1.0, -2.0, 0.5];
        let sonuc = uygula(&cezalar, &logitler, &[0, 1], &[0], &[2]).expect("kapali ceza");
        assert_eq!(sonuc.logitler, logitler);
        assert!(sonuc.yasak.is_empty());
        assert!(!sonuc.rapor.etkin);
        assert_eq!(sonuc.rapor.cezalanan_jeton, 0);
        assert_eq!(sonuc.rapor.ozet(), "ceza: kapali");
    }

    #[test]
    fn tekrar_cezasi_gorulmus_jetonu_geriletir_yon_korunur() {
        let cezalar = ayar_tekrar(2.0);
        let logitler = vec![4.0, -4.0, 1.0];
        let sonuc = uygula(&cezalar, &logitler, &[0, 1], &[0], &[]).expect("tekrar cezasi");
        let cikan = &sonuc.logitler;
        assert!((cikan[0] - 2.0).abs() < 1e-12, "{cikan:?}");
        // Negatif logit bolunmez, carpilir: ceza her zaman "daha az tercih".
        assert!((cikan[1] + 8.0).abs() < 1e-12, "{cikan:?}");
        // Gorulmemis jeton ayni kalir; hicbir logit sonsuza gitmez.
        assert!((cikan[2] - 1.0).abs() < 1e-12);
        assert!(cikan.iter().all(|l| l.is_finite()), "{cikan:?}");
        assert!(sonuc.yasak.is_empty());
        assert_eq!(sonuc.rapor.cezalanan_jeton, 2);
        assert!(sonuc.rapor.ozet().contains("2 jeton geriletilmis"), "{}", sonuc.rapor.ozet());
    }

    #[test]
    fn varlik_ve_siklik_cezasi_sayimla_olceklenir() {
        let cezalar = Cezalar {
            varlik_cezasi: 0.5,
            siklik_cezasi: 0.25,
            ..Cezalar::default()
        };
        let logitler = vec![10.0, 10.0];
        let sonuc = uygula(&cezalar, &logitler, &[0, 0, 0, 1], &[0], &[]).expect("varlik/siklik");
        // 0 uc kez goruldu: 10 - 0.5 - 0.25*3 = 8.75
        assert!(
            (sonuc.logitler[0] - 8.75).abs() < 1e-12,
            "{:?}",
            sonuc.logitler
        );
        // 1 bir kez: 10 - 0.5 - 0.25 = 9.25
        assert!(
            (sonuc.logitler[1] - 9.25).abs() < 1e-12,
            "{:?}",
            sonuc.logitler
        );
    }

    #[test]
    fn ikili_dizi_yasagi_devami_tamamen_kapatir() {
        // Baglam 0 3 0: son jeton 0; "0"dan sonra bir kez 3 gelmis, o yuzden
        // 3 yasaklanir. Baska bir devam (1, 2) serbest kalir.
        let cezalar = Cezalar {
            kac_gram: 2,
            ..Cezalar::default()
        };
        let logitler = vec![0.0, 0.0, 0.0, 0.0];
        let sonuc = uygula(&cezalar, &logitler, &[0, 3, 0], &[1], &[]).expect("n-gram");
        assert_eq!(sonuc.yasak, vec![3]);
        // Yasak logiti bozmaz: dagilim kurulabilir kalir, adayliktan cikar.
        assert!(
            sonuc.logitler.iter().all(|l| l.is_finite()),
            "{:?}",
            sonuc.logitler
        );
        assert_eq!(sonuc.rapor.yasaklanan_jeton, 1);
        assert_eq!(sonuc.rapor.cezalanan_jeton, 1);

        // Dizi tekrar ederken yasak da tekrari keser: 1 2 1 2 ... baglaminda
        // son jeton 2 ise, 2'den sonra hep 1 geldigi icin 1 yasaklanir.
        let sonuc = uygula(&cezalar, &[0.0, 0.0, 0.0], &[1, 2, 1, 2], &[1], &[]).expect("n-gram");
        assert_eq!(sonuc.yasak, vec![1]);
        assert_eq!(sonuc.rapor.yasaklanan_jeton, 1);
    }

    #[test]
    fn durma_jetonu_istenen_uzunluga_kadar_ertelenir_sonra_serbest_kalir() {
        let cezalar = Cezalar {
            en_az_jeton: 3,
            ..Cezalar::default()
        };
        let erken = uygula(&cezalar, &[1.0, 1.0], &[0], &[0, 1], &[1]).expect("erteleme");
        assert_eq!(erken.yasak, vec![1]);
        assert_eq!(erken.rapor.ertelenen_durma, 1);

        let gec = uygula(&cezalar, &[1.0, 1.0], &[0, 1, 0], &[0, 1, 0], &[1]).expect("serbest");
        assert!(gec.yasak.is_empty());
        assert_eq!(gec.rapor.ertelenen_durma, 0);
    }

    #[test]
    fn butun_adaylar_yasaklanirsa_uydurulmaz_reddedilir() {
        let cezalar = Cezalar {
            en_az_jeton: 4,
            ..Cezalar::default()
        };
        // Tek jetonlu sozluk ve o jeton durma jetonu: yasak her seyi kapatir.
        let hata = uygula(&cezalar, &[0.5], &[0], &[0], &[0]).expect_err("bos aday");
        assert_eq!(hata, CezaHatasi::BosAday);
        assert!(hata.to_string().contains("aday"));
    }

    #[test]
    fn gecersiz_ayar_reddedilir() {
        let alt = ayar_tekrar(0.5);
        assert!(matches!(alt.dogrula(), Err(CezaHatasi::GecersizCezalar(_))));
        let nan = ayar_tekrar(f64::NAN);
        assert!(nan.dogrula().is_err());
        let tek = Cezalar {
            kac_gram: 1,
            ..Cezalar::default()
        };
        assert!(tek.dogrula().is_err());
        let negatif = Cezalar {
            varlik_cezasi: -1.0,
            ..Cezalar::default()
        };
        assert!(negatif.dogrula().is_err());
        assert!(Cezalar::kapali().dogrula().is_ok());
    }

    #[test]
    fn sozluk_disindaki_jeton_sessizce_yok_sayilir() {
        let cezalar = ayar_tekrar(2.0);
        // 99 sozlukte yok: panik yok, ceza yok, digerleri islenir.
        let sonuc = uygula(&cezalar, &[1.0, 1.0], &[99, 0], &[0], &[77]).expect("sozluk disi");
        assert!((sonuc.logitler[0] - 0.5).abs() < 1e-12);
        assert!(sonuc.yasak.is_empty());
        assert_eq!(sonuc.rapor.cezalanan_jeton, 1);
    }

    #[test]
    fn raporlar_toplanabilir() {
        let mut toplam = CezaRaporu::bos();
        toplam.ekle(&CezaRaporu {
            cezalanan_jeton: 2,
            yasaklanan_jeton: 1,
            ertelenen_durma: 0,
            etkin: true,
        });
        toplam.ekle(&CezaRaporu {
            cezalanan_jeton: 3,
            yasaklanan_jeton: 0,
            ertelenen_durma: 1,
            etkin: true,
        });
        assert_eq!(toplam.cezalanan_jeton, 5);
        assert_eq!(toplam.yasaklanan_jeton, 1);
        assert_eq!(toplam.ertelenen_durma, 1);
        assert!(toplam.etkin);
    }
}
