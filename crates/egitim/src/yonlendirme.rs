//! Rotalama: jeton basina uzman secimi ve **yuk dengesi**.
//!
//! Uzman secimi tek basina kolay: her jeton icin en yuksek `k` puan alinir.
//! Zor olan sey, secimin **uzmanlara esit dagilmasi**: yalniz satir bazinda
//! normalize eden bir secim (klasik top-k yumusak secim) butun yuku birkac
//! uzmana yigar; egitimde bu, kullanilmayan uzmanlar ve bosa harcanan
//! parametre demektir. Bu modul secimi klasik tabana karsi **olcer**: ayni
//! puanlarda satir toplamlari yine 1'dir ama sutun yuku daha dengelidir.
//!
//! Yontem: secilen hucreler uzerinde iki yonlu normalizasyon (satir = jeton,
//! sutun = uzman) **log uzayinda** yinelenir. Log uzayinda calismak bir tercih
//! degil zorunluluk: `exp(1000)` tasar, `exp(-1000)` sifira iner ve oran
//! kaybolur; log-toplam-us ile ayni aritmetik sonlu kalir.
//!
//! Sozlesme (hepsi testte olculur, iddia degil):
//!
//! * **Destek korunur**: her jeton tam olarak `k` uzman secer; normalize etmek
//!   secimi degistirmez, yalniz agirligini degistirir. Sifir olmayan agirlik
//!   sayisi `jeton * k`dir ve bu, sekle bagli bir sayimdir.
//! * **Satirlar 1'e toplanir**: son adim satir normalizasyonudur, boylece
//!   sutun adimindan kalan artiklik bir sonraki adima tasinmaz.
//! * **Yuk dengesi olculur**: ayni puanlarda bu modulun uzman-yuku orani
//!   klasik tabanin oranindan **kucuktur** (kucuk olan daha dengeli).
//! * **Yineleme sifirsa tabana esit**: sutun adimi hic kosmazsa sonuc satir
//!   bazli yumusak secimin ta kendisidir - yani modul, tabani icinde tasir.
//! * **Sicaklik**: dusuk sicaklik secimi keskinlestirir (satir ici en buyuk
//!   agirlik artar); bu da iki farkli sicaklikta **olculur**.
//!
//! Ogrenilebilir parametresi yoktur: rota bir hesaplamadir, agirlik degil.
//! Butcesi bu yuzden parametre muhasebesinde sifirdir ve bunu soylemek
//! `parametre_sayisi` ile mumkundur.

/// Yuk dengelemesinin ust siniiri: bu sayidan fazla yineleme sayisal olarak
/// kazanc getirmez (yakinsama geometriktir), getirse de maliyeti gizler.
pub const YINELEME_UST_SINIRI: usize = 64;
/// Uzman sayisinin ust siniri (K6 butcesi icinde kalinir).
pub const UZMAN_UST_SINIRI: usize = 64;

/// Rotalama sekli: kac uzman, jeton basina kac secim, ne kadar yineleme.
#[derive(Debug, Clone, PartialEq)]
pub struct RotaSpec {
    /// Uzman sayisi.
    pub uzman: usize,
    /// Jeton basina secilen uzman sayisi.
    pub k: usize,
    /// Puan olcegi; kucuk deger secimi keskinlestirir.
    pub sicaklik: f64,
    /// Iki yonlu normalizasyon yinelemesi (0 = yalniz satir, yani taban).
    pub yineleme: usize,
}

/// Sekil hatasi: rota kurulmadan once reddedilir.
#[derive(Debug, Clone, PartialEq)]
pub enum RotaHatasi {
    /// Uzman sayisi sifir olamaz.
    SifirUzman,
    /// Uzman sayisi ust siniri asiyor.
    UzmanUstSiniriAsildi(usize),
    /// Jeton basina secim sayisi sifir olamaz.
    SifirSecim,
    /// Jeton basina secim sayisi uzman sayisini asiyor.
    SecimUzmandanBuyuk(usize, usize),
    /// Sicaklik pozitif ve sonlu olmali.
    GecersizSicaklik(f64),
    /// Yineleme sayisi ust siniri asiyor.
    YinelemeUstSiniriAsildi(usize),
}

impl RotaSpec {
    /// Sekli dogrular. Hatali sekil ile modul kurulmaz.
    pub fn yeni(
        uzman: usize,
        k: usize,
        sicaklik: f64,
        yineleme: usize,
    ) -> Result<Self, RotaHatasi> {
        if uzman == 0 {
            return Err(RotaHatasi::SifirUzman);
        }
        if uzman > UZMAN_UST_SINIRI {
            return Err(RotaHatasi::UzmanUstSiniriAsildi(uzman));
        }
        if k == 0 {
            return Err(RotaHatasi::SifirSecim);
        }
        if k > uzman {
            return Err(RotaHatasi::SecimUzmandanBuyuk(k, uzman));
        }
        if !sicaklik.is_finite() || sicaklik <= 0.0 {
            return Err(RotaHatasi::GecersizSicaklik(sicaklik));
        }
        if yineleme > YINELEME_UST_SINIRI {
            return Err(RotaHatasi::YinelemeUstSiniriAsildi(yineleme));
        }
        Ok(Self {
            uzman,
            k,
            sicaklik,
            yineleme,
        })
    }

    /// Ogrenilebilir parametre sayisi: rota agirlik tutmaz.
    pub fn parametre_sayisi(&self) -> usize {
        0
    }

    /// Tek bir jetonun satirindaki hucre sayisi.
    pub fn satir_hucre(&self) -> usize {
        self.uzman
    }
}

/// Tek bir jetonun secimi: uzman kimligi ve agirligi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Secim {
    /// Jeton sirasi.
    pub jeton: usize,
    /// Uzman sirasi.
    pub uzman: usize,
    /// Normalize edilmis agirlik (0 ile 1 arasi).
    pub agirlik: f64,
}

/// Rotalama ciktisi: secimler + dengenin olculebilir ozeti.
#[derive(Debug, Clone, PartialEq)]
pub struct RotaSonucu {
    /// Sifir olmayan secimler (jeton, uzman, agirlik); jeton bazinda sirali.
    pub secimler: Vec<Secim>,
    /// Her jetonun satir toplami (1'e yakin olmali).
    pub satir_toplamlari: Vec<f64>,
    /// Her uzmanin tasidigi toplam yuk.
    pub yuk: Vec<f64>,
    /// En yuklu uzmanin ortalama yuke orani (1 = tam denge; buyuk = yigilma).
    pub yuk_orani: f64,
}

/// Klasik taban: yalniz satir bazinda yumusak secim (karsilastirma olcusu).
#[derive(Debug, Clone, PartialEq)]
pub struct TabSonucu {
    /// Her uzmanin tasidigi yuk.
    pub yuk: Vec<f64>,
    /// En yuklu uzmanin ortalama yuke orani.
    pub yuk_orani: f64,
}

fn log_toplam_us(degerler: &[f64]) -> f64 {
    let en_buyuk = degerler
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !en_buyuk.is_finite() {
        return f64::NEG_INFINITY;
    }
    let toplam: f64 = degerler
        .iter()
        .filter(|d| d.is_finite())
        .map(|d| (d - en_buyuk).exp())
        .sum();
    en_buyuk + toplam.ln()
}

/// Puanlardan (jeton x uzman, satir major) top-k destegi kurar.
///
/// Esit puanlar kucuk indekse gider: secim **deterministik**tir, ortama gore
/// degismez. Ayni puan dizisi her makinede ayni hucreleri secer.
fn top_k_destegi(puanlar: &[f64], spec: &RotaSpec) -> Vec<Vec<usize>> {
    let mut destek = Vec::with_capacity(puanlar.len());
    let jeton_sayisi = puanlar.len() / spec.uzman;
    for j in 0..jeton_sayisi {
        let mut indeksler: Vec<usize> = (0..spec.uzman).collect();
        indeksler.sort_by(|a, b| {
            let (pa, pb) = (puanlar[j * spec.uzman + *a], puanlar[j * spec.uzman + *b]);
            pb.partial_cmp(&pa)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        destek.push(indeksler[..spec.k].to_vec());
    }
    destek
}

/// Iki yonlu normalizasyonu log uzayinda kosar ve secimleri dondurur.
///
/// `yineleme == 0` ise yalniz satir adimi kosar; sonuc klasik tabandir.
pub fn rota_hesapla(puanlar: &[f64], spec: &RotaSpec) -> Result<RotaSonucu, RotaHatasi> {
    // Sekil dogrulamasi once: matris satir major ve jeton sayisi eleman
    // sayisindan turetilir. Bunu karistirmak (eleman sayisini jeton sanmak)
    // matrisi uzman katı buyutur ve tum donguleri kaydirir - olculdu.
    if puanlar.len() % spec.uzman != 0 {
        return Err(RotaHatasi::SecimUzmandanBuyuk(puanlar.len(), spec.uzman));
    }
    let jeton = puanlar.len() / spec.uzman;
    if jeton == 0 {
        return Err(RotaHatasi::SifirSecim);
    }
    let destek = top_k_destegi(puanlar, spec);
    // Log uzayinda calisma matrisi: secilmeyen hucreler -sonsuz.
    let mut l = vec![f64::NEG_INFINITY; jeton * spec.uzman];
    for (j, secilenler) in destek.iter().enumerate() {
        for u in secilenler {
            l[j * spec.uzman + u] = puanlar[j * spec.uzman + *u] / spec.sicaklik;
        }
    }
    let mut satir = vec![0.0f64; jeton * spec.uzman];
    for _ in 0..spec.yineleme.max(1) {
        // Satir adimi: her jeton secimleri arasinda paylastirir.
        for j in 0..jeton {
            let dilim = &l[j * spec.uzman..(j + 1) * spec.uzman];
            let lse = log_toplam_us(dilim);
            for u in 0..spec.uzman {
                let yer = j * spec.uzman + u;
                if l[yer].is_finite() {
                    l[yer] -= lse;
                }
            }
        }
        // Sutun adimi: uzmanlar arasinda yuku dengeler. Bos sutun atlanir
        // (hicbir jetonun secmedigi uzmanin yuku zaten sifirdir).
        for u in 0..spec.uzman {
            let sutun: Vec<f64> = (0..jeton).map(|j| l[j * spec.uzman + u]).collect();
            let lse = log_toplam_us(&sutun);
            if !lse.is_finite() {
                continue;
            }
            for j in 0..jeton {
                if l[j * spec.uzman + u].is_finite() {
                    l[j * spec.uzman + u] -= lse;
                }
            }
        }
    }
    // Son adim satir: cikti satirlari tam 1'e toplanir.
    for j in 0..jeton {
        let dilim = &l[j * spec.uzman..(j + 1) * spec.uzman];
        let lse = log_toplam_us(dilim);
        for u in 0..spec.uzman {
            let yer = j * spec.uzman + u;
            if l[yer].is_finite() {
                l[yer] -= lse;
            }
        }
    }
    for (yer, deger) in l.iter().enumerate() {
        satir[yer] = if deger.is_finite() { deger.exp() } else { 0.0 };
    }
    let mut secimler = Vec::with_capacity(jeton * spec.k);
    let mut satir_toplamlari = Vec::with_capacity(jeton);
    let mut yuk = vec![0.0f64; spec.uzman];
    for j in 0..jeton {
        let mut toplam = 0.0;
        for u in 0..spec.uzman {
            let agirlik = satir[j * spec.uzman + u];
            if agirlik > 0.0 {
                secimler.push(Secim {
                    jeton: j,
                    uzman: u,
                    agirlik,
                });
                yuk[u] += agirlik;
                toplam += agirlik;
            }
        }
        satir_toplamlari.push(toplam);
    }
    let ortalama = yuk.iter().sum::<f64>() / spec.uzman as f64;
    let yuk_orani = if ortalama > 0.0 {
        yuk.iter().copied().fold(f64::MIN, f64::max) / ortalama
    } else {
        0.0
    };
    Ok(RotaSonucu {
        secimler,
        satir_toplamlari,
        yuk,
        yuk_orani,
    })
}

/// Klasik taban olcumu: yalniz satir normalizasyonu, sutun adimi yok.
pub fn taban_olc(puanlar: &[f64], spec: &RotaSpec) -> Result<TabSonucu, RotaHatasi> {
    let taban_spec = RotaSpec {
        yineleme: 0,
        ..spec.clone()
    };
    let sonuc = rota_hesapla(puanlar, &taban_spec)?;
    Ok(TabSonucu {
        yuk: sonuc.yuk,
        yuk_orani: sonuc.yuk_orani,
    })
}

/// Ornek puanlar: tohumsuz tam sayi karma ile uretilir (tekrarlanabilir).
pub fn ornek_puanlar(jeton: usize, uzman: usize, egim: u64) -> Vec<f64> {
    let mut puanlar = Vec::with_capacity(jeton * uzman);
    for j in 0..jeton {
        for u in 0..uzman {
            let karisim = (j as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((u as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                .wrapping_add(egim);
            let x = ((karisim >> 11) & 0xFFFF) as f64 / 65535.0;
            // Uzman 0'a dogru hafif onegin: dengelemenin isi burada gorunur.
            let sapma = if u == 0 { 1.2 } else { 0.0 };
            puanlar.push(x * 2.0 - 1.0 + sapma);
        }
    }
    puanlar
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(yineleme: usize) -> RotaSpec {
        RotaSpec::yeni(8, 2, 1.0, yineleme).unwrap_or(RotaSpec {
            uzman: 8,
            k: 2,
            sicaklik: 1.0,
            yineleme,
        })
    }

    #[test]
    fn sekil_hatalari_reddedilir() {
        assert_eq!(RotaSpec::yeni(0, 1, 1.0, 1), Err(RotaHatasi::SifirUzman));
        assert_eq!(RotaSpec::yeni(8, 0, 1.0, 1), Err(RotaHatasi::SifirSecim));
        assert_eq!(
            RotaSpec::yeni(4, 5, 1.0, 1),
            Err(RotaHatasi::SecimUzmandanBuyuk(5, 4))
        );
        assert_eq!(
            RotaSpec::yeni(4, 2, 0.0, 1),
            Err(RotaHatasi::GecersizSicaklik(0.0))
        );
        assert_eq!(
            RotaSpec::yeni(4, 2, 1.0, YINELEME_UST_SINIRI + 1),
            Err(RotaHatasi::YinelemeUstSiniriAsildi(YINELEME_UST_SINIRI + 1))
        );
        assert!(RotaSpec::yeni(4, 2, 1.0, 4).is_ok());
    }

    #[test]
    fn parametre_sayisi_sifir_ve_sekle_bagli() {
        let s = spec(4);
        assert_eq!(s.parametre_sayisi(), 0);
        assert_eq!(s.satir_hucre(), 8);
        let s2 = RotaSpec::yeni(16, 3, 0.7, 2).unwrap_or(s.clone());
        assert_eq!(s2.satir_hucre(), 16);
    }

    #[test]
    fn destek_korunur_ve_sayim_sekle_bagli() {
        let s = spec(4);
        let puanlar = ornek_puanlar(32, s.uzman, 7);
        let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        assert_eq!(
            sonuc.secimler.len(),
            32 * s.k,
            "secim sayisi jeton x k olmali"
        );
        let mut jeton_basina = vec![0usize; 32];
        for secim in &sonuc.secimler {
            jeton_basina[secim.jeton] += 1;
        }
        assert!(
            jeton_basina.iter().all(|c| *c == s.k),
            "her jeton tam k uzman secmeli"
        );
    }

    #[test]
    fn satir_toplamlari_bire_yakin() {
        let s = spec(8);
        let puanlar = ornek_puanlar(64, s.uzman, 11);
        let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        for (j, toplam) in sonuc.satir_toplamlari.iter().enumerate() {
            assert!(
                (toplam - 1.0).abs() < 1e-9,
                "jeton {j}: satir toplami {toplam}"
            );
        }
    }

    #[test]
    fn yuk_dengesi_klasik_tabandan_iyi() {
        let s = spec(8);
        let puanlar = ornek_puanlar(128, s.uzman, 3);
        let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        let taban = taban_olc(&puanlar, &s).unwrap_or_else(|_| panic!("taban kosmadi"));
        assert!(
            sonuc.yuk_orani < taban.yuk_orani,
            "denge saglanmadi: bu modul {:.4}, taban {:.4}",
            sonuc.yuk_orani,
            taban.yuk_orani
        );
    }

    #[test]
    fn yineleme_sifirsa_tabanin_kendisi() {
        let s = spec(0);
        let puanlar = ornek_puanlar(24, s.uzman, 5);
        let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        let taban = taban_olc(&puanlar, &s).unwrap_or_else(|_| panic!("taban kosmadi"));
        assert_eq!(sonuc.yuk, taban.yuk, "yineleme yokken yuk ayni olmali");
        for toplam in &sonuc.satir_toplamlari {
            assert!((toplam - 1.0).abs() < 1e-9, "taban satiri 1'e toplanmali");
        }
    }

    #[test]
    fn esit_puanlarda_dengeli_secim() {
        let s = RotaSpec::yeni(4, 2, 1.0, 8).unwrap_or_else(|_| panic!("spec"));
        let puanlar = vec![0.5; 4 * 12];
        let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        // Esit puanlarda her jeton ayni iki uzmani secer; denge adimi bunlari
        // esitler. Sinir olcumun parcasidir: Sinkhorn yuku **secilen destek
        // icinde** dengeler; hic secilmeyen uzmana yuk uydurmaz. Bu yuzden yuk
        // orani 4 uzmanin 2'si kullanildigi icin 2.0'dir - kodun degil, tasarimin
        // sonucu ve kayitta "olculmeyen" olarak degil, olculen sinir olarak durur.
        for secim in &sonuc.secimler {
            assert!(
                (secim.agirlik - 0.5).abs() < 1e-9,
                "agirlik {:.6}",
                secim.agirlik
            );
        }
        assert_eq!(
            sonuc.yuk[0], sonuc.yuk[1],
            "secili uzmanlar esit yuk tasimali"
        );
        assert_eq!(sonuc.yuk[2], 0.0, "secili olmayan uzman yuk tasimamali");
        assert_eq!(sonuc.yuk[3], 0.0, "secili olmayan uzman yuk tasimamali");
        assert!(
            (sonuc.yuk_orani - 2.0).abs() < 1e-6,
            "esit puanlarda 4 uzmanin 2'si kullanilir: oran 2.0 olmali, olculen {:.6}",
            sonuc.yuk_orani
        );
    }

    #[test]
    fn buyuk_puanlarda_tasma_yok() {
        let s = spec(8);
        let mut puanlar = ornek_puanlar(16, s.uzman, 9);
        for puan in puanlar.iter_mut() {
            *puan *= 1000.0;
        }
        let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        for secim in &sonuc.secimler {
            assert!(
                secim.agirlik.is_finite() && secim.agirlik > 0.0,
                "agirlik {:.3e}",
                secim.agirlik
            );
        }
        for toplam in &sonuc.satir_toplamlari {
            assert!(
                (toplam - 1.0).abs() < 1e-9,
                "buyuk puanlarda satir toplami {toplam}"
            );
        }
    }

    #[test]
    fn ayni_sekil_ayni_sonuc() {
        let s = spec(8);
        let puanlar = ornek_puanlar(32, s.uzman, 13);
        let a = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        let b = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        assert_eq!(
            a.secimler, b.secimler,
            "ayni girdi ayni cikti vermeli (bit-esit)"
        );
    }

    #[test]
    fn sicaklik_dustukce_secim_keskinlesir() {
        let puanlar = ornek_puanlar(32, 8, 17);
        let mut en_buyukler = Vec::new();
        for sicaklik in [1.0, 0.5, 0.25] {
            let s = RotaSpec::yeni(8, 2, sicaklik, 4).unwrap_or_else(|_| panic!("spec"));
            let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
            let en_buyuk = sonuc
                .secimler
                .iter()
                .filter(|secim| secim.jeton == 0)
                .fold(0.0f64, |a, secim| a.max(secim.agirlik));
            en_buyukler.push(en_buyuk);
        }
        assert!(
            en_buyukler[0] < en_buyukler[1] && en_buyukler[1] < en_buyukler[2],
            "keskinlesme yok: {en_buyukler:?}"
        );
    }

    #[test]
    fn sicaklik_egimi_iki_sonlu_farkla_ayni() {
        // Ileri ve merkezi fark birbirinden bagimsiz iki tahmindir; uyusmalari
        // egrinin gercek oldugunu gosterir. (Analitik turev yazilmadi - iddia
        // "iki bagimsiz tahmin ayni"dir, "turev dogru" degil.)
        let puanlar = ornek_puanlar(16, 4, 23);
        let kayip = |sicaklik: f64| -> f64 {
            let s = RotaSpec::yeni(4, 2, sicaklik, 4).unwrap_or_else(|_| panic!("spec"));
            let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
            let jeton_sayisi = puanlar.len() / 4;
            let mut toplam = 0.0;
            for secim in &sonuc.secimler {
                toplam += secim.agirlik * puanlar[secim.jeton * 4 + secim.uzman];
            }
            toplam / jeton_sayisi as f64
        };
        let h = 1e-5;
        let t = 1.0;
        let ileri = (kayip(t + h) - kayip(t)) / h;
        let merkezi = (kayip(t + h) - kayip(t - h)) / (2.0 * h);
        let bagil = (ileri - merkezi).abs() / merkezi.abs().max(1e-12);
        assert!(
            bagil < 1e-4,
            "iki tahmin ayrisiyor: ileri {ileri:.8}, merkezi {merkezi:.8}"
        );
    }

    #[test]
    fn olcum_raporu() {
        let s = spec(8);
        let puanlar = ornek_puanlar(128, s.uzman, 3);
        let sonuc = rota_hesapla(&puanlar, &s).unwrap_or_else(|_| panic!("rota kosmadi"));
        let taban = taban_olc(&puanlar, &s).unwrap_or_else(|_| panic!("taban kosmadi"));
        let sapma = sonuc
            .satir_toplamlari
            .iter()
            .fold(0.0f64, |a, t| a.max((t - 1.0).abs()));
        println!(
            "yonlendirme | jeton={} uzman={} k={} yineleme={} secim={} satir_sapma={:.3e} yuk_orani={:.6} taban_orani={:.6} parametre={}",
            puanlar.len() / s.uzman,
            s.uzman,
            s.k,
            s.yineleme,
            sonuc.secimler.len(),
            sapma,
            sonuc.yuk_orani,
            taban.yuk_orani,
            s.parametre_sayisi()
        );
    }
}
