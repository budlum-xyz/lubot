//! Sifir merkezli RMS norm adayi (tasarim notu 3.2, norm yolu).
//!
//! Cekirdekteki QK-norm ([`crate::kernel32`]) kafa vektorlerini birim RMS'e
//! ceker; bu modul ondan ayri bir soruya bakar ve onu **degistirmez**: RMS
//! olcegi merkezden bagimsizdir, yani girdinin tamami kaydirilirsa cikti da
//! kayar. Sifir merkezli hal once ortalama cikarir, sonra RMS'e boler:
//!
//! ```text
//! c = x - ortalama(x)
//! r = sqrt(ortalama(c^2) + eps)
//! y = (1 + s) * c / r      (s: ogrenilebilir olcek, sifir baslar)
//! ```
//!
//! Neden ikisi ayri birer iddia degil de birer **olcum**: kaydirma degismezligi
//! **tamdır** (ortalama cikarildigi icin), olcek degismezligi ise eps tabani
//! yuzunden **yaklasiktir** (r^2 = ortalama(c^2) + eps; girdiyi s ile carpinca
//! r, s*r olmaz) ve sapma eps ile birlikte kuculur. Ikisi de testte ayri ayri
//! olculur ve klasik RMS ile fark da olculur
//! (`duz_rms_degisimi`) - "merkezlemek daha iyi" gibi bir cumle bu modulde
//! gecmez, cunku o cumle bir egitim karsilastirmasi ister ve kayitta
//! `olculmeyen` olarak durur.
//!
//! Geri gecis elle turetilmistir ve **sonlu farkla** denetlenir:
//!
//! ```text
//! dy_i/dx_j = delta_ij / r - c_i c_j / (n r^3) - 1 / (n r)
//! ```
//!
//! (Ortalama terimi son adimda `sum c = 0` sayesinde sadelesir; testte
//! sadelesmenin kendisi degil, turevin sonlu farkla uyumu olculur.)
//!
//! Ogrenilebilir parametresi `s` vektorudur: parametre sayisi **genisliktir**
//! ve sekle baglidir - bir literal degil.
//!
//! Sayisal sinir fail-closed: girdi, katsayi ve gradyan NaN/Inf tasiyamaz.
//! Sonlu degerler de ara carpimda tasabilir; bu durumda hata doner, sessiz
//! sifirlama/kirpma yapilmaz. Buyuk degerlerde kararlilik iddiasi yoktur:
//! kullanilan f64 formulu temsil araligini asarsa adim reddedilir.

/// Sifir merkezli RMS normun sekli.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormSpec {
    /// Vektor genisligi (kanal sayisi).
    pub genislik: usize,
    /// Sayisal taban: sifir genislikli girdide bolmeyi engeller.
    pub eps: f64,
}

/// Sifir merkezli RMS normun sekli yanlissa ya da sekle uymayan girdi gelirse.
#[derive(Debug, Clone, PartialEq)]
pub enum NormHatasi {
    /// Genislik sifir.
    SifirGenislik,
    /// Eps pozitif ve sonlu degil.
    GecersizEps(f64),
    /// Girdi uzunlugu sekle uymuyor.
    UzunlukUyusmuyor(usize, usize),
    /// Katsayi vektoru sekle uymuyor.
    OlcekUyusmuyor(usize, usize),
    /// Vektor NaN/Inf tasiyor; alan ve ilk konum bildirilir.
    SonluOlmayan { alan: &'static str, konum: usize },
    /// Sonlu girdiler ara hesapta temsil araligini asti.
    HesapTasmasi(&'static str),
}

impl NormSpec {
    /// Sekli dogrular.
    pub fn yeni(genislik: usize, eps: f64) -> Result<Self, NormHatasi> {
        if genislik == 0 {
            return Err(NormHatasi::SifirGenislik);
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(NormHatasi::GecersizEps(eps));
        }
        Ok(Self { genislik, eps })
    }

    /// Ogrenilebilir parametre sayisi: olcek vektoru, genislik kadar.
    pub fn parametre_sayisi(&self) -> usize {
        self.genislik
    }
}

/// Ileri gecisin ara degerleri; geri gecis bunlari yeniden hesaplar ama
/// olcumun denetlenebilir olmasi icin acikca tasinirlar.
#[derive(Debug, Clone, PartialEq)]
pub struct NormCikti {
    /// Cikti vektoru (olcek uygulanmis).
    pub y: Vec<f64>,
    /// Merkezlenmis girdi.
    pub c: Vec<f64>,
    /// Merkezlenmis girdinin RMS'i (eps dahil).
    pub r: f64,
}

// Bir egitim adimi NaN'yi checkpoint'e tasimamali. Ret konumu belirgindir;
// deger kirpma ya da sifirla degistirme yapilmaz.
fn sonlu_vektor(alan: &'static str, v: &[f64]) -> Result<(), NormHatasi> {
    if let Some(konum) = v.iter().position(|x| !x.is_finite()) {
        return Err(NormHatasi::SonluOlmayan { alan, konum });
    }
    Ok(())
}

fn sonlu_ara(islem: &'static str, v: f64) -> Result<f64, NormHatasi> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(NormHatasi::HesapTasmasi(islem))
    }
}

fn _cerceve(spec: NormSpec, x: &[f64]) -> Result<(f64, Vec<f64>), NormHatasi> {
    NormSpec::yeni(spec.genislik, spec.eps)?;
    if x.len() != spec.genislik {
        return Err(NormHatasi::UzunlukUyusmuyor(x.len(), spec.genislik));
    }
    sonlu_vektor("girdi", x)?;
    let n = x.len() as f64;
    let ortalama = sonlu_ara("ortalama", x.iter().sum::<f64>() / n)?;
    let c: Vec<f64> = x.iter().map(|v| v - ortalama).collect();
    sonlu_vektor("merkezlenmis", &c)?;
    let kareler = sonlu_ara("merkez kareleri", c.iter().map(|v| v * v).sum::<f64>() / n)?;
    let r = sonlu_ara("RMS paydasi", (kareler + spec.eps).sqrt())?;
    Ok((r, c))
}

/// Ileri gecis: `y = (1 + s) * (x - ortalama) / r`.
pub fn norm_ileri(spec: NormSpec, x: &[f64], olcek: &[f64]) -> Result<NormCikti, NormHatasi> {
    if olcek.len() != spec.genislik {
        return Err(NormHatasi::OlcekUyusmuyor(olcek.len(), spec.genislik));
    }
    sonlu_vektor("olcek", olcek)?;
    let (r, c) = _cerceve(spec, x)?;
    let y: Vec<f64> = c
        .iter()
        .zip(olcek.iter())
        .map(|(ci, si)| (1.0 + si) * ci / r)
        .collect();
    sonlu_vektor("cikti", &y)?;
    Ok(NormCikti { y, c, r })
}

/// Geri gecis: `(grad_x, grad_s)` dondurur.
///
/// ```text
/// gx_j = gy_j / r - (sum_i gy_i c_i) c_j / (n r^3) - (sum_i gy_i) / (n r)
/// ```
/// burada `gy = grad_y * (1 + s)` (olcekli cikis icin), `grad_s_j = grad_y_j * c_j / r`.
pub fn norm_geri(
    spec: NormSpec,
    x: &[f64],
    olcek: &[f64],
    grad_y: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), NormHatasi> {
    if olcek.len() != spec.genislik {
        return Err(NormHatasi::OlcekUyusmuyor(olcek.len(), spec.genislik));
    }
    if grad_y.len() != spec.genislik {
        return Err(NormHatasi::UzunlukUyusmuyor(grad_y.len(), spec.genislik));
    }
    sonlu_vektor("olcek", olcek)?;
    sonlu_vektor("grad_y", grad_y)?;
    let (r, c) = _cerceve(spec, x)?;
    let n = x.len() as f64;
    let gy: Vec<f64> = grad_y
        .iter()
        .zip(olcek.iter())
        .map(|(g, s)| g * (1.0 + s))
        .collect();
    sonlu_vektor("olcekli gradyan", &gy)?;
    let gy_c = sonlu_ara(
        "gradyan merkez carpimi",
        gy.iter().zip(c.iter()).map(|(g, ci)| g * ci).sum(),
    )?;
    let gy_toplam = sonlu_ara("gradyan toplami", gy.iter().sum())?;
    let nr = sonlu_ara("gradyan paydasi", n * r)?;
    let nr3 = sonlu_ara("gradyan kubik paydasi", n * r * r * r)?;
    let gx: Vec<f64> = (0..c.len())
        .map(|j| gy[j] / r - gy_c * c[j] / nr3 - gy_toplam / nr)
        .collect();
    let grad_s: Vec<f64> = (0..c.len()).map(|j| grad_y[j] * c[j] / r).collect();
    sonlu_vektor("grad_x", &gx)?;
    sonlu_vektor("grad_s", &grad_s)?;
    Ok((gx, grad_s))
}

/// Olcekleme girdisi icin merkezlenmis girdiyi dondurur (olcum yardimcisi).
pub fn merkezle(spec: NormSpec, x: &[f64]) -> Result<Vec<f64>, NormHatasi> {
    let (_, c) = _cerceve(spec, x)?;
    Ok(c)
}

/// Klasik RMS (merkezlemez): karsilastirma olcusu. Bu modul onu kullanmaz;
/// yalnizca farki **olcmek** icin disariya aciktir.
pub fn duz_rms(spec: NormSpec, x: &[f64]) -> Result<Vec<f64>, NormHatasi> {
    NormSpec::yeni(spec.genislik, spec.eps)?;
    if x.len() != spec.genislik {
        return Err(NormHatasi::UzunlukUyusmuyor(x.len(), spec.genislik));
    }
    sonlu_vektor("girdi", x)?;
    let n = x.len() as f64;
    let r = sonlu_ara(
        "duz RMS paydasi",
        (x.iter().map(|v| v * v).sum::<f64>() / n + spec.eps).sqrt(),
    )?;
    let y: Vec<f64> = x.iter().map(|v| v / r).collect();
    sonlu_vektor("cikti", &y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ornek(genislik: usize) -> Vec<f64> {
        (0..genislik)
            .map(|i| ((i as f64) * 0.37 - 1.5).sin() * 2.0 + 0.3)
            .collect()
    }

    fn spec() -> NormSpec {
        NormSpec {
            genislik: 8,
            eps: 1e-6,
        }
    }

    fn sifir_olcek(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    #[test]
    fn sekil_hatalari_reddedilir() {
        assert_eq!(NormSpec::yeni(0, 1e-6), Err(NormHatasi::SifirGenislik));
        assert_eq!(NormSpec::yeni(8, 0.0), Err(NormHatasi::GecersizEps(0.0)));
        match NormSpec::yeni(8, f64::NAN) {
            Err(NormHatasi::GecersizEps(e)) => assert!(e.is_nan(), "yakalanan deger {e}"),
            digeri => panic!("NaN eps kabul edildi: {digeri:?}"),
        }
        assert!(NormSpec::yeni(8, 1e-6).is_ok());
        let s = spec();
        assert_eq!(
            norm_ileri(s, &[1.0; 4], &sifir_olcek(8)),
            Err(NormHatasi::UzunlukUyusmuyor(4, 8))
        );
        assert_eq!(
            norm_ileri(s, &ornek(8), &sifir_olcek(3)),
            Err(NormHatasi::OlcekUyusmuyor(3, 8))
        );
    }

    #[test]
    fn parametre_sayisi_genisliktir() {
        let s = spec();
        assert_eq!(s.parametre_sayisi(), 8);
        let s2 = NormSpec::yeni(32, 1e-5).unwrap_or_else(|_| panic!("spec"));
        assert_eq!(s2.parametre_sayisi(), 32);
    }

    #[test]
    fn cikti_sifir_ortalamali_ve_birim_rms() {
        let s = spec();
        let x = ornek(8);
        let cikti = norm_ileri(s, &x, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
        let ortalama = cikti.y.iter().sum::<f64>() / 8.0;
        assert!(ortalama.abs() < 1e-12, "cikti ortalamasi {ortalama}");
        let kareler = cikti.y.iter().map(|v| v * v).sum::<f64>() / 8.0;
        // Tam birim RMS ancak eps = 0 iken olur: r^2 = ortalama(c^2) + eps
        // oldugu icin cikti RMS karesi tam olarak 1 - eps/r^2'dir. Beklentiyi
        // gevsetmek yerine **tam iliski** olculur.
        let beklenen = 1.0 - s.eps / (cikti.r * cikti.r);
        assert!(
            (kareler - beklenen).abs() < 1e-12,
            "RMS karesi {kareler}, beklenen {beklenen}"
        );
    }

    #[test]
    fn kaydirma_degismezligi() {
        let s = spec();
        let x = ornek(8);
        let a = norm_ileri(s, &x, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
        let kaydirilmis: Vec<f64> = x.iter().map(|v| v + 12.5).collect();
        let b = norm_ileri(s, &kaydirilmis, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
        for (ya, yb) in a.y.iter().zip(b.y.iter()) {
            assert!(
                (ya - yb).abs() < 1e-12,
                "kaydirma ciktiyi degistirdi: {ya} vs {yb}"
            );
        }
    }

    #[test]
    fn olcek_degismezligi_eps_tabanina_kadar() {
        // Olcekleme degismezligi de kaydirma gibi **tam degil** olabilir: r^2 =
        // ortalama(c^2) + eps oldugu icin girdiyi s ile carpinca r, s*r olmaz.
        // Fark eps ile olceklenir ve eps kucukken sapma da kucuktur - ikisi
        // birden olculur.
        let x = ornek(8);
        let buyutulmus: Vec<f64> = x.iter().map(|v| v * 7.5).collect();
        let sapma = |eps: f64| -> f64 {
            let s = NormSpec { genislik: 8, eps };
            let a = norm_ileri(s, &x, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
            let b = norm_ileri(s, &buyutulmus, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
            a.y.iter()
                .zip(b.y.iter())
                .map(|(ya, yb)| (ya - yb).abs())
                .fold(0.0f64, f64::max)
        };
        let kaba = sapma(1e-6);
        let ince = sapma(1e-18);
        assert!(ince < 1e-9, "eps kucukken sapma kalmali mi: {ince:.3e}");
        assert!(
            kaba > ince,
            "eps buyudukce sapma buyumeli: kaba {kaba:.3e}, ince {ince:.3e}"
        );
    }

    #[test]
    fn klasik_rms_kaydirmadan_etkilenir_fark_olculur() {
        // Adayin varlik sebebi bir iddia degil bir **olcum**: klasik RMS
        // kaydirmaya duyarli, sifir merkezli hal degil.
        let s = spec();
        let x = ornek(8);
        let kaydirilmis: Vec<f64> = x.iter().map(|v| v + 25.0).collect();
        let a = duz_rms(s, &x).unwrap_or_else(|_| panic!("rms"));
        let b = duz_rms(s, &kaydirilmis).unwrap_or_else(|_| panic!("rms"));
        let fark = a
            .iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max);
        assert!(fark > 1.0, "klasik RMS kaydirmadan etkilenmedi: {fark}");
    }

    #[test]
    fn sabit_girdi_sifira_gider_patlamaz() {
        // Butun kanallar ayni ise merkezlenmis hal sifirdir; eps bolmeyi
        // kurtarir ve cikti NaN degil sifirdir.
        let s = spec();
        let x = vec![3.25; 8];
        let cikti = norm_ileri(s, &x, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
        assert!(
            cikti.y.iter().all(|v| *v == 0.0),
            "sabit girdi sifir cikmali"
        );
        assert!(cikti.r.is_finite() && cikti.r > 0.0, "r {0}", cikti.r);
    }

    #[test]
    fn olcek_sifirken_saf_norm() {
        let s = spec();
        let x = ornek(8);
        let a = norm_ileri(s, &x, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
        let b = norm_ileri(s, &x, &[0.0; 8]).unwrap_or_else(|_| panic!("ileri"));
        assert_eq!(a.y, b.y, "olcek sifirken saf norm beklenir");
        let c = norm_ileri(s, &x, &[1.0; 8]).unwrap_or_else(|_| panic!("ileri"));
        for (ya, yc) in a.y.iter().zip(c.y.iter()) {
            assert!(
                (yc - 2.0 * ya).abs() < 1e-12,
                "olcek (1+s) olarak uygulanmali"
            );
        }
    }

    #[test]
    fn geri_gecis_sonlu_farkla_uyusur() {
        // Merkezi fark, elle turetilen geri gecisle karsilastirilir: ayni
        // sayilar cikmali. Denetlenen girdi sayisi genislige baglidir.
        let s = spec();
        let x = ornek(8);
        let olcek = vec![0.15, -0.2, 0.05, 0.3, -0.1, 0.25, 0.0, 0.4];
        let gy: Vec<f64> = (0..8).map(|i| ((i as f64) * 0.7).cos()).collect();
        let (gx, gs) = norm_geri(s, &x, &olcek, &gy).unwrap_or_else(|_| panic!("geri"));
        let kayip = |girdi: &[f64]| -> f64 {
            let cikti = norm_ileri(s, girdi, &olcek).unwrap_or_else(|_| panic!("ileri"));
            cikti.y.iter().zip(gy.iter()).map(|(y, g)| y * g).sum()
        };
        let h = 1e-6;
        let mut denetlenen = 0usize;
        for j in 0..8 {
            let mut arti = x.clone();
            let mut eksi = x.clone();
            arti[j] += h;
            eksi[j] -= h;
            let fd = (kayip(&arti) - kayip(&eksi)) / (2.0 * h);
            let bagil = (fd - gx[j]).abs() / fd.abs().max(1e-9);
            assert!(bagil < 1e-5, "girdi {j}: fd {fd:.9} geri {:.9}", gx[j]);
            denetlenen += 1;
        }
        assert_eq!(
            denetlenen, s.genislik,
            "denetlenen girdi sayisi genislik olmali"
        );
        // Katsayi gradyani da ayni yolla denenir.
        let kayip_olcekli = |o: &[f64]| -> f64 {
            let cikti = norm_ileri(s, &x, o).unwrap_or_else(|_| panic!("ileri"));
            cikti.y.iter().zip(gy.iter()).map(|(y, g)| y * g).sum()
        };
        for j in 0..8 {
            let mut arti = olcek.clone();
            let mut eksi = olcek.clone();
            arti[j] += h;
            eksi[j] -= h;
            let fd = (kayip_olcekli(&arti) - kayip_olcekli(&eksi)) / (2.0 * h);
            let bagil = (fd - gs[j]).abs() / fd.abs().max(1e-9);
            assert!(bagil < 1e-5, "olcek {j}: fd {fd:.9} geri {:.9}", gs[j]);
        }
    }

    #[test]
    fn geri_gecis_toplami_sifir_verir() {
        // Cikti merkezli oldugu icin sabit bir gradyan akisi girdiye net kuvvet
        // uygulamamali: gy = 1 vektorunde girdi gradyani sifira toplanir.
        let s = spec();
        let x = ornek(8);
        let (gx, _) =
            norm_geri(s, &x, &sifir_olcek(8), &[1.0; 8]).unwrap_or_else(|_| panic!("geri"));
        let toplam: f64 = gx.iter().sum();
        assert!(toplam.abs() < 1e-12, "girdi gradyani toplami {toplam}");
    }

    #[test]
    fn ayni_girdi_ayni_cikti() {
        let s = spec();
        let x = ornek(8);
        let a = norm_ileri(s, &x, &[0.1; 8]).unwrap_or_else(|_| panic!("ileri"));
        let b = norm_ileri(s, &x, &[0.1; 8]).unwrap_or_else(|_| panic!("ileri"));
        assert_eq!(a.y, b.y, "deterministik olmali");
    }

    #[test]
    fn olcum_raporu() {
        let s = spec();
        let x = ornek(8);
        let kaydirilmis: Vec<f64> = x.iter().map(|v| v + 25.0).collect();
        let z = norm_ileri(s, &x, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
        let zk = norm_ileri(s, &kaydirilmis, &sifir_olcek(8)).unwrap_or_else(|_| panic!("ileri"));
        let merkez_fark =
            z.y.iter()
                .zip(zk.y.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
        let d = duz_rms(s, &x).unwrap_or_else(|_| panic!("rms"));
        let dk = duz_rms(s, &kaydirilmis).unwrap_or_else(|_| panic!("rms"));
        let duz_fark = d
            .iter()
            .zip(dk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        // gradyan olcumu iyi kosullandirilir: gy = 1 alinirsa cikti zaten
        // sifir ortalamali oldugu icin kayip girdiye karsi **neredeyse sabit**
        // kalir, merkezi fark gurultuye duser ve oran anlamsizlasir (olculdu:
        // gy = 1 ile sapma 8.9e-1 gorunuyordu). Farkli bir gy ile olcum
        // kosullu hale gelir.
        let olcek = vec![0.15, -0.2, 0.05, 0.3, -0.1, 0.25, 0.0, 0.4];
        let gy: Vec<f64> = (0..8).map(|i| ((i as f64) * 0.7).cos()).collect();
        let (gx, _) = norm_geri(s, &x, &olcek, &gy).unwrap_or_else(|_| panic!("geri"));
        let kayip = |girdi: &[f64]| -> f64 {
            let cikti = norm_ileri(s, girdi, &olcek).unwrap_or_else(|_| panic!("ileri"));
            cikti.y.iter().zip(gy.iter()).map(|(y, g)| y * g).sum()
        };
        let h = 1e-6;
        let mut denetlenen = 0usize;
        let sapma = (0..8)
            .map(|j| {
                denetlenen += 1;
                let mut arti = x.clone();
                let mut eksi = x.clone();
                arti[j] += h;
                eksi[j] -= h;
                let fd = (kayip(&arti) - kayip(&eksi)) / (2.0 * h);
                // Simetrik taban: fd de gx de sifira yaklasabilir; oran tek
                // tarafa gore alinirsa olcum sisirilir.
                (fd - gx[j]).abs() / fd.abs().max(gx[j].abs()).max(1e-9)
            })
            .fold(0.0f64, f64::max);
        let ortalama = z.y.iter().sum::<f64>() / 8.0;
        let kareler = z.y.iter().map(|v| v * v).sum::<f64>() / 8.0;
        println!(
            "normalizasyon | genislik={} denetlenen={} merkez_fark={:.3e} duz_rms_fark={:.3e} cikti_ort={:.3e} cikti_rms2={:.6} gradyan_sapma={:.3e} parametre={}",
            s.genislik,
            denetlenen,
            merkez_fark,
            duz_fark,
            ortalama,
            kareler,
            sapma,
            s.parametre_sayisi()
        );
    }

    #[test]
    fn sonlu_olmayan_girdi_tum_yollarda_reddedilir() {
        let s = spec();
        for konum in 0..s.genislik {
            for deger in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut x = ornek(s.genislik);
                x[konum] = deger;
                let beklenen = Err(NormHatasi::SonluOlmayan {
                    alan: "girdi",
                    konum,
                });
                assert_eq!(norm_ileri(s, &x, &[0.0; 8]), beklenen);
                assert!(norm_geri(s, &x, &[0.0; 8], &[1.0; 8]).is_err());
                assert!(duz_rms(s, &x).is_err());
                assert!(merkezle(s, &x).is_err());
            }
        }
    }

    #[test]
    fn katsayi_ve_gradyan_sonlulugu_denetlenir() {
        let s = spec();
        let x = ornek(8);
        for konum in 0..8 {
            for deger in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut v = [0.0; 8];
                v[konum] = deger;
                assert_eq!(
                    norm_ileri(s, &x, &v),
                    Err(NormHatasi::SonluOlmayan {
                        alan: "olcek",
                        konum
                    })
                );
                assert_eq!(
                    norm_geri(s, &x, &v, &[1.0; 8]),
                    Err(NormHatasi::SonluOlmayan {
                        alan: "olcek",
                        konum
                    })
                );
                assert_eq!(
                    norm_geri(s, &x, &[0.0; 8], &v),
                    Err(NormHatasi::SonluOlmayan {
                        alan: "grad_y",
                        konum
                    })
                );
            }
        }
    }

    #[test]
    fn sonlu_girdi_ara_hesapta_tasarsa_ret() {
        let s = spec();
        let x = [f64::MAX; 8];
        assert!(matches!(
            norm_ileri(s, &x, &[0.0; 8]),
            Err(NormHatasi::HesapTasmasi("ortalama"))
        ));
        assert!(duz_rms(s, &x).is_err());
        let z = [1e200, -1e200, 1e200, -1e200, 1e200, -1e200, 1e200, -1e200];
        assert!(matches!(
            norm_ileri(s, &z, &[0.0; 8]),
            Err(NormHatasi::HesapTasmasi("merkez kareleri"))
        ));
    }

    #[test]
    fn sonlu_katsayi_ciktiyi_tasirabilir() {
        let s = spec();
        let x = [7.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        assert!(norm_ileri(s, &x, &[f64::MAX; 8]).is_err());
        assert!(norm_geri(s, &x, &[f64::MAX; 8], &[2.0; 8]).is_err());
    }

    #[test]
    fn geri_gecis_ara_carpim_tasmasini_reddeder() {
        let s = spec();
        let x = [1e100, -1e100, 1e100, -1e100, 1e100, -1e100, 1e100, -1e100];
        assert!(norm_ileri(s, &x, &[0.0; 8]).is_ok());
        assert!(norm_geri(s, &x, &[0.0; 8], &[f64::MAX; 8]).is_err());
        let y = [1e110, -1e110, 1e110, -1e110, 1e110, -1e110, 1e110, -1e110];
        assert!(matches!(
            norm_geri(s, &y, &[0.0; 8], &[1.0; 8]),
            Err(NormHatasi::HesapTasmasi("gradyan kubik paydasi"))
        ));
    }

    #[test]
    fn elle_kurulan_gecersiz_norm_spec_tum_yollarda_ret() {
        for s in [
            NormSpec {
                genislik: 0,
                eps: 1e-6,
            },
            NormSpec {
                genislik: 8,
                eps: -1.0,
            },
            NormSpec {
                genislik: 8,
                eps: f64::NAN,
            },
        ] {
            let x = vec![0.0; s.genislik];
            assert!(norm_ileri(s, &x, &x).is_err());
            assert!(norm_geri(s, &x, &x, &x).is_err());
            assert!(duz_rms(s, &x).is_err());
            assert!(merkezle(s, &x).is_err());
        }
    }

    #[test]
    fn norm_hatasi_girdileri_degistirmez() {
        let s = spec();
        let x = ornek(8);
        let olcek = [f64::MAX; 8];
        let grad = [2.0; 8];
        let once = x.clone();
        assert!(norm_geri(s, &x, &olcek, &grad).is_err());
        assert_eq!(x, once);
        assert_eq!(olcek, [f64::MAX; 8]);
        assert_eq!(grad, [2.0; 8]);
    }
}
