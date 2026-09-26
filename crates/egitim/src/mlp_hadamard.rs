//! Hadamard (Monarch bloklu) MLP adayi — port bileseni 3.
//!
//! Uygulama direktifinin mimari-port listesindeki birinci bilesen (tasarim notu
//! 3.1): standart iki-matris MLP yerine, iki dogrusal projeksiyonun eleman-bazli
//! (Hadamard) carpimi ve ustune dusus projeksiyonu:
//!
//! ```text
//! u = W1 x + b1        (d_model -> d_r)
//! v = W2 x + b2        (d_model -> d_r)
//! h = u ⊙ gelu(v)      (eleman bazli; d_r)
//! y = W3 h + b3        (d_r -> d_model)
//! ```
//!
//! `W1` ve `W2` **blok-kosegen** olabilir (Monarch deseni): genislik `blok` parçaya
//! bolunur ve her parca yalnizca kendi girdi diliminden okur. Blok sayisi buyudukce
//! parametre sayisi duser; tam bloksuz hal (`blok = 1`) butun genislikleri gorur.
//!
//! # Neden bu modul ayri duruyor
//!
//! Bu bir **mimari degisiklik adayidir, uygulanmis bir mimari degil**: spec'i
//! (`training/model_spec.json`) degistirmez ve egitim dongusune baglanmaz. Kendi
//! olcumleriyle durur — gradyan sonlu farkla, parametre sayisi tam sayi
//! aritmetigiyle, blok-kosegenligi bayt-esitligiyle ve inis (loss descent)
//! gercek bir kosuyla. Baglanma karari tasarim notunun M-listesindeki isaretli
//! karardir ve olcumden once verilmez.
//!
//! # Olculen iki sey, iddia edilen hicbir sey
//!
//! 1. **Parametre muhasebesi.** `parametre_sayisi` sekilden turetilir; bloklu
//!    halin tasarrufu tam sayi aritmetigiyle gosterilir. "Daha verimli" diye bir
//!    iddia yok: bloksuz Hadamard ayni ic genislikte standart MLP'den **pahalidir**
//!    (ek bir dusus izdusumu tasir) — bu, testte yazili olan gercek.
//! 2. **Gradyan.** Elle yazilmis geri gecis, crate'in kendi olcusuyle
//!    (`GRADIENT_CHECK_*`) merkezi sonlu farka karsi denetlenir; denetlenen
//!    gradyan sayisi seklin parametre sayisina esit olmak zorundadir.
//!
//! Aktivasyon **crate'in kendi GELU'su**dur (`crate::gelu`, `crate::gelu_turev`):
//! burada ikinci bir kopya tutmak, tanh bicimli fonksiyonun tureviyle
//! uyusmamasinin klasik yoludur.

use crate::{gelu, gelu_turev};

/// Ust sinir: blok sayisi bu kadar parçaya bolunur; daha fazlasi testte degil,
/// spec kararinda tartisilir.
pub const BLOK_UST_SINIRI: usize = 64;

/// Hadamard MLP'nin sekli.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HadamardSpec {
    /// Model genisligi (girdi ve cikti).
    pub d_model: usize,
    /// Ic genislik (Hadamard carpiminin uzunlugu).
    pub d_r: usize,
    /// Blok sayisi: `1` = bloksuz (tam izdusum), `k` = blok-kosegen.
    pub blok: usize,
}

/// Neden bir sekil reddedildi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HadamardSekilHatasi {
    /// Sifir boyut.
    BosBoyut,
    /// Genislik blok sayisina bolunmuyor.
    GenislikBolmuyor,
    /// Ic genislik blok sayisina bolunmuyor.
    IcGenislikBolmuyor,
    /// Blok sayisi `1..=BLOK_UST_SINIRI` disinda.
    BlokAraligi,
    /// Ic genislik model genisliginden buyuk olamaz diye bir kural yok; ama
    /// sifir olamaz. (Ayri varyant: gelecekteki kurallar icin yer.)
    IcGenislikKucuk,
}

impl HadamardSpec {
    /// Ilk olculecek aday: `lubot-a1` genisliginde, dort bloklu, 4x ic genislik.
    #[must_use]
    pub const fn lubot_a1_adayi() -> Self {
        Self {
            d_model: 64,
            d_r: 256,
            blok: 4,
        }
    }

    /// Blok basina girdi dilimi.
    #[must_use]
    pub fn blok_girisi(&self) -> usize {
        if self.blok == 0 {
            return 0;
        }
        self.d_model / self.blok
    }

    /// Blok basina ic dilim.
    #[must_use]
    pub fn blok_ici(&self) -> usize {
        if self.blok == 0 {
            return 0;
        }
        self.d_r / self.blok
    }

    /// # Errors
    /// [`HadamardSekilHatasi`] — hangi boyutun neden reddedildigini tasir.
    pub fn dogrula(&self) -> Result<(), HadamardSekilHatasi> {
        if self.d_model == 0 || self.d_r == 0 {
            return Err(HadamardSekilHatasi::BosBoyut);
        }
        if self.blok == 0 || self.blok > BLOK_UST_SINIRI {
            return Err(HadamardSekilHatasi::BlokAraligi);
        }
        if !self.d_model.is_multiple_of(self.blok) {
            return Err(HadamardSekilHatasi::GenislikBolmuyor);
        }
        if !self.d_r.is_multiple_of(self.blok) {
            return Err(HadamardSekilHatasi::IcGenislikBolmuyor);
        }
        Ok(())
    }

    /// Bu seklin parametre sayisi: iki bloklu izdusum, bir dusus izdusumu ve
    /// yanliliklar. Sekilden turetilir — hicbir sayi burada sabit degildir.
    #[must_use]
    pub fn parametre_sayisi(&self) -> usize {
        let w_ic = 2 * self.blok * self.blok_girisi() * self.blok_ici();
        let w_dusus = self.d_r * self.d_model;
        let yanlilik = 2 * self.d_r + self.d_model;
        w_ic + w_dusus + yanlilik
    }

    /// Ayni seklin bloksuz karsiligi (`blok = 1`) — tasarruf olcusunun temeli.
    #[must_use]
    pub fn bloksuz_parametre_sayisi(&self) -> usize {
        let bloksuz = Self { blok: 1, ..*self };
        bloksuz.parametre_sayisi()
    }

    /// Ayni ic genislikte standart iki-matris MLP'nin parametre sayisi.
    /// Karsilastirma icin; "ustunluk" iddiasi degil.
    #[must_use]
    pub fn standart_mlp_parametre_sayisi(d_model: usize, ic_genislik: usize) -> usize {
        2 * d_model * ic_genislik + ic_genislik + d_model
    }
}

/// Butun agirliklar tek yerde. `w1`/`w2` blok-kosegendir: `blok * (blok_ici x
/// blok_girisi)` duz dizisi; `w3` yogun `(d_model x d_r)`.
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardAgirliklar {
    /// Yukari izdusum 1 (bloklar sirali).
    pub w1: Vec<f64>,
    /// Yukari izdusum 2 (bloklar sirali).
    pub w2: Vec<f64>,
    /// Dusus izdusumu, `[d_model][d_r]`.
    pub w3: Vec<f64>,
    /// Birinci dalin yanliligi, `d_r`.
    pub b1: Vec<f64>,
    /// Ikinci dalin yanliligi, `d_r`.
    pub b2: Vec<f64>,
    /// Cikis yanliligi, `d_model`.
    pub b3: Vec<f64>,
}

/// Ileri gecisin sakladigi her sey.
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardBellek {
    /// Birinci dal cikisi (carpim oncesi), `t x d_r`.
    pub u: Vec<f64>,
    /// Ikinci dal cikisi (carpim oncesi), `t x d_r`.
    pub v: Vec<f64>,
    /// Carpim cikisi, `t x d_r`.
    pub h: Vec<f64>,
    /// Girdi, `t x d_model` — geri gecis izdusum gradyanlari icin ister.
    pub girdi: Vec<f64>,
    /// Pencere uzunlugu.
    pub t: usize,
}

/// Geri gecisin urettigi gradyanlar.
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardGradyanlar {
    /// Yukari izdusum 1 gradyani.
    pub w1: Vec<f64>,
    /// Yukari izdusum 2 gradyani.
    pub w2: Vec<f64>,
    /// Dusus izdusumu gradyani.
    pub w3: Vec<f64>,
    /// Birinci yanlilik gradyani.
    pub b1: Vec<f64>,
    /// Ikinci yanlilik gradyani.
    pub b2: Vec<f64>,
    /// Cikis yanliligi gradyani.
    pub b3: Vec<f64>,
    /// Girdi gradyani, `t x d_model`.
    pub girdi: Vec<f64>,
}

/// Kucuk, deterministik dolgu: ayni tohum her makinede ayni sayilari verir.
#[must_use]
pub fn belirgin_doldur(spec: HadamardSpec, tohum: u64) -> HadamardAgirliklar {
    let mut sayac: u64 = tohum
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut uret = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| {
                sayac = sayac
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let ust = (sayac >> 33) as f64 / (1u64 << 31) as f64;
                (ust - 0.5) * 0.5
            })
            .collect()
    };
    let ic = 2 * spec.blok * spec.blok_girisi() * spec.blok_ici();
    HadamardAgirliklar {
        w1: uret(ic / 2),
        w2: uret(ic / 2),
        w3: uret(spec.d_r * spec.d_model),
        b1: uret(spec.d_r),
        b2: uret(spec.d_r),
        b3: uret(spec.d_model),
    }
}

/// Blok-kosegen izdusum: `girdi (t x d_model) -> cikti (t x d_r)`, yalniz kendi
/// blok diliminden okur. `w`, `blok * (blok_ici x blok_girisi)` duz dizisidir.
fn blok_izdusum(
    x: &[f64],
    w: &[f64],
    b: &[f64],
    t: usize,
    d_model: usize,
    d_r: usize,
    blok: usize,
) -> Vec<f64> {
    let girisi = d_model / blok;
    let ici = d_r / blok;
    let mut y = vec![0.0f64; t * d_r];
    for p in 0..blok {
        let w_tabani = p * ici * girisi;
        let x_tabani = p * girisi;
        let y_tabani = p * ici;
        for i in 0..t {
            for c in 0..ici {
                let mut toplam = b[y_tabani + c];
                for k in 0..girisi {
                    toplam += x[i * d_model + x_tabani + k] * w[w_tabani + c * girisi + k];
                }
                y[i * d_r + y_tabani + c] = toplam;
            }
        }
    }
    y
}

/// Ileri gecis. `girdi` uzunlugu `t * d_model` olmali.
#[must_use]
pub fn ileri(
    spec: HadamardSpec,
    a: &HadamardAgirliklar,
    girdi: &[f64],
    t: usize,
) -> (Vec<f64>, HadamardBellek) {
    let d = spec.d_model;
    let dr = spec.d_r;
    let u = blok_izdusum(girdi, &a.w1, &a.b1, t, d, dr, spec.blok);
    let v = blok_izdusum(girdi, &a.w2, &a.b2, t, d, dr, spec.blok);
    let mut h = vec![0.0f64; t * dr];
    for i in 0..t * dr {
        h[i] = u[i] * gelu(v[i]);
    }
    // Dusus izdusumu: yogun, blok yok.
    let mut cikti = vec![0.0f64; t * d];
    for i in 0..t {
        for o in 0..d {
            let mut toplam = a.b3[o];
            for j in 0..dr {
                toplam += h[i * dr + j] * a.w3[o * dr + j];
            }
            cikti[i * d + o] = toplam;
        }
    }
    let bellek = HadamardBellek {
        u,
        v,
        h,
        girdi: girdi.to_vec(),
        t,
    };
    (cikti, bellek)
}

/// Geri gecis: `d_cikti` (t x d_model) verilir.
#[must_use]
pub fn geri(
    spec: HadamardSpec,
    a: &HadamardAgirliklar,
    bellek: &HadamardBellek,
    d_cikti: &[f64],
) -> HadamardGradyanlar {
    let d = spec.d_model;
    let dr = spec.d_r;
    let blok = spec.blok;
    let girisi = spec.blok_girisi();
    let ici = spec.blok_ici();
    let t = bellek.t;

    // Dusus izdusumu geri gecisi.
    let mut d_w3 = vec![0.0f64; d * dr];
    let mut d_b3 = vec![0.0f64; d];
    let mut dh = vec![0.0f64; t * dr];
    for i in 0..t {
        for o in 0..d {
            let g = d_cikti[i * d + o];
            d_b3[o] += g;
            for j in 0..dr {
                d_w3[o * dr + j] += g * bellek.h[i * dr + j];
                dh[i * dr + j] += g * a.w3[o * dr + j];
            }
        }
    }

    // Carpim geri gecisi: h = u ⊙ gelu(v).
    let mut du = vec![0.0f64; t * dr];
    let mut dv = vec![0.0f64; t * dr];
    for i in 0..t * dr {
        let g = dh[i];
        du[i] = g * gelu(bellek.v[i]);
        dv[i] = g * bellek.u[i] * gelu_turev(bellek.v[i]);
    }

    // Yukari izdusumlerin geri gecisi (blok-kosegen), ve girdi gradyani.
    let mut d_w1 = vec![0.0f64; blok * ici * girisi];
    let mut d_w2 = vec![0.0f64; blok * ici * girisi];
    let mut d_b1 = vec![0.0f64; dr];
    let mut d_b2 = vec![0.0f64; dr];
    let mut d_girdi = vec![0.0f64; t * d];
    for p in 0..blok {
        let w_tabani = p * ici * girisi;
        let x_tabani = p * girisi;
        let y_tabani = p * ici;
        for i in 0..t {
            for c in 0..ici {
                let g1 = du[i * dr + y_tabani + c];
                let g2 = dv[i * dr + y_tabani + c];
                d_b1[y_tabani + c] += g1;
                d_b2[y_tabani + c] += g2;
                for k in 0..girisi {
                    let x = bellek.girdi[i * d + x_tabani + k];
                    d_w1[w_tabani + c * girisi + k] += g1 * x;
                    d_w2[w_tabani + c * girisi + k] += g2 * x;
                    d_girdi[i * d + x_tabani + k] +=
                        g1 * a.w1[w_tabani + c * girisi + k] + g2 * a.w2[w_tabani + c * girisi + k];
                }
            }
        }
    }

    HadamardGradyanlar {
        w1: d_w1,
        w2: d_w2,
        w3: d_w3,
        b1: d_b1,
        b2: d_b2,
        b3: d_b3,
        girdi: d_girdi,
    }
}

/// Kayip: `0.5 * sum((cikti - hedef)^2)` — crate'in egitim kaybiyla ayni bicim.
#[must_use]
pub fn kayip(cikti: &[f64], hedef: &[f64]) -> f64 {
    0.5 * cikti
        .iter()
        .zip(hedef.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
}

/// Kaybin ciktiya gore gradyani.
#[must_use]
pub fn kayip_gradyan(cikti: &[f64], hedef: &[f64]) -> Vec<f64> {
    cikti.iter().zip(hedef.iter()).map(|(a, b)| a - b).collect()
}

/// Tek bir egim-inis adimi: butun agirliklar ayni ogrenme oraniyla guncellenir.
pub fn sgd_adimi(a: &mut HadamardAgirliklar, g: &HadamardGradyanlar, lr: f64) {
    let alanlar: [(&mut Vec<f64>, &Vec<f64>); 6] = [
        (&mut a.w1, &g.w1),
        (&mut a.w2, &g.w2),
        (&mut a.w3, &g.w3),
        (&mut a.b1, &g.b1),
        (&mut a.b2, &g.b2),
        (&mut a.b3, &g.b3),
    ];
    for (agirlik, gradyan) in alanlar {
        for (w, gr) in agirlik.iter_mut().zip(gradyan.iter()) {
            *w -= lr * gr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GRADIENT_CHECK_MUTLAK_TABAN, GRADIENT_CHECK_TOLERANCE};

    fn kucuk() -> HadamardSpec {
        HadamardSpec {
            d_model: 8,
            d_r: 6,
            blok: 2,
        }
    }

    fn girdiler(t: usize, d: usize, tohum: u64) -> Vec<f64> {
        let mut sayac = tohum
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (0..t * d)
            .map(|_| {
                sayac = sayac
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((sayac >> 33) as f64 / (1u64 << 31) as f64) - 0.5
            })
            .collect()
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Alan {
        W1,
        W2,
        W3,
        B1,
        B2,
        B3,
    }

    const ALANLAR: [(&str, Alan); 6] = [
        ("w1", Alan::W1),
        ("w2", Alan::W2),
        ("w3", Alan::W3),
        ("b1", Alan::B1),
        ("b2", Alan::B2),
        ("b3", Alan::B3),
    ];

    fn alan(a: &HadamardAgirliklar, secim: Alan) -> &Vec<f64> {
        match secim {
            Alan::W1 => &a.w1,
            Alan::W2 => &a.w2,
            Alan::W3 => &a.w3,
            Alan::B1 => &a.b1,
            Alan::B2 => &a.b2,
            Alan::B3 => &a.b3,
        }
    }

    fn alan_mut(a: &mut HadamardAgirliklar, secim: Alan) -> &mut Vec<f64> {
        match secim {
            Alan::W1 => &mut a.w1,
            Alan::W2 => &mut a.w2,
            Alan::W3 => &mut a.w3,
            Alan::B1 => &mut a.b1,
            Alan::B2 => &mut a.b2,
            Alan::B3 => &mut a.b3,
        }
    }

    #[test]
    fn sekil_dogrulanir() {
        assert_eq!(kucuk().dogrula(), Ok(()));
        assert_eq!(
            HadamardSpec {
                d_model: 8,
                d_r: 6,
                blok: 0
            }
            .dogrula(),
            Err(HadamardSekilHatasi::BlokAraligi)
        );
        assert_eq!(
            HadamardSpec {
                d_model: 8,
                d_r: 6,
                blok: 3
            }
            .dogrula(),
            Err(HadamardSekilHatasi::GenislikBolmuyor)
        );
        assert_eq!(
            HadamardSpec {
                d_model: 9,
                d_r: 6,
                blok: 1
            }
            .dogrula(),
            Ok(())
        );
        assert_eq!(
            HadamardSpec {
                d_model: 9,
                d_r: 6,
                blok: 2
            }
            .dogrula(),
            Err(HadamardSekilHatasi::GenislikBolmuyor)
        );
        assert_eq!(
            HadamardSpec {
                d_model: 8,
                d_r: 7,
                blok: 2
            }
            .dogrula(),
            Err(HadamardSekilHatasi::IcGenislikBolmuyor)
        );
        assert_eq!(
            HadamardSpec {
                d_model: 0,
                d_r: 6,
                blok: 1
            }
            .dogrula(),
            Err(HadamardSekilHatasi::BosBoyut)
        );
        assert_eq!(
            HadamardSpec {
                d_model: 8,
                d_r: 6,
                blok: BLOK_UST_SINIRI + 1
            }
            .dogrula(),
            Err(HadamardSekilHatasi::BlokAraligi)
        );
    }

    #[test]
    fn parametre_sayisi_sekilden_turetilir() {
        let spec = HadamardSpec::lubot_a1_adayi();
        assert_eq!(spec.dogrula(), Ok(()));
        let beklenen = 2 * spec.blok * spec.blok_girisi() * spec.blok_ici()
            + spec.d_r * spec.d_model
            + 2 * spec.d_r
            + spec.d_model;
        assert_eq!(spec.parametre_sayisi(), beklenen);
        // Bloksuz hal daha pahali: ayni ic genislikte ek bir dusus izdusumu tasir.
        assert!(spec.parametre_sayisi() < spec.bloksuz_parametre_sayisi());
        // Tasarruf tam olarak bloklu izdusumlerden gelir: 2*d*d_r*(g-1)/g.
        let fark = spec.bloksuz_parametre_sayisi() - spec.parametre_sayisi();
        let beklenen_fark = 2 * spec.d_model * spec.d_r * (spec.blok - 1) / spec.blok;
        assert_eq!(fark, beklenen_fark);
    }

    #[test]
    fn bloklu_hal_standart_mlpyi_yalniz_blokla_gecer() {
        // Ayni ic genislikte: bloksuz Hadamard standart MLP'den pahali (dusus
        // izdusumu yuzunden), dort bloklu hali ondan ucuz. Iki gercek de yazili.
        let d = 64;
        let ic = 256;
        let standart = HadamardSpec::standart_mlp_parametre_sayisi(d, ic);
        let bloksuz = HadamardSpec {
            d_model: d,
            d_r: ic,
            blok: 1,
        }
        .parametre_sayisi();
        let bloklu = HadamardSpec {
            d_model: d,
            d_r: ic,
            blok: 4,
        }
        .parametre_sayisi();
        assert!(bloksuz > standart, "bloksuz {bloksuz} standart {standart}");
        assert!(bloklu < standart, "bloklu {bloklu} standart {standart}");
    }

    #[test]
    fn hadamard_yapisi_gercekten_carpim() {
        let spec = kucuk();
        let a = belirgin_doldur(spec, 3);
        let t = 2;
        let x = girdiler(t, spec.d_model, 5);
        let (cikti, bellek) = ileri(spec, &a, &x, t);
        assert_eq!(cikti.len(), t * spec.d_model);
        assert_eq!(bellek.h.len(), t * spec.d_r);
        // Ikinci dali sifirla: gelu(0) = 0 oldugu icin carpim sifirlanir ve
        // cikti yalniz yanlilik kalir. Standart MLP'de bu olmazdi.
        let mut a2 = a.clone();
        for v in &mut a2.w2 {
            *v = 0.0;
        }
        for v in &mut a2.b2 {
            *v = 0.0;
        }
        let (cikti2, bellek2) = ileri(spec, &a2, &x, t);
        for h in &bellek2.h {
            // Isaretli sifir da sifirdir: bir dalin isaretine gore -0.0 gelebilir.
            assert_eq!(h.abs().to_bits(), 0.0f64.to_bits(), "carpim sifirlanmadi");
        }
        for (i, deger) in cikti2.iter().enumerate() {
            assert_eq!(deger.to_bits(), a.b3[i % spec.d_model].to_bits());
        }
        assert_ne!(cikti, cikti2);
        // Yanlilik sifir degilse carpim yeniden gorunur olur: h = u * gelu(b2).
        let mut a3 = a.clone();
        for v in &mut a3.w2 {
            *v = 0.0;
        }
        let (_, bellek3) = ileri(spec, &a3, &x, t);
        let gorunur = bellek3.h.iter().any(|h| h.to_bits() != 0.0f64.to_bits());
        assert!(gorunur, "yanlilikla bile carpim gorunmedi");
    }

    #[test]
    fn blok_kosegenligi_bayt_duzeyinde_olculur() {
        let spec = kucuk();
        assert_eq!(spec.blok, 2);
        let a = belirgin_doldur(spec, 7);
        let t = 2;
        let x = girdiler(t, spec.d_model, 11);
        let (_, b1) = ileri(spec, &a, &x, t);
        // Yalniz birinci girdi blogunu degistir.
        let mut x2 = x.clone();
        for i in 0..t {
            for k in 0..spec.blok_girisi() {
                x2[i * spec.d_model + k] += 0.75;
            }
        }
        let (_, b2) = ileri(spec, &a, &x2, t);
        // Ikinci blogun ic temsilleri bit duzeyinde ayni kalmali.
        for i in 0..t {
            for c in spec.blok_ici()..spec.d_r {
                let idx = i * spec.d_r + c;
                assert_eq!(
                    b1.u[idx].to_bits(),
                    b2.u[idx].to_bits(),
                    "u[{idx}] komsu bloktan etkilendi"
                );
                assert_eq!(
                    b1.v[idx].to_bits(),
                    b2.v[idx].to_bits(),
                    "v[{idx}] komsu bloktan etkilendi"
                );
            }
        }
        // Ve birinci blok gercekten degisti (test bosa kosmuyor).
        let degisti = (0..t * spec.d_r)
            .filter(|i| i % spec.d_r < spec.blok_ici())
            .any(|i| b1.u[i].to_bits() != b2.u[i].to_bits());
        assert!(degisti, "girdi degisti ama birinci blok ayni kaldi");
    }

    #[test]
    fn ayni_girdi_ayni_baytlari_verir() {
        let spec = kucuk();
        let a = belirgin_doldur(spec, 13);
        let x = girdiler(3, spec.d_model, 17);
        let (c1, b1) = ileri(spec, &a, &x, 3);
        let (c2, b2) = ileri(spec, &a, &x, 3);
        assert_eq!(c1, c2);
        assert_eq!(b1.h, b2.h);
    }

    #[test]
    fn gelu_turevi_kodlanan_fonksiyonun_turevi() -> Result<(), String> {
        let h = 1e-6f64;
        for z in [-3.0f64, -1.0, -0.25, 0.0, 0.4, 1.5, 3.0] {
            let sayisal = (gelu(z + h) - gelu(z - h)) / (2.0 * h);
            let analitik = gelu_turev(z);
            let fark = (sayisal - analitik).abs();
            if fark > 1e-6 {
                return Err(format!(
                    "z={z}: analitik {analitik:.9e} sayisal {sayisal:.9e}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn gradyan_sonlu_farkla_uyusur() -> Result<(), String> {
        let spec = kucuk();
        spec.dogrula()
            .map_err(|e| format!("sekil reddedildi: {e:?}"))?;
        let a = belirgin_doldur(spec, 7);
        let t = 3;
        let x = girdiler(t, spec.d_model, 31);
        let hedef = girdiler(t, spec.d_model, 37);

        let (cikti, bellek) = ileri(spec, &a, &x, t);
        let g = geri(spec, &a, &bellek, &kayip_gradyan(&cikti, &hedef));

        let h = 1e-5f64;
        let mut denetlenen = 0usize;
        let mut ihlaller: Vec<String> = Vec::new();
        let mut en_kotu = 0.0f64;
        let mut en_kotu_ad = String::new();
        for (ad, secim) in ALANLAR {
            let analitik = match secim {
                Alan::W1 => g.w1.clone(),
                Alan::W2 => g.w2.clone(),
                Alan::W3 => g.w3.clone(),
                Alan::B1 => g.b1.clone(),
                Alan::B2 => g.b2.clone(),
                Alan::B3 => g.b3.clone(),
            };
            let mut deneme = a.clone();
            for (i, analitik_deger) in analitik.iter().enumerate() {
                let asil = alan(&a, secim)[i];
                alan_mut(&mut deneme, secim)[i] = asil + h;
                let (c_art, _) = ileri(spec, &deneme, &x, t);
                alan_mut(&mut deneme, secim)[i] = asil - h;
                let (c_eks, _) = ileri(spec, &deneme, &x, t);
                alan_mut(&mut deneme, secim)[i] = asil;
                let sayisal = (kayip(&c_art, &hedef) - kayip(&c_eks, &hedef)) / (2.0 * h);
                let fark = (analitik_deger - sayisal).abs();
                let sinir = GRADIENT_CHECK_MUTLAK_TABAN
                    + GRADIENT_CHECK_TOLERANCE * analitik_deger.abs().max(sayisal.abs());
                denetlenen += 1;
                if fark > sinir {
                    ihlaller.push(format!(
                        "{ad}[{i}] analitik={analitik_deger:.6e} sonlu_fark={sayisal:.6e}"
                    ));
                }
                let olcek = analitik_deger.abs().max(sayisal.abs());
                let bagil = if olcek > 0.0 { fark / olcek } else { fark };
                if bagil > en_kotu {
                    en_kotu = bagil;
                    en_kotu_ad = format!("{ad}[{i}]");
                }
            }
        }
        // Sayim bagi: yeni bir agirlik tensoru denetime girmeden eklenirse duser.
        if denetlenen != spec.parametre_sayisi() {
            return Err(format!(
                "{denetlenen} gradyan denetlendi ama sekil {} parametre sayiyor",
                spec.parametre_sayisi()
            ));
        }
        if !ihlaller.is_empty() {
            return Err(format!(
                "{} / {denetlenen} gradyan anlasmiyor: {}",
                ihlaller.len(),
                ihlaller.join("; ")
            ));
        }
        eprintln!("en kotu bagil sapma {en_kotu_ad} icin {en_kotu:.3e} ({denetlenen} parametre)");
        Ok(())
    }

    #[test]
    fn girdi_gradyani_da_sonlu_farkla_uyusur() -> Result<(), String> {
        let spec = kucuk();
        let a = belirgin_doldur(spec, 11);
        let t = 2;
        let x = girdiler(t, spec.d_model, 41);
        let hedef = girdiler(t, spec.d_model, 43);
        let (cikti, bellek) = ileri(spec, &a, &x, t);
        let g = geri(spec, &a, &bellek, &kayip_gradyan(&cikti, &hedef));
        let h = 1e-5f64;
        for i in 0..x.len() {
            let mut x_art = x.clone();
            x_art[i] += h;
            let (c_art, _) = ileri(spec, &a, &x_art, t);
            let mut x_eks = x.clone();
            x_eks[i] -= h;
            let (c_eks, _) = ileri(spec, &a, &x_eks, t);
            let sayisal = (kayip(&c_art, &hedef) - kayip(&c_eks, &hedef)) / (2.0 * h);
            let analitik = g.girdi[i];
            let fark = (analitik - sayisal).abs();
            if fark
                > GRADIENT_CHECK_MUTLAK_TABAN
                    + GRADIENT_CHECK_TOLERANCE * analitik.abs().max(sayisal.abs())
            {
                return Err(format!(
                    "girdi[{i}] analitik={analitik:.6e} sonlu_fark={sayisal:.6e}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn inis_olculur() -> Result<(), String> {
        // Gercek bir kosu: ayni sekil, sabit hedef, egim inisi. "Ogreniyor"
        // denmez; olculen sey kaybin dustugu ve dustugunun kayda gectigidir.
        let spec = kucuk();
        let mut a = belirgin_doldur(spec, 23);
        let t = 3;
        let x = girdiler(t, spec.d_model, 53);
        let hedef = girdiler(t, spec.d_model, 59);
        let (ilk, _) = ileri(spec, &a, &x, t);
        let kayip0 = kayip(&ilk, &hedef);
        let mut onceki = kayip0;
        let adim = 40;
        let lr = 0.05;
        for s in 0..adim {
            let (cikti, bellek) = ileri(spec, &a, &x, t);
            let g = geri(spec, &a, &bellek, &kayip_gradyan(&cikti, &hedef));
            sgd_adimi(&mut a, &g, lr);
            let (yeni, _) = ileri(spec, &a, &x, t);
            let k = kayip(&yeni, &hedef);
            if k >= onceki {
                return Err(format!("adim {s}: kayip dusmedi ({onceki:.8} -> {k:.8})"));
            }
            onceki = k;
        }
        eprintln!("inis: {kayip0:.8} -> {onceki:.8} ({adim} adim, lr {lr})");
        if onceki >= kayip0 {
            return Err("kayip hic dusmedi".to_string());
        }
        Ok(())
    }

    #[test]
    fn blok_sayisi_ic_temsili_degistirir() {
        // Ayni tohum, ayni girdi, farkli blok sayisi: cikti ayni olamaz, yoksa
        // blok parametresi sessizce yok sayiliyor demektir.
        let a_spec = kucuk();
        let b_spec = HadamardSpec { blok: 1, ..a_spec };
        let a = belirgin_doldur(a_spec, 29);
        let b = belirgin_doldur(b_spec, 29);
        let x = girdiler(2, a_spec.d_model, 61);
        let (ca, _) = ileri(a_spec, &a, &x, 2);
        let (cb, _) = ileri(b_spec, &b, &x, 2);
        assert_ne!(ca, cb);
    }
}
