//! Cok-seritli artik baglantilar (hyper-connections) adayi — bilesen 5.
//!
//! Uygulama direktifinin mimari bilesen listesindeki dorduncu bilesen (tasarim
//! notu 3.4): tek artık akisi yerine `k` paralel serit. Her katman, serit
//! durumundan ogrenilmis bir agirlikli toplamla **girdi** alir ve cikti, ogrenilmis
//! bir karisim matrisi ile seritlere **geri dagitilir**:
//!
//! ```text
//! x            = Σ_l alfa_l · S_l                    (okuma: serit -> katman girdisi)
//! y            = katman(x)                           (cagiranin katmani)
//! S'_l         = Σ_m karisim[l][m] · S_m + beta_l · y (yazma: katman ciktisi -> serit)
//! ```
//!
//! `karisim` **birim matris**, `beta` **sifir**, `alfa = (1, 0, …, 0)` ile
//! baslar: taze bir serit kumesi **no-op**'tur. Ayni disiplin deponun baska
//! adaylarinda da var (kivrim dokunuslari ve qk-norm identity ile baslar);
//! sebebi ayni: yeni bir mekanizma, acilirken var olan agin sayisini
//! degistirmemelidir, yoksa "iyilesme" mekanizmadan mi geldi olculemez.
//!
//! # Neden bu modul ayri duruyor
//!
//! Bu bir **mimari degisiklik adayidir, uygulanmis bir mimari degil**: spec'i
//! (`training/model_spec.json`) degistirmez ve egitim dongusune baglanmaz.
//! Tasarim notu 3.4'un bagladigi acik bulgu (μP turlarinda ileri gecis RMS
//! profilinin Θ(1) bandindan buyumesi, θ₁) icin **olcum onerir**: `rms_profili`
//! ayni yiginin serit=1 ve serit=2 hallerini yan yana olcer; hangi sayi cikarsa
//! o yazilir, hicbir sonuc varsayilmaz. Serit sayisi isaretli karardir (M3) ve
//! olcumden once verilmez.
//!
//! # Olculen seyler, iddia edilen hicbir sey
//!
//! 1. **Gradyan.** Elle yazilmis geri gecis, deponun kendi toleranslariyla
//!    (`GRADIENT_CHECK_*`) merkezi sonlu farka karsi denetlenir; denetlenen
//!    gradyan sayisi seklin parametre sayisina esit olmak zorundadir. Denetim
//!    iki parcadan olusan tek bir kayip uzerinden yapilir: hem okuma yolunun
//!    (`girdi_gradyani`) hem yazma yolunun (`durum_gradyani`) gradyani ayni
//!    olcumde gorunur.
//! 2. **Rms profili.** Yukaridaki θ₁ sorusu icin tek olcum: derinlige gore
//!    katman girdisi RMS'i, serit=1 ve serit=2 icin. Ayni tohum ayni sayilari
//!    verir (tekrarlanabilirlik testi).
//! 3. **Inis.** Gercek bir kosuda (40 adim, SGD) kaybin dustugu olculur.
//!
//! Parametre muhasebesi sekilden turetilir ve tam sayi aritmetigidir: `k` serit
//! `k + k² + k` parametre tasir; tek seritli akis 3 tasir. "Daha ucuz" diye bir
//! iddia yok — serit sayisi arttikca maliyet **artar**, ve bu testte yazilidir.

/// Ust sinir: serit sayisi bunu asarsa sekil reddedilir. Daha buyuk bir deger
/// spec kararinda tartisilir; modul kendi basina tavani buyutmez.
pub const SERIT_UST_SINIRI: usize = 8;

/// Cok-seritli artik baglanti sekli.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CokSeritSpec {
    /// Model genisligi (serit satirlarinin uzunlugu).
    pub d_model: usize,
    /// Serit sayisi (`1` = tek akis, klasik artık baglanti).
    pub serit: usize,
}

/// Neden bir sekil reddedildi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeritSekilHatasi {
    /// Genislik sifir olamaz.
    SifirGenislik,
    /// Serit sayisi sifir olamaz.
    SifirSerit,
    /// Serit sayisi `SERIT_UST_SINIRI`'ni asiyor.
    SeritUstSinirAsildi,
}

impl CokSeritSpec {
    /// Sekli dogrular.
    pub fn yeni(d_model: usize, serit: usize) -> Result<Self, SeritSekilHatasi> {
        if d_model == 0 {
            return Err(SeritSekilHatasi::SifirGenislik);
        }
        if serit == 0 {
            return Err(SeritSekilHatasi::SifirSerit);
        }
        if serit > SERIT_UST_SINIRI {
            return Err(SeritSekilHatasi::SeritUstSinirAsildi);
        }
        Ok(Self { d_model, serit })
    }

    /// Sekilden turetilen parametre sayisi: `alfa (k) + karisim (k²) + beta (k)`.
    #[must_use]
    pub fn parametre_sayisi(&self) -> usize {
        self.serit + self.serit * self.serit + self.serit
    }

    /// Karsilastirma: tek seritli klasik artık akisi `k = 1` halidir ve
    /// `1 + 1 + 1 = 3` parametre tasir.
    #[must_use]
    pub fn tek_serit_parametre_sayisi(&self) -> usize {
        3
    }
}

/// Okuma agirliklari, karisim matrisi ve yazma olcekleri.
#[derive(Debug, Clone, PartialEq)]
pub struct SeritAgirliklar {
    /// Okuma agirliklari (`serit`).
    pub alfa: Vec<f64>,
    /// Karisim matrisi, satir-baskin (`serit x serit`).
    pub karisim: Vec<f64>,
    /// Katman ciktisinin seritlere dagitim olcekleri (`serit`).
    pub beta: Vec<f64>,
}

/// Serit durumu: `serit x d_model`, satir-baskin.
#[derive(Debug, Clone, PartialEq)]
pub struct SeritDurumu {
    /// Duz durum vektoru (`serit * d_model`).
    pub satirlar: Vec<f64>,
}

impl SeritDurumu {
    /// Sifir durum.
    #[must_use]
    pub fn sifir(spec: CokSeritSpec) -> Self {
        Self {
            satirlar: vec![0.0; spec.serit * spec.d_model],
        }
    }

    /// `l`. seritin dilimi; aralik disindaysa bos dilim doner (panik yolu yok).
    #[must_use]
    pub fn satir(&self, spec: CokSeritSpec, l: usize) -> &[f64] {
        self.satirlar.chunks(spec.d_model).nth(l).unwrap_or(&[])
    }
}

/// Kimlik baslangic: taze serit kumesi no-op'tur (bkz. modul basligi).
#[must_use]
pub fn kimlik_doldur(spec: CokSeritSpec) -> SeritAgirliklar {
    let mut alfa = vec![0.0; spec.serit];
    if let Some(ilk) = alfa.first_mut() {
        *ilk = 1.0;
    }
    let mut karisim = vec![0.0; spec.serit * spec.serit];
    for l in 0..spec.serit {
        karisim[l * spec.serit + l] = 1.0;
    }
    SeritAgirliklar {
        alfa,
        karisim,
        beta: vec![0.0; spec.serit],
    }
}

/// Tohumlu doldurma: ayni tohum ayni agirliklari verir (dogrusal eslemeli uretici).
#[must_use]
pub fn tohumlu_doldur(spec: CokSeritSpec, tohum: u64) -> SeritAgirliklar {
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
    let alfa = uret(spec.serit);
    let karisim = uret(spec.serit * spec.serit);
    let beta = uret(spec.serit);
    SeritAgirliklar {
        alfa,
        karisim,
        beta,
    }
}

/// Ileri gecis: katman girdisi ve yeni serit durumu.
///
/// `katman_cikti` cagiranin katmaninin urettigi `y`'dir (`d_model` uzunlugunda;
/// eksik eleman sifir sayilir, panik yolu yok).
#[must_use]
fn okuma(spec: &CokSeritSpec, w: &SeritAgirliklar, durum: &SeritDurumu) -> Vec<f64> {
    let mut girdi = vec![0.0f64; spec.d_model];
    for l in 0..spec.serit {
        let agirlik = w.alfa.get(l).copied().unwrap_or(0.0);
        let satir = durum.satir(*spec, l);
        for (x, s) in girdi.iter_mut().zip(satir.iter()) {
            *x += agirlik * *s;
        }
    }
    girdi
}

#[must_use]
pub fn ileri(
    spec: &CokSeritSpec,
    w: &SeritAgirliklar,
    durum: &SeritDurumu,
    katman_cikti: &[f64],
) -> (Vec<f64>, SeritDurumu) {
    let d = spec.d_model;
    let k = spec.serit;
    let girdi = okuma(spec, w, durum);
    let mut yeni = vec![0.0f64; k * d];
    for l in 0..k {
        let beta = w.beta.get(l).copied().unwrap_or(0.0);
        for m in 0..k {
            let katsayi = w.karisim.get(l * k + m).copied().unwrap_or(0.0);
            if katsayi == 0.0 {
                continue;
            }
            let satir = durum.satir(*spec, m);
            for (d_i, s) in satir.iter().enumerate() {
                if let Some(hedef) = yeni.get_mut(l * d + d_i) {
                    *hedef += katsayi * *s;
                }
            }
        }
        for (d_i, y) in katman_cikti.iter().enumerate() {
            if let Some(hedef) = yeni.get_mut(l * d + d_i) {
                *hedef += beta * *y;
            }
        }
    }
    (girdi, SeritDurumu { satirlar: yeni })
}

/// Geri gecis gradyanlari.
#[derive(Debug, Clone, PartialEq)]
pub struct SeritGradyanlar {
    /// `dL/dalfa`.
    pub alfa: Vec<f64>,
    /// `dL/dkarisim`, satir-baskin.
    pub karisim: Vec<f64>,
    /// `dL/dbeta`.
    pub beta: Vec<f64>,
    /// `dL/dS` (girdi durumuna gore), satir-baskin.
    pub durum: Vec<f64>,
}

/// Elle yazilmis geri gecis.
///
/// Turevler (satir satir):
/// ```text
/// dL/dalfa[m]      = Σ_d girdi_gradyani[d] · S_m[d]
/// dL/dkarisim[l][m] = Σ_d durum_gradyani[l][d] · S_m[d]
/// dL/dbeta[l]      = Σ_d durum_gradyani[l][d] · y[d]
/// dL/dS_m[d]       = Σ_l durum_gradyani[l][d] · karisim[l][m] + girdi_gradyani[d] · alfa_l[m]
/// ```
#[must_use]
pub fn geri(
    spec: &CokSeritSpec,
    w: &SeritAgirliklar,
    durum: &SeritDurumu,
    katman_cikti: &[f64],
    girdi_gradyani: &[f64],
    durum_gradyani: &[f64],
) -> SeritGradyanlar {
    let d = spec.d_model;
    let k = spec.serit;
    let mut alfa = vec![0.0f64; k];
    let mut karisim = vec![0.0f64; k * k];
    let mut beta = vec![0.0f64; k];
    let mut durum_grad = vec![0.0f64; k * d];
    for m in 0..k {
        let satir = durum.satir(*spec, m);
        for (d_i, s) in satir.iter().enumerate() {
            let gx = girdi_gradyani.get(d_i).copied().unwrap_or(0.0);
            if let Some(hedef) = alfa.get_mut(m) {
                *hedef += gx * *s;
            }
        }
    }
    for l in 0..k {
        let g_satir_bas = l * d;
        for m in 0..k {
            let satir = durum.satir(*spec, m);
            let mut toplam = 0.0f64;
            for (d_i, s) in satir.iter().enumerate() {
                let g = durum_gradyani
                    .get(g_satir_bas + d_i)
                    .copied()
                    .unwrap_or(0.0);
                toplam += g * *s;
            }
            if let Some(hedef) = karisim.get_mut(l * k + m) {
                *hedef += toplam;
            }
        }
        for d_i in 0..d {
            let g = durum_gradyani
                .get(g_satir_bas + d_i)
                .copied()
                .unwrap_or(0.0);
            let y = katman_cikti.get(d_i).copied().unwrap_or(0.0);
            if let Some(hedef) = beta.get_mut(l) {
                *hedef += g * y;
            }
        }
    }
    for m in 0..k {
        let alfa_m = w.alfa.get(m).copied().unwrap_or(0.0);
        for d_i in 0..d {
            let gx = girdi_gradyani.get(d_i).copied().unwrap_or(0.0);
            let mut toplam = gx * alfa_m;
            for l in 0..k {
                let katsayi = w.karisim.get(l * k + m).copied().unwrap_or(0.0);
                let g = durum_gradyani.get(l * d + d_i).copied().unwrap_or(0.0);
                toplam += katsayi * g;
            }
            if let Some(hedef) = durum_grad.get_mut(m * d + d_i) {
                *hedef += toplam;
            }
        }
    }
    SeritGradyanlar {
        alfa,
        karisim,
        beta,
        durum: durum_grad,
    }
}

/// Katman girdisinin RMS'i: `sqrt(Σ x² / d)`. Sifir genislikte `0.0`.
#[must_use]
pub fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let toplam: f64 = x.iter().map(|v| v * v).sum();
    (toplam / x.len() as f64).sqrt()
}

/// Derinlige gore katman girdisi RMS profili (θ₁ sorusunun tek olcumu).
///
/// `katman`, cagiranin katmanidir: girdiyi alir, ciktiyi verir. Profilin `i`.
/// elemani `i`. katmanin **girdisinin** RMS'idir (yani katmanin ne gordugu);
/// boylece "derinlestikce girdi ne kadar buyuyor" sorusu dogrudan okunur.
#[must_use]
pub fn rms_profili<F>(
    spec: &CokSeritSpec,
    w: &SeritAgirliklar,
    baslangic: &SeritDurumu,
    katman: F,
    derinlik: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut durum = baslangic.clone();
    let mut profil = Vec::with_capacity(derinlik);
    for _ in 0..derinlik {
        let girdi = okuma(spec, w, &durum);
        profil.push(rms(&girdi));
        let cikti = katman(&girdi);
        let (_, sonraki) = ileri(spec, w, &durum, &cikti);
        durum = sonraki;
    }
    profil
}

/// Ortalama kare hata (kendi olcumumuz icin; egitim cekirdeginin kaybi model
/// cikisina bagli oldugundan burada yeniden yazilmaz, bu modul kendi olceginde
/// kalir).
#[must_use]
pub fn kayip(cikti: &[f64], hedef: &[f64]) -> f64 {
    if cikti.is_empty() {
        return 0.0;
    }
    let toplam: f64 = cikti
        .iter()
        .zip(hedef.iter())
        .map(|(c, h)| (c - h) * (c - h))
        .sum();
    toplam / cikti.len() as f64
}

/// `kayip`'in ciktiya gore gradyani.
#[must_use]
pub fn kayip_gradyan(cikti: &[f64], hedef: &[f64]) -> Vec<f64> {
    if cikti.is_empty() {
        return Vec::new();
    }
    let olcek = 2.0 / cikti.len() as f64;
    cikti
        .iter()
        .zip(hedef.iter())
        .map(|(c, h)| olcek * (c - h))
        .collect()
}

/// Yerinde SGD adimi.
pub fn sgd_adimi(w: &mut SeritAgirliklar, g: &SeritGradyanlar, lr: f64) {
    for (p, gr) in w.alfa.iter_mut().zip(g.alfa.iter()) {
        *p -= lr * *gr;
    }
    for (p, gr) in w.karisim.iter_mut().zip(g.karisim.iter()) {
        *p -= lr * *gr;
    }
    for (p, gr) in w.beta.iter_mut().zip(g.beta.iter()) {
        *p -= lr * *gr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GRADIENT_CHECK_MUTLAK_TABAN, GRADIENT_CHECK_TOLERANCE};

    fn hedef_vektor(n: usize) -> Vec<f64> {
        (0..n).map(|i| 0.2 * (i as f64) - 0.1).collect()
    }

    fn sabit_gradyan(n: usize) -> Vec<f64> {
        (0..n).map(|i| 0.05 - 0.01 * (i as f64)).collect()
    }

    fn durum_doldur(spec: CokSeritSpec, tohum: u64) -> SeritDurumu {
        let mut sayac = tohum
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let satirlar = (0..spec.serit * spec.d_model)
            .map(|_| {
                sayac = sayac
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((sayac >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.75
            })
            .collect();
        SeritDurumu { satirlar }
    }

    /// Tek kayip: okuma yolundan (hedefe uzaklik) + yazma yolundan (sabit
    /// gradyan) ayni anda. Boylece geri gecisin iki yarisi da denetlenir.
    fn toplam_kayip(
        spec: &CokSeritSpec,
        w: &SeritAgirliklar,
        durum: &SeritDurumu,
        katman_cikti: &[f64],
        hedef: &[f64],
        sabit: &[f64],
    ) -> f64 {
        let (girdi, yeni) = ileri(spec, w, durum, katman_cikti);
        let mut toplam = kayip(&girdi, hedef);
        for (s, c) in yeni.satirlar.iter().zip(sabit.iter()) {
            toplam += s * c;
        }
        toplam
    }

    #[allow(clippy::too_many_arguments)]
    fn sonlu_fark(
        spec: &CokSeritSpec,
        w: &SeritAgirliklar,
        durum: &SeritDurumu,
        katman_cikti: &[f64],
        hedef: &[f64],
        sabit: &[f64],
        degistir: impl Fn(&mut SeritAgirliklar, usize, f64),
        indeks: usize,
    ) -> f64 {
        let h = 1e-6;
        let mut arti = w.clone();
        degistir(&mut arti, indeks, h);
        let mut eksi = w.clone();
        degistir(&mut eksi, indeks, -h);
        (toplam_kayip(spec, &arti, durum, katman_cikti, hedef, sabit)
            - toplam_kayip(spec, &eksi, durum, katman_cikti, hedef, sabit))
            / (2.0 * h)
    }

    #[test]
    fn sekil_reddi_dogru_hata_verir() {
        assert_eq!(
            CokSeritSpec::yeni(0, 2),
            Err(SeritSekilHatasi::SifirGenislik)
        );
        assert_eq!(CokSeritSpec::yeni(8, 0), Err(SeritSekilHatasi::SifirSerit));
        assert_eq!(
            CokSeritSpec::yeni(8, SERIT_UST_SINIRI + 1),
            Err(SeritSekilHatasi::SeritUstSinirAsildi)
        );
        assert!(CokSeritSpec::yeni(8, SERIT_UST_SINIRI).is_ok());
    }

    #[test]
    fn parametre_sayisi_sekilden_turetilir() {
        let bir = CokSeritSpec::yeni(16, 1).expect("gecerli sekil");
        let iki = CokSeritSpec::yeni(16, 2).expect("gecerli sekil");
        let uc = CokSeritSpec::yeni(16, 3).expect("gecerli sekil");
        // Tam sayi aritmetigi: k + k² + k.
        assert_eq!(bir.parametre_sayisi(), 3);
        assert_eq!(iki.parametre_sayisi(), 8);
        assert_eq!(uc.parametre_sayisi(), 15);
        // Tek serit klasik artık akistir ve karsilastirma sabittir.
        assert_eq!(bir.tek_serit_parametre_sayisi(), 3);
        assert_eq!(bir.parametre_sayisi(), bir.tek_serit_parametre_sayisi());
        // Serit sayisi arttikca maliyet artar - "daha ucuz" iddiasi yok.
        assert!(iki.parametre_sayisi() > bir.parametre_sayisi());
        assert!(uc.parametre_sayisi() > iki.parametre_sayisi());
    }

    #[test]
    fn kimlik_baslangic_girdiyi_degistirmez() {
        let spec = CokSeritSpec::yeni(6, 1).expect("gecerli sekil");
        let w = kimlik_doldur(spec);
        let durum = durum_doldur(spec, 11);
        let y = hedef_vektor(spec.d_model);
        let (girdi, yeni) = ileri(&spec, &w, &durum, &y);
        // Tek seritte okuma, durumun kendisidir: bit duzeyinde.
        for (a, b) in girdi.iter().zip(durum.satirlar.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "okuma durumu degistirdi");
        }
        // beta sifir oldugu icin katman ciktisi duruma hic girmez.
        for (a, b) in yeni.satirlar.iter().zip(durum.satirlar.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "kimlikte durum degisti");
        }
    }

    #[test]
    fn kimlik_baslangic_serit_sayisindan_bagimsiz() {
        let y = hedef_vektor(6);
        for k in [2usize, 4, 8] {
            let spec = CokSeritSpec::yeni(6, k).expect("gecerli sekil");
            let w = kimlik_doldur(spec);
            let durum = durum_doldur(spec, 3 + k as u64);
            let (girdi, yeni) = ileri(&spec, &w, &durum, &y);
            // Kimlikte okuma yalnizca 0. seriti gorur.
            for (a, b) in girdi.iter().zip(durum.satir(spec, 0).iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "k={k}: okuma 0. serit degil");
            }
            // Ve hicbir serit hareket etmez: taze serit kumesi no-op.
            for (a, b) in yeni.satirlar.iter().zip(durum.satirlar.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "k={k}: durum degisti");
            }
        }
    }

    #[test]
    fn gradyan_sonlu_farkla_uyusur() {
        let spec = CokSeritSpec::yeni(5, 3).expect("gecerli sekil");
        let w = tohumlu_doldur(spec, 20260926);
        let durum = durum_doldur(spec, 7);
        let y = hedef_vektor(spec.d_model);
        let hedef = hedef_vektor(spec.d_model);
        let sabit = sabit_gradyan(spec.serit * spec.d_model);
        let (girdi, _) = ileri(&spec, &w, &durum, &y);
        let gx = kayip_gradyan(&girdi, &hedef);
        let g = geri(&spec, &w, &durum, &y, &gx, &sabit);
        let mut denetlenen = 0usize;
        let mut en_kotu = 0.0f64;
        for l in 0..spec.serit {
            let analitik = g.alfa[l];
            let sonlu = sonlu_fark(
                &spec,
                &w,
                &durum,
                &y,
                &hedef,
                &sabit,
                |ww, i, h| ww.alfa[i] += h,
                l,
            );
            let sinir = GRADIENT_CHECK_MUTLAK_TABAN
                + GRADIENT_CHECK_TOLERANCE * analitik.abs().max(sonlu.abs());
            assert!(
                (analitik - sonlu).abs() <= sinir,
                "alfa[{l}]: analitik {analitik}, sonlu {sonlu}"
            );
            en_kotu = en_kotu.max((analitik - sonlu).abs());
            denetlenen += 1;
            let analitik = g.beta[l];
            let sonlu = sonlu_fark(
                &spec,
                &w,
                &durum,
                &y,
                &hedef,
                &sabit,
                |ww, i, h| ww.beta[i] += h,
                l,
            );
            let sinir = GRADIENT_CHECK_MUTLAK_TABAN
                + GRADIENT_CHECK_TOLERANCE * analitik.abs().max(sonlu.abs());
            assert!(
                (analitik - sonlu).abs() <= sinir,
                "beta[{l}]: analitik {analitik}, sonlu {sonlu}"
            );
            en_kotu = en_kotu.max((analitik - sonlu).abs());
            denetlenen += 1;
        }
        for l in 0..spec.serit {
            for m in 0..spec.serit {
                let indeks = l * spec.serit + m;
                let analitik = g.karisim[indeks];
                let sonlu = sonlu_fark(
                    &spec,
                    &w,
                    &durum,
                    &y,
                    &hedef,
                    &sabit,
                    |ww, i, h| ww.karisim[i] += h,
                    indeks,
                );
                let sinir = GRADIENT_CHECK_MUTLAK_TABAN
                    + GRADIENT_CHECK_TOLERANCE * analitik.abs().max(sonlu.abs());
                assert!(
                    (analitik - sonlu).abs() <= sinir,
                    "karisim[{l}][{m}]: analitik {analitik}, sonlu {sonlu}"
                );
                en_kotu = en_kotu.max((analitik - sonlu).abs());
                denetlenen += 1;
            }
        }
        // Sayim bagi: denetlenen gradyan sayisi seklin parametre sayisina bagli.
        if denetlenen != spec.parametre_sayisi() {
            assert_eq!(denetlenen, spec.parametre_sayisi(), "denetim sayisi sasti");
        }
        assert!(en_kotu < 1e-6, "en kotu sapma {en_kotu}");
    }

    #[test]
    fn durum_gradyani_da_sonlu_farkla_uyusur() {
        let spec = CokSeritSpec::yeni(4, 2).expect("gecerli sekil");
        let w = tohumlu_doldur(spec, 4242);
        let durum = durum_doldur(spec, 9);
        let y = hedef_vektor(spec.d_model);
        let hedef = hedef_vektor(spec.d_model);
        let sabit = sabit_gradyan(spec.serit * spec.d_model);
        let (girdi, _) = ileri(&spec, &w, &durum, &y);
        let gx = kayip_gradyan(&girdi, &hedef);
        let g = geri(&spec, &w, &durum, &y, &gx, &sabit);
        let h = 1e-6;
        for i in 0..spec.serit * spec.d_model {
            let mut arti = durum.clone();
            arti.satirlar[i] += h;
            let mut eksi = durum.clone();
            eksi.satirlar[i] -= h;
            let sonlu = (toplam_kayip(&spec, &w, &arti, &y, &hedef, &sabit)
                - toplam_kayip(&spec, &w, &eksi, &y, &hedef, &sabit))
                / (2.0 * h);
            let analitik = g.durum[i];
            let sinir = GRADIENT_CHECK_MUTLAK_TABAN
                + GRADIENT_CHECK_TOLERANCE * analitik.abs().max(sonlu.abs());
            assert!(
                (analitik - sonlu).abs() <= sinir,
                "durum[{i}]: analitik {analitik}, sonlu {sonlu}"
            );
        }
    }

    #[test]
    fn serit_karisimi_gercekten_karistirir() {
        let spec = CokSeritSpec::yeni(3, 2).expect("gecerli sekil");
        let durum = durum_doldur(spec, 21);
        let y = hedef_vektor(spec.d_model);
        let mut w = kimlik_doldur(spec);
        let (_, once) = ileri(&spec, &w, &durum, &y);
        w.karisim[1] = 0.5; // karisim[0][1]
        let (_, sonra) = ileri(&spec, &w, &durum, &y);
        let fark: f64 = once
            .satirlar
            .iter()
            .zip(sonra.satirlar.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(fark > 0.0, "karisim matrisi hicbir seyi degistirmedi");
        // karisim[0][1] degisti: 0. serit artik 1. seritten besleniyor, 1. serit
        // bu degisimden etkilenmez (kimlikte satiri yalniz kendini okur).
        let satir0_degisti = once
            .satir(spec, 0)
            .iter()
            .zip(sonra.satir(spec, 0).iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(satir0_degisti, "0. serit karisimdan etkilenmedi");
        let satir1_ayni = once
            .satir(spec, 1)
            .iter()
            .zip(sonra.satir(spec, 1).iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(satir1_ayni, "1. serit karisimdan etkilenmemeliydi");
    }

    #[test]
    fn tohum_tekrarlanabilir() {
        let spec = CokSeritSpec::yeni(4, 3).expect("gecerli sekil");
        let a = tohumlu_doldur(spec, 5);
        let b = tohumlu_doldur(spec, 5);
        let c = tohumlu_doldur(spec, 6);
        assert_eq!(a, b, "ayni tohum farkli agirlik verdi");
        assert_ne!(a, c, "farkli tohum ayni agirligi verdi");
    }

    #[test]
    fn rms_profili_olculur() {
        // Katman: girdiyi 1.15 ile buyuten dogrusal bir harita (kasten > 1).
        let katman = |x: &[f64]| -> Vec<f64> { x.iter().map(|v| 1.15 * v).collect() };
        let derinlik = 8usize;
        let spec1 = CokSeritSpec::yeni(16, 1).expect("gecerli sekil");
        let spec2 = CokSeritSpec::yeni(16, 2).expect("gecerli sekil");
        let d1 = durum_doldur(spec1, 31);
        let d2 = durum_doldur(spec2, 31);
        let w1 = tohumlu_doldur(spec1, 77);
        let w2 = tohumlu_doldur(spec2, 77);
        let p1 = rms_profili(&spec1, &w1, &d1, katman, derinlik);
        let p2 = rms_profili(&spec2, &w2, &d2, katman, derinlik);
        assert_eq!(p1.len(), derinlik);
        assert_eq!(p2.len(), derinlik);
        for (i, v) in p1.iter().chain(p2.iter()).enumerate() {
            assert!(v.is_finite() && *v >= 0.0, "profil[{i}] gecersiz: {v}");
        }
        // Tekrarlanabilirlik: ayni tohum ayni profil.
        let p1b = rms_profili(&spec1, &w1, &d1, katman, derinlik);
        for (a, b) in p1.iter().zip(p1b.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "profil tohumla ayni cikmadi");
        }
        // Kimlik baslangicta serit sayisi profili degistirmez: no-op iddiasi
        // yalniz bu halde gecerlidir (serit sayisinin kendisi bir sey yapmaz).
        let k1 = kimlik_doldur(spec1);
        let k2 = kimlik_doldur(spec2);
        let ik1 = rms_profili(&spec1, &k1, &d1, katman, derinlik);
        let mut d2_tek = durum_doldur(spec2, 31);
        d2_tek.satirlar[spec2.d_model..].fill(0.0);
        let ik2 = rms_profili(&spec2, &k2, &d2_tek, katman, derinlik);
        for (a, b) in ik1.iter().zip(ik2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "kimlikte profiller ayrildi");
        }
    }

    /// Tek olcum satiri: ayni yiginin serit=1 ve serit=2 hallerinin RMS
    /// profili. Sayilar bu satirdan kayda gecer (`training/cok_serit.py`),
    /// boylece kayit ikinci bir uygulamadan degil bu modulun kendisinden gelir.
    #[test]
    fn rms_profili_raporu() {
        let derinlik = 8usize;
        let mut satirlar = Vec::new();
        for kazanc in [1.15f64, 4.0] {
            let katman = move |x: &[f64]| -> Vec<f64> { x.iter().map(|v| kazanc * v).collect() };
            for k in [1usize, 2, 4] {
                let spec = CokSeritSpec::yeni(16, k).expect("gecerli sekil");
                let w = tohumlu_doldur(spec, 77);
                let durum = durum_doldur(spec, 31);
                let profil = rms_profili(&spec, &w, &durum, katman, derinlik);
                let metin: Vec<String> = profil.iter().map(|v| format!("{v:.9e}")).collect();
                satirlar.push(format!("kazanc={kazanc:.2};serit={k}:{}", metin.join(",")));
            }
        }
        println!("cok-serit rms profili | {}", satirlar.join(" | "));
    }

    #[test]
    fn inis_olculur() {
        // Gercek bir kosu: iki serit, sabit bir buyuten katman, MSE hedefi.
        let spec = CokSeritSpec::yeni(6, 2).expect("gecerli sekil");
        let katman = |x: &[f64]| -> Vec<f64> { x.iter().map(|v| 1.05 * v).collect() };
        let hedef = hedef_vektor(spec.d_model);
        let sabit = vec![0.0f64; spec.serit * spec.d_model];
        let mut w = tohumlu_doldur(spec, 99);
        let mut durum = durum_doldur(spec, 12);
        let baslangic = {
            let (girdi, _) = ileri(&spec, &w, &durum, &katman(&[0.0; 6]));
            kayip(&girdi, &hedef)
        };
        for _ in 0..40 {
            let (girdi, _) = ileri(&spec, &w, &durum, &katman(&[0.0; 6]));
            let gx = kayip_gradyan(&girdi, &hedef);
            let g = geri(&spec, &w, &durum, &katman(&[0.0; 6]), &gx, &sabit);
            sgd_adimi(&mut w, &g, 0.05);
        }
        let bitis = {
            let (girdi, _) = ileri(&spec, &w, &durum, &katman(&[0.0; 6]));
            kayip(&girdi, &hedef)
        };
        assert!(bitis < baslangic, "inis yok: {baslangic} -> {bitis}");
        // Durum da ogrenilebilir: onu da bir adim ilerletip kaybin dustugunu gormek
        // yerine, geri gecisin durum gradyanini urettigini dogruluyoruz.
        let (girdi, _) = ileri(&spec, &w, &durum, &katman(&[0.0; 6]));
        let gx = kayip_gradyan(&girdi, &hedef);
        let g = geri(&spec, &w, &durum, &katman(&[0.0; 6]), &gx, &sabit);
        assert_eq!(g.durum.len(), durum.satirlar.len());
        assert!(
            g.durum.iter().any(|v| v.abs() > 0.0),
            "durum gradyani sifir"
        );
        durum.satirlar.iter_mut().enumerate().for_each(|(i, s)| {
            *s -= 0.05 * g.durum[i];
        });
        let sonraki = {
            let (girdi, _) = ileri(&spec, &w, &durum, &katman(&[0.0; 6]));
            kayip(&girdi, &hedef)
        };
        assert!(sonraki <= bitis + 1e-12, "durum adimi kaybi bozdu");
    }

    #[test]
    fn ileri_geri_tutarlidir() {
        // Tek bir serit ve sifir karisim: geri gecis, okumanin agirlik gradyanini
        // yalnizca girdi gradyanindan almali (yazma yolu kapali).
        let spec = CokSeritSpec::yeni(3, 1).expect("gecerli sekil");
        let durum = durum_doldur(spec, 55);
        let mut w = kimlik_doldur(spec);
        w.beta[0] = 0.0;
        let y = hedef_vektor(spec.d_model);
        let gx = sabit_gradyan(spec.d_model);
        let g = geri(&spec, &w, &durum, &y, &gx, &[0.0]);
        let beklenen: f64 = (0..spec.d_model)
            .map(|d_i| gx[d_i] * durum.satirlar[d_i])
            .sum();
        assert_eq!(
            g.alfa[0].to_bits(),
            beklenen.to_bits(),
            "alfa gradyani sasti"
        );
        // alfa = 1 ve karisim = 1 oldugu icin durum gradyani = girdi gradyani.
        for (m, beklenen) in gx.iter().enumerate() {
            assert_eq!(
                g.durum[m].to_bits(),
                beklenen.to_bits(),
                "durum gradyani sasti"
            );
        }
    }
}
