//! Gruplu-sorgu dikkati ve nedensel kivrim dokunuslari.
//!
//! Port bilesen 2 (tasarim notu 3.2). Bugunku dikkat her kafaya kendi K/V'sini
//! veriyor; bu modul sorgu kafalarini gruplayip K/V'yi grup basina bir kez
//! hesapliyor ve sorgu ile anahtara girmeden once kisa bir nedensel kivrim
//! uyguluyor. Iki sey boylece olculur hale geliyor: kac parametre tasarruf
//! edildigi ve kivrimin yerel oruntuyu tasiyip tasimadigi.
//!
//! # Neden bu modul ayri duruyor
//!
//! Bu bir mimari degisiklik adayidir, uygulanmis bir mimari degil: spec'i
//! (`training/model_spec.json`) degistirmez, egitim dongusune baglanmaz. Kendi
//! testleriyle olculur - gradyan sonlu farkla, kayit siniri bayt-esitligiyle,
//! parametre tasarrufu tam sayi aritmetigiyle. Baglanma karari olcumden sonra
//! verilir (tasarim notu M1-M3 ailesi).
//!
//! # Kayit siniri iki yerde de korunur
//!
//! dikkat penceresi baska bir kaydin jetonuna bakamaz (mevcut `dikkat_ileri`
//! kurali). Kivrim dokunuslari da ayni kurala tabidir: bir dokunus, komsu
//! kaydin jetonunu tasiyamaz. Aksi halde kayit 1'in ciktisi kayit 2'nin
//! icerigine bagli olurdu; bu, alintilanamayan bir baglamdir.

/// Ust sinir: dokunus sayisi kucuk kalir, genel kivrim altyapisi gerekmez.
pub const DOKUNUS_UST_SINIRI: usize = 7;

/// Gruplu-sorgu dikkatinin sekli.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrupSpec {
    /// Model genisligi.
    pub d_model: usize,
    /// Sorgu kafasi sayisi.
    pub n_heads: usize,
    /// Anahtar/deger kafasi sayisi; sorgu kafalarini gruplar.
    pub n_kv_heads: usize,
    /// Nedensel kivrim dokunusu sayisi (1 = dokunus yok).
    pub taps: usize,
}

/// Neden bir sekil reddedildi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrupHatasi {
    /// Sifir boyut.
    BosBoyut,
    /// Genislik kafa sayisina bolunmuyor.
    BasSayisiBolmuyor,
    /// Sorgu kafalari K/V kafalarina tam bolunmuyor.
    GrupBolmuyor,
    /// Dokunus sayisi `1..=DOKUNUS_UST_SINIRI` disinda.
    DokunusAraligi,
}

impl GrupSpec {
    /// Ilk olculecek aday: iki sorgu kafasi tek K/V kafasini paylasir, uc dokunus.
    #[must_use]
    pub const fn lubot_a2_adayi() -> Self {
        Self {
            d_model: 64,
            n_heads: 2,
            n_kv_heads: 1,
            taps: 3,
        }
    }

    /// Kafa basina genislik.
    #[must_use]
    pub fn d_k(&self) -> usize {
        if self.n_heads == 0 {
            return 0;
        }
        self.d_model / self.n_heads
    }

    /// K/V toplam genisligi (gruplu tarafta).
    #[must_use]
    pub fn kv_genislik(&self) -> usize {
        self.n_kv_heads * self.d_k()
    }

    /// Bir K/V kafasini kac sorgu kafasi paylasir.
    #[must_use]
    pub fn grup_boyu(&self) -> usize {
        if self.n_kv_heads == 0 {
            return 0;
        }
        self.n_heads / self.n_kv_heads
    }

    /// # Errors
    /// [`GrupHatasi`] - hangi boyutun neden reddedildigini tasir.
    pub fn dogrula(&self) -> Result<(), GrupHatasi> {
        if self.d_model == 0 || self.n_heads == 0 || self.n_kv_heads == 0 {
            return Err(GrupHatasi::BosBoyut);
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(GrupHatasi::BasSayisiBolmuyor);
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(GrupHatasi::GrupBolmuyor);
        }
        if self.taps == 0 || self.taps > DOKUNUS_UST_SINIRI {
            return Err(GrupHatasi::DokunusAraligi);
        }
        Ok(())
    }

    /// Bu seklin parametre sayisi: q izdusumu, gruplu k/v izdusumleri, cikis
    /// izdusumu, yanliliklar ve kanal basina dokunus agirliklari.
    #[must_use]
    pub fn parametre_sayisi(&self) -> usize {
        let d = self.d_model;
        let kv = self.kv_genislik();
        let izdusum = d * d + 2 * kv * d + d * d;
        let yanlilik = d + 2 * kv + d;
        let dokunus = self.taps * d + self.taps * kv;
        izdusum + yanlilik + dokunus
    }

    /// Ayni seklin her kafaya kendi K/V'sini verdigi karsiligi - tasarruf olcusu.
    #[must_use]
    pub fn tam_dikkat_parametre_sayisi(&self) -> usize {
        let tam = Self {
            n_kv_heads: self.n_heads,
            ..*self
        };
        tam.parametre_sayisi()
    }
}

/// Butun agirliklar tek yerde.
#[derive(Debug, Clone, PartialEq)]
pub struct Agirliklar {
    /// Sorgu izdusumu, `d_model x d_model`.
    pub wq: Vec<f64>,
    /// Gruplu anahtar izdusumu, `kv x d_model`.
    pub wk: Vec<f64>,
    /// Gruplu deger izdusumu, `kv x d_model`.
    pub wv: Vec<f64>,
    /// Cikis izdusumu, `d_model x d_model`.
    pub wo: Vec<f64>,
    /// Sorgu yanliligi, `d_model`.
    pub bq: Vec<f64>,
    /// Anahtar yanliligi, `kv`.
    pub bk: Vec<f64>,
    /// Deger yanliligi, `kv`.
    pub bv: Vec<f64>,
    /// Cikis yanliligi, `d_model`.
    pub bo: Vec<f64>,
    /// Sorgu kivrimi, `taps x d_model` (kanal basina).
    pub cq: Vec<f64>,
    /// Anahtar kivrimi, `taps x kv`.
    pub ck: Vec<f64>,
}

/// Ileri gecisin sakladigi her sey - geri gecis bunlarsiz yazilamaz.
#[derive(Debug, Clone, PartialEq)]
pub struct Bellek {
    /// Sorgu izdusumu cikisi (kivrimsiz), `t x d_model`.
    pub q_ham: Vec<f64>,
    /// Kivrimdan gecmis sorgu, `t x d_model`.
    pub q: Vec<f64>,
    /// Anahtar izdusumu cikisi (kivrimsiz), `t x kv`.
    pub k_ham: Vec<f64>,
    /// Kivrimdan gecmis anahtar, `t x kv`.
    pub k: Vec<f64>,
    /// Deger, `t x kv`.
    pub v: Vec<f64>,
    /// dikkat agirliklari, `n_heads x t x t`.
    pub agirlik: Vec<f64>,
    /// Kafalarin birlestirilmis cikisi (wo oncesi), `t x d_model`.
    pub birlesik: Vec<f64>,
    /// Girdiler - izdusum geri gecisi bunlari ister.
    pub girdi_q: Vec<f64>,
    /// Girdi (anahtar yolu).
    pub girdi_k: Vec<f64>,
    /// Girdi (deger yolu).
    pub girdi_v: Vec<f64>,
    /// Jeton basina kayit kimligi (kayit siniri).
    pub kaynak: Vec<u32>,
    /// Pencere uzunlugu.
    pub t: usize,
}

/// Geri gecisin urettigi gradyanlar.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradyanlar {
    /// Sorgu izdusumu gradyani.
    pub wq: Vec<f64>,
    /// Anahtar izdusumu gradyani.
    pub wk: Vec<f64>,
    /// Deger izdusumu gradyani.
    pub wv: Vec<f64>,
    /// Cikis izdusumu gradyani.
    pub wo: Vec<f64>,
    /// Sorgu yanliligi gradyani.
    pub bq: Vec<f64>,
    /// Anahtar yanliligi gradyani.
    pub bk: Vec<f64>,
    /// Deger yanliligi gradyani.
    pub bv: Vec<f64>,
    /// Cikis yanliligi gradyani.
    pub bo: Vec<f64>,
    /// Sorgu kivrimi gradyani.
    pub cq: Vec<f64>,
    /// Anahtar kivrimi gradyani.
    pub ck: Vec<f64>,
    /// Pencereye giren sorgu gradyani.
    pub girdi_q: Vec<f64>,
    /// Pencereye giren anahtar gradyani.
    pub girdi_k: Vec<f64>,
    /// Pencereye giren deger gradyani.
    pub girdi_v: Vec<f64>,
}

/// Kucuk, deterministik dolgu: ayni tohum her makinede ayni sayilari verir.
#[must_use]
pub fn belirgin_doldur(spec: GrupSpec, tohum: u64) -> Agirliklar {
    let d = spec.d_model;
    let kv = spec.kv_genislik();
    let taps = spec.taps;
    let mut sayac: u64 = tohum.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut uret = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| {
                sayac = sayac
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let ust = (sayac >> 33) as f64 / (1u64 << 31) as f64;
                ust - 0.5
            })
            .collect()
    };
    Agirliklar {
        wq: uret(d * d),
        wk: uret(kv * d),
        wv: uret(kv * d),
        wo: uret(d * d),
        bq: uret(d),
        bk: uret(kv),
        bv: uret(kv),
        bo: uret(d),
        cq: uret(taps * d),
        ck: uret(taps * kv),
    }
}

fn satir_matmul(x: &[f64], w: &[f64], b: &[f64], girdi: usize, cikti: usize, t: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; t * cikti];
    for i in 0..t {
        for o in 0..cikti {
            let mut toplam = b[o];
            for k in 0..girdi {
                toplam += x[i * girdi + k] * w[o * girdi + k];
            }
            y[i * cikti + o] = toplam;
        }
    }
    y
}

/// Kayit sinirini asmadan nedensel kivrim: `y[i,c] = sum_tau c[tau,c]*x[i-tau,c]`,
/// yalniz ayni kaydin jetonlari icinden.
fn kivrim(
    x: &[f64],
    c: &[f64],
    genislik: usize,
    t: usize,
    taps: usize,
    kaynak: &[u32],
) -> Vec<f64> {
    let mut y = vec![0.0f64; t * genislik];
    for i in 0..t {
        for tau in 0..taps {
            if tau > i {
                break;
            }
            let j = i - tau;
            if kaynak[j] != kaynak[i] {
                continue;
            }
            for kanal in 0..genislik {
                y[i * genislik + kanal] += c[tau * genislik + kanal] * x[j * genislik + kanal];
            }
        }
    }
    y
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let en_buyuk = x
        .iter()
        .fold(f64::NEG_INFINITY, |a, b| if *b > a { *b } else { a });
    if !en_buyuk.is_finite() {
        return vec![0.0f64; x.len()];
    }
    let mut toplam = 0.0f64;
    let mut y = vec![0.0f64; x.len()];
    for (i, v) in x.iter().enumerate() {
        if v.is_finite() {
            let e = (v - en_buyuk).exp();
            y[i] = e;
            toplam += e;
        }
    }
    if toplam > 0.0 {
        for v in &mut y {
            *v /= toplam;
        }
    }
    y
}

/// Ileri gecis. `girdi_*` dizileri `t x d_model`, `kaynak` uzunlugu `t`.
#[must_use]
pub fn ileri(
    spec: GrupSpec,
    a: &Agirliklar,
    girdi_q: &[f64],
    girdi_k: &[f64],
    girdi_v: &[f64],
    t: usize,
    kaynak: &[u32],
) -> (Vec<f64>, Bellek) {
    let d = spec.d_model;
    let kv = spec.kv_genislik();
    let dk = spec.d_k();
    let h = spec.n_heads;
    let grup = spec.grup_boyu();
    let olcek = 1.0 / (dk as f64).sqrt();

    let q_ham = satir_matmul(girdi_q, &a.wq, &a.bq, d, d, t);
    let k_ham = satir_matmul(girdi_k, &a.wk, &a.bk, d, kv, t);
    let v = satir_matmul(girdi_v, &a.wv, &a.bv, d, kv, t);
    let q = kivrim(&q_ham, &a.cq, d, t, spec.taps, kaynak);
    let k = kivrim(&k_ham, &a.ck, kv, t, spec.taps, kaynak);

    let mut birlesik = vec![0.0f64; t * d];
    let mut agirlik = vec![0.0f64; h * t * t];
    for head in 0..h {
        let g = if grup == 0 { 0 } else { head / grup } * dk;
        for i in 0..t {
            let mut skor = vec![f64::NEG_INFINITY; t];
            for j in 0..=i {
                if kaynak[j] != kaynak[i] {
                    continue;
                }
                let mut toplam = 0.0;
                for m in 0..dk {
                    toplam += q[i * d + head * dk + m] * k[j * kv + g + m];
                }
                skor[j] = toplam * olcek;
            }
            let yumusak = softmax(&skor);
            for (j, w) in yumusak.iter().enumerate() {
                agirlik[head * t * t + i * t + j] = *w;
            }
            for m in 0..dk {
                let mut toplam = 0.0;
                for j in 0..t {
                    toplam += yumusak[j] * v[j * kv + g + m];
                }
                birlesik[i * d + head * dk + m] = toplam;
            }
        }
    }
    let cikti = satir_matmul(&birlesik, &a.wo, &a.bo, d, d, t);
    let bellek = Bellek {
        q_ham,
        q,
        k_ham,
        k,
        v,
        agirlik,
        birlesik,
        girdi_q: girdi_q.to_vec(),
        girdi_k: girdi_k.to_vec(),
        girdi_v: girdi_v.to_vec(),
        kaynak: kaynak.to_vec(),
        t,
    };
    (cikti, bellek)
}

/// Geri gecis: `d_cikti` (t x d_model) verilir, butun gradyanlar doner.
#[must_use]
pub fn geri(spec: GrupSpec, a: &Agirliklar, bellek: &Bellek, d_cikti: &[f64]) -> Gradyanlar {
    let d = spec.d_model;
    let kv = spec.kv_genislik();
    let dk = spec.d_k();
    let h = spec.n_heads;
    let grup = spec.grup_boyu();
    let t = bellek.t;
    let olcek = 1.0 / (dk as f64).sqrt();

    // Cikis izdusumu.
    let mut d_wo = vec![0.0f64; d * d];
    let mut d_bo = vec![0.0f64; d];
    let mut d_birlesik = vec![0.0f64; t * d];
    for i in 0..t {
        for c in 0..d {
            let g = d_cikti[i * d + c];
            d_bo[c] += g;
            d_wo[c * d..c * d + d]
                .iter_mut()
                .zip(&bellek.birlesik[i * d..i * d + d])
                .for_each(|(acc, x)| *acc += g * x);
            for k2 in 0..d {
                d_birlesik[i * d + k2] += g * a.wo[c * d + k2];
            }
        }
    }

    // dikkat: grup yayilimi.
    let mut dq = vec![0.0f64; t * d];
    let mut dk_ = vec![0.0f64; t * kv];
    let mut dv = vec![0.0f64; t * kv];
    for head in 0..h {
        let g = if grup == 0 { 0 } else { head / grup } * dk;
        for i in 0..t {
            let mut dw = vec![0.0f64; t];
            for m in 0..dk {
                let gr = d_birlesik[i * d + head * dk + m];
                for (j, dw_deger) in dw.iter_mut().enumerate().take(i + 1) {
                    if bellek.kaynak[j] != bellek.kaynak[i] {
                        continue;
                    }
                    *dw_deger += gr * bellek.v[j * kv + g + m];
                }
            }
            for m in 0..dk {
                let gr = d_birlesik[i * d + head * dk + m];
                for j in 0..=i {
                    if bellek.kaynak[j] != bellek.kaynak[i] {
                        continue;
                    }
                    let w = bellek.agirlik[head * t * t + i * t + j];
                    dv[j * kv + g + m] += gr * w;
                }
            }
            let mut nokta = 0.0;
            for (j, dw_deger) in dw.iter().enumerate().take(i + 1) {
                nokta += bellek.agirlik[head * t * t + i * t + j] * dw_deger;
            }
            let mut ds = vec![0.0f64; t];
            for (j, ds_deger) in ds.iter_mut().enumerate().take(i + 1) {
                if bellek.kaynak[j] != bellek.kaynak[i] {
                    continue;
                }
                let w = bellek.agirlik[head * t * t + i * t + j];
                *ds_deger = w * (dw[j] - nokta);
            }
            for j in 0..=i {
                if bellek.kaynak[j] != bellek.kaynak[i] {
                    continue;
                }
                for m in 0..dk {
                    dq[i * d + head * dk + m] += ds[j] * olcek * bellek.k[j * kv + g + m];
                    dk_[j * kv + g + m] += ds[j] * olcek * bellek.q[i * d + head * dk + m];
                }
            }
        }
    }

    // Kivrim geri gecisi (kisa dongu; dokunus sayisi kucuk).
    let mut dq_ham = vec![0.0f64; t * d];
    let mut dk_ham = vec![0.0f64; t * kv];
    let mut d_cq = vec![0.0f64; spec.taps * d];
    let mut d_ck = vec![0.0f64; spec.taps * kv];
    for i in 0..t {
        for tau in 0..spec.taps {
            if tau > i {
                break;
            }
            let j = i - tau;
            if bellek.kaynak[j] != bellek.kaynak[i] {
                continue;
            }
            for kanal in 0..d {
                d_cq[tau * d + kanal] += dq[i * d + kanal] * bellek.q_ham[j * d + kanal];
                dq_ham[j * d + kanal] += a.cq[tau * d + kanal] * dq[i * d + kanal];
            }
            for kanal in 0..kv {
                d_ck[tau * kv + kanal] += dk_[i * kv + kanal] * bellek.k_ham[j * kv + kanal];
                dk_ham[j * kv + kanal] += a.ck[tau * kv + kanal] * dk_[i * kv + kanal];
            }
        }
    }

    // Izdusum geri gecisleri.
    let mut d_wq = vec![0.0f64; d * d];
    let mut d_wk = vec![0.0f64; kv * d];
    let mut d_wv = vec![0.0f64; kv * d];
    let mut d_bq = vec![0.0f64; d];
    let mut d_bk = vec![0.0f64; kv];
    let mut d_bv = vec![0.0f64; kv];
    let mut d_girdi_q = vec![0.0f64; t * d];
    let mut d_girdi_k = vec![0.0f64; t * d];
    let mut d_girdi_v = vec![0.0f64; t * d];
    for i in 0..t {
        for o in 0..d {
            let gr = dq_ham[i * d + o];
            d_bq[o] += gr;
            for k2 in 0..d {
                d_wq[o * d + k2] += gr * bellek.girdi_q[i * d + k2];
                d_girdi_q[i * d + k2] += gr * a.wq[o * d + k2];
            }
        }
        for o in 0..kv {
            let gk = dk_ham[i * kv + o];
            let gv = dv[i * kv + o];
            d_bk[o] += gk;
            d_bv[o] += gv;
            for k2 in 0..d {
                d_wk[o * d + k2] += gk * bellek.girdi_k[i * d + k2];
                d_girdi_k[i * d + k2] += gk * a.wk[o * d + k2];
                d_wv[o * d + k2] += gv * bellek.girdi_v[i * d + k2];
                d_girdi_v[i * d + k2] += gv * a.wv[o * d + k2];
            }
        }
    }

    Gradyanlar {
        wq: d_wq,
        wk: d_wk,
        wv: d_wv,
        wo: d_wo,
        bq: d_bq,
        bk: d_bk,
        bv: d_bv,
        bo: d_bo,
        cq: d_cq,
        ck: d_ck,
        girdi_q: d_girdi_q,
        girdi_k: d_girdi_k,
        girdi_v: d_girdi_v,
    }
}

/// Kayip: `0.5 * sum((cikti - hedef)^2)`. Hedefe gore gradyan `cikti - hedef`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GRADIENT_CHECK_MUTLAK_TABAN, GRADIENT_CHECK_TOLERANCE};

    fn kucuk() -> GrupSpec {
        GrupSpec {
            d_model: 8,
            n_heads: 2,
            n_kv_heads: 1,
            taps: 3,
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

    fn hedef_vektor(t: usize, d: usize, tohum: u64) -> Vec<f64> {
        girdiler(t, d, tohum)
    }

    #[test]
    fn sekil_dogrulanir() {
        assert_eq!(kucuk().dogrula(), Ok(()));
        assert_eq!(
            GrupSpec {
                d_model: 8,
                n_heads: 2,
                n_kv_heads: 3,
                taps: 3
            }
            .dogrula(),
            Err(GrupHatasi::GrupBolmuyor)
        );
        assert_eq!(
            GrupSpec {
                d_model: 8,
                n_heads: 3,
                n_kv_heads: 1,
                taps: 3
            }
            .dogrula(),
            Err(GrupHatasi::BasSayisiBolmuyor)
        );
        assert_eq!(
            GrupSpec {
                d_model: 8,
                n_heads: 2,
                n_kv_heads: 1,
                taps: 0
            }
            .dogrula(),
            Err(GrupHatasi::DokunusAraligi)
        );
        assert_eq!(
            GrupSpec {
                d_model: 0,
                n_heads: 2,
                n_kv_heads: 1,
                taps: 3
            }
            .dogrula(),
            Err(GrupHatasi::BosBoyut)
        );
    }

    #[test]
    fn gruplama_parametre_tasarrufunu_olcer() {
        let spec = GrupSpec::lubot_a2_adayi();
        assert_eq!(spec.dogrula(), Ok(()));
        let gruplu = spec.parametre_sayisi();
        let tam = spec.tam_dikkat_parametre_sayisi();
        assert!(
            gruplu < tam,
            "gruplu {gruplu} tam {tam} olmali ki tasarruf olculsun"
        );
        let d = spec.d_model as f64;
        let kv = spec.kv_genislik() as f64;
        let beklenen = ((d * d + 2.0 * kv * d + d * d)
            + (d + 2.0 * kv + d)
            + (spec.taps as f64 * d + spec.taps as f64 * kv)) as usize;
        assert_eq!(gruplu, beklenen);
        // Tasarruf tam olarak K/V izdusumlerinden ve yanliliklarindan gelir.
        let fark = tam - gruplu;
        let beklenen_fark = (2 * spec.d_model - 2 * spec.kv_genislik()) * spec.d_model
            + (2 * spec.d_model - 2 * spec.kv_genislik())
            + spec.taps * (spec.d_model - spec.kv_genislik());
        assert_eq!(fark, beklenen_fark);
    }

    #[test]
    fn kayit_siniri_dikkatte_asilmaz() {
        let spec = kucuk();
        let a = belirgin_doldur(spec, 3);
        let t = 4;
        let d = spec.d_model;
        let x1 = girdiler(t, d, 11);
        let kaynak = [1u32, 1, 2, 2];
        let (cikti1, _) = ileri(spec, &a, &x1, &x1, &x1, t, &kaynak);
        // Ikinci kaydin jetonlarini degistir; birinci kaydin ciktisi degismemeli.
        let mut x2 = x1.clone();
        for v in x2.iter_mut().skip(2 * d) {
            *v += 0.5;
        }
        let (cikti2, _) = ileri(spec, &a, &x2, &x2, &x2, t, &kaynak);
        for i in 0..2 * d {
            assert_eq!(
                cikti1[i].to_bits(),
                cikti2[i].to_bits(),
                "kayit 1'in {i}. elemani komsu kayittan etkilendi"
            );
        }
        // Ve ikinci kaydin ciktisi gercekten degisti (test bosa kosmuyor).
        let degisen = (2 * d..t * d).any(|i| cikti1[i].to_bits() != cikti2[i].to_bits());
        assert!(degisen, "komsu kayit degisikligi hicbir seyi degistirmedi");
    }

    #[test]
    fn kayit_siniri_kivrimda_da_asilmaz() {
        let spec = kucuk();
        let a = belirgin_doldur(spec, 5);
        let t = 3;
        let d = spec.d_model;
        let x = girdiler(t, d, 17);
        let iki_kayit = [1u32, 2, 2];
        let tek_kayit = [1u32, 1, 1];
        let (c_iki, _) = ileri(spec, &a, &x, &x, &x, t, &iki_kayit);
        let (c_tek, _) = ileri(spec, &a, &x, &x, &x, t, &tek_kayit);
        // 0. jetonun cikisi iki durumda da ayni: dokunus komsu kayda uzanmiyor.
        for c in 0..d {
            assert_eq!(c_iki[c].to_bits(), c_tek[c].to_bits());
        }
    }

    #[test]
    fn ayni_girdi_ayni_baytlari_verir() {
        let spec = kucuk();
        let a = belirgin_doldur(spec, 9);
        let t = 4;
        let x = girdiler(t, spec.d_model, 23);
        let kaynak = [1u32, 1, 2, 2];
        let (c1, b1) = ileri(spec, &a, &x, &x, &x, t, &kaynak);
        let (c2, b2) = ileri(spec, &a, &x, &x, &x, t, &kaynak);
        assert_eq!(c1, c2);
        assert_eq!(b1.agirlik, b2.agirlik);
    }

    #[test]
    fn dokunuslar_uygulanir() {
        let spec = kucuk();
        let a = belirgin_doldur(spec, 13);
        let t = 4;
        let x = girdiler(t, spec.d_model, 29);
        let kaynak = [1u32, 1, 1, 1];
        let (cikti, _) = ileri(spec, &a, &x, &x, &x, t, &kaynak);
        // Dokunuslari sifirla: kivrim kimlige doner, cikti degismeli.
        let mut a2 = a.clone();
        for v in &mut a2.cq {
            *v = 0.0;
        }
        for v in &mut a2.ck {
            *v = 0.0;
        }
        let (cikti2, _) = ileri(spec, &a2, &x, &x, &x, t, &kaynak);
        assert_ne!(
            cikti, cikti2,
            "dokunus agirliklari ciktiyi hic degistirmedi"
        );
    }

    /// Hangi agirlik tensoru denetleniyor - sonlu fark dongusu alanlari
    /// isimle gezsin diye.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Alan {
        Wq,
        Wk,
        Wv,
        Wo,
        Bq,
        Bk,
        Bv,
        Bo,
        Cq,
        Ck,
    }

    const ALANLAR: [(&str, Alan); 10] = [
        ("wq", Alan::Wq),
        ("wk", Alan::Wk),
        ("wv", Alan::Wv),
        ("wo", Alan::Wo),
        ("bq", Alan::Bq),
        ("bk", Alan::Bk),
        ("bv", Alan::Bv),
        ("bo", Alan::Bo),
        ("cq", Alan::Cq),
        ("ck", Alan::Ck),
    ];

    fn alan(a: &Agirliklar, secim: Alan) -> &Vec<f64> {
        match secim {
            Alan::Wq => &a.wq,
            Alan::Wk => &a.wk,
            Alan::Wv => &a.wv,
            Alan::Wo => &a.wo,
            Alan::Bq => &a.bq,
            Alan::Bk => &a.bk,
            Alan::Bv => &a.bv,
            Alan::Bo => &a.bo,
            Alan::Cq => &a.cq,
            Alan::Ck => &a.ck,
        }
    }

    fn alan_mut(a: &mut Agirliklar, secim: Alan) -> &mut Vec<f64> {
        match secim {
            Alan::Wq => &mut a.wq,
            Alan::Wk => &mut a.wk,
            Alan::Wv => &mut a.wv,
            Alan::Wo => &mut a.wo,
            Alan::Bq => &mut a.bq,
            Alan::Bk => &mut a.bk,
            Alan::Bv => &mut a.bv,
            Alan::Bo => &mut a.bo,
            Alan::Cq => &mut a.cq,
            Alan::Ck => &mut a.ck,
        }
    }

    #[test]
    fn gradyan_sonlu_farkla_uyusur() -> Result<(), String> {
        let spec = kucuk();
        spec.dogrula()
            .map_err(|e| format!("sekil reddedildi: {e:?}"))?;
        let a = belirgin_doldur(spec, 7);
        let t = 4;
        let d = spec.d_model;
        let x = girdiler(t, d, 31);
        let kaynak = [1u32, 1, 2, 2];
        let hedef = hedef_vektor(t, d, 37);

        let (cikti, bellek) = ileri(spec, &a, &x, &x, &x, t, &kaynak);
        let g = geri(spec, &a, &bellek, &kayip_gradyan(&cikti, &hedef));
        let analitik_alan = |secim: Alan| -> Vec<f64> {
            match secim {
                Alan::Wq => g.wq.clone(),
                Alan::Wk => g.wk.clone(),
                Alan::Wv => g.wv.clone(),
                Alan::Wo => g.wo.clone(),
                Alan::Bq => g.bq.clone(),
                Alan::Bk => g.bk.clone(),
                Alan::Bv => g.bv.clone(),
                Alan::Bo => g.bo.clone(),
                Alan::Cq => g.cq.clone(),
                Alan::Ck => g.ck.clone(),
            }
        };

        let h = 1e-5f64;
        let mut en_kotu: f64 = 0.0;
        let mut en_kotu_ad = String::new();
        let mut denetlenen = 0usize;
        let mut ihlaller: Vec<String> = Vec::new();
        for (ad, secim) in ALANLAR {
            let analitik = analitik_alan(secim);
            let mut deneme = a.clone();
            for (i, analitik_deger) in analitik.iter().enumerate() {
                let asil = alan(&a, secim)[i];
                alan_mut(&mut deneme, secim)[i] = asil + h;
                let (c_art, _) = ileri(spec, &deneme, &x, &x, &x, t, &kaynak);
                alan_mut(&mut deneme, secim)[i] = asil - h;
                let (c_eks, _) = ileri(spec, &deneme, &x, &x, &x, t, &kaynak);
                alan_mut(&mut deneme, secim)[i] = asil;
                let sayisal = (kayip(&c_art, &hedef) - kayip(&c_eks, &hedef)) / (2.0 * h);
                let analitik_deger = *analitik_deger;
                let fark = (analitik_deger - sayisal).abs();
                // Deponun kendi olcusu (bkz. GRADIENT_CHECK_*): mutlak taban +
                // bagil tolerans. Sonlu fark, olcek ~1e-6 altindaki gradyanlari
                // cozemez; orada gurultu "yanlis gradyan" diye okunmaz.
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

        if denetlenen == 0 {
            return Err("hicbir parametre denetlenmedi".to_string());
        }
        // Sekle bagli sayim: yeni bir agirlik tensoru eklenip denetime
        // eklenmezse bu test duser, sessizce denetimsiz kalmaz.
        if denetlenen != spec.parametre_sayisi() {
            return Err(format!(
                "{} gradyan denetlendi ama sekil {} parametre sayiyor",
                denetlenen,
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
        let t = 3;
        let d = spec.d_model;
        let x = girdiler(t, d, 41);
        let kaynak = [1u32, 1, 1];
        let hedef = hedef_vektor(t, d, 43);

        let (cikti, bellek) = ileri(spec, &a, &x, &x, &x, t, &kaynak);
        let g = geri(spec, &a, &bellek, &kayip_gradyan(&cikti, &hedef));

        let h = 1e-5f64;
        let mut en_kotu: f64 = 0.0;
        for i in 0..x.len() {
            let mut x_art = x.clone();
            x_art[i] += h;
            let (c_art, _) = ileri(spec, &a, &x_art, &x_art, &x_art, t, &kaynak);
            let mut x_eks = x.clone();
            x_eks[i] -= h;
            let (c_eks, _) = ileri(spec, &a, &x_eks, &x_eks, &x_eks, t, &kaynak);
            let sayisal = (kayip(&c_art, &hedef) - kayip(&c_eks, &hedef)) / (2.0 * h);
            let analitik = g.girdi_q[i] + g.girdi_k[i] + g.girdi_v[i];
            let olcek = analitik.abs().max(sayisal.abs());
            let bagil = if olcek > 1e-9 {
                (analitik - sayisal).abs() / olcek
            } else {
                (analitik - sayisal).abs()
            };
            en_kotu = en_kotu.max(bagil);
        }
        if en_kotu > 1e-6 {
            return Err(format!("girdi gradyani bagil hata {en_kotu:.3e}"));
        }
        Ok(())
    }
}
