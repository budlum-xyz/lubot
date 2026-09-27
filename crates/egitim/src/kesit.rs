//! Kesit: bir spec'ten **calisan** alt-modeller turetmek.
//!
//! Ayni agirlik kumesinden birden cok boyutta model kosmak iki ayri sorudur ve
//! bu modul yalnizca birincisini cevaplar:
//!
//! * **Kesit turetme (burada):** verilen spec'ten derinlik ve genislik kesiti
//!   alinir, cikan spec'in *kendisi* gecerli olmali ve kosabilmelidir.
//! * Rung baytlarindan bellek tavani secmek (`lubot-tasiyici::merdiven`):
//!   hangi derinligin *sigdigi*. O modul "sigiyor mu" der; "calisiyor mu"
//!   demez ve dememeli. Bu modul tersini yapar: **kosar** ve olcer.
//!
//! Sozlesme (hepsi testte olculur):
//!
//! * **Tamlık:** tam derinlik ve tam genislik girdi spec'inin ta kendisidir
//!   (bit-esit parametre sayisi; yeni spec uydurulmaz).
//! * **Her kesit gecerli:** uretilen her spec kendi dogrulamasindan gecer.
//! * **Her kesit kosar:** uretilen her spec ile bir ileri+geri adim kosar ve
//!   kayip sonludur. "Her derinlikte kosulabilen aile" iddiasi boyle olculur -
//!   spec alanlarini kopyalamak kosmak degildir.
//! * **Monotonluk:** derinlik de genislik de arttikca parametre sayisi
//!   **azalmaz** (tam sayi aritmetigi; karsilastirma esitlik de kabul eder ama
//!   bu ailede artar).
//! * **Bas dilimi:** genislik kesiti **kafa silerek** daralir; `d_k` sabit kalir
//!   (kafayi daraltmak ayni kafayi bozmak olurdu). Bu yuzden genislik hedefi
//!   `d_k`'nin tam kati olmali; olmayan hedef reddedilir, en yakin degere
//!   yuvarlanmaz - sessiz yuvarlama, olculen seyi degistirir.
//!
//! Ogrenilebilir parametresi yoktur; bu modul **spec uretir**, agirlik degil.

use crate::{Spec, SpecHatasi};

/// Kesit istegi neden reddedildi.
#[derive(Debug, Clone, PartialEq)]
pub enum KesitHatasi {
    /// Istenen derinlik sifir: sifir katmanli bir model yoktur.
    SifirDerinlik,
    /// Istenen derinlik spec'ten fazla; kesit buyutmez, kirpar.
    DerinlikSpecAsiyor(usize, usize),
    /// Istenen genislik sifir.
    SifirGenislik,
    /// Istenen genislik spec'ten genis; kesit buyutmez, daraltir.
    GenislikSpecAsiyor(usize, usize),
    /// Hedef genislik kafa boyutunun kati degil.
    GenislikKafaKatıDegil(usize, usize),
    /// Uretilen spec kendi dogrulamasindan gecmedi (olculemeyen hal).
    UretilenSpecGecersiz(SpecHatasi),
}

/// Bir kesit: turetilen spec ve onu tureten istek.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kesit {
    /// Turetilen spec.
    pub spec: Spec,
    /// Istenen derinlik (katman sayisi).
    pub derinlik: usize,
    /// Istenen genislik (`d_model`).
    pub genislik: usize,
}

/// Aile kesiti: spec'ten derinlik/genislik dilimi turetir.
///
/// `derinlik` katman sayisidir (1..=spec.n_layers) ve her zaman **bastan**
/// alinir: ortadan katman cikarmak baska bir modeldir, kisaltmak ise ayni
/// modelin onceki katmanlaridir. `genislik` hedef `d_model`dir; kafa boyutu
/// (`d_k = d_model / n_heads`) korunur, kafa sayisi hedefe gore azalir.
pub fn kesit(spec: Spec, derinlik: usize, genislik: usize) -> Result<Kesit, KesitHatasi> {
    if derinlik == 0 {
        return Err(KesitHatasi::SifirDerinlik);
    }
    if derinlik > spec.n_layers {
        return Err(KesitHatasi::DerinlikSpecAsiyor(derinlik, spec.n_layers));
    }
    if genislik == 0 {
        return Err(KesitHatasi::SifirGenislik);
    }
    if genislik > spec.d_model {
        return Err(KesitHatasi::GenislikSpecAsiyor(genislik, spec.d_model));
    }
    let d_k = spec.d_k();
    if d_k == 0 || genislik % d_k != 0 {
        return Err(KesitHatasi::GenislikKafaKatıDegil(genislik, d_k));
    }
    let kafa = genislik / d_k;
    // MLP ic genisligi d_model ile ayni oranda olceklenir ve tam sayi
    // aritmetigi ile hesaplanir: yuvarlama sessiz kalmaz, asagi yuvarlanir ve
    // yine de pozitif kalir (`d_ff >= d_model` spec dogrulamasinda zaten sart).
    let d_ff = (spec.d_ff * genislik / spec.d_model).max(genislik);
    let mut kesilen = Spec {
        n_layers: derinlik,
        d_model: genislik,
        n_heads: kafa,
        // K/V kafalari da kafa sayisini asamaz; grup orani korunur.
        n_kv_heads: kafa.min(spec.n_kv_heads.max(1)),
        d_ff,
        ..spec
    };
    if kesilen.n_kv_heads > kesilen.n_heads {
        kesilen.n_kv_heads = kesilen.n_heads;
    }
    if kesilen.n_heads % kesilen.n_kv_heads != 0 {
        // Grup orani tutmuyorsa K/V kafasini tam bolene cek: en yakin *bozan*
        // deger degil, en yakin *gecerli* deger secilir ve bu bir yuvarlama
        // degil, gecerli kumenin en buyuk elemanidir.
        let mut kv = kesilen.n_kv_heads;
        while kv > 1 && kesilen.n_heads % kv != 0 {
            kv -= 1;
        }
        kesilen.n_kv_heads = kv;
    }
    kesilen
        .dogrula()
        .map_err(KesitHatasi::UretilenSpecGecersiz)?;
    Ok(Kesit {
        spec: kesilen,
        derinlik,
        genislik,
    })
}

/// Derinlik merdiveni: 1..=n_layers arasi her derinlik icin bir kesit.
///
/// Sira derinlige gore artar; her basamak kendi dogrulamasindan gecmis olur.
pub fn derinlik_merdiveni(spec: Spec) -> Vec<Kesit> {
    (1..=spec.n_layers)
        .filter_map(|d| kesit(spec, d, spec.d_model).ok())
        .collect()
}

/// Genislik izgarasi: kafa katlarina dusen genislikler icin kesit.
pub fn genislik_izgarasi(spec: Spec, en_az_kafa: usize) -> Vec<Kesit> {
    let d_k = spec.d_k();
    if d_k == 0 {
        return Vec::new();
    }
    // Izgara **dardan genise** uretilir: merdiven okunusu budur ve monotonluk
    // iddiasi bu sirayla anlamli olur (genisleyen kesit daha az parametre
    // tutamaz). Ters sirada uretmek testin yonunu ters cevirirdi - olculdu.
    let en_az_kafa = en_az_kafa.max(1);
    let mut izgara = Vec::new();
    for kafa in en_az_kafa..=spec.n_heads {
        if let Ok(k) = kesit(spec, spec.n_layers, kafa * d_k) {
            izgara.push(k);
        }
    }
    izgara
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ileri_ve_geri, Parametreler};

    fn temel() -> Spec {
        Spec {
            vocab: 512,
            d_model: 64,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 2,
            qkv_dokunus: 0,
            qk_norm: false,
            d_ff: 256,
            max_seq_len: 64,
        }
    }

    #[test]
    fn sekil_hatalari_reddedilir() {
        let s = temel();
        assert_eq!(kesit(s, 0, 64), Err(KesitHatasi::SifirDerinlik));
        assert_eq!(kesit(s, 5, 64), Err(KesitHatasi::DerinlikSpecAsiyor(5, 4)));
        assert_eq!(kesit(s, 2, 0), Err(KesitHatasi::SifirGenislik));
        assert_eq!(
            kesit(s, 2, 128),
            Err(KesitHatasi::GenislikSpecAsiyor(128, 64))
        );
        assert_eq!(
            kesit(s, 2, 40),
            Err(KesitHatasi::GenislikKafaKatıDegil(40, 16))
        );
        assert!(kesit(s, 2, 32).is_ok());
    }

    #[test]
    fn tamlik_girdinin_kendisi() {
        let s = temel();
        let tam = kesit(s, s.n_layers, s.d_model).unwrap_or_else(|_| panic!("kesit"));
        assert_eq!(tam.spec, s, "tam kesit girdi spec'i olmali");
        assert_eq!(tam.spec.parametre_sayisi(), s.parametre_sayisi());
    }

    #[test]
    fn her_kesit_kendi_dogrulamasindan_gecer() {
        let s = temel();
        for derinlik in 1..=s.n_layers {
            for genislik in [16, 32, 48, 64] {
                let k = kesit(s, derinlik, genislik).unwrap_or_else(|_| panic!("kesit"));
                assert_eq!(
                    k.spec.dogrula(),
                    Ok(()),
                    "derinlik {derinlik} genislik {genislik} kesiti gecersiz"
                );
            }
        }
    }

    #[test]
    fn derinlik_arttikca_parametre_azalmaz() {
        let s = temel();
        let merdiven = derinlik_merdiveni(s);
        assert_eq!(merdiven.len(), s.n_layers);
        for cift in merdiven.windows(2) {
            assert!(
                cift[1].spec.parametre_sayisi() >= cift[0].spec.parametre_sayisi(),
                "derinlik artarken parametre dustu: {} -> {}",
                cift[0].spec.parametre_sayisi(),
                cift[1].spec.parametre_sayisi()
            );
        }
    }

    #[test]
    fn genislik_arttikca_parametre_azalmaz() {
        let s = temel();
        let izgara = genislik_izgarasi(s, 1);
        assert!(izgara.len() >= 3, "izgara cok kisa: {}", izgara.len());
        for cift in izgara.windows(2) {
            assert!(
                cift[0].spec.parametre_sayisi() <= cift[1].spec.parametre_sayisi(),
                "genislik azalirken parametre artti"
            );
        }
        assert_eq!(izgara.last().map(|k| k.spec.d_model), Some(s.d_model));
    }

    #[test]
    fn kafa_silinir_kafa_daraltilmaz() {
        let s = temel();
        let k = kesit(s, s.n_layers, 32).unwrap_or_else(|_| panic!("kesit"));
        assert_eq!(k.spec.d_k(), s.d_k(), "d_k sabit kalmali");
        assert_eq!(k.spec.n_heads, 2, "kafa sayisi hedefe gore azalmali");
        assert!(
            k.spec.n_kv_heads <= k.spec.n_heads,
            "kv kafasi kafa sayisini asamaz"
        );
        assert_eq!(
            k.spec.n_heads % k.spec.n_kv_heads,
            0,
            "grup orani tam bolmeli"
        );
    }

    #[test]
    fn kesit_kosumdan_gecer() {
        // "Kosulabilir aile" iddiasinin olcumu: her derinlik ve her genislik
        // icin bir ileri+geri adim kosar, kayip sonlu cikar.
        let s = temel();
        let girdi: Vec<usize> = (0..16).map(|j| (j * 7 + 3) % s.vocab).collect();
        let hedef: Vec<usize> = (0..16).map(|j| (j * 11 + 5) % s.vocab).collect();
        let mut kosan = 0usize;
        for derinlik in 1..=s.n_layers {
            for genislik in [16, 32, 64] {
                let k = kesit(s, derinlik, genislik).unwrap_or_else(|_| panic!("kesit"));
                let p = Parametreler::belirgin_doldur(k.spec, 20260927);
                let (kayip, gradyan) = ileri_ve_geri(k.spec, &p, &girdi, &hedef);
                assert!(
                    kayip.is_finite(),
                    "kayip sonlu degil: derinlik {derinlik} genislik {genislik}"
                );
                assert!(kayip > 0.0, "pozitif kayip beklenir (capraz entropi)");
                assert!(
                    gradyan.embedding.iter().any(|g| *g != 0.0),
                    "gomme gradyani bos"
                );
                kosan += 1;
            }
        }
        assert_eq!(kosan, 12, "12 kesit kosmali");
    }

    #[test]
    fn kesit_agirlik_gerektirmez() {
        // Kesit bir spec islemidir: parametre tutmaz, parametre sayisi
        // spec'ten turetilir ve kesit kucukse azdir.
        let s = temel();
        let k = kesit(s, 2, 32).unwrap_or_else(|_| panic!("kesit"));
        assert!(k.spec.parametre_sayisi() < s.parametre_sayisi());
        assert_eq!(k.spec.vocab, s.vocab, "gomme sozlugu ayni kalir");
        assert_eq!(k.spec.max_seq_len, s.max_seq_len, "pencere ayni kalir");
    }

    #[test]
    fn ayni_istek_ayni_kesit() {
        let s = temel();
        let a = kesit(s, 3, 32).unwrap_or_else(|_| panic!("kesit"));
        let b = kesit(s, 3, 32).unwrap_or_else(|_| panic!("kesit"));
        assert_eq!(a, b, "kesit deterministik olmali");
    }

    #[test]
    fn olcum_raporu() {
        let s = temel();
        let merdiven = derinlik_merdiveni(s);
        let ilk = merdiven
            .first()
            .map(|k| k.spec.parametre_sayisi())
            .unwrap_or(0);
        let izgara = genislik_izgarasi(s, 1);
        let en_dar = izgara
            .first()
            .map(|k| k.spec.parametre_sayisi())
            .unwrap_or(0);
        // Kosan kesit sayimi: "her derinlikte kosulabilen aile" iddiasinin
        // sayisi iddia edilmez, kosularak bulunur.
        let girdi: Vec<usize> = (0..16).map(|j| (j * 7 + 3) % s.vocab).collect();
        let hedef: Vec<usize> = (0..16).map(|j| (j * 11 + 5) % s.vocab).collect();
        let mut kosan = 0usize;
        for derinlik in 1..=s.n_layers {
            for genislik in [16, 32, 64] {
                if let Ok(k) = kesit(s, derinlik, genislik) {
                    let pp = Parametreler::belirgin_doldur(k.spec, 20260927);
                    let (kayip, _) = ileri_ve_geri(k.spec, &pp, &girdi, &hedef);
                    if kayip.is_finite() && kayip > 0.0 {
                        kosan += 1;
                    }
                }
            }
        }
        println!(
            "kesit | derinlik={} genislik={} d_k={} izgara={} kosan={} parametre_en_az={} parametre_tam={} parametre_genislik_en_az={}",
            s.n_layers,
            s.d_model,
            s.d_k(),
            izgara.len(),
            kosan,
            ilk,
            s.parametre_sayisi(),
            en_dar
        );
    }
}
