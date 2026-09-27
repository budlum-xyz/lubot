//! Kesit: bir spec'ten **calisan** alt-modeller turetmek.
//!
//! Ayni agirlik kumesinden birden cok boyutta model kosmak iki ayri sorudur ve
//! bu modul spec ve agirlik kesitini cevaplar; bellek merdiveni ayri kalir:
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
//! Yeni ogrenilebilir parametre uretmez; var olan agirliklari koordinatla keser.
//! Optimizer durumu tasinmaz, checkpoint ustune yazilmaz.

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
    /// Disaridan kurulan kesit, istekten turetilen spec ile uyusmuyor.
    KesitTutarsiz,
    /// Tensor boyutu veya toplam bayt usize araligini asti.
    BoyutTasmasi,
    /// kaynak tensor uzunlugu kaynak spec'e uymuyor.
    AgirlikSekli {
        blok: usize,
        beklenen: usize,
        gelen: usize,
    },
    /// kaynakta NaN ya da sonsuz deger var; sessiz tasinmaz.
    SonluOlmayanAgirlik { blok: usize, konum: usize },
    /// Yeni tensor verisi operatorun verdigi ek bellek tavanini asiyor.
    BellekTavani { gereken: usize, tavan: usize },
    /// Allocator talebi reddetti.
    TahsisReddedildi,
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
    spec.dogrula().map_err(KesitHatasi::UretilenSpecGecersiz)?;
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
    // Carpim u128 ile hesaplanir; genislik <= d_model oldugu icin sonuc
    // usize araligindadir. Dar MLP de gecerlidir: tam kesit d_ff'yi korur.
    let d_ff = ((spec.d_ff as u128 * genislik as u128) / spec.d_model as u128).max(1) as usize;
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
    if spec.dogrula().is_err() {
        return Vec::new();
    }
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

// Checkpoint blok sirasiyla [katman, cikis, giris]. Duz prefix kirpma
// matrislerde yanlistir: kaynak satir adimi korunarak hedef satir kopyalanir.
fn tensor_sekilleri(s: Spec) -> [[usize; 3]; 24] {
    let d = s.d_model;
    let kv = s.d_kv();
    let l = s.n_layers;
    let f = s.d_ff;
    let qn = if s.qk_norm { s.d_k() } else { 0 };
    [
        [1, s.vocab, d],
        [l, 1, d],
        [l, 1, d],
        [l, d, d],
        [l, 1, d],
        [l, s.qkv_dokunus, d],
        [l, 1, qn],
        [l, kv, d],
        [l, 1, kv],
        [l, s.qkv_dokunus, kv],
        [l, 1, qn],
        [l, kv, d],
        [l, 1, kv],
        [l, s.qkv_dokunus, kv],
        [l, d, d],
        [l, 1, d],
        [l, 1, d],
        [l, 1, d],
        [l, f, d],
        [l, 1, f],
        [l, d, f],
        [l, 1, d],
        [1, 1, d],
        [1, 1, d],
    ]
}

fn tensor_boyutu(sekil: [usize; 3]) -> Result<usize, KesitHatasi> {
    sekil.into_iter().try_fold(1usize, |n, boyut| {
        n.checked_mul(boyut).ok_or(KesitHatasi::BoyutTasmasi)
    })
}

impl Kesit {
    /// Var olan agirliklardan bastan derinlik ve satir/sutun genislik kesiti.
    ///
    /// Yeni tohumlama yoktur: her hedef eleman kaynakta bir koordinata aittir.
    /// Q/K kafa boyutu korunur; grup sayisinin degismesi fonksiyonel esdegerlik
    /// garantisi vermez. Daraltilmis modelin kalitesi ayrica sinanmalidir.
    ///
    /// `tavan_bayt`, yalnizca yeni f64 tensor verisinin ek bellek butcesidir;
    /// kaynak checkpoint, vektor basliklari ve allocator ek yuku dahil degil.
    /// Tum sekiller, kaynak sayilar ve butce tahsisten ONCE denetlenir.
    /// Girdi salt okunur; hata halinde kaynak ve optimizer degismez.
    pub fn agirliklari_al(
        self,
        kaynak: Spec,
        parametreler: &crate::Parametreler,
        tavan_bayt: usize,
    ) -> Result<crate::Parametreler, KesitHatasi> {
        let dogrulanmis = kesit(kaynak, self.derinlik, self.genislik)?;
        if dogrulanmis != self {
            return Err(KesitHatasi::KesitTutarsiz);
        }
        let kaynak_sekiller = tensor_sekilleri(kaynak);
        let hedef_sekiller = tensor_sekilleri(self.spec);
        let bloklar = parametreler.bloklar();
        let mut toplam = 0usize;
        for (i, ((blok, kaynak_sekil), hedef_sekil)) in bloklar
            .iter()
            .zip(kaynak_sekiller)
            .zip(hedef_sekiller)
            .enumerate()
        {
            let beklenen = tensor_boyutu(kaynak_sekil)?;
            if blok.len() != beklenen {
                return Err(KesitHatasi::AgirlikSekli {
                    blok: i,
                    beklenen,
                    gelen: blok.len(),
                });
            }
            if let Some(konum) = blok.iter().position(|x| !x.is_finite()) {
                return Err(KesitHatasi::SonluOlmayanAgirlik { blok: i, konum });
            }
            let boyut = tensor_boyutu(hedef_sekil)?;
            toplam = toplam.checked_add(boyut).ok_or(KesitHatasi::BoyutTasmasi)?;
        }
        let gereken = toplam
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(KesitHatasi::BoyutTasmasi)?;
        if gereken > tavan_bayt {
            return Err(KesitHatasi::BellekTavani {
                gereken,
                tavan: tavan_bayt,
            });
        }
        let mut sonuc: [Vec<f64>; 24] = std::array::from_fn(|_| Vec::new());
        for (i, hedef) in sonuc.iter_mut().enumerate() {
            let [katman, satir, sutun] = hedef_sekiller[i];
            let [_, kaynak_satir, kaynak_sutun] = kaynak_sekiller[i];
            let boyut = tensor_boyutu(hedef_sekiller[i])?;
            hedef
                .try_reserve_exact(boyut)
                .map_err(|_| KesitHatasi::TahsisReddedildi)?;
            // Kapali opsiyonlarin [l,0,d] ya da [l,1,0] sekli tahsis/kopya yapmaz.
            if boyut == 0 {
                continue;
            }
            for l in 0..katman {
                for r in 0..satir {
                    let bas = (l * kaynak_satir + r) * kaynak_sutun;
                    let dilim = bloklar[i]
                        .get(bas..bas + sutun)
                        .ok_or(KesitHatasi::KesitTutarsiz)?;
                    hedef.extend_from_slice(dilim);
                }
            }
        }
        let [embedding, ln1_olcek, ln1_sapma, wq, bq, q_dokunus, q_norm_olcek, wk, bk, k_dokunus, k_norm_olcek, wv, bv, v_dokunus, wo, bo, ln2_olcek, ln2_sapma, w1, b1, w2, b2, lnf_olcek, lnf_sapma] =
            sonuc;
        Ok(crate::Parametreler {
            embedding,
            ln1_olcek,
            ln1_sapma,
            wq,
            bq,
            q_dokunus,
            q_norm_olcek,
            wk,
            bk,
            k_dokunus,
            k_norm_olcek,
            wv,
            bv,
            v_dokunus,
            wo,
            bo,
            ln2_olcek,
            ln2_sapma,
            w1,
            b1,
            w2,
            b2,
            lnf_olcek,
            lnf_sapma,
        })
    }
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
        for bozuk in [
            Spec { n_heads: 0, ..s },
            Spec { n_kv_heads: 0, ..s },
            Spec { d_model: 0, ..s },
        ] {
            assert!(kesit(bozuk, 1, 16).is_err());
            assert!(genislik_izgarasi(bozuk, 1).is_empty());
        }
    }

    #[test]
    fn tamlik_girdinin_kendisi() {
        let s = temel();
        let tam = kesit(s, s.n_layers, s.d_model).unwrap_or_else(|_| panic!("kesit"));
        assert_eq!(tam.spec, s, "tam kesit girdi spec'i olmali");
        assert_eq!(tam.spec.parametre_sayisi(), s.parametre_sayisi());
        for d_ff in [1, 16, 63, 64, 256, usize::MAX] {
            let dar = Spec { d_ff, ..s };
            assert_eq!(
                kesit(dar, dar.n_layers, dar.d_model).map(|k| k.spec),
                Ok(dar)
            );
        }
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
        let kaynak_p = Parametreler::belirgin_doldur(s, 20260927);
        let tam = kesit(s, s.n_layers, s.d_model)
            .expect("tam kesit")
            .agirliklari_al(s, &kaynak_p, usize::MAX)
            .expect("tam agirlik");
        let tam_ozdes = usize::from(kaynak_p.bloklar().into_iter().zip(tam.bloklar()).all(
            |(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
        ));
        let mut kosan = 0usize;
        for derinlik in 1..=s.n_layers {
            for genislik in [16, 32, 64] {
                if let Ok(k) = kesit(s, derinlik, genislik) {
                    let pp = k
                        .agirliklari_al(s, &kaynak_p, usize::MAX)
                        .expect("agirlik kesiti");
                    let (kayip, _) = ileri_ve_geri(k.spec, &pp, &girdi, &hedef);
                    if kayip.is_finite() && kayip > 0.0 {
                        kosan += 1;
                    }
                }
            }
        }
        println!(
            "kesit | derinlik={} genislik={} d_k={} izgara={} kosan={} parametre_en_az={} parametre_tam={} parametre_genislik_en_az={} tam_ozdes={}",
            s.n_layers,
            s.d_model,
            s.d_k(),
            izgara.len(),
            kosan,
            ilk,
            s.parametre_sayisi(),
            en_dar,
            tam_ozdes
        );
    }

    fn agirlik_ornegi() -> Spec {
        Spec {
            vocab: 16,
            d_model: 8,
            n_layers: 3,
            n_heads: 4,
            n_kv_heads: 2,
            qkv_dokunus: 2,
            qk_norm: true,
            d_ff: 12,
            max_seq_len: 8,
        }
    }

    fn koordinatli(s: Spec) -> Parametreler {
        let mut p = Parametreler::sifir(s);
        for (b, blok) in p.bloklar_mut().into_iter().enumerate() {
            for (i, v) in blok.iter_mut().enumerate() {
                *v = (b * 10000 + i) as f64;
            }
        }
        p
    }

    #[test]
    fn tam_agirlik_kesiti_bit_ozdes() {
        let s = agirlik_ornegi();
        let mut p = koordinatli(s);
        p.embedding[0] = -0.0;
        let k = kesit(s, s.n_layers, s.d_model).expect("kesit");
        let q = k.agirliklari_al(s, &p, usize::MAX).expect("agirlik");
        for (a, b) in p.bloklar().into_iter().zip(q.bloklar()) {
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }

    #[test]
    fn matris_satir_adimi_duz_prefix_degil() {
        let s = agirlik_ornegi();
        let p = koordinatli(s);
        let k = kesit(s, 2, 4).expect("kesit");
        let q = k.agirliklari_al(s, &p, usize::MAX).expect("agirlik");
        assert_eq!(q.embedding[4], p.embedding[8]);
        assert_ne!(q.embedding[4], p.embedding[4]);
        assert_eq!(q.wq[4], p.wq[8]);
        assert_eq!(q.wq[16], p.wq[64]);
        assert_eq!(q.wk[4], p.wk[8]);
        assert_eq!(q.w1[4], p.w1[8]);
        assert_eq!(q.w2[6], p.w2[12]);
        assert_eq!(q.q_dokunus[4], p.q_dokunus[8]);
        assert_eq!(q.q_norm_olcek, p.q_norm_olcek[..4]);
        assert_eq!(q.lnf_olcek, p.lnf_olcek[..4]);
        assert_eq!(
            q.bloklar().iter().map(|b| b.len()).sum::<usize>(),
            k.spec.parametre_sayisi()
        );
    }

    #[test]
    fn her_blok_bozuk_sekli_tahsisten_once_reddeder() {
        let s = agirlik_ornegi();
        let p = koordinatli(s);
        let k = kesit(s, 2, 4).expect("kesit");
        for (b, ad) in Parametreler::blok_adlari().iter().enumerate() {
            // Alanlar public oldugundan bozuk checkpoint parcasi kurulabilir.
            let mut q = p.clone();
            match b {
                0 => q.embedding.pop(),
                1 => q.ln1_olcek.pop(),
                2 => q.ln1_sapma.pop(),
                3 => q.wq.pop(),
                4 => q.bq.pop(),
                5 => q.q_dokunus.pop(),
                6 => q.q_norm_olcek.pop(),
                7 => q.wk.pop(),
                8 => q.bk.pop(),
                9 => q.k_dokunus.pop(),
                10 => q.k_norm_olcek.pop(),
                11 => q.wv.pop(),
                12 => q.bv.pop(),
                13 => q.v_dokunus.pop(),
                14 => q.wo.pop(),
                15 => q.bo.pop(),
                16 => q.ln2_olcek.pop(),
                17 => q.ln2_sapma.pop(),
                18 => q.w1.pop(),
                19 => q.b1.pop(),
                20 => q.w2.pop(),
                21 => q.b2.pop(),
                22 => q.lnf_olcek.pop(),
                _ => q.lnf_sapma.pop(),
            };
            assert!(
                matches!(k.agirliklari_al(s, &q, 0),
                Err(KesitHatasi::AgirlikSekli { blok, .. }) if blok == b),
                "{ad}"
            );
        }
    }

    #[test]
    fn agirlik_bellek_tavani_tam_sinirda() {
        let s = agirlik_ornegi();
        let p = koordinatli(s);
        let k = kesit(s, 2, 4).expect("kesit");
        let gereken = k.spec.parametre_sayisi() * 8;
        assert!(k.agirliklari_al(s, &p, gereken).is_ok());
        assert_eq!(
            k.agirliklari_al(s, &p, gereken - 1),
            Err(KesitHatasi::BellekTavani {
                gereken,
                tavan: gereken - 1
            })
        );
        assert_eq!(p, koordinatli(s));
    }

    #[test]
    fn sahte_kesit_speci_reddedilir() {
        let s = agirlik_ornegi();
        let p = koordinatli(s);
        let mut k = kesit(s, 2, 4).expect("kesit");
        k.spec.vocab += 1;
        assert_eq!(
            k.agirliklari_al(s, &p, usize::MAX),
            Err(KesitHatasi::KesitTutarsiz)
        );
    }

    #[test]
    fn sonlu_olmayan_agirlik_hicbir_bloktan_gecmez() {
        let s = agirlik_ornegi();
        let p = koordinatli(s);
        let k = kesit(s, 2, 4).expect("kesit");
        for b in 0..24 {
            for deger in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut q = p.clone();
                q.bloklar_mut()[b][0] = deger;
                assert!(matches!(k.agirliklari_al(s, &q, usize::MAX),
                    Err(KesitHatasi::SonluOlmayanAgirlik { blok, konum: 0 }) if blok == b));
            }
        }
    }

    #[test]
    fn kapali_opsiyonlar_bos_kalir() {
        let s = Spec {
            qkv_dokunus: 0,
            qk_norm: false,
            ..agirlik_ornegi()
        };
        let p = koordinatli(s);
        let k = kesit(s, 1, 2).expect("kesit");
        let q = k.agirliklari_al(s, &p, usize::MAX).expect("agirlik");
        for b in [5, 6, 9, 10, 13] {
            assert!(q.bloklar()[b].is_empty());
        }
    }

    #[test]
    fn boyut_carpimi_tasmasi_panik_degil() {
        assert_eq!(
            tensor_boyutu([usize::MAX, 2, 1]),
            Err(KesitHatasi::BoyutTasmasi)
        );
        let s = Spec {
            vocab: usize::MAX,
            ..agirlik_ornegi()
        };
        let k = kesit(s, 1, 2).expect("spec");
        let p = koordinatli(agirlik_ornegi());
        assert_eq!(
            k.agirliklari_al(s, &p, usize::MAX),
            Err(KesitHatasi::BoyutTasmasi)
        );
    }

    #[test]
    fn mevcut_agirlik_kesitleri_ileri_geri_kosar() {
        let s = agirlik_ornegi();
        let p = Parametreler::mup_init(s, 7, 0.02);
        let once = p.clone();
        for derinlik in 1..=s.n_layers {
            for genislik in [2, 4, 6, 8] {
                let k = kesit(s, derinlik, genislik).expect("kesit");
                let q = k.agirliklari_al(s, &p, usize::MAX).expect("agirlik");
                let (kayip, g) = ileri_ve_geri(k.spec, &q, &[1, 2, 3], &[2, 3, 4]);
                assert!(kayip.is_finite() && kayip > 0.0);
                assert!(g
                    .bloklar()
                    .iter()
                    .flat_map(|b| b.iter())
                    .all(|v| v.is_finite()));
                assert!(g.embedding.iter().any(|v| *v != 0.0));
            }
        }
        assert_eq!(p, once);
    }

    #[test]
    fn tam_agirlik_kesiti_kayip_ve_gradyani_korur() {
        let s = agirlik_ornegi();
        let p = Parametreler::mup_init(s, 11, 0.02);
        let k = kesit(s, s.n_layers, s.d_model).expect("kesit");
        let q = k.agirliklari_al(s, &p, usize::MAX).expect("agirlik");
        let (a, ga) = ileri_ve_geri(s, &p, &[1, 2, 3], &[2, 3, 4]);
        let (b, gb) = ileri_ve_geri(s, &q, &[1, 2, 3], &[2, 3, 4]);
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(ga, gb);
    }
}
