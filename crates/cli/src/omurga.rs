//! # omurga - the encoder backbone, as a command
//!
//! Five verbs, each one printing a number that would otherwise only exist
//! inside a test:
//!
//! - `plan` prints the attention schedule: which layers see the whole sequence
//!   and which see a window. The rule is one integer, and a rule that can only
//!   be recovered by instrumenting a forward pass is a rule nobody checks.
//! - `sayim` prints the parameter accounting twice - once from the tensor
//!   directory that exists and once from a closed formula that never looks at
//!   it - and then the two totals side by side. A single figure can be wrong in
//!   the same direction as whatever produced it.
//! - `donme` probes the rotary map: the declared pairing against the pairing
//!   measured back out of the implementation, the norm before and after, and the
//!   inner product at two different absolute positions with the same gap. The
//!   third one is the property the method exists for.
//! - `ileri` runs a forward pass and prints the per-position RMS profile. It
//!   does not say whether the profile is good; the `lubot-a1` family has an open
//!   finding about exactly this quantity and nothing here closes it.
//! - `birlesim` builds two differently seeded models, checks their shape
//!   signatures, averages them in parameter space and prints how far the average
//!   sits from each parent. This is the operation branch-and-merge needs, so it
//!   is measured rather than assumed to work.

use lubot_omurga::dikkat::{yumusak_azami, DikkatHatasi, Gqa};
use lubot_omurga::hadamard::{
    blok_bol, karisim, kron_uygula, silu, walsh, Hadamard, HadamardHatasi, HadamardSekli,
    CIKIS_DIAGONAL_INIT, KARISIM_TOHUMLARI, KARISIM_UST_YARI_OFSETI, KOSUL_INIT_STD, KOSUL_RANK,
};
use lubot_omurga::katman::{carp, gelu, kapili_ileri, KatmanHatasi, Norm};
use lubot_omurga::konum::{Eslesme, KonumHatasi, Rope};
use lubot_omurga::pencere::{Kapsam, PencereHatasi, Plan};
use lubot_omurga::sonda::{
    birim_rms, maskeli_yumusak_azami, Bas, Sonda, SondaHatasi, SondaSekli, GOMME_SICAKLIK_INIT,
    GOMME_YANLILIK_INIT, RMS_EPS, ROTA_KALIBRASYONU, SONDA_INIT_STD,
};
use lubot_omurga::{Omurga, OmurgaHatasi, TensorKaydi, Tohum, Yapilandirma};

fn hata(e: &OmurgaHatasi) -> String {
    format!("omurga: {e}")
}

fn deger_bul(args: &[String], ad: &str) -> Option<String> {
    args.iter()
        .position(|a| a == ad)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn bayrak(args: &[String], ad: &str) -> bool {
    args.iter().any(|a| a == ad)
}

fn sayi(args: &[String], ad: &str, varsayilan: usize) -> Result<usize, String> {
    match deger_bul(args, ad) {
        None => Ok(varsayilan),
        Some(v) => v.parse().map_err(|_| format!("{ad}: `{v}` is not a count")),
    }
}

/// The configuration the verbs work on, with the shape overridable.
fn yapilandirma(args: &[String]) -> Result<Yapilandirma, String> {
    let mut yap = Yapilandirma::kucuk_aday();
    yap.d_model = sayi(args, "--genislik", yap.d_model)?;
    yap.n_katman = sayi(args, "--katman", yap.n_katman)?;
    yap.n_sorgu_kafa = sayi(args, "--kafa", yap.n_sorgu_kafa)?;
    yap.n_kv_kafa = sayi(args, "--kv-kafa", yap.n_kv_kafa)?;
    yap.d_ff = sayi(args, "--dff", yap.d_ff)?;
    yap.vocab = sayi(args, "--sozluk", yap.vocab)?;
    yap.genel_periyot = sayi(args, "--periyot", yap.genel_periyot)?;
    yap.yerel_yaricap = sayi(args, "--yaricap", yap.yerel_yaricap)?;
    if deger_bul(args, "--eslesme").as_deref() == Some("komsu") {
        yap.eslesme = Eslesme::KomsuCift;
    }
    yap.dogrula().map_err(|e| hata(&e))?;
    Ok(yap)
}

fn kullanim() -> String {
    [
        "usage:",
        "  lubot omurga plan [--katman N] [--periyot P] [--yaricap R]",
        "  lubot omurga sayim [--genislik D] [--katman N] [--dff F] [--sozluk V]",
        "  lubot omurga donme [--kafa-boyutu D] [--eslesme komsu|yariya]",
        "  lubot omurga ileri [--jeton N] [--tohum S]",
        "  lubot omurga birlesim [--tohum S] [--tohum-b S]",
        "  lubot omurga hadamard [--genislik D] [--jeton N] [--tohum S] [--yariya-ayrik]",
        "  lubot omurga sonda [--seviye L] [--sonda K] [--sorgu Q] [--genislik D] [--jeton N]",
        "  lubot omurga parca",
        "",
        "shared shape flags: --genislik --katman --kafa --kv-kafa --dff --sozluk",
        "                    --periyot --yaricap --eslesme",
    ]
    .join("\n")
}

fn plan(args: &[String]) -> Result<(), String> {
    let yap = yapilandirma(args)?;
    let plan = Plan::periyodik(yap.n_katman, yap.genel_periyot, yap.yerel_yaricap)
        .map_err(|e: PencereHatasi| format!("plan: {e}"))?;
    println!(
        "katman: {} | periyot: {} | yaricap: {}",
        plan.n_katman(),
        plan.periyot(),
        plan.yaricap()
    );
    println!("ozet: {}", plan.ozet());
    println!(
        "genel katman: {} / {}",
        plan.genel_sayisi(),
        plan.n_katman()
    );
    for katman in 0..plan.n_katman() {
        let kapsam = plan.kapsam(katman).map_err(|e| format!("plan: {e}"))?;
        let etiket = match kapsam {
            Kapsam::Genel => "genel (tum dizi)".to_string(),
            Kapsam::Yerel { yaricap } => format!("yerel (+/-{yaricap})"),
        };
        println!("  katman {katman:>3}  {etiket}");
    }
    Ok(())
}

fn sayim(args: &[String]) -> Result<(), String> {
    let yap = yapilandirma(args)?;
    let model =
        Omurga::yeni(yap.clone(), sayi(args, "--tohum", 1)? as u64).map_err(|e| hata(&e))?;
    println!("imza: {}", model.sekil_imzasi());
    println!("tensor: {}", model.dizin().len());
    let mut gruplar: Vec<(&str, usize)> = Vec::new();
    for kayit in model.dizin() {
        let rol = if kayit.ad == "gomme" {
            "gomme"
        } else if kayit.ad.ends_with("_norm") {
            "norm"
        } else if kayit.ad.ends_with("w_giris") || kayit.ad.ends_with("w_cikis") {
            "mlp"
        } else {
            "dikkat"
        };
        match gruplar.iter_mut().find(|(ad, _)| *ad == rol) {
            Some((_, toplam)) => *toplam += TensorKaydi::uzunluk(kayit),
            None => gruplar.push((rol, TensorKaydi::uzunluk(kayit))),
        }
    }
    for (rol, toplam) in &gruplar {
        println!("  {rol:<8} {toplam:>12}");
    }
    let dizinden = model.param_sayisi();
    let formulden = Omurga::beklenen_param_sayisi(&yap).map_err(|e| hata(&e))?;
    println!("dizinden: {dizinden}");
    println!("formulden: {formulden}");
    println!(
        "uyum: {}",
        if dizinden == formulden {
            "iki bagimsiz sayim ayni"
        } else {
            "AYRISIYOR"
        }
    );
    Ok(())
}

fn donme(args: &[String]) -> Result<(), String> {
    let d_head = sayi(args, "--kafa-boyutu", 32)?;
    let eslesme = match deger_bul(args, "--eslesme").as_deref() {
        Some("komsu") => Eslesme::KomsuCift,
        _ => Eslesme::YariyaBolme,
    };
    let rope =
        Rope::yeni(d_head, 10_000.0, eslesme).map_err(|e: KonumHatasi| format!("donme: {e}"))?;
    println!(
        "kafa boyutu: {} | cift: {}",
        rope.d_head(),
        rope.cift_sayisi()
    );
    println!("beyan edilen eslesme: {}", rope.eslesme().ad());
    println!("olculen eslesme:      {}", rope.olculen_eslesme().ad());
    println!(
        "beyan ile uygulama: {}",
        if rope.eslesme() == rope.olculen_eslesme() {
            "ayni"
        } else {
            "CELISIYOR"
        }
    );

    let x: Vec<f32> = (0..d_head)
        .map(|i| ((i as f32) * 0.37).sin() + 0.11 * (i as f32))
        .collect();
    let norm_of = |v: &[f32]| -> f64 {
        v.iter()
            .map(|a| f64::from(*a) * f64::from(*a))
            .sum::<f64>()
            .sqrt()
    };
    let mut dondurulmus = x.clone();
    rope.uygula(&mut dondurulmus, 7)
        .map_err(|e| format!("donme: {e}"))?;
    println!("norm once: {:.9}", norm_of(&x));
    println!("norm sonra: {:.9}", norm_of(&dondurulmus));

    // The relative property: the same gap at two different absolute positions
    // must give the same inner product.
    let k: Vec<f32> = x.iter().rev().copied().collect();
    let mut olcumler: Vec<f64> = Vec::new();
    for taban in [0usize, 40] {
        let mut q = x.clone();
        let mut kk = k.clone();
        rope.uygula(&mut q, taban + 5)
            .map_err(|e| format!("donme: {e}"))?;
        rope.uygula(&mut kk, taban + 2)
            .map_err(|e| format!("donme: {e}"))?;
        olcumler.push(
            q.iter()
                .zip(kk.iter())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum(),
        );
    }
    println!("<q_5, k_2>:   {:.9}", olcumler[0]);
    println!("<q_45, k_42>: {:.9}", olcumler[1]);
    println!("fark: {:.3e}", (olcumler[0] - olcumler[1]).abs());
    Ok(())
}

fn ileri(args: &[String]) -> Result<(), String> {
    let yap = yapilandirma(args)?;
    let tohum = sayi(args, "--tohum", 1)? as u64;
    let n = sayi(args, "--jeton", 32)?;
    let model = Omurga::yeni(yap.clone(), tohum).map_err(|e| hata(&e))?;
    let jetonlar: Vec<u32> = (0..n).map(|i| ((i * 7 + 13) % yap.vocab) as u32).collect();
    let basladi = std::time::Instant::now();
    let durum = model.ileri(&jetonlar).map_err(|e| hata(&e))?;
    let sure = basladi.elapsed();
    println!("imza: {}", model.sekil_imzasi());
    println!("jeton: {} | gizli durum: {}", jetonlar.len(), durum.len());
    println!(
        "sure: {:.3} ms (bu makinede, tek kosu)",
        sure.as_secs_f64() * 1e3
    );
    let ikinci = model.ileri(&jetonlar).map_err(|e| hata(&e))?;
    println!(
        "determinizm: {}",
        if ikinci == durum {
            "iki kosu birebir ayni"
        } else {
            "AYRISIYOR"
        }
    );
    let profil = Omurga::konum_rms(&durum, yap.d_model);
    let en_kucuk = profil.iter().copied().fold(f32::INFINITY, f32::min);
    let en_buyuk = profil.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "rms profili: {en_kucuk:.6} -> {en_buyuk:.6} ({} konum)",
        profil.len()
    );
    println!("  ilk dort: {:?}", &profil[..profil.len().min(4)]);
    println!("not: bu sayi egitilmemis bir baslangictan olculdu; kalite iddiasi degildir.");
    Ok(())
}

fn birlesim(args: &[String]) -> Result<(), String> {
    let yap = yapilandirma(args)?;
    let a = Omurga::yeni(yap.clone(), sayi(args, "--tohum", 1)? as u64).map_err(|e| hata(&e))?;
    let b = Omurga::yeni(yap.clone(), sayi(args, "--tohum-b", 2)? as u64).map_err(|e| hata(&e))?;
    println!("a imzasi: {}", a.sekil_imzasi());
    println!("b imzasi: {}", b.sekil_imzasi());
    println!(
        "uyum: {}",
        if a.imza_uyumlu(&b) {
            "ayni sekil, birlesim tanimli"
        } else {
            "AYRI SEKIL"
        }
    );
    let birlesik = Omurga::ortala(&[&a, &b]).map_err(|e| hata(&e))?;
    let uzaklik = |sol: &Omurga, sag: &Omurga| -> f64 {
        sol.agirlik()
            .iter()
            .zip(sag.agirlik().iter())
            .map(|(p, q)| f64::from((p - q).abs()))
            .fold(0.0, f64::max)
    };
    println!("parametre: {}", birlesik.param_sayisi());
    println!("|ortalama - a| en buyuk: {:.9}", uzaklik(&birlesik, &a));
    println!("|ortalama - b| en buyuk: {:.9}", uzaklik(&birlesik, &b));
    println!("|a - b| en buyuk:        {:.9}", uzaklik(&a, &b));
    let durum = birlesik.ileri(&[1, 2, 3, 4]).map_err(|e| hata(&e))?;
    println!(
        "ortalama model ileri gecisi: {} deger, hepsi sonlu: {}",
        durum.len(),
        durum.iter().all(|v| v.is_finite())
    );
    println!("not: birlesim sekil uyumunu dogrular, kaliteyi dogrulamaz; kalite kapi isidir.");
    Ok(())
}

/// The primitives, one number each, so a change in any of them is visible
/// without reading a test.
fn parca(_args: &[String]) -> Result<(), String> {
    let birim: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let carpim =
        carp(&birim, 3, 3, &[2.0, 3.0, 5.0]).map_err(|e: KatmanHatasi| format!("parca: {e}"))?;
    println!("carp (birim matris): {carpim:?}");

    let norm = Norm::yeni(4, 1e-5).map_err(|e| format!("parca: {e}"))?;
    let normlanmis = norm
        .uygula(&[1.0, 2.0, 3.0, 10.0], &[1.0; 4])
        .map_err(|e| format!("parca: {e}"))?;
    let toplam: f32 = normlanmis.iter().sum();
    println!(
        "norm (bias yok, kazanc 1): toplam {toplam:.3e}, genislik {}",
        norm.genislik()
    );

    println!(
        "gelu: g(0)={:.6} g(1)={:.6} g(-1)={:.6} g(8)={:.6}",
        gelu(0.0),
        gelu(1.0),
        gelu(-1.0),
        gelu(8.0)
    );

    let kapili = kapili_ileri(&[0.25; 2 * 2 * 3], &[0.5; 3 * 2], 3, 2, &[1.0, 1.0, 1.0])
        .map_err(|e| format!("parca: {e}"))?;
    println!("kapili ileri (3 -> 2 -> 3): {kapili:?}");

    let mut skorlar = vec![1.0f32, 2.0, f32::NEG_INFINITY, 3.0];
    yumusak_azami(&mut skorlar, 0).map_err(|e: DikkatHatasi| format!("parca: {e}"))?;
    println!(
        "yumusak azami (biri maskeli): {skorlar:?} toplam {:.6}",
        skorlar.iter().sum::<f32>()
    );

    let gqa = Gqa::yeni(8, 2, 16).map_err(|e| format!("parca: {e}"))?;
    println!(
        "gqa: {} sorgu kafasi, {} kv kafasi, grup {}, olcek {:.6}",
        gqa.n_sorgu(),
        gqa.n_kv(),
        gqa.grup_boyutu(),
        gqa.olcek()
    );

    let mut uretec = Tohum::yeni(7);
    let ornek: Vec<f64> = (0..4).map(|_| uretec.normal()).collect();
    println!("tohum 7, ilk dort normal: {ornek:?}");
    println!("not: bu degerler tohuma baglidir ve her makinede aynidir.");
    Ok(())
}

/// `lubot omurga ...`
///
/// # Errors
///
/// A usage string when the verb is missing or unknown, and whatever the
/// backbone refuses with otherwise.
/// The Kronecker-Walsh conditioned feed-forward, counted and run.
///
/// Prints what the block costs against the dense layer it would replace, then
/// runs it once: the fresh-block identities (`c == 1`, zero maps to zero) are
/// printed as measurements rather than asserted in prose.
fn hadamard(args: &[String]) -> Result<(), String> {
    let genislik = sayi(args, "--genislik", 768)?;
    let jeton = sayi(args, "--jeton", 4)?;
    let tohum_no = sayi(args, "--tohum", 1)? as u64;
    let yariya_ayrik = bayrak(args, "--yariya-ayrik");

    let sekil = HadamardSekli::yeni(genislik).map_err(|e: HadamardHatasi| format!("sekil: {e}"))?;
    let (yogun, yapisal) = sekil.carp_tasarrufu();
    println!(
        "genislik: {} | dolgulu: {} | blok: {}x{} | kosul rank: {}",
        sekil.d_model, sekil.n, sekil.ba, sekil.bb, KOSUL_RANK
    );
    println!(
        "parametre: {} (formul) | yogun esdeger {} carpim, yapisal {} carpim",
        sekil.param_sayisi(),
        yogun,
        yapisal
    );

    let mut tohum = Tohum::yeni(tohum_no);
    let blok = Hadamard::yeni(genislik, &mut tohum, yariya_ayrik)
        .map_err(|e: HadamardHatasi| format!("kurulum: {e}"))?;
    println!(
        "tutulan parametre: {} | formulle ayni mi: {}",
        blok.tutulan_param_sayisi(),
        blok.tutulan_param_sayisi() == sekil.param_sayisi()
    );

    let x: Vec<f32> = (0..jeton * genislik)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();
    let c = blok.kosul(&x[..genislik]);
    let kosul_bir = c.iter().all(|v| v.to_bits() == 1.0f32.to_bits());
    println!("taze kosul vektoru tam olarak 1 mi: {kosul_bir} (init std {KOSUL_INIT_STD})");

    let y = blok
        .ileri(&x)
        .map_err(|e: HadamardHatasi| format!("ileri: {e}"))?;
    let enb = y.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let rms =
        (y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / (y.len() as f64)).sqrt();
    println!(
        "cikis: {} sayi | en buyuk |y| {:.6} | rms {:.6} | cikis diagonali {}",
        y.len(),
        enb,
        rms,
        CIKIS_DIAGONAL_INIT
    );

    let sifir = blok
        .ileri(&vec![0.0f32; genislik])
        .map_err(|e: HadamardHatasi| format!("ileri: {e}"))?;
    let sifir_kalir = sifir.iter().all(|v| v.to_bits() == 0.0f32.to_bits());
    println!("sifir girdi tam olarak sifir mi: {sifir_kalir}");

    let (ba, bb) = blok_bol(sekil.n);
    let h = walsh(ba);
    let dik: f32 = (0..ba).map(|k| h[k] * h[k]).sum();
    println!(
        "walsh {ba}x{ba}: ilk satir normu {dik:.6} | ikinci faktor {bb}x{bb} | silu(1) {:.6}",
        silu(1.0)
    );

    // Two orthogonal factors cannot change the length of what they transform,
    // so the Kronecker stage is measured by the one number that would move if
    // a factor were unnormalised: the norm.
    let hb = walsh(bb);
    let birim: Vec<f32> = (0..sekil.n)
        .map(|i| if i == 0 { 1.0 } else { 0.0 })
        .collect();
    let donusen = kron_uygula(&birim, &h, ba, &hb, bb);
    let norm = donusen
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    println!("kronecker asamasi normu korur mu: {norm:.6} (beklenen 1)");

    let p1 = karisim(sekil.n, KARISIM_TOHUMLARI.0, yariya_ayrik);
    let p2 = karisim(sekil.n, KARISIM_TOHUMLARI.1, yariya_ayrik);
    println!(
        "karisim: iki permutasyon ayri mi {} | yariya ayrik {} | ust yari ofseti {}",
        p1 != p2,
        yariya_ayrik,
        KARISIM_UST_YARI_OFSETI
    );
    println!("olculmedi: geri gecis yok, bu blok egitilmedi; M1 damgasi bekliyor");
    Ok(())
}

/// Probe pooling, counted and run.
///
/// The three properties that make the pooling usable are printed as
/// measurements: the pooled width does not follow the sequence length, a
/// shuffle does not move the answer, and a masked token is absent rather than
/// quiet.
fn sonda(args: &[String]) -> Result<(), String> {
    let seviye = sayi(args, "--seviye", 2)?;
    let sonda_sayisi = sayi(args, "--sonda", 4)?;
    let sorgu = sayi(args, "--sorgu", 4)?;
    let genislik = sayi(args, "--genislik", 16)?;
    let jeton = sayi(args, "--jeton", 6)?;

    let sekil = SondaSekli::yeni(seviye, sonda_sayisi, sorgu, genislik)
        .map_err(|e: SondaHatasi| format!("sekil: {e}"))?;
    println!(
        "seviye: {seviye} | seviye basina sonda: {sonda_sayisi} | sorgu: {sorgu} | genislik: {genislik}"
    );
    println!(
        "havuz genisligi: {} (dizi uzunlugundan bagimsiz) | havuz parametresi: {}",
        sekil.havuz_genisligi(),
        sekil.havuz_param_sayisi()
    );

    let mut tohum = Tohum::yeni(sayi(args, "--tohum", 1)? as u64);
    for bas in [Bas::Guven, Bas::YonSecimi, Bas::Gomme { genislik: 128 }] {
        let s =
            Sonda::yeni(sekil, bas, &mut tohum).map_err(|e: SondaHatasi| format!("bas: {e}"))?;
        println!(
            "  bas {:?}: cikis {} | yanlilik {} | parametre {} (formul) / {} (yurume) | ayni mi {}",
            bas,
            bas.cikis_genisligi(),
            bas.yanlilik_var(),
            sekil.param_sayisi(bas),
            s.tutulan_param_sayisi(),
            sekil.param_sayisi(bas) == s.tutulan_param_sayisi()
        );
    }

    let s = Sonda::yeni(sekil, Bas::YonSecimi, &mut tohum)
        .map_err(|e: SondaHatasi| format!("bas: {e}"))?;
    let hucre_genislik = seviye * genislik;
    let hucreler: Vec<f32> = (0..jeton * hucre_genislik)
        .map(|i| ((i % 13) as f32) * 0.11 - 0.6)
        .collect();
    let duz = s
        .havuzla(&hucreler, jeton, None, None)
        .map_err(|e: SondaHatasi| format!("havuz: {e}"))?;

    let mut karisik = vec![0.0f32; hucreler.len()];
    for yeni in 0..jeton {
        let eski = jeton - 1 - yeni;
        karisik[yeni * hucre_genislik..(yeni + 1) * hucre_genislik]
            .copy_from_slice(&hucreler[eski * hucre_genislik..(eski + 1) * hucre_genislik]);
    }
    let ters = s
        .havuzla(&karisik, jeton, None, None)
        .map_err(|e: SondaHatasi| format!("havuz: {e}"))?;
    let sira_farki = duz
        .iter()
        .zip(ters.iter())
        .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));

    let mut maske = vec![true; jeton];
    maske[jeton - 1] = false;
    let maskeli = s
        .havuzla(&hucreler, jeton, Some(&maske), None)
        .map_err(|e: SondaHatasi| format!("havuz: {e}"))?;
    let kisa = s
        .havuzla(
            &hucreler[..(jeton - 1) * hucre_genislik],
            jeton - 1,
            None,
            None,
        )
        .map_err(|e: SondaHatasi| format!("havuz: {e}"))?;
    let maske_tam = maskeli
        .iter()
        .zip(kisa.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());

    println!("havuz cikisi: {} sayi", duz.len());
    println!("jeton sirasi ters cevrilince en buyuk fark: {sira_farki:.3e} (beklenen 0)");
    println!("maskeli jeton tam olarak disarida mi: {maske_tam}");

    let tumu_kapali = vec![false; jeton];
    let red = s.havuzla(&hucreler, jeton, Some(&tumu_kapali), None);
    println!(
        "tum jetonlar maskeliyken: {}",
        match red {
            Err(e) => format!("reddedildi ({e})"),
            Ok(_) => "SAYI URETTI (beklenmiyor)".to_string(),
        }
    );

    let mut birim = vec![3.0f32, -4.0, 0.0, 12.0];
    birim_rms(&mut birim);
    let rms = (birim
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / (birim.len() as f64))
        .sqrt();
    let dagilim = maskeli_yumusak_azami(&[1.0, 50.0, 3.0], &[true, false, true])
        .map_err(|e: SondaHatasi| format!("yumusak azami: {e}"))?;
    println!(
        "birim rms: {rms:.6} (eps {RMS_EPS:e}) | maskeli agirlik: {:.6} (beklenen 0) | init std {SONDA_INIT_STD}",
        dagilim[1]
    );
    println!("rota kalibrasyonu: {ROTA_KALIBRASYONU:?} | gomme sicakligi {GOMME_SICAKLIK_INIT} yanliligi {GOMME_YANLILIK_INIT}");
    println!("olculmedi: geri gecis yok, hicbir bas egitilmedi");
    Ok(())
}

pub fn cmd_omurga(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("plan") => plan(&args[1..]),
        Some("sayim") => sayim(&args[1..]),
        Some("donme") => donme(&args[1..]),
        Some("ileri") => ileri(&args[1..]),
        Some("birlesim") => birlesim(&args[1..]),
        Some("hadamard") => hadamard(&args[1..]),
        Some("sonda") => sonda(&args[1..]),
        Some("parca") => parca(&args[1..]),
        _ => Err(kullanim()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg(parcalar: &[&str]) -> Vec<String> {
        parcalar.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn verb_yoksa_kullanim_doner() {
        assert!(cmd_omurga(&[]).unwrap_err().contains("usage:"));
        assert!(cmd_omurga(&arg(&["yok"])).unwrap_err().contains("usage:"));
    }

    #[test]
    fn plan_calisir() {
        cmd_omurga(&arg(&["plan", "--katman", "7", "--periyot", "3"])).unwrap();
    }

    #[test]
    fn hadamard_calisir() {
        assert!(hadamard(&arg(&["--genislik", "64", "--jeton", "2"])).is_ok());
    }

    #[test]
    fn hadamard_yariya_ayrik_calisir() {
        assert!(hadamard(&arg(&["--genislik", "32", "--yariya-ayrik"])).is_ok());
    }

    #[test]
    fn hadamard_sifir_genislik_reddedilir() {
        assert!(hadamard(&arg(&["--genislik", "0"])).is_err());
    }

    #[test]
    fn bayrak_yoksa_false() {
        assert!(!bayrak(&arg(&["--genislik", "8"]), "--yariya-ayrik"));
        assert!(bayrak(&arg(&["--yariya-ayrik"]), "--yariya-ayrik"));
    }

    #[test]
    fn sonda_calisir() {
        assert!(sonda(&arg(&["--seviye", "2", "--sonda", "2", "--jeton", "4"])).is_ok());
    }

    #[test]
    fn sonda_sifir_boyut_reddedilir() {
        assert!(sonda(&arg(&["--seviye", "0"])).is_err());
    }

    #[test]
    fn sayim_calisir() {
        cmd_omurga(&arg(&[
            "sayim",
            "--genislik",
            "32",
            "--katman",
            "2",
            "--dff",
            "64",
            "--sozluk",
            "128",
            "--kafa",
            "2",
            "--kv-kafa",
            "1",
        ]))
        .unwrap();
    }

    #[test]
    fn donme_iki_eslesmede_de_calisir() {
        cmd_omurga(&arg(&["donme"])).unwrap();
        cmd_omurga(&arg(&["donme", "--eslesme", "komsu"])).unwrap();
    }

    #[test]
    fn ileri_calisir() {
        cmd_omurga(&arg(&[
            "ileri",
            "--genislik",
            "32",
            "--katman",
            "2",
            "--dff",
            "64",
            "--sozluk",
            "128",
            "--kafa",
            "2",
            "--kv-kafa",
            "1",
            "--jeton",
            "8",
        ]))
        .unwrap();
    }

    #[test]
    fn birlesim_calisir() {
        cmd_omurga(&arg(&[
            "birlesim",
            "--genislik",
            "32",
            "--katman",
            "2",
            "--dff",
            "64",
            "--sozluk",
            "128",
            "--kafa",
            "2",
            "--kv-kafa",
            "1",
        ]))
        .unwrap();
    }

    #[test]
    fn parca_calisir() {
        cmd_omurga(&arg(&["parca"])).unwrap();
    }

    #[test]
    fn bozuk_sayi_reddedilir() {
        let hata = cmd_omurga(&arg(&["plan", "--katman", "cok"])).unwrap_err();
        assert!(hata.contains("--katman"), "{hata}");
    }

    #[test]
    fn bolunmez_genislik_reddedilir() {
        let hata = cmd_omurga(&arg(&["sayim", "--genislik", "18", "--kafa", "4"])).unwrap_err();
        assert!(hata.contains("omurga:"), "{hata}");
    }
}
