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
use lubot_omurga::katman::{carp, gelu, kapili_ileri, KatmanHatasi, Norm};
use lubot_omurga::konum::{Eslesme, KonumHatasi, Rope};
use lubot_omurga::pencere::{Kapsam, PencereHatasi, Plan};
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
pub fn cmd_omurga(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("plan") => plan(&args[1..]),
        Some("sayim") => sayim(&args[1..]),
        Some("donme") => donme(&args[1..]),
        Some("ileri") => ileri(&args[1..]),
        Some("birlesim") => birlesim(&args[1..]),
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
