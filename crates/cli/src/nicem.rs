//! # nicem - the quantiser, as a command
//!
//! Four verbs, each one printing a number that would otherwise only exist
//! inside a test:
//!
//! - `kitap` solves the optimal codebook for a width and prints its levels,
//!   its predicted distortion, and how far the solution is from the two
//!   conditions an optimal quantiser must satisfy. `--ham` re-solves from
//!   scratch instead of taking the memoised answer, which is how the cache is
//!   checked against the thing it caches.
//! - `butce` prints what a tensor of a given size costs at every width. It is
//!   arithmetic, but it is the arithmetic a deployment decision is made on, and
//!   having to derive it by hand is how a wrong figure gets repeated.
//! - `dondur` measures what the rotation does to one group: energy before and
//!   after (it is orthogonal, so they must agree), and the peak-to-average
//!   ratio, which is the thing the rotation is actually there to reduce.
//! - `yarim` prints the range the group scale can represent, and what a given
//!   value becomes when stored in it. A group whose norm falls outside that
//!   range cannot be stored faithfully no matter how good the codebook is.

use std::fs;

use lubot_nicem::grup::EN_KUCUK_GRUP;
use lubot_nicem::hadamard::{enerji, wht, yogun_matris, HadamardHatasi};
use lubot_nicem::kodkitabi::{
    bolge_kutleleri, coz, coz_ham, coz_ucdeger, coz_ucdeger_ham, kosullari_sagliyor, Kodkitabi,
    KodkitabiHatasi,
};
use lubot_nicem::paket::{
    agirlik_basina_bit, agirlik_basina_bit_ucdeger, bayt_sayisi, ucdeger_bayt_sayisi,
    TRIT_BASINA_BAYT,
};
use lubot_nicem::yarim::{
    f32_to_yarim, yarim_to_f32, yuvarla, YARIM_EN_BUYUK, YARIM_EN_KUCUK_NORMAL,
    YARIM_EN_KUCUK_SUBNORMAL,
};
use lubot_nicem::{VARSAYILAN_BIT, VARSAYILAN_GRUP};

// The rotation refuses for exactly two reasons and both are the caller's
// doing, so the message says which layer produced it rather than leaving a
// bare sentence to be blamed on the file.
fn hadamard_hatasi(e: &HadamardHatasi) -> String {
    format!("donme: {e}")
}

fn kodkitabi_hatasi(e: &KodkitabiHatasi) -> String {
    format!("kod kitabi: {e}")
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

fn kullanim() -> String {
    [
        "usage:",
        "  lubot nicem kitap [--bit 1..8|ucdeger] [--ham]",
        "  lubot nicem butce --agirlik N [--grup G]",
        "  lubot nicem dondur --dosya <ham.f32> [--grup G] [--matris N]",
        "  lubot nicem yarim [--deger X]",
    ]
    .join("\n")
}

/// `lubot nicem ...`
///
/// # Errors
///
/// A usage string when the verb is missing or unknown, and whatever the
/// quantiser refuses with otherwise.
pub fn cmd_nicem(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("kitap") => kitap(&args[1..]),
        Some("butce") => butce(&args[1..]),
        Some("dondur") => dondur(&args[1..]),
        Some("yarim") => yarim(&args[1..]),
        _ => Err(kullanim()),
    }
}

/// Solve a codebook, memoised or from scratch.
fn kitabi_coz(bit: &str, ham: bool) -> Result<Kodkitabi, KodkitabiHatasi> {
    if bit == "ucdeger" || bit == "0" {
        return if ham {
            coz_ucdeger_ham()
        } else {
            coz_ucdeger()
        };
    }
    let b: u8 = bit.parse().unwrap_or(0);
    if ham {
        coz_ham(b)
    } else {
        coz(b)
    }
}

fn kitap(args: &[String]) -> Result<(), String> {
    let bit = deger_bul(args, "--bit").unwrap_or_else(|| "2".to_string());
    let ham = bayrak(args, "--ham");
    let kitap = kitabi_coz(&bit, ham).map_err(|e| kodkitabi_hatasi(&e))?;
    println!(
        "alfabe: {bit} | seviye: {} | tur: {} | cozum: {}",
        kitap.boyut(),
        kitap.tur(),
        if ham {
            "ham (onbelleksiz)"
        } else {
            "onbellekli"
        }
    );
    println!("bozulma: {:.9}", kitap.bozulma());
    println!("snr: {:.4} dB", kitap.snr_db());
    let seviyeler = kitap.seviyeler();
    println!("seviyeler:");
    for (i, s) in seviyeler.iter().enumerate() {
        println!("  {i:>3}  {s:+.9}");
    }
    // The two conditions an optimal scalar quantiser satisfies: every boundary
    // sits midway between its neighbouring levels, and every level sits at the
    // centroid of its own region. The residual is printed rather than asserted
    // because the number is the evidence; a pass/fail would hide how close it is.
    println!("kosul artigi: {:.3e}", kosullari_sagliyor(seviyeler));
    let kutleler = bolge_kutleleri(seviyeler);
    let toplam: f64 = kutleler.iter().sum();
    println!(
        "bolge kutleleri: {} bolge, toplam {:.12} (bire esit olmali)",
        kutleler.len(),
        toplam
    );
    let en_kucuk = kutleler.iter().copied().fold(f64::INFINITY, f64::min);
    println!("en seyrek bolge: {en_kucuk:.9}");
    let olcekli = kitap.birim_norm_icin(VARSAYILAN_GRUP);
    if let Some(ilk) = olcekli.first() {
        println!(
            "grup {VARSAYILAN_GRUP} icin olcekli ilk seviye: {ilk:+.9} (birim normlu gruba gore)"
        );
    }
    Ok(())
}

fn butce(args: &[String]) -> Result<(), String> {
    let agirlik = sayi(args, "--agirlik", 0)?;
    if agirlik == 0 {
        return Err("nicem butce needs --agirlik N (how many weights)".to_string());
    }
    let grup = sayi(args, "--grup", VARSAYILAN_GRUP)?;
    if grup < EN_KUCUK_GRUP {
        return Err(format!(
            "--grup: {grup} is below the smallest group the rotation is worth running on \
             ({EN_KUCUK_GRUP})"
        ));
    }
    let grup_sayisi = agirlik.div_ceil(grup);
    let olcek_bayt = grup_sayisi * 2;
    println!("agirlik: {agirlik} | grup: {grup} | grup sayisi: {grup_sayisi}");
    println!("olcek: {olcek_bayt} bayt (grup basina bir f16)");
    println!("alfabe   indeks bayt   toplam bayt   agirlik basina bit   f32'ye gore");
    let f32_bayt = agirlik * 4;
    for bit in 1..=8u8 {
        let indeks = bayt_sayisi(agirlik, bit);
        let toplam = indeks + olcek_bayt;
        #[allow(clippy::cast_precision_loss)]
        let oran = f32_bayt as f64 / toplam as f64;
        println!(
            "{:>6}   {indeks:>11}   {toplam:>11}   {:>18.4}   {oran:>10.2}x",
            format!("b{bit}"),
            agirlik_basina_bit(bit, grup)
        );
    }
    let indeks = ucdeger_bayt_sayisi(agirlik);
    let toplam = indeks + olcek_bayt;
    #[allow(clippy::cast_precision_loss)]
    let oran = f32_bayt as f64 / toplam as f64;
    println!(
        "{:>6}   {indeks:>11}   {toplam:>11}   {:>18.4}   {oran:>10.2}x",
        "uc",
        agirlik_basina_bit_ucdeger(grup)
    );
    println!(
        "ucdeger paketleme: {TRIT_BASINA_BAYT} trit / bayt; varsayilan alfabenin maliyeti \
         {VARSAYILAN_BIT} bit/agirlik"
    );
    Ok(())
}

fn dondur(args: &[String]) -> Result<(), String> {
    let grup = sayi(args, "--grup", VARSAYILAN_GRUP)?;
    if let Some(n) = deger_bul(args, "--matris") {
        let n: usize = n.parse().map_err(|_| "--matris: not a size".to_string())?;
        let m = yogun_matris(n).map_err(|e| hadamard_hatasi(&e))?;
        println!("H/sqrt({n}), satir satir (kendi tersi):");
        for satir in &m {
            let govde: Vec<String> = satir.iter().map(|v| format!("{v:+.4}")).collect();
            println!("  {}", govde.join(" "));
        }
        return Ok(());
    }
    let Some(yol) = deger_bul(args, "--dosya") else {
        return Err("nicem dondur needs --dosya <ham.f32> or --matris N".to_string());
    };
    let ham = fs::read(&yol).map_err(|e| format!("input refused ({yol}): {e}"))?;
    if ham.len() % 4 != 0 {
        return Err(format!("{yol}: not a whole number of f32 values"));
    }
    let agirlik: Vec<f32> = ham
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if agirlik.len() < grup {
        return Err(format!(
            "{yol} holds {} values, fewer than one group of {grup}",
            agirlik.len()
        ));
    }
    let mut en_kotu_once = 0.0f64;
    let mut en_kotu_sonra = 0.0f64;
    let mut en_buyuk_enerji_farki = 0.0f64;
    let mut sayac = 0usize;
    for parca in agirlik.chunks_exact(grup) {
        let once = enerji(parca);
        let mut calisma = parca.to_vec();
        wht(&mut calisma).map_err(|e| hadamard_hatasi(&e))?;
        let sonra = enerji(&calisma);
        en_buyuk_enerji_farki = en_buyuk_enerji_farki.max((once - sonra).abs());
        en_kotu_once = en_kotu_once.max(tepe_orani(parca));
        en_kotu_sonra = en_kotu_sonra.max(tepe_orani(&calisma));
        sayac += 1;
    }
    println!("dosya: {yol} | grup: {grup} | tam grup: {sayac}");
    println!(
        "enerji farki (en buyuk): {en_buyuk_enerji_farki:.3e} - ortogonal donusumde sifir olmali"
    );
    println!("tepe/ortalama orani, donmeden once: {en_kotu_once:.4}");
    println!("tepe/ortalama orani, donduktan sonra: {en_kotu_sonra:.4}");
    if en_kotu_sonra >= en_kotu_once {
        println!(
            "not: bu veride donme tepeyi dusurmedi. Donme her zaman kazandirmaz ve bu \
             gizlenmez; tek bir sivri vektor donduruldugunde butun koordinatlar esit \
             buyukluge gider ve bu iki bitlik alfabede daha kotu oturur."
        );
    }
    Ok(())
}

/// Largest absolute coordinate divided by the root-mean-square coordinate.
fn tepe_orani(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let tepe = x.iter().fold(0.0f64, |a, v| a.max(f64::from(v.abs())));
    #[allow(clippy::cast_precision_loss)]
    let rms = (enerji(x) / x.len() as f64).sqrt();
    if rms == 0.0 {
        0.0
    } else {
        tepe / rms
    }
}

fn yarim(args: &[String]) -> Result<(), String> {
    println!("f16 araligi, grup olceginin yasadigi yer:");
    println!("  en buyuk:             {YARIM_EN_BUYUK:e}");
    println!("  en kucuk normal:      {YARIM_EN_KUCUK_NORMAL:e}");
    println!("  en kucuk subnormal:   {YARIM_EN_KUCUK_SUBNORMAL:e}");
    if let Some(ham) = deger_bul(args, "--deger") {
        let x: f32 = ham
            .parse()
            .map_err(|_| format!("--deger: `{ham}` is not a number"))?;
        let kodlu = f32_to_yarim(x);
        let geri = yarim_to_f32(kodlu);
        println!("deger: {x:e}");
        println!("  f16 deseni: 0x{kodlu:04x}");
        println!("  geri okunan: {geri:e}");
        println!("  yuvarlama farki: {:e}", f64::from(geri) - f64::from(x));
        println!("  dogrudan yuvarlama: {:e}", yuvarla(x));
        if x.abs() > YARIM_EN_BUYUK {
            println!(
                "  uyari: bu deger f16 araliginin disinda; boyle bir grup normu olarak \
                 saklanamaz, kod kitabi ne kadar iyi olursa olsun."
            );
        }
    }
    Ok(())
}

#[cfg(test)]
// Exact comparisons are the point in these tests: a codebook that the cache
// changed, or a half-precision value that does not come back as itself, is a
// bug that a tolerance would hide.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_verb_prints_the_four_verbs_rather_than_choosing_one() {
        let hata = cmd_nicem(&[]).expect_err("refuses");
        for verb in ["kitap", "butce", "dondur", "yarim"] {
            assert!(hata.contains(verb), "{verb} missing from usage: {hata}");
        }
    }

    #[test]
    fn the_cached_codebook_and_the_uncached_one_are_the_same_answer() {
        // The whole point of the memo table: it must not change the result.
        let onbellekli = kitabi_coz("2", false).expect("solves");
        let ham = kitabi_coz("2", true).expect("solves");
        assert_eq!(onbellekli.seviyeler(), ham.seviyeler());
        let uc_onbellekli = kitabi_coz("ucdeger", false).expect("solves");
        let uc_ham = kitabi_coz("0", true).expect("solves");
        assert_eq!(uc_onbellekli.seviyeler(), uc_ham.seviyeler());
    }

    #[test]
    fn a_width_the_solver_does_not_have_is_refused_rather_than_approximated() {
        assert!(kitabi_coz("9", false).is_err());
        assert!(kitabi_coz("yarim", false).is_err());
    }

    #[test]
    fn the_budget_refuses_a_group_too_small_for_the_rotation_to_pay_for_itself() {
        let args = vec![
            "--agirlik".to_string(),
            "1024".to_string(),
            "--grup".to_string(),
            "8".to_string(),
        ];
        let hata = butce(&args).expect_err("refuses");
        assert!(hata.contains("smallest group"), "{hata}");
        assert!(butce(&[]).is_err(), "a budget with no size is not a budget");
    }

    #[test]
    fn a_refusal_names_the_layer_that_produced_it() {
        let e = yogun_matris(3).expect_err("not a power of two");
        assert!(hadamard_hatasi(&e).starts_with("donme: "));
        let e = coz(9).expect_err("outside 1..=8");
        assert!(kodkitabi_hatasi(&e).starts_with("kod kitabi: "));
    }

    #[test]
    fn the_peak_ratio_is_one_for_a_flat_vector_and_larger_for_a_spike() {
        let duz = vec![1.0f32; 16];
        assert!((tepe_orani(&duz) - 1.0).abs() < 1e-9);
        let mut sivri = vec![0.0f32; 16];
        sivri[3] = 1.0;
        assert!(
            (tepe_orani(&sivri) - 4.0).abs() < 1e-6,
            "{}",
            tepe_orani(&sivri)
        );
        assert_eq!(tepe_orani(&[]), 0.0);
    }

    #[test]
    fn the_scale_range_is_the_one_the_half_precision_format_actually_has() {
        // A value outside it cannot be stored as a group norm, so the numbers
        // the command prints are checked here rather than trusted.
        assert_eq!(yarim_to_f32(f32_to_yarim(YARIM_EN_BUYUK)), YARIM_EN_BUYUK);
        assert!(yarim_to_f32(f32_to_yarim(YARIM_EN_BUYUK * 2.0)).is_infinite());
        assert_eq!(yarim(&[]), Ok(()));
        assert!(yarim(&["--deger".to_string(), "abc".to_string()]).is_err());
    }
}
