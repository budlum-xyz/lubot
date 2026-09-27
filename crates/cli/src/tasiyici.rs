//! # tasiyici - the weight container, as a command
//!
//! Three verbs, and each one exists because a claim in the design needs a way
//! to be checked from outside the test suite:
//!
//! - `olc` quantises a real tensor and prints the error it cost. The
//!   compression ratio of a scheme is arithmetic and needs no command; the
//!   error is a measurement and does.
//! - `incele` opens a container and reports what is in it, rung by rung,
//!   with the honest bits-per-weight figure that includes the header and the
//!   alignment padding.
//! - `tavan` reads this machine's free memory, asks the ladder how deep it can
//!   go, and prints the one-line declaration an operator publishes. If memory
//!   is supplied by hand instead of measured, the line says so in capitals.
//!
//! The input format for `olc` is deliberately the dullest one possible: a flat
//! little-endian `f32` file. A tensor arriving in someone else's serialisation
//! format would make this command a parser for that format, and the parser
//! would then be the thing under test rather than the quantiser.

use std::fs;
use std::path::Path;

use lubot_nicem::{Genislik, NicemHatasi, Nicemleyici};
use lubot_tasiyici::bicim::{genislik_kodu, hizala};
use lubot_tasiyici::{
    BicimHatasi, Kademe, Kapsayici, Merdiven, MerdivenHatasi, Tavan, Yazici, YaziciHatasi,
    BASLIK_BAYT, HIZA, IMZA, SURUM, VARSAYILAN_PAY,
};

// Four layers can refuse, and a bare message does not say which one did. The
// layer is part of the diagnosis: a quantiser refusal means the input was
// wrong, a container refusal means the file was, and a ladder refusal means
// the machine is.
fn nicem_hatasi(e: &NicemHatasi) -> String {
    format!("nicemleyici: {e}")
}

fn bicim_hatasi(e: &BicimHatasi) -> String {
    format!("kapsayici: {e}")
}

fn yazici_hatasi(e: &YaziciHatasi) -> String {
    format!("yazici: {e}")
}

fn merdiven_hatasi(e: &MerdivenHatasi) -> String {
    format!("merdiven: {e}")
}

/// One rung of the ladder, as a line.
fn kademe_satiri(k: &Kademe) -> String {
    format!(
        "{:>6}  {:>6}  {:>12}  {:>10}",
        k.no, k.tensor, k.agirlik, k.bayt
    )
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

/// `--bit 0` means ternary: zero bits of sign-magnitude, three levels.
fn genislik_coz(args: &[String]) -> Result<Genislik, String> {
    let ham = deger_bul(args, "--bit").unwrap_or_else(|| "2".to_string());
    if ham == "ucdeger" || ham == "0" {
        return Ok(Genislik::Ucdeger);
    }
    let b: u8 = ham
        .parse()
        .map_err(|_| format!("--bit: `{ham}` is not 1..=8 or `ucdeger`"))?;
    if !(1..=8).contains(&b) {
        return Err(format!("--bit: {b} is outside 1..=8"));
    }
    Ok(Genislik::Bit(b))
}

/// Read a flat little-endian `f32` file.
fn f32_oku(yol: &str) -> Result<Vec<f32>, String> {
    let ham = fs::read(yol).map_err(|e| format!("input refused ({yol}): {e}"))?;
    if ham.len() % 4 != 0 {
        return Err(format!(
            "{yol}: {} bytes is not a whole number of f32 values",
            ham.len()
        ));
    }
    Ok(ham
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Usable memory on this machine, in bytes, measured rather than guessed.
///
/// `MemAvailable` and not `MemFree`: free memory excludes the page cache, which
/// the kernel will hand back under pressure, so `MemFree` understates what a
/// reader may have and would produce a ceiling that is too low to be useful.
/// Returns `None` on any machine that does not publish the file, and the caller
/// must then say the ceiling was not measured rather than invent one.
fn olculen_bellek() -> Option<u64> {
    let metin = fs::read_to_string("/proc/meminfo").ok()?;
    for satir in metin.lines() {
        if let Some(kalan) = satir.strip_prefix("MemAvailable:") {
            let kb: u64 = kalan.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// `lubot tasiyici ...`
///
/// # Errors
///
/// Any refusal from the quantiser, the container or the ladder, and a usage
/// string when the subcommand is missing or unknown.
pub fn cmd_tasiyici(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("olc") => olc(&args[1..]),
        Some("incele") => incele(&args[1..]),
        Some("tavan") => tavan(&args[1..]),
        _ => Err(kullanim()),
    }
}

fn kullanim() -> String {
    [
        "usage:",
        "  lubot tasiyici olc --dosya <ham.f32> --son-eksen N [--bit 1..8|ucdeger] [--grup G] [--yaz f.lubotncm] [--ad t] [--kademe k]",
        "  lubot tasiyici incele --dosya <f.lubotncm> [--dogrula]",
        "  lubot tasiyici tavan --dosya <f.lubotncm> [--bellek BAYT] [--pay 0.66]",
    ]
    .join("\n")
}

fn olc(args: &[String]) -> Result<(), String> {
    let Some(yol) = deger_bul(args, "--dosya") else {
        return Err("tasiyici olc needs --dosya <ham.f32>".to_string());
    };
    let agirlik = f32_oku(&yol)?;
    let son_eksen = sayi(args, "--son-eksen", 0)?;
    if son_eksen == 0 {
        return Err("tasiyici olc needs --son-eksen N (the reduction axis)".to_string());
    }
    let grup = sayi(args, "--grup", lubot_nicem::VARSAYILAN_GRUP)?;
    let genislik = genislik_coz(args)?;
    let nicemleyici = Nicemleyici::yeni(genislik, grup).map_err(|e| nicem_hatasi(&e))?;
    let tensor = nicemleyici
        .nicemle(&agirlik, son_eksen)
        .map_err(|e| nicem_hatasi(&e))?;
    let olcum = tensor.olc(&agirlik).map_err(|e| nicem_hatasi(&e))?;
    println!("dosya: {yol}");
    println!(
        "sekil: {} satir x {} sutun = {} agirlik",
        tensor.satir(),
        tensor.son_eksen(),
        agirlik.len()
    );
    println!("alfabe: {} | grup: {grup}", genislik.ad());
    println!("bagil hata: {:.6}", olcum.bagil_hata);
    println!(
        "snr: {:.2} dB (kod kitabi {:.2} dB)",
        olcum.snr_db, olcum.kitap_snr_db
    );
    println!("en buyuk sapma: {:.6}", olcum.en_buyuk_sapma);
    println!("agirlik basina bit: {:.4}", olcum.agirlik_basina_bit);
    println!("bayt: {} (f32'ye gore {:.2}x)", olcum.bayt, olcum.oran);
    println!(
        "calisma alani: {} f32 (model boyundan bagimsiz)",
        tensor.en_buyuk_calisma_alani()
    );
    if let Some(cikti) = deger_bul(args, "--yaz") {
        let ad = deger_bul(args, "--ad").unwrap_or_else(|| "tensor".to_string());
        let kademe: u8 = deger_bul(args, "--kademe")
            .unwrap_or_else(|| "0".to_string())
            .parse()
            .map_err(|_| "--kademe: not a rung number".to_string())?;
        let mut yazici = Yazici::yeni();
        yazici
            .ekle(&ad, kademe, &tensor)
            .map_err(|e| yazici_hatasi(&e))?;
        let baytlar = yazici.bayt();
        if let Some(ust) = Path::new(&cikti).parent() {
            if !ust.as_os_str().is_empty() {
                fs::create_dir_all(ust).map_err(|e| e.to_string())?;
            }
        }
        fs::write(&cikti, &baytlar).map_err(|e| format!("write refused ({cikti}): {e}"))?;
        println!("yazildi: {cikti} ({} bayt)", baytlar.len());
    }
    Ok(())
}

fn incele(args: &[String]) -> Result<(), String> {
    let Some(yol) = deger_bul(args, "--dosya") else {
        return Err("tasiyici incele needs --dosya <f.lubotncm>".to_string());
    };
    let ham = fs::read(&yol).map_err(|e| format!("container refused ({yol}): {e}"))?;
    let kapsayici = Kapsayici::ac(&ham).map_err(|e| bicim_hatasi(&e))?;
    println!("dosya: {yol} ({} bayt)", ham.len());
    println!(
        "bicim: {} s{SURUM} | baslik {BASLIK_BAYT} bayt | blok hizasi {HIZA} bayt",
        String::from_utf8_lossy(IMZA)
    );
    println!(
        "tensor: {} | agirlik: {} | agirlik basina bit: {:.4} (baslik ve dolgu dahil)",
        kapsayici.kayitlar().len(),
        kapsayici.agirlik(),
        kapsayici.agirlik_basina_bit(ham.len())
    );
    let merdiven = Merdiven::kur(&kapsayici).map_err(|e| e.to_string())?;
    println!("kademe  tensor      agirlik        bayt");
    for k in merdiven.kademeler() {
        println!("{}", kademe_satiri(k));
    }
    for kayit in kapsayici.kayitlar() {
        println!(
            "  {} kademe {} {} (kod {}) grup {} sekil {}x{} {} bayt",
            kayit.ad,
            kayit.kademe,
            kayit.genislik.ad(),
            genislik_kodu(kayit.genislik),
            kayit.grup,
            kayit.satir,
            kayit.son_eksen,
            kayit.bayt()
        );
    }
    let talep = kapsayici.talep_edilen_bayt();
    let hizali: u64 = kapsayici
        .kayitlar()
        .iter()
        .map(|k| {
            let olcek = hizala(usize::try_from(k.olcek_bayt).unwrap_or(usize::MAX));
            let yuk = hizala(usize::try_from(k.yuk_bayt).unwrap_or(usize::MAX));
            (olcek + yuk) as u64
        })
        .sum();
    println!(
        "dolgu: {} bayt (blok basina {HIZA} bayta hizalamanin bedeli)",
        hizali.saturating_sub(talep)
    );
    if bayrak(args, "--dogrula") {
        kapsayici.dogrula().map_err(|e| bicim_hatasi(&e))?;
        println!("ozet: tuttu (butun yuk okundu)");
    } else {
        println!("ozet: bakilmadi (--dogrula ile tam gecis yapilir)");
    }
    Ok(())
}

fn tavan(args: &[String]) -> Result<(), String> {
    let Some(yol) = deger_bul(args, "--dosya") else {
        return Err("tasiyici tavan needs --dosya <f.lubotncm>".to_string());
    };
    let ham = fs::read(&yol).map_err(|e| format!("container refused ({yol}): {e}"))?;
    let kapsayici = Kapsayici::ac(&ham).map_err(|e| bicim_hatasi(&e))?;
    let merdiven = Merdiven::kur(&kapsayici).map_err(|e| merdiven_hatasi(&e))?;
    let pay: f64 = match deger_bul(args, "--pay") {
        None => VARSAYILAN_PAY,
        Some(v) => v
            .parse()
            .map_err(|_| format!("--pay: `{v}` is not a fraction"))?,
    };
    let tavan = match deger_bul(args, "--bellek") {
        Some(v) => {
            let bayt: u64 = v
                .parse()
                .map_err(|_| format!("--bellek: `{v}` is not a byte count"))?;
            Tavan::beyandan(bayt, "--bellek ile verildi")
        }
        None => match olculen_bellek() {
            Some(bayt) => Tavan::olcumden(bayt, pay, "/proc/meminfo MemAvailable"),
            None => {
                return Err(
                    "this machine does not publish /proc/meminfo, so the ceiling cannot be \
                     measured here; pass --bellek <bayt> and the report will say it was declared"
                        .to_string(),
                )
            }
        },
    };
    let secim = merdiven.sec(&tavan).map_err(|e| merdiven_hatasi(&e))?;
    println!("{}", merdiven.beyan(&tavan, &secim));
    println!(
        "yerlesik agirlik: {} / {}",
        secim.yerlesik_agirlik, secim.toplam_agirlik
    );
    if secim.tam() {
        println!("akan kademe: yok, dosyanin tamami bellekte");
    } else {
        println!(
            "akan kademe: {:?} (dosyada kalir, gerektikce grup grup okunur; suresi olculmedi)",
            secim.akan_kademe
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_subcommand_prints_the_three_verbs_rather_than_doing_something() {
        let hata = cmd_tasiyici(&[]).expect_err("refuses");
        assert!(hata.contains("olc"), "{hata}");
        assert!(hata.contains("incele"), "{hata}");
        assert!(hata.contains("tavan"), "{hata}");
        assert!(cmd_tasiyici(&["bilinmeyen".to_string()]).is_err());
    }

    #[test]
    fn the_width_flag_accepts_the_alphabets_the_quantiser_has_and_no_others() {
        let arg = |v: &str| vec!["--bit".to_string(), v.to_string()];
        assert_eq!(genislik_coz(&arg("2")), Ok(Genislik::Bit(2)));
        assert_eq!(genislik_coz(&arg("8")), Ok(Genislik::Bit(8)));
        assert_eq!(genislik_coz(&arg("ucdeger")), Ok(Genislik::Ucdeger));
        assert_eq!(genislik_coz(&arg("0")), Ok(Genislik::Ucdeger));
        assert!(genislik_coz(&arg("9")).is_err());
        assert!(genislik_coz(&arg("yarim")).is_err());
        // The default is the one the crate documents, not a fresh opinion.
        assert_eq!(genislik_coz(&[]), Ok(Genislik::Bit(2)));
    }

    #[test]
    fn a_file_that_is_not_a_whole_number_of_floats_is_refused_not_truncated() {
        let dizin = std::env::temp_dir().join("lubot-tasiyici-cli-test");
        fs::create_dir_all(&dizin).expect("temp dir");
        let yol = dizin.join("kirik.f32");
        fs::write(&yol, [1u8, 2, 3, 4, 5]).expect("write");
        let yol = yol.to_string_lossy().to_string();
        let hata = f32_oku(&yol).expect_err("refuses");
        assert!(hata.contains("whole number"), "{hata}");
        fs::remove_file(&yol).ok();
    }

    #[test]
    fn floats_round_trip_through_the_dullest_possible_input_format() {
        let dizin = std::env::temp_dir().join("lubot-tasiyici-cli-test");
        fs::create_dir_all(&dizin).expect("temp dir");
        let yol = dizin.join("iki.f32");
        let mut ham = Vec::new();
        for v in [1.5f32, -0.25, 7.0] {
            ham.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&yol, &ham).expect("write");
        let okunan = f32_oku(&yol.to_string_lossy()).expect("reads");
        assert_eq!(okunan, vec![1.5, -0.25, 7.0]);
        fs::remove_file(&yol).ok();
    }

    #[test]
    fn a_measured_ceiling_is_not_invented_when_the_machine_will_not_say() {
        // Whatever this machine is, the reading is either a real byte count or
        // nothing at all. There is no third answer, and no default.
        // No /proc/meminfo here means the command refuses; it never guesses.
        if let Some(bayt) = olculen_bellek() {
            assert!(bayt > 0, "MemAvailable read as zero");
        }
    }

    #[test]
    fn a_refusal_says_which_layer_refused() {
        // "grup 0" alone does not tell an operator whether their file, their
        // flags or their machine is the problem.
        let e = Nicemleyici::yeni(Genislik::Bit(2), 3).expect_err("refuses");
        assert!(nicem_hatasi(&e).starts_with("nicemleyici: "));
        assert!(bicim_hatasi(&BicimHatasi::ImzaYok).starts_with("kapsayici: "));
        assert!(yazici_hatasi(&YaziciHatasi::AdBos).starts_with("yazici: "));
        assert!(merdiven_hatasi(&MerdivenHatasi::TabanYok).starts_with("merdiven: "));
    }

    #[test]
    fn counts_fall_back_to_their_default_but_a_bad_count_is_refused() {
        let args = vec!["--grup".to_string(), "64".to_string()];
        assert_eq!(sayi(&args, "--grup", 128), Ok(64));
        assert_eq!(sayi(&[], "--grup", 128), Ok(128));
        let kotu = vec!["--grup".to_string(), "yarim".to_string()];
        assert!(sayi(&kotu, "--grup", 128).is_err());
    }
}
