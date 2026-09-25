//! # sertleme - the hardening gate, as a command
//!
//! `lubot sertleme` reports what the hardening checks see on this machine and
//! prints the running binary's digest. `--zorla` refuses the run when a heavy
//! finding is present; `--beklenen-sha <hex>` verifies an artifact against a
//! digest given from outside, `--dosya <yol>` saying *which* artifact (the
//! running executable by default).
//!
//! The cheap group is run **twice** - once at the start and once after the
//! report is built - because a debugger attached while the program is already
//! running is the case a single start-up check cannot see. The deeper group
//! costs milliseconds, so it is opt-in through `--derin`; a check that is paid
//! on every run is a check that gets deleted from every run.
//!
//! A digest mismatch is not a footnote. When `--beklenen-sha` is given and the
//! measured digest does not match, the mismatch enters the same finding list
//! every other check writes to, at the same weight as a debugger: a modified
//! binary is the failure the integrity layer exists for, and a check whose
//! result only reaches the printed page is a check that changes nothing.

use std::path::Path;

use lubot_sertlestirme::butunluk::{self, OzetHatasi};
use lubot_sertlestirme::izler::{self, Bulgu, Katman};
use lubot_sertlestirme::kapi::{self, Kip, Rapor};

/// Dispatches the `sertleme` command.
pub fn cmd_sertleme(args: &[String]) -> Result<(), String> {
    let zorla = args.iter().any(|a| a == "--zorla");
    let derin = args.iter().any(|a| a == "--derin");
    let ayrinti = args.iter().any(|a| a == "--ayrinti");
    let kip = if zorla { Kip::Zorla } else { Kip::Bildir };
    if args.iter().any(|a| a == "--panik-kur") {
        kapi::panik_kancasi_kur();
    }
    let baslangic = izler::hafif_kontrol();
    let mut bulgular: Vec<Bulgu> = if derin {
        izler::derin_kontrol()
    } else {
        izler::hafif_kontrol()
    };
    // The integrity finding joins the same list the other checks write to. Kept
    // apart, the code that decides would not see it and `--zorla` would wave a
    // modified binary through.
    let mut butunluk_metni = String::new();
    if let Some(beklenen) = deger(args, "--beklenen-sha") {
        let dosya = deger(args, "--dosya");
        let (metin, bulgu) = butunluk_md(&beklenen, dosya.as_deref());
        butunluk_metni = metin;
        if let Some(bulgu) = bulgu {
            bulgular.push(bulgu);
        }
    }
    let rapor = kapi::topla(kip, bulgular);
    let kapanis = izler::hafif_kontrol();
    let mut metin = kapi::rapor_md(&rapor);
    metin.push_str(&format!(
        "\nGiris kontrolu {} bulgu, cikis kontrolu {} bulgu uretti (kontroller tek noktada degil).\n",
        baslangic.len(),
        kapanis.len()
    ));
    if ayrinti {
        metin.push_str(&katman_tablosu(&rapor));
    }
    metin.push_str(&butunluk_metni);
    print!("{metin}");
    if kapi::durdurulmali(&rapor) {
        return Err(kapi::zorla_mesaji());
    }
    Ok(())
}

/// Findings counted per layer: which family of check is doing the talking.
///
/// A total finding count says something was seen; the layer split says *what
/// kind of thing*, which is the difference between "a container" and "someone
/// is attached to this process".
fn katman_tablosu(rapor: &Rapor) -> String {
    const KATMANLAR: [Katman; 4] = [
        Katman::HataAyiklayici,
        Katman::Zamanlama,
        Katman::SanalOrtam,
        Katman::Ortam,
    ];
    let mut c = String::from("\n| katman | bulgu |\n|---|---:|\n");
    for katman in KATMANLAR {
        let grup: Vec<&Bulgu> = kapi::katman_bulgulari(rapor, katman);
        c.push_str(&format!("| {} | {} |\n", katman.ad(), grup.len()));
    }
    c
}

/// The digest section, and the finding a mismatch produces.
///
/// `--dosya` exists so the same command can check an artifact that is *not*
/// running - a copied build, say - through [`butunluk::dogrula`]; without it the
/// digest is the running executable's own, taken from [`butunluk::kendi_ozeti`].
/// The second value is the finding: `Some` only when the digest was measured
/// and disagreed, because "could not measure" is not evidence of tampering.
fn butunluk_md(beklenen: &str, dosya: Option<&str>) -> (String, Option<Bulgu>) {
    let olcum = match dosya {
        Some(yol) => butunluk::dogrula(Path::new(yol), beklenen),
        None => butunluk::kendi_ozeti().and_then(|olculen| {
            if olculen == beklenen.trim().to_ascii_lowercase() {
                Ok(olculen)
            } else {
                Err(OzetHatasi::OzetUyusmadi {
                    beklenen: beklenen.trim().to_string(),
                    olculen,
                })
            }
        }),
    };
    match olcum {
        Ok(olculen) => (format!("\nButunluk: dogrulandi (`{olculen}`).\n"), None),
        Err(OzetHatasi::OzetUyusmadi { beklenen, olculen }) => (
            format!("\nButunluk: UYUSMADI\n\n- beklenen: `{beklenen}`\n- olculen: `{olculen}`\n"),
            Some(Bulgu {
                katman: Katman::Ortam,
                aciklama: "ikilinin sha256 ozeti beklenenle uyusmadi".to_string(),
                agirlik: 3,
            }),
        ),
        Err(digeri) => (format!("\nButunluk: olculemedi ({digeri:?}).\n"), None),
    }
}

/// The value after `--ad`, if the flag is present with a value.
fn deger(args: &[String], ad: &str) -> Option<String> {
    let sira = args.iter().position(|a| a == ad)?;
    args.get(sira + 1).cloned()
}

/// A one-line summary, for the `durum` command.
#[must_use]
pub fn sertleme_ozeti(rapor: &Rapor) -> String {
    format!(
        "sertlestirme: {} bulgu, en agir {} (/{}), kip {}",
        rapor.bulgular.len(),
        rapor.en_agir(),
        kapi::ESIGIR_AGIRLIK,
        match rapor.kip {
            Kip::Bildir => "bildir",
            Kip::Zorla => "zorla",
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_and_values_are_read() {
        let args = vec![
            "--zorla".to_string(),
            "--beklenen-sha".to_string(),
            "ab".to_string(),
        ];
        assert_eq!(deger(&args, "--beklenen-sha"), Some("ab".to_string()));
        assert_eq!(deger(&args, "--yok"), None);
    }

    #[test]
    fn the_summary_states_the_weight_and_the_mode() {
        let rapor = Rapor {
            kip: Kip::Zorla,
            bulgular: Vec::new(),
            ozet: None,
        };
        let ozet = sertleme_ozeti(&rapor);
        assert!(ozet.contains("zorla"));
        assert!(ozet.contains("en agir 0"));
    }

    #[test]
    fn a_wrong_digest_is_reported_and_becomes_a_heavy_finding() {
        let (metin, bulgu) = butunluk_md(&"0".repeat(64), None);
        assert!(metin.contains("UYUSMADI") || metin.contains("olculemedi"));
        if metin.contains("UYUSMADI") {
            let bulgu = bulgu.expect("uyusmazlik bulgu uretmeli");
            assert_eq!(bulgu.agirlik, 3);
        }
    }

    #[test]
    fn the_running_digest_verifies_against_itself() {
        let yol = std::env::current_exe().unwrap_or_default();
        if let Ok(olculen) = butunluk::ozet(&yol) {
            let (metin, bulgu) = butunluk_md(&olculen, None);
            assert!(metin.contains("dogrulandi"), "{metin}");
            assert!(bulgu.is_none(), "dogru ozet bulgu uretmemeli");
        }
    }

    #[test]
    fn a_named_file_is_verified_through_the_same_section() {
        let yol = std::env::current_exe().unwrap_or_default();
        let beklenen = butunluk::ozet(&yol).unwrap_or_default();
        let (metin, bulgu) = butunluk_md(&beklenen, Some(&yol.to_string_lossy()));
        assert!(metin.contains("dogrulandi"), "{metin}");
        assert!(bulgu.is_none());
    }

    #[test]
    fn the_layer_table_counts_every_layer_even_when_it_is_empty() {
        let rapor = Rapor {
            kip: Kip::Bildir,
            bulgular: vec![Bulgu {
                katman: Katman::Zamanlama,
                aciklama: "olculdu".to_string(),
                agirlik: 1,
            }],
            ozet: None,
        };
        let tablo = katman_tablosu(&rapor);
        assert!(tablo.contains("| zamanlama | 1 |"));
        assert!(tablo.contains("| hata-ayiklayici | 0 |"));
    }
}
