//! # sir - the command that runs the mask over text
//!
//! Two uses, one rule. `maskele` prints the cleaned text and the count of what
//! was hidden; `tara` answers only whether anything *looks* like a secret, for
//! the case where the caller wants to refuse a file rather than rewrite it.
//! The exit code of `tara` carries the answer, so a shell can gate on it.

use std::io::Read;

/// Dispatches the `sir` subcommands.
pub fn cmd_sir(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("maskele") => maskele(&args[1..]),
        Some("tara") => tara(&args[1..]),
        Some(other) => Err(format!(
            "unknown sir subcommand: {other}\nusage: lubot sir [maskele|tara] [--dosya <yol>]"
        )),
    }
}

/// The text to work on: a file, or stdin when no file is named. A pipeline is
/// the normal case here - the mask sits between a producer and a consumer.
fn metin(args: &[String]) -> Result<String, String> {
    let yol = args
        .iter()
        .position(|a| a == "--dosya")
        .and_then(|i| args.get(i + 1));
    match yol {
        Some(yol) => std::fs::read_to_string(yol).map_err(|e| format!("{yol}: {e}")),
        None => {
            let mut tampon = String::new();
            std::io::stdin()
                .read_to_string(&mut tampon)
                .map_err(|e| format!("stdin: {e}"))?;
            Ok(tampon)
        }
    }
}

fn maskele(args: &[String]) -> Result<(), String> {
    let s = lubot_sir::maskele(&metin(args)?);
    print!("{}", s.metin());
    if s.rapor().degisti() && !s.metin().ends_with('\n') {
        println!();
    }
    eprintln!(
        "# sir maskeleme: {} maskeleme{}",
        s.rapor().toplam(),
        if s.rapor().sayim().is_empty() {
            String::new()
        } else {
            format!(
                " ({})",
                s.rapor()
                    .sayim()
                    .iter()
                    .map(|(tur, adet)| format!("{tur} x{adet}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );
    Ok(())
}

fn tara(args: &[String]) -> Result<(), String> {
    let ham = metin(args)?;
    let s = lubot_sir::maskele(&ham);
    if s.rapor().degisti() {
        eprintln!("# sir bulundu: {} maskeleme gerekir", s.rapor().toplam());
        // A non-zero exit is the answer: a caller that gates on this command
        // does not have to parse anything.
        std::process::exit(1);
    }
    eprintln!("# taninan sir yok");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_subcommand_is_named() {
        let hata = cmd_sir(&["bilinmeyen".to_string()]);
        assert!(hata.is_err());
    }

    #[test]
    fn the_file_flag_is_read_before_stdin() {
        let yol = std::env::temp_dir().join("lubot-sir-test.txt");
        std::fs::write(&yol, "token: abc").expect("yazilamadi");
        let okunan = metin(&["--dosya".to_string(), yol.display().to_string()]);
        assert_eq!(okunan.expect("okunamadi"), "token: abc");
    }

    #[test]
    fn a_missing_file_is_named_rather_than_guessed() {
        let hata = metin(&["--dosya".to_string(), "/tmp/lubot-yok-12345".to_string()]);
        assert!(hata.is_err());
    }
}
