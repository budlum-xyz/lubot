//! # sozluk - the checkpoint's tokenizer, as a command
//!
//! Three subcommands, and each of them exists because a claim needs a command
//! behind it: `envanter` says what is in the vocabulary file, `jetonla` turns
//! text into ids and prints them, and `coz` turns ids back into text. The id
//! line of `jetonla` is the format the cross-check tool reads, so the port and
//! the reference are compared through the same surface the operator uses.

use std::path::{Path, PathBuf};

use lubot_kodlayici::sozluk::Sozluk;

/// Dispatches the `sozluk` subcommands.
pub fn cmd_sozluk(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("envanter") => envanter(&args[1..]),
        Some("jetonla") => jetonla(&args[1..]),
        Some("coz") => coz(&args[1..]),
        Some(other) => Err(format!(
            "unknown sozluk subcommand: {other}\nusage: lubot kodlayici sozluk [envanter|jetonla|coz] --sozluk <tokenizer.json>"
        )),
    }
}

fn deger(args: &[String], ad: &str) -> Option<String> {
    let sira = args.iter().position(|a| a == ad)?;
    args.get(sira + 1).cloned()
}

fn sozluk_yolu(args: &[String]) -> Result<PathBuf, String> {
    deger(args, "--sozluk")
        .or_else(|| deger(args, "--paket").map(|p| format!("{p}/tokenizer/tokenizer.json")))
        .map(PathBuf::from)
        .ok_or("usage: --sozluk <tokenizer.json> gerekli".to_string())
}

fn oku(args: &[String]) -> Result<Sozluk, String> {
    let yol = sozluk_yolu(args)?;
    Sozluk::oku(&yol).map_err(|h| format!("{h:?}"))
}

fn envanter(args: &[String]) -> Result<(), String> {
    let sozluk = oku(args)?;
    let (bas, son) = sozluk.sarmalayici();
    println!("# Sozluk envanteri\n");
    println!("- jeton: {}", sozluk.boyut());
    println!("- birlesme kurali: {}", sozluk.kural_sayisi());
    println!("- eklenen jeton: {}", sozluk.eklenen_sayisi());
    println!("- dolgu: {}", sozluk.dolgu());
    println!("- maske: {}", sozluk.maske());
    match (bas, son) {
        (Some(b), Some(s)) => println!("- sarmalayici: {b} .. {s}"),
        _ => println!("- sarmalayici: yok"),
    }
    Ok(())
}

fn jetonla(args: &[String]) -> Result<(), String> {
    let sozluk = oku(args)?;
    // Text can also arrive as a file. A zero byte and a very long example
    // cannot go through a command line at all, and the cross-check has to be
    // able to compare exactly those: a fallback that swallowed a NUL would
    // make the comparison agree on the wrong bytes.
    let metin = match deger(args, "--metin-dosya") {
        Some(yol) => {
            let ham = std::fs::read(&yol).map_err(|e| format!("{yol}: {e}"))?;
            String::from_utf8(ham).map_err(|e| format!("{yol}: utf-8 degil: {e}"))?
        }
        None => deger(args, "--metin").unwrap_or_default(),
    };
    let idler = sozluk.jetonla(&metin);
    println!("# Jetonlama\n");
    println!("- metin uzunlugu: {} bayt", metin.len());
    println!("- jeton: {}", idler.len());
    println!(
        "- ids: {}",
        idler
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    );
    if deger(args, "--kapsamli").is_some() {
        let cekirdek = sozluk.jetonla_cekirdek(&metin);
        println!(
            "- cekirdek: {}",
            cekirdek
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        );
        let parcalar: Vec<String> = idler
            .iter()
            .map(|i| sozluk.jeton(*i).unwrap_or("?").to_string())
            .collect();
        println!("- jetonlar: {}", parcalar.join(" | "));
    }
    Ok(())
}

fn coz(args: &[String]) -> Result<(), String> {
    let sozluk = oku(args)?;
    let ham = deger(args, "--ids").ok_or("usage: --ids 1,2,3 gerekli")?;
    let idler: Vec<u32> = ham
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            p.trim()
                .parse::<u32>()
                .map_err(|h| format!("`{p}` id degil: {h}"))
        })
        .collect::<Result<Vec<u32>, String>>()?;
    let metin = sozluk.coz(&idler);
    println!("# Cozumleme\n");
    println!("- jeton: {}", idler.len());
    println!("- metin: {metin:?}");
    Ok(())
}

/// A one-line summary of a vocabulary file, for the status line.
///
/// # Errors
/// The path or the file, named.
pub fn sozluk_ozeti(yol: &Path) -> Result<String, String> {
    let sozluk = Sozluk::oku(yol).map_err(|h| format!("{h:?}"))?;
    Ok(format!(
        "sozluk: {} jeton, {} birlesme, {} eklenen",
        sozluk.boyut(),
        sozluk.kural_sayisi(),
        sozluk.eklenen_sayisi()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vocabulary_flag_is_required_and_the_package_form_is_accepted() {
        assert!(sozluk_yolu(&[]).is_err());
        assert_eq!(
            sozluk_yolu(&["--sozluk".into(), "/tmp/t.json".into()]),
            Ok(PathBuf::from("/tmp/t.json"))
        );
        assert_eq!(
            sozluk_yolu(&["--paket".into(), "/p".into()]),
            Ok(PathBuf::from("/p/tokenizer/tokenizer.json"))
        );
    }

    #[test]
    fn an_id_list_with_a_typo_is_refused_not_skipped() {
        let idler: Result<Vec<u32>, String> = "1,2,x"
            .split(',')
            .map(|p| p.parse::<u32>().map_err(|h| h.to_string()))
            .collect();
        assert!(idler.is_err());
    }

    #[test]
    fn a_missing_vocabulary_file_is_named_rather_than_guessed() {
        assert!(sozluk_ozeti(Path::new("/tmp/lubot-yok.json")).is_err());
    }
}
