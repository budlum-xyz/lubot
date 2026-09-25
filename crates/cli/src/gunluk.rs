//! # gunluk - the command that reads operator logs
//!
//! One command, two shapes: `ozet` counts what a log file contains by status or
//! severity, and `oku` filters the lines whose severity is at or above a
//! threshold. Both mask secrets on the way in - the masking happens inside
//! [`lubot_gunluk::satir_oku`], so there is no path here that reads a log
//! without it.

use std::collections::BTreeMap;

use lubot_gunluk::Satir;

/// Dispatches the `gunluk` subcommands.
pub fn cmd_gunluk(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("ozet") => ozet(&args[1..]),
        Some("oku") => oku(&args[1..]),
        Some(other) => Err(format!(
            "unknown gunluk subcommand: {other}\nusage: lubot gunluk [ozet|oku] --dosya <yol> [--en-agir <0-7>]"
        )),
    }
}

fn deger(args: &[String], ad: &str) -> Option<String> {
    let sira = args.iter().position(|a| a == ad)?;
    args.get(sira + 1).cloned()
}

fn govde(args: &[String]) -> Result<String, String> {
    match deger(args, "--dosya") {
        Some(yol) => std::fs::read_to_string(&yol).map_err(|e| format!("{yol}: {e}")),
        None => Err("usage: --dosya <yol> gerekli".to_string()),
    }
}

fn ozet(args: &[String]) -> Result<(), String> {
    let (satirlar, atilan) = lubot_gunluk::oku(&govde(args)?);
    let mut durumlar: BTreeMap<u16, usize> = BTreeMap::new();
    let mut seviyeler: BTreeMap<u8, usize> = BTreeMap::new();
    for satir in &satirlar {
        match satir {
            Satir::Nginx(s) => *durumlar.entry(s.durum).or_insert(0) += 1,
            Satir::Syslog(o) => *seviyeler.entry(o.etkin_severity()).or_insert(0) += 1,
        }
    }
    println!("# Gunluk ozeti\n");
    println!("- okunan satir: {}", satirlar.len());
    println!("- anlasilmayan satir: {atilan}");
    if !durumlar.is_empty() {
        println!("\n| durum | adet |\n|---:|---:|");
        for (durum, adet) in &durumlar {
            println!("| {durum} | {adet} |");
        }
    }
    if !seviyeler.is_empty() {
        println!("\n| seviye | ad | adet |\n|---:|---|---:|");
        for (seviye, adet) in &seviyeler {
            println!(
                "| {seviye} | {} | {adet} |",
                lubot_gunluk::severity_adi(*seviye)
            );
        }
    }
    Ok(())
}

fn oku(args: &[String]) -> Result<(), String> {
    let esik = match deger(args, "--en-agir") {
        Some(s) => s
            .parse::<u8>()
            .map_err(|e| format!("`{s}` seviye degil: {e}"))?,
        None => 4,
    };
    if esik > 7 {
        return Err(format!("seviye 0..=7 olmali, `{esik}` verildi"));
    }
    let (satirlar, _) = lubot_gunluk::oku(&govde(args)?);
    let mut yazilan = 0;
    for satir in &satirlar {
        let (seviye, metin) = match satir {
            Satir::Nginx(s) => (
                if s.durum >= 500 { 3 } else { 6 },
                format!("{} {} {} {}", s.adres, s.istek, s.durum, s.bayt),
            ),
            Satir::Syslog(o) => (
                o.etkin_severity(),
                format!(
                    "{} {}: {}",
                    lubot_gunluk::severity_adi(o.etkin_severity()),
                    o.uygulama.as_deref().unwrap_or("-"),
                    o.govde
                ),
            ),
        };
        if seviye <= esik {
            println!("{seviye}\t{metin}");
            yazilan += 1;
        }
    }
    eprintln!("# esik {esik}: {yazilan} satir");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_subcommand_is_named() {
        assert!(cmd_gunluk(&["bilinmeyen".to_string()]).is_err());
    }

    #[test]
    fn the_file_flag_is_required() {
        assert!(govde(&[]).is_err());
    }

    #[test]
    fn a_threshold_above_the_severity_range_is_refused() {
        let hata = oku(&[
            "--en-agir".to_string(),
            "9".to_string(),
            "--dosya".to_string(),
            "/yok".to_string(),
        ]);
        assert!(hata.is_err());
    }
}
