//! # sikistir - the context compression layer, as a command
//!
//! Structural adaptation of a context-compression shape: route the input
//! to the compressor that fits its type, keep every compression reversible
//! through a content-digest store, pin the lines that must survive, and
//! record every run in an append-only ledger. The original stays on disk;
//! `--geri-getir` brings it back and re-verifies it byte for byte.
//!
//! `lubot ogren` is the learn shape: it mines a log for failure patterns
//! and writes a session-layer lesson file; a pattern already seen in an
//! earlier session report is marked for promotion to a rule.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use lubot_sikistir::{
    basliktan_sha, istatistik_ekle, ogren, ogren_raporu, ozet_basligi, sikistir, CcrDepo,
    IstatistikKaydi,
};

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn deger_bul(args: &[String], ad: &str) -> Option<String> {
    args.iter()
        .position(|a| a == ad)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// `lubot sikistir --path <f> [--igne <desen> ...] [--depo <dir>]`
/// `lubot sikistir --geri-getir <ozet-dosya> [--depo <dir>]`
pub fn cmd_sikistir(args: &[String]) -> Result<(), String> {
    let depo_kok = deger_bul(args, "--depo").unwrap_or_else(|| "outputs/sikistir".to_string());
    if let Some(ozet_yolu) = deger_bul(args, "--geri-getir") {
        let ozet = fs::read_to_string(&ozet_yolu)
            .map_err(|e| format!("summary file refused ({ozet_yolu}): {e}"))?;
        let sha = basliktan_sha(&ozet).map_err(|e| e.to_string())?;
        let depo = CcrDepo::ac(PathBuf::from(&depo_kok)).map_err(|e| e.to_string())?;
        let baytlar = depo.geri_getir(&sha).map_err(|e| e.to_string())?;
        let icerik = String::from_utf8(baytlar)
            .map_err(|e| format!("retrieved original is not utf-8: {e}"))?;
        print!("{icerik}");
        return Ok(());
    }
    let Some(yol) = deger_bul(args, "--path") else {
        return Err("sikistir needs --path <f> or --geri-getir <ozet-dosya>".to_string());
    };
    let icerik = fs::read_to_string(&yol).map_err(|e| format!("input refused ({yol}): {e}"))?;
    let igneler: Vec<String> = args
        .windows(2)
        .filter(|w| w[0] == "--igne")
        .map(|w| w[1].clone())
        .collect();
    let sonuc = sikistir(&icerik, &igneler).map_err(|e| e.to_string())?;
    let depo = CcrDepo::ac(PathBuf::from(&depo_kok)).map_err(|e| e.to_string())?;
    let sha = depo.kaydet(icerik.as_bytes()).map_err(|e| e.to_string())?;
    let ozet = format!(
        "{}{}",
        ozet_basligi(&sha, sonuc.tur, sonuc.girdi_bayt, sonuc.cikti_bayt),
        sonuc.metin
    );
    let ozet_yolu = Path::new(&depo_kok).join("ozetler").join(format!(
        "{}.ozet",
        Path::new(&yol)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("girdi")
    ));
    fs::create_dir_all(ozet_yolu.parent().ok_or("summary dir has no parent")?)
        .map_err(|e| e.to_string())?;
    fs::write(&ozet_yolu, &ozet).map_err(|e| e.to_string())?;
    istatistik_ekle(
        &Path::new(&depo_kok).join("istatistik.jsonl"),
        &IstatistikKaydi {
            zaman_unix: now_seconds(),
            tur: sonuc.tur.ad().to_string(),
            dosya: yol.clone(),
            girdi_bayt: sonuc.girdi_bayt,
            cikti_bayt: sonuc.cikti_bayt,
        },
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "sikistir: {} bayt -> {} bayt ({}) ; ozet: {}",
        sonuc.girdi_bayt,
        sonuc.cikti_bayt,
        sonuc.tur.ad(),
        ozet_yolu.display()
    );
    print!("{ozet}");
    Ok(())
}

/// `lubot ogren --log <f> [--ogren-dir outputs/ogren]`
pub fn cmd_ogren(args: &[String]) -> Result<(), String> {
    let Some(yol) = deger_bul(args, "--log") else {
        return Err("ogren needs --log <f>".to_string());
    };
    let dizin = deger_bul(args, "--ogren-dir").unwrap_or_else(|| "outputs/ogren".to_string());
    let icerik = fs::read_to_string(&yol).map_err(|e| format!("log refused ({yol}): {e}"))?;
    let satirlar = ogren(&icerik);
    // Two tiers: codes seen in earlier session reports mark promotion.
    let mut onceki: Vec<String> = Vec::new();
    if let Ok(girisler) = fs::read_dir(&dizin) {
        for giris in girisler.flatten() {
            if let Ok(eski) = fs::read_to_string(giris.path()) {
                for (igne, kod) in lubot_sikistir::HATA_DESENLERI {
                    let _ = igne;
                    if eski.contains(&format!("## {kod} ")) && !onceki.iter().any(|k| k == kod) {
                        onceki.push((*kod).to_string());
                    }
                }
            }
        }
    }
    let rapor = ogren_raporu(&satirlar, &onceki);
    fs::create_dir_all(&dizin).map_err(|e| e.to_string())?;
    let out_yolu = Path::new(&dizin).join(format!(
        "{}-{}.md",
        now_seconds(),
        Path::new(&yol)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("log")
    ));
    fs::write(&out_yolu, &rapor).map_err(|e| e.to_string())?;
    eprintln!(
        "ogren: {} desen; rapor: {}",
        satirlar.len(),
        out_yolu.display()
    );
    print!("{rapor}");
    Ok(())
}
