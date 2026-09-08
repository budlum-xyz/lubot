#![forbid(unsafe_code)]
//! # lubot-sikistir - the context compression layer
//!
//! Structural adaptation of a context-compression architecture:
//! compress what an agent reads before it reaches the model, and keep every
//! compression reversible. The code here is written fresh; only the shape is
//! borrowed, the way the licence permits.
//!
//! The discipline, one line each:
//!
//! 1. **Route by content type.** JSON, code, logs, diffs and prose each get
//!    their own compressor; one universal pass would give false confidence on
//!    at least one of them.
//! 2. **Pins survive byte for byte.** A caller names lines that must not be
//!    summarised away; after compression each pin is checked, and a missing
//!    pin is an error, not a warning.
//! 3. **Compression stays reversible (CCR).** The original is stored under
//!    its SHA-256, the summary carries a header naming that digest, and
//!    retrieval re-verifies the digest before returning anything.
//! 4. **Every run is measured.** Before/after sizes land in an append-only
//!    statistics ledger, so savings are an audit, not a claim.
//! 5. **Task text is protected.** This crate compresses what it is handed;
//!    it never decides on its own that an instruction is redundant.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// What kind of content the router saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IcerikTuru {
    Json,
    Kod,
    Log,
    Diff,
    Metin,
}

impl IcerikTuru {
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            IcerikTuru::Json => "json",
            IcerikTuru::Kod => "kod",
            IcerikTuru::Log => "log",
            IcerikTuru::Diff => "diff",
            IcerikTuru::Metin => "metin",
        }
    }
}

/// A named refusal. Nothing here panics: a reader that panics is a reader
/// that stops answering.
#[derive(Debug)]
pub enum Hata {
    BosGirdi,
    Depo(String),
    GirisCikis(String),
    Pin(String),
    Dogrulama(String),
}

impl fmt::Display for Hata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Hata::BosGirdi => write!(f, "empty input is not content; nothing to compress"),
            Hata::Depo(mesaj) => write!(f, "store refused: {mesaj}"),
            Hata::GirisCikis(mesaj) => write!(f, "io refused: {mesaj}"),
            Hata::Pin(mesaj) => write!(f, "pin check failed: {mesaj}"),
            Hata::Dogrulama(mesaj) => write!(f, "verification refused: {mesaj}"),
        }
    }
}

impl From<io::Error> for Hata {
    fn from(e: io::Error) -> Self {
        Hata::GirisCikis(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// router: content type decides the compressor
// ---------------------------------------------------------------------------

/// Decide which compressor fits this content. Deterministic over the same
/// input; never reads a wall clock.
#[must_use]
pub fn yonlendir(icerik: &str) -> IcerikTuru {
    let trimmed = icerik.trim_start();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    {
        return IcerikTuru::Json;
    }
    let lines: Vec<&str> = icerik.lines().take(400).collect();
    let diff_belirtisi = lines
        .iter()
        .filter(|l| {
            l.starts_with("diff --git")
                || l.starts_with("@@")
                || l.starts_with("--- ")
                || l.starts_with("+++ ")
        })
        .count();
    let degisen = lines
        .iter()
        .filter(|l| l.starts_with('+') || l.starts_with('-'))
        .count();
    if diff_belirtisi >= 2 && degisen >= 1 {
        return IcerikTuru::Diff;
    }
    let kod_belirtisi = lines
        .iter()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("fn ")
                || t.starts_with("pub fn")
                || t.starts_with("def ")
                || t.starts_with("import ")
                || t.starts_with("use ")
                || t.starts_with("#include")
                || t.starts_with("package ")
        })
        .count();
    if kod_belirtisi >= 2 {
        return IcerikTuru::Kod;
    }
    let log_belirtisi = lines
        .iter()
        .filter(|l| {
            l.contains(" INFO ")
                || l.contains(" ERROR ")
                || l.contains(" WARN ")
                || l.contains(" DEBUG ")
                || l.contains("##[error]")
                || (l.len() > 20
                    && l[..20].chars().take(4).all(|c| c.is_ascii_digit())
                    && l[4..].starts_with('-'))
        })
        .count();
    if log_belirtisi >= 3 {
        return IcerikTuru::Log;
    }
    IcerikTuru::Metin
}

// ---------------------------------------------------------------------------
// compressors, one per type
// ---------------------------------------------------------------------------

/// The measured outcome of one compression run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sikistirma {
    pub tur: IcerikTuru,
    pub metin: String,
    pub girdi_bayt: usize,
    pub cikti_bayt: usize,
}

/// Compress content by its routed type. Pins must each appear byte for byte
/// in the output; otherwise the run is refused, because a summary that lost
/// the line you pointed at is worse than no summary.
pub fn sikistir(icerik: &str, igneler: &[String]) -> Result<Sikistirma, Hata> {
    if icerik.trim().is_empty() {
        return Err(Hata::BosGirdi);
    }
    let tur = yonlendir(icerik);
    let metin = match tur {
        IcerikTuru::Json => json_sikistir(icerik),
        IcerikTuru::Kod => kod_sikistir(icerik),
        IcerikTuru::Log => log_sikistir(icerik),
        IcerikTuru::Diff => diff_sikistir(icerik),
        IcerikTuru::Metin => metin_sikistir(icerik),
    };
    for igne in igneler {
        if !metin.contains(igne.as_str()) {
            return Err(Hata::Pin(format!(
                "pinned line is absent from the summary: {igne}"
            )));
        }
    }
    Ok(Sikistirma {
        tur,
        girdi_bayt: icerik.len(),
        cikti_bayt: metin.len(),
        metin,
    })
}

/// Repeated identical consecutive lines fold to one copy plus a count
/// marker. The first copy of every run stays, byte for byte, so a pinned
/// line that sits inside a repeated run still survives.
fn log_sikistir(icerik: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut onceki: Option<String> = None;
    let mut tekrar: usize = 0;
    fn kapat(out: &mut Vec<String>, onceki: &Option<String>, tekrar: usize) {
        if tekrar > 1 {
            if let Some(o) = onceki {
                out.push(format!("[lubot: ayni satir x{tekrar} - {o}]"));
            }
        }
    }
    for satir in icerik.lines() {
        match &onceki {
            Some(o) if o == satir => tekrar += 1,
            _ => {
                kapat(&mut out, &onceki, tekrar);
                out.push(satir.to_string());
                onceki = Some(satir.to_string());
                tekrar = 1;
            }
        }
    }
    kapat(&mut out, &onceki, tekrar);
    out.join("\n")
}

/// JSON arrays longer than eight entries keep the first three and the last,
/// with an explicit skip marker between them; objects are kept whole.
fn json_sikistir(icerik: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(icerik.trim()) else {
        return metin_sikistir(icerik);
    };
    let daraltilmis = json_daralt(&value);
    serde_json::to_string(&daraltilmis).unwrap_or_else(|_| metin_sikistir(icerik))
}

fn json_daralt(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) if items.len() > 8 => {
            let atlanan = items.len() - 4;
            let mut yeni: Vec<serde_json::Value> = items[..3].iter().map(json_daralt).collect();
            let mut marker = serde_json::Map::new();
            marker.insert(
                "lubot_atlanan".to_string(),
                serde_json::Value::Number(atlanan.into()),
            );
            yeni.push(serde_json::Value::Object(marker));
            if let Some(son) = items.last() {
                yeni.push(json_daralt(son));
            }
            serde_json::Value::Array(yeni)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(json_daralt).collect())
        }
        serde_json::Value::Object(map) => {
            let mut yeni = serde_json::Map::new();
            for (k, v) in map {
                yeni.insert(k.clone(), json_daralt(v));
            }
            serde_json::Value::Object(yeni)
        }
        other => other.clone(),
    }
}

/// Changed lines and hunk headers stay; unchanged context runs fold.
fn diff_sikistir(icerik: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut baglam: Vec<String> = Vec::new();
    let flush_baglam = |out: &mut Vec<String>, baglam: &mut Vec<String>| {
        if baglam.len() > 3 {
            out.push(baglam[0].clone());
            out.push(format!(
                "[lubot: {} degismez baglam satiri]",
                baglam.len() - 2
            ));
            out.push(baglam[baglam.len() - 1].clone());
        } else {
            out.extend(baglam.iter().cloned());
        }
        baglam.clear();
    };
    for satir in icerik.lines() {
        let degisti = satir.starts_with('+')
            || satir.starts_with('-')
            || satir.starts_with("@@")
            || satir.starts_with("diff --git")
            || satir.starts_with("--- ")
            || satir.starts_with("+++ ");
        if degisti {
            flush_baglam(&mut out, &mut baglam);
            out.push(satir.to_string());
        } else {
            baglam.push(satir.to_string());
        }
    }
    flush_baglam(&mut out, &mut baglam);
    out.join("\n")
}

/// Code stays code: only blank-line runs are folded. A compressor that
/// deletes code lines would trade correctness for size.
fn kod_sikistir(icerik: &str) -> String {
    bosluk_katla(icerik, 2)
}

/// Prose: blank-line runs longer than three fold to one.
fn metin_sikistir(icerik: &str) -> String {
    bosluk_katla(icerik, 3)
}

fn bosluk_katla(icerik: &str, sinir: usize) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut bos: usize = 0;
    for satir in icerik.lines() {
        if satir.trim().is_empty() {
            bos += 1;
            if bos <= 1 {
                out.push(satir.to_string());
            } else if bos == sinir + 1 {
                out.push(format!("[lubot: {sinir}+ bos satir katlandi]"));
            }
        } else {
            bos = 0;
            out.push(satir.to_string());
        }
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// CCR: Compress-Cache-Retrieve - originals are never lost
// ---------------------------------------------------------------------------

/// Header keys embedded in every summary so the original can be found again.
pub const CCR_ASLI: &str = "CCR-ASLI";
pub const CCR_TUR: &str = "CCR-TUR";

/// Build the header block a summary carries.
#[must_use]
pub fn ozet_basligi(
    sha_hex: &str,
    tur: IcerikTuru,
    girdi_bayt: usize,
    cikti_bayt: usize,
) -> String {
    format!(
        "{CCR_ASLI}: {sha_hex}\n{CCR_TUR}: {} ({} bayt -> {} bayt)\n",
        tur.ad(),
        girdi_bayt,
        cikti_bayt
    )
}

/// Extract the original's digest from a summary.
pub fn basliktan_sha(ozet: &str) -> Result<String, Hata> {
    for satir in ozet.lines().take(4) {
        if let Some(artik) = satir.strip_prefix(&format!("{CCR_ASLI}: ")) {
            let sha = artik.trim();
            if sha.len() == 64 && sha.chars().all(|c| c.is_ascii_hexdigit()) {
                return Ok(sha.to_string());
            }
            return Err(Hata::Dogrulama(format!(
                "summary carries a malformed digest: {sha}"
            )));
        }
    }
    Err(Hata::Dogrulama(
        "summary carries no CCR header; it is not reversible".to_string(),
    ))
}

/// On-disk store for originals, keyed by content digest.
pub struct CcrDepo {
    kok: PathBuf,
}

impl CcrDepo {
    /// Open (creating if needed) a store rooted at `kok`.
    pub fn ac(kok: impl Into<PathBuf>) -> Result<Self, Hata> {
        let kok = kok.into();
        let asli = kok.join("asli");
        fs::create_dir_all(&asli).map_err(|e| Hata::Depo(e.to_string()))?;
        Ok(Self { kok })
    }

    /// Store content under its SHA-256 and return the digest.
    pub fn kaydet(&self, icerik: &[u8]) -> Result<String, Hata> {
        let mut hasher = Sha256::new();
        hasher.update(icerik);
        let sha = format!("{:x}", hasher.finalize());
        let yol = self.kok.join("asli").join(&sha);
        if !yol.exists() {
            fs::write(&yol, icerik).map_err(|e| Hata::Depo(e.to_string()))?;
        }
        Ok(sha)
    }

    /// Retrieve an original by digest; the digest is re-verified against the
    /// bytes actually on disk, so a corrupted store is a refusal.
    pub fn geri_getir(&self, sha_hex: &str) -> Result<Vec<u8>, Hata> {
        let yol = self.kok.join("asli").join(sha_hex);
        let baytlar = fs::read(&yol)
            .map_err(|e| Hata::Depo(format!("original not found ({sha_hex}): {e}")))?;
        let mut hasher = Sha256::new();
        hasher.update(&baytlar);
        let dogrulanan = format!("{:x}", hasher.finalize());
        if dogrulanan != sha_hex {
            return Err(Hata::Dogrulama(format!(
                "stored original does not hash to its digest: {sha_hex}"
            )));
        }
        Ok(baytlar)
    }
}

// ---------------------------------------------------------------------------
// measurement: savings are an audit, not a claim
// ---------------------------------------------------------------------------

/// One measured run, appended to the statistics ledger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IstatistikKaydi {
    pub zaman_unix: u64,
    pub tur: String,
    pub dosya: String,
    pub girdi_bayt: usize,
    pub cikti_bayt: usize,
}

/// Append one run to a JSONL ledger.
pub fn istatistik_ekle(yol: &Path, kayit: &IstatistikKaydi) -> Result<(), Hata> {
    let satir = serde_json::to_string(kayit).map_err(|e| Hata::GirisCikis(e.to_string()))?;
    let onceki = fs::read_to_string(yol).unwrap_or_default();
    fs::write(yol, format!("{onceki}{satir}\n")).map_err(|e| Hata::GirisCikis(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// ogren: mine a log for failure patterns (the learn shape)
// ---------------------------------------------------------------------------

/// Failure patterns the miner recognises: (needle, stable code).
pub const HATA_DESENLERI: &[(&str, &str)] = &[
    ("##[error]", "ci-hatasi"),
    ("error[", "derleme-hatasi"),
    ("panicked", "panik"),
    ("Traceback", "iz-surme"),
    ("No such file", "yanlis-yol"),
    ("Permission denied", "izin"),
    ("FAILED", "test-basarisiz"),
    ("error:", "hata-satiri"),
];

/// One mined pattern with example lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OgrenSatiri {
    pub kod: String,
    pub igne: String,
    pub ornekler: Vec<String>,
    pub toplam: usize,
}

/// Group every failing line of a log under the pattern that caught it.
/// Deterministic; example lists are capped at five.
#[must_use]
pub fn ogren(log_icerigi: &str) -> Vec<OgrenSatiri> {
    let mut topluluk: BTreeMap<String, OgrenSatiri> = BTreeMap::new();
    for satir in log_icerigi.lines() {
        for (igne, kod) in HATA_DESENLERI {
            if satir.contains(igne) {
                let giris = topluluk
                    .entry((*kod).to_string())
                    .or_insert_with(|| OgrenSatiri {
                        kod: (*kod).to_string(),
                        igne: (*igne).to_string(),
                        ornekler: Vec::new(),
                        toplam: 0,
                    });
                giris.toplam += 1;
                if giris.ornekler.len() < 5 {
                    giris.ornekler.push(satir.trim().to_string());
                }
                break; // one pattern per line keeps the count honest
            }
        }
    }
    topluluk.into_values().collect()
}

/// Render the mining result as a session-layer lesson file. A pattern seen
/// in an earlier session report is marked for promotion to a rule: lessons
/// become rules on the second occurrence, not the first.
#[must_use]
pub fn ogren_raporu(satirlar: &[OgrenSatiri], onceki_oturumlarda: &[String]) -> String {
    let mut out = String::from("# ogren: oturum katmani dersleri\n\n");
    if satirlar.is_empty() {
        out.push_str("Kazilacak hata deseni bulunamadi.\n");
        return out;
    }
    for satir in satirlar {
        let ikinci_kez = onceki_oturumlarda.iter().any(|k| k == &satir.kod);
        out.push_str(&format!(
            "## {} ({}, {} eslesme){}\n",
            satir.kod,
            satir.igne,
            satir.toplam,
            if ikinci_kez {
                " - IKINCI KEZ: kurala yukselt"
            } else {
                ""
            }
        ));
        for ornek in &satir.ornekler {
            out.push_str(&format!("- {ornek}\n"));
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// tests: every claim above is locked here
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_lines_survive_byte_for_byte() {
        let log = "ok\n".repeat(200)
            + "FAIL [badges-are-current]: README says 2885, run measured 2896\n"
            + &"ok\n".repeat(200);
        let pin = "FAIL [badges-are-current]: README says 2885, run measured 2896".to_string();
        let sonuc = sikistir(&log, &[pin.clone()]).unwrap();
        assert!(
            sonuc.metin.contains(&pin),
            "the pinned line must survive byte for byte"
        );
        assert!(
            sonuc.cikti_bayt < sonuc.girdi_bayt,
            "repetition must actually shrink"
        );
    }

    #[test]
    fn a_missing_pin_refuses_the_run() {
        let hata = sikistir("only calm lines here\n", &["FATAL".to_string()]).unwrap_err();
        assert!(matches!(hata, Hata::Pin(_)));
    }

    #[test]
    fn compression_round_trips_through_the_store() {
        let dir = std::env::temp_dir().join(format!("lubot-ccr-{}", std::process::id()));
        let depo = CcrDepo::ac(&dir).unwrap();
        let asil = "satir 1\nsatir 2\n".repeat(50);
        let sha = depo.kaydet(asil.as_bytes()).unwrap();
        let sonuc = sikistir(&asil, &[]).unwrap();
        let ozet = format!(
            "{}{}",
            ozet_basligi(&sha, sonuc.tur, sonuc.girdi_bayt, sonuc.cikti_bayt),
            sonuc.metin
        );
        let geri_sha = basliktan_sha(&ozet).unwrap();
        let geri = depo.geri_getir(&geri_sha).unwrap();
        assert_eq!(
            geri,
            asil.as_bytes(),
            "the retrieved original must equal the input exactly"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupted_original_is_refused_not_returned() {
        let dir = std::env::temp_dir().join(format!("lubot-ccr-bozuk-{}", std::process::id()));
        let depo = CcrDepo::ac(&dir).unwrap();
        let sha = depo.kaydet(b"gercek icerik").unwrap();
        let yol = dir.join("asli").join(&sha);
        fs::write(&yol, b"degistirilmis").unwrap();
        let hata = depo.geri_getir(&sha).unwrap_err();
        assert!(matches!(hata, Hata::Dogrulama(_)));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_router_names_each_type() {
        assert_eq!(yonlendir("{\"a\": 1}"), IcerikTuru::Json);
        assert_eq!(
            yonlendir("diff --git a/x b/x\n@@ -1 +1 @@\n-eski\n+yeni\n"),
            IcerikTuru::Diff
        );
        assert_eq!(
            yonlendir("use std::fs;\npub fn main() {}\nfn yardimci() {}\n"),
            IcerikTuru::Kod
        );
        let log = (0..10)
            .map(|i| format!("2026-09-08 0{i}:00:00 INFO adim {i}\n"))
            .collect::<String>();
        assert_eq!(yonlendir(&log), IcerikTuru::Log);
        assert_eq!(yonlendir("sade bir paragraf"), IcerikTuru::Metin);
    }

    #[test]
    fn json_arrays_fold_with_an_explicit_skip_marker() {
        let dizi: Vec<i32> = (0..20).collect();
        let girdi = serde_json::to_string(&dizi).unwrap();
        let sonuc = sikistir(&girdi, &[]).unwrap();
        assert!(sonuc.metin.contains("lubot_atlanan"));
        assert!(sonuc.metin.contains("16"), "20 - 4 kept = 16 skipped");
        assert!(
            sonuc.metin.contains('0') && sonuc.metin.contains("19"),
            "head and tail survive"
        );
    }

    #[test]
    fn diff_keeps_every_changed_line() {
        let diff = "diff --git a/f b/f\n@@ -1,3 +1,3 @@\n baglam 1\n baglam 2\n baglam 3\n baglam 4\n baglam 5\n-eski\n+yeni\n";
        let sonuc = sikistir(diff, &[]).unwrap();
        assert!(sonuc.metin.contains("-eski"));
        assert!(sonuc.metin.contains("+yeni"));
        assert!(sonuc.metin.contains("[lubot:"), "long context must fold");
    }

    #[test]
    fn the_miner_finds_a_planted_failure_and_counts_it() {
        let log = "ok\n##[error]Process completed with exit code 1.\nok\n##[error]Another one\npanicked at src/x.rs:1\n";
        let satirlar = ogren(log);
        let ci = satirlar.iter().find(|s| s.kod == "ci-hatasi").unwrap();
        assert_eq!(ci.toplam, 2);
        let panik = satirlar.iter().find(|s| s.kod == "panik").unwrap();
        assert_eq!(panik.toplam, 1);
    }

    #[test]
    fn a_second_occurrence_marks_promotion_to_rule() {
        let satirlar = ogren("##[error]boom\n");
        let rapor = ogren_raporu(&satirlar, &["ci-hatasi".to_string()]);
        assert!(rapor.contains("IKINCI KEZ: kurala yukselt"));
        let ilk = ogren_raporu(&satirlar, &[]);
        assert!(
            !ilk.contains("IKINCI KEZ"),
            "a first occurrence is a session note, not a rule"
        );
    }

    #[test]
    fn the_statistics_ledger_appends_one_line_per_run() {
        let yol =
            std::env::temp_dir().join(format!("lubot-istatistik-{}.jsonl", std::process::id()));
        let kayit = IstatistikKaydi {
            zaman_unix: 1,
            tur: "log".to_string(),
            dosya: "ornek.log".to_string(),
            girdi_bayt: 100,
            cikti_bayt: 10,
        };
        istatistik_ekle(&yol, &kayit).unwrap();
        istatistik_ekle(&yol, &kayit).unwrap();
        let icerik = fs::read_to_string(&yol).unwrap();
        assert_eq!(
            icerik.lines().count(),
            2,
            "each run appends exactly one ledger line"
        );
        fs::remove_file(&yol).ok();
    }

    #[test]
    fn empty_input_is_refused() {
        assert!(matches!(sikistir("   \n ", &[]), Err(Hata::BosGirdi)));
    }
}
