//! # kalibrasyon - the calibrated confidence head, as a command
//!
//! `lubot kalibrasyon --girdi <kayitlar.jsonl>` fits one temperature to
//! recorded `(puan, dogru)` pairs, reports the training error and the expected
//! calibration error before and after the correction, and measures the bands
//! from the same records. The band edges are not written here and not written
//! in the crate: they come out of the data, and when the data cannot support
//! them the command refuses instead of inventing a threshold.
//!
//! Two inputs are possible and they are not the same thing:
//!
//! - a **real** record file: scores a head actually produced, with the outcome
//!   it actually got. This is the only input that says anything about a model;
//! - a **fixture**: declared numbers that exercise the machinery. It measures
//!   the measurement, not a model, and the report says so in its first line.
//!
//! The summary line at the end of the report is the machine-readable half:
//! `kalibrasyon: sicaklik=... nll_ham=... ...`. A record file is generated from
//! it (`training/kalibrasyon.py --kur`), so the numbers in the record come from
//! this binary rather than from a second implementation in Python.

use std::fs;
use std::path::PathBuf;

use lubot_tomurcuk::kalibrasyon::{
    bantlari_olc, basamak, ece, kalibre_puan, kova_genisligi, sicaklik_uyarla, Bantlar, Basamak,
    HamKayit, KalibrasyonHatasi, SicaklikFit, ASGARI_DESTEK, ASGARI_KAYIT, HEDEF_KESINLIK,
    KIRMIZI_TAVAN, SICAKLIK_ALT, SICAKLIK_UST,
};

/// Runs the calibration command.
pub fn cmd_kalibrasyon(args: &[String]) -> Result<(), String> {
    let Argumanlar { girdi, hedef } = Argumanlar::oku(args)?;
    let girdi = PathBuf::from(
        girdi.ok_or("kalibrasyon: --girdi is required (an unmeasured head has no bands)")?,
    );

    let metin = fs::read_to_string(&girdi)
        .map_err(|e| format!("kalibrasyon: cannot read {}: {e}", girdi.display()))?;
    let (kayitlar, kac_satir) = kayitlari_oku(&metin)?;
    let fit: SicaklikFit = sicaklik_uyarla(&kayitlar).map_err(hata_metni)?;
    let bantlar: Bantlar = bantlari_olc(&kayitlar, fit.sicaklik, hedef).map_err(hata_metni)?;
    let ece_ham = ece(&kayitlar, 1.0).ok_or("kalibrasyon: ECE is not a number for this input")?;
    let ece_duzeltilmis =
        ece(&kayitlar, fit.sicaklik).ok_or("kalibrasyon: ECE is not a number for this input")?;

    // Bant dagilimi: kac kayit hangi basamakta. Bu, bandin bos kume olmadiginin
    // olcumudur; bir esik yazmak yerine kayitlarin nereye dustugu gosterilir.
    let mut basamak_sayilari = [0u64; 3];
    for kayit in &kayitlar {
        let Some(q) = kalibre_puan(kayit.puan, fit.sicaklik) else {
            return Err(format!(
                "kalibrasyon: a record is not usable: {}",
                KalibrasyonHatasi::GecersizPuan.ad()
            ));
        };
        let basamak = basamak(q, &bantlar);
        basamak_sayilari[basamak_index(basamak)] += 1;
    }

    let mut md = String::from("# Kalibrasyon\n\n");
    md.push_str(&format!(
        "Girdi: `{}` ({} satir, {} kayit).\n\n",
        girdi.display(),
        kac_satir,
        kayitlar.len()
    ));
    md.push_str(&format!(
        "Sozlesme: sicaklik {:.2}..{:.2} araliginda aranir (altin oran aramasi, log uzayinda), en az {} kayit ister, kova genisligi {:.2}, bir kovayi olculmus saymak icin en az {} kayit ve yazi-tura esigi {:.2}.\n\n",
        SICAKLIK_ALT, SICAKLIK_UST, ASGARI_KAYIT, kova_genisligi(), ASGARI_DESTEK, KIRMIZI_TAVAN
    ));
    md.push_str(&format!(
        "| olcum | deger |\n|---|---|\n| sicaklik (uydurulan) | {:.6} |\n| egitim hatasi (T=1) | {:.6} |\n| egitim hatasi (duzeltilmis) | {:.6} |\n| ECE (T=1) | {:.6} |\n| ECE (duzeltilmis) | {:.6} |\n| hedef isabet | {:.2} |\n",
        fit.sicaklik, fit.nll_ham, fit.nll_fit, ece_ham, ece_duzeltilmis, hedef
    ));
    md.push_str(&format!(
        "\nBantlar (olcumden): kirmizi `0 .. < {:.2}`, orta `{:.2} .. < {:.2}`, yesil `{:.2} .. 1.00`.\nDestek esiginin altinda kalan bos olmayan kova: {}.\n",
        bantlar.kirmizi_ust,
        bantlar.kirmizi_ust,
        bantlar.yesil_alt,
        bantlar.yesil_alt,
        bantlar.destek_alti
    ));
    md.push_str(&format!(
        "\n| basamak | kayit |\n|---|---|\n| tek-bas (yesil) | {} |\n| konsensus (orta) | {} |\n| yukselt (kirmizi) | {} |\n",
        basamak_sayilari[0], basamak_sayilari[1], basamak_sayilari[2]
    ));
    md.push_str(&format!(
        "\nKazanc: egitim hatasi {} dustu ({:.6} -> {:.6}), ECE {} dustu ({:.6} -> {:.6}).\n",
        if fit.kazanc_var() {
            "gercekten"
        } else {
            "DUSMEDI"
        },
        fit.nll_ham,
        fit.nll_fit,
        if ece_duzeltilmis < ece_ham {
            "gercekten"
        } else {
            "DUSMEDI"
        },
        ece_ham,
        ece_duzeltilmis
    ));
    md.push_str(&format!(
        "\nkalibrasyon: sicaklik={:.6} nll_ham={:.6} nll_fit={:.6} ece_ham={:.6} ece_fit={:.6} yesil_alt={:.2} kirmizi_ust={:.2} hedef={:.2} kayit={} destek_alti={}\n",
        fit.sicaklik,
        fit.nll_ham,
        fit.nll_fit,
        ece_ham,
        ece_duzeltilmis,
        bantlar.yesil_alt,
        bantlar.kirmizi_ust,
        hedef,
        kayitlar.len(),
        bantlar.destek_alti
    ));
    crate::validate_output(md.as_bytes(), "kalibrasyon")?;
    println!("{md}");
    Ok(())
}

fn basamak_index(basamak: Basamak) -> usize {
    match basamak {
        Basamak::TekBas => 0,
        Basamak::Konsensus => 1,
        Basamak::Yukselt => 2,
    }
}

/// Hatayi kullanicinin okuyabilecegi bir satira cevirir. Karar basinin kendisi
/// metin uretmez; metin burada, komut katmaninda uretilir.
fn hata_metni(hata: KalibrasyonHatasi) -> String {
    format!("kalibrasyon: {}", hata.ad())
}

/// Bu komutun gordulu argumanlari.
struct Argumanlar {
    girdi: Option<String>,
    hedef: f64,
}

impl Argumanlar {
    /// `--girdi <yol>` ve `--hedef <sayi>`; baska her sey reddedilir.
    fn oku(args: &[String]) -> Result<Self, String> {
        let mut girdi: Option<String> = None;
        let mut hedef: Option<f64> = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--girdi" => {
                    let deger = args.get(i + 1).ok_or("kalibrasyon: --girdi needs a path")?;
                    girdi = Some(deger.clone());
                    i += 2;
                }
                "--hedef" => {
                    let deger = args
                        .get(i + 1)
                        .ok_or("kalibrasyon: --hedef needs a number")?;
                    hedef =
                        Some(deger.parse::<f64>().map_err(|_| {
                            format!("kalibrasyon: --hedef is not a number: {deger}")
                        })?);
                    i += 2;
                }
                other => {
                    return Err(format!(
                        "kalibrasyon: unexpected `{other}`\nusage: lubot kalibrasyon --girdi <kayitlar.jsonl> [--hedef 0.9]"
                    ));
                }
            }
        }
        Ok(Self {
            girdi,
            hedef: hedef.unwrap_or(HEDEF_KESINLIK),
        })
    }
}

/// JSONL kaydini okur: her satir `{"puan": 0.83, "dogru": true}`.
///
/// Bos satirlar atlanir; bozuk bir satir **tum dosyayi** reddeder. Yarim
/// okunan bir kayitla olculen bant, olculmemis bir banddir.
fn kayitlari_oku(metin: &str) -> Result<(Vec<HamKayit>, usize), String> {
    let mut kayitlar = Vec::new();
    let mut satir_sayisi = 0usize;
    for (i, satir) in metin.lines().enumerate() {
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        satir_sayisi += 1;
        let deger: serde_json::Value = serde_json::from_str(satir)
            .map_err(|e| format!("kalibrasyon: line {} is not JSON: {e}", i + 1))?;
        let puan = deger
            .get("puan")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("kalibrasyon: line {} has no `puan`", i + 1))?;
        let dogru = deger
            .get("dogru")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| format!("kalibrasyon: line {} has no `dogru`", i + 1))?;
        let kayit = HamKayit::yeni(puan, dogru).ok_or_else(|| {
            format!(
                "kalibrasyon: line {} carries an unusable score ({puan}); an endpoint has no logit",
                i + 1
            )
        })?;
        kayitlar.push(kayit);
    }
    Ok((kayitlar, satir_sayisi))
}
