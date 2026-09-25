//! # kodlayici - the ported model, as a command
//!
//! `lubot kodlayici envanter --paket <dizin>` reads a split checkpoint's header
//! and prints what is inside it: how many tensors, which types, what the
//! configuration says the model is. `lubot kodlayici kosu --paket <dizin>
//! --kimlik 1,2,3` runs the encoder over a token sequence and prints a summary
//! of the hidden states.
//!
//! Everything on stdout is Markdown. The command does **not** claim the numbers
//! are the model's answers: the encoder's output is a representation, and what
//! it means is the head's business. A run that says where it stopped is more
//! useful than a run that guesses.

use std::path::{Path, PathBuf};

use lubot_kodlayici::baslik::{Dizin, ParcaliDosya};
use lubot_kodlayici::blok::{kodla, kullanilan_adlar, ortalama_havuz, Agirliklar, PencereKurali};
use lubot_kodlayici::karar::{kullanilan_adlar as kafa_adlari, puanla, tip_indeksi, KararAgirliklari, KararYapisi, TIPLER};
use lubot_kodlayici::sozluk::Sozluk;
use lubot_kodlayici::yapilandirma::{KafaYapisi, KodlayiciYapisi};

/// Dispatches the `kodlayici` subcommands.
pub fn cmd_kodlayici(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("envanter") => envanter(&args[1..]),
        Some("kosu") => kosu(&args[1..]),
        Some("dogrula") => dogrula(&args[1..]),
        Some("sozluk") => crate::sozluk::cmd_sozluk(&args[1..]),
        Some(other) => Err(format!(
            "unknown kodlayici subcommand: {other}\nusage: lubot kodlayici [envanter|kosu|dogrula] --paket <dizin> ..."
        )),
    }
}

/// The value after `--ad`, if the flag is present with a value.
fn deger(args: &[String], ad: &str) -> Option<String> {
    let sira = args.iter().position(|a| a == ad)?;
    args.get(sira + 1).cloned()
}

/// The checkpoint directory, required by both subcommands.
fn paket(args: &[String]) -> Result<PathBuf, String> {
    deger(args, "--paket")
        .map(PathBuf::from)
        .ok_or("usage: --paket <dizin> gerekli".to_string())
}

/// The encoder configuration, read from the package.
fn yapi_oku(paket: &Path) -> Result<KodlayiciYapisi, String> {
    let yol = paket.join("encoder").join("config.json");
    // The window shape is not a checkpoint field; `kosu` can be told which one
    // to use, and every report says which one it used.
    KodlayiciYapisi::oku(&yol)
}

/// Reads the header and reports what is inside.
fn envanter(args: &[String]) -> Result<(), String> {
    let paket = paket(args)?;
    let dosya = ParcaliDosya::ac(&paket, "model.safetensors.part-")
        .map_err(|h| format!("{}: {h:?}", paket.display()))?;
    let dizin = Dizin::oku(&dosya).map_err(|h| format!("{h:?}"))?;
    let yapi = yapi_oku(&paket)?;
    let mut c = String::new();
    c.push_str("# Kodlayici envanteri\n\n");
    c.push_str(&format!("- paket: `{}`\n", paket.display()));
    c.push_str(&format!("- parca: {}\n", dosya.parca_sayisi()));
    c.push_str(&format!("- toplam bayt: {}\n", dosya.boyut()));
    c.push_str(&format!("- tensor: {}\n", dizin.uzunluk()));
    c.push_str(&format!("- eleman: {}\n", dizin.toplam_eleman()));
    let turler: Vec<String> = dizin
        .tur_dagilimi()
        .iter()
        .map(|(tur, adet)| format!("{tur} {adet}"))
        .collect();
    c.push_str(&format!("- turler: {}\n\n", turler.join(", ")));
    c.push_str("| yapi | deger |\n|---|---|\n");
    c.push_str(&format!("| katman | {} |\n", yapi.num_hidden_layers));
    c.push_str(&format!("| gizli | {} |\n", yapi.hidden_size));
    c.push_str(&format!("| kafa | {} x {} |\n", yapi.num_attention_heads, yapi.kafa_genisligi()));
    c.push_str(&format!("| sozluk | {} |\n", yapi.vocab_size));
    c.push_str(&format!("| pencere | {} ({:?}) |\n", yapi.local_attention, yapi.pencere_kurali));
    c.push_str(&format!("| theta | {} |\n", yapi.theta(lubot_kodlayici::KatmanTuru::FullAttention)));
    let tam = yapi
        .layer_types
        .iter()
        .filter(|t| **t == lubot_kodlayici::KatmanTuru::FullAttention)
        .count();
    c.push_str(&format!(
        "| dikkat | {tam} tam, {} kayan |\n",
        yapi.layer_types.len() - tam
    ));
    // The head's own file is read when it is there: a report about the encoder
    // that stayed silent about the head would describe half the artifact.
    let kafa_yolu = paket.join("rl_agent_config.json");
    if kafa_yolu.is_file() {
        match KafaYapisi::oku(&kafa_yolu) {
            Ok(kafa) => {
                c.push_str(&format!("| kafa katmani | {} |\n", kafa.head_layers));
                c.push_str(&format!("| en fazla onek | {} |\n", kafa.max_prefixes));
                if let Some(maliyet) = kafa.escalate_maliyeti() {
                    c.push_str(&format!("| escalate maliyeti | {maliyet} |\n"));
                }
                c.push_str(&format!("| yanlis aksiyon maliyeti | {} |\n", kafa.cost_wrong_act));
                if let Some(esik) = kafa.esik_dogruluk() {
                    c.push_str(&format!(
                        "| basabas dogruluk | {esik:.4} (bu esigin ustunde aksiyon, altinda escalate) |\n"
                    ));
                }
            }
            Err(hata) => c.push_str(&format!("| kafa | okunamadi: {hata} |\n")),
        }
    }
    // Which tensors nothing reads. A header that is mostly unused means the
    // weights are being read wrongly, and that is worth knowing before the
    // first number is believed.
    let mut beklenen: Vec<String> = kullanilan_adlar(&yapi);
    if let Ok(karar) = KararYapisi::oku(&paket) {
        beklenen.extend(kafa_adlari(&karar, &yapi));
    }
    let kullanilmayan: Vec<String> = dizin
        .adlar()
        .into_iter()
        .filter(|ad| !beklenen.iter().any(|b| b == ad))
        .map(str::to_string)
        .collect();
    c.push_str(&format!("\n- okunan tensor: {}\n", dizin.uzunluk() - kullanilmayan.len()));
    c.push_str(&format!("- okunmayan tensor: {}\n", kullanilmayan.len()));
    if !kullanilmayan.is_empty() {
        c.push_str(&format!("- ornek: {}\n", kullanilmayan.iter().take(8).cloned().collect::<Vec<_>>().join(", ")));
    }
    print!("{c}");
    Ok(())
}

/// The window shape named on the command line, or the checkpoint's own default.
fn pencere_kurali(ad: Option<&str>) -> Result<PencereKurali, String> {
    match ad {
        None => Ok(PencereKurali::ReferansYarisi),
        Some("referans") => Ok(PencereKurali::ReferansYarisi),
        Some("sol") => Ok(PencereKurali::SolPencere),
        Some("simetrik") => Ok(PencereKurali::Simetrik),
        Some(digeri) => Err(format!(
            "bilinmeyen pencere kurali `{digeri}`: referans | sol | simetrik"
        )),
    }
}

/// Verifies the package against an expected hash of the whole artifact.
///
/// The parts are hashed in name order as one stream, which is what the merge
/// script does; hashing them separately would prove nothing about the artifact
/// the model was trained as.
fn dogrula(args: &[String]) -> Result<(), String> {
    let paket = paket(args)?;
    let beklenen = deger(args, "--beklenen-sha")
        .ok_or("usage: lubot kodlayici dogrula --paket <dizin> --beklenen-sha <64 hex>")?;
    if beklenen.len() != 64 || !beklenen.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("`{beklenen}` 64 haneli sha256 degil"));
    }
    let dosya = ParcaliDosya::ac(&paket, "model.safetensors.part-")
        .map_err(|h| format!("{}: {h:?}", paket.display()))?;
    let ozet = dosya.sha256().map_err(|h| format!("{h:?}"))?;
    let uyum = ozet.eq_ignore_ascii_case(&beklenen);
    println!("# Paket dogrulamasi\n");
    println!("- paket: `{}`", paket.display());
    println!("- parca: {}", dosya.parca_sayisi());
    println!("- bayt: {}", dosya.boyut());
    println!("- beklenen sha256: `{beklenen}`");
    println!("- olculen sha256: `{ozet}`");
    println!("- sonuc: {}", if uyum { "ayni" } else { "FARKLI" });
    if uyum {
        Ok(())
    } else {
        Err("paket beklenen ozetle ayni degil".to_string())
    }
}

/// Runs the encoder over a token sequence.
fn kosu(args: &[String]) -> Result<(), String> {
    let paket = paket(args)?;
    // `--metin` turns the whole path on: the text is tokenized by this
    // repository's own port of the vocabulary, and the ids it produces are what
    // the encoder runs on. Without it the command runs on ids given directly,
    // which is what the cross-check compares.
    let metin = match deger(args, "--metin-dosya") {
        Some(yol) => Some(
            std::fs::read_to_string(&yol).map_err(|e| format!("{yol}: {e}"))?,
        ),
        None => deger(args, "--metin"),
    };

    // The window rule is validated before anything else is demanded: it is the
    // one flag whose value a caller can get wrong in a way that still runs, and
    // a usage error about a missing token list would hide the typo.
    let pencere = pencere_kurali(deger(args, "--pencere").as_deref())?;
    let kimlikler = deger(args, "--kimlik").unwrap_or_default();
    let jetonlar: Vec<u32> = kimlikler
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            p.trim()
                .parse::<u32>()
                .map_err(|h| format!("`{p}` jeton degil: {h}"))
        })
        .collect::<Result<Vec<u32>, String>>()?;
    let mut yapi = yapi_oku(&paket)?;
    yapi.pencere_kurali = pencere;
    let jetonlar = if let Some(metin) = &metin {
        let sozluk = Sozluk::oku(&paket.join("tokenizer").join("tokenizer.json"))
            .map_err(|h| format!("{h:?}"))?;
        let idler = sozluk.jetonla(metin);
        println!(
            "# Metin\n\n- yazi: {} bayt\n- jeton: {}\n- ids: {}\n",
            metin.len(),
            idler.len(),
            idler
                .iter()
                .take(24)
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        );
        idler
    } else {
        jetonlar
    };
    if jetonlar.is_empty() {
        return Err("usage: --kimlik 1,2,3 ya da --metin <yazi> gerekli".to_string());
    }
    let tip = match deger(args, "--tip") {
        Some(ad) => tip_indeksi(&ad)
            .ok_or_else(|| format!("bilinmeyen tip `{ad}`: {}", TIPLER.join(" | ")))?,
        None => 0,
    };
    let mut isaretler: Vec<usize> = match deger(args, "--isaret") {
        Some(liste) => liste
            .split(',')
            .filter(|p| !p.trim().is_empty())
            .map(|p| {
                p.trim()
                    .parse::<usize>()
                    .map_err(|h| format!("`{p}` konum degil: {h}"))
            })
            .collect::<Result<Vec<usize>, String>>()?,
        None => Vec::new(),
    };
    // With text and no marker named there is only one sensible position - the
    // last one - and a run that reports a decision is more useful than one that
    // asks for a position the caller thought was implied.
    if isaretler.is_empty() && metin.is_some() {
        isaretler.push(jetonlar.len() - 1);
    }
    let dosya = ParcaliDosya::ac(&paket, "model.safetensors.part-")
        .map_err(|h| format!("{}: {h:?}", paket.display()))?;
    let dizin = Dizin::oku(&dosya).map_err(|h| format!("{h:?}"))?;
    let agirliklar = Agirliklar::yukle(yapi, &dosya, &dizin).map_err(|h| format!("{h:?}"))?;
    let gizli = kodla(&agirliklar, &jetonlar).map_err(|h| format!("{h:?}"))?;
    // An optional dump of the final hidden states as raw little-endian `f32`,
    // which is the input a second implementation is given when the two are
    // compared: a text dump would round before the comparison starts.
    if let Some(yol) = deger(args, "--cikti") {
        let mut ham = Vec::with_capacity(gizli.len() * 4);
        for v in &gizli {
            ham.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&yol, &ham).map_err(|e| format!("{yol}: {e}"))?;
        println!(
            "- gizli durum dokumu: `{yol}` ({} bayt, {} x {})",
            ham.len(),
            jetonlar.len(),
            agirliklar.yapi.hidden_size
        );
    }
    let havuz = ortalama_havuz(&gizli, agirliklar.yapi.hidden_size).map_err(|h| format!("{h:?}"))?;

    let mut c = String::new();
    c.push_str("# Kodlayici kosusu\n\n");
    c.push_str(&format!("- paket: `{}`\n", paket.display()));
    c.push_str(&format!("- jeton: {} ({})\n", jetonlar.len(), kimlikler));
    c.push_str(&format!(
        "- pencere kurali: {:?}\n",
        agirliklar.yapi.pencere_kurali
    ));
    c.push_str(&format!(
        "- gizli durum: {} x {}\n",
        jetonlar.len(),
        agirliklar.yapi.hidden_size
    ));
    let ort = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let min = gizli.iter().copied().fold(f32::INFINITY, f32::min);
    let max = gizli.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let nan = gizli.iter().filter(|v| !v.is_finite()).count();
    c.push_str(&format!("- ortalama: {:.6}\n", ort(&gizli)));
    c.push_str(&format!("- en kucuk: {min:.6}\n"));
    c.push_str(&format!("- en buyuk: {max:.6}\n"));
    c.push_str(&format!("- sonsuz/NaN: {nan}\n\n"));
    c.push_str("Ilk 8 havuz degeri:\n\n");
    c.push_str("| # | deger |\n|---:|---:|\n");
    for (sira, deger) in havuz.iter().take(8).enumerate() {
        c.push_str(&format!("| {sira} | {deger:.6} |\n"));
    }
    c.push('\n');
    if isaretler.is_empty() {
        c.push_str(
            "Bu cikti bir *temsildir*: kodlayicinin son gizli durumu. Anlami kafanin isidir;\n\
             kafa kosulmadan hicbir iddia edilmez. `--isaret` verilince kafa da kosar.\n",
        );
    } else {
        let karar = KararYapisi::oku(&paket).map_err(|h| format!("{h:?}"))?;
        let kafa = KararAgirliklari::yukle(&dosya, &dizin, &agirliklar.yapi, &karar)
            .map_err(|h| format!("{h:?}"))?;
        let cevap = puanla(&kafa, &gizli, jetonlar.len(), &isaretler, tip)
            .map_err(|h| format!("{h:?}"))?;
        c.push_str("# Karar\n\n");
        c.push_str(&format!("- tip: {} ({tip})\n", TIPLER[tip]));
        c.push_str(&format!("- isaret: {isaretler:?}\n"));
        c.push_str(&format!(
            "- sicaklik: {:.4}\n",
            karar.sicaklik.get(tip).copied().unwrap_or(1.0)
        ));
        c.push_str(&format!("- basabas dogruluk: {:.4}\n\n", karar.basabas));
        c.push_str("| secenek | puan | olasilik |\n|---:|---:|---:|\n");
        for (sira, (puan, olasilik)) in cevap.puanlar.iter().zip(cevap.olasiliklar.iter()).enumerate() {
            c.push_str(&format!("| {sira} | {puan:.6} | {olasilik:.6} |\n"));
        }
        if let Some(secim) = cevap.secim() {
            c.push_str(&format!("\n- secim: {secim}\n"));
        }
        c.push_str(&format!("- beklenen indeks: {:.6}\n", cevap.beklenen_indeks()));
        if tip == 2 {
            if let Some(evet) = cevap.evet() {
                c.push_str(&format!("- noul (evet): {evet:.6}\n"));
            }
        }
        let en_buyuk = cevap.eylem_olasiliklari.iter().copied().fold(f32::MIN, f32::max);
        c.push_str(&format!("\n- eylem puanlari: {:?}\n", cevap.eylem_puanlari));
        c.push_str(&format!("- eylem olasiliklari: {:?}\n", cevap.eylem_olasiliklari));
        c.push_str(&format!("- en yuksek eylem: {en_buyuk:.6}\n"));
        c.push_str(&format!("- ozellikler: {:?}\n", cevap.ozellikler));
        c.push_str(
            "\nOlasiliklar modelin kendi dagilimidir; bir esikle aksiyona cevrilmesi\n\
             yukaridaki basabas sayisinin isidir ve onu uygulayan ayri bir adimdir.\n",
        );
    }
    print!("{c}");
    Ok(())
}

/// The hidden size and layer count of a package, for the status line.
///
/// # Errors
/// The package path or the configuration, named.
pub fn kodlayici_ozeti(paket: &Path) -> Result<String, String> {
    let yapi = yapi_oku(paket)?;
    Ok(format!(
        "kodlayici: {} katman, gizli {}, sozluk {}, pencere {}",
        yapi.num_hidden_layers, yapi.hidden_size, yapi.vocab_size, yapi.local_attention
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_package_flag_is_required_and_read() {
        assert!(paket(&[]).is_err());
        let args = vec!["--paket".to_string(), "/tmp/x".to_string()];
        assert_eq!(paket(&args).unwrap_or_default(), PathBuf::from("/tmp/x"));
        assert_eq!(deger(&args, "--paket"), Some("/tmp/x".to_string()));
        assert_eq!(deger(&args, "--yok"), None);
    }

    #[test]
    fn a_token_list_with_a_typo_is_refused_not_skipped() {
        let jetonlar: Result<Vec<u32>, String> = "1,2,x"
            .split(',')
            .map(|p| p.parse::<u32>().map_err(|h| h.to_string()))
            .collect();
        assert!(jetonlar.is_err(), "bozuk jeton sessizce atlanmamali");
    }

    #[test]
    fn an_unknown_window_rule_lists_the_ones_that_exist() {
        let args = vec![
            "--pencere".to_string(),
            "yanlis".to_string(),
            "--paket".to_string(),
            "/tmp".to_string(),
        ];
        let hata = kosu(&args).unwrap_err();
        assert!(hata.contains("referans"), "{hata}");
        assert!(pencere_kurali(None).is_ok());
        assert_eq!(pencere_kurali(Some("sol")).unwrap_or(PencereKurali::Simetrik), PencereKurali::SolPencere);
        assert!(pencere_kurali(Some("yanlis")).is_err());
    }

    #[test]
    fn the_summary_refuses_a_missing_package_rather_than_guessing() {
        assert!(kodlayici_ozeti(Path::new("/tmp/lubot-yok-boyle-dizin")).is_err());
    }
}
