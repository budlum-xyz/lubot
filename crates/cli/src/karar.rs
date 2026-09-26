//! # karar - the decision head, as a command
//!
//! `lubot karar` states the doctrine that [`lubot_tomurcuk`] implements and
//! measures the part of it that can be measured without a training run: the
//! closed decision points, the fixed tier order, the effort band, and the state
//! of the confidence ledger. `karar tek` runs one head, `karar oyla` runs the
//! k-of-n vote.
//!
//! The ledger starts empty and is reported as unmeasured rather than as clean:
//! a head with no recorded outcomes has not earned the right to decide alone.

use lubot_tomurcuk::{
    efor_butcesi, kademe_atlandi, konsensus, tek_bas, EforButcesi, EforHatasi, EvetHayirKarari,
    Guven, GuvenDefteri, Karar, KararTipi, Konsensus, Oncelik, Politika, PolitikaHatasi, Puan,
    Secenek, Sonuc, YukseltmeNedeni, EFOR_TABAN, EFOR_TAVAN, GUVEN_ESIGI, KONSENSUS_K, KONSENSUS_N,
};

/// Reports the doctrine, or runs a single head or a k-of-n vote.
pub fn cmd_karar(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("doktrin") => doktrin(),
        Some("tek") => tek(&args[1..]),
        Some("oyla") => oyla(&args[1..]),
        Some(other) => Err(format!(
            "unknown karar subcommand: {other}\nusage: lubot karar [doktrin|tek|oyla] ..."
        )),
    }
}

/// A rendered decision, in the words a report can carry.
fn sonuc_satiri(sonuc: Sonuc) -> String {
    match sonuc {
        Sonuc::Kesin(karar) => format!("kesin ({})", karar_tipi_adi(karar.tipi())),
        Sonuc::Yukselt(neden) => format!("yukselt ({})", neden_adi(neden)),
        Sonuc::Red => "red".to_string(),
    }
}

fn karar_tipi_adi(tip: KararTipi) -> &'static str {
    match tip {
        KararTipi::Secenek => "secenek",
        KararTipi::Puan => "puan",
        KararTipi::EvetHayir => "evet-hayir",
    }
}

fn neden_adi(neden: YukseltmeNedeni) -> &'static str {
    match neden {
        YukseltmeNedeni::GuvenEsikAltinda => "guven esigin altinda",
        YukseltmeNedeni::KonsensusYok => "konsensus yok",
        YukseltmeNedeni::TipKarisik => "tip karisik",
        YukseltmeNedeni::OySayisiEksik => "oy sayisi eksik",
    }
}

fn politika_hatasi_adi(hata: PolitikaHatasi) -> &'static str {
    match hata {
        PolitikaHatasi::KonsensusGecersiz => "konsensus k/n gecersiz",
        PolitikaHatasi::EsikAralikDisi => "guven esigi 0..=1 disinda",
    }
}

/// One yes-or-no vote, parsed from `evet`/`hayir` and a probability.
fn oy_oku(parca: &str) -> Result<Karar, String> {
    let (yon, olasilik) = parca
        .split_once(':')
        .ok_or_else(|| format!("oy `{parca}` `evet:0.9` biciminde degil"))?;
    let deger: f64 = olasilik
        .parse()
        .map_err(|_| format!("oy `{parca}`: `{olasilik}` bir sayi degil"))?;
    let puan =
        Puan::yeni(deger).ok_or_else(|| format!("oy `{parca}`: {deger} 0..=1 araliginda degil"))?;
    match yon {
        "evet" => Ok(Karar::EvetHayir(EvetHayirKarari {
            evet: true,
            olasilik: puan,
            guven: Guven(puan),
        })),
        "hayir" => Ok(Karar::EvetHayir(EvetHayirKarari {
            evet: false,
            olasilik: puan.tumleyen(),
            guven: Guven(puan),
        })),
        other => Err(format!("oy `{parca}`: `{other}` evet ya da hayir degil")),
    }
}

fn efor_satiri(efor: f64) -> String {
    match efor_butcesi(efor) {
        Ok(EforButcesi {
            baslik_sayisi,
            konsensus_zorunlu,
            uretken_izinli,
        }) => format!(
            "| {efor} | {baslik_sayisi} | {} | {} |",
            evet_hayir(konsensus_zorunlu),
            evet_hayir(uretken_izinli)
        ),
        Err(EforHatasi::AralikDisi) => format!("| {efor} | reddedildi: bant disi | - | - |"),
    }
}

fn evet_hayir(deger: bool) -> &'static str {
    if deger {
        "evet"
    } else {
        "hayir"
    }
}

/// The doctrine and the state of its ledger, rendered.
fn doktrin() -> Result<(), String> {
    let md = doktrin_report()?;
    crate::validate_output(md.as_bytes(), "karar")?;
    print!("{md}");
    Ok(())
}

/// Builds the doctrine report. Split from [`doktrin`] so a test can read what
/// the command would print without capturing stdout.
fn doktrin_report() -> Result<String, String> {
    let politika = Politika::varsayilan();
    if let Err(hata) = politika.dogrula() {
        return Err(format!(
            "beyan edilen politika gecersiz: {}",
            politika_hatasi_adi(hata)
        ));
    }
    let mut md = String::from("# Karar basligi (tomurcuk)\n\n");
    md.push_str("Kapali karar noktalari, cikti yuzeyi uc kapali sekil:\n\n");
    md.push_str("| # | nokta |\n|---|---|\n");
    for (i, nokta) in Secenek::HEPSI.iter().enumerate() {
        md.push_str(&format!("| {} | {} |\n", i + 1, nokta.ad()));
    }
    md.push_str("\n## Sabit sira\n\n| kademe | sira |\n|---|---|\n");
    for (i, kademe) in Oncelik::SIRALI.iter().enumerate() {
        md.push_str(&format!("| {} | {} |\n", kademe.ad(), i + 1));
    }
    md.push_str(&format!(
        "\nKademe atlandi: {} (beyan edilen siranin kendisi denetlendi)\n",
        evet_hayir(kademe_atlandi(&Oncelik::SIRALI))
    ));
    md.push_str("\n## Politika\n\n| alan | deger |\n|---|---|\n");
    md.push_str(&format!(
        "| guven esigi | {:.2} |\n",
        politika.guven_esigi.0.deger()
    ));
    md.push_str(&format!(
        "| konsensus | {}/{} |\n",
        politika.konsensus_k, politika.konsensus_n
    ));
    md.push_str(&format!("| beyan edilen esik sabiti | {GUVEN_ESIGI} |\n"));
    md.push_str(&format!("| efor bandi | {EFOR_TABAN}x-{EFOR_TAVAN}x |\n"));
    md.push_str("\n## Efor -> butce\n\n| efor | baslik | konsensus zorunlu | uretken izinli |\n|---|---|---|---|\n");
    for efor in [EFOR_TABAN, 1.0, 2.0, EFOR_TAVAN] {
        md.push_str(&format!("{}\n", efor_satiri(efor)));
    }
    md.push_str("\n## Guven defteri\n\n| olculen | deger |\n|---|---|\n");
    let defter = GuvenDefteri::yeni();
    md.push_str(&format!("| kayitli sonuc | {} |\n", defter.toplam()));
    md.push_str(&format!(
        "| en kotu bosluk | {} |\n",
        match defter.en_kotu_bosluk() {
            Some(bosluk) => format!("{:.3}", bosluk.gap),
            None => "olculmedi (egitim kosusu gerektirir)".to_string(),
        }
    ));
    md.push_str(&format!(
        "| tek basina karar verebilir | {} (defter bos: olculmemis bir baslik yalniz karar veremez) |\n",
        evet_hayir(defter.tek_bas_guvenli(0.05))
    ));
    Ok(md)
}

/// Runs one head on a yes-or-no question.
fn tek(args: &[String]) -> Result<(), String> {
    let Some(arg) = args.first() else {
        return Err("usage: lubot karar tek <evet|hayir>:<olasilik>".to_string());
    };
    let karar = oy_oku(arg)?;
    let politika = Politika::varsayilan();
    let sonuc = tek_bas(karar, &politika)
        .map_err(|hata| format!("politika gecersiz: {}", politika_hatasi_adi(hata)))?;
    let mut md = String::from("# Karar basligi: tek bas\n\n");
    md.push_str("| alan | deger |\n|---|---|\n");
    md.push_str(&format!("| oy | {arg} |\n"));
    md.push_str(&format!("| sekil | {} |\n", karar_tipi_adi(karar.tipi())));
    md.push_str(&format!("| guven | {:.2} |\n", karar.guven().0.deger()));
    md.push_str(&format!("| sonuc | {} |\n", sonuc_satiri(sonuc)));
    crate::validate_output(md.as_bytes(), "karar tek")?;
    print!("{md}");
    Ok(())
}

/// Runs the k-of-n vote over independently initialised heads.
fn oyla(args: &[String]) -> Result<(), String> {
    let Some(liste) = args.first() else {
        return Err(format!(
            "usage: lubot karar oyla evet:0.9,hayir:0.7,evet:0.8  ({KONSENSUS_K}/{KONSENSUS_N} varsayilan)"
        ));
    };
    let oylar: Vec<Karar> = liste
        .split(',')
        .map(oy_oku)
        .collect::<Result<Vec<_>, _>>()?;
    let politika = Politika::varsayilan();
    let rapor: Konsensus = konsensus(&oylar, &politika)
        .map_err(|hata| format!("politika gecersiz: {}", politika_hatasi_adi(hata)))?;
    let mut md = String::from("# Karar basligi: k-of-n\n\n");
    md.push_str("| alan | deger |\n|---|---|\n");
    md.push_str(&format!("| oylar | {liste} |\n"));
    md.push_str(&format!("| evet oyu | {} |\n", rapor.evet_oyu));
    md.push_str(&format!("| hayir oyu | {} |\n", rapor.hayir_oyu));
    md.push_str(&format!("| beklenen n | {} |\n", rapor.beklenen_n));
    md.push_str(&format!("| sonuc | {} |\n", sonuc_satiri(rapor.sonuc)));
    md.push_str("\nBu bir model-ici konsensustur; zincirin operator esigi (K5) ile ayni sayi degildir ve ayni RPC yuzeyini kullanmaz.\n");
    crate::validate_output(md.as_bytes(), "karar oyla")?;
    print!("{md}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_tomurcuk::{PuanKarari, SecenekKarari};

    #[test]
    fn the_doctrine_report_is_schema_valid_and_names_the_points() {
        let rapor = doktrin_report().unwrap();
        assert!(rapor.contains("arac-yonlendirici"));
        assert!(rapor.contains("belirlenimci-kod"));
        assert!(rapor.contains("olculmedi (egitim kosusu gerektirir)"));
    }

    #[test]
    fn a_malformed_vote_is_refused_not_parsed_as_zero() {
        assert!(oy_oku("evet").is_err());
        assert!(oy_oku("belki:0.5").is_err());
        assert!(oy_oku("evet:1.4").is_err());
        assert!(oy_oku("evet:x").is_err());
    }

    #[test]
    fn the_effort_band_edges_are_reported() {
        assert!(efor_satiri(EFOR_TABAN).contains("| 1 | hayir | hayir |"));
        assert!(efor_satiri(EFOR_TAVAN).contains("evet | evet |"));
        assert!(efor_satiri(0.1).contains("reddedildi"));
    }

    #[test]
    fn the_three_closed_shapes_render() {
        assert_eq!(
            karar_tipi_adi(
                Karar::Secenek(SecenekKarari {
                    secim: Secenek::KapsamReddi,
                    guven: Guven(Puan::SIFIR),
                })
                .tipi()
            ),
            "secenek"
        );
        assert_eq!(
            karar_tipi_adi(
                Karar::Puan(PuanKarari {
                    deger: Puan::BIR,
                    guven: Guven(Puan::BIR),
                })
                .tipi()
            ),
            "puan"
        );
    }
}
