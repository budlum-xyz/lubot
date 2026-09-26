//! # kanaat - the local decision engine, as a command
//!
//! `lubot kanaat batarya` runs the shipped battery and prints, per case, what
//! was expected and what the engine produced; `lubot kanaat ver` decides one
//! case read from a file; `lubot kanaat doktrin` prints the knobs and the
//! doctrine the engine answers to.
//!
//! Everything on stdout is Markdown, like every other command here. A failing
//! expectation is reported in the table and, with `--kati`, as a failure exit:
//! printing a table nobody fails on is how a battery rots.

use lubot_kanaat::{
    batarya_kos, batarya_oku, hukum_metni, karar_ver, Ayarlar, BataryaRaporu, Dava, Defter,
    Gerekce, Hukum, RedSebebi, SecenekPuan, VakaSonucu, Yukseltme, GOMULU_BATARYA,
};

/// Dispatches the `kanaat` subcommands.
pub fn cmd_kanaat(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("doktrin") => {
            print!("{}", doktrin_md(&Ayarlar::default()));
            Ok(())
        }
        Some("batarya") => batarya(&args[1..]),
        Some("ver") => ver(&args[1..]),
        Some("defter") => defter(&args[1..]),
        Some(other) => Err(format!(
            "unknown kanaat subcommand: {other}\nusage: lubot kanaat [doktrin|batarya|ver] ..."
        )),
    }
}

/// The value after `--ad`, if the flag is present with a value.
fn deger(args: &[String], ad: &str) -> Option<String> {
    let sira = args.iter().position(|a| a == ad)?;
    args.get(sira + 1).cloned()
}

/// The policy and the knobs, as a page a reader can check the engine against.
#[must_use]
pub fn doktrin_md(ayar: &Ayarlar) -> String {
    let mut c = String::new();
    c.push_str("# Kanaat motoru - doktrin\n\n");
    c.push_str("Motor kanittan hukum cikarir; kanit yoksa, marj yoksa ya da guven\n");
    c.push_str("esigin altinda kalirsa secmez. Red bir cevaptir.\n\n");
    c.push_str("| ayar | deger |\n|---|---|\n");
    c.push_str(&format!("| en az kanit | {} |\n", ayar.en_az_kanit));
    c.push_str(&format!("| en az kapsam | {:.2} |\n", ayar.en_az_kapsam));
    c.push_str(&format!("| marj esigi | {:.2} |\n", ayar.marj_esigi));
    c.push_str(&format!("| marj tavan | {:.2} |\n", ayar.marj_tam));
    c.push_str(&format!("| destek tavan | {} |\n", ayar.destek_tam));
    c.push_str(&format!(
        "| olumsuzluk cezasi | {:.2} |\n",
        ayar.olumsuzluk_cezasi
    ));
    c.push_str(&format!("| sayi bonusu | {:.2} |\n", ayar.sayi_bonusu));
    c.push_str(&format!(
        "| sayi celiskisi | {:.2} |\n",
        ayar.sayi_celiskisi
    ));
    c.push_str(&format!(
        "| guven esigi (doktrin) | {:.2} |\n",
        ayar.politika.guven_esigi.0.deger()
    ));
    c.push('\n');
    c.push_str("Sirasi sabittir: eksik olan, zayif olan, sonra doktrin esigi.\n");
    c
}

/// Runs the shipped battery, or one named with `--dosya`.
fn batarya(args: &[String]) -> Result<(), String> {
    // With no `--dosya` the battery compiled into the binary is run: a report
    // that depends on the working directory is a report that changes when the
    // command is run from somewhere else.
    let (kaynak, metin) = match deger(args, "--dosya") {
        Some(dosya) => {
            let metin = std::fs::read_to_string(&dosya).map_err(|h| format!("{dosya}: {h}"))?;
            (dosya, metin)
        }
        None => ("gomulu".to_string(), GOMULU_BATARYA.to_string()),
    };
    let batarya = batarya_oku(&metin).map_err(|h| format!("{kaynak}: {h:?}"))?;
    let ayar = Ayarlar::default();
    let rapor: BataryaRaporu = batarya_kos(&batarya, &ayar);
    let mut c = String::new();
    c.push_str("# Kanaat bataryasi\n\n");
    c.push_str(&format!("- dosya: `{kaynak}`\n"));
    c.push_str(&format!("- vaka: {}\n", rapor.toplam));
    c.push_str(&format!(
        "- dogru: {} ({:.1}%)\n\n",
        rapor.dogru,
        rapor.oran() * 100.0
    ));
    c.push_str("| vaka | beklenen | gelen | sonuc |\n|---|---|---|---|\n");
    for vaka in &rapor.vakalar {
        c.push_str(&vaka_satiri(vaka));
    }
    c.push('\n');
    if let Some(kayit) = deger(args, "--kayit") {
        let govde = serde_json::to_string_pretty(&rapor).map_err(|h| h.to_string())?;
        std::fs::write(&kayit, govde + "\n").map_err(|h| format!("{kayit}: {h}"))?;
        c.push_str(&format!("Kayit yazildi: `{kayit}`\n"));
    }
    print!("{c}");
    if args.iter().any(|a| a == "--kati") && rapor.yanlis > 0 {
        return Err(format!("{} vaka tutmadi", rapor.yanlis));
    }
    Ok(())
}

/// Decides one case read from a file.
fn ver(args: &[String]) -> Result<(), String> {
    let dosya = deger(args, "--dosya").ok_or("usage: lubot kanaat ver --dosya <vaka.json>")?;
    let metin = std::fs::read_to_string(&dosya).map_err(|h| format!("{dosya}: {h}"))?;
    let dava: Dava = serde_json::from_str(&metin).map_err(|h| format!("{dosya}: {h}"))?;
    let ayar = Ayarlar::default();
    let hukum = karar_ver(&dava, &ayar);
    print!("{}", hukum_md(&dava, &hukum));
    Ok(())
}

/// The append-only verdict ledger: `yaz`, `dogrula`, `liste`.
fn defter(args: &[String]) -> Result<(), String> {
    let alt = args
        .first()
        .map(String::as_str)
        .ok_or("usage: lubot kanaat defter [yaz|dogrula|liste] --dosya <defter.jsonl>")?;
    let kalan = &args[1..];
    let yol = deger(kalan, "--dosya").unwrap_or_else(|| "outputs/kanaat-defteri.jsonl".to_string());
    match alt {
        "yaz" => {
            let vaka = deger(kalan, "--vaka")
                .ok_or("usage: lubot kanaat defter yaz --dosya <defter.jsonl> --vaka <vaka.json>")?;
            let vaka_metni =
                std::fs::read_to_string(&vaka).map_err(|h| format!("{vaka}: {h}"))?;
            let dava: Dava =
                serde_json::from_str(&vaka_metni).map_err(|h| format!("{vaka}: {h}"))?;
            let hukum = karar_ver(&dava, &Ayarlar::default());
            // The ledger is read and re-verified before it is appended to: a
            // ledger that is already broken must not be extended, or every
            // later entry would be built on a link nobody checked.
            let mut mevcut = match std::fs::read_to_string(&yol) {
                Ok(metin) => Defter::oku(&metin).map_err(|h| format!("{yol}: {h:?}"))?,
                Err(_) => Defter::yeni(),
            };
            let kayit = mevcut.ekle(&dava.soru, &hukum);
            let metin = mevcut.jsonl()?;
            if let Some(klasor) = std::path::Path::new(&yol).parent() {
                if !klasor.as_os_str().is_empty() {
                    std::fs::create_dir_all(klasor).map_err(|h| format!("{}: {h}", klasor.display()))?;
                }
            }
            std::fs::write(&yol, metin).map_err(|h| format!("{yol}: {h}"))?;
            print!(
                "# Kanaat defteri\n\n- dosya: `{yol}`\n- sira: {}\n- hukum: {}\n- ozet: `{}`\n- zincir ucu: `{}`\n",
                kayit.sira,
                kayit.hukum,
                kayit.ozet,
                mevcut.uc()
            );
            Ok(())
        }
        "dogrula" => {
            let metin = std::fs::read_to_string(&yol).map_err(|h| format!("{yol}: {h}"))?;
            let defter = Defter::oku(&metin).map_err(|h| format!("{yol}: {h:?}"))?;
            let mut c = format!(
                "# Kanaat defteri dogrulamasi\n\n- dosya: `{yol}`\n- kayit: {}\n- zincir ucu: `{}`\n",
                defter.uzunluk(),
                defter.uc()
            );
            // An anchor given from outside turns "internally consistent" into
            // "the same ledger I recorded", which is the only way to notice a
            // truncated tail.
            if let Some(capa) = deger(kalan, "--capa") {
                match defter.dogrula_uc(&capa) {
                    Ok(()) => {
                        c.push_str(&format!(
                            "- capa: dogrulandi (`{}`)\n",
                            capa.trim().to_ascii_lowercase()
                        ));
                        print!("{c}");
                        Ok(())
                    }
                    Err(hata) => {
                        c.push_str(&format!("- capa: UYUSMADI ({hata:?})\n"));
                        print!("{c}");
                        Err(format!("{hata:?}"))
                    }
                }
            } else {
                print!("{c}");
                Ok(())
            }
        }
        "liste" => {
            let metin = std::fs::read_to_string(&yol).map_err(|h| format!("{yol}: {h}"))?;
            let defter = Defter::oku(&metin).map_err(|h| format!("{yol}: {h:?}"))?;
            let mut c = format!("# Kanaat defteri\n\n- dosya: `{yol}`\n\n| sira | hukum | soru | ozet |\n|---:|---|---|---|\n");
            for kayit in defter.kayitlar() {
                c.push_str(&format!(
                    "| {} | {} | `{}` | `{}` |\n",
                    kayit.sira,
                    kayit.hukum,
                    &kayit.soru_damgasi[..8.min(kayit.soru_damgasi.len())],
                    &kayit.ozet[..12.min(kayit.ozet.len())]
                ));
            }
            print!("{c}");
            Ok(())
        }
        other => Err(format!(
            "unknown kanaat defter subcommand: {other}\nusage: lubot kanaat defter [yaz|dogrula|liste] --dosya <defter.jsonl>"
        )),
    }
}

/// Renders one verdict as Markdown.
#[must_use]
pub fn hukum_md(dava: &Dava, hukum: &Hukum) -> String {
    let mut c = String::new();
    c.push_str("# Kanaat\n\n");
    c.push_str(&format!("- soru: {}\n", dava.soru));
    c.push_str(&format!("- karar: {}\n", hukum_metni(hukum)));
    match hukum {
        Hukum::Secim(secim) => {
            let secilen = dava
                .secenekler
                .get(secim.indeks)
                .map_or("(yok)", String::as_str);
            c.push_str(&format!("- secilen: `{secilen}`\n"));
            c.push_str(&format!("- puan: {:.4}\n", secim.puan.deger()));
            c.push_str(&format!("- guven: {:.4}\n", secim.guven.0.deger()));
            c.push_str(&gerekce_satirlari(&secim.gerekce));
            c.push_str(&aday_tablosu(&secim.puanlar, dava));
        }
        Hukum::Yukselt(neden) => c.push_str(&format!("- neden: {}\n", yukseltme_metni(neden))),
        Hukum::Red(neden) => c.push_str(&format!("- neden: {}\n", red_metni(*neden))),
    }
    c.push('\n');
    c
}

/// The reasoning lines of a verdict: the numbers the decision read.
fn gerekce_satirlari(gerekce: &Gerekce) -> String {
    let mut c = format!(
        "- marj: {:.4} | kapsam: {:.4} | destek: {}\n",
        gerekce.marj, gerekce.kapsam, gerekce.destek
    );
    if !gerekce.dayanaklar.is_empty() {
        c.push_str(&format!("- dayanak: {}\n", gerekce.dayanaklar.join(", ")));
    }
    c
}

/// Every candidate's score, in the engine's own order.
///
/// The losing candidates are printed with the winner on purpose: a verdict
/// without the alternatives is a claim that nothing else was close, and the
/// margin is only readable next to the numbers it came from.
fn aday_tablosu(puanlar: &[SecenekPuan], dava: &Dava) -> String {
    let mut c = String::from("\n| # | aday | puan | kapsam | destek | olumsuzluk | sayi |\n");
    c.push_str("|---:|---|---:|---:|---:|---|---|\n");
    for puan in puanlar {
        let aday = dava
            .secenekler
            .get(puan.indeks)
            .map_or("(yok)", String::as_str);
        c.push_str(&format!(
            "| {} | {} | {:.4} | {:.2} | {} | {} | {} |\n",
            puan.indeks,
            aday,
            puan.puan,
            puan.kapsam,
            puan.destek,
            if puan.olumsuzluk_celiskisi {
                "celiski"
            } else {
                "-"
            },
            if puan.sayi_celiskisi { "celiski" } else { "-" }
        ));
    }
    c
}

/// One battery case as a table row.
fn vaka_satiri(vaka: &VakaSonucu) -> String {
    format!(
        "| {} | {} | {} | {} |\n",
        vaka.ad,
        vaka.beklenen,
        vaka.gelen,
        if vaka.dogru { "tuttu" } else { "TUTMADI" }
    )
}

fn yukseltme_metni(neden: &Yukseltme) -> String {
    match neden {
        Yukseltme::DestekYetersiz { destek, gereken } => {
            format!("destek yetersiz ({destek}/{gereken})")
        }
        Yukseltme::KapsamDusuk { kapsam, gereken } => {
            format!("kapsam dusuk ({kapsam:.2} < {gereken:.2})")
        }
        Yukseltme::MarjYetersiz { marj, gereken } => {
            format!("marj yetersiz ({marj:.2} < {gereken:.2})")
        }
        Yukseltme::GuvenEsigi(neden) => format!("doktrin esigi ({neden:?})"),
    }
}

fn red_metni(neden: RedSebebi) -> &'static str {
    match neden {
        RedSebebi::SoruBos => "soru bos",
        RedSebebi::SecenekYok => "secenek yok",
        RedSebebi::KanitYok => "kanit yok",
        RedSebebi::EslesmeYok => "kanitla ortusen aday yok",
        RedSebebi::GuvenRed => "doktrin reddetti",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_kanaat::{Batarya, BataryaHatasi, Beklenen, Kanit, Vaka, BATARYA_SURUMU};

    fn dava(soru: &str, secenekler: &[&str], kanitlar: &[(&str, &str)]) -> Dava {
        Dava {
            soru: soru.to_string(),
            secenekler: secenekler.iter().map(|s| (*s).to_string()).collect(),
            kanitlar: kanitlar
                .iter()
                .map(|(kimlik, metin)| Kanit {
                    kimlik: (*kimlik).to_string(),
                    metin: (*metin).to_string(),
                    agirlik: None,
                })
                .collect(),
        }
    }

    #[test]
    fn the_doctrine_page_names_the_thresholds() {
        let metin = doktrin_md(&Ayarlar::default());
        assert!(metin.starts_with("# Kanaat motoru"));
        assert!(metin.contains("guven esigi"));
        assert!(metin.contains("marj esigi"));
    }

    #[test]
    fn a_verdict_renders_as_markdown_with_the_candidate_table() {
        let dava = dava(
            "kayit acildi mi",
            &["kayit acildi", "kayit kapandi"],
            &[("k1", "kayit acildi ve surdu"), ("k2", "kayit acildi")],
        );
        let hukum = karar_ver(&dava, &Ayarlar::default());
        let metin = hukum_md(&dava, &hukum);
        assert!(metin.contains("- karar: secim:0"), "{metin}");
        assert!(metin.contains("| # | aday |"));
        assert!(metin.contains("k1, k2"));
    }

    #[test]
    fn a_refusal_renders_its_reason() {
        let dava = dava("kayit acildi mi", &["kayit"], &[]);
        let metin = hukum_md(&dava, &karar_ver(&dava, &Ayarlar::default()));
        assert!(metin.contains("- karar: ret"));
        assert!(metin.contains("kanit yok"));
    }

    #[test]
    fn an_escalation_renders_its_reason() {
        // Iki aday ayni: soru tam kapsanir, destek yeter, tek eksik marjdir.
        let dava = dava(
            "kayit acildi mi",
            &["kayit acildi", "kayit acildi"],
            &[("k1", "kayit acildi")],
        );
        let metin = hukum_md(&dava, &karar_ver(&dava, &Ayarlar::default()));
        assert!(metin.contains("- karar: yukselt"), "{metin}");
        assert!(metin.contains("marj yetersiz"), "{metin}");
    }

    #[test]
    fn a_battery_of_our_own_runs_and_a_broken_one_is_refused() {
        let batarya = Batarya {
            surum: BATARYA_SURUMU,
            vakalar: vec![Vaka {
                ad: "acik-kapi".to_string(),
                dava: dava(
                    "kayit acildi mi",
                    &["kayit acildi", "kayit kapandi"],
                    &[("k1", "kayit acildi")],
                ),
                beklenen: Beklenen::Secim { indeks: 0 },
            }],
        };
        let rapor: BataryaRaporu = batarya_kos(&batarya, &Ayarlar::default());
        assert_eq!(rapor.toplam, 1);
        assert!(rapor.vakalar[0].dogru, "{}", vaka_satiri(&rapor.vakalar[0]));
        assert!(vaka_satiri(&rapor.vakalar[0]).contains("acik-kapi"));
        assert!(matches!(
            batarya_oku("{").unwrap_err(),
            BataryaHatasi::Json(_)
        ));
        let eski = batarya_oku(r#"{"surum":0,"vakalar":[]}"#).unwrap_err();
        assert!(matches!(eski, BataryaHatasi::Surum(0)), "{eski:?}");
    }

    #[test]
    fn the_compiled_in_battery_agrees_with_every_expectation() {
        let batarya = batarya_oku(GOMULU_BATARYA).expect("gomulu batarya okunmali");
        assert_eq!(batarya.surum, BATARYA_SURUMU);
        let rapor = batarya_kos(&batarya, &Ayarlar::default());
        assert_eq!(rapor.toplam, batarya.vakalar.len());
        let tutmayan: Vec<&VakaSonucu> = rapor.vakalar.iter().filter(|v| !v.dogru).collect();
        assert!(tutmayan.is_empty(), "tutmayan: {tutmayan:?}");
    }

    #[test]
    fn the_ledger_appends_verifies_and_notices_an_edit() {
        let yol = std::env::temp_dir().join("lubot-kanaat-defter-testi.jsonl");
        let _ = std::fs::remove_file(&yol);
        let vaka = std::env::temp_dir().join("lubot-kanaat-vaka-testi.json");
        let dava = dava(
            "kayit acildi mi",
            &["kayit acildi", "kayit kapandi"],
            &[("k1", "kayit acildi")],
        );
        std::fs::write(&vaka, serde_json::to_string(&dava).expect("vaka yazilmali"))
            .expect("vaka dosyasi");
        let yol_str = yol.to_string_lossy().to_string();
        let vaka_str = vaka.to_string_lossy().to_string();
        for _ in 0..2 {
            defter(&[
                "yaz".to_string(),
                "--dosya".to_string(),
                yol_str.clone(),
                "--vaka".to_string(),
                vaka_str.clone(),
            ])
            .expect("defter yazilmali");
        }
        let metin = std::fs::read_to_string(&yol).expect("defter okunmali");
        let okunan = Defter::oku(&metin).expect("zincir tutarli olmali");
        assert_eq!(okunan.uzunluk(), 2);
        assert_eq!(okunan.kayitlar()[1].onceki, okunan.kayitlar()[0].ozet);
        defter(&[
            "dogrula".to_string(),
            "--dosya".to_string(),
            yol_str.clone(),
        ])
        .expect("dogrulama gecmeli");
        // Kurcalama: bir kaydin hukum etiketi degistirilir.
        let bozuk = metin.replacen("secim:0", "ret", 1);
        std::fs::write(&yol, bozuk).expect("bozuk defter yazilmali");
        assert!(defter(&["dogrula".to_string(), "--dosya".to_string(), yol_str]).is_err());
        let _ = std::fs::remove_file(&yol);
        let _ = std::fs::remove_file(&vaka);
    }

    #[test]
    fn flags_are_read_with_their_values() {
        let args = vec![
            "--dosya".to_string(),
            "a.json".to_string(),
            "--kati".to_string(),
        ];
        assert_eq!(deger(&args, "--dosya"), Some("a.json".to_string()));
        assert_eq!(deger(&args, "--yok"), None);
    }
}
