//! # gecis - the transition line, as a command
//!
//! Three verbs, one per crate verb, and each prints the JSON it measured so
//! the pipeline can be piped: `sayim` walks a source tree, `plan` turns a
//! census into a realization plan, `durum` measures how much of that plan a
//! realized crate covers. The plumbing lines the plan carries are the
//! repository's own lines; the gate compares them byte for byte.
//!
//! What the command deliberately does not accept is a name for the source
//! repository: paths are read relative to the tree that is given, and which
//! repository a component came from is provenance for the intake line, not
//! architecture.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use lubot_gecis::{durum, plan, say, GecisHatasi, Plan, Sayim};

fn hata_mesaji(e: &GecisHatasi) -> String {
    format!("gecis: {e}")
}

fn deger_bul(args: &[String], ad: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == ad {
            return iter.next().cloned();
        }
    }
    None
}

fn deger_zorunlu(args: &[String], ad: &str, kimlik: &str) -> Result<String, String> {
    deger_bul(args, ad).ok_or_else(|| format!("gecis {kimlik}: {ad} <deger> zorunlu"))
}

fn cikti<D: serde::Serialize>(deger: &D, yaz: Option<&str>) -> Result<(), String> {
    let metin =
        serde_json::to_string_pretty(deger).map_err(|e| format!("gecis: cikti yazilamadi: {e}"))?;
    if let Some(yol) = yaz {
        fs::write(yol, &metin).map_err(|e| format!("gecis: yazilamadi: {yol}: {e}"))?;
    }
    println!("{metin}");
    Ok(())
}

/// `lubot gecis sayim|plan|durum` komutunun kendisi.
pub fn cmd_gecis(args: &[String]) -> Result<(), String> {
    let alt = args.first().cloned().unwrap_or_default();
    let kalan: Vec<String> = args.iter().skip(1).cloned().collect();
    match alt.as_str() {
        "sayim" => cmd_sayim(&kalan),
        "plan" => cmd_plan(&kalan),
        "durum" => cmd_durum(&kalan),
        _ => Err("gecis: alt komut yok: sayim | plan | durum".to_string()),
    }
}

fn cmd_sayim(args: &[String]) -> Result<(), String> {
    let kaynak = deger_zorunlu(args, "--kaynak", "sayim")?;
    let lisans = deger_zorunlu(args, "--lisans", "sayim")?;
    let s = say(Path::new(&kaynak), &lisans).map_err(|e| hata_mesaji(&e))?;
    cikti(&s, deger_bul(args, "--yaz").as_deref())
}

fn cmd_plan(args: &[String]) -> Result<(), String> {
    let sayim_yolu = deger_zorunlu(args, "--sayim", "plan")?;
    let ad = deger_zorunlu(args, "--ad", "plan")?;
    let metin = fs::read_to_string(&sayim_yolu)
        .map_err(|e| format!("gecis plan: sayim okunamadi: {sayim_yolu}: {e}"))?;
    let s: Sayim =
        serde_json::from_str(&metin).map_err(|e| format!("gecis plan: sayim bozuk JSON: {e}"))?;
    let p = plan(&s, &ad).map_err(|e| hata_mesaji(&e))?;
    cikti(&p, deger_bul(args, "--yaz").as_deref())
}

fn cmd_durum(args: &[String]) -> Result<(), String> {
    let plan_yolu = deger_zorunlu(args, "--plan", "durum")?;
    let metin = fs::read_to_string(&plan_yolu)
        .map_err(|e| format!("gecis durum: plan okunamadi: {plan_yolu}: {e}"))?;
    let p: Plan =
        serde_json::from_str(&metin).map_err(|e| format!("gecis durum: plan bozuk JSON: {e}"))?;
    let kok =
        deger_bul(args, "--gerceklesen").unwrap_or_else(|| format!("crates/{}", p.hedef_crate));
    let esleme = match deger_bul(args, "--esleme") {
        None => None,
        Some(yol) => {
            let e_metin = fs::read_to_string(&yol)
                .map_err(|e| format!("gecis durum: esleme okunamadi: {yol}: {e}"))?;
            Some(
                serde_json::from_str::<BTreeMap<String, String>>(&e_metin)
                    .map_err(|e| format!("gecis durum: esleme bozuk JSON: {e}"))?,
            )
        }
    };
    let d = durum(&p, Path::new(&kok), esleme.as_ref()).map_err(|e| hata_mesaji(&e))?;
    cikti(&d, deger_bul(args, "--yaz").as_deref())
}
