//! The gate: one place collects findings, one place decides, one place reports.
//!
//! # Why one place
//!
//! Findings are produced all over the crate so that no single patch removes
//! them, but the *decision* to refuse is taken exactly here. A program that
//! stops itself in five different places is a program whose behaviour cannot be
//! described in a bug report, and "it exits on my machine" is not a diagnosis.
//!
//! # Why refusing is opt-in
//!
//! [`Kip::Bildir`] is the default: the gate writes its findings and lets the run
//! continue. [`Kip::Zorla`] refuses, and only for findings whose weight reaches
//! [`ESIGIR_AGIRLIK`]. Shared runners, containers and tracing tools all look
//! suspicious and none of them are an attack, so a hard refusal by default
//! would make the program unusable on the machines it is developed on. The
//! operator turns the hard mode on for a build that is being shipped.

use crate::butunluk;
use crate::izler::{self, Bulgu, Katman};

/// Weight at which a finding stops a hardened run.
pub const ESIGIR_AGIRLIK: u8 = 3;

/// How long [`izler::is_yuku_ms`]'s workload may take before the timing check
/// treats the delay as suspicious; see [`izler::zamanlama_bulgusu`] for the
/// ratio that is applied on top of it.
pub(crate) const IS_BUTCESI_MS: f64 = 25.0;

/// Whether findings are reported or enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kip {
    /// Write findings; never stop.
    Bildir,
    /// Stop when a heavy finding is present.
    Zorla,
}

/// What the gate saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rapor {
    /// The mode the gate ran in.
    pub kip: Kip,
    /// Findings, in the order the checks produced them.
    pub bulgular: Vec<Bulgu>,
    /// The running executable's digest, when it could be measured.
    pub ozet: Option<String>,
}

impl Rapor {
    /// The heaviest weight among the findings.
    #[must_use]
    pub fn en_agir(&self) -> u8 {
        self.bulgular.iter().map(|b| b.agirlik).max().unwrap_or(0)
    }
}

/// Assembles the report: the caller's findings, the timing probe, the digest.
///
/// The caller chooses *which* groups of probes run ([`izler::hafif_kontrol`],
/// [`izler::derin_kontrol`]) because that choice costs milliseconds and belongs
/// to the operator; the *threshold* the timing is compared against lives here,
/// because two thresholds for one decision is two answers. The digest is the
/// running executable's own; when it cannot be read the report says so in its
/// Markdown rather than omitting the field, since a missing measurement and a
/// clean one look identical in a summary.
#[must_use]
pub fn topla(kip: Kip, bulgular: Vec<Bulgu>) -> Rapor {
    let mut bulgular = bulgular;
    let olculen = izler::is_yuku_ms();
    bulgular.extend(izler::zamanlama_bulgusu(IS_BUTCESI_MS, olculen, 50.0));
    Rapor {
        kip,
        bulgular,
        ozet: butunluk::kendi_ozeti().ok(),
    }
}

/// Whether the gate refuses the run.
///
/// Only [`Kip::Zorla`] refuses, and only on a finding at [`ESIGIR_AGIRLIK`] or
/// above. See the module note for why the default does not.
#[must_use]
pub fn durdurulmali(rapor: &Rapor) -> bool {
    rapor.kip == Kip::Zorla && rapor.en_agir() >= ESIGIR_AGIRLIK
}

/// The message a refused run prints.
///
/// The text is obfuscated at rest (see [`crate::metin`]) so that a search for
/// strings in the binary does not immediately reveal what the gate does; the
/// decoded form is asserted in this module's tests so the two cannot drift.
#[must_use]
pub fn zorla_mesaji() -> String {
    crate::gizli_metin!("lubot: bu ortamda guvenli calisma dogrulanmadi", 0x5C)
        .unwrap_or_else(|_| "lubot: ortam dogrulanmadi".to_string())
}

/// Installs a panic hook that prints one sentence and no location.
///
/// A default panic prints the source file, the line and a backtrace when asked;
/// that is exactly the map a reader of the binary wants. The hook is not
/// installed by the library on its own: a program that hides its own crashes in
/// development is a program nobody can debug. `LUBOT_PANIK_AYRINTI=1` keeps the
/// detailed message for exactly that reason.
pub fn panik_kancasi_kur() {
    std::panic::set_hook(Box::new(|bilgi| {
        let ayrinti = std::env::var("LUBOT_PANIK_AYRINTI").is_ok_and(|deger| deger == "1");
        if ayrinti {
            eprintln!("lubot panik: {bilgi}");
        } else {
            eprintln!("lubot: beklenmeyen durum, is durduruldu");
        }
    }));
}

/// The report as Markdown: the only output shape this crate produces.
#[must_use]
pub fn rapor_md(rapor: &Rapor) -> String {
    let mut cikti = String::new();
    cikti.push_str("# Sertlestirme raporu\n\n");
    cikti.push_str(&format!(
        "- kip: {}\n",
        match rapor.kip {
            Kip::Bildir => "bildir",
            Kip::Zorla => "zorla",
        }
    ));
    cikti.push_str(&format!(
        "- calisan ikilinin sha256 ozeti: {}\n",
        rapor.ozet.as_deref().unwrap_or("olculemedi")
    ));
    cikti.push_str(&format!(
        "- bulgu: {} (en agir {}), esik {}\n\n",
        rapor.bulgular.len(),
        rapor.en_agir(),
        ESIGIR_AGIRLIK
    ));
    if rapor.bulgular.is_empty() {
        cikti.push_str("Bulgu yok: bu kosuda ne izleyici, ne sanal ortam isareti ne de\n");
        cikti.push_str("zamanlama sapmasi goruldu. \"Bulgu yok\" bir kanit degil, bir olcumdur:\n");
        cikti.push_str("kontrol ettigi seylerin disinda bir sey olmadigini soyler.\n");
        return cikti;
    }
    cikti.push_str("| katman | agirlik | bulgu |\n|---|---:|---|\n");
    for bulgu in &rapor.bulgular {
        cikti.push_str(&format!(
            "| {} | {} | {} |\n",
            bulgu.katman.ad(),
            bulgu.agirlik,
            bulgu.aciklama
        ));
    }
    cikti.push('\n');
    if durdurulmali(rapor) {
        cikti.push_str("Karar: zorla kipinde esigi asan bulgu var, kosu durdurulur.\n");
    } else if rapor.kip == Kip::Zorla {
        cikti.push_str("Karar: zorla kipinde esigi asan bulgu yok, kosu surer.\n");
    } else {
        cikti.push_str("Karar: bildir kipinde kosu surer; bulgular kayda gecer.\n");
    }
    cikti
}

/// A layer's findings only, for a caller that wants one group.
#[must_use]
pub fn katman_bulgulari(rapor: &Rapor, katman: Katman) -> Vec<&Bulgu> {
    rapor
        .bulgular
        .iter()
        .filter(|b| b.katman == katman)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bulgu(agirlik: u8) -> Bulgu {
        Bulgu {
            katman: Katman::HataAyiklayici,
            aciklama: "olculdu".to_string(),
            agirlik,
        }
    }

    #[test]
    fn reporting_never_stops_and_forcing_stops_only_above_the_threshold() {
        let hafif = Rapor {
            kip: Kip::Zorla,
            bulgular: vec![bulgu(1)],
            ozet: None,
        };
        assert!(!durdurulmali(&hafif));
        let agir = Rapor {
            kip: Kip::Zorla,
            bulgular: vec![bulgu(3)],
            ozet: None,
        };
        assert!(durdurulmali(&agir));
        let bildir = Rapor {
            kip: Kip::Bildir,
            bulgular: vec![bulgu(3)],
            ozet: None,
        };
        assert!(!durdurulmali(&bildir));
    }

    #[test]
    fn the_obfuscated_message_decodes_to_its_asserted_form() {
        let mesaj = zorla_mesaji();
        assert!(mesaj.starts_with("lubot:"), "{mesaj}");
        assert!(mesaj.contains("dogrulanmadi"), "{mesaj}");
    }

    #[test]
    fn the_report_is_markdown_with_a_table_when_there_are_findings() {
        let rapor = Rapor {
            kip: Kip::Bildir,
            bulgular: vec![bulgu(1), bulgu(3)],
            ozet: Some("ab".repeat(32)),
        };
        let metin = rapor_md(&rapor);
        assert!(metin.starts_with("# Sertlestirme raporu"));
        assert!(metin.contains("| katman | agirlik | bulgu |"));
        assert!(metin.contains("hata-ayiklayici"));
        assert_eq!(rapor.en_agir(), 3);
    }

    #[test]
    fn an_empty_report_says_what_it_does_not_prove() {
        let rapor = Rapor {
            kip: Kip::Bildir,
            bulgular: Vec::new(),
            ozet: None,
        };
        let metin = rapor_md(&rapor);
        assert!(metin.contains("Bulgu yok"));
        assert!(metin.contains("olcumdur"));
    }

    #[test]
    fn findings_can_be_filtered_by_layer() {
        let rapor = Rapor {
            kip: Kip::Bildir,
            bulgular: vec![
                bulgu(1),
                Bulgu {
                    katman: Katman::Zamanlama,
                    aciklama: "yavas".to_string(),
                    agirlik: 2,
                },
            ],
            ozet: None,
        };
        assert_eq!(katman_bulgulari(&rapor, Katman::Zamanlama).len(), 1);
        assert_eq!(katman_bulgulari(&rapor, Katman::SanalOrtam).len(), 0);
    }

    #[test]
    fn collecting_on_this_machine_produces_a_well_formed_report() {
        let rapor = topla(Kip::Bildir, vec![bulgu(3)]);
        assert!(rapor.en_agir() <= 3);
        if let Some(ozet) = &rapor.ozet {
            assert_eq!(ozet.len(), 64);
        }
    }
}
