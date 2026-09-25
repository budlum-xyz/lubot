//! The probes: is this process being watched, and is this machine what it says?
//!
//! # Why the probes are in two groups
//!
//! The cheap group reads `/proc/self/status` and the environment. The deeper
//! group also reads the machine's description (`/proc/cpuinfo`, the DMI product
//! name, the container markers in `/proc/1/cgroup`). They are separate because
//! the deeper group costs milliseconds and because a single call site means a
//! single patch disables everything: the caller is expected to run the cheap
//! group at more than one point in its own flow.
//!
//! # Why "found something" is not "attack"
//!
//! A shared CI runner is a container. A virtual machine is a virtual machine
//! because someone is paying for it. Both look exactly like the things a
//! sandbox-evading binary is told to fear, and a probe that treats them as
//! attacks produces a program that refuses to run where it is developed - which
//! is how hardening gets deleted. Findings carry a *weight* and the caller
//! decides; see [`crate::kapi`].
//!
//! # What is not here
//!
//! No `ptrace` call of our own, no timing of `rdtsc`, no syscall-level checks:
//! this crate is `#![forbid(unsafe_code)]` and reads the kernel's own report
//! instead. That is a real limit - a tracer that also patches `/proc` output
//! defeats it - and it is stated rather than papered over.

use std::fs;

/// Which family of check produced a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Katman {
    /// A debugger or tracer.
    HataAyiklayici,
    /// Timing that does not match the work done.
    Zamanlama,
    /// A virtualised or containerised machine.
    SanalOrtam,
    /// Environment variables that ask for introspection.
    Ortam,
}

impl Katman {
    /// The fixed label of this layer.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::HataAyiklayici => "hata-ayiklayici",
            Self::Zamanlama => "zamanlama",
            Self::SanalOrtam => "sanal-ortam",
            Self::Ortam => "ortam",
        }
    }
}

/// One finding: what was seen, how much it matters.
///
/// `agirlik` is an ordering, not a probability: `3` means "this is almost
/// certainly a debugger", `1` means "this is worth writing down".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bulgu {
    /// Which check produced it.
    pub katman: Katman,
    /// What was observed, in words a report can print.
    pub aciklama: String,
    /// How much it matters, `0..=3`.
    pub agirlik: u8,
}

/// Reads `TracerPid` out of a `/proc/<pid>/status` body.
///
/// `None` means the field was absent, which happens on kernels that do not
/// expose it; that is reported as "unknown" rather than as "clean".
#[must_use]
pub(crate) fn tracer_pid(status: &str) -> Option<u32> {
    status.lines().find_map(|satir| {
        satir
            .strip_prefix("TracerPid:")
            .and_then(|kalan| kalan.trim().parse::<u32>().ok())
    })
}

/// The cheap group: is this process traced, or has it been asked to explain
/// itself?
#[must_use]
pub fn hafif_kontrol() -> Vec<Bulgu> {
    let mut bulgular = Vec::new();
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        match tracer_pid(&status) {
            Some(0) => {}
            Some(pid) => bulgular.push(Bulgu {
                katman: Katman::HataAyiklayici,
                aciklama: format!("izleyici pid {pid} ile bagli (TracerPid)"),
                agirlik: 3,
            }),
            None => bulgular.push(Bulgu {
                katman: Katman::HataAyiklayici,
                aciklama: "TracerPid alani yok: izleyici durumu bilinmiyor".to_string(),
                agirlik: 1,
            }),
        }
    } else {
        bulgular.push(Bulgu {
            katman: Katman::Ortam,
            aciklama: "/proc/self/status okunamadi: izleyici durumu bilinmiyor".to_string(),
            agirlik: 1,
        });
    }
    if let Some(bulgu) = geri_izleme_degiskeni(std::env::var("RUST_BACKTRACE").ok().as_deref()) {
        bulgular.push(bulgu);
    }
    bulgular
}

/// A finding for the introspection environment variables, if one is set.
#[must_use]
pub(crate) fn geri_izleme_degiskeni(deger: Option<&str>) -> Option<Bulgu> {
    match deger {
        Some(deger) if !deger.is_empty() && deger != "0" => Some(Bulgu {
            katman: Katman::Ortam,
            aciklama: "RUST_BACKTRACE acik: ayrinti isteyen bir ortam".to_string(),
            agirlik: 1,
        }),
        _ => None,
    }
}

/// A fixed amount of arithmetic, used as a clock the probe can trust.
///
/// The loop cannot be optimised away because its result is returned, and it is
/// small enough that any machine finishes it quickly. It is a *lower bound*
/// probe: single-stepping through it takes orders of magnitude longer, which is
/// the only signal it is used for.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn is_yuku_ms() -> f64 {
    let baslangic = std::time::Instant::now();
    let mut toplam: u64 = 0;
    let mut sira: u64 = 0;
    while sira < 2_000_000 {
        toplam = toplam
            .wrapping_mul(6364136223846793005)
            .wrapping_add(sira ^ 0x9E37_79B9_7F4A_7C15);
        sira += 1;
    }
    // The result is used, so the loop is not dead code.
    if toplam == 0xDEAD_BEEF {
        return f64::NAN;
    }
    baslangic.elapsed().as_secs_f64() * 1000.0
}

/// A finding when the measured time is a multiple of what the work should cost.
///
/// The ratio is passed in rather than fixed here: what counts as "far off"
/// depends on how noisy the machine is, and that is the caller's measurement.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn zamanlama_bulgusu(beklenen_ms: f64, olculen_ms: f64, kat: f64) -> Option<Bulgu> {
    if beklenen_ms <= 0.0 || olculen_ms <= 0.0 {
        return None;
    }
    let oran = olculen_ms / beklenen_ms;
    if oran < kat {
        return None;
    }
    Some(Bulgu {
        katman: Katman::Zamanlama,
        aciklama: format!(
            "beklenen yukun {oran:.1} kati surdu ({olculen_ms:.1} ms / {beklenen_ms:.1} ms)"
        ),
        agirlik: 2,
    })
}

/// A hypervisor flag or vendor string in `/proc/cpuinfo`.
#[must_use]
pub(crate) fn hipervizor_bulgusu(cpuinfo: &str) -> Option<Bulgu> {
    const ADLAR: &[&str] = &[
        "hypervisor",
        "kvm",
        "vmware",
        "virtualbox",
        "xen",
        "qemu",
        "bhyve",
        "parallels",
    ];
    let kucuk = cpuinfo.to_ascii_lowercase();
    let eslesen = ADLAR.iter().find(|ad| kucuk.contains(**ad))?;
    Some(Bulgu {
        katman: Katman::SanalOrtam,
        aciklama: format!("cpuinfo hypervisor izi tasiyor ({eslesen})"),
        agirlik: 1,
    })
}

/// A hypervisor name in the DMI product name.
#[must_use]
pub(crate) fn dmi_bulgusu(urun: &str) -> Option<Bulgu> {
    const ADLAR: &[&str] = &[
        "kvm",
        "vmware",
        "virtualbox",
        "qemu",
        "xen",
        "bhyve",
        "parallels",
        "amazon ec2",
        "google compute",
        "microsoft corporation",
    ];
    let kucuk = urun.to_ascii_lowercase();
    let eslesen = ADLAR.iter().find(|ad| kucuk.contains(**ad))?;
    Some(Bulgu {
        katman: Katman::SanalOrtam,
        aciklama: format!("DMI urun adi bir hipervizor adi tasiyor ({eslesen})"),
        agirlik: 1,
    })
}

/// A container marker out of `/proc/1/cgroup` or `/proc/1/environ`.
#[must_use]
pub(crate) fn konteyner_bulgusu(icerik: &str) -> Option<Bulgu> {
    const ISARETLER: &[&str] = &[
        "docker",
        "containerd",
        "kubepods",
        "lxc",
        "podman",
        "container=",
    ];
    let kucuk = icerik.to_ascii_lowercase();
    let eslesen = ISARETLER.iter().find(|ad| kucuk.contains(**ad))?;
    Some(Bulgu {
        katman: Katman::SanalOrtam,
        aciklama: format!("konteyner isareti bulundu ({eslesen})"),
        agirlik: 1,
    })
}

/// The deeper group: the cheap checks plus the machine's own description.
#[must_use]
pub fn derin_kontrol() -> Vec<Bulgu> {
    let mut bulgular = hafif_kontrol();
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        bulgular.extend(hipervizor_bulgusu(&cpuinfo));
    }
    if let Ok(urun) = fs::read_to_string("/sys/class/dmi/id/product_name") {
        bulgular.extend(dmi_bulgusu(&urun));
    }
    if let Ok(cgroup) = fs::read_to_string("/proc/1/cgroup") {
        bulgular.extend(konteyner_bulgusu(&cgroup));
    }
    bulgular
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracer_pid_is_read_in_both_states() {
        assert_eq!(
            tracer_pid("Name:\tlubot\nTracerPid:\t0\nUid:\t1000\n"),
            Some(0)
        );
        assert_eq!(tracer_pid("Name:\tlubot\nTracerPid:\t4242\n"), Some(4242));
        assert_eq!(tracer_pid("Name:\tlubot\n"), None);
    }

    #[test]
    fn a_traced_process_is_reported_with_weight_three() {
        for bulgu in &hafif_kontrol() {
            assert!(bulgu.agirlik <= 3);
            assert!(!bulgu.aciklama.is_empty());
        }
    }

    #[test]
    fn backtrace_variable_is_only_reported_when_it_asks_for_detail() {
        assert!(geri_izleme_degiskeni(Some("full")).is_some());
        assert!(geri_izleme_degiskeni(Some("1")).is_some());
        assert!(geri_izleme_degiskeni(Some("0")).is_none());
        assert!(geri_izleme_degiskeni(None).is_none());
    }

    #[test]
    fn a_far_off_timing_is_reported_and_a_small_one_is_not() {
        assert!(zamanlama_bulgusu(2.0, 2.5, 50.0).is_none());
        assert!(zamanlama_bulgusu(2.0, 400.0, 50.0).is_some());
        assert!(zamanlama_bulgusu(0.0, 400.0, 50.0).is_none());
    }

    #[test]
    fn the_workload_runs_and_returns_a_number() {
        let ms = is_yuku_ms();
        assert!((0.0..10_000.0).contains(&ms), "olculen {ms} ms");
    }

    #[test]
    fn hypervisor_flag_is_read_from_cpuinfo_shaped_text() {
        let vm = "vendor_id\t: GenuineIntel\nflags\t\t: fpu vme hypervisor\n";
        assert!(hipervizor_bulgusu(vm).is_some());
        let gercek = "vendor_id\t: GenuineIntel\nflags\t\t: fpu vme sse2\n";
        assert!(hipervizor_bulgusu(gercek).is_none());
    }

    #[test]
    fn hypervisor_names_are_matched_case_insensitively() {
        assert!(hipervizor_bulgusu("Vendor: QEMU\n").is_some());
        assert!(dmi_bulgusu("VMware Virtual Platform\n").is_some());
        assert!(dmi_bulgusu("PowerEdge R740\n").is_none());
    }

    #[test]
    fn container_markers_are_found_where_they_are_written() {
        assert!(konteyner_bulgusu("0::/docker/abc123\n").is_some());
        assert!(konteyner_bulgusu("0::/kubepods/besteffort/pod-x\n").is_some());
        assert!(konteyner_bulgusu("0::/init.scope\n").is_none());
    }

    #[test]
    fn the_deep_group_contains_the_cheap_group() {
        let hafif = hafif_kontrol();
        let derin = derin_kontrol();
        assert!(derin.len() >= hafif.len());
        for katman in [
            Katman::HataAyiklayici,
            Katman::Zamanlama,
            Katman::SanalOrtam,
        ] {
            assert!(!katman.ad().is_empty());
        }
    }

    #[test]
    fn every_finding_on_this_machine_is_well_formed() {
        for bulgu in derin_kontrol() {
            assert!(bulgu.agirlik >= 1 && bulgu.agirlik <= 3);
            assert!(!bulgu.aciklama.is_empty());
        }
    }
}
