//! # hiz - counting the work instead of timing it
//!
//! "Fast" is a property of a machine, a compiler and an afternoon. Counted
//! multiply-accumulates are a property of the model, and two people on two
//! machines get the same number. This module counts; nothing here reads a
//! clock, and nothing here should: a benchmark that is quoted without its
//! hardware is a number with no meaning attached, and a benchmark that is
//! re-run on a busy machine is a different number tomorrow.
//!
//! ## What is counted
//!
//! Per layer, per token, for a sequence of `n` tokens:
//!
//! - **The projections.** `q`, `k`, `v` and the output projection are dense
//!   multiplies, so their cost is linear in the sequence and quadratic in the
//!   width.
//! - **The scores and the mixing.** Both are quadratic in *how much of the
//!   sequence a token may see*, which is where the alternating schedule earns
//!   its keep: a local layer of radius `r` sees `2r + 1` positions, not `n`.
//!   [`Maliyet::dikkat_carpim`] therefore takes the schedule, not just the
//!   depth.
//! - **The feed-forward.** Two dense multiplies, gated, so `2 * d * d_ff`
//!   with the gate's halves counted once each.
//!
//! ## What is measured about the counting itself
//!
//! - `yerel_kat_genelden_ucuz` - a local layer costs strictly less than a
//!   global one at the same width, for every sequence longer than the window.
//!   If that were ever false the schedule would be a decoration.
//! - `uzun_dizide_dikkat_baskin` - past some length the attention term
//!   overtakes the projections; the crossover is reported rather than assumed,
//!   because "attention is the bottleneck" is only true above it.
//! - `sayim_toplamsal` - the whole stack's count equals the sum of its
//!   layers', so no term is counted twice and none is forgotten.

use crate::pencere::{Kapsam, PencereHatasi, Plan};

/// Multiply-accumulates, counted as `u128` because a real configuration
/// overflows `u64` at a few thousand tokens and a silently wrapped cost model
/// is worse than none.
pub type Carpim = u128;

/// The cost of one configuration on one sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maliyet {
    pub izdusum: Carpim,
    pub dikkat: Carpim,
    pub ileri_besleme: Carpim,
}

impl Maliyet {
    #[must_use]
    pub fn toplam(&self) -> Carpim {
        self.izdusum + self.dikkat + self.ileri_besleme
    }

    /// The attention share, between zero and one.
    #[must_use]
    pub fn dikkat_payi(&self) -> f64 {
        let toplam = self.toplam();
        if toplam == 0 {
            return 0.0;
        }
        (self.dikkat as f64) / (toplam as f64)
    }

    #[must_use]
    pub fn topla(&self, diger: &Self) -> Self {
        Self {
            izdusum: self.izdusum + diger.izdusum,
            dikkat: self.dikkat + diger.dikkat,
            ileri_besleme: self.ileri_besleme + diger.ileri_besleme,
        }
    }
}

/// How many keys a query at `konum` may read under `kapsam`.
///
/// Closed form, clipped at both ends. The obvious loop over the sequence is
/// kept as the reference the closed form is checked against
/// (`kapali_form_saymayla_uzlasir`): the clipping at the edges is exactly
/// where an arithmetic shortcut goes wrong by one.
#[must_use]
pub fn gorulen_anahtar(kapsam: Kapsam, konum: usize, dizi: usize) -> usize {
    match kapsam {
        Kapsam::Genel => dizi,
        Kapsam::Yerel { yaricap } => {
            if konum >= dizi {
                return 0;
            }
            let sol = konum.saturating_sub(yaricap);
            let sag = konum.saturating_add(yaricap).min(dizi - 1);
            sag - sol + 1
        }
    }
}

/// The same count, done the slow and obvious way.
#[must_use]
pub fn gorulen_anahtar_sayarak(kapsam: Kapsam, konum: usize, dizi: usize) -> usize {
    (0..dizi).filter(|k| kapsam.gorulebilir(konum, *k)).count()
}

/// The attention multiplies of one layer: scores and the mixing that follows.
///
/// Counted position by position rather than with a closed formula, because
/// the closed formula for a clipped window is the place an off-by-one hides.
#[must_use]
pub fn dikkat_carpim(kapsam: Kapsam, dizi: usize, d_head: usize, n_sorgu_kafa: usize) -> Carpim {
    let mut gorulen_toplam: u128 = 0;
    for konum in 0..dizi {
        gorulen_toplam += gorulen_anahtar(kapsam, konum, dizi) as u128;
    }
    // Scores: d_head per (query, visible key). Mixing: the same again.
    2 * gorulen_toplam * (d_head as u128) * (n_sorgu_kafa as u128)
}

/// The dense projections of one layer.
#[must_use]
pub fn izdusum_carpim(
    dizi: usize,
    d_model: usize,
    d_head: usize,
    n_sorgu_kafa: usize,
    n_kv_kafa: usize,
) -> Carpim {
    let n = dizi as u128;
    let d = d_model as u128;
    let q = (n_sorgu_kafa * d_head) as u128;
    let kv = (n_kv_kafa * d_head) as u128;
    // q, k, v and the output projection.
    n * d * (q + 2 * kv + q)
}

/// The gated feed-forward of one layer.
#[must_use]
pub fn ileri_besleme_carpim(dizi: usize, d_model: usize, d_ff: usize) -> Carpim {
    let n = dizi as u128;
    let d = d_model as u128;
    let f = d_ff as u128;
    // Up projection produces two halves, down projection consumes one.
    n * d * (2 * f) + n * f * d
}

/// The whole stack, layer by layer, under a schedule.
///
/// # Errors
///
/// Whatever [`Plan::kapsam`] refuses.
pub fn yigin_maliyeti(
    plan: &Plan,
    dizi: usize,
    d_model: usize,
    d_head: usize,
    n_sorgu_kafa: usize,
    n_kv_kafa: usize,
    d_ff: usize,
) -> Result<Maliyet, PencereHatasi> {
    let mut toplam = Maliyet {
        izdusum: 0,
        dikkat: 0,
        ileri_besleme: 0,
    };
    for katman in 0..plan.n_katman() {
        let kapsam = plan.kapsam(katman)?;
        toplam = toplam.topla(&Maliyet {
            izdusum: izdusum_carpim(dizi, d_model, d_head, n_sorgu_kafa, n_kv_kafa),
            dikkat: dikkat_carpim(kapsam, dizi, d_head, n_sorgu_kafa),
            ileri_besleme: ileri_besleme_carpim(dizi, d_model, d_ff),
        });
    }
    Ok(toplam)
}

/// The shortest sequence at which attention overtakes everything else.
///
/// Searched rather than solved: the window clipping makes the closed form a
/// piecewise thing, and a wrong closed form would answer confidently. The
/// search is a bisection, which assumes the crossover happens once - true
/// while the attention term grows at least as fast as the linear ones, and
/// the boundary is verified either side by `uzun_dizide_dikkat_baskin`.
/// Returns `None` when attention never overtakes up to `tavan`.
#[must_use]
pub fn dikkatin_bastigi_uzunluk(
    plan: &Plan,
    d_model: usize,
    d_head: usize,
    n_sorgu_kafa: usize,
    n_kv_kafa: usize,
    d_ff: usize,
    tavan: usize,
) -> Option<usize> {
    let baskin = |dizi: usize| -> bool {
        match yigin_maliyeti(plan, dizi, d_model, d_head, n_sorgu_kafa, n_kv_kafa, d_ff) {
            Ok(m) => m.dikkat > m.izdusum + m.ileri_besleme,
            Err(_) => false,
        }
    };
    if tavan == 0 || !baskin(tavan) {
        return None;
    }
    let (mut alt, mut ust) = (1usize, tavan);
    while alt < ust {
        let orta = alt + (ust - alt) / 2;
        if baskin(orta) {
            ust = orta;
        } else {
            alt = orta + 1;
        }
    }
    Some(alt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(n: usize, periyot: usize, yaricap: usize) -> Plan {
        match Plan::periyodik(n, periyot, yaricap) {
            Ok(p) => p,
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn kapali_form_saymayla_uzlasir() {
        // Two implementations, one number. The edges are where a closed form
        // slips by one, so every position of several sequences is compared.
        for dizi in [1usize, 2, 7, 33] {
            for yaricap in [0usize, 1, 3, 50] {
                for konum in 0..dizi {
                    let kapsam = Kapsam::Yerel { yaricap };
                    assert_eq!(
                        gorulen_anahtar(kapsam, konum, dizi),
                        gorulen_anahtar_sayarak(kapsam, konum, dizi),
                        "dizi={dizi} yaricap={yaricap} konum={konum}"
                    );
                }
                assert_eq!(
                    gorulen_anahtar(Kapsam::Genel, 0, dizi),
                    gorulen_anahtar_sayarak(Kapsam::Genel, 0, dizi)
                );
            }
        }
    }

    #[test]
    fn dizinin_disindaki_konum_hicbir_sey_gormez() {
        assert_eq!(gorulen_anahtar(Kapsam::Yerel { yaricap: 2 }, 9, 5), 0);
    }

    #[test]
    fn gorulen_anahtar_genelde_tum_dizi() {
        assert_eq!(gorulen_anahtar(Kapsam::Genel, 0, 10), 10);
        assert_eq!(gorulen_anahtar(Kapsam::Genel, 9, 10), 10);
    }

    #[test]
    fn gorulen_anahtar_yerelde_pencereyle_sinirli() {
        assert_eq!(gorulen_anahtar(Kapsam::Yerel { yaricap: 2 }, 5, 20), 5);
        // Clipped at the edges, which is exactly where a closed formula slips.
        assert_eq!(gorulen_anahtar(Kapsam::Yerel { yaricap: 2 }, 0, 20), 3);
        assert_eq!(gorulen_anahtar(Kapsam::Yerel { yaricap: 2 }, 19, 20), 3);
    }

    #[test]
    fn yerel_kat_genelden_ucuz() {
        // If this were ever false the alternating schedule would be
        // decoration rather than a saving.
        for dizi in [8usize, 64, 512] {
            let genel = dikkat_carpim(Kapsam::Genel, dizi, 32, 8);
            let yerel = dikkat_carpim(Kapsam::Yerel { yaricap: 3 }, dizi, 32, 8);
            assert!(yerel < genel, "dizi={dizi}: {yerel} >= {genel}");
        }
    }

    #[test]
    fn kisa_dizide_yerel_genele_esitlenir() {
        // A window wider than the sequence sees everything: the saving is a
        // property of long sequences, and the count says so.
        let dizi = 5;
        let genel = dikkat_carpim(Kapsam::Genel, dizi, 16, 4);
        let yerel = dikkat_carpim(Kapsam::Yerel { yaricap: 99 }, dizi, 16, 4);
        assert_eq!(genel, yerel);
    }

    #[test]
    fn dikkat_dizi_karesiyle_buyur() {
        let a = dikkat_carpim(Kapsam::Genel, 10, 8, 2);
        let b = dikkat_carpim(Kapsam::Genel, 20, 8, 2);
        assert_eq!(b, 4 * a);
    }

    #[test]
    fn izdusum_dizide_dogrusal() {
        let a = izdusum_carpim(10, 128, 32, 4, 2);
        let b = izdusum_carpim(20, 128, 32, 4, 2);
        assert_eq!(b, 2 * a);
    }

    #[test]
    fn gruplu_kv_izdusumu_ucuzlatir() {
        let tam = izdusum_carpim(16, 128, 32, 4, 4);
        let gruplu = izdusum_carpim(16, 128, 32, 4, 1);
        assert!(gruplu < tam, "{gruplu} >= {tam}");
    }

    #[test]
    fn ileri_besleme_genislikle_dogrusal() {
        let a = ileri_besleme_carpim(8, 64, 128);
        let b = ileri_besleme_carpim(8, 64, 256);
        assert_eq!(b, 2 * a);
    }

    #[test]
    fn sayim_toplamsal() {
        // The stack's count is the sum of its layers': nothing counted twice,
        // nothing forgotten.
        let p = plan(6, 3, 4);
        let tam = match yigin_maliyeti(&p, 32, 128, 32, 4, 2, 256) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        let mut elle = Maliyet {
            izdusum: 0,
            dikkat: 0,
            ileri_besleme: 0,
        };
        for katman in 0..p.n_katman() {
            let kapsam = match p.kapsam(katman) {
                Ok(k) => k,
                Err(e) => panic!("{e}"),
            };
            elle = elle.topla(&Maliyet {
                izdusum: izdusum_carpim(32, 128, 32, 4, 2),
                dikkat: dikkat_carpim(kapsam, 32, 32, 4),
                ileri_besleme: ileri_besleme_carpim(32, 128, 256),
            });
        }
        assert_eq!(tam, elle);
    }

    #[test]
    fn dikkat_payi_sifirla_bir_arasinda() {
        let p = plan(4, 2, 8);
        let m = match yigin_maliyeti(&p, 64, 128, 32, 4, 2, 256) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        let pay = m.dikkat_payi();
        assert!((0.0..=1.0).contains(&pay), "{pay}");
        assert!(m.toplam() == m.izdusum + m.dikkat + m.ileri_besleme);
    }

    #[test]
    fn bos_maliyetin_payi_sifir() {
        let bos = Maliyet {
            izdusum: 0,
            dikkat: 0,
            ileri_besleme: 0,
        };
        assert_eq!(bos.dikkat_payi(), 0.0);
        assert_eq!(bos.toplam(), 0);
    }

    #[test]
    fn seyrek_cizelge_yogun_cizelgeden_ucuz() {
        // One global layer in three against all-global: the same stack, the
        // same width, a strictly smaller count.
        let seyrek = plan(6, 3, 8);
        let yogun = plan(6, 2, 8);
        let a = match yigin_maliyeti(&seyrek, 256, 128, 32, 4, 2, 256) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        let b = match yigin_maliyeti(&yogun, 256, 128, 32, 4, 2, 256) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        assert!(a.dikkat < b.dikkat, "{} >= {}", a.dikkat, b.dikkat);
        assert_eq!(a.izdusum, b.izdusum);
        assert_eq!(a.ileri_besleme, b.ileri_besleme);
    }

    #[test]
    fn uzun_dizide_dikkat_baskin() {
        // "Attention is the bottleneck" is only true above a length; the
        // length is searched for and reported rather than assumed.
        let p = plan(6, 3, 16);
        let esik = dikkatin_bastigi_uzunluk(&p, 128, 32, 4, 2, 256, 20000);
        match esik {
            Some(n) => {
                let altinda = match yigin_maliyeti(&p, n - 1, 128, 32, 4, 2, 256) {
                    Ok(m) => m,
                    Err(e) => panic!("{e}"),
                };
                let ustunde = match yigin_maliyeti(&p, n, 128, 32, 4, 2, 256) {
                    Ok(m) => m,
                    Err(e) => panic!("{e}"),
                };
                assert!(altinda.dikkat <= altinda.izdusum + altinda.ileri_besleme);
                assert!(ustunde.dikkat > ustunde.izdusum + ustunde.ileri_besleme);
            }
            None => panic!("dikkat 20000 jetona kadar baskin olmadi"),
        }
    }

    #[test]
    fn kisa_dizide_dikkat_baskin_degil() {
        let p = plan(6, 3, 16);
        let m = match yigin_maliyeti(&p, 8, 128, 32, 4, 2, 256) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        assert!(m.dikkat < m.izdusum + m.ileri_besleme);
    }

    #[test]
    fn buyuk_yapilandirmada_sayim_sarmaz() {
        // A real configuration: the count must be large and *exact*. It is
        // compared with an independent f64 estimate, because a wrapped
        // integer would still look like a number.
        let p = plan(24, 3, 128);
        let dizi = 8192usize;
        let m = match yigin_maliyeti(&p, dizi, 1024, 64, 16, 4, 4096) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        assert!(m.toplam() > 1e12 as u128, "{}", m.toplam());
        let tahmin_izdusum =
            24.0 * (dizi as f64) * 1024.0 * (16.0 * 64.0 + 2.0 * 4.0 * 64.0 + 16.0 * 64.0);
        let olculen = m.izdusum as f64;
        assert!(
            (olculen - tahmin_izdusum).abs() / tahmin_izdusum < 1e-9,
            "{olculen} vs {tahmin_izdusum}"
        );
    }

    #[test]
    fn maliyet_toplama_bilesenleri_karistirmaz() {
        let a = Maliyet {
            izdusum: 1,
            dikkat: 2,
            ileri_besleme: 3,
        };
        let b = Maliyet {
            izdusum: 10,
            dikkat: 20,
            ileri_besleme: 30,
        };
        let c = a.topla(&b);
        assert_eq!(c.izdusum, 11);
        assert_eq!(c.dikkat, 22);
        assert_eq!(c.ileri_besleme, 33);
    }

    #[test]
    fn sifir_dizi_sifir_maliyet() {
        let p = plan(4, 2, 4);
        let m = match yigin_maliyeti(&p, 0, 128, 32, 4, 2, 256) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(m.toplam(), 0);
    }
}
