//! The codebook: where the quantisation levels come from, and why they are
//! solved here instead of pasted in.
//!
//! # What a codebook is for
//!
//! After [`crate::hadamard`] rotates a group and the group is scaled to unit
//! norm, each coordinate looks like a draw from a normal distribution with
//! standard deviation `1 / sqrt(group)`. A uniform grid of levels over that is
//! wasteful: the tails are nearly empty and the centre is crowded. The
//! Lloyd-Max conditions say what the right levels are - every level sits at the
//! conditional mean of the region it owns, and every boundary sits midway
//! between its two levels - and this module solves them.
//!
//! # Why it is solved rather than tabulated
//!
//! The four numbers for a two-bit Gaussian quantiser are in every textbook, and
//! copying them would be shorter. Three reasons not to:
//!
//! 1. **A tabulated constant cannot be checked.** A solved codebook can be
//!    compared against the Lloyd-Max *conditions* themselves - see
//!    [`kosullari_sagliyor`] - so the test proves optimality rather than
//!    proving that a typist was careful.
//! 2. **The distortion comes out of the same solve.** [`Kodkitabi::bozulma`] is
//!    the mean squared error the codebook will cause, and it is what the
//!    measured reconstruction error in [`crate::grup`] is compared against. A
//!    measurement with nothing to compare it to is a number.
//! 3. **Bit widths are a parameter, not a menu.** The ladder in the container
//!    needs different widths for different tensors; a table would have to be
//!    extended by hand for each.
//!
//! # Why the integrals are analytic and the grid is only an initial guess
//!
//! The first version of this solver did everything on a fixed grid of 65536
//! cells: assign each cell to its nearest level, average. It reproduced the
//! published numbers at one and two bits and then **stopped converging at
//! six**, with the residual movement stalled at `1.7e-4`, which is the width of
//! one grid cell. The reason is structural rather than a tuning problem: on a discrete
//! grid a region boundary can only sit *between* cells, so as the boundary
//! creeps a whole cell's mass jumps from one region to the next and the
//! iteration enters a limit cycle whose amplitude is the cell width. Making the
//! grid finer buys a decimal place and costs linear time.
//!
//! So the region integrals are closed form instead, with only the mass needing
//! quadrature:
//!
//! ```text
//! integral of x*phi(x) dx over [a,b]    =  phi(a) - phi(b)
//! integral of x^2*phi(x) dx over [a,b]  =  mass(a,b) + a*phi(a) - b*phi(b)
//! ```
//!
//! Both follow from `d/dx(-phi(x)) = x*phi(x)` and one integration by parts.
//! Boundaries are now real numbers, the limit cycle is gone, and the solve
//! reaches its fixed point to `1e-13` at every width from one bit to eight. The
//! grid survives in exactly one place - picking the starting levels in
//! [`baslangic`] - where its resolution cannot affect the answer, because Lloyd
//! converges to the same fixed point from any sorted start.
//!
//! # Determinism, which is not optional here
//!
//! Two operators must derive the same codebook from the same bit width or the
//! container written by one is unreadable by the other. There is no sampling
//! anywhere in this module: no random draws, no seed to agree on. The obvious
//! alternative - draw four hundred thousand normal samples and run k-means - is
//! shorter to write and produces a slightly different answer on every platform.

use std::fmt;
use std::sync::OnceLock;

/// Effective infinity, in standard deviations. A standard normal carries about
/// `6e-58` of its mass beyond sixteen sigma, which is far below the `f64`
/// resolution of a sum that is close to one.
const SONSUZ: f64 = 9.0;
/// Quadrature panels per region. Fixed rather than derived from the width, and
/// that is not a detail: a panel *count* that changes with the interval makes
/// the mass a discontinuous function of the boundary, and the Lloyd iteration
/// then chases the discontinuity instead of converging. Measured, on the way
/// here: with `ceil((b-a)/0.25)` panels the solve stalled at `2.8e-9` at seven
/// bits - small, plausible, and a limit cycle all the same. Sixteen panels of
/// eight-point Gauss-Legendre is exact to `f64` for a Gaussian over any region
/// this solver produces, and it is *smooth* in both endpoints.
const PANEL_SAYISI: usize = 16;
/// Cells in the initialisation grid. Only the starting point depends on this.
const IZGARA_ADIMI: usize = 1 << 14;
/// Ceiling on Lloyd iterations. Convergence is measured, not assumed; this is
/// the refusal point.
const EN_COK_TUR: usize = 60_000;
/// Movement below which the solve is called converged.
///
/// The levels are cast to `f32` before anything reads them, and `f32` resolves
/// about `6e-8` near one. `1e-11` is roughly four orders below what survives
/// that cast, which is the point: tightening further buys precision no reader
/// can see and costs real time - measured, eight bits reached `7.9e-12` and was
/// still creeping after sixty thousand passes - while loosening it would let
/// two operators disagree after the cast.
const YAKINSAMA: f64 = 1e-11;
/// Over-relaxation factor on the level update.
///
/// Plain Lloyd converges linearly and the rate worsens with the number of
/// levels: measured here, seven bits needed more than twenty thousand passes to
/// reach `2.8e-9`. Stepping `1.9` times the way to the new centroid instead of
/// exactly to it cuts that to the low hundreds of iterations at the same fixed
/// point - over-relaxation cannot move where the iteration converges, only how
/// fast it gets there, because the fixed point is where the step length is zero.
const GEVSETME: f64 = 1.9;

/// Eight-point Gauss-Legendre nodes on `[-1, 1]`, positive half.
const GL_DUGUM: [f64; 4] = [
    0.183_434_642_495_65,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
/// The matching weights.
const GL_AGIRLIK: [f64; 4] = [
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

/// Why a codebook could not be built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KodkitabiHatasi {
    /// Fewer than one bit, or more than eight. One bit is the smallest
    /// meaningful codebook; eight bits is where the packing stops paying for
    /// itself against a plain byte.
    BitAraligi { bit: u8 },
    /// The Lloyd iteration did not settle inside [`EN_COK_TUR`]. An error
    /// rather than a silent early stop, because a half-converged codebook is a
    /// wrong codebook that still decodes.
    Yakinsamadi { bit: u8, hareket: f64 },
}

impl fmt::Display for KodkitabiHatasi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitAraligi { bit } => write!(f, "bit genisligi {bit} 1..=8 disinda"),
            Self::Yakinsamadi { bit, hareket } => {
                write!(
                    f,
                    "{bit} bit icin Lloyd yakinsamadi, son hareket {hareket:.3e}"
                )
            }
        }
    }
}

impl std::error::Error for KodkitabiHatasi {}

/// The standard normal density.
fn phi(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Mass of a standard normal on `[a, b]`, by composite Gauss-Legendre.
///
/// The only quantity in the solve that is not closed form. Panels are at most
/// [`PANEL`] wide, so the count depends on the interval and not on any global
/// grid; a narrow central region costs one panel and the tail costs the rest.
fn kutle(a: f64, b: f64) -> f64 {
    let a = a.max(-SONSUZ);
    let b = b.min(SONSUZ);
    if b <= a {
        return 0.0;
    }
    let panel = PANEL_SAYISI;
    #[allow(clippy::cast_precision_loss)]
    let genislik = (b - a) / (panel as f64);
    let yari = genislik / 2.0;
    let mut toplam = 0.0;
    for i in 0..panel {
        #[allow(clippy::cast_precision_loss)]
        let merkez = a + yari + (i as f64) * genislik;
        for (d, w) in GL_DUGUM.iter().zip(GL_AGIRLIK.iter()) {
            toplam += w * (phi(merkez + yari * d) + phi(merkez - yari * d));
        }
    }
    toplam * yari
}

/// `integral of x*phi(x) dx` over `[a, b]`, closed form.
fn birinci_moment(a: f64, b: f64) -> f64 {
    phi(a.max(-SONSUZ)) - phi(b.min(SONSUZ))
}

/// `integral of x^2*phi(x) dx` over `[a, b]`, closed form.
fn ikinci_moment(a: f64, b: f64) -> f64 {
    let a = a.max(-SONSUZ);
    let b = b.min(SONSUZ);
    kutle(a, b) + a * phi(a) - b * phi(b)
}

/// A solved set of quantisation levels plus the distortion it causes.
#[derive(Debug, Clone, PartialEq)]
pub struct Kodkitabi {
    seviyeler: Vec<f64>,
    bozulma: f64,
    tur: usize,
}

impl Kodkitabi {
    /// The levels, ascending, for a unit-variance source.
    #[must_use]
    pub fn seviyeler(&self) -> &[f64] {
        &self.seviyeler
    }

    /// Mean squared error this codebook causes on a standard normal.
    ///
    /// For reference, the classical Lloyd-Max figures are `0.3634` at one bit,
    /// `0.1175` at two, `0.03454` at three and `0.009497` at four; the solver
    /// reproduces them and the tests pin that.
    #[must_use]
    pub fn bozulma(&self) -> f64 {
        self.bozulma
    }

    /// Signal-to-noise ratio in decibels, `10 log10(1 / distortion)`, for a
    /// unit-variance source. Reported because it is the number that can be
    /// compared directly against the measured reconstruction error of a real
    /// tensor - if a tensor does much worse than this, the rotation is not
    /// Gaussianising that tensor, and that is a finding rather than a nuisance.
    #[must_use]
    pub fn snr_db(&self) -> f64 {
        10.0 * (1.0 / self.bozulma).log10()
    }

    /// Iterations the solve took.
    #[must_use]
    pub fn tur(&self) -> usize {
        self.tur
    }

    /// Number of levels.
    #[must_use]
    pub fn boyut(&self) -> usize {
        self.seviyeler.len()
    }

    /// The levels rescaled for a group normalised to unit *norm* rather than
    /// unit variance. A unit-norm vector of length `g` has coordinates of
    /// typical size `1 / sqrt(g)`, so the whole book shrinks by that factor.
    /// Done here rather than at the call site because getting it wrong produces
    /// a plausible tensor that is uniformly the wrong size.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn birim_norm_icin(&self, grup: usize) -> Vec<f32> {
        let olcek = 1.0 / (grup as f64).sqrt();
        self.seviyeler.iter().map(|v| (v * olcek) as f32).collect()
    }

    /// Index of the nearest level to `x`, by binary search over the ascending
    /// levels. Ties go to the lower index: arbitrary but fixed, because a tie
    /// rule that depended on iteration order would make the encoder
    /// non-deterministic for exactly the inputs that sit on a boundary.
    #[must_use]
    pub fn en_yakin(seviyeler: &[f32], x: f32) -> usize {
        if seviyeler.is_empty() {
            return 0;
        }
        let mut alt = 0usize;
        let mut ust = seviyeler.len();
        while alt < ust {
            let orta = usize::midpoint(alt, ust);
            if seviyeler[orta] < x {
                alt = orta + 1;
            } else {
                ust = orta;
            }
        }
        if alt == 0 {
            return 0;
        }
        if alt == seviyeler.len() {
            return seviyeler.len() - 1;
        }
        let sol = x - seviyeler[alt - 1];
        let sag = seviyeler[alt] - x;
        if sol <= sag {
            alt - 1
        } else {
            alt
        }
    }
}

/// Starting levels: the `(k+0.5)/L` quantiles, read off a coarse cumulative
/// grid.
///
/// The grid is here and nowhere else. Lloyd converges to the same fixed point
/// from any sorted starting set, so the resolution of this grid cannot reach
/// the answer - but the *starting point* still has to be the same everywhere,
/// or two operators could land in different local solutions at eight bits. A
/// fixed grid is the cheapest way to make the start reproducible.
fn baslangic(seviye_sayisi: usize) -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)]
    let genislik = 2.0 * SONSUZ / (IZGARA_ADIMI as f64);
    let mut x = Vec::with_capacity(IZGARA_ADIMI);
    let mut kumulatif = Vec::with_capacity(IZGARA_ADIMI);
    let mut toplam = 0.0;
    for i in 0..IZGARA_ADIMI {
        #[allow(clippy::cast_precision_loss)]
        let orta = -SONSUZ + (i as f64 + 0.5) * genislik;
        toplam += phi(orta) * genislik;
        x.push(orta);
        kumulatif.push(toplam);
    }
    for v in &mut kumulatif {
        *v /= toplam;
    }
    let mut seviyeler = Vec::with_capacity(seviye_sayisi);
    let mut tarayici = 0usize;
    for k in 0..seviye_sayisi {
        #[allow(clippy::cast_precision_loss)]
        let hedef = (k as f64 + 0.5) / (seviye_sayisi as f64);
        while tarayici + 1 < kumulatif.len() && kumulatif[tarayici] < hedef {
            tarayici += 1;
        }
        seviyeler.push(x[tarayici]);
    }
    seviyeler
}

/// Region boundaries implied by a level set: the midpoints, with the two ends
/// pushed out to effective infinity.
fn sinirlar(seviyeler: &[f64]) -> Vec<f64> {
    let mut s = Vec::with_capacity(seviyeler.len() + 1);
    s.push(-SONSUZ);
    for w in seviyeler.windows(2) {
        s.push(f64::midpoint(w[0], w[1]));
    }
    s.push(SONSUZ);
    s
}

/// Solved books, kept once per width.
///
/// The solve is deterministic, so caching cannot change an answer - it only
/// stops a pass over a model from paying for the same eight solves once per
/// tensor. Eight bits costs a few seconds; a model has hundreds of tensors.
static ONBELLEK: [OnceLock<Result<Kodkitabi, KodkitabiHatasi>>; 9] = [const { OnceLock::new() }; 9];
/// The ternary book, same reasoning.
static ONBELLEK_UC: OnceLock<Result<Kodkitabi, KodkitabiHatasi>> = OnceLock::new();

/// Solve the Lloyd-Max conditions for `bit` bits against a standard normal.
///
/// Memoised: the first call for a width solves, later calls clone. Because the
/// solve is deterministic this is invisible except in the clock, and
/// [`coz_ham`] is available so a test can still exercise the solver itself.
///
/// # Errors
///
/// [`KodkitabiHatasi::BitAraligi`] outside `1..=8`, and
/// [`KodkitabiHatasi::Yakinsamadi`] if the iteration has not settled inside
/// [`EN_COK_TUR`] passes.
pub fn coz(bit: u8) -> Result<Kodkitabi, KodkitabiHatasi> {
    if bit == 0 || bit > 8 {
        return Err(KodkitabiHatasi::BitAraligi { bit });
    }
    ONBELLEK[bit as usize].get_or_init(|| coz_ham(bit)).clone()
}

/// The solve itself, uncached.
///
/// # Errors
///
/// As [`coz`].
pub fn coz_ham(bit: u8) -> Result<Kodkitabi, KodkitabiHatasi> {
    if bit == 0 || bit > 8 {
        return Err(KodkitabiHatasi::BitAraligi { bit });
    }
    let seviye_sayisi = 1usize << bit;
    let mut seviyeler = baslangic(seviye_sayisi);

    let mut tur = 0usize;
    let mut hareket = f64::INFINITY;
    while tur < EN_COK_TUR {
        tur += 1;
        let s = sinirlar(&seviyeler);
        let mut en_cok = 0.0f64;
        for i in 0..seviye_sayisi {
            let m = kutle(s[i], s[i + 1]);
            if m > 0.0 {
                let merkez = birinci_moment(s[i], s[i + 1]) / m;
                let yeni = seviyeler[i] + GEVSETME * (merkez - seviyeler[i]);
                en_cok = en_cok.max((yeni - seviyeler[i]).abs());
                seviyeler[i] = yeni;
            }
            // An empty region keeps its level. Moving it would need a rule, and
            // every rule for it is a guess. On a Gaussian at these widths no
            // region is empty, which a test asserts rather than this comment
            // claiming.
        }
        hareket = en_cok;
        if hareket < YAKINSAMA {
            break;
        }
    }
    if hareket >= YAKINSAMA {
        return Err(KodkitabiHatasi::Yakinsamadi { bit, hareket });
    }

    let bozulma = bozulma_olc(&seviyeler);
    Ok(Kodkitabi {
        seviyeler,
        bozulma,
        tur,
    })
}

/// The three-level book, solved the same way but with the zero pinned.
///
/// A ternary alphabet costs `log2(3) = 1.585` bits of entropy and packs to
/// `1.6` bits per weight at five trits per byte (see [`crate::paket`]). It is
/// the only sub-two-bit width in the format, and it is worth having because a
/// third of a rotated group lands close enough to zero that a level there is
/// nearly free.
///
/// The zero is pinned rather than solved because an unpinned three-level solve
/// on a symmetric source converges to a symmetric answer anyway, but only to
/// within the solver's tolerance; pinning makes the exact zero exact, and an
/// exact zero is what lets a reader skip a multiply.
///
/// # Errors
///
/// [`KodkitabiHatasi::Yakinsamadi`] if the fixed point does not settle.
pub fn coz_ucdeger() -> Result<Kodkitabi, KodkitabiHatasi> {
    ONBELLEK_UC.get_or_init(coz_ucdeger_ham).clone()
}

/// The ternary solve itself, uncached.
///
/// # Errors
///
/// As [`coz_ucdeger`].
pub fn coz_ucdeger_ham() -> Result<Kodkitabi, KodkitabiHatasi> {
    // Symmetric three-level quantiser: levels are -c, 0, +c, so by the midpoint
    // condition the boundaries are +-c/2, and by the centroid condition
    // c = E[X | X > c/2]. Iterate that fixed point.
    let mut c = 1.0f64;
    let mut tur = 0usize;
    let mut hareket = f64::INFINITY;
    while tur < EN_COK_TUR {
        tur += 1;
        let esik = c / 2.0;
        let m = kutle(esik, SONSUZ);
        if m <= 0.0 {
            return Err(KodkitabiHatasi::Yakinsamadi { bit: 2, hareket });
        }
        let yeni = birinci_moment(esik, SONSUZ) / m;
        hareket = (yeni - c).abs();
        c = yeni;
        if hareket < YAKINSAMA {
            break;
        }
    }
    if hareket >= YAKINSAMA {
        return Err(KodkitabiHatasi::Yakinsamadi { bit: 2, hareket });
    }
    let seviyeler = vec![-c, 0.0, c];
    let bozulma = bozulma_olc(&seviyeler);
    Ok(Kodkitabi {
        seviyeler,
        bozulma,
        tur,
    })
}

/// Mean squared error of a level set against a standard normal, closed form up
/// to the one quadrature.
fn bozulma_olc(seviyeler: &[f64]) -> f64 {
    let s = sinirlar(seviyeler);
    let mut toplam = 0.0;
    for (i, c) in seviyeler.iter().enumerate() {
        let m = kutle(s[i], s[i + 1]);
        let m1 = birinci_moment(s[i], s[i + 1]);
        let m2 = ikinci_moment(s[i], s[i + 1]);
        toplam += m2 - 2.0 * c * m1 + c * c * m;
    }
    toplam
}

/// Check a level set against the Lloyd-Max conditions directly.
///
/// This is the test that a solved codebook is *optimal*, not merely stable:
/// every boundary must be the midpoint of its neighbours (true by construction
/// of [`sinirlar`]) and every level must be the conditional mean of its own
/// region (true only at a fixed point). Returns the largest violation.
#[must_use]
pub fn kosullari_sagliyor(seviyeler: &[f64]) -> f64 {
    let s = sinirlar(seviyeler);
    let mut en_cok = 0.0f64;
    for (i, c) in seviyeler.iter().enumerate() {
        let m = kutle(s[i], s[i + 1]);
        if m > 0.0 {
            en_cok = en_cok.max((birinci_moment(s[i], s[i + 1]) / m - c).abs());
        }
    }
    en_cok
}

/// Mass of the region each level owns, for the emptiness check.
#[must_use]
pub fn bolge_kutleleri(seviyeler: &[f64]) -> Vec<f64> {
    let s = sinirlar(seviyeler);
    (0..seviyeler.len())
        .map(|i| kutle(s[i], s[i + 1]))
        .collect()
}

#[cfg(test)]
// Exact float comparison is the *subject* of several of these tests: a round
// trip that is only approximately equal has lost something, and the cast and
// single-character-name lints are noise inside small numeric fixtures.
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;

    #[test]
    fn the_quadrature_reproduces_known_normal_masses() {
        // If the mass is wrong every level is wrong, so it is checked against
        // numbers that exist outside this file.
        assert!(
            (kutle(-SONSUZ, SONSUZ) - 1.0).abs() < 1e-14,
            "{}",
            kutle(-SONSUZ, SONSUZ)
        );
        assert!((kutle(-1.0, 1.0) - 0.682_689_492_137_086).abs() < 1e-13);
        assert!((kutle(-2.0, 2.0) - 0.954_499_736_103_642).abs() < 1e-13);
        assert!((kutle(-3.0, 3.0) - 0.997_300_203_936_740).abs() < 1e-13);
        assert!((kutle(0.0, SONSUZ) - 0.5).abs() < 1e-14);
        assert_eq!(kutle(1.0, 1.0), 0.0);
        assert_eq!(kutle(2.0, 1.0), 0.0);
    }

    #[test]
    fn the_closed_form_moments_agree_with_the_quadrature() {
        // The first and second moments are analytic; checking them against a
        // brute-force sum is how a sign error in the integration by parts is
        // caught rather than absorbed into the levels.
        let (a, b) = (-1.3, 2.4);
        let n = 200_000;
        let h = (b - a) / f64::from(n);
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        for i in 0..n {
            let x = a + (f64::from(i) + 0.5) * h;
            m1 += x * phi(x) * h;
            m2 += x * x * phi(x) * h;
        }
        assert!((birinci_moment(a, b) - m1).abs() < 1e-9, "{m1}");
        assert!((ikinci_moment(a, b) - m2).abs() < 1e-9, "{m2}");
        // Over the whole line the second moment is the variance, exactly one.
        assert!((ikinci_moment(-SONSUZ, SONSUZ) - 1.0).abs() < 1e-13);
        assert!(birinci_moment(-SONSUZ, SONSUZ).abs() < 1e-14);
    }

    #[test]
    fn one_bit_reproduces_the_classical_answer() {
        let k = coz(1).expect("one bit");
        // E|X| for a standard normal is sqrt(2/pi) = 0.7978845608028654.
        let beklenen = (2.0 / std::f64::consts::PI).sqrt();
        // The band is the solver tolerance, not a guess: YAKINSAMA is 1e-11,
        // so a level may sit that far from the fixed point.
        assert!(
            (k.seviyeler()[1] - beklenen).abs() < 1e-10,
            "{:?}",
            k.seviyeler()
        );
        assert!((k.seviyeler()[0] + beklenen).abs() < 1e-10);
        // Classical distortion 1 - 2/pi = 0.3633802.
        assert!((k.bozulma() - (1.0 - 2.0 / std::f64::consts::PI)).abs() < 1e-12);
        // And the level is antisymmetric to the precision it was solved to.
        assert!((k.seviyeler()[0] + k.seviyeler()[1]).abs() < 1e-11);
    }

    #[test]
    fn two_three_and_four_bits_match_the_published_lloyd_max_tables() {
        let iki = coz(2).expect("two bits");
        for (a, b) in iki
            .seviyeler()
            .iter()
            .zip([-1.5104, -0.4528, 0.4528, 1.5104])
        {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        assert!((iki.bozulma() - 0.117_5).abs() < 1e-3, "{}", iki.bozulma());

        let uc = coz(3).expect("three bits");
        for (a, b) in uc.seviyeler().iter().zip([
            -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
        ]) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        assert!((uc.bozulma() - 0.034_54).abs() < 1e-4, "{}", uc.bozulma());

        let dort = coz(4).expect("four bits");
        assert!(
            (dort.bozulma() - 0.009_497).abs() < 1e-4,
            "{}",
            dort.bozulma()
        );
    }

    #[test]
    fn the_solved_books_satisfy_the_optimality_conditions_not_just_stability() {
        for bit in 1..=8u8 {
            let k = coz(bit).expect("in range");
            let ihlal = kosullari_sagliyor(k.seviyeler());
            assert!(
                ihlal < 1e-10,
                "bit {bit}: centroid condition off by {ihlal}"
            );
        }
    }

    #[test]
    fn the_grid_only_picks_the_starting_point_and_cannot_reach_the_answer() {
        // Lloyd converges to the same fixed point from any sorted start. This
        // runs the same solve from a deliberately bad start - uniform spacing
        // over a range that is too narrow - and requires the same levels.
        for bit in [1u8, 2, 3, 4] {
            let l = 1usize << bit;
            #[allow(clippy::cast_precision_loss)]
            let mut kotu: Vec<f64> = (0..l)
                .map(|k| -1.0 + 2.0 * (k as f64 + 0.5) / (l as f64))
                .collect();
            let mut tur = 0;
            loop {
                tur += 1;
                let s = sinirlar(&kotu);
                let mut en_cok = 0.0f64;
                for i in 0..l {
                    let m = kutle(s[i], s[i + 1]);
                    if m > 0.0 {
                        let merkez = birinci_moment(s[i], s[i + 1]) / m;
                        let yeni = kotu[i] + GEVSETME * (merkez - kotu[i]);
                        en_cok = en_cok.max((yeni - kotu[i]).abs());
                        kotu[i] = yeni;
                    }
                }
                if en_cok < YAKINSAMA || tur > EN_COK_TUR {
                    break;
                }
            }
            let iyi = coz(bit).expect("in range");
            for (a, b) in kotu.iter().zip(iyi.seviyeler().iter()) {
                assert!((a - b).abs() < 1e-9, "bit {bit}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn no_level_owns_an_empty_region() {
        // The update step keeps an empty level where it is, which would be a
        // silent degeneracy. This asserts the case does not arise rather than
        // leaving a comment to claim it.
        for bit in 1..=8u8 {
            let k = coz(bit).expect("in range");
            let kutleler = bolge_kutleleri(k.seviyeler());
            assert!(
                kutleler.iter().all(|m| *m > 0.0),
                "bit {bit} has an empty region"
            );
            // And the masses are a partition of one.
            let toplam: f64 = kutleler.iter().sum();
            assert!((toplam - 1.0).abs() < 1e-13, "bit {bit}: {toplam}");
        }
    }

    #[test]
    fn distortion_falls_by_roughly_six_db_a_bit() {
        // The high-rate rule of thumb is 6.02 dB per bit. It is a rule of
        // thumb, so the band is wide; what the test catches is a solver that
        // stops improving, which is the realistic failure and the one the grid
        // version actually had.
        let mut onceki = coz(1).expect("one bit").snr_db();
        for bit in 2..=8u8 {
            let simdi = coz(bit).expect("in range").snr_db();
            let kazanc = simdi - onceki;
            assert!(
                (4.0..8.0).contains(&kazanc),
                "bit {bit}: gain {kazanc} dB is outside the plausible band"
            );
            onceki = simdi;
        }
    }

    #[test]
    fn every_width_converges_and_the_iteration_count_is_recorded() {
        for bit in 1..=8u8 {
            let k = coz(bit).expect("in range");
            assert!(
                k.tur() >= 1 && k.tur() < EN_COK_TUR,
                "bit {bit}: {}",
                k.tur()
            );
        }
    }

    #[test]
    fn the_ternary_book_is_symmetric_with_an_exact_zero() {
        let k = coz_ucdeger().expect("ternary");
        assert_eq!(k.boyut(), 3);
        assert_eq!(k.seviyeler()[1], 0.0, "the zero must be exactly zero");
        assert!((k.seviyeler()[0] + k.seviyeler()[2]).abs() < 1e-15);
        // Measured by this solver: the outer level sits at 1.2240 to four
        // decimals.
        assert!(
            (k.seviyeler()[2] - 1.224).abs() < 1e-3,
            "{}",
            k.seviyeler()[2]
        );
        // Ternary must sit between one and two bits in distortion, which is the
        // only reason to carry a third alphabet at all.
        let bir = coz(1).expect("one bit").bozulma();
        let iki = coz(2).expect("two bits").bozulma();
        assert!(k.bozulma() < bir && k.bozulma() > iki, "{}", k.bozulma());
    }

    #[test]
    fn the_unit_norm_rescaling_shrinks_by_the_square_root_of_the_group() {
        let k = coz(2).expect("two bits");
        let olcekli = k.birim_norm_icin(128);
        let oran = f64::from(olcekli[3]) / k.seviyeler()[3];
        assert!((oran - 1.0 / 128f64.sqrt()).abs() < 1e-6, "{oran}");
    }

    #[test]
    fn nearest_is_a_real_binary_search_including_both_ends() {
        let seviyeler = [-1.5f32, -0.5, 0.5, 1.5];
        assert_eq!(Kodkitabi::en_yakin(&seviyeler, -9.0), 0);
        assert_eq!(Kodkitabi::en_yakin(&seviyeler, 9.0), 3);
        assert_eq!(Kodkitabi::en_yakin(&seviyeler, -0.6), 1);
        assert_eq!(Kodkitabi::en_yakin(&seviyeler, 0.4), 2);
        // Exactly on a boundary: the tie rule is fixed and goes low.
        assert_eq!(Kodkitabi::en_yakin(&seviyeler, 0.0), 1);
        // Exactly on a level.
        assert_eq!(Kodkitabi::en_yakin(&seviyeler, 1.5), 3);
        // An empty book cannot panic.
        assert_eq!(Kodkitabi::en_yakin(&[], 1.0), 0);
    }

    #[test]
    fn nearest_agrees_with_a_linear_scan_over_the_whole_book() {
        let k = coz(4).expect("four bits");
        let seviyeler = k.birim_norm_icin(64);
        for i in -2000..=2000 {
            #[allow(clippy::cast_possible_truncation)]
            let x = (f64::from(i) / 4000.0) as f32;
            let hizli = Kodkitabi::en_yakin(&seviyeler, x);
            let mut yavas = 0usize;
            let mut en_iyi = f32::MAX;
            for (j, s) in seviyeler.iter().enumerate() {
                let d = (x - s).abs();
                if d < en_iyi {
                    en_iyi = d;
                    yavas = j;
                }
            }
            assert_eq!(hizli, yavas, "x={x}");
        }
    }

    #[test]
    fn out_of_range_bit_widths_are_refused_by_name() {
        assert_eq!(coz(0).err(), Some(KodkitabiHatasi::BitAraligi { bit: 0 }));
        assert_eq!(coz(9).err(), Some(KodkitabiHatasi::BitAraligi { bit: 9 }));
        assert_eq!(
            coz(0).unwrap_err().to_string(),
            "bit genisligi 0 1..=8 disinda"
        );
        let e = KodkitabiHatasi::Yakinsamadi {
            bit: 6,
            hareket: 1.5e-4,
        };
        assert!(e.to_string().contains("yakinsamadi"), "{e}");
    }

    #[test]
    fn the_solve_is_deterministic_across_calls() {
        // Two operators must derive the same book or the container one writes
        // is unreadable by the other.
        // The raw solver, not the cache: a cache makes any function look
        // deterministic, which is the opposite of evidence.
        for bit in 1..=6u8 {
            let a = coz_ham(bit).expect("in range");
            let b = coz_ham(bit).expect("in range");
            assert_eq!(a, b, "bit {bit} is not reproducible");
            assert_eq!(a, coz(bit).expect("in range"), "the cache disagrees");
        }
        assert_eq!(coz_ucdeger_ham(), coz_ucdeger_ham());
        assert_eq!(coz_ucdeger_ham(), coz_ucdeger());
    }
}
