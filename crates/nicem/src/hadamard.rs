//! The rotation that makes two bits enough.
//!
//! # The problem it solves
//!
//! A row of trained weights is not Gaussian. It is mostly small values with a
//! handful of large ones, and a scalar quantiser sized for the outliers spends
//! all of its levels on a range almost nothing occupies. At four bits the waste
//! is affordable. At two bits there are four levels in total and the outlier
//! takes two of them, so the format collapses.
//!
//! A Walsh-Hadamard transform is an orthogonal map whose every entry has the
//! same magnitude. Applying it to a group mixes every input into every output
//! in equal measure, so a single large coordinate is spread across the whole
//! group instead of dominating it. What comes out looks far more like a
//! Gaussian than what went in - which is exactly the distribution the codebook
//! in [`crate::kodkitabi`] is built for. The transform is its own inverse under
//! the `1/sqrt(n)` normalisation used here, so the reader undoes it with the
//! same routine and no second table.
//!
//! This is a rotation, not a compression: it moves the energy around, it does
//! not remove any. The saving comes entirely from what the codebook can then do
//! with the rotated values, and that saving is measured in [`crate::grup`].
//!
//! # Why the size must be a power of two
//!
//! The fast transform is the radix-2 butterfly: `log2(n)` passes over the
//! array, `n/2` add/subtract pairs each, `O(n log n)` total with no table and
//! no multiplication until the final scale. Non-power-of-two Hadamard matrices
//! exist for some sizes, but they have no butterfly, need a stored matrix, and
//! would make the container's group size a lookup rather than a shift. The
//! refusal is [`HadamardHatasi::IkininKuvvetiDegil`] and it happens at
//! construction, not at the first tensor.

/// Why a transform was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HadamardHatasi {
    /// The length is not a power of two, so there is no butterfly for it.
    IkininKuvvetiDegil { uzunluk: usize },
    /// The length is zero. An empty group is not a group of size zero, it is a
    /// bug upstream, and returning "success" on it hides the bug.
    Bos,
}

impl std::fmt::Display for HadamardHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IkininKuvvetiDegil { uzunluk } => {
                write!(f, "grup uzunlugu {uzunluk} ikinin kuvveti degil")
            }
            Self::Bos => write!(f, "bos grup donusturulemez"),
        }
    }
}

impl std::error::Error for HadamardHatasi {}

/// The normalised fast Walsh-Hadamard transform, in place.
///
/// After the butterfly every element is scaled by `1 / sqrt(n)`, which makes the
/// map orthonormal and therefore self-inverse: `wht(wht(x)) == x` up to floating
/// point. Norm is preserved, which is what lets [`crate::grup`] store one scale
/// per group and trust it after the round trip.
///
/// # Errors
///
/// [`HadamardHatasi::Bos`] for an empty slice and
/// [`HadamardHatasi::IkininKuvvetiDegil`] for a length that has no butterfly.
pub fn wht(x: &mut [f32]) -> Result<(), HadamardHatasi> {
    let n = x.len();
    if n == 0 {
        return Err(HadamardHatasi::Bos);
    }
    if !n.is_power_of_two() {
        return Err(HadamardHatasi::IkininKuvvetiDegil { uzunluk: n });
    }
    butterfly(x);
    let olcek = normalizasyon(n);
    for v in x.iter_mut() {
        *v *= olcek;
    }
    Ok(())
}

/// The unnormalised butterfly. Separate from [`wht`] so the scale can be
/// applied once at the end instead of once per pass, and so a caller that wants
/// to fold the scale into something else can.
fn butterfly(x: &mut [f32]) {
    let n = x.len();
    let mut adim = 1;
    while adim < n {
        let mut i = 0;
        while i < n {
            for j in i..i + adim {
                let ust = x[j];
                let alt = x[j + adim];
                x[j] = ust + alt;
                x[j + adim] = ust - alt;
            }
            i += adim << 1;
        }
        adim <<= 1;
    }
}

/// `1 / sqrt(n)`, computed once per call rather than per element.
#[allow(clippy::cast_precision_loss)]
fn normalizasyon(n: usize) -> f32 {
    1.0 / (n as f32).sqrt()
}

/// Build the dense `n x n` normalised Hadamard matrix.
///
/// The fast path is the only one used in the format; this exists so the tests
/// can check the butterfly against the definition rather than against itself. A
/// fast transform verified only by its own inverse can be wrong in a way that
/// cancels - a permuted output is still self-inverse.
///
/// # Errors
///
/// Same refusals as [`wht`].
pub fn yogun_matris(n: usize) -> Result<Vec<Vec<f32>>, HadamardHatasi> {
    if n == 0 {
        return Err(HadamardHatasi::Bos);
    }
    if !n.is_power_of_two() {
        return Err(HadamardHatasi::IkininKuvvetiDegil { uzunluk: n });
    }
    let olcek = normalizasyon(n);
    let mut m = vec![vec![0.0f32; n]; n];
    for (i, satir) in m.iter_mut().enumerate() {
        for (j, hucre) in satir.iter_mut().enumerate() {
            // The sign is the parity of the population count of the bitwise
            // and: this is the Sylvester construction written directly.
            let isaret = if (i & j).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            *hucre = isaret * olcek;
        }
    }
    Ok(m)
}

/// Sum of squares, used by the tests and by the group quantiser to show that
/// the rotation preserved the norm.
#[must_use]
pub fn enerji(x: &[f32]) -> f64 {
    x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum()
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

    fn referans(x: &[f32]) -> Vec<f32> {
        let m = yogun_matris(x.len()).expect("power of two");
        (0..x.len())
            .map(|i| (0..x.len()).map(|j| m[i][j] * x[j]).sum())
            .collect()
    }

    #[test]
    fn the_butterfly_agrees_with_the_definition() {
        // A fast transform checked only against its own inverse can be a
        // permutation of the right answer. This compares against the dense
        // matrix built from the Sylvester rule.
        for log in 1..=8u32 {
            let n = 1usize << log;
            let mut x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 19) as f32) - 9.0 + (i as f32) * 0.25)
                .collect();
            let beklenen = referans(&x);
            wht(&mut x).expect("power of two");
            for (a, b) in x.iter().zip(beklenen.iter()) {
                assert!((a - b).abs() < 1e-3, "n={n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn it_is_its_own_inverse() {
        let mut x: Vec<f32> = (0..128).map(|i| ((i % 7) as f32) * 0.5 - 1.5).collect();
        let onceki = x.clone();
        wht(&mut x).expect("power of two");
        wht(&mut x).expect("power of two");
        for (a, b) in x.iter().zip(onceki.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn energy_is_preserved_which_is_what_lets_one_scale_serve_a_group() {
        let mut x: Vec<f32> = (0..64).map(|i| ((i * 13 % 31) as f32) - 15.0).collect();
        let once = enerji(&x);
        wht(&mut x).expect("power of two");
        let sonra = enerji(&x);
        assert!(
            (once - sonra).abs() / once < 1e-5,
            "energy moved: {once} -> {sonra}"
        );
    }

    #[test]
    fn a_single_spike_comes_out_flat_which_is_the_whole_point() {
        // One large coordinate in a group of 128 is the case that breaks a
        // two-bit scalar quantiser. After the rotation the energy is spread
        // evenly, so no coordinate needs a level of its own.
        let mut x = vec![0.0f32; 128];
        x[5] = 100.0;
        wht(&mut x).expect("power of two");
        let en_buyuk = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let en_kucuk = x.iter().fold(f32::MAX, |a, v| a.min(v.abs()));
        assert!(
            (en_buyuk - en_kucuk).abs() < 1e-3,
            "the spike was not spread: {en_kucuk} .. {en_buyuk}"
        );
        // Measured: the peak-to-mean ratio fell from sqrt(128) to 1.
        assert!((en_buyuk - 100.0 / 128f32.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn a_non_power_of_two_is_refused_at_the_call_not_at_the_tensor() {
        let mut x = vec![1.0f32; 96];
        assert_eq!(
            wht(&mut x),
            Err(HadamardHatasi::IkininKuvvetiDegil { uzunluk: 96 })
        );
        assert_eq!(
            yogun_matris(96).unwrap_err().to_string(),
            "grup uzunlugu 96 ikinin kuvveti degil"
        );
    }

    #[test]
    fn an_empty_group_is_an_error_not_a_quiet_success() {
        let mut x: Vec<f32> = Vec::new();
        assert_eq!(wht(&mut x), Err(HadamardHatasi::Bos));
    }

    #[test]
    fn size_one_is_the_identity() {
        let mut x = vec![3.5f32];
        wht(&mut x).expect("one is a power of two");
        assert!((x[0] - 3.5).abs() < 1e-6);
    }
}
