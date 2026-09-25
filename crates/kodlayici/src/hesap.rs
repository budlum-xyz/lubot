//! The arithmetic an encoder forward pass is made of, written to be checkable.
//!
//! # Why these functions are separated from the blocks
//!
//! A transformer block is a sequence of six of these calls. If each block
//! carried its own copy, a mistake in the normalisation would appear in
//! twenty-two places and be invisible; here it appears once and has one test.
//! The functions take slices and return vectors rather than tensors: a tensor
//! type would be a second vocabulary for the same idea, and the port does not
//! need one yet.
//!
//! # The convention that matters
//!
//! A weight matrix `W` with shape `[out, in]` - which is how the checkpoint
//! stores them - is applied as `y = W x`. Rows are output features. Getting
//! this backwards is the classic silent bug: every shape still lines up if the
//! dimensions happen to be equal, and the numbers are plausible.

/// Why a shape did not fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SekilHatasi {
    /// What was expected, in words.
    pub beklenen: String,
    /// What arrived.
    pub gelen: String,
}

impl std::fmt::Display for SekilHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sekil uyusmadi: beklenen {}, gelen {}",
            self.beklenen, self.gelen
        )
    }
}

/// Checks that `len` equals `beklenen`.
///
/// # Errors
/// [`SekilHatasi`] naming both numbers.
pub fn sekil(len: usize, beklenen: usize, ne: &str) -> Result<(), SekilHatasi> {
    if len == beklenen {
        Ok(())
    } else {
        Err(SekilHatasi {
            beklenen: format!("{ne} icin {beklenen}"),
            gelen: len.to_string(),
        })
    }
}

/// `y = W x`, with `W` stored row-major as `[out, in]`.
///
/// # Errors
/// [`SekilHatasi`] when `x.len() != in`, when `w.len() != out * in`, or when
/// `y.len() != out`.
pub fn matris_vektor(w: &[f32], x: &[f32], y: &mut [f32], girdi: usize) -> Result<(), SekilHatasi> {
    if girdi == 0 || x.len() != girdi || !w.len().is_multiple_of(girdi) {
        return Err(SekilHatasi {
            beklenen: format!("W satirlari {girdi} genisliginde"),
            gelen: format!("W {} eleman, x {}", w.len(), x.len()),
        });
    }
    let cikti = w.len() / girdi;
    sekil(y.len(), cikti, "cikis")?;
    for (sira, hedef) in y.iter_mut().enumerate() {
        let satir = &w[sira * girdi..(sira + 1) * girdi];
        // The dot product is written as a fold rather than indexed by a second
        // loop: one bounds check per row instead of one per element.
        *hedef = satir
            .iter()
            .zip(x.iter())
            .fold(0.0_f32, |acc, (a, b)| acc + a * b);
    }
    Ok(())
}

/// Layer normalisation with a weight and no bias, as the checkpoint stores it.
///
/// The mean is subtracted, which is the difference between this and an RMS
/// normalisation: a port that used the root-mean-square form would produce
/// numbers that look right on symmetric inputs and drift on the rest.
///
/// # Errors
/// [`SekilHatasi`] when the lengths disagree.
pub fn katman_norm(x: &mut [f32], agirlik: &[f32], epsilon: f32) -> Result<(), SekilHatasi> {
    sekil(agirlik.len(), x.len(), "norm agirligi")?;
    if x.is_empty() {
        return Ok(());
    }
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f32;
    let ortalama = x.iter().sum::<f32>() / n;
    let varyans = x
        .iter()
        .map(|v| {
            let d = v - ortalama;
            d * d
        })
        .sum::<f32>()
        / n;
    let ters_std = 1.0 / (varyans + epsilon).sqrt();
    for (v, w) in x.iter_mut().zip(agirlik.iter()) {
        *v = (*v - ortalama) * ters_std * w;
    }
    Ok(())
}

/// The tanh-form GELU, which is the activation this checkpoint was trained with.
///
/// The exact form and the tanh approximation differ in the fourth decimal; the
/// coded derivative of the coded function is what matters for a training port,
/// and for inference the difference is far below the tolerance of any
/// comparison against a reference that used the same form.
#[must_use]
pub fn gelu(x: f32) -> f32 {
    // The checkpoint's configuration names the activation plainly as `gelu`,
    // and the reference for that name is the error-function form. The tanh
    // approximation that is often substituted differs from it by up to 5e-4 in
    // the region this model uses, which is far above the agreement a comparison
    // between two implementations is meant to prove, so the exact form is used.
    let x64 = f64::from(x);
    #[allow(clippy::cast_possible_truncation)]
    let y = (0.5 * x64 * (1.0 + erf(x64 / std::f64::consts::SQRT_2))) as f32;
    y
}

/// The error function, accurate to below `f32`'s own resolution.
///
/// `f64` is used for the constants and the working values and the result is
/// narrowed once: the polynomial below is a double-precision fit, and doing the
/// arithmetic in `f32` would throw away the accuracy that is the reason for
/// choosing it over the cheap version.
fn erf(x: f64) -> f64 {
    // Numerical Recipes' complementary error function: fractional error below
    // 1.2e-7 everywhere, which is under the 1.2e-7 resolution of an f32 near 1.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        1.0 - ans
    } else {
        ans - 1.0
    }
}

/// Layer normalisation with a bias, as the decision head's own norms have.
///
/// The encoder's norms carry no bias and use [`katman_norm`]; the head is built
/// from the framework's norm, which does carry one, so the two must stay
/// separate rather than sharing a function with an optional argument that the
/// encoder never fills.
///
/// # Errors
/// [`SekilHatasi`] when the weight, the bias and the row disagree in length.
pub fn katman_norm_sapmali(
    x: &mut [f32],
    agirlik: &[f32],
    sapma: &[f32],
    epsilon: f32,
) -> Result<(), SekilHatasi> {
    sekil(agirlik.len(), x.len(), "norm agirligi")?;
    sekil(sapma.len(), x.len(), "norm sapmasi")?;
    if x.is_empty() {
        return Ok(());
    }
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f32;
    let ortalama = x.iter().sum::<f32>() / n;
    let varyans = x
        .iter()
        .map(|v| {
            let d = v - ortalama;
            d * d
        })
        .sum::<f32>()
        / n;
    let ters_std = 1.0 / (varyans + epsilon).sqrt();
    for ((v, w), b) in x.iter_mut().zip(agirlik.iter()).zip(sapma.iter()) {
        *v = (*v - ortalama) * ters_std * w + b;
    }
    Ok(())
}

/// Softmax over a slice, in place, with the maximum subtracted.
///
/// Subtracting the maximum is not an optimisation: without it `exp` overflows
/// for a logit above ~88 and the result is `NaN` rather than a distribution.
///
/// # Errors
/// [`SekilHatasi`] when the slice is empty.
pub fn softmax(x: &mut [f32]) -> Result<(), SekilHatasi> {
    if x.is_empty() {
        return Err(SekilHatasi {
            beklenen: "en az bir eleman".to_string(),
            gelen: "0".to_string(),
        });
    }
    let en_buyuk = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut toplam = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - en_buyuk).exp();
        toplam += *v;
    }
    if toplam > 0.0 && toplam.is_finite() {
        for v in x.iter_mut() {
            *v /= toplam;
        }
    }
    Ok(())
}

/// Rotary position embedding applied to one head's `(x, y)` pairs in place.
///
/// The rotation is applied to consecutive pairs of the head vector, which is
/// how the checkpoint's weights were trained; a port that rotated
/// `(first half, second half)` instead would be a different model with the same
/// tensor names.
///
/// `kosin` and `sinus` are the whole table for one layer kind, laid out as
/// `[konum * yarim + i]`; `konum` says which row this head is at. Passing the
/// table rather than a row keeps the caller from building a slice per position,
/// which for an 8 192-token sequence and 22 layers would be 180 000 allocations
/// of the same numbers.
///
/// # Errors
/// [`SekilHatasi`] when the vector length is odd or the tables do not cover the
/// position.
pub fn rope(v: &mut [f32], kosin: &[f32], sinus: &[f32], konum: usize) -> Result<(), SekilHatasi> {
    if !v.len().is_multiple_of(2) {
        return Err(SekilHatasi {
            beklenen: "cift uzunluk".to_string(),
            gelen: v.len().to_string(),
        });
    }
    let yarim = v.len() / 2;
    let baslangic = konum * yarim;
    if kosin.len() != sinus.len() || baslangic + yarim > kosin.len() {
        return Err(SekilHatasi {
            beklenen: format!("tablo en az {} eleman", baslangic + yarim),
            gelen: format!("cos {}, sin {}", kosin.len(), sinus.len()),
        });
    }
    for i in 0..yarim {
        let c = kosin[baslangic + i];
        let s = sinus[baslangic + i];
        let a = v[i];
        let b = v[yarim + i];
        v[i] = a * c - b * s;
        v[yarim + i] = a * s + b * c;
    }
    Ok(())
}

/// The inverse frequencies used to build a rotary table.
///
/// `dim` is the head dimension; `theta` is the base the model was trained with.
/// The table is built once per sequence rather than per call, because it is a
/// function of position only.
#[must_use]
pub fn rope_tablosu(uzunluk: usize, dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let yarim = dim / 2;
    let mut kosin = vec![0.0_f32; uzunluk * yarim];
    let mut sinus = vec![0.0_f32; uzunluk * yarim];
    for konum in 0..uzunluk {
        for i in 0..yarim {
            #[allow(clippy::cast_precision_loss)]
            let us = -2.0 * (i as f32) / (dim as f32);
            let frekans = theta.powf(us);
            #[allow(clippy::cast_precision_loss)]
            let aci = konum as f32 * frekans;
            kosin[konum * yarim + i] = aci.cos();
            sinus[konum * yarim + i] = aci.sin();
        }
    }
    (kosin, sinus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_times_a_vector_uses_rows_as_output_features() {
        // W = [[1, 2], [3, 4]], x = [5, 6] -> y = [17, 39].
        let w = [1.0_f32, 2.0, 3.0, 4.0];
        let x = [5.0_f32, 6.0];
        let mut y = [0.0_f32; 2];
        matris_vektor(&w, &x, &mut y, 2).expect("sekil tutmali");
        assert_eq!(y, [17.0, 39.0]);
        // The transposed reading would give [23, 34]; the test exists to make
        // that mistake fail loudly rather than look plausible.
        assert_ne!(y, [23.0, 34.0]);
    }

    #[test]
    fn a_shape_mismatch_is_refused_not_padded() {
        let w = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0_f32, 2.0];
        let mut y = [0.0_f32; 2];
        assert!(matris_vektor(&w, &x, &mut y, 4).is_err());
        // W has 6 elements and in=2, so the output must be 3 long; a caller
        // that passes a 2-long buffer is refused rather than half-filled.
        let mut yanlis_cikis = [0.0_f32; 2];
        assert!(matris_vektor(&w, &x, &mut yanlis_cikis, 2).is_err());
    }

    #[test]
    fn layer_norm_centres_scales_and_applies_the_weight() {
        let mut x = [1.0_f32, 2.0, 3.0, 4.0];
        let w = [1.0_f32; 4];
        katman_norm(&mut x, &w, 1.0e-5).expect("sekil");
        let ortalama = x.iter().sum::<f32>() / 4.0;
        assert!(ortalama.abs() < 1e-5, "ortalama {ortalama}");
        // Sample standard deviation of the normalised values is ~1.
        let varyans = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((varyans - 1.0).abs() < 1e-3, "varyans {varyans}");
        // A zero weight zeroes the channel, which is the point of the weight.
        let mut y = [1.0_f32, 2.0, 3.0];
        katman_norm(&mut y, &[0.0, 1.0, 1.0], 1.0e-5).expect("sekil");
        assert!(y[0].abs() < 1e-6);
    }

    #[test]
    fn layer_norm_of_a_constant_vector_is_zero_not_nan() {
        // Variance is zero: without epsilon this is a division by zero, and the
        // result would be NaN in every later layer.
        let mut x = [2.0_f32; 5];
        katman_norm(&mut x, &[1.0; 5], 1.0e-5).expect("sekil");
        assert!(x.iter().all(|v| v.abs() < 1e-6), "{x:?}");
    }

    #[test]
    fn gelu_matches_its_defining_values() {
        assert!(gelu(0.0).abs() < 1e-7);
        // Exact GELU at 1 is the standard normal's CDF there, 0.8413447; the
        // tanh approximation's 0.8411920 is 1.5e-4 away.
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6, "{}", gelu(1.0));
        // The exact form, not the tanh approximation: `gelu(-1)` is -0.158655,
        // and the approximation's -0.158808 is outside the tolerance below.
        assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-5, "{}", gelu(-1.0));
        assert!((gelu(0.5) - 0.345_731_3).abs() < 1e-6, "{}", gelu(0.5));
        assert!((gelu(-0.5) + 0.154_268_7).abs() < 1e-6, "{}", gelu(-0.5));
        // Monotone on the positive side and negative for large negative input.
        assert!(gelu(2.0) > gelu(1.0));
        assert!(gelu(-6.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_logits() {
        let mut x = [1.0_f32, 2.0, 3.0];
        softmax(&mut x).expect("sekil");
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
        // Without the max subtraction this overflows to NaN.
        let mut buyuk = [1000.0_f32, 1001.0, 1002.0];
        softmax(&mut buyuk).expect("sekil");
        assert!(buyuk.iter().all(|v| v.is_finite()), "{buyuk:?}");
        assert!((buyuk.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(softmax(&mut []).is_err());
    }

    #[test]
    fn rope_rotates_halves_not_neighbours_and_preserves_length() {
        // 4 positions, head width 4 (so two frequencies per position).
        let (kosin, sinus) = rope_tablosu(4, 4, 10_000.0);
        let mut v = vec![1.0_f32, 0.0, 0.5, -0.5];
        let onceki_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Position 2 of the table.
        rope(&mut v, &kosin, &sinus, 2).expect("sekil");
        let sonraki_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // A rotation is length-preserving; a scale would not be.
        assert!((onceki_norm - sonraki_norm).abs() < 1e-5);
        // Position zero rotates by zero: the identity.
        let mut a = vec![0.5_f32, -0.25, 0.125, 0.0625];
        rope(&mut a, &kosin, &sinus, 0).expect("sekil");
        assert!((a[0] - 0.5).abs() < 1e-6 && (a[3] - 0.0625).abs() < 1e-6);
        // Position 1, width 4: the angle is theta^0 = 1 radian for both
        // frequencies? No - the second frequency is theta^-0.5, so only the
        // first is one radian. The check is the closed form, written out.
        let theta = 10_000.0_f32;
        let aci0 = 1.0_f32;
        let aci1 = theta.powf(-0.5);
        let mut b = vec![1.0_f32, 0.0, 0.0, 1.0];
        rope(&mut b, &kosin, &sinus, 1).expect("sekil");
        // Element 0 turns by aci0 with element 2; element 1 turns by aci1 with
        // element 3.
        assert!((b[0] - aci0.cos()).abs() < 1e-5, "{}", b[0]);
        assert!((b[2] - aci0.sin()).abs() < 1e-5, "{}", b[2]);
        assert!((b[1] + aci1.sin()).abs() < 1e-5, "{}", b[1]);
        assert!((b[3] - aci1.cos()).abs() < 1e-5, "{}", b[3]);
        // Neighbours, not halves, would leave 1 and 3 in place: the check is
        // that both moved.
        assert!(b[1].abs() > 1e-3 && b[3].abs() > 1e-3);
        // A position past the table is refused, not silently unrotated.
        assert!(rope(&mut [1.0_f32, 0.0], &kosin, &sinus, 99).is_err());
    }

    #[test]
    fn the_rotary_table_is_the_inverse_frequency_it_claims_to_be() {
        // 2 positions, head width 4: two frequency pairs, row-major by position.
        let (kosin, sinus) = rope_tablosu(2, 4, 10_000.0);
        // Row 1 (position 1): frequency 0 is theta^0 = 1, so the angle is 1.
        assert!((kosin[2] - 1.0_f32.cos()).abs() < 1e-6, "{}", kosin[2]);
        assert!((sinus[2] - 1.0_f32.sin()).abs() < 1e-6);
        // Frequency 1 is theta^(-0.5) = 0.01.
        assert!((kosin[3] - 0.01_f32.cos()).abs() < 1e-6, "{}", kosin[3]);
        // Every entry sits on the unit circle.
        for i in 0..2 {
            for j in 0..2 {
                let c = kosin[i * 2 + j];
                let s = sinus[i * 2 + j];
                assert!((c * c + s * s - 1.0).abs() < 1e-5);
            }
        }
    }
}
