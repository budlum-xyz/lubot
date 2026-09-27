//! Half precision, written out, because the group scale is the one number a
//! sub-byte format cannot afford to store at full width.
//!
//! # Why this exists at all
//!
//! A quantised group is a packed index array plus one scale. At two bits per
//! weight and a group of 128, an `f32` scale costs `32 / 128 = 0.25` bits per
//! weight - more than a tenth of the whole budget. An `f16` scale costs
//! `16 / 128 = 0.125` bits per weight, and the measured reconstruction error
//! does not move (see [`crate::grup`]'s measurement, not a claim made here).
//! So the format stores `f16`, and the conversion has to live somewhere.
//!
//! # Why it is not a cast
//!
//! Rust has `f32 as f16` only on nightly, and the obvious hand-rolled version -
//! truncate the mantissa - is biased: it always rounds toward zero, so a tensor
//! of positive scales comes back systematically small. The bias is invisible in
//! a round-trip test that uses exact halves and shows up as a slow drift in the
//! dequantised norm. This module rounds to nearest, ties to even, and the tie
//! cases are in the tests by name.
//!
//! Subnormals are handled rather than flushed. A scale that underflows to zero
//! turns a whole group into zeros, which is a silent, total loss of one group's
//! worth of weights; flushing is the kind of shortcut that is only ever noticed
//! by the model.

/// Exponent bias of binary32.
const F32_BIAS: i32 = 127;
/// Exponent bias of binary16.
const F16_BIAS: i32 = 15;
/// Largest finite binary16 value, `(2 - 2^-10) * 2^15`.
pub const YARIM_EN_BUYUK: f32 = 65504.0;
/// Smallest positive *normal* binary16 value, `2^-14`.
pub const YARIM_EN_KUCUK_NORMAL: f32 = 6.103_515_6e-5;
/// Smallest positive subnormal binary16 value, `2^-24`.
pub const YARIM_EN_KUCUK_SUBNORMAL: f32 = 5.960_464_5e-8;

/// Encode an `f32` as binary16 bits, rounding to nearest with ties to even.
///
/// Infinities and NaN survive as infinities and NaN; a NaN keeps its sign and
/// is given a non-zero payload so it cannot be mistaken for an infinity after
/// the mantissa is truncated. A finite value too large for the format becomes
/// an infinity of the same sign rather than the largest finite value: saturating
/// would turn an overflow into a plausible number, and a scale that is plausible
/// but wrong is worse than one that is obviously broken.
///
/// Every cast below is narrowing on purpose and range-checked by the branch it
/// sits in: the exponent has been bounded to `-25..=15`, the shifted mantissa
/// to ten bits, and the sign extracted before either.
#[must_use]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn f32_to_yarim(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exp == 0xff {
        // Infinity or NaN. A NaN whose payload truncates to zero would become
        // an infinity, so force a quiet bit.
        return if mantissa == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }

    let unbiased = exp - F32_BIAS;

    if unbiased > 15 {
        return sign | 0x7c00; // overflow to infinity
    }

    if unbiased >= -14 {
        // Normal in binary16: 10 mantissa bits, so drop 13.
        let yarim_exp = ((unbiased + F16_BIAS) as u32) << 10;
        let kalan = mantissa & 0x1fff;
        let mut taban = yarim_exp | (mantissa >> 13);
        // Round to nearest, ties to even.
        if kalan > 0x1000 || (kalan == 0x1000 && (taban & 1) == 1) {
            taban += 1; // may carry into the exponent, which is correct
        }
        if taban >= 0x7c00 {
            return sign | 0x7c00; // rounding pushed it over the top
        }
        return sign | (taban as u16);
    }

    // Subnormal in binary16, or a true zero.
    if unbiased < -25 {
        return sign; // rounds to zero even before the tie rule
    }
    // Rebuild the implicit leading one and shift into the subnormal grid.
    let tam = mantissa | 0x0080_0000;
    let kaydirma = (-unbiased - 14) + 13; // 14 <= kaydirma <= 24
    let taban = tam >> kaydirma;
    let kalan = tam & ((1u32 << kaydirma) - 1);
    let orta = 1u32 << (kaydirma - 1);
    let mut sonuc = taban;
    if kalan > orta || (kalan == orta && (taban & 1) == 1) {
        sonuc += 1; // may carry up into the smallest normal, which is correct
    }
    sign | (sonuc as u16)
}

/// Decode binary16 bits back to `f32`. Exact: every binary16 is an `f32`.
///
/// The exponent arithmetic is done in `i32` and lands in `1..=254` for every
/// input, so the cast back to the unsigned field cannot lose a sign.
#[must_use]
#[allow(clippy::cast_sign_loss)]
pub fn yarim_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = i32::from((h >> 10) & 0x1f);
    let mantissa = u32::from(h & 0x03ff);

    if exp == 0x1f {
        let bits = sign | 0x7f80_0000 | (mantissa << 13);
        return f32::from_bits(bits);
    }

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: renormalise into binary32, which has room for all of them.
        let mut m = mantissa;
        let mut e = -14_i32;
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        let bits = sign | (((e + F32_BIAS) as u32) << 23) | (m << 13);
        return f32::from_bits(bits);
    }

    let bits = sign | (((exp - F16_BIAS + F32_BIAS) as u32) << 23) | (mantissa << 13);
    f32::from_bits(bits)
}

/// Round an `f32` through binary16 and back, which is what the container will
/// do to every group scale. Useful in the quantiser so that the error it
/// measures is the error the reader will see, not the error of an `f32` scale
/// that never reaches disk.
#[must_use]
pub fn yuvarla(x: f32) -> f32 {
    yarim_to_f32(f32_to_yarim(x))
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
    fn exact_values_round_trip_bit_for_bit() {
        for x in [0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, 1024.0, -0.125] {
            assert_eq!(yuvarla(x), x, "{x} did not survive the round trip");
        }
    }

    #[test]
    fn the_sign_of_negative_zero_survives() {
        assert!(yuvarla(-0.0).is_sign_negative());
        assert_eq!(f32_to_yarim(-0.0), 0x8000);
    }

    #[test]
    fn ties_go_to_even_not_away_from_zero() {
        // 1 + 2^-11 sits exactly between 1.0 and the next binary16 up
        // (1 + 2^-10). The even neighbour is 1.0, so a correct tie rule keeps
        // it. Rounding away from zero - the usual shortcut - would give
        // 1.0009766 and bias every scale upward.
        let orta = 1.0f32 + 2f32.powi(-11);
        assert_eq!(yuvarla(orta), 1.0);

        // 1 + 3*2^-11 is a tie between (1 + 2^-10) and (1 + 2^-9); the even
        // neighbour is the upper one.
        let ust = 1.0f32 + 3.0 * 2f32.powi(-11);
        assert_eq!(yuvarla(ust), 1.0 + 2f32.powi(-9));
    }

    #[test]
    fn truncation_would_have_been_biased_and_is_not_what_happens() {
        // A value just below a representable point must round up, not down.
        let hemen_alt = 1.0f32 + 0.9 * 2f32.powi(-10);
        assert_eq!(yuvarla(hemen_alt), 1.0 + 2f32.powi(-10));
    }

    #[test]
    fn subnormals_are_kept_not_flushed() {
        let en_kucuk = YARIM_EN_KUCUK_SUBNORMAL;
        assert_eq!(yuvarla(en_kucuk), en_kucuk);
        assert_ne!(f32_to_yarim(en_kucuk), 0);
        // Half of the smallest subnormal is a tie against zero; zero is even.
        assert_eq!(yuvarla(en_kucuk / 2.0), 0.0);
        // Slightly more than half rounds up to the smallest subnormal.
        assert_eq!(yuvarla(en_kucuk * 0.51), en_kucuk);
    }

    #[test]
    fn a_subnormal_can_carry_up_into_the_smallest_normal() {
        let hemen_alt = YARIM_EN_KUCUK_NORMAL * (1.0 - 2f32.powi(-12));
        assert_eq!(yuvarla(hemen_alt), YARIM_EN_KUCUK_NORMAL);
    }

    #[test]
    fn overflow_becomes_infinity_rather_than_a_plausible_number() {
        assert!(yuvarla(1.0e30).is_infinite());
        assert!(yuvarla(-1.0e30).is_sign_negative());
        assert_eq!(yuvarla(YARIM_EN_BUYUK), YARIM_EN_BUYUK);
        // Just over the last finite value rounds to infinity, not to it.
        assert!(yuvarla(YARIM_EN_BUYUK * 1.001).is_infinite());
    }

    #[test]
    fn nan_stays_nan_and_does_not_become_an_infinity() {
        let h = f32_to_yarim(f32::NAN);
        assert!(yarim_to_f32(h).is_nan());
        // A NaN whose low mantissa bits are all that is set would truncate to
        // an infinity under a naive shift.
        let sinsi = f32::from_bits(0x7f80_0001);
        assert!(yarim_to_f32(f32_to_yarim(sinsi)).is_nan());
    }

    #[test]
    fn every_half_decodes_and_re_encodes_to_itself() {
        // The decode direction is exact, so encode(decode(h)) == h for every
        // bit pattern that is not a NaN. This is the strongest round trip
        // available and it covers all 65536 patterns.
        let mut kontrol = 0u32;
        for h in 0u16..=0xffff {
            let x = yarim_to_f32(h);
            if x.is_nan() {
                continue;
            }
            assert_eq!(f32_to_yarim(x), h, "pattern {h:#06x} did not survive");
            kontrol += 1;
        }
        assert_eq!(kontrol, 65536 - 2046, "NaN pattern count changed");
    }
}
