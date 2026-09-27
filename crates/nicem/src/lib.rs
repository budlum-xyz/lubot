//! # lubot-nicem - sub-byte weights, written here
//!
//! A reader that only runs on a machine with a data-centre accelerator is not a
//! reader an operator owns. `crates/egitim` already says where the ceiling is
//! ("the operator answers with the machine it actually has"), and this crate is
//! the arithmetic that moves that ceiling without moving the machine: it
//! stores a parameter in about two bits instead of thirty-two, and it does so
//! in a layout a reader can map and consume one group at a time.
//!
//! ## The four steps
//!
//! | step | module | what it is for |
//! |---|---|---|
//! | rotate | [`hadamard`] | spread outliers so a four-level alphabet is not spent on one weight |
//! | solve | [`kodkitabi`] | Lloyd-Max levels for the distribution the rotation produces |
//! | scale | [`yarim`] | one `f16` norm per group, rounded to nearest with ties to even |
//! | pack | [`paket`] | indices at exactly their width; base-3 for the ternary alphabet |
//!
//! [`grup`] assembles them and measures the result.
//!
//! ## What this crate is not
//!
//! It is **not a model**, not a runtime and not a checkpoint format. It takes
//! `&[f32]` and gives back bytes. The container that holds those bytes, the
//! tensor directory, the depth ladder and the residency decision are
//! `crates/tasiyici`'s job, and the ceiling that decides which depth an
//! operator may serve is `crates/cihaz`'s. Keeping the arithmetic separate from
//! the file means the arithmetic can be tested against closed-form answers -
//! which is what [`kodkitabi`]'s tests do - instead of against a file someone
//! wrote earlier.
//!
//! It also imports nothing. K1 puts the model, the training core and the
//! inference path on a from-scratch footing; a quantisation format is part of
//! the model's definition, so the transform, the codebook solve, the half
//! precision conversion and the bit packing are all written out here. The cost
//! is about two thousand lines. The benefit is that every number this crate
//! reports can be traced to a line in it.
//!
//! ## What is measured and what is refused
//!
//! Measured: reconstruction error, signal-to-noise ratio, worst single-weight
//! deviation, bits per weight, compression ratio, and the codebook's own
//! predicted distortion to compare them against
//! ([`grup::Olcum`]).
//!
//! Refused, loudly, rather than approximated: a group size that is not a power
//! of two, a group too small to amortise its scale, a non-finite weight, an
//! index that does not fit its declared width, a payload shorter than the shape
//! needs. Each refusal names the offending value. The alternative - clamp,
//! mask, zero-fill - produces a model that loads and is quietly wrong, and a
//! quietly wrong model is the one failure mode this repository has no test for
//! because it looks exactly like success.
//!
//! ## Not claimed
//!
//! **Not measured here:** what quantisation does to answer quality. Bits per
//! weight is a storage fact; citation accuracy and refusal discipline are the
//! exam set's business (`training/sinav.py`) and no number in this crate speaks
//! to them. A compression ratio is not an evaluation result, and this crate
//! does not let one be reported as though it were.

pub mod grup;
pub mod hadamard;
pub mod kodkitabi;
pub mod paket;
pub mod yarim;

pub use grup::{Genislik, NicemHatasi, Nicemlenmis, Nicemleyici, Olcum};

/// The group size the container defaults to.
///
/// 128 is a compromise with both ends visible: the `f16` scale costs
/// `16/128 = 0.125` bits per weight, which is small, and the rotation is a
/// seven-pass butterfly, which is cheap. A larger group makes the scale cheaper
/// still but asks one norm to serve more weights, and the measured error starts
/// to rise; a smaller group is the opposite trade. It is a default, not a
/// constant of nature, and every entry point takes the group size as a
/// parameter so the choice can be re-measured rather than inherited.
pub const VARSAYILAN_GRUP: usize = 128;

/// Bits per weight at the default configuration: two-bit indices plus an `f16`
/// scale per group of 128.
pub const VARSAYILAN_BIT: f64 = 2.125;

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
    fn the_advertised_default_is_the_arithmetic_of_the_defaults() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), VARSAYILAN_GRUP).expect("valid");
        assert!((n.genislik().agirlik_basina_bit(VARSAYILAN_GRUP) - VARSAYILAN_BIT).abs() < 1e-12);
    }

    #[test]
    fn a_whole_tensor_goes_through_the_public_surface_and_comes_back() {
        let mut durum = 7u32;
        let w: Vec<f32> = (0..VARSAYILAN_GRUP * 24)
            .map(|_| {
                durum = durum.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                #[allow(clippy::cast_precision_loss)]
                let v = ((durum >> 16) as f32) / 32_768.0 - 1.0;
                v * 0.05
            })
            .collect();
        let n = Nicemleyici::yeni(Genislik::Bit(2), VARSAYILAN_GRUP).expect("valid");
        let q = n.nicemle(&w, VARSAYILAN_GRUP).expect("quantises");
        let o = q.olc(&w).expect("measures");
        assert!(o.oran > 14.0, "compression ratio {} is too low", o.oran);
        assert!(o.bagil_hata < 0.35, "relative error {}", o.bagil_hata);
        assert!(o.en_buyuk_sapma.is_finite());
        assert_eq!(q.en_buyuk_calisma_alani(), VARSAYILAN_GRUP);
    }
}
