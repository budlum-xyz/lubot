//! Packing: turning a vector of small indices into bytes, and back, with no
//! byte wasted and no ambiguity about which end a bit came from.
//!
//! # Two packings, because two alphabets
//!
//! A power-of-two alphabet packs by bit shifting: `b` bits per index, indices
//! laid down least-significant-bit first, crossing byte boundaries freely. That
//! is [`paketle`] and it is exact - `n` indices of `b` bits take
//! `ceil(n*b / 8)` bytes and no more.
//!
//! A three-level alphabet does not. Two bits per trit wastes a quarter of the
//! space: `log2(3) = 1.585`, so bit packing gives up 26% of the saving the
//! ternary codebook bought. [`paketle_ucdeger`] packs base-3 instead - five
//! trits per byte, since `3^5 = 243 <= 256` - for `8/5 = 1.6` bits per trit.
//! The remaining `0.015` bits above the entropy is the cost of not doing
//! arithmetic coding, and arithmetic coding is not something a device should
//! have to run to read a weight.
//!
//! # Why least-significant-bit first
//!
//! Either order works as long as both ends agree. LSB-first is chosen because
//! the decode is then a shift and a mask with no dependence on the total
//! length: a reader can start at any byte boundary that happens to align and
//! does not need to know how many indices follow. MSB-first would require the
//! total count to compute the shift of the final partial byte, which is one
//! more thing the container has to get right.
//!
//! # What is deliberately not here
//!
//! There is no compression. Packed indices are not run-length encoded, not
//! entropy coded, not delta coded. The container is meant to be *mapped and
//! read in place* - a reader dequantises one group at a time straight out of
//! the mapped bytes - and every one of those schemes makes a byte offset
//! depend on the data before it, which turns a map into a decompression pass
//! with a buffer the size of the model. That trade is the whole reason a large
//! model can live on a small device here, so it is not available as an option.

use std::fmt;

/// Why packing or unpacking was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaketHatasi {
    /// A bit width outside `1..=8`.
    BitAraligi { bit: u8 },
    /// An index that does not fit the declared width. Silently masking it would
    /// store a different weight than the one the encoder chose.
    IndeksAsimi { indeks: u8, bit: u8 },
    /// A trit outside `0..=2`.
    UcdegerAsimi { indeks: u8 },
    /// The byte buffer is shorter than the declared count needs.
    Kisa { gereken: usize, var: usize },
}

impl fmt::Display for PaketHatasi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitAraligi { bit } => write!(f, "bit genisligi {bit} 1..=8 disinda"),
            Self::IndeksAsimi { indeks, bit } => {
                write!(f, "indeks {indeks} {bit} bite sigmaz")
            }
            Self::UcdegerAsimi { indeks } => write!(f, "ucdeger {indeks} 0..=2 disinda"),
            Self::Kisa { gereken, var } => {
                write!(f, "paket kisa: {gereken} bayt gerekli, {var} bayt var")
            }
        }
    }
}

impl std::error::Error for PaketHatasi {}

/// Bytes needed for `adet` indices at `bit` bits each.
#[must_use]
pub fn bayt_sayisi(adet: usize, bit: u8) -> usize {
    (adet * bit as usize).div_ceil(8)
}

/// Bytes needed for `adet` trits at five per byte.
#[must_use]
pub fn ucdeger_bayt_sayisi(adet: usize) -> usize {
    adet.div_ceil(TRIT_BASINA_BAYT)
}

/// Trits per byte: `3^5 = 243` fits in a byte, `3^6 = 729` does not.
pub const TRIT_BASINA_BAYT: usize = 5;

/// Pack indices, least-significant-bit first.
///
/// # Errors
///
/// [`PaketHatasi::BitAraligi`] for a width outside `1..=8`, and
/// [`PaketHatasi::IndeksAsimi`] for an index that does not fit the width. The
/// second is a refusal rather than a mask because a masked index decodes to a
/// different, entirely plausible weight.
///
/// The narrowing casts are bounded by the width check above: `yerlesecek` is at
/// most eight and `deger` has already been refused if it does not fit `bit`.
#[allow(clippy::cast_possible_truncation)]
pub fn paketle(indeksler: &[u8], bit: u8) -> Result<Vec<u8>, PaketHatasi> {
    if bit == 0 || bit > 8 {
        return Err(PaketHatasi::BitAraligi { bit });
    }
    let tavan: u16 = 1u16 << bit;
    let mut cikti = vec![0u8; bayt_sayisi(indeksler.len(), bit)];
    let mut bit_konumu = 0usize;
    for indeks in indeksler {
        if u16::from(*indeks) >= tavan {
            return Err(PaketHatasi::IndeksAsimi {
                indeks: *indeks,
                bit,
            });
        }
        let mut deger = u32::from(*indeks);
        let mut kalan = bit;
        let mut konum = bit_konumu;
        while kalan > 0 {
            let bayt = konum / 8;
            let ofset = (konum % 8) as u8;
            let yerlesecek = (8 - ofset).min(kalan);
            let maske = ((1u32 << yerlesecek) - 1) as u8;
            cikti[bayt] |= ((deger as u8) & maske) << ofset;
            deger >>= yerlesecek;
            kalan -= yerlesecek;
            konum += yerlesecek as usize;
        }
        bit_konumu += bit as usize;
    }
    Ok(cikti)
}

/// Unpack `adet` indices of `bit` bits from `bayt`.
///
/// # Errors
///
/// [`PaketHatasi::BitAraligi`] for a width outside `1..=8` and
/// [`PaketHatasi::Kisa`] if the buffer cannot hold that many indices. A short
/// buffer is refused rather than zero-filled: a tensor that decodes to zeros
/// past the truncation point still runs, and produces a model that is quietly
/// half there.
#[allow(clippy::cast_possible_truncation)]
pub fn coz(bayt: &[u8], bit: u8, adet: usize) -> Result<Vec<u8>, PaketHatasi> {
    if bit == 0 || bit > 8 {
        return Err(PaketHatasi::BitAraligi { bit });
    }
    let gereken = bayt_sayisi(adet, bit);
    if bayt.len() < gereken {
        return Err(PaketHatasi::Kisa {
            gereken,
            var: bayt.len(),
        });
    }
    let mut cikti = Vec::with_capacity(adet);
    let mut bit_konumu = 0usize;
    for _ in 0..adet {
        let mut deger = 0u32;
        let mut alindi = 0u8;
        let mut konum = bit_konumu;
        while alindi < bit {
            let indeks = konum / 8;
            let ofset = (konum % 8) as u8;
            let alinacak = (8 - ofset).min(bit - alindi);
            let maske = ((1u16 << alinacak) - 1) as u8;
            let parca = (bayt[indeks] >> ofset) & maske;
            deger |= u32::from(parca) << alindi;
            alindi += alinacak;
            konum += alinacak as usize;
        }
        #[allow(clippy::cast_possible_truncation)]
        cikti.push(deger as u8);
        bit_konumu += bit as usize;
    }
    Ok(cikti)
}

/// Pack trits base-3, five to a byte.
///
/// # Errors
///
/// [`PaketHatasi::UcdegerAsimi`] for a value outside `0..=2`.
///
/// The base-3 accumulator tops out at `3^5 - 1 = 242`, so the cast to a byte is
/// exact; the trit range is refused above rather than masked.
#[allow(clippy::cast_possible_truncation)]
pub fn paketle_ucdeger(trits: &[u8]) -> Result<Vec<u8>, PaketHatasi> {
    let mut cikti = Vec::with_capacity(ucdeger_bayt_sayisi(trits.len()));
    for blok in trits.chunks(TRIT_BASINA_BAYT) {
        let mut deger = 0u16;
        // Little-endian in base 3: the first trit of the block is the least
        // significant digit, matching the bit packing's LSB-first rule so the
        // two formats do not disagree about direction.
        let mut carpan = 1u16;
        for t in blok {
            if *t > 2 {
                return Err(PaketHatasi::UcdegerAsimi { indeks: *t });
            }
            deger += u16::from(*t) * carpan;
            carpan *= 3;
        }
        #[allow(clippy::cast_possible_truncation)]
        cikti.push(deger as u8);
    }
    Ok(cikti)
}

/// Unpack `adet` trits from base-3 bytes.
///
/// # Errors
///
/// [`PaketHatasi::Kisa`] if the buffer is shorter than `adet` trits need.
pub fn coz_ucdeger(bayt: &[u8], adet: usize) -> Result<Vec<u8>, PaketHatasi> {
    let gereken = ucdeger_bayt_sayisi(adet);
    if bayt.len() < gereken {
        return Err(PaketHatasi::Kisa {
            gereken,
            var: bayt.len(),
        });
    }
    let mut cikti = Vec::with_capacity(adet);
    'dis: for b in bayt.iter().take(gereken) {
        let mut deger = *b;
        for _ in 0..TRIT_BASINA_BAYT {
            cikti.push(deger % 3);
            deger /= 3;
            if cikti.len() == adet {
                break 'dis;
            }
        }
    }
    Ok(cikti)
}

/// Bits per weight actually spent, including the one `f16` scale per group.
///
/// This is the number the container reports and the one an operator budgets
/// against. At two bits and a group of 128 it is `2 + 16/128 = 2.125`; at five
/// trits per byte and the same group it is `1.6 + 0.125 = 1.725`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn agirlik_basina_bit(bit: u8, grup: usize) -> f64 {
    f64::from(bit) + 16.0 / (grup as f64)
}

/// The same figure for the ternary packing.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn agirlik_basina_bit_ucdeger(grup: usize) -> f64 {
    8.0 / (TRIT_BASINA_BAYT as f64) + 16.0 / (grup as f64)
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

    fn deterministik_indeksler(n: usize, tavan: u8) -> Vec<u8> {
        // A fixed mixing recurrence, so the case set is the same on every
        // machine and a failure is reproducible from the test name alone.
        let mut durum = 0x2026_0925u32;
        (0..n)
            .map(|_| {
                durum = durum.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_possible_truncation)]
                let v = (durum >> 16) as u8;
                v % tavan
            })
            .collect()
    }

    #[test]
    fn every_width_round_trips_over_a_thousand_indices() {
        for bit in 1..=8u8 {
            let tavan = if bit == 8 { 255u16 } else { 1u16 << bit };
            #[allow(clippy::cast_possible_truncation)]
            let tavan = tavan as u8;
            let indeksler = deterministik_indeksler(1000, tavan.max(1));
            let paket = paketle(&indeksler, bit).expect("in range");
            assert_eq!(paket.len(), bayt_sayisi(1000, bit));
            let geri = coz(&paket, bit, indeksler.len()).expect("in range");
            assert_eq!(geri, indeksler, "bit {bit} did not survive");
        }
    }

    #[test]
    fn the_packing_is_exactly_as_tight_as_the_arithmetic_says() {
        // Two bits over a group of 128 must be 32 bytes, not 33: an off-by-one
        // here costs half a percent of the whole file.
        assert_eq!(bayt_sayisi(128, 2), 32);
        assert_eq!(bayt_sayisi(128, 3), 48);
        assert_eq!(bayt_sayisi(1, 2), 1);
        assert_eq!(bayt_sayisi(0, 2), 0);
        assert_eq!(ucdeger_bayt_sayisi(128), 26);
    }

    #[test]
    fn indices_cross_byte_boundaries_in_the_documented_direction() {
        // Three bits: the first index occupies bits 0..3 of byte 0, the second
        // 3..6, the third straddles bytes 0 and 1. Pinning the exact bytes
        // means a change of direction is caught here and not by a device that
        // reads a model written last month.
        let paket = paketle(&[0b101, 0b011, 0b110], 3).expect("three bits");
        // byte 0: 101 in bits 0..3, 011 in bits 3..6, the low two bits of 110
        // in bits 6..8. byte 1: the remaining high bit of 110.
        assert_eq!(paket[0], 0b1001_1101, "{:08b}", paket[0]);
        assert_eq!(paket[1], 0b0000_0001, "{:08b}", paket[1]);
    }

    #[test]
    fn an_index_that_does_not_fit_is_refused_not_masked() {
        assert_eq!(
            paketle(&[4], 2),
            Err(PaketHatasi::IndeksAsimi { indeks: 4, bit: 2 })
        );
        assert_eq!(
            paketle(&[4], 2).unwrap_err().to_string(),
            "indeks 4 2 bite sigmaz"
        );
        // Masking would have stored index 0, which decodes to a real weight.
        assert_eq!(paketle(&[3], 2).expect("fits"), vec![0b11]);
    }

    #[test]
    fn a_short_buffer_is_refused_rather_than_zero_filled() {
        let paket = paketle(&[1, 2, 3], 2).expect("two bits");
        assert_eq!(
            coz(&paket, 2, 100),
            Err(PaketHatasi::Kisa {
                gereken: 25,
                var: 1
            })
        );
    }

    #[test]
    fn out_of_range_widths_are_refused_on_both_directions() {
        assert_eq!(paketle(&[0], 0), Err(PaketHatasi::BitAraligi { bit: 0 }));
        assert_eq!(paketle(&[0], 9), Err(PaketHatasi::BitAraligi { bit: 9 }));
        assert_eq!(coz(&[0], 0, 1), Err(PaketHatasi::BitAraligi { bit: 0 }));
        assert_eq!(coz(&[0], 12, 1), Err(PaketHatasi::BitAraligi { bit: 12 }));
    }

    #[test]
    fn ternary_packs_five_to_a_byte_and_round_trips() {
        let trits = deterministik_indeksler(997, 3);
        let paket = paketle_ucdeger(&trits).expect("valid trits");
        assert_eq!(paket.len(), 200);
        let geri = coz_ucdeger(&paket, trits.len()).expect("long enough");
        assert_eq!(geri, trits);
    }

    #[test]
    fn ternary_beats_two_bit_packing_which_is_the_only_reason_it_exists() {
        let n = 10_000usize;
        let iki_bit = bayt_sayisi(n, 2);
        let ucdeger = ucdeger_bayt_sayisi(n);
        assert!(ucdeger < iki_bit, "{ucdeger} vs {iki_bit}");
        // Measured: 2000 bytes against 2500, a fifth of the payload.
        assert_eq!(iki_bit, 2500);
        assert_eq!(ucdeger, 2000);
    }

    #[test]
    fn a_trit_above_two_is_refused() {
        assert_eq!(
            paketle_ucdeger(&[0, 1, 3]),
            Err(PaketHatasi::UcdegerAsimi { indeks: 3 })
        );
    }

    #[test]
    fn the_last_partial_block_does_not_leak_phantom_trits() {
        // 7 trits is one full block plus two; the decode must stop at seven and
        // not return the three zero digits that pad the second byte.
        let trits = vec![2, 1, 0, 2, 1, 1, 2];
        let paket = paketle_ucdeger(&trits).expect("valid");
        assert_eq!(paket.len(), 2);
        assert_eq!(coz_ucdeger(&paket, 7).expect("long enough"), trits);
    }

    #[test]
    fn the_advertised_bit_budget_is_the_arithmetic_and_not_a_slogan() {
        assert!((agirlik_basina_bit(2, 128) - 2.125).abs() < 1e-12);
        assert!((agirlik_basina_bit(4, 128) - 4.125).abs() < 1e-12);
        assert!((agirlik_basina_bit(2, 64) - 2.25).abs() < 1e-12);
        assert!((agirlik_basina_bit_ucdeger(128) - 1.725).abs() < 1e-12);
    }

    #[test]
    fn an_empty_input_packs_to_nothing_and_decodes_to_nothing() {
        assert!(paketle(&[], 2).expect("empty").is_empty());
        assert!(coz(&[], 2, 0).expect("empty").is_empty());
        assert!(paketle_ucdeger(&[]).expect("empty").is_empty());
        assert!(coz_ucdeger(&[], 0).expect("empty").is_empty());
    }
}
