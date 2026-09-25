//! String obfuscation: constants that should not be readable in the binary.
//!
//! # What this does and does not buy
//!
//! A message compiled the ordinary way sits in the binary as plain bytes, and
//! `strings` prints it. That is not a weakness of Rust; it is what a string
//! literal is. This module stores such a constant **exclusive-ored with a
//! per-call key** and reconstructs it at run time, so the plain form never
//! appears in the file.
//!
//! It does not make the message secret. Anyone who runs the binary, or reads
//! [`coz`], can recover it; the key is in the same file. The point is to remove
//! the free version of the attack - `strings` followed by a search for
//! interesting words - and force the reader to understand the code instead.
//! Anything that must stay secret does not belong in a client binary at all.
//!
//! Both directions are crate-internal: the only public door is the
//! [`gizli_metin!`] macro. Handing out the decoder would hand out the one thing
//! this module exists to keep out of a plain read.

/// Why a hidden string could not be recovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CozHatasi {
    /// The bytes are not valid UTF-8 after decoding, which means the key or the
    /// buffer is wrong rather than merely unreadable.
    GecersizUtf8(String),
}

/// The masking step: byte `i` is exclusive-ored with the key advanced by `i`.
///
/// A single repeated key would leave repeated text with a repeating pattern in
/// the binary, which is how obfuscated blobs get spotted. Mixing the offset in
/// costs one multiply per byte and removes that pattern.
///
/// A string literal's bytes are a `&[u8]` whose length is not a const generic
/// argument at the call site, so the length comes from the annotated return
/// type instead. There is exactly one form of this function on purpose: two
/// forms under one name would be two places for the masking rule to drift.
///
/// # Panics
/// Compile-time evaluation fails if the slice is shorter than `N`, which can
/// only happen if the caller lies about the type. No runtime panic exists.
#[must_use]
pub(crate) const fn gizle_dilim<const N: usize>(duz: &[u8], anahtar: u8) -> [u8; N] {
    let mut cikti = [0_u8; N];
    let mut sira = 0;
    while sira < N {
        cikti[sira] = duz[sira] ^ anahtar.wrapping_add((sira as u8).wrapping_mul(31));
        sira += 1;
    }
    cikti
}

/// Recovers a string hidden by [`gizle_dilim`].
///
/// # Errors
/// [`CozHatasi::GecersizUtf8`] when the decoded bytes are not valid UTF-8.
pub(crate) fn coz(kod: &[u8], anahtar: u8) -> Result<String, CozHatasi> {
    let duz: Vec<u8> = kod
        .iter()
        .enumerate()
        .map(|(sira, bayt)| bayt ^ anahtar.wrapping_add(((sira % 256) as u8).wrapping_mul(31)))
        .collect();
    String::from_utf8(duz).map_err(|hata| CozHatasi::GecersizUtf8(hata.to_string()))
}

/// Hides a literal at rest and decodes it where it is used.
///
/// The macro is the only public door: `gizli_metin!("...", 0x5C)` yields a
/// `Result<String, CozHatasi>` that is already decoded, so the plain text never
/// exists as a compile-time constant.
macro_rules! gizli_metin {
    ($metin:literal, $anahtar:literal) => {{
        const KOD: [u8; $metin.len()] = $crate::metin::gizle_dilim($metin.as_bytes(), $anahtar);
        $crate::metin::coz(&KOD, $anahtar)
    }};
}

pub(crate) use gizli_metin;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_text_is_not_the_plain_text() {
        const DUZ: &[u8; 11] = b"kayit yoksa";
        let kod: [u8; 11] = gizle_dilim(DUZ, 0x5A);
        assert_ne!(&kod[..], &DUZ[..]);
        assert_eq!(coz(&kod, 0x5A).unwrap_or_default(), "kayit yoksa");
    }

    #[test]
    fn the_same_byte_does_not_produce_the_same_byte() {
        // `aaaa` must not become four equal bytes, or the pattern is visible.
        let kod: [u8; 4] = gizle_dilim(b"aaaa", 0x11);
        assert_ne!(kod[0], kod[1]);
        assert_ne!(kod[1], kod[2]);
    }

    #[test]
    fn the_macro_recovers_its_literal() {
        let metin = gizli_metin!("kalici hata", 0x3C).unwrap_or_default();
        assert_eq!(metin, "kalici hata");
    }

    #[test]
    fn a_wrong_key_does_not_pretend_to_work() {
        let kod: [u8; 5] = gizle_dilim(b"dogru", 0x11);
        // Either the bytes stop being UTF-8, or they decode to something else;
        // both are refusals to return the original.
        match coz(&kod, 0x12) {
            Ok(metin) => assert_ne!(metin, "dogru"),
            Err(CozHatasi::GecersizUtf8(_)) => {}
        }
    }

    #[test]
    fn an_empty_string_round_trips() {
        let kod: [u8; 0] = gizle_dilim(b"", 0x01);
        assert_eq!(coz(&kod, 0x01).unwrap_or_default(), "");
    }
}
