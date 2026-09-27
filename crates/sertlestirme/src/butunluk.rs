//! The integrity layer: a digest of the running artifact.
//!
//! # What a digest proves and what it does not
//!
//! A SHA-256 over the file's own bytes proves that the bytes on disk are the
//! bytes that were measured. It does **not** prove that the process executing
//! is the process the file describes: an attacker who patches memory after
//! `exec` leaves the file untouched, and the digest stays correct. That is why
//! the digest is one leg of the layer and not the layer: it is checked against
//! a digest held elsewhere (the operator's record, the release notes), and it
//! is checked by comparing the *measured* value with the *expected* one rather
//! than by trusting either.
//!
//! # Why the hashing is streaming
//!
//! A binary is read block by block and fed to the hash as it arrives. Reading a
//! 70 MB debug binary into a single `Vec` to hash it would work and would also
//! double the process's peak memory for no reason; the streaming form is the
//! one that stays correct as the artifact grows.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// Block size for the streaming read; 64 KiB is the usual disk-friendly unit.
const BLOK: usize = 64 * 1024;

/// Why a digest could not be produced or did not match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OzetHatasi {
    /// The file could not be opened or read.
    Okunamadi(String),
    /// The measured digest is not the expected one.
    OzetUyusmadi {
        /// What was expected.
        beklenen: String,
        /// What was measured.
        olculen: String,
    },
}

/// The SHA-256 of a file, read in blocks.
///
/// # Errors
/// [`OzetHatasi::Okunamadi`] when the path cannot be read.
pub fn ozet(yol: &Path) -> Result<String, OzetHatasi> {
    let dosya = File::open(yol).map_err(|h| OzetHatasi::Okunamadi(h.to_string()))?;
    let mut okuyucu = BufReader::new(dosya);
    let mut seri = Sha256::new();
    let mut tampon = vec![0_u8; BLOK];
    loop {
        let okunan = okuyucu
            .read(&mut tampon)
            .map_err(|h| OzetHatasi::Okunamadi(h.to_string()))?;
        if okunan == 0 {
            break;
        }
        seri.update(&tampon[..okunan]);
    }
    Ok(format!("{:x}", seri.finalize()))
}

/// The digest of the running executable.
///
/// # Errors
/// [`OzetHatasi::Okunamadi`] when the executable's own path is unknown (the
/// process may have been started in a way that erases it) or unreadable.
pub fn kendi_ozeti() -> Result<String, OzetHatasi> {
    let yol =
        std::env::current_exe().map_err(|h| OzetHatasi::Okunamadi(format!("current_exe: {h}")))?;
    ozet(&yol)
}

/// Measures a file and compares the digest with the expected one.
///
/// The comparison is case-insensitive and whitespace-trimmed: a digest pasted
/// from a release note arrives with either, and refusing it would be a refusal
/// of a formatting difference rather than of a mismatch.
///
/// # Errors
/// [`OzetHatasi::OzetUyusmadi`] when the digests differ, or
/// [`OzetHatasi::Okunamadi`] when the file cannot be read.
pub fn dogrula(yol: &Path, beklenen: &str) -> Result<String, OzetHatasi> {
    let olculen = ozet(yol)?;
    if olculen == beklenen.trim().to_ascii_lowercase() {
        Ok(olculen)
    } else {
        Err(OzetHatasi::OzetUyusmadi {
            beklenen: beklenen.trim().to_string(),
            olculen,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn gecici(ad: &str, icerik: &[u8]) -> std::path::PathBuf {
        let yol = std::env::temp_dir().join(format!("lubot-sertlestirme-{ad}"));
        let mut dosya = File::create(&yol).expect("gecici dosya olusmali");
        dosya.write_all(icerik).expect("yazilmali");
        yol
    }

    #[test]
    fn the_digest_matches_an_independent_implementation() {
        // The empty string's SHA-256 is a published constant: checking it pins
        // the hash function, not our reading of it.
        let yol = gecici("bos", b"");
        assert_eq!(
            ozet(&yol).unwrap_or_default(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let _ = std::fs::remove_file(&yol);
    }

    #[test]
    fn one_byte_of_difference_changes_the_digest() {
        let a = gecici("a", b"lubot");
        let b = gecici("b", b"luboT");
        let oa = ozet(&a).unwrap_or_default();
        let ob = ozet(&b).unwrap_or_default();
        assert_ne!(oa, ob);
        assert_eq!(oa.len(), 64);
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }

    #[test]
    fn verification_accepts_the_measured_digest_and_refuses_another() {
        let yol = gecici("c", b"lubot sertlestirme");
        let olculen = ozet(&yol).unwrap_or_default();
        assert_eq!(dogrula(&yol, &olculen).unwrap_or_default(), olculen);
        // Case and surrounding whitespace are formatting, not a mismatch.
        assert!(dogrula(&yol, &format!("  {}  ", olculen.to_uppercase())).is_ok());
        match dogrula(&yol, &"0".repeat(64)) {
            Err(OzetHatasi::OzetUyusmadi { beklenen, olculen }) => {
                assert_eq!(beklenen, "0".repeat(64));
                assert_eq!(olculen, olculen);
            }
            digeri => panic!("uyusmazlik bekleniyordu: {digeri:?}"),
        }
        let _ = std::fs::remove_file(&yol);
    }

    #[test]
    fn a_missing_file_is_an_error_and_not_an_empty_digest() {
        let yol = std::env::temp_dir().join("lubot-sertlestirme-yok");
        let _ = std::fs::remove_file(&yol);
        assert!(matches!(ozet(&yol), Err(OzetHatasi::Okunamadi(_))));
    }
}
