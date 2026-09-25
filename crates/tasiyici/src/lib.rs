//! The weight container: a file Lubot reads in place, and the ladder that
//! decides how much of it a given machine holds.
//!
//! `lubot-nicem` answers "how few bits may a weight cost?". This crate answers
//! the two questions that follow from it, and they are the ones that decide
//! whether a large model runs on a small machine at all.
//!
//! # 1. Reading without unpacking
//!
//! A loader that parses a file into owned structures pays for the model twice:
//! once on disk and once in memory, with a peak somewhere above both while the
//! parse is in flight. On a phone that peak is the binding constraint, not the
//! model size. So the container here is laid out to be **mapped and used where
//! it lies**: a 64-byte header, a directory of fixed-meaning fields, then a
//! payload whose blocks are 64-byte aligned. [`Kapsayici::dilimler`] hands back
//! sub-slices of the caller's buffer, and the only heap allocation a load makes
//! is the directory itself, which is kilobytes.
//!
//! The digest is over the payload and is checked only when
//! [`Kapsayici::dogrula`] is called. Verifying on open would mean touching
//! every page of a file whose entire point is that most pages are never
//! touched, which would undo the design at the moment of loading. Opening
//! validates *structure* - that offsets are in range, blocks do not overlap,
//! and byte counts match the declared shapes - and that validation is
//! exhaustive, so a malformed file is refused without reading the payload.
//!
//! # 2. The ladder
//!
//! Every tensor is tagged with a rung. Rung `0` holds what every depth needs;
//! rungs `1..` are layers, in order. A reader that keeps rungs `0..=k` resident
//! and streams the rest has a working model at depth `k`. That makes a
//! machine's capacity expressible as one small integer, and gives the operator
//! rule in `crates/egitim` the term it has always been missing: a ceiling that
//! refers to the machine. See [`merdiven`] for what the ladder does and, more
//! importantly, for what it deliberately declines to decide.
//!
//! # What this crate refuses to do
//!
//! It does not convert streamed bytes into a predicted latency: that depends on
//! the storage, and a number invented here would be reported as if measured. It
//! does not decide that a shallower depth is acceptable: truncation produces a
//! different model and only the exam set may pass judgement on it. And it does
//! not report depth `0` when even rung zero will not fit - that is a refusal,
//! because a depth-zero reading looks like a working shallow model and is not
//! one.
//!
//! # Layout, in one place
//!
//! ```text
//! 0    magic "LUBOTNCM"        8 bytes
//! 8    version u32 le          1
//! 12   tensor count u32 le
//! 16   directory bytes u64 le
//! 24   payload bytes u64 le
//! 32   sha-256 of payload     32 bytes
//! 64   directory, padded to a 64-byte boundary
//! ...  payload, each block 64-byte aligned
//! ```

pub mod bicim;
pub mod merdiven;
pub mod yazici;

pub use bicim::{BicimHatasi, Kapsayici, Kayit, BASLIK_BAYT, HIZA, IMZA, SURUM};
pub use merdiven::{Kademe, Merdiven, MerdivenHatasi, Secim, Tavan};
pub use yazici::{Yazici, YaziciHatasi};

/// Fraction of measured usable memory a reader takes by default.
///
/// Two thirds, not all of it: the host process, the tokeniser and the scratch
/// buffer all need room, and a reader that claims everything it can see makes
/// the machine unusable for the thing that called it. Callers that know their
/// own budget should pass their own share to [`Tavan::olcumden`] rather than
/// take this one.
pub const VARSAYILAN_PAY: f64 = 0.66;

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn the_default_share_leaves_room_for_the_host_process() {
        // Stated as bytes on a plausible phone rather than as a bound on the
        // constant: what matters is that the reader leaves the host something
        // to run in, and a gigabyte of headroom out of three is that claim.
        let bellek = 3_000_000_000u64;
        let t = Tavan::olcumden(bellek, VARSAYILAN_PAY, "test");
        assert!(t.olculdu);
        assert!(t.yerlesik_bayt < bellek);
        assert!(bellek - t.yerlesik_bayt > 1_000_000_000);
    }

    #[test]
    fn the_header_is_the_size_the_layout_documents() {
        assert_eq!(BASLIK_BAYT, 64);
        assert_eq!(HIZA, 64);
        assert_eq!(IMZA, b"LUBOTNCM");
        assert_eq!(SURUM, 1);
    }
}
