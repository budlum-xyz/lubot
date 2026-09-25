#![forbid(unsafe_code)]
//! # sertlestirme - making the shipped binary expensive to read
//!
//! # The honest position
//!
//! Every binary can be understood by someone with enough time. The layers here
//! do not aim at "impossible"; they aim at removing the *cheap* attacks, so that
//! whatever is learned costs real effort:
//!
//! | layer | what it removes | where |
//! |---|---|---|
//! | compiler | symbol names, unwind tables, a clean function-per-file layout | workspace `[profile.release]` |
//! | strings | `strings` followed by a search for interesting words | [`metin`] |
//! | control flow | nothing, this build does not obfuscate it - see below | - |
//! | anti-debug | a debugger nobody notices, a machine nobody described | [`izler`], [`kapi`] |
//! | integrity | a patched binary that looks like the tested one | [`butunluk`] |
//! | secrets | every answer a client binary could leak | *not here* |
//!
//! **Control flow** is not flattened in this build. The honest reason: the LLVM
//! passes that do it are not part of a stock toolchain, and pulling a
//! third-party pass into the release path of a program that has to build with a
//! plain `cargo build` would trade a measurable property for an unmeasurable
//! one. What is done instead: parsing and scoring are split so that no single
//! function holds a whole algorithm, and the decision gate is one place.
//!
//! **Packing** is not done either. Section encryption needs its own loader, and
//! a loader that decrypts at start-up is a loader whose key is in the same file.
//! It buys time against a static reader and nothing against a dynamic one, and
//! saying so is more useful than shipping it and calling it hardening.
//!
//! **Secrets** are not in the client. A key, a policy that must not be read, or
//! an algorithm whose value depends on staying hidden does not belong in a
//! binary a reader can run. That is a design constraint, not a hardening step,
//! and this crate cannot enforce it: it can only be stated.
//!
//! # What is measured, not claimed
//!
//! The report from [`kapi::topla`] and the external measurements (`strings`,
//! `nm`, `objdump`, `readelf`, digest verification) are recorded per build in
//! `lubot-yerel/SERTLESTIRME-OLCUM.md`. A hardening claim that was never
//! measured is a claim; this crate exists to keep the two apart.


pub mod butunluk;
pub mod izler;
pub mod kapi;
pub mod metin;

pub use izler::{Bulgu, Katman};
pub use kapi::{Kip, Rapor};
