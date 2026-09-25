#![forbid(unsafe_code)]
//! # kodlayici - a checkpoint read, checked and run from Rust
//!
//! This crate is the port: the checkpoint's own configuration, its weights read
//! from split part files, and the arithmetic a forward pass is made of. The
//! modules are split by what can be checked independently.
//!
//! | module | what it holds | how it is checked |
//! |---|---|---|
//! | [`baslik`] | safetensors header, part-aware byte reader, half-precision conversion | round trips, ranges, overlaps, a tensor that straddles a part boundary |
//! | [`yapilandirma`] | the checkpoint's description of itself, and the head's economics | refusals for a layer count, an activation and a head width that cannot be true |
//! | [`hesap`] | matrix-vector, layer norm, gelu, softmax, rope | each against its defining values, including the cases that produce `NaN` if written the obvious way |
//! | [`blok`] | the layer stack: attention with a window, the gated feed-forward, mean pooling | depth, determinism, the window's bound, and the gate/value split |
//! | [`karar`] | the decision head: type embedding, two bidirectional layers, the scorer, the action head, temperature | where the type embedding lands, which position is read, the four features, and that temperature cannot reorder the options |
//!
//! # What is not here yet
//!
//! The tokenizer that turns text into ids, and an independent cross-check of
//! the whole stack against a second implementation. Neither is claimed: a
//! checkpoint that can be read, encoded and scored from ids is not a checkpoint
//! that can be asked a question in words, and the distinction is why the
//! modules are separate in the first place.
//!
//! # Numbers
//!
//! Everything is `f32` once loaded. The checkpoint stores half precision, and
//! [`baslik::f16_to_f32`] converts exactly; the accumulation is done in `f32`
//! because a half-precision accumulator loses a different amount depending on
//! the order of the additions, which would make two runs of the same input
//! disagree.

pub mod baslik;
pub mod blok;
pub mod hesap;
pub mod karar;
pub mod yapilandirma;

pub use baslik::{BaslikHatasi, Dizin, ParcaliDosya, TensorBasligi};
pub use blok::{kodla, ortalama_havuz, Agirliklar, GommeKaynagi, KatmanAgirliklari, PencereKurali};
pub use hesap::{
    gelu, katman_norm, katman_norm_sapmali, matris_vektor, rope, rope_tablosu, softmax, SekilHatasi,
};
pub use karar::{
    puanla, tip_indeksi, Cevap, KararAgirliklari, KararYapisi, OZELLIK_SAYISI, TIPLER, TIP_SAYISI,
};
pub use yapilandirma::{KafaYapisi, KatmanTuru, KodlayiciYapisi};
