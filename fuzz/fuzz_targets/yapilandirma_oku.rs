//! The configuration reader, fed strings rather than files.
//!
//! A checkpoint's `config.json` is untrusted in exactly the same way its
//! weights are: it arrives with the artifact. The reader refuses what it cannot
//! use instead of filling in a default, and this target checks that the
//! refusing path is total - every byte string either parses or is refused, and
//! nothing in between.

#![no_main]

use libfuzzer_sys::fuzz_target;
use lubot_kodlayici::yapilandirma::KodlayiciYapisi;

fuzz_target!(|veri: &[u8]| {
    let Ok(metin) = std::str::from_utf8(veri) else {
        return;
    };
    if let Ok(yapi) = KodlayiciYapisi::metinden(metin) {
        // A configuration that parses must survive its own consistency check,
        // and the check must not be skipped by a value that makes it vacuous.
        assert!(yapi.num_hidden_layers < 100_000, "layer count is not bounded");
        assert!(yapi.hidden_size < 100_000, "hidden size is not bounded");
        assert!(
            yapi.layer_types.len() == yapi.num_hidden_layers
                || yapi.layer_types.is_empty(),
            "layer types and layer count disagree after a successful parse"
        );
    }
});
