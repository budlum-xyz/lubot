//! The header reader takes bytes from outside this program.
//!
//! What is being looked for is not a wrong number: it is a panic, an unbounded
//! allocation, an arithmetic overflow in an offset computation, or a loop that
//! does not end. Each of those in a reader that parses somebody else's file is
//! a denial of service at best and an out-of-bounds read at worst, and the
//! second is why the reader is fuzzed rather than reasoned about.
//!
//! The target asserts only what must hold for *any* input: the call returns, and
//! whatever it returns is either a refusal or a directory whose tensor ranges
//! are internally consistent. A refusal is a perfectly good outcome.

#![no_main]

use libfuzzer_sys::fuzz_target;
use lubot_kodlayici::baslik::{Dizin, ParcaliDosya};

fuzz_target!(|veri: &[u8]| {
    let dosya = ParcaliDosya::bellekten(veri.to_vec());
    if let Ok(dizin) = Dizin::oku(&dosya) {
        // If it parsed, every range it reports must be inside the artifact and
        // ordered: a parser that returns ranges outside its own input has been
        // talked into an out-of-bounds read by a file.
        let boyut = dosya.boyut();
        for (ad, baslik) in dizin.adlar().iter().map(|ad| (ad, dizin.tensor(ad))) {
            let Some(baslik) = baslik else { continue };
            let (bas, son) = (baslik.data_offsets[0], baslik.data_offsets[1]);
            assert!(bas <= son, "{ad}: range is backwards");
            // `boyut()` is u64 and the offsets are usize; widen the offset
            // instead of narrowing the size, because a narrowing conversion
            // that panics inside a fuzz target would be a false denial of
            // service, not a finding.
            assert!(son as u64 <= boyut, "{ad}: range ends past the artifact");
        }
        // And a tensor that is not there must be a refusal rather than an
        // empty vector, which would be read as a real tensor of length zero.
        assert!(dizin.tensor_oku(&dosya, "fuzz-yok-boyle-tensor").is_err());
    }
});
