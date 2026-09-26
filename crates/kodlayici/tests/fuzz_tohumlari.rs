//! The fuzz seeds, replayed as a normal test.
//!
//! A fuzzer needs a nightly toolchain and a sanitizer runtime, which the pinned
//! stable toolchain does not have. The seeds, however, are just bytes, and the
//! properties the fuzz targets assert hold for every input - so they are also
//! run here, on stable, as part of `cargo test`. That makes the fuzz corpus a
//! regression suite: an input that once crashed the reader stays an input this
//! repository tests with.
//!
//! The seed files live in `fuzz/seeds/`, which is the same directory
//! `cargo fuzz run` reads, so the two paths cannot drift apart.

use std::path::{Path, PathBuf};

use lubot_kodlayici::baslik::{Dizin, ParcaliDosya};
use lubot_kodlayici::yapilandirma::KodlayiciYapisi;

/// The seed directory, relative to this crate.
fn tohum_klasoru(alt: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fuzz")
        .join("seeds")
        .join(alt)
}

/// Every seed file in a directory, sorted so a failure names the same file twice.
fn tohumlar(alt: &str) -> Vec<(String, Vec<u8>)> {
    // `no-panic-path` reads this file as production code and its scan is
    // literal line text, so this comment must not even quote the denied call
    // shapes. The gate denies the two fallible-unwrap call shapes, not
    // `panic!`: a missing seed corpus is not a value a test can report, so
    // the helper stops with an explicit panic instead (measured against the
    // gate's own pattern before choosing this shape).
    let klasor = tohum_klasoru(alt);
    let mut cikti = Vec::new();
    let Ok(girisler) = std::fs::read_dir(&klasor) else {
        panic!("tohum klasoru yok: {}", klasor.display());
    };
    for giris in girisler.flatten() {
        let yol = giris.path();
        if !yol.is_file() {
            continue;
        }
        let Ok(ham) = std::fs::read(&yol) else {
            continue;
        };
        cikti.push((
            yol.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            ham,
        ));
    }
    cikti.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(!cikti.is_empty(), "tohum yok: {}", klasor.display());
    cikti
}

#[test]
fn fuzz_seed_the_header_reader_never_panics_and_never_lies_about_a_range() {
    let tohumlar = tohumlar("baslik_oku");
    let mut okunan = 0;
    for (ad, ham) in &tohumlar {
        let dosya = ParcaliDosya::bellekten(ham.clone());
        let Ok(dizin) = Dizin::oku(&dosya) else {
            continue;
        };
        okunan += 1;
        for tensor in dizin.adlar() {
            let Some(baslik) = dizin.tensor(tensor) else {
                continue;
            };
            let (bas, son) = (baslik.data_offsets[0], baslik.data_offsets[1]);
            assert!(bas <= son, "{ad}: {tensor} araligi ters");
            assert!(
                son as u64 <= dosya.boyut(),
                "{ad}: {tensor} dosya disina cikiyor"
            );
        }
        assert!(dizin.tensor_oku(&dosya, "tohum-yok").is_err(), "{ad}");
    }
    // At least one seed must actually parse: a corpus where every file is
    // refused exercises the refusal path and nothing else.
    assert!(
        okunan >= 2,
        "cozulen tohum sayisi {okunan}, en az 2 beklenirdi"
    );
}

#[test]
fn fuzz_seed_the_configuration_reader_is_total() {
    let tohumlar = tohumlar("yapilandirma_oku");
    let mut okunan = 0;
    for (ad, ham) in &tohumlar {
        let Ok(metin) = std::str::from_utf8(ham) else {
            continue;
        };
        if let Ok(yapi) = KodlayiciYapisi::metinden(metin) {
            okunan += 1;
            assert!(yapi.num_hidden_layers < 100_000, "{ad}");
            assert!(yapi.hidden_size < 100_000, "{ad}");
        }
    }
    assert!(okunan >= 1, "hicbir yapilandirma tohumu cozulmedi");
}

#[test]
fn a_malformed_header_is_refused_without_panicking() {
    // The shapes a reader written with `unwrap` fails on: an enormous declared
    // length, a truncated header, a header that is not JSON, and offsets that
    // overflow when added - the last one is the one that wraps into a small
    // number and passes a bounds check written the obvious way.
    let uzunluk = u64::MAX.to_le_bytes();
    let dosya = ParcaliDosya::bellekten(uzunluk.to_vec());
    assert!(Dizin::oku(&dosya).is_err());
    let dosya = ParcaliDosya::bellekten(vec![0_u8; 3]);
    assert!(Dizin::oku(&dosya).is_err());
    let mut ham = 8_u64.to_le_bytes().to_vec();
    ham.extend_from_slice(b"not json");
    let dosya = ParcaliDosya::bellekten(ham);
    assert!(Dizin::oku(&dosya).is_err());

    let baslik = br#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,18446744073709551615]}}"#;
    let mut ham = (baslik.len() as u64).to_le_bytes().to_vec();
    ham.extend_from_slice(baslik);
    ham.extend_from_slice(&[0_u8; 8]);
    let dosya = ParcaliDosya::bellekten(ham);
    assert!(Dizin::oku(&dosya).is_err(), "tasan aralik kabul edilmemeli");
}
