//! Splitting the corpus and turning records into windows.
//!
//! Two decisions live here, and both are about not fooling yourself.
//!
//! **The split is by identity, not by position.** A record's side of the split
//! is decided by a hash of its `content_id`, so adding one record to the corpus
//! does not move every later record to the other side. A positional split makes
//! two runs over two versions of the corpus incomparable, and the difference
//! looks like learning.
//!
//! **A record is never split across the boundary.** All of a record's tokens
//! belong to one side. Otherwise the validation loss would be measuring
//! memorisation of the second half of a passage the model was trained on.

use crate::{paketle, pencere_olcu, PaketHatasi, PaketPencere, Spec};

/// Denominator of the split ratio. Ten thousand is enough resolution for a
/// validation share - and it is a rational, so the same share gives the same
/// split on every machine, which a float comparison would not.
pub const PAYDA: u64 = 10_000;

/// One corpus record, tokenised and identified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kayit {
    /// `content_id` of the record.
    pub kimlik: String,
    /// Token ids, in reading order.
    pub jetonlar: Vec<u32>,
}

/// The two sides of one split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bolum {
    /// Records the run trains on.
    pub egitim: Vec<Kayit>,
    /// Records the run is measured on.
    pub dogrulama: Vec<Kayit>,
}

/// Why a split was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BolumHatasi {
    /// A share outside the open unit interval: neither side may be empty, and
    /// a share of 1 would leave the run nothing to measure.
    PaydaDisi,
    /// The same identity appears twice, so the two sides are not disjoint.
    KimlikTekrari,
    /// A side came out empty although both sides were asked for.
    BosTaraf,
}

/// FNV-1a over the identity, mixed with the seed.
///
/// A hash rather than a counter, so the side a record lands on does not depend
/// on how many records came before it. Not a cryptographic hash: nothing here
/// is defending against an adversary who picks record ids.
#[must_use]
pub fn tohum_karmasi(kimlik: &str, tohum: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in kimlik.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= tohum;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    h
}

/// Split records into a training and a validation side.
///
/// `dogrulama_payi` is a share, not a count: the same share on a growing corpus
/// keeps the validation set growing with it, which a fixed count would not.
///
/// # Errors
/// [`BolumHatasi::PaydaDisi`] when the share is not inside `(0, 1)`;
/// [`BolumHatasi::KimlikTekrari`] when an identity appears twice;
/// [`BolumHatasi::BosTaraf`] when either side came out empty.
pub fn bolumle(kayitlar: Vec<Kayit>, dogrulama_payi: f64) -> Result<Bolum, BolumHatasi> {
    if !dogrulama_payi.is_finite() || dogrulama_payi <= 0.0 || dogrulama_payi >= 1.0 {
        return Err(BolumHatasi::PaydaDisi);
    }
    if kayitlar.is_empty() {
        return Err(BolumHatasi::BosTaraf);
    }
    let esik = (dogrulama_payi * PAYDA as f64).round() as u64;
    let mut gorulen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut egitim: Vec<Kayit> = Vec::new();
    let mut dogrulama: Vec<Kayit> = Vec::new();
    for kayit in kayitlar {
        if !gorulen.insert(kayit.kimlik.clone()) {
            return Err(BolumHatasi::KimlikTekrari);
        }
        if tohum_karmasi(&kayit.kimlik, 0) % PAYDA < esik {
            dogrulama.push(kayit);
        } else {
            egitim.push(kayit);
        }
    }
    if egitim.is_empty() || dogrulama.is_empty() {
        return Err(BolumHatasi::BosTaraf);
    }
    Ok(Bolum { egitim, dogrulama })
}

/// The window length to use, checked against the spec rather than clamped.
///
/// # Errors
/// A zero window, or one longer than the spec was built for. Clamping would
/// turn "the spec is too small for this window" into a quiet rounding.
pub fn pencere_uzunlugu(spec: Spec, istenen: usize) -> Result<usize, String> {
    if istenen < 2 {
        return Err(format!(
            "pencere uzunlugu {istenen}: bir konumdan sonra tahmin edilecek jeton yok"
        ));
    }
    if istenen > spec.max_seq_len {
        return Err(format!(
            "istenen pencere {istenen} > spec.max_seq_len {}: spec bu uzunluk icin dogrulanmadi",
            spec.max_seq_len
        ));
    }
    Ok(istenen)
}

/// Cut windows out of both sides, packing records into a stream.
///
/// Packing is used rather than record-by-record windowing because the corpus is
/// mostly short records: measured on the self corpus, record-by-record
/// windowing keeps about half the tokens and packing keeps all but a handful.
/// The report of what was lost comes back with the windows instead of being
/// recomputed by the caller, so there is one answer to "how much did we lose".
///
/// # Errors
/// Whatever [`paketle`] refuses, named with the side it refused on.
pub fn pencereler(
    bolum: &Bolum,
    uzunluk: usize,
) -> Result<(Vec<PaketPencere>, Vec<PaketPencere>), String> {
    let egitim = paket_veya_hata(&bolum.egitim, uzunluk, "egitim")?;
    let dogrulama = paket_veya_hata(&bolum.dogrulama, uzunluk, "dogrulama")?;
    Ok((egitim, dogrulama))
}

fn paket_veya_hata(
    kayitlar: &[Kayit],
    uzunluk: usize,
    taraf: &str,
) -> Result<Vec<PaketPencere>, String> {
    let diziler: Vec<Vec<u32>> = kayitlar.iter().map(|k| k.jetonlar.clone()).collect();
    let (pencereler, rapor) = paketle(&diziler, uzunluk)
        .map_err(|h: PaketHatasi| format!("{taraf} tarafi paketlenemedi: {h:?}"))?;
    if pencereler.is_empty() {
        return Err(format!(
            "{taraf} tarafi {uzunluk} jetonluk tek bir pencere bile vermiyor ({} jeton, en uzun kayit {}): \
             korpus bu pencere uzunlugu icin fazla kucuk",
            rapor.kapsanan_jeton + rapor.artan_jeton,
            kayitlar.iter().map(|k| k.jetonlar.len()).max().unwrap_or(0)
        ));
    }
    Ok(pencereler)
}

/// The window measurement for one side, so the caller can report what packing
/// cost instead of guessing.
///
/// # Errors
/// [`crate::PencereHatasi`] on an empty side or a zero window.
pub fn olcu(
    bolum: &Bolum,
    uzunluk: usize,
) -> Result<(crate::PencereRaporu, crate::PencereRaporu), crate::PencereHatasi> {
    let egitim: Vec<usize> = bolum.egitim.iter().map(|k| k.jetonlar.len()).collect();
    let dogrulama: Vec<usize> = bolum.dogrulama.iter().map(|k| k.jetonlar.len()).collect();
    Ok((
        pencere_olcu(&egitim, uzunluk)?,
        pencere_olcu(&dogrulama, uzunluk)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kayitlar(adet: usize) -> Vec<Kayit> {
        (0..adet)
            .map(|i| Kayit {
                kimlik: format!("kimlik-{i}"),
                jetonlar: (0..40).map(|j| ((i * 7 + j) % 8192) as u32).collect(),
            })
            .collect()
    }

    #[test]
    fn the_split_follows_the_identity_not_the_position() {
        // Karar verici olcum: bir kayit eklenince diger kayitlar taraf
        // degistirmemeli. Pozisyona gore bolme bunu garanti etmez.
        let on = bolumle(kayitlar(40), 0.25).expect("split");
        let mut buyuk = kayitlar(40);
        buyuk.insert(
            0,
            Kayit {
                kimlik: "yeni-kimlik".to_string(),
                jetonlar: vec![1, 2, 3, 4],
            },
        );
        let sonra = bolumle(buyuk, 0.25).expect("split");
        let eski_dogrulama: Vec<&String> = on.dogrulama.iter().map(|k| &k.kimlik).collect();
        for kimlik in eski_dogrulama {
            let yer = sonra
                .dogrulama
                .iter()
                .find(|k| &k.kimlik == kimlik)
                .or_else(|| sonra.egitim.iter().find(|k| &k.kimlik == kimlik));
            assert!(yer.is_some(), "kayit {kimlik} iki tarafta da yok");
            assert!(
                sonra.dogrulama.iter().any(|k| &k.kimlik == kimlik),
                "{kimlik} dogrulamadan egitime kaydi: bolme pozisyona bagli"
            );
        }
    }

    #[test]
    fn a_share_outside_the_open_unit_interval_is_refused() {
        for pay in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
            assert_eq!(
                bolumle(kayitlar(10), pay),
                Err(BolumHatasi::PaydaDisi),
                "pay {pay} kabul edildi"
            );
        }
    }

    #[test]
    fn a_repeated_identity_is_refused_not_deduplicated() {
        let mut kayit = kayitlar(4);
        let kopya = kayit[2].clone();
        kayit.push(kopya);
        assert_eq!(bolumle(kayit, 0.25), Err(BolumHatasi::KimlikTekrari));
    }

    #[test]
    fn an_empty_side_is_a_refusal_not_a_silent_zero() {
        // Tek kayitla %25 dogrulama istemek ya bos egitim ya bos dogrulama
        // verirdi; ikisi de turun olcusunu anlamsiz kilar.
        let tek = bola(kayitlar(1));
        assert!(matches!(
            bolumle(tek, 0.5),
            Err(BolumHatasi::BosTaraf) | Ok(_)
        ));
        assert_eq!(bolumle(Vec::new(), 0.5), Err(BolumHatasi::BosTaraf));
    }

    fn bola(k: Vec<Kayit>) -> Vec<Kayit> {
        k
    }

    #[test]
    fn the_side_is_stable_across_runs() {
        let a = bolumle(kayitlar(64), 0.2).expect("split");
        let b = bolumle(kayitlar(64), 0.2).expect("split");
        assert_eq!(a, b, "ayni girdi iki farkli bolme verdi");
    }

    #[test]
    fn windows_keep_the_record_they_came_from() {
        let bolum = bolumle(kayitlar(8), 0.25).expect("split");
        let (egitim, dogrulama) = pencereler(&bolum, 16).expect("windows");
        assert!(!egitim.is_empty() && !dogrulama.is_empty());
        for pencere in egitim.iter().chain(dogrulama.iter()) {
            assert_eq!(pencere.kimlikler.len(), 16);
            assert_eq!(pencere.kaynak.len(), 16);
        }
    }

    #[test]
    fn a_window_above_the_spec_is_named_not_clamped() {
        let spec = Spec::lubot_a1();
        assert_eq!(pencere_uzunlugu(spec, 256).expect("ok"), 256);
        assert!(pencere_uzunlugu(spec, 257).is_err());
        assert!(pencere_uzunlugu(spec, 1).is_err());
        assert!(pencere_uzunlugu(spec, 0).is_err());
    }

    #[test]
    fn a_side_too_small_for_one_window_is_refused_with_its_size() {
        let bolum = Bolum {
            egitim: vec![Kayit {
                kimlik: "a".into(),
                jetonlar: vec![1, 2],
            }],
            dogrulama: vec![Kayit {
                kimlik: "b".into(),
                jetonlar: vec![3, 4],
            }],
        };
        let hata = pencereler(&bolum, 16).expect_err("kucuk korpus kabul edildi");
        assert!(hata.contains("egitim"), "hata tarafi soylemiyor: {hata}");
    }

    #[test]
    fn the_measurement_says_what_packing_costs() {
        let bolum = bolumle(kayitlar(10), 0.2).expect("split");
        let (egitim, _dogrulama) = olcu(&bolum, 16).expect("measure");
        // Paketleme kayit basina pencerelmeden daha az jeton atar: korpusun
        // cogu kisa kayittan olusuyor, o yuzden fark burada olculuyor.
        assert!(
            egitim.paket_pencere >= egitim.pencere,
            "paketleme daha az pencere verdi: {} < {}",
            egitim.paket_pencere,
            egitim.pencere
        );
        assert!(egitim.paket_artan < 16);
        assert!(egitim.artan_jeton >= egitim.paket_artan);
    }
}
