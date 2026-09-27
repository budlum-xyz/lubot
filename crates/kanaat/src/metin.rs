//! Turkish text preparation: folding, tokenizing, negation.
//!
//! # Why folding comes first
//!
//! Turkish has four letters that all lowercase to two different characters
//! depending on the locale: `İ`, `I`, `ı` and `i`. A matcher that lowercases
//! with the system locale gets a different token stream on a Turkish machine
//! than on an English one, and the difference looks like a scoring change
//! rather than a locale bug. Nothing here calls a locale-aware conversion:
//! [`katla`] maps each character by hand and the mapping is total, so the same
//! sentence folds the same way everywhere.
//!
//! The fold is deliberately lossy - `İ` and `I` both become `i`, `ı` becomes
//! `i` as well. That is what makes "KAYIT", "Kayıt" and "kayıt" one token. The
//! price is that a word pair differing only in dotted/dotless `i` can no longer
//! be told apart, which is why the fold is applied to *matching* only: the
//! original text is what a report quotes.
//!
//! # Why the stop list is written folded
//!
//! The list is compared against folded tokens, so every entry here is already
//! folded: `için` is stored as `icin`, `çok` as `cok`. An entry written with
//! its diacritics would never match and would silently be dead weight - a bug
//! that is invisible because the word it was meant to filter still gets scored,
//! just at full weight instead of none.

/// Words that carry no discriminating power in a verdict.
///
/// Folded, ASCII-only, lowercase. See the module note for why.
pub(crate) const DUR_ILETLERI: &[&str] = &[
    "acaba",
    "ama",
    "ancak",
    "aslinda",
    "az",
    "bazi",
    "belki",
    "bile",
    "bircok",
    "birkac",
    "birsey",
    "bu",
    "butun",
    "cok",
    "cunku",
    "cuney",
    "da",
    "daha",
    "de",
    "defa",
    "degil",
    "diger",
    "diye",
    "dort",
    "eger",
    "elbette",
    "en",
    "gibi",
    "hem",
    "her",
    "hersey",
    "hic",
    "icin",
    "ile",
    "ise",
    "iste",
    "kadar",
    "karsi",
    "ki",
    "kim",
    "madem",
    "mi",
    "mu",
    "nasil",
    "ne",
    "neden",
    "nerde",
    "nerede",
    "nicin",
    "niye",
    "o",
    "olarak",
    "oldugu",
    "olmak",
    "olmasi",
    "olsun",
    "oyle",
    "ozellikle",
    "ragmen",
    "sadece",
    "sanki",
    "sey",
    "seyler",
    "siz",
    "sonra",
    "soyle",
    "su",
    "tum",
    "uc",
    "uzere",
    "var",
    "ve",
    "veya",
    "ya",
    "yani",
    "yine",
];

/// Words that carry a negation.
///
/// Folded, like [`DUR_ILETLERI`]. This list and [`OLUMSUZLUK_SONEKLERI`] are
/// both needed: `gecersiz` is negated by its suffix and appears in no list, and
/// `yok` is negated by the list and has no suffix.
pub(crate) const OLUMSUZLUK_ILETLERI: &[&str] = &[
    "asla", "degil", "hayir", "hic", "olmadi", "olmaz", "olmuyor", "yok", "yoktur", "yoksun",
];

/// Suffixes that negate what they are attached to.
///
/// A suffix, not a word: `acilmadi` and `acilmaz` are negations of `acildi` and
/// `acilir`. Matching on the ending rather than on a word list is the only way
/// to cover a language that builds negation by suffixing; the cost is that a
/// word which merely *ends* like a negation is read as one, which is why the
/// stems are kept long enough to be unambiguous.
pub(crate) const OLUMSUZLUK_SONEKLERI: &[&str] = &[
    // -me/-ma with the tense endings that follow it
    "madi", "madi", "maz", "mazlik", "mayacak", "mayan", "mamis", "medi", "mez", "mezlik",
    "meyecek", "meyen", "memis", "miyor", "muyor",
    // -siz: "without", carried by a suffix rather than a separate word
    "siz", "sizlik", "sizdir", "suz", "suzluk",
];

/// The fold: one character in, one character out, no locale consulted.
///
/// Non-alphabetic characters are returned unchanged; the caller decides what
/// to do with them.
#[must_use]
pub(crate) fn katla(harf: char) -> char {
    match harf {
        'İ' | 'I' | 'ı' | 'i' => 'i',
        'Ş' | 'ş' => 's',
        'Ğ' | 'ğ' => 'g',
        'Ü' | 'ü' => 'u',
        'Ö' | 'ö' => 'o',
        'Ç' | 'ç' => 'c',
        'Â' | 'â' => 'a',
        'Î' | 'î' => 'i',
        'Û' | 'û' => 'u',
        'É' | 'é' => 'e',
        digeri => digeri.to_ascii_lowercase(),
    }
}

/// Folds and squeezes: every token separator becomes one space.
///
/// Digits are kept, because a verdict often turns on a number, and a period
/// inside a number is kept for the same reason (`3.5` is one token, not two).
#[must_use]
pub(crate) fn sadeles(metin: &str) -> String {
    let mut cikti = String::with_capacity(metin.len());
    let mut son_bosluk = true;
    let harfler: Vec<char> = metin.chars().collect();
    for (sira, ham) in harfler.iter().enumerate() {
        let harf = katla(*ham);
        let sayi_noktasi = harf == '.'
            && sira > 0
            && sira + 1 < harfler.len()
            && harfler[sira - 1].is_ascii_digit()
            && harfler[sira + 1].is_ascii_digit();
        if harf.is_ascii_alphanumeric() || sayi_noktasi {
            cikti.push(harf);
            son_bosluk = false;
        } else if !son_bosluk {
            cikti.push(' ');
            son_bosluk = true;
        }
    }
    if cikti.ends_with(' ') {
        cikti.pop();
    }
    cikti
}

/// The tokens a piece of text is matched by.
#[must_use]
pub(crate) fn jetonlar(metin: &str) -> Vec<String> {
    sadeles(metin)
        .split(' ')
        .filter(|jeton| !jeton.is_empty())
        .map(str::to_string)
        .collect()
}

/// Adjacent token pairs, joined with `\u{1f}` so a bigram can never collide
/// with a single token that happens to contain a space.
#[must_use]
pub(crate) fn ikili_gramlar(jetonlar: &[String]) -> Vec<String> {
    jetonlar
        .windows(2)
        .map(|ikili| format!("{}\u{1f}{}", ikili[0], ikili[1]))
        .collect()
}

/// Whether one token carries a negation.
#[must_use]
pub(crate) fn olumsuz_mu(jeton: &str) -> bool {
    if OLUMSUZLUK_ILETLERI.contains(&jeton) {
        return true;
    }
    OLUMSUZLUK_SONEKLERI
        .iter()
        .any(|sonek| jeton.len() > sonek.len() + 1 && jeton.ends_with(sonek))
}

/// The share of tokens that carry a negation.
#[must_use]
pub(crate) fn olumsuzluk_orani(jetonlar: &[String]) -> f64 {
    if jetonlar.is_empty() {
        return 0.0;
    }
    let sayi = jetonlar.iter().filter(|jeton| olumsuz_mu(jeton)).count();
    sayi as f64 / jetonlar.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_collapses_the_four_i_letters() {
        for harf in ['İ', 'I', 'ı', 'i'] {
            assert_eq!(katla(harf), 'i', "{harf} katlanmadi");
        }
        assert_eq!(sadeles("KAYIT AÇILDI"), "kayit acildi");
        assert_eq!(sadeles("Kayıt Açıldı"), "kayit acildi");
        assert_eq!(sadeles("kayıt açıldı"), "kayit acildi");
    }

    #[test]
    fn folding_does_not_consult_the_locale() {
        // A locale-aware lowercase would turn `I` into `ı` in Turkish and into
        // `i` elsewhere; here both spellings land on the same token.
        assert_eq!(jetonlar("ILIŞKI"), jetonlar("ılışkı"));
        assert_eq!(jetonlar("ILIŞKI"), vec!["iliski"]);
    }

    #[test]
    fn a_decimal_number_stays_one_token_and_a_sentence_dot_does_not() {
        assert_eq!(
            jetonlar("oran 3.5 oldu. bitti"),
            vec!["oran", "3.5", "oldu", "bitti"]
        );
    }

    #[test]
    fn punctuation_becomes_a_single_separator() {
        assert_eq!(jetonlar("kayit,,,  acildi!!!"), vec!["kayit", "acildi"]);
        assert_eq!(sadeles("  bas  son  "), "bas son");
    }

    #[test]
    fn bigrams_do_not_collide_with_single_tokens() {
        let jetonlar = jetonlar("kayit acildi ve surdu");
        let gramlar = ikili_gramlar(&jetonlar);
        assert_eq!(gramlar.len(), jetonlar.len() - 1);
        assert_eq!(gramlar[0], "kayit\u{1f}acildi");
        assert!(
            !jetonlar.contains(&gramlar[0]),
            "bigram tek jetonla cakisti"
        );
    }

    #[test]
    fn negation_is_read_from_both_the_list_and_the_suffixes() {
        assert!(olumsuz_mu("degil"));
        assert!(olumsuz_mu("yok"));
        assert!(olumsuz_mu("acilmadi"));
        assert!(olumsuz_mu("gecersiz"));
        assert!(olumsuz_mu("olmayacak"));
        assert!(!olumsuz_mu("acildi"));
        // A short token that merely ends like a suffix is not a negation.
        assert!(!olumsuz_mu("maz"));
    }

    #[test]
    fn the_stop_list_is_written_folded_and_stays_that_way() {
        for giris in DUR_ILETLERI {
            let geri = sadeles(giris);
            assert_eq!(*giris, geri, "`{giris}` katlanmis degil: `{geri}` olmali");
        }
        for giris in OLUMSUZLUK_ILETLERI {
            assert_eq!(*giris, sadeles(giris), "`{giris}` katlanmis degil");
        }
    }

    #[test]
    fn the_negation_ratio_counts_the_share_not_the_number() {
        let jetonlar = jetonlar("kayit acilmadi ve surdu");
        assert!((olumsuzluk_orani(&jetonlar) - 0.25).abs() < 1e-12);
        assert!((olumsuzluk_orani(&[])).abs() < 1e-12);
    }
}
