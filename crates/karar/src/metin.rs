//! Metin isleme — Turkce katlama, jetonlama, sayilar.
//!
//! Kanaat crate'indeki metin.rs'den esinlendi, ama sifirdan yazildi (K1).
//! Dis bagimlilik yok, sadece stdlib.

/// Turkce karakterleri normalize et: kucult, I/İ -> i, ğ->g, ü->u, ş->s, ö->o, ç->c
#[must_use]
pub fn normalize(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'İ' | 'I' => 'i',
            'Ğ' => 'g',
            'Ü' => 'u',
            'Ş' => 's',
            'Ö' => 'o',
            'Ç' => 'c',
            'ğ' => 'g',
            'ü' => 'u',
            'ş' => 's',
            'ö' => 'o',
            'ç' => 'c',
            'ı' => 'i',
            _ => c.to_ascii_lowercase(),
        })
        .collect()
}

/// Jetonlara ayir: a-z, 0-9 dizileri.
#[must_use]
pub fn jetonlar(text: &str) -> Vec<String> {
    let norm = normalize(text);
    let mut jetonlar = Vec::new();
    let mut current = String::new();

    for ch in norm.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            if current.len() >= 2 {
                jetonlar.push(current.clone());
            }
            current.clear();
        }
    }
    if !current.is_empty() && current.len() >= 2 {
        jetonlar.push(current);
    }

    jetonlar
}

/// Sayilari cikar.
#[must_use]
pub fn sayilar(text: &str) -> Vec<String> {
    let mut sayilar = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            sayilar.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        sayilar.push(current);
    }

    sayilar
}

/// Olumsuzluk var mi?
#[must_use]
pub fn olumsuzluk_var(text: &str) -> bool {
    let norm = normalize(text);
    ["degil", "yok", "hayir", "red", "değil"]
        .iter()
        .any(|w| norm.contains(*w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_turkce() {
        assert_eq!(normalize("İstanbul"), "istanbul");
        assert_eq!(normalize("ĞÜŞİÖÇ"), "gusioc");
        assert_eq!(normalize("çalışma"), "calisma");
    }

    #[test]
    fn jetonlar_bolme() {
        let toks = jetonlar("Public content is read without asking.");
        assert!(toks.contains(&"public".to_string()));
        assert!(toks.contains(&"content".to_string()));
    }

    #[test]
    fn sayilar_cikar() {
        let nums = sayilar("74830 * 1291 kac eder?");
        assert_eq!(nums, vec!["74830", "1291"]);
    }

    #[test]
    fn olumsuzluk() {
        assert!(olumsuzluk_var("bu degil"));
        assert!(!olumsuzluk_var("bu dogru"));
    }
}
