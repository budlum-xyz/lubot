//! Otonom veri-tetiklemeli egitim dongusu (calisma direktifi bolum 6).
//!
//! Bu crate bir zamanlayici degil, bir **red zinciri**. Dongunun her halkasi
//! "ne zaman devam edilir" sorusundan cok "hangi durumda durulur" sorusunu
//! cevaplar, cunku otonom bir sistemi guvenli yapan sey ilerleme yetenegi
//! degil, durma yetenegidir.
//!
//! Halkalar ve her birinin **tek cumlesi**:
//!
//! | modul | 6.x | tek cumle |
//! |---|---|---|
//! | [`tetikleyici`] | 6.1 | Zaman gecmesi veri degildir: bos parti egitilmez. |
//! | [`veri_yolu`] | 6.2 | Taninmayan etiket omurgaya dusurulmez. |
//! | [`kosum`] | 6.3 | Egitim sayisi ile servis sayisi ayni sey degildir. |
//! | [`regresyon`] | 6.4 | Once ratchet okunur, sonra kayip. |
//! | [`yayin`] | 6.5 | Soyu dogrulanamayan kontrol noktasi yayimlanmaz. |
//! | [`capraz`] | 6.6 | Bir dogrulayici cogunluk degildir. |
//! | [`secim`] | 6.7 | Esit skor esit raporlanir. |
//! | [`koken`] | 6.8 | Katki kaydedilir, odullendirilmez. |
//! | [`dallanma`] | 6.9 | Ortak atasi olmayan modeller ortalanmaz. |
//!
//! Hicbir modul agirlik tutmaz. Hepsinde `parametre_sayisi() == 0` ve bu bir
//! yorum degil, kapinin denetledigi bir metin sozlesmesi: bu crate dongunun
//! **kararlarini** tasir, ogrendiklerini degil. Ogrenen taraf
//! `crates/egitim` ve `crates/omurga`; aralarindaki sinir bilincli.
//!
//! Hicbir modul gercek saat okumaz, gercek dosya acmaz, ag'a cikmaz. Zaman ve
//! girdi disaridan verilir. Bunun sebebi test kolayligi degil: otonom bir
//! dongunun bir kararini **yeniden uretebilmek**, o kararin denetlenebilir
//! olmasinin on kosulu. Yeniden uretilemeyen bir karar, gerekcesi ne olursa
//! olsun, denetlenmemis bir karardir.

pub mod capraz;
pub mod dallanma;
pub mod koken;
pub mod kosum;
pub mod regresyon;
pub mod secim;
pub mod tetikleyici;
pub mod veri_yolu;
pub mod yayin;

/// Crate genelinde gecerli: bu crate'in hicbir modulu parametre tutmaz.
///
/// Tek bir yerden sorulabilmesi, kapinin tek bir yere bakabilmesi demek.
#[must_use]
pub const fn parametre_sayisi() -> usize {
    0
}

/// Dongudeki halka sayisi (direktif 6.1 - 6.9).
pub const HALKA_SAYISI: usize = 9;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_parametre_tutmaz() {
        assert_eq!(parametre_sayisi(), 0);
        assert_eq!(tetikleyici::Biriktirici::parametre_sayisi(), 0);
        assert_eq!(veri_yolu::Ayirici::parametre_sayisi(), 0);
        assert_eq!(kosum::Bolme::parametre_sayisi(), 0);
        assert_eq!(regresyon::Fren::parametre_sayisi(), 0);
        assert_eq!(yayin::Yayinci::parametre_sayisi(), 0);
        assert_eq!(capraz::Kurul::parametre_sayisi(), 0);
        assert_eq!(secim::Kapisma::parametre_sayisi(), 0);
        assert_eq!(koken::Defter::parametre_sayisi(), 0);
        assert_eq!(dallanma::Corba::parametre_sayisi(), 0);
    }

    #[test]
    fn dokuz_halka() {
        assert_eq!(HALKA_SAYISI, 9);
    }
}
