//! # lubot-sema-cozucu - şemayı geçemeyen çıktıyı üretmeyen decode
//!
//! Lubot'un anayasası çıktıyı iki yerden sıkıştırır: üretmez, okur; ve
//! kullanıcının önüne çıkan tek biçim **şema doğrulanmış Markdown**'dır —
//! şemayı geçemeyen çıktı reddedilir, "en yakın formata düşürülmez"
//! (`lubot-read::output_schema`). Bugüne kadar bu kural çıktıyı *ürettikten
//! sonra* denetliyordu: üret, doğrula, reddet. Bu crate aynı kuralı decode'un
//! **içine** taşır: geçerli çıktı uzayı bayt seviyesinde bir otomatla daraltılır
//! ve modelin dağılımından yalnız o uzaya düşen jetonlar seçilebilir.
//!
//! Doğrudan madde 7.3'ün `schema_decoder` bileşenidir: şema-kısıtlı,
//! bayt-seviye, dilbilgisi-maskeli decode. Yöntem ilhamı dışarıdandır; buradaki
//! otomat, maskeleme ve red yolu bu ağaçta sıfırdan yazılmıştır (K1).
//!
//! ## Üç sözleşme
//!
//! 1. **Daraltma, gevşetme yok.** Otomat, doğrulayıcının kabul ettiği dilin bir
//!    **alt kümesini** kabul eder. Bir jeton otomatın durumundan geçmiyorsa
//!    maskelenir; maskelenecek hiçbir jeton kalmadıysa decode **reddeder**
//!    (`CozumHatasi::Cikmaz`). Reddedilen bir decode, yumuşatılmış bir çıktıya
//!    dönüştürülmez — bu, anayasadaki "asla yumuşatılmaz"ın decode tarafıdır.
//! 2. **Bayt seviyesi.** Otomat jeton değil bayt okur; bu yüzden çok baytlı
//!    UTF-8 dizileri de durumun parçasıdır. Geçersiz bir UTF-8 dizisini
//!    tamamlayacak bir jeton, başındaki bayt kabul edilmiş olsa bile bir daha
//!    kabul edilmez.
//! 3. **Ölçülen daraltma.** Otomatın doğrulayıcıdan daha dar olduğu yerler
//!    (satır başı `#` koşusu başlık olmak zorundadır, `\r` yoktur, tablo satırı
//!    `|` ile bitmek zorundadır) saklanmaz; [`olcum`] raporunda daraltmanın
//!    kendisi değil ama **kaçışın olmadığı** ölçülür: otomatın kabul ettiği
//!    hiçbir belge doğrulayıcıdan geçemez durumda değildir.
//!
//! ## Ne değildir
//!
//! Bu crate bir model, bir örnekleyici ve bir servis değildir. Logitleri
//! dışarıdan alır, sözlüğü dışarıdan alır, ürettiği belgeyi dışarı verir.
//! Sıcaklık/top-k gibi dağıtım işleri `lubot-cikarim::ornekleyici`'nindir;
//! buradaki tek seçim kuralı maskelenmiş küme içinde en yüksek logittir
//! (eşitlikte küçük indeks kazanır — iki eşit sayı arasına sıra uydurmak fark
//! uydurmaktır).

pub mod cozucu;
pub mod otomat;

pub use cozucu::{adim, coz, Adim, CozumHatasi, CozumOlcum};
pub use otomat::Yuruyus;

/// Otomatın kabul ettiği en büyük belge.
///
/// Doğrulayıcının kendi tavanıyla aynı sayı: iki tavan birbirinden farklı
/// olsaydı, dar olanı hangisiyse sözleşme o olurdu ve diğeri yalan söylerdi.
pub const EN_FAZLA_BAYT: usize = 64 * 1024;

/// Decode'un nasıl daraltılacağının ayarı.
///
/// Tek knobsuz bir ayar bilinçli: daraltma kararı bir ayar değil, şemanın
/// kendisidir. Ayar olarak duran tek şey boyut tavanı, çünkü o şemanın zaten
/// söylediği sayı.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sema {
    /// Kabul edilen en çok bayt.
    pub en_fazla_bayt: usize,
}

impl Sema {
    /// Doğrulayıcıyla aynı tavanı taşıyan varsayılan ayar.
    #[must_use]
    pub const fn varsayilan() -> Self {
        Self {
            en_fazla_bayt: EN_FAZLA_BAYT,
        }
    }
}

impl Default for Sema {
    fn default() -> Self {
        Self::varsayilan()
    }
}

/// Otomatın bir baytı neden reddettiği.
///
/// Her varyant bir kuraldır ve her biri bir testte üretilir: üretilmeyen bir
/// red yolu, olmayan bir kuraldır.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemaHatasi {
    /// Bayt, geçerli bir UTF-8 dizisinin parçası olamaz.
    GecersizUtf8 {
        /// Reddedilen bayt.
        bayt: u8,
        /// Hangi satırda.
        satir: usize,
    },
    /// Belge bittiğinde içinde boşluk dışında bir şey yoktu.
    Bos,
    /// Belge tavanı aştı.
    CokBuyuk {
        /// Ulaşılan bayt sayısı.
        bayt: usize,
    },
    /// Başlık seviyesi inerken bir basamak atladı.
    BaslikAtlama {
        /// Bir önceki başlığın seviyesi.
        onceki: u8,
        /// İstenen seviye.
        seviye: u8,
    },
    /// Satır başındaki `#` koşusunu boşluk izlemedi.
    ///
    /// Doğrulayıcı bunu düz metin sayar; otomat başlık sayar ve boşluk ister.
    /// Bilinçli daraltma: `#x` gibi bir satır bu gramerin dışında.
    BaslikBoslukIster {
        /// Hangi satırda.
        satir: usize,
    },
    /// Belge bittiğinde açık bir kod çiti vardı.
    CitDengesiz {
        /// Çitin açıldığı satır.
        satir: usize,
    },
    /// Tablo satırı ayracı tutmadı ya da sütun sayısı başlıkla uyuşmadı.
    TabloUyumsuz {
        /// Bloğun ilk satırı.
        satir: usize,
    },
}

impl std::fmt::Display for SemaHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GecersizUtf8 { bayt, satir } => {
                write!(f, "line {satir}: byte {bayt:#04x} is not valid UTF-8 here")
            }
            Self::Bos => write!(f, "the document holds nothing but whitespace"),
            Self::CokBuyuk { bayt } => write!(f, "the document reached {bayt} bytes"),
            Self::BaslikAtlama { onceki, seviye } => write!(
                f,
                "a heading descends from level {onceki} to {seviye}, skipping a level"
            ),
            Self::BaslikBoslukIster { satir } => write!(
                f,
                "line {satir} starts with '#' and no space follows: outside this grammar"
            ),
            Self::CitDengesiz { satir } => {
                write!(f, "a code fence opened on line {satir} never closed")
            }
            Self::TabloUyumsuz { satir } => {
                write!(f, "the table block starting on line {satir} is malformed")
            }
        }
    }
}

impl std::error::Error for SemaHatasi {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tavan_dogrulayicinin_tavani() {
        // İki tavanın aynı sayı olduğu bir iddia değil, bir bağ: biri değişirse
        // bu test düşer ve daraltmanın hangi tarafta olduğu yeniden yazılır.
        assert_eq!(EN_FAZLA_BAYT, lubot_read::output_schema::MAX_OUTPUT_BYTES);
        assert_eq!(Sema::varsayilan().en_fazla_bayt, EN_FAZLA_BAYT);
        assert_eq!(Sema::default(), Sema::varsayilan());
    }

    #[test]
    fn her_red_varyanti_bir_mesaj_tasir() {
        let hatalar = [
            SemaHatasi::GecersizUtf8 {
                bayt: 0xff,
                satir: 1,
            },
            SemaHatasi::Bos,
            SemaHatasi::CokBuyuk { bayt: 65537 },
            SemaHatasi::BaslikAtlama {
                onceki: 1,
                seviye: 3,
            },
            SemaHatasi::BaslikBoslukIster { satir: 4 },
            SemaHatasi::CitDengesiz { satir: 2 },
            SemaHatasi::TabloUyumsuz { satir: 3 },
        ];
        for h in hatalar {
            let mesaj = h.to_string();
            assert!(!mesaj.is_empty());
            assert!(mesaj.len() > 10, "mesaj bir şey söylemiyor: {mesaj}");
        }
    }
}
