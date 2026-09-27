//! Bayt seviyesinde şema otomatı: hangi baytın hangi durumda kabul edildiği.
//!
//! Otomat, `lubot-read::output_schema`'nın denetlediği Markdown sözleşmesinin
//! **ön yüzüdür**: doğrulayıcı bir belgeyi bitmiş hâliyle reddeder, otomat ise
//! aynı kuralları bayt bayt uygular ve bir belgeyi hiç bitirmeye gerek
//! kalmadan daraltır. İkisinin ilişkisi tek yönlüdür ve ölçülür: otomatın kabul
//! ettiği her belge doğrulayıcıdan geçer; doğrulayıcının kabul ettiği bazı
//! belgeler otomatın dışındadır (daraltma). Daraltmanın nerede olduğu
//! [`crate`] dokümanında ve [`Yuruyus`] üzerindeki notlarda yazılıdır.
//!
//! ## Durum neden `Copy`
//!
//! [`Yuruyus::izinli`] her jeton için çağrılır ve bir jeton birden çok bayt
//! taşıyabilir. Durumun tamamı skaler olduğu için denemek bir kopya + bir
//! geçiş turudur; belge (üretilmiş baytlar) durumun dışında tutulur, yani
//! maskeleme belgenin uzunluğuyla büyümez.

use crate::{Sema, SemaHatasi};

/// Otomatın bütün durumu: yığın alanı yok, bu yüzden `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Durum {
    /// Üretilen toplam bayt.
    bayt: usize,
    /// Şu anki satırın numarası (1 tabanlı).
    satir: u32,
    /// Bu satırda kaç bayt üretildi.
    satir_ici: u16,
    /// Satır başındaki boşluk baytı sayısı (doğrulayıcı satırı kırpar).
    on_bosluk: u16,
    /// Satırdaki son boşluk-olmayan bayt; `0` henüz yok demek.
    son_anlamli: u8,
    /// Beklenen UTF-8 devam baytı sayısı.
    utf8_kalan: u8,
    /// Beklenen toplam devam baytı sayısı (aralık denetimi için).
    utf8_beklenen: u8,
    /// Çok baytlı dizinin ilk baytı.
    utf8_bas: u8,
    /// Çok baytlı dizinin baytları (karakter tamamlanınca çözülür).
    utf8_tampon: [u8; 4],
    /// Belgede boşluk dışında bir şey görüldü mü.
    icerik_var: bool,
    /// Daha önce bir başlık görüldü mü (iniş kuralı yalnız ondan sonra).
    baslik_goruldu: bool,
    /// Son başlığın seviyesi.
    baslik_seviyesi: u8,
    /// Satır başındaki `#` koşusunun içinde miyiz.
    kare_modu: bool,
    /// Satır başındaki `#` sayısı.
    kare_sayisi: u8,
    /// Açık bir kod çitinin içinde miyiz.
    cit_icinde: bool,
    /// Açık çitin eşik uzunluğu.
    cit_uzunlugu: u8,
    /// Çitin açıldığı satır.
    cit_satiri: u32,
    /// Satır başındaki ters-kare koşusunun içinde miyiz.
    tk_modu: bool,
    /// Satır başındaki ters-kare sayısı.
    tk_sayisi: u8,
    /// Bu satır bir çit satırı mı (açan ya da kapatan).
    cit_satiri_mi: bool,
    /// Bu satır çiti kapatacak aday mı (satır sonuna kadar yalnız boşluk).
    cit_kapanis: bool,
    /// Bu satır bir tablo satırı mı.
    satir_modu: bool,
    /// Bu satırdaki `|` sayısı.
    satir_pipe: u16,
    /// Açık tablo bloğundaki satır sayısı.
    tablo_satir: u16,
    /// Tablo bloğunun başlık satırındaki sütun sayısı.
    tablo_sutun: u16,
    /// Tablo bloğunun ilk satırının numarası (red mesajı için).
    tablo_ilk_satir: u32,
    /// Şu anki hücredeki boşluk-olmayan bayt sayısı.
    hucre_anlamli: u16,
    /// Şu anki hücre `-` içeriyor mu.
    hucre_tire: bool,
    /// Şu anki hücre yalnız `-`, `:`, boşluk mu içeriyor.
    hucre_ayirac: bool,
}

impl Durum {
    const fn baslangic() -> Self {
        Self {
            bayt: 0,
            satir: 1,
            satir_ici: 0,
            on_bosluk: 0,
            son_anlamli: 0,
            utf8_kalan: 0,
            utf8_beklenen: 0,
            utf8_bas: 0,
            utf8_tampon: [0; 4],
            icerik_var: false,
            baslik_goruldu: false,
            baslik_seviyesi: 0,
            kare_modu: false,
            kare_sayisi: 0,
            cit_icinde: false,
            cit_uzunlugu: 0,
            cit_satiri: 0,
            tk_modu: false,
            tk_sayisi: 0,
            cit_satiri_mi: false,
            cit_kapanis: false,
            satir_modu: false,
            satir_pipe: 0,
            tablo_satir: 0,
            tablo_sutun: 0,
            tablo_ilk_satir: 0,
            hucre_anlamli: 0,
            hucre_tire: false,
            hucre_ayirac: true,
        }
    }

    /// Bir baytı kabul et ve yeni durumu ver.
    ///
    /// Bu fonksiyon otomatın kendisidir: kuralların hepsi burada, başka bir
    /// yerde ikinci bir kopyası yok.
    fn gecis(self, gelen: u8, sema: &Sema) -> Result<Self, SemaHatasi> {
        let mut d = self;
        if d.bayt + 1 > sema.en_fazla_bayt {
            return Err(SemaHatasi::CokBuyuk { bayt: d.bayt + 1 });
        }
        // Satırın anlam başı, sayaç artmadan **önce** okunur: doğrulayıcı satırı
        // kırpar, otomat da aynı yeri başlangıç sayar. Sayaç arttıktan sonra
        // okumak satır başını bir bayt kaydırır ve ne `#`, ne `|`, ne ters-kare
        // kendi satırında tanınır.
        let satir_basi = d.satir_ici == d.on_bosluk;
        d.bayt += 1;
        d.satir_ici += 1;

        // --- UTF-8: bayt seviyesinde decode'un varlık sebebi ---------------
        if d.utf8_kalan > 0 {
            let (alt, ust) = devam_araligi(d.utf8_bas, d.utf8_kalan, d.utf8_beklenen);
            if gelen < alt || gelen > ust {
                return Err(SemaHatasi::GecersizUtf8 {
                    bayt: gelen,
                    satir: d.satir as usize,
                });
            }
            let konum = (d.utf8_beklenen - d.utf8_kalan) as usize;
            d.utf8_tampon[konum] = gelen;
            d.utf8_kalan -= 1;
            if d.utf8_kalan == 0 {
                let uzunluk = d.utf8_beklenen as usize + 1;
                let bosluk = std::str::from_utf8(&d.utf8_tampon[..uzunluk])
                    .ok()
                    .and_then(|s| s.chars().next())
                    .is_some_and(char::is_whitespace);
                // Çok baytlı bir karakter satır yapısını değiştirmez: ne `#`,
                // ne `|`, ne ters-kare. Tek yaptığı şey metin olmak.
                if bosluk {
                    d.bosluk_isle(satir_basi);
                } else {
                    d.icerik_var = true;
                    if d.kare_modu {
                        // Cok baytli karakter de bir bayttir: `#` kosusunu
                        // bosluk izlemeli kurali icin ASCII olup olmamasi fark
                        // etmez, yoksa daraltma yalniz yari belgede isirirdi.
                        return Err(SemaHatasi::BaslikBoslukIster {
                            satir: d.satir as usize,
                        });
                    }
                    if d.tk_modu {
                        // Olculen kacis buradaydi: "```" kosusunu cok baytli bir
                        // karakter izlerse kosu ASCII yolundan gecmedigi icin
                        // hic kapanmiyor, cit acilmamis sayiliyor ve belge
                        // "kabul" gorunurken dogrulayici onu dengesiz cit diye
                        // reddediyordu. Kural tek: kosu bitti, karar verildi.
                        let kosu = d.tk_sayisi;
                        d.tk_modu = false;
                        d.ters_kare_kosu_bitti(kosu);
                    }
                    d.metin_bayti();
                }
            }
            return Ok(d);
        }
        match gelen {
            0x00..=0x7f => {}
            0xc2..=0xdf => {
                d.utf8_basla(gelen, 1);
                return Ok(d);
            }
            0xe0..=0xef => {
                d.utf8_basla(gelen, 2);
                return Ok(d);
            }
            0xf0..=0xf4 => {
                d.utf8_basla(gelen, 3);
                return Ok(d);
            }
            _ => {
                return Err(SemaHatasi::GecersizUtf8 {
                    bayt: gelen,
                    satir: d.satir as usize,
                })
            }
        }

        // --- ASCII: satır yapısı ------------------------------------------
        if gelen == b'\n' {
            d.satir_sonu()?;
            return Ok(d);
        }
        if gelen == b' ' || gelen == b'\t' {
            // Başlığı boşluk onaylar ve bu dal başka hiçbir yere uğramaz:
            // onay burada yapılmazsa `# ` koşusu satır sonunda "boşluksuz
            // başlık" diye reddedilir. Doğrulayıcı `starts_with(' ')` diye
            // bakar, yani sekme başlığı onaylamaz — otomat da onaylamıyor.
            if d.kare_modu {
                if gelen != b' ' {
                    return Err(SemaHatasi::BaslikBoslukIster {
                        satir: d.satir as usize,
                    });
                }
                d.baslik_goruldu = true;
                d.baslik_seviyesi = d.kare_sayisi;
                d.kare_modu = false;
            }
            d.bosluk_isle(satir_basi);
            return Ok(d);
        }
        // `\r` bu gramerde yok: doğrulayıcı `\r\n`'i kabul eder, otomat satır
        // sonunu yalnız `\n` sayar. Bilinçli daraltma, `crate` dokümanında.
        if gelen == b'\r' {
            return Err(SemaHatasi::GecersizUtf8 {
                bayt: gelen,
                satir: d.satir as usize,
            });
        }
        d.icerik_var = true;
        d.yapi_bayti(gelen, satir_basi)?;
        Ok(d)
    }

    fn utf8_basla(&mut self, bayt: u8, devam: u8) {
        self.utf8_bas = bayt;
        self.utf8_kalan = devam;
        self.utf8_beklenen = devam;
        self.utf8_tampon[0] = bayt;
    }

    fn bosluk_isle(&mut self, satir_basi: bool) {
        if satir_basi {
            self.on_bosluk += 1;
        }
        // Boşluk hücrenin "anlamlı" sayılmasını sağlamaz ama ayracı bozmaz:
        // `| :- |` bir ayraç hücresidir, `| - x |` değildir.
    }

    /// Metin baytı: satır yapısını değiştirmeyen her şey.
    ///
    /// `son_anlamli` gerçek baytı değil `b'm'` taşır: satır sonunda sorulan tek
    /// soru "satır `|` ile mi bitti", yani hangi metin baytı olduğu değil.
    fn metin_bayti(&mut self) {
        self.son_anlamli = b'm';
        if self.tk_modu {
            self.tk_modu = false;
        }
        if self.satir_modu {
            self.hucre_anlamli += 1;
            self.hucre_ayirac = false;
        }
    }

    /// Satır yapısı taşıyan bir ASCII baytı.
    fn yapi_bayti(&mut self, b: u8, satir_basi: bool) -> Result<(), SemaHatasi> {
        // Ters-kare koşusu: satırın anlam başında başlar, ilk başka baytta biter.
        if self.tk_modu {
            if b == b'`' {
                self.tk_sayisi += 1;
                self.son_anlamli = b;
                return Ok(());
            }
            let kosu = self.tk_sayisi;
            self.tk_modu = false;
            self.ters_kare_kosu_bitti(kosu);
        } else if satir_basi {
            // Ters-kare koşusu çitin **içinde de** izlenir: çiti kapatan satır
            // orada başlar. `#` ve `|` ise yalnız dışarıda yapı taşır.
            if b == b'`' {
                self.tk_modu = true;
                self.tk_sayisi = 1;
                self.son_anlamli = b;
                return Ok(());
            }
            if !self.cit_icinde {
                match b {
                    b'#' => {
                        self.kare_modu = true;
                        self.kare_sayisi = 1;
                        self.son_anlamli = b;
                        return Ok(());
                    }
                    b'|' => {
                        self.satir_modu = true;
                        self.satir_pipe = 1;
                        if self.tablo_satir == 0 {
                            self.tablo_ilk_satir = self.satir;
                        }
                        self.son_anlamli = b;
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

        if self.kare_modu {
            if b == b'#' {
                // İniş kuralı baytın kendisinde ısırır: üçüncü `#` üretildikten
                // sonra satırı kurtaracak bir yol kalmaz, o yüzden burada red.
                if self.baslik_goruldu && self.kare_sayisi + 1 > self.baslik_seviyesi + 1 {
                    return Err(SemaHatasi::BaslikAtlama {
                        onceki: self.baslik_seviyesi,
                        seviye: self.kare_sayisi + 1,
                    });
                }
                self.kare_sayisi += 1;
                self.son_anlamli = b;
                return Ok(());
            }
            // Başka bir bayt: `#` koşusu başlık olmak zorunda (daraltma).
            return Err(SemaHatasi::BaslikBoslukIster {
                satir: self.satir as usize,
            });
        }

        // Çit kapanış adayı: satırın geri kalanı yalnız boşluk olmalı; başka bir
        // bayt gelirse satır çit değil metindir (doğrulayıcıyla aynı okuma).
        if self.cit_kapanis && b != b'`' {
            self.cit_kapanis = false;
        }

        if self.cit_icinde && !self.cit_satiri_mi && !self.cit_kapanis {
            // Çitin içi: hiçbir satır yapısı yok, her şey metin.
            self.metin_bayti();
            return Ok(());
        }

        if self.satir_modu {
            if b == b'|' {
                if self.satir_pipe >= 1 && self.tablo_satir == 1 {
                    // İkinci satır ayraç olmak zorunda: her hücre boş değil,
                    // `-` içeriyor ve yalnız `-`/`:`/boşluk taşıyor.
                    if self.hucre_anlamli == 0 || !self.hucre_tire || !self.hucre_ayirac {
                        return Err(SemaHatasi::TabloUyumsuz {
                            satir: self.tablo_ilk_satir as usize,
                        });
                    }
                }
                self.satir_pipe += 1;
                self.hucre_anlamli = 0;
                self.hucre_tire = false;
                self.hucre_ayirac = true;
                self.son_anlamli = b;
                return Ok(());
            }
            self.hucre_anlamli += 1;
            if b == b'-' {
                self.hucre_tire = true;
            } else if b != b':' {
                self.hucre_ayirac = false;
            }
            self.son_anlamli = b;
            return Ok(());
        }

        self.son_anlamli = b;
        Ok(())
    }

    /// Satır başındaki ters-kare koşusu bitti: çit mi, metin mi.
    fn ters_kare_kosu_bitti(&mut self, kosu: u8) {
        if kosu < 3 {
            return;
        }
        if !self.cit_icinde {
            self.cit_icinde = true;
            self.cit_uzunlugu = kosu;
            self.cit_satiri = self.satir;
            self.cit_satiri_mi = true;
        } else if kosu >= self.cit_uzunlugu {
            self.cit_kapanis = true;
            self.cit_satiri_mi = true;
        }
    }

    /// Satır sonu: satırın taşıdığı iddialar burada karara bağlanır.
    fn satir_sonu(&mut self) -> Result<(), SemaHatasi> {
        // Satır yalnız ters-kare ile bittiyse koşu hâlâ açık sayılır.
        if self.tk_modu {
            let kosu = self.tk_sayisi;
            self.tk_modu = false;
            self.ters_kare_kosu_bitti(kosu);
        }
        if self.kare_modu {
            return Err(SemaHatasi::BaslikBoslukIster {
                satir: self.satir as usize,
            });
        }
        if self.satir_modu {
            // Doğrulayıcı satırı kırpar: satır `|` ile bitmek zorunda, yoksa bu
            // bir tablo satırı değil düz metindir ve bu gramer onu kabul etmez.
            if self.son_anlamli != b'|' {
                return Err(SemaHatasi::TabloUyumsuz {
                    satir: self.tablo_ilk_satir as usize,
                });
            }
            let sutun = self.satir_pipe.saturating_sub(1);
            if self.tablo_satir == 0 {
                self.tablo_sutun = sutun;
            } else if self.tablo_satir == 1 {
                if sutun != self.tablo_sutun || sutun == 0 {
                    return Err(SemaHatasi::TabloUyumsuz {
                        satir: self.tablo_ilk_satir as usize,
                    });
                }
            } else if sutun != self.tablo_sutun {
                return Err(SemaHatasi::TabloUyumsuz {
                    satir: self.tablo_ilk_satir as usize,
                });
            }
            self.tablo_satir += 1;
        } else if self.tablo_satir > 0 {
            // Satır tablo satırı değil: blok kapanır. Bloğun doğruluğu satır
            // satır denetlendiği için burada ayrıca bir denetim yok.
            self.tablo_satir = 0;
            self.tablo_sutun = 0;
        }
        if self.cit_kapanis {
            self.cit_icinde = false;
            self.cit_kapanis = false;
            self.cit_uzunlugu = 0;
        }
        self.satir += 1;
        self.satir_ici = 0;
        self.on_bosluk = 0;
        self.son_anlamli = 0;
        self.kare_modu = false;
        self.kare_sayisi = 0;
        self.cit_satiri_mi = false;
        self.satir_modu = false;
        self.satir_pipe = 0;
        self.hucre_anlamli = 0;
        self.hucre_tire = false;
        self.hucre_ayirac = true;
        Ok(())
    }

    /// Belge burada bitse kabul edilir mi; edilmiyorsa neden.
    fn kapat(&self) -> Result<(), SemaHatasi> {
        if self.utf8_kalan > 0 {
            return Err(SemaHatasi::GecersizUtf8 {
                bayt: self.utf8_bas,
                satir: self.satir as usize,
            });
        }
        // Son satır `\n` ile bitmemiş olabilir; aynı kararlar orada da geçerli.
        let mut kopya = *self;
        if kopya.satir_ici > 0 {
            kopya.satir_sonu()?;
        }
        if kopya.cit_icinde {
            return Err(SemaHatasi::CitDengesiz {
                satir: kopya.cit_satiri as usize,
            });
        }
        if !self.icerik_var {
            return Err(SemaHatasi::Bos);
        }
        Ok(())
    }
}

/// Çok baytlı bir dizinin şu anki baytı için geçerli aralık.
///
/// Aşırı uzun kodlamalar ve vekil (surrogate) bölge burada reddedilir: `0xC0`
/// ve `0xC1` hiç kabul edilmez, `0xED`'yi `0x80..=0x9F` izler, `0xF4`'ü
/// `0x80..=0x8F`. Bu aralıklar olmadan otomat "geçerli UTF-8" değil "UTF-8
/// biçimli" olurdu ve fark doğrulayıcıda görünürdü.
fn devam_araligi(bas: u8, kalan: u8, beklenen: u8) -> (u8, u8) {
    if kalan != beklenen {
        return (0x80, 0xbf);
    }
    match bas {
        0xe0 => (0xa0, 0xbf),
        0xed => (0x80, 0x9f),
        0xf0 => (0x90, 0xbf),
        0xf4 => (0x80, 0x8f),
        _ => (0x80, 0xbf),
    }
}

/// Decode'un şema tarafındaki durumu: otomat + üretilmiş belge.
///
/// Bir `Yuruyus` **ön ek**tir. [`Yuruyus::kapat`] çağrılmadan kabul edilmiş
/// sayılmaz: açık bir çit, yarım kalmış bir UTF-8 dizisi ya da boş bir belge
/// ancak orada görünür.
#[derive(Debug, Clone)]
pub struct Yuruyus {
    sema: Sema,
    durum: Durum,
    belge: Vec<u8>,
}

impl Yuruyus {
    /// Boş bir ön ekle başla.
    #[must_use]
    pub fn yeni(sema: Sema) -> Self {
        Self {
            sema,
            durum: Durum::baslangic(),
            belge: Vec::new(),
        }
    }

    /// Bu baytlar şu anki duruma eklenebilir mi.
    ///
    /// Maskenin tek kaynağı budur: decode başka bir yerde ikinci bir kural
    /// tutmaz, o yüzden "kabul edilen çıktı doğrulayıcıdan geçer" iddiası tek
    /// bir fonksiyona bağlıdır.
    #[must_use]
    pub fn izinli(&self, bayt: &[u8]) -> bool {
        let mut d = self.durum;
        for b in bayt {
            match d.gecis(*b, &self.sema) {
                Ok(yeni) => d = yeni,
                Err(_) => return false,
            }
        }
        true
    }

    /// Baytları ekle; şema reddederse durum değişmez.
    ///
    /// # Errors
    ///
    /// [`SemaHatasi`] — hangi kuralın hangi satırda reddettiğini taşır.
    pub fn ilerle(&mut self, bayt: &[u8]) -> Result<(), SemaHatasi> {
        let mut d = self.durum;
        for b in bayt {
            d = d.gecis(*b, &self.sema)?;
        }
        self.durum = d;
        self.belge.extend_from_slice(bayt);
        Ok(())
    }

    /// Şu anki durumda tek bir bayt bile kabul ediliyor mu.
    ///
    /// `false` bir çıkmazdır: decode'un reddetmesi gerekir, gevşemesi değil.
    #[must_use]
    pub fn cikmaz(&self) -> bool {
        (0u16..=255).all(|b| {
            #[allow(clippy::cast_possible_truncation)]
            let bayt = b as u8;
            !self.izinli(&[bayt])
        })
    }

    /// Belge burada bitse şema kabul eder mi.
    #[must_use]
    pub fn kabul(&self) -> bool {
        self.kapat().is_ok()
    }

    /// Belge burada bitse şemanın söyleyeceği şey.
    ///
    /// # Errors
    ///
    /// [`SemaHatasi`] — açık çit, boş belge, yarım UTF-8, satır sonunda
    /// kapanmamış başlık ya da bozuk tablo.
    pub fn kapat(&self) -> Result<(), SemaHatasi> {
        self.durum.kapat()
    }

    /// Şu ana kadar üretilmiş baytlar.
    #[must_use]
    pub fn belge(&self) -> &[u8] {
        &self.belge
    }

    /// Üretilen bayt sayısı.
    #[must_use]
    pub fn bayt_sayisi(&self) -> usize {
        self.durum.bayt
    }

    /// Şu anki satır numarası (1 tabanlı).
    #[must_use]
    pub fn satir(&self) -> usize {
        self.durum.satir as usize
    }

    /// Açık bir kod çitinin içinde miyiz.
    ///
    /// Ölçüm için dışarı açık: "belge çit içinde bitti" ile "belge hiç kapanmayan
    /// bir çitle bitti" aynı şey değil ve rapor ikisini ayırmak istiyor.
    #[must_use]
    pub fn cit_icinde(&self) -> bool {
        self.durum.cit_icinde
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kos(metin: &str) -> Result<Yuruyus, SemaHatasi> {
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        for b in metin.as_bytes() {
            y.ilerle(&[*b])?;
        }
        Ok(y)
    }

    fn kabul(metin: &str) -> bool {
        kos(metin).is_ok_and(|y| y.kabul())
    }

    #[test]
    fn bos_ve_bosluklu_belge_reddedilir() {
        assert_eq!(kos("").and_then(|y| y.kapat()), Err(SemaHatasi::Bos));
        assert_eq!(
            kos("   \n\t\n").and_then(|y| y.kapat()),
            Err(SemaHatasi::Bos)
        );
        assert!(kabul("tek cumle"));
    }

    #[test]
    fn baslik_inisi_bayt_seviyesinde_rededilir() {
        assert!(kabul("# Baslik\n\n## Alt\n\nmetin\n"));
        // `###` üçüncü baytında red gelir: belge hiç bozulmadan durur.
        let y = kos("# Baslik\n\n").expect("öneki geçerli");
        assert!(y.izinli(b"#"));
        let iki = {
            let mut y = y.clone();
            y.ilerle(b"#").expect("birinci kare");
            y
        };
        assert!(iki.izinli(b"#"));
        let mut uc = {
            let mut y = iki.clone();
            y.ilerle(b"#").expect("ikinci kare");
            y
        };
        assert!(!uc.izinli(b"#"), "seviye atlayan başlık maskelenmeli");
        assert_eq!(
            uc.ilerle(b"#").err(),
            Some(SemaHatasi::BaslikAtlama {
                onceki: 1,
                seviye: 3
            })
        );
        // Yükselmek serbest: üçten bire dönüş bir iniş değildir.
        assert!(kabul("### Derin\n\n# Geri\n\n## Bir basamak\n"));
    }

    #[test]
    fn baslik_bosluk_ister_daraltmasi() {
        // Doğrulayıcı `#x`'i düz metin sayar; otomat saymaz. Daraltma ölçülür,
        // saklanmaz: red adı kendi adıyla var.
        assert!(kos("#x").is_err(), "`#x` bu gramerin disinda");
        assert!(kabul("# x\n"), "boslukla baslik gecerli");
        let y = kos("").expect("boş öneki");
        let y = {
            let mut y = y;
            y.ilerle(b"#").expect("kare");
            y
        };
        assert_eq!(
            {
                let mut y = y.clone();
                y.ilerle(b"x").err()
            },
            Some(SemaHatasi::BaslikBoslukIster { satir: 1 })
        );
    }

    #[test]
    fn cit_dengesi_olculur() {
        assert!(kabul("metin\n\n```\nkod\n```\n"));
        assert!(kabul("````\n```\nic ice kisa cit\n```\n````\n"));
        let acik = kos("metin\n```\nkod\n").expect("öneki geçerli");
        assert!(acik.cit_icinde());
        assert_eq!(
            acik.kapat().err(),
            Some(SemaHatasi::CitDengesiz { satir: 2 })
        );
        // Kısa koşu çiti kapatmaz: doğrulayıcıyla aynı okuma.
        assert!(kabul("````\n```\n````\n"));
    }

    /// Belgeyi koş ve sonundaki kararı ver: bazı reddler satır sonunda, bazıları
    /// belge sonunda görünür; ikisini ayırmak testi yanlış yere bağlar.
    fn sonuc(metin: &str) -> Result<(), SemaHatasi> {
        kos(metin).and_then(|y| y.kapat())
    }

    #[test]
    fn tablo_ayraci_ve_sutun_sayisi() {
        assert!(kabul("| a | b |\n|---|---|\n| 1 | 2 |\n"));
        assert!(kabul("| a |\n")); // tek satırlık blok tablo değil
                                   // Ayraç olmayan ikinci satır reddedilir.
        let h = sonuc("| a | b |\n| 1 | 2 |");
        assert!(matches!(h, Err(SemaHatasi::TabloUyumsuz { .. })), "{h:?}");
        // Sütun sayısı başlıkla uyuşmalı: red satırın bittiği yerde görünür.
        let h = sonuc("| a | b |\n|---|---|\n| 1 | 2 | 3 |");
        assert!(matches!(h, Err(SemaHatasi::TabloUyumsuz { .. })), "{h:?}");
        // Ayraç hücresi `-` içermeli.
        let h = sonuc("| a | b |\n| : | : |");
        assert!(matches!(h, Err(SemaHatasi::TabloUyumsuz { .. })), "{h:?}");
        // `|` ile bitmeyen satır bu gramerde tablo satırı değil, düz metin de
        // değil: daraltma.
        let h = sonuc("| a | b\n");
        assert!(matches!(h, Err(SemaHatasi::TabloUyumsuz { .. })), "{h:?}");
    }

    #[test]
    fn utf8_araliklari_isirir() {
        // Geçerli: iki baytlı `ı`, üç baytlı `—`, dört baytlı bir emoji.
        assert!(kabul("ılık — 🌱\n"));
        // Aşırı uzun kodlama: `0xC0 0xAF` bir `/` değildir.
        let y = Yuruyus::yeni(Sema::varsayilan());
        assert!(!y.izinli(&[0xc0, 0xaf]));
        // Vekil bölge: `0xED 0xA0 0x80` reddedilir.
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        y.ilerle(&[0xed]).expect("lead byte accepted");
        assert!(!y.izinli(&[0xa0]));
        // Yarım kalan dizi belge sonunda yakalanır.
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        y.ilerle(b"a").expect("ascii");
        y.ilerle(&[0xc4]).expect("lead byte");
        assert_eq!(
            y.kapat().err(),
            Some(SemaHatasi::GecersizUtf8 {
                bayt: 0xc4,
                satir: 1
            })
        );
        // `\r` bu gramerde yok.
        assert!(!Yuruyus::yeni(Sema::varsayilan()).izinli(b"\r\n"));
    }

    #[test]
    fn tavan_asilinca_red_gelir() {
        let sema = Sema { en_fazla_bayt: 4 };
        let mut y = Yuruyus::yeni(sema);
        y.ilerle(b"abcd").expect("tavaa kadar");
        assert_eq!(y.ilerle(b"e").err(), Some(SemaHatasi::CokBuyuk { bayt: 5 }));
    }

    #[test]
    fn izinli_ve_ilerle_ayni_karari_verir() {
        // İki fonksiyon ayrı yollar: biri kopyalar, biri yazar. Aynı baytta
        // ayrı karar verirlerse maske yalan söylemiş olur.
        let ornekler = [
            "# B\n\nm\n",
            "```\nx\n```\n",
            "| a |\n|-|\n",
            "ılık\n",
            "#x",
            "a\r\n",
        ];
        for metin in ornekler {
            let a = Yuruyus::yeni(Sema::varsayilan());
            let mut b = Yuruyus::yeni(Sema::varsayilan());
            let mut gecen = 0usize;
            for (i, bayt) in metin.as_bytes().iter().enumerate() {
                // İki yol aynı öneki tutmalı: `a` hiç ilerlemez, `b` ilerler;
                // kararlar ayrılırsa maske yalan söylüyor demektir.
                let mut a_kopya = a.clone();
                a_kopya
                    .ilerle(&metin.as_bytes()[..gecen])
                    .expect("öneki geçer");
                let izin = a_kopya.izinli(&[*bayt]);
                let sonuc = b.ilerle(&[*bayt]);
                assert_eq!(
                    izin,
                    sonuc.is_ok(),
                    "{metin:?} bayt {i} ({bayt:#04x}) için kararlar ayrıldı"
                );
                if sonuc.is_err() {
                    break;
                }
                gecen += 1;
            }
            assert_eq!(b.belge(), &metin.as_bytes()[..gecen]);
        }
    }

    #[test]
    fn durum_belgeden_bagimsiz_kopyalanir() {
        // `izinli` belgeyi kopyalamaz: uzun bir belgede maskeleme maliyeti
        // belgeyle büyümesin diye. Bu test o bağın koptuğunu ölçer.
        let mut y = Yuruyus::yeni(Sema::varsayilan());
        y.ilerle(&vec![b'a'; 5000]).expect("uzun metin");
        let once = y.bayt_sayisi();
        assert!(y.izinli(b"\n"));
        assert_eq!(y.bayt_sayisi(), once, "izinli durumu değiştirdi");
        assert_eq!(y.belge().len(), once);
    }

    #[test]
    fn cikmaz_tavanda_ve_baslik_esiginde_olculur() {
        // Tavan: tek bir bayt bile sığmıyorsa çıkmaz gerçektir ve decode'un
        // reddetmesi gerekir (gevşemek değil).
        let mut y = Yuruyus::yeni(Sema { en_fazla_bayt: 2 });
        y.ilerle(b"ab").expect("tavana kadar");
        assert!(y.cikmaz(), "tavan doluyken hiçbir bayt kabul edilemez");
        // Başlık eşiğinde çıkmaz **yoktur**: `#` ve `\n` reddedilir ama boşluk
        // başlığı kapatır. Bu ölçülür, çünkü "red" ile "çıkmaz" aynı şey değil.
        let mut y = kos("# B\n\n").expect("öneki geçerli");
        y.ilerle(b"##").expect("iki kare");
        assert!(!y.izinli(b"#"), "seviye atlayan üçüncü kare maskelenir");
        assert!(!y.izinli(b"\n"), "boşluksuz başlık satırı reddedilir");
        assert!(y.izinli(b" "), "boşluk başlığı kapatır, çıkmaz değil");
        assert!(!y.cikmaz());
    }

    #[test]
    fn cikmazi_olmayan_bir_cikmaz_yolu_adim_sinirina_duser() {
        // Sıfır sütunlu bir başlık satırından sonra gelen her satır tabloyu
        // bozar: tek baytlık bir çıkmaz yok ama **tamamlayan** bir yol da yok.
        // Bu ayrım ölçülür, çünkü decode'un red nedeni ikisinde başka.
        let mut y = kos("|\n").expect("sıfır sütunlu tek satır geçerli");
        y.ilerle(b"|").expect("satır başı");
        y.ilerle(b"-").expect("hücre");
        assert!(!y.izinli(b"\n"), "sütun sayısı başlıkla uyuşmuyor");
        assert!(y.izinli(b"-"), "içerik baytı hâlâ kabul edilir");
        assert!(y.izinli(b"|"), "hücreyi kapatmak tek baytta geçerli");
        assert!(!y.cikmaz(), "çıkmaz değil: çıkmaz tek baytta görünür");
        // Ama satırı bitiren her yol reddedilir: belge asla kabul edilmez.
        let mut k = y.clone();
        k.ilerle(b"|").expect("hücre sonu");
        assert!(!k.izinli(b"\n"), "sütun 1, başlık 0");
        assert!(!k.kabul(), "bu ön ek kabul edilemez");
    }

    #[test]
    fn cok_baytli_karakter_ters_kare_kosusunu_kapatir() {
        // Ölçülen kaçışın kalıcı testi: "```" kosusunu cok baytli bir karakter
        // izlerse kosu ASCII yolundan gecmiyordu, cit hic acilmamis sayiliyor
        // ve belge "kabul" gorunurken dogrulayici onu dengesiz cit diye
        // reddediyordu. Iki yol ayni karari vermek zorunda.
        let cok = kos("```\u{fe}EpVU").expect("oneki gecerli");
        let ascii = kos("```xEpVU").expect("oneki gecerli");
        assert!(cok.cit_icinde(), "cok baytli karakter citi acmadi");
        assert!(ascii.cit_icinde(), "ascii karakter citi acmadi");
        assert_eq!(
            cok.kapat().err(),
            Some(SemaHatasi::CitDengesiz { satir: 1 }),
            "cok baytli yolda cit kapanmadi"
        );
        assert_eq!(
            ascii.kapat().err(),
            Some(SemaHatasi::CitDengesiz { satir: 1 }),
            "ascii yolda cit kapanmadi"
        );
    }
}
