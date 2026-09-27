//! Engram bellegi adayi — tasarim notu 3.3.
//!
//! KV'nin bir bolumu, diziden **hesaplanan** K/V yerine, jeton
//! dizisinden cikan **n-gram karmasiyla adreslenen** ogrenilmis tablodan
//! `gather` ile okunur. Model sik n-gram'larin belleğini tabloya tasir; dikkat
//! hesabinin kendisi degismez - degisen sey K/V'nin nereden geldigidir; burada
//! olculen sey adresleme, sinir ve geri gecistir.
//!
//! ```text
//! hucre          = ngram_karmasi(tokens[i-n+1 ..= i]) % tablo      (tam sayi, tohumsuz)
//! anahtar, deger = tablo[hucre]                                    (gather)
//! ```
//!
//! # Neden bu modul ayri duruyor
//!
//! Bu bir **mimari degisiklik adayidir, uygulanmis bir mimari degil**: spec'i
//! (`training/model_spec.json`) degistirmez ve skorlama cagrisina baglanmaz.
//! Tasarim notu 3.3'un isaretli karari ("engram bu aileye girer mi") bu
//! olcumle **verilmez**; modul yalnizca o kararin gerektirdigi sayilari uretir.
//!
//! # Tasarim sartlari ve her birinin olcumu
//!
//! 1. **karma deterministik ve tohumsuz.** karma yalniz tam sayi aritmetigidir
//!    (float yok, rastgelelik yok): ayni jeton dizisi her makinede ayni hucreyi
//!    adresler. Testte ayni dizinin tekrar tekrar ayni hucreyi vermesi ve
//!    farkli bir "tohum" diye bir seyin **olmamasi** (imzada parametre yok)
//!    uzerinden gosterilir.
//! 2. **`gather`'in geri gecisi seyrek toplayicidir.** Yalniz okunan hucrelere
//!    gradyan yazilir; toplama **hucre sirasinda** yapilir, boylece okuma
//!    sirasi degisse de sonuc bit-esit kalir (`seyrek_toplam_hucre_sirasinda`).
//! 3. **Kayit-siniri maskesi engram okumalarina da uygulanir.** Bir pencere
//!    komsu kaydin n-gram'indan okuyamaz: `kaynak` kimligi degisen yerde okuma
//!    **yok**'tur ve bu, komsunun jetonlari degistirilerek olculur.
//! 4. **Cakisma fail-closed degil, olculen kayip.** Ayni hucreye dussen farkli
//!    n-gram sayisi `cakisma` olarak raporlanir; modul bunu yasaklamaz, sayar.
//!
//! tablo boyutu dogrudan parametre butcesinden yer (K6): `2·tablo·d_kv`.
//! Muhasebe testte tam sayi aritmetigidir; aday izgarasi olcum ister.

/// n-gram uzunlugu ust siniri. Daha buyugu spec kararinda tartisilir.
pub const NGRAM_UST_SINIRI: usize = 8;

/// Engram tablosunun sekli.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngramSpec {
    /// n-gram uzunlugu (kac jetonun karmasi adres uretir).
    pub n: usize,
    /// hucre sayisi (tablo boyu).
    pub tablo: usize,
    /// hucre basina anahtar/deger genisligi.
    pub d_kv: usize,
}

/// Neden bir sekil reddedildi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngramSekilHatasi {
    /// n-gram uzunlugu sifir olamaz.
    SifirN,
    /// n-gram uzunlugu ust siniri asiyor.
    NUstSinirAsildi,
    /// hucre sayisi sifir olamaz.
    SifirTablo,
    /// Genislik sifir olamaz.
    SifirGenislik,
}

impl EngramSpec {
    /// Sekli dogrular.
    pub fn yeni(n: usize, tablo: usize, d_kv: usize) -> Result<Self, EngramSekilHatasi> {
        if n == 0 {
            return Err(EngramSekilHatasi::SifirN);
        }
        if n > NGRAM_UST_SINIRI {
            return Err(EngramSekilHatasi::NUstSinirAsildi);
        }
        if tablo == 0 {
            return Err(EngramSekilHatasi::SifirTablo);
        }
        if d_kv == 0 {
            return Err(EngramSekilHatasi::SifirGenislik);
        }
        Ok(Self { n, tablo, d_kv })
    }

    /// Sekilden turetilen parametre sayisi: anahtar + deger, her hucre icin.
    #[must_use]
    pub fn parametre_sayisi(&self) -> usize {
        2 * self.tablo * self.d_kv
    }

    /// hucre basi: anahtarlar `[0, tablo*d_kv)`, degerler ondan sonra.
    #[must_use]
    pub fn anahtar_tabani(&self, hucre: usize) -> usize {
        hucre * self.d_kv
    }

    /// deger diliminin basi.
    #[must_use]
    pub fn deger_tabani(&self, hucre: usize) -> usize {
        self.tablo * self.d_kv + hucre * self.d_kv
    }
}

/// Bir konumda okunan hucre ve icerigi.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramOkuma {
    /// Konum (dizideki jeton indeksi).
    pub konum: usize,
    /// Adreslenen hucre.
    pub hucre: usize,
    /// Okunan anahtar.
    pub anahtar: Vec<f64>,
    /// Okunan deger.
    pub deger: Vec<f64>,
}

/// Okuma ozeti: kac konum okundu, kac konum atlandi ve neden.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramOzet {
    /// Okunan konumlar.
    pub okumalar: Vec<EngramOkuma>,
    /// Gecmis yetersiz oldugu icin atlanan konum sayisi.
    pub gecmis_yok: usize,
    /// Kayit siniri gectigi icin atlanan konum sayisi.
    pub kayit_siniri: usize,
}

/// Tohumsuz, tam sayi n-gram karmasi.
///
/// karma yalniz jeton kimliklerine ve konuma baglidir: float yok, rastgelelik
/// yok, ortam degiskeni yok. Ayni n-gram her makinede ayni `u64`'u verir.
#[must_use]
pub fn ngram_karmasi(ngram: &[usize]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, jeton) in ngram.iter().enumerate() {
        let t = u64::try_from(*jeton).unwrap_or(u64::MAX);
        let konum = u64::try_from(i).unwrap_or(u64::MAX);
        let karisim = t
            .wrapping_mul(0x0000_0100_0000_01b3)
            .wrapping_add(konum.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        h = (h ^ karisim).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// n-gram'in dustugu hucre.
///
/// Uzunluk `spec.n` degilse `None` doner: yanlis uzunlukta bir adres uretmek
/// sessiz bir hata olurdu.
#[must_use]
pub fn hucre(spec: EngramSpec, ngram: &[usize]) -> Option<usize> {
    if ngram.len() != spec.n {
        return None;
    }
    Some((ngram_karmasi(ngram) % u64::try_from(spec.tablo).unwrap_or(u64::MAX)) as usize)
}

/// Tabloyu tohumlu doldurur (olcum icin; gercek modelde ogrenilir).
#[must_use]
pub fn belirgin_doldur(spec: EngramSpec, tohum: u64) -> Vec<f64> {
    let mut sayac: u64 = tohum
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..2 * spec.tablo * spec.d_kv)
        .map(|_| {
            sayac = sayac
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let ust = (sayac >> 33) as f64 / (1u64 << 31) as f64;
            (ust - 0.5) * 0.5
        })
        .collect()
}

/// Diziden okumalar: her konum icin nedensel n-gram adresi.
///
/// Iki atlama nedeni vardir ve ikisi ayri sayilir:
/// - konumun gecmisi `n` jetona yetmiyor (`gecmis_yok`);
/// - pencere kayit sinirini geciyor (`kayit_siniri`) - bir pencere komsu kaydin
///   n-gram'indan okuyamaz.
#[must_use]
pub fn oku(spec: EngramSpec, tablo: &[f64], tokens: &[usize], kaynak: &[u32]) -> EngramOzet {
    let mut okumalar = Vec::new();
    let mut gecmis_yok = 0usize;
    let mut kayit_siniri = 0usize;
    for konum in 0..tokens.len() {
        if konum + 1 < spec.n {
            gecmis_yok += 1;
            continue;
        }
        let bas = konum + 1 - spec.n;
        let ayni_kayit = kaynak
            .get(bas..=konum)
            .map(|dilim| dilim.windows(2).all(|p| p[0] == p[1]))
            .unwrap_or(false);
        if !ayni_kayit {
            kayit_siniri += 1;
            continue;
        }
        let pencere = &tokens[bas..=konum];
        let bazi_hucre = match hucre(spec, pencere) {
            Some(c) => c,
            None => continue,
        };
        let k_bas = spec.anahtar_tabani(bazi_hucre);
        let v_bas = spec.deger_tabani(bazi_hucre);
        let anahtar = tablo.get(k_bas..k_bas + spec.d_kv).unwrap_or(&[]).to_vec();
        let deger = tablo.get(v_bas..v_bas + spec.d_kv).unwrap_or(&[]).to_vec();
        okumalar.push(EngramOkuma {
            konum,
            hucre: bazi_hucre,
            anahtar,
            deger,
        });
    }
    EngramOzet {
        okumalar,
        gecmis_yok,
        kayit_siniri,
    }
}

/// Tek bir okumanin gradyani: hangi hucreye, anahtar/deger icin ne kadar.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramKatkisi {
    /// Hedef hucre.
    pub hucre: usize,
    /// anahtar gradyani (`d_kv`).
    pub anahtar: Vec<f64>,
    /// deger gradyani (`d_kv`).
    pub deger: Vec<f64>,
}

/// Seyrek geri gecis: yalniz okunan hucrelere yazar, toplamayı **hucre
/// sirasinda** yapar.
///
/// Toplama sirasi bilinclidir: girdi katkilarinin sirasi degisse de ayni hucreye
/// gelen terimler ayni sirada toplanir, boylece sonuc bit-esit kalir. Bu,
/// cok-is parcacikli bir kosuda geri gecisin tekrar uretilebilirligidir.
#[must_use]
pub fn geri(spec: EngramSpec, katkilar: &[EngramKatkisi]) -> Vec<f64> {
    let mut g = vec![0.0f64; 2 * spec.tablo * spec.d_kv];
    let mut sirali: Vec<&EngramKatkisi> = katkilar.iter().collect();
    sirali.sort_by_key(|k| k.hucre);
    for katki in sirali {
        let k_bas = spec.anahtar_tabani(katki.hucre);
        let v_bas = spec.deger_tabani(katki.hucre);
        for i in 0..spec.d_kv {
            if let (Some(hedef), Some(kaynak)) = (g.get_mut(k_bas + i), katki.anahtar.get(i)) {
                *hedef += *kaynak;
            }
            if let (Some(hedef), Some(kaynak)) = (g.get_mut(v_bas + i), katki.deger.get(i)) {
                *hedef += *kaynak;
            }
        }
    }
    g
}

/// Cakisma olcumu: kac **farkli** n-gram kac hucreye dustu.
///
/// Fail-closed degil, olculen kayip: cakisma yasak degildir, sayilir. Ayni
/// hucreye dusen iki farkli n-gram, iki farkli bellegi tek hucrede paylasir;
/// bunun model uzerindeki etkisi ayri bir egitim olcumudur.
#[derive(Debug, Clone, PartialEq)]
pub struct CakismaOlcumu {
    /// Gorulen n-gram (pencere) sayisi.
    pub ngram: usize,
    /// Bunlarin dustugu farkli hucre sayisi.
    pub dolu_hucre: usize,
    /// Ayni hucreye birden fazla n-gram dusen hucre sayisi.
    pub cakisan_hucre: usize,
    /// tablo doluluk orani (`dolu_hucre / tablo`).
    pub doluluk: f64,
}

/// Bir jeton dizisindeki n-gram cakismasini olcer (kayit siniri gozetilir).
#[must_use]
pub fn cakisma_olc(spec: EngramSpec, tokens: &[usize], kaynak: &[u32]) -> CakismaOlcumu {
    let mut hucreler: Vec<usize> = Vec::new();
    for konum in 0..tokens.len() {
        if konum + 1 < spec.n {
            continue;
        }
        let bas = konum + 1 - spec.n;
        let ayni_kayit = kaynak
            .get(bas..=konum)
            .map(|dilim| dilim.windows(2).all(|p| p[0] == p[1]))
            .unwrap_or(false);
        if !ayni_kayit {
            continue;
        }
        if let Some(c) = hucre(spec, &tokens[bas..=konum]) {
            hucreler.push(c);
        }
    }
    let ngram = hucreler.len();
    let mut sirali = hucreler;
    sirali.sort_unstable();
    let mut dolu = 0usize;
    let mut cakisan = 0usize;
    let mut onceki: Option<usize> = None;
    let mut bu_hucrede = 0usize;
    for c in sirali {
        if Some(c) == onceki {
            bu_hucrede += 1;
            if bu_hucrede == 2 {
                cakisan += 1;
            }
        } else {
            if onceki.is_some() {
                dolu += 1;
            }
            onceki = Some(c);
            bu_hucrede = 1;
        }
    }
    if onceki.is_some() {
        dolu += 1;
    }
    let doluluk = if spec.tablo == 0 {
        0.0
    } else {
        dolu as f64 / spec.tablo as f64
    };
    CakismaOlcumu {
        ngram,
        dolu_hucre: dolu,
        cakisan_hucre: cakisan,
        doluluk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GRADIENT_CHECK_MUTLAK_TABAN, GRADIENT_CHECK_TOLERANCE};

    fn tohumlu_jeton(n: usize, tohum: u64, sinir: usize) -> Vec<usize> {
        let mut sayac = tohum
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (0..n)
            .map(|_| {
                sayac = sayac
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((sayac >> 33) as usize) % sinir
            })
            .collect()
    }

    fn kayip(okumalar: &[EngramOkuma]) -> f64 {
        okumalar
            .iter()
            .map(|o| {
                o.anahtar
                    .iter()
                    .zip(o.deger.iter())
                    .map(|(k, v)| k * v)
                    .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn sekil_reddi_dogru_hata_verir() {
        assert_eq!(EngramSpec::yeni(0, 8, 4), Err(EngramSekilHatasi::SifirN));
        assert_eq!(
            EngramSpec::yeni(NGRAM_UST_SINIRI + 1, 8, 4),
            Err(EngramSekilHatasi::NUstSinirAsildi)
        );
        assert_eq!(
            EngramSpec::yeni(2, 0, 4),
            Err(EngramSekilHatasi::SifirTablo)
        );
        assert_eq!(
            EngramSpec::yeni(2, 8, 0),
            Err(EngramSekilHatasi::SifirGenislik)
        );
        assert!(EngramSpec::yeni(2, 8, 4).is_ok());
    }

    #[test]
    fn karma_deterministik_ve_tohumsuz() {
        let spec = EngramSpec::yeni(3, 64, 4).expect("gecerli sekil");
        let dizi = [7usize, 11, 13];
        let birinci = hucre(spec, &dizi);
        for _ in 0..8 {
            assert_eq!(
                hucre(spec, &dizi),
                birinci,
                "ayni n-gram farkli hucre verdi"
            );
        }
        // Uzunluk tutmazsa adres uretilmez (sessiz hata yok).
        assert_eq!(hucre(spec, &[7, 11]), None);
        assert_eq!(hucre(spec, &[7, 11, 13, 17]), None);
        // Farkli n-gram genelde farkli hucreye duser; bu bir olasilik ifadesidir,
        // iddia degil: burada yalnizca en az bir ayrimin oldugu olculur.
        let baska = hucre(spec, &[13, 11, 7]);
        assert_ne!(birinci, baska);
    }

    #[test]
    fn parametre_muhasebesi_tam_sayi() {
        let spec = EngramSpec::yeni(2, 4096, 32).expect("gecerli sekil");
        assert_eq!(spec.parametre_sayisi(), 2 * 4096 * 32);
        // tablo boyutu dogrudan butceden yer: iki kat tablo, iki kat parametre.
        let buyuk = EngramSpec::yeni(2, 8192, 32).expect("gecerli sekil");
        assert_eq!(buyuk.parametre_sayisi(), 2 * spec.parametre_sayisi());
        // dilim tabanlari cakismaz: anahtarlar once, degerler sonra.
        assert_eq!(spec.anahtar_tabani(0), 0);
        assert_eq!(spec.deger_tabani(0), spec.tablo * spec.d_kv);
        assert_eq!(
            spec.anahtar_tabani(spec.tablo - 1) + spec.d_kv,
            spec.deger_tabani(0)
        );
        assert_eq!(
            spec.deger_tabani(spec.tablo - 1) + spec.d_kv,
            spec.parametre_sayisi()
        );
    }

    #[test]
    fn gecmis_yok_ve_kayit_siniri_ayri_sayilir() {
        let spec = EngramSpec::yeni(3, 64, 2).expect("gecerli sekil");
        let tablo = belirgin_doldur(spec, 5);
        let tokens = [1usize, 2, 3, 4, 5];
        let kaynak = [7u32; 5];
        let ozet = oku(spec, &tablo, &tokens, &kaynak);
        // n = 3: ilk iki konumun gecmisi yetmez.
        assert_eq!(ozet.gecmis_yok, 2);
        assert_eq!(ozet.kayit_siniri, 0);
        assert_eq!(ozet.okumalar.len(), 3);
        // Kayit siniri: 2. konumdan sonra kaynak degisiyor.
        let karisik = [7u32, 7, 7, 9, 9];
        let ozet2 = oku(spec, &tablo, &tokens, &karisik);
        assert_eq!(ozet2.gecmis_yok, 2);
        assert_eq!(
            ozet2.okumalar.len(),
            1,
            "kayit sinirini gecen pencere okundu"
        );
        assert!(ozet2.kayit_siniri >= 1);
    }

    #[test]
    fn kayit_sinirinda_komsu_jetonlar_okumayi_degistirmez() {
        // Bir pencere komsu kaydin n-gram'indan okuyorsa, komsunun jetonlarini
        // degistirmek okumayi degistirirdi. Degistirmiyorsa sinir tutuyor.
        let spec = EngramSpec::yeni(2, 32, 3).expect("gecerli sekil");
        let tablo = belirgin_doldur(spec, 9);
        let tokens = [3usize, 5, 8, 13];
        let kaynak = [1u32, 1, 2, 2];
        let ozet = oku(spec, &tablo, &tokens, &kaynak);
        let mut degismis = tokens;
        degismis[0] = 999; // yalniz 1. kaydin jetonu degisti
        let ozet2 = oku(spec, &tablo, &degismis, &kaynak);
        assert_eq!(ozet.okumalar.len(), ozet2.okumalar.len());
        // 2. kaydin konumlari (konum >= 2) hic etkilenmemeli...
        let ikinci: Vec<bool> = ozet
            .okumalar
            .iter()
            .zip(ozet2.okumalar.iter())
            .filter(|(a, _)| a.konum >= 2)
            .map(|(a, b)| a.hucre == b.hucre && a.anahtar == b.anahtar && a.deger == b.deger)
            .collect();
        assert!(!ikinci.is_empty(), "2. kayit hic okunmadi");
        assert!(
            ikinci.iter().all(|v| *v),
            "kayit sinirini gecen okuma komsudan etkilendi: {ikinci:?}"
        );
        // ...ve degisiklik gercekten gecerli olmali: 1. kaydin kendi okumasi degisir.
        let birinci_degisti = ozet
            .okumalar
            .iter()
            .zip(ozet2.okumalar.iter())
            .filter(|(a, _)| a.konum < 2)
            .any(|(a, b)| a.hucre != b.hucre);
        assert!(
            birinci_degisti,
            "kendi kaydinin okumasi da degismedi: olcum etkisiz"
        );
    }

    #[test]
    fn gradyan_sonlu_farkla_uyusur() {
        let spec = EngramSpec::yeni(2, 16, 3).expect("gecerli sekil");
        let tablo = belirgin_doldur(spec, 20260926);
        let tokens = tohumlu_jeton(9, 7, 5);
        let kaynak = [3u32; 9];
        let ozet = oku(spec, &tablo, &tokens, &kaynak);
        assert!(!ozet.okumalar.is_empty(), "hic okuma yok");
        // Kayip = Σ anahtar·deger; gradyanlar: dL/danahtar = deger, dL/ddeger = anahtar.
        let katkilar: Vec<EngramKatkisi> = ozet
            .okumalar
            .iter()
            .map(|o| EngramKatkisi {
                hucre: o.hucre,
                anahtar: o.deger.clone(),
                deger: o.anahtar.clone(),
            })
            .collect();
        let g = geri(spec, &katkilar);
        assert_eq!(g.len(), spec.parametre_sayisi());
        let h = 1e-6;
        let mut denetlenen = 0usize;
        let mut en_kotu = 0.0f64;
        for i in 0..spec.parametre_sayisi() {
            let mut arti = tablo.clone();
            let mut eksi = tablo.clone();
            arti[i] += h;
            eksi[i] -= h;
            let sonlu = (kayip(&oku(spec, &arti, &tokens, &kaynak).okumalar)
                - kayip(&oku(spec, &eksi, &tokens, &kaynak).okumalar))
                / (2.0 * h);
            let analitik = g[i];
            let sinir = GRADIENT_CHECK_MUTLAK_TABAN
                + GRADIENT_CHECK_TOLERANCE * analitik.abs().max(sonlu.abs());
            assert!(
                (analitik - sonlu).abs() <= sinir,
                "tablo[{i}]: analitik {analitik}, sonlu {sonlu}"
            );
            en_kotu = en_kotu.max((analitik - sonlu).abs());
            denetlenen += 1;
        }
        // Sayim bagi: denetlenen gradyan sayisi seklin parametre sayisina bagli.
        if denetlenen != spec.parametre_sayisi() {
            assert_eq!(denetlenen, spec.parametre_sayisi(), "denetim sayisi sasti");
        }
        assert!(en_kotu < 1e-6, "en kotu sapma {en_kotu}");
        // Okunmayan hucreler sifir gradyan almali: seyrek toplayici.
        let okunan: Vec<usize> = ozet.okumalar.iter().map(|o| o.hucre).collect();
        let mut sifir_hucre = None;
        for c in 0..spec.tablo {
            if !okunan.contains(&c) {
                sifir_hucre = Some(c);
                break;
            }
        }
        if let Some(c) = sifir_hucre {
            assert_eq!(g[spec.anahtar_tabani(c)].to_bits(), 0.0f64.to_bits());
            assert_eq!(g[spec.deger_tabani(c)].to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn seyrek_toplam_hucre_sirasinda() {
        let spec = EngramSpec::yeni(2, 8, 2).expect("gecerli sekil");
        let katkilar = vec![
            EngramKatkisi {
                hucre: 3,
                anahtar: vec![0.1, 0.2],
                deger: vec![0.3, 0.4],
            },
            EngramKatkisi {
                hucre: 1,
                anahtar: vec![0.5, 0.6],
                deger: vec![0.7, 0.8],
            },
            EngramKatkisi {
                hucre: 3,
                anahtar: vec![0.9, 1.0],
                deger: vec![1.1, 1.2],
            },
        ];
        let mut ters = katkilar.clone();
        ters.reverse();
        let g1 = geri(spec, &katkilar);
        let g2 = geri(spec, &ters);
        for (a, b) in g1.iter().zip(g2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "toplama sirasi sonucu degistirdi");
        }
        // 3. hucrede iki katki var; toplam dogru mu?
        let k_bas = spec.anahtar_tabani(3);
        let beklenen = (0.1f64 + 0.9).to_bits();
        assert_eq!(g1[k_bas].to_bits(), beklenen);
        assert_eq!(g1[spec.anahtar_tabani(1)].to_bits(), 0.5f64.to_bits());
    }

    #[test]
    fn cakisma_olculur_yasaklanmaz() {
        // Kucuk tablo, cok n-gram: cakisma beklenir ve olculur.
        let spec = EngramSpec::yeni(1, 4, 2).expect("gecerli sekil");
        let tokens = tohumlu_jeton(64, 3, 4);
        let kaynak = [1u32; 64];
        let c = cakisma_olc(spec, &tokens, &kaynak);
        assert_eq!(c.ngram, 64);
        assert!(c.dolu_hucre <= spec.tablo, "dolu hucre tabloyu asti");
        assert!(c.cakisan_hucre <= c.dolu_hucre);
        assert!(c.cakisan_hucre > 0, "kucuk tabloda cakisma beklenirdi");
        assert!((c.doluluk - c.dolu_hucre as f64 / spec.tablo as f64).abs() < 1e-12);
        // Kucuk tablo ile buyuk tablo: ayni n-gram kumesi, daha cok hucre.
        let buyuk = EngramSpec::yeni(1, 4096, 2).expect("gecerli sekil");
        let c2 = cakisma_olc(buyuk, &tokens, &kaynak);
        assert_eq!(c2.ngram, c.ngram);
        assert!(c2.dolu_hucre >= c.dolu_hucre);
    }

    #[test]
    fn okuma_anahtar_deger_dilimlerinden_gelir() {
        let spec = EngramSpec::yeni(1, 4, 2).expect("gecerli sekil");
        let tablo = belirgin_doldur(spec, 11);
        let tokens = [2usize];
        let kaynak = [0u32];
        let ozet = oku(spec, &tablo, &tokens, &kaynak);
        assert_eq!(ozet.okumalar.len(), 1);
        let okuma = &ozet.okumalar[0];
        let k_bas = spec.anahtar_tabani(okuma.hucre);
        let v_bas = spec.deger_tabani(okuma.hucre);
        assert_eq!(okuma.anahtar, tablo[k_bas..k_bas + spec.d_kv].to_vec());
        assert_eq!(okuma.deger, tablo[v_bas..v_bas + spec.d_kv].to_vec());
    }

    /// Tek olcum satiri: tablo muhasebesi, seyreklik, cakisma ve gradyan
    /// sapmasi. Sayilar bu satirdan kayda gecer (`training/engram.py`).
    #[test]
    fn olcum_raporu() {
        let spec = EngramSpec::yeni(3, 512, 16).expect("gecerli sekil");
        let tablo = belirgin_doldur(spec, 4242);
        let tokens = tohumlu_jeton(256, 99, 64);
        let kaynak = [1u32; 256];
        let ozet = oku(spec, &tablo, &tokens, &kaynak);
        let katkilar: Vec<EngramKatkisi> = ozet
            .okumalar
            .iter()
            .map(|o| EngramKatkisi {
                hucre: o.hucre,
                anahtar: o.deger.clone(),
                deger: o.anahtar.clone(),
            })
            .collect();
        let g = geri(spec, &katkilar);
        let h = 1e-6;
        let mut en_kotu = 0.0f64;
        let ornek: Vec<usize> = (0..spec.parametre_sayisi()).step_by(37).collect();
        for i in ornek {
            let mut arti = tablo.clone();
            let mut eksi = tablo.clone();
            arti[i] += h;
            eksi[i] -= h;
            let sonlu = (kayip(&oku(spec, &arti, &tokens, &kaynak).okumalar)
                - kayip(&oku(spec, &eksi, &tokens, &kaynak).okumalar))
                / (2.0 * h);
            en_kotu = en_kotu.max((g[i] - sonlu).abs());
        }
        let c = cakisma_olc(spec, &tokens, &kaynak);
        println!(
            "engram | parametre={} okuma={} atlanan_gecmis={} atlanan_kayit_siniri={} tablo={} dolu_hucre={} cakisan_hucre={} doluluk={:.6} grad_sapma={:.3e}",
            spec.parametre_sayisi(),
            ozet.okumalar.len(),
            ozet.gecmis_yok,
            ozet.kayit_siniri,
            spec.tablo,
            c.dolu_hucre,
            c.cakisan_hucre,
            c.doluluk,
            en_kotu
        );
    }
}
