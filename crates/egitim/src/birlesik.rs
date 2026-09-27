//! Birlesik port blogu: alti tekil aday tek bir blokta.
//!
//! # Neden bu modul var
//!
//! Port bilesen bilesen bitmisti ve **kompozisyon olarak bitmemisti**.
//! `mlp_hadamard`, `engram`, `cok_serit`, `yonlendirme`, `normalizasyon` ve
//! `kesit` ayri ayri yazildi, ayri ayri olculdu, ayri ayri kapilandi; ama
//! `docs/CRATES.md` her birinin yaninda ayni cumleyi tasiyordu: *bagli degil*.
//! Alti modulun her biri kendi basina dogru olabilir ve birlesik hâlleri yine
//! de calismayabilir - sekiller tutmayabilir, gradyan bir yerde kopabilir,
//! parametre muhasebesi cift sayabilir. Bu modul o bosluktur: bileseni degil
//! **birlesimi** olcer.
//!
//! # Blogun sirasi
//!
//! ```text
//! serit durumu -> okuma -> sifir merkezli RMS norm -> rotali Hadamard
//!   uzmanlari -> engram deger bellegi -> serit yazma -> yeni serit durumu
//! ```
//!
//! Her ok gercek bir cagridir; hicbir adim burada yeniden yazilmadi. Okuma
//! [`cok_serit::okuma`], norm [`normalizasyon::norm_ileri`], uzman
//! [`mlp_hadamard::ileri`], rota [`yonlendirme::rota_hesapla`], bellek
//! [`engram::oku`], yazma [`cok_serit::ileri`]. Kompozisyonun kendi
//! aritmetigi yalniz uc yerdedir ve ucu de burada yazili: uzman ciktilarinin
//! agirlikli toplami, engram katkisinin blok cikisina eklenmesi, ve yazma
//! yolundan gelen `dL/dy`.
//!
//! # Kapali bilesen = yoklugu (olculur, iddia edilmez)
//!
//! Deponun kendi deseni burada da gecerli: bir bilesen kapatildiginda sonuc,
//! o bilesen **hic yazilmamis** hâlin bit-ozdesi olmalidir. Uc anahtar vardir
//! ve ucu de `f64::to_bits` ile olculur:
//!
//! * `serit = 1` -> tek akisli klasik artık baglanti;
//! * `rota = None` -> tek uzman, agirlik carpimi yok;
//! * `engram = None` -> bellek terimi hic hesaplanmaz.
//!
//! # Rotalama gradyani: yazilmadi ve iddia edilmiyor
//!
//! Rotalama puanlari blogun **disindan** gelir (cagiran verir); rotanin kendisi
//! parametre tutmaz (`RotaSpec::parametre_sayisi() == 0`). Bunun somut sonucu
//! sudur: agirliklar bu blogun hicbir parametresine bagli olmadigi icin
//! buradaki butun parametre gradyanlari, rota acikken de **tam**dir - sonlu
//! farkla olculur. Yazilmayan tek sey `dL/dpuanlar`'dir; [`geri`] onu
//! dondurmez ve bu modul onu hesapladigini soylemez.
//!
//! # Bu modulun degistirmedikleri
//!
//! `training/model_spec.json` degismez, `lubot-a1` ailesi degismez, hicbir
//! egitim cagrisi bu modulden gecmez. Hangi bilesenin hangi aileye girecegi
//! (`docs/MIMARI-TASARIM.md` M1/M2/M3) isaretli mimari karardir ve bu modul o
//! karari **vermez**: bir aile degil, bir kompozisyon yuzeyi kurar.

use crate::cok_serit::{self, CokSeritSpec, SeritAgirliklar, SeritDurumu};
use crate::engram::{self, EngramKatkisi, EngramOkuma, EngramSpec};
use crate::mlp_hadamard::{self, HadamardAgirliklar, HadamardBellek, HadamardSpec};
use crate::normalizasyon::{self, NormSpec};
use crate::yonlendirme::{self, RotaSpec};

/// Blogun sekli: alti adayin sekilleri tek yerde.
#[derive(Debug, Clone, PartialEq)]
pub struct BirlesikSpec {
    /// Model genisligi; uc alt seklin de genisligiyle ayni olmali.
    pub d_model: usize,
    /// Artık akis sekli (`serit = 1` klasik tek akis).
    pub serit: CokSeritSpec,
    /// Sifir merkezli RMS norm sekli.
    pub norm: NormSpec,
    /// Uzman basina Hadamard MLP sekli (butun uzmanlar ayni sekli tasir).
    pub hadamard: HadamardSpec,
    /// Rotalama sekli; `None` ise tek uzman vardir ve rota hic kosmaz.
    pub rota: Option<RotaSpec>,
    /// Engram bellegi; `None` ise bellek terimi hic hesaplanmaz.
    pub engram: Option<EngramSpec>,
}

/// Neden bir sekil ya da bir cagri reddedildi.
///
/// Reddetme noktalari bilerek erken: sekil tutmuyorsa tahsis bile yapilmaz.
#[derive(Debug, Clone, PartialEq)]
pub enum BirlesikHatasi {
    /// Genislik sifir.
    SifirGenislik,
    /// Alt seklin genisligi blogun genisligiyle ayni degil.
    GenislikUyusmuyor {
        /// Hangi alt sekil.
        alan: &'static str,
        /// Alt seklin genisligi.
        bulunan: usize,
        /// Blogun genisligi.
        beklenen: usize,
    },
    /// Engram anahtar/deger genisligi model genisliginden buyuk: kapi
    /// vektoru `h`'nin ilk `d_kv` kanalindan okunur, o yuzden sigmali.
    EngramGenisligiBuyuk(usize, usize),
    /// Uzman sayisi ile verilen uzman agirlik kumesi sayisi ayni degil.
    UzmanSayisiUymuyor(usize, usize),
    /// Bir vektorun uzunlugu sekle uymuyor; alan adiyla birlikte.
    UzunlukUyusmuyor {
        /// Hangi vektor.
        alan: &'static str,
        /// Gelen uzunluk.
        bulunan: usize,
        /// Sekilden turetilen uzunluk.
        beklenen: usize,
    },
    /// Rotalama acikken puan matrisi verilmedi.
    PuanYok,
    /// Alt modulun kendi reddi, adiyla birlikte tasinir.
    AltModul(&'static str, String),
}

impl BirlesikSpec {
    /// Olcum ve test icin kucuk, deterministik bir sekil: her bilesen acik.
    ///
    /// Sayilar kucuk secildi cunku bu seklin uzerinde her parametre merkezi
    /// sonlu farkla denetlenir; buyuk sekil olcumu degil sabri olcer.
    ///
    /// # Errors
    /// Alt sekillerden biri kendi kuralini reddederse.
    pub fn kucuk_aday() -> Result<Self, BirlesikHatasi> {
        let d_model = 8;
        Ok(Self {
            d_model,
            serit: CokSeritSpec::yeni(d_model, 2).map_err(|e| alt("cok_serit", e))?,
            norm: NormSpec::yeni(d_model, 1e-6).map_err(|e| alt("normalizasyon", e))?,
            hadamard: HadamardSpec {
                d_model,
                d_r: 8,
                blok: 2,
            },
            rota: Some(RotaSpec::yeni(3, 2, 1.0, 4).map_err(|e| alt("yonlendirme", e))?),
            engram: Some(EngramSpec::yeni(2, 5, 4).map_err(|e| alt("engram", e))?),
        })
    }

    /// Uzman sayisi: rota kapaliysa tektir.
    #[must_use]
    pub fn uzman_sayisi(&self) -> usize {
        self.rota.as_ref().map_or(1, |r| r.uzman)
    }

    /// Engram izdusumunun eleman sayisi (`d_model x d_kv`), kapaliysa sifir.
    #[must_use]
    pub fn engram_izdusum_sayisi(&self) -> usize {
        self.engram.map_or(0, |e| self.d_model * e.d_kv)
    }

    /// Sekli dogrular: alt sekiller kendi kurallarini, blok da aralarindaki
    /// tutarliligi denetler.
    ///
    /// # Errors
    /// [`BirlesikHatasi`] — hangi alanin neden reddedildigini tasir.
    pub fn dogrula(&self) -> Result<(), BirlesikHatasi> {
        if self.d_model == 0 {
            return Err(BirlesikHatasi::SifirGenislik);
        }
        CokSeritSpec::yeni(self.serit.d_model, self.serit.serit)
            .map_err(|e| alt("cok_serit", e))?;
        NormSpec::yeni(self.norm.genislik, self.norm.eps).map_err(|e| alt("normalizasyon", e))?;
        self.hadamard
            .dogrula()
            .map_err(|e| alt("mlp_hadamard", e))?;
        if let Some(r) = &self.rota {
            RotaSpec::yeni(r.uzman, r.k, r.sicaklik, r.yineleme)
                .map_err(|e| alt("yonlendirme", e))?;
        }
        if let Some(e) = self.engram {
            EngramSpec::yeni(e.n, e.tablo, e.d_kv).map_err(|x| alt("engram", x))?;
            if e.d_kv > self.d_model {
                return Err(BirlesikHatasi::EngramGenisligiBuyuk(e.d_kv, self.d_model));
            }
        }
        for (alan, bulunan) in [
            ("cok_serit", self.serit.d_model),
            ("normalizasyon", self.norm.genislik),
            ("mlp_hadamard", self.hadamard.d_model),
        ] {
            if bulunan != self.d_model {
                return Err(BirlesikHatasi::GenislikUyusmuyor {
                    alan,
                    bulunan,
                    beklenen: self.d_model,
                });
            }
        }
        Ok(())
    }

    /// Sekilden turetilen parametre sayisi.
    ///
    /// Toplam, alt modullerin kendi sayimlarindan kurulur; burada tek bir sabit
    /// yoktur. Rotalama sifir tasir ve terim olarak **yazilir**: toplamdan sessizce
    /// dusmek, rotanin parametre tutmadigi bilgisini gorunmez yapardi.
    #[must_use]
    pub fn parametre_sayisi(&self) -> usize {
        let rota = self.rota.as_ref().map_or(0, RotaSpec::parametre_sayisi);
        self.serit.parametre_sayisi()
            + self.norm.parametre_sayisi()
            + self.uzman_sayisi() * self.hadamard.parametre_sayisi()
            + rota
            + self.engram.map_or(0, |e| e.parametre_sayisi())
            + self.engram_izdusum_sayisi()
    }
}

fn alt<E: core::fmt::Debug>(ad: &'static str, hata: E) -> BirlesikHatasi {
    BirlesikHatasi::AltModul(ad, format!("{hata:?}"))
}

/// Blogun butun agirliklari.
#[derive(Debug, Clone, PartialEq)]
pub struct BirlesikAgirliklar {
    /// Serit okuma/karisim/yazma agirliklari.
    pub serit: SeritAgirliklar,
    /// Norm olcegi (`d_model`).
    pub norm_olcek: Vec<f64>,
    /// Uzman basina bir Hadamard agirlik kumesi.
    pub uzmanlar: Vec<HadamardAgirliklar>,
    /// Engram tablosu (`2 * tablo * d_kv`); kapaliysa bos.
    pub engram_tablo: Vec<f64>,
    /// Engram degerini model genisligine tasiyan izdusum (`d_model x d_kv`);
    /// kapaliysa bos.
    pub engram_izdusum: Vec<f64>,
}

/// Ileri gecisin geri gecis icin sakladigi her sey.
#[derive(Debug, Clone, PartialEq)]
pub struct BirlesikBellek {
    /// Pencere uzunlugu (konum sayisi).
    pub t: usize,
    /// Serit okumasi, `t x d_model`.
    pub okuma: Vec<f64>,
    /// Norm cikisi, `t x d_model`.
    pub h: Vec<f64>,
    /// Rotalama agirliklari, `t x uzman` (secilmeyen hucre sifir).
    pub agirlik: Vec<f64>,
    /// Kullanilan uzmanlarin ileri gecis bellekleri; kullanilmayan `None`.
    pub uzman_bellek: Vec<Option<HadamardBellek>>,
    /// Kullanilan uzmanlarin ciktilari (`t x d_model`); kullanilmayan bos.
    pub uzman_cikti: Vec<Vec<f64>>,
    /// Engram okumalari (konum, hucre, anahtar, deger).
    pub engram_okumalar: Vec<EngramOkuma>,
    /// Okuma basina kapi skalari `s`.
    pub engram_kapi: Vec<f64>,
    /// Blok cikisi `y`, `t x d_model` (serit yazmasina giren).
    pub y: Vec<f64>,
}

/// Engram okumasinin ozet sayimlari: kac konum atlandi ve neden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngramSayim {
    /// Okunan konum sayisi.
    pub okunan: usize,
    /// Gecmisi `n` jetona yetmeyen konum sayisi.
    pub gecmis_yok: usize,
    /// Kayit sinirini gectigi icin atlanan konum sayisi.
    pub kayit_siniri: usize,
}

/// Ileri gecisin girdileri tek yerde.
///
/// Dort alan ayri argüman olarak da tasinabilirdi; yapi halinde tasinmalarinin
/// sebebi cagri yerinde sirayi karistirmanin sessiz olmasidir: `tokens` ile
/// `kaynak` ayni uzunlukta iki dizidir ve yer degistirdiklerinde kod calisir,
/// yalniz sonuc yanlis cikar. Alan adi bu hatayi derleme zamanina tasir.
#[derive(Debug, Clone, Copy)]
pub struct BirlesikGirdi<'a> {
    /// Konum basina serit durumu; uzunlugu pencere uzunlugunu belirler.
    pub durumlar: &'a [SeritDurumu],
    /// Engram adresi icin jeton dizisi (engram kapaliysa bos gecilebilir).
    pub tokens: &'a [usize],
    /// Konum basina kayit kimligi; engram okumasi kayit sinirini gecemez.
    pub kaynak: &'a [u32],
    /// Rotalama puanlari (`t x uzman`, satir major); rota acikken zorunlu.
    pub puanlar: Option<&'a [f64]>,
}

/// Ileri gecisin ciktisi.
#[derive(Debug, Clone, PartialEq)]
pub struct BirlesikCikti {
    /// Yeni serit durumlari, konum basina bir tane.
    pub durumlar: Vec<SeritDurumu>,
    /// Geri gecisin istedigi ara degerler.
    pub bellek: BirlesikBellek,
    /// Engram okuma sayimlari (kapaliysa hepsi sifir).
    pub engram: EngramSayim,
}

/// Geri gecisin urettigi gradyanlar; sekli agirliklarin aynisi.
#[derive(Debug, Clone, PartialEq)]
pub struct BirlesikGradyanlar {
    /// `dL/dalfa`.
    pub serit_alfa: Vec<f64>,
    /// `dL/dkarisim`.
    pub serit_karisim: Vec<f64>,
    /// `dL/dbeta`.
    pub serit_beta: Vec<f64>,
    /// `dL/dnorm_olcek`.
    pub norm_olcek: Vec<f64>,
    /// Uzman basina `dL/dW`.
    pub uzmanlar: Vec<mlp_hadamard::HadamardGradyanlar>,
    /// `dL/dengram_tablo` (seyrek toplanmis, hucre sirasinda).
    pub engram_tablo: Vec<f64>,
    /// `dL/dengram_izdusum`.
    pub engram_izdusum: Vec<f64>,
    /// `dL/dS` — giren serit durumlarina gore, konum basina `serit * d_model`.
    pub durumlar: Vec<f64>,
}

/// Kucuk, deterministik dolgu: ayni tohum her makinede ayni sayilari verir.
///
/// Kimlik/sifir baslangic bilerek **secilmedi** burada: kompozisyonun gradyani
/// ancak butun kollar sifirdan farkliyken denetlenebilir. Kapali-bilesen
/// bit-ozdeslik testleri ise ayri sekillerle kosar.
#[must_use]
pub fn belirgin_doldur(spec: &BirlesikSpec, tohum: u64) -> BirlesikAgirliklar {
    let k = spec.serit.serit;
    let mut sayac = tohum;
    let mut sonraki = move || {
        sayac = sayac
            .wrapping_mul(0x5851_F42D_4C95_7F2D)
            .wrapping_add(0x1405_7B7E_F767_814F);
        let ham = ((sayac >> 33) as f64) / ((1u64 << 31) as f64);
        (ham - 0.5) * 0.5
    };
    let alfa: Vec<f64> = (0..k).map(|_| 0.5 + sonraki()).collect();
    let karisim: Vec<f64> = (0..k * k)
        .map(|i| if i % (k + 1) == 0 { 1.0 } else { 0.0 } + sonraki())
        .collect();
    let beta: Vec<f64> = (0..k).map(|_| 0.5 + sonraki()).collect();
    let norm_olcek: Vec<f64> = (0..spec.d_model).map(|_| sonraki()).collect();
    let uzmanlar: Vec<HadamardAgirliklar> = (0..spec.uzman_sayisi())
        .map(|e| mlp_hadamard::belirgin_doldur(spec.hadamard, tohum.wrapping_add(e as u64 + 1)))
        .collect();
    let (engram_tablo, engram_izdusum) = match spec.engram {
        Some(e) => (
            engram::belirgin_doldur(e, tohum.wrapping_add(97)),
            (0..spec.d_model * e.d_kv).map(|_| sonraki()).collect(),
        ),
        None => (Vec::new(), Vec::new()),
    };
    BirlesikAgirliklar {
        serit: SeritAgirliklar {
            alfa,
            karisim,
            beta,
        },
        norm_olcek,
        uzmanlar,
        engram_tablo,
        engram_izdusum,
    }
}

/// Tohumsuz, deterministik baslangic serit durumlari (`t` konum).
#[must_use]
pub fn belirgin_durumlar(spec: &BirlesikSpec, t: usize, tohum: u64) -> Vec<SeritDurumu> {
    let mut sayac = tohum;
    (0..t)
        .map(|_| {
            let satirlar = (0..spec.serit.serit * spec.d_model)
                .map(|_| {
                    sayac = sayac
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(0x6C8E_9CF5_7003_9C1D);
                    (((sayac >> 33) as f64) / ((1u64 << 31) as f64) - 0.5) * 2.0
                })
                .collect();
            SeritDurumu { satirlar }
        })
        .collect()
}

fn uzunluk(alan: &'static str, bulunan: usize, beklenen: usize) -> Result<(), BirlesikHatasi> {
    if bulunan == beklenen {
        Ok(())
    } else {
        Err(BirlesikHatasi::UzunlukUyusmuyor {
            alan,
            bulunan,
            beklenen,
        })
    }
}

fn sekil_denetle(
    spec: &BirlesikSpec,
    w: &BirlesikAgirliklar,
    durumlar: &[SeritDurumu],
) -> Result<(), BirlesikHatasi> {
    spec.dogrula()?;
    let k = spec.serit.serit;
    uzunluk("serit.alfa", w.serit.alfa.len(), k)?;
    uzunluk("serit.karisim", w.serit.karisim.len(), k * k)?;
    uzunluk("serit.beta", w.serit.beta.len(), k)?;
    uzunluk("norm_olcek", w.norm_olcek.len(), spec.d_model)?;
    if w.uzmanlar.len() != spec.uzman_sayisi() {
        return Err(BirlesikHatasi::UzmanSayisiUymuyor(
            w.uzmanlar.len(),
            spec.uzman_sayisi(),
        ));
    }
    uzunluk(
        "engram_tablo",
        w.engram_tablo.len(),
        spec.engram.map_or(0, |e| e.parametre_sayisi()),
    )?;
    uzunluk(
        "engram_izdusum",
        w.engram_izdusum.len(),
        spec.engram_izdusum_sayisi(),
    )?;
    for durum in durumlar {
        uzunluk("durum", durum.satirlar.len(), k * spec.d_model)?;
    }
    Ok(())
}

/// Rotalama agirliklarini `t x uzman` duz matrise acar; rota kapaliysa tek sutun.
fn agirlik_matrisi(
    spec: &BirlesikSpec,
    t: usize,
    puanlar: Option<&[f64]>,
) -> Result<Vec<f64>, BirlesikHatasi> {
    let Some(rota) = spec.rota.as_ref() else {
        return Ok(vec![1.0; t]);
    };
    let Some(puanlar) = puanlar else {
        return Err(BirlesikHatasi::PuanYok);
    };
    uzunluk("puanlar", puanlar.len(), t * rota.uzman)?;
    let sonuc = yonlendirme::rota_hesapla(puanlar, rota).map_err(|e| alt("yonlendirme", e))?;
    let mut a = vec![0.0f64; t * rota.uzman];
    for secim in &sonuc.secimler {
        if let Some(h) = a.get_mut(secim.jeton * rota.uzman + secim.uzman) {
            *h = secim.agirlik;
        }
    }
    Ok(a)
}

/// Ileri gecis.
///
/// Girdinin dort alani [`BirlesikGirdi`]'de; pencere uzunlugu
/// `girdi.durumlar` uzunlugundan turetilir.
///
/// # Errors
/// [`BirlesikHatasi`] — sekil tutmuyorsa, rota acikken puan yoksa ya da bir
/// alt modul kendi girdisini reddederse. Hicbir durumda yaklasik bir sonuc
/// uretilmez.
pub fn ileri(
    spec: &BirlesikSpec,
    w: &BirlesikAgirliklar,
    girdi: &BirlesikGirdi<'_>,
) -> Result<BirlesikCikti, BirlesikHatasi> {
    let BirlesikGirdi {
        durumlar,
        tokens,
        kaynak,
        puanlar,
    } = *girdi;
    sekil_denetle(spec, w, durumlar)?;
    let d = spec.d_model;
    let t = durumlar.len();
    let uzman = spec.uzman_sayisi();

    // 1) Serit okumasi ve norm.
    let mut okuma = Vec::with_capacity(t * d);
    let mut h = Vec::with_capacity(t * d);
    for durum in durumlar {
        let x = cok_serit::okuma(&spec.serit, &w.serit, durum);
        let cikti = normalizasyon::norm_ileri(spec.norm, &x, &w.norm_olcek)
            .map_err(|e| alt("normalizasyon", e))?;
        okuma.extend_from_slice(&x);
        h.extend_from_slice(&cikti.y);
    }

    // 2) Rotalama agirliklari (bu blogun parametresi degil: disaridan gelen puan).
    let agirlik = agirlik_matrisi(spec, t, puanlar)?;

    // 3) Uzmanlar. Yalniz yuku olan uzman kosar; kosmayanin bellegi `None`
    //    kalir ve geri gecis onu atlar - bos tensor uretip carpmak, sifirla
    //    carpmanin maliyetini gizlemenin pahali yoludur.
    let mut uzman_bellek: Vec<Option<HadamardBellek>> = Vec::with_capacity(uzman);
    let mut uzman_cikti: Vec<Vec<f64>> = Vec::with_capacity(uzman);
    let mut y = vec![0.0f64; t * d];
    for e in 0..uzman {
        let yuklu = (0..t).any(|i| agirlik.get(i * uzman + e).copied().unwrap_or(0.0) != 0.0);
        if !yuklu {
            uzman_bellek.push(None);
            uzman_cikti.push(Vec::new());
            continue;
        }
        let agirliklar = w
            .uzmanlar
            .get(e)
            .ok_or(BirlesikHatasi::UzmanSayisiUymuyor(w.uzmanlar.len(), uzman))?;
        let (cikti, bellek) = mlp_hadamard::ileri(spec.hadamard, agirliklar, &h, t);
        for i in 0..t {
            let a = agirlik.get(i * uzman + e).copied().unwrap_or(0.0);
            if a == 0.0 {
                continue;
            }
            for o in 0..d {
                if let (Some(hedef), Some(deger)) = (y.get_mut(i * d + o), cikti.get(i * d + o)) {
                    *hedef += a * *deger;
                }
            }
        }
        uzman_bellek.push(Some(bellek));
        uzman_cikti.push(cikti);
    }

    // 4) Engram: deger bellegi, anahtardan gelen parametresiz kapi ile.
    let mut engram_okumalar = Vec::new();
    let mut engram_kapi = Vec::new();
    let mut sayim = EngramSayim::default();
    if let Some(espec) = spec.engram {
        let ozet = engram::oku(espec, &w.engram_tablo, tokens, kaynak);
        let olcek = (espec.d_kv as f64).sqrt();
        for okumasi in &ozet.okumalar {
            let konum = okumasi.konum;
            let mut s = 0.0f64;
            for j in 0..espec.d_kv {
                let p = h.get(konum * d + j).copied().unwrap_or(0.0);
                s += okumasi.anahtar.get(j).copied().unwrap_or(0.0) * p;
            }
            s /= olcek;
            for o in 0..d {
                let mut katki = 0.0f64;
                for j in 0..espec.d_kv {
                    katki += w
                        .engram_izdusum
                        .get(o * espec.d_kv + j)
                        .copied()
                        .unwrap_or(0.0)
                        * okumasi.deger.get(j).copied().unwrap_or(0.0);
                }
                if let Some(hedef) = y.get_mut(konum * d + o) {
                    *hedef += s * katki;
                }
            }
            engram_kapi.push(s);
        }
        sayim = EngramSayim {
            okunan: ozet.okumalar.len(),
            gecmis_yok: ozet.gecmis_yok,
            kayit_siniri: ozet.kayit_siniri,
        };
        engram_okumalar = ozet.okumalar;
    }

    // 5) Serit yazmasi.
    let mut yeni = Vec::with_capacity(t);
    for (i, durum) in durumlar.iter().enumerate() {
        let dilim = y.get(i * d..(i + 1) * d).unwrap_or(&[]);
        let (_, sonraki) = cok_serit::ileri(&spec.serit, &w.serit, durum, dilim);
        yeni.push(sonraki);
    }

    Ok(BirlesikCikti {
        durumlar: yeni,
        bellek: BirlesikBellek {
            t,
            okuma,
            h,
            agirlik,
            uzman_bellek,
            uzman_cikti,
            engram_okumalar,
            engram_kapi,
            y,
        },
        engram: sayim,
    })
}

/// Elle yazilmis geri gecis.
///
/// `d_durumlar` cikan serit durumlarina gore gradyandir (`t x serit x d_model`,
/// konum basina duz). Turevler, ileri gecisin sirasinin tersi:
///
/// ```text
/// dL/dy_i[o]      = Σ_l beta[l] · g_yeni[i][l][o]          (yazma yolundan)
/// dL/dizdusum[o][j] += gy[o] · s · deger[j]                 (engram)
/// dL/ddeger[j]      += s · Σ_o gy[o] · izdusum[o][j]
/// ds                 = Σ_j deger[j] · Σ_o gy[o] · izdusum[o][j]
/// dL/danahtar[j]    += ds · h[i][j] / sqrt(d_kv)
/// dL/dh[i][j]       += ds · anahtar[j] / sqrt(d_kv)   (j < d_kv)
/// dL/dy_e[i][o]     = agirlik[i][e] · gy[o]                 (uzmanlar)
/// ```
///
/// Geri kalan her sey alt modullerin kendi geri geciselerine devredilir.
/// `dL/dpuanlar` **yazilmadi**: rota puanlari blogun disindan gelir.
///
/// # Errors
/// [`BirlesikHatasi`] — sekil tutmuyorsa ya da bir alt modul reddederse.
pub fn geri(
    spec: &BirlesikSpec,
    w: &BirlesikAgirliklar,
    bellek: &BirlesikBellek,
    durumlar: &[SeritDurumu],
    d_durumlar: &[f64],
) -> Result<BirlesikGradyanlar, BirlesikHatasi> {
    sekil_denetle(spec, w, durumlar)?;
    let d = spec.d_model;
    let k = spec.serit.serit;
    let t = bellek.t;
    let uzman = spec.uzman_sayisi();
    uzunluk("d_durumlar", d_durumlar.len(), t * k * d)?;
    uzunluk("bellek.h", bellek.h.len(), t * d)?;

    // 1) Yazma yolundan `dL/dy`.
    let mut dy = vec![0.0f64; t * d];
    for i in 0..t {
        for l in 0..k {
            let beta = w.serit.beta.get(l).copied().unwrap_or(0.0);
            for o in 0..d {
                let g = d_durumlar
                    .get(i * k * d + l * d + o)
                    .copied()
                    .unwrap_or(0.0);
                if let Some(hedef) = dy.get_mut(i * d + o) {
                    *hedef += beta * g;
                }
            }
        }
    }

    let mut dh = vec![0.0f64; t * d];

    // 2) Engram geri gecisi.
    let mut d_izdusum = vec![0.0f64; spec.engram_izdusum_sayisi()];
    let mut d_tablo = vec![0.0f64; spec.engram.map_or(0, |e| e.parametre_sayisi())];
    if let Some(espec) = spec.engram {
        let olcek = (espec.d_kv as f64).sqrt();
        let mut katkilar: Vec<EngramKatkisi> = Vec::with_capacity(bellek.engram_okumalar.len());
        for (idx, okumasi) in bellek.engram_okumalar.iter().enumerate() {
            let konum = okumasi.konum;
            let s = bellek.engram_kapi.get(idx).copied().unwrap_or(0.0);
            // q[j] = Σ_o gy[o] · izdusum[o][j]
            let mut q = vec![0.0f64; espec.d_kv];
            for o in 0..d {
                let gy = dy.get(konum * d + o).copied().unwrap_or(0.0);
                if gy == 0.0 {
                    continue;
                }
                for j in 0..espec.d_kv {
                    let iz = w
                        .engram_izdusum
                        .get(o * espec.d_kv + j)
                        .copied()
                        .unwrap_or(0.0);
                    if let Some(hedef) = q.get_mut(j) {
                        *hedef += gy * iz;
                    }
                    let deger = okumasi.deger.get(j).copied().unwrap_or(0.0);
                    if let Some(hedef) = d_izdusum.get_mut(o * espec.d_kv + j) {
                        *hedef += gy * s * deger;
                    }
                }
            }
            let mut ds = 0.0f64;
            let mut d_deger = vec![0.0f64; espec.d_kv];
            for j in 0..espec.d_kv {
                let qj = q.get(j).copied().unwrap_or(0.0);
                let deger = okumasi.deger.get(j).copied().unwrap_or(0.0);
                ds += deger * qj;
                if let Some(hedef) = d_deger.get_mut(j) {
                    *hedef = s * qj;
                }
            }
            let mut d_anahtar = vec![0.0f64; espec.d_kv];
            for j in 0..espec.d_kv {
                let hj = bellek.h.get(konum * d + j).copied().unwrap_or(0.0);
                if let Some(hedef) = d_anahtar.get_mut(j) {
                    *hedef = ds * hj / olcek;
                }
                let anahtar = okumasi.anahtar.get(j).copied().unwrap_or(0.0);
                if let Some(hedef) = dh.get_mut(konum * d + j) {
                    *hedef += ds * anahtar / olcek;
                }
            }
            katkilar.push(EngramKatkisi {
                hucre: okumasi.hucre,
                anahtar: d_anahtar,
                deger: d_deger,
            });
        }
        d_tablo = engram::geri(espec, &katkilar);
    }

    // 3) Uzmanlarin geri gecisi.
    let mut d_uzmanlar: Vec<mlp_hadamard::HadamardGradyanlar> = Vec::with_capacity(uzman);
    for e in 0..uzman {
        let (Some(Some(bellek_e)), Some(agirliklar)) =
            (bellek.uzman_bellek.get(e), w.uzmanlar.get(e))
        else {
            d_uzmanlar.push(bos_hadamard_gradyani(spec));
            continue;
        };
        let mut d_cikti = vec![0.0f64; t * d];
        for i in 0..t {
            let a = bellek.agirlik.get(i * uzman + e).copied().unwrap_or(0.0);
            if a == 0.0 {
                continue;
            }
            for o in 0..d {
                if let (Some(hedef), Some(g)) = (d_cikti.get_mut(i * d + o), dy.get(i * d + o)) {
                    *hedef = a * *g;
                }
            }
        }
        let g = mlp_hadamard::geri(spec.hadamard, agirliklar, bellek_e, &d_cikti);
        for (yer, deger) in g.girdi.iter().enumerate() {
            if let Some(hedef) = dh.get_mut(yer) {
                *hedef += *deger;
            }
        }
        d_uzmanlar.push(g);
    }

    // 4) Norm ve serit geri gecisi, konum konum.
    let mut d_norm_olcek = vec![0.0f64; d];
    let mut d_alfa = vec![0.0f64; k];
    let mut d_karisim = vec![0.0f64; k * k];
    let mut d_beta = vec![0.0f64; k];
    let mut d_giren_durumlar = vec![0.0f64; t * k * d];
    for (i, durum) in durumlar.iter().enumerate() {
        let x = bellek.okuma.get(i * d..(i + 1) * d).unwrap_or(&[]);
        let dh_i = dh.get(i * d..(i + 1) * d).unwrap_or(&[]);
        let (d_okuma, d_olcek) = normalizasyon::norm_geri(spec.norm, x, &w.norm_olcek, dh_i)
            .map_err(|e| alt("normalizasyon", e))?;
        for (yer, deger) in d_olcek.iter().enumerate() {
            if let Some(hedef) = d_norm_olcek.get_mut(yer) {
                *hedef += *deger;
            }
        }
        let y_i = bellek.y.get(i * d..(i + 1) * d).unwrap_or(&[]);
        let g_durum = d_durumlar.get(i * k * d..(i + 1) * k * d).unwrap_or(&[]);
        let g = cok_serit::geri(&spec.serit, &w.serit, durum, y_i, &d_okuma, g_durum);
        for (yer, deger) in g.alfa.iter().enumerate() {
            if let Some(hedef) = d_alfa.get_mut(yer) {
                *hedef += *deger;
            }
        }
        for (yer, deger) in g.karisim.iter().enumerate() {
            if let Some(hedef) = d_karisim.get_mut(yer) {
                *hedef += *deger;
            }
        }
        for (yer, deger) in g.beta.iter().enumerate() {
            if let Some(hedef) = d_beta.get_mut(yer) {
                *hedef += *deger;
            }
        }
        for (yer, deger) in g.durum.iter().enumerate() {
            if let Some(hedef) = d_giren_durumlar.get_mut(i * k * d + yer) {
                *hedef = *deger;
            }
        }
    }

    Ok(BirlesikGradyanlar {
        serit_alfa: d_alfa,
        serit_karisim: d_karisim,
        serit_beta: d_beta,
        norm_olcek: d_norm_olcek,
        uzmanlar: d_uzmanlar,
        engram_tablo: d_tablo,
        engram_izdusum: d_izdusum,
        durumlar: d_giren_durumlar,
    })
}

fn bos_hadamard_gradyani(spec: &BirlesikSpec) -> mlp_hadamard::HadamardGradyanlar {
    let h = spec.hadamard;
    let blok_agirlik = h.blok * h.blok_girisi() * h.blok_ici();
    mlp_hadamard::HadamardGradyanlar {
        w1: vec![0.0; blok_agirlik],
        w2: vec![0.0; blok_agirlik],
        w3: vec![0.0; h.d_model * h.d_r],
        b1: vec![0.0; h.d_r],
        b2: vec![0.0; h.d_r],
        b3: vec![0.0; h.d_model],
        girdi: Vec::new(),
    }
}

/// Kayip: `0.5 * Σ (cikan durum - hedef)²` — crate'in diger aday modulleriyle
/// ayni bicim, boylece sonlu fark denetimi ayni tolerans tablosuna oturur.
#[must_use]
pub fn kayip(durumlar: &[SeritDurumu], hedef: &[f64]) -> f64 {
    let mut toplam = 0.0f64;
    let mut yer = 0usize;
    for durum in durumlar {
        for deger in &durum.satirlar {
            let h = hedef.get(yer).copied().unwrap_or(0.0);
            toplam += 0.5 * (deger - h) * (deger - h);
            yer += 1;
        }
    }
    toplam
}

/// Kaybin cikan durumlara gore gradyani.
#[must_use]
pub fn kayip_gradyan(durumlar: &[SeritDurumu], hedef: &[f64]) -> Vec<f64> {
    let mut g = Vec::new();
    let mut yer = 0usize;
    for durum in durumlar {
        for deger in &durum.satirlar {
            g.push(deger - hedef.get(yer).copied().unwrap_or(0.0));
            yer += 1;
        }
    }
    g
}

/// Tek bir gradyan inisi adimi.
pub fn sgd_adimi(w: &mut BirlesikAgirliklar, g: &BirlesikGradyanlar, lr: f64) {
    fn dus(hedef: &mut [f64], grad: &[f64], lr: f64) {
        for (h, d) in hedef.iter_mut().zip(grad.iter()) {
            *h -= lr * *d;
        }
    }
    dus(&mut w.serit.alfa, &g.serit_alfa, lr);
    dus(&mut w.serit.karisim, &g.serit_karisim, lr);
    dus(&mut w.serit.beta, &g.serit_beta, lr);
    dus(&mut w.norm_olcek, &g.norm_olcek, lr);
    for (uzman, grad) in w.uzmanlar.iter_mut().zip(g.uzmanlar.iter()) {
        mlp_hadamard::sgd_adimi(uzman, grad, lr);
    }
    dus(&mut w.engram_tablo, &g.engram_tablo, lr);
    dus(&mut w.engram_izdusum, &g.engram_izdusum, lr);
}

/// Butun skaler parametrelere duz erisim: `(ad, indeks)` sirasi sabittir ve
/// sekilden turetilir. Sonlu fark denetimi bu sirayi kullanir, boylece sonradan
/// eklenen bir tensor denetimin disinda kalamaz.
#[must_use]
pub fn parametre_yollari(spec: &BirlesikSpec) -> Vec<(&'static str, usize, usize)> {
    let h = spec.hadamard;
    let blok_agirlik = h.blok * h.blok_girisi() * h.blok_ici();
    let mut yollar = vec![
        ("serit.alfa", 0, spec.serit.serit),
        ("serit.karisim", 0, spec.serit.serit * spec.serit.serit),
        ("serit.beta", 0, spec.serit.serit),
        ("norm_olcek", 0, spec.d_model),
    ];
    for e in 0..spec.uzman_sayisi() {
        yollar.push(("uzman.w1", e, blok_agirlik));
        yollar.push(("uzman.w2", e, blok_agirlik));
        yollar.push(("uzman.w3", e, h.d_model * h.d_r));
        yollar.push(("uzman.b1", e, h.d_r));
        yollar.push(("uzman.b2", e, h.d_r));
        yollar.push(("uzman.b3", e, h.d_model));
    }
    yollar.push((
        "engram_tablo",
        0,
        spec.engram.map_or(0, |e| e.parametre_sayisi()),
    ));
    yollar.push(("engram_izdusum", 0, spec.engram_izdusum_sayisi()));
    yollar
}

fn parametre_dilimi<'a>(
    w: &'a mut BirlesikAgirliklar,
    ad: &str,
    uzman: usize,
) -> Option<&'a mut Vec<f64>> {
    match ad {
        "serit.alfa" => Some(&mut w.serit.alfa),
        "serit.karisim" => Some(&mut w.serit.karisim),
        "serit.beta" => Some(&mut w.serit.beta),
        "norm_olcek" => Some(&mut w.norm_olcek),
        "engram_tablo" => Some(&mut w.engram_tablo),
        "engram_izdusum" => Some(&mut w.engram_izdusum),
        "uzman.w1" => w.uzmanlar.get_mut(uzman).map(|u| &mut u.w1),
        "uzman.w2" => w.uzmanlar.get_mut(uzman).map(|u| &mut u.w2),
        "uzman.w3" => w.uzmanlar.get_mut(uzman).map(|u| &mut u.w3),
        "uzman.b1" => w.uzmanlar.get_mut(uzman).map(|u| &mut u.b1),
        "uzman.b2" => w.uzmanlar.get_mut(uzman).map(|u| &mut u.b2),
        "uzman.b3" => w.uzmanlar.get_mut(uzman).map(|u| &mut u.b3),
        _ => None,
    }
}

fn gradyan_dilimi<'a>(g: &'a BirlesikGradyanlar, ad: &str, uzman: usize) -> Option<&'a Vec<f64>> {
    match ad {
        "serit.alfa" => Some(&g.serit_alfa),
        "serit.karisim" => Some(&g.serit_karisim),
        "serit.beta" => Some(&g.serit_beta),
        "norm_olcek" => Some(&g.norm_olcek),
        "engram_tablo" => Some(&g.engram_tablo),
        "engram_izdusum" => Some(&g.engram_izdusum),
        "uzman.w1" => g.uzmanlar.get(uzman).map(|u| &u.w1),
        "uzman.w2" => g.uzmanlar.get(uzman).map(|u| &u.w2),
        "uzman.w3" => g.uzmanlar.get(uzman).map(|u| &u.w3),
        "uzman.b1" => g.uzmanlar.get(uzman).map(|u| &u.b1),
        "uzman.b2" => g.uzmanlar.get(uzman).map(|u| &u.b2),
        "uzman.b3" => g.uzmanlar.get(uzman).map(|u| &u.b3),
        _ => None,
    }
}

/// Sonlu fark denetiminin ozeti.
///
/// Olcut crate'in kendi tablosudur ve burada yeniden yazilmadi:
/// `fark <= GRADIENT_CHECK_MUTLAK_TABAN + GRADIENT_CHECK_TOLERANCE *
/// max(|analitik|, |sonlu|)`. [`GradyanDenetimi::en_kotu_oran`] bu sinira
/// gore normalize edilmis en kotu degerdir - birimsizdir ve **1'in altinda**
/// olmalidir. Ham sapmayi tek basina raporlamak, cozunmeyen kucuk
/// gradyanlarda gurultuyu hata gibi gostermenin klasik yoludur.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradyanDenetimi {
    /// Denetlenen skaler parametre sayisi.
    pub denetlenen: usize,
    /// Olcutu gecemeyen parametre sayisi.
    pub ihlal: usize,
    /// En kotu `fark / sinir` orani (1'in altinda olmali).
    pub en_kotu_oran: f64,
    /// En kotu oranin dustugu tensorun adi.
    pub en_kotu_alan: &'static str,
}

/// Bu kompozisyonun her parametresini sonlu farkla denetler.
///
/// Denetlenen sayi sekilden turetilir ([`parametre_yollari`]), yani sonradan
/// eklenen bir tensor sessizce denetimin disinda kalamaz. Tolerans crate'in
/// kendi tablosudur ve burada gevsetilmez.
///
/// # Neden iki noktali degil dort noktali fark
///
/// Iki noktali merkezi fark `O(h²)` kesme hatasi tasir. Alti bilesenli bir
/// kompozisyonda kayip yuzeyinin egriligi tek bir modulunkinden buyuk oldugu
/// icin bu kesme hatasi, crate'in `1e-6` bagil toleransinin **ustune**
/// cikabiliyor: olculdu, en kotu oran 1.233 idi ve sapma gradyanin degil
/// yontemin hatasiydi. Toleransi gevsetmek bu bilgiyi silerdi. Onun yerine
/// olcum aleti duzeltildi: dort noktali (Richardson) fark `O(h⁴)` kesme
/// hatasi tasir ve geriye yalniz yuvarlama kalir.
///
/// ```text
/// f'(x) ≈ [8(f(x+h) - f(x-h)) - (f(x+2h) - f(x-2h))] / (12h)
/// ```
///
/// Kesme hatasi bu kadar kucuk olunca adim **buyutulebilir** ve yuvarlama
/// hatasi (`O(eps·|f|/h)`) dusurulur. Nerede durulacagi tahmin degil olcum:
/// `adim_taramasi_u_egrisi_cizer` testi adimi `1e-2`'den `1e-5`'e tarar ve
/// olculen en kotu oran (sinira gore normalize) su egriyi cizer - `1.2e0`,
/// `1.0e-2`, `7.9e-3`, `3.2e-2`, `8.9e-2`, `1.6e0`. Iki ucu da kotu, ortasi
/// genis bir plato: analitik gradyan dogru oldugunda beklenen sekil budur.
/// Varsayilan adim `1e-3`, yani olculen platonun dibi.
///
/// # Errors
/// [`BirlesikHatasi`] — ileri ya da geri gecis reddederse.
pub fn gradyan_denetle(
    spec: &BirlesikSpec,
    w: &BirlesikAgirliklar,
    girdi: &BirlesikGirdi<'_>,
    hedef: &[f64],
    adim: f64,
) -> Result<GradyanDenetimi, BirlesikHatasi> {
    let ileri_cikti = ileri(spec, w, girdi)?;
    let g_cikis = kayip_gradyan(&ileri_cikti.durumlar, hedef);
    let analitik = geri(spec, w, &ileri_cikti.bellek, girdi.durumlar, &g_cikis)?;

    let mut denetlenen = 0usize;
    let mut ihlal = 0usize;
    let mut en_kotu_oran = 0.0f64;
    let mut en_kotu_alan = "yok";
    for (ad, uzman, uzunluk) in parametre_yollari(spec) {
        for idx in 0..uzunluk {
            let mut kayiplar = [0.0f64; 4];
            let mut yazilabilir = true;
            for (yer, kaydirma) in [adim, -adim, 2.0 * adim, -2.0 * adim].iter().enumerate() {
                let mut kopya = w.clone();
                {
                    let Some(dilim) = parametre_dilimi(&mut kopya, ad, uzman) else {
                        yazilabilir = false;
                        break;
                    };
                    let Some(hucre) = dilim.get_mut(idx) else {
                        yazilabilir = false;
                        break;
                    };
                    *hucre += *kaydirma;
                }
                if let Some(hedef_yer) = kayiplar.get_mut(yer) {
                    *hedef_yer = kayip(&ileri(spec, &kopya, girdi)?.durumlar, hedef);
                }
            }
            if !yazilabilir {
                continue;
            }
            let sayisal =
                (8.0 * (kayiplar[0] - kayiplar[1]) - (kayiplar[2] - kayiplar[3])) / (12.0 * adim);
            let cozum = gradyan_dilimi(&analitik, ad, uzman)
                .and_then(|v| v.get(idx))
                .copied()
                .unwrap_or(0.0);
            let fark = (cozum - sayisal).abs();
            let sinir = crate::GRADIENT_CHECK_MUTLAK_TABAN
                + crate::GRADIENT_CHECK_TOLERANCE * cozum.abs().max(sayisal.abs());
            let oran = fark / sinir;
            if oran > 1.0 {
                ihlal += 1;
            }
            if oran > en_kotu_oran {
                en_kotu_oran = oran;
                en_kotu_alan = ad;
            }
            denetlenen += 1;
        }
    }
    Ok(GradyanDenetimi {
        denetlenen,
        ihlal,
        en_kotu_oran,
        en_kotu_alan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test ortami: sahip olunan diziler. `girdi()` bunlardan odunc kurar.
    struct Ortam {
        durumlar: Vec<SeritDurumu>,
        tokens: Vec<usize>,
        kaynak: Vec<u32>,
        puanlar: Vec<f64>,
    }

    impl Ortam {
        fn girdi(&self) -> BirlesikGirdi<'_> {
            BirlesikGirdi {
                durumlar: &self.durumlar,
                tokens: &self.tokens,
                kaynak: &self.kaynak,
                puanlar: if self.puanlar.is_empty() {
                    None
                } else {
                    Some(&self.puanlar)
                },
            }
        }

        /// Butun konumlar tek kayitta: engram penceresi kapali kalmaz.
        fn tek_kayit(mut self) -> Self {
            self.kaynak = vec![0; self.kaynak.len()];
            self
        }
    }

    fn ortam(spec: &BirlesikSpec, t: usize) -> Ortam {
        Ortam {
            durumlar: belirgin_durumlar(spec, t, 12345),
            tokens: (0..t).map(|i| (i * 7 + 3) % 11).collect(),
            kaynak: (0..t).map(|i| u32::from(i >= t / 2)).collect(),
            puanlar: match spec.rota.as_ref() {
                Some(r) => yonlendirme::ornek_puanlar(t, r.uzman, 3),
                None => Vec::new(),
            },
        }
    }

    fn hedef_uret(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i % 5) as f64 - 2.0) * 0.25).collect()
    }

    #[test]
    fn sekil_tutmayan_alt_modul_reddedilir() {
        let mut spec = BirlesikSpec::kucuk_aday().unwrap();
        spec.norm.genislik = spec.d_model + 1;
        assert!(matches!(
            spec.dogrula(),
            Err(BirlesikHatasi::GenislikUyusmuyor {
                alan: "normalizasyon",
                ..
            })
        ));
    }

    #[test]
    fn engram_genisligi_modelden_buyuk_olamaz() {
        let mut spec = BirlesikSpec::kucuk_aday().unwrap();
        spec.engram = Some(EngramSpec::yeni(2, 5, spec.d_model + 1).unwrap());
        assert!(matches!(
            spec.dogrula(),
            Err(BirlesikHatasi::EngramGenisligiBuyuk(..))
        ));
    }

    #[test]
    fn rota_acikken_puansiz_cagri_reddedilir() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let w = belirgin_doldur(&spec, 7);
        let o = ortam(&spec, 6);
        let mut girdi = o.girdi();
        girdi.puanlar = None;
        assert_eq!(ileri(&spec, &w, &girdi), Err(BirlesikHatasi::PuanYok));
    }

    #[test]
    fn parametre_sayisi_bilesenlerin_toplami() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let engram = spec.engram.unwrap();
        let beklenen = spec.serit.parametre_sayisi()
            + spec.norm.parametre_sayisi()
            + spec.uzman_sayisi() * spec.hadamard.parametre_sayisi()
            + engram.parametre_sayisi()
            + spec.d_model * engram.d_kv;
        assert_eq!(spec.parametre_sayisi(), beklenen);
        // Rotanin sifir terimi toplami degistirmez ama yazilidir.
        assert_eq!(spec.rota.as_ref().unwrap().parametre_sayisi(), 0);
    }

    #[test]
    fn parametre_yollari_sayimi_sekle_bagli() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let toplam: usize = parametre_yollari(&spec).iter().map(|(_, _, n)| n).sum();
        assert_eq!(toplam, spec.parametre_sayisi());
    }

    #[test]
    fn dolgu_sekle_uyar() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let w = belirgin_doldur(&spec, 11);
        let o = ortam(&spec, 4);
        assert!(sekil_denetle(&spec, &w, &o.durumlar).is_ok());
    }

    #[test]
    fn ileri_tekrarlanabilir() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let w = belirgin_doldur(&spec, 5);
        let o = ortam(&spec, 6);
        let a = ileri(&spec, &w, &o.girdi()).unwrap();
        let b = ileri(&spec, &w, &o.girdi()).unwrap();
        for (x, y) in a.bellek.y.iter().zip(b.bellek.y.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn serit_okumasi_ileri_ile_bit_ozdes() {
        // `okuma` disari acildi; ikinci bir kopya olmadigini olcen test budur.
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let w = belirgin_doldur(&spec, 3);
        let o = ortam(&spec, 3);
        for durum in &o.durumlar {
            let a = cok_serit::okuma(&spec.serit, &w.serit, durum);
            let (b, _) = cok_serit::ileri(&spec.serit, &w.serit, durum, &vec![0.0; spec.d_model]);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }

    #[test]
    fn engram_kapali_hal_bit_ozdes() {
        let mut acik = BirlesikSpec::kucuk_aday().unwrap();
        acik.engram = None;
        let w = belirgin_doldur(&acik, 9);
        let o = ortam(&acik, 6);
        let a = ileri(&acik, &w, &o.girdi()).unwrap();
        // Ayni sekil, engram tablosu bos: bellek terimi hic hesaplanmaz, yani
        // cikti engram hic yazilmamis hâlin ta kendisidir.
        assert!(a.bellek.engram_okumalar.is_empty());
        assert_eq!(a.engram, EngramSayim::default());
        // Uzman toplamini elle kur ve bit-ozdesligi olc.
        let uzman = acik.uzman_sayisi();
        let mut elle = vec![0.0f64; a.bellek.t * acik.d_model];
        for e in 0..uzman {
            let cikti = &a.bellek.uzman_cikti[e];
            if cikti.is_empty() {
                continue;
            }
            for i in 0..a.bellek.t {
                let ag = a.bellek.agirlik[i * uzman + e];
                if ag == 0.0 {
                    continue;
                }
                for o in 0..acik.d_model {
                    elle[i * acik.d_model + o] += ag * cikti[i * acik.d_model + o];
                }
            }
        }
        for (x, y) in elle.iter().zip(a.bellek.y.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn tek_serit_klasik_artik_akis() {
        let mut spec = BirlesikSpec::kucuk_aday().unwrap();
        spec.serit = CokSeritSpec::yeni(spec.d_model, 1).unwrap();
        assert_eq!(spec.serit.parametre_sayisi(), 3);
        assert_eq!(
            spec.serit.parametre_sayisi(),
            spec.serit.tek_serit_parametre_sayisi()
        );
        let w = belirgin_doldur(&spec, 4);
        let o = ortam(&spec, 5);
        let cikti = ileri(&spec, &w, &o.girdi()).unwrap();
        // Tek akista okuma yalniz alfa[0] ile olceklenmis durumdur.
        for (i, durum) in o.durumlar.iter().enumerate() {
            for o in 0..spec.d_model {
                let beklenen = w.serit.alfa[0] * durum.satirlar[o];
                assert_eq!(
                    cikti.bellek.okuma[i * spec.d_model + o].to_bits(),
                    beklenen.to_bits()
                );
            }
        }
    }

    #[test]
    fn rota_kapali_hal_tek_uzman() {
        let mut spec = BirlesikSpec::kucuk_aday().unwrap();
        spec.rota = None;
        assert_eq!(spec.uzman_sayisi(), 1);
        let w = belirgin_doldur(&spec, 8);
        let o = ortam(&spec, 5);
        let cikti = ileri(&spec, &w, &o.girdi()).unwrap();
        assert!(cikti.bellek.agirlik.iter().all(|a| *a == 1.0));
        assert_eq!(cikti.bellek.agirlik.len(), cikti.bellek.t);
    }

    #[test]
    fn engram_kayit_sinirini_gecmez() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let w = belirgin_doldur(&spec, 6);
        let t = 8;
        let mut o = ortam(&spec, t);
        // Her konum ayri kayit: hicbir n-gram penceresi kapali kalmaz.
        o.kaynak = (0..t).map(|i| i as u32).collect();
        let a = ileri(&spec, &w, &o.girdi()).unwrap();
        assert_eq!(a.engram.okunan, 0);
        assert!(a.engram.kayit_siniri > 0);
        // Tek kayit: okuma sayisi, gecmisi yeten konum sayisi kadar.
        let o = o.tek_kayit();
        let b = ileri(&spec, &w, &o.girdi()).unwrap();
        let n = spec.engram.unwrap().n;
        assert_eq!(b.engram.okunan, t - (n - 1));
        assert_eq!(b.engram.gecmis_yok, n - 1);
        assert_eq!(b.engram.kayit_siniri, 0);
    }

    #[test]
    fn gradyan_sonlu_farkla_uyusur() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let w = belirgin_doldur(&spec, 21);
        let t = 6;
        let o = ortam(&spec, t).tek_kayit();
        let hedef = hedef_uret(t * spec.serit.serit * spec.d_model);
        let denetim = gradyan_denetle(&spec, &w, &o.girdi(), &hedef, 1e-3).unwrap();
        assert_eq!(denetim.denetlenen, spec.parametre_sayisi());
        assert_eq!(
            denetim.ihlal, 0,
            "{} gradyan olcutu gecemedi; en kotu oran {:.3e} ({})",
            denetim.ihlal, denetim.en_kotu_oran, denetim.en_kotu_alan
        );
    }

    #[test]
    fn gradyan_engramsiz_halde_de_uyusur() {
        let mut spec = BirlesikSpec::kucuk_aday().unwrap();
        spec.engram = None;
        let w = belirgin_doldur(&spec, 33);
        let t = 5;
        let o = ortam(&spec, t);
        let hedef = hedef_uret(t * spec.serit.serit * spec.d_model);
        let denetim = gradyan_denetle(&spec, &w, &o.girdi(), &hedef, 1e-3).unwrap();
        assert_eq!(denetim.denetlenen, spec.parametre_sayisi());
        assert_eq!(
            denetim.ihlal, 0,
            "en kotu oran {:.3e} ({})",
            denetim.en_kotu_oran, denetim.en_kotu_alan
        );
    }

    #[test]
    fn gradyan_rotasiz_halde_de_uyusur() {
        let mut spec = BirlesikSpec::kucuk_aday().unwrap();
        spec.rota = None;
        spec.serit = CokSeritSpec::yeni(spec.d_model, 1).unwrap();
        let w = belirgin_doldur(&spec, 44);
        let t = 5;
        let o = ortam(&spec, t).tek_kayit();
        let hedef = hedef_uret(t * spec.serit.serit * spec.d_model);
        let denetim = gradyan_denetle(&spec, &w, &o.girdi(), &hedef, 1e-3).unwrap();
        assert_eq!(denetim.denetlenen, spec.parametre_sayisi());
        assert_eq!(
            denetim.ihlal, 0,
            "en kotu oran {:.3e} ({})",
            denetim.en_kotu_oran, denetim.en_kotu_alan
        );
    }

    #[test]
    fn inis_gercek_bir_kosuda_gorunur() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let mut w = belirgin_doldur(&spec, 21);
        let t = 6;
        let o = ortam(&spec, t).tek_kayit();
        let hedef = hedef_uret(t * spec.serit.serit * spec.d_model);
        let ilk = kayip(&ileri(&spec, &w, &o.girdi()).unwrap().durumlar, &hedef);
        let mut son = ilk;
        for _ in 0..40 {
            let c = ileri(&spec, &w, &o.girdi()).unwrap();
            son = kayip(&c.durumlar, &hedef);
            let g_cikis = kayip_gradyan(&c.durumlar, &hedef);
            let g = geri(&spec, &w, &c.bellek, &o.durumlar, &g_cikis).unwrap();
            sgd_adimi(&mut w, &g, 0.02);
        }
        assert!(son < ilk, "inis yok: {ilk} -> {son}");
    }

    /// Olcum satiri: `training/birlesik.py` bu satiri kosar ve okur.
    #[test]
    fn olcum_raporu() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let mut w = belirgin_doldur(&spec, 21);
        let t = 6;
        let o = ortam(&spec, t).tek_kayit();
        let hedef = hedef_uret(t * spec.serit.serit * spec.d_model);
        let denetim = gradyan_denetle(&spec, &w, &o.girdi(), &hedef, 1e-3).unwrap();
        let ilk = kayip(&ileri(&spec, &w, &o.girdi()).unwrap().durumlar, &hedef);
        let mut son = ilk;
        let mut okunan = 0usize;
        for _ in 0..40 {
            let c = ileri(&spec, &w, &o.girdi()).unwrap();
            son = kayip(&c.durumlar, &hedef);
            okunan = c.engram.okunan;
            let g_cikis = kayip_gradyan(&c.durumlar, &hedef);
            let g = geri(&spec, &w, &c.bellek, &o.durumlar, &g_cikis).unwrap();
            sgd_adimi(&mut w, &g, 0.02);
        }
        println!(
            "birlesik | bilesen={} parametre={} denetlenen={} uzman={} serit={} \
             engram_okuma={} ihlal={} oran={:.6e} ilk_kayip={:.8} son_kayip={:.8}",
            6,
            spec.parametre_sayisi(),
            denetim.denetlenen,
            spec.uzman_sayisi(),
            spec.serit.serit,
            okunan,
            denetim.ihlal,
            denetim.en_kotu_oran,
            ilk,
            son,
        );
    }

    /// Sapmanin kaynagi gradyan degil olcum aletidir: U egrisi bunu olcer.
    ///
    /// Sonlu farkin toplam hatasi iki terimden gelir - kesme hatasi adimla
    /// **buyur** (dort noktali fark icin `O(h⁴)`), yuvarlama hatasi adimla
    /// **kucuIur** (`O(eps·|f|/h)`). Analitik gradyan dogruysa ikisinin
    /// toplami bir U cizer ve dibi genis bir plato olur; analitik gradyan
    /// yanlissa egri duzlesir ve hicbir adimda dibe inmez. Bu test o platoyu
    /// olcer: ortadaki dort adimda hicbir ihlal olmamali ve plato dibi, her
    /// iki uctan da belirgin sekilde iyi olmali.
    #[test]
    fn adim_taramasi_u_egrisi_cizer() {
        let spec = BirlesikSpec::kucuk_aday().unwrap();
        let w = belirgin_doldur(&spec, 21);
        let t = 6;
        let o = ortam(&spec, t).tek_kayit();
        let hedef = hedef_uret(t * spec.serit.serit * spec.d_model);
        let mut oranlar = Vec::new();
        for h in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5] {
            let d = gradyan_denetle(&spec, &w, &o.girdi(), &hedef, h).unwrap();
            assert_eq!(d.denetlenen, spec.parametre_sayisi());
            oranlar.push((h, d.ihlal, d.en_kotu_oran));
        }
        // Plato: ortadaki dort adim temiz.
        for (h, ihlal, oran) in &oranlar[1..5] {
            assert_eq!(
                *ihlal, 0,
                "h={h:.1e} plato icinde ihlal verdi (oran {oran:.3e})"
            );
        }
        // Uclar: buyuk adimda kesme, kucuk adimda yuvarlama baskin - ikisi de
        // platonun dibinden **kotu** olmali, yoksa egri U degildir.
        let dip = oranlar[1..5]
            .iter()
            .map(|(_, _, o)| *o)
            .fold(f64::INFINITY, f64::min);
        assert!(oranlar[0].2 > dip, "buyuk adim ucu platodan iyi cikti");
        assert!(oranlar[5].2 > dip, "kucuk adim ucu platodan iyi cikti");
    }
}
