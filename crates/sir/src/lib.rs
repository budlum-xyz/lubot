//! # `sir` - Lubot'a giren metinde sir kalmasin
//!
//! Kural tek cumlede: **Lubot'a giren hicbir metin anahtar tasimaz.** Bir
//! istegin, bir kayit defterinin ya da bir logun icinde bir API anahtari
//! geciyorsa, o metin Lubot'un icine girmeden once temizlenmis olmali - sonra
//! temizlemek, sizmis olani geri almaz.
//!
//! Modul iki katmanda temizler ve ikisi farkli seyleri yakalar:
//!
//! 1. **Ad kalibi**: `api_key: ...`, `token = "..."` gibi satirlarda *ad*
//!    sirra isaret eder. Ad taniyorsa degeri gormeye gerek yoktur; deger ne
//!    olursa olsun maskelenir. Bu katman, sekli bilinmeyen sirlari yakalar.
//! 2. **Sekil kalibi**: `sk-...`, `ghp_...`, `AKIA...`, JWT gibi *taninan
//!    sekiller*. Bu katman, adin gecmedigi serbest metinde calisir - bir
//!    yorum satirina yapistirilmis anahtar da sizintidir.
//!
//! Ikisi ayri ayri yeterli degil: yalniz ad kalibi, `not: ghp_...` gibi bir
//! satiri kacirir; yalniz sekil kalibi, sekli bilinmeyen bir sirri kacirir.
//!
//! ## Ne degismez
//!
//! Maskeleme **bicimi korur**: satir sayisi, kodun iskeleti ve anahtarin
//! cevresindeki noktalama yerinde kalir. Bunun sebebi estetik degil: temizlenen
//! metin cogu zaman bir modele girdi olarak gidiyor ve `fn main()` satirinin
//! kaybolmasi, temizlemenin bedelini isin kendisine odetmek olurdu.
//!
//! ## Neyi iddia etmez
//!
//! Bu modul *taninan* sekilleri ve *taninan* adlari maskeler. Tanimadigi bir
//! sirri yakaladigini iddia etmez ve "metin temiz" diye bir sonuc dondurmez:
//! [`Rapor::degisti`] yalnizca "bir sey maskelendi" der, "baska sir yok" demez.
//! Bir sir tarayicisinin verebilecegi en yanlis cevap, temiz oldugunu soyleyen
//! cevaptir.

use std::collections::BTreeMap;

/// Maskelenen degerin yerine gecen isaret.
///
/// Sabit, ikilinin icinde **duz metin olarak durmaz**: derleme zamani
/// gizlenir, ilk kullanimda cozulur ve tek bir yerde tutulur. Sebep, maskeyi
/// gizlemek degil, tarayicinin *hangi onekleri tanidigini* ikiliden
/// okuyabilmesini engellemektir; maskeyi bilmek bir saldirgana hicbir sey
/// kazandirmaz ama taniyicinin sozlugunu bilmek kazandirir.
#[must_use]
pub fn maske() -> &'static str {
    static MASKE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MASKE.get_or_init(|| {
        lubot_sertlestirme::gizli_metin!("<SIR:GIZLENDI>", 0x5C).unwrap_or_else(|_| String::new())
    })
}

/// Taniyici onekler, gizli tutulur.
///
/// Her cagride cozmek yerine bir kez cozulur: `OnceLock`, ilk `maskele`
/// cagrisinda maliyeti oder ve sonra ayni `&'static str` paylasilir.
fn onek(no: usize) -> &'static str {
    static ONEKLER: std::sync::OnceLock<[String; 12]> = std::sync::OnceLock::new();
    let liste = ONEKLER.get_or_init(|| {
        use lubot_sertlestirme::gizli_metin as g;
        let bos = |h: Result<String, _>| h.unwrap_or_else(|_| String::new());
        [
            bos(g!("sk-ant-", 0x11)),
            bos(g!("sk-proj-", 0x22)),
            bos(g!("sk-svcacct-", 0x33)),
            bos(g!("sk_live_", 0x44)),
            bos(g!("AIza", 0x55)),
            bos(g!("AKIA", 0x66)),
            bos(g!("ghp_", 0x77)),
            bos(g!("xoxb-", 0x88)),
            bos(g!("eyJ", 0x99)),
            bos(g!("sk-", 0xAA)),
            bos(g!("sk_test_", 0xAB)),
            bos(g!("rk_live_", 0xAC)),
        ]
    });
    &liste[no]
}

/// Adi sirra isaret eden kelimeler; ad eslesmesi bunlara bakmaz, *icerir*
/// kuralina bakar (`client_secret` da `secret` icerir).
const SIR_ADLARI: &[&str] = &[
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "token",
    "password",
    "passwd",
    "pwd",
    "private_key",
    "client_secret",
    "auth",
    "credential",
    "anahtar",
    "parola",
    "sifre",
];

/// Bir degeri taninan sekle gore siniflandiran yordam.
type Sekil = fn(&str) -> bool;

/// Sekil kalibi: `(etiket, yordam)`. Etiket rapora yazilir, boylece "kac tane
/// maskelendi" degil "ne tur maskelendi" de gorunur.
const SEKILLER: &[(&str, Sekil)] = &[
    ("anthropic", anthropic_mi),
    ("openai", openai_mi),
    ("stripe", stripe_mi),
    ("google", google_mi),
    ("sk_onekli", sk_onekli_mi),
    ("aws", aws_mi),
    ("github", github_mi),
    ("slack", slack_mi),
    ("jwt", jwt_mi),
    ("noktali_jeton", noktali_jeton_mu),
];

/// GitHub ailesinin onekleri. Aile genisledikce burada buyur; her uye ayri
/// gizlenir, cunku birlikte gomulseydi `strings` cikardigi tek dizede hepsi
/// birden gorunurdu.
static ONEK_GH: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| {
    use lubot_sertlestirme::gizli_metin as g;
    let bos = |h: Result<String, _>| h.unwrap_or_else(|_| String::new());
    vec![
        bos(g!("ghp_", 0xB1)),
        bos(g!("gho_", 0xB2)),
        bos(g!("ghu_", 0xB3)),
        bos(g!("ghs_", 0xB4)),
        bos(g!("ghr_", 0xB5)),
        bos(g!("github_pat_", 0xB6)),
    ]
});

/// Slack ailesinin onekleri.
static ONEK_SLACK: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| {
    use lubot_sertlestirme::gizli_metin as g;
    let bos = |h: Result<String, _>| h.unwrap_or_else(|_| String::new());
    vec![
        bos(g!("xoxb-", 0xC1)),
        bos(g!("xoxa-", 0xC2)),
        bos(g!("xoxp-", 0xC3)),
        bos(g!("xoxr-", 0xC4)),
        bos(g!("xoxs-", 0xC5)),
    ]
});

/// Yalniz ASCII harf/rakam, `-` ve `_` iceren bir gövde mi.
fn govde_mi(deger: &str, en_az: usize, on_ek: &str) -> bool {
    deger.len() >= en_az
        && deger.starts_with(on_ek)
        && deger
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn sk_onekli_mi(deger: &str) -> bool {
    govde_mi(deger, 18, onek(9))
}

fn anthropic_mi(deger: &str) -> bool {
    govde_mi(deger, 20, onek(0))
}

fn openai_mi(deger: &str) -> bool {
    govde_mi(deger, 20, onek(1)) || govde_mi(deger, 20, onek(2))
}

fn stripe_mi(deger: &str) -> bool {
    // `sk_test_` ve `rk_live_` ayni ailedendir; ucu de ayri ayri gizlenir ki
    // tabloyu okuyan biri hangi ailelerin tanindigini tek bakista cikarmasin.
    deger.len() >= 24
        && (deger.starts_with(onek(3))
            || deger.starts_with(onek(10))
            || deger.starts_with(onek(11)))
        && deger.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Google anahtarlari tam 39 karakterdir ve `AIza` ile baslar. Bu, yanlis
/// pozitifi dusuk bir onek: 39 karakterlik, `AIza` ile baslayan alakasiz bir
/// dize neredeyse yoktur.
fn google_mi(deger: &str) -> bool {
    deger.len() == 39
        && deger.starts_with(onek(4))
        && deger
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn aws_mi(deger: &str) -> bool {
    deger.len() == 20
        && deger.starts_with(onek(5))
        && deger
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

fn github_mi(deger: &str) -> bool {
    deger.len() >= 22
        && ONEK_GH.iter().any(|o| deger.starts_with(o.as_str()))
        && deger.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn slack_mi(deger: &str) -> bool {
    deger.len() >= 18
        && ONEK_SLACK.iter().any(|o| deger.starts_with(o.as_str()))
}

/// Uc parcasi nokta ile ayrilmis, basligi `eyJ` ile baslayan jeton.
fn jwt_mi(deger: &str) -> bool {
    let parcalar: Vec<&str> = deger.split('.').collect();
    parcalar.len() == 3
        && parcalar[0].starts_with(onek(8))
        && parcalar
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
}

/// Basligi `eyJ` olmayan ama sekli jeton olan uzun noktali dize. Ihtiyatli
/// tutuldu: toplam uzunluk esigi, uc parcasi da base64url olmayi sart kosuyor.
fn noktali_jeton_mu(deger: &str) -> bool {
    let parcalar: Vec<&str> = deger.split('.').collect();
    parcalar.len() == 3
        && deger.len() >= 40
        && parcalar
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
}

/// Neyin maskelendigi: tur -> adet. Rapor "temiz" demez, "su turden su kadar
/// maskelendi" der; aradaki fark, bir tarayicinin en tehlikeli yalanina
/// (`temiz`) izin vermemektir.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rapor {
    sayim: BTreeMap<String, usize>,
}

impl Rapor {
    fn ekle(&mut self, tur: &str) {
        *self.sayim.entry(tur.to_string()).or_insert(0) += 1;
    }

    /// Toplam maskeleme sayisi.
    #[must_use]
    pub fn toplam(&self) -> usize {
        self.sayim.values().sum()
    }

    /// Tur bazinda sayim.
    #[must_use]
    pub fn sayim(&self) -> &BTreeMap<String, usize> {
        &self.sayim
    }

    /// Herhangi bir maskeleme yapildi mi. `false` "sir yok" demek **degildir**;
    /// "taninan bir sir maskelenmedi" demektir.
    #[must_use]
    pub fn degisti(&self) -> bool {
        self.toplam() > 0
    }
}

/// Maskelenmis metin ve ne yapildiginin raporu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sonuc {
    metin: String,
    rapor: Rapor,
}

impl Sonuc {
    /// Maskelenmis metin.
    #[must_use]
    pub fn metin(&self) -> &str {
        &self.metin
    }

    /// Rapor.
    #[must_use]
    pub fn rapor(&self) -> &Rapor {
        &self.rapor
    }

    /// Metni sahiplenerek dondurur.
    #[must_use]
    pub fn metne(self) -> String {
        self.metin
    }
}

/// Ayiriciyi bulur: `:` ya da `=`, tirnak icindeki degil, satirin ilk
/// gecerli olani.
fn ayirici_bul(satir: &str) -> Option<usize> {
    let mut tirnak = false;
    for (i, c) in satir.char_indices() {
        if c == '"' || c == '\'' {
            tirnak = !tirnak;
            continue;
        }
        if !tirnak && (c == ':' || c == '=') {
            return Some(i);
        }
    }
    None
}

/// Ad sirra isaret ediyorsa degeri maskeler; bicim (tirnak, bosluk, sondaki
/// noktalama) korunur.
fn ad_degeri_maskesi(satir: &str, rapor: &mut Rapor) -> String {
    let Some(pos) = ayirici_bul(satir) else {
        return satir.to_string();
    };
    let ad = satir[..pos].trim();
    let ad_norm = ad.to_lowercase().replace('-', "_");
    if !SIR_ADLARI.iter().any(|k| ad_norm.contains(k)) {
        return satir.to_string();
    }
    let sonrasi = &satir[pos + 1..];
    let kirpik = sonrasi.trim_start();
    let bosluk = &sonrasi[..sonrasi.len() - kirpik.len()];
    let tirnak = kirpik.chars().next().filter(|c| *c == '"' || *c == '\'');
    let deger_basi = tirnak.map_or(0, |_| 1);
    let govde = &kirpik[deger_basi..];
    // Degerin sonundaki noktalama ve bosluk degerin parcasi degil, satirin
    // bicimidir; oldugu yerde birakilir.
    let deger_sonu = govde
        .char_indices()
        .rev()
        .find(|(_, c)| !matches!(c, ',' | ';' | ')' | '}' | ']' | ' ' | '\t'))
        .map_or(0, |(i, _)| i + 1);
    let kuyruk = &govde[deger_sonu..];
    let deger = &govde[..deger_sonu];
    if deger.is_empty() || deger.starts_with(maske()) {
        return satir.to_string();
    }
    rapor.ekle("ad_degeri");
    let ayirici = &satir[pos..pos + 1];
    let tirnak_str = tirnak.map_or("", |_| "\"");
    format!("{ad}{ayirici}{bosluk}{tirnak_str}{}{tirnak_str}{kuyruk}", maske())
}

/// Serbest metinde taninan sekilleri maskeler. Kelime sinirlari korunur: bir
/// sirrin icindeki noktalama sirra aittir, sirrin cevresindeki degil.
fn sekil_maskesi(satir: &str, rapor: &mut Rapor) -> String {
    let mut cikti = String::new();
    let mut simdiki = String::new();
    fn bosalt(simdiki: &mut String, cikti: &mut String, rapor: &mut Rapor) {
        if simdiki.is_empty() {
            return;
        }
        match SEKILLER
            .iter()
            .find(|(_, tani)| tani(simdiki))
            .map(|(tur, _)| *tur)
        {
            Some(tur) => {
                rapor.ekle(tur);
                cikti.push_str(maske());
            }
            None => cikti.push_str(simdiki),
        }
        simdiki.clear();
    }
    for ch in satir.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
            simdiki.push(ch);
        } else {
            bosalt(&mut simdiki, &mut cikti, rapor);
            cikti.push(ch);
        }
    }
    bosalt(&mut simdiki, &mut cikti, rapor);
    cikti
}

/// Metni temizler: satir sayisi ve kod iskeleti korunur.
#[must_use]
pub fn maskele(metin: &str) -> Sonuc {
    // Dagitik kontrol noktasi: metin temizleme, urune giren her seyin gectigi
    // yerdir; buraya konan kontrol, tek bir yamayla kapatilacak bir nokta
    // degil, akisin kendisinde duran bir noktadir.
    let _ = lubot_sertlestirme::izler::nokta("sir.maskele");
    let mut rapor = Rapor::default();
    let satirlar: Vec<String> = metin
        .lines()
        .map(|satir| {
            let s = ad_degeri_maskesi(satir, &mut rapor);
            sekil_maskesi(&s, &mut rapor)
        })
        .collect();
    Sonuc {
        metin: satirlar.join("\n"),
        rapor,
    }
}

/// Bir onekin hangi aileye ait oldugu.
///
/// Sinif, urunun baska yerlerindeki tarayicilara **onek metnini yazmadan**
/// kural uygulama imkani verir: cagiran "GitHub klasik ailesi" der, gereken
/// uzunlugu kendi bilir, onegi buradan alir. Boylece ikinci bir tablo olusmaz
/// ve `strings` hicbir oneki gormez.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnekSinifi {
    /// `sk-` ile baslayan model servisi anahtarlari (genel).
    ModelApi,
    /// Anthropic ailesi.
    Anthropic,
    /// OpenAI kapsamli anahtarlar.
    OpenAi,
    /// Stripe ailesi.
    Stripe,
    /// Google API anahtari.
    Google,
    /// AWS erisim anahtari kimligi.
    Aws,
    /// GitHub ailelerinin tamami (klasik ve ince taneli).
    GitHub,
    /// Slack ailelerinin tamami.
    Slack,
    /// JWT basligi.
    Jwt,
}

/// Taninan onekler ve siniflari: **tek kaynak**.
#[must_use]
pub fn taninan_onekler_sinifli() -> Vec<(&'static str, OnekSinifi)> {
    let mut liste: Vec<(&'static str, OnekSinifi)> = vec![
        (onek(9), OnekSinifi::ModelApi),
        (onek(0), OnekSinifi::Anthropic),
        (onek(1), OnekSinifi::OpenAi),
        (onek(2), OnekSinifi::OpenAi),
        (onek(3), OnekSinifi::Stripe),
        (onek(10), OnekSinifi::Stripe),
        (onek(11), OnekSinifi::Stripe),
        (onek(4), OnekSinifi::Google),
        (onek(5), OnekSinifi::Aws),
        (onek(8), OnekSinifi::Jwt),
    ];
    liste.extend(ONEK_GH.iter().map(|o| (o.as_str(), OnekSinifi::GitHub)));
    liste.extend(ONEK_SLACK.iter().map(|o| (o.as_str(), OnekSinifi::Slack)));
    liste
}

/// Taninan oneklerin tamami: **tek kaynak**.
///
/// Urunun baska bir yerinde sir taniyan bir kod varsa oneklerini buradan
/// almali. Iki ayri tablo, iki ayri gercek demektir: biri guncellenip digeri
/// unutuldugunda tarayicilardan biri sessizce korlesir. Liste gizli
/// sabitlerden uretilir, yani `strings` cikarmaz.
#[must_use]
pub fn taninan_onekler() -> Vec<String> {
    let mut liste: Vec<String> = (0..12).map(|i| onek(i).to_string()).collect();
    liste.extend(ONEK_GH.iter().cloned());
    liste.extend(ONEK_SLACK.iter().cloned());
    liste
}

/// Bir deger sirra benziyor mu. Ad kalibi ya da sekil kalibi yeter; maskelenmis
/// bir deger `false` doner, cunku zaten temizdir ve yeniden maskelemek bilgi
/// kaybi olur.
#[must_use]
pub fn sirli_mi(deger: &str) -> bool {
    let kucuk = deger.to_lowercase();
    if kucuk.contains(&maske().to_lowercase()) {
        return false;
    }
    if SIR_ADLARI.iter().any(|k| kucuk.contains(k)) {
        return true;
    }
    SEKILLER.iter().any(|(_, tani)| tani(deger))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test verisi parca parca kurulur: statik kaynakta sir kalibi gecmez, boylece
    /// bir sir tarayicisi bu test dosyasini sizinti sanmaz.
    fn ornek_sk() -> String {
        format!("sk-{}", "abcdefghijklmnopqrstuvwxyz123")
    }

    #[test]
    fn ad_sirra_isaret_ediyorsa_deger_maskelenir() {
        let sir = ornek_sk();
        let s = maskele(&format!("api_key: {sir}"));
        assert!(s.metin().contains(maske()));
        assert!(!s.metin().contains(&sir));
        assert!(s.rapor().degisti());
    }

    #[test]
    fn bicim_korunur() {
        let girdi = "let api_key = \"gizli123\";\nprintln!(\"ok\");";
        let s = maskele(girdi);
        assert_eq!(s.metin().lines().count(), girdi.lines().count());
        assert!(!s.metin().contains("gizli123"));
        assert!(s.metin().contains("println!"));
    }

    #[test]
    fn sade_metin_degismez() {
        let girdi = "fn main() { println!(\"merhaba\"); }";
        let s = maskele(girdi);
        assert!(!s.rapor().degisti());
        assert_eq!(s.metin(), girdi);
    }

    #[test]
    fn github_jetonu_maskelenir() {
        let sir = format!("ghp_{}", "abcdefghijklmnopqrstuvwxyz123456");
        let s = maskele(&format!("token={sir}"));
        assert!(s.metin().contains(maske()));
        assert!(!s.metin().contains("ghp_"));
    }

    #[test]
    fn jwt_maskelenir() {
        let jwt = format!("eyJ{}.eyJ{}.{}", "hbGciOiJIUzI1NiJ9", "zdWIiOiIxIn0", "imza");
        let s = maskele(&format!("auth: {jwt}"));
        assert!(s.metin().contains(maske()));
        assert!(!s.metin().contains("eyJhbGciOiJIUzI1NiJ9"));
    }

    #[test]
    fn maske_sonrasi_metin_bir_daha_degismez() {
        let girdi = format!("password = {}", maske());
        let s = maskele(&girdi);
        assert_eq!(s.metin(), girdi);
        assert!(!s.rapor().degisti());
    }

    #[test]
    fn google_anahtari_serbest_metinde_maskelenir() {
        let anahtar = format!("AIza{}", "x".repeat(35));
        assert_eq!(anahtar.len(), 39);
        let s = maskele(&format!("not: {anahtar} burada"));
        assert!(s.metin().contains(maske()));
        assert!(!s.metin().contains(&anahtar));
    }

    #[test]
    fn anthropic_anahtari_serbest_metinde_maskelenir() {
        let anahtar = format!("sk-ant-api03-{}", "abcdefghijklmnopqrstuv");
        let s = maskele(&format!("credential {anahtar} eklendi"));
        assert!(s.metin().contains(maske()));
        assert!(!s.metin().contains(&anahtar));
    }

    #[test]
    fn openai_kapsamli_anahtar_maskelenir() {
        let anahtar = format!("sk-proj-{}", "abcdefghijklmnopqrstuv");
        let s = maskele(&format!("{anahtar} kullaniliyor"));
        assert!(s.metin().contains(maske()));
    }

    #[test]
    fn stripe_canli_anahtar_maskelenir() {
        let anahtar = format!("sk_live_{}", "abcdefghijklmnopqrstuvwx");
        let s = maskele(&format!("{anahtar} tahsilat icin"));
        assert!(s.metin().contains(maske()));
    }

    #[test]
    fn kisa_onekler_yanlis_pozitif_uretmez() {
        for zararsiz in [
            "sk-ant- bir belge onekidir",
            "AIza kisa",
            "sk-proj- kisa",
            "eyJ sadece uc harf",
        ] {
            let s = maskele(zararsiz);
            assert!(!s.rapor().degisti(), "yanlis pozitif: {zararsiz}");
        }
    }

    #[test]
    fn sirli_mi_maskelenmisi_temiz_sayar() {
        let sir = ornek_sk();
        assert!(sirli_mi(&sir));
        assert!(sirli_mi("parola ipucu"));
        assert!(!sirli_mi(&format!("x {} y", maske())));
        assert!(!sirli_mi("merhaba dunya"));
    }

    #[test]
    fn turkce_adlar_da_yakalanir() {
        let s = maskele("sifre: cok-gizli-bir-deger");
        assert!(s.metin().contains(maske()));
        assert!(!s.metin().contains("cok-gizli-bir-deger"));
        let s = maskele("parola = \"a1b2c3\"");
        assert!(s.metin().contains(maske()));
    }

    #[test]
    fn rapor_turu_soyler_ve_toplam_tutar() {
        let sir = ornek_sk();
        let s = maskele(&format!("api_key: {sir}\nnot: {sir}"));
        assert_eq!(s.rapor().toplam(), 2);
        assert!(s.rapor().sayim().contains_key("ad_degeri"));
        assert!(s.rapor().sayim().contains_key("sk_onekli"));
    }

    #[test]
    fn iki_katman_birbirinin_isi_bozmaz() {
        // Ad kalibi once calisir; deger maskelendigi icin sekil kalibi ayni
        // satirda ikinci bir maskeleme yapmaz.
        let sir = ornek_sk();
        let s = maskele(&format!("token = {sir}"));
        assert_eq!(s.rapor().toplam(), 1);
    }
}
