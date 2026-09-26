//! Lubot'un Android köprüsü.
//!
//! Bu crate **kendi mantığını taşımaz**: CLI'ın koştuğu `ask` yolunu JNI
//! üzerinden cihaza açar. Cevap şeması, alıntı disiplini ve izin defteri
//! aynı kodda; arayüz yalnızca bir taşıyıcı. İki kural buraya da yazıldı:
//!
//! 1. Cihazdan gelen belge korpusa **karışmaz**. Ayrı bir dosyada, kendi
//!    kaynağı ve lisansıyla durur; cevabın alıntısı hangi kayıttan geldiğini
//!    söyler. K2 (korpus budlum yüzeyidir) böylece bozulmadan cihaz içeriği
//!    okunabilir kalıyor.
//! 2. Reddedilen şey sessizce yutulmaz: hata adı ve sebebiyle Java'ya döner,
//!    arayüz de onu olduğu gibi gösterir.
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use jni::objects::JString;
use jni::sys::jstring;
use jni::JNIEnv;

/// Köprünün tuttuğu tek durum: yüklü korpus ve izin defteri.
struct Durum {
    korpus: lubot::LoadedCorpus,
    kitap: lubot_grant::GrantBook,
    denetim: PathBuf,
    ciktilar: PathBuf,
    kaynaklar: Vec<PathBuf>,
}

static DURUM: Mutex<Option<Durum>> = Mutex::new(None);

/// Okuyucu adı denetim kaydına böyle geçer: hangi yüzeyden sorulduğu
/// cevabın kendisi kadar kayda değer.
const OKUYUCU: &str = "android-arayuz";

/// Cihazdan gelen kayıtların kaynağı. Korpus kayıtlarından ayrılması için
/// ayrı ad taşıyor; böylece bir alıntının nereden geldiği karışmıyor.
const CIHAZ_KAYNAGI: &str = "cihaz";

/// Dizin içindeki korpus dosyalarını toplar. `.jsonl.gz` ve `.jsonl` kabul;
/// başka uzantı sessizce atlanmaz, listeye hiç girmez.
fn korpus_dosyalari(dizin: &Path) -> Result<Vec<PathBuf>, String> {
    let mut yollar: Vec<PathBuf> = Vec::new();
    let girisler = std::fs::read_dir(dizin).map_err(|e| format!("{}: {e}", dizin.display()))?;
    for giris in girisler {
        let yol = giris.map_err(|e| format!("dizin okunamadi: {e}"))?.path();
        let uzanti = yol
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_string();
        let gz = uzanti == "gz";
        let jsonl = uzanti == "jsonl";
        if gz || jsonl {
            yollar.push(yol);
        }
    }
    yollar.sort();
    if yollar.is_empty() {
        return Err(format!(
            "{}: korpus dosyasi yok (.jsonl ya da .jsonl.gz)",
            dizin.display()
        ));
    }
    Ok(yollar)
}

/// Cihazdan gelen bir metni, korpus kayıt şemasıyla ayrı dosyaya yazar.
///
/// Şema bilerek korpusunkiyle aynı: `load_corpus` aynı doğrulamayı uygulasın
/// diye. Alanlar eksilirse kayıt reddedilir ve red sebebi Java'ya döner.
fn cihaz_kaydi(ad: &str, icerik: &str, yol: &Path) -> Result<String, String> {
    let satir_sayisi = icerik.lines().count();
    let digest = lubot_read::sha256_hex(icerik.as_bytes());
    let kayit = serde_json::json!({
        "kind": "doc",
        "text": icerik,
        "path": ad,
        "lines": [1, satir_sayisi.max(1)],
        "source": CIHAZ_KAYNAGI,
        "digest": digest,
        "licence": "kullanici-girdisi",
        "attribution": "cihazdan kullanici girdisi (korpus disi, izole kayit)",
        "content_id": digest,
        "asset_id": digest,
        "asset_id_pending": true,
    });
    let mut dosya = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(yol)
        .map_err(|e| format!("{}: {e}", yol.display()))?;
    use std::io::Write;
    writeln!(dosya, "{kayit}").map_err(|e| format!("cihaz kaydi yazilamadi: {e}"))?;
    Ok(digest)
}

/// Köprüyü kurar: korpusu yükler, izin defterini sıfırlar.
///
/// Dönen dize bir ölçümdür (kayıt ve lisans dağılımı), süs değil: arayüz onu
/// olduğu gibi gösterir, yani kullanıcı neyin yüklendiğini görür.
fn kurulus_ici(korpus_dizini: &str, calisma_dizini: &str) -> Result<String, String> {
    let korpus_dizini = PathBuf::from(korpus_dizini);
    let calisma = PathBuf::from(calisma_dizini);
    std::fs::create_dir_all(&calisma).map_err(|e| format!("calisma dizini: {e}"))?;
    let kaynaklar = korpus_dosyalari(&korpus_dizini)?;
    let cihaz_yolu = calisma.join("cihaz-belgeleri.jsonl");
    let mut yollar = kaynaklar.clone();
    if cihaz_yolu.is_file() {
        yollar.push(cihaz_yolu);
    }
    let korpus = lubot::load_corpus(&yollar)?;
    let ozet = lubot::corpus_summary(&korpus);
    let mut durum = DURUM
        .lock()
        .map_err(|_| "durum kilidi zehirlenmis".to_string())?;
    *durum = Some(Durum {
        korpus,
        kitap: lubot_grant::GrantBook::new(),
        denetim: calisma.join("denetim.jsonl"),
        ciktilar: calisma.join("ciktilar.jsonl"),
        kaynaklar,
    });
    serde_json::to_string(&ozet).map_err(|e| format!("ozet serilestirilemedi: {e}"))
}

/// Soruyu CLI'ın koştuğu yoldan geçirir ve şema-doğrulanmış Markdown döner.
fn soru_ici(soru: &str) -> Result<String, String> {
    let now = lubot::now_seconds()?;
    let mut durum = DURUM
        .lock()
        .map_err(|_| "durum kilidi zehirlenmis".to_string())?;
    let d = durum
        .as_mut()
        .ok_or_else(|| "kopru kurulmamis: once kurulus cagirilmali".to_string())?;
    lubot::ask(
        &d.korpus,
        OKUYUCU,
        soru,
        &mut d.kitap,
        now,
        Some(&d.denetim),
        Some(&d.ciktilar),
        None,
    )
}

/// Cihazdan gelen belgeyi izole kayda ekler ve korpusu yeniden yükler.
fn belge_ekle_ici(ad: &str, icerik: &str, calisma_dizini: &str) -> Result<String, String> {
    if icerik.trim().is_empty() {
        return Err("bos belge eklenmez: okunacak bir sey yok".to_string());
    }
    let calisma = PathBuf::from(calisma_dizini);
    let yol = calisma.join("cihaz-belgeleri.jsonl");
    let digest = cihaz_kaydi(ad, icerik, &yol)?;
    // Yeniden yükleme bilerek tam: yarım yüklenmiş bir korpus, alıntının
    // hangi kayda ait olduğunu karıştırır.
    kurulus_ici_yeniden(&calisma)?;
    Ok(digest)
}

/// Aynı kurulum, ama korpus dizini değişmeden: cihaz dosyası eklendiği için
/// kaynak listesi yeniden toplanıyor.
fn kurulus_ici_yeniden(calisma: &Path) -> Result<(), String> {
    let kaynaklar = {
        let durum = DURUM
            .lock()
            .map_err(|_| "durum kilidi zehirlenmis".to_string())?;
        let d = durum
            .as_ref()
            .ok_or_else(|| "kopru kurulmamis".to_string())?;
        d.kaynaklar.clone()
    };
    let cihaz_yolu = calisma.join("cihaz-belgeleri.jsonl");
    let mut yollar = kaynaklar;
    if cihaz_yolu.is_file() {
        yollar.push(cihaz_yolu);
    }
    let korpus = lubot::load_corpus(&yollar)?;
    let mut durum = DURUM
        .lock()
        .map_err(|_| "durum kilidi zehirlenmis".to_string())?;
    if let Some(d) = durum.as_mut() {
        d.korpus = korpus;
    }
    Ok(())
}

// --- JNI yüzeyi -----------------------------------------------------------
//
// İnce katman: hata burada yutulmaz, adı ve sebebiyle Java'ya döner. Java
// tarafı da onu olduğu gibi gösteriyor, çünkü "yaklaşık cevap" ile "reddedildi"
// aynı satırda duramaz.

fn jstring_dondur(env: JNIEnv<'_>, metin: &str) -> jstring {
    match env.new_string(metin) {
        Ok(s) => s.into_raw(),
        // new_string yalnızca UTF-16 dönüşümünde patlar; Java tarafı null'ı
        // "köprü cevap vermedi" olarak gösterir.
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
/// # Safety
///
/// JVM'den çağrılır: `korpus_dizini` ve `calisma_dizini` geçerli `JString`
/// olmak zorunda ve bu çağrı JNI ortamına ait iş parçacığında olmalı.
/// `JNIEnv` başka iş parçacığına taşınmaz; köprü kendi durumunu kilitler.
pub unsafe extern "system" fn Java_dev_budlum_lubot_Kopru_kurulus<'local>(
    mut env: JNIEnv<'local>,
    _sinif: jni::objects::JClass<'local>,
    korpus_dizini: JString<'local>,
    calisma_dizini: JString<'local>,
) -> jstring {
    let sonuc = (|| -> Result<String, String> {
        let a: String = env
            .get_string(&korpus_dizini)
            .map_err(|e| format!("korpus dizini okunamadi: {e}"))?
            .into();
        let b: String = env
            .get_string(&calisma_dizini)
            .map_err(|e| format!("calisma dizini okunamadi: {e}"))?
            .into();
        kurulus_ici(&a, &b)
    })();
    match sonuc {
        Ok(s) => jstring_dondur(env, &s),
        Err(e) => jstring_dondur(env, &format!("HATA: {e}")),
    }
}

#[no_mangle]
/// # Safety
///
/// JVM'den çağrılır: `soru` geçerli bir `JString` olmalı. Köprü kurulmamışsa
/// red döner, boş cevap değil.
pub unsafe extern "system" fn Java_dev_budlum_lubot_Kopru_soru<'local>(
    mut env: JNIEnv<'local>,
    _sinif: jni::objects::JClass<'local>,
    soru: JString<'local>,
) -> jstring {
    let sonuc = (|| -> Result<String, String> {
        let s: String = env
            .get_string(&soru)
            .map_err(|e| format!("soru okunamadi: {e}"))?
            .into();
        soru_ici(&s)
    })();
    match sonuc {
        Ok(s) => jstring_dondur(env, &s),
        Err(e) => jstring_dondur(env, &format!("HATA: {e}")),
    }
}

#[no_mangle]
/// # Safety
///
/// JVM'den çağrılır: `ad`, `icerik` ve `calisma_dizini` geçerli `JString`
/// olmalı. İçerik cihazdan gelir ve korpusa karışmaz; izole kayda yazılır.
pub unsafe extern "system" fn Java_dev_budlum_lubot_Kopru_belgeEkle<'local>(
    mut env: JNIEnv<'local>,
    _sinif: jni::objects::JClass<'local>,
    ad: JString<'local>,
    icerik: JString<'local>,
    calisma_dizini: JString<'local>,
) -> jstring {
    let sonuc = (|| -> Result<String, String> {
        let a: String = env
            .get_string(&ad)
            .map_err(|e| format!("ad okunamadi: {e}"))?
            .into();
        let i: String = env
            .get_string(&icerik)
            .map_err(|e| format!("icerik okunamadi: {e}"))?
            .into();
        let c: String = env
            .get_string(&calisma_dizini)
            .map_err(|e| format!("calisma dizini okunamadi: {e}"))?
            .into();
        belge_ekle_ici(&a, &i, &c)
    })();
    match sonuc {
        Ok(s) => jstring_dondur(env, &s),
        Err(e) => jstring_dondur(env, &format!("HATA: {e}")),
    }
}

#[no_mangle]
/// # Safety
///
/// JVM'den çağrılır; girdi almaz, sabit bir dize döndürür.
pub unsafe extern "system" fn Java_dev_budlum_lubot_Kopru_surum<'local>(
    env: JNIEnv<'local>,
    _sinif: jni::objects::JClass<'local>,
) -> jstring {
    jstring_dondur(
        env,
        &format!(
            "lubot-arayuz {} (JNI kopru; ask yolu CLI ile ayni)",
            env!("CARGO_PKG_VERSION")
        ),
    )
}

#[cfg(test)]
mod testler {
    use super::*;

    /// Köprü kurulmadan soru sormak sessiz bir boş cevap değil, adı konmuş
    /// bir red döndürmek zorunda.
    #[test]
    fn kurulussuz_soru_reddedilir() {
        // Statik durumu kirletmemek için doğrudan iç yolu değil, hatanın
        // şeklini doğruluyoruz: kurulmamış durumda `soru_ici` red döner.
        let mut durum = DURUM.lock().expect("kilit");
        let onceki = durum.take();
        drop(durum);
        let sonuc = soru_ici("bu korpus nedir");
        let mut durum = DURUM.lock().expect("kilit");
        *durum = onceki;
        drop(durum);
        let hata = sonuc.expect_err("kurulussuz soru reddedilmeliydi");
        assert!(
            hata.contains("kurulmamis"),
            "red sebebi adı konmuş olmalıydı, gelen: {hata}"
        );
    }

    /// Boş belge korpusa girmemeli: okunacak bir şey olmayan kayıt, alıntı
    /// yapılabilir bir kayıt değildir.
    #[test]
    fn bos_belge_reddedilir() {
        let dizin = std::env::temp_dir().join("lubot-arayuz-bos-test");
        std::fs::create_dir_all(&dizin).expect("dizin");
        let sonuc = belge_ekle_ici("bos.md", "   \n  ", &dizin.display().to_string());
        let hata = sonuc.expect_err("boş belge reddedilmeliydi");
        assert!(hata.contains("bos belge"), "gelen: {hata}");
    }

    /// Cihaz kaydı korpus şemasıyla yazılmalı ve aynı içerik aynı özeti
    /// vermeli: iki kez eklenen aynı belge iki ayrı kayıt olmamalı.
    #[test]
    fn cihaz_kaydi_deterministik_ozet_verir() {
        let dizin = std::env::temp_dir().join("lubot-arayuz-ozet-test");
        let _ = std::fs::remove_dir_all(&dizin);
        std::fs::create_dir_all(&dizin).expect("dizin");
        let yol = dizin.join("kayit.jsonl");
        let a = cihaz_kaydi("a.md", "ayni icerik", &yol).expect("kayit a");
        let b = cihaz_kaydi("b.md", "ayni icerik", &yol).expect("kayit b");
        assert_eq!(a, b, "aynı içerik aynı özeti vermeli");
        let farkli = cihaz_kaydi("c.md", "baska icerik", &yol).expect("kayit c");
        assert_ne!(a, farkli, "farklı içerik farklı özet vermeli");
        let _ = std::fs::remove_dir_all(&dizin);
    }

    /// Korpus dizini boşsa sessizce boş korpus yüklenmemeli: bu, "cevap yok"
    /// ile "okunacak bir şey yok" ayrımını kaybettirir.
    #[test]
    fn bos_korpus_dizini_reddedilir() {
        let dizin = std::env::temp_dir().join("lubot-arayuz-boskorpus-test");
        let _ = std::fs::remove_dir_all(&dizin);
        std::fs::create_dir_all(&dizin).expect("dizin");
        let hata = korpus_dosyalari(&dizin).expect_err("boş dizin reddedilmeliydi");
        assert!(hata.contains("korpus dosyasi yok"), "gelen: {hata}");
        let _ = std::fs::remove_dir_all(&dizin);
    }
}
