//! `lubot-a2` spec'i ile uygulanan bloğun **bağı**.
//!
//! `training/model_spec_a2.json` bir şekil beyanı; `crates/egitim/src/*` o
//! şekli gerçekten kuran kod. İkisi ayrı yerlerde durduğu sürece ayrışırlar ve
//! ayrıştıkları gün spec bir belge, kod başka bir şey olur. Bu dosya o ayrımı
//! imkânsız kılıyor: JSON okunur, ondan bir [`BirlesikSpec`] kurulur, ve
//! Python'un mimari tanımdan türettiği sayı ile Rust'ın alt modüllerden
//! topladığı sayı **karşılaştırılır**.
//!
//! Su nokta onemli: bu test Python'un formülünü tekrar etmiyor. Python
//! `2*blok*(d_model/blok)*(d_r/blok) + d_r*d_model + 2*d_r + d_model` yazıyor;
//! burada o formül hiç yok — `HadamardSpec` kurulup kendi
//! `parametre_sayisi()`'si soruluyor. İki yol, tek sayı; formül bir yerde
//! değişirse test kırmızı yanar.
//!
//! Bağlı olmadığı da burada yazılı: bu spec hiçbir eğitim çağrısından
//! geçmiyor, `model_spec.json` (a1) değişmedi.

use lubot_egitim::birlesik::BirlesikSpec;
use lubot_egitim::cok_serit::CokSeritSpec;
use lubot_egitim::engram::EngramSpec;
use lubot_egitim::mlp_hadamard::HadamardSpec;
use lubot_egitim::normalizasyon::NormSpec;
use lubot_egitim::yonlendirme::RotaSpec;

fn spec_json() -> serde_json::Value {
    let yol =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/model_spec_a2.json");
    let ham = std::fs::read_to_string(&yol)
        .unwrap_or_else(|e| panic!("a2 spec okunamadi ({}): {e}", yol.display()));
    serde_json::from_str(&ham).unwrap_or_else(|e| panic!("a2 spec gecerli JSON degil: {e}"))
}

fn usize_al(v: &serde_json::Value, yol: &[&str]) -> usize {
    let mut su_an = v;
    for anahtar in yol {
        su_an = su_an
            .get(*anahtar)
            .unwrap_or_else(|| panic!("a2 spec'te alan yok: {}", yol.join(".")));
    }
    usize::try_from(
        su_an
            .as_u64()
            .unwrap_or_else(|| panic!("{} bir tam sayi degil", yol.join("."))),
    )
    .unwrap_or_else(|e| panic!("alan usize'a sigmadi: {e}"))
}

/// JSON'daki şekilden gerçek bir blok kurar.
fn blok_kur(engramli: bool) -> BirlesikSpec {
    let j = spec_json();
    let d_model = usize_al(&j, &["sekil", "d_model"]);
    let serit = usize_al(&j, &["sekil", "serit"]);
    let d_r = usize_al(&j, &["sekil", "hadamard", "d_r"]);
    let blok = usize_al(&j, &["sekil", "hadamard", "blok"]);
    let uzman = usize_al(&j, &["sekil", "rota", "uzman"]);
    let k = usize_al(&j, &["sekil", "rota", "k"]);
    let yineleme = usize_al(&j, &["sekil", "rota", "yineleme"]);
    let e_n = usize_al(&j, &["sekil", "engram", "n"]);
    let e_tablo = usize_al(&j, &["sekil", "engram", "tablo"]);
    let e_dkv = usize_al(&j, &["sekil", "engram", "d_kv"]);

    BirlesikSpec {
        d_model,
        serit: CokSeritSpec::yeni(d_model, serit)
            .unwrap_or_else(|e| panic!("serit sekli reddedildi: {e:?}")),
        norm: NormSpec::yeni(d_model, 1e-6)
            .unwrap_or_else(|e| panic!("norm sekli reddedildi: {e:?}")),
        hadamard: HadamardSpec { d_model, d_r, blok },
        rota: Some(
            RotaSpec::yeni(uzman, k, 1.0, yineleme)
                .unwrap_or_else(|e| panic!("rota sekli reddedildi: {e:?}")),
        ),
        engram: if engramli {
            Some(
                EngramSpec::yeni(e_n, e_tablo, e_dkv)
                    .unwrap_or_else(|e| panic!("engram sekli reddedildi: {e:?}")),
            )
        } else {
            None
        },
    }
}

/// Beyan edilen şekil gerçekten kurulabiliyor mu? Kurulamayan bir şekil,
/// üzerinde konuşulan ama var olmayan bir mimaridir.
#[test]
fn a2_sekli_gercekten_kurulabiliyor() {
    assert!(
        blok_kur(false).dogrula().is_ok(),
        "engramsiz blok reddedildi"
    );
    assert!(blok_kur(true).dogrula().is_ok(), "engramli blok reddedildi");
}

/// Bu dosyanın asıl işi: Python'un mimari tanımdan türettiği blok sayıları ile
/// Rust'ın alt modüllerden topladığı sayılar **aynı**.
#[test]
fn blok_parametreleri_iki_yoldan_ayni() {
    let j = spec_json();
    let beklenen_engramsiz = usize_al(&j, &["params", "blok_engramsiz"]);
    let beklenen_engramli = usize_al(&j, &["params", "blok_engramli"]);

    assert_eq!(
        blok_kur(false).parametre_sayisi(),
        beklenen_engramsiz,
        "engramsiz blok: Python ile Rust ayrismis"
    );
    assert_eq!(
        blok_kur(true).parametre_sayisi(),
        beklenen_engramli,
        "engramli blok: Python ile Rust ayrismis"
    );
}

/// Engram bir **ek**: açıldığında blok büyümeli, ve büyüme tam olarak tablo +
/// izdüşüm kadar olmalı. Aradaki fark başka bir şeyse, engram sessizce başka
/// bir terimi de değiştiriyor demektir.
#[test]
fn engram_sadece_kendi_terimini_ekliyor() {
    let j = spec_json();
    let d_model = usize_al(&j, &["sekil", "d_model"]);
    let tablo = usize_al(&j, &["sekil", "engram", "tablo"]);
    let d_kv = usize_al(&j, &["sekil", "engram", "d_kv"]);
    let fark = blok_kur(true).parametre_sayisi() - blok_kur(false).parametre_sayisi();
    assert_eq!(
        fark,
        2 * tablo * d_kv + d_model * d_kv,
        "engramin katkisi tablo + izdusumden farkli"
    );
}

/// Rotalama parametre tutmuyor ve bu **ölçülerek** doğrulanıyor: rotayı açıp
/// kapatmak blok sayısını değiştirmemeli.
#[test]
fn rota_parametre_tasimiyor() {
    let mut rotasiz = blok_kur(false);
    let rotali = rotasiz.parametre_sayisi();
    rotasiz.rota = None;
    // Rotalama kapaninca uzman sayisi 1'e duser, yani Hadamard payi da duser.
    // Olculen sey rotanin *kendi* payi: uzman sayisini sabit tutarak bak.
    let uzman = blok_kur(false).uzman_sayisi();
    let hadamard_payi = blok_kur(false).hadamard.parametre_sayisi();
    assert_eq!(
        rotali - rotasiz.parametre_sayisi(),
        (uzman - 1) * hadamard_payi,
        "rota acilinca Hadamard payindan baska bir sey de degisti: rota parametre tutuyor"
    );
}

/// Toplam, bileşenlerin toplamıyla tutuyor mu? Spec'in kendi aritmetiği.
#[test]
fn toplam_bilesenlerle_tutuyor() {
    let j = spec_json();
    let toplam = usize_al(&j, &["params", "toplam"]);
    let parcalar = usize_al(&j, &["params", "embedding_bagli"])
        + usize_al(&j, &["params", "dikkat_gqa"])
        + usize_al(&j, &["params", "bloklar"])
        + usize_al(&j, &["params", "son_norm"]);
    assert_eq!(
        toplam, parcalar,
        "a2 spec'inin toplami parcalariyla tutmuyor"
    );
}

/// Bloklar satırı, katman sayısı ve engram katmanı sayısıyla tutuyor mu?
#[test]
fn blok_toplami_katman_sayisiyla_tutuyor() {
    let j = spec_json();
    let n = usize_al(&j, &["sekil", "n_layers"]);
    let e = usize_al(&j, &["sekil", "engram_katmanlari"]);
    assert!(e <= n, "engram katmani sayisi katman sayisindan buyuk");
    let beklenen =
        (n - e) * blok_kur(false).parametre_sayisi() + e * blok_kur(true).parametre_sayisi();
    assert_eq!(
        usize_al(&j, &["params", "bloklar"]),
        beklenen,
        "bloklar satiri katman dagilimiyla tutmuyor"
    );
}

/// K6: toplam, ölçülen donanım tavanının altında. Tavan a1 spec'inden okunur,
/// burada yeniden yazılmaz.
#[test]
fn k6_tavaninin_altinda() {
    let j = spec_json();
    let toplam = usize_al(&j, &["params", "toplam"]);
    let ust = usize_al(&j, &["ceiling_reference", "max_params_train_fp32_adamw"]);
    assert!(toplam < ust, "K6 ihlali: {toplam} >= {ust}");

    let a1_yol =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/model_spec.json");
    let a1: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(a1_yol).unwrap_or_else(|e| panic!("a1 spec okunamadi: {e}")),
    )
    .unwrap_or_else(|e| panic!("a1 spec JSON degil: {e}"));
    assert_eq!(
        usize_al(&a1, &["ceiling_reference", "max_params_train_fp32_adamw"]),
        ust,
        "a2'deki tavan a1'dekiyle ayni degil: iki dosyada iki tavan"
    );
}

/// Engram parametreleri matmul görmüyor; bu ailenin bütün iddiası bu ayrımda,
/// o yüzden ayrı raporlanıyor ve toplamın altında kalıyor.
#[test]
fn matmul_etkin_pay_ayri_raporlaniyor() {
    let j = spec_json();
    let toplam = usize_al(&j, &["params", "toplam"]);
    let matmul = usize_al(&j, &["params", "matmul_etkin"]);
    let tablolar = usize_al(&j, &["params", "engram_tablolari"]);
    assert!(
        matmul < toplam,
        "matmul payi toplamla ayni: ayrim yapilmamis"
    );
    assert_eq!(matmul + tablolar, toplam, "matmul + tablo toplami tutmuyor");
    assert!(
        tablolar > 0,
        "engram tablosu sifir: aile a1'den ayirt edilemez"
    );
}

/// a1 ailesi değişmedi. Bu test, a2'yi eklerken a1'e dokunulmadığını çiviler.
#[test]
fn a1_ailesi_degismedi() {
    let a1_yol =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/model_spec.json");
    let a1: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(a1_yol).unwrap_or_else(|e| panic!("a1 spec okunamadi: {e}")),
    )
    .unwrap_or_else(|e| panic!("a1 spec JSON degil: {e}"));
    assert_eq!(
        a1.get("name").and_then(serde_json::Value::as_str),
        Some("lubot-a1-derin-dar")
    );
    assert_eq!(usize_al(&a1, &["params", "toplam"]), 924_288);
    assert_eq!(usize_al(&a1, &["d_ff"]), 256, "a1'in FFN'i degistirilmis");
}
