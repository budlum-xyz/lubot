//! `lubot sohbet` — kullanıcıya dönük üretim yüzeyi.
//!
//! Sözleşme (anayasa maddesi `no-generation`): model her türden veriyi
//! okuyabilir ve inceleyebilir; **kullanıcıya sunduğu çıktı yalnızca şema
//! doğrulamalı Markdown'dır.** Bu komut o sözleşmenin tek kapısıdır:
//!
//! 1. Sorgu jetonlanır ([`lubot_jeton::Sozluk`]).
//! 2. Üretim [`lubot_cikarim::uretim::Uretic`] ile yapılır — yerel ağırlıklar,
//!    yerel örnekleyici; dışarıya tek bayt gitmez, tek bayt gelmez.
//! 3. Üretilen jetonlar metne çözülür ve **Markdown şemasından geçirilir**;
//!    geçmezse komut hata verir. Yumuşatma, kırpma, "yakınını dene" yok:
//!    şemaya girmeyen çıktı yazılmaz.
//!
//! Üretim boşsa (durma jetonu ilk adımda) bu da bir cevaptır: boş Markdown
//! gövdesi yazılmaz, açık ret döner.

use std::path::Path;

use lubot_cikarim::cezalar::{CezaHatasi, Cezalar};
use lubot_cikarim::ornekleyici::{Ayarlar, OrnekHatasi};
use lubot_cikarim::uretim::{Durma, KaydirmaModu, Uretic, UretimAyari, UretimHatasi, UretimRaporu};
use lubot_cikarim::{Cikarim, CikarimHatasi};

use crate::egitim_kosu::Bayraklar;

/// `lubot sohbet` giriş noktası.
///
/// # Errors
/// Bayrak eksikliği, kontrol noktası/sözlük reddi ve şema reddi dâhil her
/// başarısızlık [`String`] olarak döner; komut sıfırdan farklı kodla biter.
pub fn cmd_sohbet(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(
        args,
        &[
            "--ckpt",
            "--vocab",
            "--sorgu",
            "--tohum",
            "--sicaklik",
            "--top-k",
            "--top-p",
            "--en-cok",
            "--dur-jetonu",
            "--kayit",
            "--tekrar-cezasi",
            "--varlik-cezasi",
            "--siklik-cezasi",
            "--kac-gram",
            "--en-az-jeton",
            "--kaydirma",
        ],
    )?;
    let ckpt = b.zorunlu("--ckpt")?.to_string();
    let sorgu = b.zorunlu("--sorgu")?.to_string();
    let vocab = b
        .metin("--vocab")
        .unwrap_or("training/tokenizer/lubot-bpe-v2.json")
        .to_string();
    let tohum = b.sayi::<u64>("--tohum")?.unwrap_or(20_260_924);
    let sicaklik = b.ondalik("--sicaklik")?.unwrap_or(0.8);
    let top_k = b.sayi::<usize>("--top-k")?.unwrap_or(40);
    let top_p = b.ondalik("--top-p")?.unwrap_or(0.95);
    let en_cok = b.sayi::<usize>("--en-cok")?.unwrap_or(64);
    let dur_jetonu = b.sayi::<u32>("--dur-jetonu")?;
    let kayit_yolu = b.metin("--kayit").map(str::to_string);
    // Cezalar ve kayan pencere yolu: her biri dogrulanir, sonra rapora gecer.
    // Sessiz varsayilan yok - kapatilmis her ceza raporda "kapali" gorunur.
    let ceza = Cezalar {
        tekrar_cezasi: b.ondalik("--tekrar-cezasi")?.unwrap_or(1.0),
        varlik_cezasi: b.ondalik("--varlik-cezasi")?.unwrap_or(0.0),
        siklik_cezasi: b.ondalik("--siklik-cezasi")?.unwrap_or(0.0),
        kac_gram: b.sayi::<usize>("--kac-gram")?.unwrap_or(0),
        en_az_jeton: b.sayi::<usize>("--en-az-jeton")?.unwrap_or(0),
    };
    ceza.dogrula()
        .map_err(|e: CezaHatasi| format!("ceza ayari reddedildi: {e}"))?;
    let kaydirma_modu = match b.metin("--kaydirma") {
        None | Some("yeniden") => KaydirmaModu::YenidenKur,
        Some("onbellek") => KaydirmaModu::OnbellektenDus,
        Some(baska) => {
            return Err(format!(
                "--kaydirma `{baska}`: `yeniden` (tam) ya da `onbellek` (yaklasim) olmali"
            ))
        }
    };

    let cikarim = Cikarim::yukle(Path::new(&ckpt))
        .map_err(|e: CikarimHatasi| format!("kontrol noktasi yuklenemedi ({ckpt}): {e}"))?;
    let sozluk = lubot_jeton::Sozluk::yukle(Path::new(&vocab))
        .map_err(|e| format!("sozluk reddedildi ({vocab}): {e}"))?;

    let baglam = sozluk.kodla(&sorgu);
    if baglam.is_empty() {
        return Err("sorgu jetonlanmadi: bos girdiden uretim yok".to_string());
    }
    let ayar = UretimAyari {
        ayar: Ayarlar {
            sicaklik,
            top_k,
            top_p,
        },
        tohum,
        en_cok_jeton: en_cok,
        dur_jetonlari: dur_jetonu.into_iter().collect(),
        pencere_kaydir: true,
        kaydirma_modu,
        ceza,
    };
    // Ayar once dogrulanir: gecersiz sicaklik/top-p ile uretime girmek, hatayi
    // jeton urettikten sonra fark etmek olurdu.
    ayar.ayar
        .dogrula()
        .map_err(|e: OrnekHatasi| format!("ornekleme ayari reddedildi: {e}"))?;
    let mut uretic = Uretic::yeni(cikarim, ayar);
    let (jetonlar, rapor): (Vec<u32>, UretimRaporu) = uretic
        .uret(&baglam)
        .map_err(|e: UretimHatasi| format!("uretim reddedildi: {e}"))?;
    if jetonlar.is_empty() {
        return Err("uretim bos: durma jetonu ilk adimda geldi, yazilacak metin yok".to_string());
    }
    let uretilen = sozluk
        .coz(&jetonlar)
        .map_err(|e| format!("uretilen jetonlar cozulemedi: {e}"))?;

    // Kullanıcıya dönük tek biçim: şema doğrulamalı Markdown.
    //
    // Başlık "Soru/Cevap" değil "Girdi/Üretim": bu model bir dil modeli olarak
    // *devam ettirir*, soruyu yanıtlamak üzere eğitilmiş bir asistan değildir.
    // Yanlış başlık, çıktıyı olduğundan yetenekli gösterirdi.
    let md = format!(
        "# Sohbet\n\n## Girdi\n\n{girdi}\n\n## Uretim\n\n{cevap}\n\n## Olcum\n\n| olcu | deger |\n|---|---|\n| jeton | {jeton} |\n| durma | {durma} |\n| ortalama log-olasilik | {log:.4} |\n| tohum | {tohum} |\n| sicaklik | {sicaklik} |\n| top-k | {top_k} |\n| top-p | {top_p} |\n| pencere kaydirma | {kaydirma} |\n| kaydirma modu | {kaydirma_modu} |\n| onbellek yeniden kurma | {yeniden_kurma} |\n| ceza | {ceza} |\n",
        girdi = govde(&sorgu),
        cevap = govde(&uretilen),
        jeton = rapor.jeton,
        durma = match rapor.durma { Durma::Uzunluk => "uzunluk", Durma::DurJetonu => "dur-jetonu" },
        log = rapor.ortalama_log_olasilik,
        kaydirma = rapor.kaydirma,
        kaydirma_modu = match rapor.kaydirma_modu {
            KaydirmaModu::YenidenKur => "yeniden-kur (tam)",
            KaydirmaModu::OnbellektenDus => "onbellekten-dus (yaklasim)",
        },
        yeniden_kurma = rapor.yeniden_kurma,
        ceza = rapor.ceza.ozet(),
    );
    crate::validate_output(md.as_bytes(), "sohbet")?;
    print!("{md}");
    if let Some(yol) = kayit_yolu {
        let kayit = serde_json::json!({
            "surum": 1,
            "kosucu": "model",
            "olcut": {"ad": "uretim_markdown_semasindan_gecti", "sonuc": true},
            "kaynaklar": {"sure_saniye": 0.0, "girdi_jetonlari": baglam.len(),
                          "onbellekli_jetonlari": 0, "cikti_jetonlari": jetonlar.len(),
                          "maliyet": 0.0},
            "ckpt": ckpt,
            "tohum": tohum,
            "cevap_satir": govde(&uretilen).lines().count(),
        });
        std::fs::write(&yol, format!("{kayit:#}\n"))
            .map_err(|e| format!("kayit yazilamadi ({yol}): {e}"))?;
    }
    Ok(())
}

/// Üretilen metni Markdown gövdesine çevirir.
///
/// Model ne yazdıysa o yazılır: satır sonları korunur, başlığa karışmasın diye
/// satır başı `#` işaretleri kaçırılır. Kırpma ya da düzeltme yok — metin
/// modelin, biçim şemanın.
fn govde(metin: &str) -> String {
    let temiz: Vec<String> = metin
        .lines()
        .map(|s| {
            if s.starts_with('#') {
                format!("\\{s}")
            } else {
                s.to_string()
            }
        })
        .collect();
    let birlesik = temiz.join("\n");
    let kirpik = birlesik.trim();
    if kirpik.is_empty() {
        "(bos)".to_string()
    } else {
        kirpik.to_string()
    }
}

#[cfg(test)]
mod testler {
    use super::*;

    #[test]
    fn govde_basligi_kacirir_ve_bos_govdeyi_isaretler() {
        assert_eq!(govde("## baslik\nmetin"), "\\## baslik\nmetin");
        assert_eq!(govde("   \n\n "), "(bos)");
        assert_eq!(govde("satir\n"), "satir");
    }

    #[test]
    fn bayraklar_zorunlu_alanlari_ister() {
        let hata = cmd_sohbet(&[]).expect_err("ckpt olmadan kosmamali");
        assert!(hata.contains("--ckpt"), "beklenen hata: {hata}");
    }

    #[test]
    fn gecersiz_kaydirma_modu_ve_ceza_reddedilir() {
        // Kontrol noktasi okunmadan once ayar reddedilir: kullanici iki kez
        // kosmasin diye, hata ayarin kendisini adiyla soyler.
        let hata = cmd_sohbet(&[
            "--ckpt".into(),
            "/yok".into(),
            "--sorgu".into(),
            "selam".into(),
            "--kaydirma".into(),
            "hizli".into(),
        ])
        .expect_err("bilinmeyen mod kabul edilmemeli");
        assert!(hata.contains("--kaydirma"), "beklenen hata: {hata}");

        let hata = cmd_sohbet(&[
            "--ckpt".into(),
            "/yok".into(),
            "--sorgu".into(),
            "selam".into(),
            "--tekrar-cezasi".into(),
            "0.5".into(),
        ])
        .expect_err("1.0 alti ceza reddedilmeli");
        assert!(
            hata.contains("ceza ayari reddedildi"),
            "beklenen hata: {hata}"
        );
    }
}
