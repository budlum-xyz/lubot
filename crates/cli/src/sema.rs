//! # sema - şema-kısıtlı decode, komut olarak
//!
//! İki alt komut, ikisi de aynı sebeple var: bir sözleşme ancak koşulursa
//! sözleşmedir.
//!
//! - `sozlesme`: otomatın tavanını, maskesini ve **red yollarını** bu makinede
//!   yeniden ölçer. Bir red yolu hiç koşulmadıysa o kuralın çalıştığı bilinmez;
//!   bu komut üç reddi (seviye atlayan başlık, dengesiz çit, geçersiz UTF-8)
//!   ve bir çıkmazı gerçekten koşturur.
//! - `coz`: maskeli bir decode koşar, üretilen belgeyi şemanın kendisine sorar
//!   ve sayaçları yazar.
//!
//! ## Burada eğitilmiş bir model yok
//!
//! Logitler deterministik bir karışımdan gelir, bir kontrol noktasından değil.
//! Ölçülen şey modelin kalitesi değil **maskenin ne yaptığı**dır: hangi jetonlar
//! hiç seçilemedi, belge şemadan geçti mi, geçmediyse decode ne dedi.

use lubot_read::output_schema::validate_markdown_output;
use lubot_sema_cozucu::{
    adim, coz, Adim, CozumHatasi, CozumOlcum, Sema, SemaHatasi, Yuruyus, EN_FAZLA_BAYT,
};

/// 0..=255 için tek baytlık jetonlar.
static TEK_BAYT: [[u8; 1]; 256] = {
    let mut tablo = [[0u8; 1]; 256];
    let mut i = 0;
    while i < 256 {
        #[allow(clippy::cast_possible_truncation)]
        {
            tablo[i] = [i as u8];
        }
        i += 1;
    }
    tablo
};

/// Komutun kullandığı sözlük: 256 tek bayt + birkaç çok baytlı jeton.
///
/// Gerçek sözlük (`lubot-bpe-v2`) donmuş bir dosyadır ve `lubot sozluk`
/// komutunun işidir; burada ölçülen şey maske olduğu için sözlüğün kendisi
/// bilinçli olarak küçük ve sabit tutulur.
fn sozluk() -> Vec<&'static [u8]> {
    let mut s: Vec<&'static [u8]> = Vec::with_capacity(260);
    for b in 0..=255u16 {
        #[allow(clippy::cast_possible_truncation)]
        let bayt = b as u8;
        s.push(&TEK_BAYT[bayt as usize]);
    }
    s.extend_from_slice(&[b"## ", b"\n\n", b"|---|", b"```"]);
    s
}

/// Deterministik logit karışımı: aynı tohum aynı belgeyi verir.
fn logit_kaynagi(tohum: u64, boyut: usize) -> impl FnMut(&Yuruyus) -> Vec<f64> {
    let mut durum = tohum
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    move |_y: &Yuruyus| {
        (0..boyut)
            .map(|_| {
                durum = durum
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((durum >> 33) as f64 / (1u64 << 31) as f64) - 0.5
            })
            .collect()
    }
}

fn deger(args: &[String], ad: &str) -> Option<String> {
    let sira = args.iter().position(|a| a == ad)?;
    args.get(sira + 1).cloned()
}

fn sayi(args: &[String], ad: &str, varsayilan: usize) -> Result<usize, String> {
    match deger(args, ad) {
        Some(metin) => metin
            .parse::<usize>()
            .map_err(|e| format!("{ad} bir sayı değil: {e}")),
        None => Ok(varsayilan),
    }
}

/// `sema` alt komutlarını dağıtır.
///
/// # Errors
///
/// Bilinmeyen alt komut ya da sayı olarak çözülemeyen bir seçenek metin olarak
/// döner; ikili bunu stderr'e yazıp sıfır olmayan kodla çıkar.
pub fn cmd_sema(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("sozlesme") => sozlesme(),
        Some("coz") => coz_komut(&args[1..]),
        Some(other) => Err(format!(
            "unknown sema subcommand: {other}\nusage: lubot sema [sozlesme|coz] [--adim N] [--tohum N]"
        )),
    }
}

fn sozlesme() -> Result<(), String> {
    let s = sozluk();
    let sema = Sema::varsayilan();
    println!("# Sema cozucu sozlesmesi\n");
    println!(
        "- tavan: {EN_FAZLA_BAYT} bayt (ayar {})",
        sema.en_fazla_bayt
    );
    println!("- sozluk: {} jeton", s.len());

    // Maske boş bir belgede neyi eliyor: bu sayı UTF-8 kuralının kendisidir.
    let bos = Yuruyus::yeni(sema);
    let izinli = s.iter().filter(|j| bos.izinli(j)).count();
    println!(
        "- bos belge: {} jeton izinli, {} jeton maskeli",
        izinli,
        s.len() - izinli
    );

    // Üç red yolu, gerçekten koşulur.
    let mut y = Yuruyus::yeni(sema);
    for b in b"# B\n\n##" {
        y.ilerle(&[*b]).map_err(|e| e.to_string())?;
    }
    let atlama = y.ilerle(b"#").err();
    println!("- seviye atlayan baslik: {}", rapor(atlama));

    let mut y = Yuruyus::yeni(sema);
    for b in b"metin\n```\nkod\n" {
        y.ilerle(&[*b]).map_err(|e| e.to_string())?;
    }
    println!("- dengesiz cit: {}", rapor(y.kapat().err()));

    let y = Yuruyus::yeni(sema);
    println!(
        "- gecersiz utf8 (0xC0 0xAF): {}",
        if y.izinli(&[0xc0, 0xaf]) {
            "KABUL EDILDI (sozlesme bozuk)".to_string()
        } else {
            "reddedildi".to_string()
        }
    );

    // Çıkmaz: tavan doluyken tek bir bayt bile geçmez.
    let mut y = Yuruyus::yeni(Sema { en_fazla_bayt: 2 });
    y.ilerle(b"ab").map_err(|e| e.to_string())?;
    println!(
        "- cikmaz (tavan dolu): {}",
        if y.cikmaz() {
            "tek bayt bile kabul edilmiyor".to_string()
        } else {
            "KABUL EDILEN BAYT VAR (sozlesme bozuk)".to_string()
        }
    );
    let h = adim(&mut y, &vec![0.0; s.len()], &s).err();
    match h {
        Some(CozumHatasi::Cikmaz { adim, satir }) => {
            println!("- cikmaz reddi: adim {adim}, satir {satir}");
        }
        Some(other) => return Err(format!("çıkmaz başka bir red olarak döndü: {other}")),
        None => return Err("çıkmazda decode bir jeton seçti".to_string()),
    }
    Ok(())
}

fn rapor(hata: Option<SemaHatasi>) -> String {
    match hata {
        Some(h) => format!("reddedildi ({h})"),
        None => "KABUL EDILDI (sozlesme bozuk)".to_string(),
    }
}

fn coz_komut(args: &[String]) -> Result<(), String> {
    let en_fazla_adim = sayi(args, "--adim", 200)?;
    let tohum = sayi(args, "--tohum", 7)?;
    let s = sozluk();
    let mut y = Yuruyus::yeni(Sema::varsayilan());
    let kaynak = logit_kaynagi(tohum as u64, s.len());
    let dur = |y: &Yuruyus| y.kabul() && y.belge().ends_with(b"\n");
    println!("# Sema kisitli decode\n");
    println!(
        "- uyarı: logitler deterministik bir karışımdan geliyor, eğitilmiş bir kontrol noktası yok"
    );
    match coz(&mut y, kaynak, &s, en_fazla_adim, dur) {
        Ok(o) => {
            yaz(&y, &o);
            Ok(())
        }
        Err(CozumHatasi::KabulEdilmez { sebep }) => {
            println!("- sonuç: REDDEDİLDİ — {sebep}");
            println!("- bayt: {}", y.bayt_sayisi());
            println!("- satır: {}", y.satir());
            println!("- çit içinde bitti: {}", y.cit_icinde());
            Ok(())
        }
        Err(other) => Err(format!("decode koşmadı: {other}")),
    }
}

fn yaz(y: &Yuruyus, o: &CozumOlcum) {
    // Adım izinden iki sayı: maskenin işlediği adım sayısı ve en yoğun tek
    // adım. Toplam tek başına ikisini birbirinden ayıramaz.
    let maskeli_adim = o
        .adimlar
        .iter()
        .filter(|a: &&Adim| a.maskelenen > 0)
        .count();
    let en_yogun = o.adimlar.iter().map(|a| a.maskelenen).max().unwrap_or(0);
    println!("- adım: {}", o.adim);
    println!("- maskelenen jeton (toplam): {}", o.maskelenen);
    println!("- maskenin işlediği adım: {maskeli_adim}");
    println!("- tek adımda en çok maskelenen: {en_yogun}");
    println!("- bayt: {}", o.bayt);
    println!("- satır: {}", y.satir());
    println!("- şema kabulü: {}", o.kabul);
    println!(
        "- doğrulayıcı: {}",
        match validate_markdown_output(y.belge()) {
            Ok(()) => "geçti".to_string(),
            Err(e) => format!("KALDI ({e})"),
        }
    );
    println!("\n## Belge\n");
    println!("{}", String::from_utf8_lossy(y.belge()));
}
