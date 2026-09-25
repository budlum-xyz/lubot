//! The training-run command, the inference commands and the exam run.
//!
//! # What lives here and why it is the CLI that owns it
//!
//! `lubot-egitim` knows how to take one step and how to judge a run;
//! `lubot-cikarim` knows how to read a checkpoint. Neither of them touches the
//! filesystem, reads a corpus, or decides what a run's identity is. Those are
//! operator decisions, and this module is where an operator's command line
//! becomes one.
//!
//! # The three refusals, and why they are refusals
//!
//! 1. **No run without a declared corpus stamp.** `--damga` must equal the
//!    SHA-256 this command computes over the corpus's `content_id`s and the
//!    vocabulary family. A run that trains on data other than the data it
//!    announced cannot be compared with any earlier run, and the file it writes
//!    would carry the wrong provenance. Fail closed.
//! 2. **No run over the exam set.** `--sinav` names the held-out exam set; its
//!    stamped `content_id`s are read out of `eval-only.json` and the corpus is
//!    searched for them. One stamped record present is enough to refuse - the
//!    leak is binary, not a rate.
//! 3. **No continuation across a different corpus or vocabulary.** `--devam`
//!    compares the checkpoint's stamp and family against this run's. A resumed
//!    run that silently changes its data is a new run wearing an old name.
//!
//! # What is written, and what is not
//!
//! The report is Markdown because everything Lubot prints is Markdown, and it
//! is validated against the output schema before it reaches stdout. The
//! checkpoint is written once, at the end, from the state the run left behind.
//! The evaluation record is JSON in the schema `training/eval/SONUC_SEMASI.md`
//! describes: one mechanical boolean criterion, full resource accounting.

use std::path::Path;

use lubot_cikarim::{Aday, Cikarim, CikarimHatasi, OnbellekRaporu, CACHE_TOLERANCE};
use lubot_egitim::kontrol::{hex, Hassasiyet, Kontrol, SIHIR, SURUM};
use lubot_egitim::kosu::{
    egitim_kosu, AdimKaydi, DogrulamaKaydi, DurmaNedeni, KosuAyari, KosuHatasi, KosuRaporu,
};
use lubot_egitim::veri::{bolumle, pencere_uzunlugu, pencereler, Bolum, BolumHatasi, Kayit};
use lubot_egitim::{Adamw, Parametreler, Spec, BLOK_ADLARI, INIT_STD_EMBEDDING};
use sha2::{Digest, Sha256};

/// Step every this many steps when `--bildir` is not given.
const BILDIRIM_VARSAYILAN: u64 = 25;

/// The flags this module understands, split into a lookup so a typo is a
/// refusal rather than a silently ignored word.
pub(crate) struct Bayraklar {
    degerler: Vec<(String, String)>,
}

impl Bayraklar {
    pub(crate) fn ayikla(args: &[String], gecerli: &[&str]) -> Result<Self, String> {
        let mut degerler: Vec<(String, String)> = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let ad = args[i].clone();
            if !gecerli.contains(&ad.as_str()) {
                return Err(format!(
                    "bilinmeyen secenek {ad}; gecerli olanlar: {}",
                    gecerli.join(", ")
                ));
            }
            if ad == "--f32" || ad == "--sessiz" {
                degerler.push((ad, String::new()));
                i += 1;
                continue;
            }
            i += 1;
            let deger = args
                .get(i)
                .ok_or_else(|| format!("{ad} bir deger istiyor"))?
                .clone();
            degerler.push((ad, deger));
            i += 1;
        }
        Ok(Self { degerler })
    }

    pub(crate) fn metin(&self, ad: &str) -> Option<&str> {
        self.degerler
            .iter()
            .find(|(a, _)| a == ad)
            .map(|(_, d)| d.as_str())
    }

    pub(crate) fn zorunlu(&self, ad: &str) -> Result<&str, String> {
        self.metin(ad).ok_or_else(|| format!("{ad} zorunlu"))
    }

    pub(crate) fn sayi<T: std::str::FromStr>(&self, ad: &str) -> Result<Option<T>, String> {
        match self.metin(ad) {
            None => Ok(None),
            Some(ham) => ham
                .parse::<T>()
                .map(Some)
                .map_err(|_| format!("{ad} bir sayi istiyor, alinan `{ham}`")),
        }
    }

    pub(crate) fn ondalik(&self, ad: &str) -> Result<Option<f64>, String> {
        match self.metin(ad) {
            None => Ok(None),
            Some(ham) => ham
                .parse::<f64>()
                .map(Some)
                .map_err(|_| format!("{ad} bir ondalik istiyor, alinan `{ham}`")),
        }
    }

    pub(crate) fn var_mi(&self, ad: &str) -> bool {
        self.degerler.iter().any(|(a, _)| a == ad)
    }
}

/// Read a corpus, tokenise it and keep each record's identity with its ids.
///
/// The `content_id` is read from the record rather than recomputed: the corpus
/// builder is the one place that decides what a record's identity is, and a
/// second implementation of that rule would be a second answer to the question
/// "is this the same text".
fn korpus_oku(yol: &str, sozluk: &lubot_jeton::Sozluk) -> Result<Vec<Kayit>, String> {
    let dosya = std::fs::File::open(yol).map_err(|e| format!("korpus acilamadi {yol}: {e}"))?;
    let okuyucu: Box<dyn std::io::BufRead> =
        if Path::new(yol).extension().is_some_and(|e| e == "gz") {
            Box::new(std::io::BufReader::new(flate2::read::GzDecoder::new(dosya)))
        } else {
            Box::new(std::io::BufReader::new(dosya))
        };
    let mut kayitlar: Vec<Kayit> = Vec::new();
    for (sira, satir) in std::io::BufRead::lines(okuyucu).enumerate() {
        let satir = satir.map_err(|e| format!("korpus okunamadi ({sira}): {e}"))?;
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        let deger: serde_json::Value = serde_json::from_str(satir)
            .map_err(|e| format!("korpus kaydi {sira} JSON degil: {e}"))?;
        let metin = deger["text"]
            .as_str()
            .ok_or_else(|| format!("korpus kaydi {sira}: `text` alani yok"))?;
        let kimlik = deger["content_id"]
            .as_str()
            .ok_or_else(|| format!("korpus kaydi {sira}: `content_id` alani yok"))?;
        kayitlar.push(Kayit {
            kimlik: kimlik.to_string(),
            jetonlar: sozluk.kodla(metin),
        });
    }
    if kayitlar.is_empty() {
        return Err(format!("korpus bos: {yol}"));
    }
    Ok(kayitlar)
}

/// SHA-256 over the sorted `content_id`s plus the vocabulary family.
///
/// Sorted, because the stamp has to be a property of the *set* of records: a
/// rebuild that emits the same records in a different order is the same corpus,
/// and a stamp that changed with the order would break every comparison for a
/// reason that has nothing to do with the data. The family is inside the digest
/// because the same text under a different tokenizer is different training
/// data.
fn korpus_ozeti(kayitlar: &[Kayit], aile: &str, surum: &str) -> String {
    let mut kimlikler: Vec<&str> = kayitlar.iter().map(|k| k.kimlik.as_str()).collect();
    kimlikler.sort_unstable();
    let mut ozet = Sha256::new();
    ozet.update(b"lubot-korpus-ozeti-v1\n");
    ozet.update(aile.as_bytes());
    ozet.update(b"\n");
    ozet.update(surum.as_bytes());
    ozet.update(b"\n");
    for kimlik in kimlikler {
        ozet.update(kimlik.as_bytes());
        ozet.update(b"\n");
    }
    hex(&ozet.finalize())
}

/// The stamped `content_id`s of the held-out eval set.
///
/// `--sinav` names the exam set; its stamps live next to it in
/// `eval-only.json`. A missing file is a refusal, not an empty list: "I could
/// not find the stamps" and "there are no stamps" are different statements and
/// only one of them is safe to train past.
fn sinav_damgalari(sinav_yolu: &str) -> Result<Vec<String>, String> {
    let sinav = Path::new(sinav_yolu);
    if !sinav.is_file() {
        return Err(format!("sinav seti yok: {sinav_yolu}"));
    }
    let damga_yolu = sinav
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("eval-only.json");
    let metin = std::fs::read_to_string(&damga_yolu)
        .map_err(|e| format!("{} okunamadi: {e}", damga_yolu.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&metin).map_err(|e| format!("eval-only.json JSON degil: {e}"))?;
    let liste = json["digests"]
        .as_array()
        .ok_or_else(|| "eval-only.json: `digests` listesi yok".to_string())?;
    let mut damgalar: Vec<String> = Vec::new();
    for deger in liste {
        let kimlik = deger
            .as_str()
            .ok_or_else(|| "eval-only.json: digests icinde dize olmayan deger".to_string())?;
        damgalar.push(kimlik.to_string());
    }
    Ok(damgalar)
}

/// Take the eval-only records out of the training stream, and count them.
///
/// Not a filter that hides something: the number removed is reported, because
/// "the exam set was never trained on" is a claim that needs a quantity behind
/// it. Zero removed is a legitimate result - it says the held-out passages are
/// not in this corpus at all - but it is a *result*, not a silence.
fn damgalilari_cikar(kayitlar: Vec<Kayit>, damgalar: &[String]) -> (Vec<Kayit>, usize) {
    if damgalar.is_empty() {
        return (kayitlar, 0);
    }
    let kume: std::collections::HashSet<&str> = damgalar.iter().map(String::as_str).collect();
    let once = kayitlar.len();
    let kalan: Vec<Kayit> = kayitlar
        .into_iter()
        .filter(|k| !kume.contains(k.kimlik.as_str()))
        .collect();
    let cikarilan = once - kalan.len();
    (kalan, cikarilan)
}

/// Evenly spaced subset of the validation windows.
///
/// Validation costs one forward pass per window per measurement; on the self
/// corpus the full set makes a 1500-step run spend more time measuring than
/// training. The subset is spread evenly across the whole set rather than taken
/// from the front: the windows are in content order, and a front slice would
/// measure one region of the corpus and call it the corpus.
fn dogrulama_seyrelt(
    pencereler: &[lubot_egitim::PaketPencere],
    adet: Option<usize>,
) -> Vec<lubot_egitim::PaketPencere> {
    let tavan = adet.unwrap_or(usize::MAX);
    if tavan == 0 || pencereler.len() <= tavan {
        return pencereler.to_vec();
    }
    let adim = pencereler.len() as f64 / tavan as f64;
    let mut secilen: Vec<lubot_egitim::PaketPencere> = Vec::with_capacity(tavan);
    for i in 0..tavan {
        let konum = ((i as f64 + 0.5) * adim) as usize;
        let konum = konum.min(pencereler.len() - 1);
        secilen.push(pencereler[konum].clone());
    }
    secilen
}

/// Write a JSON file with a trailing newline.
fn json_yaz(yol: &str, deger: &serde_json::Value) -> Result<(), String> {
    let metin = format!(
        "{}\n",
        serde_json::to_string_pretty(deger).map_err(|e| e.to_string())?
    );
    std::fs::write(yol, metin).map_err(|e| format!("{yol} yazilamadi: {e}"))
}

/// `lubot korpus-damgasi` - print the stamp a run must declare.
///
/// The stamp is computed here and *declared* on the command line of the run, in
/// that order, so the value on the run's command line is one an operator has
/// seen rather than one the tool filled in for them. A run whose stamp was
/// filled in automatically would carry provenance nobody ever looked at.
pub fn cmd_korpus_damgasi(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(args, &["--corpus", "--vocab"])?;
    let korpus_yolu = b.zorunlu("--corpus")?.to_string();
    let vocab_yolu = b
        .metin("--vocab")
        .unwrap_or("training/tokenizer/lubot-bpe-v2.json")
        .to_string();
    let sozluk = lubot_jeton::Sozluk::yukle(Path::new(&vocab_yolu))
        .map_err(|e| format!("sozluk reddedildi ({vocab_yolu}): {e}"))?;
    let kayitlar = korpus_oku(&korpus_yolu, &sozluk)?;
    let ozet = korpus_ozeti(&kayitlar, sozluk.aile(), &spec_etiketi()?);
    println!("{ozet}");
    eprintln!(
        "{} kayit, sozluk ailesi `{}`: bu damgayi --damga ile bildirin",
        kayitlar.len(),
        sozluk.aile()
    );
    Ok(())
}

/// `lubot egitim-kosu` - train, measure, checkpoint.
pub fn cmd_egitim_kosu(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(
        args,
        &[
            "--corpus",
            "--vocab",
            "--ckpt",
            "--devam",
            "--damga",
            "--sinav",
            "--rapor",
            "--kayit",
            "--adim",
            "--epoch",
            "--yigin",
            "--iplik",
            "--pencere",
            "--dogrulama-payi",
            "--dogrulama-pencere",
            "--dogrulama-her",
            "--ogrenme-orani",
            "--agirlik-sonumu",
            "--kirpma",
            "--tohum",
            "--isinma",
            "--bildir",
            "--sessiz",
            "--f32",
        ],
    )?;
    let spec = Spec::lubot_a1();
    let korpus_yolu = b.zorunlu("--corpus")?.to_string();
    let vocab_yolu = b
        .metin("--vocab")
        .unwrap_or("training/tokenizer/lubot-bpe-v2.json")
        .to_string();
    let ckpt_yolu = b.zorunlu("--ckpt")?.to_string();
    let damga = b.zorunlu("--damga")?.to_string();
    let sinav_yolu = b.zorunlu("--sinav")?.to_string();
    let rapor_yolu = b.metin("--rapor").map(str::to_string);
    let kayit_yolu = b.metin("--kayit").map(str::to_string);
    let sessiz = b.var_mi("--sessiz");

    let sozluk = lubot_jeton::Sozluk::yukle(Path::new(&vocab_yolu))
        .map_err(|e| format!("sozluk reddedildi ({vocab_yolu}): {e}"))?;
    if sozluk.boyut() != spec.vocab {
        return Err(format!(
            "sozluk {vocab_yolu} {} kimlik tasiyor ama spec {} istiyor: vocab spec'in parcasi",
            sozluk.boyut(),
            spec.vocab
        ));
    }
    let kayitlar = korpus_oku(&korpus_yolu, &sozluk)?;
    let ozet = korpus_ozeti(&kayitlar, sozluk.aile(), &spec_etiketi()?);
    if ozet != damga {
        return Err(format!(
            "korpus damgasi tutmuyor: hesaplanan {ozet}, bildirilen {damga}\n\
             ayni veri uzerinde kosmadiginiz bir tur, onceki turlarla karsilastirilamaz"
        ));
    }
    let damgalar = sinav_damgalari(&sinav_yolu)?;
    // Damgali kayitlar egitim akisindan CIKARILIR ve sayisi rapora yazilir:
    // sinav kumesi uzerinde egitim yapilmadigi iddiasi bir sayi ister.
    let (kayitlar, dislanan) = damgalilari_cikar(kayitlar, &damgalar);
    if kayitlar.is_empty() {
        return Err("butun korpus eval-only damgali cikti: egitilecek kayit kalmadi".to_string());
    }

    // Pencere uzunlugu spec'e karsi dogrulanir, kirpilmaz.
    let istenen = b.sayi::<usize>("--pencere")?.unwrap_or(128);
    let pencere = pencere_uzunlugu(spec, istenen)?;
    let pay = b.ondalik("--dogrulama-payi")?.unwrap_or(0.05);
    let kayit_sayisi = kayitlar.len();
    // Bolme kimlige gore yapilir ve bolumlemeden sonra kayit asla ikiye
    // bolunmez; iki adim, iki ayri cevap vermesin diye ayni yerde duruyor.
    let bolum: Bolum = bolumle(kayitlar, pay)
        .map_err(|h: BolumHatasi| format!("bolumleme reddedildi: {h:?} (pay {pay})"))?;
    let (egitim_pencereleri, dogrulama_hepsi) = pencereler(&bolum, pencere)?;
    let dogrulama = dogrulama_seyrelt(&dogrulama_hepsi, b.sayi::<usize>("--dogrulama-pencere")?);
    let tohum = b.sayi::<u64>("--tohum")?.unwrap_or(20_260_924);
    let lr = b.ondalik("--ogrenme-orani")?.unwrap_or(0.01);
    let sonum = b.ondalik("--agirlik-sonumu")?.unwrap_or(0.1);
    let kirpma = b.ondalik("--kirpma")?.unwrap_or(1.0);
    let isinma = b.sayi::<u64>("--isinma")?.unwrap_or(50);
    let adim = b.sayi::<u64>("--adim")?.unwrap_or(1_500);
    let epoch_tavani = b.sayi::<u32>("--epoch")?.unwrap_or(8);
    let yigin = b.sayi::<usize>("--yigin")?.unwrap_or(2);
    // 0 = makineye sor. Sonucu degistirmez (toplama sirasi korunur), yalniz
    // sureyi degistirir; raporda ikisi de yazili.
    let iplik = b.sayi::<usize>("--iplik")?.unwrap_or(0);
    // --f32 hem adimin hem kontrol noktasinin hassasiyetidir: f32'de egitilen
    // bir turun sayilari her adimda yuvarlanmistir, saklama da bunu soyler.
    let istenen_hassasiyet = if b.var_mi("--f32") {
        Hassasiyet::F32
    } else {
        Hassasiyet::F64
    };
    let mut hassasiyet = istenen_hassasiyet;
    let dogrulama_her = b
        .sayi::<u64>("--dogrulama-her")?
        .unwrap_or(BILDIRIM_VARSAYILAN);
    let bildirim = b.sayi::<u64>("--bildir")?.unwrap_or(BILDIRIM_VARSAYILAN);
    // Epoch tavani grant'tan gelir: turun butcesi bu crate'in degil, zincir
    // tarafinin soyledigi bir sayidir.
    let epoch_tavani = lubot_egitim::epoch_butcesi(epoch_tavani).map_err(|e| e.to_string())?;

    let (mut p, tasinan, baslangic_adim, baslangic_epoch, baslangic_konum, baslangic_en_iyi) =
        match b.metin("--devam") {
            None => (
                Parametreler::mup_init(spec, tohum, INIT_STD_EMBEDDING),
                None,
                0u64,
                0u32,
                0usize,
                None,
            ),
            Some(yol) => {
                let k = Kontrol::yukle(Path::new(yol))
                    .map_err(|e| format!("devam edilecek kontrol noktasi reddedildi: {e}"))?;
                if k.korpus_ozeti != ozet {
                    return Err(format!(
                        "kontrol noktasi baska bir korpusun ({}) bu kosunun ({ozet}) degil",
                        k.korpus_ozeti
                    ));
                }
                if k.sozluk_aile != sozluk.aile() {
                    return Err(format!(
                        "kontrol noktasi sozluk ailesi `{}` ama bu kosu `{}`",
                        k.sozluk_aile,
                        sozluk.aile()
                    ));
                }
                // Devam eden tur, biraktigi hassasiyetle devam eder: f32'de
                // egitilmis bir kosuyu f64'te surdurmek iki farkli sayi
                // rejimini tek egriye yazmak olurdu. Celiski susmaz.
                if k.hassasiyet != istenen_hassasiyet {
                    return Err(format!(
                        "kontrol noktasi {} hassasiyetinde ama bu cagri {}: devam eden tur kendi hassasiyetiyle surer",
                        k.hassasiyet.etiket(),
                        istenen_hassasiyet.etiket()
                    ));
                }
                hassasiyet = k.hassasiyet;
                let opt = k.optimizer_yeniden().ok_or_else(|| {
                    "kontrol noktasi optimiser durumu tasimiyor: devam edilemez".to_string()
                })?;
                (
                    k.parametreler,
                    Some(opt),
                    k.adim,
                    k.epoch,
                    k.devam_konum,
                    k.en_iyi_dogrulama,
                )
            }
        };
    let mut opt = match tasinan {
        Some(o) => o,
        None => Adamw::yeni(p.toplam_ogeler(), lr, sonum)?,
    };
    if !p.sekil_dogru(spec) {
        return Err(
            "parametreler spec'in sekline uymuyor: kontrol noktasi baska bir mimarinin".to_string(),
        );
    }
    let ayar = KosuAyari {
        spec,
        pencere_uzunlugu: pencere,
        tohum,
        ogrenme_orani: lr,
        agirlik_sonumu: sonum,
        isinma_adimi: isinma,
        toplam_adim: adim,
        planlanan_adim: baslangic_adim + adim,
        baslangic_adim,
        baslangic_en_iyi,
        baslangic_epoch,
        baslangic_konum,
        yigin,
        hassasiyet,
        iplik,
        kirpma,
        dogrulama_her,
        epoch_tavani,
    };
    let rapor: KosuRaporu = egitim_kosu(
        &ayar,
        &egitim_pencereleri,
        &dogrulama,
        &mut p,
        &mut opt,
        |adim_kaydi: &AdimKaydi, olcum: Option<&DogrulamaKaydi>| {
            if sessiz {
                return;
            }
            if let Some(d) = olcum {
                eprintln!(
                    "adim {} epoch {}: kayip {:.6}, dogrulama {:.6}",
                    adim_kaydi.adim, adim_kaydi.epoch, adim_kaydi.kayip, d.kayip
                );
            } else if adim_kaydi.adim.is_multiple_of(bildirim) {
                eprintln!(
                    "adim {} epoch {}: kayip {:.6} (lr {:.5}, gradyan {:.4})",
                    adim_kaydi.adim,
                    adim_kaydi.epoch,
                    adim_kaydi.kayip,
                    adim_kaydi.ogrenme_orani,
                    adim_kaydi.gradyan_normu
                );
            }
        },
    )
    .map_err(|e: KosuHatasi| format!("kosu reddedildi: {e:?}"))?;

    let kontrol = Kontrol::kosudan(&rapor, &p, &opt, sozluk.aile(), &ozet, hassasiyet);
    let ckpt_ozeti = kontrol
        .yaz(Path::new(&ckpt_yolu))
        .map_err(|e| format!("kontrol noktasi yazilamadi ({ckpt_yolu}): {e}"))?;

    let markdown = kosu_markdown(
        &rapor,
        &ayar,
        &ozet,
        &ckpt_yolu,
        &ckpt_ozeti,
        kayit_sayisi,
        egitim_pencereleri.len(),
        dogrulama.len(),
        dogrulama_hepsi.len(),
        &sinav_yolu,
        damgalar.len(),
        dislanan,
    );
    crate::validate_output(markdown.as_bytes(), "egitim-kosu")?;
    if let Some(yol) = &rapor_yolu {
        std::fs::write(yol, &markdown).map_err(|e| format!("rapor yazilamadi {yol}: {e}"))?;
    }
    if let Some(yol) = &kayit_yolu {
        let sonuc = rapor.son_kaybi.is_finite()
            && rapor.baslangic_kaybi.is_finite()
            && rapor.son_kaybi < rapor.baslangic_kaybi;
        json_yaz(
            yol,
            &serde_json::json!({
                "kosucu": "model",
                "tarih": tarih(),
                "is": "egitim-turu",
                "olcut": {
                    "ad": "kosunun_son_egitim_kaybi_baslangic_kaybindan_dusuk",
                    "sonuc": sonuc,
                },
                "kaynaklar": {
                    "sure_saniye": rapor.sure_ms as f64 / 1000.0,
                    "girdi_jetonlari": rapor.jeton,
                    "onbellekli_jetonlari": 0,
                    "cikti_jetonlari": rapor.jeton,
                    "maliyet": 0.0,
                },
                "kanit": format!(
                    "adim {} -> {}; kayip {:.6} -> {:.6}; dogrulama {}; ckpt sha256 {}",
                    ayar.baslangic_adim,
                    rapor.adim,
                    rapor.baslangic_kaybi,
                    rapor.son_kaybi,
                    rapor
                        .en_iyi_dogrulama
                        .map_or("olculmedi".to_string(), |d| format!("{d:.6}")),
                    ckpt_ozeti
                ),
            }),
        )?;
    }
    print!("{markdown}");
    Ok(())
}

/// The spec's label from `training/model_spec.json`, checked against the Rust
/// constant: two places naming the same architecture must agree.
fn spec_etiketi() -> Result<String, String> {
    let metin = std::fs::read_to_string("training/model_spec.json")
        .map_err(|e| format!("model_spec.json okunamadi: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&metin).map_err(|e| format!("model_spec.json JSON degil: {e}"))?;
    let etiket = json["ad"]
        .as_str()
        .or_else(|| json["name"].as_str())
        .ok_or_else(|| "model_spec.json: ad alani yok".to_string())?;
    let spec = Spec::lubot_a1();
    let beyan = json["parametre_sayisi"].as_u64().unwrap_or(0);
    if beyan != 0 && beyan != spec.parametre_sayisi() as u64 {
        return Err(format!(
            "model_spec.json parametre sayisi {beyan} ama lubot-egitim {} diyor",
            spec.parametre_sayisi()
        ));
    }
    Ok(etiket.to_string())
}

/// Today's date, from the clock, as `YYYY-AA-GG`.
fn tarih() -> String {
    let saniye = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let gun = saniye / 86_400;
    // Sivil tarih, Howard Hinnant'in gun sayisi algoritmasiyla; kutuphane
    // cekmeden takvim yapragi.
    let z = gun as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let yil = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let gun_ay = doy - (153 * mp + 2) / 5 + 1;
    let ay = if mp < 10 { mp + 3 } else { mp - 9 };
    let yil = if ay <= 2 { yil + 1 } else { yil };
    format!("{yil:04}-{ay:02}-{gun_ay:02}")
}

#[allow(clippy::too_many_arguments)]
fn kosu_markdown(
    rapor: &KosuRaporu,
    ayar: &KosuAyari,
    ozet: &str,
    ckpt_yolu: &str,
    ckpt_ozeti: &str,
    kayit: usize,
    egitim_pencere: usize,
    dogrulama_pencere: usize,
    dogrulama_hepsi: usize,
    sinav_yolu: &str,
    damga_sayisi: usize,
    dislanan: usize,
) -> String {
    let mut md = String::new();
    md.push_str("# Egitim Turu: olculen adimlar\n\n");
    md.push_str("Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil\n");
    md.push_str("gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan\n");
    md.push_str("geliyor; hicbiri burada yeniden hesaplanmiyor.\n\n");
    md.push_str("| alan | deger |\n| --- | --- |\n");
    md.push_str(&format!("| kayit | {kayit} |\n"));
    md.push_str(&format!(
        "| pencere | {pencere} jeton; egitim {egitim_pencere}, dogrulama {dogrulama_pencere} (havuz {dogrulama_hepsi}) |\n",
        pencere = ayar.pencere_uzunlugu
    ));
    md.push_str(&format!(
        "| adim | {} -> {} (bu cagri {} adim) |\n",
        ayar.baslangic_adim,
        rapor.adim,
        rapor.harcanan_adim()
    ));
    md.push_str(&format!(
        "| epoch | {} -> {} (tavan {}) |\n",
        ayar.baslangic_epoch, rapor.epoch, ayar.epoch_tavani
    ));
    md.push_str(&format!(
        "| yigin | {} pencere/adim (iplik {}); kirpma {:.3}; lr {:.5}; sonum {:.3}; hesap {} |\n",
        ayar.yigin,
        lubot_egitim::kosu::iplik_sayisi(ayar),
        ayar.kirpma,
        ayar.ogrenme_orani,
        ayar.agirlik_sonumu,
        ayar.hassasiyet.etiket()
    ));
    md.push_str(&format!("| jeton | {} |\n", rapor.jeton));
    // Olcum catisi: kayip eğrisinin özeti, karmaşıklık ve hız. Hepsi koşunun
    // kayitlarindan; hata verirse rapor "olculmedi" yazar, sayi uydurmaz.
    // Olcum catisi disaridan adiyla cagrilir: turun hizi, EMA'si ve bit/bayt
    // degeri bu turdan gelir, tahmin edilmez.
    let olculer: Result<lubot_egitim::olcum::KosuOlculeri, lubot_egitim::olcum::OlcumHatasi> =
        rapor.olculer();
    match olculer {
        Ok(olcum) => {
            md.push_str(&olcum.markdown_satirlari().join("\n"));
            let ema: &lubot_egitim::olcum::KayipIstatistigi = &olcum.kayip;
            md.push_str(&format!(
                "| EMA | {} |\n",
                if ema.dolu() {
                    format!("{:.6}", ema.ema())
                } else {
                    "olculmedi".into()
                }
            ));
            let sayac: &lubot_egitim::olcum::JetonSayaci = &olcum.sayac;
            md.push_str(&format!(
                "| hiz | {} |\n",
                match (sayac.jeton_saniye(), sayac.milisaniye_jeton()) {
                    (Some(js), Some(ms)) => format!("{js:.1} jeton/s ({ms:.3} ms/jeton)"),
                    _ => "olculmedi".to_string(),
                }
            ));
            if let Some(bpb) = lubot_egitim::olcum::bit_basina_bayt(
                rapor.son_kaybi,
                rapor.jeton,
                olcum.bayt.unwrap_or(0),
            ) {
                md.push_str(&format!("| bit/bayt | {bpb:.4} |\n"));
            }
        }
        Err(hata) => md.push_str(&format!("| olcum | olculemedi: {hata} |\n")),
    }
    md.push('\n');
    md.push_str(&format!(
        "| kayip | {:.6} -> {:.6} |\n",
        rapor.baslangic_kaybi, rapor.son_kaybi
    ));
    if let Some(d) = rapor.dusus_orani() {
        md.push_str(&format!("| dusus | {:.2}% |\n", d * 100.0));
    }
    md.push_str(&format!(
        "| en iyi dogrulama | {} |\n",
        rapor.en_iyi_dogrulama.map_or("olculmedi".to_string(), |d| {
            format!(
                "{d:.6} (adim {})",
                rapor.en_iyi_dogrulama_adimi.unwrap_or(0)
            )
        })
    ));
    let durma: DurmaNedeni = rapor.durma_nedeni;
    md.push_str(&format!(
        "| durma | {} ({}) |\n",
        durma.etiket(),
        match durma {
            DurmaNedeni::AdimButcesi => "butce doldu",
            DurmaNedeni::EpochKurali => "dogrulama iyilesmedi",
            DurmaNedeni::EpochTavani => "epoch tavani",
        }
    ));
    md.push_str(&format!("| kirpilan adim | {} |\n", rapor.kirpilan_adim));
    // Perplexity bir kaybin tek basina soylemedigi seyi soyler: 3.85'in
    // "kac jeton arasinda tereddut" demek oldugunu. Donusum tanimsizsa
    // "olculmedi" yazilir, sayi uydurulmaz.
    md.push_str(&format!(
        "| perplexity | {} |\n",
        match lubot_egitim::olcum::perplexity(rapor.son_kaybi) {
            Some(p) => format!("{p:.4}"),
            None => "olculmedi".to_string(),
        }
    ));
    md.push_str(&format!(
        "| sure | {:.1} sn (makineye bagli; ratchet'e girmez) |\n",
        rapor.sure_ms as f64 / 1000.0
    ));
    md.push_str(&format!("| korpus ozeti | `{ozet}` |\n"));
    md.push_str(&format!(
        "| kontrol noktasi | `{ckpt_yolu}` sha256 `{ckpt_ozeti}` ({} v{SURUM}, {} blok) |\n",
        String::from_utf8_lossy(SIHIR),
        BLOK_ADLARI.len()
    ));
    md.push_str(&format!(
        "| held-out | `{sinav_yolu}`; {damga_sayisi} damga, {dislanan} kayit egitim akisindan cikarildi |\n"
    ));
    md.push_str(&format!(
        "| devam konumu | {} (bitmemis epoch icin) |\n\n",
        rapor.devam_konum
    ));
    md.push_str("## Epoch egrisi\n\n| epoch | kayip |\n| --- | --- |\n");
    for (i, kayip) in rapor.epoch_kaybi.iter().enumerate() {
        md.push_str(&format!(
            "| {} | {kayip:.6} |\n",
            ayar.baslangic_epoch as usize + i + 1
        ));
    }
    md.push_str("\n## Dogrulama egrisi\n\n| adim | epoch | kayip | pencere | jeton |\n| --- | --- | --- | --- | --- |\n");
    for d in &rapor.dogrulama_egrisi {
        md.push_str(&format!(
            "| {} | {} | {:.6} | {} | {} |\n",
            d.adim, d.epoch, d.kayip, d.pencere, d.jeton
        ));
    }
    md
}

/// `lubot cikarim denetle|puanla|sirala` - the inference surface.
pub fn cmd_cikarim(args: &[String]) -> Result<(), String> {
    let alt = args
        .first()
        .ok_or_else(|| "cikarim alt komut istiyor: denetle | puanla | sirala".to_string())?
        .clone();
    let kalan = &args[1..];
    match alt.as_str() {
        "denetle" => cikarim_denetle(kalan),
        "puanla" => cikarim_puanla(kalan),
        "sirala" => cikarim_sirala(kalan),
        other => Err(format!(
            "bilinmeyen cikarim alt komutu `{other}`: denetle | puanla | sirala"
        )),
    }
}

fn kimlikleri_coz(ham: &str, sozluk: usize) -> Result<Vec<u32>, String> {
    let mut kimlikler: Vec<u32> = Vec::new();
    for parca in ham.split(',') {
        let parca = parca.trim();
        if parca.is_empty() {
            continue;
        }
        let deger: u32 = parca
            .parse()
            .map_err(|_| format!("jeton kimligi sayi degil: `{parca}`"))?;
        if deger as usize >= sozluk {
            return Err(format!("jeton {deger} sozluk disinda (0..{sozluk})"));
        }
        kimlikler.push(deger);
    }
    Ok(kimlikler)
}

/// Metni jetonlara çevirir: `@` ile başlayan değer **metindir** (dosya yolu
/// varsa dosyadan, yoksa düz metin olarak), aksi hâlde virgüllü kimlik listesi.
///
/// Bu ayrım olmadan kıyas ölçümü (iki model aynı şıkları görsün) çağıranın her
/// taraf için jeton listesi üretmesini gerektirirdi; jetonlayıcı iki yerde
/// yaşasaydı iki cevap olurdu.
fn metin_veya_kimlik(ham: &str, c: &Cikarim) -> Result<Vec<u32>, String> {
    let Some(metin) = ham.strip_prefix('@') else {
        return kimlikleri_coz(ham, c.sozluk_boyutu());
    };
    let icerik = if Path::new(metin).is_file() {
        std::fs::read_to_string(metin).map_err(|e| format!("{metin} okunamadi: {e}"))?
    } else {
        metin.to_string()
    };
    let sozluk = lubot_jeton::Sozluk::yukle(Path::new("training/tokenizer/lubot-bpe-v2.json"))
        .map_err(|e| format!("sozluk reddedildi: {e}"))?;
    let jetonlar = sozluk.kodla(&icerik);
    if jetonlar.is_empty() {
        return Err("metin jetonlanmadi: bos girdi puanlanmaz".to_string());
    }
    Ok(jetonlar)
}

fn cikarim_yukle(b: &Bayraklar) -> Result<Cikarim, String> {
    let yol = b.zorunlu("--ckpt")?;
    Cikarim::yukle(Path::new(yol))
        .map_err(|e: CikarimHatasi| format!("kontrol noktasi yuklenemedi ({yol}): {e}"))
}

fn cikarim_denetle(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(args, &["--ckpt", "--kimlikler"])?;
    let c = cikarim_yukle(&b)?;
    let kimlikler = kimlikleri_coz(b.zorunlu("--kimlikler")?, c.sozluk_boyutu())?;
    let rapor: OnbellekRaporu = c
        .onbellek_denetimi(&kimlikler)
        .map_err(|e| format!("onbellek denetimi reddedildi: {e}"))?;
    let md = format!(
        "# Cikarim Onbellek Denetimi\n\n\
         Ayni dizi uc yoldan gecti: onbellekli artimli gecis, her oneki sifirdan\n\
         yeniden isleyen tam gecis ve egitim cekirdeginin kendi kaybi. Ucuncu\n\
         gorus olmadan ilk ikisi ayni hatayi paylasabilir.\n\n\
         | alan | deger |\n| --- | --- |\n\
         | adim | {} |\n\
         | sozluk ailesi | {} |\n\
         | konum | {} |\n\
         | onbellekli ortalama log-olasilik | {:.12} |\n\
         | tam gecis ortalamasi | {:.12} |\n\
         | egitim cekirdegi | {:.12} |\n\
         | en buyuk fark (onbellek/tam) | {:.3e} (tolerans {:.0e}) |\n\
         | en buyuk fark (onbellek/egitim) | {:.3e} |\n",
        c.adim(),
        c.sozluk_aile(),
        rapor.konum,
        rapor.onbellekli,
        rapor.tam_gecis,
        rapor.egitim_cekirdegi,
        rapor.en_buyuk_fark_onbellek,
        CACHE_TOLERANCE,
        rapor.en_buyuk_fark_egitim
    );
    crate::validate_output(md.as_bytes(), "cikarim-denetle")?;
    print!("{md}");
    Ok(())
}

fn cikarim_puanla(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(args, &["--ckpt", "--baglam", "--metin"])?;
    let c = cikarim_yukle(&b)?;
    let baglam = match b.metin("--baglam") {
        Some(ham) if !ham.is_empty() => metin_veya_kimlik(ham, &c)?,
        _ => Vec::new(),
    };
    let metin = metin_veya_kimlik(b.zorunlu("--metin")?, &c)?;
    let puan = c
        .puanla(&baglam, &metin)
        .map_err(|e| format!("puanlama reddedildi: {e}"))?;
    let md = format!(
        "# Cikarim Puani\n\n\
         Puan, metnin jeton basina ortalama log-olasiligidir: baglamdan sonra bu\n\
         metnin ne kadar beklenir oldugu. Sifira yakin daha iyi. Baglamsiz ilk\n\
         jeton puanlanmaz - o sayi sozluk onselidir, metnin olcusu degil.\n\n\
         | alan | deger |\n| --- | --- |\n\
         | baglam jetonu | {} |\n\
         | metin jetonu | {} |\n\
         | jeton basina ortalama log-olasilik | {:.12} |\n\
         | adim | {} |\n",
        baglam.len(),
        metin.len(),
        puan,
        c.adim()
    );
    crate::validate_output(md.as_bytes(), "cikarim-puanla")?;
    print!("{md}");
    Ok(())
}

fn cikarim_sirala(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(args, &["--ckpt", "--baglam", "--adaylar"])?;
    let c = cikarim_yukle(&b)?;
    let baglam = kimlikleri_coz(b.metin("--baglam").unwrap_or(""), c.sozluk_boyutu())?;
    let yol = b.zorunlu("--adaylar")?;
    let metin = std::fs::read_to_string(yol).map_err(|e| format!("{yol} okunamadi: {e}"))?;
    let mut adaylar: Vec<Vec<u32>> = Vec::new();
    for (sira, satir) in metin.lines().enumerate() {
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        adaylar.push(
            kimlikleri_coz(satir, c.sozluk_boyutu())
                .map_err(|e| format!("{yol} satir {}: {e}", sira + 1))?,
        );
    }
    let siralama: Vec<Aday> = c
        .pasaj_sirala(&baglam, &adaylar)
        .map_err(|e| format!("siralama reddedildi: {e}"))?;
    let mut md = String::from(
        "# Aday Siralamasi\n\n\
         Adaylar baglama karsi jeton basina ortalama log-olasilikla siralanir.\n\
         Esit puanli adaylar esittir; aralarinda sira uydurulmaz, cagiranin\n\
         verdigi sira numarasi kullanilir.\n\n\
         | sira | aday | puanlanan jeton | jeton basina puan |\n| --- | --- | --- | --- |\n",
    );
    for (i, aday) in siralama.iter().enumerate() {
        md.push_str(&format!(
            "| {} | {} | {} | {:.12} |\n",
            i + 1,
            aday.sira,
            aday.jeton,
            aday.puan
        ));
    }
    crate::validate_output(md.as_bytes(), "cikarim-sirala")?;
    print!("{md}");
    Ok(())
}

/// `lubot egitim-karsilastir` - the two compute kernels, measured against each
/// other on a real corpus window.
///
/// Why this exists as a command and not only as a test: the f32 kernel's
/// only honest justification is that it agrees with the f64 one, and on a
/// small synthetic spec that agreement is easy. This runs both kernels on the
/// window length the spec actually trains with, from a checkpoint or from the
/// run's own initialisation, and prints the worst relative deviation per
/// tensor. A machine whose f32 path drifts badly says so here, before a run
/// is spent on it.
///
/// The output is markdown, like every other measurement this project reports.
pub fn cmd_egitim_karsilastir(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(
        args,
        &[
            "--corpus",
            "--vocab",
            "--pencere",
            "--tohum",
            "--ckpt",
            "--rapor",
            "--sessiz",
        ],
    )?;
    let spec = Spec::lubot_a1();
    let korpus_yolu = b.zorunlu("--corpus")?.to_string();
    let vocab_yolu = b
        .metin("--vocab")
        .unwrap_or("training/tokenizer/lubot-bpe-v2.json")
        .to_string();
    let tohum = b.sayi::<u64>("--tohum")?.unwrap_or(20_260_924);
    let pencere = match b.sayi::<usize>("--pencere")? {
        Some(n) => n,
        None => spec.max_seq_len,
    };
    if pencere < 2 || pencere > spec.max_seq_len {
        return Err(format!(
            "pencere {pencere} spec ile uyusmuyor (2..={})",
            spec.max_seq_len
        ));
    }
    let sozluk = lubot_jeton::Sozluk::yukle(Path::new(&vocab_yolu))
        .map_err(|e| format!("sozluk yuklenemedi {vocab_yolu}: {e}"))?;
    let kayitlar = korpus_oku(&korpus_yolu, &sozluk)?;
    let parametreler: Parametreler = match b.metin("--ckpt") {
        Some(yol) => {
            Kontrol::yukle(Path::new(yol))
                .map_err(|e| format!("kontrol noktasi reddedildi: {e}"))?
                .parametreler
        }
        None => Parametreler::mup_init(spec, tohum, INIT_STD_EMBEDDING),
    };
    // Pencereyi dolduracak ilk kayit secilir: kisa kayitla olcmek, modelin
    // gordugu baglami olcmemek olurdu. Kayit kimligi rapora yazilir, boylece
    // "hangi metinde olctun" sorusu cevapsiz kalmaz.
    let kayit = kayitlar
        .iter()
        .find(|k| k.jetonlar.len() >= pencere)
        .ok_or_else(|| {
            let en_uzun = kayitlar.iter().map(|k| k.jetonlar.len()).max().unwrap_or(0);
            format!("korpusda {pencere} jetonluk kayit yok (en uzun {en_uzun})")
        })?;
    let jetonlar = &kayit.jetonlar;
    let dilim = &jetonlar[..pencere];
    let girdi: Vec<usize> = dilim[..pencere - 1].iter().map(|j| *j as usize).collect();
    let hedef: Vec<usize> = dilim[1..].iter().map(|j| *j as usize).collect();
    let kaynak = vec![0u32; girdi.len()];

    let p32 = lubot_egitim::kernel32::Parametreler32::indir(&parametreler);
    let (k64, g64) =
        lubot_egitim::ileri_ve_geri_paket(spec, &parametreler, &girdi, &hedef, &kaynak);
    let (k32, g32) =
        lubot_egitim::kernel32::ileri_ve_geri_paket_32(spec, &p32, &girdi, &hedef, &kaynak);
    let g32_f64 = g32.geri_f64();
    let tensors: Vec<(&str, &Vec<f64>, &Vec<f64>)> = vec![
        ("embedding", &g64.embedding, &g32_f64.embedding),
        ("wq", &g64.wq, &g32_f64.wq),
        ("wk", &g64.wk, &g32_f64.wk),
        ("wv", &g64.wv, &g32_f64.wv),
        ("wo", &g64.wo, &g32_f64.wo),
        ("w1", &g64.w1, &g32_f64.w1),
        ("w2", &g64.w2, &g32_f64.w2),
        ("ln1", &g64.ln1_olcek, &g32_f64.ln1_olcek),
        ("lnf", &g64.lnf_sapma, &g32_f64.lnf_sapma),
    ];
    let mut satirlar = String::new();
    let mut en_kotu = 0.0f64;
    let mut en_kotu_ad = "";
    for (ad, a, c) in tensors {
        let buyukluk = a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let fark = a
            .iter()
            .zip(c.iter())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        // Sifira yakin tensorde mutlak taban: goreli oran tek basina
        // yaniltici olur (1e-12'lik bir gradyanda 1e-15'lik fark %100'dur).
        let oran = fark / buyukluk.max(1e-6);
        if oran > en_kotu {
            en_kotu = oran;
            en_kotu_ad = ad;
        }
        satirlar.push_str(&format!(
            "| {ad} | {buyukluk:.3e} | {fark:.3e} | {oran:.3e} |\n"
        ));
    }
    let mut md = String::new();
    md.push_str("# Hesap cekirdegi karsilastirmasi\n\n");
    md.push_str(
        "Ayni pencere, ayni parametreler; f64 referans, f32 olculen. Tolerans\n\
         `gates`teki gibi goreli, sifira yakin tensorde mutlak tabanli.\n\n",
    );
    md.push_str("| olcu | deger |\n| --- | --- |\n");
    md.push_str(&format!("| korpus | `{korpus_yolu}` |\n"));
    md.push_str(&format!("| pencere | {pencere} jeton |\n"));
    md.push_str(&format!("| kayit | `{}` |\n", kayit.kimlik));
    md.push_str(&format!(
        "| kaynak | {} |\n",
        if b.metin("--ckpt").is_some() {
            "kontrol noktasi"
        } else {
            "mup init (egitilmemis)"
        }
    ));
    md.push_str(&format!(
        "| kayip | f64 {k64:.6} | \n| kayip (f32) | {k32:.6} |\n"
    ));
    md.push_str(&format!(
        "| kayip farki | {:.3e} |\n",
        (k64 - f64::from(k32)).abs()
    ));
    md.push_str(&format!(
        "| en kotu tensorsel oran | {en_kotu:.3e} ({en_kotu_ad}) |\n\n"
    ));
    md.push_str("| tensor | buyukluk | fark | oran |\n| --- | --- | --- | --- |\n");
    md.push_str(&satirlar);
    md.push_str(&format!(
        "\n## Karar\n\nEn kotu oran {en_kotu:.3e} ({en_kotu_ad}); tolerans 2e-3. {}\n",
        if en_kotu < 2e-3 {
            "Iki cekirdek bu pencerede uyusuyor."
        } else {
            "Iki cekirdek bu pencerede AYRISTI: f32 secenegi bu makinede kapatilmali."
        }
    ));
    crate::validate_output(md.as_bytes(), "egitim-karsilastir")?;
    if !b.var_mi("--sessiz") {
        print!("{md}");
    }
    if let Some(yol) = b.metin("--rapor") {
        std::fs::write(yol, &md).map_err(|e| format!("rapor yazilamadi: {e}"))?;
    }
    Ok(())
}

/// `lubot sinav-kosu` - the held-out exam, graded by ranking.
///
/// The exam questions were written against specific passages; each question's
/// stamped `content_id` names the passage that answers it. This run scores that
/// passage against distractors drawn from the corpus and asks whether it came
/// first - a mechanical criterion, and the only one the A1 model can be
/// measured on, because it ranks and does not write.
///
/// The score is an upper bound and the record says so: the question text was
/// derived from the passage's own opening line during construction, so part of
/// the question is visible in the candidate.
pub fn cmd_sinav_kosu(args: &[String]) -> Result<(), String> {
    let b = Bayraklar::ayikla(
        args,
        &[
            "--ckpt", "--sinav", "--corpus", "--vocab", "--aday", "--tohum", "--rapor", "--kayit",
        ],
    )?;
    let ckpt = b.zorunlu("--ckpt")?;
    let sinav_yolu = b.zorunlu("--sinav")?.to_string();
    let korpus_yolu = b.zorunlu("--corpus")?.to_string();
    let vocab_yolu = b
        .metin("--vocab")
        .unwrap_or("training/tokenizer/lubot-bpe-v2.json")
        .to_string();
    let aday_sayisi = b.sayi::<usize>("--aday")?.unwrap_or(4).max(1);
    let tohum = b.sayi::<u64>("--tohum")?.unwrap_or(20_260_924);
    let rapor_yolu = b.metin("--rapor").map(str::to_string);
    let kayit_yolu = b.metin("--kayit").map(str::to_string);

    let c = Cikarim::yukle(Path::new(ckpt))
        .map_err(|e: CikarimHatasi| format!("kontrol noktasi yuklenemedi ({ckpt}): {e}"))?;
    let sozluk = lubot_jeton::Sozluk::yukle(Path::new(&vocab_yolu))
        .map_err(|e| format!("sozluk reddedildi ({vocab_yolu}): {e}"))?;
    let kayitlar = korpus_oku(&korpus_yolu, &sozluk)?;
    let metinler: std::collections::HashMap<&str, &Kayit> =
        kayitlar.iter().map(|k| (k.kimlik.as_str(), k)).collect();

    let sinav_metni =
        std::fs::read_to_string(&sinav_yolu).map_err(|e| format!("{sinav_yolu} okunamadi: {e}"))?;
    let mut sorular: Vec<(String, String)> = Vec::new(); // (content_id, soru)
    for (sira, satir) in sinav_metni.lines().enumerate() {
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        let deger: serde_json::Value = serde_json::from_str(satir)
            .map_err(|e| format!("{sinav_yolu} satir {}: {e}", sira + 1))?;
        let kimlik = deger["content_id"]
            .as_str()
            .ok_or_else(|| format!("{sinav_yolu} satir {}: content_id yok", sira + 1))?;
        let soru = deger["soru"]
            .as_str()
            .ok_or_else(|| format!("{sinav_yolu} satir {}: soru yok", sira + 1))?;
        sorular.push((kimlik.to_string(), soru.to_string()));
    }
    if sorular.is_empty() {
        return Err(format!("sinav seti bos: {sinav_yolu}"));
    }

    let pencere = c.spec().max_seq_len;
    let mut dogru = 0usize;
    let mut satirlar: Vec<(String, usize, usize, f64)> = Vec::new();
    let mut jeton_sayisi: u64 = 0;
    for (sira, (kimlik, soru)) in sorular.iter().enumerate() {
        let Some(pasaj) = metinler.get(kimlik.as_str()) else {
            return Err(format!(
                "sinav {kimlik} korpusta yok: soru held-out bir pasaja dayaniyor ama pasaj veride degil"
            ));
        };
        let baglam = sozluk.kodla(soru);
        let yer = pencere.saturating_sub(baglam.len()).max(1);
        // Soru pasajin acilis satirindan turetildi; adayin BAStan kesilmesi
        // gerekir, sondan degil - cevap pasajin basindadir.
        let aday_pasaj: Vec<u32> = pasaj.jetonlar.iter().take(yer).copied().collect();
        let mut adaylar: Vec<Vec<u32>> = vec![aday_pasaj];
        let mut eklenen = 0usize;
        let mut adim = 1usize;
        while eklenen < aday_sayisi.saturating_sub(1) {
            let konum = (karisim_konum(tohum, sira, adim)) % kayitlar.len();
            let aday_kayit = &kayitlar[konum];
            adim += 1;
            if aday_kayit.kimlik == *kimlik {
                continue;
            }
            let parca: Vec<u32> = aday_kayit.jetonlar.iter().take(yer).copied().collect();
            if parca.is_empty() {
                continue;
            }
            adaylar.push(parca);
            eklenen += 1;
        }
        let siralama = c
            .pasaj_sirala(&baglam, &adaylar)
            .map_err(|e| format!("sinav sorusu {} reddedildi: {e}", sira + 1))?;
        jeton_sayisi += adaylar.iter().map(|a| a.len() as u64).sum::<u64>();
        let basarili = siralama.first().is_some_and(|en_iyi| en_iyi.sira == 0);
        if basarili {
            dogru += 1;
        }
        let dogru_puan = siralama
            .iter()
            .find(|a| a.sira == 0)
            .map_or(f64::NAN, |a| a.puan);
        let dogru_sira = siralama
            .iter()
            .position(|a| a.sira == 0)
            .map_or(0, |p| p + 1);
        satirlar.push((kimlik.clone(), dogru_sira, adaylar.len(), dogru_puan));
    }

    let soru_sayisi = sorular.len();
    let mut md = String::new();
    md.push_str("# Sinav Kosusu: held-out sorular, siralama olcumu\n\n");
    md.push_str("Her soru, damgali pasajina karsi korpustan cekilen celdiricilerle\n");
    md.push_str("siralanir; olcut dogru pasajin ilk sirada cikmasidir. Bu bir UST\n");
    md.push_str("SINIRDIR: soru metni pasajin acilis satirindan turetildi, yani sorunun\n");
    md.push_str("bir parcasi adayin icinde gorunuyor. Model uretmez; bu yuzden alinti\n");
    md.push_str("dogrulugu degil, pasaj siralama dogrulugu olculuyor.\n\n");
    md.push_str("| alan | deger |\n| --- | --- |\n");
    md.push_str(&format!(
        "| kontrol noktasi | `{ckpt}` (adim {}) |\n",
        c.adim()
    ));
    md.push_str(&format!("| sorus | {soru_sayisi} |\n"));
    md.push_str(&format!(
        "| ilk sirada dogru pasaj | {dogru} / {soru_sayisi} |\n"
    ));
    md.push_str(&format!("| celdirici | {aday_sayisi} aday/soru |\n"));
    md.push_str(&format!("| puanlanan jeton | {jeton_sayisi} |\n\n"));
    md.push_str(
        "| soru | dogru pasajin sirasi | aday | dogru pasajin puani |\n| --- | --- | --- | --- |\n",
    );
    for (i, (kimlik, sira, aday, puan)) in satirlar.iter().enumerate() {
        md.push_str(&format!(
            "| {} (`{}`) | {} | {} | {puan:.6} |\n",
            i + 1,
            &kimlik[..8.min(kimlik.len())],
            sira,
            aday
        ));
    }
    crate::validate_output(md.as_bytes(), "sinav-kosu")?;
    if let Some(yol) = &rapor_yolu {
        std::fs::write(yol, &md).map_err(|e| format!("rapor yazilamadi {yol}: {e}"))?;
    }
    if let Some(yol) = &kayit_yolu {
        let _ = sinav_damgalari(&sinav_yolu)?;
        json_yaz(
            yol,
            &serde_json::json!({
                "kosucu": "model",
                "tarih": tarih(),
                "is": "sinav-kosu",
                "olcut": {
                    "ad": "sinav_sorularinda_dogru_pasajin_ilk_sirada_cikmasi",
                    "sonuc": dogru == soru_sayisi,
                },
                "kaynaklar": {
                    "sure_saniye": 0.0,
                    "girdi_jetonlari": 0,
                    "onbellekli_jetonlari": 0,
                    "cikti_jetonlari": jeton_sayisi,
                    "maliyet": 0.0,
                },
                "kanit": format!(
                    "held-out {soru_sayisi} soru; dogru pasaj ilk sirada {dogru}; \
                     soru metni pasajin acilis satirindan turetildigi icin skor ust sinirdir; \
                     puanlanan jeton {jeton_sayisi}"
                ),
            }),
        )?;
    }
    print!("{md}");
    Ok(())
}

/// A deterministic position for distractor selection: same seed, same question
/// index and same attempt give the same record on every machine.
fn karisim_konum(tohum: u64, soru: usize, deneme: usize) -> usize {
    let mut x = tohum
        ^ (soru as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (deneme as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x % 1_000_003) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kayit(kimlik: &str, uzunluk: usize) -> Kayit {
        Kayit {
            kimlik: kimlik.to_string(),
            jetonlar: (0..uzunluk).map(|i| (i % 29) as u32).collect(),
        }
    }

    #[test]
    fn the_corpus_stamp_ignores_the_order_and_notices_a_missing_record() {
        let a = vec![kayit("a", 4), kayit("b", 4), kayit("c", 4)];
        let mut b = vec![kayit("c", 4), kayit("a", 4), kayit("b", 4)];
        assert_eq!(
            korpus_ozeti(&a, "aile", "lubot-a1"),
            korpus_ozeti(&b, "aile", "lubot-a1")
        );
        b.pop();
        assert_ne!(
            korpus_ozeti(&a, "aile", "lubot-a1"),
            korpus_ozeti(&b, "aile", "lubot-a1")
        );
        assert_ne!(
            korpus_ozeti(&a, "aile", "lubot-a1"),
            korpus_ozeti(&a, "baska-aile", "lubot-a1")
        );
    }

    #[test]
    fn the_eval_only_records_leave_the_training_stream_and_are_counted() {
        let kayitlar = vec![kayit("a", 4), kayit("b", 4), kayit("c", 4)];
        let (kalan, cikarilan) =
            damgalilari_cikar(kayitlar.clone(), &["b".to_string(), "zzz".to_string()]);
        assert_eq!(cikarilan, 1);
        assert_eq!(kalan.len(), 2);
        assert!(kalan.iter().all(|k| k.kimlik != "b"));
        let (hepsi, sifir) = damgalilari_cikar(kayitlar.clone(), &[]);
        assert_eq!(sifir, 0);
        assert_eq!(hepsi.len(), kayitlar.len());
        let (hepsi, sifir) = damgalilari_cikar(kayitlar, &["zzz".to_string()]);
        assert_eq!(sifir, 0);
        assert_eq!(hepsi.len(), 3);
    }

    #[test]
    fn the_validation_subset_is_spread_and_never_empty() {
        let pencereler: Vec<lubot_egitim::PaketPencere> = (0..20)
            .map(|i| lubot_egitim::PaketPencere {
                kimlikler: vec![i as u32; 4],
                kaynak: vec![0u32; 4],
            })
            .collect();
        let hepsi = dogrulama_seyrelt(&pencereler, None);
        assert_eq!(hepsi.len(), 20);
        let secilen = dogrulama_seyrelt(&pencereler, Some(5));
        assert_eq!(secilen.len(), 5);
        assert_eq!(secilen[0].kimlikler[0], pencereler[2].kimlikler[0]);
        assert_eq!(secilen[4].kimlikler[0], pencereler[18].kimlikler[0]);
        let az = dogrulama_seyrelt(&pencereler, Some(0));
        assert_eq!(az.len(), 20, "sifir adet 'hepsi' demektir, 'bos' degil");
        let fazla = dogrulama_seyrelt(&pencereler, Some(100));
        assert_eq!(fazla.len(), 20);
    }

    #[test]
    fn the_flag_reader_refuses_a_typo_rather_than_ignoring_it() {
        let args = vec![
            "--corpus".to_string(),
            "a".to_string(),
            "--corpu".to_string(),
        ];
        assert!(Bayraklar::ayikla(&args, &["--corpus", "--vocab"]).is_err());
        let eksik = vec!["--corpus".to_string()];
        assert!(Bayraklar::ayikla(&eksik, &["--corpus"]).is_err());
        let bayrak = vec!["--f32".to_string()];
        let b = Bayraklar::ayikla(&bayrak, &["--f32"]).expect("ayikla");
        assert!(b.var_mi("--f32"));
        assert_eq!(b.metin("--f32"), Some(""));
    }

    #[test]
    fn a_token_id_outside_the_vocabulary_is_refused_by_the_parser() {
        assert_eq!(kimlikleri_coz("1,2,3", 8), Ok(vec![1, 2, 3]));
        assert!(kimlikleri_coz("1,x", 8).is_err());
        assert!(kimlikleri_coz("1,99", 8).is_err());
        assert_eq!(kimlikleri_coz(" 1 , 2 ", 8), Ok(vec![1, 2]));
    }

    #[test]
    fn the_distractor_position_is_a_function_of_its_inputs() {
        assert_eq!(karisim_konum(7, 0, 1), karisim_konum(7, 0, 1));
        assert_ne!(karisim_konum(7, 0, 1), karisim_konum(7, 1, 1));
        assert!(karisim_konum(7, 3, 9) < 1_000_003);
    }

    #[test]
    fn the_date_is_a_civil_date() {
        let t = tarih();
        assert_eq!(t.len(), 10, "tarih bicimi bozuk: {t}");
        let yil: i32 = t[..4].parse().expect("yil");
        assert!((2024..2100).contains(&yil), "yil beklenmedik: {t}");
    }
}
