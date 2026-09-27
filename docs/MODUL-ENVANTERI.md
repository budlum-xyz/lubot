# Modul envanteri: sekiz modul nerede duruyor

Bu belge tek bir soruya cevap verir: **hangi mimari modul bu agacta var, kaniti
nerede, ve hangisi hala yok.** Cevap iddia degil, dosya yoludur — her satirdaki
her yol `modul-envanteri-kapisi` tarafindan denetlenir: kaynak dosya yoksa,
kapi adi kayitli degilse ya da olcum kaydi eksikse kapi kirmizi yanar.

Belgenin varlik sebebi sudir: mimari moduller iki isim kumesinde yasir —
kavramsal adlar (`hadamard_mlp`, `sinkhorn_router`) ve bu agacin somut
dosyalari (`mlp_hadamard.rs`, `yonlendirme.rs`). Iki kume arasinda yazili bir
karsilik olmadigi surece "mimari tamamlandi mi" sorusu **olculemez** bir
sorudur; herkes kendi listesine bakar ve farkli cevap verir.

## Okuma kurallari

- **kapi** sutunu, `gates/check.py` icinde kayitli bir kapi adi tasir. Bos
  hucre (`—`) o modulun deposal bir kapisi olmadigini soyler; bu bir eksiktir
  ve gizlenmez.
- **olcum kaydi** sutunu `training/eval/sonuclar/` altindaki bir dosyadir.
  `—` ise olcum modulun kendi testlerinin icindedir ve ayri bir kayda
  gecmemistir.
- **durum** sutunu yalniz uc degerden birini alir: `bagli` (bir egitim ya da
  cikarim cagrisindan gerçekten geciliyor), `bagimsiz` (yazildi ve olculdu ama
  hicbir cagri yolundan gecmiyor), `yok`.

## Ana mimarinin sekiz modulu

| modul | bu depodaki yeri | kapi | olcum kaydi | durum |
|---|---|---|---|---|
| `kodlayici_omurga` | `crates/kodlayici/src/lib.rs`, `crates/kodlayici/src/blok.rs`, `crates/transformer/src/lib.rs` | `kodlayici-kapisi` | `kodlayici-2026-09-27.json`, `dikkat-kadansi-2026-09-27.json`, `norm-yeri-2026-09-27.json` | bagli |
| `hadamard_mlp` | `crates/egitim/src/mlp_hadamard.rs` | `hadamard-mlp-kapisi` | `hadamard-mlp-2026-09-27.json` | bagimsiz |
| `gqa_engram_attention` | `crates/egitim/src/lib.rs`, `crates/egitim/src/engram.rs` | `engram-kapisi` | `engram-2026-09-26.json` | bagli |
| `hyperconnections` | `crates/egitim/src/cok_serit.rs` | `cok-serit-kapisi` | `cok-serit-2026-09-26.json` | bagimsiz |
| `sinkhorn_router` | `crates/egitim/src/yonlendirme.rs` | `yonlendirme-kapisi` | `sinkhorn-yonlendirme-2026-09-27.json` | bagimsiz |
| `decision_head` | `crates/tomurcuk/src/lib.rs`, `crates/tomurcuk/src/kalibrasyon.rs` | `kalibrasyon-bandi-kapisi`, `omurga-karar-kayitlari` | `kalibrasyon-2026-09-26.json`, `tipli-karar-2026-09-27.json` | bagli |
| `schema_decoder` | `crates/read/src/output_schema.rs`, `crates/egitim/src/sema_cozucu.rs`, `crates/egitim/src/sema_jeton.rs` | `ai-output-schema-enforced`, `sema-cozucu-reddeder`, `sema-jeton-maskeler` | `sema-kapsam-2026-09-24.json`, `sema-jeton-2026-09-27.json` | bagli |
| `cq2_quant` | `crates/nicem/src/lib.rs`, `crates/nicem/src/grup.rs`, `crates/tasiyici/src/lib.rs` | `bit-budget-is-arithmetic` | `nicem-2026-09-27.json` | bagli |

`gqa_engram_attention` satiri iki parcalidir ve durumu **parca parca** okunur:
GQA egitim cekirdeginde baglidir (`n_kv_heads`, `qkv_dokunus`, `qk_norm` ayni
ileri/geri gecisten geciyor), engram tablosu ise bagimsizdi — 3.8 blogu onu bir
kompozisyon icine aldi ama o blok da hicbir aileye bagli degil. Satirin durumu
bu yuzden GQA'ya gore `bagli` yazilir; engramin kendi satiri asagidadir.

`schema_decoder` satiri da iki parcalidir ve bir **itiraz** tasiyor. Mimari
bu modulden "gecerli cikti uzayini grammar/sema ile daraltarak decode"
etmesini bekler; `read::output_schema` bunu yapmaz, **bitmis** bir ciktiyi
dogrular. Ikisi ayni kuralin iki yarisidir ve biri otekinin yerine gecmez:
dogrulayici "bu gecerli miydi?" sorusuna cevap verir, cozucunun ihtiyaci olan
soru "bu hala gecerli olabilir mi?"dir ve o soru her konumda, bir logit
orneklenmeden once sorulmak zorundadir. Cozme tarafi bu agacta **yoktu**;
`crates/egitim/src/sema_cozucu.rs` onu ekliyor. Satirin durumu dogrulayici
yarisina gore `bagli` yazili kalir - cozucu yarisi **bagimsizdir** ve hicbir
cagri yolundan gecmez. Bu, envanterin "sekizin sekizi de var" sonucunu
degistirmez ama onu daha dar okutur: sekizinci modulun iki yarisindan biri
2026-09-27'de eklendi, oteki zaten duruyordu.

## Ana listede olmayan, bu agacta olan

| modul | bu depodaki yeri | kapi | olcum kaydi | durum |
|---|---|---|---|---|
| aile kesitleri | `crates/egitim/src/kesit.rs` | `kesit-kapisi` | `kesit-2026-09-27.json` | bagimsiz |
| sifir merkezli RMS norm | `crates/egitim/src/normalizasyon.rs` | `normalizasyon-kapisi` | `normalizasyon-2026-09-27.json` | bagimsiz |
| birlesik blok (3.8) | `crates/egitim/src/birlesik.rs` | `birlesik-kapisi` | `birlesik-2026-09-27.json` | bagimsiz |
| engram tasima (tablo model durumunun parcasi) | `crates/egitim/src/engram_tasima.rs` | `omurga-karar-kayitlari` | `engram-tasima-2026-09-27.json` | bagimsiz |
| kademe egitimi (3.5 kalem 1) | `crates/egitim/src/kademe.rs` | `kademe-kapisi` | `kademe-2026-09-27.json` | bagimsiz |

## Bu envanterin soyledigi

1. Sekiz modulun **sekizi de** bu agacta var. Mimari yazilacak bir is degil;
   kalan is **baglama** isidir.
2. `kodlayici_omurga`'nin deposal kapisi `kodlayici-kapisi`dir (bu belgenin
   ilk surumunde eksik olarak isaretlenmisti). Kapi pencere kararinin tek
   fonksiyonda kaldigini, sinir/belirlenimcilik/red testlerinin adiyla
   durdugunu, test disi govdede ucuncu taraf adi ve `unwrap` olmadigini
   denetler; crate'in lib testlerini kosturur ve kaynaktaki `#[test]`
   sayisiyla birebir esler. Kaydi `kodlayici-2026-09-27.json`: 8 konumluk
   dizide 64 (sorgu, degisen) ciftinin tamami olculur, pencere disi etki 0,
   pencere ici etki 15/15.
3. `hadamard_mlp` ve `cq2_quant` icin ayri olcum kayitlari
   `hadamard-mlp-2026-09-27.json` ve `nicem-2026-09-27.json`; ikisi de
   `omurga-karar-kayitlari` kapisinda her turda yeniden olculur.
4. Bagimsiz kalan dort aday (`hadamard_mlp`, `hyperconnections`,
   `sinkhorn_router`, ve engram kolu) artik tek bir blokta birlikte kosuyor
   (3.8), ama **o blok da hicbir aileye bagli degil**. Hangisinin hangi aileye
   girecegi `docs/MIMARI-TASARIM.md` 5. bolumunde M1/M2/M3 olarak isaretli
   operator kararidir ve bu envanter onu vermez. `kademe` adayi bu blogun
   disindadir: o bir katman degil egitim hedefidir ve baglanmasi M4 kararidir.

## Bu envanterin soylemedigi

- Hicbir modulun **kalitesi** hakkinda bir sey demez. "Var" demek "iyi" demek
  degildir; kalite iddiasi egitilmis bir kontrol noktasi ve sinav seti ister.
- Satir sayisi ya da ilerleme yuzdesi tasimaz. Bir modulun buyuklugu
  onun tamamlanmisligi degildir.
