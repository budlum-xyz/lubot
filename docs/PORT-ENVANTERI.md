# Port envanteri: sekiz modul nerede duruyor

Bu belge tek bir soruya cevap verir: **hangi port modulu bu agacta var, kanıti
nerede, ve hangisi hala yok.** Cevap iddia degil, dosya yoludur — her satirdaki
her yol `port-envanteri-kapisi` tarafindan denetlenir: kaynak dosya yoksa,
kapi adi kayitli degilse ya da olcum kaydi eksikse kapi kirmizi yanar.

Belgenin varlik sebebi sudur: modul adlari calisma direktifinde Ingilizce ve
kavramsal (`hadamard_mlp`, `sinkhorn_router`), bu agacta ise Turkce ve somut
(`mlp_hadamard.rs`, `yonlendirme.rs`). Iki isim kumesi arasinda yazili bir
karsilik olmadigi surece "port bitti mi" sorusu **olculemez** bir sorudur;
herkes kendi listesine bakar ve farkli cevap verir.

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

## 7.4'un sekiz modulu

| 7.4 modulu | bu depodaki yeri | kapi | olcum kaydi | durum |
|---|---|---|---|---|
| `modernbert_encoder` | `crates/kodlayici/src/lib.rs`, `crates/transformer/src/lib.rs` | — | — | bagli |
| `hadamard_mlp` | `crates/egitim/src/mlp_hadamard.rs` | `hadamard-mlp-kapisi` | — | bagimsiz |
| `gqa_engram_attention` | `crates/egitim/src/lib.rs`, `crates/egitim/src/engram.rs` | `engram-kapisi` | `engram-2026-09-26.json` | bagli |
| `hyperconnections` | `crates/egitim/src/cok_serit.rs` | `cok-serit-kapisi` | `cok-serit-2026-09-26.json` | bagimsiz |
| `sinkhorn_router` | `crates/egitim/src/yonlendirme.rs` | `yonlendirme-kapisi` | `sinkhorn-yonlendirme-2026-09-27.json` | bagimsiz |
| `decision_head` | `crates/tomurcuk/src/lib.rs`, `crates/tomurcuk/src/kalibrasyon.rs` | `kalibrasyon-bandi-kapisi` | `kalibrasyon-2026-09-26.json` | bagli |
| `schema_decoder` | `crates/read/src/output_schema.rs` | `ai-output-schema-enforced` | `sema-kapsam-2026-09-24.json` | bagli |
| `cq2_quant` | `crates/nicem/src/lib.rs`, `crates/tasiyici/src/lib.rs` | `bit-budget-is-arithmetic` | — | bagli |

`gqa_engram_attention` satiri iki parcalidir ve durumu **parca parca** okunur:
GQA egitim cekirdeginde baglidir (`n_kv_heads`, `qkv_dokunus`, `qk_norm` ayni
ileri/geri gecisten geciyor), engram tablosu ise bagimsizdi — 3.8 blogu onu bir
kompozisyon icine aldi ama o blok da hicbir aileye bagli degil. Satirin durumu
bu yuzden GQA'ya gore `bagli` yazilir; engramin kendi satiri asagidadir.

## 7.4'te olmayan, bu agacta olan

| modul | bu depodaki yeri | kapi | olcum kaydi | durum |
|---|---|---|---|---|
| aile kesitleri | `crates/egitim/src/kesit.rs` | `kesit-kapisi` | `kesit-2026-09-27.json` | bagimsiz |
| sifir merkezli RMS norm | `crates/egitim/src/normalizasyon.rs` | `normalizasyon-kapisi` | `normalizasyon-2026-09-27.json` | bagimsiz |
| birlesik blok (3.8) | `crates/egitim/src/birlesik.rs` | `birlesik-kapisi` | `birlesik-2026-09-27.json` | bagimsiz |

## Bu envanterin soyledigi

1. Sekiz modulun **sekizi de** bu agacta var. "Port yazilacak" bir is degil;
   kalan is **baglama** isidir.
2. `modernbert_encoder`in deposal bir kapisi **yok**. Kaynak ve testler var,
   ama kapi yok: yani bir regresyon kapiya takilmaz. Bu, bu belgenin urettigi
   somut bir eksiktir ve baska bir turun isidir.
3. `hadamard_mlp` ve `cq2_quant` icin ayri bir olcum kaydi yok; olcum modul
   testlerinin icinde. Kayit dosyasi, olcumun tarihini ve tazeligini disaridan
   denetlenebilir yapar — bunlarin ikisi de simdilik ic olcumdur.
4. Bagimsiz kalan dort aday (`hadamard_mlp`, `hyperconnections`,
   `sinkhorn_router`, ve engram kolu) artik tek bir blokta birlikte kosuyor
   (3.8), ama **o blok da hicbir aileye bagli degil**. Hangisinin hangi aileye
   girecegi `docs/MIMARI-TASARIM.md` 5. bolumunde M1/M2/M3 olarak isaretli
   operator kararidir ve bu envanter onu vermez.

## Bu envanterin soylemedigi

- Hicbir modulun **kalitesi** hakkinda bir sey demez. "Var" demek "iyi" demek
  degildir; kalite iddiasi egitilmis bir kontrol noktasi ve sinav seti ister.
- Satir sayisi ya da ilerleme yuzdesi tasimaz. Bir port modulunun buyuklugu
  onun tamamlanmisligi degildir.
