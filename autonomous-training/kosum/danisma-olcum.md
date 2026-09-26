# K7 danışma katmanı — ölçüm ve karar (2026-09-24)

Anayasa maddesi: **K7** (`INVARIANTS.md`, damga `c2f8ce3f5e0e` yenilendi).
Karar kuralı: tut/at kararı kabul eşiğinin bandına düşerse (`|marj| ≤ eşik × bant_carpanı`)
danışma katmanına **oy** sorulur. Oy bir tavsiyedir; eşik, marj ve tut/at mantığı
`dongu.py` ile `danisma.py`'de yazılıdır. Oy gelmez/geçersiz olursa ya da bilinen
cevaplı **kalibrasyon probunu** geçemezse karar insana gider ve döngü **S2** ile durur.

## Kurulum (depo içi)

| parça | yol | ne yapar |
| --- | --- | --- |
| karar modülü | `autonomous-training/danisma.py` | bant, marj, prob, oy→insan kuralı; **hiç yazmaz** |
| döngü bağı | `autonomous-training/dongu.py` → `danisma_oyu()` | oyu sorar, kararı koda bırakır, çelişkiyi kayda geçer |
| arka uç servisi | `training/danisma/sunucu.py` | modeli **bellekte tutar**; `GET /saglik`, `POST /oy` |
| kartlar | `training/danisma/kararlar.json` | sorulacak soru; karar değil |
| kapı | `gates/check.py → danisma-layer-is-closed` | sınırı mekanik denetler (83. kapı) |

Servis komutu: `python3 training/danisma/sunucu.py --port 8790 --arka-uc otomatik`
(`TYPESAFE_API_KEY` varsa uzak Jev, yoksa yerel Laya).

## Ölçüm (bu makine: 2 vCPU, 1,94 GiB, GPU yok)

| ölçüm | değer |
| --- | --- |
| model yükleme (soğuk, önbellekten) | **10,2 s** (ilk kez 16,4 s) |
| oy gecikmesi (servis sıcak) | **351–457 ms** (sunucu içi 350–445 ms) |
| tepe RSS | ~1,4 GB (bf16; fp32 bu makinede OOM) |
| ilk çağrı (soğuk process) | 2,6 s |
| maliyet | $0 |

**Arka planda çalışıyor:** servis ayrı süreç olarak açılıyor, model bir kez
yükleniyor, sonraki oylar 0,35–0,46 s. Karar başına model yükleyen yol 10–16 s
olurdu; servis bunu ortadan kaldırdı.

## Oylar — sabit oy veren bir model

| girdi | oy | güven | gecikme |
| --- | --- | --- | --- |
| skor 6,671741 (sınırda) | tut | 0,678 | 445 ms |
| skor 6,500000 (açıkça iyi) | tut | 0,620 | 352 ms |
| skor 9,000000 (açıkça kötü) | tut | 0,613 | 351 ms |

Üç durumda da "tut": model sabit oy veriyor. Güven 0,61–0,68 — yani 0,55
eşiğini **geçiyor**. Sabit bir oy, eşiği geçse bile bilgi taşımaz; onu karara
soksaydık karar gizlice "hep tut"a dönerdi.

## Kalibrasyon probu — devreye giren güvenlik

Bilinen cevaplı iki prob sorulur (6,50 → tut; 9,00 → at). Yerel ölçüm: **1/2**
(prob-2'de "tut" dedi). Prob düşünce oy **geçersiz** sayılır:

```
$ python3 autonomous-training/danisma.py --kart tut_at --skor 6.6717405 --mevcut 6.671741 --esik 0.001
insan_gerekli: True | oy: None | kalibrasyon: 1/2
gerekce: "kalibrasyon probu dustu (1/2): oy bilgi tasimiyor"   (çıkış kodu 1)
```

Yani K7 bağlı ve çalışıyor; bu makinedeki yerel model ise **kendi oyunu geçersiz
ilan ediyor** ve karar insana kalıyor. Jev anahtarı gelince aynı kartlar, aynı
problar uzak arka uçla koşar; hüküm ölçümle yenilenir.

## Yol boyunca yakalanan üç hata (hepsi kanaryaya bağlandı)

1. **`choice` etiketi metin değil anahtar** (`o0`): oy "None" düşüyordu. Eşleme
   artık bizim gönderdiğimiz anahtarla **birebir**; sayıdan taban tahmini yok
   (`criteria_metne_cevir` + kanarya).
2. **`not prob["gecen"]` — sayaç boolean sanıldı.** Prob düştüğü hâlde karar
   oyla verildi; kanarya yalnız prob fonksiyonunu sınadığı için görmedi. Artık
   karar `prob["gecti"]`ye bağlı ve kanarya **karar yolunu** da sınıyor
   ("probu düşen arka uç -> insan").
3. **Kapı yorumla ateş etti:** ilk hatayı anlatan yorum, "sayaca bağlanmış"
   denetimini kırdı. Kapı artık yalnız kodu okur (yorum sıyrılır) — yorumla
   ateş eden kapı, yorumu silmeye zorlar.
