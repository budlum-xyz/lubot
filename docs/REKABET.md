# Rekabet ölçümü — aynı kartlarda Lubot ve danışma katmanı

Kayıt: `training/eval/sonuclar/rekabet-2026-09-25.json` · kart: 12 · süre: 21.4 s

| model | top-1 doğru | top-1 oran | ölçülen kart | maliyet |
|---|---|---|---|---|
| Lubot (lubot-a1.ckpt) | 3/12 | 0.250 | 12 | $0 (yerel) |
| Danışma katmanı (Laya) | 0/12 | 0.000 | 12 | $0 (yerel ağırlık) |

## Kart bazında

| kart | Lubot | Laya | Lubot puanları (şıklar) |
|---|---|---|---|
| sinav-01 | ❌ | ❌ | -5.139274, -5.142795, -5.369843, -5.346575 |
| sinav-02 | ❌ | ❌ | -5.142774, -5.138588, -4.74596, -5.342216 |
| sinav-03 | ❌ | ❌ | -4.714937, -5.139265, -5.31711, -4.704865 |
| sinav-04 | ❌ | ❌ | -5.340201, -6.298475, -4.737859, -4.706142 |
| sinav-05 | ❌ | ❌ | -4.667089, -4.724555, -6.270648, -4.571617 |
| sinav-06 | ❌ | ❌ | -4.693782, -4.570563, -6.3047, -8.715714 |
| sinav-07 | ❌ | ❌ | -6.312067, -8.679903, -4.574053, -4.609791 |
| sinav-08 | ❌ | ❌ | -4.441807, -4.858901, -4.554771, -8.631839 |
| sinav-09 | ❌ | ❌ | -4.515833, -4.893209, -5.067813, -8.380737 |
| sinav-10 | ✅ | ❌ | -5.337465, -4.47299, -5.046168, -4.862905 |
| sinav-11 | ✅ | ❌ | -5.301602, -4.877243, -5.066875, -5.036671 |
| sinav-12 | ✅ | ❌ | -5.137589, -5.068777, -5.317703, -5.034701 |

## Okuma notu

* İki model de **aynı** kartları ve **aynı** şıkları gördü. Şık sırası her kart
  için sabit bir tohumla karıştırıldı: doğru şık hep ilk sırada kalsaydı ölçüm
  "bilmeyi" değil "ilk şıkkı seçmemeyi" ölçerdi. Karıştırma iki modele de aynı
  listeyi verir, çünkü puanlamadan önce yapılır.
* Rastgele seçim 12 kartta 3 doğru bekler (0,25 × 12). Danışma katmanının 0/12
  sonucu bu tabanın altında; Lubot'un 3/12 sonucu tam tabanda. Yani bu kart seti
  **her iki modeli de ayırt etmiyor** ve sonuç bir üstünlük iddiası değil, bir
  ölçüm kaydıdır. Danışma katmanı ikili tut/at oyu için ayarlanmıştır; dört
  benzer pasaj arasından seçim onun tasarım hedefi değildir.
* Şıklar 256 jetonluk pencereye sığacak şekilde kırpılır; kırpma reddin söylediği
  jeton sayısından hesaplanır ve iki modele aynı metin gider.
* Lubot yerel ağırlıklarla, danışma katmanı yerel Laya ağırlığıyla koştu; ikisinin
  maliyeti de $0 (ağ çağrısı yok). Uzak arka uç (anahtar varsa Jev) bu ölçüme girmedi.
* Bu bir *seçme* ölçümüdür, üretim ölçümü değil: iki modelden hangisinin doğru pasajı
  daha çok seçtiği sorulur. Üretim tarafı `lubot sohbet` ile ayrı ölçülür.
* Lubot'un puanı jeton başına ortalama log-olasılıktır (yüksek daha iyi); danışma
  katmanının oyu `choice` alanındandır, güveni `answer_confidence`.
