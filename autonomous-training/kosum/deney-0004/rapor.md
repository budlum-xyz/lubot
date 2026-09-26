# Egitim Turu: olculen adimlar

Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil
gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan
geliyor; hicbiri burada yeniden hesaplanmiyor.

| alan | deger |
| --- | --- |
| kayit | 5557 |
| pencere | 128 jeton; egitim 2832, dogrulama 153 (havuz 153) |
| adim | 0 -> 140 (bu cagri 140 adim) |
| epoch | 0 -> 0 (tavan 8) |
| yigin | 2 pencere/adim (iplik 2); kirpma 1.000; lr 0.01000; sonum 0.163; hesap f64 |
| jeton | 35560 |
| adim | 140 |
| kayip (son EMA) | 7.098974 |
| kayip (ortalama) | 7.875967 |
| en iyi adim | 131 (6.541881) |
| perplexity | 1210.724 |
| dogrulama kaybi | 7.711115 |
| dogrulama perplexity | 4366.191 |
| jeton | 35560 |
| jeton/saniye | 325.89 |
| bit/bayt | olculmedi |
| EMA | 7.098974 |
| hiz | 325.9 jeton/s (3.068 ms/jeton) |
| kayip | 9.035798 -> 6.647107 |
| dusus | 26.44% |
| en iyi dogrulama | 6.885483 (adim 140) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| perplexity | 770.5520 |
| sure | 109.1 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `0730c46db2616935a0ad88b02a5d6bc015c1cb57724ff7f6f224813dcb5abd70` |
| kontrol noktasi | `/home/user/port/lubot/autonomous-training/kosum/deney-0004/ckpt.bin` sha256 `8560c4d96fd321ee045a5dba5ce2138b2ffb81b0315a22501d8c5c5cd9f76a69` (LUBOTCKPT v1, 24 blok) |
| held-out | `/home/user/port/lubot/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 280 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.700884 | 153 | 19431 |
| 40 | 1 | 8.509424 | 153 | 19431 |
| 60 | 1 | 8.102979 | 153 | 19431 |
| 80 | 1 | 7.600682 | 153 | 19431 |
| 100 | 1 | 7.199543 | 153 | 19431 |
| 120 | 1 | 6.978811 | 153 | 19431 |
| 140 | 1 | 6.885483 | 153 | 19431 |
