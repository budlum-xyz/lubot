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
| yigin | 2 pencere/adim (iplik 2); kirpma 1.000; lr 0.01306; sonum 0.100; hesap f64 |
| jeton | 35560 |
| adim | 140 |
| kayip (son EMA) | 6.404037 |
| kayip (ortalama) | 7.496540 |
| en iyi adim | 131 (5.620040) |
| perplexity | 604.279 |
| dogrulama kaybi | 7.267611 |
| dogrulama perplexity | 3744.331 |
| jeton | 35560 |
| jeton/saniye | 220.13 |
| bit/bayt | olculmedi |
| EMA | 6.404037 |
| hiz | 220.1 jeton/s (4.543 ms/jeton) |
| kayip | 9.035798 -> 5.821471 |
| dusus | 35.57% |
| en iyi dogrulama | 6.123750 (adim 140) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| perplexity | 337.4680 |
| sure | 161.5 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `0730c46db2616935a0ad88b02a5d6bc015c1cb57724ff7f6f224813dcb5abd70` |
| kontrol noktasi | `/home/user/port/lubot/autonomous-training/kosum/deney-0007/ckpt.bin` sha256 `ff1115e343d5c813fc47af80aeeaa43e0d59ab81557f79421619668ba76f1365` (LUBOTCKPT v1, 24 blok) |
| held-out | `/home/user/port/lubot/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 280 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.684840 | 153 | 19431 |
| 40 | 1 | 8.419585 | 153 | 19431 |
| 60 | 1 | 7.820289 | 153 | 19431 |
| 80 | 1 | 7.077946 | 153 | 19431 |
| 100 | 1 | 6.515029 | 153 | 19431 |
| 120 | 1 | 6.231840 | 153 | 19431 |
| 140 | 1 | 6.123750 | 153 | 19431 |
