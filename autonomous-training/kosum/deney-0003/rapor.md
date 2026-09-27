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
| yigin | 2 pencere/adim (iplik 2); kirpma 1.000; lr 0.01000; sonum 0.038; hesap f64 |
| jeton | 35560 |
| adim | 140 |
| kayip (son EMA) | 7.099191 |
| kayip (ortalama) | 7.876091 |
| en iyi adim | 131 (6.541616) |
| perplexity | 1210.986 |
| dogrulama kaybi | 7.711121 |
| dogrulama perplexity | 4366.261 |
| jeton | 35560 |
| jeton/saniye | 330.13 |
| bit/bayt | olculmedi |
| EMA | 7.099191 |
| hiz | 330.1 jeton/s (3.029 ms/jeton) |
| kayip | 9.035798 -> 6.644045 |
| dusus | 26.47% |
| en iyi dogrulama | 6.885977 (adim 140) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| perplexity | 768.1962 |
| sure | 107.7 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `0730c46db2616935a0ad88b02a5d6bc015c1cb57724ff7f6f224813dcb5abd70` |
| kontrol noktasi | `/home/user/port/lubot/autonomous-training/kosum/deney-0003/ckpt.bin` sha256 `20199c168813d9d740b70caf02b40ba4d39c78a1be84bc0856707101b614e1c9` (LUBOTCKPT v1, 24 blok) |
| held-out | `/home/user/port/lubot/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 280 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.700908 | 153 | 19431 |
| 40 | 1 | 8.510271 | 153 | 19431 |
| 60 | 1 | 8.102914 | 153 | 19431 |
| 80 | 1 | 7.600316 | 153 | 19431 |
| 100 | 1 | 7.198785 | 153 | 19431 |
| 120 | 1 | 6.978678 | 153 | 19431 |
| 140 | 1 | 6.885977 | 153 | 19431 |
