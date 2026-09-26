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
| yigin | 2 pencere/adim (iplik 2); kirpma 1.000; lr 0.00694; sonum 0.100; hesap f64 |
| jeton | 35560 |
| adim | 140 |
| kayip (son EMA) | 7.762996 |
| kayip (ortalama) | 8.224649 |
| en iyi adim | 131 (7.384396) |
| perplexity | 2351.940 |
| dogrulama kaybi | 8.120096 |
| dogrulama perplexity | 5035.440 |
| jeton | 35560 |
| jeton/saniye | 234.82 |
| bit/bayt | olculmedi |
| EMA | 7.762996 |
| hiz | 234.8 jeton/s (4.259 ms/jeton) |
| kayip | 9.035798 -> 7.450104 |
| dusus | 17.55% |
| en iyi dogrulama | 7.631849 (adim 140) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| perplexity | 1720.0424 |
| sure | 151.4 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `0730c46db2616935a0ad88b02a5d6bc015c1cb57724ff7f6f224813dcb5abd70` |
| kontrol noktasi | `/home/user/port/lubot/autonomous-training/kosum/deney-0006/ckpt.bin` sha256 `a6f4789fbd21faa89b4234c9413d5331b818253c7c863eb164620587acb2ed4d` (LUBOTCKPT v1, 24 blok) |
| held-out | `/home/user/port/lubot/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 280 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.716346 | 153 | 19431 |
| 40 | 1 | 8.589348 | 153 | 19431 |
| 60 | 1 | 8.339037 | 153 | 19431 |
| 80 | 1 | 8.051052 | 153 | 19431 |
| 100 | 1 | 7.824534 | 153 | 19431 |
| 120 | 1 | 7.688509 | 153 | 19431 |
| 140 | 1 | 7.631849 | 153 | 19431 |
