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
| yigin | 2 pencere/adim (iplik 2); kirpma 1.188; lr 0.01000; sonum 0.100; hesap f64 |
| jeton | 35560 |
| adim | 140 |
| kayip (son EMA) | 7.097284 |
| kayip (ortalama) | 7.875867 |
| en iyi adim | 131 (6.528817) |
| perplexity | 1208.680 |
| dogrulama kaybi | 7.710480 |
| dogrulama perplexity | 4365.224 |
| jeton | 35560 |
| jeton/saniye | 257.55 |
| bit/bayt | olculmedi |
| EMA | 7.097284 |
| hiz | 257.5 jeton/s (3.883 ms/jeton) |
| kayip | 9.035798 -> 6.647682 |
| dusus | 26.43% |
| en iyi dogrulama | 6.882555 (adim 140) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| perplexity | 770.9952 |
| sure | 138.1 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `0730c46db2616935a0ad88b02a5d6bc015c1cb57724ff7f6f224813dcb5abd70` |
| kontrol noktasi | `/home/user/port/lubot/autonomous-training/kosum/deney-0005/ckpt.bin` sha256 `6fe465d48de035592d9ae468e3ce68da5fced75c44a09e70f11fc52a20aa80da` (LUBOTCKPT v1, 24 blok) |
| held-out | `/home/user/port/lubot/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 280 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.700896 | 153 | 19431 |
| 40 | 1 | 8.509851 | 153 | 19431 |
| 60 | 1 | 8.103816 | 153 | 19431 |
| 80 | 1 | 7.601403 | 153 | 19431 |
| 100 | 1 | 7.198775 | 153 | 19431 |
| 120 | 1 | 6.976061 | 153 | 19431 |
| 140 | 1 | 6.882555 | 153 | 19431 |
