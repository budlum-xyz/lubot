# Egitim Turu: olculen adimlar

Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil
gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan
geliyor; hicbiri burada yeniden hesaplanmiyor.

| alan | deger |
| --- | --- |
| kayit | 5629 |
| pencere | 128 jeton; egitim 2874, dogrulama 154 (havuz 154) |
| adim | 0 -> 140 (bu cagri 140 adim) |
| epoch | 0 -> 0 (tavan 8) |
| yigin | 2 pencere/adim (iplik 2); kirpma 1.000; lr 0.01306; sonum 0.100; hesap f64 |
| jeton | 35560 |
| adim | 140 |
| kayip (son EMA) | 6.360304 |
| kayip (ortalama) | 7.487058 |
| en iyi adim | 140 (5.634027) |
| perplexity | 578.422 |
| dogrulama kaybi | 7.270319 |
| dogrulama perplexity | 3753.930 |
| jeton | 35560 |
| jeton/saniye | 366.41 |
| bit/bayt | olculmedi |
| EMA | 6.360304 |
| hiz | 366.4 jeton/s (2.729 ms/jeton) |
| kayip | 9.047749 -> 5.634027 |
| dusus | 37.73% |
| en iyi dogrulama | 6.117361 (adim 140) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| perplexity | 279.7867 |
| sure | 97.1 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `e0eb1f85808d71ab31ed7e94094a31debbe88cb0cb0d51715d0344d6ff003ae0` |
| kontrol noktasi | `/home/user/port/lubot/autonomous-training/kosum/deney-0007/ckpt.bin` sha256 `ad03c2b7c43b59f740631e2e3f17cffcf15d44129ef33676354bc757c69ecb97` (LUBOTCKPT v1, 24 blok) |
| held-out | `/home/user/port/lubot/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 280 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.687512 | 154 | 19558 |
| 40 | 1 | 8.428893 | 154 | 19558 |
| 60 | 1 | 7.825655 | 154 | 19558 |
| 80 | 1 | 7.085361 | 154 | 19558 |
| 100 | 1 | 6.518544 | 154 | 19558 |
| 120 | 1 | 6.228906 | 154 | 19558 |
| 140 | 1 | 6.117361 | 154 | 19558 |
