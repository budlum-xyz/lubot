# Egitim Turu: olculen adimlar

Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil
gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan
geliyor; hicbiri burada yeniden hesaplanmiyor.

| alan | deger |
| --- | --- |
| kayit | 2779 |
| pencere | 128 jeton; egitim 1410, dogrulama 70 (havuz 70) |
| adim | 0 -> 265 (bu cagri 265 adim) |
| epoch | 0 -> 0 (tavan 8) |
| yigin | 2 pencere/adim; kirpma 1.000; lr 0.01000; sonum 0.100 |
| jeton | 67310 |
| kayip | 9.034512 -> 5.538451 |
| dusus | 38.70% |
| en iyi dogrulama | 4.980211 (adim 260) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| sure | 398.9 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `05803214a179949ae943472e2d77b8a3f0101c6475362873d2b7802d4cadfe2b` |
| kontrol noktasi | `/home/user/lubot-git/autonomous-training/kosum/tekrar-1/ckpt.bin` sha256 `471b42087ce725ccb7e2bf911f1fb344539932d405174bffc712eb8e331a141f` (LUBOTCKPT v1, 19 blok) |
| held-out | `/home/user/lubot-git/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 530 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.656957 | 70 | 8890 |
| 40 | 1 | 8.450811 | 70 | 8890 |
| 60 | 1 | 8.008350 | 70 | 8890 |
| 80 | 1 | 7.395628 | 70 | 8890 |
| 100 | 1 | 6.726230 | 70 | 8890 |
| 120 | 1 | 6.100413 | 70 | 8890 |
| 140 | 1 | 5.618636 | 70 | 8890 |
| 160 | 1 | 5.318430 | 70 | 8890 |
| 180 | 1 | 5.159120 | 70 | 8890 |
| 200 | 1 | 5.088512 | 70 | 8890 |
| 220 | 1 | 5.034689 | 70 | 8890 |
| 240 | 1 | 5.000839 | 70 | 8890 |
| 260 | 1 | 4.980211 | 70 | 8890 |
