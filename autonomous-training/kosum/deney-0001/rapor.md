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
| yigin | 2 pencere/adim; kirpma 1.000; lr 0.01613; sonum 0.100 |
| jeton | 67310 |
| kayip | 9.034512 -> 5.356378 |
| dusus | 40.71% |
| en iyi dogrulama | 4.653110 (adim 260) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| sure | 397.5 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `05803214a179949ae943472e2d77b8a3f0101c6475362873d2b7802d4cadfe2b` |
| kontrol noktasi | `/home/user/lubot-git/autonomous-training/kosum/deney-0001/ckpt.bin` sha256 `a19bbb8779ace8f30f30b68af6ba367daa9cc6a1ac6b486d459f687e6e3a3831` (LUBOTCKPT v1, 19 blok) |
| held-out | `/home/user/lubot-git/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 530 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.622322 | 70 | 8890 |
| 40 | 1 | 8.253449 | 70 | 8890 |
| 60 | 1 | 7.355028 | 70 | 8890 |
| 80 | 1 | 6.155600 | 70 | 8890 |
| 100 | 1 | 5.316417 | 70 | 8890 |
| 120 | 1 | 5.125650 | 70 | 8890 |
| 140 | 1 | 4.994272 | 70 | 8890 |
| 160 | 1 | 4.895681 | 70 | 8890 |
| 180 | 1 | 4.808079 | 70 | 8890 |
| 200 | 1 | 4.769331 | 70 | 8890 |
| 220 | 1 | 4.714046 | 70 | 8890 |
| 240 | 1 | 4.676265 | 70 | 8890 |
| 260 | 1 | 4.653110 | 70 | 8890 |
