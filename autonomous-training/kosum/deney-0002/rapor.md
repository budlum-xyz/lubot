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
| yigin | 2 pencere/adim; kirpma 1.000; lr 0.00387; sonum 0.100 |
| jeton | 67310 |
| kayip | 9.034512 -> 7.537633 |
| dusus | 16.57% |
| en iyi dogrulama | 7.357786 (adim 260) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| sure | 393.4 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `05803214a179949ae943472e2d77b8a3f0101c6475362873d2b7802d4cadfe2b` |
| kontrol noktasi | `/home/user/lubot-git/autonomous-training/kosum/deney-0002/ckpt.bin` sha256 `fb879dc658f076851baae7031a65ae01f7d73e61881449e0d56df566b7a3c1a1` (LUBOTCKPT v1, 19 blok) |
| held-out | `/home/user/lubot-git/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 530 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.695779 | 70 | 8890 |
| 40 | 1 | 8.616523 | 70 | 8890 |
| 60 | 1 | 8.478238 | 70 | 8890 |
| 80 | 1 | 8.315188 | 70 | 8890 |
| 100 | 1 | 8.146184 | 70 | 8890 |
| 120 | 1 | 7.978663 | 70 | 8890 |
| 140 | 1 | 7.830927 | 70 | 8890 |
| 160 | 1 | 7.687880 | 70 | 8890 |
| 180 | 1 | 7.573307 | 70 | 8890 |
| 200 | 1 | 7.488068 | 70 | 8890 |
| 220 | 1 | 7.427436 | 70 | 8890 |
| 240 | 1 | 7.386470 | 70 | 8890 |
| 260 | 1 | 7.357786 | 70 | 8890 |
