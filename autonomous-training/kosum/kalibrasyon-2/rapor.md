# Egitim Turu: olculen adimlar

Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil
gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan
geliyor; hicbiri burada yeniden hesaplanmiyor.

| alan | deger |
| --- | --- |
| kayit | 2779 |
| pencere | 128 jeton; egitim 1410, dogrulama 70 (havuz 70) |
| adim | 0 -> 100 (bu cagri 100 adim) |
| epoch | 0 -> 0 (tavan 8) |
| yigin | 2 pencere/adim; kirpma 1.000; lr 0.01000; sonum 0.100 |
| jeton | 25400 |
| kayip | 9.034512 -> 7.441784 |
| dusus | 17.63% |
| en iyi dogrulama | 7.476994 (adim 100) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| sure | 145.2 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `05803214a179949ae943472e2d77b8a3f0101c6475362873d2b7802d4cadfe2b` |
| kontrol noktasi | `/home/user/lubot-git/autonomous-training/kosum/kalibrasyon-2/ckpt.bin` sha256 `30ade38b4ed38f33325015ea1ef62f922f827a587ce04f5f4da79e56f8f6bf8f` (LUBOTCKPT v1, 19 blok) |
| held-out | `/home/user/lubot-git/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 200 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.656957 | 70 | 8890 |
| 40 | 1 | 8.450811 | 70 | 8890 |
| 60 | 1 | 8.016636 | 70 | 8890 |
| 80 | 1 | 7.613171 | 70 | 8890 |
| 100 | 1 | 7.476994 | 70 | 8890 |
