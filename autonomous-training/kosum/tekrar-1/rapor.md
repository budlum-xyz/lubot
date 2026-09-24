# Egitim Turu: olculen adimlar

Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil
gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan
geliyor; hicbiri burada yeniden hesaplanmiyor.

| alan | deger |
| --- | --- |
| kayit | 2779 |
| pencere | 128 jeton; egitim 1410, dogrulama 70 (havuz 70) |
| adim | 0 -> 140 (bu cagri 140 adim) |
| epoch | 0 -> 0 (tavan 8) |
| yigin | 2 pencere/adim; kirpma 1.000; lr 0.01000; sonum 0.100 |
| jeton | 35560 |
| kayip | 9.034512 -> 6.921743 |
| dusus | 23.39% |
| en iyi dogrulama | 6.671741 (adim 140) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| sure | 197.4 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `05803214a179949ae943472e2d77b8a3f0101c6475362873d2b7802d4cadfe2b` |
| kontrol noktasi | `/home/user/lubot-git/autonomous-training/kosum/tekrar-1/ckpt.bin` sha256 `5e30569161dfa4a38af159749c1b6d70c41b8408e0afc572c88553ec2babb7c1` (LUBOTCKPT v1, 19 blok) |
| held-out | `/home/user/lubot-git/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 280 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.656957 | 70 | 8890 |
| 40 | 1 | 8.450811 | 70 | 8890 |
| 60 | 1 | 8.010617 | 70 | 8890 |
| 80 | 1 | 7.462990 | 70 | 8890 |
| 100 | 1 | 7.023536 | 70 | 8890 |
| 120 | 1 | 6.773814 | 70 | 8890 |
| 140 | 1 | 6.671741 | 70 | 8890 |
