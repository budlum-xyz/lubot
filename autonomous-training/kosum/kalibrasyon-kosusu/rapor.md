# Egitim Turu: olculen adimlar

Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil
gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan
geliyor; hicbiri burada yeniden hesaplanmiyor.

| alan | deger |
| --- | --- |
| kayit | 2779 |
| pencere | 128 jeton; egitim 1410, dogrulama 70 (havuz 70) |
| adim | 0 -> 20 (bu cagri 20 adim) |
| epoch | 0 -> 0 (tavan 8) |
| yigin | 2 pencere/adim; kirpma 1.000; lr 0.01000; sonum 0.100 |
| jeton | 5080 |
| kayip | 9.034512 -> 8.742536 |
| dusus | 3.23% |
| en iyi dogrulama | 8.656957 (adim 20) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| sure | 27.3 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `05803214a179949ae943472e2d77b8a3f0101c6475362873d2b7802d4cadfe2b` |
| kontrol noktasi | `/home/user/lubot-git/autonomous-training/kosum/kalibrasyon-kosusu/ckpt.bin` sha256 `3de83fc35cf074b0382a6b0176dd3d8409b081d18fead6f3ec5fca6c66044381` (LUBOTCKPT v1, 19 blok) |
| held-out | `/home/user/lubot-git/training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 40 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 20 | 1 | 8.656957 | 70 | 8890 |
