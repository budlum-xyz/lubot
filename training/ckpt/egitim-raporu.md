# Egitim Turu: olculen adimlar

Bu rapor bir turun ne yaptigini soyler: kac adim atti, kaybi nasil
gitti, nerede durdu ve neden durdu. Sayilar kosunun kendi kayitlarindan
geliyor; hicbiri burada yeniden hesaplanmiyor.

| alan | deger |
| --- | --- |
| kayit | 81138 |
| pencere | 256 jeton; egitim 52594, dogrulama 32 (havuz 2758) |
| adim | 0 -> 350 (bu cagri 350 adim) |
| epoch | 0 -> 0 (tavan 1) |
| yigin | 8 pencere/adim; kirpma 1.000; lr 0.01000; sonum 0.100 |
| jeton | 714000 |
| kayip | 9.038323 -> 3.652348 |
| dusus | 59.59% |
| en iyi dogrulama | 3.847723 (adim 300) |
| durma | adim-butcesi (butce doldu) |
| kirpilan adim | 0 |
| sure | 1962.7 sn (makineye bagli; ratchet'e girmez) |
| korpus ozeti | `ff2d02eeb24907b48c927324da59ca8f6b52fca101dce5104465ea81847dfaf4` |
| kontrol noktasi | `training/ckpt/lubot-a1.ckpt` sha256 `59009ba1bb310bd9f4b798f59e82ccfa17d7cdde143de486403932f39ea2eab1` (LUBOTCKPT v1, 19 blok) |
| held-out | `training/eval/sinav-seti.jsonl`; 12 damga, 12 kayit egitim akisindan cikarildi |
| devam konumu | 2800 (bitmemis epoch icin) |

## Epoch egrisi

| epoch | kayip |
| --- | --- |

## Dogrulama egrisi

| adim | epoch | kayip | pencere | jeton |
| --- | --- | --- | --- | --- |
| 100 | 1 | 6.690512 | 32 | 8160 |
| 200 | 1 | 4.477911 | 32 | 8160 |
| 300 | 1 | 3.847723 | 32 | 8160 |
