# Sohbet ornekleri (olculen cikti)

Bu belge, `lubot sohbet` komutunun **gercek** ciktisidir: her blok bir kosunun
stdout'undan alindi, elle duzeltilmedi. Model `training/ckpt/lubot-a1.ckpt`
kontrol noktasidir (924.288 parametre, 350 adim, ~917 bin jeton, dogrulama kaybi
9,03 -> 3,85). Kucuk bir dil modeli *devam ettirir*; soruyu yanitlamaz. Cikti bu
yuzden akici cumleler degil, korpusun doku ve parcalaridir.

| ayar | deger |
|---|---|
| sicaklik | 0,8 |
| top-k | 40 |
| top-p | 0,95 |
| uretilen jeton | 64 |
| cikti bicimi | sema dogrulamali Markdown |

## Girdi: `Korpus nedir?` (tohum 7)

```
[] <u :   a an zone /
[20  the /ildi a a  a and  text  gT:  the you g:  the  in and the A
```

Ortalama log-olasilik: **-1.9337** (jeton basina; sifira yakin daha iyi).

## Girdi: `Korpus nedir?` (tohum 42)

```
[ a ar I kaydını g :   g A3::: kaydını 
 Bul[
 Yele  . ile  m, kezinde merYağm   ş S,   Er
```

Ortalama log-olasilik: **-2.1467** (jeton basina; sifira yakin daha iyi).

## Girdi: `Lubot nasil ogrenir?` (tohum 7)

```
[al: I and  to that zır:10 is .  the ks to to  to m  on  gi   the ında g:
[ m in m a ında
```

Ortalama log-olasilik: **-1.9933** (jeton basina; sifira yakin daha iyi).

## Girdi: `Lubot nasil ogrenir?` (tohum 42)

```
[ the ında I is in in   in kayd  the  merkez -. sonuç T:
[ et çalış kaydını kaydını ga   kaydını geçirone mer, Y
```

Ortalama log-olasilik: **-1.9186** (jeton basina; sifira yakin daha iyi).

## Girdi: `Sicaklik nedir?` (tohum 7)

```
st  kaydını  ,  ş:0010
antıdan Bul  kaydını 
 ki baş kaydını -TE-000 sonuç.
  k.
ki şar kaydını ,  A-000 kayded
```

Ortalama log-olasilik: **-1.8911** (jeton basina; sifira yakin daha iyi).

## Girdi: `Sicaklik nedir?` (tohum 42)

```
ile st T-TE-000, : şar merkezinde:  merkezindeki Y[
normal_gTE- kaydını kaydını  merkezinde kaydını  kaydını kaydını ş:  kayd çalışan
```

Ortalama log-olasilik: **-1.6492** (jeton basina; sifira yakin daha iyi).

## Okuma notu

* Cikti **bilerek** duzeltilmedi: modelin bugunku hali budur.
* Hicbir blokta veri kaynagi adi, lisans adi ya da boru hatti etiketi gecmiyor
  (`id:`, `split:`, `provenance:` gibi). Bu, alim hattinin arindirma kuralinin
  olculen sonucudur: ilk iki egitim kosusunda bu etiketler ciktiya siziyordu.
* Bu bir *yetenek* iddiasi degil, *yuzeyin calistiginin* kanitidir: girdi
  jetonlanir, yerel agirliklardan dagilim kurulur, orneklenir ve cikti Markdown
  semasindan gecer. Sema gecmezse komut hata verir - yumusatma yok.
* Daha iyi metin icin olcek gerekir: daha cok adim, daha buyuk model (K6 tavanina
  kadar) ve daha cok jeton. Bu turun butcesi 350 adimdi; sayilar
  `training/ckpt/egitim-raporu.md` icindedir.
