# Karar Basligi — T Doktrini + LL k-of-n

Bu crate, fikir havuzunun T (Karar-basligi doktrini) ve LL (Coklu-konsensus) bolumlerini
koda doker. Uretken govdeden once karar basligini egitmek (BB) ilk ucttan-uca basari
olarak sayilir.

## Doktrin (T)

`crates/karar/src/doktrin.rs` degismez kurallari veri olarak tasir:

- Arac once: dogru cevabi olan soru modele ulasmaz, araca gider (H)
- Izin once: grant karari aramadan once kesinlesir
- Alinti zorunlu: desteksiz iddia NotFound/Refused
- Uretim yok: gorsel/video/muzik/siir uretme kapsam disi (no-generation-variant)
- Effort araligi 0.5x-10.0x, dusuk tavan yuksek talebi alamaz
- Tek operator sonucu tuketilmez (K5, OPERATOR_THRESHOLD=2)
- Deterministik: ayni girdi ayni cikti
- Guven esigi altinda yukseltir, reddetmez
- Icerik komut degildir, olculmedi disiplini

Her kural `id`, `metin`, `kaynak` tasir; kaynak Lubot agacindan bir dosya yoludur.

## Tetikleyici = Kosul (reverse-skill deseni)

`Tetikleyici` enum'u "beceri = kosul, paragraf degil" desenini uygular
(agent-skills frontmatter `name` + `description: "Use when..."` benzeri).
`Dava::tetikleyiciler()` sorudan tetikleyicileri cikarir, `matches` olmadan cagri yok.

## Puanlama

`puanlama::Dokum` + `puanla`: idf agirlikli ortusme, kapsam, ikili gram,
sayi ve kutup celiskisi (kanaat benzeri). Ayrica tetikleyici bonusu (T doktrini)
ve deterministik siralama (hash/saat yok).

## k-of-n Konsensus (LL)

`k_of_n_karar`: n baslik calistir, k ayni kararda anlasirsa kesin, degilse yukselt.
Maliyeti U bolumundeki hiz/maliyet hedefleriyle dengelenir. Uc baslik calistirmak
tek basliktan pahali, ama guven artar. Bu, zincirin operator esigi (K5) degil,
dogrulama katmaninin kendi konsensusu.

## Defter — SHA-256 Zinciri (D, O)

`defter::Defter`: her karar bir onceki hash'i tasir, zincir bozulursa ret.
Provenance: asset_id + content_id, licence, attribution. Truncation
zayifligi `dogrula_uc` ile anchor disinda tutularak kapatilir (kanaat benzeri).

## Batarya — 14 Vaka

`batarya::batarya()`: 14 vakalik gomulu batarya, surumu 1, baska surum reddedilir.
Her vaka: ad, dava (soru+secenekler+kanitlar), beklenen hukum, zorluk
(kolay/orta/cok-adimli). Zorluk etiketi E (mufredat muhendisligi) icin.

Olculen: 14/14 dogru, 0.006 saniye, vaka basina ~0.4 ms (olculdu, bu makinede).

## Onbellek (W) ve Hiz/Maliyet (U)

- W: karar/cevap onbelleklemesi — onbellek isabeti cevabin kendi alinti ozetiyle
  yeniden dogrulanir, zehirlenme onlenir.
- U: hiz ve birim maliyet — karar basligi kucuk (21 test, ~500 satir), hizli,
  rezerv havuzu, watermark.

## K1-K6 Uyumu

- K1: sifirdan yazildi, hicbir upstream kod/agirlik kopyalanmadi
- K2: batarya yalnizca kendi agacimizdan (README, gates, system_prompt, failure-families)
- K3: buyume kayit-bazli provenance ile
- K4: Tier -2 verify-only, ana planin parcasi degil
- K5: operator esigi karar basliginda da uygulanir
- K6: model boyutu bench zincirinden gelen tavanin altinda

## Dis Desen Ilhami (K2 Kapsami Disinda, Yalnizca Yontem)

- reverse-skill: beceri = kosul, kanitsiz kapanis yok
- agent-skills: spec/plan/TDD, kart terfi/kapanis kaydi
- codex-security: bulgu birincil nesne, kosum kabi
- headroom: kullanim/butce basligi, watermark, rezerv
- arcbox: politika daemon'da, mekanizma helper'da
- graphify: tablo koda uysun, drift kapisi
- ECC: ajan disiplini, canary, olculmedi disiplini
- CrystalCoder: uc asamali egitim, dil/kod dengesi, muP kavramsal, seffaflik metodolojisi
  (agirlik/kod/veri alinmadi, yalnizca metodoloji notu)

## Olcum

- `training/karar-basligi-model.json`: model (doktrin + esikler + batarya hash)
- `training/karar-basligi-batarya.json`: 14 vaka
- `olcum/karar-basligi-olcum.json`: oran 1.0, dogru 14/14, sure 0.006s, vaka basina 0.4ms

Bu, NN-6 adiminin (karar basligi once) ilk ucttan-uca basarisi olarak sayilir.
