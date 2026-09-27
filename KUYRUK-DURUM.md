# Kuyruk durumu — 2026-09-25 (veri + konuşan model + rekabet turu)

Bu tur, önceki turun 9 kapanan maddesi ve K7 üzerine **yeni operatör görevini**
uyguladı: (1) Lubot'un eğitimi için veri doğrudan alındı, (2) model konuşan bir
yüzeye kavuştu, (3) rekabet ölçümü kuruldu, (4) ajanın kendi çalışma desenleri
veri olarak aktarıldı, (5) açık kaynaklı modeller yalnız incelendi.

## Bu turda kapananlar

| # | İş | Kanıt |
|---|---|---|
| A | Kamu malı veri alım hattı | `training/kamu_verisi.py` (lisans doğrulaması indirmeden önce, ad yazmadan, revizyon+sha256 damgası) |
| B | Korpus inşası tek komut | `training/corpus_insa.py`; izlenen veri dosyası `veri/kamu-mali.jsonl.gz` (**78.103 kayıt**) |
| C | K2 anayasa maddesi genişledi | `autonomous-training/INVARIANTS.md` + mühür `187b5b6f…` |
| D | no-generation yeniden yazıldı | "her türden veriyi okur ve inceler; kullanıcıya yalnız şema doğrulamalı Markdown" |
| E | Örnekleyici çekirdeği | `crates/cikarim/src/ornekleyici.rs` (sıcaklık/top-k/nucleus + deterministik akış) |
| F | Üretim döngüsü | `crates/cikarim/src/uretim.rs` (artımlı döngü, sızıntı kuralı) |
| G | Konuşan yüzey | `lubot sohbet` (çıktı Markdown şemasından geçer; geçmezse ret) |
| H | Çalışma deseni devri | `docs/CALISMA-DESENI.md` (20 desen, kendi eser) |
| I | Açık model incelemesi | `docs/ACIK-MODEL-INCELEMESI.md` (kod/veri alınmadı) |
| J | Yeni sertleştirme kapısı | `ingestion-refuses-unlicensed-sources` (kanaryalı) → **86 kapı** |
| K | Veri kabul kararı | `docs/VERI-KABUL.md` (kanıt zinciri, sınırlar) |
| L | Eğitim koşusu | 350 adım, doğrulama kaybı **3,847723**, etiket sızıntısı yok → `training/ckpt/` |
| M | Sohbet örnekleri | `docs/SOHBET-ORNEKLERI.md` (3 soru × 2 tohum, gerçek çıktı) |
| N | Rekabet ölçümü | `docs/REKABET.md` + kayıt: Lubot **3/12**, danışma katmanı 0/12 |
| O | Arınma sertleşti | Alan beyaz listesi + ad/adres silme + **görünmez ayırıcı onarımı** (`--onar`) |

## Ölçüler

| Ölçü | Değer |
|---|---|
| Korpus | **81.166** kayıt (3.063 kendi ağaç + 78.103 kamu malı) |
| Korpus damgası | `5da0b2aa…` (bozuk satır: 0) |
| Jeton | 14.173.000 |
| Kapı | 86 kapı; tam koşu **79 OK / 7 FAIL** (düzeltme sürüyor) |
| Test | 557 |
| Ratchet | `training/ratchet.json` ölçülenle tazelendi |

## Süren / sıradaki

- Tam koşuda düşen 7 kapının onarımı: `comparison-class-is-declared`,
  `crates-doc-is-measured`, `data-mix-is-declared`, `fmt-clean`,
  `pub-api-is-used`, `retrieval-at-k-is-measured`,
  `training-runner-engineering-vs-data`.
- Push + CI yeşili (son başarılı koşu `36045105638`, bu commit için değil).
- Damga kararı: eğitim `ff2d02ee…` (81.150) ile koştu; güncel korpus
  `5da0b2aa…` (81.166) — fark docs'ların kendi ağaca girmesinden.
- Kuyruk maddeleri 21, 3, 32/33/34/A.

## Bilinen maliyet

Korpus 2.931 → 81.166 kayda çıktı; kapı süitinin cevap yolu ölçümleri korpusla
ölçekleniyor (`tekrar_maliyeti.py` ~30 `ask` çağrısı yapar). Bu turda süitin
duvar saati kayda geçti; CI süresi buna göre uzar. Ölçüm küçültülmedi: ölçümü
korumak için süre kabul edildi.

## Bu turda öğrenilen tuzak

Satır tabanlı okuma ile JSON üretimi çakışabiliyor: `U+2028` gibi görünmez
ayırıcılar JSON'da kaçışlanmaz, ama `splitlines` onları satır sonu sayar; bir
kayıt iki satıra bölünür ve iki yarım kayıt da bozuk görünür. Arınma artık bu
karakterleri metinden çıkarıyor (`_gorunmezleri_sil`) ve her kayıt yazılmadan
önce tek satırda çözülebildiği doğrulanıyor; eldeki veri dosyası `--onar` ile
onarıldı (91 kayıt).
