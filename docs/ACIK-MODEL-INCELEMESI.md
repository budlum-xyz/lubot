# Açık kaynaklı AI'lar: inceleme notu (kod alınmadı)

Operatör talimatı: "diğer açık kaynak kodlu AI'lara da bak ama kodları alma lisans
gereği." Bu belge **incelemenin** kaydıdır. Buradan bu depoya **tek satır kod,
tek bayt ağırlık, tek kayıt veri girmedi**; giren tek şey, başkalarının
yayımlanmış *davranış* ve *yaklaşım* notlarından çıkarılan kavramsal kıyaslamadır.
K1 (from-scratch) maddesi bu depoda yürürlüktedir: model, eğitim döngüsü,
örnekleyici ve kapılar bu ağaçta yazılmıştır.

## İncelenen sınıflar ve çıkardığımız ders

### 1) Danışma modeli (Laya ailesi, Apache-2.0)

| Ne yapıyor | Bizim karşılığımız |
|---|---|
| Tek ileri geçişte karar; encoder tabanlı, üretim yok | `crates/cikarim` puanlama/sıralama; üretim ayrı yüzeyde (`uretim.rs`) |
| Ölçülmüş bir "kalibrasyon denetimi" (judge-audit): modelin verdiği oyun bilgi taşıyıp taşımadığı | K7: oy bilgi taşımıyorsa **geçersiz** sayılır, karar insana gider; ölçüm `autonomous-training/kosum/danisma-olcum.md` |
| Küçük yardımcı sınıflandırıcı, büyük model yerine | Lubot'un kendisi küçük (924.288 parametre), ölçüm odaklı |

**Almadığımız:** ağırlıkları, tokenizer'ı, kod. Yerel çalıştırma yalnız
*karşılaştırma* içindir (danışma katmanı, yalnız oy verir; K7).

### 2) Tipli karar sistemi (System One / Jev)

| Ne yapıyor | Bizim karşılığımız |
|---|---|
| Üç tipli çıktı: `choice` / `score` / `noul` — serbest metin yok | `Karar` enum'u üç kapalı şekil; `decision-head-has-no-generation-surface` kapısı `String`, `format!`, `write!` yasaklar |
| Kural öncesi sınıflandırıcı: hangi soru tipinin hangi yol olduğunu önceden ayırır | `criteria_metne_çevir` birebir eşleme; bilinmeyen etiket → oy geçersiz |
| Güven bantları; para harcayan kararda eşik 0,95 | Eşik ve marj **kodda** sabit (K7); danışma katmanı eşiği değiştiremez |
| "$0,042 / 1M girdi, 70–500 ms" gibi açık maliyet tablosu | Ölçüm kayıtları aynı şeyi kendi ekseninde tutar: `sure_saniye`, `girdi_jetonlari`, `maliyet` (yerel modelde $0) |

**Almadığımız:** prompt şablonları, SDK'lar, ağırlıklar. Kıyas, aynı kartlarda
ölçümle yapılır (`docs/REKABET.md`), kod alışverişiyle değil.

### 3) Açık kaynak dil modelleri (genel)

| Yaygın desen | Bizim kararımız |
|---|---|
| Hazır ağırlıkla başla (fine-tune) | K1 gereği yok; ağırlık bu ağaçta üretilir |
| Onlarca milyar parametre, GPU | K6: donanım kapasitesi tavanı bu makinede 924.288 parametre |
| Sampling: sıcaklık + top-k + nucleus + tekrarlama cezası | Sıcaklık/top-k/nucleus uygulandı (`ornekleyici.rs`); tekrar cezası **eklenmedi** — ölçülmeden davranış eklememek için |
| KV önbelleği ile artımlı üretim | Pencere 256 için her adımda baştan ileri geçiş; ölçüm gösterirse önbellek eklenir (şimdilik sadelik) |

## Neden bu not "iş bitmiş" sayılmaz

Bir davranışı ödünç almak (örneğin "güven eşiği olmadan serbest karar yok")
lisans zincirini ilgilendirmez: fikirler telif konusu değildir. Ama bu depoya
giren her **ifade** (kod, metin, veri) lisansa tabidir ve kapı da bunu mekanik
denetler (`corpus-records-carry-licence`, kapalı lisans kümesi). İnceleme
notunun sınırı bu: **bakıldı, öğrenildi, kopyalanmadı.**

Ölçüm tarafı ayrıca durur: kıyas iddiası `docs/REKABET.md` içinde kart bazında
sayıyla yazılır; bu belge yalnız "ne baktık" sorusunu yanıtlar.
