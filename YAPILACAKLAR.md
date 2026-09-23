# Yapılacaklar — tek liste

Bu dosya işin tek listesidir: yapılan işaretlenir, kalan işaretlenmez. Rapor
disiplini değişmez: her "bitti" bir commit SHA'sı ve bir CI koşusu ister.

**Dış repolar yalnız YÖNTEM esinidir (K1).** Aşağıdaki listede adı geçen
depoların hiçbirisinden kod, veri veya ağırlık alınmaz; alınan şey, bir kararın
nasıl bölündüğü ve nasıl ölçüldüğüdür. Lisans PolyForm Shield 1.0.0, eser
Lubot'un kendisi.

## NN §8 iskeleti

- [x] **Adım 1 — Ölçüm.** `training/bench_hardware.py`, K6 donanım tavanı.
- [x] **Adım 2 — Sözlük.** Donmuş `lubot-bpe-v1/v2`, `tokenizer-vocab-is-frozen` kapısı.
- [x] **Adım 3 — Mimari.** μP init ölçeği ölçüldü (dikkat ölçeği oranı 0.497,
      beklenen 0.500; bağlı readout sapması 0.0124, bant 0.10; parametre 924.288).
- [x] **Adım 4 — Veri karışımı.** `training/veri_karisimi.py`, katman pay bandı,
      kimlik-şekli filtresi (kapı 52).
- [x] **Adım 5 — Düzenlileştirme.** `training/duzenlilestirme.json` +
      `training/egitim_butcesi.py`; jeton bütçesi ratchet'te (kapı 53).
- [x] **Adım 6 — Karar başlığı önce.** `crates/tomurcuk`: üç kapalı çıktı şekli,
      sabit kademe sırası, kalibre güven, k-of-n; `lubot karar` (kapı 54).
- [x] **Adım 7 — Kendinden-damıtma turu.** `training/onyukleme.py`: geçenler
      sayılır, geçmeyenler **nedeniyle** negatif havuza yazılır, yeterlilik farkı
      iki kayıttan yeniden hesaplanır (kapı 55). Tur 1: 88/88 satır geçti,
      0 negatif, fark karşılaştırılabilir değil (ilk tur bir tabandır).
- [x] **Adım 8a — Kıyas sınıfı beyanı.** `training/kiyas_sinifi.py` (kapı 56):
      parametre `model_spec.json`'dan, jeton `egitim_butcesi.py --olc`'ten
      okunuyor (ikinci literal yok). Ölçülen: **924.288 parametre**, sınırın
      (SmolLM2-135M = 135.000.000) **altında**; 96.645 benzersiz jeton;
      **0.1046 jeton/param** (yani parametre başına 9.56 jeton). Kural:
      "kapışma" iddiası yalnız **görev ekseninde** (alıntı doğruluğu + red
      disiplini) yapılabilir; parametre ekseninde bir iddia kaydı kapı reddediyor.
      Ratchet'e `exam` anahtarı eklendi (sınav sorusu sayısı, yalnız yükselir).
- [ ] **Adım 8b — Held-out sınav setini yaz.** `training/eval/sinav-seti.jsonl`
      bugün **0 soru**; damga mekanizması kurulu ve kanaryalı (damgasız bir soru
      reddediliyor). Kalan iş: soruları yazmak ve her sorunun dayandığı pasajı
      `eval-only.json`'a damgalamak.
- [ ] **Adım 8c — İlk kapışma ölçümü.** Görev ekseninde, taban olarak raporla —
      zafer ilanı değil. Eğitilmiş kontrol noktası yokken (K6) ölçülemez.

## Dış repolardan derlenen yöntemler

- [x] **jev-drone → kademe disiplini.** Uçuş kontrolü dengeyi tutar, üst
      katman yalnız karar verir. Lubot'ta karşılığı zaten var: `Oncelik::SIRALI`
      (belirlenimci kod → karar başlığı → üretken) ve `kademe_atlandi()`.
- [x] **jev-curate → önyükleme turu.** "Veriyi yargıla, sonra hangi satırın
      bir sonraki tura gireceğine karar ver." `training/onyukleme.py` bu yöntemi
      uyguluyor; dış ödül modeli yok, ölçüt mekanik.
- [x] **killmyidea → kapalı karar sözlüğü.** Serbest görüş yerine KILL/FIX/SHIP.
      Karşılığı `Sonuc::{Kesin, Yukselt, Red}` ve üç kapalı `Karar` şekli.
- [x] **agent-desktop / typesafe-mario → ham algı yerine tiplenmiş durum.**
      Karşılığı `crates/read` üç kanalı ve `dosya` yönlendiricisi.
- [ ] **Canny → "bitti" iddiasının denetimi.** Bir tamamlanma iddiası ancak
      mekanik kanıtla geçer. Lubot'ta disiplin zaten bu (CI tek doğrulayıcı,
      ratchet yalnız yükselir); kalan iş: bir commit/PR gövdesindeki her
      "bitti" ifadesinin yanında SHA + koşu numarası arayan bir kapı.
- [ ] **jev-trader → karar gecikmesi ekseni.** Karar yolu üretken yoldan ayrı
      ölçülmeli. Kalan iş: `lubot karar` için tekrarlanabilir bir gecikme
      tabanı (N koşu, medyan) ve bunun ratchet'e `karar_gecikme_ms` olarak
      girmesi — U bölümünün hız ekseni.
- [ ] **Prism → mevcut hattı değiştirmeden önce yargılayan katman.** Kalan iş:
      karar başlığını `ask`'ten önce bir kabul filtresi olarak bağlamak; kapsam
      reddi ve enjeksiyon tespiti başlığa devredilirken kalite bataryasının
      düşmediğini ölçmek (T'nin "yalnız ölçümle devir" kuralı).
- [ ] **neo4jev → kenarı yargılayarak gezinme.** `graf` ve `mufredat` bugün
      sabit sıra veriyor. Kalan iş: müfredat sırasını statik listeden, puanlı
      kenar seçimine taşımak; her adımın neden seçildiği kayda girer.
- [ ] **OneVOneJev → kararı bağımsız eksenlere bölmek.** Kalan iş: kapsam reddi,
      ilgililik sıralaması, dil tespiti ve efor önerisi karar noktalarını
      `tomurcuk`'un kapalı `Secenek` kümesine gerçek çağrı sahibi olarak bağlamak
      (bugün küme tanımlı, çağrı sahipleri kademeli gelecek).

## Operatör kararı bekleyen bulgular (kod değiştirilmedi)

- [ ] **`bulgu_veri_butcesi`.** Spec'in 1.94 token/param beyanı **yüzey**
      korpusuna (1.791.712 jeton) ait; CI'ın kurduğu **self** korpusuyla ölçülen
      0.1024 — ~19 kat fark. Hangi korpusun eğitileceği operatör kararı.
- [ ] **`bulgu_mufredat_isareti`.** `training/curriculum/format.jsonl` içinde 2
      satır `kind: "negative"` taşıyor; `make_sft.py:84` damgayı koşulsuz eziyor.
      Veri sızıntısı değil (geçersiz Markdown `user` turunda, cevap doğru bir
      red); kaybolan şey işaretin kendisi. Hangi işaretin korunacağı operatör
      kararı.

## Süreç notları

- [x] **CI sırasını yerelde birebir koşmak.** `Gate self-tests` korpus
      kurulmadan **önce** koşar; korpus isteyen bir self-test yerelde geçer,
      CI'da düşer (run 39'un kök nedeni). Artık her turda self-testler korpus
      silinmiş hâlde de koşuluyor.
- [x] **`concurrency.cancel-in-progress: true`.** Ana dal dışındaki bir dala
      push, kuyruktaki koşuyu iptal eder (run 40 böyle iptal oldu). Bir koşunun
      sonucu gerekiyorsa push'tan önce beklenir.
