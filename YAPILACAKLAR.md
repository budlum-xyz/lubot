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
- [x] **Eğitim çekirdeği** (iskelette ayrı adı yok, 5 ile 8 arasında duruyor).
      `crates/egitim` + `lubot egitim`. Repoda backward, optimizer ve train step
      **yoktu**; spec tamdı ama onu koşacak çekirdek yoktu. K1 gereği sıfırdan:
      otograd kütüphanesi yok, üçüncü taraf ağırlık yok. `Spec::lubot_a1()`
      spec'i birebir kuruyor (924.288 parametre), `parametre_sayisi()` bunu
      testle spec'in beyanına bağlıyor. Doğruluk **savunulmuyor, ölçülüyor**: 19
      tensör alanının **344 parametresinin tamamı** merkezi sonlu farkla
      karşılaştırılıyor ve beklenen sayı spec'ten alınıyor, yani sonradan eklenen
      bir tensör denetimsiz kalamıyor. Sonuç **344/344**. Tolerans göreli +
      mutlak taban: ölçüldü, en kötü parametrede analitik `-1.280e-6` ve sonlu
      fark `-1.280e-6` (dört anlamlı basamak aynı) ama naif göreli hata 8.6e-5 —
      sebebi sonlu farkın yuvarlama tabanı, yanlış gradyan değil. Ölçülen iniş:
      30 adım, kayıp **4.156951 → 2.501726** (bellek içi dizi; komut bunun
      korpus ölçümü olmadığını aynı satırda söylüyor). Epoch tavanı buraya
      yazılmadı, `lubot-grant`'ten okunuyor.
- [x] **Rust jetonlayıcı** — `crates/jeton` + `lubot jetonla` (kapı 61).
      Donmuş `lubot-bpe-v2` sözlüğünü okuyor (fail-closed: `vocab_size == 256 +
      merge`, birleştirme DAG'ı, ve **uygulayamadığı ön-işlem deseni reddediyor**
      — yaklaşık uygulamak yerine). Ölçülen: **1767/1767 kayıt Python'la birebir
      aynı jetonlandı, 102654 jeton**, 93712 ön-jeton. İki kolay hata ölçülerek
      yakalandı: (a) dört sınıfı karakter bölümlemesi sanmak — `[\W_]+` açgözlü
      ve `\W` boşluğu içerdiği için `"; oku"` → `"; "` + `"oku"`, üç ön-jeton
      değil; (b) `\d` yerine `is_numeric()` kullanmak Nl/No'yu (½, ², Romen
      rakamı) içeri alırdı, Nd kategorisi kullanıldı. Kalan: bu jetonlayıcıyı
      eğitim çekirdeğine bağlayıp korpus üzerinde gerçek bir tur koşmak.
- [ ] **Eğitim çekirdeğinin korpusla ilk gerçek turu.** Jetonlayıcı hazır,
      çekirdek hazır; eksik olan ikisini birleştiren veri yolu (korpus → jeton
      dizisi → pencere) ve K6 donanımında ölçülecek ilk kayıp eğrisi. Sınav
      skoru hâlâ ölçülemez: eğitilmiş kontrol noktası yok.
- [x] **Ölçümün kendine geri beslenmemesi kuralı** (kapı 60
      `measurements-do-not-feed-back`). Bulundu ve kurala bağlandı: `README.md`
      korpusun içinde olduğu için ratchet satırındaki korpus türevi sayılar
      ölçümü kendine bağlıyordu — sabit nokta **yok** (101443 yazınca 101444,
      101444 yazınca 101443 ölçülüyor; ölçüldü). Sayılar
      `training/ratchet.json`'a taşındı (korpusa girmiyor). Kapı şimdi 53
      korpus-içi dosyayı korpusun kendi sayılarına karşı denetliyor. Kendi
      yazarını yakaladı: ilk sürümde self-test kanaryası gerçek sayıları sabit
      olarak içeriyordu ve `gates/check.py` de korpus-içi bir dosya — kanarya
      artık gerçek olamayacak sayılar kullanıyor. Üç kanarya: yakalanan dosya,
      temiz dosya, uzun bir sayının içinde geçen alt dizi.
- [x] **Adım 8a — Kıyas sınıfı beyanı.** `training/kiyas_sinifi.py` (kapı 56):
      parametre `model_spec.json`'dan, jeton `egitim_butcesi.py --olc`'ten
      okunuyor (ikinci literal yok). Ölçülen: **924.288 parametre**, sınırın
      (SmolLM2-135M = 135.000.000) **altında**; 96.645 benzersiz jeton;
      **0.1046 jeton/param** (yani parametre başına 9.56 jeton). Kural:
      "kapışma" iddiası yalnız **görev ekseninde** (alıntı doğruluğu + red
      disiplini) yapılabilir; parametre ekseninde bir iddia kaydı kapı reddediyor.
      Ratchet'e `exam` anahtarı eklendi (sınav sorusu sayısı, yalnız yükselir).
- [x] **Adım 8b — Held-out sınav seti.** `training/sinav.py` (kapı 57):
      **12 soru**, her biri kendi pasajının `content_id`'sini
      `eval-only.json`'a damgalıyor. Seçim kuralı yazılı (elle seçilmiş sınav
      seçeni ölçer): `doc` kayıtları, `content_id`'ye göre sıralı, dosya başına
      en fazla bir soru, eşit aralıklı 12 adet. **Eksik olan önleme kapatıldı**:
      `make_sft.py` artık damgalı pasajları eliyor (`dropped_eval_only`), yani
      sınav seti gerçekten held-out; `eval_sft`'in reddi ikinci duvar olarak
      duruyor. Ölçülen: 1701 grounded satır, damgalı 0; SFT 1789 satır
      (1701 + 88 mufredat). Ne ölçtüğü dar ve kayıtlı: getirme + alıntı; soru
      metni pasajın ilk satırından türediği için skor bir **üst sınır**.
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
- [x] **Canny → iddianın kanıtını taşıması.** Bir iddia ancak mekanik kanıtla
      geçer. Kapı 58 `claims-carry-their-evidence`: her `bulgu_*` alanı
      `olculen` (içinde **sayı** olmak zorunda) + `hukum` + `yapilmayan`
      taşımak zorunda; "ölçüldü" deyip sayı taşımayan bir cümle reddediliyor.
      Bu turda yakalanan iki gerçek kusur: `bulgu_veri_butcesi.hukum` içinde
      **sabit "89.443"** yazıyordu (ölçülen 97.515) — dinamikleştirildi; ve
      alan `olculen`/`yapilmayan` taşımıyordu — normalize edildi. Eşleşme
      `bulgu_` önekiyle sınırlı, çünkü mufredat sınıflarından birinin adı
      `bulgular` ve bir sınıf adı iddia değildir.
- [ ] **Canny'nin kalan yarısı:** commit/PR gövdesindeki "bitti" ifadelerinin
      yanında SHA + koşu numarası aramak. Çalışma ağacında olmadığı için kapı
      olarak değil, süreç kuralı olarak duruyor.
- [x] **jev-trader → karar gecikmesi ekseni.** `training/karar_gecikme.py`
      (kapı 59): `lubot karar tek evet:0.9` **50 koşu**, medyan **2.237 ms**,
      en düşük 2.059 / en yüksek 3.507 ms, sapma 0.357 ms. Ölçülen süre **süreç
      başlatmayı içeriyor** ve bu, kaydın kapı tarafından zorunlu tutulan
      `uyari` alanında duruyor: okunacak sayı bir çağırının ödeyeceği **üst
      sınır**, başlığın kendi maliyeti değil. **Bilerek ratchet'e konmadı**:
      duvar saati makineye bağlıdır, burada ölçülen sayı daha yavaş bir CI
      makinesinde "gerileme" gibi görünürdü. Eksen beyan ediliyor, sayı eşik
      yapılmıyor.
- [ ] **jev-trader'ın kalanı:** karar başına enerji ve tekrarlı sorularda
      marjinal maliyet. Ölçüm düzeneği yok; donanım tarafı K6'da.
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
