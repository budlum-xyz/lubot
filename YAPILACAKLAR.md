# Yapılacaklar — tek liste

Bu dosya işin tek listesidir: yapılan işaretlenir, kalan işaretlenmez. Rapor
disiplini değişmez: her "bitti" bir commit SHA'sı ve bir CI koşusu ister.

**Dış repolar yalnız YÖNTEM esinidir (K1).** Aşağıdaki listede adı geçen
depoların hiçbirisinden kod, veri veya ağırlık alınmaz; alınan şey, bir kararın
nasıl bölündüğü ve nasıl ölçüldüğüdür. Lisans PolyForm Shield 1.0.0, eser
Lubot'un kendisi.

**Bu dosya tek kaynaktır.** Beş havuz burada birleşiyor: eğitim stratejisi
fikir havuzu (A–FF), genişletme promptu (GG–RR), awesome eşleştirme kataloğu,
dış repo kaydı (kullanıcı repoları + Shizuku + Stagehand + budlum-xyz ailesi)
ve direktif §4.1'in onaylı sekiz deposu.
Her madde bitirilecek bir iştir; katalog linkleri olduğu gibi taşınmadı, lubot'a
düşenler somut işe çevrildi, düşmeyenler gerekçesiyle kayıt defterinde duruyor.
Başka bir yerde plan listesi tutulmuyor: bir iş buraya yazılmadıysa yoktur.

İşaretler: `[x]` bitti (kanıt dosya ya da kapı adıyla maddede), `[~]` kısmen
(neyin eksik olduğu maddede yazılı), `[ ]` açık. **KAPSAM DIŞI** yazan maddeler
yapılmayacak değil, kararı verilmiş ve gerekçesi K1–K6'ya bağlı olanlardır.
Ölçülmeyen hiçbir sayı ölçülmüş gibi yazılmaz.

**Ölçülen kod hacmi** (bu ağaçta, `target/` ve `.git/` hariç): Rust **23.592**
satır / 45 dosya, Python **9.403** satır / 19 dosya — kod toplamı **32.995**
satır; belge ve veri dosyalarıyla birlikte 80.344. Crate bazında döküm ve
test sayıları `docs/CRATES.md`'de duruyor ve oradaki rakamları
`crates-doc-is-measured` kapısı kaynağa karşı doğruluyor — buradaki sayı
yalnızca büyüklük sırası için, yetkili kaynak o tablo. Operatörün koyduğu hedef yüz
binlerce satır. Bu sayı bir kalite ölçütü değil ve şişirmek için satır
üretilmeyecek: büyüme, sıfırdan yazılması gereken gerçek bileşenlerden gelecek —
eğitim döngüsünün tamamı (optimizer, scheduler, checkpoint, resume), veri hattı,
ölçme çatısı, çıkarım/servis yolu, karar başlığının çağrı sahipleri, retrieval
indeksinin ölçülen tarafı. Hangi bileşenin hangi maddede olduğu aşağıda yazılı.

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
- [x] **Veri yolu ölçümü** — `lubot egitim-veri` + `pencere_olcu` +
      `bulgu_veri_yolu`. Bu madde korpus türevi sayı taşımaz, çünkü korpus bu
      dosyayı da okuyor ve sayı buraya yazılırsa ölçüm kendi girdisini
      değiştirir (kapı `measurements-do-not-feed-back` bunu buldu, öngörmedi).
      Güncel rakamlar `lubot egitim-veri` çıktısında ve
      `training/eval/sonuclar/egitim-butcesi-*.json`'da. Sabit kalan hükümler:
      (1) spec'in `max_seq_len` = 256 beyanı ölçülerek doğrulanıyor ve p95 onu
      aşarsa "beyan YANLIS, spec yeniden dogrulanmali" satırı basılıyor;
      (2) asıl kısıt pencere uzunluğu değil **pencereleme stratejisi** — kayıt
      başına pencereleme jetonların büyük kısmını kuyruklarda atıyor, o yüzden
      rapor iki kapsamayı yan yana yazıyor ve fark 1 puanı aşınca bulgu
      satırı düşürülüyor. Yüzdelik yöntemi adı ile yazıldı (en yakın-rank),
      çünkü "p95" yöntem söylenmeden tek bir sayı değil.
- [x] **Kayıtlar arası paketleme** — `paketle` + `ileri_ve_geri_paket`.
      Pencere ve kapsama sayıları korpus türevi, bu yüzden burada değil
      `lubot egitim-veri` çıktısında duruyor; rapor her koşuda tek kaynaklı /
      çok kaynaklı pencere ayrımını ve bir pencerede birleşen en çok kayıt
      sayısını yazıyor. Kalıcı olan iki önlem: (1) her pencere **konum başına
      kaynak izi** taşıyor (`kaynak[i]`, `kimlikler[i]` ile aynı uzunlukta) —
      alıntı hangi kayda ait olduğunu kaybetmiyor, yani sınav setini yeniden
      damgalamaya gerek kalmadı; (2) **dikkat kayıt sınırını aşmıyor**, ileri
      ve geri geçişte. İkincisi savunulmuyor, ölçülüyor: paketli koşunun kaybı
      parçaların tek başına koşularına **eşit** olmalı (1.942361282232); maske
      kaldırıldığında aynı test **1.945561023341** okuyor ve düşüyor — yani
      test maskenin varlığını ölçüyor, yokluğunu değil. Bu iki sayı korpusa
      değil crate içindeki sabit dizilere ait, o yüzden burada durabilir.
      CLI artık paketli adımı gerçek korpus verisiyle koşuyor ve maskenin kaç
      konumda devreye girdiğini raporluyor.
- [ ] **Eğitim çekirdeğinin korpusla ilk gerçek turu.** Jetonlayıcı ve veri yolu
      ölçümü hazır; eksik olan paketlenmiş pencerelerle koşan tur ve K6
      donanımında ölçülecek ilk kayıp eğrisi. Sınav skoru hâlâ ölçülemez:
      eğitilmiş kontrol noktası yok.
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
      (SmolLM2-135M = 135.000.000) **altında**; 138.009 benzersiz jeton;
      **0.149314 jeton/param** (yani parametre başına 6.70 jeton). Kural:
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
      duruyor. Ölçülen: 2120 grounded satır, damgalı 0; SFT 2208 satır
      (2120 + 88 mufredat). Ne ölçtüğü dar ve kayıtlı: getirme + alıntı; soru
      metni pasajın ilk satırından türediği için skor bir **üst sınır**.
- [x] **Eğitim koşusu yüzeyi.** `lubot egitim-kosu`, `lubot cikarim
      {denetle,puanla,sirala}`, `lubot sinav-kosu`, `lubot korpus-damgasi`;
      çekirdekte `crates/egitim::{veri,kosu,kontrol}`, çıkarımda
      `crates/cikarim`. Üç kural fail-closed: damga beyan edilmeden koşu yok;
      `eval-only` damgalı kayıtlar eğitim akışından çıkarılır ve **sayısı
      rapora yazılır**; devam eden tur adımı, epoch'u, epoch içindeki pencere
      konumunu ve devralınan en iyiyi taşır (ölçüldü: 6+6 adım, kesintisiz 12
      adımın kayıp eğrisini 1e-12 içinde yeniden üretiyor). Kontrol noktası
      biçimi: `LUBOTCKPT` v1, başlık JSON + adlandırılmış bloklar + tek baytlık
      bozulmayı yakalayan SHA-256; `--f32` dosyaya yazılır. Dört yeni kapı
      (64–67): `checkpoint-round-trips`, `inference-cache-agrees`,
      `training-run-is-measured`, `reranker-is-measured`. Ölçülen: test 478 →
      **529**, kapı 63 → **67**; çıkarım önbelleği ile tam geçiş ve eğitim
      çekirdeği aynı dizide 1e-12'nin altında anlaşıyor.
- [x] **Adım 8d — Belge turları: GG–RR'nin kapıya bağlanması.** Uygulama
      promptu bölümleri ölçülebilir hâle geldi: MM
      `training-runner-engineering-vs-data` (kapı 68; koşucu yalnız
      stdlib+kardeş modül, veri yalnız kendi ağaçtan, provenance zorunlu),
      JJ `corpus-carries-structure` (kapı 69; api-doc-pair 429, trait-impl
      31, dependency-edge 85), RR `gap-report-is-measured` (kapı 70; audit
      günlüğünden bilgi-boşluğu haritası + `kind: gap-report` kaydı), QQ
      `doc-diagram-feeds-corpus` (kapı 71; Mermaid/SVG girdisi; görüntü
      dosyası metne çevrilmez), KK `gate-pairs-carry-referee` (kapı 72;
      iddia ↔ kanarya sayısı eşleşir, "kanaryada red yok" reddedilir).
      Ölçülen: kapı 67 → **72**. LL (k-of-n karar konsensüsü) açık: en az
      iki eğitilmiş karar başlığı ister; maliyet ekseni zaten
      `decision-latency-is-recorded` ile ölçülüyor.
- [x] **Adım 8e — Alma yüzeyinin ölçümü (I).**
      `training/erisim_geri_cagirma.py` sınav setindeki her sorunun
      damgalanmış pasajını `lubot ara` sıralamasında arar; isabet kuralı
      dosya + pasaj metni üzerinden kurulur (alıntıdaki sayı pasaj sırası,
      dosya satırı değil). Ölçülen (2026-09-24, korpus 2755 kayıt): tam
      soru metniyle **10/12 ilk sırada, 10/12 ilk 3'te**; yalnız çekirdek
      cümleyle 5/12 ilk sırada, **11/12 ilk 3'te**. Kayıt
      `training/eval/sonuclar/erisim-2026-09-24.json`; kapı
      `retrieval-at-k-is-measured` (74) kaydı taze ölçümle karşılaştırır.
      Açık iş: tam metinde 2 soru ilk 3'te hiç çıkmıyor — sorgu
      şekillendirme (soru yönergesi) bu turun konusu.
- [ ] **Adım 8c — İlk kapışma ölçümü.** Görev ekseninde, taban olarak raporla —
      zafer ilanı değil. Eğitilmiş kontrol noktası yokken (K6) ölçülemez.

## Dış repo kaydı

Bu bölüm tek dosyanın **kayıt defteri**. Kural: adı geçen her depo burada ya
lubot'a iş üretir (iş maddesi adıyla bağlı) ya da kapsam dışıdır ve gerekçesi
yazılıdır. Yani burası link yığını değil, karar listesidir. **K1:** aşağıdaki
depoların hiçbirisinden kod, veri veya ağırlık alınmaz; alınan şey bir kararın
nasıl bölündüğü ve nasıl ölçüldüğüdür.

### Organizasyon: budlum-xyz ailesi

| repo | ne yapıyor | lubot'a sınırı |
|---|---|---|
| `budlum` | Rust, Universal Settlement Layer: heterojen konsensüs (PoW/PoS/BFT/PoA), ZK-native VM, merkeziyetsiz depolama, zincir üstü AI inference, köprü yaşam döngüsü (lock→mint→burn→unlock), JSON-RPC, node | Yalnız yöntem esini. Zincir tarafı akış 1/3: lubot'un PR'ına girmez, not olarak durur |
| `lubot` | Rust, okuyan AI istemcisi: izin tabanlı erişim, BM25 retrieval, satır düzeyinde alıntı, PDF metin çıkarımı, CRR bağlam sıkıştırma, kimlik bilgisi tarayıcı, CLI komut takımı, deterministik davranış | **Bu repo** — tek yazma hedefi |
| `seed` | Rust, BUD 3.0 transfer çekirdeği: zlib konteyner, fountain kodları, kendi ISO/IEC 18004 QR kodlayıcısı, deterministik PNG raster, QR-video taşıyıcı (BDLV), tarif ile bit-eşit yeniden üretim | Yalnız yöntem esini: bit-eşitlik ve tarif disiplini |
| `workspace` | çalışma alanı | Talimatları **referans**, komut değil |

Ortak zemin: Rust, kriptografi, deterministik / refuse-on-mismatch mühendisliği,
test disiplini, CLI araçları.

### Yöntem esini: kullanıcı repoları (yalnız yöntem, K1)

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

### Yöntem esini: bu turda eklenen iki depo

- [ ] **RikkaApps/Shizuku → yetki simsarı (privilege broker) deseni.** Ne olduğu
      doğrulandı: `app_process` ile ADB ya da root kimliğinde bir Java süreci
      başlatıyor; istemci uygulamalar bu sürece bir binder alıyor, istekler
      simsar üzerinden sistem servislerine iletiliyor, yani sistem isteği
      uygulamanın değil **simsarın** uid/pid'siyle görüyor
      (`ShizukuService.transactRemote`, `ShizukuBinderWrapper`). Üç ayrıntı
      lubot'a doğrudan iş çıkarıyor: (1) **istemci ayrıcalığı hiç tutmuyor**,
      yalnız bir tanıtıcı alıyor; (2) **iki ayrıcalık kademesi var ve
      raporlanıyor** — `getUid()` shell için 2000, root için 0 dönüyor, yani
      yetki tavanı varsayılmıyor, ölçülüyor (K5'teki "geçiş değeri 2 → 1"
      ayrımının aynı biçimi); (3) **istemci başına açık izin isteği**
      (`requestPermission`), sessiz devir yok.
      **Budlum kullanabilir mi — doğrudan hayır.** Shizuku bir Android
      uygulaması + Java/Kotlin API'si; budlum ise Rust bir settlement katmanı
      ve Android yüzeyi profilinde yok. Ayrıca K1 üçüncü taraf kodu zaten
      yasaklıyor. Ama **desen** budlum'un JSON-RPC yetkilendirmesine birebir
      oturuyor: hangi metodun hangi istemciye açık olduğu, her çağrının çağıran
      kimliğiyle ilişkilendirilmesi, köprü yaşam döngüsünde her adımın ayrı
      yetki denetimi taşıması.
      Lubot'ta iş: `crates/grant` defterini ve `it`'in kısıtlı push'unu bu
      desenle denetlemek — sır tek yerde mi duruyor, çağıran kimliği her
      iletmede korunuyor mu, yetki tavanı beyan ediliyor mu. Ölçülen çıktı bir
      `docs/GRANT-KIYAS.md` kaydı.
      **Kırmızı takım girdisi:** Shizuku'nun kendi kötüye kullanım literatürü,
      tek seferlik bir ADB onayının yeniden başlatma sonrası kendini doğuran
      bir cihaz-içi yetki simsarına dönüştüğünü belgeliyor. Yani simsar
      deseninin üç zorunluluğu var: geri alma, süreç yeniden başlatmaya
      dayanıklı çağıran doğrulaması, taze onay olmadan kendini doğurmama.
      Bu üçü `kirmizi-senaryolar` bataryasına madde olarak girecek (Y).
- [ ] **browserbase/stagehand → gözle, doğrula, önbellekle, tekrar oynat.** Ne
      olduğu doğrulandı: Playwright'ı üç LLM-destekli ilkel ile genişleten
      TypeScript SDK — `act()` doğal dilde eylem, `extract()` Zod şemasıyla
      doğrulanmış yapısal çıkarım, `observe()` eylemi **yapmadan** aday eylem
      listesi (seçici + metod + argümanlar). Altında DOM'a betik enjekte edip
      aday öğeleri (yaprak ya da etkileşimli) topluyor, görünmeyenleri eliyor
      ve modele ham DOM yerine **numaralı öğe listesi** veriyor. Dört somut iş:
      (1) **gözle→doğrula→önbellekle→tekrar oynat.** Model bir kez keşfeder,
      doğrulanan eylem deterministik artefakta dönüşür, sonraki koşular modeli
      çağırmaz. Bu, W maddesinin (karar/cevap önbellekleme) ve önyükleme
      döngüsünün (G) tam karşılığı: kapıdan geçen çıktı bir sonraki turun
      deterministik girdisi oluyor. İş: `observe` benzeri bir önizleme yüzeyi,
      doğrulanmış kararın önbelleğe yazılması, tekrarlı soruda marjinal
      maliyetin ölçülmesi.
      (2) **Önbellek kaçırınca kendini onarma.** Önbellekli seçici tutmazsa
      modele dönülüyor, önbellek yenileniyor ve bu **bildiriliyor**. Lubot'ta
      karşılığı: önbellekli kararın dayanağı (korpus özeti) değiştiğinde kararı
      yeniden türet ve yenilendiğini kayda yaz. Ölçülebilir olan: bayat
      önbellek yakalama oranı.
      (3) **Şemayla doğrulanmış çıkarım.** Tiplenmiş çıktı ya da red — lubot'un
      `ai-output-schema-enforced` ve "en yakın biçime düşme yok" kuralıyla
      aynı çizgi. İş: şemanın kapsanmayan durumlarını saymak.
      (4) **Modelden önce yüzeyi küçült.** Ham DOM değil aday listesi.
      Karşılığı `sikistir` (CCR) ve JJ maddesindeki AST-farkında parçalama:
      modele giden bağlamın ne kadarının elendiği ölçülecek.
      **K1 notu:** Stagehand dış LLM sağlayıcılarına bağımlı; lubot dışarıdan
      model çağırmıyor (`reads-not-generates`, `no-generation-variant`).
      Alınan şey desendir, bağımlılık değil.

### Yöntem esini: onaylı sekiz depo (direktif §4.1)

Sekiz depo bu turda tek tek tarandı. Statü X araştırmasıyla aynı: kod, veri
veya ağırlık alınmaz (K1); bu depoların çalıştırılabilir olanları workspace'in
kendi `/skills` altyapısına aittir, lubot'un model/eğitim kodu bunlardan yalnız
desen çıkarımıyla beslenir. Kullanıcının bu tur attığı on Jev'li liste de
tekrar kontrol edildi: on madde de önceki turlarda bu listeye işlenmiş durumda
(agent-desktop ve typesafe-mario ortak maddede; OneVOneJev, Canny'nin kalan
yarısı, jev-trader'ın kalanı, Prism ve neo4jev yukarıda açık maddeler olarak
duruyor) — tekrar açılmadı, açık olanlar bu turun sırasına girdi.

- [ ] **Graphify-Labs/graphify → kenarı açıklanmış bilgi grafiği.** Ne olduğu
      doğrulandı: kod tabanını, belgeleri ve şemaları sorgulanabilir bilgi
      grafiğine çeviriyor; ayrıştırma yerel ve deterministik, vektör deposu
      yok ve her kenarın gerekçesi duruyor. Lubot'a düşen iş: `crates/index`
      BM25 üstünde; JJ'nin AST maddesiyle birleşip pasajlar arası kenar
      katmanının taslağı — her kenar türü ve üretim kuralı kaydedilir,
      alıntı kaynağı kaybolmaz (paketlemedeki konum-başına-iz ilkesinin
      grafik hâli). Ölçülecek şey: kenar katmanının retrieval bataryasındaki
      (Z) skora etkisi. Vektör-deposu deseni alınmıyor: benzerlik BM25 ve bu
      karar GG'deki veri-sınırlı sınıf kalibrasyonuna bağlı.
- [ ] **addyosmani/agent-skills → beceri kaydının şema disiplini.** Üretim
      sınıfı beceriler derlenmiş bir artefakt olarak duruyor; not değil.
      Lubot'taki karşılığı `crates/yetenek` (kendi kuralı: beyan edilmiş,
      çalışıyor demek değildir — self-test geçmeden kullanılamaz). İş:
      `training/curriculum/yetenek.jsonl` kayıtlarına kanıt alanının zorunlu
      tutulması taslağı — tetikleyici, kapsam dışı ve ölçülen kanıt alanı
      eksik kayıt reddedilir (fail-closed; eval-only damgasının kayıt
      disiplinindeki hâli).
- [ ] **openai/codex-security → bul → doğrula → düzelt döngüsü.** Güvenlik
      açığını bulan, doğrulayan ve düzelten CLI+SDK. Lubot bulgu disiplinini
      uyguluyor (Strix turları); eksik olan resmileştirme. İş: bulgu rapor
      şemasına **yeniden-üretme adımını** zorunlu alan eklemek — yeniden
      üretilemeyen bulgu kapanamaz. Taslak `docs/failure-families.md`
      yanına ayrı kayıt olarak.
- [ ] **headroomlabs-ai/headroom → sıkıştırmada kayıp ölçümü.** Araç
      çıktılarını ve parçaları modele girmeden sıkıştırıyor ve tasarrufu
      ölçülen olarak raporluyor. Lubot karşılığı `crates/sikistir` (pinlenen
      satırlar bayt baytta, CCR deposu digest yeniden doğruluyor). İş:
      sıkıştırma sonrası **alıntı koruma ölçümü** — sıkıştırılmış bağlamla
      üretilen alıntı özgün kaydı hâlâ buluyor mu; bulamıyorsa sıkıştırma
      kararı reddedilir. Tasarruf rakamı, kayıp ölçülmeden başarı sayılmaz.
- [ ] **arcboxlabs/arcbox → her deneye ayrı kök.** İzole makineler: kendi
      çekirdeği, dosya sistemi, ağı; yerel-first, OCI uyumlu. Lubot
      karşılığı `crates/izolasyon` (oturum boş çalışma alanına açılır,
      sonuçlar kopya çıkar). İş: izolasyon self-test'lerine **ağ-yok
      kanıtı** taslağı — izole oturumda dış kaynak okuma denemesinin
      reddi ölçülür; Android beyanı "ağ izni yok" satırının manifest'ten
      düşmediği CI kontrol listesine girer.
- [ ] **zhaoxuya520/reverse-skill → beceri yönlendirici + isteğe bağlı araç
      zinciri.** Doğrulandı: tersine mühendislik / yetkili sızma beceri
      paketi; yönlendirici beceriyi seçiyor, araç zinciri ihtiyaç anında
      kuruluyor, deneyim tabanı kendini büyütüyor. İçerik kapsam dışı
      (M'nin kırmızı takım sınırı: saldırı becerisi lubot'un iş alanı
      değil). Alınan desen: **seçimin kanıtı** — yönlendirici hangi
      beceriyi neden seçtiğini kayda yazar. İş: `karar`/`yetenek`
      çağrılarında seçim günlüğü taslağı; önyükleme turunun (G) "hangi
      satır ikinci tura girer" kararına bağlanır.
- [ ] **affaan-m/ECC → performans bütçesi disiplini.** Koşum takımı
      performans sistemi: beceriler, sezgiler, bellek, güvenlik. Lubot'a
      düşen eksen: her yeni bileşenin başlatma ve ikili-boyut maliyeti beyan
      edilir. Kapı 59 gecikmeyi, `derle.sh` APK içeriğini doğruluyor; eksik
      olan crate başına ikili bütçesi. İş: `--release` ikili boyutunun
      kapıya bağlanması taslağı (sayısal eşik değil, kayıtlı beyan +
      gerileme ratchet'i).
- [ ] **1jehuang/jcode → bellek bütçesi ölçümü.** RAM-verimi iddialı Rust
      koşum takımı. K6 donanım tavanı olan projede bellek ikinci kısıt:
      `egitim` çekirdeğinin ve `ask` yolunun en yüksek RSS'i ölçülmüyor.
      İş: RSS zirvesi ölçüm taslağı — makineye bağlı, kapı-59 disiplini:
      eksen beyan edilir, sayı eşik yapılmaz, ratchet'e konmaz.

### Awesome katalog kaydı (213 benzersiz depo, 6 grup)

Katalog `uploads/budlum-awesome-eslestirme.md`. Sayılar dosyadan ölçüldü,
tahmin değil. Her grubun lubot kararı tek satır; grubun içinde lubot'a iş
üreten depolar "Awesome listesinden lubot'a düşenler" bölümünde tek tek
maddeye çevrildi.

| katalog grubu | benzersiz depo | lubot kararı |
|---|---|---|
| 1. Doğrudan isabet — çekirdek teknolojiler | 74 | Kısmen: Rust / IR / QA / XAI / enjeksiyon / NLG / fuzzing / güvenlik lubot'a iş üretiyor. Konsensüs, ZK, post-kuantum, merkeziyetsiz depolama, P2P **budlum**; QR/video/codec **seed** |
| 2. Güçlü destek — mimari, teori, kalite | 37 | Kısmen: yazılım mimarisi, test, statik analiz, ampirik yazılım mühendisliği lubot'a iş üretiyor. Dağıtık sistemler ve NoSQL/veri depolama **budlum** |
| 3. Geliştirme ortamı, CLI ve iş akışı | 32 | Kısmen: CLI, kabuk, git kancaları, CI/CD saldırıları, düzenli ifadeler lubot'a iş üretiyor. k8s/ansible/terraform/SRE operatör altyapısı, kapsam dışı |
| 4. Dokümantasyon, açık kaynak yönetimi, ürünleşme | 30 | Kısmen: README, Markdown, lisans/kamu malı kaynaklar, adlandırma, çeviri lubot'a iş üretiyor. Ürünleşme/pazarlama maddeleri kapsam dışı |
| 5. Öğrenme ve topluluk | 12 | Yalnız bağlam: `mufredat`'ın insan versiyonu olarak not edildi, kod işi üretmiyor |
| 6. Kıyıda ama gerekirse | 27 | Kapsam dışı: web sitesi, PWA, istemci çatıları, çevre araçlar — lubot CLI ve akış 2 kapsamında |

Kapsam dışı bırakılanların ortak gerekçesi: ya başka bir reponun alanı
(budlum/seed), ya operatör altyapısı, ya da akış 1/3. Hiçbiri "bakılmadı"
değil, "karar verildi ve gerekçesi yazıldı".

## Fikir havuzu A–FF (lubot-egitim-stratejisi-fikir-havuzu.md)

Kaynak dosya `uploads/`ta; buradaki her madde o bölümden çıkarılan **bitirilecek
iş**. Durumlar bu repoda doğrulanmış artefaktlara bağlanıyor, tahmine değil.

- [~] **A — Veri kaynağı: tek repo yerine tüm budlum yüzeyi.** `build_corpus.py`
      self korpusunu deterministik kuruyor (`corpus-build-is-deterministic`),
      sözlük ise yüzey korpusundan (`budlum-yuzeyi.jsonl.gz`) kesilmiş.
      **Açık olan:** hangi korpusun eğitileceği — bu bir operatör kararı ve
      `bulgu_veri_butcesi` olarak kayıtlı: yüzey korpusu self korpusunun yaklaşık
      18 katı; iki korpusun rakamları `docs/spec.md`'de ölçülmüş hâliyle duruyor.
- [x] **B — Zincir-kaynaklı canlı veri akışları. KAPSAM DIŞI.** Karar: zincir
      tarafı `TrainingDataGrant` dışa aktarımı kapalı konu; yalnız akış 2
      kapsamında çalışılıyor. `crates/grant::training` epoch defterini
      doğruluyor, zincire yazmıyor.
- [x] **C — Sentetik veri. KARAR: 0.** `veri_karisimi.py` sentetik katmanı
      ölçüyor ve **0** beyan ediyor (`data-mix-is-declared`); dışarıdan öğretmen
      yok, çoğalma kendinden-damıtmayla yapılıyor (G).
- [x] **D — Küratörlük ve kalite kapıları.** 61 kapı + `findings-are-disciplined`
      + `claims-carry-their-evidence` (her bulgu ölçülen sayı + hüküm +
      dokunulmayanları taşır). `onyukleme.py` geçenleri sayıyor, geçmeyenleri
      **nedeniyle** negatif havuza yazıyor.
- [x] **E — Müfredat mühendisliği.** `training/curriculum/*.jsonl` (ajan,
      behaviour, bulgular…), SFT'ye 88 satır olarak giriyor; kapı müfredat
      satırlarının sayısını ratchet'te tutuyor.
- [x] **F — Küçük-veri rejimine uygun mimari.** `model_spec.json`: derin-dar,
      924.288 parametre, 8 katman, d_model 64; `model-spec-is-consistent` ve
      `crates/egitim`'deki `parametre_sayisi()` testi beyanı birbirine bağlıyor.
- [x] **G — Kendinden-damıtma / önyükleme döngüsü.** `onyukleme.py`, tur 1:
      88/88 satır geçti, 0 negatif, yeterlilik farkı karşılaştırılabilir değil
      (ilk tur taban). **Tur 2 eğitilmiş kontrol noktası istiyor (K6).**
- [x] **H — Gate'leri ödül sinyaline dönüştürmek.** Mekanizma kurulu: geçen
      çıktılar 2. tura müfredat satırı, geçmeyenler neden etiketiyle negatif
      havuza. **Ölçülmeyen:** gerçek bir koşuda ödül şekillendirmenin etkisi.
- [ ] **I — Retrieval'ın kendisini güçlendirmek.** `crates/index` BM25 + satır
      düzeyinde alıntı var, ama **getirme kalitesi ölçülmüyor**. İş: sınav seti
      üzerinde getirme@k ölçen `training/getirme_olcumu.py` + kapı; BM25
      parametreleri ölçülmeden değiştirilmeyecek.
- [x] **J — Donanım ve verimlilik.** `bench_hardware.py` + `recommend_model_size.py`;
      kalıcı tavan owner donanımında (K6). Sandbox tavanının 105 kat altında
      kalındığı spec'te kayıtlı.
- [x] **K — Dağıtık / işbirlikçi eğitim. KAPSAM DIŞI (şimdilik).** K5/K6:
      geçiş değeri zkVM içerik kanıtı canlıya çıkana dek 2; hesaplama owner
      donanımı. Operatör havuzu bu turun kapsamı dışında.
- [x] **L — "Küçük modellerle kapışma" ölçütü.** `kiyas_sinifi.py` + kapı 56:
      parametre 924.288, sınır SmolLM2-135M'nin altında, ikinci akran
      Qwen3-0.6B; kapışma iddiası **yalnız görev ekseninde**, parametre
      ekseninde iddia kaydı kapı reddediyor.
- [x] **M — Kırmızı takım ve kapsam disiplini.** `kirmizi-senaryolar` kapısı +
      kapsam reddi; genişletmesi Y maddesinde.
- [ ] **N — Çok dillilik: TR/EN kombinasyonu.** Ölçülmüyor. İş: korpus üzerinde
      TR ve EN metinler için **jeton/karakter** oranını ölçmek (sözlük Türkçe
      ağırlıklı kesildi); fark büyükse EN ağırlıklı kayıtların bütçe maliyeti
      beyan edilmeli.
- [~] **O — Sürümleme, provenance, zincir kaydı.** Provenance ve digest
      zorunlu (`corpus-records-carry-provenance`, `provenance-fails-closed`),
      `asset_id` hesaplanıyor ama **`asset_id_pending` her kayıtta dolu** —
      zincir çıpası akış 1/3'te, kapsam dışı.
- [x] **P — Topluluk / pollen kaynaklı veri büyümesi. KAPSAM DIŞI.** K3: büyüme
      yalnız reponun kendi geliştirmesi + `doc` ile kabul edilmiş kapalı
      lisanslı belgeler, provenance kaydıyla. Aday listesi Awesome bölümünde.
- [x] **R — Ölçüm ve izleme sistematiği.** `training/ratchet.json` 7 anahtar,
      `ratchet-holds` kapısı, `findings.py`, `lubot olc`/`lubot durum`.
- [ ] **S — Uzun ufuk / spekülatif yönler.** Bilerek açık: bu turun işi değil.
- [x] **T — Karar modeli doktrini.** `crates/tomurcuk`: üç kapalı çıktı şekli,
      üretim yüzeyi yok (kapı 54), sabit kademe sırası, k-of-n; `lubot karar`.
- [~] **U — Hız ve birim maliyet.** Gecikme ölçüldü (kapı 59: 50 koşu, medyan
      2.237 ms, süreç başlatma dahil ve kayıtta öyle yazıyor). **Ölçülmeyen:**
      karar başına enerji ve tekrarlı soruda marjinal maliyet (donanım düzeneği
      yok).
- [~] **V — Yerel-first çıkarım yığınını derinleştirmek.** İlk yarı kapandı:
      `training/ilk_cevap_gecikme.py` + kapı `first-answer-latency-is-recorded`
      — `lubot ask` **30 kez soğuk** koşuldu (bu yolda sıcak bileşen yok: her
      çağrı süreç başlatma + korpus ayrıştırma ödüyor). Medyan ve aralık
      `training/eval/sonuclar/ilk-cevap-gecikme-2026-09-24.json` dosyasında;
      `uyari` alanı kapı tarafından zorunlu, beş kanaryalı self-test var.
      Kapı-59 disiplini: eksen beyan edilir, sayı eşik yapılmaz, ratchet'e
      konmaz. Kalan ikinci yarı: tekrarlı sorunun marjinal maliyeti (W) ve
      yolun derinleşmesi (yerel skorlayıcılar, kontrol noktası K6'da).
- [ ] **W — Karar ve cevap önbelleklemesi.** İş: aynı sorunun 2. kez sorulduğunda
      marjinal maliyetin ölçülmesi; U maddesinin "tekrarlı soruda maliyet → 0"
      iddiası ancak bununla ölçülebilir.
- [x] **X — Dış veri bağlayıcıları. KAPSAM DIŞI.** K2: korpus yalnız budlum
      yüzeyi; dışarıdan veri yok. Giriş kapısı deseni `doc` kabulüyle sınırlı.
- [ ] **Y — Kırmızı takım çalışmalarını genişletmek.** İş: enjeksiyon bataryası
      (aşağıda Awesome Prompt Injection maddesiyle aynı iş) + ölçülen red oranı.
- [x] **Z — Adı konmuş değerlendirme bataryaları.** `training/soru-bataryasi.json`
      + `soru-bataryasi-gecerli` + held-out sınav seti (12 soru, kapı 57).
- [x] **AA — Rakip küçük modellerle kapışma protokolü.** L ile aynı mekanizma;
      **ölçüm 8c'de**, eğitilmiş kontrol noktası yokken yapılamaz.
- [x] **BB — Eğitim aşamalarını adımlara bölmek.** Bu dosya + NN §8 iskeleti.
- [x] **CC — Sonsuz döngü tasarımı.** Süreç notlarında: bitiş çizgisi yok,
      ratchet yalnız yükselir, her tur bir ölçüm bırakır.
- [x] **DD — Zincir entegrasyonu ve model kaydı. KAPSAM DIŞI.** Akış 2 dışındaki
      akışlar bu repoda yalnız not olarak durur, PR'a girmez.
- [~] **EE — Öngörülebilir tuzaklar ve karşı önlemler.** Birçoğu kapıya dönüştü
      (`corpus-build-is-deterministic`, `tokenizer-vocab-is-frozen`,
      `mup-measurement-reproduced`, `measurements-do-not-feed-back`). OO
      ek riskleri aşağıda.
- [ ] **FF — Taze fikirler.** Somutlaştırılacak üç tanesi: (1) cevap
      okunabilirlik/tutarlılık skoru düşükse reddeden mekanik kural;
      (2) zaman damgalı müfredat satırları — Budlum terminolojisi yeniden
      adlandırıldığı için "bu karar hangi tarihte hangi isimle geçerliydi";
      (3) "bulunamadı" yerine **hangi veri eksik** bilgisini döndüren cevap
      biçimi. Kalanlar (NFT oranı izleme, topluluk oylaması) akış 1/3 ve K3
      kapsamında, bu turun dışında.

## Genişletme promptu GG–RR (lubot-egitim-genisletme-promptu.md)

- [x] **GG — Gerçekçi ölçek sınıfı kalibrasyonu.** `kiyas_sinifi.py` + kapı 56;
      iki eksen ayrıldı: parametre-eşleneği (ham dil) ve **görev-eşleneği**
      (Budlum-alanı alıntı doğruluğu + red disiplini). Kapışma iddiası yalnız
      ikincisinde.
- [ ] **HH — Veri-sınırlı ön-eğitim bilimi.** İş: düzenlileştirme
      ablasyonları (weight_decay, dropout) — **eğitim koşusu gerektiriyor**,
      o yüzden `olculmeyen` listesinde duruyor.
- [~] **II — μP / hiperparametre transferi.** `mup_olcum.py` + kapı
      `mup-measurement-reproduced`: dikkat ölçeği oranı **0.497** (beklenen
      0.500), bağlı readout sapması **0.0124** (bant 0.10). **Ölçülmeyen:**
      akışı RMS 1.231→5.440 (4.418×), `theta_1_bandinda=false`.
- [ ] **JJ — Rust'ın kendi sözdizim ağacını korpus inşasında kullanmak.** Ölçüm
      bunu gerekli kılıyor: kayıt uzunluğu p99 **546**, en uzun **2581** jeton;
      spec bu kayıtların AST-farkında parçalamaya kalacağını söylüyor. İş:
      parçalayıcı + parçaların `content_id` izini koruması.
- [x] **KK — Derleyiciyi ve test takımını hakem olarak kullanmak.** Veri
      karışımında `derleyici-hakem` katmanı **433 satır**; `epoch_ledger`
      fail-closed.
- [x] **LL — Çoklu-konsensüs metaforunu modelin doğrulama katmanına taşımak.**
      `crates/tomurcuk` k-of-n (bağımsız başlatılmış başlıklar); zincirin
      operatör eşiğiyle bilerek karıştırılmıyor.
- [~] **MM — Mühendislik iskeleti ile veri arasındaki ayrım.** `bulgu_mufredat_isareti`
      kayıtlı: `make_sft.py` `kind="curriculum"` yazıyor, `format.jsonl`'deki
      `kind:"negative"` işareti aşağı akışta kayboluyor (sızıntı değil, operatör
      kararı).
- [x] **NN — İlk sıfırdan koşu için başlangıç defteri.** Bu dosyanın NN §8
      bölümü; 8 adım, ölçümleriyle.
- [ ] **OO — EE'ye ek riskler.** İş: her risk için ya bir kapı ya bir
      `olculmeyen` kaydı. Şu an kısmen kapılarda, kısmen dağınık.
- [x] **PP — Değerlendirme setinin sızmasını fiziksel olarak imkânsız kılmak.**
      Üç duvar: sınav seti damgalıyor (`sinav.py`, kapı 57), üretici eliyor
      (`make_sft.py` `dropped_eval_only`), okuyucu reddediyor (`eval_sft`,
      `eval-set-never-trained`). Ölçülen: 1701 grounded satır, damgalı 0.
- [ ] **QQ — Görsel/diyagram okuma (girdi, üretim değil).** Durum ölçüldü:
      korpus kayıtlarının tamamı `kind: markdown`, görsel kayıt yok. İş: görsel
      varlıkların korpusa hangi şemayla gireceğine karar vermek + PDF dışı
      görsel metin çıkarımı. Üretim tarafı kapsam dışı (`reads-not-generates`).
- [ ] **RR — "Ölçülmedi" yanıtlarını bilgi-boşluğu haritasına çevirmek.**
      `olculmeyen` alanları kayıtlarda duruyor ama **tek bir haritada
      toplanmıyor**. İş: tüm `olculmeyen` girdilerini toplayıp tek dosyaya
      yazan betik + kapı.

## Awesome listesinden lubot'a düşenler (budlum-awesome-eslestirme.md)

Kaynak bir eşleştirme listesi; burada yalnız **lubot'a değen** maddeler var ve
her biri somut bir işe çevrildi. K1 gereği bunlar **yöntem ilhamı**: hiçbir
liste, crate ya da veri içe alınmayacak. budlum/seed'e ait bölümler (konsensüs,
QR/codec, merkeziyetsiz depolama, site/WASM, k8s/terraform) bilerek alınmadı.

- [ ] **Bilgi getirimi (Awesome Information Retrieval).** Kendi arama motorunu
      geliştirme literatürü; lubot'un BM25 + alıntı hattının tam karşılığı. İş:
      I maddesindeki getirme@k ölçümü.
- [ ] **Soru-cevap (Awesome QA).** `ask`/`batch` akışının literatürü. İş: sınav
      bataryasını soru tipi bazında genişletmek ve her tipin skorunu ayrı
      raporlamak (tek ortalama skoru tip bazlı gerilemeyi gizler).
- [ ] **Açıklanabilirlik (Awesome XAI).** "Her cümlenin kaynağını göster"
      ilkesinin ölçülebilir hali. İş: üretilen cevapta **alıntısız cümle
      oranını** ölçen betik + eşik aşımında red.
- [ ] **İstem enjeksiyonu (Awesome Prompt Injection).** Fail-closed/kapsam reddi
      tasarımı için tehdit kataloğu. İş: enjeksiyon bataryası + ölçülen red
      oranı; `kirmizi-senaryolar` kapısının genişletilmiş hali (Y maddesi).
- [ ] **Belirsiz girdi / fuzzing (Awesome Fuzzing).** Yeni bağımlılık eklemek
      yok: ayrıştırıcılar (`lubot-jeton` sözlük yükleyici, `lubot-read` korpus
      yükleyici, çıktı şema doğrulayıcı) için **tekrarlanabilir tohumlu**
      düşmanca girdi bataryası. Ölçülen: panik/refus sayısı; `no-panic-path`
      kapısıyla aynı çizgide.
- [ ] **Kimlik bilgisi biçimleri (Awesome Password Cracking).** Saldırgan
      tarafı: `no-secret-material` kapısının tarayıcısının bilinen biçimlere
      karşı **yakalama oranı** ölçülecek (şu an oran ölçülmüyor).
- [ ] **IAM / OpenID Connect kıyası (Awesome IAM).** `crates/grant` grant
      defterinin standart IAM desenleriyle karşılaştırması. İş: `docs/GRANT-KIYAS.md`
      — kod değil, karar kaydı.
- [ ] **Düzenli ifadeler (Awesome Regex).** Kimlik bilgisi tarayıcısının ve lint
      kurallarının desen denetimi; yanlış-pozitif/negatif ölçümü.
- [ ] **CI/CD saldırıları (Awesome CI/CD Attacks).** `it` komutunun kısıtlı
      push tasarımı tedarik zinciri kaygısıyla aynı yere bakıyor. İş: tehdit
      notu + gerekiyorsa kapı.
- [x] **SECURITY.md yazmak (Awesome AppSec / Security).** Yazıldı:
      `SECURITY.md` — özel raporlama yolu (GitHub private advisory, yedeği
      kör kamu kaydı), kapsam bu kod tabanının ölçülen tehdit yüzeyiyle
      tanımlı (grant bypass, şema bypass, eval sızıntısı, `it` allowlist
      kaçışı, kapalı listedeki tarama kaçağı; debug keystore'un yetkisizliği
      beyanlı kapsam dışı). SLA uydurulmadı: ölçülmemiş taahhüt yazılmadı,
      \"güvenlik raporları özellik işinin önüne alınır\" hükmü duruyor.
- [x] **ARCHITECTURE.md yazmak (Awesome Software Architecture).** Yazıldı:
      `docs/ARCHITECTURE.md` — katman haritası (ilkel → birleşik → okuma →
      giriş), okuma döngüsünün sırası, eğitim yolu, doğrulama otoritesi,
      Android kabuğu ve sınırlar. Rakam taşımıyor; ölçülen tablonun
      `docs/CRATES.md` olduğunu beyan ediyor (korpus geri-besleme kuralı).
      \"Saparsa kapı düşmeli\" yarısı da mekanikleşti: kapı
      `architecture-doc-tracks-layer-rule` belgenin 25 crate'in tamamını ve
      katman kuralının kendi kelimelerini (`never above`, `topological`)
      taşımasını zorluyor; kanarya: belgeyi silmek ya da bir crate adını
      düşürmek kapıyı düşürüyor (self-test'te).
- [ ] **README turu (Awesome README / Translations).** Ölçüldü: kaynak listede
      "README.tr.md zaten var" deniyor ama bu repoda **README.tr.md yok**. İş:
      ya yazmak ya da listeyi düzeltmek; ikisi de karar gerektiriyor.
- [ ] **Markdown şeması (Awesome Markdown / NLG).** `ai-output-schema-enforced`
      var; iş şemanın kurallarını tek tek sayıp **kapsanmayan durum** kalıp
      kalmadığını ölçmek.
- [ ] **Lisansı temiz kaynak adayları (Awesome Uncopyright / Public Datasets).**
      K3 kapsamında: kamu malı/lisansı temiz kaynakların **aday listesi**,
      provenance ve lisans alanlarıyla. Karar operatörün; bu madde yalnız
      listeyi ve alanları hazırlar.
- [ ] **Yanlış inanışlar denetimi (Awesome Falsehood).** Zaman, isim, para ve
      kodlama hakkında. İş: lubot'ta ısıranlara test — Türkçe büyük/küçük harf
      dönüşümü ve Unicode normalizasyonu jetonlayıcıyı doğrudan etkiliyor
      (`crates/jeton`'daki sınıf ayrımı).
- [ ] **Kanıta dayalı mühendislik (Awesome Empirical Software Engineering).**
      Ratchet felsefesinin yöntem notu: ölçülen taban yalnız yükselir, iddia
      ölçümle taşınır. İş: kısa bir yöntem belgesi + mevcut kapılarla eşleme.
- [x] **Statik analiz (Awesome Static Analysis). KARAR: clippy çizgisi.**
      `-D warnings` + `-W pedantic` ratchet'te 0'da; yeni lint eklemek serbest,
      `#[allow]` ile gevşetmek yasak. Ek araç eklemek CI'yi zayıflatma riski
      taşıdığı için alınmadı.
- [x] **Üretken AI / sohbet listeleri (Awesome Generative AI, Conversational AI,
      ChatGPT, AGI). YALNIZ BAĞLAM.** Lubot bilerek üretmiyor; çizgi
      `reads-not-generates` ve `no-generation-variant` kapılarıyla çizilmiş
      durumda. Bu listeler rakip/emsal taramasıdır, iş üretmez.
- [x] **Ses ve görüntü listeleri (Awesome Whisper, VLM, Computer Vision).
      KISMEN KAPSAM DIŞI.** Lubot bir okuma istemcisi: PDF metin çıkarımı var
      (`doc-pdf-feeds-corpus`), ses girdisi kapsam dışı. Görsel **girdi** QQ
      maddesinde ayrıca duruyor.

## Android arayüzü (APK)

- [x] **JNI köprüsü** — `crates/arayuz`, 367 satır, 4 test. Kendi mantığı yok:
      CLI'ın koştuğu `ask` yolunu cihaza açıyor. Dört dışa aktarım: `kurulus`,
      `soru`, `belgeEkle`, `surum`. Hata yutulmuyor, `HATA:` önekiyle Java'ya
      dönüyor ve arayüz onu olduğu gibi gösteriyor.
- [x] **Sade arayüz** — `android/`: tek ekran, AndroidX yok, framework
      bileşenleriyle. Soru kutusu, iki düğme (Sor / Belge al), kaydırılabilir
      ve seçilebilir cevap alanı, durum satırı. Ağ izni yok: Lubot cihaz
      dışına çıkmıyor, korpus cihazda.
- [x] **Cihazdan içerik alma** — `ACTION_OPEN_DOCUMENT` ile seçilen metin
      okunuyor ve izole kayda ekleniyor. **K2 korunuyor:** cihaz belgesi
      korpusa karışmıyor; `source: cihaz`, `licence: kullanici-girdisi` ile
      ayrı dosyada duruyor, alıntı hangi kayda ait olduğunu kaybetmiyor.
- [x] **Gradle'siz APK derlemesi** — `android/derle.sh`: aapt2 → javac
      (UTF-8) → d8 → zip → zipalign → apksigner. AGP 8 Java 17 istiyor,
      zincirde Java 11 var; o yüzden Android'in kendi araçları kullanılıyor.
      Betik paketin içeriğini doğrulamadan "bitti" demiyor: dex, `.so`,
      korpus, manifest ve arsc pakette mi diye bakıyor.
      Ölçülen çıktı: **784 KB, imzalı, arm64-v8a**, Rust cdylib 1.4 MB.
- [ ] **Cihazda çalışma doğrulaması.** Burada ölçülemedi: sandbox'ta cihaz ya
      da emülatör yok. APK imzalı ve içeriği doğrulanmış, ama JNI çağrısının
      gerçek bir cihazda cevap döndürdüğü **ölçülmedi**.
- [ ] **Diğer ABI'ler.** Yalnız `arm64-v8a` derlendi; `armeabi-v7a` ve
      `x86_64` (emülatör) betikte parametre olarak durmuyor.
- [ ] **OPERATÖR KARARI — cihazdan gelen belge kalıcı korpusa girebilir mi?**
      Şu an girmiyor, izole duruyor. K2 korpusun budlum yüzeyi olduğunu
      söylüyor; cihazdan gelen içerik dışarıdan geliyor. Kalıcı kabul K3'ün
      `doc` yolunu ve lisans/provenans kaydını gerektirir. Karar operatörün;
      bu madde o kararı bekliyor, kod o kararı vermeden ilerlemiyor.

## Operatör kararı bekleyen bulgular (kod değiştirilmedi)

- [ ] **`bulgu_veri_butcesi`.** Spec'in 1.94 token/param beyanı **yüzey**
      korpusuna (1.791.712 jeton) ait; CI'ın kurduğu **self** korpusuyla ölçülen
      0.1024 — ~19 kat fark. Hangi korpusun eğitileceği operatör kararı.
- [ ] **`bulgu_mufredat_isareti`.** `training/curriculum/format.jsonl` içinde 2
      satır `kind: "negative"` taşıyor; `make_sft.py:84` damgayı koşulsuz eziyor.
      Veri sızıntısı değil (geçersiz Markdown `user` turunda, cevap doğru bir
      red); kaybolan şey işaretin kendisi. Hangi işaretin korunacağı operatör
      kararı.

## Operatör kararları (uygulandı)

**2026-09-24 — Süreç belgeleri cevap yüzeyinden damgalı.** Ölçülen arıza:
"ilk eğitim spec'i nedir, hangi parametreler?" sorusu bu dosyanın iş
sırası bölümlerinden üç dev pasaj döküyordu (gerçek cevap korpusun
içindeyken). Karar: plan/süreç belgeleri korpustan **çıkarılmaz**,
damgalanır (`served: false`) — arşiv ve ciro bütünlüğü korunur (kayıt
sayısı gerilemez), ama damgalı kayıt ne aranabilir ne alıntılanabilir.

- `training/servis-politikasi.json` tek gerçek listedir; builder
  fail-closed (politika yok/bozuk/hiçbir kayda dokunmuyor → kurulum reddi).
- Zincir: builder damgası → `LoadedCorpus.served_ids()` → `ara`/`ask`
  aynı id listesinden indekslenir; `lubot corpus` servis-dışı sayısını basar.
- Kanarya kapısı `unserved-records-never-cited` damganın kendisinin
  dışladığını ispatlar: damgalı kanarya ASLA alıntılanamaz; aynı kanarya
  damgasız bulunur (kör taramayı yeşil gösteremez). Kapı sayısı 68.
- Ölçüm: 2777 kayıttan 30'u damgalı; aynı spec sorusu artık
  `crates/egitim`'den alıntılı cevap veriyor (lubot-a1-derin-dar, d_model 64,
  8 katman, 2 başlık, 924.288 parametre).
- Geri dönüş: politika girdisini kaldır + korpus kurulumunu yenile —
  tek satırlık veri değişikliği, kalıcı out-of-band itiraz kalmaz.
- Eşlik eden bulgu ve düzeltme: damga sonrası Türkçe soru kelimeleri
  (`hangi`, `neden`...) içerik terimi sayılıp kapsama tabanını düşürüyor
  ve sorgu çoğulları tekil geçişleri kaçırıyordu
  (`parametreler`/`parametre`). Stoplist genişledi ve paylaşılan-kök
  eşleşmesi eklendi (yanlış-pozitif duvarı testli:
  `sistematik`/`sistemimiz` eşleşmez).
- Kapsam notu: TUR5'teki kabataslak `model egit/oku` denemesi eğitim-kosu +
  çikarim yüzeyi (LUBOTCKPT v1, resume kimliği, cache kanıtlı) karşısında
  GERİDE kaldığı için dahil edilmedi — üstün uygulama korundu.

## Süreç notları

- [ ] **`lubot ratchet --set` yedi anahtarın dördünü yazıyor (bulgu, bu turda
      yakalandı).** `crates/cli`'deki `ratchet::Measured` 4 alan ölçüyor
      (tests/gates/pedantic/corpus) ve `as_baseline()` tablonun tamamıymış
      gibi yazıyor; `gates/check.py` tarafının `RATCHET_KEYS`'i 7 anahtar
      (tokens, bootstrap, exam Python tarafında yaşıyor). Sonuç: `--set`
      koşusu `tokens`/`bootstrap`/`exam` satırlarını düşürüyor ve bir
      sonraki `ratchet-holds` `ratchet baseline lost` ile düşüyor —
      doğuştan kırmızı bir "tabanı yenile" yolu. Bu turda uydurma sayıyla
      değil, sahiplerinin ölçümüyle (`egitim_butcesi.py --olc` + sonuç
      dizini ve sınav seti sayımı) birleştirilerek onarıldı; iki tarafın
      tek anahtar listesine bağlanması işi açık. Koruma yönü mevcut: kayıp
      anahtar taban güncellemesi kapıda reddediliyor.

- **Doğrulama CI'ın sabitlediği toolchain ile yapılır, en yenisiyle değil
  (ölçüldü: bu turda CI koşusu 50 bu yüzden düştü).** CI `rustup default
  1.88.0` kuruyor; yerelde 1.98.1 ile koşunca clippy temiz görünüyordu, ama
  1.88'in clippy'si `crates/egitim`'deki bir test mesajında
  `uninlined_format_args` istiyordu ve "Lints are errors" adımı düştü —
  ardından test, kapı ve korpus adımları hiç koşmadı. Kural: doğrulamadan
  önce `rustc --version`'ı `.github/workflows/ci.yml`'deki sürümle
  karşılaştır; gate'ler de `cargo`'yu çağırdığı için `rustup default`
  sabitlenen sürüme çekilmeli, yalnız `cargo +sürüm` yetmiyor.

- **Ortam toolchain'i periyodik siliyor (ölçüldü, bu turda iki kez).**
  `~/.cargo` ve `~/.rustup` oturum ortasında kayboldu; `.cargo/registry/src`
  yarım kalınca hata "jetonlayıcılar uyuşmuyor" gibi değil,
  `couldn't read .../rustversion-1.0.23/build/build.rs` olarak görünüyor.
  Kural: doğrulamadan önce `cargo --version` çalıştır, çalışmıyorsa kurulumu
  ve tüm doğrulamayı **tek çağrıda** yap; registry yarım silindiyse
  `rm -rf ~/.cargo/registry/src` yeterli, ağ varsa cargo yeniden açıyor.
  Bir kapının `could not run: FileNotFoundError: 'cargo'` demesi kapının
  kırık olduğu anlamına gelmez — araç eksiktir.

- [x] **CI sırasını yerelde birebir koşmak.** `Gate self-tests` korpus
      kurulmadan **önce** koşar; korpus isteyen bir self-test yerelde geçer,
      CI'da düşer (run 39'un kök nedeni). Artık her turda self-testler korpus
      silinmiş hâlde de koşuluyor.
- [x] **`concurrency.cancel-in-progress: true`.** Ana dal dışındaki bir dala
      push, kuyruktaki koşuyu iptal eder (run 40 böyle iptal oldu). Bir koşunun
      sonucu gerekiyorsa push'tan önce beklenir.
