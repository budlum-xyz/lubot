# Veri kabul kararı ve kanıt zinciri

Tarih: 2026-09-24. Dayanak: operatör talimatı ("ihtiyacın olan ne varsa direkt
al", "bu repolarda lazım ne varsa Lubot'un parametrelerine işle, repo ismi ya da
lisans vs belirtmiyoruz", "Lubot her türden veriyi okuyabilmeli inceleyebilmeli
ama kullanıcıya md formatı dışında bir çıktı sunamaz").

## Ne değişti

| Konu | Önce | Şimdi |
|---|---|---|
| Eğitim verisi kaynağı (K2) | yalnız kendi repo ağacı | kendi ağaç **+** sahibine ait ya da kamu malı sınıfındaki dış kaynaklar |
| Lisans kümesi | MIT, Apache-2.0, PolyForm-Shield-1.0.0 | aynı küme **+** `kamu-mali` sınıfı |
| Model yüzeyi (no-generation) | okuyan, üretmeyen | her türden veriyi **okuyabilir ve inceleyebilir**; kullanıcıya sunduğu çıktı **yalnız şema doğrulamalı Markdown** |
| Kaynak adı | korpus kaydında kaynak adı geçer | **ne kaynak adı ne lisans adı** depo ağacına yazılır |

Anayasa değişikliği `autonomous-training/INVARIANTS.md` içinde yapıldı; mühür
`INVARIANTS.sha256` yenilendi (damga `187b5b6f190df5ef…`). Kapı
`invariants-are-frozen` mührü her koşuda yeniden hesaplar.

## Kanıt zinciri: adı yazmadan doğrulanabilirlik

Kaynak **adı** depoda yok ama kaynak **kanıtı** var:

1. **Alım manifesti** (`/home/user/kamu-kaynak.json`, depo dışı) kaynak adını,
   beklenen lisans sınıfını ve sabitlenmiş revizyonu taşır.
2. **Lisans, indirmeden önce** kaynağın kendi metadata kaydından okunur
   (`training/kamu_verisi.py::lisans_dogrula`). İzinli sınıfta değilse alım
   orada durur; hiçbir dosya inmez. Kapı değil, **kod yolu** bunu yapar:
   doğrulama geçmeyen kaynağın baytı diske düşmez.
3. **Her kayıt** korpus kaydı biçimindedir: `licence` sınıfı, `attribution`,
   `content_id` (metnin sha256'sı), `asset_id` (kaynağın *kimlik özeti*: ad +
   revizyonun sha256'sının ilk 16 hanesi), `digest`, `path` (`kamu/<özet>/…`).
4. **Alım kaydı** `training/eval/sonuclar/kamu-veri-2026-09-24.json`: kaynak
   başına kimlik özeti, revizyon, indirilen bayt, kayıt sayısı ve **dosya
   özetleri**. Adı bilen biri manifestle eşleştirip her dosyayı doğrulayabilir;
   adı bilmeyen yalnızca "şu özetli dosyalar alınmış" bilgisini görür.
5. **Tekrar üretilebilirlik**: indirilen ham dosyalar `/var/tmp/lubot-veri`
   önbelleğinde durur (commit'e girmez). Aynı revizyonlar için ikinci koşu ağdan
   okumaz; `python3 training/corpus_insa.py` korpusu baştan kurar.

## Alımın ölçüsü (bu koşu)

| Ölçü | Değer |
|---|---|
| Kaynak sayısı | 9 (hepsinin lisansı indirmeden önce doğrulandı) |
| İndirilen kayıt | 38.156 |
| Korpus toplamı | 41.178 kayıt (3.022 kendi ağaç + 38.156 kamu malı) |
| Metin bütçesi | 30.000.000 karakter hedef; adil pay ile kaynak başına ~3,3 M |
| Süre | 30,3 s (önbellekli koşuda ~7 s) |

## Arındırma kuralı (ölçülerek öğrenildi)

İlk eğitim koşusunun çıktısı, veri setlerinin **provenans başlıklarını** ezberledi:
model `provenance: ...`, `generator_sha256: ...` satırlarını üretiyordu. Yani
kaynak adı, verinin *içinden* çıkıp **kullanıcıya dönük çıktıya** sızıyordu —
operatör kararı bunu yasaklar ("repo ismi ya da lisans belirtmiyoruz") ve bu,
ölçülmeden görülemeyecek bir sızıntıdır.

Düzeltme iki kural olarak kodda durur (`training/kamu_verisi.py`):

1. **Süslemeli alanlar düşer**: kimlik/ayrım/katalog/provenans alanları
   (`id`, `split`, `source_type`, `provenance`, `generator*`, `license`,
   `citation`, `url`, `author`, `publisher`, …) korpusa girmez.
2. **Adres taşıyan satır düşer**: satırda bir kaynak adresi kalıbı geçiyorsa
   (`huggingface.co`, `github.com`, `doi.org`, …) o satır alınmaz.

Ölçüm: arındırılmış veri dosyasında 57.564 kaydın **0**'ında ad/sızıntı kalıbı
var (kural: `huggingface.co|github.com|GoktugD|provenance:|generator_`).
Bu düzeltme, veri hacmini 38.156 → 57.564 kayda çıkardı: süsleme satırları
düşünce satır bazlı kayıt sınırları değişti ve gövde metni daha çok yer buldu.

## Sınırlar (kendini kandırmama notları)

- **Adil pay** her kaynağa eşit yer verir; bu bir *çeşitlilik* kararıdır, veri
  kalitesi iddiası değil. Kaynak başına ~3,3 M karakter, büyük bir kaynağın
  tamamını almadığımız anlamına gelir.
- **Dil**: alınan veri çoğunlukla Türkçe; IRC kaynağı İngilizce ağırlıklı.
  Modelin dil dağılımı bu karışımı yansıtır, ayrı bir dengeleme yapılmadı.
- **Kalite süzgeci yok**: kayıtlar geldiği gibi alındı (tekilleştirme dışında).
  Bu bir "temiz korpus" iddiası değildir; ölçüm `egitim-veri` komutunun
  raporunda durur.
- **K2 genişlemesi** izinli kümeyi sınıf bazında açar, kaynak bazında değil:
  yeni bir kaynak eklemek manifest değişikliğidir ve doğrulama yine kod yolunda
  çalışır.
