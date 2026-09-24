# Lubot Otonom Sürekli Eğitim Döngüsü

Bu dizin, insan müdahalesi olmadan çalışan sürekli eğitim otomasyonunu barındırır.
Direktifin dört katmanı burada uygulanır; kabul edilen kararlar ve gerekçeleri
aşağıdadır. Kararların sayısal hâli `ayarlar.json`'dadır, gerekçesi bu dosyadır:
ikisi çelişirse **`ayarlar.json` bağlayıcıdır**.

## 0. Özet

| katman | ne yapar | nerede |
| --- | --- | --- |
| 1 — deneysel döngü | sabit süreli deney: öneri üret, eğit, ölç, tut ya da at | `dongu.py --tek-tur` / `--kos` |
| 2 — kalıcı hafıza | ideation + experimentation kaydı, oturumlar arası taşınır | `hafiza/ideation.jsonl`, `hafiza/experimentation.jsonl` |
| 3 — kendini geliştirme | yalnız **kendi arama politikasını** değiştirir; her değişiklik loglu ve geri alınabilir | `kendini-gelistirme.json` + `kendini_gelistir()` |
| 4 — genişletilmiş keşif | veri işleme/biçim denemeleri; ikincil ve isteğe bağlı; K2 sabit | bu sürümde **kapalı** (aşağıda §7) |
| **danışma (K7)** | belirsizlik bandındaki tut/at kararına **oy** verir; karar kuralı kodda kalır, oy yoksa insan | `danisma.py` + `training/danisma/` servisi |

Yön (hangi soruyu kovaladığımız) insan yazımıdır: `program.md`. "Hangi yönü
araştırayım" diye karar veren bir katman bilinçli olarak **yoktur** (AI Scientist
tarzı yön seçimi K1 kapsamı dışında bırakıldı).

## 1. Dokunulmazlar ve anayasa kilidi

`INVARIANTS.md` donmuş anayasadır: K1–K6 + no-generation, durdurma koşulları
(S1–S5), dokunulmaz dosya listesi. `INVARIANTS.sha256` bu dosyanın damgasıdır.

- `anayasa_kilidi()` her komutun ilk adımında damgayı doğrular; damga tutmazsa
  döngü hiç başlamaz.
- `yazma_reddi(yol)` yazma yolunu dokunulmazlara karşı denetler: `INVARIANTS.md`,
  `INVARIANTS.sha256`, `ayarlar.json`, `program.md`, `olcut.md`.
  Bu dosyalara yazma girişimi tek başına **S2** durdurma koşuludur.
- Anayasa değişikliği otomasyonun yetkisinde değildir: yeni sürüm ancak insan
  eliyle, damga yeniden üretilerek yazılır.

Kapılar: `invariants-are-frozen` (damga + makine bloğu + "beyan edilen her durdurma
koşulu kodda `dur()` ile çağrılıyor mu" + dokunulmazlara yazma taraması + kanarya),
`mutation-surface-is-closed` (mutasyon alanı ∩ dokunulmazlar = ∅; düğmeler ikilinin
gerçekten kabul ettiği bayraklar) ve `credential-shapes-are-measured` (bu ağacın
ölçüm betiği için; bkz. §12). Süit toplamı 84 kapı.

## 2. Ölçüt (`olcut.md` bağlayıcıdır)

- Metrik: en iyi doğrulama kaybı, yön: düşük iyi, asgari iyileşme **0.001**.
- Oturum başında **tekrarlanabilirlik kapısı**: aynı yapılandırma iki kez koşar ve
  aynı sayıyı vermek zorundadır; vermezse ölçüm geçersiz sayılır ve DURULUR (S1).
  Aynı tohum + aynı veri sırası ile koşular deterministiktir; bu kapı ölçüm
  düzeninin kendisini her oturumda sınar.
- Bir deney **tutulur** ancak: skor iyileşir **ve** tam doğrulama (fmt + clippy +
  test + tüm kapılar) kırmızısız geçer. Aksi hâlde atılır.

### Gerekçe: neden 0.001?
Doğrulama kaybı 6 ondalıkla yazılır ve koşular deterministik olduğu için küçük
farklar "gürültü" değildir; yine de bir taban eşik şart, çünkü 1e-6 mertebesindeki
farklar sayı biçimlendirmesinden gelir. 0.001, ölçülen skala ~8.4 üzerinde göreli
~1.2e-4 demektir: sayı biçimlendirmesinin çok üstünde, ama gerçek bir iyileşmeyi
kaçırmayacak kadar altında.

## 3. Deney bütçesi: 300 s, günde 40 deney, adım kalibrasyondan

Ölçülen hız: **≈0.80 adım/sn** (20 adım 11.2 s; 200 adım 249.7 s; 60 adım +
`--dogrulama-her 20` 86.5 s — bu makinede, tek süreç, 2 çekirdek).

- Adım sayısı elle yazılmaz: `kalibrasyon()` **iki noktalı prob** koşar — `dogrulama_her`
  adım ve onun beş katı — ve iki süreden hem sabit yükü (korpus yükleme + jetonlama)
  hem **marjinal adım maliyetini** (eğim) ayırır:
  `adim_butcesi = floor((sure_butcesi - yük) / eğim × 0.8)`, sonra `dogrulama_her`
  katına aşağı yuvarlanır. Sonuç `kosum/kalibrasyon.json`'da durur.
- Neden iki nokta? Ölçüldü: tek noktalı 20 adımlık prob 0.874 s/adım dedi, bütçe
  265 adım çıktı, gerçek koşu **399 s** sürdü (300 s bütçeye karşı: 1.33× aşım).
  Kısa koşu sabit yükle karışıyordu; iki nokta eğimiyle sabit yük ayrışır.
- Kalibrasyon **kapalı çevrimlidir**: gerçek bir koşu bütçeyi %10'dan fazla aşarsa
  `kosum/durum.json → butce_gozlemi`'ne yazılır ve sonraki oturumda eğim olarak
  `max(prob eğimi, gözlem eğimi)` kullanılır.
- Alt sınır `dogrulama_her`: doğrulama kadansından önce duran koşu hiç doğrulama
  kaybı üretmez, yani ölçülecek metrik kalmaz.
- Güvenlik payı 0.8: gözlenen hız dalgalanması ve doğrulama adımlarının ek maliyeti
  için. Bütçe değişirse kalibrasyon **bayatlar** (adım bütçesi bütçeden türetilir).
- `gunluk_deney_ust_siniri = 40`: 300 s × 40 ≈ 3.3 saat/gün eğitim. Gerekçe: aynı
  makinede kapılar (fmt/clippy/test/`gates --all`) ve insan incelemesi için yer
  kalmalı; ayrıca bir gün içinde üretilebilecek en fazla commit/PR sayısı insan
  denetlenebilir kalmalı.
- `tohum = 20260924`: sabit. Tohum **deneyin değişkeni olamaz** — karşılaştırmanın
  geçerlilik koşulu (bkz. `mutasyon_alani.json → kapsam_disi_dugmeler`).

## 4. Mutasyon alanı: 5 düğme, kapsam dışı 6 düğme

Düğmeler (`mutasyon_alani.json`): `--ogrenme-orani` [0.001, 0.05],
`--agirlik-sonumu` [0, 0.5], `--kirpma` [0.5, 2.0], `--isinma` [0, 200], `--yigin` [1, 4].

Gerekçe: bunlar `training/model_spec.json`'daki **donmuş mimari ekseni** içinde
kalan, karşılaştırmayı bozmayan hiperparametrelerdir. Mimariyi (`--pencere`,
`--vocab`) veya ölçüm düzenini (`--adim`, `--tohum`, `--dogrulama-her`), veri
kaynağını (`--corpus`) değiştirmek deneyi başka bir soruya çevirir; bunlar kapsam
dışıdır ve girişim S1/S2'ye düşer. Kaynak yaması yalnız şu dosyalara izinlidir:
`crates/egitim/src/`, `crates/cli/src/egitim_kosu.rs`, `training/make_sft.py` —
dokunulmazlara ya da kapsam dışı alana dokunan yama reddedilir.

## 5. Checkpoint, geri alma, regresyon

| karar | değer | gerekçe |
| --- | --- | --- |
| checkpoint sıklığı | her **5** tutmada | geri alma granülaritesi ≈ bir PR konsolidasyonu; checkpoint **durum + politika + manifest** taşır (model ağırlığı taşımaz), boyutu kilobayt mertebesindedir |
| ek checkpoint | kendini geliştirme **öncesi ve sonrası** | katman-3 politika değişikliği her zaman geri alınabilir olmalı |
| regresyon tutma eşiği | **0 kırmızı** | "iyileşme" ile "kırmızı array" aynı kayıtta tutulamaz: kapı/test kırmızısı olan bir tutma, ölçütü kirletir |
| art arda kırmızı | **2** → S4 durur | tek kırmızı ortam kaynaklı olabilir; ikincisi sistemik demektir |
| üst üste kötü pencere | **3** (pencere 3) → S5 durur | katman-3 politikasının işe yaramadığını en az 9 deneyle göstermek gerekir; daha erken durmak aramayı sığ yapar |

Geri alma: `--geri-al` son manifestli checkpoint'e döner ve bunu günlüğe yazar.
Manifest yalnız listelediği dosyaları geri yazar (kapsam dışı bir dosyaya dokunmak
"geri alma" değil gizli bir yazma olurdu).

### Koşu yükü saklanmaz (ölçüldü: 22,2 MB/koşu)
Her eğitim koşusu bir `ckpt.bin` üretir: ölçülen boyut 22.183.751 B. Bu bir **ara
ürün**, kanıt değil: `kayit.json` zaten `ckpt sha256 <özet>` satırını taşır ve
oturum başı tekrarlanabilirlik kapısı aynı imzanın aynı sayıyı verdiğini kanıtlar,
yani koşu gerektiğinde 300 saniyede yeniden üretilebilir. Günde 40 deneyde
saklansaydı ~890 MB/gün ederdi; bu yüzden metrik okunduktan sonra yük silinir
(`ckpt_yuku_silindi`), `.gitignore` de yolu kapsar.

## 6. Katman 3 uzayı: yalnız 5 sayı/alan

`kendini-gelistirme.json → arama_politikasi`:
`kesif_orani`, `adim_carpani`, `sogutma`, `oncelik_sirasi`, `geri_donme_esigi`.

Gerekçe: katman 3 **yalnız arama politikasını** değiştirebilir. Ölçüt, veri, donanım
sınırı, dokunulmazlar ve mutasyon alanının sınırları bu uzayın dışındadır; yani
"sistemi iyileştirmek" adına hedefi değiştirmek mümkün değildir. Her değişiklik
`gecmis` listesine (eski → yeni), `hafiza/ideation.jsonl`'a ve oturum günlüğüne
yazılır; `geri_donme_esigi` aşılırsa değişiklik geri alınır. Aralık **5 deney**,
pencere **3**.

## 7. Katman 4 (genişletilmiş keşif): bu sürümde kapalı

Direktif katman 4'ü ikincil ve isteğe bağlı bıraktı; veri işleme/biçim denemelerini
kapsıyordu, veri kaynağı ise K2 ile sabit (yalnız bu depo ağacı). Bu sürümde
katman 4 **devre dışıdır**: veri kaynağı donmuşken genişletilmiş keşfin ekleyebileceği
tek şey biçim gürültüsü olurdu. Açmak isteyen operatör önce K2'yi değiştirmek
zorundadır; bu ise otomasyonun yetkisinde değildir. Kod tarafında ayrı bir bayrak
açılmadı — yarım bir keşif kolu, ölçülmemiş bir iddia olurdu.

## 8. Kanıt, CI ve PR

- Kanıt tek otorite: **CI**. Yerel yeşil yeterli değildir.
- Tutulan deneyin kaydı `kanit_durumu: ci-bekliyor` ile yazılır; CI koşusu
  `--ci-onayla <RUN_ID>` ile işlenir ve durum `ci-onayli` olur. Commit SHA'sı
  commit sonrası kayda işlenir.
- Dal `egitim-dongusu`; **main'e doğrudan push yok**. PR konsolidasyonu her **5**
  tutmada: checkpoint kadansıyla aynı, böylece her PR'ın bir geri dönüş noktası var.
- Rapor sohbete değil dosyaya yazılır (quiet): `kosum/oturum.md` (günlük),
  `hafiza/*.jsonl` (kalıcı kayıt), `kosum/deney-*/rapor.md` (eğitim raporu).

## 9. Durdurma koşulları

| koşul | tetikleyici | davranış |
| --- | --- | --- |
| S1 | ölçüm geçersiz: tekrarlanabilirlik kapısı düştü, koşu düştü | DUR + `ask_user` onayı |
| S2 | dokunulmazlara/kurallara/değerlendirme tanımına yazma girişimi, politika anahtarı sınırı aşımı | DUR + onay |
| S3 | K6: spec parametre sayısı ölçülen donanım tavanını aştı | DUR + onay |
| S4 | art arda 2 deney tam doğrulamadan kırmızı | DUR + onay |
| S5 | katman 3: art arda 3 kötü pencere | DUR + onay |

DURUŞ `kosum/DURUS.json`'a yazılır; onay `--onayla "<gerekçe>"` ile verilir ve
gerekçe hafızaya işlenir. Otomasyon **kendiliğinden devam etmez**.

## 10. Donanım ve K6 kararı (2026-09-24, e2b.local)

Ölçüm: 2 çekirdek, 1.94 GiB RAM (2.081.390.592 B), boş disk 17.53 GiB, GPU yok,
python 3.13.14 (`training/bench_hardware.py` + `training/recommend_model_size.py`).

- Muhasebe: `train_fp32_adamw` — fp32 ağırlık + AdamW momentleri + aktivasyonlar.
- Kullanılabilir bütçe: RAM × 0.75 = 1.561.042.944 B → **tavan 97.565.184 parametre**.
- Bugünkü spec (`training/model_spec.json`): **924.288 parametre** — tavanın ~%0.95'i.
  Gerekçe: mimari insan kararıdır (S1 sınırı); otomasyon parametre sayısını
  büyüterek "iyileşme" üretemez, yalnız mevcut eksende arama yapar.
- Denetim her oturum başında `k6_kontrol()` ile koşar ve `kosum/k6.json`'a yazılır;
  parametre sayısı `params.toplam` alanından okunur ve grup toplamıyla tutarlılığı
  denetlenir (tek bir literal yetmez).

## 11. Nasıl koşulur

```bash
python3 autonomous-training/dongu.py --kendini-test      # durdurma kanaryaları
python3 autonomous-training/dongu.py --durum              # durum özeti (dosyadan)
python3 autonomous-training/dongu.py --tek-tur            # tek deney
python3 autonomous-training/dongu.py --kos --deney 5      # 5 deney
python3 autonomous-training/dongu.py --geri-al            # son iyi checkpoint
python3 autonomous-training/dongu.py --onayla "<gerekçe>" # açık DURUS'u kapat
python3 autonomous-training/dongu.py --ci-onayla <RUN_ID> # CI kanıtını işle
```

`cargo` PATH'te değilse döngü başlamaz ve açık mesajla durur (araç yokluğu bir
regresyon değil, ortam eksiğidir). `--butce <saniye>` yalnız o oturum için deney
bütçesini değiştirir ve loglanır; politika dosyası değişmez.

## 12. İlk koşular, bulunan kusurlar ve onarımlar (2026-09-24)

Kuruluş gününde döngü iki kez durdu, üç kusur ortaya çıktı; hepsi onarıldı ve her
biri bir kanaryayla kilitlendi.

**S3 (ilk duruş).** K6 denetimi spec'ten parametre sayısını okuyamadı ve döngü
başlamadı: teşhis edemediği bir ölçümle devam etmedi, `DURUS.json` yazdı, insan
onayı bekledi. Onarım: sayı `params.toplam`'dan okunur ve grupların toplamıyla
tutarlılığı denetlenir (tek bir literal yetmez).

**Taban adım bütçesine bağlı değildi (geçerlilik kusuru).** İlk tamamlanan turun
ölçümü 4.653110'du ve kayıtlı "en iyi" 8.450811'e göre kocaman bir "iyileşme" gibi
görünüyordu; oysa taban 42 adımlık bir koşudan, deney 265 adımlık bir koşudan
geliyordu. Fark düğmeden değil adım bütçesinden geliyordu — yani karşılaştırma
geçersizdi. Onarım: tekrarlanabilirlik kapısının ürettiği taban skoru artık
`taban.adim` ile birlikte saklanır, bütçe değişince taban yeniden ölçülür; deney
sırasında taban ile adım bütçesi ayrışırsa döngü **S1** ile durur.

**S2 ve S5 beyan edilmiş ama bağlanmamıştı.** Yeni kapı `invariants-are-frozen`
anayasadaki beş durdurma koşulunu kodun `dur()` çağrılarıyla karşılaştırdı ve
kırmızı yandı: yazma reddi bir DURUŞ değil sade çıkıştı, katman-3 kötüleşme sayacı
hiçbir eşiğe bağlı değildi. Onarım: `s2_dur()` yazma reddinde `DURUS.json` yazar,
`kendini_gelistir()` eşik aşılınca S5 ile durur; ikisi de artık kanaryalı.

**Üçüncü kırmızılar.** Aynı turda `no-secret-material` da kırmızıydı: yeni ölçüm
betiği `training/kimlik_bicimleri.py` özel anahtar başlığını literal taşıyordu.
Fikstür artık parçalardan kurulur (fikstür gerçek bir biçim, kaynak dosya bir
anahtar bloğu değil) ve `readme-is-measured` kapı sayısını güncelledi. O günün
kaydı kırmızıları **adıyla** yazmıyordu; artık `kirmizilar` alanı ve oturum günlüğü
hangi kapının/testin düştüğünü söylüyor — teshis edilemeyen bir ret, ölçüm değil.

**Zaman bütçesi tutmuyordu (ölçüm kusuru).** İlk doğrulanmış turun tabanı iki kez
ölçüldü (265 adım, skor 4.980211) ve öneri `--ogrenme-orani 0.01 → 0.003875` skor
7.357786 verdi: aynı adım bütçesinde olduğu için bu kez **karşılaştırma geçerliydi**
ve deney haklı olarak atıldı. Ama üç koşu da 393–399 s sürdü; bütçe 300 s.
Sebep: tek noktalı prob. Onarım: iki noktalı eğim ölçümü (§3) + bütçe gözlemi.
Aynı gün ölçülen `ckpt_yuku_silindi: true` ile koşu yükü temizliği de doğrulandı.

Ders, kural olarak: bu otomasyonun tam doğrulaması (fmt + clippy + test + tüm
kapılar) deney sırasında **eller serbest bırakılmaz**; ağaçta eşzamanlı düzenleme
varsa tur kırmızı çıkar ve haklı olarak atılır.
