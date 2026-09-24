# Lubot

<p align="center">
  <img src="assets/lubot-banner.png" alt="lubot banner" width="100%" />
</p>

Lubot **okuyan** bir yapay zekâdır. Ağın sakladığı şeyden, ağın zaten uyguladığı
izinler altında cevap verir ve her cümlenin nereden geldiğini gösterir.

Altındaki doğrulama düzeneği **değildir**. Operatör kaydı, işlem teminatı, bir
modeli girdisi ve hesabıyla bağlayan üç bağ — bunlar zincirin AI çıkarım
katmanıdır ve düğümde yaşar. Lubot onun üstünde koşan tek bir yapay zekâdır:
katman değil, istemci.

**Okur; üretmez.** Metin, görüntü, ses ve video girdidir. Burada görüntü, video
ya da müzik üreten bir yol yoktur. Bu bir eksik özellik değil bir **kabul
kuralıdır**: üretilmiş bir eserin doğruluğu tanımsızdır ve denetleyemediği şeyi
kabul etmeyen bir sistemin, üretimi denetleyecek hiçbir şeyi olmaz.

Bu belge `README.md`'nin Türkçe karşılığıdır. İkisi çelişirse **`README.md`
bağlayıcıdır**; sayılar orada ölçülür ve kapılarla tazelenir.

## Bugün çalışanlar

| yetenek | crate | kanıt |
|---|---|---|
| bayttan önce izin; epoch sınırlı eğitim bütçeleri (fail-closed) | `crates/grant` | 19 test |
| üç kanal, özetle doğrulanır, dördüncüsü yok; Markdown cevap şeması; okumadan önce sihirli-bayt dosya türü ve rota reddi | `crates/read` | 28 test |
| tahmin yerine tam aritmetik; zincir kayıt istemcisi; operatör eşitleme kuralları (teminat, tek `model_hash`, tavanla özetlenmiş efor kademesi, kontrol noktası penceresi); kapsam redleri (üretim, gizli avı); eforla sınırlı cevap bütçesi; deterministik komut-risk biçimleri; korpusa kabulde kapalı lisans kümesi; kimlik-bilgisi tarayıcısı (kapalı liste, tam uzunluklar, bir **anma** asla sızıntı değildir) | `crates/tools` | 47 test |
| satır düzeyinde alıntıyla retrieval, yazma yolunda maskeleme; kapsama tabanlı ölçekli BM25 ve tek-düzenleme toleransı; karakter bütçesi altında deterministik bağlam sıkıştırma | `crates/index` | 18 test |
| birleştirilmiş okuma döngüsü, şemadan geçmiş çıkış; kapsam redleri; sonlandırılmış çıktı devri (`ai-inference` etiketi) | `crates/answer` | 14 test |
| zengin belge okuma: PDF metin çıkarımı, paragraf farkındalıklı parçalama | `crates/doc` | 4 test |
| bağlam sıkıştırma: içerik türüne göre rota, sabitlenen satırlar bayt bayt korunur, özet yeniden doğrulamalı CCR deposu, append-only tasarruf defteri | `crates/sikistir` | 11 test |
| karar başlığı: üç kapalı çıkış şekli ve metin üreten yüzey yok, sabit kademe sırası (önce deterministik kod, sonra başlık, sonra üretim), tahmin etmek yerine yükselen kalibre güven, bağımsız başlatılmış başlıklar üzerinde k-of-n uyumu | `crates/tomurcuk` | 10 test |
| koşabilir ikili: korpus yükleme, `ask`, grant defteri, çıktı denetimi, kapalı-döngü devri; `ceilings`; çok sorulu `batch`; kesintisiz iş kuyruğu (devam, bütçe, iş başına kapı denetimi, yüksek sesli duruş); yalnız yükselebilen ölçülmüş tabanlar (`ratchet`); depo `envanter`; kısıtlı `it` (yalnız listelenen yollar işlenir ve push edilir); dört adımlı `olc` doğrulama zinciri; `durum`; manifest haritası `graf`; kimlik taraması `guvenlik`; dosya türü yönlendiricisi `dosya`; ask_user biçimli karar bataryası `soru`; içerik araması `ara`, ölçülmüş `indeks`, sıralı okuma planı `mufredat`, efor kıyası `karsilastir`; bağlam sıkıştırma `sikistir`; hata madenciliği `ogren`; karar başlığı `karar` (`doktrin`/`tek`/`oyla`); kuyruk operatörü (`queue ls`, `queue iptal` — iptal edilen iş asla koşmaz); `batch`, `ask` ile aynı denetimi ve kapalı-döngü izini yazar | `crates/cli` | 37 test |

538 test, `clippy -D warnings` temiz, test dışında `unwrap`/`expect` yasak.
**83 kapı**, her birinin kendi öz-testi var; ratchet yedi anahtarda tutuyor:
testler, kapılar, pedantic uyarılar, korpus kayıtları, korpus jetonları,
bootstrap turları, sınav soruları. Korpustan türeyen sayılar burada bilerek
**yeniden yazılmaz**: bu dosya da korpusun parçasıdır, buraya yazılan bir sayı
kendisini üreten ölçüme geri beslenir. O sayıları `training/ratchet.json`
taşır ve o dosya korpusta değildir.

## İzin, bir kabul kararıdır

Herkese açık içerik sorulmadan okunur. Geri kalan her şey bir **görüntüleme
izni** ile açılır: izin verilen taraf ve bir içerik anahtarı kimliği, ve bir
bitiş anı. Birine doğrudan mesaj göndermek o izni vermektir.

Burada anahtar malzemesi saklanmaz — izin bir yetki kaydıdır; baytları açmak
depolama katmanının işidir. Geri alma **yeni** açmaları durdurur; okunmuş olanı
geri çağırmaz. Bu yüzden `Decision::Revoked`, `Decision::NoGrant`'tan farklı bir
cevaptır; ikisini birleştirmek geçmiş hakkında yalan olurdu.

Redler, izinlerle **aynı biçimde** loglanır. Canlı içerik üstünde sıfır red
bildiren bir dağıtım, denetimlerinin hiç koşmadığını bildiriyordur.

## Aritmetik hesaplanır, tahmin edilmez

```
route("74830 * 1291 kac eder?")  -> Computed { calculator, "96605530" }
route("what does revocation do?") -> the reading path
route("what is 1 / 0")            -> ToolRefused { "division by zero" }
```

Hesap makinesi `i128` üzerinde tam rasyoneldir: `0.1 + 0.2` sonucu `0.3`, `1/3`
`1/3` olarak yazılır, `2^3^2` `512`'dir ve taşma sarma değil **hata**dır.
Modelin tahmin etmesini engellemek için var olan bir araç tahmin edemez.

## Döngünün sırası

```
soru
  -> araç yönlendirici    (doğru cevabı olan soru asla bir modele varmaz)
  -> izin kararları       (arama başlamadan önce verilir)
  -> dizin araması        (yalnız açılabilen üzerinde)
  -> cevap + alıntılar    (kaynak + satır aralığı, ya da NotFound)
```

`NotFound` birinci sınıf bir cevaptır. `Refused` da öyledir ve izin defterinin
kullandığı kelimeyi taşır; böylece "geri alınmış" hiçbir zaman "bulunamadı"
olarak raporlanmaz.

## Yerleşim

| yol | ne durur |
|---|---|
| `crates/grant` | görüntüleme izinleri, geri alma, bitiş, denetim günlüğü |
| `crates/read` | üç kaynak kanalı, SHA-256 kaynak izi, korpus yüzeyi |
| `crates/index` | satır aralıklı pasajlar, gizli maskeleme, terim araması |
| `crates/tools` | tam-rasyonel hesap makinesi ve yönlendirici |
| `crates/answer` | dördünü birleştiren okuma döngüsü |
| `gates/check.py` | CI'ın uyguladığı depo kapıları |
| `training/` | korpus kurucu, gözetimli küme kurucu, donanım ölçümü, model boyu önericisi ve dondurulmuş BPE sözlük eğiticisi |
| `corpus/` | türetilmiş, kendi kendine kurulan korpus (gitignore'da; CI kapılardan önce kurar) |

## Derleme

```
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
python3 training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz
python3 gates/check.py --all
```

## İkili (binary)

`cargo run -p lubot -- <komut>` koşabilir okuyucudur. `ask` stdout'a yalnız
render edilmiş Markdown yazar — cevap yüzeyinin tek çıkışı budur — ve soru
başına bir JSONL denetim satırı ekler (okuyucu, soru, cevap türü, alıntılar,
karar, red sayısı).

```
lubot corpus corpus/knowledge-self.jsonl.gz
lubot ask --corpus corpus/knowledge-self.jsonl.gz --reader ayaz --audit audit.jsonl "what does a view grant name?"
lubot ask --corpus corpus/knowledge-self.jsonl.gz --reader ayaz --effort 0.5x "how does the epoch ledger refuse an expired grant?"
lubot grant issue --reader ayaz --key dm-1 --expires-at 2000000000
lubot grant list
lubot audit --path audit.jsonl --limit 5
lubot prompt
```

Bir korpus kaydı ancak şu koşullarla kabul edilir: baytlarıyla eşleşen bir
özet, bir içerik kimliği + varlık kimliği çifti, bir lisans, dört kapalı
türden biri ve okuma tavanının altında metin; **tek bir reddedilen kayıt
yüklemeyi düşürür**.

`--effort <etiket>` cevabı sınırlar: kademe operatörün donanım tavanıdır ve
kabul ettiği bütçe sabit bir eşlemedir (0.5x – 10.0x); yani düşük tavanlı bir
koşudan uzun cevap istenemez.

`lubot prompt` Budlum'a özgü sistem promptunu basar. Her cevabın geçtiği şema
doğrulayıcısından o da geçer; `gates/check.py --system-prompt-is-true` burada
ölçülmemiş bir olgu bildiren bir promptu reddeder — dört tavan, sekiz RPC adı,
efor aralığı ve tüketim eşiği taşıyıcı sayılardır.

`--outputs <file>` her dayanaklı ya da hesaplanmış cevap için sonlandırılmış
çıktı devrini ekler: `content_id` (baytların SHA-256'sı), `digest`, bekleyen
`asset_id`, sabit `ai-inference` etiketi ve zaman damgası. Kayıt oluştuğunda
şema doğrulayıcısı zaten koşmuştur; dosya, reddedeceği hiçbir şeyi tutmaz.

## Korpus ve eğitim

Korpus budlum-xyz yüzeyidir. `training/build_corpus.py` bu depoyu (CI'da her
zaman) ve operatör tarafında bir kaynak manifesti üzerinden budlum çekirdeğini
ve çalışma alanı kök belgelerini gezer; her kaynak kendi lisansını ve depo
başına kaynak-izi çiftini damgalar, kaynaklar arası tekrarlar bir kez girer ve
kuruluşun kendi ağaçlarının dışından hiçbir şey girmez. İzinli lisansı olmayan
bir kayıt korpusa asla girmez — ret **kapıda**dır, sonraki bir süzgeçte değil.

Sözlük dondurulmuş ve sürümlenmiştir: `training/train_tokenizer.py` bu
korpustan sıfırdan bayt düzeyinde BPE sözlüğü keser (yalnız standart kütüphane),
`training/tokenizer/` altına işler (v1: self korpus; v2: yüzey korpusu) ve yeni
bir korpus ailesi yeni bir kesimdir, sessiz kayma değil. `training/bench_hardware.py`
koşu makinesini ölçer (K6), `training/recommend_model_size.py` bu ölçümü bir boy
tavanına çevirir ve `training/model_spec.py` işlenmiş mimari spec'ini muP
tablosuna, kendi parametre sayısına ve o tavana karşı doğrular; her sayı
etiketiyle durur: ölçülmüş, türetilmiş (formül yazılı) ya da ölçülmemiş.

Korpusun ne tuttuğu (ölçülmüş): kayıt ve lisans dökümü için
`docs/CRATES.md` ve `corpus/` altındaki türetilmiş dosyalar esastır; bu belge
sayı taşımaz, çünkü kendisi de korpusun parçasıdır (geri-besleme kuralı).

Veriyi üç kapı korur: `corpus-records-carry-licence` (her kayıt izinli bir
lisans ve atıf taşır), `corpus-records-carry-provenance` (her kayıt asset_id +
content_id çiftini taşır) ve `ratchet-holds` (kayıt sayısı yalnız yükselebilir).

Ölçülen sayılar eskir ve biri CI'da yeniden ölçülür: karşılaştırma sınıfı
kaydı her koşuda bu ağaçtan yeniden kurulur, bu yüzden bir commit'in sırası
şudur: düzenle → `python3 training/build_corpus.py --repo .` →
`python3 training/kiyas_sinifi.py --kur` → `python3 gates/check.py --all` →
commit. Son düzenlemeden önce üretilen kayıt tanımı gereği bayattır ve kapı
bunu iki sayıyla birlikte söyler.

Epoch muhasebesi fail-closed'dır: `training/epoch_ledger.py`, zincirdeki
`TrainingDataGrant`'ın (zaman + azami epoch) hat tarafındaki yarısıdır; süresi
geçmiş ya da tükenmiş bir izinle korpus geçişi başlamayı reddeder ve her epoch
tüketilmelidir. Zincir tarafında izin verme işi gelecek iştir.

## Temel model

Temel modelden bağımsızdır. Kademe bir yetenek sınıfıdır; sunulan adlar
bizimdir: `ai_inference-light` (varsayılan) ve `ai_inference-normal`. Bir
kademenin arkasındaki kontrol noktası bir operatör yapılandırma değeridir;
yani çalışma zamanı, operatörün hangi ağırlıkları yüklediği hakkında hiçbir şey
söylemez.

## Otonom eğitim döngüsü ve karar katmanı

Bu depo, insan yazımı bir yön (`program.md`) altında kesintisiz koşan bir
eğitim döngüsünü de barındırır: `autonomous-training/` (dört katman, S1–S5
durdurma koşulları, `INVARIANTS.md` anayasası). Karar katmanı **K7** ile
sınırlıdır: dış model (Jev/Laya) yalnız **oy** verir; eşik, marj ve tut/at
kuralı kodda kalır, oy yoksa karar insana gider. Ölçüm ve gerekçe:
`autonomous-training/kosum/danisma-olcum.md`.

## Lisans

PolyForm Shield 1.0.0 — bkz. [`LICENSE.md`](LICENSE.md).
