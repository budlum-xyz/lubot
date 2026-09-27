# Mimari tasarım notları

Bu belge bir **tasarım notudur; kod değildir**. Uygulama direktifinin (akış 2, 2026-09-26)
mimari bileşen sırasındaki "her bileşen için ayrı Rust modül tasarım notu yaz — henüz kod
değil" adımının karşılığıdır ve sıfırdan-eğitim defterinin (NN) 10-13 adımlarına eşlenir.

Bu belge `served: false` damgalıdır (bkz. `training/servis-politikasi.json`): tasarım
notu süreç belgesidir; uygulanan her bileşen kendi kodunu, kendi kapısını ve kendi
ölçüm kaydını getirdiğinde damgası kalkar.

## 1. Kurallar

- Bu ağaçta üçüncü taraf adı geçmez (K1; veri kabul kararı, 2026-09-24: "ne kaynak adı
  ne lisans adı depo ağacına yazılır"). Yöntem ilhamı kaynaklarının tam künyesi,
  lisansları ve doğrulama sonuçları workspace'tedir:
  `arastirma/2026-09-26-lubot-mimari-referans-dogrulama.md`.
- Yalnızca **desen** alınır: hiçbir dış ağırlık, kod, veri, çıktı bu ağaca girmez (K1/K2).
  `kodlayici` dış kontrol noktasını okuyup çalıştıran tek crate olarak kalır; sıfırdan
  eğitilen çekirdek dış kontrol noktası okumaz.
- Bu belgede hiçbir sayi ölçülmüş gibi yazılmaz. Ölçülmüş değerlerin tek kaynağı
  `training/ratchet.json`, `training/model_spec.json` ve `training/eval/sonuclar/`
  kayıtlarıdır; burada yalnızca beyanlara atıf yapılır.
- Mimari değişiklik yeni model ailesi demektir (EE/NN-3 çerçevesi: sözlük donmuş ve
  sürümlüdür; mimari spec commit edilir, kafada taşınmaz). Bu notlardan hiçbiri
  `model_spec.json`'u değiştirmez.

## 2. Zemin: hangi desen zaten nerede duruyor

| desen | bugünkü yeri | durum |
|---|---|---|
| donmuş, sürümlü bayt-düzeyi BPE | `jeton`, `sozluk`, `bpe-gelismis`, `training/tokenizer/` | var (NN-2) |
| derin-dar spec + bağlı embedding + param muhasebesi | `derin`, `training/model_spec.json` | var (NN-3) |
| μP parametrizasyonu veri olarak + init ölçümleri | `mu`, `training/mup_olcum.py`, ölçüm kayıtları | var (NN-4 dilimler) |
| ileri geçiş + el yazımı geri geçiş + paketleme + kontrol noktası | `transformer`, `egitim` (LUBOTCKPT), `cikarim` | var |
| tiplediğim karar başlığı: kapalı seçim/puan/noul-türevi, k-of-n | `tomurcuk`, `karar` | var (T, LL) |
| kanıttan hüküm, ölçülü batarya | `kanaat` | var |
| alt-bayt ağırlık nicemlemesi + yerinde okunan kap + derinlik merdiveni | `nicem` (2,125 bit/ağırlık ölçüldü), `tasiyici` (LUBOTNCM) | var |
| şema-doğrulamalı tek Markdown çıkış, üretim yüzeyi yok | `read/output_schema`, `answer`, kapılar | var |
| dış kontrol noktasını okuyup çalıştırma | `kodlayici` | var (sözlük tarafı eksik) |
| sıfırdan yazılan kodlayıcı omurgası: dönel konum + dönüşümlü yerel/genel dikkat + bias'sız blok + birleştirilebilir ağırlık dizini | `omurga` | var (2026-09-27; tasarım notu değil kod) |
| kendinden-damıtma, mekanik jüri | `kendinden` | var |
| kıyas protokolü, rakip çıktısı korpusa girmez | `kapisma` | var |

Eksik olan beş deseni bu notlar tasarlar: (a) Hadamard/Monarch MLP, (b) GQA +
nedensel kıvrım dokunuşları, (c) engram belleği, (d) çok-şeritli artık bağlantılar,
(e) kalibre edilmiş güven başlığı. Altıncı desen (dilbilgisi-kısıtlı çözümleme) bu
deponun anayasası tarafından zaten büyük ölçüde karşılanır; kalan boşluk notlanır.

**2026-09-27 eki.** Yukarıdaki maddelerin hiçbiri bileşenlerin üzerine oturacağı
*yığını* tarif etmiyordu. §3.9 o yığını ekler ve bir tasarım notu değil kod
getirir (`crates/omurga`, `omurga-sozlesmesi` kapısı).

## 3. Bileşen tasarım notları

### 3.1 Hadamard / Monarch MLP (FFN yerine)

**Ne:** standart iki-matris MLP yerine, iki (veya Monarch bloğuyla kademeli) doğrusal
projeksiyonun eleman-bazlı (Hadamard) çarpımı: `h = (W1·x) ⊙ act(W2·x)`, ardından
düşüş projeksiyonu. Aynı param bütçesinde daha az derin-fermat çarpımı; küçük
genişliklerde param/yarar oranı standart MLP'ye karşı ölçülerek karar verilir.

**Rust yeri:** `transformer` içine ayrı modül (`mlp/hadamard.rs` adayı); spec tarafında
`derin`'e yeni MLP çeşidi, `model_spec.json`'a yeni `mlp_turu` alanı (aile değişikliği).

**Param muhasebesi:** `say_params`'a yeni terim: yükseliş `2·d·d_r` (iki doğrusal, d_r
ara boyut) + düşüş `d_r·d`; grup parametrelemesi `mu`'ya yeni parametre grubu olarak
girer — Hadamard çarpımı fan_in tanımını değiştirir (her eleman iki girdiden gelir);
init std ve LR formülü μP tablosuna **yeni satır olarak** eklenir, mevcut satır
bozulmaz.

**Geri geçiş (el yazımı):** `∂/∂W1 = (∂h/∂u ⊙ v)·xᵀ`, `∂/∂W2 = (u ⊙' …)` — eleman-bazlı
çarpımın her iki koluna standart çarpım-zinciri; `egitim` çekirdeğindeki mevcut
lineer-geri geçiş blokları yeniden kullanılır, yalnız `⊙` gradyanı yeni.

**K-çapraz:** K1 uygun (desen bizim); K6: param formülü değişir, tavan yeniden türetilir
(`recommend_model_size` zinciri otomatik); EE: aile değişikliği — lubot-a2 adayı.

**2026-09-27 düzeltmesi: bu notun tarif ettiği blok, port kartının bloğu değil.**
Yukarıdaki `h = (W1·x) ⊙ act(W2·x)` bir *kapılı* MLP'dir ve `egitim::mlp_hadamard`
onu uygular. Port kartının bloğu ise başka bir şeydir ve adı yüzünden aynı
sanılıyordu: **Kronecker çarpanlı Walsh-Hadamard dönüşümü**. Kartın aritmetiği
sayısal olarak birebir alındı ve `crates/omurga/src/hadamard.rs`'e yazıldı
(adlar ve dosya düzeni bu deponun; hiçbir dosya ithal edilmedi, lisans notu port
kartında durur):

| | bu notun bloğu (`egitim::mlp_hadamard`) | kartın bloğu (`omurga::hadamard`) |
|---|---|---|
| çekirdek | iki projeksiyonun eleman-bazlı çarpımı | üç aşamalı Kronecker-Walsh dönüşümü |
| ara genişlik | `d_r` seçilir | `n` = `d_model`'in üstündeki ikinin kuvveti, seçim yok |
| çarpan maliyeti | `3·d·d_r` | `3·(ba² + bb²)`, `ba·bb = n` |
| 768 genişlikte | `d_r`'ye bağlı | 6144 çarpan sayısı, yoğun eşdeğeri 1.048.576 |
| aktivasyon | GELU kolu | SiLU, ikinci aşamada |
| koşullama | yok | `c = 1 + softmax(x·W_v)·W_u`, rank 8, `W_u` sıfır → taze blokta `c = 1` (bit-özdeş) |
| karıştırma | yok | iki donmuş permütasyon, aşamalar arasında |
| çıkış ölçeği | init kuralı | çıkış diagonali 0.02, yani blok sessiz başlar |

**Beyan edilen tek sapma:** iki donmuş permütasyon kartın üretecinden değil bu
deponun `Tohum`'undan çekiliyor. Permütasyon bir *bijeksiyon* olduğu sürece
dönüşümün yapısı aynıdır; hangi bijeksiyon olduğu ise farklıdır, yani eğitilmiş
ağırlıklarda iki uygulama bit bit uyuşmaz. Bu saklanmadı: kapı permütasyonun
gerçekten bijeksiyon olduğunu ölçüyor.

**Ölçülmeyen:** geri geçiş yazılmadı (bu crate ileri-yönlü). Hangi bloğun aileye
gireceği hâlâ M1'dir ve bu iki blok artık **ayrı iki aday**tır, tek aday değil.

**Açık sorular (işaretli, mimari karar):** d_r seçimi ölçüm ister (aday ızgara yöntemi,
NN-3'ün 63 adaylı desenine eşdeğer); Hadamard MLP'nin bu korpusun api/behaviour ağırlığı
altında standart MLP'den iyi olup olmadığı **ölçülmeden varsayılmaz**.

### 3.2 GQA + nedensel kıvrım dokunuşları

**Ne:** tam dikkat yerine gruplanmış-sorgu dikkat (kafa başına ayrı Q, gruplaşmış K/V);
Q/K'ya girmeden önce kısa (3-7 dokunuş) nedensel konvolüsyon: yerel örüntüyü taşır,
uzun bağlamda K/V benzerliğini yumuşatır.

**Rust yeri:** `transformer` dikkat modülü; `egitim` geri geçişine konvolüsyon gradyanı.

**Uyumluluk şartları:** kayıt-sınırı maskeleri bozulmaz (dikkat penceresi kayıt
sınırını aşamaz — `egitim`'in kayıt-sınırı testleri aynen kalmalı); f64/f32 çapraz
doğrulama çifti (HESAP-ÇEKİRDEĞİ deseni) yeni op için de koşar; iplik bölünmesi
pencere-bağımsızlığı korunarak yapılır (ölçülmüş bit-bit-aynı sonuç özelliği şart).

**Geri geçiş:** GQA grup yayılımı (broadcast gradyan) + konvolüsyon için doğrudan
kısa-döngü gradyan (dokunuş sayısı sabit ve küçük; genel conv altyapısı gerekmez).

**K-çapraz:** K6: param tasarrufu (K/V paylaşımı) tavana yer açar — ölçülünce spec'e
yansır. İşaretli karar: kafa/grup oranı.

### 3.3 Engram belleği (n-gram karmasıyla adreslenen KV tablosu)

**Ne:** dikkat K/V'sinin bir bölümü, diziden hesaplanan K/V yerine, donmuş jetonlayıcı
üzerinden hesaplanan n-gram karmasıyla adreslenen öğrenilmiş tablodan `gather` ile
okunur. Model, sık n-gram'ların belleğini parametreye taşır; dikkat hesabı aynı kalır.

**Rust yeri:** yeni modül (`memory/engram.rs` adayı) `transformer`'ın altında;
tablo, kontrol noktasında ayrı tensör olarak yaşar (LUBOTCKPT'e yeni alan).

**Tasarım şartları:**
- Karma **deterministik ve tohumsuz** olmalı (aynı jeton dizisi her makinede aynı
  hücreyi adresler; aksi halde çapraz-donanım determinizmi kapısı düşer).
- `gather`'ın geri geçişi seyrek toplayıcıdır: yalnız okunan hücrelere gradyan yazar;
  `egitim`'in yığın toplama sırası korunursa determinizm korunur (işaret: ölçülmeden
  iddia edilmez, kapı-kanıt istenir).
- Kayıt-sınırı maskesi engram okumalarına da uygulanır: bir pencere, komşu kaydın
  n-gram'ından okuyamaz (provenance-bağımsızlık korunur).
- Çakışma politikası fail-closed tercih edilir (aynı hücreye düşen farklı n-gram
  ayrımının **yasak** değil **ölçülen** bir kayıp olarak raporlanması; ikincisi
  tercih, kapı kanıtı gerekir).

**K-çapraz:** K6: tablo boyutu doğrudan param bütçesinden yer — tablo boyutu aday
ızgarası ölçüm ister (öneri: spec'e `engram` bloğu, boyut `ölçülmedi` etiketiyle).
K2: engram yalnız kendi korpusumuzun n-gram'larını öğrenir; dış tablo alınmaz.

**İşaretli mimari karar:** engram bu aileye girer mi, yoksa engram'sız lubot-a2 önce
mi koşar? Öneri: **sonra** — GQA ve MLP değişiklikleri ölçülmeden engram tablosu
bütçeyi belirsizleştirir. Operatör onayı bekler.

### 3.4 Çok-şeritli artık bağlantılar (hyper-connections)

**Ne:** tek artık akış yerine N paralel akış; her katman girdiyi ve katman çıktısını
şeritler arasında küçük öğrenilmiş bir karışım matrisiyle dağıtır. Artık ölçek
öğrenilir; derin yığınların ileri-geçiş varyans büyümesi (θ₁ bandı bulgusu) bu
desenle kısıtlanabilir.

**Bağlantı — bilinen açık bulgu:** μP ölçüm turlarında ileri geçiş RMS profili Θ(1)
bandının dışında büyüdü ve bu, `model_spec.json`'da değiştirilmeden kayda geçti.
Çok-şeritli bağlantılar bu soruna **olası bir cevap adayıdır**; ama init kuralı
değişikliği gibi bu da mimari karardır — bu not yalnızca adayı isimlendirir,
spekülatif ölçüm önerir (proxy genişliklerde şerit=2 koşusu, mevcut ölçüm şablonuyla),
hiçbir sonuç varsaymaz.

**Maliyet:** artık durum ve karışım hesabı ×şerit büyür; K6 tavanıyla çelişir mi
sorusu ölçüm ister. İşaretli karar: şerit sayısı (2 önerisi ölçülmeden kural değil).

### 3.5 Alt-bayt nicem + derinlik merdiveni: eksik kalan tek parça

`nicem` (2,125 bit/ağırlık, Lloyd-Max kod kitabı, Walsh-Hadamard dönüşümü) ve
`tasiyici` (LUBOTNCM, yerinde okuma, derinlik merdiveni, cihaz tavanı) bu deseni
ölçülü olarak taşır. Kalan boşluk iki kalem:

1. **Kademe eğitimi:** "her derinlik dağıtılabilir bir modeldir" hedefi — eğitim
   sırasında ara derinlik çıkışlarının da kayıp taşıması. Eğitim hedefini değiştirir;
   işaretli mimari karar, öneri olarak durur.
2. **Eğitim-zamanı nicem farkındalığı (J):** ölçülmeden önerilmez; nicemleme
   doğruluk/maliyet tablosu zaten ölçülü duruyor, QAT ancak o tablo bir kapıya
   bağlanınca gündeme gelir.

### 3.6 Dilbilgisi-kısıtlı çözümleme (byte-level grammar)

Bu depo açısından desen **büyük ölçüde anayasayla karşılanmış** durumdadır: üretim
yüzeyi yok (`no-generation-variant`), çıkış tek kapıdan şema doğrulamalı Markdown
(`ai-output-schema-enforced`, `output-finalize-closed-loop`), karar başlığının
çıktıları kapalı biçimlerdir (`tomurcuk`, `decision-head-has-no-generation-surface`).

Kalan boşluk: şema doğrulaması bugün **çıktıdan sonra** reddeder. Desenin tam
karşılığı, bileşenlerin **çözümleme anında** şemadan türeyen bir kısıtla üretmesi
(başlık-başlık değil alan-alan kısıt) olurdu — ki bu yalnızca üretim yüzeyi olsaydı
anlamlıydı. Okuyan-yapıda karşılığı şudur ve zaten vardır: reddedilen çıktı en yakın
formata **düşürülmez**, yeniden üretilir. Bu bileşen için kod önerilmez; not,
"desenin adayı zaten karşılanıyor" kaydıdır. Tek somut öneri: `output_schema`'nın
ret sınıfları tablo-başlığı/hierarchy düzeyinde ayrıştırılıp ret istatistiğinin
`ogren` hata ailelerine beslenmesi (FF ile uyumlu; küçük, kapılı artım).

### 3.7 Kalibre edilmiş güven başlığı

**Ne:** karar başlığının her kararıyla birlikte kalibre edilmiş bir güven skoru
taşması; güven eşik altındaysa karar otomatik olarak yükseltilir (önce kural/ret,
gerekirse insan).

**Zemin ve ders:** karar katmanının ölçülü deneyimi zaten var (workspace Jev karar
kaydı, 2026-09-24: yerel modelin karar katmanı olarak kullanılmaması hükmü; hata
yönünün tek taraflı olduğu ölçümü). Bu ders doğrudan tasarıma yazılır: güven başlığı
**hüküm vermez**, yalnızca skor; eşik ve yükseltme zinciri kodda ve kapıdadır
(`kanaat`'in "kanıttan hüküm, ret bir cevaptır" düzeniyle aynı yerleşim).

**Tasarım:** sıcaklık kalibrasyonu (doğrulama kümesinde ölçülür, kayıt
`training/eval/sonuclar/` altına mekanik-koşu şemasıyla); skora eşik değil **bant**
(Kırmızı-orta-yeşil): kırmızı → ret/yükseltme, orta → k-of-n ikinci başlık (LL),
yeşil → tek geçiş. Bant sınırları ölçümle oturur, varsayılan değer bu notta yazmaz.

**K-çapraz:** U (gecikme: başlık ölçümü tek ileri geçişe ek bir vektör — maliyet
ölçülür); W ile uyum (önbellek anahtarına güven bandı da girer).

### 3.8 Birleşik blok: altı adayın kompozisyonu

**Neden ayrı bir bileşen:** 3.1–3.7 bileşen bileşen yazıldı, bileşen bileşen
ölçüldü ve bileşen bileşen kapılandı; `docs/CRATES.md` her birinin yanına aynı
cümleyi koyuyordu: *bağlı değil*. Altı modülün her biri tek başına doğru olup
birleşik hâlleri yine de çalışmayabilir — şekiller tutmayabilir, gradyan bir
yerde kopabilir, parametre muhasebesi çift sayabilir. Bu bileşen o boşluktur:
bileşeni değil **birleşimi** ölçer.

**Sıra:** şerit durumu → okuma → sıfır merkezli RMS norm → rotalı Hadamard
uzmanları → engram değer belleği → şerit yazma → yeni şerit durumu. Her ok
gerçek bir alt modül çağrısıdır; hiçbir adım yeniden yazılmadı ve kapı bunu
denetler. Kompozisyonun kendi aritmetiği yalnız üç yerdedir: uzman çıktılarının
ağırlıklı toplamı, engram katkısının blok çıkışına eklenmesi, ve yazma
yolundan gelen `dL/dy = Σ_l beta_l · g_yeni[l]`.

**Engram bu blokta değer belleğidir.** Anahtar, parametresiz bir kapı olarak
kullanılır (`s = <anahtar, h[:d_kv]> / sqrt(d_kv)`), böylece tablonun iki yarısı
da gradyan alır; dikkat tarafındaki kullanım (3.3) ayrı bir bağlamadır ve bu
blokta yoktur.

**Rota gradyanı yazılmadı ve iddia edilmiyor.** Rota puanları bloğun *dışından*
gelir ve rota parametre tutmaz; bunun somut sonucu, buradaki bütün parametre
gradyanlarının rota açıkken de **tam** olmasıdır. Yazılmayan tek şey
`dL/dpuanlar`'dır.

**Kapalı bileşen = yokluğu.** `serit = 1` klasik tek akış, `rota = None` tek
uzman, `engram = None` bellek terimi hiç hesaplanmaz — üçü de `f64::to_bits`
ile ölçülür.

**Ölçülen:** 544 parametrenin tamamı dört noktalı (Richardson) sonlu farkla
denetlendi, ihlal 0, en kötü oran 7.93e-3; iniş gerçek bir 40 adımlık koşuda
29.44711861 → 4.93556415; engram okuması kayıt sınırını geçmiyor. İki noktalı
fark bu kompozisyonda toleransı **geçemiyordu** (en kötü oran 1.233): tolerans
gevşetilmedi, ölçüm aleti düzeltildi ve adım taraması U eğrisini kayda geçirdi
(`1.2e0, 1.0e-2, 7.9e-3, 3.2e-2, 8.9e-2, 1.6e0`).

**Bu bileşenin vermediği karar:** hangi bileşenin hangi aileye gireceği M1/M2/M3
olarak işaretli kalır. Blok bir aile değil, bir **kompozisyon yüzeyidir**;
`training/model_spec.json` ve `lubot-a1` değişmedi, hiçbir eğitim çağrısı
buradan geçmiyor.

**K-çapraz:** K1 (her satır bu ağaçta yazıldı, hiçbir üçüncü taraf adı geçmez),
K5/K6 (yeni bir çıktı yüzeyi yok), ve 4. bölümün sırası: bu blok 4–6. satırların
*ön koşuludur*, yerine geçmez — hangi bileşenin aileye gireceği hâlâ ölçümle ve
damgayla karara bağlanır.

### 3.9 Kodlayıcı omurgası (uygulandı, `crates/omurga`)

**Ne:** gömme katmanı + N blok; her blokta ön-norm, bias'sız doğrusal katmanlar,
gruplanmış-sorgu dikkat, kapılı ileri-besleme. Konum bilgisi dönel (rotary);
dikkat her `periyot` katmanda bir tüm diziyi, geri kalan katmanlarda simetrik bir
pencereyi görür. Çift yönlüdür, nedensel maske yoktur: bu depo okur, metin
sürdürmez.

**Neden tasarım notu değil kod:** §3.1-3.7'deki bileşenlerin her biri bir
omurganın *içine* takılan parçalar; §3.8 onları tek blokta kompoze ediyor. İkisi
de bir kodlayıcı yığınının *parçası*, yığının kendisi değil. Bu bölüm yığını
yazar: gömmeden çıkışa, katman çizelgesiyle birlikte. Bileşenlerin bu yığına
takılması (özellikle §3.1 Hadamard MLP'nin kapılı ileri-beslemenin yerine
geçmesi) hâlâ M1 kararına bağlı ve **yapılmadı**.

**Kapsam çakışması, gizlenmedi (M9).** Bu ağaçta artık üç kodlayıcı yüzeyi var:
`egitim` (sıfırdan eğitilen çekirdek, elle yazılmış geri geçiş), `kodlayici`
(dış yapıtı okuyup koşturan yükleyici) ve `omurga` (bu bölüm: sıfırdan yazılan,
birleştirilebilir omurga, yalnızca ileri geçiş). Üçü bugün farklı amaç taşıyor
ama aynı aritmetiğin parçalarını ayrı ayrı yazıyorlar; iki uygulama bir gün
sessizce ayrışabilir. Tek uygulamaya indirme ya da "aynı girdi, aynı sayı"
uzlaşma ölçümü bir sonraki karttır ve bu turda yapılmadı. Bir gözlem şimdiden
kayda değer: `kodlayici` dış yapıtın dönel eşleşmesini **yarıya bölme** olarak
ölçtü, `transformer` ise bugün **komşu çift** döndürüyor (M7).

**Sessizce karar verilmeyenler.** Omurga üç noktada `training/model_spec.json`'un
`lubot-a1` ailesinden ayrılır ve bunları **beyan eder, spec'i değiştirmez**:
dikkat ölçeği `1/sqrt(d_head)` (spec: `1/d_k`); ileri-besleme kapılı (spec: iki
matrisli MLP); bias yok (spec param gruplarında bias sayıyor). Bunlar ayrı bir
aile demektir; ailenin benimsenip benimsenmeyeceği operatör damgası bekler.

**Ölçülmeyenler.** Bu omurga eğitilmedi. Hiçbir ağırlık dışarıdan okunmaz;
`Omurga::yeni` tohumdan üretir. Kalite, hız, `Θ(1)` bandı ve pencere yarıçapının
alma başarısına etkisi **ölçülmedi**; `lubot omurga ileri` yalnızca ölçüm aletini
verir, hüküm vermez.

**Kapı ne kanıtlıyor (`omurga-kapisi`):** dönme eşleşmesi beyanının uygulamadan
bağımsız ölçülüp karşılaştırıldığını; tümü-maskeli softmax satırının reddedildiğini
(NaN üretilmediğini); her jetonun kendini gördüğünü (softmax'ın tanımlı kaldığını);
parametre sayımının biri dizinden biri kapalı formülden olmak üzere iki bağımsız
yoldan türetildiğini; şekil imzası uymayan iki modelin birleştirilmediğini. Kapının
kendi self-test'i bu maddelerin her birini sökülmüş bir kopyada yakaladığını
gösterir.

**6.9 bağlantısı (dallanma/birleştirme):** ağırlıklar tek düz tampon + adlandırılmış
dizin olarak durur ve `sekil_imzasi()` iki kontrol noktasının birleştirilebilir
olmasının şartıdır. `Omurga::ortala` parametre uzayında ortalama alır, imza
uymuyorsa reddeder. Bu, dal-birleştir işleminin *tanımlı* olduğunu gösterir;
birleşmenin kaliteyi düşürüp düşürmediği kapı işidir ve ölçülmedi.

## 4. Birleşik taslak: lubot-a2 adayı (yön, değil taahhüt)

Sıra, ölçüm disiplinine göre kurulur; her satır ayrı artım, kendi self-test'i ve
ratchet satırıyla:

1. **Önce zemin ölçümü:** owner donanımında NN-1 zinciri yeniden (sandbox tavanı
   geçiciydi; kalıcı tavan K6'nın şartı). Spec bu tavana göre doğrulanır.
2. **GQA** (en küçük yüzey, param tasarrufu ölçülebilir, geri geçiş lokal).
3. **Karar başlığı önce** (T, BB): `tomurcuk` zaten var; güven başlığı (3.7) ona ek.
4. **Hadamard MLP** aday ızgarası (proxy genişlik, μP transferi, mevcut ölçüm şablonu).
5. **Çok-şeritli bağlantı** yalnızca θ₁ bulgusu kapanmadıysa aday olarak.
6. **Engram** en son; tablo bütçesi ancak önceki adımların ölçümleriyle bilinir.
7. Her adım: aynı korpus, aynı sınav seti, aynı kıyas sınıfı (GG); ratchet satırı;
   lubot-a2 ailesi spec olarak commit.

Hiçbir adımda değişmeyenler: K1-K6, donmuş sözlük ailesi, okuyan-yapı, tek Markdown
çıkış, üretim yüzeyinin yokluğu.

## 5. İşaretli mimari kararlar (operatör damgası bekleyen)

| # | karar | öneri (varsayılan) |
|---|---|---|
| M1 | Hadamard MLP lubot-a2'ye girer mi | aday ızgara ölçümü sonrası karar; ölçülmeden girmez |
| M2 | engram bu ailede mi, sonraki ailede mi | sonraki (bütçe belirsizliği) |
| M3 | şerit sayısı (çok-şeritli bağlantı) | yalnız θ₁ bulgusu kapanmadıysa 2 şerit adayı; ölçüm şart |
| M4 | kademe eğitimi (her derinlik dağıtılabilir) | hedef olarak işaretli, eğitim hedefi değişikliği damga ister |
| M5 | güven bandı sınırları | ölçümle oturur; kodda sabit değer yok |
| M6 | dikkat ölçeği / init kuralı (θ₁ bulgusu) | önceki turlarda zaten açık; bu not yalnızca 3.4'le ilişkisini kaydeder |
| M7 | `transformer`'daki dönel eşleşme: komşu çift mi kalsın, yarıya bölmeye mi çevrilsin | **bulgu (2026-09-27):** beyan "yarıya bölme" diyordu, uygulama komşu çift döndürüyordu, testi aynı sabiti tekrar ettiği için hiçbir koşu göremiyordu. Beyan ölçüme bağlandı, **uygulama değişmedi**: değiştirmek aile kararıdır. Bağımsız doğrulama: `kodlayici` dış yapıttaki eşleşmeyi yarıya bölme olarak ölçtü, yani iki kodlayıcı bugün farklı eşleşme kullanıyor. |
| M8 | `omurga` ailesi benimsensin mi (dikkat ölçeği `1/sqrt(d_head)`, kapılı ileri-besleme, bias yok) | üç sapma beyan edildi, spec değiştirilmedi; benimseme ölçüm + damga ister |
| M9 | üç kodlayıcı yüzeyi (`egitim` çekirdeği, `kodlayici` dış yapıt okuyucusu, `omurga` sıfırdan omurga) tek yüzeye indirilsin mi | üçü de bugün ayrı amaç taşıyor; birleştirme ya da uzlaşma ölçümü (aynı girdi, aynı sayı) bir sonraki kart. Kapsam çakışması operatöre bildirildi. |

## 6. Kaynak işaretleri

- Yöntem ilhamı kaynaklarının künyesi, lisansları, doğrulama yöntemi ve sapmalar:
  workspace `arastirma/2026-09-26-lubot-mimari-referans-dogrulama.md` (kendi ağacımız).
- Bu notta adı geçen ölçümlerin kayıtları: `training/eval/sonuclar/`,
  `training/ratchet.json`, `training/model_spec.json`, `docs/HESAP-CEKIRDEGI.md`.
- Ağırlıkların içerik-adresli nesne olarak BUD üzerinde taşınması ve validator
  doğrulama deseni ayrı bir öneridir: workspace
  `arastirma/2026-09-26-veri-akisi-tasarim-onerisi.md` (kod bu ağaca girmez;
  sınır `tasiyici` + `mimari` katman kuralıdır).
