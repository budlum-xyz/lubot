# Hesap çekirdeği: iki çekirdek, iplikler ve ölçülen sonuç (2026-09-25)

Bu belge bir ölçüm kaydıdır. İçindeki her sayı bu makinede (2 çekirdek, x86-64,
`cargo 1.98.1`, `--release` değil `debug` profil) koşularak alındı; olmayan bir
sayı yok.

## 1. Neden ikinci bir çekirdek

`crates/egitim/src/lib.rs` içindeki çekirdek f64: gradyan sınamasının referansı,
kontrol noktası biçiminin taşıyıcısı ve "acaba yuvarlama mı" sorusunun cevabı.
`crates/egitim/src/kernel32.rs` ise aynı mimarinin f32 sürümü: aynı toplama
sırası, aynı kayıt-sınırı maskesi, aynı katman düzeni — yalnız aritmetik f32.

İki çekirdek olmasının dürüst gerekçesi hız değil **çapraz doğrulama**: iki
çekirdek aynı gradyanı vermek zorunda ve bu ölçülüyor.

| test | ne ölçer | sonuç |
|---|---|---|
| `f32_gradyani_f64_ile_uyusur` | dokuz tensörün göreli sapması | < 2e-3 (geçti) |
| `f32_kayip_maske_ile_birlikte_uyusur` | maskeli pencerede kayıp | \|Δ\| < 1e-4 (geçti) |
| `f32_kayit_siniri_asilmaz` | uzak kuyruk değişince ilk kayıt sabit | sabit (geçti) |
| `indir_geri_tur_f32_sinirinda_dogru` | f64→f32→f64 turu | \|Δ\| < 1e-6 (geçti) |

## 2. f32 hız kazandırmadı: ölçüldü

`hiz_olcumu_f64_vs_f32` (ignored test; 128 bağlam, `d_model` 128, 4 katman,
3 tekrar ortalaması):

| çekirdek | süre | oran |
|---|---|---|
| f64 | 8736 ms | 1,00x |
| f32 | 8096 ms | **1,08x** |

Yani skaler Rust'ta f32, f64'ün iki katı hızlı **değil**. Sebep yapısal:
x86-64'te skaler `add`/`mul` f32 ve f64 için aynı gecikmede; kazanç ancak
vektörleşmeyle (`f32x8`) gelir ve bu depoda o yol yok. Sonuç: f32 çekirdeği
hız için değil, çapraz doğrulama için duruyor ve kodda da öyle yazıyor. Bir hız
iddiası ölçülmeden yazılmayacak.

### 2b. Ölçüm artık bir komut: `lubot egitim-karsilastir`

Tolerans bir testte yaşadığı sürece "bu makinede ne oluyor" sorusu cevapsız
kalıyordu. Komut, gerçek korpustan pencereyi dolduracak **ilk kaydı** seçer
(kısa kayıtla ölçmek modelin gördüğü bağlamı ölçmemek olurdu), iki çekirdeği
koşar ve tensör tensör yazıdır:

```
lubot egitim-karsilastir --corpus corpus/knowledge-self.jsonl.gz --rapor f.md
```

Eğitilmemiş `lubot-a1` ile, 256 jetonluk pencerede ölçüldü:

| ölçü | değer |
|---|---|
| kayıp (f64) | 9,057966 |
| kayıp (f32) | 9,057968 |
| kayıp farkı | 1,646e-6 |
| en kötü tensörel oran | 2,555e-6 (`wk`) |

Yani ölçülen sapma, testteki 2e-3 toleransının binde birinden küçük. Komut
aynı zamanda bir red yolu: pencereyi dolduracak kayıt yoksa tahmin yürütmez,
"korpusda 256 jetonluk kayit yok (en uzun N)" der.

## 3. İplikler: gerçek kazanç burada, ölçüldü

Bir adımın pencereleri birbirinden bağımsızdır: dikkat kayıt sınırını aşmaz,
adım yalnız toplar. Bu yüzden `KosuAyari::iplik` ile pencereler iş parçalarına
bölünüyor (`yigin_gradyanlari`) ve sonuçlar **pencere sırasında** toplanıyor.
Toplama sırası korunduğu için paralel yol, sıralı yolun aynısını üretir: aynı
sayılar, bit bit.

Ölçüm: gerçek korpus (81.349 kayıt / 14.185.482 jeton), 6 adım, `--yigin 8`,
`--pencere 256`, aynı tohum.

| koşu | iplik | duvar saati | kontrol noktası sha256 | kayıp izi |
|---|---|---|---|---|
| A | 1 | 630,8 s | `d73ea5c04ed3…` | 9,036236 → 8,903560 |
| B | 2 | 325,6 s | `d73ea5c04ed3…` | 9,036236 → 8,903560 |

**1,94x hız, aynı sonuç.** İki kontrol noktasının sha256'sı birebir aynı; kayıp
izi birebir aynı. Not: süreler korpusun yüklenmesi/paketlenmesi gibi tek
iplikli bir ön bölümü de içeriyor, yani adım başına kazanç 1,94x'ten **az
değil**.

Bunu iddia değil ölçüm yapan testler:

* `iplik_sayisi_sonucu_degistirmez` — aynı yığın, 1 ve 2 iplikle, gradyanlar
  `assert_eq!` ile karşılaştırılır (f64::EPSILON bile fark yok).
* `iplik_sayisi_yiginla_sinirlanir` — 4 pencerede 8 iplik istemek 4 iş parçası
  demektir; boş iş üretilmez.

## 4. Kullanım ve sınırlar

```
lubot egitim-kosu ... --yigin 8 --iplik 2      # iki iş parçası
lubot egitim-kosu ... --iplik 0                # makineye sor (varsayılan)
```

* `--iplik 1` sıralı yolu zorlar; sonuç değişmez, süre değişir.
* İş parçası paniklerse adım **rapor edilmez**: `KosuHatasi::IsParcasiDustu`.
  Yarısı hesaplanmış gradyanla adım atmak, ölçülen bir tur değildir.
* Bu makinede 2 çekirdek var: tavan 2x. Daha geniş kutuda `--iplik` yükselir,
  `--yigin` de yükselmek zorundadır; aksi hâlde iş parçası sayısı pencere
  sayısına takılır.

## 5. Sırada ne var (iddia değil, plan)

Ölçülen darboğaz hesap değil: 6 adım 325 saniye sürüyor ve bunun neredeyse
tamamı matris çarpımlarında geçiyor. Sıradaki gerçek kazanç yerleri:

1. Vektörleşmiş matris çarpımı (`f32x8` benzeri bloklama).
2. Paketleme kapsaması zaten %99,9999; veri tarafında kazanç yok.
3. Doğrulama geçişini de iş parçalarına bölmek (şu an tek iplikli).
