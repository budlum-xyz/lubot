# Geçiş hattı

Bir dış kaynak ağacını bu workspace'e getirmenin maliyeti iki parçadır:
gerçekleme (fikirin Rust'ta, ölçülerek yazılması) ve bunun dışındaki her şey
(envanter, tesisat, kayıt, takip). İkinci parça gündür ve mekaniktir; bu
hat onu komut yapar. Üç fiil, `crates/gecis`'te:

## `lubot gecis sayim --kaynak <dizin> --lisans <ad> [--yaz f.json]`

Ağacı deterministik gezer ve sayar:

- dosya cinsi (uzantı; bilinmeyen uzantıda ilk 256 baytın NUL'u),
- satır sayısı (ikilide 0),
- herkese açık semboller (Python: satır başında `def`/`class`/sabit; Rust:
  `pub fn`/`struct`/`enum`/`const`/`trait`/`type`; Markdown: başlıklar),
- içe aktarmalar, ilk bölümleriyle.

Tüm dosyaların yol+özetinden tek bir **ağaç özeti** (sha256) üretilir; aynı
baytlar aynı özeti verir, dolayısıyla bir plan hangi kaynaktan kurulduysa
onu isimlendirir. Kayıt **yalnızca göreli yollar** taşır: kaynağın kök adı
girmez, çünkü bir bileşenin hangi depodan geldiği provenance'dır (alım
hattının işi), mimari değil. Lisans alım hattının kapalı kümesiyle
doğrulanır. Sınırlar (dosya, sembol, bayt) aşıldığında red, aşılan sınırın
adını söyler; sembolik bağ izlenmez.

## `lubot gecis plan --sayim f.json --ad <crate> [--yaz f.json]`

Sayımdan gerçekleme planı:

- sembol taşıyan her kod dosyası bir **modül**; normalleştirilmiş ad
  (Türkçe harfler ASCII'ye iner, yol bileşenleri tireyle birleşir),
- modülün **sözleşmesi**: gerçeklenmesi beklenen sembol adları,
- içe aktarmaların iç/dış ayrımı (ilk bölüm ağaçtaki bir dosya ya da dizin
  adıyla eşleşirse iç),
- `test_amaci`: sözleşmenin toplam sembolü — herkese açık her sembol için
  en az bir test hedefi. Hedef, iddia değil,
- **tesisat satırları**: kök `Cargo.toml` üye satırı, cli bağımlılık
  satırı, CRATES.md satır şablonu, kapı kaydı — ağacın kendi biçiminde.
  `gecis-hatti-kapisi` üye ve bağımlılık satırlarını repodaki gerçek
  satırlarla bayt bayt karşılaştırır; biçim iki yönde de kayarsa yakalanır.

İki dosya aynı moda indirgerse plan, her ikisini adıyla söyleyerek reddeder.

## `lubot gecis durum --plan f.json [--gerceklesen <dizin>] [--esleme f.json]`

Gerçekleşen crate'in kapsamı:

- modül düzeyi: `src/<modül>.rs` var mı; varsa kaç `pub` öğe, kaç `#[test]`,
- sembol düzeyi (yalnızca eşleme verilmişse): `"modül:kaynak sembolü" ->
  "rust adı"` eşlemesiyle, eşlenen sözleşmenin ne kadarı geçiyor.

Bekleyen modüller listelenir; eksik sözleşme saklanmaz. Eşleme
verilmeden sembol düzeyi ölçülmez, çünkü kaynak ve gerçekleşen adlar farklı
dillerdedir ve tahmin kapsam değildir.

## Ne yok, bilinçli olarak

**Gövde üretimi yok.** Üretilmiş bir iskelet derlenir, lint'leri geçer ve
hiçbir fikir taşımaz; bu reponun anayasası ölçülmüş, sıfırdan
gerçeklemedir. Gerçekleme, planın sözleşmesi altında elle yazılır ve kapsam
sayısı ne kadarının açık olduğunu söyler. Hız, disiplini çiğnemeden
mekanizmadan gelir.

Bilinen sınırlar, gizli değil: dizin düzeyi paket içe aktarması dizin adıyla
eşleşir (noktalı yol çözümlenmez), virgüllü içe aktarma yalnız ilk adı
kaydeder, sembol düzeyi kapsam eşleme ister.

## Döngü, operatör için

```
lubot gecis sayim --kaynak <kaynak-agaci> --lisans MIT --yaz sayim.json
lubot gecis plan --sayim sayim.json --ad <crate-adi> --yaz plan.json
# gerçekleme: plan.json'daki sözleşme, elle, testleriyle
lubot gecis durum --plan plan.json
```

Kapı: `gecis-hatti-kapisi` (sayım yinelenebilir, tesisat satırları ağaçla
bayt eşit, kapsam kayıtla aynı, redler adlı). Kayıt:
`training/eval/sonuclar/gecis-hatti-2026-09-27.json`.
