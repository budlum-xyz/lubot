# Örnekleme cezaları — ölçülen hâli (2026-09-25)

Bu belge bir **ölçüm kaydıdır**, tasarım notu değil. Cezalar (`tekrar`, `varlık`,
`sıklık`, n-gram yasağı, durma erteleme) `crates/cikarim/src/cezalar.rs` içinde
durur; burada yazılan şey, açık cezalarla üretilen metnin kapalı cezalı metinden
nasıl ayrıldığı.

## Neden var

350 adımlık kontrol noktası serbest üretimde kısa dizilere takılıp kalıyordu:
çıktı tekrarlardan ibaretti (`... ki the I ki kaydını I S. y gs the a . y . I
yürütülen kaydını ...`). Bu bir *örnekleme* hatasıdır: dağılımın kuyruğu kendi
kendini besler. Cezalar tam olarak bu kuyruğu kesmek için var.

## Ölçüm

Aynı kontrol noktası (`training/ckpt/lubot-a1.ckpt`), aynı soru
(`Kalibrasyon kapani nedir?`), aynı tohum (`20260924`), 48 jeton:

| koşu | ayar | çıktı (kırpılmamış) | süre |
|---|---|---|---|
| A | cezasız (`tekrar 1.0`, `kac_gram 0`) | `s ki the I ki kaydını I S.` / `y gs the a . y . I yürütülen kaydını   [[0` | ~1,6 s |
| B | `--tekrar-cezasi 1.3 --kac-gram 3 --en-az-jeton 8` | `answer: arylelen kaydını, zTE-00020k.10_singT: > 0 f to üm-09[4 ha` / `[-ır:000000gat` | ~1,8 s |

B koşusunun rapor satırı:

```
| ceza | ceza: adim basina toplam 1427 jeton geriletilmis, 1 yasak, 0 erteleme |
```

## Ne ölçüldü, ne ölçülmedi

**Ölçüldü:** ceza açıkken dizi tekrarı kesiliyor (aynı `ki`/`I` döngüsü yok);
ceza kapalıyken üretim bit bit aynı kalıyor; aynı tohum aynı diziyi veriyor
(`ceza_acikken_uretim_tekrarlanabilir_ve_yasak_sayilir` testi); cezanın kaç
jetona dokunduğu ve kaç jetonu yasakladığı raporda yazılı.

**Ölçülmedi:** çıktının *anlamlı* olduğu. B koşusunun metni de bozuk; cezalar
tekrarı kesiyor, dili düzeltmiyor. Dil, 350 adımlık bir koşunun ürünü ve bu
belgenin konusu değil. "Ceza ile model düzeldi" gibi bir cümle burada
yazılmayacak: ölçülen şey tekrarın kesilmesi, dilin düzelmesi değil.

## Sözleşme

* Cezalar **varsayılan kapalıdır**; açık olan her ceza rapora yazılır.
* Yasak, logiti `-inf` yapmaz: örnekleyiciye **ayrı liste** olarak geçer.
  Sebep ölçülerek görüldü — örnekleyici sonlu olmayan logiti "dağılım
  kurulamaz" diye reddediyor ve bu ret *doğru*: `-inf` ile yazılan bir yasak,
  "bu logit bozuk" ile "bu jeton yasak" ayrımını silerdi.
* Bütün adaylar yasaklanırsa örnekleme uydurmaz: `CezaHatasi::BosAday` döner.

## Kayan pencere yolu da bu turda ölçüldü

Üretim döngüsü artık önbelleği her adımda baştan kurmuyor: bağlam bir kez
okunuyor, sonra adım başına tek jeton ilerliyor. Kayan pencere dolduğunda iki
yol var ve **ikisi aynı şey değil**:

| yol | ne yapar | bedel | rapor alanı |
|---|---|---|---|
| `yeniden` (varsayılan) | düşen konumdan sonrasını yeniden hesaplar | pencere dolduktan sonra adım başına `O(pencere)` | `onbellek yeniden kurma` |
| `onbellek` | yalnız önbellekten düşürür | adım başına `O(1)`; geçmiş gizli durumları eski pencereyle hesaplanmış kalır (**yaklaşım**) | `onbellek yeniden kurma: 0` |

Varsayılan, ölçümü sessizce değiştirmeyen yoldur; hızlı yol `--kaydirma
onbellek` ile açılır ve raporda "yaklasim" olarak işaretlenir.
