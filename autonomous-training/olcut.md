# Değerlendirme kriterinin tanımı (ölçüt)

Bu dosya **ölçütün tanımıdır** ve dokunulmazlar listesindedir. Otomasyon bu
tanımı değiştiremez; değiştirme girişimi durdurma koşulu S2'dir. Tanım
değişecekse operatör değiştirir ve `INVARIANTS.sha256` yanındaki kayıt gibi bu
dosyanın da yeni hâli bir commit olur.

## Deney nedir

Sabit süreli bir koşu: mevcut yapılandırmaya, `mutasyon_alani.json`'daki
**düğme uzayı**ndan bir değişiklik önerilir, eğitim çekirdeği o
yapılandırmayla koşar, sonuç değerlendirilir. Deney bütçesi duvar saatidir ve
`ayarlar.json`'dan gelir; adım sayısı, ölçülen adım hızından **türetilir**
(kalibrasyon), sabit yazılmaz.

## Ölçüt (tek cümle)

> Bir deney, **en iyi doğrulama kaybını** eşikten fazla düşürüyorsa tutulur;
> düşürmüyorsa atılır ve atılma sebebi hafızaya yazılır.

## Karşılaştırmanın geçerlilik koşulları

Bir karşılaştırma ancak şu dört şey sabitken yapılır:

1. **tohum** — koşu tekrarlanabilir olmalı;
2. **adım bütçesi** — iki koşu aynı adımda durur;
3. **korpüs ve damga** — `lubot korpus-damgasi` ile ölçülen damga aynı olmalı,
   yoksa `egitim-kosu` zaten reddeder;
4. **doğrulama düzeni** — `--dogrulama-her` ve `--dogrulama-payi` sabit; yoksa
   doğrulama kaybı iki tarafta farklı kümeler üzerinde ölçülür.

Dördünden biri değişirse karşılaştırma geçersizdir ve deney **atılır**
(başarısız sayılır, hafızaya "geçersiz karşılaştırma" diye yazılır).

## Tekrarlanabilirlik kapısı (oturum başında)

Oturum, burakımı almadan önce mevcut yapılandırmayı **iki kez** koşar ve iki
kaybın bit-bit aynı olmasını şart koşar. Aynı değilse ölçüm gürültülüdür:
otomasyon durur (`S1` değil, kendi iç durdurması: *ölçüm tekrarlanabilir
değil*) ve insan onayı bekler. Gürültülü bir düzenekte "iyileşme" iddiası
kanıtsızdır.

## Şema doğrulaması

Her koşu raporu ve kaydı, `lubot`'un kendi çıktı şemasından geçer (yumuşatma
yok). Rapor tablosu okunamıyorsa ya da kayıt `olcut`/`kaynaklar` alanlarını
taşımıyorsa deney **başarısız** sayılır — kayıpsız bir koşu, iyi bir koşu
değildir.

## Regresyon

Tam doğrulama (fmt + clippy + test + tüm kapılar) her **tutmadan önce** koşar.
Tek bir başarısız test ya da kapı, o deneyin tutulmasını engeller (deney
atılır; bu normal akıştır). Buna karşılık **art arda iki deney** tam
doğrulamadan kırmızı çıkarsa otomasyon durur (S4): artık tek bir deneyin
suçu değil, düzeneğin kendisi şüphelidir.

## Kendini geliştirme (katman 3)

Otomasyon yalnızca `kendini-gelistirme.json`'daki **beş sayıyı** değiştirebilir
(arama politikası). Her değişiklik bir deneydir: loglanır, geri alınabilir ve
sonraki pencere onun etkisiyle ölçülür. Art arda üç değişiklik, pencere
ortalamasını kötüleştirirse otomasyon durur (S5).
