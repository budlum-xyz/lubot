# Var olan agirliklardan model kesiti

Kesit artik yalnizca yeni bir spec ve yeni rastgele agirlik uretmek degildir.
`Kesit::agirliklari_al` kaynak modelin 24 tensorunu hedef koordinatlara tasir.
Yeni tohumlama, optimizer devam ettirme veya checkpoint ustune yazma yapmaz.

## Koordinat sozlesmesi

Her blok `[katman, cikis, giris]` olarak okunur. Embedding bir katmanli
`[1, vocab, d_model]` matrisidir. Iki MLP matrisi farkli sekillerdedir:
`w1=[katman,d_ff,d_model]`, `w2=[katman,d_model,d_ff]`.
Hedef satirin elemanlari kaynak satir adimiyla okunur. Duz vektorun ilk N
sayisini almak genislik kesiti DEGILDIR; ikinci satirdan itibaren yanlis olur.

- Derinlik bastan kesilir.
- Genislikte kafa boyutu sabit, kafa sayisi azalir.
- K/V kafa sayisi hedef kafa sayisinin bir boleni olur. Grup orani degisebilir;
  kaynak modelle fonksiyonel esdegerlik iddia edilmez.
- Konvolusyon dokunuslari ve QK norm parametreleri de kaynak koordinatlarindan
  alinir; kapali opsiyonlar bos kalir.
- Tam kesit, negatif sifir dahil, kaynak tensorlerin bitlerini korur.

## Bellek ve hata siniri

Tahsisten once spec, tum kaynak tensor uzunluklari ve tum agirliklarin
sonlulugu denetlenir. Tensor boyutu, toplam eleman ve toplam f64 bayti checked
aritmetikle hesaplanir. Operatorun `tavan_bayt` degeri asilirsa kesit reddedilir.
Allocator reddi de hata olarak doner; kaynak degistirilmez.

Bu tavan yeni tensor verisidir: mevcut kaynak checkpoint, allocator ek yuku,
vektor basliklari ve ileri/geri gecis gecicileri dahil DEGILDIR. Toplam surec
bellek tavani olarak sunulamaz.

## CLI

```
lubot kesit-incele --ckpt model.ckpt --derinlik 2 --genislik 32 --tensor-tavani 8388608 --girdi-tavani 16777216
```

Butun bayraklar tam bir kez verilmelidir. Komut dosyayi sinirli okur, mevcut
checkpoint okuyucusuyla checksum dogrular, kesiti kurar ve yalniz semadan gecmis
Markdown basar. Her tensorun kaynak/hedef sayisi ve kesit tensorlerinin SHA-256
ozeti raporlanir. Kaynak dosyayi degistirmez; yeni checkpoint yazmaz.

## Olcumler ve sinirlar

Rust regresyonlari: tam kesit bit ozdesligi, stride, 24 bozuk tensor sekli,
NaN/Inf, tam bellek siniri ve bir bayt eksigi, sahte kesit, boyut tasmasi,
kapali opsiyonlar, mevcut agirliklarla 12 alt-modelde ileri/geri gecis ve tam
kesitte kayip/gradyan ozdesligi.

`training/kesit.py --kur` artik ayni kaynak agirliklari keserek 12 ileri/geri
kosuyu olcer. `tam_ozdes` kaynak ve tam kesit bitlerinden hesaplanir. Kalite
korunumu, optimizer moment aktarimi ve paylasimli alt-model egitimi olculmedi.

CI onayi ile aday uygulama birbirinden ayridir; sadece bu dosyanin varligi
bir tamamlama kaniti degildir.
