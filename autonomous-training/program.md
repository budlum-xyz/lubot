# Yön dosyası (program) — insan yazar, otomasyon okur

Bu dosya otomasyonun **neyi** araştıracağını söyler; **nasıl** araştıracağına
otomasyon karar verir. AI Scientist tarzı "hangi yönü araştırayım" katmanı
bilinçli olarak yoktur: yön buradan gelir, arama politikası döngünün kendi
işidir. Otomasyonun bu dosyaya yazması yasaktır (§2, dokunulmaz liste).

## Amaç

Lubot'un eğitim çekirdeğini, insan müdahalesi olmadan, ölçülen bir kayıp
eğrisi üzerinde sürekli iyileştirmek — ve her adımı geri alınabilir, kanıtlı,
denetlenebilir tutmak.

## Bu oturumun yönü

1. **Öncelik: doğrulama kaybı.** Birincil metrik *en iyi doğrulama kaybı*dır
   (düşük iyi). Karşılaştırma eşleştirilmiş yapılır: aynı tohum, aynı adım
   bütçesi, aynı korpus ve aynı damga; yoksa sayılar karşılaştırılamaz.
2. **Sıra: önce mütevazı arama.** Öğrenme oranı ve ağırlık sönümü ekseninde
   küçük adımlarla başla; sıcaklık (warmup) ve yığın (yığın/accum) eksenleri
   ancak ilk eksen tükendiğinde açılır.
3. **Yasak yönler.** Aşağıdakiler bu oturumun yönü dışındadır ve bir öneri
   bunlara dokunuyorsa deney değil, insan kararıdır (durdurma koşulu S1):
   mimari (katman/genişlik/pencere), sözlük ailesi, korpus kaynağı,
   değerlendirme kümesi, donanım bütçesi.
4. **Bitiş çizgisi yok.** Oturum, durdurma koşullarından biri gerçekleşene ya
   da operatör durdurana kadar döner; her tur kendi kanıtını yazar.

## Kabul edilen kanıt

- Yerel doğrulama yeşili (fmt + clippy -D warnings + test + kapılar) bir
  ön koşuldur, kanıt değildir.
- Kanıt: commit SHA + CI koşu numarası. CI onaylanmadan hiçbir kayıt
  "iyileşme" demez; `kanit_durumu: ci-bekliyor` der.
- Metrik sayıları korpustan gelen sayılar değildir (model/kayıp ölçümüdür) ve
  kayıtta dururlar; korpus türevi sayılar bu dosyaya yazılmaz.
