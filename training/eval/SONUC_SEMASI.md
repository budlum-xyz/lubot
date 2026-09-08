# Sonuç Şeması: Bir Değerlendirme Koşusu Nasıl Ölçülür

Kalıp kaynağı: üçüncü taraf mekanik ölçüt şeması (kavram uyarlaması);
kod alınmadı. Oradaki her ajan koşusu tek bir makine doğrulamalı başarı
ölçütüyle (sonucun bayrağı çıkarabildiği gerçeği) ve kaynak muhasebesiyle
(jeton, maliyet, süre) kapanır. Bu şema aynı disiplini Lubot
değerlendirmelerine uyarlar.

## Kurallar

1. Tek ölçüt. Her koşu tam olarak bir ölçüt taşır: `olcut.ad`
   (makine doğrulamalı ad) ve `olcut.sonuc` (yalnızca true/false).
   Kısmi puan, izlenim ve "büyük ölçüde" yoktur.
2. Ölçüt adı davranışı söyler, yargıyı değil. "iki bağımsız inşanın
   özet eşitliği" bir ölçüttür; "çıktı iyi görünüyor" değildir.
   Yargı kelimeleri (iyi, güzel, başarılı, sorunsuz) kapı tarafından
   reddedilir.
3. Kaynak muhasebesi zorunludur: `sure_saniye`, `girdi_jetonlari`,
   `onbellekli_jetonlari`, `cikti_jetonlari`, `maliyet`. Betik
   koşucuları jeton ve maliyet alanlarını sıfır bildirir; model
   koşucuları sıfır çıktı jetonu bildiremez.
4. Kanıt alanı ölçütün nasıl doğrulandığını tek cümleyle söyler.
5. Koşu kayıtları `training/eval/sonuclar/` altında tek JSON dosyası
   olarak yaşar ve `eval-runs-are-mechanical` kapısı her commit'te
   şemayı denetler.

## Örnek

```json
{
  "kosucu": "betik",
  "tarih": "2026-09-08",
  "is": "corpus-build-is-deterministic",
  "olcut": {
    "ad": "iki_bagimsiz_korpus_insasinin_sha256_esitligi",
    "sonuc": true
  },
  "kaynaklar": {
    "sure_saniye": 4.2,
    "girdi_jetonlari": 0,
    "onbellekli_jetonlari": 0,
    "cikti_jetonlari": 0,
    "maliyet": 0.0
  },
  "kanit": "build_corpus.py iki kez bağımsız koşuldu; sha256 özetleri eşit."
}
```
