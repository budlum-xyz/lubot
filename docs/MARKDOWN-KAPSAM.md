# Markdown şeması — kural kapsama dökümü

Cevap yüzeyinin şeması `crates/read/src/output_schema.rs` içindedir ve
`ai-output-schema-enforced` kapısı **varlığını** denetler. Bu belge farklı bir
soruyu cevaplar (madde 26): *kuralların her biri kendini hedefleyen bir girdiyle
gerçekten reddediyor mu, ve kapsanmayan bir durum kaldı mı?*

Ölçüm `training/sema_kapsam.py` ile yapılır; kural listesi **Rust kaynağından
okunur** (elle yazılmaz), her kural için tek bir kötü örnek üretilir ve
`lubot prompt --path` üzerinden denenir — o komut cevabın geçtiği **aynı** şema
doğrulayıcısından geçer. Kayıt:
`training/eval/sonuclar/sema-kapsam-2026-09-24.json`. Kapı:
`markdown-schema-is-covered` (85. kapı).

## Kurallar ve kapsama (ölçülmüş)

| kural | kötü örnek | hangi kapı reddediyor | durum |
|---|---|---|---|
| `Empty` | boş belge | şema (`output is empty`) | ✅ |
| `HeadingSkip` | `#` sonra `###` | şema (`skips a level`) | ✅ |
| `UnbalancedFence` | kapanmamış ``` | şema (`unbalanced code fence`) | ✅ |
| `TableMismatch` | satır hücre sayısı uyuşmayan tablo | şema (`malformed table`) | ✅ |
| `TooLarge` | 64 KiB + 1 bayt | şema (`output too large: … bytes`) | ✅ |
| `NotUtf8` | bozuk UTF-8 baytları | **okuma kapısı** (`stream did not contain valid UTF-8`) + kasa testi `invalid_utf8_is_refused` | ✅ (atıflı) |

Sonuç: **6/6 kural hesap veriyor**; 5'i şema kapısında ısırıyor, `NotUtf8` ise
şemaya varmadan okuma kapısında reddediliyor ve gerekçesi kasa testine
atfediliyor. Kapsanmayan **durum yok**, kanıtsız atıf yok.

## İki kapsam tuzağı (ikisi de ölçüm sırasında yakalandı)

1. **"İfade bulundu" ≠ "o kapı reddetti".** Bozuk UTF-8 mesajı da "valid UTF-8"
   içerir; ifadeyi tek başına aramak okuma kapısındaki reddi *şema kapsamı*
   saydı. Ölçüm artık iki kapıyı ayırır: şema reddi için `rejected by schema`
   de, ifade de aranır. Kanarya bu ayrımı kilitler.
2. **Vakası olmayan kural sessizce kapsanmış görünür.** Kural listesi kaynaktan
   okunmazsa, yeni eklenen bir ret dalı ölçümde hiç görünmez. Bu yüzden liste
   koddan okunur ve kayıtla karşılaştırılır; eşleşmezse kapı "kaydı yeniden
   üret" der. Kanarya: vakası silinen kural `kapsanmayan` sayılır.

## Şemanın kapsamadığı şeyler (bilerek)

- **İçerik doğruluğu**: şema *biçimi* doğrular, doğru cevabı değil. Doğruluk
  tarafı `ask` yolundadır (alıntı zorunluluğu, `NotFound` birinci sınıf cevap).
- **Dil bilgisi / imla**: imla denetimi yok; Türkçe metin ayrıştırıcıdan geçer,
  yargılanmaz.
- **HTML/LaTeX gömme**: kapsam dışı; Markdown'ın dört sınıfı ve tablo kuralı
  yeterlidir.
