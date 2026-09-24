# Lisansı temiz kaynak adayları — liste ve karar alanları

Madde 27 (Awesome Uncopyright / Public Datasets). K3 gereği korpus **yalnız**
budlum yüzeyinden beslenir: izinli lisans kümesi dışındaki hiçbir kayıt kapıdan
girmez. Bu belge yeni kaynak **kabul etmez**; aday listesini ve kararın hangi
alanlara bakacağını hazırlar. Her satır bir *öneridir*, karar operatöründür.

Alanlar (her aday için doldurulacak): **lisans** (kapalı küme uyumu), **provenance**
(asset_id + content_id çifti üretilebilir mi), **ATTRIBUTION** (atıf metni),
**korpus katmanı** (kendi eser / kamu malı / izinli), **K2 uyumu** (budlum yüzeyi
mi, değilse neden girebilir), **K6** (jeton maliyeti bütçeye sığar mı).

## Aday sınıfları

| sınıf | örnek kaynak tipi | lisans | K2'ye göre durum |
|---|---|---|---|
| **Kamu malı standart metinler** | RFC/IETF belgeleri (IETF Trust), W3C spesifikasyonları, Unicode standart ekleri | kopyalama serbest, bazıları atıf ister | ⚖️ karar: dış kaynak; K2 istisnası açık bir K3 gerekçesi ister |
| **Kamu malı dil verisi** | kamu malı sözlükler, tarihsel metinler (Project Gutenberg'in kamu malı kümesi) | kamu malı | ⚖️ dış kaynak; jeton maliyeti K6'ya çarpar |
| **Açık lisanslı teknik belge** | CC0 / CC-BY-4.0 lisanslı yazılım mimarisi notları | CC0 uyumlu; CC-BY atıf ister | ⚖️ izinli küme bugün PolyForm + MIT; **genişletme kararı** |
| **Kendi eserimiz** | bu depo, budlum çekirdeği, çalışma alanı kök belgeleri | PolyForm Shield 1.0.0 | ✅ bugünkü tek yol |
| **Kendi ürettiğimiz türev** | korpustan öz-damıtma (G maddesi), sentetik **0** kararıyla | kendi eser | ✅ (çoğaltma kendinden; dış öğretmen yok) |

## Karar alanlarının bugünkü değerleri

- **İzinli lisans kümesi**: `crates/tools` içindeki kapalı küme (PolyForm Shield,
  MIT). Yeni bir lisans girmesi **kod değişikliği** ve kapı güncellemesi ister;
  belgeyle değişmez.
- **Korpus katmanı**: self korpusu (bu ağaçtan, CI'da) + operatör tarafında
  yüzey korpusu. Üçüncü bir katman bugün yok.
- **Jeton bütçesi**: K6/ratchet tarafında ölçülür; dış kaynak eklemek
  `ratchet-holds` altındaki jeton sayısını yükseltir ve `data-mix-is-declared`
  karışımı yeniden beyan ister.
- **Sentetik veri**: **0** (karar verilmiş; `veri_karisimi.py` bunu ölçer).

## Neden bu liste "iş bitmiş" sayılmaz

Bir adayın listeye girmesi, korpusa girmesi değildir. Kapıda bugün izinli küme
kapalıdır; liste açık kalan tek şeyi görünür kılar: **hangi dış kaynağın hangi
lisansla ve hangi atıfla kabul edileceği bir operatör kararıdır**, ve karar
verilmeden ölçüm düzenine dokunulmaz.

Bu belge ölçülmüş sayı taşımaz (korpus geri-besleme kuralı); sayılar
`training/ratchet.json`, `training/eval/sonuclar/veri-karisimi-2026-09-23.json`
ve `corpus/` altındadır.
