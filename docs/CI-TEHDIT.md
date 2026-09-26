# CI/CD tehdit notu — tedarik zinciri ve push yüzeyi

`it` komutunun kısıtlı push tasarımı, tedarik zinciri kaygısıyla **aynı yere**
bakar: kim, hangi yoldan, neyi yazabiliyor. Bu belge o bakışı tehdit başına
yazar ve her tehdit için "kapı var mı, yok mu" sorusunu dürüstçe cevaplar.
Ölçülen sayıları yeniden yazmaz; kanıt kapı adları ve `gates/check.py`'dir.

## Varlıklar

| varlık | neden hedef |
|---|---|
| `gates/check.py` ve `training/*.py` | Kapılar ve ölçüm; buradaki bir gevşetme her şeyi gevşetir. |
| `autonomous-training/INVARIANTS.md` + damga | Anayasa; değişirse döngü başka bir döngü olur. |
| `.github/workflows/ci.yml` | CI'ın ne koştuğu; tek otorite burasıdır. |
| `training/ratchet.json` | Tabanlar; düşürülürse regresyon sessizleşir. |
| Push kimlik bilgisi (jeton) | Depoya yazılırsa sızıntı; `no-secret-material` kapısı bunu arar. |

## Tehditler ve durum

| tehdit | saldırı yolu | mevcut savunma | durum |
|---|---|---|---|
| **Yazma yüzeyini genişletmek** | Yeni bir betik `guvenli_yaz` dışında dosya yazar | `invariants-are-frozen` içindeki yazma-kapısı taraması (`_yazma_kapisi_bulgu`) otomasyon ağacında kapı dışı yazımı yakalar | ✅ |
| **Dokunulmaza yazmak** | `INVARIANTS.md` ya da `ayarlar.json`'a yazan satır | `_dokunulmaz_yazma_bulgu` + `yazma_reddi` + kanarya; girişim **S2** duruşudur | ✅ |
| **Anayasayı sessizce değiştirmek** | Dosyayı düzenleyip damgayı yenilememek | `invariants-are-frozen` özeti yeniden hesaplar; damga tutmazsa döngü başlamaz | ✅ |
| **Kapıyı gevşetmek** | Bir kapıyı silmek ya da `--self-test`i boşaltmak | `gate-pairs-carry-referee`: her kapının kanaryası olmalı ve kanarya **reddetmeli**; `ratchet-holds` kapı sayısının düşmesini yasaklar | ✅ |
| **Tabanı düşürmek** | `training/ratchet.json`'da sayıyı küçültmek | `ratchet-holds` yalnız yükselmeye izin verir; `ratchet --set` yabancı anahtarları korur | ✅ |
| **Kimlik bilgisini sızdırmak** | Jetonu bir betiğe/dosyaya gömmek | `no-secret-material` + `credential-shapes-are-measured` (kapalı liste, tam uzunluk; anma sızıntı değil) | ✅ |
| **Push'un kapsamını genişletmek** | `it` dışındaki bir yolun commit edilmesi | `it-is-restricted`: yalnız listelenen yollar işlenir ve push edilir | ✅ |
| **Korpustan veri sızdırmak** | Üretilmiş kaydı korpusa sokmak | `no-generation` + `unserved-records-never-cited` + karma kapalı lisans kümesi | ✅ |
| **CI'da sahte yeşil** | Kapıyı koşmadan "geçti" demek | CI adımları kapıları doğrudan koşar; yerel bayat ölçüm CI'da kırmızıya döner (2026-09-24'te oldu: `language-cost-is-declared`) | ✅ |
| **Bağımlılık üzerinden yürütme** | Yeni bir paket, `build.rs` içinde kod koşar | Yeni bağımlılık eklenmiyor; `dependencies-are-used` kullanılmayanı, `cargo` kilidi sürümü tutar. **Kısmi:** bağımlılık denetimi (audit) aracı yok | ⚠️ |
| **İş akışı enjeksiyonu** (`pull_request_target`, kullanıcı girdisiyle `run:`) | Workflow'da ifade enjeksiyonu | Workflow yalnız kendi betiklerini koşar, olay verisini `run:` içine gömmez | ✅ (gözle doğrulandı) |
| **Önbellek zehirlenmesi** | Kopyalanabilir bağımlılık önbelleği | `actions/cache` kullanılmıyor; araç zinciri sabit sürümle kurulur | ✅ |
| **Uzun ömürlü jeton** | Depoda duran bir jetonun sızması | Jeton yalnız tek push için araç çağrısında kullanılır, dosyaya yazılmaz; `~/.gitconfig` credential helper `.git` dışında tutar | ✅ |
| **Etiket/sürüm taklidi** | Zararlı bir sürümün "resmî" gibi push edilmesi | Sürüm yayınlama akışı **yok**; tek dal `egitim-dongusu`, `main`'e doğrudan push yok | ✅ (yoklukla) |

## Kapsam dışı bırakılanlar ve gerekçesi

- **Bağımlılık güvenlik denetimi (audit/vet aracı)**: ek araç, CI'ı zayıflatma
  riski taşıdığı için alınmadı (statik analiz maddesindeki kararın aynısı).
  Kayıt burada duruyor; karar değişirse bu satır güncellenir.
- **İmzalı commit / SLSA kanıtı**: bugünkü tehdit modelinde tek operatör var ve
  imza anahtarı yönetimi yeni bir sızıntı yüzeyi açardı. Karar, K5 operatör
  anlaşması maddesiyle birlikte yeniden değerlendirilecek.
- **Sır tarayıcısının dış repo taraması**: `no-secret-material` `crates/`,
  `training/`, `gates/` ağaçlarını tarar — korpusa giren yüzey orasıdır.
