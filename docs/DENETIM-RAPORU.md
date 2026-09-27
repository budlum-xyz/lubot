# Uyum Denetimi — 2026-09-24

İstenen: *"talimatlarıma ve tüm dosyalarıma harfiyen uyduğuna emin ol, gözden geçir."*
Bu dosya o denetimin kaydıdır. Her satır bir **kanıta** bağlı; kanıt yeniden
üretilebilir (bkz. §E). Kanıtsız iddia yok, ölçülmemiş hüküm yok.

Denetlenen ağaç: `budlum-xyz/lubot`, dal `egitim-dongusu`, head **`08d8d50`**,
uzakla senkron. CI: **run `36038074925` → success** (10/10 adım yeşil).

---

## A. Operatörün ayakta duran talimatları

| # | talimat | kanıt | durum |
| --- | --- | --- | --- |
| A1 | Verilen jetonla push et | `4f21f93..08d8d50 egitim-dongusu`; `ls-remote` = `08d8d50…` | ✅ |
| A2 | Jeton repoya/dosyalara **asla** yazılmaz | `gates/no-secret-material` yeşil; ağaçta ve `PUSH-TALIMATI.md`'de `ghp_` yok; jeton yalnız araç çağrısında | ✅ |
| A3 | Talimatlara ve dosyalara harfiyen uyum → gözden geçir | bu dosya; bulunan 2 kusur §C'de kapatıldı | ✅ |
| A4 | Jev/System One bağla ve çalış; anahtar yoksa **sistemi kendin yükle** | `/home/user/jev/` kurulu: `jev.py` + `laya_yerel.py` + `kararlar.json`; uç nokta anahtarsız 403 doğrulandı | ✅ (ölçüm: §D1) |
| A5 | Döngünün kararları (tut/at dahil) da Jev'e/Laya'ya girsin | **anayasa değişikliği** gerektirir → `INVARIANTS.md` madde + damga yenileme + yeni kapı + **operatör damgası** | ⏸ onay bekliyor |
| A6 | main'e asla doğrudan push yok; çalışma dalı `egitim-dongusu` | `git log --oneline main..egitim-dongusu` dolu; push'lar yalnız dala | ✅ |
| A7 | Quiet: raporlar sohbete değil dosyaya | `autonomous-training/kosum/oturum.md`, `hafiza/*.jsonl`, `kosum/*/rapor.md`; commit'li | ✅ |
| A8 | CI tek otorite | CI run `36038074925` yeşil; yerel ölçüm CI'ın ağaçtan kurduğu korpusla hizalandı (§C1) | ✅ |
| A9 | Durdurma koşullarında DUR + `ask_user` | S1–S5 `autonomous-training/README.md` §9; `kosum/DURUS.json` yok (aktif duruş yok) | ✅ |
| A10 | 3 yükleme belgesindeki tüm görevler bitecek (34 açık madde) | `YAPILACAKLAR.md` + `uploads/*.md` kutuları; 34 açık | ⏳ devam |
| A11 | Kapsam: L1–L4 döngü, K1–K6, S1–S5, 4 trafik koşulu, 4 katman | `ayarlar.json` bağlayıcı; §B tablosu | ✅ |

## B. Anayasa ve dokunulmazlar

| madde | nerede yazılı | onu tutan kapı | durum |
| --- | --- | --- | --- |
| K1 sıfırdan eğitim, adlar bize ait | `TRAINING.md` §K | `ratchet-holds`, `model-spec-is-consistent` | ✅ |
| K2 korpus = budlum yüzeyi, dış veri yok | `TRAINING.md`, `SECURITY.md` | `corpus-records-carry-provenance`, `unserved-records-never-cited` | ✅ |
| K3 büyüme yalnız kendi gelişimi + `doc` kapalı lisansla | `TRAINING.md` | `corpus-records-carry-licence`, `doc-pdf-feeds-corpus` | ✅ |
| K5 operatör anlaşması (geçiş değeri 2) | `TRAINING.md` | `operator-sync-rules` | ✅ |
| K6 donanım tavanı | `autonomous-training/README.md` §10 | `training-gate-epoch-ledger-fail-closed`, `k6_kontrol()` → `kosum/k6.json` | ✅ (tavan 97.565.184; spec 924.288) |
| no-generation | `INVARIANTS.md` | `reads-not-generates`, `no-generation-variant`, `decision-head-has-no-generation-surface`, `chain-surface-fixed` | ✅ |
| anayasa donmuş, damga tutar | `INVARIANTS.md` + `INVARIANTS.sha256` | `invariants-are-frozen`, `mutation-surface-is-closed` | ✅ |
| S1–S5 duruş + onay | `autonomous-training/README.md` §9 | `kirmizi-senaryolar`, `DURUS.json` akışı | ✅ |
| 4 katman; katman 4 kapalı | `ayarlar.json` + README §0/§7 | `architecture-doc-tracks-layer-rule` | ✅ |
| katman 3 yalnız arama politikasını değiştirir | README §6 | `kendini-gelistirme.json` + geri alma; `measurements-do-not-feed-back` | ✅ |

Stok (bu turda ölçülen): **82/82 kapı**, **538 test**, clippy pedantic 0, fmt temiz,
korpus **2861 kayıt / 200.367 jeton**, ratchet yedi anahtarda tutuyor.

## C. Denetimin bulduğu iki gerçek kusur ve kapanışı

### C1 — CI kırmızıydı: bayat ölçüm (kapatıldı)
CI run `36031303585` **failure**: adım 9 (kayıt kapıları). Kök neden: ağaç
değiştikçe türetilmiş kayıtlar bayatlıyor ve CI korpusu **ağaçtan yeniden
kuruyor**; yerelde kalan bayat `corpus/` artefaktı yüzünden kapı yerelde yeşil,
CI'da kırmızı görünüyordu. Üç kapı düştü: `language-cost-is-declared` (tr 89/138),
`data-mix-is-declared`, `comparison-class-is-declared` (jeton bütçesi).

Kapanış: kayıtlar ağaçtan yeniden üretildi (`cok_dillilik --kur`,
`veri_karisimi --kur`, `kiyas_sinifi --kur`), korpus yeniden kuruldu
(2791 → 2861 kayıt; en 1799 → 1802), yerel ağaç CI ile aynı sayıları verir hâle
geldi. **Temiz klonda** (yalnız commit'li içerik) tüm kapılar yeşil: run
`36038074925` success. Kural olarak yazıldı: *ağaç = commit*; izlenmeyen koşu
kaydı bırakılmaz.

### C2 — Güvenlik tarayıcısı Türkçe metinde panikliyordu (kapatıldı)
`crates/tools/src/secrets.rs:54`: `rest[..MIN]` sabit **bayt** dilimi.
`ü` iki bayt olduğu için pencere karakterin içine düşüyor ve
`byte index 20 is not a char boundary` ile **panik**. Etki: `veri_karisimi.py`
tarayıcıyı çağırınca koşu komple düşüyordu — fail-**silent**, fail-closed değil.

Kapanış: `head_after()` yardımcısı (ilk *n karakter*, `char_indices` sınır
kontrolü) beş çağrı yerine ve `pem_block`'a bağlandı; regresyon testi
`multibyte_text_does_not_panic_the_scanner` eklendi (test 537 → 538);
`README.md` 538'e, `docs/CRATES.md` tools 1760/48'e tazelendi.
Kanıt: `cargo test -p lubot-tools secrets` → 5/5.

## D. Jev / System One — ölçüm özeti (ayrıntı: `/home/user/jev/KARAR.md`)

| deney | sonuç |
| --- | --- |
| Uç nokta, anahtarsız `GET /v1/models` | 403 `authentication_error` → ağ açık, anahtar yok |
| Yerel Laya (Apache-2.0), TR batarya 8 madde | ham oy **5/8**, kapılı **5/8** |
| Yerel Laya, EN batarya 8 madde | ham oy **4/8**, kapılı **4/8** (yalnız kapı 1 maddeyi daha düşürdü) |
| Hata yönü | **hep "evet/act" yönünde**: riskli işi güvenli sayıyor, yetersiz kanıtı "bitti" sayıyor |
| TR noul | **dejenere** (hep ~1,0); EN'de ayrım var |

Hüküm: yerel model **karar katmanı olarak kullanılmaz**. Yazılım ve ölçüm duruyor;
`TYPESAFE_API_KEY` gelince aynı kartlar `--arka-uc jev` ile koşar ve aynı batarya
yeniden ölçülür (karşılaştırma tek komut).

## E. Yeniden üretilebilir doğrulama

```bash
# 1) kapılar (82) ve stok
python3 gates/check.py --all

# 2) test / biçim / lint
cargo test --workspace && cargo fmt --all --check && \
  cargo clippy --workspace --all-targets -- -D warnings

# 3) CI'ın yaptığını yap: korpus ağaçtan, sonra kapılar
python3 training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz
python3 gates/check.py --all

# 4) yerel karar katmanı bataryası (Jev/Laya)
cd /home/user/jev
python3 yerel_kiyas.py --kaydet
python3 yerel_kiyas.py --batarya kanaat_bataryasi_en.json --alt-klasor "" --dtype float16

# 5) anahtar gelince: aynı kartlar, uzak arka uç
export TYPESAFE_API_KEY=...   # yalnız ortam değişkeni; dosyaya yazılmaz
python3 jev.py --kart sonraki_madde --durum-dosya durum/kuyruk.json --gonder --arka-uc jev
```

## F. Açık kalanlar

1. **Onay bekleyen tek madde (A5):** döngünün ölçüt/durdurma kararlarının
   (tut/at dahil) Jev/Laya'ya bağlanması. Anayasa değişikliği; damga yenileme ve
   yeni kapı gerektirir. Onaylanmadan uygulanmadı. Önerilen metin ve karşı
   gerekçe `/home/user/jev/KARAR.md` §4'te.
2. 34 açık madde (fuzzing, QA tip skoru, IAM kıyası, CI/CD, README.tr, markdown
   şeması, kamu malı, yanlış inanışlar, empirik SE …).
3. `TYPESAFE_API_KEY` gelince: bataryanın uzak arka uçla yeniden ölçümü (aynı
   kartlar, aynı eşikler) ve karşılaştırma tablosunun `defter/`e işlenmesi.


---

# Ek — aynı gün, ikinci tur (head `a97325d`, CI `36043827896` success)

## Kimlik düzeltmesi (operatör talimatı)

İki commit `lubot dongusu <dongu@lubot.local>` kimliğiyle atılmıştı; operatör
"commit'leri benim kimliğimle atacaktın" dedi. Commit'ler `--reset-author` ile
**Ayaz `<ayazkussann@gmail.com>`** kimliğine çevrildi, ağaç birebir aynı kaldı
(`tree` hash'i değişmedi) ve dal `--force-with-lease` ile push edildi (main'e
dokunulmadı). Bundan sonra kimlik `~/.gitconfig`'e yazılıdır.

## Anayasa değişikliği: K7 (operatör onaylı)

"Damgala ve uygula" onayı üzerine: `INVARIANTS.md`'ye **K7** maddesi eklendi,
damga yenilendi (`c2f8ce3f5e0e`), S2 tanımı danışma kararını da kapsayacak şekilde
genişletildi, `autonomous-training/danisma.py` + `training/danisma/sunucu.py`
yazıldı, kapı `danisma-layer-is-closed` (84. kapı) eklendi.

**Sınır korunmuştur:** model oy verir, karar kuralı (eşik, marj, tut/at) kodda
kalır; oy yok/geçersiz/probu düşükse karar insana gider ve döngü S2 ile durur.

## Danışma katmanı ölçümü (depo içi, arka plan servisi)

| ölçüm | değer |
|---|---|
| model yükleme (tek sefer, soğuk) | 10,2 s (ilk kez 16,4 s) |
| oy gecikmesi (servis sıcak) | **351–457 ms** |
| tepe RSS | ~1,4 GB (bf16; fp32 bu makinede OOM) |
| maliyet | $0 |
| **oy kalitesi** | **sabit "tut"**: 6,5 / 6,67 / 9,0 skorlarının üçünde de "tut" (güven 0,61–0,68) |
| kalibrasyon probu | **1/2** → oy geçersiz → **karar insana** (çıkış kodu 1) |

Yani K7 bağlı ve çalışıyor; bu makinedeki yerel model **kendi oyunu geçersiz ilan
ediyor**. Jev anahtarı gelince aynı kartlar ve problar uzak arka uçla koşacak.

## Yol boyunca yakalanan dört hata (hepsi kanaryaya bağlandı)

1. **`choice` etiketi metin değil anahtar** (`o0`) → oy `None` düşüyordu; eşleme
   artık *bizim gönderdiğimiz* anahtarla birebir, sayıdan taban tahmini yok.
2. **`not prob["gecen"]` — sayaç boolean sanıldı**: prob düştüğü hâlde karar oyla
   verildi; kanarya yalnız prob fonksiyonunu sınadığı için görmedi. Artık karar
   `prob["gecti"]`ye bağlı ve kanarya **karar yolunu** da sınıyor.
3. **Kapı yorumla ateş etti**: hatayı anlatan yorum "sayaca bağlanmış" denetimini
   kırdı; kapı artık yalnız kodu okur (yorumlar sıyrılır).
4. **Ölçüm kaydı biçimi**: fuzzing kaydı depo standardında değildi
   (`eval-runs-are-mechanical` reddetti) → tek mekanik boolean ölçüt + kaynak
   muhasebesi biçimine çevrildi.

## Kuyruk

Bu turda **7 madde** kapandı (22, 23, 24, 25, 28, 29 + K7), 4 madde sıraya
girdi (21, 26, 27, 3), kalanlar engelli ya da operatör kararı bekliyor.
Tam döküm: `/home/user/KUYRUK-DURUM.md`.
