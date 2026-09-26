# Alım hattı — arayüz, alanlar ve retler

Bu belge `crates/alim` hattının ne yaptığını, hangi alanda hangi kuralı
uyguladığını ve bir redin hangi kuralı adlandırdığını yazar. Kapı:
`gates/check.py → alim-hatti-kapali`; kapının kendi self-test'i vardır.

## Ne için

Uygulama direktifinin altıncı bölümü dört parça istiyor: alım (ingestion),
alımın beslediği eğitim adımı, kayıt-bazlı provenance ve donanım. Bu hat
birinci ile üçüncünün **depo tarafıdır**: baytlar geldikten sonra, eğitim
adımından önce duran kapı. Ne korpus okur, ne eğitir, ne zincire bağlanır —
gelen baytı kuralına karşı ölçer ve nereden geldiğini yazar.

Hattın üç kapısı vardır ve her kapı tek bir soruyu cevaplar:

| kapı | soru | modül |
|---|---|---|
| manifest kabulü | gelen manifest, korpusun kabul kurallarını geçiyor mu? | `manifest` |
| provenance defteri | bu kayıt hangi manifestten, hangi yükleyiciyle, hangi adıma girdi? | `provenance` |
| ağırlık manifesti | nesnenin içerik adresi, bölünme planı ve yerleşimi tutarlı mı? | `agirlik` |

## `lubot alim dogrula` — hangi fonksiyon tetiklenir, hangi alan okunur

Komut satırı: `lubot alim dogrula --manifest <f.json> --veri <dir> --defter <f.jsonl> [--adim n]`

| adım | fonksiyon | okunan alanlar |
|---|---|---|
| manifest baytları | `std::fs::read` | `--manifest` yolu |
| ayrıştırma | `manifest::parse` | `schema`, `manifest_id`, `source_class`, `loader`, `created_at`, `records[]` |
| kayıt baytları | `resolve_records` + `std::fs::read` | her kaydın `path` alanı, `--veri` kökü altında |
| kabul | `manifest::admit` | `digest`, `content_id`, `asset_id`, `kind`, `licence`, `attribution`, `path` |
| defter satırları | `provenance::rows_for` | `manifest_id`, `loader`, `verified_at`, `admitted_step`, `content_id`, `path`, `kind`, `licence` |
| defter yazımı | `provenance::append` | `--defter` yolu; satır başına bir JSON nesnesi |
| çıktı | `lubot::validate_output` | şema doğrulamalı Markdown, tek çıkış |

Defter **okunmadan** hiçbir bayt kabul edilmez: `--adim`, defterin son
satırındaki adımın gerisindeyse ret, manifest dosyası hiç açılmadan verilir.
`--defter` verilmezse komut çalışmaz: kaydedilmeyen bir alım, alım değildir —
"bu veri hangi adıma girdi" sorusunun cevabı kalmaz (K3).

## Kabul kuralları ve retler

Her ret, uyguladığı kuralı adıyla taşır (`[K2]`, `[K3]`, `[O]`, `[-]`).

| kural | ret | ne zaman |
|---|---|---|
| K2 | `UnknownSourceClass` | `source_class` kapalı kümenin dışında (`self_tree`, `doc_admitted`). **En yakın üyeye düşürülmez**: sessiz düşürme, kimsenin onaylamadığı bir genişletmedir. |
| K3 | `UnknownLicence` | `licence`, kapalı kümenin (`MIT`, `Apache-2.0`, `PolyForm-Shield-1.0.0`) dışında |
| K3 | `MissingAttribution` | `attribution` boş |
| O | `MissingProvenancePair` | `content_id` / `asset_id` çifti eksik |
| O | `BadManifestId`, `BadDigest` | alan bir `sha256` onaltılık özeti biçiminde değil |
| O | `DigestMismatch` | baytlar, manifestin iddia ettiği özete hash'lenmiyor |
| — | `UnsafePath` | `path` mutlak, boş ya da ağaçtan yukarı çıkıyor |
| — | `MissingBytes` | kaydın yolu için bayt çözülemedi (atlama yok, red var) |
| — | `UnsupportedSchema`, `EmptyManifest`, `MissingLoader`, `Malformed` | manifest düzeyi |
| — | `MalformedLedger`, `StepWentBackwards`, `Io` | defter düzeyi |
| — | `EmptyArtefact`, `BadErasureParams`, `ShardOutOfRange`, `TooFewHolders`, `HolderOverload` | ağırlık düzeyi |

Manifest **bütün olarak** kabul edilir ya da hiç kabul edilmez: ilk reddedilen
kayıt, koşunun tamamını reddeder. Bilinmeyen alan taşıyan manifest de reddedilir
(`deny_unknown_fields`): bu derlemenin hesabına katamadığı bir alan, kimsenin
denetlemediği bir alandır.

## Ağırlık manifesti: ne hesaplanıyor, ne hesaplanmıyor

`agirlik` modülü üç şeyi **hesaplar**:

1. İçerik adresi — `weight_id(bytes)`, nesnenin kendi baytlarının `sha256`'sı.
2. Bölünme planı — `plan(EcParams, total_len)`: shard boyu, veri shard sayısı,
   toplam shard sayısı ve **beyan edilen dolgu** (`pad_len`). Her shard aynı
   uzunluğa doldurulur, böylece shard özeti hangi shard olduğuna bağlı olmaz
   ve yeniden birleştirme, dolgu düşülerek doğrulanabilir.
3. Yerleşim kuralı — `place(plan, holders)`: hiçbir tutucu, kodun
   kaybetmeye dayandığından fazla shard tutamaz. Kural `data` + `parity`
   shard ve `parity` kayıp toleransı için en az `1 + ceil(data / parity)`
   tutucu ister; sağlanmıyorsa yerleşim **reddedilir**, çünkü tek bir
   tutucunun kaybı nesneyi kaybettirir.

**Hesaplanmayan:** eşlik (parity) shard'larının kendisi. Bu turda plan ve
yerleşim kuralı yazıldı, kod çözücü yazılmadı; Reed-Solomon kodlayıcı
yazılmadan "erasure-coded" demek ölçülmemiş bir iddia olurdu. Aynı şekilde
yerleşim bir **kuraldır**, ağ işlemi değildir: hiçbir tutucuya bağlanılmaz.

## K2'nin bu hattaki okuması

Direktifin beşinci bölümündeki K2 - *"korpus yalnızca budlum-xyz yüzeyi:
Lubot'un ağacı, budlum ve workspace dahil; dışarıdan hiçbir şey girmez"* - bu
hat için bağlayıcı kabul edildi (operatör kararı, 2026-09-26: taban `main`,
K2 direktif sürümü). `source_class` kapalı kümesi tam olarak bu yüzeyi
yansıtır ve korpustaki kaynak adlarıyla aynı sözcükleri kullanır:

| sınıf | ne | nereden |
|---|---|---|
| `lubot` | bu depo | `build_corpus.py --repo .` |
| `budlum` | budlum çekirdek ağacı | operatör tarafı kaynak manifesti |
| `workspace` | workspace kök belgeleri | operatör tarafı kaynak manifesti |
| `doc` | K3'ün `doc` yoluyla kabul edilen kayıtlar | `lubot doc` |

Hattın hiçbir yolu bu kümeyi genişletemez: yeni bir sınıf eklemek, anayasa
değişikliğiyle aynı sınıfta bir iştir ve kod içinde yapılabilecek bir şey
değildir. Özellikle **depo dışı kaynaklar** (sahibine ait ya da kamu malı
sınıfı) bu sürümün kapsamı dışındadır; böyle bir genişletme K2'nin metnini
değiştirmek demektir. Taşıma katmanı (B.U.D.) bu sınıflandırmayı değiştirmez:
yeni bir manifest, sınıfı kümenin dışındaysa adıyla reddedilir.

## Varsayım kaydı (direktif 3.1)

- **Görev yorumu.** Operatör "oradan devam et" dedi; bu, devam eden akışın
  bıraktığı noktadan (mimari port tasarım notları; NN adımları 10-13) sonraki
  adımlar olarak okundu: NN 14-16, yani alım arayüzü, provenance şeması ve
  ağırlık manifesti. Tasarım notlarının kendisi başka bir açık PR'da
  yürüdüğü için burada tekrar yazılmadı.
- **Taban dal.** `main`. Bunun bedeli bilinerek kabul edildi: `main`'de eğitim
  çekirdeği (`crates/egitim`), `nicem`, `tasiyici` ve `autonomous-training/`
  yok; onlar açık PR'larda. Bu hat hiçbirine bağlı değildir ve `main`'in kendi
  kapılarıyla yeşildir.
- **Sayılar.** Bu belgede ölçülmüş gibi yazılmış hiçbir sayı yoktur; ölçümün
  tek kaynağı `training/ratchet.json` ile kapı çıktılarıdır.
