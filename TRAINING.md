# Lubot training pipeline

Decisions behind the pipeline, the measured corpus, the gates, and what is
still open. Everything the pipeline learns from is Lubot's own tree: the
corpus is built from this repository and from nothing outside it.

## Decisions

| # | Decision |
|---|---|
| K1 | Base model: trained from scratch. The served names are ours; no upstream name is used as a served name anywhere in this repository. |
| K2 | The corpus is the budlum-xyz surface: this repository always, and (operator-side, via a sources manifest) the budlum core plus workspace root documents, each source stamped with its own licence and per-repo provenance. Nothing from outside the organization's own trees enters. (2026-09-22: the single-tree rule widened to the surface; workspace subdirectories enter only after curation.) |
| K3 | Growth policy: the corpus grows by the repository's own development and by documents admitted through the `doc` command under the closed licence set, with per-record provenance. |
| K4 | Tier -2 (STARK-provable) path: verify-only, evaluated separately for high-stakes outputs; not part of the main training plan. |
| K5 | Operator agreement: target 1/1 when the zkVM content proof is live for this model class; until then the transition value is 2 (a single-operator outcome is not consumed). See the transition note in the workspace record. |
| K6 | Compute for the from-scratch run: the owner's own hardware. Model size is therefore capped by measured hardware; the measurement is the next step. |

## Pipeline

```
this repository
   -> training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz
      (per record: path, lines, digest, licence, asset_id, content_id;
       a chunk that fails refusal never enters)
   -> training/train_tokenizer.py --corpus corpus/knowledge-self.jsonl.gz
      --out training/tokenizer/lubot-bpe-v1.json
      (frozen, versioned byte-level BPE; a new vocab is a cut, never a drift)
   -> (operator, surface) training/build_corpus.py --sources manifest.json
      --out corpus/budlum-yuzeyi.jsonl.gz
      (budlum + workspace root documents + this repository; per-source
       asset_id, licence, attribution; cross-source dedup)
   -> training/epoch_ledger.py --check ledger.json --now <block>      (fail-closed grant)
   -> training/make_sft.py --corpus ... --curriculum training/curriculum --out corpus/sft.jsonl
      (grounded rows + curriculum rows; a row that cannot cite is dropped)
   -> training run (from scratch; compute = owner hardware, measured next)
```

Both writers produce real gzip when `--out` ends in `.gz`. Every record
carries `asset_id` + `content_id`; pre-issuance canonical asset id with
`asset_id_pending: true` until a chain TrainingDataGrant is issued.

## Measured corpus

One corpus, one source: this repository, rebuilt by CI on every run.

| file | records | kinds | tokens (approx) |
|---|---|---|---|
| `corpus/knowledge-self.jsonl.gz` (CI, this repository) | 893 | api 293 · behaviour 199 · doc 300 · markdown 101 | ~31K |
| `corpus/karisim.jsonl` (NN-4: real + synthetic + compiler-refereed + curriculum) | 1138 | doc 300 · api 293 · behaviour 199 · markdown 101 · synthetic 152 · compiler-refereed 5 · curriculum 88 | ~40K |
| `corpus/eval-only.jsonl` (PP: held-out, never trained) | 113 | mixed | ~4K |
| `corpus/budlum-yuzeyi.jsonl.gz` (operator, sources manifest: lubot + budlum + workspace root) | 23604 | api 6015 · behaviour 4684 · doc 7323 · markdown 5582 | ~1.2M |

Every record carries the provenance pair; the provenance gate measures 100%
of the corpus files present. Refused-at-the-door is measured by the same
gates that load the corpus: a single bad record fails the load.

Chain core: the Rust side is `crates/tools/src/chain.rs` - it builds the
JSON-RPC request, parses a `bud_aiGetOutcome` response (missing text is a
refusal, never a guess) and emits the exact one-line record format the
corpus builder consumes. Chain records enter as `kind: doc` (chain analysis
text under `chain/<type>`, `request_id` kept), licence PolyForm Shield
1.0.0, own work.

## Gates (in `gates/check.py`, each with a self-test)

| gate | what it refuses |
|---|---|
| `reads-not-generates` | No generation surface exists, and the reading path has a refusal for everything it cannot open |
| `no-fourth-channel` | Content enters through exactly three channels, and a fourth is refused |
| `provenance-fails-closed` | A record's bytes are checked against its digest, and a mismatch refuses |
| `mask-before-storage` | The secret mask is applied on the write path, not on the way out |
| `no-panic-path` | `unwrap` and `expect` are denied outside tests, and the denial is real |
| `readme-is-measured` | The test count in the README is the count the suite reports |
| `system-prompt-is-true` | The Budlum-specific system prompt states only measured facts: the four ceilings, the eight RPC names, the effort range, the threshold, the ai-inference tag - and no superlative or proof claim that nothing here produced |
| `operator-sync-rules` | The report's operator rules are checks Lubot can run: non-zero bond above the floor, one model_hash among active operators, effort tier hashed into the request within 0.5x-10.0x, and a checkpoint transition window with a real retirement moment |
| `output-finalize-closed-loop` | A finalized output is sealed only after schema validation, carries the fixed 'ai-inference' tag, and the answer type has no binary/visual/video return variant to begin with |
| `cli-asks-and-renders-markdown` | The binary answers a real question against a real file and prints a Markdown document as its only stdout, and the tool route answers arithmetic without touching the index |
| `corpus-records-carry-licence` | Every corpus record carries an allowed licence and an attribution |
| `no-multiplier-labels` | The effort tier range is the education report's canonical naming (0.5x-10.0x) and is stated in the README; short or uppercased variants (10x, 0.5X) never appear as labels |
| `training-gate-epoch-ledger-fail-closed` | The epoch ledger refuses an exhausted or expired grant |
| `corpus-records-carry-provenance` | Every corpus record carries asset_id and content_id (AÇIK-4 rule) |
| `training-grant-crate-validates` | The grant crate holds the epoch rules; the scripts are not the authority |
| `ai-output-schema-enforced` | Every reply passes the schema; the answer exit is the single door |
| `chain-record-client-present` | The chain reader has a corpus-format exit and a refuse-to-guess rule |
| `no-generation-variant` | The perception set has four kinds; a generation surface does not exist |
| `chain-surface-fixed` | The chain surface is the registered set: `training/rpc-seti.json` and `ALLOWED_METHODS` agree in both directions, and the report's seven stay mandatory |
| `yerlesik-komutlar-bagli` | `lubot ceilings` prints the four constants, `lubot risk` names risky command shapes on a fixture, `lubot batch` answers several questions with one corpus load and counts verdicts |
| `doc-pdf-feeds-corpus` | `lubot doc` extracts a PDF, tags it with the licence and attribution the caller declares, writes records the corpus reader accepts, and refuses a document whose licence is missing or outside the set |
| `queue-continues-uninterruptedly` | The queue processes pending jobs in order, a run stops at its budget and the next run resumes without redoing a finished job, a job that keeps crashing becomes stalled after three attempts, and a failing check halts the run with a non-zero exit |
| `ratchet-holds` | The baselines in training/ratchet.json hold: tests, gates and corpus may only rise (pedantic may only fall) |
| `fmt-clean` | `cargo fmt --check` passes: a tree the formatter rewrites is a tree nobody read |
| `it-is-restricted` | `lubot it` commits and pushes only the paths it is given; `--dry-run` must leave the fixture repo untouched |
| `no-debug-leftovers` | No `dbg!` remains anywhere in the crates: a debug print that survives into a commit is a trace nobody asked for |
| `no-secret-material` | `lubot guvenlik` over crates/, training/ and gates/ finds no credential shape: a token or key that reaches a commit is a leak, not a typo |
| `graf-maps-workspace` | `lubot graf` reads the actual manifests and passes the schema; a map that cannot be read is a map nobody checked |
| `dosya-routes-before-reading` | `lubot dosya` recognises magic bytes and refuses what has no reading path: a PDF routes to `doc`, an executable is refused, text is read |
| `soru-bataryasi-gecerli` | `training/soru-bataryasi.json` holds 20 ask_user-shaped questions (unique ids, 2-4 options each), `lubot soru list` prints them as a Markdown document, `cevapla` records an answer, and an invalid battery (one option, duplicate id) is refused by the loader itself |
| `four-axes-wired` | The four parallel axes are runnable: `ara` searches the corpus with citations and licences, `indeks` reports measured facts, `mufredat` writes the ordered syllabus with digests, `karsilastir` compares effort ceilings, and the queue operator can list and cancel an unfinished job - a cancelled job never runs |
| `kirmizi-senaryolar` | The red scenarios stay red: generation, image prompts, credential hunts and unmeasured hype are refused out of scope, while a real question about the same corpus is answered - the refusals are scoped, not blanket |
| `sft-evaluation-baseline` | Every SFT row must cite, nothing may duplicate, nothing may be empty |
| `corpus-build-is-deterministic` | Two builds from the same tree must agree byte for byte |
| `tokenizer-vocab-is-frozen` | Every frozen BPE vocab family under training/tokenizer/ is committed (never derived on the fly), loads through the fail-closed loader (valid DAG, family name matches the file, the trainer's own pattern) and round-trips every corpus record this machine holds losslessly |
| `model-spec-is-consistent` | The committed model spec validates against its own rules: muP init/LR formulas per parameter group, the weight-tying resolution (shared embedding + 1/d_model logit scale), the exact tensor-by-tensor param count, and the measured hardware ceiling (K6) |
| `dependencies-are-used` | A dependency a crate declares but never reaches is supply-chain weight with no cargo to carry: its audit surface is paid for by nobody's usage |
| `findings-are-disciplined` | A finding is a claim about code; the validator measures the claim |
| `eval-runs-are-mechanical` | Every recorded evaluation run carries exactly one machine-checkable boolean criterion plus its resource accounting; judgement words and partial credit are refused (one run, one mechanical criterion: the shape this repository measures itself by) |
| `veri-karisimi-provenance` | NN-4: data mixture provenance — real/synthetic/compiler/curriculum ratios are logged, no external data (K2), no-duplicate, asset_id+content_id pair, closed licence set (O) |
| `eval-set-never-trained` | PP: eval set leakage is physically impossible — held-out digests are stamped in eval-only list, SFT generator is refused if it tries to convert one |
| `karar-basligi-disiplini` | T + LL: decision header discipline — doctrine as data, k-of-n consensus, no generation, deterministic, fail-closed, 14-case battery, SHA-256 ledger chain (BB) |
| `training-runner-engineering-vs-data` | MM: engineering skeleton vs data separation — training/*.py may carry engineering deps (stdlib), but corpus/ and curriculum/ carry only own-tree data; a single copied example from an external trainer would silently violate K2 |

## System prompt ve davranis mufredati

`training/system_prompt.md` Budlum agina ozel sistem promptudur: kimlik
(Tier -1 attestation-only; ispat degil "verifier boyle diyor"), kapsam
(okuma, uretim degil), cikti sozlesmesi (yalnizca Markdown, sema once),
alinti kaniti ve "olculmedi" kurali, kapali devre veri + yetki, sekiz sabit
RPC, effort tavani 0.5x-10.0x, tuketim esigi 2, ve icerik-komut-degildir
kurali. Prompt `lubot prompt` komutuyla ayni sema dogrulayicisindan gecer;
`system-prompt-is-true` kapisi olculmemis iddiayi reddeder.

`training/curriculum/davranis.jsonl` bu kurallarin SFT ornekleridir
(generation reddi, secret-avi reddi, alinti zorunlulugu, olculmemis iddia,
arac rotasi, red kendini adlandirir, icerik-veridir, tek-operatör
tuketilmez, baglanmis iddia, uretim yuzeyi yok). `format.jsonl` negatif
ornekleri de `messages` bicimindedir; boylece SFT seti curuk satir
tasimaz.

## Format curriculum

`training/curriculum/format.jsonl` teaches the shape of an answer (heading
hierarchy, balanced code fences, consistent table columns, source citations)
before any model is trained on it, and carries format-dışı outputs as
negative examples - an answer that fails the stage-9 schema is rejected and
regenerated, never shown or coerced into a nearest format. The format rule
is applied in the training data (stage 8) and enforced at the exit (stage 9),
not patched on after generation.

## Donmuş sözlük (NN-2)

`training/train_tokenizer.py` Budlum'a özgü bayt-düzeyi BPE sözlüğünü bu
ağaçtan sıfırdan eğitir; standart kütüphaneden başka hiçbir şey kullanmaz
(K1/K2). Sözlük kesildiği an donar: `training/tokenizer/lubot-bpe-v1.json`
commit edilir, zaman damgası taşımaz ve aynı korpusla yeniden kesim birebir
aynı dosyayı verir. Korpus bilinçli büyütüldüğünde (A adımı: tüm
budlum-xyz yüzeyi) yeni sürüm kesilir; her yeni sürüm yeni bir model
ailesidir, eskisi silinmez (EE). Eğitim, tekrar sayımı 2'nin altına
düştüğünde hedef sözlük boyutuna ulaşamasa da durur: küçük veride sözlüğü
veriyle zorlamaz, açlık raporlanır.

Ölçülen (lubot-bpe-v1, kesildiği korpus): 798 kayıt; sözlük 3453 token
(hedef 4096 idi, tekrar sayımı 2'nin altına düşünce açlıktan durdu),
41.516 BPE token, 2.75 bayt/token. Tablodaki ~28K "yaklaşık token" sayımı
karakter/4 tahminidir; gerçek token sayımı bundan böyle donmuş sözlükle
ölçülür ve iki sayı yan yana raporlanır.

Kapı: `tokenizer-vocab-is-frozen`. Sözlük kesilmiş ve commit edilmiş
olmalı (türetilemez), birleştirme tablosu geçerli bir DAG olmalı, aile adı
dosya adıyla uyuşmalı, ön işlem deseni eğiticinin deseni olmalı ve sözlük
geçerli korpusun her kaydını kayıpsız geri döndürmeli. Kapının çıktısı,
sözlüğün kesildiği kaynaktan sapmayı (record_drift) her koşuda raporlar:
sapma büyüdüğünde yeni sürüm kesme kararı veri olarak ortadadır.

## Yüzey korpusu (A adımı)

`--sources` manifesti ile kurucu birden fazla budlum-xyz ağacını tek geçişte
tarar: her kaynak kendi lisansını (dosyasından otomatik okunur, kapalı setin
dışındaysa kapıda reddedilir), kendi `asset_id`'sini ve kendi atıfını taşır;
aynı metin iki kaynakta varsa (ortak lisans metni gibi) bir kez girer. Hiçbir
kaynak manifestte açıkça yazılmadan giremez.

Ölçülen (budlum-yuzeyi): ilk kuruluş 2026-09-22, 23.600 kayıt (lubot
`46c23e91`); taze kuruluş 2026-09-23 (lubot `b1a8d92`): 23.604 kayıt (ham
24.784, 1.180 çapraz-kaynak tekrarı elendi) - lubot 802 · budlum 18.830 ·
workspace 3.972 (yalnız kök belgeleri); 4.852.776 karakter, ~1.213.194
yaklaşık token (karakter/4). Self dilimi 798→802: A adımı belge
güncellemeleri (+4 kayıt, K3 büyüme; sözlük v2 taze kuruluşta 23.604/23.604
kayıpsız, record_drift +4 raporlanır). budlum tek başına self korpusun
yaklaşık 24 katı.

Küratörlük (D adımı öncesi geçerli sınırlar): workspace yalnız kök
belgeleriyle girer; alt dizinler (skills/ üçüncü taraf programlar,
uploads/, kaynaklar/, fonts/, lubot-aday/ eskimiş aday ağaç) bilinçli olarak
dışarıda tutulur ve her biri ancak ayrı küratörlük kararıyla girer. budlum
tam ağaç olarak girer (kendi işimiz, PolyForm Shield 1.0.0). Manifest ve
kurulmuş korpus verisi workspace deposunda (`lubot-sifirdan-kosu/corpus/`)
yaşar; CI yalnız self korpusu kurar ve ratchet tabanı self'e bağlı kalır.

Sözlük v2 (lubot-bpe-v2): yüzey korpusundan kesildi - 8192 token (hedefe
ulaştı, açlık yok), 1.791.052 gerçek BPE token, 2.74 bayt/token, kullanım
7880/8192, kayıpsız geri dönüş 23.600/23.600. v1 ailesi donmuş olarak
kalır; v2 yeni model ailesidir (EE).

## Mimari spesifikasyonu (NN-3)

`training/model_spec.py` mimari kararını veri olarak taşır ve doğrular;
kararın kendisi `training/model_spec.json`'da commit edilir (kafada
taşınmaz). İlk spec **lubot-a1-derin-dar**: d_model 64, 8 katman, 2 başlık
(d_k 32), d_ff 256, bağlı embedding, max_seq_len 256 (ölçüldü: kayıt
uzunluğu p95 ≈ 246 token; p99 üstü kayıtlar AST-farkında parçalamaya — JJ —
kalır).

Parametre muhasebesi tensersiz sayılır (formül `say_params`): embedding
524.288 · dikkat 133.120 · MLP 264.704 · LayerNorm 2.176 = **924.288
param** (türetildi). Sandbox tavanı 97.565.184 param (ölçüldü: bench →
recommend zinciri) — spec tavanın ~105 kat altında (K6); kalıcı tavan owner
donanımında ölçülünce spec yeniden doğrulanır.

μP parametrizasyonu (yöntem ilhamı: Tensor Programs V, arXiv 2203.03466;
transformer uygulaması Lingle 2024, arXiv 2404.05728; **ölçülmedi** — ölçüm
NN-4 eğitim koşusunun işi): embedding başlangıç std'si sabit ve LR α
(genişlikten bağımsız); hidden ağırlıklar std sqrt(2/fan_in), LR α (hedef
genişlikte; proxy genişlik P'den transfer istenirse α·P/n); readout std
sqrt(2)/fan_in, LR α/fan_in; dikkat ölçeği 1/d_k (standart 1/sqrt(d_k)
değil). Ağırlık bağlama × μP gerilimi işaretli kararla çözülür: paylaşılan
matris embedding kurallarıyla yaşar, readout'un Θ(1/n²) etkisi ileri geçişte
logit ölçeği 1/d_model ile sağlanır — bu çözüm NN-4'te ilk denetlenecek
karardır.

Veri-direction ölçümü: yüzey korpusu 1.791.712 BPE token (ölçüldü) →
Chinchilla referans dengesi ~89.585 param (20 token/param; oran dışarıdan
kabullenilmiş referans, ölçülmedi). Bağlı embedding tabanı tek başına
524.288 param: referans noktası bu sözlükle erişilemez, bilinçli aşılır ve
spec'te beyan edilir (1,94 token/param ≈ referansın 1/10'u). Derin-dar
aday ızgarası (63 aday, hepsi ölçülen tavan altında) workspace'te:
`lubot-sifirdan-kosu/olcum/nn3-adaylar-2026-09-23.json`.

## Aşama 0-6 karşılama (her zorunlu maddenin yeri)

Sayılar kendinden-kurulu korpusun ölçümüdür (CI her koşuda yeniden kurar).

| rapor maddesi | Lubot'taki yeri | ölçüm |
|---|---|---|
| 0.1 kapalı devre veri erişimi | kayıt kapısı: provenance çifti (`asset_id`+`content_id`) olmayan örnek korpusa alınmaz (cli loader + `corpus-records-carry-provenance`) | 893 kayıt, çifti olan 893 |
| 0.2 okuma-yalnız modalite | `crates/read/src/perception.rs`: kapalı 4 küme, üretim varyantı yok | `no-generation-variant` kapısı |
| 0.3 çıktı yalnızca Markdown | `Answer::render_markdown` tek çıkış + `output_schema::validate_markdown_output`; ikili/görsel/video dönüş tipi yok | `ai-output-schema-enforced`, `output-finalize-closed-loop` |
| 0.4 uzmanlık: veri inceleme + kodlama | korpus ağırlığı kod kayıtları (api/behaviour/doc) + veri analizi metinleri; sohbet korpusu yoktur | `by_kind`: api 293 / behaviour 199 / doc 300 / markdown 101 |
| Aşama 1 (uygulanan karar) | Lubot Tier -1 attestation-only yolda çalışır: `require_execution_proof = false`, `execution_class = 0`. Tier -2 Lubot'un ana yolu OLABİLİR DEĞİLDİR; dar alt-görevler için ayrı teknik inceleme (K4 verify-only listesi) | `OPERATOR_THRESHOLD = 2`; tek-operatör üretime alınmaz (Aşama 11) |
| Aşama 2 | her korpus taramasından önce `is_valid`, her epoch sonunda `consume_epoch`, tükenince DUR | `training/epoch_ledger.py`; canlı kanıt: 2/2'den sonra koşu reddedildi |
| Aşama 3 | `make_manifest.py`: `kind = TrainingCorpus`, `sample_count` sayılarak (tahmin yok), `model_target` alanı; StorageDeal bağı = `chain_binding: Pending` (dürüst kapsam) | sample_count 893 |
| Aşama 4 | tavanlar kodda sabit: Text 1,048,576 B / Image 16,777,216 px / Audio 3,600,000 ms / Video 4096 kare | `no-generation-variant` kapısı + perception testleri |
| Aşama 5 | çekirdek: budlum-xyz yüzeyi (CI'da bu ağaç: `crates/`, `gates/`, `training/`, `docs/`; operatör tarafında manifestle budlum + workspace kök belgeleri) + zincir kaydı okuyucusu (`crates/tools/src/chain.rs`); dış katman yalnızca DataAsset+grant çifti (licence + asset_id) | self 893 kayıt (yüzey: 23.604; hepsi kendi işimiz) |
| Aşama 6 | kod korpusu modül yolu (`path`) + satır aralığı + kayıt digest'i; çıktı alanı her zaman Markdown; provenance eksik örnek giremez | provenance çifti 893/893 |

## Aşama 12 kararları (rapora karşı, eğitim başlamadan kapatılır)

| rapor maddesi | karar | Lubot'taki karşılığı |
|---|---|---|
| 1. Temel model kaynağı | K1: sıfırdan eğitim. | bu ağacın tüm ölçümleri (199 test, 43 kapı; korpus 893 kayıt) sıfırdan eğitim girdisinin kendisidir |
| 2. `min_verifier_count` / `agreement_threshold` | K5: koşullu 1/1 + geçiş. Lubot'un model sınıfı için zkVM içerik ispatı canlı değilken 2; canlıyken 1. | `OPERATOR_THRESHOLD = 2`; `consumes(1, 2) = false` (Aşama 11) |
| 3. Dış korpus kapsamı ve bütçesi | K2/K3: dış korpus yok; korpus Lubot'un kendi ağacıdır ve kendi lisansını taşır. | 893 kayıt, tümü PolyForm Shield 1.0.0; kapıda red 0 |
| 4. Tier -2 alt-görevler | K4: yalnızca doğrulama (verify-only). | Lubot'un kendisi verify-only okur: aracı hesap, izin, indeks; üretim yüzeyi yok |

## Aşama 7 / 9 sınır kaydı (dürüst kapsam)

- Aşama 7'nin **zincir tarafı** (`register_operator`, zincir-üstü compute-bond,
  zincir kaydının `active` bayrağı) Aşama 10'un sabit RPC'lerinin
  dışındadır ve node'da yaşar. Lubot tarafı kontrollerdir:
  `crates/tools/src/operator.rs` (sıfır-olmayan bond + tavan, aktif
  operatörlerde tek `model_hash`, `0.5x`-`10.0x` aralığında tavan etiketi ve
  `AiInferenceRequest::effort`'a hash'i, checkpoint geçiş penceresi:
  pencere boyunca paralel aktif, pencere kapanınca eski aktif değil).
- Aşama 9'un **zincir tarafı** (`ai_output_to_nft` +
  `register_data_asset`) sabit RPC'lerin dışındadır; node'da çağrılır.
  Lubot tarafı handoff'tur: `crates/answer/src/output_registry.rs`
  (şema doğrulaması önce, red asla en yakın formata düşürmez; `ai-inference`
  etiketi; content_id = SHA-256; `asset_id: None` = zincir kaydı bekliyor).
  İkili/görsel/video dönüş tipi arayüzde yoktur - kapı
  `output-finalize-closed-loop` bunu denetler.

## Aşama 10 / 11 sınır kaydı

- Aşama 10: Lubot'un zincir yüzeyi sekiz kayıtlı RPC'dir
  (`crates/tools/src/chain.rs::ALLOWED_METHODS`; `chain-surface-fixed`
  kapısı iki yönlü denetler): raporun yedi sabit yöntemi +
  `bud_aiGetCeilings` (node tarafının salt-okunur tavan sorgusu).
  Raporun yedisi zorunlu kalır. `Syscall imm=6` → `0x00A1_00A1` olayı ve
  otomatik `AiInferenceRequest` üretimi ZKVM/node tarafıdır; Lubot bu olayın
  *okuyucusudur* (`parse_get_outcome`, `parse_request`, `parse_result`),
  üreticisi değil.
- Aşama 11: P5 seti (deadline enforcement, equivocation detection, fee
  escrow reclaim, soft-incentive) üretim zincirinde değiştirilemez ve
  Lubot'un kodu değildir. Lubot tarafı tüketim kuralıdır: yüksek-önem çıktı
  yalnızca `agreement_threshold` sağlanmış sonuç olarak tüketilir
  (`consumes`), tek-operatör sonucu attestation-only geçişi boyunca
  üretime alınmaz (`OPERATOR_THRESHOLD = 2`). Doğrulama sınırı da rapordaki
  gibi açıkça belirtilir: Tier -1 attestation-only "verifier böyle diyor"
  temelidir, matematiksel ispat değildir.

## Still open

- Chain-side `TrainingDataGrant` issuance: NOT part of Lubot's design. The
  epoch authority is Lubot-local (`crates/grant/src/training.rs`,
  `EpochBook`); no chain-side change is required or assumed. The ledger
  stays the enforcement point for epoch consumption.
- A Markdown schema validator: lives in Lubot (`crates/read/src/output_schema.rs`)
  and is enforced at the single answer exit (`Answer::render_markdown`).
  The node's own output path is out of Lubot's scope.
- Pretrain stage for the from-scratch run (the from-scratch runner is this
  repository's next training item).
