# Lubot training pipeline

Decisions behind the pipeline, the measured corpus, the gates, and what is
still open. Everything the pipeline learns from is Lubot's own tree: the
corpus is built from this repository and from nothing outside it.

## Decisions

| # | Decision |
|---|---|
| K1 | Base model: trained from scratch. The served names are ours; no upstream name is used as a served name anywhere in this repository. |
| K2 | The corpus is Lubot's own tree. The builder walks this repository; every record carries the repository's own licence and provenance. Nothing from outside enters. |
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
| `corpus/knowledge-self.jsonl.gz` | 793 | api 241 · behaviour 178 · doc 285 · markdown 89 | ~27K |

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
| `system-prompt-is-true` | The Budlum-specific system prompt states only measured facts: the four ceilings, the seven RPC names, the effort range, the threshold, the ai-inference tag - and no superlative or proof claim that nothing here produced |
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
| `dependencies-are-used` | A dependency a crate declares but never reaches is supply-chain weight with no cargo to carry: its audit surface is paid for by nobody's usage |
| `findings-are-disciplined` | A finding is a claim about code; the validator measures the claim |
| `eval-runs-are-mechanical` | Every recorded evaluation run carries exactly one machine-checkable boolean criterion plus its resource accounting; judgement words and partial credit are refused (one run, one mechanical criterion: the shape this repository measures itself by) |

## System prompt ve davranis mufredati

`training/system_prompt.md` Budlum agina ozel sistem promptudur: kimlik
(Tier -1 attestation-only; ispat degil "verifier boyle diyor"), kapsam
(okuma, uretim degil), cikti sozlesmesi (yalnizca Markdown, sema once),
alinti kaniti ve "olculmedi" kurali, kapali devre veri + yetki, yedi sabit
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

## Aşama 0-6 karşılama (her zorunlu maddenin yeri)

Sayılar kendinden-kurulu korpusun ölçümüdür (CI her koşuda yeniden kurar).

| rapor maddesi | Lubot'taki yeri | ölçüm |
|---|---|---|
| 0.1 kapalı devre veri erişimi | kayıt kapısı: provenance çifti (`asset_id`+`content_id`) olmayan örnek korpusa alınmaz (cli loader + `corpus-records-carry-provenance`) | 793 kayıt, çifti olan 793 |
| 0.2 okuma-yalnız modalite | `crates/read/src/perception.rs`: kapalı 4 küme, üretim varyantı yok | `no-generation-variant` kapısı |
| 0.3 çıktı yalnızca Markdown | `Answer::render_markdown` tek çıkış + `output_schema::validate_markdown_output`; ikili/görsel/video dönüş tipi yok | `ai-output-schema-enforced`, `output-finalize-closed-loop` |
| 0.4 uzmanlık: veri inceleme + kodlama | korpus ağırlığı kod kayıtları (api/behaviour/doc) + veri analizi metinleri; sohbet korpusu yoktur | `by_kind`: api 241 / behaviour 178 / doc 285 / markdown 89 |
| Aşama 1 (uygulanan karar) | Lubot Tier -1 attestation-only yolda çalışır: `require_execution_proof = false`, `execution_class = 0`. Tier -2 Lubot'un ana yolu OLABİLİR DEĞİLDİR; dar alt-görevler için ayrı teknik inceleme (K4 verify-only listesi) | `OPERATOR_THRESHOLD = 2`; tek-operatör üretime alınmaz (Aşama 11) |
| Aşama 2 | her korpus taramasından önce `is_valid`, her epoch sonunda `consume_epoch`, tükenince DUR | `training/epoch_ledger.py`; canlı kanıt: 2/2'den sonra koşu reddedildi |
| Aşama 3 | `make_manifest.py`: `kind = TrainingCorpus`, `sample_count` sayılarak (tahmin yok), `model_target` alanı; StorageDeal bağı = `chain_binding: Pending` (dürüst kapsam) | sample_count 793 |
| Aşama 4 | tavanlar kodda sabit: Text 1,048,576 B / Image 16,777,216 px / Audio 3,600,000 ms / Video 4096 kare | `no-generation-variant` kapısı + perception testleri |
| Aşama 5 | çekirdek: Lubot'un kendi ağacı (`crates/`, `gates/`, `training/`, `docs/`) + zincir kaydı okuyucusu (`crates/tools/src/chain.rs`); dış katman yalnızca DataAsset+grant çifti (licence + asset_id) | 793 kayıt, tümü PolyForm Shield 1.0.0 (kendi işimiz) |
| Aşama 6 | kod korpusu modül yolu (`path`) + satır aralığı + kayıt digest'i; çıktı alanı her zaman Markdown; provenance eksik örnek giremez | provenance çifti 793/793 |

## Aşama 12 kararları (rapora karşı, eğitim başlamadan kapatılır)

| rapor maddesi | karar | Lubot'taki karşılığı |
|---|---|---|
| 1. Temel model kaynağı | K1: sıfırdan eğitim. | bu ağacın tüm ölçümleri (178 test, 37 kapı; korpus 793 kayıt) sıfırdan eğitim girdisinin kendisidir |
| 2. `min_verifier_count` / `agreement_threshold` | K5: koşullu 1/1 + geçiş. Lubot'un model sınıfı için zkVM içerik ispatı canlı değilken 2; canlıyken 1. | `OPERATOR_THRESHOLD = 2`; `consumes(1, 2) = false` (Aşama 11) |
| 3. Dış korpus kapsamı ve bütçesi | K2/K3: dış korpus yok; korpus Lubot'un kendi ağacıdır ve kendi lisansını taşır. | 793 kayıt, tümü PolyForm Shield 1.0.0; kapıda red 0 |
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

- Aşama 10: Lubot'un zincir yüzeyi yedi kayıtlı RPC'dir
  (`crates/tools/src/chain.rs::ALLOWED_METHODS`; `chain-surface-fixed`
  kapısı iki yönlü denetler): raporun yedi sabit yöntemi +
  `bud_aiGetCeilings` **geri alındı** (2026-09-11): çağrısızdı, budlum tarafında implementasyonu
  yok ve `chain.rs` testi izinli olmaması gerektiğini söylüyor. Sekiz → yedi; bu düzeltme
  `rpc-seti.json`, `system_prompt.md` ve `gates/check.py` token listesiyle birlikte yapıldı.
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
