
<p align="center">
  <img src="assets/lubot-banner.png" alt="lubot banner" width="100%" />
</p>

Lubot is an AI that reads. It answers from what the network stores, under the
permissions the network already enforces, and it shows where every sentence
came from.

It is **not** the verification machinery underneath it. Operator registration,
the compute bond, the three bindings that tie a model, its input and its
computation together - that is the chain's AI inference layer, and it lives in
the node. Lubot is one AI running on top of it: a client, not the layer.

**It reads; it does not generate.** Text, image, audio and video are inputs.
There is no path here that produces images, video or music. That is an
admission rule rather than a missing feature: the correctness of a generated
work is undefined, and a system that accepts nothing it cannot check has
nothing to check a generation against.

## What works today

| capability | crate | evidence |
|---|---|---|
| permission before bytes; epoch-bounded training budgets (fail-closed) | `crates/grant` | 19 tests |
| three channels, digest-verified, no fourth; the Markdown reply schema; magic-byte file kind with route refusals before reading | `crates/read` | 28 tests |
| exact arithmetic instead of a guessed number; chain record client; operator sync rules (bond, one `model_hash`, ceiling-hashed effort tier, checkpoint window); scope refusals (generation, secret hunts); effort-bounded answer budget; deterministic command-risk shapes; the closed licence set for anything admitted to the corpus; the credential-shape scanner (closed list, exact lengths, a mention is never a leak) | `crates/tools` | 47 tests |
| retrieval with line-accurate citations, masking on the write path; normalized BM25 with a coverage floor and one-edit tolerance; deterministic context compaction under a character budget | `crates/index` | 18 tests |
| the assembled reading loop, schema-validated exit; scope refusals; the finalized-output handoff (`ai-inference` tag) | `crates/answer` | 14 tests |
| rich-document reading: PDF text extraction, paragraph-aware chunking | `crates/doc` | 4 tests |
| context compression: route by content type, pins survive byte for byte, CCR store with digest re-verification, append-only savings ledger | `crates/sikistir` | 11 tests |
| decision header (T doctrine): doctrine as data, k-of-n consensus (LL), decision ledger SHA-256 chain, 14-case embedded battery, fail-closed, no generation, deterministic | `crates/karar` | 21 tests |
| BPE tokenizer, JJ AST-aware: 8192 vocab, merge rules, code vs text split, deterministic | `crates/sozluk` | 20 tests |
| μP scaling: Tensor Programs V + Lingle 2024 method inspiration (measured), init std and LR table, logit_scale 1/d_model, weight_decay separation | `crates/mu` | 15 tests |
| deep-narrow architecture: config 64/8/2/256 924K param (spec) + 512/12/8/2048 37M param (K6 ceiling 97M), weight tying, param counting | `crates/derin` | 16 tests |
| training core Rust: AdamW, sparse embedding update (U speed), forward skeleton, checkpoint, 924K param | `crates/egitim` | 16 tests |
| inference engine: deterministic, temperature, KV-cache (W), speed measurement, no generation early detection | `crates/cikarim` | 15 tests |
| data mixture Rust: real 893 + synthetic 152 + compiler 5 + curriculum 88 = 1138, ratios O, provenance asset_id+content_id, dedup 0, eval split %10 | `crates/veri` | 15 tests |
| measurement: 14 batteries (Z) each single mechanical criterion, GG calibration, RR gap map | `crates/olcum` | 15 tests |
| self-distillation: self-instruct loop, mechanical jury (citation, schema, licence, provenance), error mining | `crates/kendinden` | 15 tests |
| contest measurement: AA protocol, rival output only in comparison report, never corpus, eval-set-never-trained PP, gap map RR | `crates/kapisma` | 14 tests |
| 3-stage training system: Stage1 15K token real 50% (dil temeli), Stage2 26K token real 50% + synthetic 2 epoch + compiler (kod+dil dengesi), Stage3 6K token curriculum Python/web + real 10% FIM 0.3 SPM 0.5, 143 checkpoints, data bucket per checkpoint, transparency report (LLM360 methodology inspiration) | `crates/uc-asama` | 18 tests |
| transformer: LLaMA-like with muP (embedding scale, output scale, QK^T/d not sqrt, LR/WD groups), RoPE only first 25% hidden (rope fix: half split not neighbor), LayerNorm not RMSNorm, GELU MLP, deterministic | `crates/transformer` | 21 tests |
| advanced BPE: 8192 base + 4 FIM + 14 code metadata + 4 instruction = 8214 vocab, special token preservation, FIM application, deterministic | `crates/bpe-gelismis` | 17 tests |
| full system: 13 components (tokenizer, model, data, training, inference, measurement, checkpoint, metrics, preprocessing, data-bucket, training-code, eval, analysis) 143 checkpoints, 47K token, transparency report (LLM360 methodology) | `crates/sistem` | 11 tests |
| parallelism: 224 GPU batch 2240 (Crystal-like), 1 GPU batch 8 (Lubot), token per batch, mixed-precision BF16 activ/grad FP32 weights, CG-1 4 exaFLOPS 54M core 64-node scale (name-free), speed ratio | `crates/paralel` | 11 tests |
| data mixture + preprocessing: real 893 + synthetic 152 + compiler 5 + curriculum 88 = 1138, 3-stage 15K/26K/6K, data bucket per checkpoint, FIM 0.3 SPM 0.5, token estimate | `crates/karma` | 11 tests |
| combined system: 29 crates unified (grant, read, index, tools, answer, cli, doc, sikistir, karar, sozluk, mu, derin, egitim, cikarim, veri, olcum, kendinden, kapisma, uc-asama, transformer, bpe-gelismis, sistem, paralel, karma, tumu, checkpoint, metrics, preprocess, eval) 143 checkpoints, 47K token, all code in one PR, transparency report (LLM360) | `crates/tumu` | 11 tests |
| checkpoint management: 143 checkpoints (Crystal 143, Amber 360), data bucket per checkpoint, loss decreases, grad_norm decreases, lr decreases, FIM 0.3 last 23 checkpoints, weight + optimizer path, transparency | `crates/checkpoint` | 12 tests |
| metrics: 143 checkpoints, loss, grad_norm, lr, token/sec, eval, average, report, transparency per checkpoint (Crystal-like) | `crates/metrics` | 12 tests |
| preprocessing: 8214 vocab (Crystal 32032), FIM 0.3 SPM 0.5, 4 FIM + 14 code + 4 instruction = 22 special, StarCoder method inspiration Lubot names, encode, preprocess, max_seq 256 (Crystal 2048) | `crates/preprocess` | 12 tests |
| eval: 14 batteries (Z) mechanical criteria, deterministic score, average, pass count, report, GG calibration, RR gap map, Crystal eval per checkpoint | `crates/eval` | 12 tests |
| the runnable binary: corpus load, `ask`, grant book, output audit, closed-loop handoff; `ceilings`; multi-question `batch`; the uninterrupted-work queue (resume, budget, per-job gate check, loud halt); measured baselines that may only rise (`ratchet`); repository `envanter`; restricted `it` (only the listed paths are committed and pushed); the four-step `olc` verification chain; `durum`; the manifest map `graf`; the credential scan `guvenlik`; the file-kind router `dosya`; the ask_user-shaped decision battery `soru` (list/get/cevapla/durum); content search `ara`, measured `indeks`, the ordered reading plan `mufredat`, effort comparison `karsilastir`; context compression `sikistir` (--path/--geri-getir: typed routing, pinned lines, reversible CCR store, measured ledger) and failure mining `ogren` (pattern grouping, two-tier promotion); the queue operator (`queue ls`, `queue iptal` - a cancelled job never runs); batch writes the same audit and closed-loop trace as `ask` | `crates/cli` | 37 tests |

488 tests, `clippy -D warnings` clean, `unwrap`/`expect` denied outside tests. 43 gates, each with its own self-test; the ratchet holds (488 tests, 43 gates, 0 pedantic warnings, 893 corpus records).

## Permission is an admission decision

Public content is read without asking. Everything else opens through a **view
grant** naming a grantee and a content key id, with an expiry. Sending someone
a direct message is issuing that grant.

No key material is stored here - a grant is a permission record, and opening
bytes is the storage layer's job. Revocation stops **new** opens; it does not
recall what was already read, so `Decision::Revoked` is a different answer from
`Decision::NoGrant`. Collapsing the two would be a lie about the past.

Refusals are logged with the same shape as allowances. A deployment reporting
zero refusals over live content is reporting that its checks never ran.

## Arithmetic is computed, not predicted

```
route("74830 * 1291 kac eder?")  -> Computed { calculator, "96605530" }
route("what does revocation do?") -> the reading path
route("what is 1 / 0")            -> ToolRefused { "division by zero" }
```

The calculator is exact rationals over `i128`: `0.1 + 0.2` is `0.3`, `1/3`
prints as `1/3`, `2^3^2` is `512`, and an overflow is an error rather than a
wrap. A tool that exists to stop a model from guessing must not guess.

## The order of the loop

```
question
  -> tool router          (a question with a right answer never reaches a model)
  -> grant decisions      (settled before anything is searched)
  -> index search         (only over what may be opened)
  -> answer + citations   (origin plus line range, or NotFound)
```

`NotFound` is a first-class answer. So is `Refused`, which carries the word the
grant book used, so "revoked" is never reported as "not found".

## Layout

| path | what lives there |
|---|---|
| `crates/grant` | view grants, revocation, expiry, the audit log |
| `crates/read` | the three source channels, SHA-256 provenance, the corpus surface |
| `crates/index` | passages with line ranges, secret masking, term search |
| `crates/tools` | the exact-rational calculator and the router |
| `crates/answer` | the reading loop that puts the four together |
| `gates/check.py` | the repository gates CI enforces |
| `training/` | the corpus builder, the supervised-set builder, the hardware bench, the model-size recommender and the frozen BPE tokenizer trainer |
| `corpus/` | derived self-built corpus (gitignored; CI builds it before the gates) |

## Build

```
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
python3 training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz
python3 gates/check.py --all
```

## The binary

`cargo run -p lubot -- <command>` is the runnable reader. `ask` writes only the
rendered Markdown to stdout - that is the single exit of the answer surface -
and appends one JSONL audit line per question (reader, question, answer kind,
citations, decision, refusal count).

```
lubot corpus corpus/knowledge-self.jsonl.gz
lubot ask --corpus corpus/knowledge-self.jsonl.gz --reader ayaz --audit audit.jsonl "what does a view grant name?"
lubot ask --corpus corpus/knowledge-self.jsonl.gz --reader ayaz --effort 0.5x "how does the epoch ledger refuse an expired grant?"
lubot grant issue --reader ayaz --key dm-1 --expires-at 2000000000
lubot grant list
lubot audit --path audit.jsonl --limit 5
lubot prompt
```

A corpus record is admitted only with a digest that matches its bytes, a
content id + asset id pair, a licence, one of the four closed kinds, and text
under the reading ceiling; a single refused record fails the load.

`--effort <tag>` bounds the answer: the tier is the operator's hardware
ceiling, and the budget it admits is a fixed mapping (0.5x to 10.0x), so a
low-ceiling run cannot be asked for a long answer.

`lubot prompt` prints the Budlum-specific system prompt. It goes through the
same schema validator every answer goes through, and
`gates/check.py --system-prompt-is-true` refuses a prompt that states a fact
nothing here measured - the four ceilings, the eight RPC names, the effort
range and the consumption threshold are the load-bearing numbers.

`--outputs <file>` appends the finalized-output handoff for every grounded or
computed answer: `content_id` (SHA-256 of the bytes), `digest`, pending
`asset_id`, the fixed `ai-inference` tag and the timestamp. The schema
validator has already run by the time the record exists, so the file holds
nothing it would reject.

## Corpus and training

The corpus is the budlum-xyz surface. `training/build_corpus.py` walks this
repository (CI, always) and, operator-side through a sources manifest, the
budlum core and the workspace root documents; each source stamps its own
licence and per-repo provenance pair, cross-source duplicates enter once,
and nothing outside the organization's own trees enters it. A record
without an allowed licence never enters the corpus - the refusal is at the
door, not a later filter.

The vocab is frozen and versioned: `training/train_tokenizer.py` cuts a
byte-level BPE vocab from this corpus from scratch (standard library only),
commits it under `training/tokenizer/` (v1: self corpus; v2: the surface
corpus), and a new corpus family is a new cut, never a silent drift. `training/bench_hardware.py` measures the run
machine (K6), `training/recommend_model_size.py` turns that measurement into a size ceiling, and `training/model_spec.py` validates the committed architecture spec against the muP table, its own param count and that ceiling
into a parameter ceiling where every number carries its label: measured,
derived (formula stated) or not measured.

What the corpus holds (measured, 2026-09-22):

| corpus | records | licence |
|---|---|---|
| `corpus/knowledge-self.jsonl.gz` (built by CI from this repository) | 807 | PolyForm Shield 1.0.0 (own work) |
| `corpus/budlum-yuzeyi.jsonl.gz` (operator, sources manifest: this repo + budlum + workspace root) | 23604 | PolyForm Shield 1.0.0 + MIT (own work) |

Three gates guard the data: `corpus-records-carry-licence` (every record in
`corpus/` carries an allowed licence and an attribution),
`corpus-records-carry-provenance` (every record carries the asset_id +
content_id pair), and `ratchet-holds` (the record count may only rise).

Epoch accounting is fail-closed: `training/epoch_ledger.py` is the
pipeline-side half of the chain `TrainingDataGrant` (time + max epochs); a
corpus pass refuses to start on an expired or exhausted grant, and each
epoch must be consumed. Chain-side grant issuance is future work.

## Base model

Base-model agnostic. A tier is a capability class; the served names are ours:
`ai_inference-light` (default) and `ai_inference-normal`. The checkpoint
behind a tier is an operator configuration value, so the runtime says nothing
about which weights an operator chose to load.

## Licence

PolyForm Shield 1.0.0 - see [`LICENSE.md`](LICENSE.md).
