
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
| exact arithmetic instead of a guessed number; chain record client; effort tier as a hashed ceiling (bond, `model_hash` and window rules live in the node, not here); scope refusals (generation, secret hunts); effort-bounded answer budget; deterministic command-risk shapes; the closed licence set for anything admitted to the corpus; the credential-shape scanner (closed list, exact lengths, a mention is never a leak) | `crates/tools` | 41 tests |
| retrieval with line-accurate citations, masking on the write path; normalized BM25 with a coverage floor and one-edit tolerance; deterministic context compaction under a character budget | `crates/index` | 18 tests |
| the assembled reading loop, schema-validated exit; scope refusals; the finalized-output handoff (`ai-inference` tag) | `crates/answer` | 14 tests |
| rich-document reading: PDF text extraction, paragraph-aware chunking | `crates/doc` | 4 tests |
| context compression: route by content type, pins survive byte for byte, CCR store with digest re-verification, append-only savings ledger | `crates/sikistir` | 11 tests |
| the isolation boundary as a checkable contract: a session opens from an empty workspace under a session-scoped identity, with exactly one contract in force, and results leave only as copies | `crates/izolasyon` | 4 tests |
| the security review as an evidenced ledger (scan, validate, fix): every finding ends in a disposition, a fix closes only with the check that proves it plus the commit that is it, no waiver at the attestation floor without an attester, and a finding that changes on re-scan voids its stale closure | `crates/denetim` | 9 tests |
| the runnable binary: corpus load, `ask`, grant book, output audit, closed-loop handoff; `ceilings`; multi-question `batch`; the uninterrupted-work queue (resume, budget, per-job gate check, loud halt); measured baselines that may only rise (`ratchet`); repository `envanter`; restricted `it` (only the listed paths are committed and pushed); the four-step `olc` verification chain; `durum`; the manifest map `graf`; the credential scan `guvenlik`; the file-kind router `dosya`; the ask_user-shaped decision battery `soru` (list/get/cevapla/durum); content search `ara`, measured `indeks`, the ordered reading plan `mufredat`, effort comparison `karsilastir`; context compression `sikistir` (--path/--geri-getir: typed routing, pinned lines, reversible CCR store, measured ledger) and failure mining `ogren` (pattern grouping, two-tier promotion); the queue operator (`queue ls`, `queue iptal` - a cancelled job never runs); batch writes the same audit and closed-loop trace as `ask` | `crates/cli` | 37 tests |

331 tests, `clippy -D warnings` clean, `unwrap`/`expect` denied outside tests. 40 gates, each with its own self-test; the ratchet holds (331 tests, 40 gates, 0 pedantic warnings, 793 corpus records) - the count moved 328 to 334 when the settlement layer joined with its own six tests, 334 to 332 when `olcek`'s callerless report surface left with the two tests that were its only assertions, and 332 to 328 when operator's registry rules - bond, single-hash sync, the transition window - left with their four fixtures and the gate's name-list shrank to the one rule with a caller, and 0028 moved it to 331 with three `usl` refusal tests written against a forger who can recompute the seal, measured on the applied tree, not retyped.

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
| `crates/izolasyon` | the isolation boundary: empty workspace, session-scoped identity, one contract, copy-only exit |
| `crates/denetim` | the review ledger: scan, validate, fix, evidenced dispositions |
| `gates/check.py` | the repository gates CI enforces |
| `training/` | the corpus builder and the supervised-set builder |
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
`gates/check.py system-prompt-is-true` refuses a prompt that states a fact
nothing here measured - the four ceilings, the seven RPC names, the effort
range and the consumption threshold are the load-bearing numbers.

`--outputs <file>` appends the finalized-output handoff for every grounded or
computed answer: `content_id` (SHA-256 of the bytes), `digest`, pending
`asset_id`, the fixed `ai-inference` tag and the timestamp. The schema
validator has already run by the time the record exists, so the file holds
nothing it would reject.

## Corpus and training

The corpus is Lubot's own. `training/build_corpus.py` walks this repository,
chunks the text-bearing files at the record budget, hashes every chunk, and
stamps every record with the repository's own licence and a pre-issuance
provenance pair; nothing outside this tree enters it. A record without an
allowed licence never enters the corpus - the refusal is at the door, not a
later filter.

What the corpus holds (measured, 2026-09-08):

| corpus | records | licence |
|---|---|---|
| `corpus/knowledge-self.jsonl.gz` (built by CI from this repository) | 793 | PolyForm Shield 1.0.0 (own work) |

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
