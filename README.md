# Lubot

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
| permission before bytes | `crates/grant` | 10 tests |
| three channels, digest-verified, no fourth | `crates/read` | 7 tests |
| exact arithmetic instead of a guessed number | `crates/tools` | 13 tests |
| retrieval with line-accurate citations, masking on the write path | `crates/index` | 9 tests |
| the assembled reading loop | `crates/answer` | 9 tests |

48 tests, `clippy -D warnings` clean, `unwrap`/`expect` denied outside tests.

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
| `training/` | corpus builder and supervised-set builder |

## Build

```
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
python3 gates/check.py --all
```

## Corpus and training

The runtime does not depend on the training side. `training/build_corpus.py`
walks a checkout and emits records that keep their provenance - documentation
for intent, public signatures for surface, test names for proven behaviour -
so any answer can be walked back to a file and a line range. Raw source is not
fed in: a model trained on raw source learns to autocomplete, not to explain a
protocol.

Order of material: our own repositories first, then the working principles,
then the open web - and only from sources whose licence permits collection,
with the source and its licence stored on every record.

## Base model

Base-model agnostic. A tier is a capability class (`light`, `normal`); the
checkpoint behind it is an operator configuration value, so the runtime says
nothing about which weights an operator chose to load.

## Licence

PolyForm Shield 1.0.0 - see [`LICENSE.md`](LICENSE.md).
