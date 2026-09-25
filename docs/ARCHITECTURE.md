# Architecture

Lubot is a reader, not a generator. This document names the rules the code
enforces and where each one lives; it deliberately carries no test counts,
line counts or corpus figures, because the measured table for those is
[`docs/CRATES.md`](CRATES.md) and the `crates-doc-is-measured` gate compares
that table against the sources. A number restated here would be a second
answer for one fact, and this file is part of the corpus: a corpus-derived
figure written here feeds back into the measurement that produced it, so
figures stay out.

## The one rule that shapes everything else

`crates/mimari` holds the layer rule in code: **a component may depend on its
own layer or below, never above**. The rule is not a convention; it is
enforced where an upward call would be written. Start order is a topological
sort of that graph and is then verified against every edge, and shutdown is
the exact reverse rather than a second computation.

The layers, bottom to top:

A note on `kodlayici`: it depends on no workspace crate above it and is
depended on by none; it is the one crate that runs somebody else's trained
weights and it says so. Nothing in the reading loop reaches it yet, and the
cross-check tool (`tools/kodlayici_capraz.py`) is not part of the binary.

1. **Primitives** — single-rule crates with no workspace dependencies:
   `muhur` (seal chains), `esik` (quorum arithmetic), `izolasyon` (fresh
   sessions), `kanit` (proofs about a state root), `kuyruk` (a queue that
   refuses rather than evicts), `erisim` (capabilities, attenuation only),
   `takip` (monotonic progress), `olcek` (flap-resistant scaling), `yetenek`
   (a capability only exists after its own self-test), `jeton` (the frozen
   BPE vocab, applied fail-closed), `sertlestirme` (hardening checks that are
   measured rather than asserted) and `mimari` itself.
2. **Composed** — crates built on the primitives: `denetim` (append-only
   audit), `usl` (media re-verified byte for byte), `anlama` (classification
   that may decline), `tomurcuk` (the decision head: closed output shapes,
   fixed tier order, k-of-n), `egitim` (the from-scratch training core:
   forward pass, hand-written backward pass, packing with per-position
   provenance, the run loop and the checkpoint format) and `cikarim` (the
   inference surface: score and rank with a trained checkpoint, and never a
   generation surface), `kanaat` (evidence to verdict: a choice, an
   escalation or a refusal, with a battery compiled into the binary) and
   `kodlayici` (a checkpoint from outside this repository, read and run from
   Rust: the split-file header reader, the configuration, the encoder stack and
   the decision head, with a tokenizer still to come).
3. **Reading** — `read` (three channels, digest-verified), `index` (BM25
   with line-accurate citations), `grant` (permission settled before bytes),
   `tools` (exact-rational calculator, router), `doc` (PDF and rich
   documents), `sikistir` (reversible context compaction), and `answer`,
   which assembles the reading loop with a schema-validated exit.
4. **Entry** — `cli` (the `lubot` binary; the only command surface) and
   `arayuz` (the Android JNI bridge, a thin carrier with no logic of its
   own).

## The order of the reading loop

```
question
  -> tool router   (a question with a deterministic right answer never
                    reaches a model: exact arithmetic, exact refusals)
  -> grant         (permission is settled before anything is searched;
                    Refused and NotFound are different answers)
  -> index         (search runs only over what may be opened)
  -> answer        (citations carry origin and line range; the output is
                    schema-validated Markdown or it does not leave)
```

## The training path

`training/` is the from-scratch pipeline: corpus builder, frozen tokenizer
trainer, hardware bench, model-size recommender, the muP measurement, the
data-mix declaration, the bootstrap round and the evaluation records under
`training/eval/`. The base model is trained from scratch on the budlum-xyz
corpus only (K1/K2); no third-party weights, code or data enter it. The
epoch ledger is fail-closed: an expired or exhausted grant refuses to start
a pass.

## Verification is the authority

`gates/check.py` is the only verifier the repository trusts: every gate
carries a self-test that proves it catches its own violation, and CI runs
fmt + clippy (-D warnings) + the suite + the corpus build + all gates.
`training/ratchet.json` holds the baselines that may only rise. A claim that
cannot point at a gate, a measurement record or a commit is prose, and this
repository treats prose as unverified by default.

## The Android shell

`android/` is a Gradle-free build (`android/derle.sh`: aapt2, javac, d8,
zipalign, apksigner) around `crates/arayuz`. The app requests no network
permission, the corpus ships as an app asset, and device-supplied documents
stay in an isolated record (`source: cihaz`) so K2 is never diluted by what
a user points the app at.

## Boundaries

What Lubot is *not* lives elsewhere by design: operator registration, the
compute bond and the proof layer are the Budlum chain's node software, not
this repository. Lubot is one client on top of that layer, and every chain
interaction here is a client of a fixed surface (the `chain-surface-fixed`
gate holds the method list).
