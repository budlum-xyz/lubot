# Crates

What each crate is for, what it depends on, what reaches it, and how much of it
has been measured.

## Reading the measurement column

The test figures are counts of `#[test]` functions in the crate. They are **not**
counts of tests that have been run. This file was written in an environment with
no Rust toolchain, so no `cargo test` has executed against any of it; the figure
says how much checking is written, and `gates/check.py` says what was actually
verified. The distinction matters because a written test that has never run is a
claim, not a result.

What *has* been verified mechanically, by `python3 gates/check.py --all`:

| gate | what it proves |
|---|---|
| `every-crate-is-a-member` | no crate directory is missing from the workspace, so nothing compiles in isolation |
| `assert-arity` | every `assert_eq!`/`assert_ne!` carries an argument count that compiles |
| `no-dead-error-variant` | every variant of every `*Error` enum is produced by some code path |
| `doc-links-resolve` | every intra-doc link names something that exists |
| `no-bool-comparison` | no `== true` / `== false`, which the workspace lints deny |
| `delimiters-balance` | braces, parens and brackets balance in every source file |
| `no-panic-path` | no `unwrap`/`expect` outside `#[cfg(test)]` |
| `crates-are-reachable` | every crate is reachable from the binary, or it is on a baseline that may only shrink |

Sixteen further gates depend on `cargo` or on a built corpus and cannot run
without them. They report as failures rather than aborting the run.

## The primitives

No dependencies inside the workspace. Each one holds a single rule.

| crate | lines | tests | the rule it holds |
|---|---|---|---|
| `muhur` | 348 | 10 | A seal is not a signature - it has no key. It is a chain, not a concatenation, so it can name *where* it broke rather than merely that something did. |
| `esik` | 585 | 14 | A quorum over a set that includes the requester is not a quorum. `n >= 3f + 1`. A threshold that would drop below the Byzantine floor is refused, so `5 of 7` cannot become `5 of 40`. |
| `izolasyon` | 424 | 12 | A session opens onto an empty workspace, with a **fresh** identity checked against a register of issued ones. Results leave as copies. Documented limit: a checkable contract cannot observe what happens between open and close. |
| `kanit` | 576 | 13 | A proof is a fact *about a state root*, so a valid proof against a stale state is refused. There is **no path from Verified back to Pending** - not guarded, absent. Expiry is not failure: a pending record that goes stale is rejected as never-verified, never marked expired. |
| `kuyruk` | 611 | 15 | A full queue **refuses rather than evicting**, because which item to lose belongs to the submitter. Aging stops starvation. `take()` removes on take, so an in-flight item cannot go to a second worker. |
| `erisim` | 603 | 17 | Capabilities, not access lists: bearer-held, holder identity not checked. Attenuation only - widening is refused on scope, actions and expiry as separate axes. Expiry is checked at use, not at issue. This is **binding, not unforgeability**; there is no key here and the module says so. |
| `takip` | 638 | 15 | Progress is monotonic. A completed task that goes back to pending has lost a fact, and the refusal distinguishes that from a deliberate, recorded re-open. A dependency cycle is refused when the edge is written, not when the plan stalls. |
| `olcek` | 642 | 18 | Scale-up and scale-down thresholds must differ, or the controller alternates forever. No action inside the cooldown. Decisions are driven by *sustained* load, so one spike does not buy capacity. Reversals are counted, because flapping is a fact about the policy. |
| `yetenek` | 642 | 14 | Declared is not working. A capability only becomes usable after its own self-test passes. Versions are exact - a caller asking for v2 never silently gets v1. Degraded is a state reported to the caller, not an error hidden from it. |
| `mimari` | 616 | 17 | A component may depend on its own layer or below, never above, enforced where the upward call is written. Start order is a topological sort that is then **verified against every edge**. Shutdown is the exact reverse, not a second computation. |

| `kanaat` | 1851 | 40 | `tomurcuk` | The evidence-to-verdict engine: an answer is chosen, escalated or refused, and every verdict carries its reasoning. Turkish folding is applied before matching (`I`/`İ`/`ı`/`i` collapse to one letter) and never through a locale, so the same sentence folds the same way on every machine. The stop-word list is stored **folded**, and a test asserts it stays that way: an entry written with its diacritics would never match and would silently carry full weight instead of none. Negation is read from a closed list and from the `-me`/`-siz` suffix families, because `gecersiz` and `olmayacak` are negated by their endings and appear in no list. Scoring is idf-weighted overlap over the candidate's own token mass, so a candidate cannot buy points by being long; coverage, support and margin are separate floors with separate numbers, and a phrase match is scored apart from two unrelated word matches. A contradicting polarity and a different figure in the same unit are both marked and both penalised, while a figure in a *different* unit is not called a contradiction: it is a comparison the engine cannot make. The three verdict shapes are closed, and the order of the floors is a contract, because the report names the first check that fired. The verdict ledger (`defter`) records decisions as a SHA-256 chain: an edited entry is caught by its own digest, a removed one by the link to its neighbour, and the module states the one failure a chain cannot see - truncation - which is why `dogrula_uc` takes an anchor held elsewhere. The battery is compiled into the binary and a battery of another version is refused rather than run. |

## Built on the primitives

| crate | lines | tests | depends on | what it adds |
|---|---|---|---|---|
| `denetim` | 672 | 16 | `muhur` | An append-only audit trail. There is **no update method and no delete method**; corrections are appended entries. The chain detects edits and names the first index, but **cannot detect truncation** - `head()` has to be anchored elsewhere, and `verify_against_anchor` is what catches it. |
| `usl` | 710 | 14 | `muhur` | A read media verified by recomputation. `from_media` re-parses every line, verifies the seal, **and compares the rebuilt file byte for byte** - the check most implementations omit. Amounts are minor units; `1.5`, `007`, a swapped line order and a duplicate payout are all refused. |
| `arayuz` | 367 | 4 | The Android bridge: the same `ask` path the CLI runs, reached over JNI. It carries no logic of its own - a thin carrier, and device-supplied documents stay in an isolated record with their own source and licence so a citation never loses its origin. |
| `cikarim` | 2479 | 41 | `egitim` | The inference surface of a trained checkpoint: score and rank, never generate. A token is scored from the hidden state of the position *before* it - scoring it from a state that already contains it is leakage that stays invisible, because the number is still a plausible log-probability - and a request with no context and one token is refused rather than answered with a vocabulary prior. The cached incremental path is measured against a full recomputation *and* against the training kernel's own loss (three opinions: two paths inside this crate could share one mistake), agreeing to 1e-15 with a 1e-9 tolerance chosen three orders above the noise floor. Candidates that score equally are reported as equal and tie-broken by the caller's index, because inventing an order between two equal numbers is inventing a difference. |
| `anlama` | 688 | 16 | `read` | Classification that can decline. Abstention distinguishes *no support* from *contradictory* from *below the floor*, because they need different responses. Ties are not resolved by category spelling. Calibration reports the gap between confidence claimed and accuracy observed, which is the number that says whether the confidence is usable. |

| `tomurcuk` | 764 | 10 | `anlama` | The decision head: three closed output shapes and no text-producing surface, checked by a gate rather than by convention. A fixed tier order - deterministic code, then the head, then generation - and a route that skips a tier is refused. Confidence below the threshold escalates instead of deciding, and an empty ledger means the head may not decide alone, so moving a decision to the head stays a measured step. k-of-n agreement over independently initialised heads; deliberately not the chain's operator threshold. |
| `egitim` | 6147 | 69 | `grant` | The training core, written from scratch: forward pass, cross-entropy, and a hand-written backward pass with no autograd library. Correctness is **measured rather than argued** - every parameter of the small configuration (344 of them, all 19 tensor fields) is compared against a central finite difference, and the expected count is taken from the spec itself so a tensor added later cannot go unchecked. The tolerance is relative with an absolute floor, because a finite difference cannot resolve a gradient of 1e-6: measured, the worst-scoring parameter agrees to four significant digits (-1.280e-6 against -1.280e-6) while its naive relative error reads 8.6e-5. GELU is the tanh form so the coded derivative is the derivative of the coded function. Weight decay is a per-call decision, since decaying a LayerNorm scale shrinks a scale rather than regularising. The epoch ceiling comes from `lubot-grant` and is not restated. What it does **not** have: no tokenizer, so it cannot read the corpus yet, and no tokenizer of its own, and no trained checkpoint until a run writes one - `lubot egitim` reports a measured descent on an in-memory sequence and says so in the same line. Three more modules hold the discipline around the arithmetic: `veri` splits by a hash of `content_id` (a positional split makes two corpus versions incomparable, and the difference looks like learning) and refuses a repeat, an empty side or a window above the spec instead of clamping; `kosu` carries the step, epoch and schedule horizon across a resumed call so a run split in two produces one loss curve (measured: 6+6 steps reproduce 12 to 1e-12), stops on an epoch that fails to beat the carried-in best, and reports *why* it stopped; `kontrol` writes the checkpoint as header JSON plus named binary blocks plus a SHA-256 over every byte before it, and carries the optimizer moments too - a checkpoint that can be loaded but not continued is exactly the difference a resumed run is supposed to erase. | It also measures the corpus against the spec's window length (`pencere_olcu`), nearest-rank percentiles named as such because "p95" is not one number without a method: measured on the self corpus, p95 is 168 tokens against the spec's 256 (the claim holds), while record-by-record windowing covers only 18.94% of the tokens and packing covers 99.78% - so the spec's sequence length was never the constraint, the windowing strategy is. Packing is here too (`paketle`), and it carries provenance position by position because a packed window joins records: measured, 369 of 416 windows span more than one record and one window joins 12, so without per-position source ids a citation could not be attributed. Attention therefore does not cross a record boundary, forward or backward, and that is measured rather than asserted - a packed run must equal its records run alone (loss 1.942361282232 either way); with the mask removed the same test reads 1.945561023341 and fails.
| `jeton` | 592 | 13 | - | The frozen BPE vocab, read and applied rather than re-cut. The loader is fail-closed on structure (`vocab_size == 256 + merges`, and every merge may only refer to ids defined before it) and **string-compares the pretoken pattern** against the one this file implements - a pattern it cannot apply is a refusal, never an approximation, because a different segmentation would produce plausible ids that mean something else. The segmentation is the subtle part and it is pinned by a test: `[\W_]+` is greedy and `\W` includes whitespace, so `"; oku"` is `"; "` then `"oku"`, not three pretokens - treating the four classes as a partition of characters is the easy way to get this wrong. `\d` is read as the Unicode Nd category; the comment records why each lint-suggested replacement (`is_ascii_digit`, `is_numeric`) would be wrong. Three falsehood tests pin it: case folding is not locale-independent (Turkish `İ` lowercases to two characters, so the tokenizer never lowercases and `I`/`ı`/`İ`/`i` stay four distinct token paths); the two Unicode normalization forms of `é` are **not** merged - NFC is one pretoken, NFD is two (`e` + U+0301), because the combining mark falls into the punctuation class; and a grapheme is not a user-perceived character (a flag emoji is two code points and eight bytes, and the ceilings that call it are in bytes). Agreement with the Python tokenizer that cut the vocab is **measured, not assumed**: gate 61 runs both over the whole corpus and compares ids record by record (1767/1767 identical, 102654 tokens). |
| `kodlayici` | 2776 | 42 | `sha2` | A checkpoint from outside this repository, read and run from Rust. The header reader is part-aware: the artifact is seven files and a tensor may straddle a boundary, so ranges are read across parts and two tensors claiming the same bytes are refused by name rather than resolved silently. Two measured facts about the real header are built in as rules: it is **not** written in offset order (the single fp32 buffer arrives alphabetically before fp16 tensors whose bytes precede it), so overlap is checked by sorting ranges rather than by trusting the listing order; and **layer 0 has no attention norm** while the other 21 do, so the loader demands the absence there and the presence everywhere else - an absent norm on a later layer is refused, because running it as an identity produces plausible numbers from the wrong model. The rope pairing is by halves and not by neighbours, which the first version of this port had the other way round: the independent cross-check disagreed at the first layer, and the test that now pins it says so in the comment. The decision head is here too: the type embedding is added to **every** position, the two head layers are bidirectional (no causal mask), the action head reads position 0 of the post-head state together with four features, and the runner-up in those features is the second *position* in value order so a tie gives a gap of zero. Temperature is applied to the scores before the softmax. Nothing here generates text. | 
| `sertlestirme` | 942 | 25 | - | The hardening layers that can be measured from inside the process. Layer 1 is the release profile in the workspace root and is not restated here. Layer 2 stores the literals that `strings` would otherwise read first as exclusive-ored blocks and decodes them at the point of use; the key is in the same file, so this removes the free attack and not the message. The integrity layer hashes the file in blocks and compares the measured digest with one held elsewhere, and the module states what that proves (the bytes on disk) and what it does not (a process patched after `exec`). The probes read the kernel's own report - `TracerPid`, a fixed workload's wall-clock, the hypervisor flag, the DMI product name, the container markers - and carry a weight rather than a verdict, because a shared runner is a container and a container is not an attack; the decision lives in exactly one gate, refusing only in `Zorla` mode at weight 3. Not here, and said so: control-flow flattening (no stock pass), packing (a loader decrypting at start-up carries its key in the same file) and secrets (a client binary cannot keep one). |


## The crates that were here first

These predate this work and are listed so the table covers the whole workspace
rather than only the part that was rewritten. The rule column is what reading
each one shows it to hold, not a claim about how it was written.

| crate | lines | tests | what it holds |
|---|---|---|---|
| `read` | 899 | 29 | The three source channels, SHA-256 provenance, the corpus surface, magic-byte file kind with route refusals before reading. |
| `index` | 617 | 21 | Passages with line ranges, secret masking on the write path, normalized BM25 with a coverage floor. |
| `grant` | 719 | 19 | View grants, revocation, expiry, the audit log. Permission is settled before the index is searched, so a refused item is never scored. |
| `tools` | 1760 | 48 | The exact-rational calculator, the command router, deterministic command-risk shapes. |
| `sikistir` | 672 | 11 | Context compression: typed routing, pins that survive byte for byte, a CCR store with digest re-verification, an append-only savings ledger. |
| `doc` | 106 | 4 | Rich-document reading: PDF text extraction, paragraph-aware chunking. |
| `answer` | 590 | 15 | The assembled reading loop with a schema-validated exit. |

## The binary

| crate | lines | tests | what it holds |
|---|---|---|---|
| `cli` | 11540 | 151 | The runnable binary, and the only crate that reaches everything else. `kodlayici envanter` prints what is inside a package and which tensors nothing read; `kodlayici kosu` runs the encoder and, with `--isaret`, the head; `kodlayici dogrula` streams the parts through SHA-256 and compares the digest with one given, so a run can say which artifact it ran. |

Four modules carry the wiring:

| module | command | what it wires |
|---|---|---|
| `activation` | (library) | Binds a run to an exact corpus digest, grant epoch and policy. A grant issued against one corpus does not authorize another. The policy never comes from the request. A run is not re-activated in place. |
| `kosum` | `lubot kosum denetle`, `lubot kosum dogrula` | `activation` + `erisim` + `kuyruk` + `denetim` + `muhur` + `izolasyon` + `kanit` + `yetenek`. Each item runs inside an isolated session with a fresh identity and leaves a proof accepted against the state it was made for. The record is verified by recomputation, not by trusting the writer. |
| `olcum` | `lubot olcum mimari\|esik\|takip\|sinif` | `mimari` + `esik` + `takip` + `anlama`. `olcum mimari` reads every `crates/*/Cargo.toml` and checks the real dependency graph against the declared layering in `ARCHITECTURE`. |
| `odeme` | `lubot odeme yaz`, `lubot odeme dogrula` | `usl` + `muhur`. Verification re-parses, re-checks, re-seals and re-renders, naming the first check that fails. |

## Running on one small device

Two crates written together, for the one question the rest of the repository
cannot answer: what has to be true for a model with a large parameter count to
run on a machine somebody actually owns.

| crate | lines | tests | deps | what it holds |
|---|---|---|---|---|
| `nicem` | 2697 | 61 | - | Sub-byte weight quantisation, written here and depending on nothing. Four stages, each measurable on its own: an fp16 codec verified across all 65536 bit patterns; an in-place normalised Walsh-Hadamard transform (`H/sqrt(n)` is symmetric and orthogonal, so the inverse is the transform itself and no second routine exists to disagree with the first); an analytic Lloyd-Max codebook for a Gaussian source at 1..=8 bits and ternary, solved from closed-form moments with quadrature used only for the region mass; and LSB-first bit packing with a base-3 packer at five trits per byte. The error is **measured, not claimed**: `Olcum` reports relative error, SNR, the largest single-weight deviation and the bits actually spent, and the largest deviation is reported beside the aggregate because a small Frobenius error can still hide one badly placed weight. Two findings are kept as passing tests rather than tidied away: the rotation is not universally better than absmax (a single-spike vector measures 0.5052 rotated against 0.3340 unrotated, because a spike maps to a constant-magnitude vector whose coordinates land between the two-bit levels), and the solver needs over-relaxation rather than a finer grid - a fixed-grid Lloyd iteration stalls at one grid cell width by construction, and the residual that remains at seven bits is Lloyd's linear convergence, not the quadrature. |
| `tasiyici` | 1725 | 32 | `nicem` | The container the weights are read from and the ladder that decides how much of it a machine holds. The file is mapped and used where it lies: a 64-byte header, a directory of fixed-meaning fields, then a payload whose blocks are 64-byte aligned, and a load's only allocation is the directory. Opening validates structure exhaustively - offsets in range, no overlapping blocks, byte counts matching the declared shapes - but deliberately does **not** verify the digest, because touching every page of a file designed so most pages are never touched would undo the design at the moment of loading; `dogrula()` is the explicit full pass. The writer is a pure function of insertion order, so two builds of the same model are byte-identical and the digest means something. Every tensor carries a rung: rung 0 is what every depth needs, and a reader holding rungs `0..=k` has a working model at depth `k`, which is what finally gives an operator a ceiling that refers to the machine. What it refuses to do is written into the crate: it does not turn streamed bytes into a predicted latency (that belongs to the storage, and an invented number would be reported as measured), it does not decide that a shallower depth is good enough (that is the exam set's judgement), and it does not report depth 0 when rung zero will not fit - that is a refusal, because a depth-zero reading looks like a working shallow model. A ceiling carries whether it was measured or declared, and the declaration line says `OLCULMEDI` when it was not. |

Both are reached from the binary: `lubot nicem kitap|butce|dondur|yarim` and
`lubot tasiyici olc|incele|tavan`.

## Reachability

`crates-are-reachable` walks the `path = "..."` dependencies from `crates/cli`.
As of this writing **all 24 crates are reachable and the baseline is empty**.

That is a ratchet, not a permission: the gate fails the moment an unwired crate
appears, and fails again when a baselined crate becomes reachable and the
baseline is not shrunk. A crate that nothing calls is code that has never been
run, and the number is only worth having if it cannot go up.

## The layering `olcum mimari` enforces

| layer | crates |
|---|---|
| 0 | `sertlestirme`, `read`, `muhur`, `esik`, `izolasyon`, `kanit`, `kuyruk`, `erisim`, `takip`, `olcek`, `yetenek`, `mimari` |
| 1 | `denetim`, `anlama`, `usl`, `index`, `grant`, `tools`, `sikistir` |
| 2 | `doc`, `answer`, `kanaat` |
| 3 | `cli` |

A crate with no entry in `ARCHITECTURE` is reported as unclassified and left out
of the layer check rather than assumed into a layer, because an assumed layer
produces violations that mean nothing. It is still checked for cycles, because a
cycle makes the graph unstartable regardless of which layer anything sits in.
