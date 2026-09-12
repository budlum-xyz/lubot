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
| `esik` | 540 | 13 | A quorum over a set that includes the requester is not a quorum. `n >= 3f + 1`. A threshold that would drop below the Byzantine floor is refused, so `5 of 7` cannot become `5 of 40`. |
| `izolasyon` | 424 | 12 | A session opens onto an empty workspace, with a **fresh** identity checked against a register of issued ones. Results leave as copies. Documented limit: a checkable contract cannot observe what happens between open and close. |
| `kanit` | 563 | 13 | A proof is a fact *about a state root*, so a valid proof against a stale state is refused. There is **no path from Verified back to Pending** - not guarded, absent. Expiry is not failure: a pending record that goes stale is rejected as never-verified, never marked expired. |
| `kuyruk` | 613 | 15 | A full queue **refuses rather than evicting**, because which item to lose belongs to the submitter. Aging stops starvation. `take()` removes on take, so an in-flight item cannot go to a second worker. |
| `erisim` | 527 | 16 | Capabilities, not access lists: bearer-held, holder identity not checked. Attenuation only - widening is refused on scope, actions and expiry as separate axes. Expiry is checked at use, not at issue. This is **binding, not unforgeability**; there is no key here and the module says so. |
| `takip` | 627 | 15 | Progress is monotonic. A completed task that goes back to pending has lost a fact, and the refusal distinguishes that from a deliberate, recorded re-open. A dependency cycle is refused when the edge is written, not when the plan stalls. |
| `olcek` | 644 | 18 | Scale-up and scale-down thresholds must differ, or the controller alternates forever. No action inside the cooldown. Decisions are driven by *sustained* load, so one spike does not buy capacity. Reversals are counted, because flapping is a fact about the policy. |
| `yetenek` | 642 | 14 | Declared is not working. A capability only becomes usable after its own self-test passes. Versions are exact - a caller asking for v2 never silently gets v1. Degraded is a state reported to the caller, not an error hidden from it. |
| `mimari` | 616 | 17 | A component may depend on its own layer or below, never above, enforced where the upward call is written. Start order is a topological sort that is then **verified against every edge**. Shutdown is the exact reverse, not a second computation. |

## Built on the primitives

| crate | lines | tests | depends on | what it adds |
|---|---|---|---|---|
| `denetim` | 582 | 14 | `muhur` | An append-only audit trail. There is **no update method and no delete method**; corrections are appended entries. The chain detects edits and names the first index, but **cannot detect truncation** - `head()` has to be anchored elsewhere, and `verify_against_anchor` is what catches it. |
| `usl` | 701 | 14 | `muhur` | A read media verified by recomputation. `from_media` re-parses every line, verifies the seal, **and compares the rebuilt file byte for byte** - the check most implementations omit. Amounts are minor units; `1.5`, `007`, a swapped line order and a duplicate payout are all refused. |
| `anlama` | 688 | 16 | `read` | Classification that can decline. Abstention distinguishes *no support* from *contradictory* from *below the floor*, because they need different responses. Ties are not resolved by category spelling. Calibration reports the gap between confidence claimed and accuracy observed, which is the number that says whether the confidence is usable. |

## The crates that were here first

These predate this work and are listed so the table covers the whole workspace
rather than only the part that was rewritten. The rule column is what reading
each one shows it to hold, not a claim about how it was written.

| crate | lines | tests | what it holds |
|---|---|---|---|
| `read` | 848 | 28 | The three source channels, SHA-256 provenance, the corpus surface, magic-byte file kind with route refusals before reading. |
| `index` | 552 | 18 | Passages with line ranges, secret masking on the write path, normalized BM25 with a coverage floor. |
| `grant` | 719 | 19 | View grants, revocation, expiry, the audit log. Permission is settled before the index is searched, so a refused item is never scored. |
| `tools` | 1729 | 47 | The exact-rational calculator, the command router, deterministic command-risk shapes. |
| `sikistir` | 672 | 11 | Context compression: typed routing, pins that survive byte for byte, a CCR store with digest re-verification, an append-only savings ledger. |
| `doc` | 106 | 4 | Rich-document reading: PDF text extraction, paragraph-aware chunking. |
| `answer` | 527 | 14 | The assembled reading loop with a schema-validated exit. |

## The binary

| crate | lines | tests | what it holds |
|---|---|---|---|
| `cli` | 7009 | 98 | The runnable binary, and the only crate that reaches everything else. |

Four modules carry the wiring:

| module | command | what it wires |
|---|---|---|
| `activation` | (library) | Binds a run to an exact corpus digest, grant epoch and policy. A grant issued against one corpus does not authorize another. The policy never comes from the request. A run is not re-activated in place. |
| `kosum` | `lubot kosum denetle`, `lubot kosum dogrula` | `activation` + `erisim` + `kuyruk` + `denetim` + `muhur` + `izolasyon` + `kanit` + `yetenek`. Each item runs inside an isolated session with a fresh identity and leaves a proof accepted against the state it was made for. The record is verified by recomputation, not by trusting the writer. |
| `olcum` | `lubot olcum mimari\|esik\|takip\|sinif` | `mimari` + `esik` + `takip` + `anlama`. `olcum mimari` reads every `crates/*/Cargo.toml` and checks the real dependency graph against the declared layering in `ARCHITECTURE`. |
| `odeme` | `lubot odeme yaz`, `lubot odeme dogrula` | `usl` + `muhur`. Verification re-parses, re-checks, re-seals and re-renders, naming the first check that fails. |

## Reachability

`crates-are-reachable` walks the `path = "..."` dependencies from `crates/cli`.
As of this writing **all 21 crates are reachable and the baseline is empty**.

That is a ratchet, not a permission: the gate fails the moment an unwired crate
appears, and fails again when a baselined crate becomes reachable and the
baseline is not shrunk. A crate that nothing calls is code that has never been
run, and the number is only worth having if it cannot go up.

## The layering `olcum mimari` enforces

| layer | crates |
|---|---|
| 0 | `read`, `muhur`, `esik`, `izolasyon`, `kanit`, `kuyruk`, `erisim`, `takip`, `olcek`, `yetenek`, `mimari` |
| 1 | `denetim`, `anlama`, `usl`, `index`, `grant`, `tools`, `sikistir` |
| 2 | `doc`, `answer` |
| 3 | `cli` |

A crate with no entry in `ARCHITECTURE` is reported as unclassified and left out
of the layer check rather than assumed into a layer, because an assumed layer
produces violations that mean nothing. It is still checked for cycles, because a
cycle makes the graph unstartable regardless of which layer anything sits in.
