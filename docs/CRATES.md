# Crate inventory, four entries added in this series

| crate | holds | fails closed on |
|---|---|---|
| `yetenek` | skill cards and the ledger of what proved them | a card with no trigger, no steps, or prose-only exit evidence |
| `olcek` | ceiling, watermark, reserved floor, and the drop ledger | an admission past the ceiling, an uncounted eviction, a reserve spent by reading |
| `kanit` | scope, evidence, findings, paths, closure | observation before a plan, evidence outside the scope, closure without an observation that can carry it |
| `mimari` | the pipeline table and the tree it must match | a row naming a symbol that is not declared, an empty contract, an empty table |

None of the four depends on another. All are `std` only, no I/O, no clock, no
randomness: a run that reads them can be reproduced from its inputs, which is
the only property that makes a review finding reviewable a second time.
