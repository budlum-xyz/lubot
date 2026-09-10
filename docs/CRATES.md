# Crate inventory, twelve entries added in this series

| crate | holds | fails closed on |
|---|---|---|
| `izolasyon` | the arcbox boundary: what a session may touch, and the refusal when it does not hold | a workspace that is not empty at session start, a path outside the declared root, a boundary widened after the fact without a record |
| `denetim` | the scan -> validate -> fix ledger, with the proof each step must carry | a fix with no finding behind it, a finding recorded before the scan that produced it, a claim of cleanliness with no evidence row |
| `yetenek` | skill cards and the ledger of what proved them | a card with no trigger, no steps, or prose-only exit evidence |
| `olcek` | ceiling, watermark, reserved floor, and the drop ledger | an admission past the ceiling, an uncounted eviction, a reserve spent by reading |
| `kanit` | scope, evidence, findings, paths, closure | observation before a plan, evidence outside the scope, closure without an observation that can carry it |
| `mimari` | the pipeline table and the tree it must match | a row naming a symbol that is not declared, an empty contract, an empty table |
| `takip` | wiring claims about reachability, and the commit each was written at | a wired claim with no call site, a site whose call text is gone, an unwired label the code outgrew, an empty registry |
| `muhur` | an append-only digest-sealed record and its optional actor index | a rewritten entry, a removed tail, a numbering gap, an index that points at nothing, an unfinalized chain |
| `kuyruk` | bounded maintenance work: pending, done, dead letters, and the eviction ledger | a zero capacity, a duplicate key, an arrival that cannot displace anything, an early take, and any job the sums cannot find |
| `erisim` | issued capability grants, their narrowing by delegation, and the audit trail | a zero ttl or use count, an empty scope or capability, a `starts_with` scope escape, a delegation that widens, a use after revocation or expiry, a trail that disagrees with the records |
| `anlama` | what each word of the command contributed, and what had to be added to get there | a scope no word supports, a loose match with no correction on record, an assumption nobody listed, an instruction read out of attached text |
| `esik` | the activation schedule: plans, the flags inside them, and the epoch each opens at | an activation at genesis, a change whose epoch is not its plan's, a flag the plan never promised, a plan opened twice at one epoch, a due change nobody ratified, a retirement that never ran, a rewritten epoch |

None of the twelve depends on another. All are `std` only, no I/O, no clock, no
randomness: a run that reads them can be reproduced from its inputs, which is
the only property that makes a review finding reviewable a second time.
