# Failure families and the rules that now gate them

This file is the system's own incident record. Every family below happened,
was fixed, and left a rule behind it. The corpus teaches these as grounded
records so the next model of this system inherits the scars, not just the
code.

## Contract read after change

A behaviour was changed on the strength of a suggestion without reading the
test that locked the old contract. The locked test then failed in CI. The
rule: read the locking test before the code it locks; a locked test outranks
a reviewer suggestion, and a reasoned skip cites the lock.

## Whole-file rewrite of a multi-list file

A file holding several separate lists was rewritten in one pass, and the
lists melted into each other; a well-formedness rule broke far from the hand
that broke it. The rule: restore from a known-good state, then add or remove
exactly the named lines; the diff is shown before the commit.

## Refusal placed at the wrong door

A refusal was added where the producer runs, while the contract placed it
where the record is admitted. Legitimate tiny inputs then tripped the
refusal. The rule: a refusal lives at the door the contract names; moving a
door is a contract change, not a fix.

## Green without a canary

A control reported pass without ever demonstrating it could refuse its own
defect. The rule: every control ships with a canary, a defect deliberately
injected and refused; green carries information only after red has been
shown. See gates/check.py, where each gate pairs with a self-test.

## Decorative wiring

A reference that is never called was offered as wiring so a count would
pass. The rule: wiring is a call path from production code; a bare mention,
an uncalled function, or a discarded result is decoration and is refused by
the wiring gate.

## Panic inside gate code

A test inside gate code used expect, and the gate that bans panic points in
gate code caught its own author. The rule: gate code returns Result even in
its canaries; a panicking checker prints a backtrace instead of a finding.

## Equality where the contract allows a window

An admission rule demanded exact equality on a clock field, while the locked
lifecycle tests sign records a few blocks away from the applying block. The
rule: bound untrusted clocks with a window derived from an existing horizon;
the attack value (the maximum integer) stays refused and the signed-but-late
record stays admissible.

## Appending past a closed module

A test was appended to the end of a file whose module was already closed, so
the test landed outside `mod tests` and could no longer see the helpers it
called. The rule: before appending, find the closing brace of the module the
code belongs to; file position is scope, and the compiler only reports the
miss later, in CI.

## The root tool does not see sub-manifests

The root formatter and linter only cover the root workspace, while crates
with their own manifests are checked by their own CI steps. A clean root next
to a dirty sub-manifest still ships red. The rule: run each manifest's
checks exactly as CI lists them; one green command is not evidence for the
others.

## How the families are kept

Each family above maps to a curriculum row in training/curriculum and to a
gate canary in gates/check.py. A family that recurs is not bad luck; it is a
missing gate, and the second occurrence is written here, in the curriculum,
and in the gate, in the same commit.
