# Security Policy

## Reporting a vulnerability

Please report vulnerabilities **privately**:

- Open a private vulnerability report on GitHub:
  `https://github.com/budlum-xyz/lubot/security/advisories/new`
- If private reporting is unavailable to you, open a public issue that says
  only that a security-relevant finding exists, with no details, and a
  maintainer will move the conversation to a private channel.

Please do not open a detailed public issue for an unfixed vulnerability, and
do not test against deployments you do not operate.

## Scope

This policy covers the contents of this repository. The Budlum chain's node
software is a different repository with its own surface.

Findings that matter here, in the shapes this codebase actually has:

- **Permission bypass** — a path that returns content a `grant` decision
  refused (including reporting `Revoked` as `NotFound`, or settling
  permission after the search).
- **Schema bypass** — output that leaves the answer surface without passing
  the Markdown schema validator, or any "closest format" degradation. The
  closed loop is fail-closed by design.
- **Evaluation contamination** — a record stamped eval-only reaching the
  training side (`eval-set-never-trained` exists because this class is
  binary: leakage is not a ratio).
- **Supply-chain escape** — the restricted `it` command committing or
  pushing paths outside its allowlist; new third-party code, weights or
  data entering the corpus (K1/K2 make this a licence violation, not a bug
  report to negotiate).
- **Credential scanner misses** — a credential shape in the closed,
  exact-length list that the scanner does not catch. Note the deliberate
  rule here: a *mention* of a token-like string is never itself a finding.

Out of scope: vulnerabilities requiring a compromised device or a modified
binary; the debug keystore produced by `android/derle.sh` (it signs debug
builds and holds no release authority by design); findings about the chain
node itself (report those to the Budlum repository).

## What a report gets back

This repository's discipline is mechanical: a finding is closed by a fix
plus a gate (or a stronger self-test on an existing gate) that would have
caught it, verified by CI. No response-time commitment is stated here,
because none has been measured; the honest statement is that security
reports are triaged ahead of feature work.

After a fix lands, reporters are credited in the fix's commit message
unless they ask not to be.

## Design principles you can rely on when reporting

- **Fail-closed.** Malformed input is refused, never approximated - the
  tokenizer refuses pretoken patterns it cannot apply, the corpus loader
  fails on a single malformed record, and a threshold below the Byzantine
  floor cannot be expressed.
- **No secrets in the tree.** Keys, tokens and keystores do not belong in
  this repository; the `no-secret-material` gate scans for them and a
  report that finds one is in scope.
- **Refusals are data.** A deployment that reports zero refusals is
  reporting that its checks never ran. Suspicious *absence* of refusals is
  as reportable as a wrong acceptance.
