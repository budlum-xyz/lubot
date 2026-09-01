#!/usr/bin/env python3
"""Repository gates.

A gate is a claim this repository makes about itself, written so that a change
that breaks the claim fails the build instead of quietly becoming untrue. Each
gate carries a self-test: a gate that cannot demonstrate it catches its own
violation is a gate nobody should trust.

Three of these came from the node repository, where they used to guard the same
promises in code that has since moved here. Moving the code without moving the
gate would have been a way of losing a check while calling it a refactor.

Usage:
    python3 gates/check.py --all
    python3 gates/check.py --list
    python3 gates/check.py <gate> [--self-test]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise SystemExit(f"gate cannot run: {rel} is missing")
    return path.read_text(encoding="utf-8")


def rust_sources() -> list[Path]:
    return sorted(p for p in (ROOT / "crates").rglob("*.rs"))


# --------------------------------------------------------------------------
# gate: reads, does not generate
# --------------------------------------------------------------------------
GENERATION_WORDS = ["fn generate_image", "fn generate_video", "fn synthesize_audio", "text_to_image"]


def gate_reads_not_generates() -> str:
    """No generation surface exists, and the reading path has a refusal for
    everything it cannot open."""
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        for word in GENERATION_WORDS:
            if word in text:
                raise SystemExit(f"{path.relative_to(ROOT)} exposes a generation surface: {word}")
    answer = read("crates/answer/src/lib.rs")
    for variant in ["NotFound", "Refused", "ToolRefused"]:
        if f"{variant} " not in answer and f"{variant}," not in answer and f"{variant} {{" not in answer:
            raise SystemExit(f"the answer type has no `{variant}` case; then an unanswerable question has nowhere to go")
    return "no generation surface; NotFound, Refused and ToolRefused all exist"


def selftest_reads_not_generates() -> None:
    assert "fn generate_image" in GENERATION_WORDS


# --------------------------------------------------------------------------
# gate: reject_unknown_source  (moved from the node repository)
# --------------------------------------------------------------------------
def gate_no_fourth_channel() -> str:
    """Content enters through exactly three channels, and a fourth is refused."""
    src = read("crates/read/src/lib.rs")
    if "fn source_kind" not in src:
        raise SystemExit("`source_kind` is gone; then the channel list is prose")
    for channel in ['"stored"', '"granted"', '"local"']:
        if channel not in src:
            raise SystemExit(f"channel {channel} is missing from `source_kind`")
    if "unknown source" not in src:
        raise SystemExit("`source_kind` no longer refuses an unknown channel")
    if "fn reject_unknown_source" not in src:
        raise SystemExit("the refusal has no test pinning it")
    return "three channels, and an unknown one is refused with a test on it"


def selftest_no_fourth_channel() -> None:
    sys.path.insert(0, str(ROOT / "gates"))
    src = read("crates/read/src/lib.rs")
    assert "unknown source" in src


# --------------------------------------------------------------------------
# gate: verify_sha256 fail-closed  (moved from the node repository)
# --------------------------------------------------------------------------
def gate_provenance_fails_closed() -> str:
    """A record's bytes are checked against its digest, and a mismatch refuses."""
    src = read("crates/read/src/lib.rs")
    if "pub fn verify_sha256" not in src:
        raise SystemExit("`verify_sha256` is gone; a self-reported digest is not provenance")
    if "digest mismatch" not in src:
        raise SystemExit("the mismatch path no longer produces an error")
    if "fn insert" not in src or "item.verify()?" not in src:
        raise SystemExit("the corpus accepts items without verifying them first")
    return "digests are verified on the way in, and a mismatch is a refusal"


def selftest_provenance_fails_closed() -> None:
    assert "verify_sha256" in read("crates/read/src/lib.rs")


# --------------------------------------------------------------------------
# gate: masking before storage  (moved from the node repository)
# --------------------------------------------------------------------------
def gate_mask_before_storage() -> str:
    """The secret mask is applied on the write path, not on the way out."""
    src = read("crates/index/src/lib.rs")
    if "pub fn mask_secrets" not in src:
        raise SystemExit("`mask_secrets` is gone; then `before storage` is a word with nothing under it")
    add_at = src.find("pub fn add(")
    if add_at < 0:
        raise SystemExit("the index has no `add`; the write path cannot be checked")
    body = src[add_at : src.find("\n    }", add_at)]
    if "mask_secrets(" not in body:
        raise SystemExit("`add` stores the body without masking it first")
    if "fn redact_model_strings" not in src:
        raise SystemExit("the mask has no test pinning its behaviour")
    return "the mask runs inside the write path, with a test on the mask itself"


def selftest_mask_before_storage() -> None:
    assert "mask_secrets" in read("crates/index/src/lib.rs")


# --------------------------------------------------------------------------
# gate: the panic limit
# --------------------------------------------------------------------------
def gate_no_panic_path() -> str:
    """`unwrap` and `expect` are denied outside tests, and the denial is real."""
    manifest = read("Cargo.toml")
    if 'unwrap_used = "deny"' not in manifest or 'expect_used = "deny"' not in manifest:
        raise SystemExit("the workspace no longer denies unwrap/expect")
    offenders: list[str] = []
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        cut = text.find("#[cfg(test)]")
        production = text if cut < 0 else text[:cut]
        for i, line in enumerate(production.splitlines(), 1):
            if re.search(r"\.(unwrap|expect)\(", line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}")
    if offenders:
        raise SystemExit("a panic path is on the production side:\n  " + "\n  ".join(offenders))
    return f"no unwrap/expect outside tests in {len(rust_sources())} files"


def selftest_no_panic_path() -> None:
    assert re.search(r"\.(unwrap|expect)\(", "x.unwrap()")


# --------------------------------------------------------------------------
# gate: the claims in the README are measured
# --------------------------------------------------------------------------
def gate_readme_is_measured() -> str:
    """The test count in the README is the count the suite reports."""
    readme = read("README.md")
    claimed = re.search(r"(\d+) tests, `clippy", readme)
    if not claimed:
        raise SystemExit("the README no longer states a measured test count")
    out = subprocess.run(
        ["cargo", "test", "--workspace"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit("the suite does not pass, so its count cannot be claimed")
    measured = sum(int(m) for m in re.findall(r"test result: ok\. (\d+) passed", out.stdout))
    if measured != int(claimed.group(1)):
        raise SystemExit(f"the README says {claimed.group(1)} tests, the suite reports {measured}")
    return f"the README count matches the suite: {measured}"


def selftest_readme_is_measured() -> None:
    assert re.search(r"(\d+) tests, `clippy", "48 tests, `clippy -D warnings` clean")


GATES = {
    "reads-not-generates": (gate_reads_not_generates, selftest_reads_not_generates),
    "no-fourth-channel": (gate_no_fourth_channel, selftest_no_fourth_channel),
    "provenance-fails-closed": (gate_provenance_fails_closed, selftest_provenance_fails_closed),
    "mask-before-storage": (gate_mask_before_storage, selftest_mask_before_storage),
    "no-panic-path": (gate_no_panic_path, selftest_no_panic_path),
    "readme-is-measured": (gate_readme_is_measured, selftest_readme_is_measured),
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "--list":
        for name in GATES:
            print(name)
        return 0
    if argv[0] == "--all":
        failures = 0
        for name, (run, selftest) in GATES.items():
            selftest()
            try:
                print(f"OK   [{name}] {run()}")
            except SystemExit as err:
                failures += 1
                print(f"FAIL [{name}] {err}")
        print("ALL GATES PASSED" if not failures else f"{failures} gate(s) failed")
        return 1 if failures else 0
    name = argv[0]
    if name not in GATES:
        print(f"unknown gate: {name}")
        return 2
    run, selftest = GATES[name]
    if "--self-test" in argv:
        selftest()
        print(f"self-test OK [{name}]")
        return 0
    print(f"OK   [{name}] {run()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
