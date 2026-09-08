#!/usr/bin/env python3
"""Examine the SFT set before any hardware spends an epoch on it.

A training row is a claim the corpus makes about itself; this measures the
claim instead of trusting the builder. Three refusals, each a leak the
builder's contract says cannot happen:

* a grounded row whose answer carries no `Source:` citation - the builder
  drops unciteable rows, so one surviving is a regression, not a style;
* a duplicate row (same user + assistant bytes) - a duplicate teaches
  repetition, not knowledge;
* an answer body under the measured floor - the smallest body in the pinned
  self-corpus is 41 characters, so 20 is generous and only truly empty or
  truncated rows trip it.

Measured on the self-corpus (2026-09-08): 776 rows (748 grounded + 28
curriculum), zero uncited, zero duplicates, minimum body 41.

Usage:
    python3 training/eval_sft.py --sft corpus/sft.jsonl
    python3 training/eval_sft.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MIN_BODY_CHARS = 20


def digest_of(row: dict) -> str:
    user = row["messages"][0]["content"]
    assistant = row["messages"][1]["content"]
    return hashlib.sha256((user + "\x00" + assistant).encode("utf-8")).hexdigest()


def evaluate(rows: list[dict]) -> dict:
    """Measure the rows; `findings` empty means the set is trainable."""
    findings: list[str] = []
    grounded = [r for r in rows if r.get("kind") != "curriculum"]
    curriculum = [r for r in rows if r.get("kind") == "curriculum"]
    seen: set[str] = set()
    duplicates = 0
    empty = 0
    uncited = 0
    for row in rows:
        assistant = row["messages"][1]["content"]
        if row.get("kind") != "curriculum":
            body = assistant.rsplit("\n\nSource: ", 1)[0]
            if "\n\nSource: " not in assistant:
                uncited += 1
        else:
            body = assistant
        if len(body.strip()) < MIN_BODY_CHARS:
            empty += 1
        digest = digest_of(row)
        if digest in seen:
            duplicates += 1
        seen.add(digest)
    if uncited:
        findings.append(f"{uncited} grounded row(s) without a Source citation")
    if duplicates:
        findings.append(f"{duplicates} duplicate row(s)")
    if empty:
        findings.append(
            f"{empty} row(s) with an answer body under {MIN_BODY_CHARS} characters"
        )
    characters = sum(len(m["content"]) for r in rows for m in r["messages"])
    return {
        "rows": len(rows),
        "grounded": len(grounded),
        "curriculum": len(curriculum),
        "unique": len(seen),
        "approx_tokens": characters // 4,
        "findings": findings,
    }


def selftest() -> None:
    good = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "an answer long enough to count on its own\n\nSource: a/b.rs:1",
                },
            ],
            "kind": "doc",
            "citation": "a/b.rs:1",
        }
    ]
    report = evaluate(good)
    assert report["findings"] == [] and report["grounded"] == 1
    uncited = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "long enough text but no source line"},
            ],
            "kind": "doc",
            "citation": "x",
        }
    ]
    assert any("citation" in f for f in evaluate(uncited)["findings"])
    assert any("duplicate" in f for f in evaluate(good + good)["findings"])
    tiny = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "x\n\nSource: a:1"},
            ],
            "kind": "doc",
            "citation": "a:1",
        }
    ]
    assert any("under" in f for f in evaluate(tiny)["findings"])
    print("self-test OK")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        selftest()
        return 0
    rows = [
        json.loads(line)
        for line in Path(args.sft).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, ensure_ascii=False))
    if report["findings"]:
        for finding in report["findings"]:
            print(f"FINDING: {finding}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
