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
  truncated rows trip it;
* a grounded row whose passage carries an eval-only stamp (PP) - a passage
  marked as a held-out evaluation source never becomes a training row, and
  the leak is binary: one row is enough to refuse the set.

Measured on the self-corpus (2026-09-08): 776 rows (748 grounded + 28
curriculum), zero uncited, zero duplicates, minimum body 41.

Usage:
    python3 training/eval_sft.py --sft corpus/sft.jsonl
    python3 training/eval_sft.py --sft corpus/sft.jsonl --eval-only training/eval/eval-only.json
    python3 training/eval_sft.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MIN_BODY_CHARS = 20


def digest_of(row: dict) -> str:
    user = row["messages"][0]["content"]
    assistant = row["messages"][1]["content"]
    return hashlib.sha256((user + "\x00" + assistant).encode("utf-8")).hexdigest()


def load_eval_only(path: Path) -> set[str]:
    """The eval-only stamp list (PP). Fail closed: a list that cannot be
    read as a list of unique lowercase sha256 digests is refused, never
    treated as empty - an unreadable stamp wearing the shape of silence is
    exactly the leak this list exists to make impossible."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"{path}: not JSON: {err}") from err
    digests = data.get("digests") if isinstance(data, dict) else None
    if not isinstance(digests, list):
        raise SystemExit(f"{path}: `digests` must be a list")
    for digest in digests:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit(f"{path}: not a sha256 digest: {digest!r}")
    if len(set(digests)) != len(digests):
        raise SystemExit(f"{path}: duplicate digest")
    return set(digests)


def evaluate(rows: list[dict], eval_only: set[str] | None = None) -> dict:
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
    stamps = set(eval_only or ())
    leaked = sum(
        1 for row in grounded
        if row.get("content_id") and row["content_id"] in stamps
    )
    if leaked:
        findings.append(f"{leaked} grounded row(s) whose passage is stamped eval-only")
    characters = sum(len(m["content"]) for r in rows for m in r["messages"])
    return {
        "rows": len(rows),
        "grounded": len(grounded),
        "curriculum": len(curriculum),
        "unique": len(seen),
        "approx_tokens": characters // 4,
        "eval_only_stamps": len(stamps),
        "leaked": leaked,
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
    stamped = "a" * 64
    leak = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a body long enough to count on its own\n\nSource: a/b.rs:1",
                },
            ],
            "kind": "doc",
            "citation": "a/b.rs:1",
            "content_id": stamped,
        }
    ]
    assert any(
        "eval-only" in f for f in evaluate(leak, {stamped})["findings"]
    ), "a stamped row passed the evaluator: the leak check is decoration"
    assert not evaluate(leak, {"b" * 64})["findings"], (
        "an unstamped row was refused: the stamp list is not being read as a set"
    )
    print("self-test OK")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft")
    parser.add_argument(
        "--eval-only",
        default=str(Path(__file__).resolve().parent / "eval" / "eval-only.json"),
        help="eval-only stamp list (PP); passages listed here never train",
    )
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
    report = evaluate(rows, load_eval_only(Path(args.eval_only)))
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, ensure_ascii=False))
    if report["findings"]:
        for finding in report["findings"]:
            print(f"FINDING: {finding}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
