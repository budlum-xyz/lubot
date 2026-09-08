#!/usr/bin/env python3
"""Examine a finding before it is allowed to exist.

A finding is a claim the reader makes about code; this module measures the
claim instead of trusting it. The discipline is ours: evidence before
reporting, written for this repository's reading identity.

Four refusals, each a leak the discipline says cannot happen:

* no location - a finding without `path:line` points at nothing, so it is
  a suspicion, not a finding;
* a static-only reading reported as proven - without a reproduction, the
  record must carry the unverified tag, or the report is the very false
  positive the discipline exists to prevent;
* a severity rating with no change condition - the rating must state the
  one concrete piece of evidence that would raise or lower it, otherwise
  the number is a guess wearing a badge;
* a duplicate - two findings at the same sink collapse into one, because
  repetition teaches repetition, not knowledge.

Usage:
    python3 training/findings.py --file findings.jsonl
    python3 training/findings.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys

LOCATION_RE = re.compile(r"^[A-Za-z0-9_./-]+:\d+(-\d+)?$")
SEVERITIES = {"low", "medium", "high", "critical"}


def validate(findings: list[dict]) -> dict:
    """Measure the findings; `findings` empty means the set is reportable."""
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(findings):
        title = str(record.get("title", "")).strip()
        who = f"finding {index + 1}" + (f" ({title})" if title else "")

        location = str(record.get("location", "")).strip()
        if not LOCATION_RE.match(location):
            problems.append(f"{who}: no usable location - a finding must point at `path:line`")

        evidence = str(record.get("evidence", "")).strip()
        status = str(record.get("status", "")).strip()
        if status not in {"unverified", "reproduced"}:
            problems.append(f"{who}: status must be `reproduced` or `unverified`, not `{status}`")
        elif status == "reproduced" and len(evidence) < 20:
            problems.append(
                f"{who}: claimed reproduced without a reproduction to show for it"
            )

        severity = str(record.get("severity", "")).strip()
        if severity:
            if severity not in SEVERITIES:
                problems.append(f"{who}: severity `{severity}` is not one of {sorted(SEVERITIES)}")
            condition = str(record.get("severity_change_conditions", "")).strip()
            if not condition:
                problems.append(
                    f"{who}: rated `{severity}` without the condition that would change it"
                )

        sink = str(record.get("sink", "")).strip()
        if location and sink:
            key = (location.split(":")[0], sink)
            if key in seen:
                problems.append(f"{who}: duplicate of an earlier finding at the same sink")
            seen.add(key)

    return {"rows": len(findings), "findings": problems}


def self_test() -> int:
    """The canaries: every refusal must fire, and a clean finding must pass."""
    clean = {
        "title": "tenant filter missing in shared query helper",
        "location": "src/db/query.rs:41",
        "sink": "execute_query",
        "status": "reproduced",
        "evidence": "request with tenant id 2 returns rows owned by tenant id 1",
        "severity": "high",
        "severity_change_conditions": "drops if the helper is unreachable from a public route",
    }
    assert validate([clean])["findings"] == [], "a clean finding must pass"

    no_location = dict(clean, location="somewhere in auth")
    assert any("location" in p for p in validate([no_location])["findings"]), (
        "a finding without a location must be refused"
    )

    untagged_static = dict(clean, status="proven", evidence="it looks unsafe")
    assert any("status" in p for p in validate([untagged_static])["findings"]), (
        "a static reading reported as proven must be refused"
    )

    rating_without_condition = dict(clean, severity_change_conditions="")
    assert any("condition" in p for p in validate([rating_without_condition])["findings"]), (
        "a rating without its change condition must be refused"
    )

    duplicate = dict(clean)
    assert any("duplicate" in p for p in validate([clean, duplicate])["findings"]), (
        "two findings at the same sink must collapse"
    )
    print("self-test: five canaries, every refusal fires")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.file:
        parser.error("either --file or --self-test")
    rows = [json.loads(line) for line in open(args.file, encoding="utf-8") if line.strip()]
    report = validate(rows)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
