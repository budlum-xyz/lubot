#!/usr/bin/env python3
"""Turn the knowledge corpus into a supervised set.

Two kinds of record come out:

* **grounded** - built from the corpus. The answer is the corpus text and the
  citation is the path and line range it came from. A record whose answer
  cannot cite is dropped rather than smoothed over.
* **curriculum** - hand written, in `training/curriculum/*.jsonl`. These teach
  the shape of an answer rather than a fact: call the calculator instead of
  predicting a product, refuse without a grant, say "not measured" instead of
  producing a number that was never measured.

    python3 training/make_sft.py --corpus corpus/knowledge.jsonl \\
        --curriculum training/curriculum --out corpus/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUESTION_BY_KIND = {
    "doc": "Why does {path} work the way it does?",
    "api": "What does {path} expose?",
    "behaviour": "What behaviour is proven in {path}?",
    "markdown": "What does {path} say about this?",
    "gate": "Which rule blocks a merge here?",
}


def grounded(record: dict) -> dict | None:
    template = QUESTION_BY_KIND.get(record["kind"])
    if not template:
        return None
    first, last = record["lines"]
    citation = f"{record['path']}:{first}" if first == last else f"{record['path']}:{first}-{last}"
    return {
        "messages": [
            {"role": "user", "content": template.format(path=record["path"])},
            {"role": "assistant", "content": f"{record['text']}\n\nSource: {citation}"},
        ],
        "kind": record["kind"],
        "citation": citation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    dropped = 0

    with Path(args.corpus).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            row = grounded(record)
            if row is None:
                dropped += 1
                continue
            rows.append(row)

    curriculum_dir = Path(args.curriculum)
    curriculum = 0
    if curriculum_dir.is_dir():
        for path in sorted(curriculum_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["kind"] = "curriculum"
                    rows.append(row)
                    curriculum += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "rows": len(rows),
        "grounded": len(rows) - curriculum,
        "curriculum": curriculum,
        "dropped_without_citation": dropped,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
