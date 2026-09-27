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
import sys
from pathlib import Path

QUESTION_BY_KIND = {
    "doc": "Why does {path} work the way it does?",
    "api": "What does {path} expose?",
    "behaviour": "What behaviour is proven in {path}?",
    "markdown": "What does {path} say about this?",
    # On-chain records: the evidence is the chain
    # itself, so the question asks what the record proves, not what it "says".
    "chain": "What does the on-chain record at {path} prove?",
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
        # The passage's own digest travels with the row so a leak check stays
        # mechanical: the evaluator compares this against the eval-only stamp
        # list (PP) instead of re-parsing a citation string.
        "content_id": record.get("content_id") or record.get("digest"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--eval-only",
        default="training/eval/eval-only.json",
        help="held-out stamp list; stamped passages never become training rows",
    )
    args = parser.parse_args()

    # Prevention, not only detection. `eval_sft` refuses a set that contains a
    # stamped passage, but a refusal after the set is built is a refused run,
    # not a held-out exam set. Dropping the row here is what makes the exam set
    # actually held out; the evaluator's refusal stays as the second wall.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_sft  # noqa: PLC0415  (the reader that owns the stamp contract)

    stamps = eval_sft.load_eval_only(Path(args.eval_only))

    rows: list[dict] = []
    dropped = 0
    dropped_eval_only = 0

    with Path(args.corpus).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("content_id") in stamps:
                dropped_eval_only += 1
                continue
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
        "dropped_eval_only": dropped_eval_only,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
