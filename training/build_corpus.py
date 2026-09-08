#!/usr/bin/env python3
"""Build the knowledge corpus from a checkout.

What goes in, and why:

* **documentation** - intent. Why a rule exists, which is the part source code
  never states.
* **public signatures** - surface. What can be called, and with what.
* **test names** - proven behaviour. A test name is the one sentence in a
  repository that someone had to make true.
* **gate names** - the rules that block a merge.

What stays out: raw function bodies. A model trained on raw source learns to
autocomplete source; the job here is to explain a protocol.

Every record carries where it came from - path and line range - so an answer
built from it can be walked back to the file. Records built from this
repository are stamped with its own licence and a pre-issuance provenance
pair (asset_id_pending), the same rule the v2 builder applies before a
chain TrainingDataGrant is issued.

    python3 training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

SKIP_DIRS = {".git", "target", "node_modules", "corpus", ".github"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


SELF_LICENCE = "PolyForm-Shield-1.0.0"
SELF_ATTRIBUTION = "lubot (kendi eser; LICENSE.md PolyForm Shield 1.0.0)"
# v2 ile ayni kural: grant cikmadan kanonik yer tutucu + pending damgasi.
SELF_ASSET_ID = digest("BDLM_LUBOT_CORPUS_ASSET_V1|self")


def walk(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def rust_records(path: Path, rel: str):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    doc_block: list[str] = []
    doc_start = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        if stripped.startswith("//!") or stripped.startswith("///"):
            if not doc_block:
                doc_start = i
            doc_block.append(stripped.lstrip("/!").strip())
            continue

        signature = re.match(r"pub (fn|struct|enum|trait|const|type) ([A-Za-z0-9_]+)", stripped)
        test_name = re.match(r"fn ([a-z0-9_]+)\(\)", stripped) if "#[test]" in "".join(lines[max(0, i - 2) : i]) else None

        if doc_block and (signature or test_name or stripped == ""):
            text = " ".join(w for w in doc_block if w)
            if len(text) > 40:
                yield {
                    "kind": "doc",
                    "text": text,
                    "path": rel,
                    "lines": [doc_start, i - 1],
                }
            doc_block = []

        if signature:
            yield {
                "kind": "api",
                "text": f"{rel} exposes `{stripped.rstrip(' {')}`.",
                "path": rel,
                "lines": [i, i],
            }
        if test_name:
            sentence = test_name.group(1).replace("_", " ")
            yield {
                "kind": "behaviour",
                "text": f"Proven in {rel}: {sentence}.",
                "path": rel,
                "lines": [i, i],
            }


def markdown_records(path: Path, rel: str):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    buffer: list[str] = []
    start = 1
    for i, line in enumerate(lines, 1):
        if line.strip():
            if not buffer:
                start = i
            buffer.append(line.strip())
            continue
        if buffer:
            text = " ".join(buffer)
            if len(text) > 60 and not text.startswith("|"):
                yield {"kind": "markdown", "text": text, "path": rel, "lines": [start, i - 1]}
            buffer = []
    if buffer:
        text = " ".join(buffer)
        if len(text) > 60:
            yield {"kind": "markdown", "text": text, "path": rel, "lines": [start, len(lines)]}


def gate_records(root: Path):
    gate_file = root / "gates" / "check.py"
    if not gate_file.is_file():
        return
    text = gate_file.read_text(encoding="utf-8")
    for match in re.finditer(r'def (gate_[a-z0-9_]+)\(\) -> str:\n    """(.+?)"""', text, re.S):
        name = match.group(1).removeprefix("gate_").replace("_", "-")
        summary = " ".join(match.group(2).split())
        yield {
            "kind": "doc",
            "text": f"The `{name}` gate blocks a merge unless: {summary}",
            "path": "gates/check.py",
            "lines": [text[: match.start()].count("\n") + 1, text[: match.end()].count("\n") + 1],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-name", default=None)
    args = parser.parse_args()

    root = Path(args.repo).resolve()
    source = args.source_name or root.name
    records = []

    for path in walk(root):
        rel = str(path.relative_to(root))
        if path.suffix == ".rs":
            records.extend(rust_records(path, rel))
        elif path.suffix == ".md":
            records.extend(markdown_records(path, rel))
    records.extend(gate_records(root))

    seen: set[str] = set()
    unique = []
    for record in records:
        key = digest(record["text"])
        if key in seen:
            continue
        seen.add(key)
        record["source"] = source
        record["digest"] = key
        record["licence"] = SELF_LICENCE
        record["attribution"] = SELF_ATTRIBUTION
        record["content_id"] = key
        record["asset_id"] = SELF_ASSET_ID
        record["asset_id_pending"] = True
        unique.append(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gz":
        handle = gzip.open(out, "wt", encoding="utf-8")
    else:
        handle = out.open("w", encoding="utf-8")
    with handle:
        for record in unique:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_kind: dict[str, int] = {}
    characters = 0
    for record in unique:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
        characters += len(record["text"])
    print(json.dumps({
        "records": len(unique),
        "by_kind": by_kind,
        "characters": characters,
        "approx_tokens": characters // 4,
        "source": source,
        "asset_id": SELF_ASSET_ID,
        "asset_id_pending": True,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
