#!/usr/bin/env python3
"""TrainingDataGrant epoch ledger - fail-closed epoch accounting.

The epoch-budget rule of the education report (stage 2): a corpus pass may
start only while the grant is valid, and every completed epoch must be
consumed. The authority is the Rust `TrainingDataGrant` / `EpochBook` in
`crates/grant/src/training.rs`; this ledger is the pipeline-side mirror of
the same field set (asset_id, owner, grantee, issued_at_block,
expires_at_block, max_epochs, epochs_used) and refuses to run once the
limit is reached. A runner that skips either side is not consuming epochs,
and a corpus pass that never consumes a grant is exactly the
unbounded-read hole the grant exists to close. The grant book lives inside
Lubot; no chain-side issuance is required (and none is assumed).

Usage:
    python3 epoch_ledger.py --check ledger.json --now 1000
    python3 epoch_ledger.py --consume ledger.json --now 1000
    python3 epoch_ledger.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = [
    "asset_id",
    "owner",
    "grantee",
    "issued_at_block",
    "expires_at_block",
    "max_epochs",
    "epochs_used",
]


def open_if_valid(ledger: dict, now_block: int) -> None:
    """Fail-closed: only a grant that is in time and not exhausted opens."""
    for field in REQUIRED:
        if field not in ledger:
            raise SystemExit(f"epoch ledger malformed: missing {field}")
    if now_block > ledger["expires_at_block"]:
        raise SystemExit(
            f"training-data grant expired: now {now_block} > "
            f"expires_at_block {ledger['expires_at_block']}"
        )
    if ledger["epochs_used"] >= ledger["max_epochs"]:
        raise SystemExit(
            "training-data grant epochs exhausted: "
            f"{ledger['epochs_used']}/{ledger['max_epochs']}"
        )


def consume(ledger: dict, now_block: int) -> dict:
    """Consume one epoch; refuses at the limit (mirrors `consume_epoch`)."""
    open_if_valid(ledger, now_block)
    out = dict(ledger)
    out["epochs_used"] += 1
    return out


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--self-test":
        good = {
            "asset_id": "a" * 64, "owner": "o", "grantee": "g",
            "issued_at_block": 0, "expires_at_block": 10,
            "max_epochs": 2, "epochs_used": 0,
        }
        open_if_valid(good, 10)
        after = consume(good, 10)
        assert after["epochs_used"] == 1, "consume must increment"
        try:
            consume(consume(after, 10), 10)
            raise AssertionError("third epoch must be refused")
        except SystemExit:
            pass
        try:
            open_if_valid(good, 11)
            raise AssertionError("past expiry must be refused")
        except SystemExit:
            pass
        print("self-test OK")
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True)
    ap.add_argument("--consume", action="store_true")
    ap.add_argument("--now", type=int, required=True)
    args = ap.parse_args(argv)

    ledger = json.loads(Path(args.check).read_text(encoding="utf-8"))
    if args.consume:
        ledger = consume(ledger, args.now)
        Path(args.check).write_text(
            json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"epoch consumed: {ledger['epochs_used']}/{ledger['max_epochs']}")
    else:
        open_if_valid(ledger, args.now)
        print(
            f"grant open: {ledger['epochs_used']}/{ledger['max_epochs']} epochs used"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
