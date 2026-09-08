#!/usr/bin/env python3
"""Asama 3: korpus manifesti - gercek sample_count, tahmin yok.

AiDatasetMetadata::training(model_target, sample_count) kuralinin Lubot
tarafi: manifest `kind: TrainingCorpus` etiketiyle korpusun gercek ornek
sayisini ve lisans dagilimini tasir. `sample_count` korpus dosyasindan
SAYILIR (tahmin kabul edilmez). Zincir baglantisi (StorageDeal +
register_data_asset) dagitim tarafinda yapilir; burada hazirlanan alanlar
aynen tasinir.

Kullanim:
    python3 make_manifest.py --corpus corpus/knowledge-*.gz --out corpus/manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-target", default=None,
                    help="hesaplanmis model hash'i; egitim baslayinca doldurulur")
    args = ap.parse_args(argv)

    files = {}
    total = 0
    licenses: dict[str, int] = {}
    kinds: dict[str, int] = {}
    provenance_ok = 0
    for name in args.corpus:
        path = Path(name)
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                count += 1
                lic = rec.get("licence", "?")
                licenses[lic] = licenses.get(lic, 0) + 1
                kind = rec.get("kind", "?")
                kinds[kind] = kinds.get(kind, 0) + 1
                if rec.get("asset_id") and rec.get("content_id"):
                    provenance_ok += 1
        files[path.name] = count
        total += count

    manifest = {
        "kind": "TrainingCorpus",
        "sample_count": total,
        "model_target": args.model_target,
        "measured_at": "2026-09-07",
        "files": files,
        "by_licence": licenses,
        "by_kind": kinds,
        "records_with_provenance_pair": provenance_ok,
        "licence_notice": "PolyForm-Shield-1.0.0 (own work)",
        "chain_binding": "Pending: register_data_asset + StorageDeal at deployment; "
                         "asset_id pre-issuance stamp until then",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": total, "provenance_ok": provenance_ok, "files": files}))
    return 0 if provenance_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
