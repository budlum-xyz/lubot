#!/usr/bin/env python3
"""NN-1: olculen donanimdan model boyutu tavanini turetir.

Girdi: bench_hardware.py ciktisi (--bench, zorunlu). Korpus istege baglidir
(--corpus): olculen kayit ve karakter sayilari oran raporuna girer; donmus
sozluk de verilebilir (--vocab) ki gercek BPE token sayisi olculebilsin.

Kural (HH, U): her sayi kaynagiyla etiketlenir.
  olculdu    = girdi dosyadan okundu; olcumu yapan bench_hardware.py'dir
  turetildi  = asagida yazili sabit formulle hesaplandi; formulu metinde tasir
  olculmedi  = bilinmiyor; hicbir varsayimla doldurulmaz

Muhasebe varsayimlari (turetildi etiketinin parcasi, koda gomulu sabit):
  egitim fp32 AdamW     16 B/param  (agirlik 4 + gradyan 4 + m 4 + v 4)
  egitim bf16 agirlik   12 B/param  (agirlik 2 + gradyan 2 + m 4 + v 4)
  cikarim fp32           4 B/param
  cikarim bf16           2 B/param
Aktivasyon bellegi sayilmaz (olculmedi): dizi uzunluguna ve mimariye bagli;
kapasite planina girmeden once ayrica olculmesi gerekir.

Kullanim:
    python3 recommend_model_size.py --bench bench.json
    python3 recommend_model_size.py --bench bench.json --corpus corpus/knowledge-self.jsonl.gz \
        --vocab training/tokenizer/lubot-bpe-v1.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

# Sabit muhasebe: bayt/parametre. Formuller acik yazilir; kapali sabit yok.
BYTES_PER_PARAM = {
    "train_fp32_adamw": {"bytes": 16, "formula": "4 agirlik + 4 gradyan + 4 m + 4 v"},
    "train_bf16_weights": {"bytes": 12, "formula": "2 agirlik + 2 gradyan + 4 m + 4 v"},
    "serve_fp32": {"bytes": 4, "formula": "yalnizca agirliklar"},
    "serve_bf16": {"bytes": 2, "formula": "yalnizca agirliklar"},
}

# Disaridan alinan tek oran: klasik token/parametre dengesi. Bu makinede
# olculmus bir sayi DEGILDIR; yaygin kabullenilmis bir referanstir ve raporda
# hep bu etiketle yazilir (U: yayinlanmis dis sayi asla kendi olcumu gibi
# sunulmaz).
REFERENCE_TOKENS_PER_PARAM = 20


def load_bench(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"bench raporu yok: {path} (once bench_hardware.py --out calistir)")
    return json.loads(p.read_text(encoding="utf-8"))


def corpus_stats(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"korpus yok: {path}")
    opener = gzip.open if p.suffix == ".gz" else open
    records = 0
    characters = 0
    sha = hashlib.sha256()
    with opener(p, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            sha.update(line.encode("utf-8"))
            rec = json.loads(line)
            records += 1
            characters += len(rec.get("text", ""))
    return {
        "records": records,
        "characters": characters,
        "approx_tokens_chars_over_4": characters // 4,
        "corpus_sha256": sha.hexdigest(),
    }


def bpe_token_count(corpus_path: str, vocab_path: str) -> dict:
    """Donmus sozlukle gercek token sayimi (olculdu)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_tokenizer as tt  # yerel, std-only

    vocab = tt.load_vocab(vocab_path)
    opener = gzip.open if Path(corpus_path).suffix == ".gz" else open
    total = 0
    with opener(corpus_path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            total += len(tt.encode(json.loads(line).get("text", ""), vocab))
    return {"bpe_tokens": total, "vocab_family": vocab["vocab_family"]}


def candidate_rows(usable_bytes: int, corpus: dict | None, bpe: dict | None) -> list[dict]:
    rows = []
    for params in (1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000):
        row: dict = {"params": params}
        for regime, spec in BYTES_PER_PARAM.items():
            need = params * spec["bytes"]
            row[regime] = {
                "bytes": need,
                "fits": need <= usable_bytes,
            }
        if bpe:
            row["bpe_tokens_per_param"] = round(bpe["bpe_tokens"] / params, 4)
        elif corpus:
            row["approx_tokens_per_param"] = round(corpus["approx_tokens_chars_over_4"] / params, 4)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="bench_hardware.py cikti JSON dosyasi")
    ap.add_argument("--corpus", default=None, help="korpus jsonl(.gz); oran raporu icin")
    ap.add_argument("--vocab", default=None, help="donmus sozluk JSON'u; gercek token sayimi icin")
    ap.add_argument("--ram-fraction", type=float, default=0.75,
                    help="egitim icin RAM'in kullanilacak payi (varsayilan 0.75)")
    args = ap.parse_args(argv)

    bench = load_bench(args.bench)
    ram = int(bench.get("ram_bytes") or 0)
    if ram <= 0:
        raise SystemExit("bench raporunda gecerli ram_bytes yok; olcum eksik")
    usable = int(ram * args.ram_fraction)

    measured_inputs = {
        "ram_bytes": {"value": ram, "label": "olculdu"},
        "ram_gib": {"value": bench.get("ram_gib"), "label": "olculdu"},
        "cpu_cores": {"value": bench.get("cpu_cores"), "label": "olculdu"},
        "disk_free_gib": {"value": bench.get("disk_free_gib"), "label": "olculdu"},
        "throughput": {"value": bench.get("throughput", "olculmedi"), "label":
                       "olculdu" if bench.get("throughput") else "olculmedi"},
        "gpu": {"value": bench.get("gpu", "olculmedi"), "label":
                "olculdu" if bench.get("gpu") else "olculmedi"},
    }

    corpus = corpus_stats(args.corpus) if args.corpus else None
    bpe = bpe_token_count(args.corpus, args.vocab) if (args.corpus and args.vocab) else None

    derived: dict = {
        "usable_bytes": {
            "value": usable,
            "formula": f"ram_bytes * {args.ram_fraction}",
            "label": "turetildi",
        },
        "max_params": {},
    }
    for regime, spec in BYTES_PER_PARAM.items():
        derived["max_params"][regime] = {
            "value": usable // spec["bytes"],
            "formula": f"usable_bytes // {spec['bytes']} ({spec['formula']})",
            "label": "turetildi",
        }

    report: dict = {
        "purpose": "NN-1: K6 donanim tavanindan model boyutu turetimi",
        "measured_inputs": measured_inputs,
        "derived": derived,
        "candidates": candidate_rows(usable, corpus, bpe),
        "not_measured": [
            "aktivasyon bellegi (dizi uzunlugu ve mimariye bagli; ayrica olculmeli)",
            "egitim suresi (is verimi sinyali sha256/memcpy dolaylidir; GEMS olcumu yok)",
        ],
    }

    if corpus:
        report["corpus"] = {k: {"value": v, "label": "olculdu"} for k, v in corpus.items()}
    if bpe:
        report["bpe"] = {k: {"value": v, "label": "olculdu"} for k, v in bpe.items()}
        tokens = bpe["bpe_tokens"]
    elif corpus:
        tokens = corpus["approx_tokens_chars_over_4"]
    else:
        tokens = None

    if tokens is not None:
        balanced = tokens / REFERENCE_TOKENS_PER_PARAM
        report["reference_balance"] = {
            "tokens": tokens,
            "reference_tokens_per_param": REFERENCE_TOKENS_PER_PARAM,
            "params_at_reference_balance": int(balanced),
            "label": "turetildi; oran disaridan kabullenilmis referans, bu makinede olculmedi",
            "note": ("korpus token sayisi bu referansla karsilastirildiginda parametre "
                     "tavani veri tarafindan baglaniyorsa GG'deki siniflandirma gecerlidir: "
                     "kapisma iddiasi gorev-eslenegi eksende tanimlanir"),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
