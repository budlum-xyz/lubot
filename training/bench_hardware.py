#!/usr/bin/env python3
"""K6 destegi: sifirdan egitim kosusunun donanim sinavini olcer.

K6 karari: sifirdan egitim sahibin kendi donaniminda kosar; model boyutu
olculen donanim ile sinirlanir. Bu betik kosu makinesinin somut sinirlarini
raporlar (sadece sayar; model secmez). Cikti, egitim planindaki
model-boyutu kararinin girdisidir.

Kullanim:
    python3 bench_hardware.py            # CPU + RAM + disk
    python3 bench_hardware.py --gpu      # nvidia-smi varsa GPU raporu

Not: olcum kosu makinesinindir; sahibin donanimi FARKLI olabilir. Betik
makineyi degil, K6 kosu listesindeki adimi yapar.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def cpu_cores() -> int:
    return os.cpu_count() or 0


def ram_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def disk_free_bytes() -> int:
    try:
        usage = shutil.disk_usage(".")
        return usage.free
    except OSError:
        return 0


def gpu_report() -> dict | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return {"gpus": [{"name": ln.split(",")[0].strip(), "memory_total": ln.split(",")[1].strip()} for ln in lines]}
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args(argv)

    report = {
        "host": platform.node() or "unknown",
        "python": platform.python_version(),
        "cpu_cores": cpu_cores(),
        "ram_bytes": ram_bytes(),
        "ram_gib": round(ram_bytes() / (1024 ** 3), 2),
        "disk_free_bytes": disk_free_bytes(),
        "disk_free_gib": round(disk_free_bytes() / (1024 ** 3), 2),
        "with_gpu": args.gpu,
    }
    if args.gpu:
        report["gpu"] = gpu_report() or "nvidia-smi yok / GPU bulunamadi"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
