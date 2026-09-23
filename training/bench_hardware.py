#!/usr/bin/env python3
"""K6 destegi: sifirdan egitim kosusunun donanim sinavini olcer.

K6 karari: sifirdan egitim sahibin kendi donaniminda kosar; model boyutu
olculen donanim ile sinirlanir. Bu betik kosu makinesinin somut sinirlarini
sayar ve (--throughput ile) iki is verimi sinyali olcer. Model secmez; o
karar training/recommend_model_size.py'nin isidir ve bu betiklerin ciktisi
o kararın girdisidir.

Kullanim:
    python3 bench_hardware.py                          # CPU + RAM + disk sayimi
    python3 bench_hardware.py --throughput             # + sha256 ve memcpy olcumleri
    python3 bench_hardware.py --gpu                    # nvidia-smi varsa GPU raporu
    python3 bench_hardware.py --throughput --out bench.json

Olculen is verimi sinyalleri (--throughput):
    sha256_mib_per_s   64 MiB tampon uzerinde tek cekirdek SHA-256, 3 kosunun medyani
    memcpy_gib_per_s   64 MiB tampon kopyalama, toplam 1 GiB, 3 kosunun medyani

Not: olcum kosu makinesinindir; sahibin donanimi FARKLI olabilir. Betik
makineyi degil, K6 kosu listesindeki adimi yapar. Burada yazilan her is
verimi sayisi bu makinede olculmustur; baska makinenin degeri ancak o
makinede olculebilir. Sayilar tek basina bir iddia tasimaz; asil olan
recommend_model_size.py'nin bu sayilari etiketiyle tuketmesidir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time

SCRIPT_VERSION = 2


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


def measure_sha256(runs: int = 3, mib: int = 64) -> dict:
    """Tek cekirdek SHA-256 is verimi: 64 MiB tampon, medyan."""
    buffer = bytes(mib * 1024 * 1024)
    speeds = []
    for _ in range(runs):
        started = time.perf_counter()
        hashlib.sha256(buffer).digest()
        elapsed = time.perf_counter() - started
        if elapsed > 0:
            speeds.append(mib / elapsed)
    if not speeds:
        return {"sha256_mib_per_s": None, "method": "olcum kosamadi: sure sifir"}
    return {
        "sha256_mib_per_s": round(statistics.median(speeds), 2),
        "method": f"tek cekirdek, {mib} MiB tampon, {runs} kosu medyani",
    }


def measure_memcpy(runs: int = 3, chunk_mib: int = 64, total_gib: float = 1.0) -> dict:
    """Bellek kopyalama bant genisligi: 64 MiB parcalarla toplam 1 GiB, medyan."""
    src = bytearray(chunk_mib * 1024 * 1024)
    dst = bytearray(chunk_mib * 1024 * 1024)
    copies = max(1, int(total_gib * 1024 / chunk_mib))
    speeds = []
    for _ in range(runs):
        started = time.perf_counter()
        for _ in range(copies):
            dst[:] = src
        elapsed = time.perf_counter() - started
        if elapsed > 0:
            speeds.append((copies * chunk_mib / 1024) / elapsed)
    if not speeds:
        return {"memcpy_gib_per_s": None, "method": "olcum kosamadi: sure sifir"}
    return {
        "memcpy_gib_per_s": round(statistics.median(speeds), 2),
        "method": f"{chunk_mib} MiB parca, toplam {total_gib:g} GiB, {runs} kosu medyani",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--throughput", action="store_true",
                    help="sha256 ve memcpy is verimi olcumlerini de kapsa")
    ap.add_argument("--out", default=None,
                    help="raporu ayrica bu dosyaya yaz (JSON)")
    args = ap.parse_args(argv)

    report = {
        "script_version": SCRIPT_VERSION,
        "host": platform.node() or "unknown",
        "python": platform.python_version(),
        "cpu_cores": cpu_cores(),
        "ram_bytes": ram_bytes(),
        "ram_gib": round(ram_bytes() / (1024 ** 3), 2),
        "disk_free_bytes": disk_free_bytes(),
        "disk_free_gib": round(disk_free_bytes() / (1024 ** 3), 2),
        "with_gpu": args.gpu,
    }
    if args.throughput:
        report["throughput"] = {
            "sha256": measure_sha256(),
            "memcpy": measure_memcpy(),
        }
    if args.gpu:
        report["gpu"] = gpu_report() or "nvidia-smi yok / GPU bulunamadi"

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
