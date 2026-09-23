#!/usr/bin/env python3
"""Rust ve Python jetonlayicilarinin ayni seyi yaptigini olcer.

Iki jetonlayicinin "anlasmasi" bir uzlasma degildir: sozlugu kesen taraf
Python, okuyan taraf Rust ve ikisi de ayni donmus dosyayi kullaniyor. Bu
betik ikisini korpusun tamaminda, kayit kayit, id id karsilastirir.

Kullanim:
    python3 training/jeton_capraz.py --vocab training/tokenizer/lubot-bpe-v2.json \
        --corpus corpus/knowledge-self.jsonl.gz [--limit N] [--json]

Cikis: uyusmazlik varsa 0 olmayan kod ve ilk uyusmaz kayitlar; yoksa olculen
sayilar. Hicbir sayi tahmin edilmez.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "training"))

from train_tokenizer import encode, load_vocab  # noqa: E402


def kayitlar(korpus: Path) -> list[str]:
    ac = gzip.open if korpus.suffix == ".gz" else open
    metinler: list[str] = []
    with ac(korpus, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for satir in fh:
            satir = satir.strip()
            if satir:
                metinler.append(json.loads(satir)["text"])
    return metinler


def rust_kimlikleri(vocab: Path, korpus: Path, limit: int, cargo: bool = False) -> list[list[int]]:
    """`lubot jetonla --tam` ciktisini okur.

    Rust tarafi mutlaka CALISTIRILIR; bu betik Rust kodunu yeniden yazip
    karsilastirmaz, o zaman kendi taklidini olcmus olurdu."""
    if cargo:
        komut = ["cargo", "run", "-q", "--bin", "lubot", "--"]
    else:
        ikili = KOK / "target" / "debug" / "lubot"
        if not ikili.is_file():
            ikili = KOK / "target" / "release" / "lubot"
        if not ikili.is_file():
            raise SystemExit(
                "lubot ikilisi bulunamadi: once `cargo build`, ya da --cargo verin"
            )
        komut = [str(ikili)]
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as gecici:
        yol = gecici.name
    try:
        with open(yol, "w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [*komut, "jetonla", "--vocab", str(vocab), "--corpus", str(korpus),
                 "--limit", str(limit), "--tam"],
                cwd=KOK, stdout=fh, text=True, check=False,
            )
        if proc.returncode != 0:
            ipucu = (proc.stderr or "").strip()
            if "rustup" in ipucu or "cargo" in ipucu.splitlines()[:1]:
                raise SystemExit(
                    "cargo calistirilamadi (PATH'te degil ya da toolchain kurulu "
                    f"degil). Rust tarafi olculemedi, jetonlayici karsilastirmasi "
                    f"YAPILMADI. Ayrıntı: {ipucu[-300:]}"
                )
            raise SystemExit(
                f"`lubot jetonla` basarisiz (kod {proc.returncode}): {ipucu[-300:]}"
            )
        with open(yol, encoding="utf-8") as fh:
            return [json.loads(l)["ids"] for l in fh if l.strip()]
    finally:
        Path(yol).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", default="training/tokenizer/lubot-bpe-v2.json")
    ap.add_argument("--corpus", default="corpus/knowledge-self.jsonl.gz")
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cargo", action="store_true",
                    help="hazir ikili yerine `cargo run` kullan")
    args = ap.parse_args(argv)

    vocab_yolu = (KOK / args.vocab).resolve()
    korpus_yolu = (KOK / args.corpus).resolve()
    sozluk = load_vocab(str(vocab_yolu))
    metinler = kayitlar(korpus_yolu)
    rust = rust_kimlikleri(vocab_yolu, korpus_yolu, args.limit, cargo=args.cargo)

    karsilastirilan = min(len(metinler), len(rust))
    uyusmaz: list[dict[str, object]] = []
    python_toplam = 0
    for i in range(karsilastirilan):
        py = encode(metinler[i], sozluk)
        python_toplam += len(py)
        if py != rust[i]:
            if len(uyusmaz) < 5:
                ilk = next(
                    (j for j in range(min(len(py), len(rust[i]))) if py[j] != rust[i][j]),
                    min(len(py), len(rust[i])),
                )
                uyusmaz.append({
                    "kayit": i,
                    "python_uzunluk": len(py),
                    "rust_uzunluk": len(rust[i]),
                    "ilk_fark_indeksi": ilk,
                    "python_cevresi": py[max(0, ilk - 3):ilk + 3],
                    "rust_cevresi": rust[i][max(0, ilk - 3):ilk + 3],
                    "metin_basi": metinler[i][:60],
                })
    rust_toplam = sum(len(r) for r in rust)
    sonuc = {
        "sozluk": sozluk["vocab_family"],
        "korpus_kayit": len(metinler),
        "karsilastirilan_kayit": karsilastirilan,
        "python_toplam_jeton": python_toplam,
        "rust_toplam_jeton": rust_toplam,
        "uyusmaz_kayit": len(uyusmaz),
        "ornekler": uyusmaz,
    }
    if args.json:
        print(json.dumps(sonuc, ensure_ascii=False, indent=1))
    else:
        print(
            f"{sonuc['sozluk']}: {karsilastirilan}/{len(metinler)} kayit karsilastirildi, "
            f"python {python_toplam} jeton, rust {rust_toplam} jeton, "
            f"uyusmaz kayit {len(uyusmaz)}"
        )
        for ornek in uyusmaz:
            print(f"  kayit {ornek['kayit']}: {ornek['metin_basi']!r}")
            print(f"    ilk fark {ornek['ilk_fark_indeksi']}: "
                  f"py {ornek['python_cevresi']} vs rust {ornek['rust_cevresi']}")
    # Rust tarafi --limit kadar kayit dondurur; sinir korpusun tamamina
    # yetmiyorsa daha az kayit donmesi beklenir, kusur degildir. Kusur olan,
    # sinir yeterken eksik kayit donmesi veya karsilastirilan sayinin Rust
    # tarafindan uretilenden fazla olmasi.
    eksik = karsilastirilan != len(rust) or (args.limit >= len(metinler) and len(rust) != len(metinler))
    if uyusmaz or eksik:
        print(
            "KIRMIZI: iki jetonlayici ayni sozlukle ayni metni ayni bicimde "
            "jetonlamiyor", file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
