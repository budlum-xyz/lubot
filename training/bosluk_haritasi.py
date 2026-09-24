#!/usr/bin/env python3
"""Audit logundan bilgi-boslugu haritasi (RR).

`lubot ask`/`lubot batch` her soruyu `--audit` dosyasina bir JSONL satiri
olarak yazar (at, reader, question, answer, citations, decision, refusals,
effort, budget). Bu betik o gunlugu okur ve "en cok sorulan ama korpusta en
az karsiligi olan" konulari siralar - varsayimla degil, olcumle.

Kural (olcut tek yerde, tek cumlede):
    cevapsiz = refusals > 0 ya da citations bos.

Cikti saf bir fonksiyondur: ayni gunluk + ayni korpus -> ayni rapor; saat,
makine adi, dosya yolu disinda gecici hicbir sey tasimaz. Rapor ayrica
`kind: gap-report` kaydi olarak korpusa verilebilir; boylece "hangi
bosluklar kapandi, hangileri kroniklesti" sorusu kendi tarihini tutar.

    python3 training/bosluk_haritasi.py --audit outputs/audit.jsonl \
        --corpus corpus/knowledge-self.jsonl.gz \
        --out training/eval/sonuclar/bosluk-haritasi.json \
        --kayit training/eval/sonuclar/bosluk-haritasi.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

KONU_DESENI = re.compile(r"[0-9A-Za-z_]{4,}")

DURAK_KELIMELER = {
    # Turkce
    "icin", "ile", "olan", "olarak", "hangi", "nasil", "neden", "nerede",
    "kadar", "daha", "veya", "ancak", "yani", "bir", "bu", "su", "o",
    "mi", "mu", "midir", "mudur", "var", "yok", "the", "and",
    # Ingilizce
    "what", "which", "where", "when", "does", "with", "that", "this",
    "from", "into", "have", "has", "are", "was", "were", "how", "why",
}

SIRALAMA_UZUNLUK = 7


def konular(soru: str) -> set[str]:
    """Sorudan konu adaylari: kucuk harfli, 4+ karakter, durak kelimesiz."""
    adaylar = {k.lower() for k in KONU_DESENI.findall(soru)}
    return {k for k in adaylar if k not in DURAK_KELIMELER}


def cevapsiz(kayit: dict) -> bool:
    """Olcut: red varsa ya da alinti yoksa soru cevapsizdir."""
    return bool(kayit.get("refusals", 0)) or not kayit.get("citations")


def korpus_kapsamasi(korpus: Path | None, istenen: set[str]) -> dict[str, int]:
    """Konu basina korpusta eslesen kayit sayisi (yoksa 0, tahmin yok)."""
    sayim = dict.fromkeys(istenen, 0)
    if korpus is None:
        return sayim
    with gzip.open(korpus, "rt", encoding="utf-8") as fh:
        for satir in fh:
            satir = satir.strip()
            if not satir:
                continue
            kayit = json.loads(satir)
            metin = str(kayit.get("text", "")).lower()
            for konu in istenen:
                if konu in metin:
                    sayim[konu] += 1
    return sayim


def harita(audit_yolu: Path, korpus: Path | None) -> dict:
    """Gunlukten rapor: soru/cevapsiz sayilari ve sirali bosluk listesi."""
    satirlar = [
        json.loads(satir)
        for satir in audit_yolu.read_text(encoding="utf-8").splitlines()
        if satir.strip()
    ]
    sorular = {i: konular(str(s.get("question", ""))) for i, s in enumerate(satirlar)}
    tum_konular: set[str] = set().union(*sorular.values()) if sorular else set()
    kapsama = korpus_kapsamasi(korpus, tum_konular)

    kayitlar = []
    for konu in sorted(tum_konular):
        ilgili = [s for i, s in enumerate(satirlar) if konu in sorular[i]]
        kayitlar.append(
            {
                "konu": konu,
                "soru": len(ilgili),
                "cevapsiz": sum(1 for s in ilgili if cevapsiz(s)),
                "korpus_kaydi": kapsama.get(konu, 0),
            }
        )
    # Once "korpus karsiligi yok", sonra "cevapsizlik", sonra "talep".
    kayitlar.sort(key=lambda k: (k["korpus_kaydi"], -k["cevapsiz"], -k["soru"], k["konu"]))
    karsiligi_olan = sorted(
        (k for k in kayitlar if k["korpus_kaydi"] > 0),
        key=lambda k: (-k["korpus_kaydi"], k["konu"]),
    )
    return {
        "kaynak": str(audit_yolu),
        "soru": len(satirlar),
        "cevapsiz": sum(1 for s in satirlar if cevapsiz(s)),
        "konu_sayisi": len(tum_konular),
        "bosluklar": kayitlar[:SIRALAMA_UZUNLUK],
        "karsiligi_olan": karsiligi_olan[:SIRALAMA_UZUNLUK],
        "olcut": "cevapsiz = refusals > 0 ya da citations bos",
    }


def kayit_uret(rapor: dict, rapor_yolu: Path, kok: Path) -> dict:
    """Raporu `kind: gap-report` korpus kaydina cevirir (provenance agac ici)."""
    rel = rapor_yolu.resolve().relative_to(kok.resolve())
    bosluklar = ", ".join(k["konu"] for k in rapor["bosluklar"][:5]) or "yok"
    return {
        "kind": "gap-report",
        "text": (
            f"Bilgi boslugu haritasi ({rapor['soru']} soru, {rapor['cevapsiz']} cevapsiz): "
            f"en zayif konular {bosluklar}."
        ),
        "path": str(rel),
        "lines": [1, 1],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", help="ask/batch gunluk dosyasi (JSONL)")
    parser.add_argument("--corpus", default=None, help="kapsama icin korpus (jsonl.gz)")
    parser.add_argument("--out", default=None, help="rapor JSON yolu")
    parser.add_argument("--kayit", default=None, help="gap-report kaydi yolu (JSONL)")
    parser.add_argument("--kok", default=".", help="provenance icin agac koku")
    args = parser.parse_args(argv)

    if not args.audit:
        raise SystemExit("--audit gerekli: harita olculmeden yazilmaz")
    audit = Path(args.audit)
    if not audit.is_file():
        raise SystemExit(f"gunluk yok: {audit}")
    korpus = Path(args.corpus) if args.corpus else None

    rapor = harita(audit, korpus)
    metin = json.dumps(rapor, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(metin, encoding="utf-8")
    else:
        sys.stdout.write(metin)
    if args.kayit:
        kayit = kayit_uret(rapor, Path(args.out or audit), Path(args.kok))
        Path(args.kayit).write_text(
            json.dumps(kayit, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
