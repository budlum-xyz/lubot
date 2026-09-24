#!/usr/bin/env python3
"""Alma (retrieval) yuzeyinin olcumu: soru basina sira (I bolumu).

Sinav setindeki her soru, damgalanmis bir pasaja dayanir (content_id).
`lubot ara` ayni korpusu BM25 ile tarayip en iyi n pasaji dondurur; bu
betik damgalanmis pasajin kacinci sirada ciktigini olcer ve ilk-sira /
ilk-n isabetini raporlar. Olcum model cagirmaz: egitilmis kontrol noktasi
beklemez, cunku alma katmani agirlik kullanmaz.

Olcut tek cumlede:
    isabet = damgalanmis content_id'nin (path, satir) cifti, donen
    alintilarin ilk n tanesinden birinin path'ine ve satir araligina duser.

Bulunamayan soru "0" diye sayilmaz; ayri bir listede adiyla durur - "yok"
ile "olculmedi" ayni sey degildir.

    python3 training/erisim_geri_cagirma.py \
        --corpus corpus/knowledge-self.jsonl.gz \
        --sorular training/eval/sinav-seti.jsonl \
        --bin target/release/lubot --n 3 --out training/eval/sonuclar/erisim.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ALINTI = re.compile(r"^- `([^`]+)` \(licence `[^`]*`\)")
PASAJ = re.compile(r"```\n(.*?)\n```", re.S)


def korpus_dizini(corpus: Path) -> dict[str, tuple[str, str]]:
    """content_id -> (path, kayit metni)."""
    dizin: dict[str, tuple[str, str]] = {}
    with gzip.open(corpus, "rt", encoding="utf-8") as fh:
        for satir in fh:
            satir = satir.strip()
            if not satir:
                continue
            kayit = json.loads(satir)
            cid = kayit.get("content_id")
            yol = kayit.get("path")
            if cid and yol:
                dizin[str(cid)] = (str(yol), str(kayit.get("text", "")))
    return dizin


def alintilari_ayikla(markdown: str) -> list[tuple[str, str]]:
    """Donen pasajlar: (alinti, metin) ciftleri, sirasi korunarak."""
    alintilar = [m.group(1) for m in (ALINTI.match(s) for s in markdown.splitlines()) if m]
    pasajlar = PASAJ.findall(markdown)
    return list(zip(alintilar, pasajlar))


def eslesme(alinti: str, metin: str, yol: str, kayit_metni: str) -> bool:
    """Isabet: ayni dosya + donen pasaj, damgalanmis kaydin metnini tasiyor.

    Alintinin `path:sayi` kismindaki sayi pasajin kayit icindeki siradir,
    dosya satiri degil; bu yuzden eslesme dosya adi + metin uzerinden
    kurulur (satir numarasi iki tarafta ayni seyi gostermiyor)."""
    if not alinti.startswith(yol + ":"):
        return False
    sade = " ".join(kayit_metni.split())
    donen = " ".join(metin.split())
    return bool(sade) and (sade in donen or donen in sade)


def soru_sirasi(binary: list[str], corpus: Path, soru: str, n: int) -> list[tuple[str, str]]:
    kosu = subprocess.run(
        [*binary, "ara", "--corpus", str(corpus), "--n", str(n), soru],
        capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(f"`ara` dustu: {(kosu.stderr or kosu.stdout)[-300:]}")
    return alintilari_ayikla(kosu.stdout)


def sorgu_metni(soru: str, alan: str) -> str:
    """`tam`: sorunun tamami; `alinti`: son ':' sonrasi cekirdek metin.

    Sinav sorulari bir yonerge ile baslar ("... kaynagini gostererek aktar:")
    ve ardindan pasajin acilis cumlesini tasir. Iki alani ayri olcmek,
    kaybip kaybetmedigimizi gosterir: alma cekirdegi ne kadar buluyor,
    yonerge onu ne kadar seyreltiyor."""
    if alan == "alinti" and ":" in soru:
        return soru.rsplit(":", 1)[1].strip() or soru
    return soru


def olc(binary: list[str], corpus: Path, sorular: list[dict], n: int, alan: str = "tam") -> dict:
    dizin = korpus_dizini(corpus)
    sonuclar = []
    bulunamayan = []
    ilk_sirada = 0
    ilk_n = 0
    for kayit in sorular:
        cid = str(kayit.get("content_id", ""))
        hedef = dizin.get(cid)
        if hedef is None:
            bulunamayan.append(
                {"soru": kayit.get("soru_kimligi"), "neden": "content_id korpusta yok"}
            )
            continue
        yol, kayit_metni = hedef
        alintilar = soru_sirasi(binary, corpus, sorgu_metni(str(kayit.get("soru", "")), alan), n)
        sira = next(
            (i + 1 for i, (a, metin) in enumerate(alintilar) if eslesme(a, metin, yol, kayit_metni)),
            None,
        )
        if sira is None:
            bulunamayan.append({"soru": kayit.get("soru_kimligi"), "neden": "ilk n icinde yok"})
        else:
            if sira == 1:
                ilk_sirada += 1
            ilk_n += 1
        sonuclar.append(
            {
                "soru_kimligi": kayit.get("soru_kimligi"),
                "kaynak_dosya": yol,
                "sira": sira,
            }
        )
    return {
        "soru": len(sorular),
        "n": n,
        "ilk_sirada": ilk_sirada,
        "ilk_n_icinde": ilk_n,
        "bulunamayan": bulunamayan,
        "sirali": sonuclar,
        "sorgu_alani": alan,
        "olcut": "isabet = content_id'nin (path, satir)'i ilk n alintidan birine duser",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--sorular", required=True)
    parser.add_argument("--bin", default="target/release/lubot")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sorgu-alani", choices=("tam", "alinti"), default="tam")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    binary = shlex.split(args.bin)
    if not corpus.is_file():
        raise SystemExit(f"korpus yok: {corpus}")
    if not binary:
        raise SystemExit("--bin bos")
    if "/" in binary[0] and not Path(binary[0]).is_file():
        raise SystemExit(f"ikili yok: {binary[0]} (once cargo build --release -p lubot)")
    sorular = [
        json.loads(satir)
        for satir in Path(args.sorular).read_text(encoding="utf-8").splitlines()
        if satir.strip()
    ]
    if not sorular:
        raise SystemExit("soru seti bos: olcumsuz rapor yazilmaz")

    rapor = olc(binary, corpus, sorular, args.n, args.sorgu_alani)
    metin = json.dumps(rapor, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(metin, encoding="utf-8")
    else:
        sys.stdout.write(metin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
