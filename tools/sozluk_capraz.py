#!/usr/bin/env python3
"""The tokenizer cross-check: the Rust port against the reference, id by id.

Why a corpus and not a handful of examples: the interesting behaviour of this
vocabulary is in the newline runs, the added-token list, the space markers and
the byte fallback, and those interact. A port can pass ten hand-picked strings
and fail on the eleventh. This runs both implementations over a corpus and
compares every id.

The corpus is built from the repository's own text plus a list of the shapes
that were measured to be tricky, and it is *written out* by this script so that
the same bytes can be replayed later without needing Python again.

Usage:
    python3 tools/sozluk_capraz.py --sozluk <tokenizer.json> --ikili <lubot> \
        --metin <dosya> [--turev-koru <adet>]

Exit code 0 only if every id of every text agrees.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys

# The shapes that were measured to be decisive. Each one is here because it
# exercises a mechanism the port had to get right, and the comment says which.
ZOR_ORNEKLER = [
    "Merhaba dünya",            # the ordinary path: two words, one merge each
    "hello world",              # ASCII
    "a",                        # the marker a single word gets at the start
    " a",                       # the same word with a space: identical output
    "",                         # no ids at all, not a marker
    " ",                        # one space is one marker
    "  ",                       # two spaces are two markers
    "   ",                      # and three
    "\n",                       # newline runs are added tokens
    "\n\n",
    "\n\n\n",
    "\n\n\n\n\n\n\n\n\n\n\n\n",  # the longest run the file lists
    "a\nb",                     # a newline inside a word: three pieces
    "a <mask> b",               # an added token with lstrip
    "<mask>",
    "<bos><eos>",               # the wrapping tokens as literal text
    "\u0000",                   # a zero byte: byte fallback
    "x\u0000y",
    "\U0001f680 roket",         # an emoji in the vocabulary
    "\U0001f9d1\u200d\U0001f4bb",  # a ZWJ sequence: three fallback-ish tokens
    "\u00e9",                   # NFC
    "e\u0301",                  # NFD: the combining mark is its own token
    "\u0e01\u0e02",             # Thai
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",  # Japanese
    "\u0130stanbul'da hava \u00e7ok g\u00fczel.",  # Turkish with a dotted capital
    "\u00c7\u011e\u0130\u00d6\u015e\u00dc\u00e7\u011f\u0131\u00f6\u015f\u00fc",
    "12345 67,89",              # digits, which merge one at a time
    "def f(x):\n    return x+1",  # code: indentation is space markers
    "\t",                       # a tab is not a space
    "x\ty",
    "\u2026\u2014\u2019\u201c\u201d",
    "fi" ,                      # adjacent letters that are not a ligature
    "fi\u0301che",
    "  bo\u015fluklu   metin  ",  # a run of spaces in the middle
    "a" * 64,                   # repetition: 64 characters of one letter
    ("ab" * 32),
    "\u00fc" * 33,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sozluk", required=True, help="tokenizer.json")
    ap.add_argument("--ikili", required=True, help="the lubot binary")
    ap.add_argument("--metin", default="", help="extra text to slice into examples")
    ap.add_argument("--parca", type=int, default=120, help="how many slices from --metin")
    ap.add_argument("--tohum", type=int, default=20260926)
    ap.add_argument("--yaz", default="", help="write the corpus here, one JSON string per line")
    a = ap.parse_args()

    ornekler = list(ZOR_ORNEKLER)
    if a.metin:
        with open(a.metin, encoding="utf-8") as f:
            metin = f.read()
        rastgele = random.Random(a.tohum)
        # Slices of the real text, taken at random offsets so the shapes are not
        # all Word-Salad: real text has the punctuation and newline distribution
        # the model was trained on.
        for _ in range(a.parca):
            if len(metin) < 40:
                break
            bas = rastgele.randrange(0, len(metin) - 40)
            uzunluk = rastgele.choice([8, 16, 40, 120, 400])
            ornekler.append(metin[bas : bas + uzunluk])
    # Duplicates removed, order kept: a corpus that repeats itself wastes runs.
    gorulen = set()
    benzersiz = []
    for o in ornekler:
        if o in gorulen:
            continue
        gorulen.add(o)
        benzersiz.append(o)

    if a.yaz:
        import json as _json

        with open(a.yaz, "w", encoding="utf-8") as f:
            for o in benzersiz:
                f.write(_json.dumps(o, ensure_ascii=False) + "\n")
        print(f"- korpus yazildi: {a.yaz} ({len(benzersiz)} ornek)")

    from tokenizers import Tokenizer

    t = Tokenizer.from_file(a.sozluk)
    import tempfile

    gecici = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    yol_gecici = gecici.name
    gecici.close()
    uyusmazlik = 0
    toplam_jeton = 0
    for sira, metin in enumerate(benzersiz):
        beklenen = t.encode(metin).ids
        # Through a file, not an argument: a NUL and a long example cannot be
        # passed on a command line, and those are exactly the cases a fallback
        # would paper over.
        with open(yol_gecici, "w", encoding="utf-8") as f:
            f.write(metin)
        proc = subprocess.run(
            [a.ikili, "sozluk", "jetonla", "--sozluk", a.sozluk, "--metin-dosya", yol_gecici],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(f"KOMUT HATASI [{sira}] {metin!r}: {proc.stderr.strip()[:200]}")
            uyusmazlik += 1
            continue
        satir = [s for s in proc.stdout.splitlines() if s.startswith("- ids:")]
        if not satir:
            print(f"CIKTI YOK [{sira}] {metin!r}: {proc.stdout[:200]!r}")
            uyusmazlik += 1
            continue
        olculen = [int(x) for x in satir[0].split(":", 1)[1].split()]
        toplam_jeton += len(beklenen)
        if olculen != beklenen:
            uyusmazlik += 1
            # The first disagreement is printed in full, and the rest as a count:
            # a port that is wrong is wrong in a pattern, and the pattern is
            # what the message has to show.
            if uyusmazlik <= 5:
                print(f"FARK [{sira}] {metin!r}")
                print(f"   referans: {beklenen[:24]}")
                print(f"   port    : {olculen[:24]}")
                for i, (x, y) in enumerate(zip(beklenen, olculen)):
                    if x != y:
                        print(f"   ilk fark indeks {i}: referans {x}, port {y}")
                        break
    import os

    os.unlink(yol_gecici)
    print(f"- ornek: {len(benzersiz)}, jeton: {toplam_jeton}, uyusmazlik: {uyusmazlik}")
    if uyusmazlik:
        print("- SONUC: FARKLI")
        return 1
    print("- SONUC: ayni (butun idler birebir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
