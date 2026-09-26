#!/usr/bin/env python3
"""Pakete giden korpusu suzer: `served: false` kayitlar APK'ya girmez.

Neden ayri bir adim var
-----------------------
Depodaki korpus bir **arsivdir**: surec belgeleri (is kuyrugu, tur raporu,
denetim raporu) orada durur ve `served: false` ile damgalanir, boylece cevap
yuzeyine cikmaz. Telefondaki paket ise arsiv degil **hizmet**tir: uygulamanin
icinde tasinan korpus, kullanicinin eline gecen dosyadir. Arsiv kayitlarini
pakete koymak, damgayi anlamsiz kilardi - kayit okuma indeksine girmese bile
metni cihazda dururdu.

Bu yuzden paketlenen kopya suzulur:

    python3 android/korpus_suz.py corpus/knowledge-self.jsonl.gz /tmp/suzulmus.jsonl.gz

Suzgec **damgaya** bakar (`served is false`), dosya adina degil: politika
`training/servis-politikasi.json` icinde yasar ve korpus kurucusu damgayi oradan
vurur. Boylece yeni bir surec belgesi eklendiginde tek yer degisir.

Cikti: JSON ozet (okunan, kalan, atilan, atilan yollar). Cikti dosyasi bos
kalirsa komut **duser**: sessizce bos bir korpus paketlemek, telefondaki
uygulamayi cevapsiz birakmak olurdu.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def suz(girdi: Path, cikti: Path) -> dict:
    """`served: false` kayitlari cikarir; kalanlari yeni dosyaya yazar."""
    if not girdi.is_file():
        raise SystemExit(f"korpus yok: {girdi}")
    gecici = cikti.with_suffix(cikti.suffix + ".gecici")
    okunan = kalan = 0
    atilan_yollar: dict[str, int] = {}
    with gzip.open(girdi, "rt", encoding="utf-8") as kaynak:
        with gzip.open(gecici, "wt", encoding="utf-8", compresslevel=9) as hedef:
            for satir in kaynak:
                if not satir.strip():
                    continue
                okunan += 1
                kayit = json.loads(satir)
                if kayit.get("served") is False:
                    yol = str(kayit.get("path", "?"))
                    atilan_yollar[yol] = atilan_yollar.get(yol, 0) + 1
                    continue
                hedef.write(satir)
                kalan += 1
    if kalan == 0:
        gecici.unlink(missing_ok=True)
        raise SystemExit("suzme sonrasi korpus bos: paketlenmez")
    gecici.replace(cikti)
    return {
        "okunan": okunan,
        "kalan": kalan,
        "atilan": okunan - kalan,
        "atilan_yollar": dict(sorted(atilan_yollar.items())),
        "cikti": str(cikti),
    }


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("girdi", type=Path, help="depodaki korpus (jsonl.gz)")
    ayristirici.add_argument("cikti", type=Path, help="pakete girecek korpus (jsonl.gz)")
    args = ayristirici.parse_args(argv)
    print(json.dumps(suz(args.girdi, args.cikti), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
