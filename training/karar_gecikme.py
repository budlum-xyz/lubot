#!/usr/bin/env python3
"""The decision path's cost, measured as a baseline rather than a promise.

U adds speed and unit cost as a second axis next to "small"; jev-trader's shape
is the same idea - a decision-only path is worth what its latency says it is.
This measures the one decision path that exists today: `lubot karar tek`, which
parses a vote, applies the threshold and renders. No weights are involved, so
this is the cost of the *machinery*, not of a trained head.

The number includes process startup, because the measurement runs the binary the
way a caller would. Reporting it as "decision latency" without that would be a
smaller-looking number than the one a caller sees, so the record carries the
caveat as a field the gate requires rather than as prose a reader may skip.

It is a baseline, not a ratchet: wall time depends on the machine, so a number
measured here would "regress" on any slower CI runner. The gate checks that the
baseline exists, is mechanical, and carries its caveat - not that the number
holds.

Usage:
    python3 training/karar_gecikme.py --olc
    python3 training/karar_gecikme.py --kur
    python3 training/karar_gecikme.py --dogrula
    python3 training/karar_gecikme.py --self-test
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
IKILI = ROOT / "target" / "debug" / "lubot"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "karar-gecikme-2026-09-23.json"
KOSU_SAYISI = 50

# Startup is part of what a caller pays, and it is most of what is measured
# here. The field exists so the number cannot be read as the head's own cost.
UYARI = (
    "Olculen sure surec baslatmayi (process spawn) icerir: bu, karar basinin "
    "kendi maliyeti degil bir cagiranin odeyecegi ust sinirdir. Egitilmis bir "
    "baslik kosuldugunda ayni olcum yeniden alinacak ve ikisi ayri satirlarda "
    "duracak."
)


def ikiliyi_bul() -> Path:
    if IKILI.is_file():
        return IKILI
    surum = ROOT / "target" / "release" / "lubot"
    if surum.is_file():
        return surum
    raise SystemExit(
        "lubot ikilisi yok: `cargo build` kosulmadan gecikme olculemez "
        "(tahmin edilmis bir sure olcum degildir)"
    )


def olc() -> dict[str, Any]:
    """Run the decision path KOSU_SAYISI times and summarise."""
    baslangic = time.monotonic()
    ikili = ikiliyi_bul()
    sureler: list[float] = []
    basarisiz = 0
    for _ in range(KOSU_SAYISI):
        t0 = time.perf_counter()
        kosu = subprocess.run(
            [str(ikili), "karar", "tek", "evet:0.9"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        sureler.append((time.perf_counter() - t0) * 1000.0)
        if kosu.returncode != 0:
            basarisiz += 1
    if basarisiz:
        raise SystemExit(
            f"{basarisiz}/{KOSU_SAYISI} karar kosusu basarisiz: gecikme olcumu "
            "calismayan bir yolun suresini soyleyemez"
        )
    return {
        "sure_saniye": round(time.monotonic() - baslangic, 3),
        "kosu_sayisi": len(sureler),
        "medyan_ms": round(statistics.median(sureler), 3),
        "en_dusuk_ms": round(min(sureler), 3),
        "en_yuksek_ms": round(max(sureler), 3),
        "standart_sapma_ms": round(statistics.pstdev(sureler), 3),
        "ikili": str(ikili.relative_to(ROOT)),
        "olculen_yol": "lubot karar tek evet:0.9",
        "uyari": UYARI,
    }


def kayit_olustur(olcum: dict[str, Any]) -> dict[str, Any]:
    return {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "decision-latency-is-recorded",
        "olcut": {
            "ad": (
                f"karar_yolu_{KOSU_SAYISI}_kez_kosuldu_ve_medyan_sure_kaydedildi"
            ),
            "sonuc": True,
        },
        "kaynaklar": {
            "sure_saniye": olcum["sure_saniye"],
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": (
            f"`{olcum['olculen_yol']}` {olcum['kosu_sayisi']} kez kosuldu; "
            f"medyan {olcum['medyan_ms']} ms, en dusuk {olcum['en_dusuk_ms']} ms, "
            f"en yuksek {olcum['en_yuksek_ms']} ms."
        ),
        "gecikme": {
            "olculen_yol": olcum["olculen_yol"],
            "kosu_sayisi": olcum["kosu_sayisi"],
            "medyan_ms": olcum["medyan_ms"],
            "en_dusuk_ms": olcum["en_dusuk_ms"],
            "en_yuksek_ms": olcum["en_yuksek_ms"],
            "standart_sapma_ms": olcum["standart_sapma_ms"],
            "ikili": olcum["ikili"],
        },
        "uyari": olcum["uyari"],
        "olculmeyen": [
            "egitilmis bir basinin karar maliyeti (agirlik yok, K6)",
            "uretim yolunun gecikmesi (uretim yuzeyi yok)",
            "karar basina enerji (olcum duzenegi yok)",
        ],
        "not": (
            "Bu bir TABANDIR ve ratchet'e konmadi: duvar saati makineye baglidir, "
            "burada olculen sayi daha yavas bir CI makinesinde 'gerileme' gibi "
            "gorunurdu. Eksen beyan ediliyor, sayi esik yapilmiyor."
        ),
    }


def dogrula() -> int:
    if not KAYIT.is_file():
        raise SystemExit(f"{KAYIT.relative_to(ROOT)} yok: gecikme tabani kaydedilmedi")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    hatalar: list[str] = []
    gecikme = kayit.get("gecikme", {})
    if gecikme.get("kosu_sayisi", 0) < 30:
        hatalar.append(
            f"{gecikme.get('kosu_sayisi')} kosu: tek kosuluk bir sure taban degildir"
        )
    if not kayit.get("uyari", "").strip():
        hatalar.append(
            "uyari alani bos: surec baslatmayi icermeyen bir 'karar gecikmesi' "
            "okunur, oysa olculen ust sinirdir"
        )
    for alan in ("medyan_ms", "en_dusuk_ms", "en_yuksek_ms"):
        deger = gecikme.get(alan)
        if not isinstance(deger, (int, float)) or deger <= 0:
            hatalar.append(f"{alan} olculmus bir sayi degil")
    if (
        isinstance(gecikme.get("en_dusuk_ms"), (int, float))
        and isinstance(gecikme.get("en_yuksek_ms"), (int, float))
        and gecikme["en_dusuk_ms"] > gecikme["en_yuksek_ms"]
    ):
        hatalar.append("en dusuk sure en yuksek sureden buyuk: olcum tutarsiz")
    for hata in hatalar:
        print(f"RED: {hata}")
    if hatalar:
        return 1
    print(
        f"gecikme tabani dogrulandi: {gecikme['kosu_sayisi']} kosu, medyan "
        f"{gecikme['medyan_ms']} ms (surec baslatma dahil, ust sinir)."
    )
    return 0


def self_test() -> int:
    """Canaries: a one-run 'baseline', a missing caveat and an inconsistent
    interval must each be refused."""
    temel = {
        "gecikme": {
            "kosu_sayisi": KOSU_SAYISI,
            "medyan_ms": 1.0,
            "en_dusuk_ms": 0.5,
            "en_yuksek_ms": 2.0,
        },
        "uyari": UYARI,
    }
    kayit_var = KAYIT.is_file()
    eski = KAYIT.read_text(encoding="utf-8") if kayit_var else None
    try:
        def yaz(veri: dict) -> None:
            KAYIT.write_text(
                json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # Canary 1: a single run sold as a baseline.
        tek = json.loads(json.dumps(temel))
        tek["gecikme"]["kosu_sayisi"] = 1
        yaz(tek)
        if dogrula() != 1:
            raise SystemExit("self-test: tek kosuluk sure taban sayildi")

        # Canary 2: no caveat, so the number reads as the head's own cost.
        uyarisiz = json.loads(json.dumps(temel))
        uyarisiz["uyari"] = ""
        yaz(uyarisiz)
        if dogrula() != 1:
            raise SystemExit("self-test: uyarisiz gecikme kabul edildi")

        # Canary 3: an interval that contradicts itself.
        celiskili = json.loads(json.dumps(temel))
        celiskili["gecikme"]["en_dusuk_ms"] = 9.0
        celiskili["gecikme"]["en_yuksek_ms"] = 3.0
        yaz(celiskili)
        if dogrula() != 1:
            raise SystemExit("self-test: celiskili aralik kabul edildi")

        # The declared shape itself must pass, or the canaries above would be
        # passing on a rule that refuses everything.
        yaz(temel)
        if dogrula() != 0:
            raise SystemExit("self-test: beyan edilen taban kendi kuralini ihlal ediyor")
    finally:
        if eski is not None:
            KAYIT.write_text(eski, encoding="utf-8")
        elif KAYIT.is_file():
            KAYIT.unlink()
    print("self-test OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--olc", action="store_true")
    grup.add_argument("--kur", action="store_true")
    grup.add_argument("--dogrula", action="store_true")
    grup.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.dogrula:
        return dogrula()
    olcum = olc()
    if args.olc:
        print(json.dumps(olcum, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(
        json.dumps(kayit_olustur(olcum), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"gecikme tabani yazildi: {olcum['kosu_sayisi']} kosu, medyan "
        f"{olcum['medyan_ms']} ms -> {KAYIT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
