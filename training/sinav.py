#!/usr/bin/env python3
"""The held-out exam set, and the stamp that keeps it held out.

L/Z/AA want an exam the training run has never seen; PP wants the separation to
be physical rather than intentional. This script is both halves:

* it picks passages from the corpus this repo builds, writes one question per
  passage, and stamps each passage's `content_id` into
  `training/eval/eval-only.json`;
* `make_sft.py` now drops stamped passages, so a stamped passage cannot become a
  training row. Before this, the mechanism only *detected* the leak
  (`eval_sft` refuses the set) - a refused run is not a held-out exam set.

The selection rule is written down because a hand-picked exam measures the
picker: `doc` records only, sorted by `content_id`, at most one per file, taken
evenly across the sorted list. Evenly, because taking the first twelve would
measure whichever file happens to sort first.

What this set measures is narrow, and the record says so: retrieval plus
citation on passages that were removed from training. The question text is
derived from the passage's own first line, which makes retrieval easier than a
paraphrase would, so any score against this set is an **upper bound**, not an
ability estimate. Reasoning is not measured here at all.

Usage:
    python3 training/sinav.py --olc
    python3 training/sinav.py --kur
    python3 training/sinav.py --dogrula
    python3 training/sinav.py --self-test
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
KORPUS = ROOT / "corpus" / "knowledge-self.jsonl.gz"
SET = ROOT / "training" / "eval" / "sinav-seti.jsonl"
DAMGA = ROOT / "training" / "eval" / "eval-only.json"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "sinav-seti-2026-09-23.json"

# Twelve questions: enough for a per-class number to mean something, small
# enough that removing them from training costs little. Not tuned to a result.
HEDEF_SORU = 12


def _eval_sft():
    spec = importlib.util.spec_from_file_location(
        "eval_sft", str(ROOT / "training" / "eval_sft.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def korpus_kayitlari() -> list[dict]:
    if not KORPUS.is_file():
        raise SystemExit(
            f"{KORPUS.relative_to(ROOT)} yok: sinav seti korpusdan kurulur, "
            "korpus kapilardan once kurulur"
        )
    with gzip.open(KORPUS, "rt", encoding="utf-8") as akis:
        return [json.loads(line) for line in akis if line.strip()]


def secim() -> list[dict]:
    """The exam passages, by the written rule."""
    adaylar = [
        k for k in korpus_kayitlari()
        if k.get("kind") == "doc" and k.get("content_id") and k.get("text", "").strip()
    ]
    if not adaylar:
        raise SystemExit("korpusda `doc` turunde kayit yok: sinav seti kurulamaz")
    adaylar.sort(key=lambda k: k["content_id"])
    # At most one question per file, so one document cannot be the whole exam.
    tekil: list[dict] = []
    gorulen_yol: set[str] = set()
    for kayit in adaylar:
        yol = kayit.get("path", "?")
        if yol in gorulen_yol:
            continue
        gorulen_yol.add(yol)
        tekil.append(kayit)
    if len(tekil) <= HEDEF_SORU:
        return tekil
    adim = len(tekil) / HEDEF_SORU
    return [tekil[int(i * adim)] for i in range(HEDEF_SORU)]


def soru_satirlari(kayitlar: list[dict]) -> list[dict]:
    """One question per passage; the expected criterion is a citation."""
    satirlar = []
    for i, kayit in enumerate(kayitlar, start=1):
        ilk_satir = kayit["text"].strip().splitlines()[0].strip().lstrip("#").strip()
        konu = ilk_satir[:120]
        satirlar.append(
            {
                "soru_kimligi": f"sinav-{i:02d}",
                "sinif": kayit.get("kind", "doc"),
                "soru": (
                    "Korpusu oku ve su konuda ne soylendigini kaynagini "
                    f"gostererek aktar: {konu}"
                ),
                "content_id": kayit["content_id"],
                "beklenen": {
                    "tur": "alintili-cevap",
                    "olcut": "cevap bu content_id'yi alintiliyor",
                },
                "kaynak_dosya": kayit.get("path", "?"),
            }
        )
    return satirlar


def damga_yaz(content_idler: list[str]) -> None:
    veri = json.loads(DAMGA.read_text(encoding="utf-8"))
    veri["digests"] = sorted(set(content_idler))
    DAMGA.write_text(
        json.dumps(veri, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def damgalar() -> list[str]:
    veri = json.loads(DAMGA.read_text(encoding="utf-8"))
    return list(veri.get("digests", []))


def set_oku() -> list[dict]:
    if not SET.is_file():
        return []
    return [
        json.loads(line)
        for line in SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def olc() -> dict[str, Any]:
    baslangic = time.monotonic()
    satirlar = set_oku()
    damgali = set(damgalar())
    soru_kimlikleri = {s["content_id"] for s in satirlar}
    return {
        "sure_saniye": round(time.monotonic() - baslangic, 3),
        "soru_sayisi": len(satirlar),
        "siniflar": sorted({s.get("sinif", "?") for s in satirlar}),
        "damga_sayisi": len(damgali),
        "damgasiz_soru": sorted(soru_kimlikleri - damgali),
        # A stamp with no question silently shrinks the training set for
        # nothing, so it is reported rather than ignored.
        "yetim_damga": sorted(damgali - soru_kimlikleri),
    }


def ihlaller(olcum: dict[str, Any]) -> list[str]:
    bulunan: list[str] = []
    if olcum["damgasiz_soru"]:
        bulunan.append(
            f"{len(olcum['damgasiz_soru'])} sinav sorusunun pasaji damgalanmamis: "
            "o soru egitime sizabilir"
        )
    if olcum["yetim_damga"]:
        bulunan.append(
            f"{len(olcum['yetim_damga'])} damga sorusuz duruyor: egitim seti "
            "bosuna kuculuyor"
        )
    return bulunan


def kayit_olustur(olcum: dict[str, Any]) -> dict[str, Any]:
    bulunan = ihlaller(olcum)
    return {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "exam-set-is-held-out",
        "olcut": {
            "ad": "her_sinav_sorusunun_pasaji_damgali_ve_her_damganin_bir_sorusu_var",
            "sonuc": not bulunan,
        },
        "kaynaklar": {
            "sure_saniye": olcum["sure_saniye"],
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": (
            "Sinav seti ile eval-only damga listesi karsilikli okundu: damgasiz "
            "soru ve sorusuz damga arandi; make_sft.py damgali pasajlari "
            "eledigi icin bu pasajlar egitim satirina donusemiyor."
        ),
        "sinav_seti": {
            "dosya": str(SET.relative_to(ROOT)),
            "soru_sayisi": olcum["soru_sayisi"],
            "siniflar": olcum["siniflar"],
            "damga_sayisi": olcum["damga_sayisi"],
            "secim_kurali": (
                "doc kayitlari, content_id'ye gore sirali, dosya basina en fazla "
                f"bir soru, sirali liste uzerinden esit aralikla {HEDEF_SORU} adet"
            ),
        },
        "ihlaller": bulunan,
        "olculmeyen": [
            "hicbir modelin bu setteki skoru (egitilmis kontrol noktasi yok, K6)",
            "alinti dogrulugu orani (set kuruldu, kosulmadi)",
            "akil yurutme: bu set yalniz getirme + alinti olcer",
        ],
        "not": (
            "Soru metni pasajin kendi ilk satirindan turedigi icin getirme, bir "
            "parafraza gore daha kolaydir: bu sete karsi alinacak her skor bir "
            "UST SINIRDIR, yetenek tahmini degil. Set bos degil ama skor yok; "
            "ikisi ayni sey degil."
        ),
    }


def dogrula() -> int:
    if not KAYIT.is_file():
        raise SystemExit(f"{KAYIT.relative_to(ROOT)} yok: sinav seti beyan edilmedi")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    taze = olc()
    hatalar: list[str] = []
    if kayit["sinav_seti"]["soru_sayisi"] != taze["soru_sayisi"]:
        hatalar.append("kayittaki soru sayisi dosyayla eslesmiyor")
    if kayit["sinav_seti"]["damga_sayisi"] != taze["damga_sayisi"]:
        hatalar.append("kayittaki damga sayisi listeyle eslesmiyor")
    hatalar.extend(ihlaller(taze))
    for hata in hatalar:
        print(f"RED: {hata}")
    if hatalar:
        return 1
    print(
        f"sinav seti dogrulandi: {taze['soru_sayisi']} soru, "
        f"{taze['damga_sayisi']} damga, siniflar {taze['siniflar']}, "
        "damgasiz soru 0, yetim damga 0."
    )
    return 0


def self_test() -> int:
    """Canaries: an unstamped question and an orphan stamp must each be refused,
    and the stamp list must really be what make_sft reads."""
    temel = {
        "sure_saniye": 0.0,
        "soru_sayisi": 1,
        "siniflar": ["doc"],
        "damga_sayisi": 1,
        "damgasiz_soru": [],
        "yetim_damga": [],
    }
    damgasiz = dict(temel)
    damgasiz["damgasiz_soru"] = ["a" * 64]
    if not ihlaller(damgasiz):
        raise SystemExit("self-test: damgasiz sinav sorusu kabul edildi")
    yetim = dict(temel)
    yetim["yetim_damga"] = ["b" * 64]
    if not ihlaller(yetim):
        raise SystemExit("self-test: sorusuz damga kabul edildi")
    if ihlaller(temel):
        raise SystemExit("self-test: temiz bir set kendi kuralini ihlal ediyor")

    # The stamp key is `digests` and make_sft must read the same list; an
    # earlier reader looked for a key that does not exist and would have called
    # every question unstamped.
    ev = _eval_sft()
    if not hasattr(ev, "load_eval_only"):
        raise SystemExit("self-test: eval_sft damga okuyucusu yok")
    kaynak = (ROOT / "training" / "make_sft.py").read_text(encoding="utf-8")
    if "load_eval_only" not in kaynak:
        raise SystemExit(
            "self-test: make_sft damga listesini okumuyor; sizinti onlenmiyor"
        )

    if not SET.is_file():
        print("self-test OK (sinav seti yok: dosya kanaryalari atlandi)")
        return 0
    gercek_damga = DAMGA.read_text(encoding="utf-8")
    try:
        # Canary: remove one stamp from a real set; the question is now
        # trainable and the rule must say so.
        veri = json.loads(gercek_damga)
        if veri.get("digests"):
            veri["digests"] = veri["digests"][:-1]
            DAMGA.write_text(
                json.dumps(veri, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
            if not ihlaller(olc()):
                raise SystemExit("self-test: eksik damga gercek sette yakalanmadi")
    finally:
        DAMGA.write_text(gercek_damga, encoding="utf-8")
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
    if args.olc:
        print(json.dumps(olc(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    kayitlar = secim()
    satirlar = soru_satirlari(kayitlar)
    damga_yaz([s["content_id"] for s in satirlar])
    SET.parent.mkdir(parents=True, exist_ok=True)
    with SET.open("w", encoding="utf-8") as akis:
        for satir in satirlar:
            akis.write(json.dumps(satir, ensure_ascii=False, sort_keys=True) + "\n")
    kayit = kayit_olustur(olc())
    KAYIT.write_text(json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"sinav seti kuruldu: {len(satirlar)} soru, {len(damgalar())} damga, "
        f"ihlal {len(kayit['ihlaller'])} -> {SET.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
