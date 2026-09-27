#!/usr/bin/env python3
"""The comparison class, declared with numbers instead of adjectives.

GG's calibration: the honest opponent is not "a small model" in the abstract but
a *parameter-matched* one, and today's "small" open models start around 135M-600M
parameters - far above this repo's 924.288. So two things are fixed here and
re-measured every commit rather than argued once:

* the class boundary. The parameter count is read from `model_spec.json`, never
  restated, and must stay below the smallest named peer. The day it does not, the
  declaration is stale and this script refuses rather than quietly comparing
  upwards;
* the axis a "match" may be claimed on. Parameter-matched comparison measures raw
  language ability; the claim of a match belongs to the *task* axis - citation
  accuracy and refusal discipline on Budlum-domain questions, which does not
  depend on size. A record claiming a match on the parameter axis is refused.

The head-to-head itself is not measured here: there is no trained checkpoint
(K6 leaves the compute to the owner's hardware) and no peer is run on this
machine. The exam set is counted, and it is empty, and the record says so -
an empty exam set reported as a baseline would be an unmeasured number dressed
as a measurement.

Usage:
    python3 training/kiyas_sinifi.py --olc
    python3 training/kiyas_sinifi.py --kur
    python3 training/kiyas_sinifi.py --dogrula
    python3 training/kiyas_sinifi.py --self-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "training" / "model_spec.json"
KORPUS = ROOT / "corpus" / "knowledge-self.jsonl.gz"
BUTCE = ROOT / "training" / "eval" / "sonuclar" / "egitim-butcesi-2026-09-23.json"
SINAV = ROOT / "training" / "eval" / "sinav-seti.jsonl"
DAMGA = ROOT / "training" / "eval" / "eval-only.json"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "kiyas-sinifi-2026-09-23.json"

# The named peers are the class boundary, not a target. Both are named so the
# boundary is checkable; neither is run here and neither's weights are used.
EN_KUCUK_AKRAN = {"ad": "SmolLM2-135M", "parametre": 135_000_000}
IKINCI_AKRAN = {"ad": "Qwen3-0.6B", "parametre": 600_000_000}

EKSEN_GOREV = "gorev-eslenegi"
EKSEN_PARAM = "parametre-eslenegi"


def parametre_sayisi() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    toplam = spec["params"]["toplam"]
    if not isinstance(toplam, int) or toplam <= 0:
        raise SystemExit(f"{SPEC.name}: params.toplam bir sayi degil")
    return toplam


def butce_olc() -> dict[str, Any]:
    """The token budget, measured by the script that owns the tokenizer."""
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "egitim_butcesi.py"), "--olc"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(f"egitim butcesi olculemedi: {kosu.stderr[-200:]}")
    return json.loads(kosu.stdout)


def sinav_satirlari() -> list[dict]:
    if not SINAV.is_file():
        return []
    return [
        json.loads(line)
        for line in SINAV.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _eval_sft():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_sft", str(ROOT / "training" / "eval_sft.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def damgali_kimlikler() -> set[str]:
    """The stamped passages, read by the reader that owns the contract.

    `eval_sft.load_eval_only` fails closed on a malformed list and reads the
    `digests` key. An earlier version of this script read a key named
    `damgalar`, which does not exist, so the leak check always saw an empty set
    and would have called every exam question unstamped the day the first one
    was written. One reader, one key.
    """
    return _eval_sft().load_eval_only(DAMGA)


def olc() -> dict[str, Any]:
    """Measure the class: parameter count, token budget, exam set, leak check."""
    baslangic = time.monotonic()
    param = parametre_sayisi()
    butce = butce_olc()
    jeton = butce["benzersiz_jeton"]
    satirlar = sinav_satirlari()
    damgalar = damgali_kimlikler()
    damgasiz = [
        s.get("soru_kimligi", "?")
        for s in satirlar
        if s.get("content_id") and s["content_id"] not in damgalar
    ]
    return {
        "sure_saniye": round(time.monotonic() - baslangic, 3),
        "parametre": param,
        "en_kucuk_akran": EN_KUCUK_AKRAN,
        "ikinci_akran": IKINCI_AKRAN,
        "benzersiz_jeton": jeton,
        "korpus_kayit_sayisi": butce["korpus_kayit_sayisi"],
        "jeton_basina_parametre": round(jeton / param, 6) if param else 0.0,
        "parametre_basina_jeton": round(param / jeton, 4) if jeton else 0.0,
        "sinav_soru_sayisi": len(satirlar),
        "sinav_siniflari": sorted({s.get("sinif", "?") for s in satirlar}),
        "damgasiz_soru": damgasiz,
    }


def ihlaller(olcum: dict[str, Any]) -> list[str]:
    """What would make this declaration false."""
    bulunan: list[str] = []
    if olcum["parametre"] >= olcum["en_kucuk_akran"]["parametre"]:
        bulunan.append(
            f"parametre {olcum['parametre']} artik en kucuk adlandirilmis akrana "
            f"({olcum['en_kucuk_akran']['ad']}, "
            f"{olcum['en_kucuk_akran']['parametre']}) esit ya da buyuk: sinif "
            "beyani bayat, yeniden beyan edilmeli"
        )
    if olcum["damgasiz_soru"]:
        bulunan.append(
            f"{len(olcum['damgasiz_soru'])} sinav sorusunun dayandigi pasaj "
            "eval-only listesine damgalanmamis: sizinti mekanizmasi devre disi"
        )
    return bulunan


def kayit_olustur(olcum: dict[str, Any]) -> dict[str, Any]:
    bulunan = ihlaller(olcum)
    return {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "comparison-class-is-declared",
        "olcut": {
            "ad": (
                "parametre_sayisi_adlandirilmis_en_kucuk_akranin_altinda_ve_"
                "her_sinav_sorusunun_pasaji_damgali"
            ),
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
            "Parametre model_spec.json'dan, jeton egitim_butcesi.py --olc'den "
            "okundu (ikinci literal yok); sinav seti sayildi ve her sorunun "
            "content_id'si eval-only.json damgalariyla karsilastirildi."
        ),
        "sinif": {
            "parametre": olcum["parametre"],
            "benzersiz_jeton": olcum["benzersiz_jeton"],
            "korpus_kayit_sayisi": olcum["korpus_kayit_sayisi"],
            "jeton_basina_parametre": olcum["jeton_basina_parametre"],
            "parametre_basina_jeton": olcum["parametre_basina_jeton"],
            "sinir": [olcum["en_kucuk_akran"], olcum["ikinci_akran"]],
        },
        "eksen_kurali": {
            "kapisma_iddiasi_yalniz": EKSEN_GOREV,
            "reddedilen": EKSEN_PARAM,
            "gerekce": (
                "Parametre-eslenegi katmani ham dil yeterliligini olcer; bu "
                "deponun 924.288 parametresi 135M-600M sinifinin ALTINDA. "
                "Kapisma iddiasi boyuttan bagimsiz olan gorev ekseninde "
                "(alinti dogrulugu + red disiplini) tanimlidir; kucuk olmak "
                "dezavantaj degil cunku rakip de kucuk."
            ),
        },
        "sinav_seti": {
            "dosya": str(SINAV.relative_to(ROOT)),
            "soru_sayisi": olcum["sinav_soru_sayisi"],
            "siniflar": olcum["sinav_siniflari"],
            "damgasiz_soru": olcum["damgasiz_soru"],
        },
        "ihlaller": bulunan,
        "olculmeyen": [
            "hicbir karsilastirmali sonuc: egitilmis kontrol noktasi yok (K6)",
            "adlandirilan akranlarin bu makinedeki performansi (kosulmadi)",
            "gorev ekseninde alinti dogrulugu ve red orani (sinav seti bos)",
        ],
        "not": (
            "Bu bir TABAN BEYANIDIR, zafer ilani degil. Sinav seti bos; ilk "
            "karsilastirma olculdugunde bu kayit onun sinifini soyleyecek. "
            "GG'nin istedigi gibi beyan tek seferlik degil: korpus her "
            "buyudugunde kapi yeniden olcer."
        ),
    }


def dogrula() -> int:
    if not KAYIT.is_file():
        raise SystemExit(f"{KAYIT.relative_to(ROOT)} yok: sinif beyan edilmedi")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    taze = olc()
    hatalar: list[str] = []
    if kayit["sinif"]["parametre"] != taze["parametre"]:
        hatalar.append("kayittaki parametre sayisi spec ile eslesmiyor")
    if kayit["sinif"]["benzersiz_jeton"] != taze["benzersiz_jeton"]:
        hatalar.append("kayittaki jeton sayisi butce olcumuyle eslesmiyor")
    if kayit["sinav_seti"]["soru_sayisi"] != taze["sinav_soru_sayisi"]:
        hatalar.append("kayittaki sinav sayisi dosyayla eslesmiyor")
    for hata in ihlaller(taze):
        hatalar.append(hata)
    # A match claim on the wrong axis is the failure this record exists to
    # prevent, so it is checked wherever it might be written.
    for dosya in sorted(KAYIT.parent.glob("*.json")):
        veri = json.loads(dosya.read_text(encoding="utf-8"))
        iddia = veri.get("kapisma_iddiasi")
        if isinstance(iddia, dict) and iddia.get("eksen") == EKSEN_PARAM:
            hatalar.append(
                f"{dosya.name}: kapisma iddiasi parametre ekseninde; iddia "
                f"yalniz {EKSEN_GOREV} ekseninde yapilabilir"
            )
    for hata in hatalar:
        print(f"RED: {hata}")
    if hatalar:
        return 1
    print(
        f"sinif beyani dogrulandi: {taze['parametre']} parametre "
        f"(< {taze['en_kucuk_akran']['ad']} {taze['en_kucuk_akran']['parametre']}), "
        f"{taze['benzersiz_jeton']} jeton, jeton/param "
        f"{taze['jeton_basina_parametre']}, sinav {taze['sinav_soru_sayisi']} soru."
    )
    return 0


def _sentetik_olcum() -> dict[str, Any]:
    """A measurement with the shape of a real one, for canaries that must run
    where no corpus exists (CI runs the gate self-tests before the corpus is
    built, so a canary that had to tokenize could never run there)."""
    return {
        "sure_saniye": 0.0,
        "parametre": 924_288,
        "en_kucuk_akran": EN_KUCUK_AKRAN,
        "ikinci_akran": IKINCI_AKRAN,
        "benzersiz_jeton": 1,
        "korpus_kayit_sayisi": 1,
        "jeton_basina_parametre": 0.0,
        "parametre_basina_jeton": 0.0,
        "sinav_soru_sayisi": 0,
        "sinav_siniflari": [],
        "damgasiz_soru": [],
    }


def self_test() -> int:
    """Canaries: a param count over the boundary, an unstamped exam question and
    a match claim on the parameter axis must each be refused.

    The first three need no corpus, so they run in CI's self-test step. The
    record round-trip does, and is skipped loudly rather than silently.
    """
    temel = _sentetik_olcum()
    # Canary 1: a param count at or over the boundary makes the class stale.
    buyuk = dict(temel)
    buyuk["parametre"] = EN_KUCUK_AKRAN["parametre"]
    if not ihlaller(buyuk):
        raise SystemExit("self-test: siniri asan parametre kabul edildi")
    # Canary 2: an unstamped exam question is a leak, whatever the sizes.
    damgasiz = dict(temel)
    damgasiz["damgasiz_soru"] = ["kanarya-soru"]
    if not ihlaller(damgasiz):
        raise SystemExit("self-test: damgasiz sinav sorusu kabul edildi")
    # Canary 3: the declared shape itself must be clean, or the canaries above
    # would be passing on a rule that refuses everything.
    if ihlaller(temel):
        raise SystemExit("self-test: beyan edilen sinif kendi kuralini ihlal ediyor")

    if not KORPUS.is_file():
        # CI runs the gate self-tests before the corpus exists. Skipping loudly
        # is the honest move; the record round-trip runs in the gate itself,
        # after the corpus is built.
        print("self-test OK (korpus yok: kayit kanaryalari atlandi)")
        return 0

    kayit_var = KAYIT.is_file()
    sinav_var = SINAV.is_file()
    damga_var = DAMGA.is_file()
    eski_kayit = KAYIT.read_text(encoding="utf-8") if kayit_var else None
    eski_sinav = SINAV.read_text(encoding="utf-8") if sinav_var else None
    eski_damga = DAMGA.read_text(encoding="utf-8") if damga_var else None
    try:
        olcum = olc()
        kayit = kayit_olustur(olcum)
        KAYIT.write_text(json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8")
        if dogrula() != 0:
            raise SystemExit("self-test: taze olcum kendi beyanini dogrulamadi")

        # Canary 4: a real unstamped question in the real file.
        SINAV.write_text(
            json.dumps(
                {
                    "soru_kimligi": "kanarya-1",
                    "sinif": "okuma",
                    "content_id": "damgalanmamis-bir-ozet",
                    "soru": "Kanarya sorusu.",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if not ihlaller(olc()):
            raise SystemExit("self-test: gercek dosyadaki damgasiz soru kabul edildi")
        if eski_sinav is None:
            SINAV.unlink(missing_ok=True)
        else:
            SINAV.write_text(eski_sinav, encoding="utf-8")

        # Canary 5: a match claim on the parameter axis.
        iddia = {
            "kosucu": "betik",
            "tarih": "2026-09-23",
            "is": "kanarya",
            "olcut": {"ad": "kanarya_olcutu", "sonuc": True},
            "kaynaklar": {
                "sure_saniye": 0.0, "girdi_jetonlari": 0,
                "onbellekli_jetonlari": 0, "cikti_jetonlari": 0, "maliyet": 0.0,
            },
            "kapisma_iddiasi": {"eksen": EKSEN_PARAM, "rakip": "kanarya"},
        }
        iddia_dosya = KAYIT.parent / "kanarya-iddia.json"
        iddia_dosya.write_text(
            json.dumps(iddia, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            if dogrula() != 1:
                raise SystemExit(
                    "self-test: parametre ekseninde kapisma iddiasi kabul edildi"
                )
        finally:
            iddia_dosya.unlink(missing_ok=True)
    finally:
        if eski_kayit is not None:
            KAYIT.write_text(eski_kayit, encoding="utf-8")
        elif KAYIT.is_file():
            KAYIT.unlink()
        if eski_sinav is not None:
            SINAV.write_text(eski_sinav, encoding="utf-8")
        elif SINAV.is_file():
            SINAV.unlink()
        if eski_damga is not None:
            DAMGA.write_text(eski_damga, encoding="utf-8")
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
    kayit = kayit_olustur(olcum)
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"sinif beyani yazildi: {olcum['parametre']} parametre, "
        f"{olcum['benzersiz_jeton']} jeton, sinav {olcum['sinav_soru_sayisi']} soru, "
        f"ihlal {len(kayit['ihlaller'])} -> {KAYIT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
