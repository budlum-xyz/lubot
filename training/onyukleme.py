#!/usr/bin/env python3
"""Bootstrap rounds: keep what passed, label what did not, measure the delta.

G describes a self-distillation loop: run a round, feed the outputs that clear
the schema and citation gates into the next round as new curriculum rows, and
put the failures into a negative pool *labelled with why they were refused*.
Two things follow, and both are measured here rather than assumed:

* a round that cannot say why it refused a row teaches nothing about the
  refusal, so every negative row carries a reason, and `--dogrula` refuses a
  pool with one unlabelled entry;
* a round that only solves what the previous round already solved is not
  progress. The competence delta is recomputed from the two records; with no
  previous round there is no delta, and "no delta" is reported rather than 0.

The checks themselves are not reimplemented. Each row is handed to
`eval_sft.evaluate`, which is the same mechanical checker the SFT gate uses, so
a threshold change there changes this round too - one rule, one answer.

Round 1's participant is the deterministic reader, not a trained checkpoint: no
training run has happened (K6 leaves the compute to the owner's hardware), so
this round is a baseline. The record says so in `katilimci` and lists what a
trained round would add under `olculmeyen`. Presenting it as a model round
would be an unmeasured number dressed as a measurement.

Usage:
    python3 training/onyukleme.py --olc
    python3 training/onyukleme.py --kur
    python3 training/onyukleme.py --dogrula
    python3 training/onyukleme.py --self-test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MUFREDAT = ROOT / "training" / "curriculum"
HAVUZ = ROOT / "training" / "eval" / "negatif-havuz.jsonl"
SONUCLAR = ROOT / "training" / "eval" / "sonuclar"
KAYIT = SONUCLAR / "onyukleme-2026-09-23.json"
TUR = 1


def _modul(ad: str, dosya: str):
    spec = importlib.util.spec_from_file_location(ad, str(ROOT / "training" / dosya))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sinif_satirlari() -> dict[str, list[dict]]:
    """One question class per curriculum file; the class is the file name.

    Each row is stamped `kind = "curriculum"` on the way in, exactly as
    `make_sft.py` stamps it (line 84) and `veri_karisimi.py` stamps `mufredat`
    (line 90): the stratum is established by the loader, not stored in the file.
    Without the stamp `eval_sft.evaluate` reads these as *grounded* rows and
    applies the Source-citation rule to them, which measures a category error
    and reports it as a finding - 78 of 88 rows "uncited" on the first run was
    that mistake, not a property of the curriculum.
    """
    if not MUFREDAT.is_dir():
        raise SystemExit(f"{MUFREDAT.relative_to(ROOT)} yok: siniflar sayilamaz")
    siniflar: dict[str, list[dict]] = {}
    for dosya in sorted(MUFREDAT.glob("*.jsonl")):
        satirlar = [
            json.loads(line)
            for line in dosya.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not satirlar:
            raise SystemExit(f"{dosya.name}: bos sinif, olculecek bir sey yok")
        for satir in satirlar:
            # The file's own marker is kept for reporting, then the stratum is
            # set from where the row came from. Two rows in format.jsonl carry
            # `kind: "negative"`: they are red-team fixtures whose *user* turn
            # holds invalid markdown and whose answer correctly refuses it, so
            # they are curriculum rows and must not be measured against the
            # grounded citation rule.
            satir["alt_tur"] = satir.get("kind")
            satir["kind"] = "curriculum"
        siniflar[dosya.stem] = satirlar
    if not siniflar:
        raise SystemExit("hic curriculum sinifi yok")
    return siniflar


def nedenler(satir: dict, ev) -> list[str]:
    """Why this row did not pass. Empty means it passed.

    The rules belong to `eval_sft`; this only names them. A finding this
    function does not recognise is reported as unrecognised rather than
    silently dropped, because a dropped reason is a row that looks clean.
    """
    rapor = ev.evaluate([satir])
    bulunan: list[str] = []
    for bulgu in rapor["findings"]:
        if "Source citation" in bulgu:
            bulunan.append("alintisiz")
        elif "characters" in bulgu:
            bulunan.append("govde-taban-altinda")
        elif "eval-only" in bulgu:
            bulunan.append("eval-only-sizintisi")
        elif "duplicate" in bulgu:
            bulunan.append("yinelenen")
        else:
            bulunan.append(f"taninmayan:{bulgu[:48]}")
    return bulunan


def tur_olc() -> dict[str, Any]:
    """Run one round over every class and split the rows."""
    baslangic = time.monotonic()
    ev = _modul("eval_sft", "eval_sft.py")
    siniflar: dict[str, Any] = {}
    negatif: list[dict] = []
    gorulen: dict[str, str] = {}
    toplam = 0
    for sinif, satirlar in sorted(sinif_satirlari().items()):
        gecen = 0
        for i, satir in enumerate(satirlar):
            toplam += 1
            sebepler = nedenler(satir, ev)
            ozet = ev.digest_of(satir)
            kimlik = f"{sinif}:{i}"
            if ozet in gorulen:
                # A duplicate teaches repetition, not knowledge; the reason
                # names the row it duplicates so the pool is actionable.
                sebepler.append(f"yinelenen:{gorulen[ozet]}")
            else:
                gorulen[ozet] = kimlik
            if sebepler:
                negatif.append(
                    {
                        "tur": TUR,
                        "sinif": sinif,
                        "kimlik": kimlik,
                        "ozet": ozet,
                        "neden": sebepler,
                    }
                )
            else:
                gecen += 1
        siniflar[sinif] = {"toplam": len(satirlar), "gecen": gecen}
    sure = round(time.monotonic() - baslangic, 3)
    neden_sayilari: dict[str, int] = {}
    for satir in negatif:
        for neden in satir["neden"]:
            anahtar = neden.split(":", 1)[0]
            neden_sayilari[anahtar] = neden_sayilari.get(anahtar, 0) + 1
    alt_turler: dict[str, int] = {}
    for satirlar in sinif_satirlari().values():
        for satir in satirlar:
            ad = satir.get("alt_tur") or "duz"
            alt_turler[ad] = alt_turler.get(ad, 0) + 1
    return {
        "alt_turler": alt_turler,
        "tur": TUR,
        "sure_saniye": sure,
        "toplam_satir": toplam,
        "siniflar": siniflar,
        "gecen_toplam": sum(v["gecen"] for v in siniflar.values()),
        "negatif": negatif,
        "neden_sayilari": neden_sayilari,
    }


def yeterlilik_farki(onceki: dict | None, simdiki: dict) -> dict:
    """Did this round solve what the previous one could not?

    Recomputed from the two records, never read from a claim. With no previous
    round the answer is "not comparable", which is the measurement - a zero
    would say the round improved nothing, and nothing was measured.
    """
    if onceki is None:
        return {
            "karsilastirilabilir": False,
            "neden": "onceki tur kaydi yok: bu tur bir tabandir, fark olculemez",
            "yeni_cozulen_siniflar": [],
            "gerileyen_siniflar": [],
            "yalniz_tekrar": None,
        }
    oncekiler = onceki.get("siniflar", {})
    yeni: list[str] = []
    geri: list[str] = []
    for sinif, veri in simdiki["siniflar"].items():
        eski = oncekiler.get(sinif, {}).get("gecen", 0)
        if veri["gecen"] > eski:
            yeni.append(sinif)
        elif veri["gecen"] < eski:
            geri.append(sinif)
    return {
        "karsilastirilabilir": True,
        "onceki_tur": onceki.get("tur"),
        "yeni_cozulen_siniflar": sorted(yeni),
        "gerileyen_siniflar": sorted(geri),
        # A round that improves no class is repeating examples, and that is a
        # finding about the round, not a neutral outcome.
        "yalniz_tekrar": not yeni,
    }


def onceki_kayit() -> dict | None:
    """The newest recorded round before this one, if any."""
    oncekiler = []
    for dosya in sorted(SONUCLAR.glob("onyukleme-*.json")):
        if dosya == KAYIT:
            continue
        veri = json.loads(dosya.read_text(encoding="utf-8"))
        if isinstance(veri.get("tur"), int):
            oncekiler.append(veri)
    return oncekiler[-1] if oncekiler else None


def kayit_olustur(olcum: dict) -> dict:
    """The record the gate checks: one criterion, full accounting."""
    return {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "bootstrap-round-is-measured",
        "tur": TUR,
        "katilimci": (
            "deterministik-okuyucu (egitilmis kontrol noktasi YOK; K6'ya gore "
            "hesap sahibin donaniminda ve henuz kosulmadi)"
        ),
        "olcut": {
            "ad": (
                "negatif_havuzun_her_satiri_bir_neden_tasiyor_ve_yeterlilik_farki_"
                "kayitlardan_yeniden_hesaplaniyor"
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
            "Her curriculum satiri make_sft.py'nin vurdugu ayni "
            "kind=curriculum damgasiyla eval_sft.evaluate'e verildi (ayni "
            "mekanik denetleyici, ikinci kural yok); gecenler sayildi, "
            "gecmeyenler nedeniyle havuza yazildi, yeterlilik farki iki kayit "
            "yeniden okunarak hesaplandi."
        ),
        "siniflar": olcum["siniflar"],
        "gecen_toplam": olcum["gecen_toplam"],
        "toplam_satir": olcum["toplam_satir"],
        "negatif_havuz": {
            "dosya": str(HAVUZ.relative_to(ROOT)),
            "toplam": len(olcum["negatif"]),
            "nedenlere_gore": olcum["neden_sayilari"],
        },
        "alt_turler": olcum["alt_turler"],
        "yeterlilik_farki": yeterlilik_farki(onceki_kayit(), olcum),
        "bulgu_mufredat_isareti": {
            "olculen": (
                "training/curriculum/format.jsonl icinde 2 satir `kind: "
                "\"negative\"` ve bir `rule` alani tasiyor; make_sft.py:84 "
                "ayni dosyalari okurken `row[\"kind\"] = \"curriculum\"` ile "
                "damgayi KOSULSUZ eziyor."
            ),
            "hukum": (
                "BULGUDUR, veri sizintisi DEGIL. Iki satirin gecersiz Markdown'i "
                "user turunda; asistan cevabi dogru bir red ('Invalid: ... "
                "fails the schema; the answer is rejected and regenerated'). "
                "Yani egitime giren satirlar gecerli. Kaybolan sey isaretin "
                "kendisi:asagidaki bir tuketici kirmizi-senaryo fiksturlerini duz "
                "mufredat satirlarindan ayiramaz."
            ),
            "yapilmayan": (
                "make_sft.py degistirilmedi, format.jsonl degistirilmedi. Hangi "
                "isaretin korunacagi bir operator karari; kayit onu tasir."
            ),
        },
        "olculmeyen": [
            "egitilmis bir kontrol noktasinin cevap kalitesi (egitim kosusu gerektirir)",
            "iki kontrol noktasi arasindaki yeterlilik farki (bu tur ilk tur)",
            "best-of-N seciminin maliyeti (aday uretimi yok)",
        ],
        "not": (
            "Bu tur bir TABANDIR: katilimci egitilmis bir model degil, mevcut "
            "deterministik okuyucudur. Ilk egitilmis kontrol noktasi kosuldugunda "
            "tur 2 bu kayitla karsilastirilacak ve 'yalniz_tekrar' o zaman anlam "
            "kazanacak."
        ),
    }


def havuz_yaz(negatif: list[dict]) -> None:
    HAVUZ.parent.mkdir(parents=True, exist_ok=True)
    with HAVUZ.open("w", encoding="utf-8") as akis:
        for satir in negatif:
            akis.write(json.dumps(satir, ensure_ascii=False, sort_keys=True) + "\n")


def havuz_oku() -> list[dict]:
    if not HAVUZ.is_file():
        return []
    return [
        json.loads(line)
        for line in HAVUZ.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dogrula() -> int:
    """The record must match a fresh measurement, not merely exist."""
    if not KAYIT.is_file():
        raise SystemExit(f"{KAYIT.relative_to(ROOT)} yok: tur olculmedi")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    hatalar: list[str] = []

    havuz = havuz_oku()
    nedenleri_olmayan = [s for s in havuz if not s.get("neden")]
    if nedenleri_olmayan:
        hatalar.append(
            f"{len(nedenleri_olmayan)} negatif satir neden tasimiyor: "
            "nedeni olmayan bir ret ogretilemez"
        )
    if len(havuz) != kayit["negatif_havuz"]["toplam"]:
        hatalar.append(
            f"havuz {len(havuz)} satir, kayit "
            f"{kayit['negatif_havuz']['toplam']} diyor"
        )

    taze = tur_olc()
    if taze["siniflar"] != kayit["siniflar"]:
        hatalar.append("kayittaki sinif sayilari yeniden olcumle eslesmiyor")
    if taze["gecen_toplam"] != kayit["gecen_toplam"]:
        hatalar.append("kayittaki gecen toplami yeniden olcumle eslesmiyor")
    if len(taze["negatif"]) != len(havuz):
        hatalar.append("yeniden olculen negatif sayisi havuzla eslesmiyor")

    # The delta is recomputed from the records; a stored delta that disagrees
    # with the arithmetic is a claim, not a measurement.
    fark = yeterlilik_farki(onceki_kayit(), taze)
    if fark != kayit["yeterlilik_farki"]:
        hatalar.append("yeterlilik farki kayitlardan yeniden hesaplanamiyor")

    for hata in hatalar:
        print(f"RED: {hata}")
    if hatalar:
        return 1
    print(
        f"tur {kayit['tur']} dogrulandi: {kayit['toplam_satir']} satir, "
        f"{kayit['gecen_toplam']} gecti, {len(havuz)} negatif "
        f"(nedenler: {kayit['negatif_havuz']['nedenlere_gore']}), "
        f"fark karsilastirilabilir={kayit['yeterlilik_farki']['karsilastirilabilir']}."
    )
    return 0


def self_test() -> int:
    """Canaries: an unlabelled refusal, a drifted count and a fabricated delta
    must each be refused; and the checker must be the one `eval_sft` has."""
    kayit_var = KAYIT.is_file()
    havuz_var = HAVUZ.is_file()
    eski_kayit = KAYIT.read_text(encoding="utf-8") if kayit_var else None
    eski_havuz = HAVUZ.read_text(encoding="utf-8") if havuz_var else None
    try:
        olcum = tur_olc()
        if olcum["toplam_satir"] <= 0:
            raise SystemExit("self-test: tur hic satir olcmedi")
        kayit = kayit_olustur(olcum)
        KAYIT.write_text(
            json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        havuz_yaz(olcum["negatif"])
        if dogrula() != 0:
            raise SystemExit("self-test: taze olcum kendi kaydini dogrulamadi")

        # Canary 1: a refusal with no reason is not teachable.
        kirik = [dict(s) for s in olcum["negatif"]]
        kirik.append({"tur": TUR, "sinif": "okuma", "kimlik": "okuma:99", "neden": []})
        havuz_yaz(kirik)
        if dogrula() != 1:
            raise SystemExit("self-test: nedensiz ret kabul edildi")

        # Canary 2: a record whose counts drifted from the data.
        havuz_yaz(olcum["negatif"])
        kayit["gecen_toplam"] = kayit["gecen_toplam"] + 1
        KAYIT.write_text(
            json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if dogrula() != 1:
            raise SystemExit("self-test: kaymis sayi kabul edildi")
        kayit["gecen_toplam"] = olcum["gecen_toplam"]

        # Canary 3: a delta invented rather than recomputed. With one round
        # there is nothing to compare, so a comparable delta is a fabrication.
        kayit["yeterlilik_farki"] = {
            "karsilastirilabilir": True,
            "onceki_tur": 0,
            "yeni_cozulen_siniflar": ["okuma"],
            "gerileyen_siniflar": [],
            "yalniz_tekrar": False,
        }
        KAYIT.write_text(
            json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if dogrula() != 1:
            raise SystemExit("self-test: uydurma yeterlilik farki kabul edildi")

        # Canary 4: the reasons come from eval_sft, not from a second rule set.
        ev = _modul("eval_sft", "eval_sft.py")
        if not hasattr(ev, "evaluate") or not hasattr(ev, "digest_of"):
            raise SystemExit("self-test: eval_sft yuzeyi degismis")
        if ev.MIN_BODY_CHARS <= 0:
            raise SystemExit("self-test: govde tabani anlamini yitirmis")
    finally:
        if eski_kayit is not None:
            KAYIT.write_text(eski_kayit, encoding="utf-8")
        elif KAYIT.is_file():
            KAYIT.unlink()
        if eski_havuz is not None:
            HAVUZ.write_text(eski_havuz, encoding="utf-8")
        elif HAVUZ.is_file():
            HAVUZ.unlink()
    print("self-test OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--olc", action="store_true", help="run a round and print it")
    grup.add_argument("--kur", action="store_true", help="write the round record and pool")
    grup.add_argument("--dogrula", action="store_true", help="verify the written record")
    grup.add_argument("--self-test", action="store_true", help="run the canaries")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.dogrula:
        return dogrula()
    olcum = tur_olc()
    if args.olc:
        cikti = {k: v for k, v in olcum.items() if k != "negatif"}
        cikti["negatif_ornek"] = olcum["negatif"][:3]
        print(json.dumps(cikti, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    kayit = kayit_olustur(olcum)
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8")
    havuz_yaz(olcum["negatif"])
    print(
        f"tur {TUR} yazildi: {kayit['toplam_satir']} satir, "
        f"{kayit['gecen_toplam']} gecti, {len(olcum['negatif'])} negatif -> "
        f"{HAVUZ.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
