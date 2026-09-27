#!/usr/bin/env python3
"""Aile kesitinin (derinlik/genislik dilimleri) olcumunu kayda gecirir.

Olcum Rust modulunun icindedir (`crates/egitim/src/kesit.rs`, `olcum_raporu`):
her derinlik ve genislik icin bir kesit turetildi mi, **kosuyor** mu (ileri+geri
adim, sonlu pozitif kayip), parametre sayisi derinlik/genislik ile monoton mu ve
tam kesit girdi spec'inin kendisi mi. Bu betik o satiri kosar ve okur.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    kosan = derinlik sayisi x 3 (her derinlikte uc genislik) VE
    tam kesitin parametre sayisi girdi spec'ininkine esit VE
    en dar kesitin parametresi tam kesitten kucuk.

"Kesitler model kalitesini korur" iddiasi bu kayitta **yoktur**: o iddia egitim
ve sinav karsilastirmasi ister ve `olculmeyen` listesinde durur.

Kullanim:

    python3 training/kesit.py --olc
    python3 training/kesit.py --kur
    python3 training/kesit.py --dogrula
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "kesit-2026-09-27.json"
ETIKET = "kesit |"
TAM_ALANLAR = ("derinlik", "genislik", "d_k", "izgara", "kosan",
               "parametre_en_az", "parametre_tam", "parametre_genislik_en_az", "tam_ozdes")


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-egitim",
         "kesit::tests::olcum_raporu", "--", "--nocapture"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return kosu.returncode, kosu.stdout + kosu.stderr


def _satiri_coz(cikti: str) -> dict[str, float]:
    satir = next((a for a in cikti.splitlines() if ETIKET in a), None)
    if satir is None:
        raise SystemExit("olcum satiri bulunamadi:\n" + cikti[-600:])
    olcum: dict[str, float] = {}
    for alan in TAM_ALANLAR:
        eslesme = re.search(rf"{alan}=([0-9.eE+-]+)", satir)
        if not eslesme:
            raise SystemExit(f"olcum satirinda {alan} yok:\n{satir[:400]}")
        olcum[alan] = float(eslesme.group(1))
    return olcum


def olc() -> dict:
    basla = time.monotonic()
    kod, cikti = _test_kos()
    if kod != 0:
        raise SystemExit("olcum testi kirmizi:\n" + cikti[-800:])
    olcum = _satiri_coz(cikti)
    olcum["sure_saniye"] = round(time.monotonic() - basla, 2)
    olcum["olcut_sonucu"] = bool(
        olcum["kosan"] == olcum["derinlik"] * 3
        and olcum["tam_ozdes"] == 1
        and olcum["parametre_tam"] > 0
        and olcum["parametre_genislik_en_az"] < olcum["parametre_tam"]
    )
    return olcum


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Aile kesiti (derinlik/genislik dilimleri): spec'ten her derinlik ve genislik "
            "icin kosan alt-modeller; tamlik, monotonluk ve kafa silme olculdu"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "tasarim_karti": "workspace:tasarim-kartlari/derinlik-genislik.md",
        "olcut": {
            "ad": "her_derinlikte_kesit_kosar_ve_parametre_monoton",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "derinlik=4, genislik=64: kosan = 4 x 3 = 12 kesit (ileri+geri adim, "
                "sonlu pozitif kayip) VE tam agirliklar bit ozdes (tam_ozdes=1) VE "
                "en dar genislik kesitinin parametresi tam kesitten kucuk"
            ),
        },
        "kaynaklar": {
            "girdi_jetonlari": 0,
            "cikti_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "maliyet": 0.0,
            "sure_saniye": olcum["sure_saniye"],
        },
        "kanit": olcum,
        "uyari": (
            "Kesit ayni kaynak agirliklardan koordinatla kirpilir. Olculen sey "
            "'bu spec ile bir adim kosuyor' ve 'parametre sayisi monoton' - 'kesilmis "
            "model kaliteyi koruyor' degil."
        ),
        "olculmeyen": [
            "kesilmis modelin sinav/kalite davranisi (egitim ve sinav kosusu ister)",
            "optimizer momentlerini alt-modele tasima ve ortak agirlik egitimi olculmedi",
            "ortadan katman cikarma (sadece bastan kisaltma olculdu)",
            "K/V grup orani degisebilir: dar kesitin kaynakla fonksiyonel esdegerligi iddia edilmez",
        ],
    }


def kur() -> Path:
    kayit = _kayit(olc())
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(
        json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KAYIT


def _bulgu(kayit: dict) -> str | None:
    if not isinstance(kayit, dict):
        return "kayit nesne degil"
    if not isinstance(kayit.get("olcut"), dict):
        return "olcut bolumu yok"
    if not isinstance(kayit["olcut"].get("sonuc"), bool):
        return "olcut.sonuc mantiksal degil"
    kanit = kayit.get("kanit")
    if not isinstance(kanit, dict):
        return "kanit nesne degil"
    for alan in TAM_ALANLAR:
        if alan not in kanit:
            return f"kanit alani yok: {alan}"
        deger = kanit[alan]
        # bool, int alt sinifi olsa da sayisal olcum degildir. NaN ve sonsuz
        # karsilastirmalarla sessizce gecmemeli; once tur ve sonluluk denetlenir.
        if type(deger) not in (int, float):
            return f"sayisal olmayan kanit: {alan}"
        if isinstance(deger, float) and not math.isfinite(deger):
            return f"sonlu olmayan kanit: {alan}"
        if alan in TAM_ALANLAR and (deger <= 0 or deger != int(deger)):
            return f"pozitif tam sayi olmayan kanit: {alan}"
    sonuc = bool(
        kanit["kosan"] == kanit["derinlik"] * 3
        and kanit["tam_ozdes"] == 1
        and kanit["parametre_tam"] > 0
        and kanit["parametre_genislik_en_az"] < kanit["parametre_tam"]
    )
    if sonuc != kayit["olcut"]["sonuc"]:
        return "olcut ile kanit celisiyor"
    if kanit["d_k"] <= 0 or kanit["genislik"] % kanit["d_k"] != 0:
        return "d_k tam sayi kafa bolmesi degil"
    return None


def dogrula(yol: Path = KAYIT) -> str:
    if not yol.is_file():
        raise SystemExit(f"kayit yok: {yol}")
    kayit = json.loads(yol.read_text(encoding="utf-8"))
    bulgu = _bulgu(kayit)
    if bulgu:
        raise SystemExit(f"kayit semasi bozuk: {bulgu}")
    taze = olc()
    eski = kayit["kanit"]
    for alan in TAM_ALANLAR:
        if float(eski[alan]) != float(taze[alan]):
            raise SystemExit(
                f"{alan} kayittan farkli cikti: kayit {eski[alan]}, olcum {taze[alan]}"
            )
    if kayit["olcut"]["sonuc"] != taze["olcut_sonucu"]:
        raise SystemExit("olcut sonucu degisti")
    return (
        "kayit taze: "
        f"{taze['kosan']:.0f} kesit kosuyor ({taze['derinlik']:.0f} derinlik x 3 genislik), "
        f"parametre {taze['parametre_en_az']:.0f}..{taze['parametre_tam']:.0f}, "
        f"d_k={taze['d_k']:.0f}, {taze['sure_saniye']} s"
    )


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--olc", action="store_true", help="olcumu kos ve bas")
    ayristirici.add_argument("--kur", action="store_true", help="kaydi yaz")
    ayristirici.add_argument("--dogrula", action="store_true", help="kayit taze mi")
    ayristirici.add_argument("--kayit", type=Path, default=KAYIT, help="kayit yolu")
    args = ayristirici.parse_args(argv)
    if args.olc:
        print(json.dumps(olc(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.kur:
        print(f"kayit yazildi: {kur().relative_to(ROOT)}")
        return 0
    if args.dogrula:
        print(dogrula(args.kayit))
        return 0
    ayristirici.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
