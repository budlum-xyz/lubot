#!/usr/bin/env python3
"""Hadamard (Monarch bloklu) MLP adayinin karta-ozel olcum kaydi.

Olcum Rust testinin icindedir (`crates/egitim/src/mlp_hadamard.rs`,
`mlp_hadamard::tests::olcum_raporu_hadamard`): parametre muhasebesi, agirlik ve
girdi gradyanlarinin merkezi sonlu farkla denetimi, 40 adimlik inis ve blok
tasarrufu. Bu betik o satiri **kosar ve okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    parametre = denetlenen VE agirlik_sapma < 1e-5 VE girdi_sapma < 1e-5 VE
    inis_son < inis_baslangic VE blok_tasarruf > 0 VE bloksuz_standart_fark > 0

Son madde bir ustunluk iddiasi degil, kayda gecen bir gercektir: ayni ic
genislikte **bloksuz** Hadamard standart iki-matris MLP'den pahali oldugu icin
fark isaretli yazilir; tasarruf yalniz blok-kosegen halinde vardir.

"Bu modul modeli iyilestirir" iddiasi bu kayitta **yoktur**: o iddia uzmanli bir
egitim karsilastirmasi ister ve `olculmeyen` listesinde durur.

Kullanim:

    python3 training/hadamard_mlp.py --olc
    python3 training/hadamard_mlp.py --kur
    python3 training/hadamard_mlp.py --dogrula
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "hadamard-mlp-2026-09-27.json"
ETIKET = "hadamard-mlp |"
TAM_ALANLAR = (
    "d_model",
    "d_r",
    "blok_sayisi",
    "parametre",
    "denetlenen",
    "blok_tasarruf",
    "bloksuz_standart_fark",
)
KESIRLI_ALANLAR = (
    "agirlik_sapma",
    "girdi_sapma",
    "inis_baslangic",
    "inis_son",
)


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-egitim",
         "mlp_hadamard::tests::olcum_raporu_hadamard", "--", "--nocapture"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return kosu.returncode, kosu.stdout + kosu.stderr


def _satiri_coz(cikti: str) -> dict[str, float]:
    satir = next((a for a in cikti.splitlines() if ETIKET in a), None)
    if satir is None:
        raise SystemExit("olcum satiri bulunamadi:\n" + cikti[-600:])
    olcum: dict[str, float] = {}
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
        eslesme = re.search(rf"\b{alan}=([0-9.eE+-]+)", satir)
        if not eslesme:
            raise SystemExit(f"olcum satirinda {alan} yok:\n{satir[:400]}")
        olcum[alan] = float(eslesme.group(1))
    return olcum


def _olcut(k: dict[str, float]) -> bool:
    return bool(
        k["parametre"] == k["denetlenen"]
        and k["agirlik_sapma"] < 1e-5
        and k["girdi_sapma"] < 1e-5
        and k["inis_son"] < k["inis_baslangic"]
        and k["blok_tasarruf"] > 0
        and k["bloksuz_standart_fark"] > 0
    )


def olc() -> dict:
    basla = time.monotonic()
    kod, cikti = _test_kos()
    if kod != 0:
        raise SystemExit("olcum testi kirmizi:\n" + cikti[-800:])
    olcum = _satiri_coz(cikti)
    olcum["sure_saniye"] = round(time.monotonic() - basla, 2)
    olcum["olcut_sonucu"] = _olcut(olcum)
    return olcum


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Hadamard MLP adayi (tasarim 3.1): iki dogrusal projeksiyonun eleman-bazli "
            "carpimi, blok-kosegen ust izdusumler; parametre muhasebesi sekilden, "
            "gradyanlar merkezi sonlu farktan, inis gercek bir 40 adimlik kosudan"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:skills/port-hatti/port-kartlari/needle-hadamard-mlp.md",
        "olcut": {
            "ad": "parametre_muhasebesi_sekilden_ve_gradyan_sonlu_farkla",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "d_model=8, d_r=6, blok=2: parametre = denetlenen VE agirlik_sapma < 1e-5 "
                "VE girdi_sapma < 1e-5 VE inis_son < inis_baslangic VE blok_tasarruf > 0 "
                "VE bloksuz_standart_fark > 0"
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
            "Olcum kucuk ve tohumlu bir sekildedir (d_model=8, d_r=6, blok=2, 116 "
            "parametre); lubot-a1 aday sekli (64/256/4) bu kayitta kosulmadi. Inis "
            "bellege yazilmis sabit bir hedefe karsi olculur, korpus kosusu degildir. "
            "Modul hicbir spec'i ve hicbir egitim cagrisini degistirmez: bagli degil."
        ),
        "olculmeyen": [
            "kaynaktaki sayisal sonuclar (bu kayit yalniz kendi kosumumuzu olcer)",
            "model kalitesine etkisi: uzmanli egitim kosusu ve sinav seti ister",
            "lubot-a1 aday seklinde (64/256/4, 25152 parametre) ayni denetim: bu kayit "
            "kucuk sekilde kosulur, buyuk sekilde denetim maliyeti olculmedi",
            "blok sayisi ve ic genislik secimi izgarasi: isaretli mimari karar (M-listesi)",
            "omurgaya baglandiktan sonraki gecikme/bellek etkisi (modul bagli degil)",
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
        return "kayit bir sozluk degil"
    if not isinstance(kayit.get("olcut"), dict):
        return "olcut bolumu yok"
    if not isinstance(kayit["olcut"].get("sonuc"), bool):
        return "olcut.sonuc mantiksal degil"
    if not kayit.get("olculmeyen"):
        return "olculmeyen listesi bos"
    kanit = kayit.get("kanit")
    if not isinstance(kanit, dict):
        return "kanit bolumu yok"
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
        deger = kanit.get(alan)
        if not isinstance(deger, (int, float)) or isinstance(deger, bool):
            return f"kanit alani sayisal degil: {alan}"
    if _olcut({a: float(kanit[a]) for a in TAM_ALANLAR + KESIRLI_ALANLAR}) != kayit["olcut"]["sonuc"]:
        return "olcut ile kanit celisiyor"
    return None


def dogrula(yol: Path = KAYIT) -> str:
    if not yol.is_file():
        raise SystemExit(f"kayit yok: {yol}")
    kayit = json.loads(yol.read_text(encoding="utf-8"))
    bulgu = _bulgu(kayit)
    if bulgu:
        raise SystemExit(f"kayit semasi bozuk: {bulgu}")
    if not kayit["olcut"]["sonuc"]:
        raise SystemExit("kayit olcutu tutmuyor: kart bu kayitla ilerleyemez")
    taze = olc()
    eski = kayit["kanit"]
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
        if float(eski[alan]) != float(taze[alan]):
            raise SystemExit(
                f"{alan} kayittan farkli cikti: kayit {eski[alan]}, olcum {taze[alan]}"
            )
    if kayit["olcut"]["sonuc"] != taze["olcut_sonucu"]:
        raise SystemExit("olcut sonucu degisti")
    return "kayit taze: " + " ".join(
        f"{alan}={taze[alan]:g}" for alan in TAM_ALANLAR + KESIRLI_ALANLAR
    ) + f", {taze['sure_saniye']} s"


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
