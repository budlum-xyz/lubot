#!/usr/bin/env python3
"""Birlesik port blogunun (tasarim notu 3.8) olcumunu kayda gecirir.

Olcum Rust modulunun icindedir (`crates/egitim/src/birlesik.rs`,
`olcum_raporu`): alti bilesen tek blokta kosuyor mu, parametre muhasebesi
bilesenlerin toplami mi, her parametre sonlu farkla denetlendi mi, ve
kompozisyon gercekten iniyor mu. Bu betik o satiri **kosar ve okur**; sayilari
kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    alti bilesen VE denetlenen parametre sayisi = sekilden turetilen parametre
    sayisi VE sonlu fark ihlali sifir VE son kayip ilk kayiptan kucuk.

"Birlesik blok modeli iyilestirir" iddiasi bu kayitta **yoktur**: o iddia
egitilmis bir kontrol noktasi ve sinav seti ister ve `olculmeyen` listesinde
durur. Bu kayit yalniz kompozisyonun **kostugunu ve gradyaninin dogru
oldugunu** olcer.

Kullanim:

    python3 training/birlesik.py --olc
    python3 training/birlesik.py --kur
    python3 training/birlesik.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "birlesik-2026-09-27.json"
ETIKET = "birlesik |"
TAM_ALANLAR = ("bilesen", "parametre", "denetlenen", "uzman", "serit", "engram_okuma")
KESIRLI_ALANLAR = ("ihlal", "oran", "ilk_kayip", "son_kayip")
# Kompozisyonun kac tekil adaydan kuruldugu: serit, norm, hadamard, rota,
# engram, kesit ailesi. Sayi burada sabit degil, kayitla karsilastirilan bir
# beklenti; degisirse kayit da degismek zorunda.
BEKLENEN_BILESEN = 6


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        [
            "cargo",
            "test",
            "-q",
            "-p",
            "lubot-egitim",
            "birlesik::tests::olcum_raporu",
            "--",
            "--nocapture",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
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


def _olcut_sonucu(olcum: dict) -> bool:
    return bool(
        olcum["bilesen"] == BEKLENEN_BILESEN
        and olcum["denetlenen"] == olcum["parametre"]
        and olcum["ihlal"] == 0
        and olcum["son_kayip"] < olcum["ilk_kayip"]
    )


def olc() -> dict:
    basla = time.monotonic()
    kod, cikti = _test_kos()
    if kod != 0:
        raise SystemExit("olcum testi kirmizi:\n" + cikti[-800:])
    olcum = _satiri_coz(cikti)
    olcum["sure_saniye"] = round(time.monotonic() - basla, 2)
    olcum["olcut_sonucu"] = _olcut_sonucu(olcum)
    return olcum


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Birlesik port blogu (tasarim 3.8): serit okuma -> sifir merkezli RMS norm "
            "-> rotali Hadamard uzmanlari -> engram deger bellegi -> serit yazma; "
            "elle yazilmis geri gecis dort noktali sonlu farkla denetlendi"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:port-kartlari/birlesik-mimari-7-3.md",
        "olcut": {
            "ad": "alti_bilesen_tek_blokta_ve_gradyan_ihlali_sifir",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "bilesen=6 VE denetlenen = sekilden turetilen parametre sayisi VE "
                "sonlu fark ihlali = 0 VE son kayip < ilk kayip"
            ),
        },
        "kaynaklar": {
            "girdi_jetonlari": 0,
            "cikti_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "maliyet": 0.0,
            "sure_saniye": olcum["sure_saniye"],
        },
        "kanit": {
            alan: olcum[alan]
            for alan in TAM_ALANLAR + KESIRLI_ALANLAR + ("sure_saniye", "olcut_sonucu")
        },
        "uyari": (
            "Olcum sentetik durumlar ve sentetik rota puanlari uzerindedir: egitilmis "
            "kontrol noktasi, korpus ve sinav seti yoktur. Blok hicbir model ailesine "
            "bagli degildir; model_spec.json degismedi. Rota puanlari blogun disindan "
            "gelir, dL/dpuanlar yazilmadi."
        ),
        "olculmeyen": [
            "kompozisyonun model kalitesine etkisi (egitilmis kontrol noktasi ister)",
            "hangi bilesenin hangi aileye girecegi (M1/M2/M3 isaretli mimari karar)",
            "gercek korpus uzerinde kosu suresi ve bellek tavani",
            "rota puanlarinin gradyani (blok disindan gelir)",
            "cok katmanli yigin: burada tek blok olculdu",
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
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
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
    for alan in ("ihlal", "oran", "ilk_kayip", "son_kayip"):
        if kanit[alan] < 0:
            return f"negatif buyukluk: {alan}"
    if kanit["ihlal"] != int(kanit["ihlal"]):
        return "ihlal sayisi tam sayi degil"
    if kanit["bilesen"] != BEKLENEN_BILESEN:
        return f"bilesen sayisi {BEKLENEN_BILESEN} degil"
    if kanit["denetlenen"] != kanit["parametre"]:
        return "denetlenen parametre sayisi sekilden turetilen sayiya esit degil"
    if kanit["engram_okuma"] <= 0:
        return "engram hic okumamis: bellek kolu olculmemis sayilir"
    if kanit["uzman"] <= 1:
        return "tek uzman: rota kolu olculmemis sayilir"
    if kanit["serit"] <= 1:
        return "tek serit: cok-serit kolu olculmemis sayilir"
    sonuc = _olcut_sonucu(kanit)
    if sonuc != kayit["olcut"]["sonuc"]:
        return "olcut ile kanit celisiyor"
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
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
        if float(eski[alan]) != float(taze[alan]):
            raise SystemExit(
                f"{alan} kayittan farkli cikti: kayit {eski[alan]}, olcum {taze[alan]}"
            )
    if kayit["olcut"]["sonuc"] != taze["olcut_sonucu"]:
        raise SystemExit("olcut sonucu degisti")
    return (
        "kayit taze: "
        f"{taze['bilesen']:.0f} bilesen, {taze['parametre']:.0f} parametre "
        f"({taze['denetlenen']:.0f} denetlendi, {taze['ihlal']:.0f} ihlal), "
        f"en kotu oran {taze['oran']:.3e}, kayip {taze['ilk_kayip']:.6} -> "
        f"{taze['son_kayip']:.6}, {taze['sure_saniye']} s"
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
