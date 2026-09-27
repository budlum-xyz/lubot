#!/usr/bin/env python3
"""Engram bellegi adayinin (tasarim notu 3.3) olcumunu kayda gecirir.

Olcumun kendisi Rust modulunun icindedir (`crates/egitim/src/engram.rs`,
`olcum_raporu`): tablo muhasebesi, kac konumun okundugu, kayit sinirinda kac
konumun atlandigi, hucre dolulugu, **cakisma** (olculen kayip) ve orneklenen
tablo gradyanlarinin sonlu fark sapmasi. Bu betik olsa olsa o satiri **kosar ve
okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    cakisan hucre sayisi sifirdan buyuk (cakisma yasak degil, olculur) VE
    orneklenen gradyan sapmasi 1e-6'nin altinda.

"Engram modeli iyilestirir" gibi bir iddia bu kayitta **yoktur**: o iddia
korpuslu bir egitim karsilastirmasi ister.

Kullanim:

    python3 training/engram.py --olc
    python3 training/engram.py --kur
    python3 training/engram.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "engram-2026-09-26.json"
ETIKET = "engram |"
TAM_ALANLAR = (
    "parametre",
    "okuma",
    "atlanan_gecmis",
    "atlanan_kayit_siniri",
    "tablo",
    "dolu_hucre",
    "cakisan_hucre",
)
KESIRLI_ALANLAR = ("doluluk", "grad_sapma")


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        [
            "cargo",
            "test",
            "-q",
            "-p",
            "lubot-egitim",
            "engram::tests::olcum_raporu",
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
        olcum["cakisan_hucre"] > 0.0 and olcum["grad_sapma"] < 1e-6
    )
    return olcum


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Engram bellegi adayi (tasarim 3.3): n-gram karmasiyla adreslenen KV tablosu; "
            "kayit siniri okumalara uygulandi, cakisma olculdu, seyrek geri gecis sonlu "
            "farkla denetlendi"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "cakisan_hucre_var_ve_gradyan_sapmasi_esik_altinda",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "n=3, tablo=512, d_kv=16, 256 jeton: cakisan_hucre > 0 VE orneklenen "
                "tablo gradyanlarinin sonlu fark sapmasi < 1e-6"
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
            "olcum tek bir sentetik dizidir: korpus yok, dikkat cagrisi yok, spec baglanmadi. "
            "Cakisma sayisi bir kayip olarak **olculdu**, yasaklanmadi."
        ),
        "olculmeyen": [
            "cakismanin model kalitesine etkisi (egitim kosusu ister)",
            "tablo boyutu aday izgarasi (isaretli karar; K6 butcesinden yer)",
            "kontrol noktasi bicimi (LUBOTCKPT'e yeni alan) ve dikkat cagrisina baglanma",
            "cok is parcacikli kosuda geri gecis determinizmi (tek is parcasinda olculdu)",
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
    if not isinstance(kayit.get("olcut"), dict):
        return "olcut bolumu yok"
    if not isinstance(kayit["olcut"].get("sonuc"), bool):
        return "olcut.sonuc mantiksal degil"
    kanit = kayit.get("kanit") or {}
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
        if alan not in kanit:
            return f"kanit alani yok: {alan}"
    sonuc = bool(kanit["cakisan_hucre"] > 0.0 and kanit["grad_sapma"] < 1e-6)
    if sonuc != kayit["olcut"]["sonuc"]:
        return "olcut ile kanit celisiyor"
    if kanit["okuma"] + kanit["atlanan_gecmis"] + kanit["atlanan_kayit_siniri"] != 256:
        return "okuma sayimi diziyi tutmuyor (okuma + atlanan != jeton sayisi)"
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
        f"{taze['parametre']:.0f} parametre, {taze['okuma']:.0f}/{256} konum okundu, "
        f"tablo {taze['tablo']:.0f} icinde {taze['dolu_hucre']:.0f} hucre dolu, "
        f"{taze['cakisan_hucre']:.0f} cakisan hucre, grad sapma {taze['grad_sapma']:.3e} "
        f"({taze['sure_saniye']} s)"
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
