#!/usr/bin/env python3
"""Sifir merkezli RMS norm adayinin olcumunu kayda gecirir.

Olcum Rust modulunun icindedir (`crates/egitim/src/normalizasyon.rs`,
`olcum_raporu`): kaydirma degismezligi, klasik RMS ile fark, cikti ortalamasi ve
RMS karesinin **eps tabanli tam iliskisi**, elle turetilen geri gecisin merkezi
farkla uyumu. Bu betik o satiri kosar ve okur.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    kaydirma ile cikti degismiyor (merkez_fark < 1e-12) VE klasik RMS
    kaydirmadan etkileniyor (duz_rms_fark > 1.0) VE gradyan sapmasi < 1e-6.

"Merkezleme kaliteyi artirir" iddiasi bu kayitta **yoktur**: o iddia bir egitim
karsilastirmasi ister ve `olculmeyen` listesinde durur.

Kullanim:

    python3 training/normalizasyon.py --olc
    python3 training/normalizasyon.py --kur
    python3 training/normalizasyon.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "normalizasyon-2026-09-27.json"
ETIKET = "normalizasyon |"
TAM_ALANLAR = ("genislik", "denetlenen", "parametre")
KESIRLI_ALANLAR = ("merkez_fark", "duz_rms_fark", "cikti_ort", "cikti_rms2", "gradyan_sapma")


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-egitim",
         "normalizasyon::tests::olcum_raporu", "--", "--nocapture"],
        cwd=ROOT, capture_output=True, text=True, check=False,
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
        olcum["merkez_fark"] < 1e-12
        and olcum["duz_rms_fark"] > 1.0
        and olcum["gradyan_sapma"] < 1e-6
        and olcum["denetlenen"] == olcum["genislik"]
    )
    return olcum


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Sifir merkezli RMS norm adayi (tasarim 3.2 norm yolu): kaydirma degismezligi, "
            "klasik RMS ile olculen fark, eps tabanli cikti iliskisi ve elle turetilen "
            "geri gecisin sonlu farkla uyumu"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:port-kartlari/needle-normalizasyon.md",
        "olcut": {
            "ad": "kaydirma_degismez_klasik_rms_degisken_ve_gradyan_fd_ile_uyumlu",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "genislik=8, eps=1e-6: merkez_fark < 1e-12 VE duz_rms_fark > 1.0 VE "
                "gradyan_sapma < 1e-6 VE denetlenen girdi sayisi = genislik"
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
            "Olcum sentetik bir vektordur; hicbir katmana baglanmadi ve hicbir egitim "
            "kosusunda kullanilmadi. Olcekleme degismezligi **yaklasiktir**: sapma eps "
            "tabanindan gelir ve eps kuculdukce kuculur (testte iki eps degeri ile olculur)."
        ),
        "olculmeyen": [
            "norm seciminin egitim kaybina etkisi (karsilastirmali kosu ister)",
            "cekirdekteki QK-norm ile degistirilebilirlik (q/k yoluna baglanmadi)",
            "derin yiginlarda karisik hassasiyet davranisi",
            "eps secimi izgarasi (yalniz 1e-6 ve 1e-18 karsilastirildi)",
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
    sonuc = bool(
        kanit["merkez_fark"] < 1e-12
        and kanit["duz_rms_fark"] > 1.0
        and kanit["gradyan_sapma"] < 1e-6
        and kanit["denetlenen"] == kanit["genislik"]
    )
    if sonuc != kayit["olcut"]["sonuc"]:
        return "olcut ile kanit celisiyor"
    if kanit["parametre"] != kanit["genislik"]:
        return "parametre sayisi genislik degil (olcek vektoru sekle bagli olmali)"
    if kanit["cikti_ort"] >= 1e-12:
        return "cikti sifir ortalamali degil"
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
        f"genislik {taze['genislik']:.0f}, kaydirma farki {taze['merkez_fark']:.3e}, "
        f"klasik RMS farki {taze['duz_rms_fark']:.3f}, gradyan sapmasi "
        f"{taze['gradyan_sapma']:.3e}, {taze['sure_saniye']} s"
    )


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--olc", action="store_true", help="olcumu kos ve bas")
    ayristirici.add_argument("--kur", action="store_true", help="kaydi yaz")
    ayristirici.add_argument("--dogrula", action="store_true", help="kayit taze mi")
    ayristirici.add_argument("--kayit", type=Path, default=KAYIT, help="kayit yolu")
    args = ayristirici.parse_args(args=argv)
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
