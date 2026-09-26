#!/usr/bin/env python3
"""Cok-seritli artik baglanti (bilesen 5) olcumunu kayda gecirir.

Olcumun kendisi Rust modulunun icindedir (`crates/egitim/src/cok_serit.rs`,
`rms_profili_raporu`): ayni yiginin serit=1/2/4 hallerinin derinlige gore
katman-girdisi RMS profili, iki katman kazanci icin. Bu betik olsa olsa o
satiri **kosar ve okur**; sayilari kendisi uretmez - ikinci bir uygulama,
olcumun iki farkli yerden cikmasi demektir.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    her katman kazanci icin, adim basina geometrik oran (son/ilk)^(1/(n-1))
    serit sayisiyla monoton artar.

Kalibrasyon kaydinda oldugu gibi "olculmeyen" listesi kayitta acik durur.

Kullanim:

    python3 training/cok_serit.py --olc      # satiri kos, sayilari bas
    python3 training/cok_serit.py --kur      # kaydi yaz
    python3 training/cok_serit.py --dogrula  # kayit taze mi (yeniden olc)
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "cok-serit-2026-09-26.json"
SATIR = re.compile(r"kazanc=([0-9.]+);serit=(\d+):([0-9eE.,+-]+)")
ETIKET = "cok-serit rms profili"


def _test_kos() -> tuple[int, str]:
    """Olcum satirini ureten testi kosar (tek test, --nocapture)."""
    kosu = subprocess.run(
        [
            "cargo",
            "test",
            "-q",
            "-p",
            "lubot-egitim",
            "cok_serit::tests::rms_profili_raporu",
            "--",
            "--nocapture",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return kosu.returncode, kosu.stdout + kosu.stderr


def _satirlari_coz(cikti: str) -> dict[str, dict[str, list[float]]]:
    """`kazanc -> serit -> profil` sozlugu; satir yoksa/bozuksa hata."""
    satir = None
    for aday in cikti.splitlines():
        if ETIKET in aday:
            satir = aday
            break
    if satir is None:
        raise SystemExit("olcum satiri bulunamadi:\n" + cikti[-600:])
    sonuc: dict[str, dict[str, list[float]]] = {}
    for kazanc, serit, degerler in SATIR.findall(satir):
        profil = [float(v) for v in degerler.rstrip(",").split(",") if v]
        if len(profil) < 2:
            raise SystemExit(f"profil cok kisa: kazanc={kazanc} serit={serit}")
        sonuc.setdefault(kazanc, {})[serit] = profil
    if not sonuc:
        raise SystemExit("olcum satiri cozulemedi:\n" + satir[:400])
    return sonuc


def _oran(profil: list[float]) -> float:
    """Adim basina geometrik zayiflama orani (ilk -> son)."""
    ilk, son = profil[0], profil[-1]
    if ilk <= 0.0 or son <= 0.0:
        raise SystemExit("profil sifir iceriyor: oran hesaplanamaz")
    return (son / ilk) ** (1.0 / (len(profil) - 1))


def olc() -> dict:
    """Taze olcum: profiller + turetilen oranlar + olcut sonucu."""
    basla = time.monotonic()
    kod, cikti = _test_kos()
    if kod != 0:
        raise SystemExit("olcum testi kirmizi:\n" + cikti[-800:])
    profiller = _satirlari_coz(cikti)
    oranlar = {
        kazanc: {serit: _oran(profil) for serit, profil in sorted(seritler.items())}
        for kazanc, seritler in sorted(profiller.items())
    }
    # Olcut: her kazanc noktasinda oran, serit sayisiyla birlikte 1'e yaklasir.
    monoton = all(
        all(a < b for a, b in zip(list(or_serit.values()), list(or_serit.values())[1:]))
        for or_serit in oranlar.values()
    )
    return {
        "profiller": profiller,
        "oranlar": {k: {s: round(v, 6) for s, v in d.items()} for k, d in oranlar.items()},
        "serit_sayilari": sorted({int(s) for d in profiller.values() for s in d}),
        "kazanclar": sorted(float(k) for k in profiller),
        "sure_saniye": round(time.monotonic() - basla, 2),
        "olcut_sonucu": monoton,
    }


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Cok-seritli artik baglanti adayi (tasarim 3.4): serit sayisinin derinlik "
            "boyunca katman-girdisi RMS profiline etkisi olculdu; gradyan sonlu farkla, "
            "ins gercek kosuyla"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "serit_sayisi_ile_adim_basina_geometrik_oran_monoton_artar",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "her katman kazanci icin: oran(serit=1) < oran(serit=2) < oran(serit=4), "
                "oran = (son/ilk)^(1/(n-1))"
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
            "Olcum bir PROXY'dir: 16 genislik, 8 derinlik, tohumlu agirliklar ve kasten "
            "buyuten/azaltan dogrusal 'katman'. Gercek yiginin (spec) serit sayisi karari "
            "bu olcumle verilmez; isaretli karar M3'tur."
        ),
        "olculmeyen": [
            "egitilmis bir yiginda serit etkisi (spec baglanmadi, K6)",
            "muP/transfer tarafi (proxy genislik olcegi ayri is)",
            "serit sayisinin bellek/maliyet etkisi gercek cihazda (K6 tavani disinda)",
        ],
    }


def kur() -> Path:
    kayit = _kayit(olc())
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    return KAYIT


def _bulgu(kayit: dict) -> str | None:
    """Kaydin kendi iddiasini tasiyip tasimadigi: tek olcutlu sema."""
    if not isinstance(kayit.get("olcut"), dict):
        return "olcut bolumu yok"
    if not isinstance(kayit["olcut"].get("sonuc"), bool):
        return "olcut.sonuc mantiksal degil"
    kanit = kayit.get("kanit") or {}
    if not kanit.get("profiller") or not kanit.get("oranlar"):
        return "kanit profilleri/oranlari yok"
    if kanit.get("olcut_sonucu") != kayit["olcut"]["sonuc"]:
        return "olcut ile kanit celisiyor"
    return None


def dogrula(yol: Path = KAYIT) -> str:
    """Kayit taze mi: ayni olcum simdi de ayni sayilari veriyor mu."""
    if not yol.is_file():
        raise SystemExit(f"kayit yok: {yol}")
    kayit = json.loads(yol.read_text(encoding="utf-8"))
    bulgu = _bulgu(kayit)
    if bulgu:
        raise SystemExit(f"kayit semasi bozuk: {bulgu}")
    taze = olc()
    eski = kayit["kanit"]
    if eski.get("profiller") != taze["profiller"]:
        raise SystemExit("profil kayittan farkli cikti: olcum taze degil")
    if eski.get("oranlar") != taze["oranlar"]:
        raise SystemExit("oranlar kayittan farkli cikti: olcum taze degil")
    if taze["olcut_sonucu"] != kayit["olcut"]["sonuc"]:
        raise SystemExit("olcut sonucu degisti")
    oranlar = taze["oranlar"]
    ozet = "; ".join(
        f"kazanc={k}: " + ", ".join(f"serit={s} {v:.6f}" for s, v in d.items())
        for k, d in sorted(oranlar.items())
    )
    return f"kayit taze: {ozet} ({taze['sure_saniye']} s)"


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
        yol = kur()
        print(f"kayit yazildi: {yol.relative_to(ROOT)}")
        return 0
    if args.dogrula:
        print(dogrula(args.kayit))
        return 0
    ayristirici.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
