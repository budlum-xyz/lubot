#!/usr/bin/env python3
"""Rota adayinin (tasarim notu 3.5) olcumunu kayda gecirir.

Olcum Rust modulunun icindedir (`crates/egitim/src/yonlendirme.rs`,
`olcum_raporu`): top-k destegi korunuyor mu, satirlar 1'e toplaniyor mu, yuk
dengesi klasik tabandan iyi mi, sayimlar sekle bagli mi. Bu betik o satiri
**kosar ve okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    satir sapmasi < 1e-12 VE bu modulun yuk orani klasik tabanin oranindan
    KUCUK VE secim sayisi = jeton x k.

"Rota modeli iyilestirir" iddiasi bu kayitta **yoktur**: o iddia uzmanli bir
egitim karsilastirmasi ister ve `olculmeyen` listesinde durur.

Kullanim:

    python3 training/yonlendirme.py --olc
    python3 training/yonlendirme.py --kur
    python3 training/yonlendirme.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "sinkhorn-yonlendirme-2026-09-27.json"
ETIKET = "yonlendirme |"
TAM_ALANLAR = ("jeton", "uzman", "k", "yineleme", "secim", "parametre")
KESIRLI_ALANLAR = ("satir_sapma", "yuk_orani", "taban_orani")


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-egitim",
         "yonlendirme::tests::olcum_raporu", "--", "--nocapture"],
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
        olcum["satir_sapma"] < 1e-12
        and olcum["yuk_orani"] < olcum["taban_orani"]
        and olcum["secim"] == olcum["jeton"] * olcum["k"]
    )
    return olcum


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Sinkhorn rota adayi (tasarim 3.5): top-k destegi uzerinde log uzayinda iki "
            "yonlu normalizasyon; yuk dengesi klasik tabana karsi olculdu"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:port-kartlari/needle-sinkhorn-yonlendirme.md",
        "olcut": {
            "ad": "satir_toplamlari_bir_ve_yuk_orani_tabandan_kucuk",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "jeton=128, uzman=8, k=2, yineleme=8: satir sapmasi < 1e-12 VE yuk orani "
                "klasik tabanin oranindan kucuk VE secim sayisi = jeton x k"
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
            "Olcum sentetik puanlardir: uzmanli bir aga baglanmadi, egitim kosusu yok. "
            "Ogrenilebilir parametre sayisi sifirdir - rota bir hesaplamadir."
        ),
        "olculmeyen": [
            "rotanin model kalitesine etkisi (uzmanli egitim kosusu ister)",
            "uzman sayisi ve k secimi izgarasi (isaretli karar)",
            "dilim (jeton blogu) bazli dengeleme: denge burada tum yigin uzerinde olculdu",
            "egitim sirasinda yuk dagiliminin adim adim izlenmesi",
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
        kanit["satir_sapma"] < 1e-12
        and kanit["yuk_orani"] < kanit["taban_orani"]
        and kanit["secim"] == kanit["jeton"] * kanit["k"]
    )
    if sonuc != kayit["olcut"]["sonuc"]:
        return "olcut ile kanit celisiyor"
    if kanit["parametre"] != 0:
        return "rota parametre tutmamali (parametre sayisi sifir olmali)"
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
        f"{taze['jeton']:.0f} jeton x {taze['uzman']:.0f} uzman, k={taze['k']:.0f}, "
        f"satir sapmasi {taze['satir_sapma']:.3e}, yuk orani {taze['yuk_orani']:.6} "
        f"(taban {taze['taban_orani']:.6}), {taze['sure_saniye']} s"
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
