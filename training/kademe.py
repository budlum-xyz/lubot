#!/usr/bin/env python3
"""Kademe egitimi adayinin (tasarim notu 3.5, kalem 1) olcumunu kayda gecirir.

Olcum Rust modulunun icindedir (`crates/egitim/src/kademe.rs`,
`olcum_raporu`): son kademe agirligi 1 iken kademeli kayip taban kaybin ta
kendisi mi (bit-ozdes), gradyanlar sonlu farkla uyumlu mu, agirliklar bire
normalize mi, kayip bicimi parametre tutuyor mu. Bu betik o satiri **kosar ve
okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    derinlik=6, ara kademeler=[2, 4] (kademe=3), agirliklar [0.25, 0.25, 0.5]:
    kademe sayisi 3 VE agirlik toplami 1e-12 icinde 1 VE taban farki tam 0 VE
    gradyan sapmasi < 1e-6 VE parametre 0.

"Kademe kaybi model kalitesini iyilestirir" iddiasi bu kayitta **yoktur**: o
iddia uzmanli bir egitim kosusu ister ve `olculmeyen` listesinde durur. Modul
adaydir; egitim hedefine baglanmasi M4 damgasi bekler (spec degismedi).

Kullanim:

    python3 training/kademe.py --olc
    python3 training/kademe.py --kur
    python3 training/kademe.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "kademe-2026-09-27.json"
ETIKET = "kademe |"
TAM_ALANLAR = ("derinlik", "kademe")
KESIRLI_ALANLAR = ("agirlik_toplam", "taban_fark", "gradyan_sapma", "kayip")
# Kayip biciminin parametre muhasebesi sifirdir; sayim degil, sifir oldugu
# icin TAM_ALANLAR'da degil ayri denetlenir (sayimlar pozitif tam sayidir).
SIFIR_ALANLAR = ("parametre",)


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-egitim",
         "kademe::tests::olcum_raporu", "--", "--nocapture"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return kosu.returncode, kosu.stdout + kosu.stderr


def _satiri_coz(cikti: str) -> dict[str, float]:
    satir = next((a for a in cikti.splitlines() if ETIKET in a), None)
    if satir is None:
        raise SystemExit("olcum satiri bulunamadi:\n" + cikti[-600:])
    olcum: dict[str, float] = {}
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR + SIFIR_ALANLAR:
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
        olcum["kademe"] == 3
        and abs(olcum["agirlik_toplam"] - 1.0) < 1e-12
        and olcum["taban_fark"] == 0.0
        and olcum["gradyan_sapma"] < 1e-6
        and olcum["parametre"] == 0
    )
    return olcum


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Kademe egitimi adayi (tasarim 3.5 kalem 1): her derinlik dagitilabilir - "
            "ara kademeler kayip tasir, taban kayip modulun icinde bit-ozdes tasinir, "
            "gradyanlar elle yazildi ve sonlu farkla denetlendi"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "kademe_3_agirlik_bir_taban_fark_sifir_gradyan_esik_alti",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "derinlik=6, ara kademeler=[2,4], agirliklar [0.25,0.25,0.5]: kademe=3 VE "
                "agirlik toplami 1e-12 icinde 1 VE taban farki tam 0 VE gradyan sapmasi "
                "< 1e-6 VE parametre=0"
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
            "Olcum sentetik kademe cikislaridir: uzmanli bir yigina baglanmadi, egitim "
            "kosusu yok. Modul adaydir ve bagli degildir; egitim hedefine baglanmasi "
            "M4 damgasi bekler (model_spec.json degismedi)."
        ),
        "olculmeyen": [
            "kademe kaybinin gercek egitim kosusunda model kalitesine etkisi (uzmanli kosu ister; M4 isaretli karar)",
            "agirlik semasinin secimi (uniform/azalan/harmonik: olcum izgarasi ister, kodda sabit deger yok)",
            "ara kademelerin hangi derinliklere dusacagi ile bellek merdiveni basamaklarinin eslenmesi (olculmedi)",
            "kademe kaybinin gradyan girisimi ve unutma etkisi (olculmedi)",
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
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR + SIFIR_ALANLAR:
        if alan not in kanit:
            return f"kanit alani yok: {alan}"
        deger = kanit[alan]
        # bool, int alt sinifi olsa da sayisal olcum degildir; NaN ve sonsuz
        # karsilastirmalarla sessizce gecmemeli (test_olcum_kayitlari.py).
        if type(deger) not in (int, float):
            return f"sayisal olmayan kanit: {alan}"
        if isinstance(deger, float) and not math.isfinite(deger):
            return f"sonlu olmayan kanit: {alan}"
        if alan in TAM_ALANLAR and (deger <= 0 or deger != int(deger)):
            return f"pozitif tam sayi olmayan kanit: {alan}"
    if kanit["parametre"] != 0:
        return "kayip bicimi parametre tutmamali (sayi sifir olmali)"
    # Sapma negatif olamaz; kayip pozitif olcumek. Ikisi de olcute girer.
    if kanit["gradyan_sapma"] < 0:
        return "negatif gradyan sapmasi"
    if kanit["kayip"] <= 0:
        return "pozitif olmayan kayip"
    sonuc = bool(
        kanit["kademe"] == 3
        and abs(kanit["agirlik_toplam"] - 1.0) < 1e-12
        and kanit["taban_fark"] == 0.0
        and kanit["gradyan_sapma"] < 1e-6
        and kanit["parametre"] == 0
    )
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
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR + SIFIR_ALANLAR:
        if float(eski[alan]) != float(taze[alan]):
            raise SystemExit(
                f"{alan} kayittan farkli cikti: kayit {eski[alan]}, olcum {taze[alan]}"
            )
    if kayit["olcut"]["sonuc"] != taze["olcut_sonucu"]:
        raise SystemExit("olcut sonucu degisti")
    return (
        "kayit taze: "
        f"derinlik {taze['derinlik']:.0f}, kademe {taze['kademe']:.0f}, "
        f"agirlik toplami {taze['agirlik_toplam']}, taban fark {taze['taban_fark']}, "
        f"gradyan sapma {taze['gradyan_sapma']:.3e}, {taze['sure_saniye']} s"
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
