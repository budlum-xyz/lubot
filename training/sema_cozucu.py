#!/usr/bin/env python3
"""Sema cozucu (sema-kisitli decode) karta-ozel olcum kaydi.

Olcum Rust testinin icindedir (`crates/sema-cozucu/src/cozucu.rs`,
`cozucu::tests::olcum_raporu_sema_cozucu`): otomatin maskesiyle kosulan
decode'lar, uretilen belgelerin **cikti dogrulayicisiyla** capraz denetimi,
maskesiz taban karsilastirmasi ve red yollari. Bu betik o satiri **kosar ve
okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    dogrulayici_kacis = 0 VE rastgele_kacis = 0 VE naif_gecersiz > 0 VE
    cikmaz_red >= 1 VE adim_bitti_red >= 1 VE utf8_red = 4 VE belge > 0 VE
    bayt > 0

Iki tarafin anlami:

* **Kacis yok** (ilk iki madde): otomatin kabul ettigi her belge
  `lubot-read::output_schema` dogrulayicisindan da gecer. Gramer,
  dogrulayicinin bir **alt kumesi** olmak zorunda; tek bir kacis bile bu kartin iddiasini
  dusurur.
* **Maske bos degil** (ucuncu madde): maske kaldirildiginda ayni decode
  gecersiz belgeler uretiyor olmali. Aksi halde "gecerlilik" maskeden degil
  sans eseri sozlukten geliyor demektir.

"Bu modul cevap kalitesini iyilestirir" iddiasi bu kayitta **yoktur**: logitler
deterministik bir karisimdan gelir, egitilmis bir kontrol noktasi yoktur.
Olculen sey modelin kalitesi degil maskenin ne yaptigidir; kalite iddiasi
`olculmeyen` listesinde durur.

Kullanim:

    python3 training/sema_cozucu.py --olc
    python3 training/sema_cozucu.py --kur
    python3 training/sema_cozucu.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "sema-cozucu-2026-09-27.json"
ETIKET = "sema-cozucu |"
TAM_ALANLAR = (
    "sozluk",
    "belge",
    "adim",
    "maskelenen",
    "bayt",
    "rastgele_belge",
    "dogrulayici_kacis",
    "rastgele_kacis",
    "naif_belge",
    "naif_gecersiz",
    "cikmaz_red",
    "adim_bitti_red",
    "utf8_red",
)


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-sema-cozucu",
         "cozucu::tests::olcum_raporu_sema_cozucu", "--", "--nocapture"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return kosu.returncode, kosu.stdout + kosu.stderr


def _satiri_coz(cikti: str) -> dict[str, float]:
    satir = next((a for a in cikti.splitlines() if ETIKET in a), None)
    if satir is None:
        raise SystemExit("olcum satiri bulunamadi:\n" + cikti[-600:])
    olcum: dict[str, float] = {}
    for alan in TAM_ALANLAR:
        eslesme = re.search(rf"\b{alan}=([0-9.eE+-]+)", satir)
        if not eslesme:
            raise SystemExit(f"olcum satirinda {alan} yok:\n{satir[:400]}")
        olcum[alan] = float(eslesme.group(1))
    return olcum


def _olcut(k: dict[str, float]) -> bool:
    return bool(
        k["dogrulayici_kacis"] == 0
        and k["rastgele_kacis"] == 0
        and k["naif_gecersiz"] > 0
        and k["cikmaz_red"] >= 1
        and k["adim_bitti_red"] >= 1
        and k["utf8_red"] == 4
        and k["belge"] > 0
        and k["bayt"] > 0
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
            "Sema cozucu (tasarim 3.7): cikti sozlesmesini bayt otomatina ceviren "
            "maske, maskeli decode ve uretilen belgenin cikti dogrulayicisiyla "
            "capraz denetimi; maskesiz taban karsilastirmasi ve red yollari"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:skills/port-hatti/port-kartlari/sema-cozucu.md",
        "olcut": {
            "ad": "gramer_dogrulayicinin_alt_kumesi_ve_maske_bos_degil",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "dogrulayici_kacis = 0 VE rastgele_kacis = 0 VE naif_gecersiz > 0 VE "
                "cikmaz_red >= 1 VE adim_bitti_red >= 1 VE utf8_red = 4 VE belge > 0 "
                "VE bayt > 0"
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
            "Logitler deterministik bir karisimdan geliyor: egitilmis bir kontrol "
            "noktasi yok, yani bu kayit cevap kalitesi olcmuyor. Sozluk sentetiktir "
            "(256 tek bayt + 8 cok baytli jeton); gercek sozluk lubot-bpe-v2'dir ve "
            "bu kayitta kosulmadi. Otomat, dogrulayicidan bilerek dardir: satir sonu "
            "bosluk/sekme ayrimi, `\\r` reddi ve asiri uzun/vekil UTF-8 reddi "
            "dogrulayicinin suskun kaldigi yerlerde karar verir. "
            "Otomat cikti uretimini lubot-read'in tek cikisina baglamaz: crate "
            "yalniz maske ve decode verir, uretim yuzeyi yoktur."
        ),
        "olculmeyen": [
            "cevap kalitesine etkisi: uzmanli egitim kosusu ve sinav seti ister",
            "gercek sozlukle (lubot-bpe-v2) maske maliyeti ve red orani",
            "omurgaya baglandiktan sonraki gecikme: adim basina maske maliyeti "
            "sozluk boyutuyla dogrusal, buyuk sozlukte olculmedi",
            "kaynaktaki sayisal sonuclar (bu kayit yalniz kendi kosumumuzu olcer)",
            "dagitim disi (egitilmemis) logitlerde maskenin pratik etkisi: "
            "sentetik karisim yalniz maske mekanigini olcer",
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
    for alan in TAM_ALANLAR:
        deger = kanit.get(alan)
        if not isinstance(deger, (int, float)) or isinstance(deger, bool):
            return f"kanit alani sayisal degil: {alan}"
    if _olcut({a: float(kanit[a]) for a in TAM_ALANLAR}) != kayit["olcut"]["sonuc"]:
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
    for alan in TAM_ALANLAR:
        if float(eski[alan]) != float(taze[alan]):
            raise SystemExit(
                f"{alan} kayittan farkli cikti: kayit {eski[alan]}, olcum {taze[alan]}"
            )
    if kayit["olcut"]["sonuc"] != taze["olcut_sonucu"]:
        raise SystemExit("olcut sonucu degisti")
    return "kayit taze: " + " ".join(
        f"{alan}={taze[alan]:g}" for alan in TAM_ALANLAR
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
