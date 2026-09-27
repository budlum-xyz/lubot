#!/usr/bin/env python3
"""Tipli karar basligi port kaydi (uc kapali sekil, uretim yuzeyi yok, esik/k-of-n/kalibrasyon davranisi).

Olcum Rust testinin icindedir (`crates/tomurcuk/src/lib.rs`, `tests::olcum_raporu_tipli_karar`). Bu betik o satiri
**kosar ve okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    cikis yuzeyi tam uc kapali sekil (secim/puan/evet-hayir) VE 6 kapali karar noktasi VE olasilik disi 4 deger 4/4 reddedilir VE esik altinda tek bas yukseltir VE 2/3 oy karar verir, karisik tip yukselir VE kalibre puan sicaklikta monoton VE k=2, n=3

Kullanim:

    python3 training/tipli_karar.py --olc
    python3 training/tipli_karar.py --kur
    python3 training/tipli_karar.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "tipli-karar-2026-09-27.json"
ETIKET = "tipli-karar |"
TAM_ALANLAR = ('sekil', 'secenek', 'gecersiz_red', 'esik_alti_yukseltir', 'k_of_n', 'kalibre_monoton', 'konsensus_k', 'konsensus_n')
KESIRLI_ALANLAR = ()


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-tomurcuk", "tests::olcum_raporu_tipli_karar", "--", "--nocapture"],
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
    return bool(k['sekil'] == 3 and k['secenek'] == 6 and k['gecersiz_red'] == 4 and k['esik_alti_yukseltir'] == 1 and k['k_of_n'] == 1 and k['kalibre_monoton'] == 1 and k['konsensus_k'] == 2 and k['konsensus_n'] == 3)


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
        "is": 'Tipli karar basligi: non-autoregressive, tek geciste kapali-sekil karar (secim/puan/evet-hayir); metin yuzeyi yok; esik, k-of-n ve kalibrasyon davranisi olculdu',
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:skills/port-hatti/port-kartlari/laya-karar-basligi.md",
        "olcut": {
            "ad": 'karar_yuzeyi_uc_kapali_sekil_ve_uretim_yok',
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": 'cikis yuzeyi tam uc kapali sekil (secim/puan/evet-hayir) VE 6 kapali karar noktasi VE olasilik disi 4 deger 4/4 reddedilir VE esik altinda tek bas yukseltir VE 2/3 oy karar verir, karisik tip yukselir VE kalibre puan sicaklikta monoton VE k=2, n=3',
        },
        "kaynaklar": {
            "girdi_jetonlari": 0,
            "cikti_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "maliyet": 0.0,
            "sure_saniye": olcum["sure_saniye"],
        },
        "kanit": olcum,
        "uyari": 'Olcum yapisal ve davranissaldir: egitilmis bir karar basligi yok, etiketli karar kumesi uzerinde dogruluk olculmedi. Uretim yuzeyinin yoklugu ayrica decision-head-has-no-generation-surface kapisiyla denetlenir.',
        "olculmeyen": ['kaynaktaki sayisal sonuclar (ornek: tipli karar dogrulugu) bu kayitla dogrulanmis sayilmaz', 'etiketli karar kumesinde bu basligin dogrulugu (egitilmis baslik yok)', 'cok dilli omurga uzerinde karar davranisi (agirlik indirilmedi, olculmedi)', 'omurga temsili ile karar basliginin baglanmasi (kodlayici -> tomurcuk): ayri artim'],
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
    if not kayit.get("olculmeyen"):
        return "olculmeyen listesi bos"
    kanit = kayit.get("kanit") or {}
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
        if alan not in kanit:
            return f"kanit alani yok: {alan}"
    if _olcut(kanit) != kayit["olcut"]["sonuc"]:
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
