#!/usr/bin/env python3
"""Omurga norm yeri port kaydi (on-norm pinli: norm dalin oncesinde, artik disarida; bias yok).

Olcum Rust testinin icindedir (`crates/kodlayici/src/blok.rs`, `blok::tests::olcum_raporu_norm_yeri`). Bu betik o satiri
**kosar ve okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    norm agirligi sifirken artik akis bit-ozdes gecer (norm dalin icinde, artik disinda) VE birim normla dal katkida bulunur (> 1e-3) VE katman_norm ciktisi ortalama 0 (< 1e-6), varyans 1 (< 1e-5) VE katman agirliklari 6 tensor, 0 bias

Kullanim:

    python3 training/norm_yeri.py --olc
    python3 training/norm_yeri.py --kur
    python3 training/norm_yeri.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "norm-yeri-2026-09-27.json"
ETIKET = "norm-yeri |"
TAM_ALANLAR = ('genislik', 'uzunluk', 'artik_bit_ozdes', 'tensor_sayisi', 'bias_tensoru')
KESIRLI_ALANLAR = ('dal_katkisi', 'ortalama_sapma', 'varyans_sapma')


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-kodlayici", "blok::tests::olcum_raporu_norm_yeri", "--", "--nocapture"],
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
    return bool(k['artik_bit_ozdes'] == 1 and k['dal_katkisi'] > 1e-3 and k['ortalama_sapma'] < 1e-6 and k['varyans_sapma'] < 1e-5 and k['bias_tensoru'] == 0 and k['tensor_sayisi'] == 6)


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
        "is": "Omurga norm yeri: on-norm duzeni mekanik olarak pinlenir (sifir norm agirligi = bit-ozdes artik gecis), katman_norm tanim degerleri, bias'siz katman duzeni",
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:skills/port-hatti/port-kartlari/modernbert-normalizasyon.md",
        "olcut": {
            "ad": 'pre_norm_duzeni_pinli',
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": 'norm agirligi sifirken artik akis bit-ozdes gecer (norm dalin icinde, artik disinda) VE birim normla dal katkida bulunur (> 1e-3) VE katman_norm ciktisi ortalama 0 (< 1e-6), varyans 1 (< 1e-5) VE katman agirliklari 6 tensor, 0 bias',
        },
        "kaynaklar": {
            "girdi_jetonlari": 0,
            "cikti_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "maliyet": 0.0,
            "sure_saniye": olcum["sure_saniye"],
        },
        "kanit": olcum,
        "uyari": 'Olcum kucuk, tohumlu agirliklarla yapilir. On-norm ile son-norm arasindaki derinlik-kararlilik farki egitimle olculmedi; bu kayit yalniz duzenin yerini kanitlar.',
        "olculmeyen": ['kaynaktaki sayisal sonuclar (bu kayit yalniz kendi kosumumuzu olcer)', 'on-norm/son-norm farkinin derin yiginda egitim kararliligina etkisi (egitim kosusu ister)', 'sifirdan egitilen cekirdekteki sifir merkezli RMS adayi ile bu katman_norm arasindaki kalite farki (olculmedi)'],
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
