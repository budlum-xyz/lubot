#!/usr/bin/env python3
"""Hadamard (Monarch bloklu) MLP adayi port kaydi (tasarim 3.1): parametre muhasebesi, gradyan, inis, carpim.

Olcum Rust testinin icindedir (`crates/egitim/src/mlp_hadamard.rs`, `mlp_hadamard::tests::olcum_raporu`). Bu betik o satiri
**kosar ve okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    d_model=8, d_r=6, blok=2: denetlenen gradyan sayisi = sekilden turetilen parametre sayisi VE ihlal 0 VE en kotu bagil sapma < 1e-6 VE 40 adimda kayip her adimda duser VE ikinci dal sifirken cikti = b3 (carpim gercek) VE bloklu parametre < bloksuz

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
TAM_ALANLAR = ('d_model', 'd_r', 'blok', 'parametre', 'parametre_bloksuz', 'denetlenen', 'ihlal', 'monoton', 'carpim_sifir')
KESIRLI_ALANLAR = ('en_kotu_bagil', 'kayip0', 'kayip40')


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-egitim", "mlp_hadamard::tests::olcum_raporu", "--", "--nocapture"],
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
    return bool(k['denetlenen'] == k['parametre'] and k['ihlal'] == 0 and k['en_kotu_bagil'] < 1e-6 and k['kayip40'] < k['kayip0'] and k['monoton'] == 1 and k['carpim_sifir'] == 1 and k['parametre'] < k['parametre_bloksuz'])


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
        "is": "Hadamard/Monarch MLP adayi: elle yazilmis geri gecis merkezi sonlu farka karsi, parametre muhasebesi tam sayi, 40 adim inis, carpim yapisi; spec'e baglanmadi",
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:skills/port-hatti/port-kartlari/needle-hadamard-mlp.md",
        "olcut": {
            "ad": 'gradyan_sonlu_farkla_ve_sayim_sekle_bagli_ve_inis',
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": 'd_model=8, d_r=6, blok=2: denetlenen gradyan sayisi = sekilden turetilen parametre sayisi VE ihlal 0 VE en kotu bagil sapma < 1e-6 VE 40 adimda kayip her adimda duser VE ikinci dal sifirken cikti = b3 (carpim gercek) VE bloklu parametre < bloksuz',
        },
        "kaynaklar": {
            "girdi_jetonlari": 0,
            "cikti_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "maliyet": 0.0,
            "sure_saniye": olcum["sure_saniye"],
        },
        "kanit": olcum,
        "uyari": 'Olcum kucuk tohumlu sekilde (8/6/2) yapilir; hicbir egitim kosusuna bagli degil. "Standart MLP\'den iyi" iddiasi yok: bloksuz Hadamard ayni ic genislikte standart MLP\'den pahalidir (ek dusus izdusumu), bu modul testinde yazilidir.',
        "olculmeyen": ['d_r ve blok secimi izgarasi (isaretli karar M1; olculmeden aileye girmez)', 'bu korpusun api/behaviour agirligi altinda standart MLP ile kalite kiyasi (egitim kosusu ister)', 'muP tablosunda Hadamard fan_in satiri (init/LR aktarimi olculmedi)', 'spec genisliginde (d_model 64, d_ff 256) parametre/hiz olcumu'],
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
