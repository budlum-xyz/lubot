#!/usr/bin/env python3
"""Alt-bayt agirlik nicemleme port kaydi (tasarim 3.5): bit/agirlik, yuvarlak yol hatasi, donusun kazanci, determinizm.

Olcum Rust testinin icindedir (`crates/nicem/src/grup.rs`, `grup::tests::olcum_raporu`). Bu betik o satiri
**kosar ve okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    2 bit + 128 grup, 4096 agirlik: agirlik basina bit = 2.125 (tam aritmetik) VE 1088 bayt VE olculen SNR kod kitabi tahminine 3 dB icinde VE agir kuyrukta donuslu hata absmax tabanindan kucuk VE 1..6 bit arasinda hata tekduze duser VE ayni tensor iki kez ayni baytlari verir VE bagil hata < 0.35

Kullanim:

    python3 training/nicem.py --olc
    python3 training/nicem.py --kur
    python3 training/nicem.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "nicem-2026-09-27.json"
ETIKET = "nicem |"
TAM_ALANLAR = ('bit', 'grup', 'agirlik', 'bayt', 'tekduze', 'bayt_esit')
KESIRLI_ALANLAR = ('agirlik_basina_bit', 'oran', 'bagil_hata', 'snr_db', 'kitap_snr_db', 'snr_fark', 'agir_donuslu', 'agir_absmax')


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-nicem", "grup::tests::olcum_raporu", "--", "--nocapture"],
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
    return bool(k['agirlik_basina_bit'] == 2.125 and k['bayt'] == 1088 and k['snr_fark'] < 3.0 and k['agir_donuslu'] < k['agir_absmax'] and k['tekduze'] == 1 and k['bayt_esit'] == 1 and k['bagil_hata'] < 0.35)


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
        "is": 'Alt-bayt nicemleme: Walsh-Hadamard donusu + Lloyd-Max kod kitabi + grup olcegi; bit butcesi aritmetik, yuvarlak yol hatasi ve donusun kazanci sentetik tensorde olculdu; servis oncesi nicemleme, egitim yuksek hassasiyette kalir',
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:skills/port-hatti/port-kartlari/needle-kuantalama.md",
        "olcut": {
            "ad": 'bit_butcesi_aritmetik_ve_yuvarlak_yol_olculu',
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": '2 bit + 128 grup, 4096 agirlik: agirlik basina bit = 2.125 (tam aritmetik) VE 1088 bayt VE olculen SNR kod kitabi tahminine 3 dB icinde VE agir kuyrukta donuslu hata absmax tabanindan kucuk VE 1..6 bit arasinda hata tekduze duser VE ayni tensor iki kez ayni baytlari verir VE bagil hata < 0.35',
        },
        "kaynaklar": {
            "girdi_jetonlari": 0,
            "cikti_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "maliyet": 0.0,
            "sure_saniye": olcum["sure_saniye"],
        },
        "kanit": olcum,
        "uyari": "Tensorler sentetiktir (tohumlu normalimsi dagilim ve agir kuyruk fixture'i); egitilmis bir kontrol noktasinin nicemleme sonrasi kalite kaybi olculmedi. Tek buyuk agirlikli grup (spike) donusun en kotu halidir ve modul testinde kayitlidir.",
        "olculmeyen": ['egitilmis lubot-a1 agirliklarinda nicemleme sonrasi sinav kaybi (kontrol noktasi yok)', 'egitim-zamani nicem farkindaligi (QAT): olculmeden onerilmez (tasarim 3.5)', 'kademe egitimi ile birlikte derinlik merdiveni davranisi (M4)', 'gercek donanimda 2 bit carpim hizi (olculmedi)'],
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
