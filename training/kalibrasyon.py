#!/usr/bin/env python3
"""Kalibre edilmis guven bandi: olcum kaydi (tasarim notu 3.7).

Tasarim notu guven basligi icin uc sey ister: sicaklik kalibrasyonu
(dogrulama kumesinde olculur), esik yerine **bant** (kirmizi/orta/yesil) ve
kaydin `training/eval/sonuclar/` altina mekanik-kosu semasiyla yazilmasi.

Olcum Rust tarafindadir: `lubot kalibrasyon` komutu sicakligi uydurur, ECE'yi
hesaplar ve bantlari kayitlardan olcer. Bu betik ikinci bir uygulama yazmaz;
komutu kosar, ozet satirini okur ve kaydi yazar. Sayilar boylece tek bir
kaynakatan gelir - Python'da yeniden hesaplanan bir sayi, iki uygulamanin
ayrisma ihtimalini satin alirdi.

Olcut tek cumlede:
    ayni kayit dosyasi icin uydurulan sicaklik egitim hatasini ve ECE'yi
    dusurur, bantlar olcumden cikar ve hedef isabet olculemedigi yerde
    komut REDDEDER (fail-closed).

Beyan edilmesi gereken sinir: kayit dosyasi bir **duzenek kanaryasidir**
(`training/kalibrasyon-fixture.jsonl`), egitilmis bir basin skorlari degil -
bugun egitilmis bir bas yok (K6: sahibin donanimi). Gercek skorlar geldiginde
ayni betik ayni yoldan kosar; asagidaki "olculmeyen" listesi o gun kısalır.

    python3 training/kalibrasyon.py --kur      # kaydi yazar
    python3 training/kalibrasyon.py --dogrula  # kayit taze mi
    python3 training/kalibrasyon.py --olc      # kaydi yazmadan olcum
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
FIXTURE = ROOT / "training" / "kalibrasyon-fixture.jsonl"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "kalibrasyon-2026-09-26.json"
# Kayit dosyasi asiri guvenli bir basi taklit eder: hedef 0.75'te bantlar
# olculebilir, varsayilan 0.9'da yesil band YOKTUR (bu da bir olcumdur).
HEDEF_OLCULEN = 0.75
# Ozet satiri: `ad=deger` ciftleri; sira degisirse betik kirilmaz.
OZET = re.compile(r"kalibrasyon:\s+(.+)")
CIFT = re.compile(r"([a-z_]+)=([-\d.]+)")


def _kos(hedef: float) -> tuple[int, str, dict[str, float]]:
    """Komutu kosar; (donus kodu, cikti, ozet) doner."""
    islem = subprocess.run(
        [
            "cargo", "run", "--quiet", "-p", "lubot", "--",
            "kalibrasyon", "--girdi", str(FIXTURE.relative_to(ROOT)),
            "--hedef", f"{hedef}",
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    cikti = islem.stdout + islem.stderr
    eslesme = OZET.search(cikti)
    ozet: dict[str, float] = {}
    if eslesme:
        ozet = {ad: float(deger) for ad, deger in CIFT.findall(eslesme.group(1))}
    return islem.returncode, cikti.strip(), ozet


def olc() -> dict:
    basla = time.monotonic()
    kod_olculen, cikti_olculen, ozet = _kos(HEDEF_OLCULEN)
    kod_hedef, cikti_hedef, _ = _kos(0.9)
    sure = round(time.monotonic() - basla, 2)
    if kod_olculen != 0:
        raise SystemExit(f"olculen kosu kirmizi ({kod_olculen}):\n{cikti_olculen[-800:]}")
    if not ozet:
        raise SystemExit(f"ozet satiri okunamadi:\n{cikti_olculen[-800:]}")
    if kod_hedef == 0:
        raise SystemExit(
            "hedef 0.9 kosusu gecti: bu kayit dosyasinda yesil band OLMAMALIYDI "
            "(olcum degisti mi, yoksa komut mu gevsetildi?)"
        )
    if "yesil-bant-yok" not in cikti_hedef:
        raise SystemExit(
            f"hedef 0.9 reddi beklenen gerekceyle gelmedi:\n{cikti_hedef[-400:]}"
        )
    kayit_sayisi = sum(
        1 for satir in FIXTURE.read_text(encoding="utf-8").splitlines() if satir.strip()
    )
    return {
        "girdi": str(FIXTURE.relative_to(ROOT)),
        "girdi_kayit": kayit_sayisi,
        "hedef_olculen": HEDEF_OLCULEN,
        "hedef_reddedilen": 0.9,
        "sicaklik": ozet["sicaklik"],
        "nll_ham": ozet["nll_ham"],
        "nll_fit": ozet["nll_fit"],
        "ece_ham": ozet["ece_ham"],
        "ece_fit": ozet["ece_fit"],
        "yesil_alt": ozet["yesil_alt"],
        "kirmizi_ust": ozet["kirmizi_ust"],
        "destek_alti": int(ozet.get("destek_alti", 0)),
        "kayit": int(ozet["kayit"]),
        "hedef_0_9_reddi": cikti_hedef.splitlines()[-1].strip(),
        "sure_saniye": sure,
    }


def _kayit(olcum: dict) -> dict:
    return {
        "is": (
            "Kalibre guven bandi (tasarim 3.7): ham (puan, dogru) kayitlarindan "
            "sicaklik uyduruldu, ECE duzeltme oncesi/sonrasi hesaplandi ve bantlar "
            "(kirmizi/orta/yesil) kayitlardan olculdu. Sayilar `lubot kalibrasyon` "
            "komutundan gelir; Python tarafinda ikinci bir hesaplama yok."
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "uydurma_egitim_hatasini_ve_eceyi_dusurdu_bantlar_olcumden_cikti",
            "sonuc": True,
        },
        "kaynaklar": {
            "sure_saniye": olcum["sure_saniye"],
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": olcum,
        "uyari": (
            "Kayit dosyasi bir DUZENEK KANARYASIDIR: egitilmis bir basin skorlari "
            "degil, olcum duzenegini sinayan beyanli veridir. Duzeltme bilgi "
            "uretemez - kanaryada hedef 0.9'da yesil band cikmaz ve komut bunu "
            "reddeder; olculen sey mekanizmanin kendisidir."
        ),
        "olculmeyen": [
            "egitilmis bir karar basinin kalibrasyonu (agirlik yok, K6)",
            "gercek dogrulama kumesinde bant genisligi (ilk egitim turundan sonra)",
            "bant basina yukseltme maliyeti (gecikme kaydi ayri tutuluyor)",
        ],
    }


def kur() -> Path:
    olcum = olc()
    kayit = _kayit(olcum)
    if _bulgu(kayit) is not None:
        raise SystemExit(f"kayit semasi tutmuyor: {_bulgu(kayit)}")
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(
        json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KAYIT


def _bulgu(kayit: dict) -> str | None:
    """Mekanik kosu semasinin kendi denetimi (kapiya guvenmeden once)."""
    if kayit.get("kosucu") != "betik":
        return "kosucu `betik` degil"
    olcut = kayit.get("olcut") or {}
    if not isinstance(olcut.get("sonuc"), bool):
        return "olcut sonucu boolean degil"
    if " " in str(olcut.get("ad", "")):
        return "olcut adi cumle gibi"
    for alan in ("sure_saniye", "girdi_jetonlari", "onbellekli_jetonlari",
                 "cikti_jetonlari", "maliyet"):
        if alan not in (kayit.get("kaynaklar") or {}):
            return f"kaynaklar.{alan} yok"
    kanit = kayit.get("kanit") or {}
    if kanit.get("kayit") != kanit.get("girdi_kayit"):
        return "kayit sayisi girdi satir sayisina esit degil"
    if not kanit.get("nll_fit", 9) < kanit.get("nll_ham", 0):
        return "duzeltilmis egitim hatasi hamdan kucuk degil"
    if not kanit.get("ece_fit", 9) < kanit.get("ece_ham", 0):
        return "duzeltilmis ECE hamdan kucuk degil"
    if not kanit.get("kirmizi_ust", 9) <= kanit.get("yesil_alt", -1):
        return "kirmizi band yesil bandin ustune tasmis"
    return None


def dogrula() -> str:
    if not KAYIT.is_file():
        raise SystemExit(f"kayit yok: {KAYIT.relative_to(ROOT)} (once --kur)")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    bulgu = _bulgu(kayit)
    if bulgu:
        raise SystemExit(f"{KAYIT.name}: {bulgu}")
    taze = olc()
    eski = kayit["kanit"]
    for alan in ("sicaklik", "nll_ham", "nll_fit", "ece_ham", "ece_fit"):
        if abs(float(eski[alan]) - float(taze[alan])) > 1e-6:
            raise SystemExit(
                f"{KAYIT.name}: {alan} bayat (kayit {eski[alan]}, olcum {taze[alan]})"
            )
    for alan in ("yesil_alt", "kirmizi_ust", "kayit", "girdi_kayit"):
        if eski[alan] != taze[alan]:
            raise SystemExit(
                f"{KAYIT.name}: {alan} bayat (kayit {eski[alan]}, olcum {taze[alan]})"
            )
    return (
        f"kayit taze: sicaklik {taze['sicaklik']:.6f}, bantlar "
        f"[0,{taze['kirmizi_ust']:.2f}) / [{taze['kirmizi_ust']:.2f},{taze['yesil_alt']:.2f}) / "
        f"[{taze['yesil_alt']:.2f},1], {taze['kayit']} kayit"
    )


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    grup = ayristirici.add_mutually_exclusive_group(required=True)
    grup.add_argument("--kur", action="store_true", help="kaydi yaz")
    grup.add_argument("--dogrula", action="store_true", help="kayit taze mi")
    grup.add_argument("--olc", action="store_true", help="kaydi yazmadan olc")
    secim = ayristirici.parse_args(argv)
    if secim.kur:
        yol = kur()
        sys.stdout.write(f"kayit yazildi: {yol.relative_to(ROOT)}\n")
    elif secim.dogrula:
        sys.stdout.write(dogrula() + "\n")
    else:
        sys.stdout.write(json.dumps(olc(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
