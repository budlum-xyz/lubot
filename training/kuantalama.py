#!/usr/bin/env python3
"""Nicem (alt-bayt kuantalama) port karti icin karta-ozel olcum kaydi.

Olcum Rust testinin icindedir (`crates/nicem/src/paket.rs`,
`paket::tests::olcum_raporu_bit_butcesi`): beyan edilen agirlik-basina-bit
butcesi iki bagimsiz yoldan yeniden hesaplanir, paket sikiligi ve yuvarlak yol
her genislikte **sayilarak** denetlenir ve sabit bir tensör dort adimlik yolun
(cevir / coz / olcekle / paketle) tamamindan gecirilip geri kurma hatasi
olculur. Bu betik o satiri **kosar ve okur**; sayilari kendisi uretmez.

Hedef olcut (kayittan once yazilir, sonuc kayitta olculur):

    bit_fark = 0 VE paket_ihlal = 0 VE yuvarlak_yol_ihlal = 0 VE
    ucdeger_ihlal = 0 VE cozulen_ihlal = 0 VE adet_sayisi = 44 VE
    14 < sikistirma <= 32/2.125 VE snr_db > kitap_snr_db

Son madde iki yonde isirir: sikistirma aritmetik tavani (32 bit / 2.125 bit)
asamaz, asarsa bayt sayaci yalan soyluyordur; olculen SNR ise kod kitabinin
**kendi** on gordugu bozulmadan kotu olamaz, olursa cozucu kitaptan sapmistir.

"Kuantalama kaliteyi dusurmez" iddiasi bu kayitta **yoktur**: agirliklar egitilmis
bir tensör degil, sabit sentetik bir dagilimdir; kalite sinav setinin isidir.

Kullanim:

    python3 training/kuantalama.py --olc
    python3 training/kuantalama.py --kur
    python3 training/kuantalama.py --dogrula
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
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "kuantalama-2026-09-27.json"
ETIKET = "kuantalama |"
TAM_ALANLAR = (
    "grup",
    "seviye",
    "uzunluk",
    "eksen",
    "adet_sayisi",
    "paket_ihlal",
    "yuvarlak_yol_ihlal",
    "ucdeger_ihlal",
    "cozulen_ihlal",
)
KESIRLI_ALANLAR = (
    "bit_beyan",
    "bit_paket",
    "bit_alfabe",
    "bit_fark",
    "ucdeger_bit",
    "bagil_hata",
    "snr_db",
    "kitap_snr_db",
    "en_buyuk_sapma",
    "sikistirma",
)
# 32 bit / 2.125 bit: beyan edilen butcenin aritmetik tavani.
SIKISTIRMA_TAVANI = 32.0 / 2.125


def _test_kos() -> tuple[int, str]:
    kosu = subprocess.run(
        ["cargo", "test", "-q", "-p", "lubot-nicem",
         "paket::tests::olcum_raporu_bit_butcesi", "--", "--nocapture"],
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
    return bool(
        k["bit_fark"] == 0
        and k["paket_ihlal"] == 0
        and k["yuvarlak_yol_ihlal"] == 0
        and k["ucdeger_ihlal"] == 0
        and k["cozulen_ihlal"] == 0
        and k["adet_sayisi"] == 44
        and 14 < k["sikistirma"] <= SIKISTIRMA_TAVANI + 1e-9
        and k["snr_db"] > k["kitap_snr_db"]
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
            "Alt-bayt kuantalama (tasarim 3.6): beyan edilen agirlik-basina-bit butcesi "
            "iki yoldan yeniden hesaplandi, paket sikiligi ve yuvarlak yol her genislikte "
            "sayildi, sabit bir tensör dort adimlik yoldan gecirilip geri kurma hatasi olculdu"
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "port_karti": "workspace:skills/port-hatti/port-kartlari/needle-kuantalama.md",
        "olcut": {
            "ad": "bit_butcesi_tam_sayi_aritmetigi",
            "sonuc": bool(olcum["olcut_sonucu"]),
            "ifade": (
                "grup=128, iki bit: bit_fark = 0 VE paket_ihlal = 0 VE yuvarlak_yol_ihlal = 0 "
                "VE ucdeger_ihlal = 0 VE cozulen_ihlal = 0 VE adet_sayisi = 44 VE "
                "14 < sikistirma <= 32/2.125 VE snr_db > kitap_snr_db"
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
            "Agirliklar egitilmis bir tensör degil: 1024 elemanlik sabit sentetik dagilim, "
            "dort bilinen aykiri degerle (bir grup agir kuyruk tasisin diye). Hata "
            "sayilari bu dagilimda gecerlidir, cevap kalitesi hakkinda hicbir sey soylemez. "
            "Crate hicbir dis bagimlilik almaz; dort adimin tamami bu agacta yazilidir."
        ),
        "olculmeyen": [
            "kaynaktaki sayisal sonuclar (bu kayit yalniz kendi kosumumuzu olcer)",
            "kuantalamanin cevap kalitesine etkisi: sinav seti ve egitilmis kontrol noktasi ister",
            "egitilmis agirlik dagiliminda hata: bu kayit sentetik dagilimda kosuldu",
            "grup boyutu izgarasi (16/32/64/256) ve ucdeger alfabesinin uctan uca hatasi: "
            "ucdeger yalniz paket sikiligi ve yuvarlak yol uzerinden olculdu",
            "servis gecikmesi ve bellege yerlesim: tasiyici/cihaz crate'lerinin isi, burada olculmedi",
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
    for alan in TAM_ALANLAR + KESIRLI_ALANLAR:
        deger = kanit.get(alan)
        if not isinstance(deger, (int, float)) or isinstance(deger, bool):
            return f"kanit alani sayisal degil: {alan}"
    if _olcut({a: float(kanit[a]) for a in TAM_ALANLAR + KESIRLI_ALANLAR}) != kayit["olcut"]["sonuc"]:
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
