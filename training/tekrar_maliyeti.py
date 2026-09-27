#!/usr/bin/env python3
"""Tekrarli sorunun marjinal maliyeti: W maddesinin olcumu.

U maddesi "tekrarli soruda maliyet sifira yaklasir" diyor. Bu iddia ancak
olculurse dogru ya da yanlis olur: bugun `ask` yolunda ne onbellek ne sicak
sunucu var (`ilk_cevap_gecikme.py` kaydi bunu soyluyor).

Olcumun tuzagi iki tane ve ikisi de tasarimla kapatildi:

1. **Isletim sisteminin sayfa onbellegi.** Ikinci cagri, soru ayni oldugu
   icin degil, korpus dosyasi artik RAM'de oldugu icin hizli olabilir.
   Kontrol: fiyati esitlenmis BASKA bir soru. Ikinci cagri, baska sorunun
   ilk cagrisindan belirgin sekilde hizli degilse uygulama duzeyinde
   yeniden kullanim YOK.
2. **Soru maliyetinin farkli olmasi.** Kontrol sorusu once taranir: ilk
   cagri fiyati referansa en yakin aday secilir ve eslesme farki kayda
   gecer; fark buyurse olcum kullanilamaz sayilir ve kosu durur.

Olcut tek cumlede:
    tekrar/kontrol = ikinci ayni-cagri / baska-sorunun-ilk-cagrisi;
    bu oran REUSE_ESIGI'nin altindaysa yeniden kullanim var sayilir.

Maliyet duvar saatidir ve makineye baglidir: kayit ratchet'e girmez (kiyas
da girmedi). Kapi sayilarin degismedigini degil, kaydin mekanik oldugunu ve
sonucun taze olcumle ayni yonde ciktigini denetler.

    python3 training/tekrar_maliyeti.py --olc
    python3 training/tekrar_maliyeti.py --kur
    python3 training/tekrar_maliyeti.py --dogrula
    python3 training/tekrar_maliyeti.py --self-test
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IKILI = ROOT / "target" / "debug" / "lubot"
KORPUS = ROOT / "corpus" / "knowledge-self.jsonl.gz"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "tekrar-maliyeti-2026-09-24.json"

SORU = "what does a view grant name?"
ADAYLAR = [
    "which crate scans for credentials?",
    "what does the queue log hold?",
    "where is the corpus stamp read?",
    "which command prints the inventory?",
    "what does the effort axis compare?",
]
KOSU = 10
ESLESME_TAVANI = 0.25
REUSE_ESIGI = 0.6

UYARI = (
    "Olculen sureler duvar saatidir ve makineye baglidir; kayit bu yuzden "
    "ratchet'e girmez. Sayilar surec baslatma + korpus ayristirmayi icerir, "
    "cunku cagiranin odedigi maliyet budur. Kontrol sorusu once fiyat "
    "eslesmesi icin taranir; eslesme farki kayitta durur."
)


def _kosu(soru: str) -> float:
    """Tek `ask` cagrisinin duvar saati (ms)."""
    basla = time.monotonic()
    kosu = subprocess.run(
        [str(IKILI), "ask", "--corpus", str(KORPUS), "--reader", "olcum",
         "--effort", "1.0x", soru],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    gecen = (time.monotonic() - basla) * 1000.0
    if kosu.returncode != 0:
        raise SystemExit(f"ask dustu: {(kosu.stderr or kosu.stdout)[-200:]}")
    if "# Answer" not in kosu.stdout:
        raise SystemExit("ask ciktisi cevap basligi tasimiyor: olcum kayit degil")
    return gecen


def _eslestir(referans_ms: float) -> tuple[str, float]:
    """Referansa fiyati en yakin adayi sec; eslesme farkini da dondur.

    Fiyat eslesmesi olcumun on sartidir: iki soru farkli maliyetteyse
    'ikinci cagri daha hizli' sonucu onbellekten mi soru kolayligindan mi
    geldigi ayrilmaz."""
    en_iyi = ("", float("inf"))
    for aday in ADAYLAR:
        fiyat = statistics.median([_kosu(aday) for _ in range(3)])
        fark = abs(fiyat - referans_ms) / referans_ms if referans_ms else float("inf")
        if fark < en_iyi[1]:
            en_iyi = (aday, fark)
    return en_iyi


def olc() -> dict:
    if not IKILI.is_file():
        raise SystemExit(f"{IKILI.relative_to(ROOT)} yok: once cargo build -p lubot")
    if not KORPUS.is_file():
        raise SystemExit(f"{KORPUS.relative_to(ROOT)} yok: once korpus kurulur")

    referans_ms = statistics.median([_kosu(SORU) for _ in range(3)])
    kontrol, eslesme = _eslestir(referans_ms)
    if eslesme > ESLESME_TAVANI:
        raise SystemExit(
            f"kontrol sorusu fiyat eslesmesi tutmadi (fark {round(eslesme * 100, 1)}% > "
            f"{round(ESLESME_TAVANI * 100, 1)}%): once aday listesine fiyati yakin bir soru ekle"
        )

    a1: list[float] = []
    b1: list[float] = []
    a2: list[float] = []
    for _ in range(KOSU):
        a1.append(_kosu(SORU))
        b1.append(_kosu(kontrol))
        a2.append(_kosu(SORU))
    medyan_a1 = statistics.median(a1)
    medyan_b1 = statistics.median(b1)
    medyan_a2 = statistics.median(a2)
    tekrar_bolu_ilk = medyan_a2 / medyan_a1 if medyan_a1 else 0.0
    tekrar_bolu_kontrol = medyan_a2 / medyan_b1 if medyan_b1 else 0.0
    yeniden_kullanim = tekrar_bolu_kontrol < REUSE_ESIGI
    return {
        "soru": SORU,
        "kontrol_soru": kontrol,
        "kontrol_eslesme_farki": round(eslesme, 4),
        "kosu": KOSU,
        "ilk_ms": round(medyan_a1, 3),
        "kontrol_ms": round(medyan_b1, 3),
        "tekrar_ms": round(medyan_a2, 3),
        "tekrar_bolu_ilk": round(tekrar_bolu_ilk, 4),
        "tekrar_bolu_kontrol": round(tekrar_bolu_kontrol, 4),
        "yeniden_kullanim_esigi": REUSE_ESIGI,
        "onbellek_var": yeniden_kullanim,
        "uyari": UYARI,
    }


def kayit_yaz(olcum: dict) -> Path:
    """Kayit: tek mekanik olcut + kaynak muhasebesi + durust hukum."""
    destekleniyor = olcum["onbellek_var"]
    kayit = {
        "is": (
            "Ayni soru arka arkaya iki kez soruldu; ikinci cagri, fiyati "
            "esitlenmis BASKA bir sorunun ilk cagrisiyla karsilastirildi. "
            "Kontrol, isletim sistemi sayfa onbellegi ile uygulama duzeyi "
            "yeniden kullanimi ayirmak icin var."
        ),
        "kosucu": "ask",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "tekrar_bolu_kontrol_orani_esigin_altinda",
            "sonuc": bool(destekleniyor),
        },
        "kaynaklar": {
            "sure_saniye": round((3 + 3 * len(ADAYLAR) + 3 * olcum["kosu"]) * olcum["ilk_ms"] / 1000.0, 2),
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": olcum,
        "hukum": (
            "tekrarli soruda maliyet sifira yaklasmiyor: ikinci ayni cagri "
            f"({olcum['tekrar_ms']} ms), fiyati esitlenmis baska sorunun ilk "
            f"cagrisindan ({olcum['kontrol_ms']} ms) ucuz degil "
            f"(tekrar/kontrol {olcum['tekrar_bolu_kontrol']})"
            if not destekleniyor
            else f"yeniden kullanim var: tekrar/kontrol {olcum['tekrar_bolu_kontrol']}"
        ),
        "yapilmayan": (
            "cevap onbellegi YAZILMADI; U maddesinin iddiasi olculdu ve bugun "
            "desteklenmiyor. Onbellek eklemek operator karari: kalicilik, "
            "gecersizlestirme ve damga tazeligi kurallarini da getirir."
        ),
        "uyari": UYARI,
    }
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(
        json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KAYIT


def dogrula() -> str:
    """Kayit mekanik mi ve sonuc taze olcumle ayni yonde mi."""
    if not KAYIT.is_file():
        raise SystemExit(f"kayit yok: {KAYIT.relative_to(ROOT)} (once --kur)")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    if not isinstance(kayit.get("uyari"), str) or "duvar saati" not in kayit["uyari"]:
        raise SystemExit("kayit duvar saati uyarisini tasimiyor")
    if kayit.get("olcut", {}).get("ad") != "tekrar_bolu_kontrol_orani_esigin_altinda":
        raise SystemExit("kayit beklenen mekanik olcutu tasimiyor")
    kanit = kayit.get("kanit", {})
    if not isinstance(kanit.get("kontrol_eslesme_farki"), (int, float)):
        raise SystemExit("kayit kontrol fiyat eslesmesini tasimiyor")
    taze = olc()
    if kayit.get("kanit", {}).get("onbellek_var") != taze["onbellek_var"]:
        raise SystemExit(
            f"kayit bayat: kayitta onbellek_var={kayit.get('kanit', {}).get('onbellek_var')}, "
            f"taze olcum {taze['onbellek_var']} (tekrar/kontrol {taze['tekrar_bolu_kontrol']}) "
            "- kaydi yeniden uret"
        )
    return (
        f"tekrarli soru olculdu: ilk {taze['ilk_ms']} ms, tekrar {taze['tekrar_ms']} ms, "
        f"kontrol {taze['kontrol_ms']} ms ({taze['kontrol_soru']}, fiyat farki "
        f"{round(taze['kontrol_eslesme_farki'] * 100, 1)}%); tekrar/kontrol "
        f"{taze['tekrar_bolu_kontrol']} (esik {REUSE_ESIGI}) -> onbellek_var={taze['onbellek_var']}"
    )


def _self_test() -> None:
    """Kanarya: uyarisiz kayit, yanlis olcut adi ve eslesme farki tasimayan
    kayit reddedilir; esik yeniden kullanim olcusu olarak kalir."""
    with tempfile.TemporaryDirectory() as td:
        global KAYIT
        gercek = KAYIT
        KAYIT = Path(td) / "k.json"
        try:
            temel = {"olcut": {"ad": "tekrar_bolu_kontrol_orani_esigin_altinda"},
                     "kanit": {"onbellek_var": False, "kontrol_eslesme_farki": 0.02},
                     "uyari": "duvar saati"}
            KAYIT.write_text(json.dumps({"olcut": temel["olcut"], "kanit": temel["kanit"]}),
                             encoding="utf-8")
            try:
                dogrula()
            except SystemExit as e:
                assert "duvar saati" in str(e), f"yanlis sebep: {e}"
            else:
                raise AssertionError("uyarisiz kayit gecti")
            KAYIT.write_text(json.dumps(dict(temel, olcut={"ad": "baska_bir_sey"})),
                             encoding="utf-8")
            try:
                dogrula()
            except SystemExit as e:
                assert "mekanik olcut" in str(e), f"yanlis sebep: {e}"
            else:
                raise AssertionError("yanlis olcut adi gecti")
            KAYIT.write_text(json.dumps(dict(temel, kanit={"onbellek_var": False})),
                             encoding="utf-8")
            try:
                dogrula()
            except SystemExit as e:
                assert "eslesme" in str(e), f"yanlis sebep: {e}"
            else:
                raise AssertionError("eslesme farki tasimayan kayit gecti")
        finally:
            KAYIT = gercek
    assert REUSE_ESIGI < 1.0, "esik yeniden kullanim olcusu olmali"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--olc", action="store_true")
    g.add_argument("--kur", action="store_true")
    g.add_argument("--dogrula", action="store_true")
    g.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        print("self-test OK [tekrar-maliyeti]")
        return 0
    if args.olc:
        sys.stdout.write(json.dumps(olc(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 0
    if args.kur:
        olcum = olc()
        yol = kayit_yaz(olcum)
        print(
            f"kayit yazildi: {yol.relative_to(ROOT)} - ilk {olcum['ilk_ms']} ms, tekrar "
            f"{olcum['tekrar_ms']} ms, kontrol {olcum['kontrol_ms']} ms; tekrar/kontrol "
            f"{olcum['tekrar_bolu_kontrol']} -> onbellek_var={olcum['onbellek_var']}"
        )
        return 0
    print(dogrula())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
