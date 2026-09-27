"""Danisma katmani: oy verir, karar vermez.

Bu modul anayasanin K7 maddesinin kodudur (bkz. `INVARIANTS.md`). Döngü,
tut/at kararini **belirsiz** buldugunda (marj, kabul esiginin bant'i icindeyse)
danisma katmanina bir *oy* sorar. Danisma katmani:

* karar kuralini degistiremez - esik, marj ve tut/at mantigi bu dosyada ve
  `dongu.py`'de yazilidir;
* olcutu, veriyi, donanimi ya da degerlendirme tanimini degistiremez;
* korpusa kayit uretemez, dokunulmaz dosyalara yazamaz (bu dosyada yazma yok);
* oy esigin altindaysa ya da arka uc yoksa karar **insana** kalir ve dongu
  S2 durdurma kosulu ile durur.

Olculen gerekce (2026-09-24, bu makine): yerel Laya karar bataryasinda TR 5/8,
EN 4/8 ve hata yonu hep "evet" tarafinda. Bu yuzden oy, karari *devralmaz*;
yalniz band'i daraltir ve belirsizlikte insana goturur.

Kullanim:

    python3 autonomous-training/danisma.py --kendini-test
    python3 autonomous-training/danisma.py --kart tut_at --skor 6.6717 \
        --mevcut 6.67174 --esik 0.001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BURASI = Path(__file__).resolve().parent
AYARLAR = BURASI / "ayarlar.json"
VARSAYILAN_SERVIS = "http://127.0.0.1:8790"
ZAMAN_ASIMI_SANIYE = 20.0


def ayarlari_yukle(yol: Path | None = None) -> dict:
    return json.loads((yol or AYARLAR).read_text(encoding="utf-8")).get("danisma", {})


def bant(esik: float, carpan: float) -> float:
    """Belirsizlik bandi: kabul esiginin katı (kural, olcume uydurulmaz)."""
    return abs(esik) * carpan


def marj(skor: float, mevcut: float | None, esik: float) -> float | None:
    """Kabul sinirina uzaklik: pozitifse tutulur, negatifse atilir, 0 sinirdir."""
    if mevcut is None:
        return None
    return skor - (mevcut - esik)


def belirsiz_mi(marj_degeri: float | None, bant_degeri: float) -> bool:
    """Mevcut yoksa (ilk deney) ya da marj bant icindeyse belirsizdir."""
    return marj_degeri is None or abs(marj_degeri) <= bant_degeri


def oy_iste(kart: str, soru: str, secenekler: list[str], ayar: dict,
            zaman_asimi: float = ZAMAN_ASIMI_SANIYE) -> dict:
    """Arka uctan oy ister. Hata bir *oy yoklugu* olarak doner, istisna degil."""
    servis = ayar.get("servis_url", VARSAYILAN_SERVIS)
    govde = json.dumps({"kart": kart, "soru": soru, "secenekler": secenekler}).encode("utf-8")
    basliklar = {"Content-Type": "application/json"}
    # Sunucu belirteci (sunucu.py ile ayni ortam degiskeni): danisma sunucusu
    # belirtecsiz istek kabul etmez; belirtec yoksa istek yine gider, reddi
    # "oy yoklugu" olarak doner ve karar insana kalir (fail-closed zincir).
    belirtec = os.environ.get("LUBOT_DANISMA_TOKEN")
    if belirtec:
        basliklar["Authorization"] = f"Bearer {belirtec}"
    istek = urllib.request.Request(
        servis.rstrip("/") + "/oy", data=govde,
        headers=basliklar, method="POST")
    basla = time.monotonic()
    try:
        with urllib.request.urlopen(istek, timeout=zaman_asimi) as yanit:
            cevap = json.loads(yanit.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as hata:
        return {"oy": None, "guven": 0.0, "arka_uc": "yok",
                "gecikme_ms": int((time.monotonic() - basla) * 1000),
                "gerekce": f"arka uc yok: {type(hata).__name__}"}
    cevap.setdefault("gecikme_ms", int((time.monotonic() - basla) * 1000))
    return cevap


def kalibrasyon_probu(ayar: dict, oy_al=None) -> dict:
    """Oyun *bilgi tasiyip tasimadigini* olcer: bilinen cevapli iki prob.

    Neden gerekli (olculdu, 2026-09-24): yerel Laya iki secenekli tut/at
    sorusunda **sabit** oy verdi - "sinirda", "acikca kotu", "acikca iyi" uc
    durumun ucuunde de "tut" (guven 0,61-0,68). Sabit oy veren bir katman
    guven esigini gecse bile bilgi tasimaz; onu karara sokmak, karari gizlice
    "hep tut"a cevirirdi. Bu yuzden oy, once cevabi bilinen iki probe ile
    sinanir:

    * prob-1: en iyi skor 6.67, deney 6.50, esik 0,001 -> dogru cevap "tut";
    * prob-2: en iyi skor 6.67, deney 9.00, esik 0,001 -> dogru cevap "at".

    Ikisini de bilemeyen bir arka ucun oyu **gecersizdir**: karar insana gider.
    """
    cagri = oy_al or oy_iste
    taban, esik = 6.671741, 0.001
    problar = [
        ("prob-1", "tut", 6.50),
        ("prob-2", "at", 9.00),
    ]
    sonuclar, gecen = [], 0
    for ad, beklenen, skor in problar:
        soru = (f"Onceki en iyi skor {taban}; bu deneyin skoru {skor}; kabul esigi {esik}. "
                f"Bu deney tutulmali mi?")
        cevap = cagri(kart="tut_at", soru=soru, secenekler=["tut", "at"], ayar=ayar)
        ham = cevap.get("oy")
        dogru = ham == beklenen
        gecen += int(dogru)
        sonuclar.append({"ad": ad, "beklenen": beklenen, "oy": ham,
                         "dogru": dogru, "guven": cevap.get("guven")})
    return {"gecti": gecen == len(problar), "gecen": gecen, "toplam": len(problar),
            "sonuclar": sonuclar}


def karar_oyu(ayar: dict, skor: float, mevcut: float | None, esik: float,
              kart: str = "tut_at", oy_al=None) -> dict:
    """Belirsizlikte oy sorar; donen sey bir *oy* ve bir *insan gerekli mi* bayragidir."""
    danisma = ayar or {}
    sonuc = {"danisildi": False, "kart": kart, "insan_gerekli": False, "oy": None}
    if not danisma.get("etkin", False):
        sonuc["gerekce"] = "danisma kapali (ayarlar.json: danisma.etkin)"
        return sonuc
    bant_degeri = bant(esik, float(danisma.get("bant_carpani", 1.0)))
    marj_degeri = marj(skor, mevcut, esik)
    sonuc.update({"bant": bant_degeri, "marj": marj_degeri})
    if not belirsiz_mi(marj_degeri, bant_degeri):
        sonuc["gerekce"] = f"bant disi (|marj| {abs(marj_degeri):.6f} > {bant_degeri:.6f})"
        return sonuc
    soru = (
        f"Deney skoru {skor:.6f}; onceki en iyi {mevcut if mevcut is not None else 'yok'}; "
        f"kabul esigi {esik:.6f}. Marj {marj_degeri if marj_degeri is not None else 0:.6f}; "
        f"yon {'tut' if (marj_degeri or 0) >= 0 else 'at'}. Bu deney tutulmali mi?"
    )
    secenekler = ["tut", "at"]
    cagri = oy_al or oy_iste
    if danisma.get("kalibrasyon_prob", True) and oy_al is None:
        # Gercek arka ucta prob kosulur; kanarya (oy_al) kendi cevabini getirir.
        prob = kalibrasyon_probu(danisma)
        sonuc["kalibrasyon"] = prob
        # Dikkat: `gecen` bir SAYIDIR (1/2). `not prob["gecen"]` ilk yazimda
        # yanlislikla kullanildi ve prob dustugu halde karar oyla verildi;
        # kanarya yalniz probe fonksiyonunu sinadigi icin bunu gormedi. Karar
        # bu yuzden boolean `gecti`ye baglanir ve kanarya artik *karar yolunu*
        # da sinar (asagida "probu dusen arka uc karari oya birakmaz").
        if not prob["gecti"]:
            sonuc.update({"danisildi": True, "insan_gerekli": True, "oy": None,
                          "gerekce": f"kalibrasyon probu dustu ({prob['gecen']}/{prob['toplam']}): "
                                     f"oy bilgi tasimiyor"})
            return sonuc
    oy = cagri(kart=kart, soru=soru, secenekler=secenekler, ayar=danisma)
    guven = float(oy.get("guven") or 0.0)
    esik_guven = float(danisma.get("guven_esigi", 0.55))
    gecerli_oy = oy.get("oy") if oy.get("oy") in secenekler else None
    sonuc.update({"danisildi": True, "oy": gecerli_oy, "guven": guven,
                  "arka_uc": oy.get("arka_uc"), "gecikme_ms": oy.get("gecikme_ms"),
                  "oy_ham": oy.get("oy"), "guven_esigi": esik_guven})
    if gecerli_oy is None:
        sonuc["insan_gerekli"] = True
        sonuc["gerekce"] = "oy gecersiz/bos (arka uc karar vermedi)"
    elif guven < esik_guven:
        sonuc["insan_gerekli"] = True
        sonuc["gerekce"] = f"oy guveni {guven:.3f} < esik {esik_guven:.3f}"
    else:
        sonuc["gerekce"] = f"oy {gecerli_oy} (guven {guven:.3f}); karar kurali kodda"
    return sonuc


def _sahte(oy: str, guven: float):
    def ic(**_: object) -> dict:
        return {"oy": oy, "guven": guven, "arka_uc": "kanarya", "gecikme_ms": 1}

    return ic


def kendini_test() -> list[str]:
    """Kanarya: bant disi sorulmaz, bant ici esik alti oy insana gider."""
    bulgular: list[str] = []
    ayar = {"etkin": True, "bant_carpani": 1.0, "guven_esigi": 0.55}
    # Bant disi: hic sorulmaz.
    oy = karar_oyu(ayar, 6.5000, 6.671741, 0.001, oy_al=_sahte("tut", 0.99))
    assert oy["danisildi"] is False, "bant disi kararda oy soruldu"
    bulgular.append("bant disi sorulmaz")
    # Bant ici + esik ustu oy: danisildi, insan gerekmez.
    sinir = 6.671741 - 0.001
    oy = karar_oyu(ayar, sinir + 0.0005, 6.671741, 0.001, oy_al=_sahte("tut", 0.90))
    assert oy["danisildi"] and oy["insan_gerekli"] is False, "bant ici oy insana gitmedi"
    bulgular.append("bant ici oy")
    # Bant ici + esik alti guven: insan gerekir.
    oy = karar_oyu(ayar, sinir + 0.0005, 6.671741, 0.001, oy_al=_sahte("tut", 0.10))
    assert oy["insan_gerekli"] is True, "esik alti oy insana gitmedi"
    bulgular.append("esik alti oy -> insan")
    # Bant ici + arka uc yok: insan gerekir.
    oy = karar_oyu(ayar, sinir + 0.0005, 6.671741, 0.001, oy_al=_sahte(None, 0.0))
    assert oy["insan_gerekli"] is True, "arka uc yokken insana gitmedi"
    bulgular.append("arka uc yok -> insan")
    # Ilk deney (mevcut yok) belirsizdir.
    oy = karar_oyu(ayar, 6.5, None, 0.001, oy_al=_sahte("tut", 0.9))
    assert oy["danisildi"] is True, "ilk deney belirsiz sayilmadi"
    bulgular.append("ilk deney belirsiz")
    # Sabit oy veren arka uc: prob duser, oy gecersiz sayilir (asil kanarya).
    def sabit(**_: object) -> dict:
        return {"oy": "tut", "guven": 0.90, "arka_uc": "sabit", "gecikme_ms": 1}

    prob = kalibrasyon_probu(ayar, oy_al=sabit)
    assert prob["gecen"] == 1 and not prob["gecti"], "sabit oy veren arka uc probu gecti"
    # Kapaliysa hic sorulmaz.
    oy = karar_oyu({"etkin": False}, 6.67174, 6.671741, 0.001, oy_al=_sahte("tut", 0.9))
    assert oy["danisildi"] is False, "kapali danisma soru sordu"
    bulgular.append("kapali danisma")

    def dogru(soru: str, **_: object) -> dict:
        return {"oy": "at" if "9.0" in soru else "tut", "guven": 0.9,
                "arka_uc": "dogru", "gecikme_ms": 1}

    prob = kalibrasyon_probu(ayar, oy_al=dogru)
    assert prob["gecti"], "dogru cevap veren arka uc probu dusurdu"
    bulgular.append("kalibrasyon probu: sabit oy duser, dogru oy gecer")

    # KARAR YOLU kanaryasi: probu dusen bir arka uc, karari oya birakamaz.
    gercek = globals()["oy_iste"]
    try:
        globals()["oy_iste"] = sabit           # sabit "tut" oyu veren sahte servis
        yol = karar_oyu(ayar, 6.6717405, 6.671741, 0.001)   # oy_al None: prob kosar
        assert yol["insan_gerekli"] is True, "probu dusen arka uc karari oya birakti"
        assert yol["oy"] is None, "gecersiz oy kayda gecti"
        assert "kalibrasyon" in yol["gerekce"], f"gerekce kalibrasyondan soz etmiyor: {yol['gerekce']}"
        bulgular.append("probu dusen arka uc -> insan (karar yolu)")
        # Dogru cevap veren arka ucta karar oya gider ve kod kuralina dokunmaz.
        globals()["oy_iste"] = lambda **k: {"oy": "at" if "9.0" in k.get("soru", "") else "tut",
                                            "guven": 0.9, "arka_uc": "dogru", "gecikme_ms": 1}
        yol = karar_oyu(ayar, 6.6717405, 6.671741, 0.001)
        assert yol["insan_gerekli"] is False and yol["oy"] == "tut", "probu gecen arka uc oyu kullanilmadi"
        bulgular.append("probu gecen arka uc -> oy kayda gecer")
    finally:
        globals()["oy_iste"] = gercek
    return bulgular


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--kendini-test", action="store_true")
    ayristirici.add_argument("--kart", default="tut_at")
    ayristirici.add_argument("--skor", type=float)
    ayristirici.add_argument("--mevcut", type=float)
    ayristirici.add_argument("--esik", type=float, default=0.001)
    ayristirici.add_argument("--ayar-dosya", type=Path)
    args = ayristirici.parse_args(argv)
    if args.kendini_test:
        print("self-test OK [danisma]: " + ", ".join(kendini_test()))
        return 0
    if args.skor is None:
        ayristirici.error("--skor gerekli (ya da --kendini-test)")
    oy = karar_oyu(ayarlari_yukle(args.ayar_dosya), args.skor, args.mevcut, args.esik, args.kart)
    print(json.dumps(oy, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not oy["insan_gerekli"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
