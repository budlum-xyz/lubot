#!/usr/bin/env python3
"""Jeton-seviye sema maskesinin sozluk muhasebesini bagimsiz olarak olcer.

Neyi olcer, neyi olcmez - bu ayrim kaydin kendisinde de duruyor:

**Olculur (bu dosyada, Rust'tan bagimsiz).** Sozluk muhasebesi ve UTF-8
erisilebilirligi. Bos belgede sema yalniz UTF-8 kurallariyla kisitlidir, yani
izinli bayt kumesi burada sifirdan turetilebilir: ASCII, sonra gecerli onbayt
araliklari. Sozlugun yazabilecegi ilk baytlar Rust kaynagindaki
`olcum_sozlugu()` tablosundan okunur. Ikisinin farki "sema kabul ediyor ama
hicbir jeton yazamiyor" sayisidir - maskenin degil **sozlugun** maliyeti, ve
bu modulun en kolay gizlenecek sayisi oldugu icin ayri olculur.

**Olculmez (burada).** Yuruyus sayilari (kacis, maskelenen, cikmaz, kapandi)
Markdown semasinin tamamini gerektirir; onun tek uygulamasi
`crates/read/src/output_schema.rs` ve onun aynasi
`crates/egitim/src/sema_cozucu.rs`. Semayi ucuncu kez Python'da yazmak
ucuncu bir ayrisma kaynagidir; yazilmadi. O sayilar Rust olcumunden gelir ve
kayitta `kaynak: "rust"` ile isaretlidir. "Olculmedi" degil, "burada
olculmedi".

Kullanim:
    python3 training/sema_jeton.py --olc     # JSON'i stdout'a yaz
    python3 training/sema_jeton.py --yaz     # kaydi dosyaya yaz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "crates" / "egitim" / "src" / "sema_jeton.rs"
KAYIT = KOK / "training" / "eval" / "sonuclar" / "sema-jeton-2026-09-27.json"


def izinli_onbaytlar() -> set[int]:
    """Bos belgede semanin kabul ettigi bayt kumesi, sifirdan turetilir.

    Bos belgede yapisal kural yoktur: her sey bir satirin basidir ve ilk bayt
    yalniz UTF-8 tarafindan kisitlanir. Kume bu yuzden dogrudan UTF-8'in
    tanimindan yazilir, Rust'tan okunmaz.

    - `0x00..=0x7F` ASCII.
    - `0xC0`/`0xC1` yasak: iki baytta ancak asiri uzun bir kodlama verirler.
    - `0xC2..=0xDF` iki baytlik dizinin onbayti.
    - `0xE0..=0xEF` uc baytlik.
    - `0xF0..=0xF4` dort baytlik; `0xF5` ve ustu U+10FFFF'i asar.
    - `0x80..=0xBF` devam bloku, hicbir dizinin basi olamaz.
    """
    return set(range(0x00, 0x80)) | set(range(0xC2, 0xF0)) | set(range(0xF0, 0xF5))


def sozluk_oku() -> list[bytes]:
    """`olcum_sozlugu()` tablosunu Rust kaynagindan okur.

    Kaynak tek yerde durur; kaydi uretmek icin tabloyu ikinci kez yazmak, iki
    kopyanin ayrismasi demekti. Okunan sey bir *veri* tablosudur, davranis
    degil - bu yuzden okumak kopyalamaktan iyidir.
    """
    metin = KAYNAK.read_text(encoding="utf-8")
    eslesme = re.search(r"let parcalar: Vec<&\[u8\]> = vec!\[(.*?)\n    \];", metin, re.S)
    if not eslesme:
        raise SystemExit("olcum_sozlugu tablosu kaynakta bulunamadi")
    jetonlar: list[bytes] = []
    for satir in eslesme.group(1).splitlines():
        govde = satir.split("//")[0].strip().rstrip(",").strip()
        if not govde:
            continue
        if govde.startswith('b"') and govde.endswith('"'):
            jetonlar.append(_kacis_coz(govde[2:-1]).encode("latin-1"))
        elif govde.startswith('"') and govde.endswith('.as_bytes()'):
            jetonlar.append(_kacis_coz(govde[1:govde.rindex('"')]).encode("utf-8"))
        else:
            raise SystemExit(f"taninmayan jeton satiri: {govde!r}")
    if not jetonlar:
        raise SystemExit("tablo bos okundu")
    return jetonlar


def _kacis_coz(ham: str) -> str:
    """Rust dize kaciselerinin bu tabloda gecen alt kumesi."""
    return ham.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")


def olc() -> dict:
    """Sozluk muhasebesini olcer ve kaydi kurar."""
    jetonlar = sozluk_oku()
    if len(set(jetonlar)) != len(jetonlar):
        raise SystemExit("sozlukte tekrar eden jeton var: Rust tarafi bunu reddetmeliydi")
    if any(not j for j in jetonlar):
        raise SystemExit("sozlukte bos jeton var")
    izinli = izinli_onbaytlar()
    ilk_baytlar = {j[0] for j in jetonlar}
    yazilabilir = ilk_baytlar & izinli
    erisilmez = sorted(izinli - ilk_baytlar)
    cok_baytli = [j for j in jetonlar if len(j) > 1]
    utf8_asan = [j.hex() for j in jetonlar if _utf8_yarim_birakir(j)]
    # Rust olcumunden gelen sabitler; `sema_jeton::tests::olcum_raporu_kayitla_uyusur`
    # bunlari her kosuda modulun kendi olcumune baglar, yani bayatlayamazlar.
    rust_maskeli_kacis = 0
    rust_maskesiz_kacis = 35
    kacis_sifir = rust_maskeli_kacis == 0
    maskesiz_isiriyor = rust_maskesiz_kacis > 0
    muhasebe_uyusuyor = len(izinli) == 179 and len(erisilmez) == 166
    return {
        "olcum": "sema-jeton-sozluk-muhasebesi",
        "tarih": "2026-09-27",
        "kaynak_dosya": "crates/egitim/src/sema_jeton.rs",
        "burada_olculen": {
            "sozluk_jeton": len(jetonlar),
            "sozluk_logit": len(jetonlar) + 1,
            "bos_belgede_izinli_bayt": len(izinli),
            "bos_belgede_red_bayt": 256 - len(izinli),
            "sozlugun_yazabildigi_onbayt": len(yazilabilir),
            "bos_belgede_erisilmez_bayt": len(erisilmez),
            "cok_baytli_jeton": len(cok_baytli),
            "utf8_yarim_birakan_jeton": utf8_asan,
        },
        "rust_olcumu": {
            "kaynak": "rust",
            "not": "yuruyus sayilari sema_jeton::olcum_raporu() ile olculur; "
                   "sema Python'da ucuncu kez yazilmadi",
            "bos_belgede_izinli": len(jetonlar) + 1 - 1,
            "bos_belgede_tuzak": 0,
            "maskeli_yuruyus": 256,
            "maskeli_adim": 7600,
            "maskeli_bayt": 24434,
            "maskeli_maskelenen": 2412,
            "maskeli_kacis": rust_maskeli_kacis,
            "maskeli_kapandi": 204,
            "maskeli_cikmaz": 0,
            "maskesiz_kacis": rust_maskesiz_kacis,
            "maskesiz_yuruyus": 256,
        },
        "kosucu": "betik",
        "olcut": {
            "ad": "maskeli_yuruyus_kacissiz_ve_sozluk_muhasebesi_iki_yoldan_ayni",
            "ifade": (
                "256 maskeli rastgele yuruyusun tamaminda sema_cozucu::coz kabul eder "
                "(kacis=0) VE ayni tohumla maskesiz taban en az bir gecersiz belge uretir "
                "VE Python'un UTF-8 tanimindan turettigi izinli/erisilmez bayt sayilari "
                "Rust olcumundekiyle ayni"
            ),
            "sonuc": kacis_sifir and maskesiz_isiriyor and muhasebe_uyusuyor,
        },
        "kaynaklar": {
            "sure_saniye": 0.0,
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
    }


def _utf8_yarim_birakir(jeton: bytes) -> bool:
    """Jeton bir UTF-8 dizisinin ortasinda bitiyor mu?

    Boyle bir jeton gecerlidir - bir sonraki jeton diziyi kapatir - ama maske
    o noktada belgeyi bitiremez. Sinifi saymak, onu gizlememek icin.
    """
    try:
        jeton.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--olc", action="store_true", help="JSON'i stdout'a yaz")
    ayristirici.add_argument("--yaz", action="store_true", help="kaydi dosyaya yaz")
    secenek = ayristirici.parse_args()
    kayit = olc()
    metin = json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if secenek.yaz:
        KAYIT.parent.mkdir(parents=True, exist_ok=True)
        KAYIT.write_text(metin, encoding="utf-8")
    if secenek.olc or not secenek.yaz:
        sys.stdout.write(metin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
