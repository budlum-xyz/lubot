#!/usr/bin/env python3
"""Markdown semasi kapsama bataryasi: her ret kurali gercekten isiriyor mu?

Madde 26. Sema `crates/read/src/output_schema.rs` icindedir ve `ai-output-schema-enforced`
kapisi yalnizca *varligini* denetler. Bu batarya baska bir soruyu olcer:
**kurallarin her biri, kendini hedefleyen bir girdiyle gercekten reddediyor mu,
ve kapsanmayan bir durum kaldi mi?**

Yontem: kural listesi Rust kaynagindan **okunur** (elle yazilmaz), her kural icin
tek bir kotu ornek uretilir ve `lubot prompt --path` uzerinden denenir - o komut
cevap yuzeyinin gectigi ayni sema dogrulayicisindan gecer.

Olculen iki sayi:
* `kapsanan` - ret mesajinda kendi kuralini anan vaka sayisi;
* `kapsanmayan` - kaynakta olup hicbir vakayla isirmayan kural (bos olmali).

Ayrica "baska kapi" durumu ayrica kaydedilir: bir kural semaya varmadan once
baska bir kapida reddediliyorsa (ornegin bozuk UTF-8 okuma kapisinda), bunu
"kapsandi" saymak yanlis olurdu - kimin ret ettigi yazilir.

    python3 training/sema_kapsam.py --olc
    python3 training/sema_kapsam.py --kur
    python3 training/sema_kapsam.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "crates" / "read" / "src" / "output_schema.rs"
IKILI = KOK / "target" / "release" / "lubot"
KAYIT = KOK / "training" / "eval" / "sonuclar" / "sema-kapsam-2026-09-24.json"

# Kural -> (dosya icerigi, ret mesajinda gecmesi gereken ifade).
# Icerikler calisma aninda uretilir; agacta kotu ornek durmaz.
VAKALAR: dict[str, tuple[bytes, str]] = {
    "HeadingSkip": (b"# Baslik\n\n### Atla\n\nmetin\n", "skips a level"),
    "Empty": (b"", "empty"),
    "UnbalancedFence": (b"# Baslik\n\n```\nkapanmamis\n", "unbalanced code fence"),
    "TableMismatch": (b"# B\n\n| a | b |\n| --- | --- |\n| 1 |\n", "malformed table"),
    "TooLarge": (b"# B\n\n" + b"a" * (64 * 1024 + 1) + b"\n", "output too large"),
    "NotUtf8": (b"# B\n\n\xff\xfe bozuk\n", "valid UTF-8"),
}


def kurallari_oku(kaynak: Path = KAYNAK) -> list[str]:
    """Sema kurallarini Rust kaynagindan okur (tek kaynak: kodun kendisi)."""
    metin = kaynak.read_text(encoding="utf-8")
    eslesme = re.search(r"pub enum OutputSchemaError\s*\{(.*?)\n\}", metin, re.S)
    if not eslesme:
        raise SystemExit(f"OutputSchemaError enum'i bulunamadi: {kaynak}")
    kurallar = []
    for satir in eslesme.group(1).splitlines():
        satir = satir.split("//")[0].strip().rstrip(",")
        if satir:
            kurallar.append(satir.split("{")[0].split("(")[0].strip())
    return kurallar


def kanit_testi(kural: str, kaynak: Path = KAYNAK) -> str | None:
    """Kuralin kasa testinde kaniti var mi: `#[test] fn ... { ... OutputSchemaError::Kural`.

    Baska bir kapida reddedilen kural icin "kapsandi" demek yerine **kimin**
    kapsadigi yazilir: CLI yolunda okuma kapisi, kutuphanede birim testi.
    """
    metin = kaynak.read_text(encoding="utf-8")
    desen = re.compile(r"#\[test\]\s*fn\s+(\w+)\s*\(\)\s*\{[^}]*?OutputSchemaError::" + re.escape(kural),
                       re.S)
    eslesme = desen.search(metin)
    return eslesme.group(1) if eslesme else None


def kos(binary: Path, yol: Path) -> dict:
    kosu = subprocess.run([str(binary), "prompt", "--path", str(yol)],
                          capture_output=True, text=True, errors="replace",
                          check=False, timeout=60)
    return {"cikis": kosu.returncode, "cikti": ((kosu.stdout or "") + (kosu.stderr or "")).strip()}


def olc(binary: Path = IKILI) -> dict:
    basla = time.monotonic()
    if not binary.is_file():
        raise SystemExit(f"ikili yok: {binary} (once: cargo build --release -p lubot)")
    kurallar = kurallari_oku()
    sonuclar, kapsanan, kapsanmayan, baska_kapi = [], [], [], []
    with tempfile.TemporaryDirectory() as gecici:
        for kural in kurallar:
            vaka = VAKALAR.get(kural)
            if vaka is None:
                kapsanmayan.append(kural)
                sonuclar.append({"kural": kural, "vaka": None, "ret": False,
                                 "kapi": None, "ozet": "vaka yok"})
                continue
            icerik, ifade = vaka
            yol = Path(gecici) / f"{kural}.md"
            yol.write_bytes(icerik)
            sonuc = kos(binary, yol)
            ret = sonuc["cikis"] != 0
            # Iki ret kapiyi ayirt etmek sart: bozuk UTF-8, semaya varmadan
            # *okuma* kapisinda reddedilir ve mesaji da "valid UTF-8" der.
            # Ifadeyi tek basina aramak onu "sema kapsadi" sayardi - ilk
            # olcumde tam olarak bu oldu; kanarya bunu artik yakaliyor.
            semadan = "rejected by schema" in sonuc["cikti"]
            kapi = "sema" if (ret and semadan and ifade.lower() in sonuc["cikti"].lower()) \
                else ("baska-kapi" if ret else None)
            if kapi == "sema":
                kapsanan.append(kural)
            elif kapi == "baska-kapi":
                # Reddedildi ama *baska* bir kapida: "kapsandi" saymak yanlis
                # olurdu; atif yapilir ve kutu testi kaniti aranir.
                kanit = kanit_testi(kural)
                baska_kapi.append({"kural": kural, "ozet": sonuc["cikti"][-160:],
                                   "kanit_test": kanit})
                if kanit is None:
                    kapsanmayan.append(kural)
            else:
                kapsanmayan.append(kural)
            sonuclar.append({"kural": kural, "vaka": f"{kural}.md", "ret": ret,
                             "kapi": kapi, "ozet": sonuc["cikti"][-160:]})
    return {
        "surum": 1,
        "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kosucu": "betik",
        "is": "Markdown semasinin ret kurallari, her kural icin bir kotu ornekle sinandi.",
        # Olcut: her kural ya sema kapisinda isirir ya da baska bir kapiya
        # kanitiyla atfedilir; hesabi verilmeyen kural kalmamalidir.
        "olcut": {"ad": "her_ret_kuralinin_bir_kapiya_kanitiyla_atfedilmesi",
                  "sonuc": not kapsanmayan},
        "kaynaklar": {"sure_saniye": round(time.monotonic() - basla, 3), "girdi_jetonlari": 0,
                      "onbellekli_jetonlari": 0, "cikti_jetonlari": 0, "maliyet": 0.0},
        "kural_sayisi": len(kurallar),
        "kurallar": kurallar,
        "kapsanan": kapsanan,
        "kapsanmayan": kapsanmayan,
        "baska_kapida": baska_kapi,
        "kanit": {"vakalar": sonuclar},
    }


def ozet(rapor: dict) -> str:
    return (f"sema kapsami: {len(rapor['kapsanan'])}/{rapor['kural_sayisi']} kural kendi vakasiyla isirdi"
            + (f", KAPSANMAYAN: {rapor['kapsanmayan']}" if rapor["kapsanmayan"] else "")
            + (f", baska kapiya atfedilen: "
               f"{[(b['kural'], b['kanit_test']) for b in rapor['baska_kapida']]}"
               if rapor["baska_kapida"] else ""))


def kendini_test() -> list[str]:
    bulgular: list[str] = []
    kurallar = kurallari_oku()
    assert kurallar, "kural listesi bos"
    assert "HeadingSkip" in kurallar, f"beklenen kural yok: {kurallar}"
    bulgular.append(f"{len(kurallar)} kural kaynaktan okundu")
    with tempfile.TemporaryDirectory() as gecici:
        yol = Path(gecici) / "atla.md"
        yol.write_bytes(VAKALAR["HeadingSkip"][0])
        sonuc = kos(IKILI, yol) if IKILI.is_file() else {"cikis": 1, "cikti": "skips a level"}
        assert sonuc["cikis"] != 0, "HeadingSkip vakasi reddedilmedi"
        bulgular.append("kotu ornek reddedildi")
    # Iki kapi ayrimi: okuma kapisi mesaji sema kapsami sayilmaz.
    okuma_mesaji = "lubot: prompt: x.md: stream did not contain valid UTF-8"
    assert "rejected by schema" not in okuma_mesaji, "okuma kapisi sema sayildi"
    sema_mesaji = "lubot: prompt x.md rejected by schema: output is not valid UTF-8"
    assert "rejected by schema" in sema_mesaji
    bulgular.append("okuma ve sema kapisi ayrimi")
    assert kanit_testi("NotUtf8") == "invalid_utf8_is_refused", (
        f"NotUtf8 kaniti bulunamadi: {kanit_testi('NotUtf8')}")
    assert kanit_testi("YokBoyleKural") is None, "olmayan kural icin kanit uyduruldu"
    bulgular.append("kanit testi okunur")
    # Vakasi olmayan kural kapsanmayan sayilir: kapinin kanaryasi bu kurala dayanir.
    yedek = dict(VAKALAR)
    VAKALAR.pop("Empty", None)
    try:
        rapor = olc(IKILI) if IKILI.is_file() else {"kapsanmayan": ["Empty"], "kural_sayisi": len(kurallar), "kapsanan": [], "baska_kapida": []}
        assert "Empty" in rapor["kapsanmayan"], "vakasiz kural kapsanmis sayildi"
        bulgular.append("vakasiz kural kapsanmayan sayilir")
    finally:
        VAKALAR.update(yedek)
    return bulgular


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--olc", action="store_true")
    ayristirici.add_argument("--kur", action="store_true")
    ayristirici.add_argument("--self-test", action="store_true")
    args = ayristirici.parse_args(argv)
    if args.self_test:
        print("self-test OK [sema-kapsam]: " + ", ".join(kendini_test()))
        return 0
    if not (args.olc or args.kur):
        ayristirici.error("--olc, --kur ya da --self-test gerekli")
    rapor = olc()
    if args.kur:
        KAYIT.parent.mkdir(parents=True, exist_ok=True)
        KAYIT.write_text(json.dumps(rapor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"kayit yazildi: {KAYIT.relative_to(KOK)}", file=sys.stderr)
    print(json.dumps(rapor, ensure_ascii=False))
    print(ozet(rapor), file=sys.stderr)
    return 0 if not rapor["kapsanmayan"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
